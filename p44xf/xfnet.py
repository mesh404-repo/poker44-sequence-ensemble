"""Independent reimplementation of a hierarchical set-transformer bot detector for poker chunks.

A chunk is a SET of hands, each hand a SEQUENCE of actions. We encode actions with a per-action
Transformer, attention-pool them to a hand vector, run a per-hand Transformer over the
(permutation-invariant) set of hands, then attention-pool to a chunk vector -> logit. Chunks are
depth-capped to a fixed number of hands via even-spaced sampling (matches training chunk depth).
Our own tokenization (action_type/street/is_hero/amount_bucket/pot_flow + continuous), same raw
field names as build_seq_dataset.py so it aligns with the GRU pipeline.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

torch.backends.cudnn.enabled = False

_ACT = {"fold": 1, "check": 2, "call": 3, "bet": 4, "raise": 5,
        "post": 6, "blind": 6, "small_blind": 6, "big_blind": 6, "ante": 6, "sb": 6, "bb": 6}
_STREET = {"preflop": 1, "flop": 2, "turn": 3, "river": 4, "showdown": 4}
N_ACT, N_STREET, N_HERO, N_BUCKET, N_POTFLOW = 8, 5, 3, 8, 3
CONT_DIM = 2
MAX_HANDS = 48
MAX_ACTIONS = 12


def _i(v, d=0):
    try:
        return int(v)
    except Exception:
        return d


def _f(v, d=0.0):
    try:
        return float(v)
    except Exception:
        return d


def _bucket(amt_bb: float) -> int:
    a = max(0.0, amt_bb)
    if a <= 0.0:
        return 1
    for k, hi in enumerate((0.5, 1.5, 3.0, 6.0, 12.0, 30.0), start=2):
        if a <= hi:
            return k
    return 7


def _even_sample(total: int, limit: int):
    if total <= limit:
        return list(range(total))
    last = total - 1
    idx = sorted({int(round(i * last / (limit - 1))) for i in range(limit)})
    return idx[:limit]


def encode_chunk(hands, max_hands=MAX_HANDS, max_actions=MAX_ACTIONS):
    at = np.zeros((max_hands, max_actions), np.int64)
    st = np.zeros((max_hands, max_actions), np.int64)
    he = np.zeros((max_hands, max_actions), np.int64)
    bk = np.zeros((max_hands, max_actions), np.int64)
    pf = np.zeros((max_hands, max_actions), np.int64)
    cont = np.zeros((max_hands, max_actions, CONT_DIM), np.float32)
    amask = np.zeros((max_hands, max_actions), np.bool_)
    hmask = np.zeros(max_hands, np.bool_)
    sel = _even_sample(len(hands), max_hands)
    for hi, si in enumerate(sel):
        hand = hands[si]
        if not isinstance(hand, dict):
            continue
        md = hand.get("metadata") or {}
        hero = _i(md.get("hero_seat"), 0)
        hstack = 0.0
        for p in (hand.get("players") or []):
            if _i(p.get("seat")) == hero:
                hstack = _f(p.get("starting_stack"))
        sc = (100.0 / hstack) if hstack > 0 else 1.0
        acts = (hand.get("actions") or [])[:max_actions]
        for ai, a in enumerate(acts):
            at[hi, ai] = _ACT.get(str(a.get("action_type", "")).lower(), 0)
            st[hi, ai] = _STREET.get(str(a.get("street", "")).lower(), 0)
            he[hi, ai] = 2 if (_i(a.get("actor_seat"), -1) == hero and hero > 0) else 1
            amt = _f(a.get("normalized_amount_bb")) * sc
            bk[hi, ai] = _bucket(amt)
            pb, pa = _f(a.get("pot_before")), _f(a.get("pot_after"))
            pflow = (pa - pb) / pb if pb > 1e-6 else 0.0
            pf[hi, ai] = 0 if pflow <= 1e-6 else (1 if pflow <= 1.0 else 2)
            cont[hi, ai, 0] = np.log1p(max(0.0, amt))
            cont[hi, ai, 1] = float(np.clip(pflow, 0.0, 5.0))
            amask[hi, ai] = True
        hmask[hi] = amask[hi].any()
    return at, st, he, bk, pf, cont, amask, hmask


def encode_all(chunk_list):
    """Pre-encode a list of chunks into stacked numpy arrays (CPU)."""
    enc = [encode_chunk(c) for c in chunk_list]
    return {
        "at": np.stack([e[0] for e in enc]), "st": np.stack([e[1] for e in enc]),
        "he": np.stack([e[2] for e in enc]), "bk": np.stack([e[3] for e in enc]),
        "pf": np.stack([e[4] for e in enc]), "cont": np.stack([e[5] for e in enc]),
        "amask": np.stack([e[6] for e in enc]), "hmask": np.stack([e[7] for e in enc]),
    }


def chunks_to_batch(chunks, device="cpu"):
    e = encode_all(chunks)
    t = lambda a, d: torch.tensor(a, dtype=d, device=device)
    return {"at": t(e["at"], torch.long), "st": t(e["st"], torch.long), "he": t(e["he"], torch.long),
            "bk": t(e["bk"], torch.long), "pf": t(e["pf"], torch.long),
            "cont": t(e["cont"], torch.float32), "amask": t(e["amask"], torch.bool),
            "hmask": t(e["hmask"], torch.bool)}


class _AttnPool(nn.Module):
    """Permutation-invariant pooling via a learnable query attending over the set."""
    def __init__(self, d, h, drop):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, h, dropout=drop, batch_first=True)

    def forward(self, x, key_padding_mask=None):
        b = x.shape[0]
        q = self.q.expand(b, 1, -1)
        if key_padding_mask is not None:
            allpad = key_padding_mask.all(dim=1)
            if allpad.any():  # fully-padded rows -> unmask slot 0 to avoid NaN; output ignored upstream
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[allpad, 0] = False
        out, _ = self.attn(q, x, x, key_padding_mask=key_padding_mask)
        return out.squeeze(1)


class SetTransformer(nn.Module):
    def __init__(self, d=64, heads=4, act_layers=2, hand_layers=1, dropout=0.1, ff_mult=2):
        super().__init__()
        self.e_at = nn.Embedding(N_ACT, d, padding_idx=0)
        self.e_st = nn.Embedding(N_STREET, d, padding_idx=0)
        self.e_he = nn.Embedding(N_HERO, d, padding_idx=0)
        self.e_bk = nn.Embedding(N_BUCKET, d, padding_idx=0)
        self.e_pf = nn.Embedding(N_POTFLOW, d)
        self.e_pos = nn.Embedding(MAX_ACTIONS, d)
        self.cont_proj = nn.Linear(CONT_DIM, d)
        self.in_norm = nn.LayerNorm(d)
        self.in_drop = nn.Dropout(dropout)
        al = nn.TransformerEncoderLayer(d, heads, d * ff_mult, dropout, activation="gelu", batch_first=True)
        self.action_enc = nn.TransformerEncoder(al, act_layers)
        self.action_pool = _AttnPool(d, heads, dropout)
        hl = nn.TransformerEncoderLayer(d, heads, d * ff_mult, dropout, activation="gelu", batch_first=True)
        self.hand_enc = nn.TransformerEncoder(hl, hand_layers)
        self.chunk_pool = _AttnPool(d, heads, dropout)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, 1))

    def forward(self, at, st, he, bk, pf, cont, amask, hmask):
        B, H, A = at.shape
        pos = torch.arange(A, device=at.device).unsqueeze(0).expand(B * H, A)
        f = lambda x: x.reshape(B * H, A)
        emb = (self.e_at(f(at)) + self.e_st(f(st)) + self.e_he(f(he)) + self.e_bk(f(bk))
               + self.e_pf(f(pf)) + self.e_pos(pos) + self.cont_proj(cont.reshape(B * H, A, -1)))
        emb = self.in_drop(self.in_norm(emb))
        kp = ~amask.reshape(B * H, A)
        enc = self.action_enc(emb, src_key_padding_mask=kp)
        hand = self.action_pool(enc, key_padding_mask=kp).reshape(B, H, -1)
        hand = hand.masked_fill(~hmask.unsqueeze(-1), 0.0)
        hkp = ~hmask
        henc = self.hand_enc(hand, src_key_padding_mask=hkp)
        chunk = self.chunk_pool(henc, key_padding_mask=hkp)
        return self.head(chunk).squeeze(-1)

    @torch.no_grad()
    def score(self, chunks, device="cpu", bs=64):
        self.eval()
        out = []
        for i in range(0, len(chunks), bs):
            b = chunks_to_batch(chunks[i:i + bs], device=device)
            logit = self(b["at"], b["st"], b["he"], b["bk"], b["pf"], b["cont"], b["amask"], b["hmask"])
            out.append(torch.sigmoid(logit).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)
