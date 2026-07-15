"""Sequence-net model + hero-context action tokenization for the depth-agnostic sequence detector.

The network is a small bidirectional GRU over the ordered hero-context action tokens of a chunk. It
captures the temporal / action-order signal that the aggregate feature models discard. Tokenization
mirrors the training pipeline: per action -> (action_type, street, is_hero, log_amount_bb_rescaled,
pot_growth_frac); amounts are rescaled so the hero stack is 100bb (kills the deep-stack shift).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

# runtime cuDNN on the serving host may mismatch the bundled version; native path is fine (CPU serve).
torch.backends.cudnn.enabled = False

ACT = {"fold": 1, "check": 2, "call": 3, "bet": 4, "raise": 5,
       "post": 6, "blind": 6, "small_blind": 6, "big_blind": 6, "ante": 6, "sb": 6, "bb": 6}
STREET = {"preflop": 0, "flop": 1, "turn": 2, "river": 3, "showdown": 3}
BOUNDARY = 7
MAXLEN = 512


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


def hand_tokens(hand):
    md = hand.get("metadata") or {}
    hero = _i(md.get("hero_seat"), 0)
    hero_stack = 0.0
    for p in (hand.get("players") or []):
        if _i(p.get("seat")) == hero:
            hero_stack = _f(p.get("starting_stack"))
    sc = (100.0 / hero_stack) if hero_stack > 0 else 1.0
    out = []
    for a in (hand.get("actions") or []):
        at = ACT.get(str(a.get("action_type", "")).lower(), 0)
        st = STREET.get(str(a.get("street", "")).lower(), 0)
        ishero = 1 if (_i(a.get("actor_seat"), -1) == hero and hero > 0) else 0
        amt = np.log1p(max(0.0, _f(a.get("normalized_amount_bb")) * sc))
        pb, pa = _f(a.get("pot_before")), _f(a.get("pot_after"))
        potfrac = float(np.clip((pa - pb) / pb if pb > 1e-6 else 0.0, 0.0, 5.0))
        out.append((at, st, ishero, amt, potfrac))
    return out


def chunk_tokens(hands):
    toks = []
    for h in hands:
        toks.extend(hand_tokens(h))
        toks.append((BOUNDARY, 0, 0, 0.0, 0.0))
    return toks[:MAXLEN]


def chunks_to_tensors(chunks):
    """List of chunks (each a list of hand dicts) -> (cat[B,L,3] long, cont[B,L,2] float, lens[B] long)."""
    seqs = [chunk_tokens(c) for c in chunks]
    n = len(seqs)
    lens = np.array([max(1, len(s)) for s in seqs], dtype=np.int64)
    cat = np.zeros((n, MAXLEN, 3), dtype=np.int64)
    cont = np.zeros((n, MAXLEN, 2), dtype=np.float32)
    for i, s in enumerate(seqs):
        for j, (at, st, ih, amt, pf) in enumerate(s):
            cat[i, j, 0] = at
            cat[i, j, 1] = st
            cat[i, j, 2] = ih
            cont[i, j, 0] = amt
            cont[i, j, 1] = pf
    return torch.tensor(cat), torch.tensor(cont), torch.tensor(lens)


class SeqNet(nn.Module):
    def __init__(self, h: int = 64):
        super().__init__()
        self.ea = nn.Embedding(8, 16)
        self.es = nn.Embedding(4, 4)
        self.eh = nn.Embedding(2, 2)
        self.gru = nn.GRU(16 + 4 + 2 + 2, h, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(2 * h, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, xc, xn, ln):
        e = torch.cat([self.ea(xc[:, :, 0]), self.es(xc[:, :, 1]), self.eh(xc[:, :, 2]), xn], dim=-1)
        packed = pack_padded_sequence(e, ln.cpu(), batch_first=True, enforce_sorted=False)
        _, hh = self.gru(packed)
        z = torch.cat([hh[0], hh[1]], dim=-1)
        return self.head(z).squeeze(-1)

    @torch.no_grad()
    def score(self, chunks):
        """Return per-chunk P(bot) from the sequence signal, in [0,1]."""
        if not chunks:
            return np.zeros(0, dtype=float)
        xc, xn, ln = chunks_to_tensors(chunks)
        self.eval()
        return torch.sigmoid(self(xc, xn, ln)).cpu().numpy().astype(float)
