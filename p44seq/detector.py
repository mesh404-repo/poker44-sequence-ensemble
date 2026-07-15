"""p44seq.Detector — sequence-ensemble challenger scorer.

Independent implementation combining the recent-regime gradient-boosted fleet plus two public
reference models (the aggregate-feature signal) with a bidirectional-GRU sequence model over the
hero-context action order. The sequence signal is decorrelated from the feature signal (it reads the
ordered actions the feature models collapse), so blending the two lifts ranking quality. Blend:
final = sigmoid(0.6 * feature_ensemble_z + 0.4 * sequence_z), with fixed reference stats for
cross-query consistency. Kept deliberately separate from the baseline and depth-matched detectors.
"""
from __future__ import annotations

import os
import json
import sys
from typing import Any, Dict, List

import numpy as np

from p44seq.seqnet import SeqNet

_ENS = os.environ.get("POKER44_ENSEMBLE_DIR", "/root/poker44-heroprofiler/ensemble")
_TRAVIS = os.environ.get("POKER44_TRAVIS_DIR", "/root/poker44-heroprofiler/travis")

FLEET = ("lgb_new", "xgb_new", "cat_new", "et_new", "rf_new")

# Feature-component weights (same transfer weighting as the baseline ensemble); the sequence signal
# is added on top as a separate, decorrelated term.
WEIGHTS = {
    "lgb_new": 1.10, "xgb_new": 1.25, "cat_new": 1.00, "et_new": 0.85, "rf_new": 0.65,
    "uid174": 1.00, "uid208": 1.35,
}
_WSUM = sum(WEIGHTS.values())

_M174, _S174 = 0.494, 0.197
_M208, _S208 = 0.311, 0.344
# sequence-net reference stats over the public benchmark (fixed, for cross-query consistency)
_MSEQ, _SSEQ = 0.6500, 0.3961
# blend weights (validated held-out: sigmoid(0.6*feat_z + 0.4*seq_z) = +0.048 reward over features)
_W_FEAT, _W_SEQ = 0.6, 0.4


class Detector:
    """Sequence-ensemble detector over the fleet + public references + GRU sequence model."""

    def __init__(self, base=None):
        import joblib
        import torch

        for p in (os.path.join(_ENS, "uid174"), _TRAVIS):
            if p not in sys.path:
                sys.path.insert(0, p)
        from p44bot.features import extract_features
        from poker44_model.features import chunk_features as cf174, FEATURE_NAMES as fn174
        from poker44_ml.inference import Poker44Model

        self._extract = extract_features
        self._cf174, self._fn174 = cf174, fn174
        self.base = base  # unused; interface parity

        sub = os.environ.get("POKER44_DET_SUBDIR", "v4")
        d = os.path.join(_ENS, sub)
        self.fleet = {}
        for nm in FLEET:
            obj = joblib.load(os.path.join(d, "v4_%s.joblib" % nm))
            self.fleet[nm] = (obj["model"], obj["keys"])
        self.ref = json.load(open(os.path.join(d, "v4_ref_stats.json")))
        self.m174 = joblib.load(os.path.join(_ENS, "uid174", "poker44_model", "model.joblib"))
        self.m208 = Poker44Model(os.path.join(_ENS, "uid208_v112.joblib"))

        seq_path = os.environ.get("POKER44_SEQ_WEIGHTS",
                                  os.path.join(os.path.dirname(__file__), "seq_net.pt"))
        self.seq = SeqNet()
        self.seq.load_state_dict(torch.load(seq_path, map_location="cpu"))
        self.seq.eval()

    def _feature_z(self, chunks: List[List[Dict[str, Any]]]) -> np.ndarray:
        feats = [self._extract(c) for c in chunks]
        base_keys = self.fleet[FLEET[0]][1]
        X0 = np.array([[float(f.get(k, 0.0)) for k in base_keys] for f in feats], dtype=float)
        zsum = np.zeros(len(chunks))
        for nm in FLEET:
            model, keys = self.fleet[nm]
            X = X0 if keys == base_keys else np.array(
                [[float(f.get(k, 0.0)) for k in keys] for f in feats], dtype=float)
            p = model.predict_proba(X)[:, 1]
            r = self.ref[nm]
            zsum += WEIGHTS[nm] * (p - r["mean"]) / max(r["std"], 1e-9)
        X174 = np.array([[float(self._cf174(c).get(k, 0.0)) for k in self._fn174]
                         for c in chunks], dtype=float)
        zsum += WEIGHTS["uid174"] * (self.m174.predict_proba(X174)[:, 1] - _M174) / _S174
        p208 = np.asarray(self.m208.predict_chunk_scores(chunks), dtype=float)
        zsum += WEIGHTS["uid208"] * (p208 - _M208) / _S208
        return zsum / _WSUM

    def score_chunks(self, chunks: List[List[Dict[str, Any]]]) -> List[float]:
        if not chunks:
            return []
        z_feat = self._feature_z(chunks)
        seq_p = self.seq.score(chunks)
        z_seq = (seq_p - _MSEQ) / _SSEQ
        z = _W_FEAT * z_feat + _W_SEQ * z_seq
        z = np.clip(z, -40.0, 40.0)
        out = 1.0 / (1.0 + np.exp(-z))
        return [float(min(1.0, max(0.0, v))) for v in out]
