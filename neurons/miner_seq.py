"""Poker44 SN126 miner — sequence-ensemble bot detector.

Serves P(focus player is a bot) per chunk using the p44seq detector: the recent-regime
gradient-boosted fleet plus two public reference models blended with a bidirectional-GRU sequence
model over the hero-context action order. Independent implementation kept separate from the baseline
gradient-boosted detector and the depth-matched detector.
"""
import os
import sys
import time
from pathlib import Path
from typing import Tuple

import bittensor as bt

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from poker44.base.miner import BaseMinerNeuron
from poker44.utils.model_manifest import (
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
)
from poker44.validator.synapse import DetectionSynapse
from p44seq.detector import Detector


def _head_commit(root: Path) -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def _declared_implementation(root: Path):
    """Files that constitute this miner's implementation (hashed into identity)."""
    return [
        root / "p44seq" / "detector.py",
        root / "p44seq" / "seqnet.py",
        root / "neurons" / "miner_seq.py",
    ]


class Miner(BaseMinerNeuron):
    def __init__(self, config=None):
        super().__init__(config=config)
        self.detector = Detector(base=None)
        bt.logging.info("🃏 p44seq sequence-ensemble detector ready")

        self.model_manifest = build_local_model_manifest(
            repo_root=_ROOT,
            implementation_files=_declared_implementation(_ROOT),
            defaults={
                "model_name": os.getenv("POKER44_MODEL_NAME", "poker44-sequence-ensemble"),
                "model_version": os.getenv("POKER44_MODEL_VERSION", "1"),
                "framework": "gradient-boosting-plus-gru",
                "license": "MIT",
                "repo_url": os.getenv("POKER44_MODEL_REPO_URL", ""),
                "repo_commit": os.getenv("POKER44_MODEL_REPO_COMMIT", _head_commit(_ROOT)),
                "notes": (
                    "Gradient-boosted fleet plus public reference models blended with a "
                    "bidirectional-GRU sequence model over hero-context action order."
                ),
                "open_source": True,
                "inference_mode": "remote",
                "training_data_statement": (
                    "Trained only on the public Poker44 benchmark releases "
                    "(api.poker44.net/api/v1/benchmark). No validator-only eval data used."
                ),
                "training_data_sources": ["poker44-public-benchmark"],
                "private_data_attestation": (
                    "This model does not train on validator-only live evaluation data."
                ),
                "data_attestation": "Public Poker44 benchmark releases only.",
            },
        )
        self.manifest_compliance = evaluate_manifest_compliance(self.model_manifest)
        self.manifest_digest = manifest_digest(self.model_manifest)
        bt.logging.info(
            "manifest status=%s missing=%s repo=%s commit=%s" % (
                self.manifest_compliance["status"],
                self.manifest_compliance["missing_fields"],
                self.model_manifest.get("repo_url"),
                self.model_manifest.get("repo_commit"),
            )
        )

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        groups = synapse.chunks or []
        started = time.perf_counter()
        try:
            risk = self.detector.score_chunks(groups)
            if len(risk) != len(groups):
                raise ValueError("score/chunk length mismatch")
        except Exception as exc:
            bt.logging.warning("scoring failed (%s); neutral fallback" % exc)
            risk = [0.5] * len(groups)
        risk = [float(min(1.0, max(0.0, r))) for r in risk]
        synapse.risk_scores = risk
        synapse.predictions = [r >= 0.5 for r in risk]
        synapse.model_manifest = dict(self.model_manifest)
        elapsed = (time.perf_counter() - started) * 1000.0
        who = getattr(getattr(synapse, "dendrite", None), "hotkey", None)
        bt.logging.info("scored %d chunks in %.0fms (mean=%.3f) from %s" % (
            len(groups), elapsed, (sum(risk) / max(len(risk), 1)), str(who)[:16] if who else "?"))
        return synapse

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        return self.caller_priority(synapse)


if __name__ == "__main__":
    with Miner() as node:
        bt.logging.info("Poker44 sequence-ensemble miner running.")
        while True:
            try:
                mg = node.metagraph
                bt.logging.info("UID %s | incentive %.5f | stake %.2f | block %s" % (
                    node.uid, mg.I[node.uid], mg.S[node.uid], node.block))
            except Exception as exc:
                bt.logging.warning("status log error: %s" % exc)
            time.sleep(120)
