# poker44-sequence-ensemble

Bot-detection model for the Poker44 subnet (SN126). Combines a recent-regime gradient-boosted
model fleet plus two public reference models (the aggregate-feature signal) with a bidirectional-GRU
sequence model over the hero-context action order. The sequence signal is decorrelated from the
feature signal (it reads the ordered actions the feature models collapse), so blending the two lifts
ranking quality: `final = sigmoid(0.6 * feature_ensemble_z + 0.4 * sequence_z)`.

Trained only on the public Poker44 benchmark releases (`api.poker44.net/api/v1/benchmark`). No
validator-only evaluation data is used.

- `p44seq/detector.py` — ensemble + sequence blend
- `p44seq/seqnet.py` — GRU sequence model + hero-context action tokenization
- `neurons/miner_seq.py` — miner neuron
