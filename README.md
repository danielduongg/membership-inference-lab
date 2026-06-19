# Membership-Inference Lab

[![CI](https://github.com/danielduongg/membership-inference-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/danielduongg/membership-inference-lab/actions)

**Does a trained model leak who was in its training set? Yes — and the more it overfits, the more it leaks.**

A small, fully reproducible testbed for the **privacy** side of ML security (the third in
a trio with inference-time and training-time attacks). It trains a target classifier,
attacks it with a **naive confidence** attack and a **shadow-model** (Shokri) attack to
infer training-set membership, and shows how leakage scales with overfitting and how
regularization trades it back for accuracy. Runs **offline, no GPU, no downloads**, and
spotlights **TPR @ 1% FPR** — the low-false-positive metric that actually reflects
privacy risk.

> **Headline finding.** Membership leakage tracks overfitting: as model complexity grows,
> attack AUC climbs **0.69 → 1.000**, and an unconstrained model lets the attacker
> re-identify **100% of training members at a 1% false-positive rate.** Regularization cuts
> leakage (AUC **1.000 → 0.758**, TPR@1%FPR **100% → 11%**) but costs **8 points** of test
> accuracy — a real privacy/utility tradeoff. Full write-up: [`report/REPORT.md`](report/REPORT.md).

## Why this exists

Knowing that a specific person's record was in a training set is itself a privacy breach,
and it's the foundation of training-data *extraction* from large models. This repo
reconstructs the classic chain — Shokri's shadow attack, Yeom's overfitting connection,
Carlini's "evaluate at low FPR" lesson — honestly and end-to-end, including the part
where the defense costs you accuracy.

## Results at a glance

**Leakage vs. model complexity** (RandomForest target; 800 members, 1000 non-members)

| max_depth | Test acc | Train–test gap | Attack AUC | TPR @ 1% FPR |
|---:|---:|---:|---:|---:|
| 3 | 26.7% | 27.3% | 0.688 | 6.6% |
| 5 | 32.8% | 48.6% | 0.831 | 20.7% |
| 8 | 37.3% | 62.7% | 0.984 | 73.1% |
| 12 | 37.5% | 62.5% | 1.000 | 100% |
| ∞ | 37.7% | 62.3% | 1.000 | 100% |

**Regularization as a defense**

| Target | Test acc (utility) | Attack AUC (leakage) | TPR @ 1% FPR |
|---|---:|---:|---:|
| Unregularized (`max_depth=None`) | 37.7% | 1.000 | 100% |
| Regularized (`max_depth=4`) | 29.5% | 0.758 | 11.4% |

<p align="center">
  <img src="results/figures/leakage_vs_complexity.png" width="49%">
  <img src="results/figures/log_roc.png" width="42%"><br>
  <img src="results/figures/defense.png" width="46%">
</p>

## Quickstart

```bash
pip install -r requirements.txt
python scripts/run_all.py        # attack demo -> overfit sweep -> defense
python tests/test_smoke.py
```

Outputs in `results/` (JSON/CSV) and `results/figures/`. Seeded (`SEED = 20260617`),
byte-reproducible across `PYTHONHASHSEED`.

## How it works

- **`src/data.py`** — hard synthetic task + disjoint member / non-member / shadow-pool split.
- **`src/target.py`** — RandomForest target; `max_depth` = the overfitting knob.
- **`src/attacks.py`** — `naive_attack` (true-class confidence) and `shadow_attack`
  (Shokri shadow models → logistic-regression attacker); metrics incl. `tpr_at_fpr`.
- **`scripts/`** — `01_attack_demo` → `02_overfit_sweep` → `03_defense` (+ `run_all`).

```
membership-inference-lab/
├── README.md
├── report/REPORT.md          # research write-up (public output)
├── requirements.txt
├── src/                      # data, target, attacks, plots
├── scripts/                  # 01_attack_demo → 02_overfit_sweep → 03_defense (+ run_all)
├── results/                  # attack_demo.json, overfit_sweep.csv, defense.json, figures/
└── tests/test_smoke.py
```

## Use a real model

The attacks only need a model's `predict_proba` plus member/non-member examples. Swap
`make_dataset()` + `train_target()` for any classifier and dataset (e.g. an MNIST/CIFAR
net, or a fine-tuned text classifier) and the same `naive_attack` / `shadow_attack` apply.

## Responsible use

Trained on synthetic records — no real personal data. A **defensive** demonstration of a
privacy vulnerability and how model choices affect it.

## References

- Shokri et al., *Membership Inference Attacks* (S&P 2017): https://arxiv.org/abs/1610.05820
- Carlini et al., *MI From First Principles / LiRA* (S&P 2022): https://arxiv.org/abs/2112.03570
- Yeom et al., *Privacy Risk & Overfitting* (CSF 2018): https://arxiv.org/abs/1709.01604
- Carlini et al., *Extracting Training Data from LLMs* (USENIX 2021): https://arxiv.org/abs/2012.07805
- Abadi et al., *Deep Learning with Differential Privacy* (CCS 2016): https://arxiv.org/abs/1607.00133

## License

MIT — see [`LICENSE`](LICENSE).
