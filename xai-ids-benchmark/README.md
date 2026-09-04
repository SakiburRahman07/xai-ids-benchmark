# xai-ids-benchmark

A multi-metric, leakage-controlled benchmark of **Explainable AI (XAI) methods for Deep-Learning Network Intrusion Detection (DL-IDS)** — the implementation of the master's thesis proposal.

> **Companion documents** (place next to this repo, or see the thesis package):
> - `XAI_IDS_Gap_Analysis.md` — foundational literature review + gap ranking
> - `XAI_IDS_Thesis_Proposal.md` — the thesis proposal this code implements
> - `XAI_IDS_Preregistration_Draft.md` — OSF 21-item preregistration (hypotheses H1, H2)

This is the **#1-ranked gap** from the gap analysis: the field's "explainability" claims are largely unfalsifiable (SHAP/LIME dominate, explanation fidelity is almost never measured). This benchmark is the first apples-to-apples, multi-metric, leakage-controlled comparison.

## What it does

Runs a full factorial **4 (XAI) × 2 (DL) × 3 (dataset)** and evaluates each explanation on four metric families:

| Factor | Levels |
|---|---|
| **XAI method** | SHAP (Captum), LIME, Counterfactual (DiCE), Intrinsic reference (decision tree) |
| **DL architecture** | 1D-CNN, FT-Transformer (trained to a matched detection-performance budget) |
| **Dataset** | CICIDS2017 (2017), UNSW-NB15 (2015), CICIoT2023 (2023) — all leakage-controlled re-splits |

| Metric family | Metrics |
|---|---|
| **Faithfulness** | Deletion/Insertion AUC, Infidelity, Sufficiency, Necessity, Sparsity |
| **Stability** | Kendall τ over bootstraps, perturbation stability, Fragility Score (Vourganas 2026) |
| **Cost** | per-instance latency, scaling curve, GPU memory |
| **Proxy utility** | simulated rank-based triage (top-k attack-family recall, time-to-critical-alert) — *no human subjects, no IRB* |

**Analysis:** Friedman + Nemenyi critical-distance diagrams (H1: no single method dominates all 4 families); Wilcoxon (H2: SHAP/LIME stability degrades on UNSW-NB15).

## Quick start

```bash
pip install -r requirements.txt
# Smoke run on tiny synthetic data (no real datasets needed):
python tests/test_smoke.py
```

## Prepare real datasets

All three datasets require a **one-time manual download** (UNB / UNSW host them behind a consent form):

1. Download the zips from the official pages (links in `config/datasets.yaml`).
2. Extract CSVs into:
   ```
   data_raw/cicids2017/*.csv
   data_raw/unsw_nb15/UNSW-NB15_*.csv
   data_raw/ciciot2023/*.csv
   ```
3. Verify: `python scripts/prepare_data.py --base .`

(Kaggle mirrors work too; the Colab notebook shows the `kaggle` API path.)

## Run the benchmark

```bash
# Smoke run (subsample to 2k rows, 20 epochs — minutes on a GPU):
python scripts/run_benchmark.py --quick

# Full factorial (hours on a T4/A100):
python scripts/run_benchmark.py

# Subset:
python scripts/run_benchmark.py --datasets cicids2017 --models cnn1d --methods shap lime
```

Outputs land in `results/`:
- `raw_results.json` — per-cell raw metric traces
- `tidy_metrics.csv` — flat (dataset, model, method, metric_family, metric, value) table
- `friedman_nemenyi.json` — Friedman stats, average ranks, critical distances, pairwise decisions
- `h2_wilcoxon.json` — H2 stability-degradation test
- `cd_diagram.png` — critical distance diagram (rendered from the notebook)

## Run in Google Colab

Open `notebooks/run_in_colab.ipynb` in Colab (set runtime to GPU). The `GITHUB_REPO` cell is pre-set to `https://github.com/SakiburRahman07/xai-ids-benchmark.git`; run top-to-bottom. The notebook:

1. Clones your repo,
2. Installs deps (skips torch to avoid CUDA mismatch),
3. Guides dataset prep (Google Drive mount or Kaggle API),
4. Runs the benchmark (`--quick` by default; toggle off for the full factorial),
5. Renders the Friedman/Nemenyi CD diagram and the H2 Wilcoxon results,
6. Zips `results/` for download.

## Repository layout

```
xai-ids-benchmark/
├── config/              # YAML configs (datasets, models, xai, metrics)
├── src/xai_ids_benchmark/
│   ├── data/            # loaders, leakage-controlled splits, preprocessing
│   ├── models/          # CNN1D, FT-Transformer
│   ├── xai/             # SHAP, LIME, Counterfactual, Intrinsic wrappers
│   ├── metrics/         # faithfulness, stability, cost, proxy_utility
│   ├── analysis/        # Friedman/Nemenyi stats + CD plot
│   ├── runner.py        # full-factorial orchestrator
│   └── utils.py         # seeds, config loading, logging
├── scripts/             # run_benchmark.py, prepare_data.py, build_notebook.py
├── notebooks/           # run_in_colab.ipynb
├── tests/               # test_smoke.py (synthetic data, no downloads)
└── requirements.txt
```

## Reproducibility

- Pinned `requirements.txt`; all seeds fixed (`random_state = 20260827`).
- Leakage-controlled splits (Tolay 2026 / Bouke 2026): hash dedup + deterministic hash-level partition + label-collision removal; temporal where timestamps exist.
- Models trained to a **matched detection-performance budget** (±1 pp macro-F1) so explanation comparison is fair.
- Raw metric traces logged for every cell; preregistration (H1, H2) locks the analysis before the full factorial.

## Push to GitHub

```bash
cd D:\CyberSecurityConference\xai-ids-benchmark
git init
git add .
git commit -m "Initial: XAI-IDS benchmark (4x2x3 factorial, leakage-controlled)"
# Create an empty repo on GitHub, then:
git remote add origin https://github.com/SakiburRahman07/xai-ids-benchmark.git
git branch -M main
git push -u origin main
```

The Colab notebook (`notebooks/run_in_colab.ipynb`) is pre-configured to clone from `https://github.com/SakiburRahman07/xai-ids-benchmark.git` and auto-detects whether the code is at the repo root or nested under `xai-ids-benchmark/`.

## Scope & ethics

- **No human subjects** — the triage task is a simulated/proxy metric, so no IRB is required. A short IRB-not-needed memo is recommended at your institution.
- **Dual-use:** defensive benchmark only; no operational evasion guidance in outputs. Any fidelity finding that could aid evasion is framed defensively.
- **Stretch goals** (gated off by default; see proposal §12): LLM-conversational XAI (5th method), a small real-analyst triage study, concept-drift "explanation half-life".

## Limitations

- arXiv-heavy literature base (see `XAI_IDS_Gap_Analysis.md` limitations).
- Benchmark datasets ≠ operational traffic; findings generalize to the benchmark, with transfer to operational traffic flagged as future work.
- SHAP-on-DL uses gradient-based approximations; SHAP variant (GradientShap / Integrated Gradients / DeepShap) is pre-specified and reported as a sensitivity.
