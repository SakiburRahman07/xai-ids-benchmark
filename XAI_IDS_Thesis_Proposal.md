# Master's Thesis Proposal: A Multi-Metric, Leakage-Controlled Benchmark of Explainable AI Methods for Deep-Learning Network Intrusion Detection

**Candidate:** [Name to be completed]
**Program / Institution:** [Master's program / Institution to be completed]
**Supervisor:** [To be completed]
**Proposal date:** 2026-08-27
**Foundational literature review:** See companion file `XAI_IDS_Gap_Analysis.md` (same directory) for the full annotated bibliography, literature matrix, and gap ranking from which this proposal is derived.
**Preregistration draft:** See companion file `XAI_IDS_Preregistration_Draft.md`.
**AI-use disclosure.** AI-assisted research tools were used for the preparatory literature review and for structuring this proposal. All cited sources are web-verified arXiv IDs (see `XAI_IDS_Gap_Analysis.md`, §7).

---

## 1. Abstract

Deep-learning intrusion detection systems (DL-IDS) increasingly ship with an "explainable AI" (XAI) layer — almost always a SHAP or LIME plot — yet the field has no apples-to-apples evidence that these explanations are faithful, stable, cheap, or useful. This thesis conducts the first controlled, multi-metric benchmark of four XAI methods (SHAP, LIME, counterfactual, and an intrinsic reference model) on a fixed DL-IDS (1D-CNN and a tabular Transformer) across three leakage-controlled datasets (CICIDS2017, UNSW-NB15, CICIoT2023). Explanations are evaluated on four metric families — faithfulness (sufficiency/necessity), stability (Kendall τ over bootstraps), generation cost (latency), and a proxy of downstream triage utility (simulated rank-based alert prioritization) — with a Friedman + Nemenyi non-parametric analysis across the full 4 × 2 × 3 factorial. The study is pre-registered, fully scripted on a single workstation, uses only open data and open-source code, and requires no human-subjects/IRB approval. The intended contributions are (i) the first reproducible XAI-for-DL-IDS benchmark, (ii) evidence on whether SHAP/LIME dominance is justified, and (iii) a reusable open metric toolkit and method-selection recommendations by architecture, dataset, and attack family.

---

## 2. Problem Statement & Motivation

Network intrusion detection systems built on deep learning achieve high reported accuracy but are opaque, which blocks analyst trust and deployment in safety-critical Security Operations Centers (SOCs). The dominant response in the 2020–2026 literature is to attach a post-hoc XAI method — overwhelmingly SHAP or LIME — and report a feature-attribution plot alongside an accuracy figure (see `XAI_IDS_Gap_Analysis.md`, Themes B and the convergence row of §3).

This is the field's foundational methodological weakness. Three observations make the problem acute:

1. **"Explainability" claims are currently unfalsifiable.** Over 80% of empirical XAI-IDS papers report a SHAP/LIME plot without ever measuring whether the explanation reflects the model's reasoning. Neupane et al. (2022, arXiv:2207.06236) explicitly called for explanation-evaluation metrics in 2022; the call remains largely unanswered.
2. **When explanations are measured, they often look fragile.** Vourganas & Michala (2026, arXiv:2605.22529) prove that multicollinearity inflates SHAP/LIME attribution variance and propose an "Explanability Fragility Score." Tolay (2026, arXiv:2608.10349) shows TreeSHAP can take 700.8s vs 1.47s for XGBoost at 5,000 samples — an overlooked operational cost. Bouke et al. (2026, arXiv:2606.29797) show aggregate F1 can hide detection-rate collapse (F1 = 0.74 hiding DR = 0.48) and that SHAP fold-stability varies with protocol.
3. **No apples-to-apples comparison exists.** Only two head-to-head XAI comparisons appear in the 2020–2026 corpus: xNIDS vs EXP-SEC (Kumar 2026, arXiv:2607.12203) and a counterfactual-algorithm comparison (Galwaduge & Samarabandu 2025, arXiv:2507.17161). There is no study that fixes the DL-IDS and datasets and compares multiple XAI method families on a battery of explanation-quality metrics.

The practical risk: a misleading-but-confident SHAP plot can harm an analyst more than no explanation, and a 700s-per-batch explainer is undeployable. This thesis closes that gap.

---

## 3. Research Gap (from the gap analysis)

This proposal operationalizes **Gap 3 — "Methodological weaknesses (XAI monoculture + unevaluated explanation quality)"** — the gap ranked **#1 (strongest)** in `XAI_IDS_Gap_Analysis.md` (§5), selected because it is the field's foundational weakness, has the clearest evidence base, and the highest feasibility (fully scriptable, open-source, no IRB). The original gap text and research question are at `XAI_IDS_Gap_Analysis.md:142–148`.

**Scope decisions applied to the original RQ (documented for transparency):**
- The original RQ's "small analyst-triage task" is replaced by a **proxy/simulated triage metric** to preserve the no-IRB, fully-scriptable feasibility advantage (the reason Gap 3 ranked #1). A real human study is documented as an optional stretch goal only.
- The original 5th XAI method (LLM-conversational) is **deferred to a stretch goal** to keep the core reproducible on a single workstation with no paid API. The core uses 4 local, open-source XAI methods.
- Timeline is set to **18–24 months** (research master's / MPhil scope), allowing a thorough full-factorial benchmark plus write-up and journal submission.

---

## 4. Research Question, Sub-Questions & FINER Assessment

### 4.1 Primary Research Question
**On a fixed DL-IDS (1D-CNN and a tabular Transformer) across three leakage-controlled datasets (CICIDS2017, UNSW-NB15, CICIoT2023), how do four XAI methods (SHAP, LIME, counterfactual, and an intrinsic reference model) compare on explanation faithfulness, stability, generation cost, and a proxy of downstream triage utility — and how do the rankings shift across DL architecture, dataset vintage, and attack family?**

### 4.2 Sub-Questions
1. **Faithfulness (SQ1).** Which method's attributions best preserve the model's predictions under ablation of the top-k attributed features (sufficiency) and respond to decision-relevant features (necessity)?
2. **Stability (SQ2).** How stable are each method's attributions under prediction-preserving perturbation and across bootstrap resamples (Kendall τ), and does multicollinearity destabilize SHAP/LIME on UNSW-NB15 (per Vourganas 2026)?
3. **Cost (SQ3).** What is the per-instance explanation latency for each method, and how does it scale with sample size (extending Tolay 2026's 700.8s vs 1.47s finding to the DL-IDS setting)?
4. **Proxy utility (SQ4).** Under a simulated rank-based triage policy, does explanation-guided alert reordering improve top-k attack-family recall / time-to-critical-alert over a no-explanation baseline?
5. **Generalizability (SQ5).** Do method rankings shift across DL architecture (CNN vs Transformer), dataset vintage (2015/2017/2023), and attack family?

### 4.3 FINER Assessment

| Criterion | Score | Justification |
|-----------|-------|---------------|
| **F**easible | 5/5 | Fully scriptable on a single GPU workstation; all datasets public; all XAI methods open-source; no human subjects/IRB. |
| **I**nteresting | 4/5 | Addresses a genuine, repeatedly-flagged foundational puzzle (unfalsifiable explainability); not a settled question. |
| **N**ovel | 5/5 | No apples-to-apples, multi-metric, leakage-controlled XAI benchmark for DL-IDS exists in the 2020–2026 corpus. |
| **E**thical | 5/5 | No human subjects; defensive framing only; low dual-use risk (benchmark, no operational evasion guidance). |
| **R**elevant | 5/5 | Directly determines whether deployed XAI-IDS explanations can be trusted; informs method selection. |
| **Average** | **4.8/5** | Above the 3.0 threshold; no criterion below 2. |

---

## 5. Methodology

### 5.1 Datasets (3; leakage-controlled re-splits)
| Dataset | Vintage | Role | Leakage control |
|---|---|---|---|
| CICIDS2017 | 2017 | Canonical baseline (most-used in the literature) | Hash-based exact-feature dedup + deterministic hash-level partition (Tolay 2026, arXiv:2608.10349); remove label collisions. |
| UNSW-NB15 | 2015 | Multicollinearity stress test (Vourganas 2026) | Fold-local splits; VIF/correlation pruning reported as a measured covariate, not applied as silent cleaning. |
| CICIoT2023 | 2023 | Modern IoT, recent attack mix | Leakage-safe 39-feature-hash pipeline (Tolay 2026). |

All splits are **temporal where timestamps exist** and otherwise deterministic hash-based, with validation-only checkpoint selection and threshold calibration (Bouke 2026, arXiv:2606.29797; Guerra 2026, arXiv:2608.01454). Per-attack-family stratification is reported, not just aggregate F1.

### 5.2 DL Architectures (2; trained to a fixed detection-performance budget)
- **1D-CNN** over tabular flow features (the most common DL-IDS architecture in the literature).
- **Tabular Transformer** (e.g., FT-Transformer or comparable).

Both models are trained to a **pre-specified, matched detection-performance budget** (e.g., within ±1 pp macro-F1 on a held-out validation split) so that explanation comparison is not confounded by one detector being materially more accurate than the other. Detection metrics (accuracy, macro-F1, per-attack-family detection rate, AUC) are reported in full.

### 5.3 XAI Methods (4 core; 1–2 stretch)
| Method | Family | Implementation | Notes |
|---|---|---|---|
| **SHAP** | post-hoc, additive | DeepSHAP/GradientSHAP (exact variant pre-specified; Integrated Gradients as sensitivity) | Approximation variance reported. |
| **LIME** | post-hoc, local | Canonical LIME for tabular data | Kernel/sampling settings fixed and reported. |
| **Counterfactual** | post-hoc, actionable | DiCE (primary); Galwaduge 2025 diffusion-CF (fallback) | Coverage reported; multiple CF algorithms. |
| **Intrinsic reference** | intrinsic | Interpretable model (e.g., decision tree / attention-with-fidelity) | Bounds post-hoc vs intrinsic. |
| *Stretch:* LLM-conversational | generative | Local/open LLM only | Deferred; gated on time/API budget. |
| *Stretch:* Grad-CAM / attention-fidelity | post-hoc | — | Optional 5th local method if LLM unavailable. |

### 5.4 Metrics (4 families)
1. **Faithfulness.** Deletion/Insertion (area under prediction-vs-features-removed curve), Infidelity (Yeh et al.), Sensitivity-n, sufficiency (prediction preservation under top-k ablation), necessity (low-attribution flip should not change prediction), descriptive accuracy, sparsity.
2. **Stability.** Kendall τ rank correlation over bootstrap resamples; stability under prediction-preserving perturbation; Explanability Fragility Score (Vourganas 2026) on UNSW-NB15.
3. **Cost.** Per-instance explanation latency (wall-clock); scaling curve over N ∈ {1k, 5k, 10k, 50k}; GPU memory.
4. **Proxy utility (simulated triage).** Given a ranked alert queue, does explanation-guided reordering improve top-k attack-family recall and time-to-critical-alert vs (a) no-explanation and (b) random-reorder baselines? This is a *proxy* for a human triage study, designed to require no IRB.

### 5.5 Statistical Analysis Plan
- **Design:** full factorial 4 (method) × 2 (architecture) × 3 (dataset), repeated per attack family and per metric.
- **Primary test:** Friedman test (non-parametric rank comparison across the 4 methods per metric), followed by **post-hoc Nemenyi** critical-distance diagrams — the standard for ML-benchmark comparisons.
- **Effect sizes & uncertainty:** report full metric distributions (not only point estimates) with 95% CIs.
- **Pre-registered hypotheses** (see §10 and the preregistration draft):
  - **H1:** No single XAI method dominates across all four metric families (faithfulness, stability, cost, proxy utility).
  - **H2:** SHAP and LIME stability (Kendall τ) degrades on UNSW-NB15 relative to CICIDS2017/CICIoT2023, consistent with multicollinearity-driven fragility (Vourganas 2026).
- **Sensitivity:** leave-one-dataset-out; per-attack-family breakdown; SHAP-variant sensitivity (DeepSHAP vs GradientSHAP vs Integrated Gradients).

### 5.6 Reproducibility
- Open GitHub repository with pinned random seeds, containerized environment (Docker/Conda), public datasets, and logged raw metric traces (CSV/Parquet) for every cell.
- Pre-registered on OSF/AsPredicted **before** the full factorial runs (see companion `XAI_IDS_Preregistration_Draft.md`).
- A short "IRB-not-needed" memo documents the proxy (non-human) design and dual-use defensive framing.

---

## 6. Timeline (18–24 months)

| Phase | Months | Milestones |
|---|---|---|
| **P1 — Setup** | 1–2 | RQ finalized; preregistration submitted; repo + dataset pipelines; IRB-not-needed memo. |
| **P2 — Models** | 3–5 | 1D-CNN + Transformer trained to matched performance budget; detection metrics validated. |
| **P3 — XAI + metrics** | 6–10 | 4 XAI methods + 4 metric families implemented; pilot on CICIDS2017. |
| **P4 — Full factorial** | 11–15 | All 4 × 2 × 3 cells run; statistical analysis (Friedman/Nemenyi). |
| **P5 — Proxy utility + stretch** | 16–18 | Simulated triage experiment; optional LLM/Grad-CAM stretch if budget allows. |
| **P6 — Write-up** | 19–24 | Thesis writing; journal submission; defense preparation. |

---

## 7. Expected Contributions

1. **First reproducible, multi-metric, leakage-controlled benchmark** of XAI methods for DL-IDS — directly answering Neupane et al.'s (2022) unanswered call for explanation-evaluation metrics.
2. **Evidence on whether SHAP/LIME dominance is justified** on faithfulness, stability, and cost — or whether it is a methodological default.
3. **A reusable open artifact**: benchmark code, leakage-safe dataset splits, and a metric toolkit that downstream researchers can extend.
4. **Method-selection recommendations** by DL architecture, dataset vintage, and attack family — actionable for SOC practitioners.

---

## 8. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| **Detection-accuracy confound** (if CNN and Transformer differ in accuracy, explanation comparison is unfair) | Train both to a pre-specified matched performance budget (±1 pp macro-F1); report full detection metrics. |
| **SHAP-on-DL approximation variance** (DeepSHAP vs GradientSHAP vs IG can disagree) | Pre-specify the exact SHAP variant; report all three as a sensitivity analysis. |
| **Counterfactual coverage gaps** (CF generation can fail for some instances) | Report coverage per cell; use multiple CF algorithms; treat uncovered instances explicitly. |
| **Scope creep** (LLM method, human study) | Both deferred to gated stretch goals; core scope locked in preregistration. |
| **Dual-use** (explanations could aid evasion) | Defensive framing only; no operational evasion guidance in outputs. |
| **Dataset realism** (benchmarks ≠ operational traffic) | Acknowledged limitation; the leakage-control and per-attack-family reporting narrow (not eliminate) the gap; transfer to operational traffic flagged as future work. |

---

## 9. Ethics & Dual-Use

No human subjects are involved (the triage task is a simulated/proxy metric), so no IRB approval is required; a short IRB-not-needed memo will document this. The study is a defensive benchmark. To avoid dual-use risk, outputs will not include operational evasion guidance, and any fidelity findings that could inform adversary evasion will be framed defensively and reported with a responsible-disclosure note.

---

## 10. Preregistration Plan (summary)

A full preregistration draft following the OSF Standard 21-Item template (adapted to a CS/ML benchmark) is provided in the companion file `XAI_IDS_Preregistration_Draft.md`. Key locked elements: the primary RQ and 5 sub-questions; hypotheses H1 (no single method dominates all 4 metric families) and H2 (SHAP/LIME stability degrades on UNSW-NB15); the 4 × 2 × 3 factorial design; the Friedman + Nemenyi analysis; the four metric families and their operational definitions; exclusion criteria (e.g., instances where CF generation fails); and the sensitivity analyses. The preregistration will be submitted to OSF Registries **before** the full factorial runs in Phase P4.

---

## 11. References (foundational; full annotated bibliography in `XAI_IDS_Gap_Analysis.md` §2, §7)

- Bouke, M. A., et al. (2026). Multi-Level Distributional Entropy for Explainable Network IDS. arXiv:2606.29797.
- Galwaduge, V., & Samarabandu, J. (2025). Tabular Diffusion based Actionable Counterfactual Explanations for NIDS. arXiv:2507.17161.
- Guerra, L., et al. (2026). How Benchmarks and Evaluation Protocols Shape Conclusions in Provenance-Based IDS. NDSS 2027. arXiv:2608.01454.
- Hakim, M. A., et al. (2026). Cross-Domain Generalization Failure in Lightweight IDS for IIoT. arXiv:2607.00553.
- Himmelhuber, A., et al. (2022). Detection, Explanation and Filtering of Cyber Attacks Combining Symbolic and Sub-Symbolic Methods. IEEE SSCI 2022. arXiv:2212.13991.
- Kong, W., et al. (2026). Large Language Models as Explainable Cyberattack Detectors for Energy ICS. arXiv:2604.26079.
- Kumar, A., & Thing, V. L. L. (2026). EXP-SEC. arXiv:2607.12203.
- Neupane, J., et al. (2022). Explainable Intrusion Detection Systems (X-IDS): A Survey. arXiv:2207.06236.
- Nguyen, C. C., et al. (2026). Conversational versus Dashboard XAI for UAV IDS. arXiv:2608.10434.
- Tolay, A. (2026). Beyond Detection Accuracy: Explanation Cost, Stability, and Utility for IoT IDS. arXiv:2608.10349.
- Vourganas, I. J., & Michala, A. L. (2026). Stabilising Explainability Fragility in Cybersecurity AI. arXiv:2605.22529.

---

## 12. Appendix — Optional Stretch Goals (gated)

- **S1 (LLM-conversational XAI):** add as the 5th method if a local/open LLM is available and time permits; compare faithfulness vs the 4 core methods (addresses Gap 8 in `XAI_IDS_Gap_Analysis.md`).
- **S2 (small human analyst study):** a within-subjects tier-1 SOC analyst triage experiment replacing the proxy metric — only if IRB approval and analyst recruitment are feasible. Documented as a Phase-2 extension; not part of the core scope.
- **S3 (concept drift / "explanation half-life"):** extend the benchmark to a temporal-drift stream to measure joint fidelity–stability–cost degradation (addresses Gap 5 in `XAI_IDS_Gap_Analysis.md`). Builds naturally on the core apparatus.
