# Preregistration Draft — OSF Standard (21-Item) Adapted to a CS/ML Benchmark

**Study:** A Multi-Metric, Leakage-Controlled Benchmark of Explainable AI Methods for Deep-Learning Network Intrusion Detection.
**Companion files:** `XAI_IDS_Thesis_Proposal.md` (full proposal), `XAI_IDS_Gap_Analysis.md` (literature foundation) — same directory.
**Target registry:** OSF Registries (https://osf.io/registries) — to be submitted **before** the full factorial runs (Phase P4).
**Template note.** This draft adapts the OSF Standard Pre-Data Collection Registration template (clinical/social-science flavor) to a computer-science/ML benchmark. "Sample size" = number of test instances/experimental cells; "data collection" = dataset preparation; "manipulated variables" = XAI method, DL architecture, dataset; "measured variables" = the four explanation-quality metric families; "participants" = none (no human subjects).

---

## A. Study Information

### 1. Title [Required]
> A Multi-Metric, Leakage-Controlled Benchmark of Explainable AI Methods for Deep-Learning Network Intrusion Detection.

### 2. Authors [Required]
| Name | Institution | Role | ORCID |
|------|-------------|------|-------|
| [Candidate] | [Institution] | PI (master's researcher) | [ ] |
| [Supervisor] | [Institution] | Co-PI (supervisor) | [ ] |

### 3. Research Questions [Required]
- **RQ1 (primary):** On a fixed DL-IDS (1D-CNN and a tabular Transformer) across three leakage-controlled datasets (CICIDS2017, UNSW-NB15, CICIoT2023), how do four XAI methods (SHAP, LIME, counterfactual, intrinsic reference) compare on explanation faithfulness, stability, generation cost, and a proxy of downstream triage utility — and how do rankings shift across DL architecture, dataset vintage, and attack family?
- **RQ2–RQ6 (sub-questions):** faithfulness (SQ1), stability (SQ2), cost (SQ3), proxy utility (SQ4), generalizability (SQ5) — as in `XAI_IDS_Thesis_Proposal.md` §4.2.

### 4. Hypotheses [Required]
- **H1 (confirmatory):** No single XAI method dominates across all four metric families. (Directional prediction: the best method on faithfulness ≠ the best on stability ≠ the best on cost ≠ the best on proxy utility, in at least 3 of the 4 pairwise comparisons.)
- **H2 (confirmatory):** SHAP and LIME stability (Kendall τ over bootstraps) is lower on UNSW-NB15 than on CICIDS2017 and CICIoT2023, consistent with multicollinearity-driven fragility (Vourganas 2026, arXiv:2605.22529).
- **H3 (exploratory):** Method rankings shift across DL architecture (CNN vs Transformer) and across attack families.

---

## B. Design Plan

### 5. Study Type [Required]
- [x] Experiment — **Factorial design: 4 (XAI method) × 2 (DL architecture) × 3 (dataset)**, repeated per attack family and per metric. Fully within-stimulus (no human participants); a computational benchmark.

### 6. Randomization [Optional]
- Randomization unit: **test-instance ordering** and **bootstrap resampling seeds** for stability metrics. Random number generator: NumPy `PCG64` with pre-specified, logged seeds. Not applicable to participant allocation (no participants).

### 7. Blinding [Optional]
- Not applicable (no human participants). The analyst-proxy metric uses a fixed, pre-registered reordering policy, so there is no scorer blinding; raw metric traces are logged to prevent post-hoc tuning.

### 8. Study Design / Conditions [Required]
- **Factor 1 — XAI method (4 levels):** SHAP (DeepSHAP/GradientSHAP, exact variant pre-specified); LIME; Counterfactual (DiCE primary, diffusion-CF fallback); Intrinsic reference (interpretable model).
- **Factor 2 — DL architecture (2 levels):** 1D-CNN; tabular Transformer (FT-Transformer). Both trained to a matched detection-performance budget (±1 pp macro-F1).
- **Factor 3 — Dataset (3 levels):** CICIDS2017; UNSW-NB15; CICIoT2023 — all leakage-controlled re-splits.
- **Baseline conditions for proxy utility:** (a) no-explanation ranking; (b) random-reorder ranking. Explanation-guided reordering compared against both.

---

## C. Sampling Plan

### 9. Existing Data [Required]
- [x] Data exist but have not been examined for this study's cells (Registration prior to computing the locked metric battery on the locked splits). Datasets are public (CICIDS2017, UNSW-NB15, CICIoT2023); leakage-controlled re-splits will be generated fresh under the pre-registered protocol. No preliminary analysis of the locked metric values has been performed.

### 10. Data Collection Procedures [Required]
- **Collection method:** Archival/public datasets + computational generation of explanations and metrics.
- **Collection instruments:** Python; libraries — `captum`/`shap`/`lime`/`dice-ml`/`torch`; custom metric implementations.
- **Collection location:** Single GPU workstation (institutional / owned).
- **Collection timeline:** Phase P3–P4 (months 6–15 of the 18–24 month plan).
- **Data collectors:** The candidate (PI).

### 11. Sample Size [Required]
- **Per dataset:** all instances in the pre-registered leakage-controlled test split (no sub-sampling for the locked primary cells; full test sets used). Estimated N per test split: CICIDS2017 ~ tens of thousands of flows; UNSW-NB15 ~ tens of thousands; CICIoT2023 ~ hundreds of thousands (exact counts locked after split generation, before metric computation).
- **Bootstrap resamples for stability:** B = 100 per cell.
- **Latency scaling subsamples for cost:** N ∈ {1k, 5k, 10k, 50k}.

### 12. Sample Size Rationale [Required]
- **Method:** Feasibility + benchmarking convention. The benchmark uses full pre-registered test splits (not a power-sized sample) because the goal is method ranking, not population inference. Friedman/Nemenyi is non-parametric and appropriate for ranking across methods using per-instance metric values as repeated measures.
- **Power consideration (secondary):** with per-instance metric values as the unit of analysis, the effective N per cell is large (thousands), giving high power to detect ranking differences; the risk is the opposite (over-powering → trivially significant differences), mitigated by reporting effect sizes and full distributions, not only p-values.

### 13. Stopping Rule [Required]
- [x] Stop when all pre-registered cells (4 × 2 × 3, all metric families, all attack families present in each dataset) have been computed once with the locked code and splits. No early stopping based on results. Optional stretch cells (LLM, drift) are not part of the stopping rule.

---

## D. Variables

### 14. Manipulated Variables [Required for experiments]
- **IV1 — XAI method:** 4 levels (SHAP, LIME, Counterfactual, Intrinsic). Operationalized via fixed library configs (locked in an appendix config file).
- **IV2 — DL architecture:** 2 levels (1D-CNN, FT-Transformer). Operationalized via fixed model specs; matched detection performance.
- **IV3 — Dataset:** 3 levels. Operationalized via the pre-registered leakage-controlled split protocol.

### 15. Measured Variables [Required]
- **DV family 1 — Faithfulness:** Deletion/Insertion AUC, Infidelity, Sensitivity-n, sufficiency (prediction preservation under top-k ablation), necessity (low-attribution flip Δ-prediction), descriptive accuracy, sparsity.
- **DV family 2 — Stability:** Kendall τ over bootstrap resamples; stability under prediction-preserving perturbation; Explanability Fragility Score (Vourganas 2026) on UNSW-NB15.
- **DV family 3 — Cost:** per-instance wall-clock latency; latency scaling curve over N; GPU memory.
- **DV family 4 — Proxy utility:** top-k attack-family recall and time-to-critical-alert under explanation-guided reordering vs no-explanation and random-reorder baselines.
- **Covariates / controls:** detection macro-F1, per-attack-family detection rate, AUC; SHAP variant (sensitivity); VIF/multicollinearity of features (measured, not silently cleaned).

### 16. Indices [Required]
- Each DV computed per test instance, then aggregated per (method × architecture × dataset × attack-family) cell as: median, IQR, 95% CI. Stability indices (Kendall τ) computed over B = 100 bootstraps. Latency computed as median over 5 repeats per N. No reverse scoring. Missing/failed CF explanations handled per §20.

---

## E. Analysis Plan

### 17. Statistical Models [Required]
- **For H1 (no single method dominates):** Friedman test across the 4 methods on each metric family (per architecture × dataset), followed by **post-hoc Nemenyi** multi-comparison with critical-distance diagrams. A method is "dominant" on a family if it ranks first and the Nemenyi test finds no significant difference from the next two methods — H1 is supported if dominance does not occur on all 4 families for any single method.
- **For H2 (SHAP/LIME stability degrades on UNSW-NB15):** paired comparison of Kendall τ (UNSW-NB15 vs {CICIDS2017, CICIoT2023}) for SHAP and LIME separately, via Wilcoxon signed-rank on per-instance stability scores; one-sided test in the predicted direction; Holm correction across the 4 comparisons.
- **For H3 (exploratory):** interaction analysis — does method ranking change across architecture and attack family? Reported as exploratory with no confirmatory inference.

### 18. Transformations [Optional]
- [ ] No transformations of raw metric values. Latency reported on original scale (seconds) and log-10 scale for the scaling curve. Kendall τ is already rank-based (no transform needed).

### 19. Inference Criteria [Required]
- **Significance level:** α = .05.
- **Multiple comparison correction:** Nemenyi (for H1 Friedman post-hoc); Holm (for H2's 4 comparisons). Exploratory analyses (H3) reported without confirmatory p-value claims.
- **Effect size reporting:** for Friedman, report average rank + critical distance; for Wilcoxon, report rank-biserial correlation r; report 95% CIs for all aggregated DVs.
- **Confidence interval:** 95% CI.
- **One-tailed/Two-tailed:** H2 is one-tailed (directional prediction of degradation); H1 is two-tailed (ranking equality). Justification: H2 has a clear directional theory (multicollinearity → instability).

### 20. Data Exclusion [Required]
- **Instance-level exclusion:** instances for which counterfactual generation fails after the pre-specified iteration budget are excluded from CF-metric aggregation **and the exclusion rate reported per cell** (not silently dropped); they remain in SHAP/LIME/intrinsic aggregations.
- **Split-level exclusion:** dataset records removed by the leakage-control protocol (duplicates, label collisions per Tolay 2026) are reported as counts, not silently dropped.
- **Outliers:** metric values > 3 IQR from the cell median are flagged and reported but **not removed** in the primary analysis; a sensitivity analysis reports results with them excluded.
- **Post-exclusion procedures:** report pre- and post-exclusion instance counts per cell; compare characteristics of excluded vs retained instances for CF.

### 21. Exploratory Analyses [Optional]
- SHAP-variant sensitivity (DeepSHAP vs GradientSHAP vs Integrated Gradients).
- Per-attack-family breakdown of all metrics.
- Leave-one-dataset-out stability of method rankings.
- Optional stretch cells (LLM-conversational; concept-drift "explanation half-life") — explicitly labeled exploratory and not part of confirmatory tests.

---

## F. Other

### Ethics Review [Optional]
- **IRB review status:** Not applicable (no human subjects). The "downstream triage utility" is a simulated/proxy metric computed from alert-rank reordering, not human data. A short IRB-not-needed memo will be filed at the institution.
- **Dual-use:** Defensive benchmark; no operational evasion guidance; responsible-disclosure framing for any fidelity finding that could aid evasion.

### Data Availability [Optional]
- **Will data be made public:** Yes. Open GitHub repo (code), OSF project (splits, raw metric traces, preregistration). Public datasets referenced, not redistributed where licenses forbid.

### Supplementary Materials [Optional]
- [x] Analysis code (open repo)
- [x] Raw metric traces (CSV/Parquet per cell)
- [x] Locked XAI-method config files (appendix)
- [x] Leakage-safe split-generation script (deterministic, seeded)
- [ ] Power analysis report (N/A — benchmark, not population inference; rationale in §12)

---

## Pre-Submission Checklist
- [x] All [Required] items completed.
- [x] Hypotheses (H1, H2) are clearly stated and testable.
- [x] Analysis methods correspond to hypotheses (Friedman/Nemenyi → H1; Wilcoxon → H2).
- [x] Exclusion criteria established before metric computation (§20).
- [x] Confirmatory (H1, H2) and exploratory (H3 + stretch) analyses distinguished.
- [x] IRB review status confirmed (Not applicable; memo to be filed).
- [ ] Preregistration platform selected (OSF Registries) — to be submitted before Phase P4.
