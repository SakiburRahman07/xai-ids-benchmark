# Explainable AI-based Deep Learning for Network Intrusion Detection: A Focused Literature Review and Research-Gap Analysis for a Master's Project

**Topic.** Explainable Artificial Intelligence (XAI) applied to Deep-Learning (DL) Network Intrusion Detection Systems (IDS).
**Scope.** Global; literature 2020–2026 (pre-2020 cited only as foundational background); narrative synthesis (SWiM-style).
**Purpose.** Identify research gaps suitable for a strong, feasible master's-level research project.
**Method.** deep-research skill, `lit-review` mode (bibliography + source-verification + synthesis agents).
**AI-use disclosure.** AI-assisted research tools were used for literature search and synthesis; all cited sources were web-verified via the arXiv API.

> **Note on delivery.** This file was written to the session's writable plans directory (`C:\Users\USER\.local\share\opencode\plans\`) because a permission rule blocked writes to the requested workspace path `D:\CyberSecurityConference\`. Copy/move this file to `D:\CyberSecurityConference\XAI_IDS_Gap_Analysis.md` to place it where originally requested.

---

## 1. Search Strategy, Scope & Limitations

**Databases / sources.** arXiv API (primary, full text); Semantic Scholar API (attempted — returned HTTP 429, rate-limited; not used); Google Scholar (via arXiv mirrors). arXiv coverage is strong for this CS/security field because most relevant IEEE/ACM/workshop papers are mirrored as preprints (several entries below carry a `journal_ref`/DOI confirming peer-reviewed publication, e.g., *IEEE Access*, *IEEE ICC*, *IEEE SoutheastCon*, *IEEE SSCI*, NDSS).

**Boolean strategy.** (`"explainable" OR interpretable OR XAI OR SHAP OR LIME OR Grad-CAM OR attention OR counterfactual`) AND (`"deep learning" OR CNN OR LSTM OR Transformer OR "neural network" OR GNN`) AND (`"intrusion detection" OR IDS OR NIDS OR "network intrusion"`). Two targeted supplementary queries: `"intrusion detection" AND "counterfactual"`; `"graph neural network" AND "intrusion detection" AND (explainable OR interpretable)`.

**Inclusion.** 2020–2026; English; directly addresses XAI for DL/ML IDS (surveys, empirical studies, benchmarks, human studies). **Exclusion.** Non-IDS XAI; pure DL-IDS with no explainability component; pre-2020 unless seminal.

**PRISMA-lite flow.** Records identified (arXiv, three queries): ~60; after de-duplication: ~52; title/abstract screened: ~52; full-text assessed: ~45; included in synthesis: **41**. (Flow counts are approximate because the arXiv API returns relevance/date-ranked lists, not an exhaustive database census; see Limitations.)

**Limitations of the review process.** (1) The corpus is **arXiv-heavy**; non-mirrored IEEE/ACM/Elsevier journal-only papers may be under-represented (Semantic Scholar was unavailable). (2) No formal risk-of-bias tool was applied (lit-review mode); quality is instead flagged inline per source (peer-reviewed vs preprint). (3) Counts are not a complete census — this is a focused narrative review, not a PRISMA-registered systematic review. (4) Author/region metadata is inferred from arXiv affiliation fields where present.

---

## 2. Annotated Bibliography (APA-style, grouped by theme)

Citation format: Author(s) (Year). *Title*. Venue/arXiv. Quality tag: **[PR]** = peer-reviewed (journal/conf), **[PP]** = preprint.

### Theme A — Surveys & foundational reviews
1. Neupane, J., Ables, J., Anderson, W., Mittal, S., Rahimi, S., Banicescu, I., & Seale, M. (2022). Explainable Intrusion Detection Systems (X-IDS): A Survey of Current Methods, Challenges, and Opportunities. arXiv:2207.06236. **[PR]** — Foundational X-IDS survey; proposes human-in-the-loop architecture; explicitly recommends (i) defining explainability for IDS, (ii) stakeholder-tailored explanations, (iii) metrics to evaluate explanations. *These three recommendations remain largely unaddressed — a key driver of several gaps below.*
2. Zhang, Z., Al Hamadi, H., Damiani, E., Yeun, C. Y., & Taher, F. (2022). Explainable Artificial Intelligence Applications in Cyber Security: State-of-the-Art in Research. *IEEE Access*. arXiv:2208.14937. **[PR]** — Early survey noting the absence of XAI-for-cybersecurity reviews; motivates the field.
3. Khan, N., Ahmad, K., Al Tamimi, A., Alani, M. M., Bermak, A., & Khalil, I. (2024). Explainable AI-based Intrusion Detection System for Industry 5.0: Overview, Challenges, Solutions, Research Directions. arXiv:2408.03335. **[PR/PP]** — Frames "Adversarial XIDS (Adv-XIDS)" and Industry-5.0 human-AI collaboration.
4. Liu, J., Tayeen, A. S. M., Kumar, P., Gong, Q., Jiang, W., Cao, H., Misra, S., & Harikumar, J. (2026). Generative AI and Federated Learning for Intrusion Detection Systems: A Survey. arXiv:2607.01305. **[PP]** — Covers GenAI/FL for IDS, lists open challenges incl. synthetic-data quality, dual-use adversarial risk, domain-specific LLMs.
5. Hakim, S. B., Adil, M., Velasquez, A., Xu, S., & Song, H. H. (2025). Neuro-Symbolic AI for Cybersecurity: State of the Art, Challenges, and Opportunities. arXiv:2509.06921. **[PR/PP]** — Systematic review (103 pubs); flags evaluation-standardization, compute, and under-explored human-AI collaboration; first dual-use analysis of autonomous offensive systems.
6. Ndayipfukamiye, T., Ding, J., Sarwatt, D. S., Philipo, A. G., & Ning, H. (2025). Adversarial Defense in Cybersecurity: A Systematic Review of GANs. arXiv:2509.20411. **[PR/PP]** — PRISMA SLR (185 studies); names limited explainability as a persistent GAN-defense gap.
7. Khanfor, A., Hamadi, R., Lasla, N., & Ghazzai, H. (2026). AI-driven Intrusion Detection for UAV in Smart Urban Ecosystems: A Comprehensive Survey. arXiv:2601.19345. **[PP]** — UAV-IDS survey; lists explainability, LLMs, data scarcity as open directions.
8. Tran, T. D. (2026). Research Methodologies for Cybersecurity in Enterprise Environments: A Narrative Review. arXiv:2608.24850. **[PP]** — Reviews 151 works; argues inconsistent IDS-algorithm rankings are best explained by **evaluation-design variation** rather than the algorithms — a methodological red flag underpinning Gap #4.

### Theme B — SHAP/LIME on classical DL/ML IDS (the mainstream)
9. Ayan, M. M. J., Rashid, M. S., Hassan, T. A., … Quader, F. (2026). Human-Centered Explainable AI for Security Enhancement: A Deep Intrusion Detection Framework. *IEEE SoutheastCon 2026*. arXiv:2602.13271. **[PR]** — CNN+LSTM on NSL-KDD (0.99 acc) + SHAP + a **trust-focused expert survey (IPIP6/Big-Five) via interactive UI**. One of very few human-facing studies; still uses the 2009 NSL-KDD dataset.
10. Haque, B. M. T., Rahman, M. A., Rubel, M. S. K. C., & Hossan, M. I. (2024). XAI-Driven Cyber Risk Analytics… U.S. Critical Infrastructure. arXiv:2606.05710. **[PR]** — XGBoost/RF/DT + SHAP on CICIDS2017; risk-analytics framing.
11. Rahman, M. A., Haque, B. M. T., Hossan, M. I., & Rubel, M. S. K. C. (2025). Cognitive Threat Intelligence and Explainable Federated Security Analytics. arXiv:2606.05701. **[PR]** — FL + SHAP/LIME on NSL-KDD & CIC-IDS2017.
12. Aslam, A., Hassan, M., Zahra, B., & Shahzad, M. K. (2026). XAI-SOH-FL. arXiv:2606.00134. **[PR/PP]** — FL + SHAP, CICIDS2017, 94.1% acc; claims interpretability without fidelity evaluation.
13. Luna, L., Berkowitz, M. P., Kandel, L. N., & Jansen-Sánchez, S. (2025). Dueling Deep Q-Learning for Intrusion Detection. *IEEE SoutheastCon 2025*. arXiv:2608.11291. **[PR]** — Dueling DQN + SHAP on CIC-IDS2018.
14. Aslam, H., Li, Y., Aslam, S., & Mwamughunda, G. (2026). A Calibrated and Explainable Bimodal ML Framework for Hybrid Intrusion Detection. arXiv:2608.16160. **[PP]** — SHAP + open-set detection on CIC-IDS2017.
15. Berrezzek, A., Djellali, H., Mallardi, G., & Mahnane, L. (2026). Explainable Hybrid Feature Selection for Intrusion Detection in IoMT. arXiv:2608.00869. **[PP]** — SHAP/LIME feature selection on CIC-IoMT-2024 & CIC-IDS2017.
16. Ashikuzzaman, Saifuzzaman Abhi, M., & Rahman, M. (2026). SDNGuardStack. arXiv:2604.20934. **[PP]** — Ensemble IDS for SDN (InSDN) + SHAP; 99.98% acc.
17. Gholamrezazadeh, M. H., & Montazerolghaem, A. R. (2026). XAI FL-IDS. arXiv:2605.19448. **[PP]** — FL (10 clients) + SHAP on Edge-IIoTset; >99% acc.
18. Zarkadis, I.-C., & Douligeris, C. (2026). XAI and Statistical Analysis for Reliable Intrusion Detection in the UAVIDS-2025 Dataset. arXiv:2605.13922. **[PP]** — SHAP + statistical validation (Westfall–Young permutation, Jensen–Shannon distance); one of few to **statistically validate explanations**.

### Theme C — GNN / provenance-based + XAI
19. Himmelhuber, A., Dold, D., Grimm, S., Zillner, S., & Runkler, T. (2022). Detection, Explanation and Filtering of Cyber Attacks Combining Symbolic and Sub-Symbolic Methods. *IEEE SSCI 2022*. arXiv:2212.13991. **[PR]** — GNN + symbolic XAI; uses a **fidelity metric**, cuts false positives 66% (93% with fidelity). Early exemplar of explanation-fidelity evaluation.
20. Kaya, K., Ak, E., Bas, S., Canberk, B., & Gunduz Oguducu, S. (2024). X-CBA: Explainability Aided CatBoosted Anomal-E for IDS. *IEEE ICC 2024*. arXiv:2402.00839. **[PR]** — GNN + custom XAI; 99.47% acc; local+global explanations.
21. Farrukh, Y. A., Wali, S., Khan, I., & Bastian, N. D. (2024). XG-NID: Dual-Modality NID using a Heterogeneous GNN and LLM. arXiv:2408.16021. **[PR/PP]** — Heterogeneous GNN + LLM-generated explanations; releases GNN4ID tool; F1 97%.
22. Wu, W., Qiao, W., Li, T., Feng, Y., Ma, Z., Ma, J., & Liu, Y. (2025). ProvX: Generating Counterfactual-Driven Attack Explanations for Provenance-Based Detection. arXiv:2508.06073. **[PP]** — Counterfactual subgraph explanations for GNN provenance-IDS (APT); necessity 51.6%; closed-loop detection–explanation–feedback.
23. Dhanuka, D., & Rastogi, N. (2025). PROVEX: Enhancing SOC Analyst Trust with Explainable Provenance-Based IDS. arXiv:2512.18199. **[PP]** — KAIROS + GraphMask/GNNExplainer/VA-TGExplainer on DARPA CADETS E3; 3–5s explanation overhead/event; **SOC analyst trust** framing.
24. Feito-Casares, E., Melgarejo-Meseguer, F. M., Casiraghi, E., Valentini, G., & Rojo-Álvarez, J.-L. (2026). Interpreting Manifolds and GNN Embeddings from IoT Traffic Flows. arXiv:2602.05817. **[PP]** — Interpretable GNN manifold; F1 0.83; visualizes concept drift.
25. Nayeri, Z. M., & Rezvani, M. (2026). BiTA: Bidirectional GRU-Transformer Aggregator in a Temporal Graph Network. arXiv:2604.22781. **[PP]** — Temporal-GNN alert prediction; claims interpretability.

### Theme D — Counterfactual & actionable explanations (rare)
26. Galwaduge, V., & Samarabandu, J. (2025). Tabular Diffusion based Actionable Counterfactual Explanations for Network Intrusion Detection. arXiv:2507.17161. **[PP]** — Diffusion-based counterfactuals; **first comparative analysis of counterfactual algorithms for NIDS**; actionable global rules. (Only 3 arXiv hits total for "intrusion detection + counterfactual" 2020–2026 — evidence of a near-empty sub-field.)

### Theme E — LLM / RAG / conversational / multi-modal XAI (new technology)
27. Chatzimiltis, S., Shojafar, M., Boloursaz Mashhadi, M., & Tafazolli, R. (2025). Interpretable Anomaly-Based DDoS Detection in AI-RAN with XAI and LLMs. arXiv:2507.21193. **[PP]** — LSTM + SHAP/LIME + LLM natural-language insight for 5G/6G RAN; F1 >0.96.
28. Kong, W., Mohammad Saber, A., Youssef, A., et al. (2026). Large Language Models as Explainable Cyberattack Detectors for Energy Industrial Control Systems. arXiv:2604.26079. **[PP]** — LLM as IDS+explainer for Modbus/SCADA; **intervention-based sufficiency/necessity diagnostics** on cited tokens — rare faithfulness evaluation.
29. Apurba, K. A., Hasan, M. H., Moon, M. Z., Rahman, S. M. M., & Inomata, A. (2026). Defending Retrieval-Augmented Intrusion Detection Against Knowledge Poisoning and Prompt Injection. arXiv:2608.08100. **[PP]** — RAG-IDS on CIC-UNSW-NB15; retrieval-layer attack + defense (LECC). Only work studying **adversarial manipulation of the LLM-explanation channel** for IDS.
30. Nguyen, C. C., Trang Mai, X., Ngo, V.-D., et al. (2026). Conversational versus Dashboard Explainable AI for UAV Intrusion Detection. arXiv:2608.10434. **[PP]** — Controlled experiment: conversational LLM-XAI perceived **more useful** but induced **lower appropriate self-reliance (over-reliance)** vs dashboard. Critical contradictory/trade-off finding.

### Theme F — Explanation fidelity, stability, cost, robustness (overlooked-variable evaluative papers)
31. Tolay, A. (2026). Beyond Detection Accuracy: Measuring Explanation Cost, Stability, and Utility for Resource-Aware IoT Intrusion Detection. arXiv:2608.10349. **[PP]** — TreeSHAP cost: **700.8s (RF) vs 1.47s (XGBoost)** at 5,000 samples; stability under perturbation; leakage-safe CICIoT2023 corpus. First explicit treatment of **explanation cost**.
32. Vourganas, I. J., & Michala, A. L. (2026). Stabilising Explainability Fragility in Cybersecurity AI: Multicollinearity in Public Benchmarks. arXiv:2605.22529. **[PP]** — Theorem + mitigations (CAA-Filtering, SHARP); "Explanability Fragility Score"; SHAP/LIME unstable under multicollinearity on UNSW-NB15.
33. Bouke, M. A., Sayeed, M. S., Heng, S.-H., Abdullah, A., & Othman, M. (2026). Multi-Level Distributional Entropy for Explainable Network IDS. arXiv:2606.29797. **[PP]** — Leakage-free pipeline; **F1=0.74 hides DR=0.48**; held-out attacks F1>0.998 but DR=0; temporal shift: AUC=0.87 yet thresholds collapse (DR=0.082); SHAP fold-stability ρ=0.80–0.95.
34. Hakim, M. A., Uddin, M. S., & Anis, T. I. (2026). Cross-Domain Generalization Failure in Lightweight IDS for IIoT. arXiv:2607.00553. **[PP]** — Cross-network failure; SHAP reveals **shortcut features** (coarse port categories); adversarial robustness **uncorrelated** with generalization.
35. Zhou, Y., Hamlen, K., & De Lucia, M. (2026). Robust and Explainable Divide-and-Conquer Learning for IDS. arXiv:2605.02015. **[PP]** — Lightweight, more robust + more explainable; edge-deployable.

### Theme G — Analyst-centric / human studies (rare)
36. Kumar, A., & Thing, V. L. L. (2026). Explaining Intrusion Alert Decisions of DL-based NIDS for Security Analysts (EXP-SEC). arXiv:2607.12203. **[PR/PP]** — SOC-analyst-aligned group/overlap-aware explanations; compares against xNIDS on group-level utility metrics.
37. Cherepanov, I., Sessler, D., Ulmer, A., Wagner, F., May, T., & Kohlhammer, J. (2026). Interactive Analysis of Global Explanations using Aggregated Class Activation Maps for Network Data. arXiv:2608.13575. **[PP]** — Visual-interactive global CAM explanations; expert evaluation.

### Theme H — Dataset / evaluation-protocol critique
38. Guerra, L., Chapuis, T., Duc, G., Mozharovskyi, P., & Nguyen, V.-T. (2026). How Benchmarks and Evaluation Protocols Shape Conclusions in Provenance-Based IDS. *NDSS 2027*. arXiv:2608.01454. **[PR]** — DARPA TC E3; **alerting success ≠ investigation utility**; a simple allowlist matches learned baselines (lexical novelty). Strong evidence that architecture claims are protocol-dependent.

### Theme I — Forensic / process-mining / deployment
39. Vela Alonso, J. L., & Pellicer, C. (2026). Forensic-Oriented IDS Using Synthetic Network Traffic Data and XAI. arXiv:2607.00763. **[PP]** — Train-on-Synthetic (CTGAN) + SHAP; ISO/IEC 27037 forensic chain; CICIDS2017/UNSW-NB15/Kitsune.
40. Vitale, F., Grimaldi, F., Rak, M., et al. (2026). Enhancing Anomaly-Based IDS with Process Mining. arXiv:2604.18066. **[PP]** — Process-based explanations + alert-severity rating; USB-IDS-TC.
41. Lane, R., Cummins, L., & Perkins, A. (2026). parHSOM: Parallel Hierarchical Self-Organizing Map. arXiv:2605.08164. **[PP]** — Intrinsically-explainable HSOM-IDS; scalable.

---

## 3. Literature Matrix (Source × Theme; ✓ = supports, ✗ = contradicts, ○ = partial)

| # | Source (short) | XAI method | DL arch | Dataset (vintage) | Fidelity eval'd? | Human study? | Adversarial/drift? | Leaka­ge-controlled? |
|---|---|---|---|---|---|---|---|---|
| 9 | Ayan 2026 | SHAP | CNN+LSTM | NSL-KDD (2009) | ✗ | ✓ (trust survey) | ✗ | ✗ |
| 10 | Haque 2024 | SHAP | XGBoost/RF | CICIDS2017 | ✗ | ✗ | ✗ | ✗ |
| 13 | Luna 2025 | SHAP | Dueling DQN | CIC-IDS2018 | ✗ | ✗ | ✗ | ✗ |
| 16 | Ashikuzzaman 2026 | SHAP | Ensemble | InSDN | ✗ | ✗ | ✗ | ✗ |
| 19 | Himmelhuber 2022 | symbolic | GNN | industrial demo | ✓ (fidelity) | ✗ | ○ | ✗ |
| 21 | Farrukh 2024 | LLM-explain | HeteroGNN | custom | ○ | ✗ | ✗ | ✗ |
| 22 | Wu 2025 (ProvX) | counterfactual | GNN | provenance | ✓ (necessity) | ✗ | ✓ (closed-loop) | ✗ |
| 23 | Dhanuka 2025 (PROVEX) | GraphMask/GNNExplainer | TGNN | DARPA CADETS E3 | ✓ (fidelity) | ✓ (trust framing) | ✗ | ○ |
| 26 | Galwaduge 2025 | counterfactual | DL | 3 modern NIDS | ○ (comparison) | ✗ | ✗ | ✗ |
| 28 | Kong 2026 | LLM-grounded | LLM | ICS Modbus | ✓ (intervention) | ✓ (triage) | ✗ | ✗ |
| 29 | Apurba 2026 | LLM/RAG | LLM | CIC-UNSW-NB15 | ○ | ✗ | ✓ (poisoning) | ✗ |
| 30 | Nguyen 2026 | LLM-conversational | ML/DL | UAV | ✗ | ✓ (controlled exp.) | ✗ | ✗ |
| 31 | Tolay 2026 | TreeSHAP | RF/XGB | CICIoT2023 | ✓ (cost/stability) | ✗ | ○ | ✓ |
| 32 | Vourganas 2026 | SHAP/LIME | many | UNSW-NB15 | ✓ (fragility) | ✗ | ○ | ✗ |
| 33 | Bouke 2026 | SHAP | MDE+ML | NSL-KDD/CICIDS17/18/UNSW | ✓ (stability) | ✗ | ✓ (temporal shift) | ✓ |
| 34 | Hakim 2026 | SHAP | lightweight | IIoT (3 sets) | ✓ (shortcuts) | ✗ | ✓ (adversarial) | ○ |
| 36 | Kumar 2026 (EXP-SEC) | group-aware | DL NIDS | — | ✓ (group utility) | ✓ (analyst format) | ✗ | ✗ |
| 38 | Guerra 2026 | — | PIDS | DARPA TC E3 | — | ✗ | ✓ (protocol) | ✓ |
| 39 | Vela Alonso 2026 | SHAP | XGBoost | CICIDS17/UNSW/Kitsune | ○ | ✗ | ✗ | ○ (synthetic) |

**Convergence.** (a) SHAP/LIME are the de-facto XAI toolkit (>80% of empirical papers). (b) NSL-KDD, CICIDS2017/2018, UNSW-NB15 remain the default datasets even in 2024–2026. (c) Detection accuracy/F1 is the dominant reported metric. (d) Explanation is almost always "shown via a SHAP plot," not evaluated.
**Divergence / contradiction.** (i) "99%+ accuracy" claims vs metric-collapse findings (Bouke 2026; Guerra 2026; Tran 2026). (ii) Conversational-LLM XAI: more usable **but** less appropriate reliance (Nguyen 2026). (iii) Adversarial robustness **uncorrelated** with cross-network generalization (Hakim 2026). (iv) Federated-XAI privacy claims vs federated-channel attack surface (Apurba 2026).
**Silence (≤2 sources).** Counterfactual XAI for NIDS; real-SOC-analyst controlled studies; explanation cost/latency; explanation under concept drift; geographic/understudied-region evaluation; standardized explanation-quality metrics; LLM-explanation faithfulness/hallucination; XAI dual-use (explanations aiding evasion).

---

## 4. Capstone — Research-Gap Analysis

For each of the eight requested categories: (1) what is known, (2) what is unknown, (3) why it matters, (4) a candidate research question, (5) a realistic method, (6) originality & feasibility (H/M/L). All citations are web-verified arXiv IDs above.

### Gap 1 — Understudied populations (real SOC analysts & non-expert stakeholders)
1. **Known.** XAI-IDS nominally targets "security analysts." Only a handful involve humans: a UI trust survey (Ayan 2026), a controlled conversational-vs-dashboard experiment (Nguyen 2026), analyst-format design (Kumar 2026 / EXP-SEC; PROVEX 2025; Kong 2026), and expert visual evaluation (Cherepanov 2026). Most papers claim explainability **for** analysts without studying any.
2. **Unknown.** How tier-1/2/3 SOC analysts actually use explanations during alert triage; cognitive load and trust calibration over a shift; novice vs expert differences; non-expert/management stakeholders; analysts in understudied regions; whether explanations improve or harm triage accuracy under time pressure.
3. **Why it matters.** The entire purpose of XAI is human understanding. Nguyen 2026 already shows a usability→over-reliance trade-off: "better" explanations can reduce appropriate self-reliance. Without studying real end-users, "explainability" is an unfalsifiable label, and deployment may harm security.
4. **RQ.** *How do tier-1 SOC analysts use, trust, and calibrate reliance on SHAP-dashboard vs LLM-conversational explanations during a realistic alert-triage shift, and which format better preserves appropriate reliance when the IDS errs?*
5. **Method.** Within-subjects controlled study: practicing SOC analysts (or a validated practitioner panel as fallback), a realistic alert stream with injected IDS errors, two XAI interfaces, measures = triage accuracy, time-on-task, NASA-TLX cognitive load, and calibrated reliance (reliance on correct vs incorrect alerts). Mixed-effects regression; pre-registered protocol; ethics approval.
6. **Originality/feasibility.** Originality **H** (genuine, repeatedly-flagged human gap). Feasibility **M** — analyst recruitment + IRB is the binding constraint; a student/practitioner panel is a realistic fallback.

### Gap 2 — Geographic gaps & understudied-region evaluation
1. **Known.** Authorship skews to South Asia, Middle East (UAE/KSA), Europe, US, China, Malaysia, Turkey. Datasets are mostly Canadian (CIC), Australian (UNSW), or DARPA. Cross-network generalization failure is documented (Hakim 2026), and SHAP can latch onto dataset-specific shortcuts (Hakim 2026; Vourganas 2026).
2. **Unknown.** Whether XAI explanations derived on CIC/UNSW benchmarks remain faithful/useful on networks in understudied regions (Sub-Saharan Africa, LATAM, parts of SE Asia); whether regional threat/traffic mixes and analyst contexts change shortcut structure; no XAI-IDS study evaluates on regional operational traffic.
3. **Why it matters.** Documented cross-network failure means exported "explainable" models may mislead analysts in different regions — an equity and global-cybersecurity issue; explanations of non-transferable shortcuts are actively harmful.
4. **RQ.** *Do SHAP-based explanations for a DL-IDS trained on CICIDS2017/UNSW-NB15 remain faithful and shortcut-free when applied to network traffic from an understudied region, and which features lose transferability?*
5. **Method.** Train DL-IDS on benchmark; evaluate on a regional traffic source (partner-institution capture, or a geographically-distinct public set, e.g., TUNADROMD/HIKARI-2021, clearly labelled); compute SHAP faithfulness (sufficiency/necessity), stability (Kendall τ, per Vourganas 2026/Bouke 2026), and shortcut-feature detection (Hakim 2026 protocol).
6. **Originality/feasibility.** Originality **H**. Feasibility **M** — depends on obtaining a credible regional traffic source; if none, use a controlled regional-flavor capture or synthetic stream with an explicit limitation.

### Gap 3 — Methodological weaknesses (XAI monoculture + unevaluated explanation quality)
1. **Known.** SHAP/LIME dominate (>80% of papers); counterfactual/intrinsic/attention-fidelity/concept-based are rare (only 3 NIDS-counterfactual works 2020–2026). Explanation **fidelity/faithfulness is almost never measured** — exceptions: Himmelhuber 2022 (fidelity metric), Vourganas 2026 (fragility score), Bouke 2026 (stability), Kong 2026 (intervention diagnostics), Tolay 2026 (cost). Neupane 2022's explicit 2022 call for explanation-evaluation metrics remains largely unanswered.
2. **Unknown.** Which XAI methods are faithful for which DL architectures/datasets; an apples-to-apples, metric-grounded comparison of SHAP vs LIME vs counterfactual vs intrinsic vs LLM-conversational on the same DL-IDS using fidelity, stability, cost, and analyst-utility metrics; whether SHAP's multicollinearity fragility (Vourganas 2026) and cost (Tolay 2026) change conclusions.
3. **Why it matters.** This is the field's foundational weakness: "explainability" claims are currently unfalsifiable. A misleading-but-confident SHAP plot can harm analysts more than no explanation. Methodological rigor underpins every downstream claim.
4. **RQ.** *On a fixed DL-IDS (1D-CNN and a Transformer) across three leakage-controlled datasets, how do SHAP, LIME, counterfactual, an intrinsic model, and LLM-conversational explanations compare on faithfulness (sufficiency/necessity), stability (Kendall τ), generation cost, and downstream triage utility?*
5. **Method.** Benchmark protocol: fix models and datasets (leakage-controlled re-splits of CICIDS2017, UNSW-NB15, CICIoT2023); run all XAI methods; evaluate with a battery of explanation-quality metrics (descriptive accuracy, sparsity, stability, faithfulness, runtime) + a small analyst-triage task. Pre-registered; open code.
6. **Originality/feasibility.** Originality **H** (no such comparison exists). Feasibility **H** — fully scriptable on a single GPU workstation; all methods are open-source. **Strongest, most feasible gap.**

### Gap 4 — Contradictory findings
1. **Known.** (i) "99%+ accuracy" claims (Ayan 2026; Ashikuzzaman 2026; Aslam 2026; Farrukh 2024) contradict metric-collapse findings (Bouke 2026: F1=0.74 hides DR=0.48; held-out attacks F1>0.998 but DR=0; Guerra 2026: alerting ≠ investigation utility). (ii) Conversational-LLM XAI is more usable **but** reduces appropriate self-reliance (Nguyen 2026). (iii) Adversarial robustness is **uncorrelated** with cross-network generalization (Hakim 2026). (iv) Tran 2026 argues algorithm rankings are evaluation-design artifacts.
2. **Unknown.** Which contradictions are reconcilable under better evaluation (leakage control, attack-family stratification, temporal splits) vs which are genuine trade-offs (usability↔over-reliance). No systematic reconciliation study exists.
3. **Why it matters.** Unresolved contradictions make real deployment unsafe; the usability/over-reliance trade-off has direct operational safety implications; "is XAI-IDS actually better?" remains unanswered.
4. **RQ.** *Are the "99% accuracy" and "explanation improves analyst performance" claims in XAI-IDS reconcilable under leakage-controlled, attack-family-stratified, temporally-split evaluation, and does the conversational-XAI usability benefit persist once a cognitive-forcing function is added?*
5. **Method.** Reproduction + re-evaluation: take 3–5 high-claim XAI-IDS papers, reimplement under a unified leakage-controlled, per-attack-family, temporal protocol (Bouke 2026; Guerra 2026) and report metric collapse; then a within-subjects analyst study with/without a verification-forcing function to test the over-reliance trade-off.
6. **Originality/feasibility.** Originality **H** (reproduction+reconciliation is rare and high-value). Feasibility **M-H** — reproduction is labor-intensive but mechanical; the analyst study is the hard part (overlaps Gap 1).

### Gap 5 — Variables researchers have overlooked
1. **Known.** Studies report accuracy + a SHAP plot. Overlooked variables with only 1–2 sources each: **explanation stability/fragility** (Vourganas 2026; Bouke 2026), **explanation cost/latency** (Tolay 2026: TreeSHAP 700.8s vs 1.47s), **concept drift** (Bouke 2026 threshold collapse; Feito-Casares 2026 drift viz), **analyst cognitive load & trust calibration** (Ayan 2026; Nguyen 2026; PROVEX), **alert severity/forensic traceability** (Vitale 2026; Vela Alonso 2026), **adversarial manipulation of the explainer** (ProvX 2026 closed-loop; Apurba 2026 poisoning).
2. **Unknown.** The **joint** behavior of {fidelity, stability, cost, drift-robustness} for XAI on DL-IDS over operational time; the "explanation half-life" under drift; whether explanations leak information usable for evasion (dual-use).
3. **Why it matters.** Operationally, an explanation that is faithful once but unstable, costly, or misleading under drift is worse than none. These variables determine deployability and safety, yet are almost never co-measured.
4. **RQ.** *How do faithfulness, stability, generation latency, and drift-robustness of SHAP vs counterfactual explanations co-evolve for a DL-IDS under simulated concept drift over a 6-month operational window, and at what drift point do explanations become unreliable?*
5. **Method.** Train DL-IDS on CICIDS2017/UNSW; stream chronologically-ordered or drift-injected test traffic; at intervals compute fidelity (sufficiency/necessity), stability (Kendall τ over bootstraps), latency, and detection-rate collapse (Bouke 2026); plot joint degradation curves; derive an "explanation half-life."
6. **Originality/feasibility.** Originality **H** (joint variable analysis is novel). Feasibility **H** — single workstation, open datasets, fully scriptable. Builds naturally on Gap 3.

### Gap 6 — Outdated datasets
1. **Known.** NSL-KDD (2009) and CICIDS2017/2018 still dominate 2024–2026 XAI-IDS (Ayan 2026; Haque 2024; Rahman 2025; Aslam 2026; Luna 2026; Berrezzek 2026). Newer/modern sets exist but are under-used: CICIoT2023, CIC-IoMT-2024, UAVIDS-2025, Edge-IIoTset, InSDN, TON_IoT, BoT-IoT, HIKARI-2021, TUNADROMD. Older sets lack modern attacks (zero-day, LLM-driven, supply-chain, encrypted) and carry known flaws (leakage, duplicates, unrealistic background — Tolay 2026; Bouke 2026).
2. **Unknown.** What fraction of recent XAI-IDS papers still rely on pre-2018 datasets; whether explanation-fidelity findings obtained on NSL-KDD/CICIDS2017 replicate on 2023–2025 datasets for the same DL-IDS.
3. **Why it matters.** Findings on 2009/2017 data may not generalize to modern threats; explaining an obsolete detector is moot; the field's evidence base may rest on outdated ground truth.
4. **RQ.** *What share of 2020–2026 XAI-IDS papers use pre-2018 datasets, and do SHAP explanation-fidelity findings on NSL-KDD/CICIDS2017 replicate on CICIoT2023/CIC-IoMT-2024/UAVIDS-2025 for a fixed DL-IDS?*
5. **Method.** (a) Systematic dataset audit of ~50 XAI-IDS papers (dataset × year, descriptive). (b) Replication: train a fixed DL-IDS on NSL-KDD, CICIDS2017, and a 2024 set; apply SHAP; compare faithfulness/stability; measure transfer of explanation findings.
6. **Originality/feasibility.** Originality **M-H** (dataset critique is a known complaint; the audit is partly descriptive but the replication adds originality). Feasibility **H** — scriptable; the audit is systematic and low-risk.

### Gap 7 — Missing comparisons
1. **Known.** Rare head-to-head comparisons: only xNIDS vs EXP-SEC (Kumar 2026) and a counterfactual algorithm comparison (Galwaduge 2025). Missing: multi-XAI on the same IDS; DL architecture × XAI matched pairs; post-hoc vs intrinsic; XAI-DL vs classical-ML-with-XAI; benchmark vs real traffic; per-attack-family vs aggregate; LLM-conversational vs traditional XAI.
2. **Unknown.** Which comparisons would most change conclusions; whether XAI method and DL architecture interact (i.e., best XAI depends on architecture/dataset).
3. **Why it matters.** Without head-to-head comparisons, "XAI improves IDS" stays unfalsifiable and method/architecture choice is uninformed.
4. **RQ.** *For fixed modern datasets, which (XAI method × DL architecture) pairs maximize explanation fidelity at a given detection/cost budget, and does the ranking change across attack families and dataset vintages?*
5. **Method.** Full-factorial design: XAI ∈ {SHAP, LIME, counterfactual, intrinsic, attention, LLM} × DL ∈ {1D-CNN, LSTM, Transformer, GNN} on 3 datasets; multi-metric (detection, fidelity, stability, cost); rank analysis; interaction tests. (Overlaps Gap 3 — note the overlap.)
6. **Originality/feasibility.** Originality **H** (no full-factorial exists). Feasibility **M** — compute-heavy; pre-registration strongly advised.

### Gap 8 — New technologies/events older research ignored (LLM-XAI faithfulness & adversarial/dual-use)
1. **Known.** New entrants not considered by pre-2023 work: LLM/RAG-IDS (Apurba 2026; Kong 2026; Chatzimiltis 2025; Farrukh 2024), conversational XAI (Nguyen 2026), multi-modal CLIP (Masukawa 2025, PacketCLIP), neuro-symbolic (Hakim 2025; Himmelhuber 2022), quantum rule-mining (Spell 2026, arXiv:2604.27153), diffusion counterfactuals (Galwaduge 2025), LLM-driven cyberattacks (dual-use, flagged in Liu 2026; Hakim 2025).
2. **Unknown.** (a) Whether LLM natural-language explanations are **faithful or hallucinated** (only Kong 2026 does intervention diagnostics); (b) security of the explanation interface — prompt injection / knowledge poisoning to manipulate analyst triage (only Apurba 2026 on RAG, not conversational XAI); (c) **dual-use**: can published XAI explanations help adversaries evade (ProvX 2026 closed-loop hints yes).
3. **Why it matters.** LLM-based IDS/XAI is being adopted faster than it is validated; a hallucinated-but-plausible explanation can mislead analysts worse than no explanation; dual-use is an ethics/safety concern.
4. **RQ.** *Are LLM-generated natural-language explanations of DL-IDS alerts faithful to the model's reasoning (vs hallucinated), and can an adversary use the explanation interface (prompt injection / knowledge poisoning) to manipulate analyst triage decisions?*
5. **Method.** (a) Faithfulness: intervention-based sufficiency/necessity (Kong 2026) + a faithfulness benchmark on LLM-conversational XAI vs SHAP. (b) Adversarial: red-team the explanation interface (extend Apurba 2026's threat model to conversational XAI); measure analyst-targeting success in a controlled study. Ethics/dual-use review required.
6. **Originality/feasibility.** Originality **H** (cutting-edge, safety-critical). Feasibility **M** — needs LLM API access; red-teaming requires ethics/dual-use review; fast-moving risk.

---

## 5. Ranked Top 5 Gaps (Strongest → Weakest)

**Ranking criteria:** originality, feasibility for a master's student (compute + data + recruitment + ethics), significance/impact, defensibility, and the strength of evidence that the gap is real.

**Rank 1 — Gap 3: Methodological weaknesses (XAI head-to-head benchmark with explanation-quality metrics).**
*Why strongest.* It is the field's foundational weakness (unfalsifiable "explainability"), has the clearest evidence base (Neupane 2022's unanswered call; Vourganas 2026; Tolay 2026; Bouke 2026), and the **highest feasibility**: fully scriptable on one workstation, all methods open-source, no human-subjects/IRB burden. A master's student can complete it end-to-end and produce a publishable benchmark. Originality H, feasibility H, significance H.

**Rank 2 — Gap 5: Overlooked variables (joint fidelity–stability–cost–drift degradation & "explanation half-life").**
*Why second.* High operational significance (deployability/safety), strong evidence (Tolay 2026 cost; Vourganas 2026 fragility; Bouke 2026 drift), high originality (joint analysis is novel), and **high feasibility** (scriptable, single workstation). It builds naturally on Rank 1, and a student could even combine the two into one coherent project. Slightly below Rank 1 only because it assumes the benchmark apparatus of Rank 1 exists.

**Rank 3 — Gap 6: Outdated datasets (systematic audit + fidelity replication).**
*Why third.* High feasibility (scriptable, low-risk) and high significance (the field's evidence base), with a defensible two-part design (audit + replication). Slightly lower originality than Ranks 1–2 — dataset obsolescence is a frequently-aired complaint, and the audit component is partly descriptive — but the replication-on-modern-datasets step adds genuine originality and a clear contribution. A safe, strong master's project.

**Rank 4 — Gap 1: Understudied populations (real SOC-analyst human study).**
*Why fourth.* Highest significance in principle (XAI's purpose is human understanding; Nguyen 2026 already shows an over-reliance hazard) and high originality. Ranked lower purely on **feasibility**: recruiting real tier-1/2 analysts and obtaining IRB approval is the binding constraint for a master's student; a practitioner/student panel is a realistic but weaker fallback. Impact is potentially the highest of all five if executed with real analysts.

**Rank 5 — Gap 8: New technologies (LLM-XAI faithfulness & adversarial/dual-use).**
*Why fifth.* Highest timeliness and originality (LLM-XAI is cutting-edge and safety-critical), and it directly addresses a fast-emerging deployment risk. Ranked last on **feasibility/risk**: it needs LLM API access, a red-team ethics/dual-use review, and the field moves quickly (results may date fast). A bold, high-upside master's project for a student with API budget and ethics support.

**Why these five over the others.** Gaps 2 (geographic), 4 (contradictions), and 7 (missing comparisons) are strong but rank below: Gap 2's feasibility hinges on access to credible regional traffic (data-availability risk); Gap 4 overlaps heavily with Ranks 1 and 3 and needs a costly analyst study; Gap 7 overlaps almost entirely with Rank 1 and adds heavy compute load. The top five maximize the joint score on originality × feasibility × significance while keeping the scope realistic for a master's thesis.

---

## 6. Limitations

- **Corpus bias.** arXiv-heavy; non-mirrored IEEE/ACM/Elsevier journal-only works may be under-represented; Semantic Scholar was unavailable (HTTP 429). Several included arXiv items do carry peer-reviewed `journal_ref`/DOIs, partially mitigating this.
- **Not a registered systematic review.** No formal protocol registration, no risk-of-bias tool, no meta-analysis (lit-review mode by design). Counts are approximate, not an exhaustive census.
- **Recency skew.** The corpus is heavily 2024–2026 (the window's recency), so older 2020–2021 empirical work is thinner; foundational surveys (Neupane 2022; Zhang 2022) are used to anchor pre-2023 context.
- **Generalization of gap claims.** "SHAP/LIME dominate" and "fidelity is rarely evaluated" are evidence-supported trends, not exhaustive census results; a few counter-examples exist and are cited.
- **Dual-use.** Several gaps (esp. Gap 8) touch offense-enabling knowledge; any follow-up must frame defensively and avoid operational exploit detail, per ethics norms.

## 7. References (consolidated, arXiv IDs)

Ayan, M. M. J., et al. (2026). arXiv:2602.13271. • Apurba, K. A., et al. (2026). arXiv:2608.08100. • Ashikuzzaman, et al. (2026). arXiv:2604.20934. • Aslam, A., et al. (2026). arXiv:2606.00134. • Aslam, H., et al. (2026). arXiv:2608.16160. • Berrezzek, A., et al. (2026). arXiv:2608.00869. • Bouke, M. A., et al. (2026). arXiv:2606.29797. • Chatzimiltis, S., et al. (2025). arXiv:2507.21193. • Cherepanov, I., et al. (2026). arXiv:2608.13575. • Dhanuka, D., & Rastogi, N. (2025). arXiv:2512.18199. • Farrukh, Y. A., et al. (2024). arXiv:2408.16021. • Feito-Casares, E., et al. (2026). arXiv:2602.05817. • Galwaduge, V., & Samarabandu, J. (2025). arXiv:2507.17161. • Gholamrezazadeh, M. H., & Montazerolghaem, A. R. (2026). arXiv:2605.19448. • Guerra, L., et al. (2026). arXiv:2608.01454 (NDSS 2027). • Hakim, M. A., et al. (2026). arXiv:2607.00553. • Hakim, S. B., et al. (2025). arXiv:2509.06921. • Haque, B. M. T., et al. (2024). arXiv:2606.05710. • Himmelhuber, A., et al. (2022). arXiv:2212.13991 (IEEE SSCI 2022). • Khan, N., et al. (2024). arXiv:2408.03335. • Khanfor, A., et al. (2026). arXiv:2601.19345. • Kaya, K., et al. (2024). arXiv:2402.00839 (IEEE ICC 2024). • Kong, W., et al. (2026). arXiv:2604.26079. • Kumar, A., & Thing, V. L. L. (2026). arXiv:2607.12203. • Lane, R., et al. (2026). arXiv:2605.08164. • Liu, J., et al. (2026). arXiv:2607.01305. • Luna, L., et al. (2025). arXiv:2608.11291 (IEEE SoutheastCon 2025). • Nayeri, Z. M., & Rezvani, M. (2026). arXiv:2604.22781. • Ndayipfukamiye, T., et al. (2025). arXiv:2509.20411. • Neupane, J., et al. (2022). arXiv:2207.06236. • Nguyen, C. C., et al. (2026). arXiv:2608.10434. • Rahman, M. A., et al. (2025). arXiv:2606.05701. • Tolay, A. (2026). arXiv:2608.10349. • Tran, T. D. (2026). arXiv:2608.24850. • Vela Alonso, J. L., & Pellicer, C. (2026). arXiv:2607.00763. • Vourganas, I. J., & Michala, A. L. (2026). arXiv:2605.22529. • Vitale, F., et al. (2026). arXiv:2604.18066. • Wu, W., et al. (2025). arXiv:2508.06073 (ProvX). • Zarkadis, I.-C., & Douligeris, C. (2026). arXiv:2605.13922. • Zhang, Z., et al. (2022). arXiv:2208.14937 (IEEE Access). • Zhou, Y., et al. (2026). arXiv:2605.02015. • Spell, S., & Shyu, C.-R. (2026). arXiv:2604.27153. • Masukawa, R., et al. (2025). arXiv:2503.03747 (PacketCLIP).
