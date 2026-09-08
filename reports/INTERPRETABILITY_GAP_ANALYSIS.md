# Interpretability Criterion — Gap Analysis and Action Plan

Source of truth for the criteria: `datasources/Hackathon Problem - Die Yield Prediction.docx`
(text extracted from `word/document.xml`).

---

## 1. What the rubric actually demands

| Criterion | Weight |
| :--- | :---: |
| Prediction performance — Model A vs Model B (F1, AUC-PR on minority class) | 30% |
| Handling of imbalance (no trivial "predict all pass") | 20% |
| **Model interpretability (feature attribution, block pattern analysis, spatial heatmaps)** | **30%** |
| Creative multi-resolution fusion + Model A → B improvement analysis | 20% |

Interpretability is tied for the **largest single weight in the rubric**. Two passages define it,
and both are explicit that it must be **per-prediction**, not global:

> **Key Challenges** — "Interpretability required — for each predicted failure, the model must
> output **which features** (and **which block-level patterns**, for Model B) and **which spatial
> context** contributed most to the prediction (engineers need actionable root cause, not a black
> box)"

> **Expected Deliverables** — "Interpretability output: **per-die feature importance** + **spatial
> contribution visualization** (Model A), plus **block reading pattern analysis** (Model B)"

That decomposes into three required artifacts, all at the level of an individual die:

1. **Per-die feature attribution** — for a given predicted failure, which features drove it.
2. **Spatial contribution visualization** — heatmaps showing the spatial context's role.
3. **Block reading pattern analysis** — which sub-die block patterns drove it (Model B only).

---

## 2. What exists today

### 2.1 Markdown inventory — 15 files, 2,553 lines

| File | Lines | Interpretability content |
| :--- | ---: | :--- |
| `README.md` | 897 | One unsupported claim (line 3). "Heatmap" mentions are engine-correlation and weight-sweep heatmaps, not spatial. |
| `reports/ADIT_BRANCH_ANALYSIS.md` | 391 | Most mentions in the repo, but describes the arch-branch global-gain plots. |
| `reports/CV_ENSEMBLE_EVALUATION.md` | 197 | 1 incidental mention. |
| `reports/MODEL_D_EVALUATION.md` | 177 | Feature-importance / heatmap wording only; no per-die attribution. |
| `reports/tree_model_bakeoff_summary.md` | 141 | **None** |
| `reports/model_c2_summary.md` | 113 | 3 incidental mentions. |
| `reports/MODEL_E_EVALUATION.md` | 104 | **None** |
| `reports/FINAL_TEST_EVALUATION.md` | 103 | 1 incidental mention. |
| `PREPROCESSING_NOTES.md` | 89 | 2 mentions (feature rationale). |
| `reports/MODEL_F_PLUS_CNN_EVALUATION.md` | 77 | **None** |
| `reports/MODEL_F_EVALUATION.md` | 61 | **None** |
| `reports/MODEL_E_FINAL_TEST_EVALUATION.md` | 60 | **None** |
| `datasources/README.md` | 54 | **None** |
| `reports/MODEL_F_FINAL_TEST_EVALUATION.md` | 47 | **None** |
| `reports/MODEL_A_VS_MODEL_B_STACK_ABLATION.md` | 42 | **None** |

**8 of 15 files contain no interpretability content at all**, including every report for the
current champion (Model F) and its final-test evaluation. The documentation set is almost
entirely a performance-benchmarking narrative.

### 2.2 Code and artifacts

| Item | Status on `main` |
| :--- | :--- |
| Any `interpretability.py` | **Absent** |
| `shap` / `lime` / `captum` / `eli5` import anywhere in repo | **None** |
| `shap` in `datasources/requirements.txt` | **Absent** |
| Per-die attribution artifact | **Absent** |
| Spatial contribution heatmap | **Absent** |
| Block-pattern analysis for a specific prediction | **Absent** |
| Global feature importance | Present — `model_a`/`model_b`/`model_b_no_spatial_feature_importance.csv` |
| Channel-level gain attribution | Present — `feature_group_attribution` in 2 metrics JSONs |
| Plots / figures | 45 total, **all performance-oriented** |

All 45 figures are precision-recall curves, ROC curves, AUC-PR bar charts, engine-correlation
matrices, training curves, weight sweeps and threshold trade-offs. `plots/1`–`plots/6` are
dataset-level EDA (class balance, feature distributions, block traces, one sample wafer map).
Useful context, but none of it attributes a specific prediction.

### 2.3 The other branches don't close the gap either

`origin/adit` and `origin/sathvik2` carry six `interpretability.py` scripts and six
`*_interpretability.png` plots (arch1–arch6), never merged to `main`. Inspecting
`arch6_adversarial_resnet/interpretability.py`, what they produce is:

- top-20 **global** LightGBM gain bars,
- adversarial train-vs-val score distributions,
- a wafer process-drift scatter,
- a PR-frontier progression chart.

All four are global or wafer-level. **None is per-die attribution, a spatial heatmap, or block
pattern analysis.** Merging that branch would add polish, not rubric coverage.

### 2.4 Honest scoring estimate

Of the 30% available, the work currently supports roughly **one sixth** — global gain rankings
and channel-level attribution partially satisfy the words "feature attribution", but miss the
per-die requirement entirely, and both of the other two named artifacts are absent. Call it
**~5 of 30 points**. This is the single largest scoring gap in the project, and it is larger
than the entire margin between Model F and every model that preceded it.

---

## 3. What to do, in priority order

### P0-1 — Per-die feature attribution with TreeSHAP

The champion is a blend of CatBoost, XGBoost and LightGBM, all tree ensembles, so **exact**
TreeSHAP applies. No sampling or surrogate approximation is needed.

- Add `shap>=0.44` to `datasources/requirements.txt`.
- New `src/interpretability/per_die_attribution.py` producing, for every predicted failure:
  signed SHAP contributions, the top-N drivers, and the model's score.
- Emit `reports/per_die_attribution.parquet` keyed by `(wafer_id, die_row, die_col)`, plus an
  `explain_die(wafer_id, die_row, die_col)` helper that returns a readable explanation.
- **Aggregate SHAP into the three physical channels** (parametric / block / spatial) so each
  prediction carries a one-line summary such as *"68% block-burst evidence, 24% parametric
  drift, 8% spatial neighbourhood."* This is what makes the output actionable rather than a
  list of 1,280 opaque column names.

**Two technical points that need a decision, not a default:**

- *Attribute the blend, or an engine?* SHAP is not defined for a rank-space average, because the
  rank transform is non-linear — you cannot weight-average the three engines' SHAP values and
  stay faithful. The clean, defensible move is to attribute **CatBoost alone** (dev-val AUC-PR
  0.6312 against the blend's 0.6321, a gap of 0.0009) and state plainly that explanations come
  from the strongest single engine, which is statistically indistinguishable from the blend.
  Attempting to explain the blend directly would be the less honest choice.
- *Feature names must be translated.* Model F's matrix contains `dt_feature_*` (wafer-detrended
  residuals) and `wdev_*` / `wrank_*` / `wzscore_*` (wafer-relative transforms). Raw SHAP output
  naming `wdev_ldadt` means nothing to a process engineer. Ship a name → plain-English mapping
  so the explanation reads *"parametric drift, measured relative to this wafer's own baseline."*

### P0-2 — Spatial contribution heatmaps

Named directly in the rubric and currently absent. Cheapest high-impact item, since
`plots/6_sample_wafer_map.png` already contains the wafer-grid rendering code to build on.

For a handful of representative wafers, a 2-by-2 panel:

1. pre-test state (`old_label`),
2. predicted failure probability as a continuous heatmap,
3. true post-test outcome with true/false positives distinguished,
4. **spatial-channel SHAP contribution per die** — the panel that actually satisfies the
   criterion, showing where neighbourhood and gradient effects drove the model.

### P0-3 — Block reading pattern analysis

Also named directly, also absent. The burst *location* is already computed as a feature
(`max_rolling_mean_100_start_idx`, `burst_center_idx_600`), so the hard part is done.

For each explained failure: plot its 2,000-point trace, shade the matched window, mark the
detected burst centre, annotate burst amplitude against the die's own baseline and which window
width responded most strongly — then overlay a healthy die **from the same wafer** as a control.
That single figure answers "which block-level pattern contributed" better than any table.

### P1-1 — A worked root-cause case study

The rubric's phrase is "engineers need actionable root cause, not a black box." Judges respond
to a concrete walkthrough far more than to methodology prose. Produce one page each for four
deliberately chosen dies:

| Case | Why include it |
| :--- | :--- |
| High-confidence true positive | Shows the mechanism working cleanly |
| Marginal true positive | Shows behaviour on the 65% of failures the generator makes nearly indistinguishable |
| False positive | Demonstrates honesty and diagnostic value |
| False negative | Shows understanding of the model's limits |

Each page: the three attributions side by side, plus a plain-language root-cause sentence.

### P1-2 — Tie interpretability to the Model A → B delta (also scores the 20% criterion)

The rubric wants the A → B improvement analysed. The strongest available evidence is already
measured: block features add roughly **+0.057** AUC-PR over the parametric channel, and
`max_rolling_mean_400` is the highest-gain feature in Model B.

Isolate the dies **Model B catches that Model A misses**, then run the block-pattern analysis on
exactly that set. If their attributions are dominated by burst features, that is a direct,
quantified demonstration that the sub-die signal is what closed the gap. This single analysis
scores against both the 30% and the 20% criteria.

### P2-1 — Consolidate the documentation

- Add `reports/INTERPRETABILITY.md` as the canonical answer to criterion 3, and link it from
  `README.md` near the top.
- Add an interpretability section to `MODEL_F_EVALUATION.md`, which currently has none despite
  covering the champion.
- **Fix `README.md` line 3.** It claims the system maintains "full interpretability for process
  engineers." Nothing in the repository supports that today. Either back it or soften it — an
  unsupported claim in the opening sentence is a credibility risk with judges who will look.

### P2-2 — Resolve the calibration problem before publishing per-die numbers

Model F blends in rank space, so its output is a uniform rank, not a probability. Its Brier
score is **0.303** against roughly 0.019 for the individual engines. Presenting a die as
"92% likely to fail" would therefore be wrong, and an interpretability deliverable that misstates
its own confidence undercuts the whole submission.

Either fit isotonic regression on out-of-fold ranks to recover calibrated probabilities, or
present the score explicitly as a **risk percentile**. Either is fine; silently labelling a rank
as a probability is not.

---

## 4. Effort versus payoff

| Item | Effort | Rubric coverage |
| :--- | :--- | :--- |
| P0-1 per-die TreeSHAP | Medium | The core of the 30%; unlocks P0-2 and P0-3 |
| P0-2 spatial heatmaps | **Low** | A named artifact; highest visual impact per hour |
| P0-3 block pattern analysis | **Low** | A named artifact; burst location already computed |
| P1-1 case study | Low | Directly answers "actionable root cause" |
| P1-2 A→B attribution | Medium | Scores against 30% **and** 20% |
| P2-1 documentation | Low | Makes the work findable and removes a false claim |
| P2-2 calibration | Low | Protects the credibility of every per-die number |

P0-2 and P0-3 are both low-effort and both name-matched to the rubric. They should be done first
even though P0-1 is the deeper piece of work.

The performance criterion is already strong — Model F at 0.6321 dev-val and 0.6237 on the unseen
test. Further AUC-PR chasing competes for the 30% already largely earned, while roughly 25 of the
30 interpretability points sit untouched. **Interpretability is where the remaining marginal
return is, by a wide margin.**
