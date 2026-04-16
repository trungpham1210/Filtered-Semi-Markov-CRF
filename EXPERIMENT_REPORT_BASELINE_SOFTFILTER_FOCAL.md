# Experiment Report (Baseline vs #3 Soft Filter vs Focal)

Date: 2026-03-30  
Project: Filtered-Semi-Markov-CRF (CoNLL 2003)

> Scope note: This report intentionally excludes the `#1 + #2` experiments as requested.

## 1) Goal
Evaluate whether:
- `#3` uncertainty-aware soft filtering
- focal-loss training (with `#3` disabled)

improves over the current reproducible baseline.

---

## 2) Runs Included

### A. Baseline reproduction (no soft filter, no focal)

**Approach:** Establish a reproducible baseline with standard training (no proposed improvements). This serves as the control against which improvements are measured.

- Config: `config/conll_baseline_repro_full.yaml`
- Log metrics: `logs/baseline_repro_full/log_metrics.txt`
- Key settings:
  - `soft_filter: False`
  - no focal gamma fields (defaults used: `0.0`)
- **Runtime:** ~5h45m (10,000 training steps)

### B. #3 uncertainty-aware soft filter (full run)

**Approach:** Enable soft filtering logic that ranks null-labeled span candidates by model uncertainty and keeps only the top-K most uncertain ones. The idea is that by filtering to uncertain predictions, the model focuses on harder examples where confidence thresholding may be misleading.

- Log metrics: `logs/softfilter_k8_t09_full/log_metrics.txt`
- Key settings:
  - `soft_filter: True`
  - `null_confidence_threshold: 0.9`
  - `uncertain_top_k: 8` (keep top 8 most uncertain null-labeled spans per span width)
- **Runtime:** ~6h22m (10,000 training steps)

### C. Focal-loss run with #3 disabled (full run)

**Approach:** Use focal loss to upweight hard negatives (non-entity predictions). Focal loss downweights easy examples and focuses learning on hard-to-classify instances. This could improve performance on borderline cases without relying on confidence thresholding. Soft filter (#3) is disabled to isolate focal loss's effect.

- Log metrics: `logs/focal_no3_full/log_metrics.txt`
- Key settings:
  - `soft_filter: False`
  - `focal_gamma_entity: 1.0`
  - `focal_gamma_non_entity: 2.0`
- **Runtime:** ~5h25m (10,000 training steps)

---

## 3) Results Summary

| Experiment | Best Test F1 | Final Test F1 | Best Dev F1 | Notes |
|---|---:|---:|---:|---|
| Baseline (reproduced) | **91.77** | 91.57 | 95.54 | Best test around step 5749 |
| #3 Soft Filter | **91.85** | 91.68 | 95.51 | Slightly higher peak than baseline |
| Focal (no #3) | **91.71** | 91.63 | 95.49 | Slightly below baseline peak |

---

## 4) Interpretation

1. **No clear win from focal loss**
   - Focal run does not exceed baseline peak (91.71 vs 91.77).

2. **#3 soft filter is near-parity and slightly better at peak**
   - #3 peak is +0.08 F1 above baseline reproduction (91.85 vs 91.77).
   - This gap is very small and may be within run variance.

3. **Current reproducible performance band is ~91.6–91.9 test F1**
   - Across baseline/#3/focal full runs in this code state.

4. **Earlier 93.89 reference was not reproduced in current setup**
   - Most likely due to code/config/environment drift between earlier and current runs.

---

## 5) Conclusion

- Based on current reproducible runs:
  - `#3` soft filtering: **likely parity / marginal benefit, not conclusively better**.
  - focal loss (without `#3`): **no improvement**.
- Therefore, there is **no statistically reliable improvement claim** yet over baseline from these changes.

---

## 6) Multi-Seed Validation (April 1, 2026)

### Motivation & Approach

The single-run experiments showed a +0.08 F1 difference between #3 and baseline, but this could be noise rather than a reliable signal. To determine whether #3 truly helps, we ran both approaches across multiple random seeds (42, 43, 44) and computed **mean ± std** of test F1 scores. If the difference is smaller than the variance within each approach, then #3's benefit is not statistically reliable.

### Multi-Seed Setup
- **Seeds tested:** 42, 43, 44
- **Configs:** 
  - `config/conll_baseline_seed42/43/44.yaml` (no soft filter, no focal)
  - `config/conll_softfilter_seed42/43/44.yaml` (soft filter enabled, identical #3 settings)
- **Execution:** 6 sequential runs on single GPU
  - baseline_seed42: ~5h45m
  - baseline_seed43: ~5h40m
  - baseline_seed44: ~6h19m
  - softfilter_seed42: ~5h50m
  - softfilter_seed43: ~5h55m
  - softfilter_seed44: ~5h55m
- **Total wall-clock time:** ~35 hours (sequential runs, Mar 31 04:02 → Apr 1 13:21)

### Multi-Seed Results

| Config | Seed 42 | Seed 43 | Seed 44 | Mean ± Std |
|---|---:|---:|---:|---|
| **Baseline** | 91.62 | 91.49 | 91.67 | **91.59 ± 0.09** |
| **#3 Soft Filter** | 91.68 | 91.59 | 91.63 | **91.63 ± 0.05** |

**Difference:** +0.04 F1 (soft filter minus baseline mean)

### Interpretation of Multi-Seed Results

1. **#3 soft filter does NOT show meaningful improvement**
   - Mean difference (+0.04%) is much smaller than baseline's own variance (0.09%)
   - The small gap could easily be random variation

2. **Both approaches are stable across seeds**
   - Baseline std: 0.09% (range: 91.49–91.67)
   - Soft filter std: 0.05% (range: 91.59–91.68)
   - No systematic advantage or disadvantage observed

3. **Conclusion: #3 is not worth pursuing**
   - Single-run difference of +0.08 was noise, not signal
   - Multi-seed validation confirms #3 ≈ baseline
   - Recommend dropping #3 from future work

---

## 7) Recommended Next Step

- **#3 is ruled out** based on multi-seed evidence (no reliable improvement)
- **Focal loss did not help** (single run showed 91.71 vs 91.77 baseline)
- **Future focus:** Explore `#1` (task-aware confidence threshold) and `#2` (length-constrained filtering) if not yet tested
- Consider fundamental improvements (better features, architecture changes, hyperparameter grid search) to move beyond ~91.6% baseline
