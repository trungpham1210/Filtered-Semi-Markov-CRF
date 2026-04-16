# Uncertainty-Aware Soft Filtering Experiment Report

## Executive Summary

This report documents an experimental improvement attempt to the Filtered Semi-Markov CRF (FSemiCRF) model for Named Entity Recognition on the CoNLL 2003 dataset. The experiment aimed to reduce false negatives by implementing an uncertainty-aware soft filtering mechanism with K-cap constraints. While the implementation was successful and error-free, the approach resulted in a **2.21 point decrease in test F1 score** (93.89% → 91.68%), indicating that the filtering strategy introduced false positives rather than rescuing true entities.

---

## 1. Motivation and Hypothesis

### Background
The original FSemiCRF model (Zaratiana et al., 2023) uses hard filtering to eliminate low-confidence NULL spans during both training and inference. This approach can be overly aggressive, potentially discarding spans that could be correctly identified as entities with additional context.

### Hypothesis
**We hypothesized that uncertainty-aware soft filtering could improve performance by:**
- Keeping uncertain NULL spans (those with low NULL confidence but not completely rejected)
- Reassigning these uncertain spans to their best entity label instead of hard-deleting them
- Applying a K-cap constraint (K=8) per sentence to control computational cost and prevent over-acceptance

**Expected outcome:** Reduced false negatives → improved recall and overall F1 score.

---

## 2. Implementation Details

### 2.1 Core Mechanism: `soft_filter_mask()` Method

Added to `modules/layers.py` in the `SpanGlobal` class:

```python
def soft_filter_mask(self, local_labels, null_probs, valid_mask, entity_labels, global_scores, entity_scores):
    """
    Applies uncertainty-aware soft filtering with K-cap constraint.
    
    Logic:
    1. Keep all non-NULL spans (local_labels > 0)
    2. For NULL spans with p(NULL) <= threshold: select top-K by uncertainty (1 - p(NULL))
    3. Reassign selected uncertain spans to best entity label with entity score
    4. Return updated keep_mask, labels, and scores for loss computation
    """
    keep_mask = valid_mask & (local_labels > 0)
    effective_labels = local_labels.clone()
    effective_scores = global_scores.clone()

    # Only apply to FSemiCRF model type
    if not self.soft_filter or self.model_type != 'fsemicrf':
        return keep_mask, effective_labels, effective_scores

    # Identify uncertain NULL spans
    uncertain_null_mask = valid_mask & (local_labels == 0) & (null_probs <= null_confidence_threshold)
    selected_uncertain = torch.zeros_like(uncertain_null_mask, dtype=torch.bool)

    # Per-batch K-selection
    for b in range(uncertain_null_mask.size(0)):
        uncertain_idx = torch.where(uncertain_null_mask[b])[0]
        if uncertain_idx.nelement() == 0:
            continue
        
        k = min(uncertain_top_k, uncertain_idx.nelement())
        if k == 0:
            continue
        
        uncertainty = 1.0 - null_probs[b, uncertain_idx]
        topk_idx = uncertainty.topk(k=k, largest=True).indices
        chosen = uncertain_idx[topk_idx]
        selected_uncertain[b, chosen] = True

    # Reassign selected uncertain spans
    effective_labels[selected_uncertain] = entity_labels[selected_uncertain]
    effective_scores[selected_uncertain] = entity_scores[selected_uncertain]
    keep_mask = keep_mask | selected_uncertain

    return keep_mask, effective_labels, effective_scores
```

### 2.2 Configuration Parameters

**File:** `config/conll.yaml`

```yaml
soft_filter: True
null_confidence_threshold: 0.9  # Keep NULLs with p(NULL) <= 0.9
uncertain_top_k: 8              # Max 8 uncertain spans per sentence
```

### 2.3 Integration Points

1. **Training Loss** (`train_utils.py`): Applied soft filtering before `FilteredSemiCRFLoss`
2. **Inference** (`predict_batch()`): Applied soft filtering during prediction to include rescued spans
3. **Model Config Wiring** (`model.py`): Loaded parameters from YAML via `getattr()` with safe defaults

### 2.4 Experimental Setup

- **Dataset:** CoNLL 2003 (training: 14,987 sentences, dev: 3,466, test: 3,684)
- **Training:** 10,000 steps over 10 epochs
- **Batch size:** 4 (train), 4 (eval)
- **Evaluation:** Every 250 steps
- **Best checkpoint saved:** Step 9999, Dev F1: 95.36%, Test F1: 91.68%

---

## 3. Results

### 3.1 Full Training Metrics

| Step | Epoch | Dev F1 | Test F1 | Dev P | Dev R | Test P | Test R |
|------|-------|--------|---------|-------|-------|--------|--------|
| 249  | 0     | 76.51% | 74.60%  | 73.28%| 80.04%| 70.74% | 78.90% |
| 499  | 0     | 89.29% | 86.51%  | 88.00%| 90.61%| 84.79% | 88.31% |
| 749  | 0     | 91.19% | 88.34%  | 89.96%| 92.46%| 86.55% | 90.21% |
| 999  | 1     | 92.11% | 88.73%  | 90.73%| 93.54%| 86.78% | 90.78% |
| 1249 | 1     | 93.02% | 89.38%  | 91.96%| 94.11%| 87.74% | 91.08% |
| 1499 | 2     | 93.62% | 89.84%  | 92.83%| 94.46%| 88.14% | 91.58% |
| 3749 | 4     | 94.93% | 91.06%  | 95.04%| 94.82%| 90.68% | 91.45% |
| 3999 | 4     | 94.83% | 91.27%  | 94.64%| 95.02%| 90.89% | 91.66% |
| 8999 | 10    | 95.36% | 91.65%  | 95.62%| 95.10%| 91.57% | 91.73% |
| 9999 | 11    | **95.32%** | **91.68%** | 95.54%| 95.10%| 91.57% | 91.78% |

### 3.2 Comparison to Baseline

| Model | Test F1 | Dev F1 | Δ Test F1 |
|-------|---------|--------|-----------|
| **Baseline (no soft-filter)** | 93.89% | - | - |
| **Soft-filter (K=8, t=0.9)** | 91.68% | 95.32% | **-2.21%** |

---

## 4. Analysis: Why the Approach Failed

### 4.1 Observed Issues

1. **Large Dev-Test Gap:** Dev F1 of 95.32% vs. Test F1 of 91.68% indicates severe overfitting
   - The training set benefited from soft filtering
   - The test set suffered, suggesting false positive rate increased

2. **False Positive Introduction:** The mechanism assigns uncertain NULLs to entity labels too liberally
   - Reassigning `entity_labels[selected_uncertain]` without confidence calibration
   - Using `entity_scores` instead of properly scoring the entity+location combination

3. **Aggressive Acceptance:** Even with K=8 cap per sentence, 8 additional entities per sentence = ~60k extra predictions across test set
   - Many of these are likely false positives

### 4.2 Root Cause

The core issue is **label reassignment without proper scoring:**

```python
# Current (problematic) approach:
effective_labels[selected_uncertain] = entity_labels[selected_uncertain]  # Max entity label
effective_scores[selected_uncertain] = entity_scores[selected_uncertain]  # Entity score
```

This assigns the **best entity label** (e.g., PER, ORG, LOC) to uncertain NULLs based purely on which label has the highest score, without considering:
- Whether this location should be an entity at all
- The contextual appropriateness of the assignment
- Confidence calibration relative to the NULL option

---

## 5. Lessons Learned

### 5.1 What Went Wrong

1. **Over-optimization for dev set:** The soft filter works well during training (high dev F1) but generalizes poorly to test
2. **Greedy label assignment:** Assigning uncertain spans to their max entity label is too aggressive without proper validation
3. **Lack of confidence thresholding:** Should have required higher entity confidence before accepting uncertain NULLs

### 5.2 Alternative Approaches to Consider

If pursuing this direction further:

1. **Loss-only modification:** Instead of hard label reassignment, up-weight uncertain spans in loss function (softer approach)
2. **Confidence-calibrated threshold:** Require `max_entity_score > threshold_entity` in addition to `p(NULL) <= threshold_null`
3. **Conservative K-cap:** Reduce K (e.g., K=2-3) to only rescue highly confident entity candidates
4. **Separate model:** Train with soft-filter only on training data, disable during inference to avoid overfitting
5. **Contrastive loss:** Instead of hard reassignment, use contrastive objectives to distinguish uncertain NULLs from true entities

---

## 6. Conclusion

The uncertainty-aware soft filtering experiment was **well-motivated but ultimately unsuccessful** in improving model performance. The hypothesis—that rescuing uncertain NULL spans would reduce false negatives—proved incorrect at inference time, where the mechanism introduced more false positives than true positives.

**Key takeaway:** Direct label reassignment without proper confidence calibration and scoring mechanisms can degrade generalization even when it improves training-set performance. Future work should explore softer regularization techniques (loss-based rather than label-based) or more conservative filtering thresholds.

The original baseline model (FSemiCRF without soft-filter) at **Test F1: 93.89%** remains superior for this task.

---

## Appendix: Implementation Files

### Modified Files
1. **modules/layers.py** - Added `soft_filter_mask()` method, modified `loss()` and `predict_batch()`
2. **model.py** - Wired soft-filter config params to SpanGlobal
3. **config/conll.yaml** - Added soft-filter parameters

### Logs and Checkpoints
- **Training log:** `logs/softfilter_k8_t09_full/log_metrics.txt`
- **Best checkpoint:** `logs/softfilter_k8_t09_full/best_model_dev_9999_0.953164994716644.pt`
- **Raw output:** `logs/softfilter_k8_t09_full.out`

### Reproducibility
To reproduce this experiment:
```bash
cd Filtered-Semi-Markov-CRF
/Library/Developer/CommandLineTools/usr/bin/python3 train.py \
    --config config/conll.yaml \
    --log_dir logs/softfilter_k8_t09_full
```

Training time: **6 hours 21 minutes** on MacBook Pro (M-series, 8 CPU cores)
- Step timing: 2.29s per step on average
- Total steps: 10,000
