from typing import List, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn

from .fsemicrf import FilteredSemiCRFLoss
from .graph import IntervalGraph
from .losses import down_weight_loss


class SpanGlobal(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int, sample_rate: float = 0.5, model_type: str = 'gss',
                 transition: bool = False, null_confidence_threshold: float = 0.9,
                 uncertain_top_k: int = 8, soft_filter: bool = True,
                 entity_types: Optional[List[str]] = None,
                 entity_confidence_threshold: float = 0.0,
                 type_confidence_thresholds: Optional[Dict[str, float]] = None,
                 type_null_confidence_thresholds: Optional[Dict[str, float]] = None,
                 type_length_ranges: Optional[Dict[str, List[int]]] = None,
                 focal_gamma_entity: float = 0.0,
                 focal_gamma_non_entity: float = 0.0):
        super().__init__()
        # mode = 'gss' (global span selection) or 'fsemicrf' (filtered semi-crf)
        self.sample_rate = sample_rate
        self.model_type = model_type
        self.num_classes = num_classes
        self.null_confidence_threshold = null_confidence_threshold
        self.uncertain_top_k = uncertain_top_k
        self.soft_filter = soft_filter
        self.entity_types = entity_types or []
        self.entity_confidence_threshold = float(entity_confidence_threshold)
        self.focal_gamma_entity = float(focal_gamma_entity)
        self.focal_gamma_non_entity = float(focal_gamma_non_entity)

        self.type_aliases = {
            'person': 'person', 'per': 'person',
            'organization': 'organization', 'org': 'organization',
            'location': 'location', 'loc': 'location', 'gpe': 'location',
            'else': 'else', 'misc': 'else', 'miscellaneous': 'else',
        }

        self.type_null_confidence_thresholds = {
            self.type_aliases.get(k.lower(), k.lower()): float(v)
            for k, v in (type_null_confidence_thresholds or {}).items()
        }

        self.type_confidence_thresholds = {
            self.type_aliases.get(k.lower(), k.lower()): float(v)
            for k, v in (type_confidence_thresholds or {}).items()
        }

        # Backward-compatible fallback: if explicit type confidence thresholds are not provided,
        # reuse type_null_confidence_thresholds for entity filtering.
        if len(self.type_confidence_thresholds) == 0 and len(self.type_null_confidence_thresholds) > 0:
            self.type_confidence_thresholds = dict(self.type_null_confidence_thresholds)

        self.type_length_ranges = {
            self.type_aliases.get(k.lower(), k.lower()): (int(v[0]), int(v[1]))
            for k, v in (type_length_ranges or {}).items()
            if isinstance(v, (list, tuple)) and len(v) == 2
        }

        # Per-label thresholds and length ranges (index 0 is O)
        threshold_by_label = torch.full((num_classes,), float(null_confidence_threshold), dtype=torch.float)
        entity_threshold_by_label = torch.full((num_classes,), float(self.entity_confidence_threshold), dtype=torch.float)
        entity_threshold_by_label[0] = 1.0
        min_len_by_label = torch.ones((num_classes,), dtype=torch.long)
        max_len_by_label = torch.full((num_classes,), 10_000, dtype=torch.long)

        for idx, entity in enumerate(self.entity_types, start=1):
            ent_key = self.type_aliases.get(str(entity).lower(), str(entity).lower())
            if ent_key in self.type_null_confidence_thresholds:
                threshold_by_label[idx] = float(self.type_null_confidence_thresholds[ent_key])
            if ent_key in self.type_confidence_thresholds:
                entity_threshold_by_label[idx] = float(self.type_confidence_thresholds[ent_key])
            if ent_key in self.type_length_ranges:
                min_l, max_l = self.type_length_ranges[ent_key]
                min_len_by_label[idx] = max(1, int(min_l))
                max_len_by_label[idx] = max(int(min_len_by_label[idx].item()), int(max_l))

        self.register_buffer("threshold_by_label", threshold_by_label)
        self.register_buffer("entity_threshold_by_label", entity_threshold_by_label)
        self.register_buffer("min_len_by_label", min_len_by_label)
        self.register_buffer("max_len_by_label", max_len_by_label)

        # Span Structured Prediction (Zaratiana et al., UM-IoS 2022)
        # Local classifier with local score
        if model_type == 'standard':
            self.scorer = nn.Linear(hidden_size, num_classes)
            transition = False

        # Global Span Selection for Named Entity Recognition (Zaratiana et al., UM-IoS 2022)
        # Local classifier + global score (same score for all spans)
        if model_type == 'gss':
            self.scorer = nn.Linear(hidden_size, num_classes + 1)

        # Filtered Semi-CRF (Zaratiana et al., EMNLP 2023)
        # Local classifier + global score (score is label-dependent)
        elif model_type == 'fsemicrf':
            self.scorer = nn.Linear(hidden_size, num_classes * 2)

        if transition:  # transition score for filtered semi-crf
            var = torch.zeros((num_classes + 2, num_classes + 2))  # +2 for start and end transitions
            self.transition_score = nn.Parameter(var)
        else:
            self.transition_score = None

    def compute_scores(self, span_rep: torch.Tensor):
        scores = self.scorer(span_rep)
        B, L, K, C = scores.shape
        scores = scores.view(B, L * K, C)

        local_scores = scores[:, :, :self.num_classes]
        local_probs = local_scores.softmax(dim=-1)
        local_labels = local_scores.max(-1).indices
        label_probs = torch.gather(local_probs, -1, local_labels.unsqueeze(-1)).squeeze(-1)
        entity_labels = local_scores[:, :, 1:].max(-1).indices + 1

        if self.model_type == 'standard':
            # use local score as global score
            global_scores = torch.gather(local_scores, -1, local_labels.unsqueeze(-1)).squeeze(-1)
            entity_scores = torch.gather(local_scores, -1, entity_labels.unsqueeze(-1)).squeeze(-1)
        elif self.model_type == 'gss':
            # same score for all spans
            global_scores = scores[:, :, -1]
            entity_scores = global_scores
        elif self.model_type == 'fsemicrf':
            # score is label-dependent
            label_global_scores = scores[:, :, self.num_classes:]
            global_scores = torch.gather(label_global_scores, -1, local_labels.unsqueeze(-1)).squeeze(-1)
            entity_scores = torch.gather(label_global_scores, -1, entity_labels.unsqueeze(-1)).squeeze(-1)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        return {
            "local_scores": local_scores,
            "local_probs": local_probs,
            "local_labels": local_labels,
            "label_probs": label_probs,
            "entity_labels": entity_labels,
            "global_scores": global_scores,
            "entity_scores": entity_scores,
        }

    def soft_filter_mask(
            self,
            local_labels: torch.Tensor,
            null_probs: torch.Tensor,
            valid_mask: torch.Tensor,
            label_probs: Optional[torch.Tensor],
            entity_labels: torch.Tensor,
            global_scores: torch.Tensor,
            entity_scores: torch.Tensor,
            span_idx: Optional[torch.Tensor] = None,
    ):
        keep_mask = valid_mask & (local_labels > 0)
        effective_labels = local_labels.clone()
        effective_scores = global_scores.clone()

        # Improvement #1: task-aware confidence thresholding for predicted entity labels
        if label_probs is not None:
            pred_threshold = self.entity_threshold_by_label[torch.clamp(local_labels, min=0)]
            keep_mask = keep_mask & (label_probs >= pred_threshold)

        # Length-constrained filtering for all non-null spans
        if span_idx is not None:
            span_lengths = (span_idx[:, :, 1] - span_idx[:, :, 0] + 1).long()
            label_min = self.min_len_by_label[torch.clamp(local_labels, min=0)]
            label_max = self.max_len_by_label[torch.clamp(local_labels, min=0)]
            valid_lengths = (span_lengths >= label_min) & (span_lengths <= label_max)
            keep_mask = keep_mask & valid_lengths

        # only apply uncertainty-aware soft filter to filtered semi-crf model
        if not self.soft_filter or self.model_type != 'fsemicrf':
            return keep_mask, effective_labels, effective_scores

        per_type_threshold = self.threshold_by_label[torch.clamp(entity_labels, min=0)]
        uncertain_null_mask = valid_mask & (local_labels == 0) & (null_probs <= per_type_threshold)

        if span_idx is not None:
            span_lengths = (span_idx[:, :, 1] - span_idx[:, :, 0] + 1).long()
            ent_label_min = self.min_len_by_label[torch.clamp(entity_labels, min=0)]
            ent_label_max = self.max_len_by_label[torch.clamp(entity_labels, min=0)]
            uncertain_null_mask = uncertain_null_mask & (span_lengths >= ent_label_min) & (span_lengths <= ent_label_max)

        selected_uncertain = torch.zeros_like(uncertain_null_mask, dtype=torch.bool)

        for b in range(uncertain_null_mask.size(0)):
            uncertain_idx = torch.where(uncertain_null_mask[b])[0]
            if uncertain_idx.nelement() == 0:
                continue

            if self.uncertain_top_k is not None and self.uncertain_top_k >= 0:
                k = min(self.uncertain_top_k, uncertain_idx.nelement())
                if k == 0:
                    continue
                uncertainty = 1.0 - null_probs[b, uncertain_idx]
                topk_idx = uncertainty.topk(k=k, largest=True).indices
                chosen = uncertain_idx[topk_idx]
            else:
                chosen = uncertain_idx

            selected_uncertain[b, chosen] = True

        effective_labels[selected_uncertain] = entity_labels[selected_uncertain]
        effective_scores[selected_uncertain] = entity_scores[selected_uncertain]
        keep_mask = keep_mask | selected_uncertain

        return keep_mask, effective_labels, effective_scores

    def loss(self, span_rep: torch.Tensor, span_label: torch.Tensor, span_idx: torch.Tensor) -> torch.Tensor:
        # compute scores
        out = self.compute_scores(span_rep)
        local_scores = out["local_scores"]
        global_scores = out["global_scores"]

        # filtering loss
        loss_local = down_weight_loss(
            local_scores,
            span_label,
            sample_rate=self.sample_rate,
            focal_gamma_entity=self.focal_gamma_entity,
            focal_gamma_non_entity=self.focal_gamma_non_entity,
        )

        # Filtered Semi-CRF (Zaratiana et al., EMNLP 2023)
        if self.model_type == 'fsemicrf' or self.model_type == 'gss':
            valid_mask = span_label != -1
            _, filter_labels, filter_scores = self.soft_filter_mask(
                local_labels=out["local_labels"],
                null_probs=out["local_probs"][:, :, 0],
                valid_mask=valid_mask,
                label_probs=out["label_probs"],
                entity_labels=out["entity_labels"],
                global_scores=global_scores,
                entity_scores=out["entity_scores"],
                span_idx=span_idx,
            )

            # loss function for filtered semi-crf and gss
            fsemicrf_loss_func = FilteredSemiCRFLoss()
            # compute loss
            loss_global = fsemicrf_loss_func(
                all_segment_idx=span_idx,
                all_segment_label=span_label,
                all_label_filter=filter_labels,
                all_scores=filter_scores,
                transition_score=self.transition_score,
            )
        else:  # standard model does not use global score
            loss_global = 0

        return loss_local + loss_global

    @torch.no_grad()
    def predict_batch(self, span_rep: torch.Tensor, span_idx: torch.Tensor, mask: torch.Tensor,
                      id_to_classes: Dict[int, str], decoding: str = 'best') -> List[Dict[str, Any]]:
        """
        Make predictions for spans.

        Parameters:
            span_rep (torch.Tensor): Tensor containing span representations.
            span_idx (torch.Tensor): Tensor containing span indices.
            mask (torch.Tensor): Mask tensor to filter invalid spans.
            id_to_classes (Dict[int, str]): Dictionary mapping label indices to label strings.
            decoding (str): Decoding method to use.

        Returns:
            List[Dict[str, Any]]: List of dictionaries containing span predictions, labels, and scores.
        """

        out = self.compute_scores(span_rep)
        keep_mask, labels, scores = self.soft_filter_mask(
            local_labels=out["local_labels"],
            null_probs=out["local_probs"][:, :, 0],
            valid_mask=mask,
            label_probs=out["label_probs"],
            entity_labels=out["entity_labels"],
            global_scores=out["global_scores"],
            entity_scores=out["entity_scores"],
            span_idx=span_idx,
        )

        # get all predictions
        # contains a list of dictionaries, each dictionary contains the span prediction, label, and score
        # keys for each element: 'spans', 'labels', 'scores'
        all_predictions = []

        for i in range(labels.size(0)):
            label_slice = labels[i]

            # Skip if no non-O spans are present
            if keep_mask[i].sum().item() == 0:
                all_predictions.append([])
                continue

            # Mask out invalid and non-interesting spans
            valid_span_mask = keep_mask[i]
            valid_spans = span_idx[i]
            pred_labels = torch.masked_select(label_slice, valid_span_mask).tolist()

            # Filter and list valid spans
            span_list = torch.masked_select(valid_spans, valid_span_mask.unsqueeze(-1)).view(-1, 2).tolist()
            span_list = [tuple(item) for item in span_list]

            # Filter and list valid scores
            span_scores = torch.masked_select(scores[i].view(-1), valid_span_mask)

            # combine spans, labels, and scores into a dictionary and decode
            prediction = {'spans': span_list, 'labels': pred_labels, 'scores': span_scores}
            decoded_prediction = self.decode_prediction(prediction, id_to_classes, decoding=decoding)

            # Append to list the spans, labels, and scores
            all_predictions.append(decoded_prediction)

        return all_predictions

    def decode_prediction(self, prediction, id_to_classes, decoding="best"):

        # get spans, labels, and scores
        spans, labels, scores = prediction.values()

        # create dict: span -> label
        span_lab_dict = {span: id_to_classes[label] for span, label in zip(spans, labels)}

        if len(spans) == 0:
            return []

        # make interval graph
        interval_graph = IntervalGraph(spans, scores, labels=labels, transition_matrix=self.transition_score)

        # if no decoding, return spans as is
        output_spans = spans

        # decode
        if decoding in ["best", "gss", "fsemicrf", "global"]:
            # get set of spans with highest score
            output_spans = interval_graph.best_path()
        elif decoding == "greedy":
            # get best span iteratively
            output_spans = interval_graph.greedy_search()
        elif decoding == "global_mean":
            # get set of spans with highest average score
            output_spans = interval_graph.exhaustive_search(scoring_func=np.mean)

        # get labels
        output_labels = [span_lab_dict[span] for span in output_spans]

        # make out put in the format (start, end, label)
        outputs = [(span[0], span[1], label) for span, label in zip(output_spans, output_labels)]

        return outputs
