import torch
import torch.nn.functional as F


def _masked_ce_with_focal(logits, y, mask, gamma=0.0):
    if mask.sum() == 0:
        return logits.new_tensor(0.0)

    logits_m = logits[mask]
    y_m = y[mask]
    ce = F.cross_entropy(logits_m, y_m, reduction='none')

    if gamma and gamma > 0:
        pt = torch.exp(-ce)
        ce = ((1 - pt) ** gamma) * ce

    return ce.sum()


def down_weight_loss(logits, y, sample_rate=0.5, focal_gamma_entity=0.0, focal_gamma_non_entity=0.0):
    # Flatten the logits and y tensors to 2 dimensions.
    logits = logits.contiguous().view(-1, logits.size(-1))
    y = y.view(-1)

    # Calculate the sample rate for non-entity samples.
    rate = 1 - sample_rate

    # entity and non-entity masks
    entity_mask = y > 0
    non_entity_mask = y == 0

    # Optional focal modulation for hard examples
    loss_entity = _masked_ce_with_focal(logits, y, entity_mask, gamma=focal_gamma_entity)
    loss_non_entity = _masked_ce_with_focal(logits, y, non_entity_mask, gamma=focal_gamma_non_entity)

    # Down-weight the non-entity loss by multiplying it with the rate (1 - sample_rate).
    # A lower sample_rate will result in a higher rate, reducing the contribution of the non-entity loss.
    # This down-weighting is applied to balance the impact of entity and non-entity samples in the total loss.
    weighted_loss_non_entity = loss_non_entity * rate

    return loss_entity + weighted_loss_non_entity
