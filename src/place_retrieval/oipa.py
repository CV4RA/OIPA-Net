"""Independent OIPA retrieval heads and place-supervised training objectives."""
import torch
from torch import nn
from torch.nn import functional as F

from .models import HeadPair
from .training import objective as retrieval_objective
from .training import place_masks


ABLATIONS = {
    "B0": (False, False, False),
    "B1": (False, False, False),
    "B2": (True, False, False),
    "B3": (False, True, False),
    "B4": (False, False, True),
    "AB": (True, True, False),
    "Ours": (True, True, True),
}


class WeightedResidual(nn.Module):
    def __init__(self, width, rank):
        super().__init__()
        self.score = nn.Linear(width, 1, bias=False)
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.up.weight)

    def forward(self, tokens, mask):
        logits = self.score(tokens).squeeze(-1).masked_fill(~mask, -torch.inf)
        weights = logits.softmax(dim=-1)
        pooled = (weights.unsqueeze(-1) * tokens).sum(dim=1)
        return self.up(F.gelu(self.down(pooled))), logits


class OIPAHeads(HeadPair):
    def __init__(self, width, method, rank=64):
        super().__init__(width, residual=method != "B0")
        self.method = method
        vision, text, self.mine_negatives = ABLATIONS[method]
        self.vision_pool = WeightedResidual(width, rank) if vision else None
        self.text_pool = WeightedResidual(width, rank) if text else None

    def uses_tokens(self, modality):
        return (self.vision_pool if modality == "image" else self.text_pool) is not None

    def forward(self, features, modality):
        global_features = features["global"].float()
        if self.method == "B0":
            return global_features, None
        value = getattr(self, modality)(global_features)
        pool = self.vision_pool if modality == "image" else self.text_pool
        logits = None
        if pool is not None:
            residual, logits = pool(features["tokens"].float(), features["mask"])
            value = value + residual
        return F.normalize(value, dim=-1), logits


def view_consistency(images, same_place):
    paired = same_place.triu(diagonal=1)
    differences = (1 - images @ images.T).masked_fill(~paired, 0)
    return differences.sum() / paired.sum().clamp_min(1)


def evidence_agreement(logits, tokens, mask, visual, positives):
    # Use the worst-supported target observation as a teacher for token weighting.
    with torch.no_grad():
        similarity = F.normalize(tokens.float(), dim=-1) @ visual.float().T
        support = similarity.masked_fill(~positives.T[:, None, :], torch.inf).amin(dim=-1)
        target = (support / 0.07).masked_fill(~mask, -torch.inf).softmax(dim=-1)
        multiple_views = positives.sum(dim=0).ge(2)
    log_weights = logits.log_softmax(dim=-1).masked_fill(~mask, 0)
    cross_entropy = -(target * log_weights).sum(dim=1)
    return (cross_entropy * multiple_views).sum() / multiple_views.sum().clamp_min(1)


def hard_negative_loss(images, texts, positives, allowed, count, margin, temperature):
    with torch.no_grad():
        positive_weights = positives.float()
        visual_prototypes = F.normalize(positive_weights.T @ images, dim=-1)
        text_prototypes = F.normalize(positive_weights @ texts, dim=-1)
        affinities = (images @ visual_prototypes.T + text_prototypes @ texts.T) / 2
        eligible = allowed & ~positives
    logits = images @ texts.T

    def direction(scores, mining_scores, positive, negative):
        indices = mining_scores.masked_fill(~negative, -torch.inf).topk(min(count, scores.shape[1]), dim=1).indices
        keep = negative.gather(1, indices)
        positive_score = scores.masked_fill(~positive, 0).sum(dim=1) / positive.sum(dim=1)
        penalty = F.softplus((scores.gather(1, indices) - positive_score[:, None] + margin) / temperature)
        return penalty.masked_fill(~keep, 0).sum() / keep.sum().clamp_min(1)

    mining_scores = logits.detach() + affinities
    return (direction(logits, mining_scores, positives, eligible)
            + direction(logits.T, mining_scores.T, positives.T, eligible.T)) / 2


def objective(heads, batch, rows, options):
    images, _ = heads(batch["image"], "image")
    texts, text_logits = heads(batch["text"], "text")
    temperature, tolerance = options["temperature"], options["tolerance_m"]
    loss = retrieval_objective(images, texts, rows["image"], rows["text"], options)
    if heads.vision_pool is not None:
        ids = [row["place_id"] for row in rows["image"]]
        same = torch.tensor([[a == b for b in ids] for a in ids], device=images.device)
        loss = loss + options["view_weight"] * view_consistency(images, same)
    if heads.text_pool is not None or heads.mine_negatives:
        positives, allowed = place_masks(rows["image"], rows["text"], images.device, tolerance)
    if heads.text_pool is not None:
        features = batch["text"]
        loss = loss + options["evidence_weight"] * evidence_agreement(
            text_logits, features["tokens"], features["mask"], batch["image"]["global"], positives
        )
    if heads.mine_negatives:
        loss = loss + options["negative_weight"] * hard_negative_loss(
            images, texts, positives, allowed, options["hard_k"], options["hard_margin"], temperature
        )
    return loss
