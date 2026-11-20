"""Masked supervised and cross-view objectives for A2F-OOD.

Only observed labels contribute to supervised risks. Auxiliary representation
and generation objectives can also train on molecules without observed labels.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def fit_binary_class_weights(
    labels: Iterable[Sequence[float | None]], label_columns: Sequence[str],
) -> dict[str, Any]:
    """Fit per-task inverse-frequency weights on retained training labels only.

    N/(2*N_class) gives each class half the total training weight while keeping
    the mean sample weight at one. Missing labels are excluded per task. A task
    missing either class uses neutral weights, since its ratio is undefined.
    """
    counts = [[0, 0] for _ in label_columns]
    for row in labels:
        if len(row) != len(label_columns):
            raise ValueError("Training labels must match label_columns")
        for index, value in enumerate(row):
            if value is None or not math.isfinite(value):
                continue
            if value not in (0, 1):
                raise ValueError("Class-weight fitting requires binary 0/1 labels")
            counts[index][int(value)] += 1
    tasks = []
    for name, (negative, positive) in zip(label_columns, counts):
        both_classes = negative > 0 and positive > 0
        total = negative + positive
        weights = [total / (2 * negative), total / (2 * positive)] if both_classes else [1.0, 1.0]
        tasks.append({"label": name, "negative_count": negative, "positive_count": positive,
                      "negative_weight": weights[0], "positive_weight": weights[1],
                      "positive_to_negative_weight": weights[1] / weights[0],
                      "neutral_fallback": not both_classes})
    return {"mode": "balanced", "fitted_on": "retained_training_labels_only",
            "formula": "w_negative=N/(2*N_negative); w_positive=N/(2*N_positive)",
            "weights": [[t["negative_weight"], t["positive_weight"]] for t in tasks],
            "tasks": tasks}


def configured_binary_class_weights(config: dict[str, Any]) -> list[list[float]] | None:
    """Read fitted weights; fail clearly instead of silently training plain BCE."""
    if config["data"]["task_type"] not in {"binary", "multilabel"}:
        return None
    loss_config = config.get("loss", {})
    if loss_config.get("classification_weighting", "none") == "none":
        return None
    if loss_config.get("classification_weighting") != "balanced":
        raise ValueError("classification_weighting must be none or balanced")
    weights = loss_config.get("binary_class_weights")
    if weights is None:
        raise ValueError("Balanced BCE requires binary_class_weights fitted on training labels")
    return weights


def task_sample_losses(
    logits: Tensor,
    labels: Tensor,
    label_mask: Tensor,
    *,
    task_type: str,
    regression_loss: str = "mse",
    binary_class_weights: list[list[float]] | Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Return per-molecule risks and an observed-label molecule mask.

    Views and observed tasks are averaged within a molecule before group
    reduction. Unlabelled molecules have zero risk and are excluded by mask.
    """
    if logits.ndim == 2:
        logits = logits.unsqueeze(1)
    if logits.ndim != 3:
        raise ValueError("logits must have shape [B,O] or [B,V,O]")
    if labels.ndim == 1:
        labels = labels.unsqueeze(-1)
    if label_mask.ndim == 1:
        label_mask = label_mask.unsqueeze(-1)
    if labels.shape != label_mask.shape or labels.shape[0] != logits.shape[0]:
        raise ValueError("labels and label_mask must have matching batch/task shapes")
    observed = label_mask.bool() & torch.isfinite(labels)
    clean_labels = torch.where(observed, labels, torch.zeros_like(labels))
    if task_type == "multiclass":
        if labels.shape[1] != 1:
            raise ValueError("multiclass labels must contain one class-index column")
        sample_mask = observed[:, 0]
        classes = clean_labels[:, 0]
        valid_classes = classes[sample_mask]
        if valid_classes.numel() and (
            (valid_classes != valid_classes.round()).any()
            or (valid_classes < 0).any()
            or (valid_classes >= logits.shape[-1]).any()
        ):
            raise ValueError("multiclass labels must be integer class indices in [0, num_classes)")
        targets = classes.long().unsqueeze(1).expand(-1, logits.shape[1])
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
        ).reshape(logits.shape[0], logits.shape[1]).mean(1)
        return loss * sample_mask.to(loss.dtype), sample_mask
    if logits.shape[-1] != labels.shape[-1]:
        raise ValueError("Prediction outputs must match label columns")
    targets = clean_labels.unsqueeze(1).expand_as(logits)
    if task_type in {"binary", "multilabel"}:
        values = clean_labels[observed]
        if values.numel() and ((values != 0) & (values != 1)).any():
            raise ValueError("Binary and multilabel targets must be 0 or 1")
        element_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        if binary_class_weights is not None:
            weights = torch.as_tensor(binary_class_weights, dtype=logits.dtype, device=logits.device)
            if weights.shape != (logits.shape[-1], 2) or not torch.isfinite(weights).all() or (weights <= 0).any():
                raise ValueError("binary_class_weights must be finite positive [num_tasks, 2] negative/positive weights")
            element_loss = element_loss * torch.where(targets.bool(), weights[:, 1], weights[:, 0])
    elif task_type == "regression":
        if regression_loss == "mse":
            element_loss = F.mse_loss(logits, targets, reduction="none")
        elif regression_loss in {"huber", "smooth_l1"}:
            element_loss = F.smooth_l1_loss(logits, targets, reduction="none")
        else:
            raise ValueError("regression_loss must be 'mse' or 'huber'")
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
    task_losses = element_loss.mean(1)
    counts = observed.sum(-1)
    sample_loss = (task_losses * observed).sum(-1) / counts.clamp_min(1)
    return sample_loss, counts > 0


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    """Average observed entries, preserving a differentiable zero if empty."""
    return (values * mask.to(values.dtype)).sum() / mask.sum().clamp_min(1)


class GroupDRO(nn.Module):
    """Bounded group reweighting with an optional epoch-level adversary.

    The epoch mode starts from training group frequencies. Its minibatch
    risk uses q(group)/p_train(group), so ordinary shuffled batches estimate
    the intended group risk without renormalizing over whichever groups
    happen to appear. Each group's mean risk updates the adversary once per
    epoch, independently of group size. Batch mode preserves the old API.
    """
    def __init__(
        self, num_groups: int, step_size: float = 0.01, *,
        update_mode: str = "batch", group_counts: Tensor | list[int] | None = None,
        max_weight_ratio: float = 3.0,
    ) -> None:
        super().__init__()
        if num_groups < 1 or step_size < 0 or not torch.isfinite(torch.tensor(step_size)):
            raise ValueError("GroupDRO needs positive num_groups and nonnegative step_size")
        if update_mode not in {"batch", "epoch"}:
            raise ValueError("GroupDRO update_mode must be batch or epoch")
        if max_weight_ratio < 1 or not torch.isfinite(torch.tensor(max_weight_ratio)):
            raise ValueError("GroupDRO max_weight_ratio must be finite and >= 1")
        if update_mode == "epoch" and group_counts is None:
            raise ValueError("Epoch GroupDRO requires training group_counts")
        self.num_groups = int(num_groups)
        self.step_size = float(step_size)
        self.update_mode = update_mode
        self.max_weight_ratio = float(max_weight_ratio)
        self.has_sampling_prior = group_counts is not None
        counts = torch.ones(num_groups) if group_counts is None else torch.as_tensor(group_counts, dtype=torch.float32)
        if counts.shape != (num_groups,) or not torch.isfinite(counts).all() or (counts < 0).any() or counts.sum() <= 0:
            raise ValueError("GroupDRO group_counts must be finite nonnegative counts with a positive sum")
        prior = counts / counts.sum()
        self.register_buffer("sampling_prior", prior)
        self.register_buffer("group_weights", prior.clone())
        self.register_buffer("epoch_loss_sums", torch.zeros(num_groups))
        self.register_buffer("epoch_sample_counts", torch.zeros(num_groups))
        self.register_buffer("last_group_losses", torch.zeros(num_groups))

    @torch.no_grad()
    def begin_epoch(self) -> None:
        self.epoch_loss_sums.zero_()
        self.epoch_sample_counts.zero_()

    @torch.no_grad()
    def _update_weights(self, group_ids: Tensor, group_losses: Tensor) -> None:
        logits = self.group_weights.clamp_min(1e-30).log()
        logits[group_ids] += self.step_size * group_losses.to(logits)
        logits[self.sampling_prior == 0] = -torch.inf
        proposed = logits.softmax(0)
        # Mixing toward the prior bounds every sample's importance weight,
        # while preserving both nonnegativity and a unit total weight.
        largest_ratio = (proposed / self.sampling_prior.clamp_min(1e-30)).max()
        fraction = ((self.max_weight_ratio - 1) / (largest_ratio - 1).clamp_min(1e-12)).clamp(max=1)
        self.group_weights.copy_(self.sampling_prior + fraction * (proposed - self.sampling_prior))

    @torch.no_grad()
    def end_epoch(self, *, update: bool = True) -> dict[str, float]:
        present = self.epoch_sample_counts > 0
        self.last_group_losses[present] = self.epoch_loss_sums[present] / self.epoch_sample_counts[present]
        if self.training and update and self.update_mode == "epoch" and present.any():
            ids = present.nonzero(as_tuple=False).flatten()
            self._update_weights(ids, self.last_group_losses[ids])
        q, p = self.group_weights, self.sampling_prior
        return {
            "group_dro_max_importance": float((q / p.clamp_min(1e-30)).max()),
            "group_dro_effective_groups": float(1 / q.square().sum().clamp_min(1e-30)),
            "group_dro_kl_to_prior": float((q * (q.clamp_min(1e-30).log() - p.clamp_min(1e-30).log())).sum()),
            "group_dro_observed_groups": float(present.sum()),
        }

    def forward(
        self, sample_losses: Tensor, group_ids: Tensor, sample_mask: Tensor | None = None,
        *, update: bool = True,
    ) -> Tensor:
        if sample_losses.ndim != 1 or group_ids.shape != sample_losses.shape:
            raise ValueError("GroupDRO requires one loss and group id per molecule")
        if sample_mask is None:
            sample_mask = torch.ones_like(sample_losses, dtype=torch.bool)
        sample_mask = sample_mask.bool()
        unknown = ((group_ids < 0) | (group_ids >= self.num_groups)) & sample_mask
        if unknown.any():
            if self.training:
                raise ValueError("Training group id outside the fitted GroupDRO group set")
            return masked_mean(sample_losses, sample_mask)
        if not sample_mask.any():
            return sample_losses.sum() * 0.0
        present_ids = torch.unique(group_ids[sample_mask].long(), sorted=True)
        group_losses = torch.stack([
            sample_losses[(group_ids == group_id) & sample_mask].mean() for group_id in present_ids
        ])
        if self.training and update:
            with torch.no_grad():
                ids = group_ids[sample_mask].long()
                self.epoch_loss_sums.index_add_(0, ids, sample_losses[sample_mask].detach().to(self.epoch_loss_sums))
                self.epoch_sample_counts.index_add_(0, ids, torch.ones_like(ids, dtype=self.epoch_sample_counts.dtype))
                if self.update_mode == "batch":
                    self._update_weights(present_ids, group_losses.detach())
        if self.has_sampling_prior:
            ids = group_ids[sample_mask].long()
            if (self.sampling_prior[ids] <= 0).any():
                raise ValueError("Observed training group has zero fitted sampling probability")
            importance = (self.group_weights[ids] / self.sampling_prior[ids]).detach()
            return (importance * sample_losses[sample_mask]).mean()
        weights = self.group_weights[present_ids].detach().to(group_losses)
        weights = weights / weights.sum().clamp_min(1e-30)
        return (weights * group_losses).sum()


def atom_alignment_loss(
    graph_atoms: Tensor, smiles_atoms: Tensor, atom_mask: Tensor,
    *, temperature: float = 0.1, max_atoms: int = 512,
) -> Tensor:
    """Symmetric InfoNCE on real corresponding atoms, with bounded memory."""
    if temperature <= 0 or max_atoms < 1:
        raise ValueError("alignment temperature and max_atoms must be positive")
    if smiles_atoms.ndim == 3:
        smiles_atoms = smiles_atoms.unsqueeze(1)
    if graph_atoms.shape[:2] != atom_mask.shape:
        raise ValueError("graph atom mask shape does not match node embeddings")
    indices = atom_mask.reshape(-1).nonzero(as_tuple=False).flatten()
    if indices.numel() > max_atoms:
        positions = torch.linspace(0, indices.numel() - 1, max_atoms, device=indices.device).long()
        indices = indices[positions]
    if indices.numel() <= 1:
        return (graph_atoms.sum() + smiles_atoms.sum()) * 0.0
    graph = F.normalize(graph_atoms.reshape(-1, graph_atoms.shape[-1])[indices], dim=-1)
    labels = torch.arange(indices.numel(), device=graph.device)
    losses = []
    for view in range(smiles_atoms.shape[1]):
        smiles = F.normalize(smiles_atoms[:, view].reshape(-1, smiles_atoms.shape[-1])[indices], dim=-1)
        scores = graph @ smiles.T / temperature
        losses.append((F.cross_entropy(scores, labels) + F.cross_entropy(scores.T, labels)) * 0.5)
    return torch.stack(losses).mean()


def serialization_consistency_loss(logits: Tensor, embeddings: Tensor, *, task_type: str) -> Tensor:
    """Representation variance plus prediction Jensen-Shannon divergence."""
    if logits.ndim < 3 or logits.shape[1] < 2:
        return (logits.sum() + embeddings.sum()) * 0.0
    unit = F.normalize(embeddings, dim=-1)
    representation = (unit - unit.mean(1, keepdim=True)).square().mean()
    if task_type == "regression":
        prediction = (logits - logits.mean(1, keepdim=True)).square().mean()
    elif task_type == "multiclass":
        probabilities = logits.softmax(-1).clamp_min(1e-7)
        mixture = probabilities.mean(1, keepdim=True)
        prediction = (probabilities * (probabilities.log() - mixture.log())).sum(-1).mean()
    elif task_type in {"binary", "multilabel"}:
        positive = logits.sigmoid().clamp(1e-7, 1 - 1e-7)
        probabilities = torch.stack([positive, 1 - positive], dim=-1)
        mixture = probabilities.mean(1, keepdim=True)
        prediction = (probabilities * (probabilities.log() - mixture.log())).sum(-1).mean()
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
    return representation + prediction.clamp_min(0)


def compute_losses(
    outputs: dict[str, Any], batch: dict[str, Any], config: dict[str, Any],
    group_dro: GroupDRO | None = None, *, update_dro: bool | None = None,
) -> dict[str, Tensor]:
    """Return named losses; ``task`` includes fitted class weights when enabled."""
    loss_config = config.get("loss", {})
    fusion_config = config.get("fusion", {})
    task_type = config["data"]["task_type"]
    regression_loss = loss_config.get("regression_loss", "mse")
    class_weights = configured_binary_class_weights(config)
    logits = outputs["logits"]
    zero = logits.sum() * 0.0
    sample_losses, sample_mask = task_sample_losses(
        logits, batch["labels"], batch["label_mask"], task_type=task_type,
        regression_loss=regression_loss,
        binary_class_weights=class_weights,
    )
    task = masked_mean(sample_losses, sample_mask)
    task_unweighted = task
    if class_weights is not None:
        unweighted_losses, _ = task_sample_losses(
            logits, batch["labels"], batch["label_mask"], task_type=task_type,
        )
        task_unweighted = masked_mean(unweighted_losses, sample_mask)
    robust = task
    dro_risk = task
    if loss_config.get("use_group_dro", False):
        if group_dro is None:
            raise ValueError("use_group_dro requires a training-fitted GroupDRO instance")
        dro_risk = group_dro(sample_losses, batch["group_ids"], sample_mask,
                            update=True if update_dro is None else update_dro)
        strength = float(loss_config.get("group_dro_strength", 1.0))
        if not 0 <= strength <= 1:
            raise ValueError("loss.group_dro_strength must be in [0, 1]")
        robust = (1 - strength) * task + strength * dro_risk
    alignment = zero
    if fusion_config.get("atom_alignment", True) and loss_config.get("lambda_atom_alignment", 0) > 0:
        alignment = atom_alignment_loss(
            outputs["graph_atoms"], outputs["smiles_atoms"], batch["atom_mask"],
            temperature=float(loss_config.get("temperature", loss_config.get("alignment_temperature", 0.1))),
            max_atoms=int(loss_config.get("max_alignment_atoms", loss_config.get("alignment_max_atoms", 512))),
        )
    serial = zero
    if loss_config.get("lambda_serialization_consistency", 0) > 0:
        serial = serialization_consistency_loss(logits, outputs["fused_embedding"], task_type=task_type)
    cross_modal = zero
    if loss_config.get("lambda_cross_modal", 0) > 0:
        predicted_s = F.normalize(outputs["predicted_smiles_embedding"], dim=-1)
        predicted_g = F.normalize(outputs["predicted_graph_embedding"], dim=-1)
        target_s = F.normalize(outputs["smiles_embedding"].detach(), dim=-1)
        target_g = F.normalize(outputs["graph_embedding"].detach(), dim=-1)
        cross_modal = 0.5 * (
            F.smooth_l1_loss(predicted_s[:, None].expand_as(target_s), target_s)
            + F.smooth_l1_loss(predicted_g, target_g[:, None].expand_as(predicted_g))
        )
    generation = zero
    generation_logits = outputs.get("generation_logits")
    if generation_logits is not None and loss_config.get("lambda_generation", 0) > 0:
        targets = batch["target_ids"][:, 1:]
        if generation_logits.shape[:2] != targets.shape:
            raise ValueError("Decoder logits must predict target_ids[:, 1:]")
        valid_tokens = targets != int(config.get("data", {}).get("pad_id", 0))
        token_losses = F.cross_entropy(
            generation_logits.reshape(-1, generation_logits.shape[-1]),
            targets.reshape(-1), reduction="none",
        ).reshape_as(targets)
        generation = masked_mean(token_losses, valid_tokens)
    unimodal = zero
    if fusion_config.get("unimodal_auxiliary_heads", True) and loss_config.get("lambda_unimodal", 0) > 0:
        graph_losses, graph_mask = task_sample_losses(
            outputs["graph_logits"], batch["labels"], batch["label_mask"],
            task_type=task_type, regression_loss=regression_loss,
            binary_class_weights=class_weights,
        )
        smiles_losses, smiles_mask = task_sample_losses(
            outputs["smiles_logits"], batch["labels"], batch["label_mask"],
            task_type=task_type, regression_loss=regression_loss,
            binary_class_weights=class_weights,
        )
        unimodal = 0.5 * (masked_mean(graph_losses, graph_mask) + masked_mean(smiles_losses, smiles_mask))
    weighted = {
        "weighted_atom_alignment": float(loss_config.get("lambda_atom_alignment", 0)) * alignment,
        "weighted_serialization_consistency": float(loss_config.get("lambda_serialization_consistency", 0)) * serial,
        "weighted_cross_modal": float(loss_config.get("lambda_cross_modal", 0)) * cross_modal,
        "weighted_generation": float(loss_config.get("lambda_generation", 0)) * generation,
        "weighted_unimodal": float(loss_config.get("lambda_unimodal", 0)) * unimodal,
    }
    auxiliary = sum(weighted.values())
    total = robust + auxiliary
    return {"total": total, "task": task, "task_unweighted": task_unweighted,
            "robust": robust, "dro": dro_risk,
            "auxiliary_weighted": auxiliary, **weighted,
            "atom_alignment": alignment, "serialization_consistency": serial,
            "cross_modal": cross_modal, "generation": generation,
            "unimodal": unimodal, "num_labeled": sample_mask.sum()}
