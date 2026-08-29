import math
from typing import NamedTuple, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import OPENOCC_LOSS
from .base_loss import BaseLoss


def _get_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert normalized [w, x, y, z] quaternions to rotation matrices."""
    q = F.normalize(quaternion, dim=-1)
    w, x, y, z = q.unbind(dim=-1)
    matrix = torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    )
    return matrix.reshape(*q.shape[:-1], 3, 3)


class LocalComplexityTargets(NamedTuple):
    """Per-Gaussian supervision generated from a dense semantic GT grid."""

    complexity: torch.Tensor
    weight: torch.Tensor
    occupied_fraction: torch.Tensor
    valid: torch.Tensor
    flat_indices: torch.Tensor
    sampled_labels: torch.Tensor
    sample_mass: torch.Tensor


'''
    dict(
        type="GaussianComplexityLoss",
        weight=0.1,
        pc_range=pc_range,
        num_classes=18,
        empty_label=17,
        min_loss_weight=0.25,
    )

    # loss_input_convertion must contain:
    # control_result="control_result"
'''

class GaussianLocalComplexityTarget(nn.Module):
    """Measure local semantic complexity inside every Gaussian support.

    Dense normalized offsets from ``sampling_levels`` are transformed by the
    Gaussian scale and rotation.  Optional ``outer_sampling_levels`` add six
    axis-aligned samples per radius, extending the target's receptive field
    without paying the cubic cost of another dense sampling grid.  Their GT
    labels form a Gaussian-weighted histogram, and normalized Shannon entropy
    is used as the soft complexity label.  Uniform regions therefore have
    target 0, while mixed semantic or occupied/empty boundaries have larger
    targets.

    The target is intentionally computed without gradients.  Score gradients
    train the Score head and its input features, not the sampling geometry used
    to define the label.
    """

    def __init__(
        self,
        pc_range: Sequence[float],
        voxel_size: Optional[Union[float, Sequence[float]]] = None,
        num_classes: int = 18,
        empty_label: int = 17,
        ignore_label: int = 255,
        sampling_levels: Sequence[float] = (-1.0, 0.0, 1.0),
        outer_sampling_levels: Sequence[float] = (),
        sampling_scale: float = 1.0,
        min_loss_weight: float = 0.25,
    ) -> None:
        super().__init__()
        if pc_range is None:
            raise ValueError("pc_range is required")
        if len(pc_range) != 6:
            raise ValueError("pc_range must contain 6 values")
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than 1")
        if not 0 <= empty_label < num_classes:
            raise ValueError("empty_label must be a valid class index")
        if not sampling_levels:
            raise ValueError("sampling_levels must not be empty")
        if outer_sampling_levels is None:
            outer_sampling_levels = ()
        if any(float(level) <= 0.0 for level in outer_sampling_levels):
            raise ValueError(
                "outer_sampling_levels must contain positive radii"
            )
        if sampling_scale <= 0:
            raise ValueError("sampling_scale must be positive")
        if not 0.0 <= min_loss_weight <= 1.0:
            raise ValueError("min_loss_weight must lie in [0, 1]")

        pc_range_tensor = torch.as_tensor(pc_range, dtype=torch.float32)
        if torch.any(pc_range_tensor[3:] <= pc_range_tensor[:3]):
            raise ValueError("Each pc_range maximum must exceed its minimum")

        if voxel_size is None:
            voxel_size_tensor = torch.empty(0, dtype=torch.float32)
        else:
            voxel_size_tensor = torch.as_tensor(voxel_size, dtype=torch.float32)
            if voxel_size_tensor.numel() == 1:
                voxel_size_tensor = voxel_size_tensor.repeat(3)
            if voxel_size_tensor.numel() != 3:
                raise ValueError("voxel_size must be a scalar or 3 values")
            if torch.any(voxel_size_tensor <= 0):
                raise ValueError("voxel_size must be positive")

        levels = torch.as_tensor(sampling_levels, dtype=torch.float32)
        ox, oy, oz = torch.meshgrid(levels, levels, levels, indexing="ij")
        offsets = torch.stack([ox, oy, oz], dim=-1).reshape(-1, 3)
        if outer_sampling_levels:
            outer_offsets = []
            for radius in outer_sampling_levels:
                radius = float(radius)
                for axis in range(3):
                    positive = torch.zeros(3, dtype=torch.float32)
                    positive[axis] = radius
                    outer_offsets.extend((positive, -positive))
            offsets = torch.cat(
                [offsets, torch.stack(outer_offsets, dim=0)], dim=0
            )
            # A configured outer radius may already be present in the dense
            # grid.  Keep every physical sample exactly once.
            offsets = torch.unique(offsets, dim=0)
        offsets = offsets * float(sampling_scale)
        sample_weights = torch.exp(-0.5 * offsets.square().sum(dim=-1))

        self.num_classes = int(num_classes)
        self.empty_label = int(empty_label)
        self.ignore_label = int(ignore_label)
        self.min_loss_weight = float(min_loss_weight)
        self.register_buffer("pc_range", pc_range_tensor, persistent=False)
        self.register_buffer("voxel_size", voxel_size_tensor, persistent=False)
        self.register_buffer("sample_offsets", offsets, persistent=False)
        self.register_buffer("sample_weights", sample_weights, persistent=False)

    @staticmethod
    def _normalize_gt_shape(tensor: torch.Tensor, name: str) -> torch.Tensor:
        # Accept [B, X, Y, Z] and the occasionally used [B, 1, X, Y, Z].
        if tensor.ndim == 5 and tensor.shape[1] == 1:
            tensor = tensor[:, 0]
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 4:
            raise ValueError(
                f"{name} must have shape [B, X, Y, Z], got {tensor.shape}"
            )
        return tensor

    def _resolve_voxel_size(
        self, gt_shape: Sequence[int], device: torch.device
    ) -> torch.Tensor:
        extent = (self.pc_range[3:] - self.pc_range[:3]).to(device)
        derived = extent / torch.as_tensor(
            gt_shape, dtype=torch.float32, device=device
        )
        if self.voxel_size.numel() == 0:
            return derived
        configured = self.voxel_size.to(device)
        if not torch.allclose(configured, derived, rtol=1e-4, atol=1e-5):
            raise ValueError(
                "voxel_size is inconsistent with pc_range and GT shape: "
                f"configured={configured.tolist()}, derived={derived.tolist()}"
            )
        return configured

    @torch.no_grad()
    def forward(
        self,
        gaussians,
        gt_semantics: torch.Tensor,
        gt_valid_mask: Optional[torch.Tensor] = None,
    ) -> LocalComplexityTargets:
        gt_semantics = self._normalize_gt_shape(gt_semantics, "gt_semantics")
        if gt_valid_mask is not None:
            gt_valid_mask = self._normalize_gt_shape(
                gt_valid_mask, "gt_valid_mask"
            ).bool()

        means = gaussians.means.detach().float()
        scales = gaussians.scales.detach().float()
        rotations = gaussians.rotations.detach().float()
        if means.ndim != 3 or means.shape[-1] != 3:
            raise ValueError("Gaussian means must have shape [B, N, 3]")
        if scales.shape != means.shape:
            raise ValueError("Gaussian scales must have shape [B, N, 3]")
        if rotations.shape != (*means.shape[:2], 4):
            raise ValueError("Gaussian rotations must have shape [B, N, 4]")

        batch_size, num_gaussians = means.shape[:2]
        if gt_semantics.shape[0] != batch_size:
            raise ValueError(
                "GT and Gaussians must have the same batch size, got "
                f"{gt_semantics.shape[0]} and {batch_size}"
            )
        if gt_valid_mask is not None and gt_valid_mask.shape != gt_semantics.shape:
            raise ValueError("gt_valid_mask must have the same shape as GT")

        device = means.device
        gt_semantics = gt_semantics.to(device=device, dtype=torch.long)
        if gt_valid_mask is not None:
            gt_valid_mask = gt_valid_mask.to(device=device)

        offsets = self.sample_offsets.to(device=device, dtype=means.dtype)
        local_points = scales.unsqueeze(2) * offsets.view(1, 1, -1, 3)
        rotation_matrix = _get_rotation_matrix(rotations).transpose(-1, -2)
        rotated_points = torch.matmul(
            rotation_matrix.unsqueeze(2), local_points.unsqueeze(-1)
        ).squeeze(-1)
        sample_points = means.unsqueeze(2) + rotated_points

        grid_shape = gt_semantics.shape[-3:]
        voxel_size = self._resolve_voxel_size(grid_shape, device)
        pc_min = self.pc_range[:3].to(device)
        indices = torch.floor(
            (sample_points - pc_min.view(1, 1, 1, 3))
            / voxel_size.view(1, 1, 1, 3)
        ).long()

        spatial_valid = torch.ones(
            indices.shape[:-1], dtype=torch.bool, device=device
        )
        for axis, axis_size in enumerate(grid_shape):
            spatial_valid &= indices[..., axis] >= 0
            spatial_valid &= indices[..., axis] < axis_size

        safe_indices = indices.clone()
        for axis, axis_size in enumerate(grid_shape):
            safe_indices[..., axis].clamp_(0, axis_size - 1)
        flat_indices = (
            safe_indices[..., 0] * (grid_shape[1] * grid_shape[2])
            + safe_indices[..., 1] * grid_shape[2]
            + safe_indices[..., 2]
        )

        flat_gt = gt_semantics.reshape(batch_size, -1)
        sampled_labels = flat_gt.gather(
            1, flat_indices.reshape(batch_size, -1)
        ).reshape(batch_size, num_gaussians, -1)

        label_valid = sampled_labels != self.ignore_label
        label_valid &= sampled_labels >= 0
        label_valid &= sampled_labels < self.num_classes
        sample_valid = spatial_valid & label_valid

        if gt_valid_mask is not None:
            flat_mask = gt_valid_mask.reshape(batch_size, -1)
            sampled_mask = flat_mask.gather(
                1, flat_indices.reshape(batch_size, -1)
            ).reshape_as(sample_valid)
            sample_valid &= sampled_mask

        sample_mass = self.sample_weights.to(
            device=device, dtype=means.dtype
        ).view(1, 1, -1)
        sample_mass = sample_mass * sample_valid.to(means.dtype)

        safe_labels = sampled_labels.clamp(0, self.num_classes - 1)
        histogram = means.new_zeros(
            batch_size, num_gaussians, self.num_classes
        )
        histogram.scatter_add_(2, safe_labels, sample_mass)
        total_mass = histogram.sum(dim=-1, keepdim=True)
        valid_gaussian = total_mass.squeeze(-1) > 0
        distribution = histogram / total_mass.clamp_min(1e-8)

        entropy_terms = torch.where(
            distribution > 0,
            distribution * distribution.clamp_min(1e-8).log(),
            torch.zeros_like(distribution),
        )
        complexity = -entropy_terms.sum(dim=-1) / math.log(self.num_classes)
        complexity = torch.where(
            valid_gaussian, complexity, torch.zeros_like(complexity)
        ).clamp_(0.0, 1.0)

        occupied_fraction = 1.0 - distribution[..., self.empty_label]
        occupied_fraction = torch.where(
            valid_gaussian,
            occupied_fraction,
            torch.zeros_like(occupied_fraction),
        )
        loss_weight = self.min_loss_weight + (
            1.0 - self.min_loss_weight
        ) * occupied_fraction
        loss_weight = loss_weight * valid_gaussian.to(loss_weight.dtype)

        return LocalComplexityTargets(
            complexity=complexity,
            weight=loss_weight,
            occupied_fraction=occupied_fraction,
            valid=valid_gaussian,
            flat_indices=flat_indices,
            sampled_labels=safe_labels,
            sample_mass=sample_mass,
        )


@OPENOCC_LOSS.register_module()
class GaussianComplexityLoss(BaseLoss):
    """Entropy supervision with a training-only semantic difficulty residual.

    Local GT entropy remains the primary Score target.  Once semantic
    predictions become meaningful, the corresponding decoder prediction adds
    a small stop-gradient residual in locally misclassified occupied regions:

        target = H + lambda(t) * (1 - H) * E

    where E is the Gaussian-local mean of ``1 - p(gt_class)`` over non-empty
    voxels.  This changes neither the controller architecture nor inference;
    it only teaches Score to prioritize complex regions that the current
    occupancy representation still finds difficult.
    """

    def __init__(
        self,
        weight: float = 0.1,
        pc_range: Sequence[float] = None,
        voxel_size: Optional[Union[float, Sequence[float]]] = None,
        num_classes: int = 18,
        empty_label: int = 17,
        ignore_label: int = 255,
        sampling_levels: Sequence[float] = (-1.0, 0.0, 1.0),
        outer_sampling_levels: Sequence[float] = (),
        sampling_scale: float = 1.0,
        min_loss_weight: float = 0.25,
        use_camera_mask: bool = True,
        difficulty_max_weight: float = 0.0,
        difficulty_start_iter: int = 7000,
        difficulty_ramp_iters: int = 7000,
        difficulty_class_weights: Optional[Sequence[float]] = None,
        difficulty_class_weight_power: float = 1.0,
        input_dict=None,
        **kwargs,
    ) -> None:
        if not 0.0 <= difficulty_max_weight <= 1.0:
            raise ValueError("difficulty_max_weight must lie in [0, 1]")
        if difficulty_start_iter < 0:
            raise ValueError("difficulty_start_iter must be non-negative")
        if difficulty_ramp_iters < 0:
            raise ValueError("difficulty_ramp_iters must be non-negative")
        if difficulty_class_weight_power <= 0.0:
            raise ValueError(
                "difficulty_class_weight_power must be positive"
            )
        if input_dict is None:
            input_dict = {
                "control_result": "control_result",
                "metas": "metas",
            }
            # Preserve backward compatibility for other dataset configs that
            # use pure entropy supervision.  Prediction tensors are consumed
            # only when the correction is explicitly enabled.
            if difficulty_max_weight > 0.0:
                input_dict.update(
                    {
                        "pred_occ": "pred_occ",
                        "prediction_layers": "prediction_layers",
                        "global_iter": "global_iter",
                    }
                )
        super().__init__(weight=weight, input_dict=input_dict)
        self.use_camera_mask = bool(use_camera_mask)
        self.empty_label = int(empty_label)
        self.num_classes = int(num_classes)
        self.difficulty_max_weight = float(difficulty_max_weight)
        self.difficulty_start_iter = int(difficulty_start_iter)
        self.difficulty_ramp_iters = int(difficulty_ramp_iters)
        if difficulty_class_weights is None:
            class_weights = torch.empty(0, dtype=torch.float32)
        else:
            class_weights = torch.as_tensor(
                difficulty_class_weights, dtype=torch.float32
            )
            if class_weights.numel() != self.num_classes:
                raise ValueError(
                    "difficulty_class_weights must contain exactly "
                    f"{self.num_classes} values"
                )
            if not torch.isfinite(class_weights).all() or torch.any(
                class_weights <= 0.0
            ):
                raise ValueError(
                    "difficulty_class_weights must be finite and positive"
                )
            class_weights = class_weights.pow(
                float(difficulty_class_weight_power)
            )
            occupied_mask = torch.ones(
                self.num_classes, dtype=torch.bool
            )
            occupied_mask[self.empty_label] = False
            class_weights = class_weights / class_weights[
                occupied_mask
            ].mean().clamp_min(1e-8)
            # Empty samples are excluded below; zeroing this entry makes that
            # contract explicit and guards future refactors.
            class_weights[self.empty_label] = 0.0
        self.register_buffer(
            "difficulty_class_weights",
            class_weights,
            persistent=False,
        )
        self.target_generator = GaussianLocalComplexityTarget(
            pc_range=pc_range,
            voxel_size=voxel_size,
            num_classes=num_classes,
            empty_label=empty_label,
            ignore_label=ignore_label,
            sampling_levels=sampling_levels,
            outer_sampling_levels=outer_sampling_levels,
            sampling_scale=sampling_scale,
            min_loss_weight=min_loss_weight,
        )
        self.loss_func = self.loss_score

    @staticmethod
    def _global_iteration_value(global_iter) -> Optional[int]:
        if global_iter is None:
            return None
        if isinstance(global_iter, torch.Tensor):
            if global_iter.numel() != 1:
                raise ValueError("global_iter tensor must contain one value")
            return int(global_iter.detach().item())
        return int(global_iter)

    def _difficulty_weight(self, global_iter) -> float:
        iteration = self._global_iteration_value(global_iter)
        if iteration is None or iteration <= self.difficulty_start_iter:
            return 0.0
        if self.difficulty_ramp_iters == 0:
            return self.difficulty_max_weight
        progress = min(
            1.0,
            (iteration - self.difficulty_start_iter)
            / float(self.difficulty_ramp_iters),
        )
        return self.difficulty_max_weight * progress

    @torch.no_grad()
    def _local_prediction_error(
        self,
        prediction: torch.Tensor,
        targets: LocalComplexityTargets,
        return_unweighted: bool = False,
    ):
        """Sample detached joint class probabilities at target locations."""
        if prediction.ndim != 3:
            raise ValueError(
                "Each pred_occ tensor must have shape [B, C, V], got "
                f"{prediction.shape}"
            )
        batch_size, num_classes, num_voxels = prediction.shape
        if num_classes != self.num_classes:
            raise ValueError(
                f"Expected {self.num_classes} prediction classes, got "
                f"{num_classes}"
            )
        if targets.flat_indices.shape != targets.sampled_labels.shape:
            raise ValueError(
                "Local target indices and labels must have identical shapes"
            )
        if targets.sample_mass.shape != targets.flat_indices.shape:
            raise ValueError(
                "Local target sample_mass must match sampled indices"
            )
        if targets.flat_indices.shape[0] != batch_size:
            raise ValueError(
                "Prediction and local targets must have the same batch size"
            )
        if torch.any(targets.flat_indices >= num_voxels):
            raise ValueError(
                "Local target index exceeds flattened prediction size"
            )

        # Index in the prediction's native precision first.  Casting the
        # complete dense D1/D2 volume to FP32 would create a large temporary
        # tensor merely to retain K local samples per Gaussian.
        probabilities_by_voxel = prediction.detach().transpose(1, 2)
        batch_index = torch.arange(
            batch_size, device=prediction.device
        ).view(batch_size, 1, 1)
        true_class_probability = probabilities_by_voxel[
            batch_index,
            targets.flat_indices,
            targets.sampled_labels,
        ].float().clamp_(0.0, 1.0)
        nonempty_mass = targets.sample_mass * (
            targets.sampled_labels != self.empty_label
        ).to(targets.sample_mass.dtype)
        denominator = nonempty_mass.sum(dim=-1)
        unweighted_local_error = (
            nonempty_mass * (1.0 - true_class_probability)
        ).sum(dim=-1) / denominator.clamp_min(1e-8)
        unweighted_local_error = torch.where(
            denominator > 0.0,
            unweighted_local_error,
            torch.zeros_like(unweighted_local_error),
        ).clamp_(0.0, 1.0)

        if self.difficulty_class_weights.numel() == 0:
            balanced_local_error = unweighted_local_error
        else:
            class_weights = self.difficulty_class_weights.to(
                device=prediction.device,
                dtype=nonempty_mass.dtype,
            )
            sampled_class_weights = class_weights[
                targets.sampled_labels
            ]
            balanced_local_error = (
                nonempty_mass
                * sampled_class_weights
                * (1.0 - true_class_probability)
            ).sum(dim=-1) / denominator.clamp_min(1e-8)
            balanced_local_error = torch.where(
                denominator > 0.0,
                balanced_local_error,
                torch.zeros_like(balanced_local_error),
            ).clamp_(0.0, 1.0)

        if return_unweighted:
            return balanced_local_error, unweighted_local_error
        return balanced_local_error

    @staticmethod
    def _prediction_stage_map(pred_occ, prediction_layers):
        if not isinstance(pred_occ, (list, tuple)):
            raise TypeError("pred_occ must be a list of decoder predictions")
        if not isinstance(prediction_layers, (list, tuple)):
            raise TypeError("prediction_layers must be a list or tuple")
        if len(pred_occ) != len(prediction_layers):
            raise ValueError(
                "pred_occ and prediction_layers must have the same length, "
                f"got {len(pred_occ)} and {len(prediction_layers)}"
            )
        stage_map = {}
        for prediction, layer in zip(pred_occ, prediction_layers):
            layer = int(layer)
            if layer in stage_map:
                raise ValueError(f"Duplicate prediction layer {layer}")
            stage_map[layer] = prediction
        return stage_map

    def loss_score(
        self,
        control_result,
        metas,
        pred_occ=None,
        prediction_layers=None,
        global_iter=None,
    ):
        if isinstance(control_result, dict):
            control_result = [control_result]
        if not isinstance(control_result, (list, tuple)):
            raise TypeError(
                "control_result must be a stage list returned by "
                "GaussianOccEncoder"
            )
        if len(control_result) == 0:
            raise ValueError(
                "GaussianComplexityLoss received no adaptive-control stages"
            )

        gt_semantics = metas["occ_label"]
        gt_valid_mask = (
            metas.get("occ_cam_mask") if self.use_camera_mask else None
        )
        prediction_stage_map = {}
        if self.difficulty_max_weight > 0.0:
            prediction_stage_map = self._prediction_stage_map(
                pred_occ, prediction_layers
            )
        scheduled_difficulty_weight = self._difficulty_weight(global_iter)
        stage_losses = []
        for stage_index, stage_result in enumerate(control_result):
            if not isinstance(stage_result, dict):
                raise TypeError(
                    f"control_result[{stage_index}] must be a dictionary"
                )
            missing = {"score_logits", "gaussian"} - stage_result.keys()
            if missing:
                raise KeyError(
                    f"control_result[{stage_index}] is missing {sorted(missing)}"
                )
            logits = stage_result["score_logits"]
            stage_gaussians = stage_result["gaussian"]
            targets = self.target_generator(
                stage_gaussians,
                gt_semantics=gt_semantics,
                gt_valid_mask=gt_valid_mask,
            )
            control_stage = stage_result.get("control_stage")
            if not isinstance(control_stage, torch.Tensor):
                raise KeyError(
                    f"control_result[{stage_index}] is missing control_stage"
                )
            unique_control_stages = torch.unique(
                control_stage.detach().to(dtype=torch.long)
            )
            if unique_control_stages.numel() != 1:
                raise ValueError(
                    "Every batch item must share one control stage"
                )
            control_stage_index = int(unique_control_stages.item())

            local_prediction_error = torch.zeros_like(targets.complexity)
            unweighted_local_prediction_error = torch.zeros_like(
                targets.complexity
            )
            applied_difficulty_weight = 0.0
            stage_prediction = prediction_stage_map.get(control_stage_index)
            if (
                scheduled_difficulty_weight > 0.0
                and stage_prediction is not None
            ):
                (
                    local_prediction_error,
                    unweighted_local_prediction_error,
                ) = self._local_prediction_error(
                    stage_prediction,
                    targets,
                    return_unweighted=True,
                )
                applied_difficulty_weight = scheduled_difficulty_weight
            corrected_complexity = (
                targets.complexity
                + applied_difficulty_weight
                * (1.0 - targets.complexity)
                * local_prediction_error
            ).clamp(0.0, 1.0)

            # The monitor runs after loss construction.  Store detached
            # diagnostics without changing the encoder control contract.
            stage_result["score_semantic_entropy_target"] = (
                targets.complexity.detach()
            )
            stage_result["score_target_valid_mask"] = targets.valid.detach()
            stage_result["score_local_prediction_error_target"] = (
                local_prediction_error.detach()
            )
            stage_result[
                "score_local_prediction_error_unweighted_target"
            ] = unweighted_local_prediction_error.detach()
            stage_result["score_complexity_target"] = (
                corrected_complexity.detach()
            )
            stage_result["score_difficulty_weight"] = torch.full(
                (logits.shape[0], 1),
                applied_difficulty_weight,
                dtype=torch.float32,
                device=logits.device,
            )
            if logits.shape != targets.complexity.shape:
                raise ValueError(
                    "Score logits and targets must have identical [B, N] "
                    f"shape, got {logits.shape} and {targets.complexity.shape}"
                )
            per_gaussian = F.binary_cross_entropy_with_logits(
                logits.float(), corrected_complexity, reduction="none"
            )
            denominator = targets.weight.sum().clamp_min(1.0)
            stage_loss = (per_gaussian * targets.weight).sum() / denominator
            stage_losses.append(stage_loss)

        return torch.stack(stage_losses).mean()


@OPENOCC_LOSS.register_module()
class GaussianSplitMassLoss(BaseLoss):
    """Softly discourage learned Split from creating excessive optical mass.

    This is deliberately not a hard conservation constraint.  Occupancy loss
    may still move the two-child total mass away from the parent when useful,
    while ``abs(log(child_mass / parent_mass))`` makes large multiplicative
    changes increasingly expensive.  Only parents that are actually split
    contribute to the loss.
    """

    def __init__(
        self,
        weight: float = 0.02,
        ratio_eps: float = 1e-6,
        input_dict=None,
    ) -> None:
        if weight < 0.0:
            raise ValueError("weight must be non-negative")
        if ratio_eps <= 0.0:
            raise ValueError("ratio_eps must be positive")
        if input_dict is None:
            input_dict = {"control_result": "control_result"}
        super().__init__(weight=weight, input_dict=input_dict)
        self.ratio_eps = float(ratio_eps)
        self.loss_func = self.loss_mass

    def loss_mass(self, control_result):
        if isinstance(control_result, dict):
            control_result = [control_result]
        if not isinstance(control_result, (list, tuple)):
            raise TypeError(
                "control_result must be a stage list returned by "
                "GaussianOccEncoder"
            )
        if len(control_result) == 0:
            raise ValueError(
                "GaussianSplitMassLoss received no adaptive-control stages"
            )

        stage_losses = []
        for stage_index, stage_result in enumerate(control_result):
            if not isinstance(stage_result, dict):
                raise TypeError(
                    f"control_result[{stage_index}] must be a dictionary"
                )
            missing = {
                "learned_split_mass_ratio",
                "split_mask",
            } - stage_result.keys()
            if missing:
                raise KeyError(
                    f"control_result[{stage_index}] is missing "
                    f"{sorted(missing)}"
                )

            mass_ratio = stage_result["learned_split_mass_ratio"].float()
            split_mask = stage_result["split_mask"].bool()
            if mass_ratio.shape != split_mask.shape:
                raise ValueError(
                    "learned_split_mass_ratio and split_mask must have "
                    f"identical [B, N] shapes, got {mass_ratio.shape} and "
                    f"{split_mask.shape}"
                )
            selected_ratio = mass_ratio[split_mask]
            if selected_ratio.numel() == 0:
                # Retain a valid zero tensor on the correct device.  The
                # Split generator already has an explicit zero DDP dependency
                # for iterations without selected parents.
                stage_losses.append(mass_ratio.sum() * 0.0)
                continue
            if not torch.isfinite(selected_ratio).all():
                raise FloatingPointError(
                    "Learned Split produced a non-finite density mass ratio"
                )
            log_ratio = torch.log(selected_ratio.clamp_min(self.ratio_eps))
            stage_losses.append(log_ratio.abs().mean())

        return torch.stack(stage_losses).mean()


@OPENOCC_LOSS.register_module()
class GaussianSplitSeparationLoss(BaseLoss):
    """Keep learned sibling Gaussians spatially distinct but local.

    The adaptive Score/threshold policy still decides whether a parent is
    split.  This small stability term only acts after that decision: a hinge
    prevents the two normalized child offsets from collapsing, while a weaker
    midpoint term keeps the pair approximately centred on its parent.  It does
    not force semantic disagreement or alter optical-density mass.
    """

    def __init__(
        self,
        weight: float = 0.02,
        min_normalized_distance: float = 0.5,
        midpoint_weight: float = 0.25,
        input_dict=None,
    ) -> None:
        if weight < 0.0:
            raise ValueError("weight must be non-negative")
        if not 0.0 < min_normalized_distance <= 2.0:
            raise ValueError(
                "min_normalized_distance must lie inside (0, 2]"
            )
        if midpoint_weight < 0.0:
            raise ValueError("midpoint_weight must be non-negative")
        if input_dict is None:
            input_dict = {"control_result": "control_result"}
        super().__init__(weight=weight, input_dict=input_dict)
        self.min_normalized_distance = float(min_normalized_distance)
        self.midpoint_weight = float(midpoint_weight)
        self.loss_func = self.loss_separation

    def loss_separation(self, control_result):
        if isinstance(control_result, dict):
            control_result = [control_result]
        if not isinstance(control_result, (list, tuple)):
            raise TypeError(
                "control_result must be a stage list returned by "
                "GaussianOccEncoder"
            )
        if len(control_result) == 0:
            raise ValueError(
                "GaussianSplitSeparationLoss received no adaptive-control "
                "stages"
            )

        stage_losses = []
        for stage_index, stage_result in enumerate(control_result):
            if not isinstance(stage_result, dict):
                raise TypeError(
                    f"control_result[{stage_index}] must be a dictionary"
                )
            required = {
                "learned_split_normalized_center_distance",
                "learned_split_normalized_midpoint_distance",
                "score_logits",
            }
            missing = required - stage_result.keys()
            if missing:
                raise KeyError(
                    f"control_result[{stage_index}] is missing "
                    f"{sorted(missing)}"
                )

            center_distance = stage_result[
                "learned_split_normalized_center_distance"
            ].float()
            midpoint_distance = stage_result[
                "learned_split_normalized_midpoint_distance"
            ].float()
            if center_distance.shape != midpoint_distance.shape:
                raise ValueError(
                    "Learned split center and midpoint distances must have "
                    "identical shapes"
                )
            if center_distance.ndim != 2:
                raise ValueError(
                    "Learned split distances must have shape [B, K]"
                )
            if center_distance.numel() == 0:
                stage_losses.append(
                    stage_result["score_logits"].sum() * 0.0
                )
                continue
            if (
                not torch.isfinite(center_distance).all()
                or not torch.isfinite(midpoint_distance).all()
            ):
                raise FloatingPointError(
                    "Learned Split produced non-finite sibling distances"
                )

            collapse_penalty = F.relu(
                self.min_normalized_distance - center_distance
            ).square()
            centering_penalty = midpoint_distance.square()
            stage_losses.append(
                collapse_penalty.mean()
                + self.midpoint_weight * centering_penalty.mean()
            )

        return torch.stack(stage_losses).mean()
