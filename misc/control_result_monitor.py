from typing import Dict, List, Optional, Sequence

import torch


_REQUIRED_FIELDS = {
    "score_logits",
    "score",
    "tau_1",
    "tau_2",
    "score_mean",
    "score_std",
    "low_candidate_mask",
    "high_candidate_mask",
    "last_operation_before",
    "operation_age_before",
    "prune_evidence_before",
    "prune_evidence_after_candidate",
    "split_provenance_mask",
    "merge_provenance_mask",
    "split_reverse_block_mask",
    "scale_feasible_mask",
    "children_inside_pc_range_mask",
    "split_mask",
    "gaussian_opacity",
    "density_mass",
    "low_opacity_mask",
    "low_density_mass_mask",
    "delete_candidate_mask",
    "delete_protected_mask",
    "delete_mask",
    "merge_candidate_mask",
    "merge_reverse_block_mask",
    "merge_member_mask",
    "merge_partner_index",
    "merge_pairs",
    "merge_geometry_similarity",
    "merge_semantic_similarity",
    "opacity_delete_threshold",
    "density_mass_delete_threshold",
    "require_consecutive_prune_evidence",
    "prune_min_consecutive_stages",
    "merge_geometry_threshold",
    "merge_semantic_threshold",
    "num_gaussians_before",
    "num_split",
    "num_deleted",
    "num_merge_pairs",
    "num_split_reverse_blocked",
    "num_delete_protected",
    "num_merge_reverse_blocked",
    "num_gaussians_after",
    "control_enabled",
    "split_enabled",
    "delete_enabled",
    "merge_enabled",
    "control_stage",
    "gaussian",
}

_SPLIT_DIAGNOSTIC_FIELDS = {
    "split_gain_per_axis",
    "split_gain_ratio_per_axis",
    "best_split_gain",
    "best_split_gain_ratio",
    "best_split_axis",
    "current_split_axis",
    "current_axis_split_gain",
    "current_axis_split_gain_ratio",
    "split_gain_valid_mask",
    "split_gain_ratio_low_threshold",
}

_LEARNED_SPLIT_FIELDS = {
    "learned_split_mass_ratio",
    "learned_split_position_clamped_mask",
    "learned_split_enabled",
}

_SCORE_TARGET_FIELDS = {
    "score_semantic_entropy_target",
    "score_local_prediction_error_target",
    "score_complexity_target",
}


def _require_tensor(value, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value).__name__}")
    return value


@torch.no_grad()
def validate_control_result(
    control_result: Sequence[Dict],
    expect_score_grad: Optional[bool] = None,
    score_atol: float = 5e-4,
) -> List[Dict[str, float]]:
    """Validate the encoder control contract and return compact statistics.

    This function intentionally performs GPU-to-CPU synchronization and should
    therefore be called at the logging interval, not on every forward pass.
    It never detaches or replaces tensors stored in ``control_result``.
    """
    if not isinstance(control_result, (list, tuple)):
        raise TypeError("control_result must be a list or tuple of stage dictionaries")

    summaries = []
    for expected_stage, stage_result in enumerate(control_result):
        prefix = f"control_result[{expected_stage}]"
        if not isinstance(stage_result, dict):
            raise TypeError(f"{prefix} must be a dictionary")
        missing = _REQUIRED_FIELDS - stage_result.keys()
        if missing:
            raise KeyError(f"{prefix} is missing fields {sorted(missing)}")

        logits = _require_tensor(stage_result["score_logits"], f"{prefix}.score_logits")
        scores = _require_tensor(stage_result["score"], f"{prefix}.score")
        if logits.ndim != 2 or logits.shape != scores.shape:
            raise ValueError(
                f"{prefix} score_logits and score must share [B, N], got "
                f"{tuple(logits.shape)} and {tuple(scores.shape)}"
            )
        if scores.shape[1] == 0:
            raise ValueError(f"{prefix} contains no Gaussians")
        if expect_score_grad is True and not logits.requires_grad:
            raise RuntimeError(f"{prefix}.score_logits lost its training graph")
        if expect_score_grad is False and logits.requires_grad:
            raise RuntimeError(f"{prefix}.score_logits unexpectedly requires gradients")

        batch_size, num_gaussians = scores.shape
        score_f = scores.detach().float()
        logits_f = logits.detach().float()
        if not torch.isfinite(logits_f).all() or not torch.isfinite(score_f).all():
            raise FloatingPointError(f"{prefix} contains NaN or Inf scores")
        if torch.any(score_f < 0.0) or torch.any(score_f > 1.0):
            raise ValueError(f"{prefix}.score must lie in [0, 1]")
        if not torch.allclose(
            score_f, logits_f.sigmoid(), rtol=score_atol, atol=score_atol
        ):
            max_error = (score_f - logits_f.sigmoid()).abs().max().item()
            raise ValueError(
                f"{prefix}.score is inconsistent with sigmoid(score_logits); "
                f"max error={max_error:.3e}"
            )

        tau_1 = _require_tensor(stage_result["tau_1"], f"{prefix}.tau_1").detach().float()
        tau_2 = _require_tensor(stage_result["tau_2"], f"{prefix}.tau_2").detach().float()
        score_mean = _require_tensor(
            stage_result["score_mean"], f"{prefix}.score_mean"
        ).detach().float()
        score_std = _require_tensor(
            stage_result["score_std"], f"{prefix}.score_std"
        ).detach().float()
        expected_stat_shape = (batch_size, 1)
        for name, tensor in (
            ("tau_1", tau_1),
            ("tau_2", tau_2),
            ("score_mean", score_mean),
            ("score_std", score_std),
        ):
            if tensor.shape != expected_stat_shape:
                raise ValueError(
                    f"{prefix}.{name} must be {expected_stat_shape}, got "
                    f"{tuple(tensor.shape)}"
                )
            if not torch.isfinite(tensor).all():
                raise FloatingPointError(f"{prefix}.{name} contains NaN or Inf")
        if torch.any(tau_1 < 0.0) or torch.any(tau_2 > 1.0):
            raise ValueError(f"{prefix} thresholds must lie in [0, 1]")
        if torch.any(tau_1 >= tau_2):
            raise ValueError(f"{prefix} must satisfy tau_1 < tau_2")

        actual_mean = score_f.mean(dim=1, keepdim=True)
        actual_std = torch.sqrt(
            (score_f - actual_mean).square().mean(dim=1, keepdim=True)
        )
        if not torch.allclose(score_mean, actual_mean, rtol=1e-4, atol=1e-5):
            raise ValueError(f"{prefix}.score_mean is inconsistent with score")
        if not torch.allclose(score_std, actual_std, rtol=1e-4, atol=1e-5):
            raise ValueError(f"{prefix}.score_std is inconsistent with score")

        score_target_summary = {"score_target_available": False}
        present_score_target_fields = _SCORE_TARGET_FIELDS & stage_result.keys()
        if present_score_target_fields:
            missing_score_target_fields = (
                _SCORE_TARGET_FIELDS - stage_result.keys()
            )
            if missing_score_target_fields:
                raise KeyError(
                    f"{prefix} has an incomplete Score target diagnostic; "
                    f"missing {sorted(missing_score_target_fields)}"
                )
            semantic_entropy_target = _require_tensor(
                stage_result["score_semantic_entropy_target"],
                f"{prefix}.score_semantic_entropy_target",
            ).detach().float()
            local_error_target = _require_tensor(
                stage_result["score_local_prediction_error_target"],
                f"{prefix}.score_local_prediction_error_target",
            ).detach().float()
            complexity_target = _require_tensor(
                stage_result["score_complexity_target"],
                f"{prefix}.score_complexity_target",
            ).detach().float()
            for name, target in (
                ("score_semantic_entropy_target", semantic_entropy_target),
                ("score_local_prediction_error_target", local_error_target),
                ("score_complexity_target", complexity_target),
            ):
                if target.shape != scores.shape:
                    raise ValueError(
                        f"{prefix}.{name} must have shape {tuple(scores.shape)}"
                    )
                if not torch.isfinite(target).all():
                    raise FloatingPointError(
                        f"{prefix}.{name} contains NaN or Inf"
                    )
                if torch.any((target < 0.0) | (target > 1.0)):
                    raise ValueError(f"{prefix}.{name} must lie in [0, 1]")
            expected_complexity_target = (
                semantic_entropy_target * local_error_target
            )
            if not torch.allclose(
                complexity_target,
                expected_complexity_target,
                rtol=1e-4,
                atol=1e-5,
            ):
                raise ValueError(
                    f"{prefix}.score_complexity_target is not entropy*error"
                )
            score_target_summary = {
                "score_target_available": True,
                "score_target_entropy_mean": (
                    semantic_entropy_target.mean().item()
                ),
                "score_target_error_mean": local_error_target.mean().item(),
                "score_target_complexity_mean": (
                    complexity_target.mean().item()
                ),
                "score_target_complexity_max": (
                    complexity_target.max().item()
                ),
            }

        low_mask = _require_tensor(
            stage_result["low_candidate_mask"], f"{prefix}.low_candidate_mask"
        )
        high_mask = _require_tensor(
            stage_result["high_candidate_mask"], f"{prefix}.high_candidate_mask"
        )
        for name, mask in (("low_candidate_mask", low_mask), ("high_candidate_mask", high_mask)):
            if mask.dtype != torch.bool or mask.shape != scores.shape:
                raise ValueError(
                    f"{prefix}.{name} must be bool {tuple(scores.shape)}, got "
                    f"{mask.dtype} {tuple(mask.shape)}"
                )
        if torch.any(low_mask & high_mask):
            raise ValueError(f"{prefix} low/high candidate masks overlap")
        if torch.any(low_mask & (score_f > tau_1 + score_atol)):
            raise ValueError(f"{prefix} low candidates violate tau_1")
        if torch.any(high_mask & (score_f < tau_2 - score_atol)):
            raise ValueError(f"{prefix} high candidates violate tau_2")

        last_operation = _require_tensor(
            stage_result["last_operation_before"],
            f"{prefix}.last_operation_before",
        )
        operation_age = _require_tensor(
            stage_result["operation_age_before"],
            f"{prefix}.operation_age_before",
        )
        prune_evidence_before = _require_tensor(
            stage_result["prune_evidence_before"],
            f"{prefix}.prune_evidence_before",
        )
        prune_evidence_after = _require_tensor(
            stage_result["prune_evidence_after_candidate"],
            f"{prefix}.prune_evidence_after_candidate",
        )
        split_provenance = _require_tensor(
            stage_result["split_provenance_mask"],
            f"{prefix}.split_provenance_mask",
        )
        merge_provenance = _require_tensor(
            stage_result["merge_provenance_mask"],
            f"{prefix}.merge_provenance_mask",
        )
        split_reverse_block = _require_tensor(
            stage_result["split_reverse_block_mask"],
            f"{prefix}.split_reverse_block_mask",
        )
        if last_operation.dtype != torch.long or (
            last_operation.shape != scores.shape
        ):
            raise ValueError(
                f"{prefix}.last_operation_before must be long "
                f"{tuple(scores.shape)}"
            )
        if operation_age.dtype != torch.long or (
            operation_age.shape != scores.shape
        ):
            raise ValueError(
                f"{prefix}.operation_age_before must be long "
                f"{tuple(scores.shape)}"
            )
        for name, evidence in (
            ("prune_evidence_before", prune_evidence_before),
            ("prune_evidence_after_candidate", prune_evidence_after),
        ):
            if evidence.dtype != torch.long or evidence.shape != scores.shape:
                raise ValueError(
                    f"{prefix}.{name} must be long {tuple(scores.shape)}"
                )
            if torch.any(evidence < 0):
                raise ValueError(f"{prefix}.{name} must be non-negative")
        for name, mask in (
            ("split_provenance_mask", split_provenance),
            ("merge_provenance_mask", merge_provenance),
            ("split_reverse_block_mask", split_reverse_block),
        ):
            if mask.dtype != torch.bool or mask.shape != scores.shape:
                raise ValueError(
                    f"{prefix}.{name} must be bool {tuple(scores.shape)}"
                )
        if torch.any((last_operation < 0) | (last_operation > 2)):
            raise ValueError(f"{prefix}.last_operation_before has an invalid id")
        if torch.any(operation_age < 0):
            raise ValueError(f"{prefix}.operation_age_before must be non-negative")
        if torch.any(operation_age[last_operation == 0] != 0):
            raise ValueError(f"{prefix} KEEP provenance must have zero age")
        if torch.any(split_provenance != (last_operation == 1)):
            raise ValueError(f"{prefix}.split_provenance_mask is inconsistent")
        if torch.any(merge_provenance != (last_operation == 2)):
            raise ValueError(f"{prefix}.merge_provenance_mask is inconsistent")
        if torch.any(split_provenance & merge_provenance):
            raise ValueError(f"{prefix} has conflicting topology provenance")
        if torch.any(split_reverse_block != (high_mask & merge_provenance)):
            raise ValueError(
                f"{prefix}.split_reverse_block_mask is inconsistent"
            )

        scale_feasible = _require_tensor(
            stage_result["scale_feasible_mask"],
            f"{prefix}.scale_feasible_mask",
        )
        children_inside = _require_tensor(
            stage_result["children_inside_pc_range_mask"],
            f"{prefix}.children_inside_pc_range_mask",
        )
        split_mask = _require_tensor(
            stage_result["split_mask"], f"{prefix}.split_mask"
        )
        for name, mask in (
            ("scale_feasible_mask", scale_feasible),
            ("children_inside_pc_range_mask", children_inside),
            ("split_mask", split_mask),
        ):
            if mask.dtype != torch.bool or mask.shape != scores.shape:
                raise ValueError(
                    f"{prefix}.{name} must be bool {tuple(scores.shape)}, got "
                    f"{mask.dtype} {tuple(mask.shape)}"
                )
        if torch.any(split_mask & ~high_mask):
            raise ValueError(f"{prefix}.split_mask is not a subset of high candidates")
        if torch.any(split_mask & ~scale_feasible):
            raise ValueError(f"{prefix} split contains an infeasible child scale")
        if torch.any(split_mask & ~children_inside):
            raise ValueError(f"{prefix} split contains an out-of-range child")
        if torch.any(split_mask & merge_provenance):
            raise ValueError(f"{prefix} immediately splits a merged Gaussian")

        opacity = _require_tensor(
            stage_result["gaussian_opacity"], f"{prefix}.gaussian_opacity"
        ).detach().float()
        density_mass = _require_tensor(
            stage_result["density_mass"], f"{prefix}.density_mass"
        ).detach().float()
        if opacity.shape != scores.shape or density_mass.shape != scores.shape:
            raise ValueError(
                f"{prefix} opacity/density_mass must match {tuple(scores.shape)}"
            )
        if not torch.isfinite(opacity).all() or not torch.isfinite(density_mass).all():
            raise FloatingPointError(f"{prefix} contains non-finite deletion statistics")
        if torch.any(opacity < 0.0) or torch.any(opacity >= 1.0):
            raise ValueError(f"{prefix}.gaussian_opacity must lie inside [0, 1)")
        if torch.any(density_mass < 0.0):
            raise ValueError(f"{prefix}.density_mass must be non-negative")

        opacity_threshold = _require_tensor(
            stage_result["opacity_delete_threshold"],
            f"{prefix}.opacity_delete_threshold",
        ).detach().float()
        mass_threshold = _require_tensor(
            stage_result["density_mass_delete_threshold"],
            f"{prefix}.density_mass_delete_threshold",
        ).detach().float()
        if opacity_threshold.shape != (batch_size, 1):
            raise ValueError(f"{prefix}.opacity_delete_threshold must be [B, 1]")
        if mass_threshold.shape != (batch_size, 1):
            raise ValueError(
                f"{prefix}.density_mass_delete_threshold must be [B, 1]"
            )

        low_opacity = _require_tensor(
            stage_result["low_opacity_mask"], f"{prefix}.low_opacity_mask"
        )
        low_mass = _require_tensor(
            stage_result["low_density_mass_mask"],
            f"{prefix}.low_density_mass_mask",
        )
        delete_mask = _require_tensor(
            stage_result["delete_mask"], f"{prefix}.delete_mask"
        )
        delete_candidate = _require_tensor(
            stage_result["delete_candidate_mask"],
            f"{prefix}.delete_candidate_mask",
        )
        delete_protected = _require_tensor(
            stage_result["delete_protected_mask"],
            f"{prefix}.delete_protected_mask",
        )
        for name, mask in (
            ("low_opacity_mask", low_opacity),
            ("low_density_mass_mask", low_mass),
            ("delete_candidate_mask", delete_candidate),
            ("delete_protected_mask", delete_protected),
            ("delete_mask", delete_mask),
        ):
            if mask.dtype != torch.bool or mask.shape != scores.shape:
                raise ValueError(
                    f"{prefix}.{name} must be bool {tuple(scores.shape)}"
                )
        expected_low_opacity = opacity <= opacity_threshold
        expected_low_mass = density_mass <= mass_threshold
        if torch.any(low_opacity != expected_low_opacity):
            raise ValueError(f"{prefix}.low_opacity_mask is inconsistent")
        if torch.any(low_mass != expected_low_mass):
            raise ValueError(f"{prefix}.low_density_mass_mask is inconsistent")
        expected_delete_candidate = low_mask & low_opacity & low_mass
        if torch.any(delete_candidate != expected_delete_candidate):
            raise ValueError(f"{prefix}.delete_candidate_mask is inconsistent")
        if torch.any(delete_mask & ~delete_candidate):
            raise ValueError(
                f"{prefix}.delete_mask is not a subset of delete candidates"
            )
        require_consecutive_evidence = _require_tensor(
            stage_result["require_consecutive_prune_evidence"],
            f"{prefix}.require_consecutive_prune_evidence",
        )
        if require_consecutive_evidence.dtype != torch.bool or (
            require_consecutive_evidence.shape != (batch_size,)
        ):
            raise ValueError(
                f"{prefix}.require_consecutive_prune_evidence "
                "must be bool [B]"
            )
        prune_min_stages = _require_tensor(
            stage_result["prune_min_consecutive_stages"],
            f"{prefix}.prune_min_consecutive_stages",
        )
        if prune_min_stages.dtype != torch.long or (
            prune_min_stages.shape != (batch_size, 1)
        ):
            raise ValueError(
                f"{prefix}.prune_min_consecutive_stages must be long [B, 1]"
            )
        if torch.any(prune_min_stages <= 0):
            raise ValueError(
                f"{prefix}.prune_min_consecutive_stages must be positive"
            )
        expected_prune_evidence_after = torch.where(
            delete_candidate,
            torch.minimum(
                prune_evidence_before + 1,
                prune_min_stages.expand_as(prune_evidence_before),
            ),
            torch.zeros_like(prune_evidence_before),
        )
        if torch.any(prune_evidence_after != expected_prune_evidence_after):
            raise ValueError(
                f"{prefix}.prune_evidence_after_candidate is inconsistent"
            )
        expected_delete_protected = (
            delete_candidate
            & (last_operation != 0)
            & (prune_evidence_after < prune_min_stages)
            & require_consecutive_evidence[:, None]
        )
        if torch.any(delete_protected != expected_delete_protected):
            raise ValueError(f"{prefix}.delete_protected_mask is inconsistent")
        if torch.any(delete_mask & delete_protected):
            raise ValueError(
                f"{prefix} deletes a lineage without consecutive evidence"
            )
        if torch.any(delete_mask & ~low_mask):
            raise ValueError(f"{prefix}.delete_mask is not a subset of low candidates")
        if torch.any(delete_mask & ~low_opacity):
            raise ValueError(f"{prefix} deletes a non-low-opacity Gaussian")
        if torch.any(delete_mask & ~low_mass):
            raise ValueError(f"{prefix} deletes a non-low-mass Gaussian")
        if torch.any(delete_mask & split_mask):
            raise ValueError(f"{prefix} splits and deletes the same Gaussian")

        merge_candidate = _require_tensor(
            stage_result["merge_candidate_mask"],
            f"{prefix}.merge_candidate_mask",
        )
        merge_member = _require_tensor(
            stage_result["merge_member_mask"], f"{prefix}.merge_member_mask"
        )
        merge_partner = _require_tensor(
            stage_result["merge_partner_index"],
            f"{prefix}.merge_partner_index",
        )
        merge_reverse_block = _require_tensor(
            stage_result["merge_reverse_block_mask"],
            f"{prefix}.merge_reverse_block_mask",
        )
        for name, mask in (
            ("merge_candidate_mask", merge_candidate),
            ("merge_member_mask", merge_member),
            ("merge_reverse_block_mask", merge_reverse_block),
        ):
            if mask.dtype != torch.bool or mask.shape != scores.shape:
                raise ValueError(
                    f"{prefix}.{name} must be bool {tuple(scores.shape)}"
                )
        if merge_partner.dtype != torch.long or merge_partner.shape != scores.shape:
            raise ValueError(
                f"{prefix}.merge_partner_index must be long "
                f"{tuple(scores.shape)}"
            )
        if torch.any(merge_candidate & ~low_mask):
            raise ValueError(
                f"{prefix}.merge_candidate_mask is not a subset of low candidates"
            )
        if torch.any(merge_candidate & delete_candidate):
            raise ValueError(
                f"{prefix} proposes a delete candidate for merging"
            )
        if torch.any(
            merge_reverse_block != (low_mask & split_provenance)
        ):
            raise ValueError(
                f"{prefix}.merge_reverse_block_mask is inconsistent"
            )
        if torch.any(merge_candidate & split_provenance):
            raise ValueError(f"{prefix} immediately merges a split child")
        if torch.any(merge_member & ~merge_candidate):
            raise ValueError(f"{prefix} merges a non-candidate Gaussian")
        if torch.any(merge_member & (split_mask | delete_mask)):
            raise ValueError(f"{prefix} merge overlaps split/delete decisions")
        if torch.any((merge_partner >= 0) != merge_member):
            raise ValueError(
                f"{prefix}.merge_partner_index and merge_member_mask disagree"
            )

        merge_pairs = _require_tensor(
            stage_result["merge_pairs"], f"{prefix}.merge_pairs"
        )
        merge_geometry_similarity = _require_tensor(
            stage_result["merge_geometry_similarity"],
            f"{prefix}.merge_geometry_similarity",
        ).detach().float()
        merge_semantic_similarity = _require_tensor(
            stage_result["merge_semantic_similarity"],
            f"{prefix}.merge_semantic_similarity",
        ).detach().float()
        if merge_pairs.dtype != torch.long or (
            merge_pairs.ndim != 2 or merge_pairs.shape[-1] != 2
        ):
            raise ValueError(f"{prefix}.merge_pairs must be long [P, 2]")
        pair_count = merge_pairs.shape[0]
        if merge_geometry_similarity.shape != (pair_count,) or (
            merge_semantic_similarity.shape != (pair_count,)
        ):
            raise ValueError(
                f"{prefix} merge similarities must both have shape [P]"
            )
        if not torch.isfinite(merge_geometry_similarity).all() or not (
            torch.isfinite(merge_semantic_similarity).all()
        ):
            raise FloatingPointError(f"{prefix} has non-finite merge similarity")

        merge_geometry_threshold = _require_tensor(
            stage_result["merge_geometry_threshold"],
            f"{prefix}.merge_geometry_threshold",
        ).detach().float()
        merge_semantic_threshold = _require_tensor(
            stage_result["merge_semantic_threshold"],
            f"{prefix}.merge_semantic_threshold",
        ).detach().float()
        for name, threshold in (
            ("merge_geometry_threshold", merge_geometry_threshold),
            ("merge_semantic_threshold", merge_semantic_threshold),
        ):
            if threshold.shape != (batch_size, 1):
                raise ValueError(f"{prefix}.{name} must be [B, 1]")
            if torch.any((threshold < 0.0) | (threshold > 1.0)):
                raise ValueError(f"{prefix}.{name} must lie in [0, 1]")

        if pair_count > 0:
            if batch_size != 1:
                raise ValueError(
                    f"{prefix} dynamic merging currently requires batch size one"
                )
            if torch.any(merge_pairs < 0) or torch.any(
                merge_pairs >= num_gaussians
            ):
                raise ValueError(f"{prefix}.merge_pairs has an invalid index")
            if torch.any(merge_pairs[:, 0] >= merge_pairs[:, 1]):
                raise ValueError(
                    f"{prefix}.merge_pairs must use canonical i < j ordering"
                )
            if torch.unique(merge_pairs.flatten()).numel() != 2 * pair_count:
                raise ValueError(f"{prefix}.merge_pairs are not disjoint")
            expected_member = torch.zeros_like(merge_member[0])
            expected_member[merge_pairs.flatten()] = True
            if torch.any(expected_member != merge_member[0]):
                raise ValueError(f"{prefix}.merge_member_mask is inconsistent")
            left, right = merge_pairs[:, 0], merge_pairs[:, 1]
            if torch.any(merge_partner[0, left] != right) or torch.any(
                merge_partner[0, right] != left
            ):
                raise ValueError(
                    f"{prefix}.merge_partner_index is not symmetric"
                )
            if torch.any(
                merge_geometry_similarity
                < merge_geometry_threshold[0, 0] - score_atol
            ):
                raise ValueError(
                    f"{prefix} accepted a pair below geometry threshold"
                )
            if torch.any(
                merge_semantic_similarity
                < merge_semantic_threshold[0, 0] - score_atol
            ):
                raise ValueError(
                    f"{prefix} accepted a pair below semantic threshold"
                )
        elif torch.any(merge_member) or torch.any(merge_partner >= 0):
            raise ValueError(f"{prefix} has merge members but no merge pairs")

        enabled = _require_tensor(
            stage_result["control_enabled"], f"{prefix}.control_enabled"
        )
        split_enabled = _require_tensor(
            stage_result["split_enabled"], f"{prefix}.split_enabled"
        )
        delete_enabled = _require_tensor(
            stage_result["delete_enabled"], f"{prefix}.delete_enabled"
        )
        merge_enabled = _require_tensor(
            stage_result["merge_enabled"], f"{prefix}.merge_enabled"
        )
        stage_ids = _require_tensor(
            stage_result["control_stage"], f"{prefix}.control_stage"
        )
        if enabled.dtype != torch.bool or enabled.shape != (batch_size,):
            raise ValueError(f"{prefix}.control_enabled must be bool [B]")
        if stage_ids.shape != (batch_size,):
            raise ValueError(f"{prefix}.control_stage must have shape [B]")
        if split_enabled.dtype != torch.bool or split_enabled.shape != (batch_size,):
            raise ValueError(f"{prefix}.split_enabled must be bool [B]")
        if delete_enabled.dtype != torch.bool or delete_enabled.shape != (batch_size,):
            raise ValueError(f"{prefix}.delete_enabled must be bool [B]")
        if merge_enabled.dtype != torch.bool or merge_enabled.shape != (batch_size,):
            raise ValueError(f"{prefix}.merge_enabled must be bool [B]")
        if torch.any(split_enabled & ~enabled):
            raise ValueError(f"{prefix} enables splitting while control is disabled")
        if torch.any(delete_enabled & ~enabled):
            raise ValueError(f"{prefix} enables deletion while control is disabled")
        if torch.any(merge_enabled & ~enabled):
            raise ValueError(f"{prefix} enables merging while control is disabled")
        if torch.any(stage_ids != expected_stage):
            raise ValueError(
                f"{prefix}.control_stage must equal its list index {expected_stage}"
            )
        disabled_rows = ~enabled
        if torch.any(low_mask[disabled_rows]) or torch.any(high_mask[disabled_rows]):
            raise ValueError(f"{prefix} has candidates while control is disabled")
        if torch.any(split_mask[~split_enabled]):
            raise ValueError(f"{prefix} splits Gaussians while splitting is disabled")
        if torch.any(delete_mask[~delete_enabled]):
            raise ValueError(f"{prefix} deletes Gaussians while deletion is disabled")
        if torch.any(
            delete_mask[delete_enabled]
            != (
                delete_candidate & ~delete_protected
            )[delete_enabled]
        ):
            raise ValueError(
                f"{prefix} deletion does not match candidates minus protection"
            )
        expected_merge_candidate = (
            low_mask & ~delete_candidate & ~split_provenance
        )
        if torch.any(merge_candidate[~merge_enabled]):
            raise ValueError(
                f"{prefix} has merge candidates while merging is disabled"
            )
        if torch.any(
            merge_candidate[merge_enabled]
            != expected_merge_candidate[merge_enabled]
        ):
            raise ValueError(
                f"{prefix}.merge_candidate_mask violates candidate protection"
            )
        if torch.any(merge_member[~merge_enabled]):
            raise ValueError(f"{prefix} merges Gaussians while merging is disabled")

        num_before = _require_tensor(
            stage_result["num_gaussians_before"],
            f"{prefix}.num_gaussians_before",
        )
        num_split = _require_tensor(
            stage_result["num_split"], f"{prefix}.num_split"
        )
        num_deleted = _require_tensor(
            stage_result["num_deleted"], f"{prefix}.num_deleted"
        )
        num_merge_pairs = _require_tensor(
            stage_result["num_merge_pairs"], f"{prefix}.num_merge_pairs"
        )
        num_split_reverse_blocked = _require_tensor(
            stage_result["num_split_reverse_blocked"],
            f"{prefix}.num_split_reverse_blocked",
        )
        num_delete_protected = _require_tensor(
            stage_result["num_delete_protected"],
            f"{prefix}.num_delete_protected",
        )
        num_merge_reverse_blocked = _require_tensor(
            stage_result["num_merge_reverse_blocked"],
            f"{prefix}.num_merge_reverse_blocked",
        )
        num_after = _require_tensor(
            stage_result["num_gaussians_after"],
            f"{prefix}.num_gaussians_after",
        )
        for name, tensor in (
            ("num_gaussians_before", num_before),
            ("num_split", num_split),
            ("num_deleted", num_deleted),
            ("num_merge_pairs", num_merge_pairs),
            ("num_split_reverse_blocked", num_split_reverse_blocked),
            ("num_delete_protected", num_delete_protected),
            ("num_merge_reverse_blocked", num_merge_reverse_blocked),
            ("num_gaussians_after", num_after),
        ):
            if tensor.shape != (batch_size,) or tensor.dtype != torch.long:
                raise ValueError(f"{prefix}.{name} must be long [B]")
        expected_before = torch.full_like(num_before, num_gaussians)
        expected_split = split_mask.sum(dim=1).to(dtype=torch.long)
        expected_deleted = delete_mask.sum(dim=1).to(dtype=torch.long)
        if torch.any(num_before != expected_before):
            raise ValueError(f"{prefix}.num_gaussians_before is inconsistent")
        if torch.any(num_split != expected_split):
            raise ValueError(f"{prefix}.num_split is inconsistent with split_mask")
        if torch.any(num_deleted != expected_deleted):
            raise ValueError(f"{prefix}.num_deleted is inconsistent with delete_mask")
        if torch.any(
            num_split_reverse_blocked
            != split_reverse_block.sum(dim=1).to(dtype=torch.long)
        ):
            raise ValueError(
                f"{prefix}.num_split_reverse_blocked is inconsistent"
            )
        if torch.any(
            num_delete_protected
            != delete_protected.sum(dim=1).to(dtype=torch.long)
        ):
            raise ValueError(f"{prefix}.num_delete_protected is inconsistent")
        if torch.any(
            num_merge_reverse_blocked
            != merge_reverse_block.sum(dim=1).to(dtype=torch.long)
        ):
            raise ValueError(
                f"{prefix}.num_merge_reverse_blocked is inconsistent"
            )
        expected_merge_pairs = torch.zeros_like(num_merge_pairs)
        if batch_size == 1:
            expected_merge_pairs[0] = pair_count
        if torch.any(num_merge_pairs != expected_merge_pairs):
            raise ValueError(f"{prefix}.num_merge_pairs is inconsistent")
        if torch.any(
            num_after
            != num_before + num_split - num_deleted - num_merge_pairs
        ):
            raise ValueError(f"{prefix}.num_gaussians_after is inconsistent")

        gaussians = stage_result["gaussian"]
        if not hasattr(gaussians, "means"):
            raise TypeError(f"{prefix}.gaussian must provide a means tensor")
        means = _require_tensor(gaussians.means, f"{prefix}.gaussian.means")
        if means.shape[:2] != scores.shape or means.shape[-1] != 3:
            raise ValueError(
                f"{prefix}.gaussian and scores are misaligned: "
                f"means={tuple(means.shape)}, scores={tuple(scores.shape)}"
            )

        diagnostic_present = (
            _SPLIT_DIAGNOSTIC_FIELDS & stage_result.keys()
        )
        if diagnostic_present and (
            diagnostic_present != _SPLIT_DIAGNOSTIC_FIELDS
        ):
            missing_diagnostics = (
                _SPLIT_DIAGNOSTIC_FIELDS - diagnostic_present
            )
            raise KeyError(
                f"{prefix} has partial split diagnostics; missing "
                f"{sorted(missing_diagnostics)}"
            )

        split_diagnostic_summary = {
            "split_diagnostic_available": False,
            "split_diagnostic_valid_count": 0,
            "split_diagnostic_selected_count": 0,
            "split_population_best_gain": 0.0,
            "split_population_best_gain_ratio": 0.0,
            "split_selected_current_gain": 0.0,
            "split_selected_best_gain": 0.0,
            "split_selected_current_gain_ratio": 0.0,
            "split_selected_best_gain_ratio": 0.0,
            "split_selected_gain_lift": 0.0,
            "split_axis_match_ratio": 0.0,
            "split_direction_efficiency": 0.0,
            "split_low_gain_ratio": 0.0,
            "split_score_gain_correlation": 0.0,
            "split_current_axis_x_ratio": 0.0,
            "split_current_axis_y_ratio": 0.0,
            "split_current_axis_z_ratio": 0.0,
            "split_best_axis_x_ratio": 0.0,
            "split_best_axis_y_ratio": 0.0,
            "split_best_axis_z_ratio": 0.0,
        }
        if diagnostic_present:
            gain_per_axis = _require_tensor(
                stage_result["split_gain_per_axis"],
                f"{prefix}.split_gain_per_axis",
            ).detach().float()
            gain_ratio_per_axis = _require_tensor(
                stage_result["split_gain_ratio_per_axis"],
                f"{prefix}.split_gain_ratio_per_axis",
            ).detach().float()
            best_gain = _require_tensor(
                stage_result["best_split_gain"],
                f"{prefix}.best_split_gain",
            ).detach().float()
            best_gain_ratio = _require_tensor(
                stage_result["best_split_gain_ratio"],
                f"{prefix}.best_split_gain_ratio",
            ).detach().float()
            current_gain = _require_tensor(
                stage_result["current_axis_split_gain"],
                f"{prefix}.current_axis_split_gain",
            ).detach().float()
            current_gain_ratio = _require_tensor(
                stage_result["current_axis_split_gain_ratio"],
                f"{prefix}.current_axis_split_gain_ratio",
            ).detach().float()
            best_axis = _require_tensor(
                stage_result["best_split_axis"],
                f"{prefix}.best_split_axis",
            )
            current_axis = _require_tensor(
                stage_result["current_split_axis"],
                f"{prefix}.current_split_axis",
            )
            gain_valid = _require_tensor(
                stage_result["split_gain_valid_mask"],
                f"{prefix}.split_gain_valid_mask",
            )
            low_gain_threshold = _require_tensor(
                stage_result["split_gain_ratio_low_threshold"],
                f"{prefix}.split_gain_ratio_low_threshold",
            ).detach().float()

            axis_shape = (*scores.shape, 3)
            for name, tensor in (
                ("split_gain_per_axis", gain_per_axis),
                ("split_gain_ratio_per_axis", gain_ratio_per_axis),
            ):
                if tensor.shape != axis_shape:
                    raise ValueError(
                        f"{prefix}.{name} must be {axis_shape}, got "
                        f"{tuple(tensor.shape)}"
                    )
                if not torch.isfinite(tensor).all():
                    raise FloatingPointError(
                        f"{prefix}.{name} contains NaN or Inf"
                    )
                if torch.any((tensor < 0.0) | (tensor > 1.0)):
                    raise ValueError(f"{prefix}.{name} must lie in [0, 1]")

            for name, tensor in (
                ("best_split_gain", best_gain),
                ("best_split_gain_ratio", best_gain_ratio),
                ("current_axis_split_gain", current_gain),
                ("current_axis_split_gain_ratio", current_gain_ratio),
            ):
                if tensor.shape != scores.shape:
                    raise ValueError(
                        f"{prefix}.{name} must be {tuple(scores.shape)}"
                    )
                if not torch.isfinite(tensor).all():
                    raise FloatingPointError(
                        f"{prefix}.{name} contains NaN or Inf"
                    )
                if torch.any((tensor < 0.0) | (tensor > 1.0)):
                    raise ValueError(f"{prefix}.{name} must lie in [0, 1]")

            for name, axis in (
                ("best_split_axis", best_axis),
                ("current_split_axis", current_axis),
            ):
                if axis.dtype != torch.long or axis.shape != scores.shape:
                    raise ValueError(
                        f"{prefix}.{name} must be long "
                        f"{tuple(scores.shape)}"
                    )
                if torch.any((axis < 0) | (axis > 2)):
                    raise ValueError(f"{prefix}.{name} must lie in [0, 2]")
            if gain_valid.dtype != torch.bool or (
                gain_valid.shape != scores.shape
            ):
                raise ValueError(
                    f"{prefix}.split_gain_valid_mask must be bool "
                    f"{tuple(scores.shape)}"
                )
            if low_gain_threshold.shape != (batch_size, 1) or (
                not torch.isfinite(low_gain_threshold).all()
            ):
                raise ValueError(
                    f"{prefix}.split_gain_ratio_low_threshold must be "
                    "[B, 1] and finite"
                )
            if torch.any(
                (low_gain_threshold < 0.0)
                | (low_gain_threshold > 1.0)
            ):
                raise ValueError(
                    f"{prefix}.split_gain_ratio_low_threshold must lie "
                    "in [0, 1]"
                )

            expected_best_gain, expected_best_axis = gain_per_axis.max(
                dim=-1
            )
            if not torch.allclose(
                best_gain, expected_best_gain, rtol=1e-4, atol=1e-6
            ):
                raise ValueError(
                    f"{prefix}.best_split_gain is inconsistent"
                )
            if torch.any(best_axis != expected_best_axis):
                raise ValueError(
                    f"{prefix}.best_split_axis is inconsistent"
                )
            expected_best_ratio = gain_ratio_per_axis.gather(
                -1, best_axis.unsqueeze(-1)
            ).squeeze(-1)
            expected_current_gain = gain_per_axis.gather(
                -1, current_axis.unsqueeze(-1)
            ).squeeze(-1)
            expected_current_ratio = gain_ratio_per_axis.gather(
                -1, current_axis.unsqueeze(-1)
            ).squeeze(-1)
            if not torch.allclose(
                best_gain_ratio,
                expected_best_ratio,
                rtol=1e-4,
                atol=1e-6,
            ):
                raise ValueError(
                    f"{prefix}.best_split_gain_ratio is inconsistent"
                )
            if not torch.allclose(
                current_gain,
                expected_current_gain,
                rtol=1e-4,
                atol=1e-6,
            ):
                raise ValueError(
                    f"{prefix}.current_axis_split_gain is inconsistent"
                )
            if not torch.allclose(
                current_gain_ratio,
                expected_current_ratio,
                rtol=1e-4,
                atol=1e-6,
            ):
                raise ValueError(
                    f"{prefix}.current_axis_split_gain_ratio is "
                    "inconsistent"
                )

            gaussian_scales = _require_tensor(
                gaussians.scales, f"{prefix}.gaussian.scales"
            ).detach()
            if gaussian_scales.shape != (*scores.shape, 3):
                raise ValueError(
                    f"{prefix}.gaussian.scales must be "
                    f"{(*scores.shape, 3)}"
                )
            if torch.any(
                current_axis != gaussian_scales.argmax(dim=-1)
            ):
                raise ValueError(
                    f"{prefix}.current_split_axis does not match the "
                    "implemented largest-scale split axis"
                )
            if torch.any(
                gain_per_axis
                > best_gain.unsqueeze(-1) + score_atol
            ):
                raise ValueError(
                    f"{prefix}.split_gain_per_axis exceeds best gain"
                )
            if torch.any(gain_per_axis[~gain_valid] != 0.0) or torch.any(
                gain_ratio_per_axis[~gain_valid] != 0.0
            ):
                raise ValueError(
                    f"{prefix} invalid split diagnostics must be zero"
                )

            selected = split_mask & gain_valid
            valid_best_gain = best_gain[gain_valid]
            valid_best_ratio = best_gain_ratio[gain_valid]
            selected_current_gain = current_gain[selected]
            selected_best_gain = best_gain[selected]
            selected_current_ratio = current_gain_ratio[selected]
            selected_best_ratio = best_gain_ratio[selected]
            selected_current_axis = current_axis[selected]

            def _mean_or_zero(values: torch.Tensor) -> float:
                return (
                    values.mean().item() if values.numel() > 0 else 0.0
                )

            population_best_gain = _mean_or_zero(valid_best_gain)
            population_best_ratio = _mean_or_zero(valid_best_ratio)
            selected_best_ratio_mean = _mean_or_zero(
                selected_best_ratio
            )
            gain_lift = (
                selected_best_ratio_mean / population_best_ratio
                if population_best_ratio > 1e-12
                else 0.0
            )

            direction_valid = selected & (best_gain > 1e-8)
            direction_current_axis = current_axis[direction_valid]
            direction_best_axis = best_axis[direction_valid]
            direction_efficiency = _mean_or_zero(
                (
                    current_gain[direction_valid]
                    / best_gain[direction_valid].clamp_min(1e-8)
                ).clamp_(0.0, 1.0)
            )
            axis_match = _mean_or_zero(
                (direction_current_axis == direction_best_axis).float()
            )
            low_gain_ratio = _mean_or_zero(
                (
                    selected_current_ratio
                    <= low_gain_threshold.expand_as(
                        current_gain_ratio
                    )[selected]
                ).float()
            )

            valid_scores = score_f[gain_valid]
            if valid_scores.numel() > 1:
                centered_score = valid_scores - valid_scores.mean()
                centered_gain = (
                    valid_best_ratio - valid_best_ratio.mean()
                )
                correlation_denominator = torch.sqrt(
                    centered_score.square().sum()
                    * centered_gain.square().sum()
                )
                score_gain_correlation = (
                    (
                        centered_score * centered_gain
                    ).sum()
                    / correlation_denominator.clamp_min(1e-12)
                ).item()
                if correlation_denominator.item() <= 1e-12:
                    score_gain_correlation = 0.0
            else:
                score_gain_correlation = 0.0

            current_axis_ratios = [
                _mean_or_zero((selected_current_axis == axis).float())
                for axis in range(3)
            ]
            best_axis_ratios = [
                _mean_or_zero((direction_best_axis == axis).float())
                for axis in range(3)
            ]
            split_diagnostic_summary = {
                "split_diagnostic_available": True,
                "split_diagnostic_valid_count": int(
                    gain_valid.sum().item()
                ),
                "split_diagnostic_selected_count": int(
                    selected.sum().item()
                ),
                "split_population_best_gain": population_best_gain,
                "split_population_best_gain_ratio": (
                    population_best_ratio
                ),
                "split_selected_current_gain": _mean_or_zero(
                    selected_current_gain
                ),
                "split_selected_best_gain": _mean_or_zero(
                    selected_best_gain
                ),
                "split_selected_current_gain_ratio": _mean_or_zero(
                    selected_current_ratio
                ),
                "split_selected_best_gain_ratio": (
                    selected_best_ratio_mean
                ),
                "split_selected_gain_lift": gain_lift,
                "split_axis_match_ratio": axis_match,
                "split_direction_efficiency": direction_efficiency,
                "split_low_gain_ratio": low_gain_ratio,
                "split_score_gain_correlation": (
                    score_gain_correlation
                ),
                "split_current_axis_x_ratio": current_axis_ratios[0],
                "split_current_axis_y_ratio": current_axis_ratios[1],
                "split_current_axis_z_ratio": current_axis_ratios[2],
                "split_best_axis_x_ratio": best_axis_ratios[0],
                "split_best_axis_y_ratio": best_axis_ratios[1],
                "split_best_axis_z_ratio": best_axis_ratios[2],
            }

        learned_split_present = (
            _LEARNED_SPLIT_FIELDS & stage_result.keys()
        )
        if learned_split_present and (
            learned_split_present != _LEARNED_SPLIT_FIELDS
        ):
            missing_learned_split = (
                _LEARNED_SPLIT_FIELDS - learned_split_present
            )
            raise KeyError(
                f"{prefix} has partial learned-split statistics; missing "
                f"{sorted(missing_learned_split)}"
            )

        learned_split_summary = {
            "learned_split_available": False,
            "learned_split_mass_ratio_p10": 1.0,
            "learned_split_mass_ratio_mean": 1.0,
            "learned_split_mass_ratio_p90": 1.0,
            "learned_split_position_clamped_ratio": 0.0,
        }
        if learned_split_present:
            learned_mass_ratio = _require_tensor(
                stage_result["learned_split_mass_ratio"],
                f"{prefix}.learned_split_mass_ratio",
            ).detach().float()
            learned_position_clamped = _require_tensor(
                stage_result["learned_split_position_clamped_mask"],
                f"{prefix}.learned_split_position_clamped_mask",
            )
            learned_enabled = _require_tensor(
                stage_result["learned_split_enabled"],
                f"{prefix}.learned_split_enabled",
            )
            if learned_mass_ratio.shape != scores.shape:
                raise ValueError(
                    f"{prefix}.learned_split_mass_ratio must be "
                    f"{tuple(scores.shape)}"
                )
            if (
                not torch.isfinite(learned_mass_ratio).all()
                or torch.any(learned_mass_ratio <= 0.0)
            ):
                raise ValueError(
                    f"{prefix}.learned_split_mass_ratio must be finite "
                    "and positive"
                )
            if (
                learned_position_clamped.dtype != torch.bool
                or learned_position_clamped.shape != scores.shape
            ):
                raise ValueError(
                    f"{prefix}.learned_split_position_clamped_mask must "
                    f"be bool {tuple(scores.shape)}"
                )
            if (
                learned_enabled.dtype != torch.bool
                or learned_enabled.shape != (batch_size,)
            ):
                raise ValueError(
                    f"{prefix}.learned_split_enabled must be bool "
                    f"{(batch_size,)}"
                )

            selected_learned = split_mask & learned_enabled[:, None]
            selected_mass_ratio = learned_mass_ratio[selected_learned]
            selected_clamped = learned_position_clamped[selected_learned]
            if selected_mass_ratio.numel() > 0:
                learned_split_summary = {
                    "learned_split_available": True,
                    "learned_split_mass_ratio_p10": torch.quantile(
                        selected_mass_ratio, 0.10
                    ).item(),
                    "learned_split_mass_ratio_mean": (
                        selected_mass_ratio.mean().item()
                    ),
                    "learned_split_mass_ratio_p90": torch.quantile(
                        selected_mass_ratio, 0.90
                    ).item(),
                    "learned_split_position_clamped_ratio": (
                        selected_clamped.float().mean().item()
                    ),
                }

        total = batch_size * num_gaussians
        summaries.append(
            {
                "stage": expected_stage,
                "batch_size": batch_size,
                "num_gaussians": num_gaussians,
                "enabled_ratio": enabled.float().mean().item(),
                "score_mean": score_f.mean().item(),
                "score_std": actual_std.mean().item(),
                "score_min": score_f.min().item(),
                "score_max": score_f.max().item(),
                "tau_1": tau_1.mean().item(),
                "tau_2": tau_2.mean().item(),
                "low_count": int(low_mask.sum().item()),
                "high_count": int(high_mask.sum().item()),
                "low_ratio": low_mask.sum().item() / total,
                "high_ratio": high_mask.sum().item() / total,
                "split_count": int(split_mask.sum().item()),
                "split_ratio": split_mask.sum().item() / total,
                "split_reverse_blocked_count": int(
                    split_reverse_block.sum().item()
                ),
                "delete_count": int(delete_mask.sum().item()),
                "delete_ratio": delete_mask.sum().item() / total,
                "delete_candidate_count": int(delete_candidate.sum().item()),
                "delete_protected_count": int(delete_protected.sum().item()),
                "merge_candidate_count": int(merge_candidate.sum().item()),
                "merge_reverse_blocked_count": int(
                    merge_reverse_block.sum().item()
                ),
                "merge_pair_count": pair_count,
                "merge_reduction_ratio": pair_count / total,
                "merge_geometry_mean": (
                    merge_geometry_similarity.mean().item()
                    if pair_count > 0 else 0.0
                ),
                "merge_semantic_mean": (
                    merge_semantic_similarity.mean().item()
                    if pair_count > 0 else 0.0
                ),
                "merge_geometry_threshold": (
                    merge_geometry_threshold.mean().item()
                ),
                "merge_semantic_threshold": (
                    merge_semantic_threshold.mean().item()
                ),
                "num_gaussians_after": int(num_after.sum().item()),
                "opacity_p10": torch.quantile(opacity, 0.10).item(),
                "opacity_mean": opacity.mean().item(),
                "density_mass_p10": torch.quantile(density_mass, 0.10).item(),
                "density_mass_mean": density_mass.mean().item(),
                "opacity_delete_threshold": opacity_threshold.mean().item(),
                "density_mass_delete_threshold": mass_threshold.mean().item(),
                "near_zero_ratio": (score_f < 0.01).float().mean().item(),
                "near_one_ratio": (score_f > 0.99).float().mean().item(),
                **score_target_summary,
                **split_diagnostic_summary,
                **learned_split_summary,
            }
        )
    return summaries


def format_control_summary(summaries: Sequence[Dict[str, float]]) -> str:
    if not summaries:
        return "no adaptive-control stages"
    stage_messages = []
    for summary in summaries:
        stage_message = (
            "S{stage} N={num_gaussians} enabled={enabled_ratio:.0%} "
            "score={score_mean:.4f}+/-{score_std:.4f}"
            "[{score_min:.4f},{score_max:.4f}] "
            "tau=({tau_1:.4f},{tau_2:.4f}) "
            "low={low_count}({low_ratio:.2%}) "
            "high={high_count}({high_ratio:.2%}) "
            "split={split_count}({split_ratio:.2%}) "
            "delete={delete_count}/{delete_candidate_count}"
            "({delete_ratio:.2%}) "
            "merge={merge_pair_count}/{merge_candidate_count}"
            "({merge_reduction_ratio:.2%})->N{num_gaussians_after} "
            "lock(s/d/m)={split_reverse_blocked_count}/"
            "{delete_protected_count}/{merge_reverse_blocked_count} "
            "merge_sim(g/s)={merge_geometry_mean:.3f}/"
            "{merge_semantic_mean:.3f} "
            "merge_thr=({merge_geometry_threshold:.3f},"
            "{merge_semantic_threshold:.3f}) "
            "opa(p10/mean)={opacity_p10:.3g}/{opacity_mean:.3g} "
            "mass(p10/mean)={density_mass_p10:.3g}/{density_mass_mean:.3g} "
            "del_thr=({opacity_delete_threshold:.3g},"
            "{density_mass_delete_threshold:.3g}) "
            "sat=({near_zero_ratio:.2%},{near_one_ratio:.2%})".format(**summary)
        )
        if summary.get("score_target_available", False):
            stage_message += (
                " target(H/E/HxE)={score_target_entropy_mean:.4f}/"
                "{score_target_error_mean:.4f}/"
                "{score_target_complexity_mean:.4f}"
                "[{score_target_complexity_max:.4f}]"
            ).format(**summary)
        if summary.get("split_diagnostic_available", False):
            stage_message += (
                " IG sel/valid={split_diagnostic_selected_count}/"
                "{split_diagnostic_valid_count} "
                "abs(cur/best/pop)="
                "{split_selected_current_gain:.4f}/"
                "{split_selected_best_gain:.4f}/"
                "{split_population_best_gain:.4f} "
                "rel(cur/best/pop)="
                "{split_selected_current_gain_ratio:.3f}/"
                "{split_selected_best_gain_ratio:.3f}/"
                "{split_population_best_gain_ratio:.3f} "
                "lift={split_selected_gain_lift:.2f} "
                "axis(match/eff)="
                "{split_axis_match_ratio:.1%}/"
                "{split_direction_efficiency:.1%} "
                "axis_cur(x/y/z)="
                "{split_current_axis_x_ratio:.0%}/"
                "{split_current_axis_y_ratio:.0%}/"
                "{split_current_axis_z_ratio:.0%} "
                "axis_best(x/y/z)="
                "{split_best_axis_x_ratio:.0%}/"
                "{split_best_axis_y_ratio:.0%}/"
                "{split_best_axis_z_ratio:.0%} "
                "lowIG={split_low_gain_ratio:.1%} "
                "corr={split_score_gain_correlation:.3f}"
            ).format(**summary)
        if summary.get("learned_split_available", False):
            stage_message += (
                " learned_mass(p10/mean/p90)="
                "{learned_split_mass_ratio_p10:.3f}/"
                "{learned_split_mass_ratio_mean:.3f}/"
                "{learned_split_mass_ratio_p90:.3f} "
                "pos_clip={learned_split_position_clamped_ratio:.2%}"
            ).format(**summary)
        stage_messages.append(stage_message)
    return " | ".join(stage_messages)


@torch.no_grad()
def validate_score_head_gradients(model) -> Dict[str, float]:
    """Check Score gradients and report learned-split gradients when present."""
    if hasattr(model, "module"):
        model = model.module
    parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if "score_head" in name and parameter.requires_grad
    ]
    if not parameters:
        raise RuntimeError(
            "control_result exists but no trainable score_head parameters were found"
        )

    missing = [name for name, parameter in parameters if parameter.grad is None]
    if len(missing) == len(parameters):
        raise RuntimeError(
            "All score_head gradients are None. Check that "
            "GaussianComplexityLoss consumes control_result."
        )

    squared_norm = 0.0
    nonfinite = []
    for name, parameter in parameters:
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().float()
        if not torch.isfinite(gradient).all():
            nonfinite.append(name)
        squared_norm += gradient.square().sum().item()
    if nonfinite:
        raise FloatingPointError(
            f"Non-finite score_head gradients: {nonfinite}"
        )

    summary = {
        "grad_norm": squared_norm ** 0.5,
        "parameter_count": len(parameters),
        "missing_count": len(missing),
    }
    split_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if "split_generator" in name and parameter.requires_grad
    ]
    if split_parameters:
        split_missing = [
            name
            for name, parameter in split_parameters
            if parameter.grad is None
        ]
        split_squared_norm = 0.0
        split_nonfinite = []
        for name, parameter in split_parameters:
            if parameter.grad is None:
                continue
            gradient = parameter.grad.detach().float()
            if not torch.isfinite(gradient).all():
                split_nonfinite.append(name)
            split_squared_norm += gradient.square().sum().item()
        if split_nonfinite:
            raise FloatingPointError(
                f"Non-finite split_generator gradients: {split_nonfinite}"
            )
        summary.update(
            {
                "split_grad_norm": split_squared_norm ** 0.5,
                "split_parameter_count": len(split_parameters),
                "split_missing_count": len(split_missing),
            }
        )
    return summary


def format_score_gradient_summary(summary: Dict[str, float]) -> str:
    message = (
        "grad_norm={grad_norm:.6g} params={parameter_count} "
        "missing={missing_count}"
    ).format(**summary)
    if summary.get("split_parameter_count", 0):
        message += (
            " split_grad_norm={split_grad_norm:.6g} "
            "split_params={split_parameter_count} "
            "split_missing={split_missing_count}"
        ).format(**summary)
    return message

# from typing import Dict, List, Optional, Sequence

# import torch


# _REQUIRED_FIELDS = {
#     "score_logits",
#     "score",
#     "tau_1",
#     "tau_2",
#     "score_mean",
#     "score_std",
#     "low_candidate_mask",
#     "high_candidate_mask",
#     "scale_feasible_mask",
#     "children_inside_pc_range_mask",
#     "split_mask",
#     "gaussian_opacity",
#     "density_mass",
#     "low_opacity_mask",
#     "low_density_mass_mask",
#     "delete_candidate_mask",
#     "delete_mask",
#     "merge_candidate_mask",
#     "merge_member_mask",
#     "merge_partner_index",
#     "merge_pairs",
#     "merge_geometry_similarity",
#     "merge_semantic_similarity",
#     "opacity_delete_threshold",
#     "density_mass_delete_threshold",
#     "merge_geometry_threshold",
#     "merge_semantic_threshold",
#     "num_gaussians_before",
#     "num_split",
#     "num_deleted",
#     "num_merge_pairs",
#     "num_gaussians_after",
#     "control_enabled",
#     "split_enabled",
#     "delete_enabled",
#     "merge_enabled",
#     "control_stage",
#     "gaussian",
# }


# def _require_tensor(value, name: str) -> torch.Tensor:
#     if not isinstance(value, torch.Tensor):
#         raise TypeError(f"{name} must be a tensor, got {type(value).__name__}")
#     return value


# @torch.no_grad()
# def validate_control_result(
#     control_result: Sequence[Dict],
#     expect_score_grad: Optional[bool] = None,
#     score_atol: float = 5e-4,
# ) -> List[Dict[str, float]]:
#     """Validate the encoder control contract and return compact statistics.

#     This function intentionally performs GPU-to-CPU synchronization and should
#     therefore be called at the logging interval, not on every forward pass.
#     It never detaches or replaces tensors stored in ``control_result``.
#     """
#     if not isinstance(control_result, (list, tuple)):
#         raise TypeError("control_result must be a list or tuple of stage dictionaries")

#     summaries = []
#     for expected_stage, stage_result in enumerate(control_result):
#         prefix = f"control_result[{expected_stage}]"
#         if not isinstance(stage_result, dict):
#             raise TypeError(f"{prefix} must be a dictionary")
#         missing = _REQUIRED_FIELDS - stage_result.keys()
#         if missing:
#             raise KeyError(f"{prefix} is missing fields {sorted(missing)}")

#         logits = _require_tensor(stage_result["score_logits"], f"{prefix}.score_logits")
#         scores = _require_tensor(stage_result["score"], f"{prefix}.score")
#         if logits.ndim != 2 or logits.shape != scores.shape:
#             raise ValueError(
#                 f"{prefix} score_logits and score must share [B, N], got "
#                 f"{tuple(logits.shape)} and {tuple(scores.shape)}"
#             )
#         if scores.shape[1] == 0:
#             raise ValueError(f"{prefix} contains no Gaussians")
#         if expect_score_grad is True and not logits.requires_grad:
#             raise RuntimeError(f"{prefix}.score_logits lost its training graph")
#         if expect_score_grad is False and logits.requires_grad:
#             raise RuntimeError(f"{prefix}.score_logits unexpectedly requires gradients")

#         batch_size, num_gaussians = scores.shape
#         score_f = scores.detach().float()
#         logits_f = logits.detach().float()
#         if not torch.isfinite(logits_f).all() or not torch.isfinite(score_f).all():
#             raise FloatingPointError(f"{prefix} contains NaN or Inf scores")
#         if torch.any(score_f < 0.0) or torch.any(score_f > 1.0):
#             raise ValueError(f"{prefix}.score must lie in [0, 1]")
#         if not torch.allclose(
#             score_f, logits_f.sigmoid(), rtol=score_atol, atol=score_atol
#         ):
#             max_error = (score_f - logits_f.sigmoid()).abs().max().item()
#             raise ValueError(
#                 f"{prefix}.score is inconsistent with sigmoid(score_logits); "
#                 f"max error={max_error:.3e}"
#             )

#         tau_1 = _require_tensor(stage_result["tau_1"], f"{prefix}.tau_1").detach().float()
#         tau_2 = _require_tensor(stage_result["tau_2"], f"{prefix}.tau_2").detach().float()
#         score_mean = _require_tensor(
#             stage_result["score_mean"], f"{prefix}.score_mean"
#         ).detach().float()
#         score_std = _require_tensor(
#             stage_result["score_std"], f"{prefix}.score_std"
#         ).detach().float()
#         expected_stat_shape = (batch_size, 1)
#         for name, tensor in (
#             ("tau_1", tau_1),
#             ("tau_2", tau_2),
#             ("score_mean", score_mean),
#             ("score_std", score_std),
#         ):
#             if tensor.shape != expected_stat_shape:
#                 raise ValueError(
#                     f"{prefix}.{name} must be {expected_stat_shape}, got "
#                     f"{tuple(tensor.shape)}"
#                 )
#             if not torch.isfinite(tensor).all():
#                 raise FloatingPointError(f"{prefix}.{name} contains NaN or Inf")
#         if torch.any(tau_1 < 0.0) or torch.any(tau_2 > 1.0):
#             raise ValueError(f"{prefix} thresholds must lie in [0, 1]")
#         if torch.any(tau_1 >= tau_2):
#             raise ValueError(f"{prefix} must satisfy tau_1 < tau_2")

#         actual_mean = score_f.mean(dim=1, keepdim=True)
#         actual_std = torch.sqrt(
#             (score_f - actual_mean).square().mean(dim=1, keepdim=True)
#         )
#         if not torch.allclose(score_mean, actual_mean, rtol=1e-4, atol=1e-5):
#             raise ValueError(f"{prefix}.score_mean is inconsistent with score")
#         if not torch.allclose(score_std, actual_std, rtol=1e-4, atol=1e-5):
#             raise ValueError(f"{prefix}.score_std is inconsistent with score")

#         low_mask = _require_tensor(
#             stage_result["low_candidate_mask"], f"{prefix}.low_candidate_mask"
#         )
#         high_mask = _require_tensor(
#             stage_result["high_candidate_mask"], f"{prefix}.high_candidate_mask"
#         )
#         for name, mask in (("low_candidate_mask", low_mask), ("high_candidate_mask", high_mask)):
#             if mask.dtype != torch.bool or mask.shape != scores.shape:
#                 raise ValueError(
#                     f"{prefix}.{name} must be bool {tuple(scores.shape)}, got "
#                     f"{mask.dtype} {tuple(mask.shape)}"
#                 )
#         if torch.any(low_mask & high_mask):
#             raise ValueError(f"{prefix} low/high candidate masks overlap")
#         if torch.any(low_mask & (score_f > tau_1 + score_atol)):
#             raise ValueError(f"{prefix} low candidates violate tau_1")
#         if torch.any(high_mask & (score_f < tau_2 - score_atol)):
#             raise ValueError(f"{prefix} high candidates violate tau_2")

#         scale_feasible = _require_tensor(
#             stage_result["scale_feasible_mask"],
#             f"{prefix}.scale_feasible_mask",
#         )
#         children_inside = _require_tensor(
#             stage_result["children_inside_pc_range_mask"],
#             f"{prefix}.children_inside_pc_range_mask",
#         )
#         split_mask = _require_tensor(
#             stage_result["split_mask"], f"{prefix}.split_mask"
#         )
#         for name, mask in (
#             ("scale_feasible_mask", scale_feasible),
#             ("children_inside_pc_range_mask", children_inside),
#             ("split_mask", split_mask),
#         ):
#             if mask.dtype != torch.bool or mask.shape != scores.shape:
#                 raise ValueError(
#                     f"{prefix}.{name} must be bool {tuple(scores.shape)}, got "
#                     f"{mask.dtype} {tuple(mask.shape)}"
#                 )
#         if torch.any(split_mask & ~high_mask):
#             raise ValueError(f"{prefix}.split_mask is not a subset of high candidates")
#         if torch.any(split_mask & ~scale_feasible):
#             raise ValueError(f"{prefix} split contains an infeasible child scale")
#         if torch.any(split_mask & ~children_inside):
#             raise ValueError(f"{prefix} split contains an out-of-range child")

#         opacity = _require_tensor(
#             stage_result["gaussian_opacity"], f"{prefix}.gaussian_opacity"
#         ).detach().float()
#         density_mass = _require_tensor(
#             stage_result["density_mass"], f"{prefix}.density_mass"
#         ).detach().float()
#         if opacity.shape != scores.shape or density_mass.shape != scores.shape:
#             raise ValueError(
#                 f"{prefix} opacity/density_mass must match {tuple(scores.shape)}"
#             )
#         if not torch.isfinite(opacity).all() or not torch.isfinite(density_mass).all():
#             raise FloatingPointError(f"{prefix} contains non-finite deletion statistics")
#         if torch.any(opacity < 0.0) or torch.any(opacity >= 1.0):
#             raise ValueError(f"{prefix}.gaussian_opacity must lie inside [0, 1)")
#         if torch.any(density_mass < 0.0):
#             raise ValueError(f"{prefix}.density_mass must be non-negative")

#         opacity_threshold = _require_tensor(
#             stage_result["opacity_delete_threshold"],
#             f"{prefix}.opacity_delete_threshold",
#         ).detach().float()
#         mass_threshold = _require_tensor(
#             stage_result["density_mass_delete_threshold"],
#             f"{prefix}.density_mass_delete_threshold",
#         ).detach().float()
#         if opacity_threshold.shape != (batch_size, 1):
#             raise ValueError(f"{prefix}.opacity_delete_threshold must be [B, 1]")
#         if mass_threshold.shape != (batch_size, 1):
#             raise ValueError(
#                 f"{prefix}.density_mass_delete_threshold must be [B, 1]"
#             )

#         low_opacity = _require_tensor(
#             stage_result["low_opacity_mask"], f"{prefix}.low_opacity_mask"
#         )
#         low_mass = _require_tensor(
#             stage_result["low_density_mass_mask"],
#             f"{prefix}.low_density_mass_mask",
#         )
#         delete_mask = _require_tensor(
#             stage_result["delete_mask"], f"{prefix}.delete_mask"
#         )
#         delete_candidate = _require_tensor(
#             stage_result["delete_candidate_mask"],
#             f"{prefix}.delete_candidate_mask",
#         )
#         for name, mask in (
#             ("low_opacity_mask", low_opacity),
#             ("low_density_mass_mask", low_mass),
#             ("delete_candidate_mask", delete_candidate),
#             ("delete_mask", delete_mask),
#         ):
#             if mask.dtype != torch.bool or mask.shape != scores.shape:
#                 raise ValueError(
#                     f"{prefix}.{name} must be bool {tuple(scores.shape)}"
#                 )
#         expected_low_opacity = opacity <= opacity_threshold
#         expected_low_mass = density_mass <= mass_threshold
#         if torch.any(low_opacity != expected_low_opacity):
#             raise ValueError(f"{prefix}.low_opacity_mask is inconsistent")
#         if torch.any(low_mass != expected_low_mass):
#             raise ValueError(f"{prefix}.low_density_mass_mask is inconsistent")
#         expected_delete_candidate = low_mask & low_opacity & low_mass
#         if torch.any(delete_candidate != expected_delete_candidate):
#             raise ValueError(f"{prefix}.delete_candidate_mask is inconsistent")
#         if torch.any(delete_mask & ~delete_candidate):
#             raise ValueError(
#                 f"{prefix}.delete_mask is not a subset of delete candidates"
#             )
#         if torch.any(delete_mask & ~low_mask):
#             raise ValueError(f"{prefix}.delete_mask is not a subset of low candidates")
#         if torch.any(delete_mask & ~low_opacity):
#             raise ValueError(f"{prefix} deletes a non-low-opacity Gaussian")
#         if torch.any(delete_mask & ~low_mass):
#             raise ValueError(f"{prefix} deletes a non-low-mass Gaussian")
#         if torch.any(delete_mask & split_mask):
#             raise ValueError(f"{prefix} splits and deletes the same Gaussian")

#         merge_candidate = _require_tensor(
#             stage_result["merge_candidate_mask"],
#             f"{prefix}.merge_candidate_mask",
#         )
#         merge_member = _require_tensor(
#             stage_result["merge_member_mask"], f"{prefix}.merge_member_mask"
#         )
#         merge_partner = _require_tensor(
#             stage_result["merge_partner_index"],
#             f"{prefix}.merge_partner_index",
#         )
#         for name, mask in (
#             ("merge_candidate_mask", merge_candidate),
#             ("merge_member_mask", merge_member),
#         ):
#             if mask.dtype != torch.bool or mask.shape != scores.shape:
#                 raise ValueError(
#                     f"{prefix}.{name} must be bool {tuple(scores.shape)}"
#                 )
#         if merge_partner.dtype != torch.long or merge_partner.shape != scores.shape:
#             raise ValueError(
#                 f"{prefix}.merge_partner_index must be long "
#                 f"{tuple(scores.shape)}"
#             )
#         if torch.any(merge_candidate & ~low_mask):
#             raise ValueError(
#                 f"{prefix}.merge_candidate_mask is not a subset of low candidates"
#             )
#         if torch.any(merge_candidate & delete_candidate):
#             raise ValueError(
#                 f"{prefix} proposes a delete candidate for merging"
#             )
#         if torch.any(merge_member & ~merge_candidate):
#             raise ValueError(f"{prefix} merges a non-candidate Gaussian")
#         if torch.any(merge_member & (split_mask | delete_mask)):
#             raise ValueError(f"{prefix} merge overlaps split/delete decisions")
#         if torch.any((merge_partner >= 0) != merge_member):
#             raise ValueError(
#                 f"{prefix}.merge_partner_index and merge_member_mask disagree"
#             )

#         merge_pairs = _require_tensor(
#             stage_result["merge_pairs"], f"{prefix}.merge_pairs"
#         )
#         merge_geometry_similarity = _require_tensor(
#             stage_result["merge_geometry_similarity"],
#             f"{prefix}.merge_geometry_similarity",
#         ).detach().float()
#         merge_semantic_similarity = _require_tensor(
#             stage_result["merge_semantic_similarity"],
#             f"{prefix}.merge_semantic_similarity",
#         ).detach().float()
#         if merge_pairs.dtype != torch.long or (
#             merge_pairs.ndim != 2 or merge_pairs.shape[-1] != 2
#         ):
#             raise ValueError(f"{prefix}.merge_pairs must be long [P, 2]")
#         pair_count = merge_pairs.shape[0]
#         if merge_geometry_similarity.shape != (pair_count,) or (
#             merge_semantic_similarity.shape != (pair_count,)
#         ):
#             raise ValueError(
#                 f"{prefix} merge similarities must both have shape [P]"
#             )
#         if not torch.isfinite(merge_geometry_similarity).all() or not (
#             torch.isfinite(merge_semantic_similarity).all()
#         ):
#             raise FloatingPointError(f"{prefix} has non-finite merge similarity")

#         merge_geometry_threshold = _require_tensor(
#             stage_result["merge_geometry_threshold"],
#             f"{prefix}.merge_geometry_threshold",
#         ).detach().float()
#         merge_semantic_threshold = _require_tensor(
#             stage_result["merge_semantic_threshold"],
#             f"{prefix}.merge_semantic_threshold",
#         ).detach().float()
#         for name, threshold in (
#             ("merge_geometry_threshold", merge_geometry_threshold),
#             ("merge_semantic_threshold", merge_semantic_threshold),
#         ):
#             if threshold.shape != (batch_size, 1):
#                 raise ValueError(f"{prefix}.{name} must be [B, 1]")
#             if torch.any((threshold < 0.0) | (threshold > 1.0)):
#                 raise ValueError(f"{prefix}.{name} must lie in [0, 1]")

#         if pair_count > 0:
#             if batch_size != 1:
#                 raise ValueError(
#                     f"{prefix} dynamic merging currently requires batch size one"
#                 )
#             if torch.any(merge_pairs < 0) or torch.any(
#                 merge_pairs >= num_gaussians
#             ):
#                 raise ValueError(f"{prefix}.merge_pairs has an invalid index")
#             if torch.any(merge_pairs[:, 0] >= merge_pairs[:, 1]):
#                 raise ValueError(
#                     f"{prefix}.merge_pairs must use canonical i < j ordering"
#                 )
#             if torch.unique(merge_pairs.flatten()).numel() != 2 * pair_count:
#                 raise ValueError(f"{prefix}.merge_pairs are not disjoint")
#             expected_member = torch.zeros_like(merge_member[0])
#             expected_member[merge_pairs.flatten()] = True
#             if torch.any(expected_member != merge_member[0]):
#                 raise ValueError(f"{prefix}.merge_member_mask is inconsistent")
#             left, right = merge_pairs[:, 0], merge_pairs[:, 1]
#             if torch.any(merge_partner[0, left] != right) or torch.any(
#                 merge_partner[0, right] != left
#             ):
#                 raise ValueError(
#                     f"{prefix}.merge_partner_index is not symmetric"
#                 )
#             if torch.any(
#                 merge_geometry_similarity
#                 < merge_geometry_threshold[0, 0] - score_atol
#             ):
#                 raise ValueError(
#                     f"{prefix} accepted a pair below geometry threshold"
#                 )
#             if torch.any(
#                 merge_semantic_similarity
#                 < merge_semantic_threshold[0, 0] - score_atol
#             ):
#                 raise ValueError(
#                     f"{prefix} accepted a pair below semantic threshold"
#                 )
#         elif torch.any(merge_member) or torch.any(merge_partner >= 0):
#             raise ValueError(f"{prefix} has merge members but no merge pairs")

#         enabled = _require_tensor(
#             stage_result["control_enabled"], f"{prefix}.control_enabled"
#         )
#         split_enabled = _require_tensor(
#             stage_result["split_enabled"], f"{prefix}.split_enabled"
#         )
#         delete_enabled = _require_tensor(
#             stage_result["delete_enabled"], f"{prefix}.delete_enabled"
#         )
#         merge_enabled = _require_tensor(
#             stage_result["merge_enabled"], f"{prefix}.merge_enabled"
#         )
#         stage_ids = _require_tensor(
#             stage_result["control_stage"], f"{prefix}.control_stage"
#         )
#         if enabled.dtype != torch.bool or enabled.shape != (batch_size,):
#             raise ValueError(f"{prefix}.control_enabled must be bool [B]")
#         if stage_ids.shape != (batch_size,):
#             raise ValueError(f"{prefix}.control_stage must have shape [B]")
#         if split_enabled.dtype != torch.bool or split_enabled.shape != (batch_size,):
#             raise ValueError(f"{prefix}.split_enabled must be bool [B]")
#         if delete_enabled.dtype != torch.bool or delete_enabled.shape != (batch_size,):
#             raise ValueError(f"{prefix}.delete_enabled must be bool [B]")
#         if merge_enabled.dtype != torch.bool or merge_enabled.shape != (batch_size,):
#             raise ValueError(f"{prefix}.merge_enabled must be bool [B]")
#         if torch.any(split_enabled & ~enabled):
#             raise ValueError(f"{prefix} enables splitting while control is disabled")
#         if torch.any(delete_enabled & ~enabled):
#             raise ValueError(f"{prefix} enables deletion while control is disabled")
#         if torch.any(merge_enabled & ~enabled):
#             raise ValueError(f"{prefix} enables merging while control is disabled")
#         if torch.any(stage_ids != expected_stage):
#             raise ValueError(
#                 f"{prefix}.control_stage must equal its list index {expected_stage}"
#             )
#         disabled_rows = ~enabled
#         if torch.any(low_mask[disabled_rows]) or torch.any(high_mask[disabled_rows]):
#             raise ValueError(f"{prefix} has candidates while control is disabled")
#         if torch.any(split_mask[~split_enabled]):
#             raise ValueError(f"{prefix} splits Gaussians while splitting is disabled")
#         if torch.any(delete_mask[~delete_enabled]):
#             raise ValueError(f"{prefix} deletes Gaussians while deletion is disabled")
#         if torch.any(
#             delete_mask[delete_enabled] != delete_candidate[delete_enabled]
#         ):
#             raise ValueError(
#                 f"{prefix} does not apply all candidates while deletion is enabled"
#             )
#         expected_merge_candidate = low_mask & ~delete_candidate
#         if torch.any(merge_candidate[~merge_enabled]):
#             raise ValueError(
#                 f"{prefix} has merge candidates while merging is disabled"
#             )
#         if torch.any(
#             merge_candidate[merge_enabled]
#             != expected_merge_candidate[merge_enabled]
#         ):
#             raise ValueError(
#                 f"{prefix}.merge_candidate_mask does not exclude delete candidates"
#             )
#         if torch.any(merge_member[~merge_enabled]):
#             raise ValueError(f"{prefix} merges Gaussians while merging is disabled")

#         num_before = _require_tensor(
#             stage_result["num_gaussians_before"],
#             f"{prefix}.num_gaussians_before",
#         )
#         num_split = _require_tensor(
#             stage_result["num_split"], f"{prefix}.num_split"
#         )
#         num_deleted = _require_tensor(
#             stage_result["num_deleted"], f"{prefix}.num_deleted"
#         )
#         num_merge_pairs = _require_tensor(
#             stage_result["num_merge_pairs"], f"{prefix}.num_merge_pairs"
#         )
#         num_after = _require_tensor(
#             stage_result["num_gaussians_after"],
#             f"{prefix}.num_gaussians_after",
#         )
#         for name, tensor in (
#             ("num_gaussians_before", num_before),
#             ("num_split", num_split),
#             ("num_deleted", num_deleted),
#             ("num_merge_pairs", num_merge_pairs),
#             ("num_gaussians_after", num_after),
#         ):
#             if tensor.shape != (batch_size,) or tensor.dtype != torch.long:
#                 raise ValueError(f"{prefix}.{name} must be long [B]")
#         expected_before = torch.full_like(num_before, num_gaussians)
#         expected_split = split_mask.sum(dim=1).to(dtype=torch.long)
#         expected_deleted = delete_mask.sum(dim=1).to(dtype=torch.long)
#         if torch.any(num_before != expected_before):
#             raise ValueError(f"{prefix}.num_gaussians_before is inconsistent")
#         if torch.any(num_split != expected_split):
#             raise ValueError(f"{prefix}.num_split is inconsistent with split_mask")
#         if torch.any(num_deleted != expected_deleted):
#             raise ValueError(f"{prefix}.num_deleted is inconsistent with delete_mask")
#         expected_merge_pairs = torch.zeros_like(num_merge_pairs)
#         if batch_size == 1:
#             expected_merge_pairs[0] = pair_count
#         if torch.any(num_merge_pairs != expected_merge_pairs):
#             raise ValueError(f"{prefix}.num_merge_pairs is inconsistent")
#         if torch.any(
#             num_after
#             != num_before + num_split - num_deleted - num_merge_pairs
#         ):
#             raise ValueError(f"{prefix}.num_gaussians_after is inconsistent")

#         gaussians = stage_result["gaussian"]
#         if not hasattr(gaussians, "means"):
#             raise TypeError(f"{prefix}.gaussian must provide a means tensor")
#         means = _require_tensor(gaussians.means, f"{prefix}.gaussian.means")
#         if means.shape[:2] != scores.shape or means.shape[-1] != 3:
#             raise ValueError(
#                 f"{prefix}.gaussian and scores are misaligned: "
#                 f"means={tuple(means.shape)}, scores={tuple(scores.shape)}"
#             )

#         total = batch_size * num_gaussians
#         summaries.append(
#             {
#                 "stage": expected_stage,
#                 "batch_size": batch_size,
#                 "num_gaussians": num_gaussians,
#                 "enabled_ratio": enabled.float().mean().item(),
#                 "score_mean": score_f.mean().item(),
#                 "score_std": actual_std.mean().item(),
#                 "score_min": score_f.min().item(),
#                 "score_max": score_f.max().item(),
#                 "tau_1": tau_1.mean().item(),
#                 "tau_2": tau_2.mean().item(),
#                 "low_count": int(low_mask.sum().item()),
#                 "high_count": int(high_mask.sum().item()),
#                 "low_ratio": low_mask.sum().item() / total,
#                 "high_ratio": high_mask.sum().item() / total,
#                 "split_count": int(split_mask.sum().item()),
#                 "split_ratio": split_mask.sum().item() / total,
#                 "delete_count": int(delete_mask.sum().item()),
#                 "delete_ratio": delete_mask.sum().item() / total,
#                 "delete_candidate_count": int(delete_candidate.sum().item()),
#                 "merge_candidate_count": int(merge_candidate.sum().item()),
#                 "merge_pair_count": pair_count,
#                 "merge_reduction_ratio": pair_count / total,
#                 "merge_geometry_mean": (
#                     merge_geometry_similarity.mean().item()
#                     if pair_count > 0 else 0.0
#                 ),
#                 "merge_semantic_mean": (
#                     merge_semantic_similarity.mean().item()
#                     if pair_count > 0 else 0.0
#                 ),
#                 "merge_geometry_threshold": (
#                     merge_geometry_threshold.mean().item()
#                 ),
#                 "merge_semantic_threshold": (
#                     merge_semantic_threshold.mean().item()
#                 ),
#                 "num_gaussians_after": int(num_after.sum().item()),
#                 "opacity_p10": torch.quantile(opacity, 0.10).item(),
#                 "opacity_mean": opacity.mean().item(),
#                 "density_mass_p10": torch.quantile(density_mass, 0.10).item(),
#                 "density_mass_mean": density_mass.mean().item(),
#                 "opacity_delete_threshold": opacity_threshold.mean().item(),
#                 "density_mass_delete_threshold": mass_threshold.mean().item(),
#                 "near_zero_ratio": (score_f < 0.01).float().mean().item(),
#                 "near_one_ratio": (score_f > 0.99).float().mean().item(),
#             }
#         )
#     return summaries


# def format_control_summary(summaries: Sequence[Dict[str, float]]) -> str:
#     if not summaries:
#         return "no adaptive-control stages"
#     stage_messages = []
#     for summary in summaries:
#         stage_messages.append(
#             "S{stage} N={num_gaussians} enabled={enabled_ratio:.0%} "
#             "score={score_mean:.4f}+/-{score_std:.4f}"
#             "[{score_min:.4f},{score_max:.4f}] "
#             "tau=({tau_1:.4f},{tau_2:.4f}) "
#             "low={low_count}({low_ratio:.2%}) "
#             "high={high_count}({high_ratio:.2%}) "
#             "split={split_count}({split_ratio:.2%}) "
#             "delete={delete_count}/{delete_candidate_count}"
#             "({delete_ratio:.2%}) "
#             "merge={merge_pair_count}/{merge_candidate_count}"
#             "({merge_reduction_ratio:.2%})->N{num_gaussians_after} "
#             "merge_sim(g/s)={merge_geometry_mean:.3f}/"
#             "{merge_semantic_mean:.3f} "
#             "merge_thr=({merge_geometry_threshold:.3f},"
#             "{merge_semantic_threshold:.3f}) "
#             "opa(p10/mean)={opacity_p10:.3g}/{opacity_mean:.3g} "
#             "mass(p10/mean)={density_mass_p10:.3g}/{density_mass_mean:.3g} "
#             "del_thr=({opacity_delete_threshold:.3g},"
#             "{density_mass_delete_threshold:.3g}) "
#             "sat=({near_zero_ratio:.2%},{near_one_ratio:.2%})".format(**summary)
#         )
#     return " | ".join(stage_messages)


# @torch.no_grad()
# def validate_score_head_gradients(model) -> Dict[str, float]:
#     """Check that explicit Score supervision reaches score-head parameters."""
#     if hasattr(model, "module"):
#         model = model.module
#     parameters = [
#         (name, parameter)
#         for name, parameter in model.named_parameters()
#         if "score_head" in name and parameter.requires_grad
#     ]
#     if not parameters:
#         raise RuntimeError(
#             "control_result exists but no trainable score_head parameters were found"
#         )

#     missing = [name for name, parameter in parameters if parameter.grad is None]
#     if len(missing) == len(parameters):
#         raise RuntimeError(
#             "All score_head gradients are None. Check that "
#             "GaussianComplexityLoss consumes control_result."
#         )

#     squared_norm = 0.0
#     nonfinite = []
#     for name, parameter in parameters:
#         if parameter.grad is None:
#             continue
#         gradient = parameter.grad.detach().float()
#         if not torch.isfinite(gradient).all():
#             nonfinite.append(name)
#         squared_norm += gradient.square().sum().item()
#     if nonfinite:
#         raise FloatingPointError(
#             f"Non-finite score_head gradients: {nonfinite}"
#         )

#     return {
#         "grad_norm": squared_norm ** 0.5,
#         "parameter_count": len(parameters),
#         "missing_count": len(missing),
#     }


# def format_score_gradient_summary(summary: Dict[str, float]) -> str:
#     return (
#         "grad_norm={grad_norm:.6g} params={parameter_count} "
#         "missing={missing_count}"
#     ).format(**summary)
