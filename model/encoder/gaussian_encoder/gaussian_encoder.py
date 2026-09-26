from typing import List, Optional
import torch, torch.nn as nn

from mmseg.registry import MODELS
from mmengine import build_from_cfg
from ..base_encoder import BaseEncoder


@MODELS.register_module()
class GaussianOccEncoder(BaseEncoder):
    def __init__(
        self,
        anchor_encoder: dict,
        norm_layer: dict,
        ffn: dict,
        deformable_model: dict,
        refine_layer: dict,
        adaptive_control_layer: dict = None,
        share_adaptive_control_layer: bool = True,
        mid_refine_layer: dict = None,
        spconv_layer: dict = None,
        num_decoder: int = 6,
        operation_order: Optional[List[str]] = None,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(init_cfg)
        self.num_decoder = num_decoder

        if operation_order is None:
            operation_order = [
                "spconv",
                "norm",
                "deformable",
                "norm",
                "ffn",
                "norm",
                "refine",
            ] * num_decoder
        self.operation_order = operation_order

        # =========== build modules ===========
        def build(cfg, registry):
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)

        self.anchor_encoder = build(anchor_encoder, MODELS)
        self.op_config_map = {
            "norm": [norm_layer, MODELS],
            "ffn": [ffn, MODELS],
            "deformable": [deformable_model, MODELS],
            "refine": [refine_layer, MODELS],
            "mid_refine":[mid_refine_layer, MODELS],
            "spconv": [spconv_layer, MODELS],
            "adaptive_control": [adaptive_control_layer, MODELS],
        }
        if "score" in self.operation_order:
            raise ValueError(
                "Use one 'adaptive_control' op instead of a separate 'score' "
                "op. Score and threshold generation are now coupled."
            )
        if (
            "adaptive_control" in self.operation_order
            and adaptive_control_layer is None
        ):
            raise ValueError(
                "adaptive_control_layer must be configured when "
                "operation_order contains 'adaptive_control'"
            )

        # All controlled decoder stages share Score and threshold settings by
        # default, keeping complexity values comparable across iterations.
        shared_adaptive_control = (
            build(adaptive_control_layer, MODELS)
            if (
                share_adaptive_control_layer
                and adaptive_control_layer is not None
                and "adaptive_control" in self.operation_order
            )
            else None
        )
        layers = []
        for op in self.operation_order:
            if op == "adaptive_control" and shared_adaptive_control is not None:
                layers.append(shared_adaptive_control)
            else:
                layers.append(
                    build(*self.op_config_map.get(op, [None, None]))
                )
        self.layers = nn.ModuleList(layers)
        
    def init_weights(self):
        initialized_layers = set()
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op != "refine" and id(self.layers[i]) not in initialized_layers:
                for p in self.layers[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
            initialized_layers.add(id(self.layers[i]))
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def forward(
        self,
        representation,
        rep_features,
        ms_img_feats=None,
        metas=None,
        **kwargs
    ):
        feature_maps = ms_img_feats
        if isinstance(feature_maps, torch.Tensor):
            feature_maps = [feature_maps]
        instance_feature = rep_features
        anchor = representation
        # Persistent topology provenance, age and consecutive prune evidence.
        # AdaptiveGaussianControl updates all three in exactly the same order
        # as its variable-length anchor output.
        last_operation = torch.zeros(
            anchor.shape[:2], dtype=torch.long, device=anchor.device
        )
        operation_age = torch.zeros_like(last_operation)
        prune_evidence = torch.zeros_like(last_operation)

        anchor_embed = self.anchor_encoder(anchor)

        prediction = []
        control_results = []
        adaptive_control_stage = 0
        disable_split_at_eval = bool(
            kwargs.get("disable_split_at_eval", False)
        )
        if disable_split_at_eval and self.training:
            raise RuntimeError(
                "disable_split_at_eval is only valid for paired evaluation"
            )
        for i, op in enumerate(self.operation_order):
            if op == 'spconv':
                instance_feature = self.layers[i](
                    instance_feature,
                    anchor)
            elif op == "norm" or op == "ffn":
                instance_feature = self.layers[i](instance_feature)
            elif op == "identity":
                identity = instance_feature
            elif op == "add":
                instance_feature = instance_feature + identity
            elif op == "deformable":
                instance_feature = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    feature_maps,
                    metas,
                )
            elif "refine" in op:
                anchor, gaussian = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                )
            
                prediction.append({'gaussian': gaussian})
                if i != len(self.operation_order) - 1:
                    anchor_embed = self.anchor_encoder(anchor)
            elif op == "adaptive_control":
                if not prediction:
                    raise RuntimeError(
                        "An adaptive_control operation must appear after refine"
                    )
                (
                    instance_feature,
                    anchor,
                    last_operation,
                    operation_age,
                    prune_evidence,
                    control_result,
                ) = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    gaussian=prediction[-1]["gaussian"],
                    stage_index=adaptive_control_stage,
                    last_operation=last_operation,
                    operation_age=operation_age,
                    prune_evidence=prune_evidence,
                    global_iter=kwargs.get("global_iter"),
                    disable_split_at_eval=disable_split_at_eval,
                    collect_split_diagnostics=bool(
                        kwargs.get("collect_split_diagnostics", False)
                    ),
                )
                # Keep topology-control data separate from decoder
                # representations.  The Gaussian stored here is the exact
                # stage on which score_logits were computed, so complexity
                # supervision cannot become misaligned with another list.
                stage_control_result = dict(control_result)
                stage_control_result["gaussian"] = prediction[-1]["gaussian"]
                control_results.append(stage_control_result)
                adaptive_control_stage += 1
                # Split/delete/merge changes the anchor topology, so the next
                # decoder stage must embed the updated representation.
                anchor_embed = self.anchor_encoder(anchor)
            else:
                raise NotImplementedError(f"{op} is not supported.")

        return {
            "representation": prediction,
            "control_result": control_results,
        }
