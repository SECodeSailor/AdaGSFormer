import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmseg.registry import MODELS


@MODELS.register_module()
class GaussianComplexityScore(BaseModule):

    def __init__(
        self,
        embed_dims: int = 256,
        hidden_dims: int = None,
        dropout: float = 0.0,
        init_cfg=None,
        **kwargs,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        hidden_dims = int(hidden_dims or embed_dims)
        if embed_dims <= 0 or hidden_dims <= 0:
            raise ValueError("embed_dims and hidden_dims must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")

        self.norm = nn.LayerNorm(embed_dims)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dims, hidden_dims),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims, 1),
        )

    # def init_weight(self) -> None:
    #     nn.init.xavier_uniform_(self.mlp[0].weight)
    #     nn.init.zeros_(self.mlp[0].bias)
    #     # Start from score=0.5 for every Gaussian instead of imposing a
    #     # random complexity ordering before the GT supervision is observed.
    #     nn.init.zeros_(self.mlp[-1].weight)
    #     nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor_embed: torch.Tensor,
    ) -> torch.Tensor:
        if instance_feature.shape != anchor_embed.shape:
            raise ValueError(
                "instance_feature and anchor_embed must have the same shape, "
                f"got {instance_feature.shape} and {anchor_embed.shape}"
            )
        fused_feature = self.norm(instance_feature + anchor_embed)
        return self.mlp(fused_feature).squeeze(-1)
