import torch, torch.nn as nn
import torch.nn.functional as F

from . import OPENOCC_LOSS
from .base_loss import BaseLoss


@OPENOCC_LOSS.register_module()
class BinaryCrossEntropyLoss(BaseLoss):

    def __init__(
        self,
        weight=1.0,
        empty_label=17,
        class_weights=[1.0, 1.0],
        ignore_label=255,
        balance_pos_neg=False,
        prediction_weights=None,
        input_dict=None
    ):
        
        super().__init__()

        self.weight = weight
        if input_dict is None:
            self.input_dict = {
                'bin_logits': 'bin_logits',
                'sampled_label': 'sampled_label',
                'occ_mask': 'occ_mask'
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.loss_voxel

        self.empty_label = empty_label
        self.ignore_label = ignore_label
        self.balance_pos_neg = bool(balance_pos_neg)
        self.prediction_weights = self._validate_prediction_weights(
            prediction_weights
        )
        if len(class_weights) != 2:
            raise ValueError("class_weights must be [empty, occupied]")
        self.class_weights = torch.tensor(class_weights)
        self.class_weights = 2 * F.normalize(self.class_weights, 1, -1)
        print(self.__class__, self.class_weights)

    @staticmethod
    def _validate_prediction_weights(prediction_weights):
        if prediction_weights is None:
            return None
        weights = tuple(float(value) for value in prediction_weights)
        if not weights:
            raise ValueError("prediction_weights must not be empty")
        if any(value < 0.0 for value in weights):
            raise ValueError("prediction_weights must be non-negative")
        weight_sum = sum(weights)
        if weight_sum <= 0.0:
            raise ValueError("prediction_weights must have a positive sum")
        return tuple(value / weight_sum for value in weights)

    def _get_prediction_weights(self, num_predictions):
        if num_predictions <= 0:
            raise ValueError("bin_logits must contain at least one prediction")
        # GaussianHead emits only the final decoder prediction during
        # validation.  That single D3 prediction must receive full weight;
        # the configured sequence applies only when deep-supervision outputs
        # D1/D2/D3 are all present during training.
        if num_predictions == 1:
            return (1.0,)
        if self.prediction_weights is None:
            return (1.0 / num_predictions,) * num_predictions
        if len(self.prediction_weights) != num_predictions:
            raise ValueError(
                "prediction_weights must match the selected bin_logits: "
                f"got {len(self.prediction_weights)} weights for "
                f"{num_predictions} predictions"
            )
        return self.prediction_weights

    def loss_voxel(self, bin_logits, sampled_label, occ_mask=None):

        tot_loss = 0.
        prediction_weights = self._get_prediction_weights(len(bin_logits))
        semantic_label = sampled_label.flatten(1)
        valid_mask = semantic_label != self.ignore_label
        if occ_mask is not None:
            occ_mask = occ_mask.flatten(1)
            if occ_mask.shape != semantic_label.shape:
                raise ValueError(
                    "occ_mask and sampled_label must have the same flattened "
                    f"shape, got {occ_mask.shape} and {semantic_label.shape}"
                )
            valid_mask = valid_mask & occ_mask.bool()

        target = semantic_label != self.empty_label
        
        for prediction_index, occupancy in enumerate(bin_logits): # b, n
            occupancy = occupancy.flatten(1)
            if occupancy.shape != semantic_label.shape:
                raise ValueError(
                    "Each occupancy prediction must match sampled_label, got "
                    f"{occupancy.shape} and {semantic_label.shape}"
                )
            occupancy = occupancy[valid_mask]
            target_valid = target[valid_mask]
            if occupancy.numel() == 0:
                loss = occupancy.sum() * 0.0
                tot_loss = (
                    tot_loss
                    + prediction_weights[prediction_index] * loss
                )
                continue

            occupancy = torch.clamp(occupancy, 1e-6, 1 - 1e-6)
            if self.balance_pos_neg:
                # Equivalent to sampling equal occupied/empty counts, but uses
                # all valid voxels produced by the tiled G2V kernel.
                num_pos = target_valid.sum().to(dtype=occupancy.dtype)
                num_neg = (~target_valid).sum().to(dtype=occupancy.dtype)
                sample_weight = torch.zeros_like(occupancy)
                if num_pos > 0:
                    sample_weight[target_valid] = 0.5 / num_pos
                if num_neg > 0:
                    sample_weight[~target_valid] = 0.5 / num_neg
                if num_pos == 0 or num_neg == 0:
                    sample_weight.fill_(1.0 / occupancy.numel())
                loss = nn.functional.binary_cross_entropy(
                    occupancy,
                    target_valid.float(),
                    sample_weight,
                    reduction="sum",
                )
            else:
                class_weights = self.class_weights.to(
                    device=occupancy.device, dtype=occupancy.dtype
                )
                sample_weight = torch.where(
                    target_valid,
                    class_weights[1],
                    class_weights[0],
                )
                loss = nn.functional.binary_cross_entropy(
                    occupancy,
                    target_valid.float(),
                    sample_weight,
                )
            tot_loss = (
                tot_loss
                + prediction_weights[prediction_index] * loss
            )
        return tot_loss
    

@OPENOCC_LOSS.register_module()
class PixelDistributionLoss(BaseLoss):

    def __init__(
        self,
        weight=1.0,
        use_sigmoid=True,
        input_dict=None
    ):
        
        super().__init__(weight)

        if input_dict is None:
            self.input_dict = {
                'pixel_logits': 'pixel_logits',
                'pixel_gt': 'pixel_gt',
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.loss_voxel
        self.use_sigmoid = use_sigmoid

    def loss_voxel(self, pixel_logits, pixel_gt):
        if self.use_sigmoid:
            pixel_logits = torch.sigmoid(pixel_logits)
        else:
            pixel_logits = torch.softmax(pixel_logits, dim=-1)
        loss = nn.functional.binary_cross_entropy(pixel_logits, pixel_gt.float())
        return loss
    
@OPENOCC_LOSS.register_module()
class OccDepthLoss(BaseLoss):

    def __init__(
        self,
        weight=1.0,
        input_dict=None
    ):
        
        super().__init__(weight)

        if input_dict is None:
            self.input_dict = {
                'pixel_logits': 'pixel_logits',
                'pixel_gt': 'pixel_gt',
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.loss_voxel

    def loss_voxel(self, pixel_logits, pixel_gt):
        pixel_logits = pixel_logits.permute(0, 4, 1, 2, 3)
        ### get depth gt from occ
        occ_depth = pixel_gt.float().argmax(dim=-1) # b, n, h, w
        loss = nn.functional.cross_entropy(pixel_logits, occ_depth)
        return loss
