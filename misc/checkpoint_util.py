from collections import OrderedDict

import torch


def refine_load_from_sd(sd):
    for k in list(sd.keys()):
        if 'img_neck.' in k or 'lifter.anchor' in k:
            del sd[k]
    return sd


def adapt_spconv_weight_layout(state_dict, model_state_dict):
    """Convert legacy sparse-convolution weights to the current layout.

    Some checkpoints store 3D convolution kernels in PyTorch's
    ``[out, in, D, H, W]`` layout, while current spconv modules expose
    ``[out, D, H, W, in]``.  Convert only when the source and destination
    shapes prove that exact permutation.  All other mismatches are left
    untouched so a subsequent strict load still reports real architecture
    incompatibilities.
    """
    if not isinstance(state_dict, (dict, OrderedDict)):
        raise TypeError("state_dict must be a mapping")
    if not isinstance(model_state_dict, (dict, OrderedDict)):
        raise TypeError("model_state_dict must be a mapping")

    adapted = state_dict.copy()
    if hasattr(state_dict, "_metadata"):
        adapted._metadata = state_dict._metadata
    converted_keys = []
    for key, source in state_dict.items():
        target = model_state_dict.get(key)
        if not isinstance(source, torch.Tensor) or not isinstance(
            target, torch.Tensor
        ):
            continue
        if source.shape == target.shape or source.ndim != 5:
            continue
        legacy_to_current_shape = (
            source.shape[0],
            source.shape[2],
            source.shape[3],
            source.shape[4],
            source.shape[1],
        )
        if legacy_to_current_shape != tuple(target.shape):
            continue
        adapted[key] = source.permute(0, 2, 3, 4, 1).contiguous()
        converted_keys.append(key)
    return adapted, converted_keys
