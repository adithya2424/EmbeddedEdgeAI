import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from model_metrics import get_sparsity

# the main trick is use torch.kthtensor to find kth smallest value to prune based on sparsity
def fine_grained_prune(tensor: torch.Tensor, sparsity: float) -> torch.Tensor:
    """
    magnitude-based pruning for single tensor
    :param tensor: torch.(cuda.)Tensor, weight of conv/fc layer
    :param sparsity: float, pruning sparsity
        sparsity = #zeros / #elements = 1 - #nonzeros / #elements
    :return:
        torch.(cuda.)Tensor, mask for zeros
    """
    sparsity = min(max(0.0, sparsity), 1.0)
    if sparsity == 1.0:
        tensor.zero_()
        return torch.zeros_like(tensor)
    elif sparsity == 0.0:
        return torch.ones_like(tensor)

    num_elements = tensor.numel()

    num_zeros = round(sparsity * num_elements)

    # Calculate the importance of each weight based on its absolute value
    importance = tensor.abs()

    # Determine the threshold for pruning based on the sparsity level
    # torch.kthvalue returns the kth smallest element of the input tensor
    # Since we are pruning the smallest weights by magnitude, we find the threshold value
    threshold = importance.flatten().kthvalue(num_zeros).values.item() if num_zeros else importance.min().item() - 1
    print(threshold)
    # Create a binary mask where elements above the threshold are 1, others are 0
    mask = importance.gt(threshold).type_as(tensor)

    # Apply mask to prune the tensor
    tensor.mul_(mask)

    return mask


class FineGrainedPruner:
    def __init__(self, model, sparsity_dict):
        self.masks = FineGrainedPruner.prune(model, sparsity_dict)

    @torch.no_grad()
    def apply(self, model):
        for name, param in model.named_parameters():
            if name in self.masks:
                param *= self.masks[name]

    @staticmethod
    @torch.no_grad()
    def prune(model, sparsity_dict):
        masks = dict()
        for name, param in model.named_parameters():
            if param.dim() > 1:  # we only prune conv and fc weights
                masks[name] = fine_grained_prune(param, sparsity_dict[name])
        return masks


