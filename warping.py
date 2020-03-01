import torch
from torch import nn


def warp_forward(base_images, base_disparities, base_occlusions=None, invert=True):
    def scatter(sources, targets, indices, dim):
        targets.scatter_(dim, indices, sources)
        return targets

    if base_disparities.dim() == 4:
        base_disparities = base_disparities.squeeze(1)
    y_coords = base_disparities.new_tensor(range(base_disparities.size(-2)))
    x_coords = base_disparities.new_tensor(range(base_disparities.size(-1)))
    y_coords, x_coords = torch.meshgrid(y_coords, x_coords)
    x_coords = x_coords.unsqueeze(0).expand_as(base_disparities)
    x_coords = x_coords + (-base_disparities if invert else base_disparities)
    x_coords = torch.clamp(x_coords, -1, x_coords.size(-1))
    # -> [B, H, W]
    if base_occlusions is not None:
        if base_occlusions.dim() == 4:
            base_occlusions = base_occlusions.squeeze(1)
        x_coords = torch.where(base_occlusions > 0, x_coords, torch.full_like(x_coords, -1))
    x_indices = torch.stack((torch.floor(x_coords), torch.ceil(x_coords)), dim=-1)
    # -> [B, H, W, 2]
    unilinear_weights = torch.abs(x_coords.unsqueeze(-1) - x_indices)
    # -> [B, H, W, 2]
    zeros = unilinear_weights.new_zeros(*unilinear_weights.size()[:-1], unilinear_weights.size(-2) + 2)
    # -> [B, H, W, W + 2]
    unilinear_weights = scatter(unilinear_weights, zeros, x_indices.long() + 1, -1)
    # -> [B, H, W, W + 2]
    unilinear_weights = unilinear_weights[..., 1:-1]
    # -> [B, H, W, W]
    base_images = base_images.unsqueeze(-1)
    # -> [B, C, H, W, 1]
    unilinear_weights = unilinear_weights.unsqueeze(1)
    # -> [B, 1, H, W, W]
    match_images = torch.sum(base_images * unilinear_weights, dim=-2) / torch.sum(unilinear_weights, dim=-2)
    # -> [B, C, H, W]
    return match_images


def warp_backward(match_images, base_disparities, invert=True):
    def normalize(inputs, mean, std):
        return (inputs - mean) / std

    if base_disparities.dim() == 4:
        base_disparities = base_disparities.squeeze(1)
    y_coords = base_disparities.new_tensor(range(base_disparities.size(-2)))
    x_coords = base_disparities.new_tensor(range(base_disparities.size(-1)))
    y_coords, x_coords = torch.meshgrid(y_coords, x_coords)
    y_coords = y_coords.unsqueeze(0).expand_as(base_disparities)
    x_coords = x_coords.unsqueeze(0).expand_as(base_disparities)
    x_coords = x_coords + (-base_disparities if invert else base_disparities)
    y_coords = normalize(y_coords, (match_images.size(-2) - 1) / 2, (match_images.size(-2) - 1) / 2)
    x_coords = normalize(x_coords, (match_images.size(-1) - 1) / 2, (match_images.size(-1) - 1) / 2)
    coords = torch.stack((x_coords, y_coords), dim=-1)
    match_images = nn.functional.grid_sample(match_images, coords, mode="bilinear", padding_mode="border")
    return match_images
