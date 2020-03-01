import torch
from torch import nn


def warp_forward_dense(base_images, base_disparities, base_occlusions=None, invert=True):
    if base_disparities.dim() == 4:
        base_disparities = base_disparities.squeeze(1)
    y_coords = base_disparities.new_tensor(range(base_disparities.size(-2)))
    x_coords = base_disparities.new_tensor(range(base_disparities.size(-1)))
    y_coords, x_coords = torch.meshgrid(y_coords, x_coords)
    x_coords = x_coords.unsqueeze(0).expand_as(base_disparities)
    x_coords = x_coords + (-base_disparities if invert else base_disparities)
    # -> [B, H, W]
    if base_occlusions is not None:
        if base_occlusions.dim() == 4:
            base_occlusions = base_occlusions.squeeze(1)
        x_coords = torch.where(base_occlusions > 0, x_coords, torch.full_like(x_coords, -1))
    x_indices = torch.stack((torch.floor(x_coords), torch.floor(x_coords) + 1), dim=-1)
    # -> [B, H, W, 2]
    unilinear_weights = torch.abs(x_coords.unsqueeze(-1) - x_indices)
    unilinear_weights = torch.flip(unilinear_weights, dims=(-1,))
    # -> [B, H, W, 2]
    zeros = unilinear_weights.new_zeros(*unilinear_weights.size()[:-1], unilinear_weights.size(-2) + 2)
    # -> [B, H, W, W + 2]
    x_indices = torch.clamp(x_indices, -1, x_indices.size(-2)) + 1
    unilinear_weights = zeros.scatter(-1, x_indices.long(), unilinear_weights)
    # -> [B, H, W, W + 2]
    unilinear_weights = unilinear_weights[..., 1:-1]
    # -> [B, H, W, W]
    base_images = base_images.unsqueeze(-1)
    # -> [B, C, H, W, 1]
    unilinear_weights = unilinear_weights.unsqueeze(1)
    # -> [B, 1, H, W, W]
    match_images = torch.sum(base_images * unilinear_weights, dim=-2)
    normalization_factors = torch.sum(unilinear_weights, dim=-2)
    normalization_factors = torch.clamp(normalization_factors, 1e-6)
    match_images = match_images / normalization_factors
    # -> [B, C, H, W]
    return match_images


def warp_forward_sparse(base_images, base_disparities, base_occlusions=None, invert=True):
    if base_disparities.dim() == 3:
        base_disparities = base_disparities.unsqueeze(1)
    y_coords = base_disparities.new_tensor(range(base_disparities.size(-2)))
    x_coords = base_disparities.new_tensor(range(base_disparities.size(-1)))
    y_coords, x_coords = torch.meshgrid(y_coords, x_coords)
    x_coords = x_coords.unsqueeze(0).unsqueeze(1).expand_as(base_images)
    x_coords = x_coords + (-base_disparities if invert else base_disparities)
    # -> [B, C, H, W]
    if base_occlusions is not None:
        if base_occlusions.dim() == 3:
            base_occlusions = base_occlusions.unsqueeze(1)
        x_coords = torch.where(base_occlusions > 0, x_coords, torch.full_like(x_coords, -1))
    x_indices = torch.stack((torch.floor(x_coords), torch.floor(x_coords) + 1), dim=-1)
    # -> [B, C, H, W, 2]
    unilinear_weights = torch.abs(x_coords.unsqueeze(-1) - x_indices)
    unilinear_weights = torch.flip(unilinear_weights, dims=(-1,))
    # -> [B, C, H, W, 2]
    sparse_indices1 = torch.stack(
        torch.meshgrid(*(torch.arange(size, device=x_indices.device) for size in x_indices.size()))
    ).flatten(1, -1)
    # -> [5, B * C * H * W * 2]
    sparse_indices2 = x_indices.long().flatten()
    # -> [B * C * H * W * 2]
    sparse_indices = torch.cat((sparse_indices1[:-1], sparse_indices2.unsqueeze(0)))
    # -> [5, B * C * H * W * 2]
    sparse_values = unilinear_weights.flatten()
    # -> [B * C * H * W * 2]
    index_masks = (0 <= sparse_indices2) & (sparse_indices2 < x_indices.size(-2))
    sparse_indices = sparse_indices[..., index_masks]
    sparse_values = sparse_values[index_masks]
    sparse_size = (*unilinear_weights.size()[:-1], unilinear_weights.size(-2))
    unilinear_weights = torch.sparse_coo_tensor(sparse_indices, sparse_values, sparse_size, device=unilinear_weights.device)
    # -> [B, C, H, W, W]
    base_images = base_images.unsqueeze(-1).expand_as(x_indices)
    sparse_values = base_images.flatten()
    sparse_values = sparse_values[index_masks]
    base_images = torch.sparse_coo_tensor(sparse_indices, sparse_values, sparse_size, device=base_images.device)
    # -> [B, C, H, W, W]
    match_images = torch.sparse.sum(base_images * unilinear_weights, dim=-2).to_dense()
    normalization_factors = torch.sparse.sum(unilinear_weights, dim=-2).to_dense()
    normalization_factors = torch.clamp(normalization_factors, 1e-6)
    match_images = match_images / normalization_factors
    # -> [B, C, H, W]
    return match_images


def warp_backward(match_images, base_disparities, invert=True):
    if base_disparities.dim() == 4:
        base_disparities = base_disparities.squeeze(1)
    y_coords = base_disparities.new_tensor(range(base_disparities.size(-2)))
    x_coords = base_disparities.new_tensor(range(base_disparities.size(-1)))
    y_coords, x_coords = torch.meshgrid(y_coords, x_coords)
    y_coords = y_coords.unsqueeze(0).expand_as(base_disparities)
    x_coords = x_coords.unsqueeze(0).expand_as(base_disparities)
    x_coords = x_coords + (-base_disparities if invert else base_disparities)
    y_coords = (y_coords - ((match_images.size(-2) - 1) / 2)) / ((match_images.size(-2) - 1) / 2)
    x_coords = (x_coords - ((match_images.size(-1) - 1) / 2)) / ((match_images.size(-1) - 1) / 2)
    coords = torch.stack((x_coords, y_coords), dim=-1)
    match_images = nn.functional.grid_sample(match_images, coords, mode="bilinear", padding_mode="border")
    return match_images
