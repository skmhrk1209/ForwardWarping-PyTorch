import torch
from torch import nn


def multi_dim(reducer):

    def multi_dim_reducer(inputs, dims, keepdim=False):
        reduce_dims = [(dim + inputs.dim()) % inputs.dim() for dim in dims]
        keep_dims = [dim for dim in range(inputs.dim()) if dim not in reduce_dims]
        permute_dims = [*keep_dims, *reduce_dims]
        repermute_dims = [permute_dims.index(dim) for dim in range(inputs.dim())]
        inputs = inputs.permute(*permute_dims)
        inputs = inputs.flatten(len(keep_dims), -1)
        inputs = reducer(inputs, dim=-1)
        if keepdim:
            for i in reduce_dims:
                inputs = inputs.unsqueeze(-1)
            inputs = inputs.permute(*repermute_dims)
        return inputs

    return multi_dim_reducer


def linear_map(inputs, in_min=None, in_max=None, out_min=None, out_max=None):
    torch_min = lambda *args, **kwargs: torch.min(*args, **kwargs)[0]
    torch_max = lambda *args, **kwargs: torch.max(*args, **kwargs)[0]
    in_min = multi_dim(torch_min)(inputs, dims=range(1, inputs.dim()), keepdim=True) if in_min is None else in_min
    in_max = multi_dim(torch_max)(inputs, dims=range(1, inputs.dim()), keepdim=True) if in_max is None else in_max
    out_min = torch.zeros_like(in_min) if out_min is None else out_min
    out_max = torch.ones_like(in_max) if out_max is None else out_max
    inputs = (inputs - in_min) / (in_max - in_min) * (out_max - out_min) + out_min
    return inputs


def warp_forward(base_images, base_disparities, base_occlusions=None, invert=True):
    if base_disparities.dim() == 4:
        base_disparities = base_disparities.squeeze(1)
    coordinates_x = base_disparities.new_tensor(range(base_disparities.shape[-1]))
    coordinates_x = coordinates_x.reshape(1, 1, -1).expand_as(base_disparities)
    coordinates_x = coordinates_x + (-base_disparities if invert else base_disparities)
    if base_occlusions is not None:
        if base_occlusions.dim() == 4:
            base_occlusions = base_occlusions.squeeze(1)
        coordinates_x = torch.where(base_occlusions > 0, coordinates_x, torch.full_like(coordinates_x, -1))
    indices_x = torch.stack((torch.floor(coordinates_x), torch.floor(coordinates_x) + 1), dim=-1)
    unilinear_weights = torch.abs(coordinates_x.unsqueeze(-1) - indices_x)
    unilinear_weights = unilinear_weights.flip(-1)
    zeros = unilinear_weights.new_zeros(*unilinear_weights.shape[:-1], unilinear_weights.shape[-2] + 2)
    indices_x = torch.clamp(indices_x, -1, indices_x.shape[-2]) + 1
    unilinear_weights = zeros.scatter(-1, indices_x.long(), unilinear_weights)
    unilinear_weights = unilinear_weights[..., 1:-1]
    # NOTE: estimate occlusions
    if base_occlusions is None:
        coordinates_x = base_disparities.new_tensor(range(base_disparities.shape[-1]))
        coordinates_x = coordinates_x.reshape(1, 1, -1, 1)
        weight_masks = (unilinear_weights > 0).float()
        mean_coordinates_x = torch.sum(coordinates_x * weight_masks, dim=-2, keepdim=True)
        mean_coordinates_x = mean_coordinates_x / torch.sum(weight_masks, dim=-2, keepdim=True)
        std_coordinates_x = torch.sum((coordinates_x - mean_coordinates_x) ** 2 * weight_masks, dim=-2, keepdim=True)
        std_coordinates_x = std_coordinates_x / torch.sum(weight_masks, dim=-2, keepdim=True)
        coarse_occlusions = std_coordinates_x <= 1
        fine_occlusions = (coordinates_x > mean_coordinates_x) if invert else (coordinates_x < mean_coordinates_x)
        unilinear_weights = unilinear_weights * (coarse_occlusions | fine_occlusions).float()
    base_images = base_images.unsqueeze(-1)
    unilinear_weights = unilinear_weights.unsqueeze(1)
    match_images = torch.sum(base_images * unilinear_weights, dim=-2)
    match_images = match_images / torch.sum(unilinear_weights, dim=-2)
    match_images = torch.where(torch.isfinite(match_images), match_images, torch.zeros_like(match_images))
    return match_images


def warp_backward(match_images, base_disparities, invert=True):
    if base_disparities.dim() == 4:
        base_disparities = base_disparities.squeeze(1)
    coordinates_y = base_disparities.new_tensor(range(base_disparities.shape[-2]))
    coordinates_x = base_disparities.new_tensor(range(base_disparities.shape[-1]))
    coordinates_y, coordinates_x = torch.meshgrid(coordinates_y, coordinates_x)
    coordinates_y = coordinates_y.unsqueeze(0).expand_as(base_disparities)
    coordinates_x = coordinates_x.unsqueeze(0).expand_as(base_disparities)
    coordinates_x = coordinates_x + (-base_disparities if invert else base_disparities)
    coordinates_y = linear_map(coordinates_y, 0, match_images.shape[-2] - 1, -1, 1)
    coordinates_x = linear_map(coordinates_x, 0, match_images.shape[-1] - 1, -1, 1)
    coordinates = torch.stack((coordinates_x, coordinates_y), dim=-1)
    match_images = nn.functional.grid_sample(match_images, coordinates, mode="bilinear", padding_mode="border")
    # NOTE: `nn.functional.grid_sample` doesn't correspond to its second derivative
    # match_images = unilinear_sampler(match_images, coordinates_x, dim=-1)
    return match_images
