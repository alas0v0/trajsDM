import torch

def squash_unit_interval(image):
    return 1.0 - torch.exp(-image)


def image_encode_soft(points, valid_mask, height, width, temperature):
    batch_size = points.shape[0]
    device = points.device
    dtype = points.dtype

    x = points[..., 0] / GRID_SCALE
    y = points[..., 1] / GRID_SCALE
    center_x = torch.floor(x).to(torch.long)
    center_y = torch.floor(y).to(torch.long)
    valid_bool = valid_mask > 0
    temp = max(float(temperature), 1e-12)

    neighbor_offsets = torch.tensor(
        [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)],
        device=device,
        dtype=torch.long,
    )
    offset_x = neighbor_offsets[:, 0].view(1, 1, -1)
    offset_y = neighbor_offsets[:, 1].view(1, 1, -1)

    ngx = center_x.unsqueeze(-1) + offset_x
    ngy = center_y.unsqueeze(-1) + offset_y
    inside = ( valid_bool.unsqueeze(-1) & (ngx >= 0) & (ngx < height) & (ngy >= 0) & (ngy < width) )

    ndist2 = (x.unsqueeze(-1) - (ngx.to(dtype) + 0.5)) ** 2 + (y.unsqueeze(-1) - (ngy.to(dtype) + 0.5)) ** 2
    # logits = -ndist2 / temp
    # logits = torch.where(inside, logits, torch.full_like(logits, -1e9))
    # weights = torch.softmax(logits, dim=-1)
    weights=1/(1+ (ndist2 / temp)**2)
    weights = torch.where(inside, weights, torch.zeros_like(weights))

    linear_idx = ngx * width + ngy
    linear_idx = torch.where(inside, linear_idx, torch.zeros_like(linear_idx))

    image_flat = torch.zeros((batch_size, height * width), device=device, dtype=dtype)
    image_flat.scatter_add_(
        dim=1,
        index=linear_idx.reshape(batch_size, -1),
        src=weights.reshape(batch_size, -1),
    )

    output = 1.0 - torch.exp(-image_flat.view(batch_size, height, width))
    return output