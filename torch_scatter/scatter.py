import torch

def scatter_add(src, index, dim_size):
    out = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
    out.index_add_(0, index, src)
    return out


def scatter_min(src, index, dim_size):
    # initialize with +inf
    out = torch.full((dim_size,), float('inf'), dtype=src.dtype, device=src.device)
    out.index_reduce_(0, index, src, reduce="amin")
    # replace inf (bins with no entries)
    out = torch.where(out == float('inf'), torch.zeros_like(out), out)
    return out


def scatter_max(src, index, dim_size):
    # initialize with -inf
    out = torch.full((dim_size,), float('-inf'), dtype=src.dtype, device=src.device)
    out.index_reduce_(0, index, src, reduce="amax")
    # replace -inf (bins with no entries)
    out = torch.where(out == float('-inf'), torch.zeros_like(out), out)
    return out


def scatter_mean(src, index, dim_size):
    sum_val = scatter_add(src, index, dim_size)
    count = torch.bincount(index, minlength=dim_size).to(src.device)
    return sum_val / torch.clamp(count, min=1)
