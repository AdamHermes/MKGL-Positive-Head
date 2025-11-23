import torch
from .scatter import scatter_add, scatter_max

def scatter_softmax(src, index, dim_size):
    max_val = scatter_max(src, index, dim_size)
    exp = torch.exp(src - max_val[index])
    denom = scatter_add(exp, index, dim_size)
    return exp / denom[index]

def scatter_log_softmax(src, index, dim_size):
    max_val = scatter_max(src, index, dim_size)
    exp = torch.exp(src - max_val[index])
    denom = scatter_add(exp, index, dim_size)
    return (src - max_val[index]) - torch.log(denom[index])
