import torch

def sparse_coo_tensor(indices, values, size, device=None, dtype=None):
    """
    Wrap torch.sparse_coo_tensor. indices should be a tensor of shape (ndim, nnz).
    """
    if device is not None:
        indices = indices.to(device)
        values = values.to(device)
    return torch.sparse_coo_tensor(indices, values, size).coalesce()
