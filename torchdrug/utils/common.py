import torch

def sparse_coo_tensor(indices, values, size, device=None, dtype=None):
    """
    Wrap torch.sparse_coo_tensor. indices should be a tensor of shape (ndim, nnz).
    """
    if device is not None:
        indices = indices.to(device)
        values = values.to(device)
    return torch.sparse_coo_tensor(indices, values, size).coalesce()

class cached_property(object):
    """
    Simple cached_property implementation compatible with TorchDrug.
    Usage:
        @cached_property
        def x(self): ...
    """
    def __init__(self, func):
        self.func = func
        self.__doc__ = getattr(func, "__doc__")

    def __get__(self, instance, owner):
        if instance is None:
            return self
        value = self.func(instance)
        setattr(instance, self.func.__name__, value)
        return value

