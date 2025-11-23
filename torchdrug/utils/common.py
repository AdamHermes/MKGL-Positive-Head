import torch
import os
import urllib.request

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

def get_line_count(path):
    """
    Return number of lines in a file (used for tqdm total).
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            # use a fast method
            return sum(1 for _ in f)
    except Exception:
        return 0

def download(url, folder, save_file=None, timeout=30):
    """
    Minimal download utility used by dataset loaders.
    Downloads `url` into `folder/save_file` and returns local path.
    """
    if save_file is None:
        save_file = os.path.basename(url)
    os.makedirs(folder, exist_ok=True)
    dst = os.path.join(folder, save_file)
    if os.path.exists(dst):
        return dst
    # attempt to download
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response, open(dst, "wb") as out_file:
            out_file.write(response.read())
        return dst
    except Exception:
        # fallback: try urllib.request.urlretrieve (older)
        try:
            urllib.request.urlretrieve(url, dst)
            return dst
        except Exception as e:
            raise RuntimeError(f"Failed to download {url}: {e}")
