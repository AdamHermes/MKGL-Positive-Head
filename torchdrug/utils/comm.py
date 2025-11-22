"""
Minimal distributed utilities to satisfy:
    from torchdrug.utils import comm
"""

import torch

def get_rank():
    # No distributed support — always return 0
    return 0

def get_world_size():
    return 1

def is_main():
    return True

def barrier():
    # No-op
    return

def synchronize():
    # No-op
    return

class DummyMetricMeter:
    """A no-op metric gatherer to mimic TorchDrug behavior."""
    def __init__(self, *args, **kwargs):
        pass

    def add(self, *args, **kwargs):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, d):
        pass
