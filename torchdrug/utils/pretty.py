"""
Minimal pretty-printing utilities.
TorchDrug uses this for formatting model configs / training logs.
"""

import pprint

def format_dict(d):
    """Pretty-print a dict."""
    return pprint.pformat(d, indent=2, width=80)

def format(model):
    """Format a model/config with pretty output."""
    if hasattr(model, "config_dict"):
        return format_dict(model.config_dict())
    if isinstance(model, dict):
        return format_dict(model)
    return str(model)

def print_dict(d):
    print(format_dict(d))
