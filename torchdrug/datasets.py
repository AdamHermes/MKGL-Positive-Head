# torchdrug/datasets.py
"""
Minimal stub for `torchdrug.datasets` so that pickle loading works
and dataset code depending on Dataset / KnowledgeGraphDataset works.
"""

from .data import KnowledgeGraphDataset

class Dataset:
    """Base stub dataset class."""
    pass


class KnowledgeGraphDatasetStub(KnowledgeGraphDataset):
    """
    TorchDrug has datasets inheriting from KnowledgeGraphDataset.
    Your custom datasets (FB15k237Inductive, WN18RRInductive)
    also inherit from it, so we re-export it here.
    """
    pass


__all__ = [
    "Dataset",
    "KnowledgeGraphDatasetStub",
]
