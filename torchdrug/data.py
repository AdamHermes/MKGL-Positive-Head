import torch
from contextlib import contextmanager
from .utils import sparse_coo_tensor

# --- Existing PackedGraph (keeps previous functionality) ---
class PackedGraph:
    """
    A lightweight graph object sufficient for the code in layer.py and model.py.

    NOTE: This is a simplified representation:
      - nodes are 0..num_node-1
      - edges provided by edge_list (tensor [3, E] or separate arrays)
      - adjacency is obtainable via sparse_coo_tensor
    """
    def __init__(self, node_num, edge_index=None, edge_weight=None, num_relation=1, edge2graph=None, node2graph=None):
        self.num_node = node_num
        self.num_nodes = node_num  # alias used in places
        self.edge_list = edge_index if edge_index is not None else torch.zeros((3,0), dtype=torch.long)
        self.edge_weight = edge_weight if edge_weight is not None else torch.zeros((self.edge_list.shape[1],), dtype=torch.float)
        self.num_relation = num_relation
        self.edge2graph = edge2graph if edge2graph is not None else torch.zeros((self.edge_list.shape[1],), dtype=torch.long)
        self.node2graph = node2graph if node2graph is not None else torch.zeros((self.num_node,), dtype=torch.long)
        # placeholders for dynamic properties set by user code
        self.query = None
        self.boundary = None
        self.adjacency = None
        self.degree_out = None
        self.pna_degree_out = None
        self.num_edge = self.edge_list.shape[1]
        self.num_cum_nodes = self.num_node  # for RepeatGraph offsets in model.py

    def to(self, device):
        # move tensors to device
        self.edge_list = self.edge_list.to(device)
        self.edge_weight = self.edge_weight.to(device)
        self.edge2graph = self.edge2graph.to(device)
        self.node2graph = self.node2graph.to(device)
        if self.boundary is not None:
            self.boundary = self.boundary.to(device)
        if self.query is not None:
            self.query = self.query.to(device)
        return self

    def undirected(self, add_inverse=True):
        # create and return a new PackedGraph with inverse edges if requested
        if not add_inverse:
            return self
        if self.edge_list.numel() == 0:
            return self
        # edge_list shape: (3, E) => [src, dst, rel]
        src = self.edge_list[0]
        dst = self.edge_list[1]
        rel = self.edge_list[2]
        inv_src = dst.clone()
        inv_dst = src.clone()
        inv_rel = rel + self.num_relation  # second half relations are inverse
        edge_list = torch.cat([self.edge_list, torch.stack([inv_src, inv_dst, inv_rel], dim=0)], dim=1)
        edge_weight = torch.cat([self.edge_weight, self.edge_weight], dim=0)
        # simple new graph; shallow copy of other attributes
        new = PackedGraph(self.num_node, edge_index=edge_list, edge_weight=edge_weight,
                          num_relation=self.num_relation * 2)
        # keep dynamic attributes if present
        for k in ("boundary","query","adjacency","degree_out","pna_degree_out","node2graph","edge2graph","num_cum_nodes"):
            if hasattr(self, k):
                setattr(new, k, getattr(self, k))
        return new

    # context managers used in model.py (safe no-op implementations)
    @contextmanager
    def graph(self):
        yield

    @contextmanager
    def node(self):
        yield

    @contextmanager
    def edge(self):
        yield

    def adjacency_from_edges(self):
        # Build a 3D sparse adjacency: shape (num_node, num_node, num_relation_total)
        if self.edge_list.numel() == 0:
            # empty adjacency
            shape = (self.num_node, self.num_node, max(1, getattr(self, "num_relation", 1)))
            self.adjacency = sparse_coo_tensor(torch.zeros((3,0), dtype=torch.long), torch.zeros((0,)), shape)
            return self.adjacency
        idx = self.edge_list  # (3, E)
        values = self.edge_weight
        shape = (self.num_node, self.num_node, max(1, getattr(self, "num_relation", 1)))
        # indices shape expected (3, nnz)
        self.adjacency = sparse_coo_tensor(idx, values, shape)
        return self.adjacency

    # helpers used by model.py
    def num_neighbors(self, node_idx):
        # return counts per node in node_idx (1D tensor of node ids)
        src = self.edge_list[0]
        # neighbors are out edges from node_idx
        # produce list corresponding to node_idx as flattened vector
        mask = (src.unsqueeze(1) == node_idx.unsqueeze(0)).any(dim=0) if False else None
        # simpler: compute counts globally
        binc = torch.bincount(src, minlength=self.num_node)
        return binc[node_idx]

    def edge_mask(self, mask):
        """
        Return a subgraph keeping only edges where mask==True (mask len == num_edge).
        If compact True requested by caller, they expect nodes reindexed; here we return a new PackedGraph
        with the filtered edges but keep original node ids (i.e., not compacted) because full compacting is complex.
        """
        if mask.numel() != self.edge_list.shape[1]:
            raise ValueError("mask length mismatch")
        keep_idx = mask.nonzero(as_tuple=False).squeeze(-1)
        if keep_idx.numel() == 0:
            return PackedGraph(self.num_node)
        new_edge_list = self.edge_list[:, keep_idx]
        new_edge_weight = self.edge_weight[keep_idx]
        # keep relation count the same
        g = PackedGraph(self.num_node, edge_index=new_edge_list, edge_weight=new_edge_weight, num_relation=self.num_relation)
        # propagate some meta
        g.node2graph = getattr(self, "node2graph", torch.zeros((self.num_node,), dtype=torch.long))
        g.num_cum_nodes = self.num_cum_nodes
        return g

    def neighbors(self, node_idx):
        """
        Return (edge_index_tensor, node_out_indices) for edges originating from node_idx.
        edge_index tensor has shape (E_selected, 3) referencing original edge_list columns.
        node_out is a 1D tensor of node indices corresponding to each element.
        """
        src = self.edge_list[0]
        mask = (src.unsqueeze(-1) == node_idx.unsqueeze(0)).any(dim=0) if False else None
        # easier: compute boolean per edge whether its source is in node_idx
        src_expand = src.unsqueeze(1).expand(-1, node_idx.shape[0])
        matched = (src_expand == node_idx.unsqueeze(0))
        matched_any = matched.any(dim=1)
        selected_cols = matched_any.nonzero(as_tuple=False).squeeze(-1)
        if selected_cols.numel() == 0:
            return torch.zeros((0,3), dtype=torch.long), torch.zeros((0,), dtype=torch.long)
        edge_index = self.edge_list[:, selected_cols].t().contiguous()  # shape (Esel, 3)
        node_out = self.edge_list[1, selected_cols]
        return edge_index, node_out

    def match(self, pattern):
        """
        Pattern is expected shaped (N,3) where rows are (h,t,r) triples to match.
        Returns indices of matching edges. Simplified: we compare against edge_list.
        """
        pat = pattern.view(-1, 3)
        E = self.edge_list.shape[1]
        edges = self.edge_list.t()  # (E,3)
        # do naive matching
        matches = []
        for p in pat:
            # p can have -1 sentinel meaning "any"
            cond = torch.ones((E,), dtype=torch.bool)
            for i in range(3):
                if p[i].item() != -1:
                    cond = cond & (edges[:, i] == p[i].item())
            matches.append(cond.nonzero(as_tuple=False).squeeze(-1))
        if matches:
            idx = torch.cat(matches)
        else:
            idx = torch.tensor([], dtype=torch.long)
        return idx, None

    def data_by_meta(self, key):
        # placeholder used in model.py; return empty dicts
        return {}, {}

# --- New minimal Graph & KnowledgeGraphDataset classes for datasets.py usage ---
class Graph:
    """
    Minimal Graph container used by dataset loaders.
    Stores triplets as (h, t, r) list/iterable and basic metadata.
    """
    def __init__(self, triplets, num_node=None, num_relation=None):
        # accept list of tuples, or tensor of shape (N,3)
        if isinstance(triplets, torch.Tensor):
            self.edge_list = triplets.t().contiguous() if triplets.numel() else torch.zeros((3,0), dtype=torch.long)
        else:
            if len(triplets) == 0:
                self.edge_list = torch.zeros((3,0), dtype=torch.long)
            else:
                arr = torch.tensor(triplets, dtype=torch.long)
                self.edge_list = arr.t().contiguous()
        # expose similar attributes used elsewhere
        self.num_node = int(num_node) if num_node is not None else int(self.edge_list.max().item() + 1) if self.edge_list.numel() else 0
        self.num_relation = int(num_relation) if num_relation is not None else int(self.edge_list[2].max().item() + 1) if self.edge_list.numel() else 0
        # For compatibility with PackedGraph, provide edge_list, edge_weight, etc.
        self.edge_weight = torch.ones((self.edge_list.shape[1],), dtype=torch.float)
        self.adjacency = None
        self.boundary = None
        self.query = None
        self.edge2graph = torch.zeros((self.edge_list.shape[1],), dtype=torch.long) if self.edge_list.numel() else torch.zeros((0,), dtype=torch.long)
        self.node2graph = torch.zeros((self.num_node,), dtype=torch.long)

    def to(self, device):
        self.edge_list = self.edge_list.to(device)
        self.edge_weight = self.edge_weight.to(device)
        self.edge2graph = self.edge2graph.to(device)
        self.node2graph = self.node2graph.to(device)
        return self

    def match(self, pattern):
        # reuse simple matching like PackedGraph.match
        pat = pattern.view(-1, 3)
        E = self.edge_list.shape[1]
        edges = self.edge_list.t()  # (E,3)
        matches = []
        for p in pat:
            cond = torch.ones((E,), dtype=torch.bool)
            for i in range(3):
                if p[i].item() != -1:
                    cond = cond & (edges[:, i] == p[i].item())
            matches.append(cond.nonzero(as_tuple=False).squeeze(-1))
        if matches:
            idx = torch.cat(matches)
        else:
            idx = torch.tensor([], dtype=torch.long)
        return idx, None

    def edge_mask(self, mask):
        # similar to PackedGraph.edge_mask
        if mask.numel() != self.edge_list.shape[1]:
            raise ValueError("mask length mismatch")
        keep_idx = mask.nonzero(as_tuple=False).squeeze(-1)
        if keep_idx.numel() == 0:
            return Graph([], num_node=self.num_node, num_relation=self.num_relation)
        new_edge_list = self.edge_list[:, keep_idx]
        new_edge_weight = self.edge_weight[keep_idx]
        g = Graph([], num_node=self.num_node, num_relation=self.num_relation)
        g.edge_list = new_edge_list
        g.edge_weight = new_edge_weight
        g.edge2graph = getattr(self, "edge2graph", torch.zeros((new_edge_list.shape[1],), dtype=torch.long))
        g.node2graph = getattr(self, "node2graph", torch.zeros((self.num_node,), dtype=torch.long))
        return g

# Base dataset used in your datasets.py
class KnowledgeGraphDataset:
    """
    Minimal base to support InductiveKnowledgeGraphDataset in datasets.py.
    Provides a small helper to standardize vocabs.
    """
    def __init__(self):
        # placeholders often used by code
        self.graph = None
        self.fact_graph = None
        self.inductive_graph = None
        self.inductive_fact_graph = None
        self.kgdata = None

    def _standarize_vocab(self, vocab, inv_vocab):
        """
        Convert inv_vocab (token -> id) to vocab (id -> token).
        If vocab provided, return as-is.
        """
        if inv_vocab is None:
            inv_vocab = {}
        if vocab is None:
            # build id->token list aligned by indices
            vocab = [None] * len(inv_vocab)
            for token, idx in inv_vocab.items():
                vocab[idx] = token
        return vocab, inv_vocab
