# mini_torchdrug.py
# Minimal replacement for the subset of torchdrug used in your code.
# Provides: core.Configurable, core._MetaContainer, utils.cached_property,
# utils.sparse_coo_tensor, and data.PackedGraph with the minimal API used.

import torch
import math
from functools import cached_property as _cached_property

# ------------------- core ---------------------------------
class _MetaContainer:
    def __init__(self, **kwargs):
        # store any meta fields users may pass
        self.meta_dict = {}
        for k, v in kwargs.items():
            setattr(self, k, v)

class Configurable:
    """
    Minimal Configurable loader:
    - load_config_dict expects {'class': 'ClassName', ...}
    - will instantiate the class from globals() by name using kwargs
    """
    @classmethod
    def load_config_dict(cls, cfg):
        if cfg is None:
            return None
        if not isinstance(cfg, dict):
            return cfg
        class_name = cfg.get("class")
        kwargs = {k: v for k, v in cfg.items() if k != "class"}
        if class_name is None:
            raise ValueError("Config dict must provide a 'class' key")
        # Look up in globals - user code should have imported classes into module globals
        if class_name in globals():
            cls_obj = globals()[class_name]
            return cls_obj(**kwargs)
        # Fallback: maybe the config provided a callable already
        raise ValueError(f"Class '{class_name}' not found in current globals for Configurable.load_config_dict")

class core:
    Configurable = Configurable
    _MetaContainer = _MetaContainer

# ------------------- utils --------------------------------
class utils:
    # cached_property implemented with functools.cached_property if available
    cached_property = _cached_property

    @staticmethod
    def sparse_coo_tensor(indices, values, size):
        """
        Lightweight wrapper around torch.sparse_coo_tensor.
        indices: 2 x N or (ndim, N)
        values: shape [N] or [N, *]
        size: tuple
        """
        if indices.ndim != 2:
            indices = indices.t()
        return torch.sparse_coo_tensor(indices, values, size)

# ------------------- data ---------------------------------
class data:
    class PackedGraph:
        """
        Minimal PackedGraph that aims to present the fields used by your
        RepeatGraph implementation.

        Expected simple input graph type: an object (like torch_geometric.data.Data)
        with at least:
          - edge_index: shape [2, E] (torch.LongTensor)
          - optionally edge_weight: shape [E] or None
          - num_nodes: scalar int (or attribute x with x.shape[0])
          - optionally num_relation (int)

        PackedGraph.pack(list_of_graphs) returns a PackedGraph that contains:
          - edge_list: (sumE x 3) tensor: [src, dst, rel] (rel filled with 0 if missing)
          - edge_weight: concatenated weights or ones
          - num_nodes (tensor per graph)
          - num_edges (tensor per graph)
          - num_cum_nodes, num_cum_edges
          - num_node (total nodes), num_edge (total edges)
          - batch_size
          - input: original single-graph object if packing one graph (keeps minimal compatibility)
        """
        def __init__(self, edge_list=None, edge_weight=None, num_nodes=None, num_edges=None,
                     num_relation=0, offsets=None, meta_dict=None, **kwargs):
            # edge_list: [E, 3] (src, dst, rel)
            self.edge_list = edge_list if edge_list is not None else torch.zeros((0, 3), dtype=torch.long)
            self.edge_weight = edge_weight if edge_weight is not None else torch.zeros(self.edge_list.shape[0], dtype=torch.float)
            # num_nodes/num_edges are tensors (per graph) or scalars if single
            self.num_nodes = torch.as_tensor(num_nodes) if num_nodes is not None else torch.tensor([0], dtype=torch.long)
            self.num_edges = torch.as_tensor(num_edges) if num_edges is not None else torch.tensor([0], dtype=torch.long)
            self.num_cum_nodes = self.num_nodes.cumsum(0) if self.num_nodes.numel() > 0 else torch.tensor([], dtype=torch.long)
            self.num_cum_edges = self.num_edges.cumsum(0) if self.num_edges.numel() > 0 else torch.tensor([], dtype=torch.long)
            self.num_node = int(self.num_nodes.sum().item()) if self.num_nodes.numel() > 0 else 0
            self.num_edge = int(self.num_edges.sum().item()) if self.num_edges.numel() > 0 else 0
            self.num_relation = int(num_relation)
            self.meta_dict = meta_dict or {}
            # offsets for edge indexing if provided
            self.offsets = offsets
            # store input for convenience
            self.input = kwargs.get("input", None)
            # placeholders used by RepeatGraph: degree_in/degree_out, node2graph, edge2graph
            # compute if possible:
            if self.edge_list is not None and self.edge_list.numel() > 0:
                src = self.edge_list[:, 0].long()
                dst = self.edge_list[:, 1].long()
                # build degree_in/out for whole packed graph
                N = max(int(self.num_node), int(self.edge_list.max().item() + 1))
                deg_in = torch.zeros(N, dtype=torch.long)
                deg_out = torch.zeros(N, dtype=torch.long)
                deg_out.scatter_add_(0, src, torch.ones_like(src))
                deg_in.scatter_add_(0, dst, torch.ones_like(dst))
                self.degree_in = deg_in
                self.degree_out = deg_out
            else:
                self.degree_in = torch.tensor([], dtype=torch.long)
                self.degree_out = torch.tensor([], dtype=torch.long)

            # The fields below – node2graph/edge2graph – will be computed on demand.
            self._node2graph = None
            self._edge2graph = None

        # -------- packing helper ----------
        @staticmethod
        def pack(graph_list):
            """
            Pack a list of simple graph-like objects into a PackedGraph.
            Each graph in graph_list is expected to have:
              - edge_index: [2, E] LongTensor
              - optional edge_weight: [E] FloatTensor
              - num_nodes: int or attribute x with x.shape[0]
              - optional num_relation (int)
            """
            if not isinstance(graph_list, (list, tuple)):
                graph_list = [graph_list]

            edge_lists = []
            edge_weights = []
            num_nodes_list = []
            num_edges_list = []
            num_relation = 0
            cum_nodes = 0
            for g in graph_list:
                # try discover num_nodes
                if hasattr(g, "num_nodes"):
                    n = int(g.num_nodes)
                elif hasattr(g, "x"):
                    n = int(g.x.shape[0])
                else:
                    # fallback: infer from edge_index
                    if hasattr(g, "edge_index") and g.edge_index.numel() > 0:
                        n = int(g.edge_index.max().item()) + 1
                    else:
                        n = 0
                num_nodes_list.append(n)
                if hasattr(g, "edge_index"):
                    ei = g.edge_index
                    if ei.ndim == 2 and ei.size(0) == 2:
                        src = ei[0].long()
                        dst = ei[1].long()
                        # try relation if provided as 1D attr
                        if hasattr(g, "edge_rel") and g.edge_rel is not None:
                            rel = g.edge_rel.long()
                        else:
                            # default 0 relation
                            rel = torch.zeros(src.numel(), dtype=torch.long, device=src.device)
                        edge_list = torch.stack([src + cum_nodes, dst + cum_nodes, rel], dim=-1)
                        edge_lists.append(edge_list)
                        num_edges_list.append(edge_list.shape[0])
                        if hasattr(g, "edge_weight") and g.edge_weight is not None:
                            edge_weights.append(g.edge_weight.float())
                        else:
                            edge_weights.append(torch.ones(edge_list.shape[0], dtype=torch.float, device=edge_list.device))
                    else:
                        # no edges
                        num_edges_list.append(0)
                else:
                    num_edges_list.append(0)
                cum_nodes += n
                if hasattr(g, "num_relation"):
                    num_relation = max(num_relation, int(g.num_relation))

            if len(edge_lists) > 0:
                edge_list = torch.cat(edge_lists, dim=0)
                edge_weight = torch.cat(edge_weights, dim=0)
            else:
                edge_list = torch.zeros((0, 3), dtype=torch.long)
                edge_weight = torch.zeros((0,), dtype=torch.float)

            pg = data.PackedGraph(edge_list=edge_list,
                                  edge_weight=edge_weight,
                                  num_nodes=torch.tensor(num_nodes_list, dtype=torch.long),
                                  num_edges=torch.tensor(num_edges_list, dtype=torch.long),
                                  num_relation=num_relation)
            # also keep the original single-graph input if length==1
            if len(graph_list) == 1:
                pg.input = graph_list[0]
            return pg

        # helpers that RepeatGraph expects:
        def _standarize_index(self, index, num_total):
            # accept boolean mask, int list, or tensor of indices
            if isinstance(index, torch.BoolTensor):
                return index.nonzero(as_tuple=False).view(-1)
            if isinstance(index, (list, tuple)):
                return torch.as_tensor(index, dtype=torch.long)
            if isinstance(index, torch.Tensor):
                return index
            raise ValueError("Unsupported index type for _standarize_index")

        def data_mask(self, node_index=None, edge_index=None):
            """
            Minimal data_mask that returns (data_dict, meta_dict). We cannot
            reconstruct arbitrary per-edge/per-node data, so we return empty dicts.
            Keep signatures compatible.
            """
            # For compact mode the caller expects to be able to construct
            # a new graph type(self.input)(edge_list=..., edge_weight=..., num_nodes=..., num_edges=..., num_relation=..., offsets=..., meta_dict=meta_dict, **data_dict)
            # So we return empty data_dict (user code must not rely on these extras)
            return {}, {}

        def data_by_meta(self, key):
            # placeholder: returns empty dicts consistent with usage in code
            return {}, {}

        # neighbors and num_neighbors expected by RepeatGraph
        def neighbor_inverted_index(self):
            # build order and ranges similar to original code
            # node_in = self.input.edge_list[:, 0]
            if self.edge_list.numel() == 0:
                return torch.tensor([], dtype=torch.long), torch.zeros((0, 2), dtype=torch.long)
            node_in = self.edge_list[:, 0].long()
            node_in_sorted, order = node_in.sort()
            degree_in = bincount(node_in_sorted, minlength=self.num_node)
            ends = degree_in.cumsum(0)
            starts = ends - degree_in
            ranges = torch.stack([starts, ends], dim=-1)
            # note: order currently indexes into this packed edge list
            return order, ranges

        def neighbors(self, index):
            order, ranges = self.neighbor_inverted_index()
            if order.numel() == 0:
                return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)
            starts, ends = ranges[index].t()
            num_neighbors = ends - starts
            offsets = num_neighbors.cumsum(0) - num_neighbors
            ranges_idx = torch.arange(num_neighbors.sum(), device=self.edge_list.device)
            ranges_idx = ranges_idx + (starts - offsets).repeat_interleave(num_neighbors)
            edge_index = order[ranges_idx]
            node_out = self.edge_list[edge_index, 1]
            return edge_index, node_out

        def num_neighbors(self, index):
            order, ranges = self.neighbor_inverted_index()
            if ranges.numel() == 0:
                return torch.zeros_like(index)
            starts, ends = ranges[index].t()
            return ends - starts

        # lazy cached properties used by RepeatGraph
        @property
        def node2graph(self):
            if self._node2graph is None:
                # make per-node graph id mapping
                if self.num_nodes.numel() == 0:
                    self._node2graph = torch.zeros((self.num_node,), dtype=torch.long)
                else:
                    # repeat graph ids for each node
                    ids = []
                    for i, n in enumerate(self.num_nodes.tolist()):
                        if n > 0:
                            ids.append(torch.full((n,), i, dtype=torch.long))
                    self._node2graph = torch.cat(ids, dim=0) if len(ids) > 0 else torch.zeros((0,), dtype=torch.long)
            return self._node2graph

        @property
        def edge2graph(self):
            if self._edge2graph is None:
                if self.num_edges.numel() == 0:
                    self._edge2graph = torch.zeros((self.num_edge,), dtype=torch.long)
                else:
                    ids = []
                    for i, e in enumerate(self.num_edges.tolist()):
                        if e > 0:
                            ids.append(torch.full((e,), i, dtype=torch.long))
                    self._edge2graph = torch.cat(ids, dim=0) if len(ids) > 0 else torch.zeros((0,), dtype=torch.long)
            return self._edge2graph

        # convenience: allow attribute access for missing attributes on input
        def __getattr__(self, name):
            if "input" in self.__dict__ and self.input is not None:
                attr = getattr(self.__dict__["input"], name, None)
                if isinstance(attr, torch.Tensor):
                    return attr
                return attr
            raise AttributeError(f"'PackedGraph' object has no attribute '{name}'")

# ------------------- helper functions used by data.PackedGraph -----------------
def bincount(input, minlength=0):
    # reuse the user's bincount implementation semantics
    if isinstance(input, torch.Tensor) and input.numel() == 0:
        return torch.zeros(minlength, dtype=torch.long, device=input.device)
    # assume sorted no-check for speed
    try:
        return input.bincount(minlength=minlength)
    except Exception:
        # fallback
        mx = int(input.max().item()) if input.numel() else 0
        m = max(minlength, mx + 1)
        out = torch.zeros(m, dtype=torch.long, device=input.device)
        for v in input.tolist():
            out[v] += 1
        return out
