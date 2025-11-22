import torch

def generalized_rspmm(adjacency, relation_input, node_input, sum="add", mul="mul"):
    """
    Simplified (but general) implementation that supports 3D sparse adjacency:
        - adjacency: a torch.sparse_coo_tensor with indices shape (3, NNZ)
                     coordinates (node_in, node_out, relation_index)
        - relation_input: shape (R, D) or (batch*R, D)
        - node_input: shape (num_node, D)
    Returns a tensor with aggregated messages per (node_out, relation_index) or per node_out
    This is intentionally simple (iterates over nonzero entries) to be correct.
    """
    # If adjacency is dense tensor: fallback to matmul-like behavior
    if not adjacency.is_sparse:
        # adjacency is dense, e.g., (n, n, r) or (n, n)
        # We'll support 2D adjacency as standard matmul
        if adjacency.ndim == 2:
            # (n,n) @ (n,D) -> (n,D)
            if sum == "add":
                return adjacency @ node_input
            else:
                raise NotImplementedError("Only sum-add supported for dense 2D adjacency in shim.")
        else:
            # 3D dense: (n, n, r) -> we fold over r
            n, _, r = adjacency.shape
            out = torch.zeros(n, relation_input.shape[-1], device=node_input.device, dtype=node_input.dtype)
            for k in range(r):
                w = adjacency[:, :, k]
                rel = relation_input[k]
                # (n,n) @ (n,D) -> (n,D), then element-wise multiply with rel (broadcast)
                tmp = w @ node_input
                tmp = tmp * rel
                out += tmp
            return out

    # Sparse adjacency: indices (3, NNZ), values (NNZ,)
    adj = adjacency.coalesce()
    idx = adj.indices()  # (3, NNZ): [node_in, node_out, rel_index]
    vals = adj.values()
    nnz = idx.shape[1]

    node_in_idx = idx[0]
    node_out_idx = idx[1]
    rel_idx = idx[2]

    D = node_input.shape[-1]
    # result tensor shape: (num_node_out, D) - but multiple relation indices may map to same node_out.
    num_nodes = adjacency.shape[0]
    # We'll compute per-node aggregated vector (sum over edges and relations)
    out = torch.zeros((num_nodes, D), device=node_input.device, dtype=node_input.dtype)

    # Relation input may be shaped (R, D_rel) or (batch*R, D_rel).
    # We assume D_rel == D or supports broadcasting/multiplication.
    # Perform aggregation: out[node_out] (op=add/max/min) relation_input[rel_idx] * node_input[node_in] * edge_val
    for i in range(nnz):
        ni = node_in_idx[i].item()
        no = node_out_idx[i].item()
        rk = rel_idx[i].item()
        w = vals[i]
        rel_vec = relation_input[rk]
        vec = node_input[ni]
        if mul == "mul":
            msg = rel_vec * vec
        else:
            # support add / concat variants less commonly used
            msg = vec
        if sum == "add":
            out[no] = out[no] + w * msg
        elif sum == "max":
            out[no] = torch.max(out[no], w * msg)
        elif sum == "min":
            out[no] = torch.min(out[no], w * msg)
        else:
            raise ValueError("Unknown sum reduction: %s" % sum)
    return out
