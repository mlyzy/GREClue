
from typing import Dict, Any, List, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_OK = True
except Exception:
    _TORCH_OK = False

from utils.hashing import hash_text

NODE_KIND_VOCAB = {
    "method": 0,
    "codeline": 1,
    "var": 2,
    "test": 3,
    "other": 4,
}

def _build_adjacency(n: int, edges: List[Tuple[int,int]]) -> np.ndarray:
    A = np.zeros((n, n), dtype=np.float32)
    for a,b in edges:
        if 1 <= a <= n and 1 <= b <= n:
            A[a-1, b-1] = 1.0
            A[b-1, a-1] = 1.0  # treat as undirected for encoding
    # add self loops
    for i in range(n):
        A[i,i] = 1.0
    # row-normalize
    Dinv = 1.0 / (A.sum(axis=1, keepdims=True) + 1e-8)
    return A * Dinv

def _node_feature(node: Dict[str, Any], hash_dim: int = 128) -> np.ndarray:
    kind = node.get("kind", "other").lower()
    kind_id = NODE_KIND_VOCAB.get(kind, NODE_KIND_VOCAB["other"])
    kind_onehot = np.zeros(len(NODE_KIND_VOCAB), dtype=np.float32)
    kind_onehot[kind_id] = 1.0

    txt = str(node.get("content", ""))
    h = hash_text(txt, dim=hash_dim)  # 128 dims

    sus = np.array([float(node.get("sus", 0.0))], dtype=np.float32)

    return np.concatenate([kind_onehot, h, sus], axis=0)  # dim = 5 + 128 + 1 = 134

class _SageLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin_neigh = torch.nn.Linear(in_dim, out_dim, bias=False)
        self.lin_self = torch.nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, X, Ahat):
        neigh = torch.matmul(Ahat, X)  # aggregate neighbors (mean due to row-normalized A)
        h = self.lin_neigh(neigh) + self.lin_self(X)
        return torch.relu(h)

class GraphEncoder(torch.nn.Module):
    def __init__(self, in_dim=134, hidden=128, out_dim=256):
        super().__init__()
        self.g1 = _SageLayer(in_dim, hidden)
        self.g2 = _SageLayer(hidden, out_dim)

    def forward(self, X, Ahat):
        h1 = self.g1(X, Ahat)
        h2 = self.g2(h1, Ahat)
        # graph-level readout: mean
        g = h2.mean(dim=0)  # [out_dim]
        return F.normalize(g, p=2, dim=0)

class GNNEmbedder:
    """
    Simple GraphSAGE-like encoder to produce a graph embedding without training.
    """
    def __init__(self, in_dim=134, hidden=128, out_dim=256, device: str = "cpu"):
        if not _TORCH_OK:
            raise ImportError("PyTorch is required for GNNEmbedder")
        self.model = GraphEncoder(in_dim, hidden, out_dim).to(device)
        self.device = device

    def embed_graph(self, graph: Dict[str, Any]) -> np.ndarray:
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        n = len(nodes)
        if n == 0:
            return np.zeros((256,), dtype=np.float32)

        X = np.stack([_node_feature(nd) for nd in nodes], axis=0)  # [N, F]
        Ahat = _build_adjacency(n, edges)                          # [N, N]

        import torch
        X_t = torch.from_numpy(X).to(self.device)
        A_t = torch.from_numpy(Ahat).to(self.device)
        with torch.no_grad():
            g = self.model(X_t, A_t)  # [D]
        return g.cpu().numpy().astype("float32")
