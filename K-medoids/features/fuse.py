
import numpy as np

def fuse_features(text_vec: np.ndarray, graph_vec: np.ndarray, mode: str = "concat", alpha: float = 0.5):
    """
    mode:
      - 'concat'   -> [text_vec, graph_vec]
      - 'weighted' -> alpha * text_vec + (1-alpha) * graph_vec (requires same dims)
    """
    if mode == "concat":
        return np.concatenate([text_vec, graph_vec], axis=-1)
    elif mode == "weighted":
        d1 = text_vec.shape[-1]
        d2 = graph_vec.shape[-1]
        if d1 != d2:
            # align by trunc/pad to min dim
            d = min(d1, d2)
            tv = text_vec[:d]
            gv = graph_vec[:d]
        else:
            tv, gv = text_vec, graph_vec
        return alpha * tv + (1.0 - alpha) * gv
    else:
        raise ValueError(f"Unknown fusion mode: {mode}")
