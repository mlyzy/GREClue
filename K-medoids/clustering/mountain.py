
import numpy as np

def mountain_densities(X: np.ndarray, ra: float = 1.0) -> np.ndarray:
    """
    Compute mountain function M_i = sum_j exp(-||x_i - x_j||^2 / (2 * ra^2))
    """
    N = X.shape[0]
    D2 = np.sum((X[:,None,:] - X[None,:,:])**2, axis=-1)  # [N, N]
    M = np.exp(-D2 / (2.0 * (ra**2)))
    return M.sum(axis=1)  # [N]

def select_peaks(X: np.ndarray, k: int, ra: float = 1.0, rb: float = 1.5):
    """
    Iteratively select k peaks by mountain reduction:
    After choosing peak p, reduce M by:
      M_j <- M_j - M_p * exp(-||x_j - x_p||^2 / (2 * rb^2))
    Returns indices of selected peaks.
    """
    N = X.shape[0]
    if N == 0:
        return []
    k = min(k, N)
    D2 = np.sum((X[:,None,:] - X[None,:,:])**2, axis=-1)
    M = np.exp(-D2 / (2.0 * (ra**2))).sum(axis=1)
    peaks = []
    for _ in range(k):
        p = int(np.argmax(M))
        peaks.append(p)
        # reduce
        red = M[p] * np.exp(-D2[:, p] / (2.0 * (rb**2)))
        M = M - red
    return peaks
