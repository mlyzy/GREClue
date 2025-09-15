
import numpy as np

def pairwise_distances(X: np.ndarray) -> np.ndarray:
    D2 = np.sum((X[:,None,:] - X[None,:,:])**2, axis=-1)
    return np.sqrt(D2 + 1e-12)

def pam_kmedoids(X: np.ndarray, k: int, init_medoids=None, max_iter: int = 100):
    """
    Partitioning Around Medoids.
    X: [N, D]
    k: number of clusters
    init_medoids: list of indices, optional; if None, choose k random.
    Returns: (medoids_idx, labels, total_cost)
    """
    N = X.shape[0]
    if N == 0:
        return [], np.array([], dtype=int), 0.0
    k = min(k, N)

    D = pairwise_distances(X)

    if init_medoids is None or len(init_medoids) < k:
        rng = np.random.default_rng(0)
        rest = [i for i in range(N) if i not in (init_medoids or [])]
        extra = rng.choice(rest, size=k - (len(init_medoids or [])), replace=False).tolist()
        medoids = (init_medoids or []) + extra
    else:
        medoids = list(init_medoids[:k])

    # assign
    def assign(D, medoids):
        # nearest medoid per point
        sub = D[:, medoids]  # [N, k]
        labels = np.argmin(sub, axis=1)
        cost = np.sum(np.min(sub, axis=1))
        return labels, cost

    labels, cost = assign(D, medoids)
    improved = True
    it = 0
    while improved and it < max_iter:
        improved = False
        it += 1
        for i in range(N):
            if i in medoids:
                continue
            for mi, m in enumerate(list(medoids)):
                trial = medoids.copy()
                trial[mi] = i
                trial_labels, trial_cost = assign(D, trial)
                if trial_cost + 1e-9 < cost:
                    medoids = trial
                    labels, cost = trial_labels, trial_cost
                    improved = True
        # end for i
    return medoids, labels, cost
