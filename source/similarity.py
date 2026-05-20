import numpy as np
from scipy.spatial.distance import minkowski, jensenshannon, wasserstein_distance
from scipy.stats import pearsonr, spearmanr
from scipy.special import rel_entr


class EmbeddingSimilarityVector:
    """
    Given two embeddings, compute a vector of different similarity/distance measures.

    Metrics (in order):
        0: cosine
        1: dot
        2: pearson
        3: spearman
        4: euclidean
        5: manhattan
        6: minkowski_p3
        7: angular
        8: kl_divergence
        9: js_divergence
       10: wasserstein
       11: jaccard
    """

    def __init__(self, v1, v2):
        self.v1 = self._to_numpy(v1)
        self.v2 = self._to_numpy(v2)

        if self.v1.shape != self.v2.shape:
            raise ValueError(f"Embeddings must have same shape, got {self.v1.shape} and {self.v2.shape}")

        self._names = [
            "cosine",
            "dot",
            "pearson",
            "spearman",
            "euclidean",
            "manhattan",
            "minkowski_p3",
            "angular",
            "kl",
            "js",
            "wasserstein",
            "jaccard",
        ]

    # ---------- helpers ----------
    def _to_numpy(self, v):
        if isinstance(v, np.ndarray):
            return v.astype(float)
        try:
            import torch
            if isinstance(v, torch.Tensor):
                return v.detach().cpu().numpy().astype(float)
        except ImportError:
            pass
        return np.array(v, dtype=float)

    def _safe_prob(self, x, eps=1e-12):
        x = np.asarray(x, dtype=float)
        x = np.clip(x, eps, None)
        s = np.sum(x)
        if s == 0:
            # uniform if completely zero
            x = np.ones_like(x) / len(x)
        else:
            x = x / s
        return x

    # ---------- core metrics ----------
    def cosine(self):
        num = np.dot(self.v1, self.v2)
        denom = (np.linalg.norm(self.v1) * np.linalg.norm(self.v2) + 1e-12)
        return num / denom

    def dot(self):
        return float(np.dot(self.v1, self.v2))

    def pearson(self):
        # if constant vectors, pearsonr returns nan; handle that
        if np.allclose(self.v1, self.v1[0]) or np.allclose(self.v2, self.v2[0]):
            return 0.0
        return float(pearsonr(self.v1, self.v2)[0])

    def spearman(self):
        # similar issue as pearson
        if np.allclose(self.v1, self.v1[0]) or np.allclose(self.v2, self.v2[0]):
            return 0.0
        return float(spearmanr(self.v1, self.v2)[0])

    def euclidean(self):
        return float(np.linalg.norm(self.v1 - self.v2))

    def manhattan(self):
        return float(np.sum(np.abs(self.v1 - self.v2)))

    def minkowski_p3(self):
        return float(minkowski(self.v1, self.v2, p=3))

    def angular(self):
        cos = np.clip(self.cosine(), -1.0, 1.0)
        return float(np.arccos(cos))

    def kl(self):
        """KL(v1 || v2) treating v1, v2 as probability vectors."""
        p = self._safe_prob(self.v1)
        q = self._safe_prob(self.v2)
        return float(np.sum(rel_entr(p, q)))

    def js(self):
        """Jensen–Shannon divergence."""
        p = self._safe_prob(self.v1)
        q = self._safe_prob(self.v2)
        return float(jensenshannon(p, q))

    def wasserstein(self):
        return float(wasserstein_distance(self.v1, self.v2))

    def jaccard(self, threshold=0.0):
        """
        Jaccard similarity of support sets (indices where value > threshold).
        Works for sparse-ish embeddings or binary/bag-of-words.
        """
        a = set(np.where(self.v1 > threshold)[0])
        b = set(np.where(self.v2 > threshold)[0])
        if not a and not b:
            return 1.0  # both empty → identical support
        if not (a or b):
            return 0.0
        return float(len(a & b) / len(a | b))

    # ---------- public API ----------
    def vector(self) -> np.ndarray:
        """Return a 1D numpy array of all metrics in fixed order."""
        vals = [
            self.cosine(),
            self.dot(),
            self.pearson(),
            self.spearman(),
            self.euclidean(),
            self.manhattan(),
            self.minkowski_p3(),
            self.angular(),
            self.kl(),
            self.js(),
            self.wasserstein(),
            self.jaccard(),
        ]
        return np.array(vals, dtype=float)

    def names(self):
        """Return list of metric names in the same order as vector()."""
        return list(self._names)

    def as_dict(self):
        """Optional: get a dict {name: value}."""
        v = self.vector()
        return {name: val for name, val in zip(self._names, v)}
