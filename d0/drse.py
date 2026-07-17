"""Drift-robust subspace ensemble (DRSE) — poker44-draco's v2-view member.

Bagged ExtraTrees / HistGBM, each fit on a random FEATURE subspace + bootstrap
rows. Averaging over subspaces makes the model robust to per-feature
distribution drift between the benchmark and the live feed (no single feature
can dominate). Member mix and subspace geometry are per-variant. Picklable.
"""
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier


class DRSE:
    def __init__(self, n=10, ff=0.7, seed=0, et_n=300, et_depth=14,
                 hgb_depth=4, hgb_lr=0.04, hgb_iter=350, hgb_l2=2.0, n_jobs=4):
        self.n, self.ff, self.seed = n, ff, seed
        self.et_n, self.et_depth = et_n, et_depth
        self.hgb_depth, self.hgb_lr = hgb_depth, hgb_lr
        self.hgb_iter, self.hgb_l2 = hgb_iter, hgb_l2
        self.n_jobs = n_jobs

    def fit(self, X, y):
        X = np.asarray(X, float)
        rng = np.random.RandomState(self.seed % (2 ** 31))
        nf = X.shape[1]
        k = max(1, int(self.ff * nf))
        self.mem = []
        for b in range(self.n):
            fi = np.sort(rng.choice(nf, k, replace=False))
            rows = rng.choice(len(X), len(X), replace=True)
            m = (ExtraTreesClassifier(self.et_n, max_depth=self.et_depth,
                                      min_samples_leaf=2, n_jobs=self.n_jobs,
                                      random_state=b,
                                      class_weight="balanced_subsample")
                 if b % 2 == 0 else
                 HistGradientBoostingClassifier(max_depth=self.hgb_depth,
                                                learning_rate=self.hgb_lr,
                                                max_iter=self.hgb_iter,
                                                l2_regularization=self.hgb_l2,
                                                random_state=b))
            m.fit(X[np.ix_(rows, fi)], y[rows])
            self.mem.append((fi, m))
        return self

    def predict_proba(self, X):
        X = np.asarray(X, float)
        P = np.column_stack([m.predict_proba(X[:, fi])[:, 1] for fi, m in self.mem])
        a = P.mean(1)
        return np.column_stack([1.0 - a, a])
