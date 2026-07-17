"""D0Draco (ported verbatim from UID172 poker44-draco) — D0Draco: weighted 4-member rank-blend on three feature views.

  * stack — benchmark-supervised StackingClassifier on the phasberg view.
  * mono  — monotone-constrained XGBoost committee on the phasberg view
    (sign-constrained, drift-stable).
  * mlp   — neural PCA->MLP committee on the v2+phasberg UNION (a different
    model family — decorrelated from the trees).
  * drse  — drift-robust subspace ensemble on the v2 view (a feature space the
    others don't use alone; guards against per-feature live drift).

Rank-averaging is calibration-agnostic and overfit-resistant: only each
member's chunk ORDERING matters, which is what the AP-dominated reward scores.

Weighted 4-member rank-blend at 0.28/0.24/0.28/0.2: a 56-leaf benchmark-supervised stack (cv4) and a 3-seed depth-5 monotone XGBoost on the behavioral view, a 4-seed PCA-44 MLP (64, 32) committee on the union view, and a drift-robust subspace ensemble (n=8, feature-fraction 0.75) on the v2 view.
"""
import numpy as np

# per-variant blend weights chosen at spec time (walk-forward verified)
W = {"stack": 0.28, "mono": 0.24, "mlp": 0.28, "drse": 0.2}


class D0Draco:
    def __init__(self, stack, mono, mlp, drse, cols_ph, cols_v2, weights=None):
        self.stack = stack
        self.mono = mono
        self.mlp = mlp
        self.drse = drse
        self.cols_ph = cols_ph
        self.cols_v2 = cols_v2
        self.weights = dict(weights) if weights else dict(W)

    @staticmethod
    def _rank(s):
        s = np.asarray(s, dtype=float)
        n = s.size
        if n <= 1:
            return s
        return np.argsort(np.argsort(s, kind="stable"), kind="stable").astype(float) / (n - 1)

    def score(self, Xph, Xv2):
        """Xph: phasberg matrix (cols_ph order); Xv2: v2 matrix (cols_v2 order)."""
        Xph = np.asarray(Xph, float)
        Xv2 = np.asarray(Xv2, float)
        Xun = np.hstack([Xv2, Xph])
        w = self.weights
        r = (w["stack"] * self._rank(self.stack.predict_proba(Xph)[:, 1])
             + w["mono"] * self._rank(self.mono.predict_proba(Xph)[:, 1])
             + w["mlp"] * self._rank(self.mlp.predict_proba(Xun)[:, 1])
             + w["drse"] * self._rank(self.drse.predict_proba(Xv2)[:, 1]))
        return r / sum(w.values())
