"""
bandit/model.py — Auto-Dimension LinUCB (Option A Compatible)
-------------------------------------------------------------
✓ No 'arms' kwarg bug
✓ Uses n_arms + dim parameters
✓ Auto-expansion of A and b matrices when feature dimension changes
✓ Fully shape-safe import/export
✓ Compatible with policy.py (Option A, Router-first)
-------------------------------------------------------------
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any


class LinUCBBandit:
    """
    A shape-safe LinUCB implementation with automatic dimension expansion.
    """

    def __init__(self, n_arms: int, dim: int, alpha: float = 0.3):
        self.n_arms = n_arms
        self.dim = dim
        self.alpha = alpha

        # A: (n_arms × dim × dim)
        self.A = np.array([np.eye(dim) for _ in range(n_arms)], dtype=float)

        # b: (n_arms × dim)
        self.b = np.zeros((n_arms, dim), dtype=float)

    # ---------------------------------------------------------------
    # Utility: Expand dimension safely if feature vector grows
    # ---------------------------------------------------------------
    def _ensure_dim(self, d_new: int):
        if d_new == self.dim:
            return

        # Expand each A and b matrix
        A_new = []
        for i in range(self.n_arms):
            oldA = self.A[i]
            pad = d_new - self.dim

            # Pad identity to bottom-right
            padded = np.pad(
                oldA,
                pad_width=((0, pad), (0, pad)),
                mode="constant",
                constant_values=0.0
            )
            # Add identity on new dims
            for j in range(self.dim, d_new):
                padded[j][j] = 1.0

            A_new.append(padded)

        self.A = np.stack(A_new)
        self.b = np.pad(self.b, pad_width=((0, 0), (0, d_new - self.dim)), constant_values=0.0)

        self.dim = d_new
        print(f"[LinUCB] 🔄 Auto-expanded dim → {d_new}")

    # ---------------------------------------------------------------
    # Predict arm (UCB score)
    # ---------------------------------------------------------------
    def predict(self, x: np.ndarray) -> str:
        x = x.astype(float)
        d_new = len(x)
        self._ensure_dim(d_new)

        p_vals = []
        for i in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[i])
            theta = A_inv @ self.b[i]
            p = float(theta @ x + self.alpha * np.sqrt(x @ A_inv @ x))
            p_vals.append(p)

        arm_index = int(np.argmax(p_vals))
        return arm_index

    # ---------------------------------------------------------------
    # Update with new reward
    # ---------------------------------------------------------------
    def update(self, arm: str | int, reward: float, x: np.ndarray):
        if isinstance(arm, str):
            raise ValueError("LinUCB uses arm index, not label.")

        arm_index = int(arm)

        x = x.astype(float)
        d_new = len(x)
        self._ensure_dim(d_new)

        x = x.reshape(-1, 1)  # column vector

        self.A[arm_index] += x @ x.T
        self.b[arm_index] += reward * x.flatten()

    # ---------------------------------------------------------------
    # Export persistent state
    # ---------------------------------------------------------------
    def export_state(self) -> Dict[str, Any]:
        return {
            "n_arms": self.n_arms,
            "dim": self.dim,
            "A": self.A.tolist(),
            "b": self.b.tolist(),
            "alpha": self.alpha
        }

    # ---------------------------------------------------------------
    # Import persistent state (shape-safe)
    # ---------------------------------------------------------------
    def import_state(self, state: Dict[str, Any]):
        self.n_arms = state["n_arms"]
        self.dim = state["dim"]
        self.alpha = state.get("alpha", 0.3)

        self.A = np.array(state["A"], dtype=float)
        self.b = np.array(state["b"], dtype=float)

        print(f"[LinUCB] 📥 State loaded (n_arms={self.n_arms}, dim={self.dim})")
