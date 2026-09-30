"""Synthetic non-stationary environmental stream used for reproducible evaluation."""
from __future__ import annotations
import numpy as np
from sklearn.preprocessing import StandardScaler

FEATURES = ["rainfall", "river_level", "temperature", "seismic"]

def _phase(n: int, phase: int, rng: np.random.Generator):
    centers = np.array([
        [35, 25, 28, 1.0],
        [70, 55, 32, 1.8],
        [110, 85, 38, 2.8],
    ], dtype=float)
    # Distribution shift: feature means and class priors change by phase.
    shift = np.array([
        [0, 0, 0, 0],
        [12, 8, 3, 0.25],
        [-8, 14, 6, 0.55],
        [18, -10, 9, -0.15],
    ])[phase]
    priors = [
        np.array([0.58, 0.30, 0.12]),
        np.array([0.40, 0.38, 0.22]),
        np.array([0.22, 0.42, 0.36]),
        np.array([0.47, 0.30, 0.23]),
    ][phase]
    y = rng.choice(3, size=n, p=priors)
    X = np.empty((n, 4))
    scales = np.array([10.0, 9.0, 3.5, 0.45])
    for c in range(3):
        idx = y == c
        X[idx] = rng.normal(centers[c] + shift, scales, size=(idx.sum(), 4))
    # Mild nonlinear interaction makes a small MLP useful.
    X[:, 1] += 5.0 * np.sin(X[:, 0] / 35.0)
    X[:, 3] += 0.25 * np.tanh((X[:, 1] - 55) / 25)
    return X, y

def make_stream(samples_per_phase: int = 300, seed: int = 42):
    """Return four ordered phases; phase boundaries contain intentional drift."""
    rng = np.random.default_rng(seed)
    phases = [_phase(samples_per_phase, p, rng) for p in range(4)]
    X = np.vstack([x for x, _ in phases])
    y = np.concatenate([yy for _, yy in phases])
    phase_ids = np.concatenate([
        np.full(samples_per_phase, p, dtype=int) for p in range(4)
    ])
    return X, y, phase_ids

def scale_train_test(X_train, X_test):
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test), scaler
