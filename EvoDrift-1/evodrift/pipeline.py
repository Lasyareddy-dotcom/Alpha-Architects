"""End-to-end experiment: baseline vs adaptive evolutionary DNN on a drifting stream."""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from .data import make_stream, FEATURES
from .drift import PageHinkley
from .evolution import optimize

def _fit_baseline(X, y, seed=42):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = MLPClassifier(
        hidden_layer_sizes=(16,), alpha=1e-3, learning_rate_init=0.01,
        max_iter=140, early_stopping=True, validation_fraction=.15,
        n_iter_no_change=12, random_state=seed
    )
    model.fit(Xs, y)
    return model, scaler

def _predict(model, scaler, X):
    return model.predict(scaler.transform(X))

def run_experiment(samples_per_phase=300, seed=42, population_size=8, generations=3):
    X, y, phase = make_stream(samples_per_phase, seed)
    warmup = int(samples_per_phase * 0.73)
    X_train, y_train = X[:warmup], y[:warmup]
    stream_X, stream_y = X[warmup:], y[warmup:]
    stream_phase = phase[warmup:]

    base_model, base_scaler = _fit_baseline(X_train, y_train, seed)
    adaptive_model, adaptive_scaler = base_model, base_scaler
    detector = PageHinkley(delta=.003, threshold=5.0, min_instances=35)
    buffer_X, buffer_y = list(X_train[-140:]), list(y_train[-140:])
    rows, drifts = [], []
    last_retrain = -10_000

    for i, (xi, yi, ph) in enumerate(zip(stream_X, stream_y, stream_phase)):
        xi2 = xi.reshape(1, -1)
        pred_base = int(_predict(base_model, base_scaler, xi2)[0])
        pred_adapt = int(_predict(adaptive_model, adaptive_scaler, xi2)[0])
        error = float(pred_adapt != yi)
        drift = detector.update(error)
        if drift and (i - last_retrain >= 35) and len(buffer_y) >= 80:
            bx = np.asarray(buffer_X[-160:])
            by = np.asarray(buffer_y[-160:])
            cut = max(40, int(len(by) * .72))
            t0 = time.perf_counter()
            model, ind = optimize(
                bx[:cut], by[:cut], bx[cut:], by[cut:],
                population_size=population_size, generations=generations, seed=seed+i
            )
            adaptive_model = model
            adaptive_scaler = StandardScaler().fit(bx[:cut])
            # Refit the selected architecture on all available buffer data to use labels efficiently.
            adaptive_model.fit(adaptive_scaler.transform(bx), by)
            elapsed = time.perf_counter() - t0
            last_retrain = i
            drifts.append({"stream_index": i, "phase": int(ph), "f1": ind.f1, "latency_ms": ind.latency_ms, "cost": ind.cost, "retrain_seconds": elapsed})
        buffer_X.append(xi); buffer_y.append(yi)
        if len(buffer_y) > 180:
            buffer_X.pop(0); buffer_y.pop(0)
        rows.append({
            "index": i, "phase": int(ph), "y": int(yi),
            "baseline_pred": pred_base, "adaptive_pred": pred_adapt,
            "baseline_correct": int(pred_base == yi), "adaptive_correct": int(pred_adapt == yi),
            "drift": int(drift),
        })

    step = pd.DataFrame(rows)
    phase_rows = []
    for ph, g in step.groupby("phase"):
        phase_rows.append({
            "phase": int(ph),
            "baseline_accuracy": accuracy_score(g.y, g.baseline_pred),
            "adaptive_accuracy": accuracy_score(g.y, g.adaptive_pred),
            "baseline_f1": f1_score(g.y, g.baseline_pred, average="macro"),
            "adaptive_f1": f1_score(g.y, g.adaptive_pred, average="macro"),
            "drift_signals": int(g.drift.sum()),
        })
    phase_metrics = pd.DataFrame(phase_rows)
    summary = {
        "baseline_accuracy": float(accuracy_score(step.y, step.baseline_pred)),
        "adaptive_accuracy": float(accuracy_score(step.y, step.adaptive_pred)),
        "baseline_f1": float(f1_score(step.y, step.baseline_pred, average="macro")),
        "adaptive_f1": float(f1_score(step.y, step.adaptive_pred, average="macro")),
        "drift_events": len(drifts),
        "stream_samples": len(step),
        "features": FEATURES,
    }
    return {"summary": summary, "phase_metrics": phase_metrics, "drift_events": pd.DataFrame(drifts), "stream": step}

if __name__ == "__main__":
    result = run_experiment()
    print(pd.Series(result["summary"]).to_string())
    print("\nPhase metrics:\n", result["phase_metrics"].to_string(index=False))
