"""Small, dependency-light multi-objective evolutionary optimizer.

Objectives:
1. maximize macro-F1 (implemented as minimizing 1-F1)
2. minimize a latency/complexity proxy

The implementation uses Pareto dominance, crossover and mutation. It is deliberately
small enough to run on CPU for a hackathon evaluation while remaining reproducible.
"""
from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

@dataclass(frozen=True)
class Genome:
    hidden: int
    alpha: float
    learning_rate_init: float
    max_iter: int

@dataclass
class Individual:
    genome: Genome
    error: float = 1.0
    cost: float = 1.0
    f1: float = 0.0
    latency_ms: float = 0.0

def random_genome(rng):
    return Genome(
        hidden=int(rng.choice([8, 12, 16, 24, 32])),
        alpha=float(10 ** rng.uniform(-5, -2)),
        learning_rate_init=float(10 ** rng.uniform(-3.0, -1.2)),
        max_iter=int(rng.choice([80, 120, 160])),
    )

def _latency_ms(model, X):
    reps = max(1, min(5, 2000 // max(1, len(X))))
    t0 = time.perf_counter()
    for _ in range(reps):
        model.predict(X)
    return (time.perf_counter() - t0) * 1000.0 / reps

def evaluate(genome, X_train, y_train, X_val, y_val, scaler=None, seed=42):
    if scaler is None:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
    model = MLPClassifier(
        hidden_layer_sizes=(genome.hidden,),
        alpha=genome.alpha,
        learning_rate_init=genome.learning_rate_init,
        max_iter=genome.max_iter,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=12,
        random_state=seed,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    f1 = f1_score(y_val, pred, average="macro")
    latency = _latency_ms(model, X_val[: min(64, len(X_val))])
    # Cost combines normalized inference latency and parameter count proxy.
    params = genome.hidden * X_train.shape[1] + genome.hidden * 3 + genome.hidden + 3
    cost = 0.65 * (latency / 10.0) + 0.35 * (params / 250.0)
    return model, f1, latency, cost

def dominates(a: Individual, b: Individual) -> bool:
    return (a.error <= b.error and a.cost <= b.cost) and (a.error < b.error or a.cost < b.cost)

def pareto_front(population):
    return [x for x in population if not any(dominates(y, x) for y in population if y is not x)]

def tournament(pop, rng):
    a, b = rng.choice(pop, 2, replace=False)
    return a if (a.error + a.cost) <= (b.error + b.cost) else b

def crossover(a: Genome, b: Genome, rng):
    return Genome(
        hidden=int(a.hidden if rng.random() < 0.5 else b.hidden),
        alpha=float((a.alpha * b.alpha) ** 0.5),
        learning_rate_init=float((a.learning_rate_init * b.learning_rate_init) ** 0.5),
        max_iter=int(a.max_iter if rng.random() < 0.5 else b.max_iter),
    )

def mutate(g: Genome, rng, rate=0.30):
    vals = dict(hidden=g.hidden, alpha=g.alpha, learning_rate_init=g.learning_rate_init, max_iter=g.max_iter)
    if rng.random() < rate: vals["hidden"] = int(np.clip(g.hidden + rng.choice([-8, 8]), 8, 32))
    if rng.random() < rate: vals["alpha"] = float(np.clip(g.alpha * (10 ** rng.normal(0, .45)), 1e-5, 1e-2))
    if rng.random() < rate: vals["learning_rate_init"] = float(np.clip(g.learning_rate_init * (10 ** rng.normal(0, .30)), 1e-3, .08))
    if rng.random() < rate: vals["max_iter"] = int(np.clip(g.max_iter + rng.choice([-40, 40]), 80, 160))
    return Genome(**vals)

def optimize(X_train, y_train, X_val, y_val, population_size=10, generations=4, seed=42):
    rng = np.random.default_rng(seed)
    population = [random_genome(rng) for _ in range(population_size)]
    best_model, best_ind = None, None
    scaler = StandardScaler()
    Xt = scaler.fit_transform(X_train)
    Xv = scaler.transform(X_val)
    for gen in range(generations):
        evaluated = []
        for i, g in enumerate(population):
            model, f1, latency, cost = evaluate(g, X_train, y_train, X_val, y_val, scaler, seed + gen*100 + i)
            ind = Individual(g, 1.0 - f1, cost, f1, latency)
            evaluated.append(ind)
            if best_ind is None or (ind.error + ind.cost) < (best_ind.error + best_ind.cost):
                best_ind, best_model = ind, model
        front = pareto_front(evaluated)
        elites = sorted(front, key=lambda z: (z.error + z.cost))[: max(2, population_size // 3)]
        new_pop = [e.genome for e in elites]
        while len(new_pop) < population_size:
            p1, p2 = tournament(evaluated, rng), tournament(evaluated, rng)
            new_pop.append(mutate(crossover(p1.genome, p2.genome, rng), rng))
        population = new_pop
    return best_model, best_ind
