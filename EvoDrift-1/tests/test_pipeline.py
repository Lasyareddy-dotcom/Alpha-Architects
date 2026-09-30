from evodrift.data import make_stream
from evodrift.drift import PageHinkley
from evodrift.evolution import Genome, dominates, Individual
from evodrift.pipeline import run_experiment

def test_stream_has_four_phases():
    X, y, p = make_stream(40, seed=1)
    assert X.shape == (160, 4)
    assert set(p) == {0, 1, 2, 3}

def test_drift_detector_can_signal_shift():
    d = PageHinkley(delta=0.0, threshold=1.0, min_instances=5)
    signals = [d.update(0.0) for _ in range(8)]
    signals += [d.update(1.0) for _ in range(12)]
    assert any(signals)

def test_pareto_dominance():
    a = Individual(Genome(8,.001,.01,80), error=.1, cost=.2)
    b = Individual(Genome(16,.001,.01,80), error=.2, cost=.3)
    assert dominates(a, b)

def test_end_to_end_small():
    r = run_experiment(samples_per_phase=80, seed=2, population_size=4, generations=2)
    assert r["stream"].shape[0] > 100
    assert 0 <= r["summary"]["adaptive_f1"] <= 1
    assert "adaptive_accuracy" in r["phase_metrics"].columns
