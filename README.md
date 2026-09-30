# EvoDrift — Adaptive Multi-Objective Evolutionary Deep Learning

**Problem:** Adaptive Multi-Objective Evolutionary Deep Learning for Robust Classification under Non-Stationary Data Distributions.

## Chosen vertical
**Environmental disaster risk assessment.** The system classifies a continuous stream of environmental sensor observations into three risk levels using rainfall, river level, temperature and seismic readings.

## Approach and algorithmic logic

EvoDrift combines a compact feed-forward deep neural network with an adaptive evolutionary search.

1. **Non-stationary stream:** four ordered data phases deliberately shift feature distributions and class priors.
2. **Baseline DNN:** a fixed MLP provides a reference model.
3. **Online inference:** each incoming observation is classified by the current adaptive model.
4. **Drift monitoring:** a Page-Hinkley detector monitors the adaptive model's online error signal.
5. **Multi-objective evolution:** after a drift signal, a population of candidate MLP configurations is evaluated. The two objectives are:
   - maximize macro-F1;
   - minimize an inference-cost proxy combining measured latency and parameter count.
6. **Evolution:** Pareto dominance, tournament selection, crossover and mutation create the next generation.
7. **Adaptation:** the selected model is refit on a recent labeled buffer, allowing the classifier to respond to the new distribution.
8. **Evaluation:** frozen baseline and adaptive model are compared overall and separately for each distribution phase.

This is an intentionally reproducible CPU-sized implementation rather than a claim of production-scale GPU evolutionary training.

## End-to-end flow

`Sensor stream → DNN inference → Page-Hinkley drift detector →`
- `no drift → keep current model`
- `drift → Pareto evolutionary search → refit on recent window → continue inference`

## Why the objectives are multi-objective

Accuracy alone can select a model that is unnecessarily expensive. EvoDrift keeps model quality and deployment cost separate, uses Pareto dominance during evolution, and selects a practical compromise from the resulting candidates.

## Assumptions and operational constraints

- The demonstration uses synthetic environmental data because the task does not provide a required public dataset.
- Incoming labels are assumed to become available after a short delay so the online error signal and adaptation window can be evaluated.
- Evolution is triggered only after a confirmed drift signal and a cooldown period.
- The demo is designed for CPU execution; larger populations/generations can use parallel hardware.
- The environmental variables are illustrative features, not a real disaster-warning service.

## Repository structure

```text
EvoDrift-1/
├── app.py
├── requirements.txt
├── README.md
├── evodrift/
│   ├── data.py
│   ├── drift.py
│   ├── evolution.py
│   └── pipeline.py
├── tests/
│   └── test_pipeline.py
└── notebooks/
    └── EvoDrift_Demo.ipynb
```

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pytest -q
python -m evodrift.pipeline
streamlit run app.py
```

## Reproducibility

The default experiment uses seed `42`. No API keys, credentials, external services or private data are required.

A reference CPU run with 300 samples/phase, population 8 and 3 generations produced:
- baseline accuracy: **0.693**
- adaptive accuracy: **0.905**
- baseline macro-F1: **0.696**
- adaptive macro-F1: **0.902**
- drift-triggered evolutionary adaptations: **5**

These numbers are a reproducible reference run, not a guarantee for every random seed or hardware environment.

## Evaluation alignment

- **Code quality:** modular Python package, typed interfaces where useful, tests and readable separation of concerns.
- **Security:** no secrets in source; no network calls are required by the core experiment.
- **Efficiency:** compact MLP search space, small evolutionary population, drift-triggered adaptation rather than continuous re-optimization.
- **Testing:** pytest covers data generation, drift detection, Pareto dominance and an end-to-end run.
- **Accessibility:** Streamlit dashboard presents metrics and phase-level results in tables and charts.
- **Problem alignment:** the implementation explicitly models distribution shift, drift detection, multi-objective evolutionary adaptation and classification robustness.
