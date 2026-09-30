"""Streamlit dashboard for EvoDrift."""
import streamlit as st
import pandas as pd
from evodrift.pipeline import run_experiment

st.set_page_config(page_title="EvoDrift", page_icon="🧬", layout="wide")
st.title("🧬 EvoDrift")
st.caption("Adaptive multi-objective evolutionary deep learning under non-stationary data distributions")

with st.sidebar:
    st.header("Experiment")
    seed = st.number_input("Random seed", 0, 9999, 42)
    samples = st.slider("Samples per phase", 160, 500, 300, 20)
    population = st.slider("Evolution population", 4, 12, 8)
    generations = st.slider("Evolution generations", 2, 6, 3)

if st.button("Run adaptive evaluation", type="primary"):
    with st.spinner("Running baseline, drift detection and evolutionary adaptation..."):
        result = run_experiment(samples, seed, population, generations)
    s = result["summary"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Baseline F1", f"{s['baseline_f1']:.3f}")
    c2.metric("Adaptive F1", f"{s['adaptive_f1']:.3f}", f"{s['adaptive_f1']-s['baseline_f1']:+.3f}")
    c3.metric("Baseline accuracy", f"{s['baseline_accuracy']:.3f}")
    c4.metric("Drift events", s["drift_events"])
    st.subheader("Performance by distribution phase")
    st.dataframe(result["phase_metrics"], use_container_width=True)
    st.subheader("Detected drift events")
    st.dataframe(result["drift_events"], use_container_width=True)
    st.subheader("Accuracy over stream")
    plot_df = result["stream"][["index", "baseline_correct", "adaptive_correct"]].set_index("index").rolling(30).mean()
    st.line_chart(plot_df.rename(columns={"baseline_correct":"Baseline (30-sample mean)", "adaptive_correct":"Adaptive (30-sample mean)"}))
    st.info("The data stream is synthetic and deliberately contains four distribution phases. This makes the experiment reproducible without exposing private or proprietary data.")
else:
    st.markdown("**Workflow:** stream → drift detector → stable inference or evolutionary re-optimization → robust classification.")
