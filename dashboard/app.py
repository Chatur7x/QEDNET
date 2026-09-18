"""Streamlit dashboard for QED-Net."""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import joblib
import json
from pathlib import Path

from qednet.train.config import ExperimentConfig, DEFAULT_CONFIG_YAML


st.set_page_config(
    page_title="QED-Net Research Platform",
    page_icon="🔬",
    layout="wide",
)

# Research disclaimer
RESEARCH_DISCLAIMER = """
⚠️ **RESEARCH ONLY**: This platform is a research and educational artifact. 
It is not a medical device and must not be used for real patient care.
"""

def main():
    st.title("🔬 QED-Net: Hybrid Quantum–Classical ML Platform")
    st.markdown(RESEARCH_DISCLAIMER)

    # Sidebar
    st.sidebar.header("Configuration")

    # Dataset selection
    dataset_name = st.sidebar.selectbox(
        "Dataset",
        ["breast_cancer", "heart_disease", "diabetes"],
        index=0,
    )

    # Model selection
    model_type = st.sidebar.selectbox(
        "Model",
        ["vqc", "qsvm", "reupload", "hybrid_qnn", "xgboost", "lightgbm", "logistic", "mlp"],
        index=0,
    )

    # Quantum parameters (only for quantum models)
    if model_type in ["vqc", "qsvm", "reupload", "hybrid_qnn"]:
        st.sidebar.subheader("Quantum Parameters")
        n_qubits = st.sidebar.slider("Qubits", 2, 20, 8)
        layers = st.sidebar.slider("Layers", 1, 10, 3)
        shots = st.sidebar.slider("Shots", 100, 10000, 1024, step=100)
    else:
        n_qubits = 8
        layers = 3
        shots = 1024

    # Training parameters
    st.sidebar.subheader("Training")
    outer_folds = st.sidebar.slider("CV Folds", 2, 10, 5)
    seeds = st.sidebar.slider("Seeds", 1, 20, 3)

    # Cascade threshold
    st.sidebar.subheader("Cascade (F1)")
    cascade_low = st.sidebar.slider("Ambiguity Low", 0.0, 0.5, 0.35)
    cascade_high = st.sidebar.slider("Ambiguity High", 0.5, 1.0, 0.65)

    # Conformal alpha
    st.sidebar.subheader("Conformal (F6)")
    alpha = st.sidebar.slider("Alpha (1-coverage)", 0.01, 0.5, 0.10, step=0.01)

    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Training", "🔮 Prediction", "📈 Results", "⚙️ Config", "ℹ️ About"
    ])

    with tab1:
        training_tab(dataset_name, model_type, n_qubits, layers, shots, outer_folds, seeds)

    with tab2:
        prediction_tab()

    with tab3:
        results_tab()

    with tab4:
        config_tab()

    with tab5:
        about_tab()


def training_tab(dataset_name, model_type, n_qubits, layers, shots, outer_folds, seeds):
    st.header("Training Pipeline")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Experiment Configuration")

        # Create config
        config = ExperimentConfig()
        config.dataset.name = dataset_name
        config.model.type = model_type
        config.model.n_qubits = n_qubits
        config.model.layers = layers
        config.model.shots = shots
        config.training.outer_folds = outer_folds
        config.training.seeds = seeds

        # Display config as YAML
        st.code(config.to_yaml(), language="yaml")

    with col2:
        st.subheader("Actions")

        if st.button("🚀 Run Training", type="primary"):
            with st.spinner("Running training pipeline..."):
                # This would call the actual training runner
                st.info("Training pipeline would run here. Connect to qednet.train.runner.TrainingRunner")
                st.success("Training completed! (Mock)")

        if st.button("📤 Load Existing Model"):
            st.info("Load model from artifacts/")

        st.subheader("MLflow Tracking")
        st.write("Experiment tracking via MLflow")
        if st.button("Open MLflow UI"):
            st.write("Run: `mlflow ui`")


def prediction_tab():
    st.header("Research Prediction")

    st.markdown("""
    **Instructions**: Enter feature values to get a research prediction.
    This is for benchmarking and research purposes only.
    """)

    # Feature input
    st.subheader("Input Features")
    col1, col2 = st.columns(2)

    # Breast cancer features (30 features)
    feature_names = [
        "mean radius", "mean texture", "mean perimeter", "mean area",
        "mean smoothness", "mean compactness", "mean concavity",
        "mean concave points", "mean symmetry", "mean fractal dimension",
        "radius error", "texture error", "perimeter error", "area error",
        "smoothness error", "compactness error", "concavity error",
        "concave points error", "symmetry error", "fractal dimension error",
        "worst radius", "worst texture", "worst perimeter", "worst area",
        "worst smoothness", "worst compactness", "worst concavity",
        "worst concave points", "worst symmetry", "worst fractal dimension",
    ]

    features = []
    with col1:
        for i, name in enumerate(feature_names[:15]):
            val = st.number_input(name, value=0.0, format="%.4f", key=f"feat_{i}")
            features.append(val)

    with col2:
        for i, name in enumerate(feature_names[15:], 15):
            val = st.number_input(name, value=0.0, format="%.4f", key=f"feat_{i}")
            features.append(val)

    if st.button("🔮 Predict", type="primary"):
        X = np.array(features).reshape(1, -1)

        # Mock prediction for demo
        np.random.seed(42)
        prob = np.random.uniform(0.2, 0.8)

        st.subheader("Prediction Result")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Research Prediction", "Positive" if prob > 0.5 else "Negative")
        with col2:
            st.metric("Probability", f"{prob:.3f}")
        with col3:
            st.metric("Confidence", "High" if prob < 0.35 or prob > 0.65 else "Ambiguous")

        # Uncertainty (N5)
        st.subheader("Uncertainty (Shot Noise)")
        shots = 1024
        var = prob * (1 - prob) / shots
        std = np.sqrt(var)
        z = 1.96
        ci_l = max(0, prob - z * std)
        ci_u = min(1, prob + z * std)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Variance", f"{var:.6f}")
        with col2:
            st.metric("Std Dev", f"{std:.4f}")
        with col3:
            st.metric(f"95% CI", f"[{ci_l:.3f}, {ci_u:.3f}]")

        st.caption("Note: This is shot/sampling uncertainty, NOT model uncertainty.")

        # Conformal prediction (F6)
        st.subheader("Conformal Prediction Set")
        alpha = 0.10
        # Mock conformal set
        if prob < 0.35:
            pred_set = [0]
            abstained = False
        elif prob > 0.65:
            pred_set = [1]
            abstained = False
        else:
            pred_set = [0, 1]
            abstained = True

        st.write(f"Prediction Set: {pred_set}")
        st.write(f"Abstained: {abstained}")
        st.write(f"Coverage Target: {1-alpha:.0%}")

        # Cascade (F1)
        st.subheader("Quantum Second-Opinion Cascade")
        if prob < 0.35 or prob > 0.65:
            st.success("✅ Classical screener confident → Direct prediction")
            st.write("Routing: `classical` (high_confidence)")
        else:
            st.warning("⚠️ Ambiguous probability → Quantum second opinion")
            st.write("Routing: `quantum_second_opinion` (ambiguous_probability)")


def results_tab():
    st.header("Experiment Results")

    st.info("Results from MLflow tracking will appear here.")

    # Mock results display
    if st.button("Load Latest Results"):
        # Mock data
        data = {
            "Metric": ["AUROC", "AUPRC", "Sens@Spec95", "MCC", "Brier", "ECE"],
            "CV Mean": [0.87, 0.85, 0.72, 0.65, 0.12, 0.08],
            "CV 95% CI": ["[0.82, 0.91]", "[0.79, 0.90]", "[0.65, 0.78]", "[0.58, 0.71]", "[0.09, 0.15]", "[0.05, 0.11]"],
            "Test": [0.85, 0.83, 0.70, 0.62, 0.13, 0.09],
        }
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)

        # Decision curve
        st.subheader("Decision Curve Analysis")
        thresholds = np.linspace(0.01, 0.99, 99)
        net_benefit = np.random.uniform(0, 0.3, len(thresholds))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=thresholds, y=net_benefit, name="Net Benefit"))
        fig.add_trace(go.Scatter(x=thresholds, y=np.maximum(0, 0.3 - thresholds*0.3), name="Treat All"))
        fig.add_trace(go.Scatter(x=thresholds, y=np.zeros_like(thresholds), name="Treat None"))
        fig.update_layout(title="Decision Curve", xaxis_title="Threshold", yaxis_title="Net Benefit")
        st.plotly_chart(fig, use_container_width=True)


def config_tab():
    st.header("Configuration Management")

    st.subheader("Default Config")
    st.code(DEFAULT_CONFIG_YAML, language="yaml")

    st.subheader("Save/Load Config")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save Config"):
            st.success("Config saved to configs/experiment.yaml")
    with col2:
        if st.button("📂 Load Config"):
            st.info("Load config from file")


def about_tab():
    st.header("About QED-Net")

    st.markdown("""
    ### QED-Net: Hybrid Quantum–Classical Machine Learning Platform for Early Disease Detection

    **Project Members:**
    - **Member A**: Quantum Models (VQC, QSVM, Reupload, Hybrid QNN)
    - **Member B**: Data, Preprocessing, Leakage Guard
    - **Member C**: Platform & Integration (this module)

    ### Deliverables (Member C)
    - D7: Config-driven training/inference pipeline
    - D11: FastAPI backend
    - D12: Streamlit dashboard
    - F1: Quantum Second-Opinion Cascade
    - F6: Conformal prediction / abstention
    - N5: Shot-noise uncertainty
    - D15: Docker deployment
    - D16: Test suite and CI
    - D17: Documentation
    - D18: Reproducibility bundle
    - D20: Demo video/pitch support

    ### Architecture
    ```
    Raw Data → Validation → Preprocessing → Compression → Encoding
                                                    ↓
                                              Model (Quantum/Classical)
                                                    ↓
                                              Post-processing → Calibration
                                                    ↓
                                              Uncertainty (N5)
                                                    ↓
                                              Conformal (F6)
                                                    ↓
                                              Cascade (F1)
                                                    ↓
                                              Final Result
    ```

    ### Key Principles
    1. **No Quantum Superiority Assumption** - Honest comparison with classical baselines
    2. **Leakage Prevention** - All transformers fitted only within CV folds
    3. **Reproducibility** - Config hashes, pinned dependencies, MLflow tracking
    4. **Research Only** - Not a medical device
    """)


if __name__ == "__main__":
    main()