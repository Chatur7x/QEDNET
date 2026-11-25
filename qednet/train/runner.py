"""Training runner for config-driven experiments."""

import time
import warnings
from pathlib import Path
from typing import Any

try:
    import mlflow
    import mlflow.sklearn
except ImportError:  # ponytail: tracking optional; runs still return results
    mlflow = None
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from qednet.data.loader import Dataset, load_dataset
from qednet.evaluation.metrics import compute_all_metrics
from qednet.models.factory import create_model
from qednet.preprocess.pipeline import create_preprocessing_pipeline
from qednet.train.config import ExperimentConfig


class TrainingRunner:
    """Run training experiments from configuration."""

    def __init__(
        self,
        config: ExperimentConfig,
        use_mock: bool = False,
        mlflow_uri: str | None = None,
        artifact_dir: str = "artifacts",
    ):
        self.config = config
        self.use_mock = use_mock
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        if mlflow is not None:
            if mlflow_uri:
                mlflow.set_tracking_uri(mlflow_uri)
            mlflow.set_experiment(config.experiment_name)

    def run(self) -> dict[str, Any]:
        """Run the complete training pipeline."""
        start_time = time.time()

        # 1. Load dataset
        print(f"Loading dataset: {self.config.dataset.name}")
        train_ds, test_ds = load_dataset(self.config.dataset.__dict__)

        # 2. Create preprocessing pipeline
        print("Creating preprocessing pipeline...")
        preprocess_config = {
            "scaler": "standard",
            "imputer": "mean",
            "compression": self.config.dataset.compression,
            "target_dim": self.config.dataset.target_dim,
            "random_state": self.config.dataset.random_state,
        }
        preprocessor = create_preprocessing_pipeline(preprocess_config)

        # 3. Create model
        print(f"Creating model: {self.config.model.type}")
        model = create_model(self.config.model, use_mock=self.use_mock)

        # 4. Build full pipeline
        full_pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        # 5. Cross-validation
        print(f"Running {self.config.training.outer_folds}-fold CV...")
        cv = StratifiedKFold(
            n_splits=self.config.training.outer_folds,
            shuffle=True,
            random_state=self.config.training.seeds,
        )

        cv_predictions = cross_val_predict(
            full_pipeline,
            train_ds.X,
            train_ds.y,
            cv=cv,
            method="predict_proba",
            n_jobs=self.config.training.n_jobs,
        )

        # 6. Compute CV metrics
        cv_metrics = compute_all_metrics(
            train_ds.y,
            cv_predictions,
            metrics=self.config.evaluation.metrics,
            bootstrap_ci=self.config.evaluation.bootstrap_ci,
        )

        # 7. Final training on full training set
        print("Training final model on full training set...")
        full_pipeline.fit(train_ds.X, train_ds.y)

        # 8. Test evaluation
        test_proba = full_pipeline.predict_proba(test_ds.X)
        test_metrics = compute_all_metrics(
            test_ds.y,
            test_proba,
            metrics=self.config.evaluation.metrics,
        )

        # 9. Log to MLflow
        mlflow_run_id = self._log_mlflow(
            full_pipeline, train_ds, test_ds, cv_metrics, test_metrics, time.time() - start_time
        )

        # 10. Save artifacts
        self._save_artifacts(full_pipeline, train_ds, test_ds, cv_metrics, test_metrics)

        total_time = time.time() - start_time

        return {
            "config_hash": self.config.compute_hash(),
            "cv_metrics": cv_metrics,
            "test_metrics": test_metrics,
            "training_time": total_time,
            "mlflow_run_id": mlflow_run_id,
            "model": full_pipeline,
        }

    def _log_mlflow(
        self,
        pipeline: Pipeline,
        train_ds: Dataset,
        test_ds: Dataset,
        cv_metrics: dict,
        test_metrics: dict,
        training_time: float,
    ) -> str | None:
        """Log experiment to MLflow."""
        if mlflow is None:
            warnings.warn("mlflow not installed; skipping tracking", stacklevel=2)
            return None
        try:
            with mlflow.start_run(run_name=self.config.experiment_name) as run:
                # Log parameters
                mlflow.log_params(self.config.to_mlflow_params())
                mlflow.log_param("training_time", training_time)

                # Log metrics
                for name, value in cv_metrics.items():
                    if isinstance(value, dict):
                        for k, v in value.items():
                            mlflow.log_metric(f"cv_{name}_{k}", v)
                    else:
                        mlflow.log_metric(f"cv_{name}", value)

                for name, value in test_metrics.items():
                    if isinstance(value, dict):
                        for k, v in value.items():
                            mlflow.log_metric(f"test_{name}_{k}", v)
                    else:
                        mlflow.log_metric(f"test_{name}", value)

                # Log model
                mlflow.sklearn.log_model(pipeline, "model")

                return run.info.run_id
        except Exception as e:
            warnings.warn(f"MLflow logging failed: {e}", stacklevel=2)
            return None

    def _save_artifacts(
        self,
        pipeline: Pipeline,
        train_ds: Dataset,
        test_ds: Dataset,
        cv_metrics: dict,
        test_metrics: dict,
    ) -> None:
        """Save model and metadata artifacts."""
        import json

        import joblib

        run_dir = self.artifact_dir / f"{self.config.experiment_name}_{self.config.compute_hash()}"
        run_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(pipeline, run_dir / "model.pkl")
        self.config.to_yaml(run_dir / "config.yaml")

        metadata = {
            "config_hash": self.config.compute_hash(),
            "dataset": train_ds.name,
            "train_samples": len(train_ds.X),
            "test_samples": len(test_ds.X),
            "n_features": train_ds.X.shape[1],
            "cv_metrics": cv_metrics,
            "test_metrics": test_metrics,
        }
        with open(run_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)


def run_experiment(config_path: str, **kwargs) -> dict[str, Any]:
    """Run experiment from config file."""
    config = ExperimentConfig.from_yaml(config_path)
    runner = TrainingRunner(config, **kwargs)
    return runner.run()
