"""FastAPI backend for QED-Net."""

from contextlib import asynccontextmanager
from typing import Optional
import numpy as np
from numpy.typing import NDArray

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import joblib

from qednet.deploy.serving import ServingPipeline, PredictionResult
from qednet.models.base import BaseEstimator


# Global serving pipeline
serving_pipeline: Optional[ServingPipeline] = None
model: Optional[BaseEstimator] = None


class PredictionRequest(BaseModel):
    """Request model for prediction."""
    features: list[float] = Field(..., description="Input features")
    return_uncertainty: bool = True
    return_conformal: bool = True


class BatchPredictionRequest(BaseModel):
    """Request model for batch prediction."""
    features: list[list[float]] = Field(..., description="Batch of input features")
    return_uncertainty: bool = True
    return_conformal: bool = True


class PredictionResponse(BaseModel):
    """Response model for prediction."""
    prediction: int
    probability: float
    calibrated_probability: float
    routing: str
    uncertainty: dict | None = None
    conformal_set: list[int] | None = None
    abstained: bool
    model_name: str
    config_hash: str
    disclaimer: str = "RESEARCH ONLY: This platform is a research and educational artifact. It is not a medical device and must not be used for real patient care."


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_name: str | None = None


class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_name: str
    config_hash: str
    model_type: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global model, serving_pipeline
    # Load model on startup
    try:
        model = joblib.load("artifacts/model.pkl")
        # Create serving pipeline with loaded model
        serving_pipeline = ServingPipeline(
            model=model,
            model_name="loaded_model",
            config_hash="unknown",
        )
    except Exception as e:
        print(f"Warning: Could not load model: {e}")
    yield
    # Cleanup on shutdown
    pass


app = FastAPI(
    title="QED-Net API",
    description="Hybrid Quantum–Classical ML Platform for Early Disease Detection",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        model_name="loaded_model" if model is not None else None,
    )


@app.get("/models", response_model=list[ModelInfoResponse])
async def list_models():
    """List available models."""
    if model is None:
        raise HTTPException(status_code=503, detail="No model loaded")
    return [ModelInfoResponse(
        model_name="loaded_model",
        config_hash="unknown",
        model_type=type(model.named_steps.get("model", model)).__name__,
    )]


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Single prediction endpoint."""
    if serving_pipeline is None:
        raise HTTPException(status_code=503, detail="Serving pipeline not initialized")

    X = np.array(request.features).reshape(1, -1)
    result = serving_pipeline.predict(X)[0]

    return PredictionResponse(
        prediction=result.prediction,
        probability=result.probability,
        calibrated_probability=result.calibrated_probability,
        routing=result.routing,
        uncertainty={
            "variance": result.uncertainty.variance,
            "std": result.uncertainty.std,
            "ci_lower": result.uncertainty.ci_lower,
            "ci_upper": result.uncertainty.ci_upper,
            "shots": result.uncertainty.shots,
        } if result.uncertainty else None,
        conformal_set=result.conformal.prediction_set if result.conformal else None,
        abstained=result.abstained,
        model_name=result.model_name,
        config_hash=result.config_hash,
    )


@app.post("/batch-predict", response_model=list[PredictionResponse])
async def batch_predict(request: BatchPredictionRequest):
    """Batch prediction endpoint."""
    if serving_pipeline is None:
        raise HTTPException(status_code=503, detail="Serving pipeline not initialized")

    X = np.array(request.features)
    results = serving_pipeline.predict(X)

    return [
        PredictionResponse(
            prediction=r.prediction,
            probability=r.probability,
            calibrated_probability=r.calibrated_probability,
            routing=r.routing,
            uncertainty={
                "variance": r.uncertainty.variance,
                "std": r.uncertainty.std,
                "ci_lower": r.uncertainty.ci_lower,
                "ci_upper": r.uncertainty.ci_upper,
                "shots": r.uncertainty.shots,
            } if r.uncertainty else None,
            conformal_set=r.conformal.prediction_set if r.conformal else None,
            abstained=r.abstained,
            model_name=r.model_name,
            config_hash=r.config_hash,
        )
        for r in results
    ]


@app.get("/experiments/{experiment_id}")
async def get_experiment(experiment_id: str):
    """Get experiment information."""
    # Placeholder - would connect to MLflow
    return {"experiment_id": experiment_id, "status": "not_implemented"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)