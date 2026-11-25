"""FastAPI backend for QED-Net."""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import os

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator


class LazyResearchApp:
    """Import the optional research stack only when a research route is used."""

    def __init__(self) -> None:
        self._app = None

    async def __call__(self, scope, receive, send) -> None:
        if self._app is None:
            from api.research import app as research_app

            self._app = research_app
        await self._app(scope, receive, send)

# Global serving pipeline
logger = logging.getLogger(__name__)
serving_pipeline: ServingPipeline | None = None
model: BaseEstimator | None = None


class PredictionRequest(BaseModel):
    """Request model for prediction."""

    features: list[float] = Field(..., description="Input features")
    return_uncertainty: bool = True
    return_conformal: bool = True

    @field_validator("features")
    @classmethod
    def validate_features(cls, values: list[float]) -> list[float]:
        if not values or not np.isfinite(values).all():
            raise ValueError("features must be a non-empty list of finite numbers")
        return values


class BatchPredictionRequest(BaseModel):
    """Request model for batch prediction."""

    features: list[list[float]] = Field(..., description="Batch of input features")
    return_uncertainty: bool = True
    return_conformal: bool = True

    @field_validator("features")
    @classmethod
    def validate_features(cls, values: list[list[float]]) -> list[list[float]]:
        if not values or any(not row for row in values):
            raise ValueError("features must contain at least one non-empty row")
        width = len(values[0])
        if any(len(row) != width for row in values):
            raise ValueError("all feature rows must have the same width")
        if not np.isfinite(np.asarray(values, dtype=float)).all():
            raise ValueError("features must contain only finite numbers")
        return values


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
        from qednet.deploy.serving import ServingPipeline

        # Create serving pipeline with loaded model
        serving_pipeline = ServingPipeline(
            model=model,
            model_name="loaded_model",
            config_hash="unknown",
        )
    except Exception as e:
        logger.warning("Could not load model artifact: %s", e)
    yield
    # Cleanup on shutdown
    pass


app = FastAPI(
    title="QED-Net API",
    description="Research and educational API for hybrid quantum-classical ML. Not for clinical use.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.getenv(
            "QEDNET_CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173",
        ).split(",")
        if origin.strip()
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/research", LazyResearchApp())


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        model_name="loaded_model" if model is not None else None,
    )


@app.get("/live")
async def liveness_check():
    """Process liveness; intentionally independent of model availability."""
    from qednet import __version__

    return {"status": "ok", "research_only": True, "version": __version__}


@app.get("/ready")
async def readiness_check():
    """Readiness for predictions; returns 503 until a serving model is loaded."""
    if serving_pipeline is None:
        raise HTTPException(status_code=503, detail="Serving pipeline is not initialized")
    return {"status": "ready", "model_loaded": True}


@app.get("/models", response_model=list[ModelInfoResponse])
async def list_models():
    """List available models."""
    if model is None:
        raise HTTPException(status_code=503, detail="No model loaded")
    return [
        ModelInfoResponse(
            model_name="loaded_model",
            config_hash="unknown",
            model_type=type(model.named_steps.get("model", model)).__name__,
        )
    ]


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Single prediction endpoint."""
    if serving_pipeline is None:
        raise HTTPException(status_code=503, detail="Serving pipeline not initialized")

    X = np.array(request.features).reshape(1, -1)
    result = serving_pipeline.predict(
        X,
        return_uncertainty=request.return_uncertainty,
        return_conformal=request.return_conformal,
    )[0]

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
        }
        if result.uncertainty
        else None,
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
    results = serving_pipeline.predict(
        X,
        return_uncertainty=request.return_uncertainty,
        return_conformal=request.return_conformal,
    )

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
            }
            if r.uncertainty
            else None,
            conformal_set=r.conformal.prediction_set if r.conformal else None,
            abstained=r.abstained,
            model_name=r.model_name,
            config_hash=r.config_hash,
        )
        for r in results
    ]


@app.get("/experiments/{experiment_id}")
async def get_experiment(experiment_id: str):
    """Experiment lookup is not available in this serving API."""
    raise HTTPException(status_code=501, detail="Experiment lookup is not implemented")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
