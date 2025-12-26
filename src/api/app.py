"""FastAPI application for Boston House Price Prediction"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .predict import predictor
from .schemas import HealthResponse, HouseFeatures, PredictionResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    # Startup
    logger.info("=" * 70)
    logger.info("🚀 Starting Boston House Price Prediction API")
    logger.info("=" * 70)

    success = predictor.load_artifacts()

    if not success:
        logger.error("❌ Failed to load model artifacts!")
        logger.error("API will start but predictions will fail.")
    else:
        logger.info("=" * 70)
        logger.info("✅ API ready to serve predictions!")
        logger.info(f"   Model: {predictor.model_alias}")
        logger.info(f"   Run ID: {predictor.run_id}")
        logger.info("=" * 70)

    yield

    # Shutdown
    logger.info("🛑 Shutting down API...")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Boston House Price Prediction API",
        "version": settings.VERSION,
        "description": settings.DESCRIPTION,
        "endpoints": {"health": "/health", "predict": "/predict", "docs": "/docs"},
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API health status"""
    artifacts_status = predictor.get_artifacts_status()
    model_loaded = all(artifacts_status.values())

    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy", model_loaded=model_loaded
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(features: HouseFeatures):
    """
    Predict Boston house price based on 13 input features.

    Returns predicted house price in USD.

    **Example Request:**
    ```
    {
        "AGE": 65.2,
        "B": 396.90,
        "CHAS": 0,
        "CRIM": 0.00632,
        "DIS": 4.0900,
        "INDUS": 2.31,
        "LSTAT": 4.98,
        "NOX": 0.538,
        "PTRATIO": 15.3,
        "RAD": 1,
        "RM": 6.575,
        "TAX": 296.0,
        "ZN": 18.0
    }
    ```
    """
    try:
        # Check if model is loaded
        if predictor.model is None:
            raise HTTPException(
                status_code=503, detail="Model not loaded. Please try again later."
            )

        # Convert Pydantic model to dict
        features_dict = features.model_dump()

        # Make prediction
        predicted_price_thousands = predictor.predict(features_dict)

        # Convert to dollars
        predicted_price_dollars = predicted_price_thousands * 1000

        return PredictionResponse(predicted_price=round(predicted_price_dollars, 2))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Please check your input and try again.",
        )
