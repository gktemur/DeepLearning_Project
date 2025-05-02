from fastapi import FastAPI
from .routes import churn, iade, product

app = FastAPI(
    title="Customer Analytics API",
    description="API for customer analytics including churn prediction, return risk prediction, and product purchase prediction",
    version="1.0.0"
)

# Include routers
app.include_router(churn.router, prefix="/churn", tags=["churn"])
app.include_router(iade.router, prefix="/iade", tags=["iade"])
app.include_router(product.router, prefix="/product", tags=["product"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Customer Analytics API",
        "endpoints": {
            "/churn/predict": "POST - Make churn predictions",
            "/iade/predict": "POST - Predict return risk",
            "/product/predict": "POST - Predict product purchase probabilities",
            "/health": "GET - Check API health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "services": {
            "churn": churn.is_healthy(),
            "iade": iade.is_healthy(),
            "product": product.is_healthy()
        }
    } 