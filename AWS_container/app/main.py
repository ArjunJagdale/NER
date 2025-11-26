from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.predictor import NERPredictor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = "models/roberta-lora-fewnerd-merged"

app = FastAPI(
    title="NER API", 
    version="1.0",
    description="Named Entity Recognition API using fine-tuned RoBERTa-LoRA model"
)

# Initialize predictor
try:
    predictor = NERPredictor(MODEL_PATH)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    predictor = None

class NERRequest(BaseModel):
    text: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Apple Inc. was founded by Steve Jobs in Cupertino, California."
            }
        }

@app.get("/")
def home():
    return {
        "message": "NER API is running",
        "version": "1.0",
        "model": "roberta-lora-fewnerd-merged",
        "status": "healthy" if predictor else "model_load_failed"
    }

@app.get("/health")
def health_check():
    """Health check endpoint for monitoring"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "healthy",
        "model_loaded": True
    }

@app.post("/predict")
def predict_ner(req: NERRequest):
    """
    Extract named entities from input text
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        entities = predictor.predict(req.text)
        return {
            "text": req.text,
            "entities": entities,
            "count": len(entities)
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
