from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.predict import GrievancePredictor
import uvicorn
import os

app = FastAPI(title="Grievance Classification API", description="API to classify Indian public grievances.", version="1.0")

# Lazy loading of model to avoid load on import if running other things
predictor = None

class GrievanceRequest(BaseModel):
    text: str

class GrievanceResponse(BaseModel):
    category: str
    priority: str

@app.on_event("startup")
def load_predictor():
    global predictor
    if os.path.exists('best_model_state.bin'):
        predictor = GrievancePredictor()
    else:
        print("Warning: Model not found. API will fail on prediction.")

@app.post("/predict", response_model=GrievanceResponse)
def predict_grievance(request: GrievanceRequest):
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    category, priority = predictor.predict(request.text)
    return GrievanceResponse(category=category, priority=priority)

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": predictor is not None}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
