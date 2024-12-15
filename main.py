
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import joblib
import numpy as np

# Load the hybrid model
model = joblib.load('final_model.pkl')

# Initialize FastAPI
app = FastAPI()

# Root endpoint for testing
@app.get("/")
async def root():
    return {"message": "API is running"}

# Define the request schema
class NewsArticle(BaseModel):
    headline: str
    body_text: str

# Define the endpoint for predictions
@app.post("/predict")
async def predict(article: NewsArticle):
    try:
        # Example feature extraction (replace with your preprocessing pipeline)
        headline_features = len(article.headline)  # Simplified example
        body_text_features = len(article.body_text)  # Simplified example

        # Combine features into a single array (adjust as per your model's needs)
        features = np.array([headline_features, body_text_features]).reshape(1, -1)

        # Predict with the hybrid model
        prediction = model.predict(features)
        confidence = model.predict_proba(features).max()

        # Generate feedback
        if prediction[0] == 1:
            return {
                "prediction": "fake",
                "confidence": f"{confidence * 100:.2f}%",
                "feedback": "This news is classified as fake. Please verify with trusted sources."
            }
        else:
            return {
                "prediction": "real",
                "confidence": f"{confidence * 100:.2f}%",
                "feedback": "This news is classified as real. Proceed with caution."
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
