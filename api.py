from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from typing import List
import joblib
from src.training_pipline import training
import os
from src.predict import load_model

app = FastAPI()

# Define the request body format
class PredictionRequest(BaseModel):
    data: List[List[float]]  # Expecting a list of lists (e.g., multiple rows of features)

@app.get("/")
def read_root():
    return {"message": "Model API is running!"}

@app.post("/train")
def train():
    training()
    return "Model Training completed"

@app.post("/predict")
def predict(request: PredictionRequest):
    # Convert input data to DataFrame
    data = pd.DataFrame(request.data, columns=[
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ])
    
    # Perform prediction
    
    model_path='models/random_forest_model.pkl'
    
    model = load_model(model_path)

    predictions = model.predict(data)
    
    # Return predictions as a list
    return {"predictions": predictions.tolist()}

