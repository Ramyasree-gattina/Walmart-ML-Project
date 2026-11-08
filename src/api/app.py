import joblib
from fastapi import FastAPI, HTTPException
import pandas as pd
from pydantic import BaseModel

app = FastAPI(title="Walmart Weekly Sales MLOps API")


class SalesPredictionRequest(BaseModel):
  Store: int
  Holiday_Flag: int
  Temperature: float
  Fuel_Price: float
  CPI: float
  Unemployment: float
  Year: int
  Month: int
  Week: int
  Day: int


@app.get("/")
def root():
  return {"message": "Walmart Weekly Sales MLOps API is running!"}


@app.get("/health")
def health():
  return {"status": "healthy"}


@app.post("/predict")
def predict(data: SalesPredictionRequest):
  try:
    model = joblib.load("models/random_forest.joblib")
    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)[0]
    return {"predicted_weekly_sales": float(prediction)}
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))