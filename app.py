from fastapi import FastAPI
from pydantic import BaseModel
from inference_onnx import ColaONNXPredictor

app = FastAPI(title="MLOps Basics App")

predictor = ColaONNXPredictor("./models/model.onnx")

class PredictRequest(BaseModel):
    text: str

@app.get("/")
async def home_page():
    return "<h2>Sample prediction API</h2>"


@app.post("/predict")
async def get_prediction(query: PredictRequest):
    result =  predictor.predict(query.text)
    return result