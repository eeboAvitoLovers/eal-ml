from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

app = FastAPI()

class InputData(BaseModel):
    message: str

class OutputData(BaseModel):
    message: str
    cluster: int

@app.post("/predict/")
async def predict_cluster(data: InputData) -> OutputData:
    input_string = [data.message]
    preds = model.predict(input_string)
    return OutputData(message=data.message, cluster=preds[0])