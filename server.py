from fastapi import FastAPI
from pydantic import BaseModel
# import numpy as np
# import pandas as pd
# from nltk.corpus import stopwords as sw
import pickle
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

app = FastAPI()

class InputData(BaseModel):
    message: str

class OutputData(BaseModel):
    message: str
    cluster: str

@app.post("/predict/")
async def predict_cluster(data: InputData) -> OutputData:
    input_string = data.message
    cluster_id = "5"
    return  OutputData(message=input_string, cluster=cluster_id)


def predict(data):
    input_string = data.message
    preds = model.predict(input_string)
    new['cluster'] = preds
    
