from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel

class Item(BaseModel):
    text: str

app = FastAPI()
classifier = pipeline("sentiment-analysis", model="blanchefort/rubert-base-cased-sentiment")

@app.get("/")
def root():
    return {'message': 'DEMO APPLICATION for AI/DevOps course'}

@app.post("/predict/")
def predict(item: Item):
    return classifier(item.text)[0]

 ---------- INITIAL CODE ---------------
 classifier = pipeline("sentiment-analysis", "blanchefort/rubert-base-cased-sentiment")
 classifier("Я обожаю рыбалку!")
