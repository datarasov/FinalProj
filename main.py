# from fastapi import FastAPI
# from transformers import pipeline
# from pydantic import BaseModel

#class Item(BaseModel):
#    text: str

#app = FastAPI()
#classifier = pipeline("sentiment-analysis", model="blanchefort/rubert-base-cased-sentiment")

#@app.get("/")
#def root():
#    return ("message": "DEMO APPLICATION for AI/DevOps course")

#@app.post("/predict/")
#def predict(item: Item):
#    return classifier(item.text)[0]

# ---------- INITIAL CODE ---------------
# classifier = pipeline("sentiment-analysis", "blanchefort/rubert-base-cased-sentiment")
# classifier("Я обожаю рыбалку!")

import io
import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions

@st.cache(allow_output_mutation=True)
def load_model():
    return EfficientNetB0(weights='imagenet')


def preprocess_image(img):
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x
  

def load_image():
    uploaded_file = st.file_uploader(label='SELECT AN IMAGE FOR RECOGNITION')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None


def print_predictions(preds):
    classes = decode_predictions(preds, top=3)[0]
    for cl in classes:
        st.write(cl[1], cl[2])


model = load_model()
st.title('IMAGE CLASSIFICATION IN STREAMLIT CLOUD')
img = load_image()
result = st.button('RECOGNIZE THE IMAGE')
if result:
    x = preprocess_image(img)
    preds = model.predict(x)
    st.write('RECOGNITION RESULT -->')
    print_predictions(preds)
