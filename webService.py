from flask import Flask, json, request
import numpy as np
import base64
import io
from PIL import Image
from google.cloud import storage
from keras.models import load_model
from keras.preprocessing import image
import logging
import os
import h5py

app = Flask(__name__)
CLOUD_STORAGE_BUCKET = os.environ['CLOUD_STORAGE_BUCKET']

def loadmodel():
  storage_client = storage.Client()
  bucket = storage_client.get_bucket(CLOUD_STORAGE_BUCKET)
  blob = bucket.blob('classes_model_finalone.h5')
  blob.download_to_filename('./classes_model_finalone.h5')
  
  model = load_model('./classes_model_finalone.h5')
 
  return model

def decode_image(image_base64):
  image_decode = base64.b64decode(image_base64)
  image_convert = Image.open(io.BytesIO(image_decode))
  final_image = image.img_to_array(image_convert)
  final_image = Image.Image.resize(image_convert, (200, 200))
  predict_image = np.expand_dims(final_image, axis=0)
  
  return predict_image

def get_monument(data):
  prop = np.max(data)
  index = np.argmax(data)
  
  if index == 0:
     monument = 'Great Pyramid of Giza'
  elif index == 1:
     monument = 'Abu Simbel temples'
  elif index == 2:
     monument = 'Great Sphinx of Giza'
  elif index == 3:
     monument = 'Mortuary Temple of Hatshepsut'
      
  return monument
  

@app.route('/', methods=['POST'])
def getPredection():
  if request.headers['Content-Type'] == 'application/json':       
     data = request.json
     newdata = data['Name']

     image = decode_image(newdata)
     model = loadmodel()
     data = model.predict(image)
     monument = get_monument(data)
    
     return monument
    
if __name__ == "__main__":
  app.run(host='127.0.0.1', port=8080, debug=True)
