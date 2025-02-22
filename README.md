# Alzheimer-Predict

🧠 Alzheimer Prediction Model  
This repository contains a deep learning model designed for Alzheimer's disease prediction using EfficientNetB0. The model was trained on medical imaging data and can classify different stages of Alzheimer’s disease.  

📂 Model Files  
### efficientnet_model.h5  

This is the saved trained model in HDF5 format.  
It can be loaded directly in TensorFlow/Keras for inference.  
Usage:  
from tensorflow.keras.models import load_model  

model = load_model("efficientnet_model.h5")  

### efficientnet_model/  

This directory contains the TensorFlow SavedModel format version of the model.  
It includes model architecture, weights, and training configuration.  
Usage:  
import tensorflow as tf  

model = tf.keras.models.load_model("efficientnet_model")  
📌 How to Use the Model  
To use the model for prediction:  

import numpy as np  

### Load the model
model = load_model("efficientnet_model.h5")  

### Example input (adjust shape to match your dataset)
sample_input = np.random.rand(1, 224, 224, 3)  # Assuming 224x224 images  

### Get prediction
prediction = model.predict(sample_input)  
print("Prediction:", prediction)  
