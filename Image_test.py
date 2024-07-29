import os
import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained model
model_path = '/content/drive/MyDrive/sunglasses_detector_model.h5'
model = tf.keras.models.load_model(model_path)

# Define the function to preprocess the images
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (64, 64))  # Resize to the input shape of the model
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Predict if the person in the image is wearing sunglasses
def predict_image(image_path):
    preprocessed_image = preprocess_image(image_path)
    prediction = model.predict(preprocessed_image)[0][0]
    label = 'Sunglasses' if prediction > 0.5 else 'No Sunglasses'
    return label

# Sample usage
image_path = '/content/Z2RP9D7WD23I.jpg'
label = predict_image(image_path)
print(f'The person in the image is: {label}')
