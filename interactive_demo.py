# Setup
import cv2
import os
import sys
import tensorflow as tf
import numpy as np
import PIL
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime

# Load all required resources
root_path = os.getcwd()
assert root_path.endswith("mask-detection"), "The root path does not end with mask-detection: " + root_path 
sys.path.insert(0, root_path)
face_cascade = cv2.CascadeClassifier(root_path + '/haarcascade_frontalface_default.xml')
mask_detection_model = tf.keras.models.load_model(root_path + '/models/detection_cnn')
mask_judge_model = tf.keras.models.load_model(root_path + '/models/judge_cnn')

# Define functions
def process_img(frame, size:tuple):
    frame = cv2.resize(frame, size)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_array = frame[np.newaxis, ...] # Ellipsis object
    return img_array

def predict_detection(img_array):
    prediction_val = np.matrix.item(mask_detection_model.predict(img_array))
    return prediction_val > 0.5, prediction_val

def predict_judgement(img_array):
    prediction_val = mask_judge_model.predict(img_array)
    return prediction_val[0][0] > 0.5, prediction_val[0][0]

# Start video capture and start retrieving frames/images from video capture
vc = cv2.VideoCapture(0)
if vc.isOpened(): 
    rval, frame = vc.read()
else:
    rval = False

while rval:
    # Read frame
    rval, frame = vc.read() 
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect the face(s)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw rectangle around each face
    for (x, y, w, h) in faces: 
        face_img = frame[y:y + h, x:x + w]
        img_array = process_img(face_img, (200, 200))
        detected, val = predict_detection(img_array)
        # Visualising
        color = (255,0,0) # (B,G,R)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        if detected:
            img_array = process_img(face_img, (128, 128))
            correct, val = predict_judgement(img_array)
            if correct:
                color = (0,255,0)
                text = "MASK - CORRECTLY WORN" 
            else:
                color = (0,0,255)
                text = "MASK - INCORRECTLY WORN"
        else:
            text = "NO MASK"
        cv2.putText(frame, f'{text}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    # Display
    cv2.imshow("Mask Detection - Interactive Demo", frame)
    # Exit on ESC
    if cv2.waitKey(20) == 27:
        break
cv2.destroyAllWindows()

# Show predicted outcome on screen

# Example for testing (I did not push the images in the repo)
# img_path = "Figure_2.png"
# img = Image.open(img_path)
# img = img.resize((200, 200))
# img = img.convert('RGB')
# img_array = np.array(img)
# img_array = img_array[np.newaxis, ...]
# result = model.predict(img_array)
# result = result > 0.5
# print(result)