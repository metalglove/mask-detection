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

# Load machine learning model(s)
root_path = os.getcwd()
assert root_path.endswith("mask-detection"), "The root path does not end with mask-detection: " + root_path 
sys.path.insert(0, root_path)
model = tf.keras.models.load_model(root_path + '/models/cnn1')

# Define functions
def process_img(frame):
    frame = cv2.resize(frame, (200, 200))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_array = frame[np.newaxis, ...] # Ellipsis object used in slice notation
    return img_array

def predict(img_array):
    prediction_val = np.matrix.item(model.predict(img_array))
    return prediction_val > 0.5, prediction_val

# Start video capture and start retrieving frames/images from video capture
vc = cv2.VideoCapture(0)
if vc.isOpened():
    rval, frame = vc.read()
else:
    rval = False

window_name = "Mask Detection - Interactive Demo"
while rval:
    rval, frame = vc.read()
    img_array = process_img(frame)
    pred, val = predict(img_array)
    cv2.imshow(window_name, frame)
    print(datetime.now().time(), pred, val)
    if cv2.waitKey(20) == 27: # exit on ESC
        break
cv2.destroyWindow(window_name)

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