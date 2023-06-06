import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
import uuid   # Unique identifier
import os
import time
import pandas
from os import listdir
from yolov5 import utils
import pygame
# import utils
# from utils.plots import plot_results

# Check if CUDA is available
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/yolov5s_results/weights/best.pt', force_reload=False)
model.eval()

# Move the model to the GPU
# model.to(device)

target_label = "drowsy"
confidence_threshold = 0.70
drowsy_frames = []
total_frames = 1
frame_delay = 10

alarm_sound = "alarm/alarm.mp3"

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # set buffer size to reduce delay

# Initialize Pygame
pygame.init()
pygame.mixer.init()

def process_frame(frame):
    # resize the frame to a smaller size to speed up the inference process
    resized = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)

    # Convert the image to BGR format
    bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
    # Normalize the image
    # normalized = bgr.astype(np.float32) / 255.0

    results = model(bgr)

    # Retrieve the detected objects' information
    df = results.pandas().xyxy[0]
    # num_objects = len(df)  # Number of detected objects

    if not df.empty and len(df) > 0:
        label = df.iloc[0]['name']  # class label
        confidence = df.iloc[0]['confidence']  # Confidence score
        print(f"Object : Class={label}, Confidence={confidence}")

        pygame.mixer.music.load(alarm_sound)
        if label == target_label and confidence > confidence_threshold:
            pygame.mixer.music.play()
            drowsy_frames.append(confidence*100)
        else:
            pygame.mixer.music.stop()
        # total_frames+=1
    
    return np.squeeze(results.render())

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.imshow('YOLO', process_frame(frame))
    
    if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
