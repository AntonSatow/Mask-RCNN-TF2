
import numpy as np
import mrcnn
import mrcnn.config
import mrcnn.model
import mrcnn.visualize
import cv2 as cv
import os
import threading
import requests

import tkinter as tk
from functools import partial


class Flag:
    def __init__(self):
        self._flag = False

    def set(self):
        self._flag = True

    def is_set(self):
        return self._flag

# load the class label names from disk, one label per line
# CLASS_NAMES = open("coco_labels.txt").read().strip().split("\n")

CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench',
               'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
               'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
               'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
               'sports ball', 'kite', 'baseball bat', 'baseball glove',
               'skateboard', 'surfboard', 'tennis racket', 'bottle', 
               'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 
               'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
               'couch', 'potted plant', 'bed', 'dining table', 'toilet',
               'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
               'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
               'book', 'clock', 'vase', 'scissors', 'teddy bear', 
               'hair drier', 'toothbrush']

class SimpleConfig(mrcnn.config.Config):
    # Give the configuration a recognizable name
    NAME = "coco_inference"
    
    # set the number of GPUs to use along with the number of images per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

	# Number of classes = number of classes + 1 (+1 for the background). The background class is named BG
    NUM_CLASSES = len(CLASS_NAMES)
    
    USE_MINI_MASK = True
    
# Initialize the Mask R-CNN model for inference and then load the weights.
# This step builds the Keras model architecture.
model = mrcnn.model.MaskRCNN(mode="inference", 
                                config=SimpleConfig(),
                                model_dir=os.getcwd())

IP = "169.254.215.104" #Replace with current IP

urls = [
    "rtsp://" + IP + "/avc/ch1",
    "rtsp://" + IP + "/mjpg/ch1",
    "rtsp://" + IP + "/mpeg4/ch1",
    "rtsp://" + IP + "/mpeg4/",
    "rtsp://" + IP + "/avc/",
    "rtsp://" + IP + "/mjpg/",
    "rtsp://" + IP + "/bundles/userweb/img/set.svg#spot-white",
    0
]

frame = None
frame_det = None
labels = None
squares = None

def start_vid_capt(url):
  global frame
  i = 0
  cap = cv.VideoCapture(url)
  while True:
    ret, frame = cap.read()
    frame = cv.resize(frame, (1024, 768))
    if(labels is not None and squares is not None):
      for square in squares:
        frame_det, (startX, startY), (endX, endY), color, thickness = square
        cv.rectangle(frame, (startX, startY), (endX, endY), color, thickness)
        cv.putText(frame, labels, (startX, startY - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
    if frame is not None:
        cv.imshow('Video Stream', frame)
    
    # Press C on keyboard to save Image
    if 0xFF == ord('c') | ret:
        i += 1
        cv.imwrite(str(i) + '_frame.jpg', frame)
        
    # Press Q on keyboard to  exit
    if 0xFF == ord('q') | ret:
      exit_flag.set()
      cap.release()
 
def on_button_click(url):
    threading.Thread(target=start_vid_capt, args=(url,), daemon=True).start()
    root.destroy()

root = tk.Tk()

for url in urls:
    button = tk.Button(root, text=url, command=lambda url=url: on_button_click(url))
    button.pack()

root.mainloop()
exit_flag = Flag() # Flag to stop the thread
# Load the weights into the model.
model.load_weights(filepath="mask_rcnn_coco.h5", 
                    by_name=True)

while True:
  if frame is not None:
    # Perform a forward pass of the network to obtain the results
    r = model.detect([frame])

    # Get the results for the first image.
    r = r[0]
    squares = [None] * r["rois"].shape[0]
  
    for i in range(0, r["rois"].shape[0]):
        (startY, startX, endY, endX) = r["rois"][i]
        labels = CLASS_NAMES[r["class_ids"][i]]
        squares[i] = frame_det,(startX, startY), (endX, endY), (0, 255, 0), 2
  if(exit_flag.is_set()):
    break    
cv.destroyAllWindows()