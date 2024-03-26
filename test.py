
import numpy as np
import mrcnn
import mrcnn.config
import mrcnn.model
import mrcnn.visualize
import cv2 as cv
import os
import threading
import openpyxl
from openpyxl import Workbook

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

#url = "rtsp://169.254.23.163/mpeg4/ch0"
#url = "rtsp://169.254.23.163/avc/ch0"
#url = "rtsp://169.254.23.163/mjpg/ch0"
#url = "rtsp://169.254.23.163/mpeg4/ch1"
#url = "rtsp://169.254.23.163/avc/ch1"
#url = "rtsp://169.254.23.163/mjpg/ch1"
#url = "rtsp://169.254.23.163/mpeg4/"
#url = "rtsp://169.254.23.163/avc/"
#url = "rtsp://169.254.23.163/mjpg/"

IP = "169.254.93.52" #Replace with current IP

urls = [
    "rtsp://" + IP + "/avc/ch1",
    "rtsp://" + IP + "/mjpg/ch1",
    "rtsp://" + IP + "/mpeg4/ch1",
    "rtsp://" + IP + "/mpeg4/",
    "rtsp://" + IP + "/avc/",
    "rtsp://" + IP + "/mjpg/",
    0
]
frame = None
frame_det = None
labels = None
squares = None

def start_vid_capt(url):
  global frame
  cap = cv.VideoCapture(url)
  while True:
    ret, frame = cap.read()
    
    #sharpening the image
    #kernel = np.array([[0,-1,0], [-1,9,-1], [0,-1,0]])
    #frame = cv.filter2D(frame, -1, kernel)
    
    #frame = cv.resize(frame, (640, 480))
    #Testing
    #write frame to excel sheet, starting at B1 for testing
    #frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    #workbook = Workbook()
    #sheet = workbook.active
    
    #for i in range(len(frame)):
        #for j in range(len(frame[i])):
            #rgb_str = str(frame[i][j][0]) + "," + str(frame[i][j][1]) + "," + str(frame[i][j][2])
            #sheet.cell(row = i + 1, column = j+2, value = rgb_str)
    #workbook.save("test.xlsx")
    #exit_flag.set()
    #break
    
    if(labels is not None and squares is not None):
      for square in squares:
        frame_det, (startX, startY), (endX, endY), color, thickness = square
        cv.rectangle(frame, (startX, startY), (endX, endY), color, thickness)
        cv.putText(frame, labels, (startX, startY - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv.imshow('Video Stream', frame)
    # Press Q on keyboard to  exit
    if cv.waitKey(15) & 0xFF == ord('q') | ret:
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