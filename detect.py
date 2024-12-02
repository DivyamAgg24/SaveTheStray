# Import the required libraries
import spidev
import RPi.GPIO as GPIO
import time
import cv2
import numpy as np
from picamera2 import Picamera2
import tflite_runtime.interpreter as tflite
import time
import os
import platform
import sys
from pathlib import Path
import PIL
from utils.torch_utils import select_device, smart_inference_mode
from utils.general import non_max_suppression, scale_boxes
from models.common import DetectMultiBackend
from utils.augmentations import letterbox
import supervision as sv
import torch

# AD9833 register setup
AD9833_FREQ0_REGISTER = 0x4000
AD9833_PHASE0_REGISTER = 0xC000
AD9833_CONTROL_REGISTER = 0x2000
AD9833_RESET = 0x0100  # Reset bit

# Reference clock frequency (in Hz)
REFERENCE_CLOCK = 25000000  # 25 MHz

# Setup SPI for AD9833
spi = spidev.SpiDev()
spi.open(0, 0)  # Open bus 0, device 0
spi.max_speed_hz = 5000000

# Function to write data to AD9833
def write_data(data):
    spi.xfer2([data >> 8, data & 0xFF])

# Function to initialize AD9833 to a known state
def initialize_ad9833():

    # Reset the AD9833 to ensure it starts in a known state
    write_data(AD9833_CONTROL_REGISTER | AD9833_RESET)
    
    # Set control register to disable output initially
    write_data(AD9833_CONTROL_REGISTER)

# Function to set frequency
def set_frequency(freq):
    
    if freq == 0:
        # Disable output
        write_data(AD9833_CONTROL_REGISTER)
    
    else:
        # Calculate frequency word
        freq_word = int((freq * 2**28) / REFERENCE_CLOCK)

        # Write frequency register
        write_data(AD9833_CONTROL_REGISTER)
        write_data(AD9833_FREQ0_REGISTER | (freq_word & 0x3FFF))  # Lower 14 bits
        write_data(AD9833_FREQ0_REGISTER | ((freq_word >> 14) & 0x3FFF))  # Upper 14 bits

# Function call to Iinitialize the AD9833 to a known state
initialize_ad9833()

# Select the device "CPU"/"GPU"
device = select_device('CPU')

# Load the model
model = DetectMultiBackend('/home/username/¬/tflite_model/ntcc/weights/yolov9-e.pt', device=device, fp16=False, data='data/coco.yaml')


stride, names, pt = model.stride, model.names, model.pt


conf_thres=0.1
iou_thres=0.45

# Initialize the PiCamera
picam2 = Picamera2()
config = picam2.create_preview_configuration({'format': 'BGR888'})
picam2.configure(config)
picam2.start()
count = 0

# Function for object detection
def detect_objects(frame):
    img = letterbox(frame, 640, stride=stride, auto=True)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    bounding_box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)
    img0 = frame

    # Inference
    pred = model(img, augment=False, visualize=False)

    # Apply NMS
    pred = non_max_suppression(pred[0][0], conf_thres, iou_thres, classes=None, max_det=1000)

    # Process detections
    for det in pred:
        det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
        
        for *xyxy, conf, cls in reversed(det):
            label = f'{names[int(cls)]} {conf:.2f}'

            # Transform detections to supervisions detections
            detections = sv.Detections(
                xyxy=torch.stack(xyxy).cpu().numpy().reshape(1, -1),
                class_id=np.array([int(cls)]),
                confidence=np.array([float(conf)])
            )
            
            # Labels
            labels = [
                f"{class_id} {confidence:0.2f}"
                for class_id, confidence
                in zip(detections.class_id, detections.confidence)
            ]
            
            img0 = bounding_box_annotator.annotate(img0, detections)
            img0 = label_annotator.annotate(img0, detections, labels)
        
        if names[int(cls)] == 'dog':
                set_frequency(25000)  # Set to 25kHz for the detected object
        
        else:
            set_frequency(0)  # Turn off frequency if detected object is not a dog

    return frame

try:
    while True:
        
        # Capture frame-by-frame
        frame = picam2.capture_array()
        count+=1

        # Perform object detection every 10 frames to reduce computation
        if count % 10 == 0:
            frame = detect_objects(frame)
            
            # Display the frame
            cv2.imshow('Object Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

finally:
    picam2.stop()
    cv2.destroyAllWindows()
    spi.close()
    GPIO.cleanup()