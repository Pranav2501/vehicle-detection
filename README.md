# Project Title
Final Project - Comparison of Vehicle Detection Techniques

## Team Members
- Pranav Raj Sowrirajan Balaji

## Description

This project compares the results & efficiency of vehicle detection between classic computer vision techniques & deep-learning models. 

Classic Computer Vision Techniques: involves using the following steps
- Frame Differencing
- Image Thresholding
- Contour Detection
- Image Dilation

Deep Learning based models: YOLOv4 & YOLOv8

YOLOv4 (Pre-trained model)
- Model specification: yolov4-tiny
- Dataset : COCO dataset (pre-trained)
- Link : https://pjreddie.com/darknet/yolo/

YOLOv8 (Pre-trained model)
- Model specification: yolov8-seg
- Dataset: COCO8 dataset (pre-trained)
- Link: https://docs.ultralytics.com/quickstart/

## Executing Models
The YOLOv4 model is available in Darknet website, and YOLOv8 is available in the Ultralytics website. You will either to need to clone/install these repositories to use the model, or download the specifics files such as weights, model config files etc.

Here's a brief overview on how to use YOLOv4:
- Clone the repository: git clone https://github.com/pjreddie/darknet
- Move to the directory: cd darknet
- If using Cmake files: make
- To get the weights file for the model: wget https://pjreddie.com/media/files/yolov4.weights
- To detect an image using command-line: ./darknet detect cfg/yolov4.cfg yolov4.weights data/dog.jpg
- To train the model: ./darknet detector train cfg/voc.data cfg/yolov4-voc.cfg darknet53.conv.74

To use YOLOv8:
- Install ultralytics: pip install ultralytics
- Or clone from Github: pip install git+https://github.com/ultralytics/ultralytics.git@main
- Make sure you have Pytorch Installed
- Train from command-line: yolo train data=coco8.yaml model=yolov8n-seg.pt epochs=10 lr0=0.01
- Predict from command-line: yolo predict model=yolov8n-seg.pt source='https://youtu.be/LNwODJXcvt4' imgsz=320
- Validate: yolo val model=yolov8n.pt data=coco8.yaml batch=1 imgsz=640



If you are having trouble, please visit the websites to learn more on these processes.

## Demo Videos

To view  the recorded videos of live detection, please use this google drive link:
https://drive.google.com/drive/folders/1Jg5dKVrYQ-reMU0UzK7MLv5__lKKVOG3?usp=sharing


## Presentation Video
- https://northeastern.zoom.us/rec/share/VZZsVCClNT41McoUNZL4Ka-BbdSBn9AkmEXOGld9ev7aVVTtwrMTCUbCL676GJ_h.6AKRtITpmawsUNNe 
- Passcode: rnU1AV+K

## Files
- COCO Dataset: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml.
- COCO8 Dataset: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8.yaml.
- Google Drive Link for demo, presentation & code: 
https://drive.google.com/drive/folders/1BECx8uNyOeHJ2UW6AGxQlgV9Pv0PsomM?usp=drive_link


## Development Environment
- Operating System: Mac OS Ventura
- IDE: Visual Studio Code 
- Compiler: Cmake files with VS Code
- CPU: Apple M1 Chip with integrated GPU
- RAM: 16GB

## Time Travel Days
- Number of time travel days used: NIL 

## How to Run the Programs

### Prerequisites

Ensure you have OpenCV installed on your system.
The following codes are implemented using Cmake Files with VS Code.

Run the following commands:
- 1. mkdir build
- 2. cd build
- 3. cmake ..
- 4. make YourPythonScriptName

