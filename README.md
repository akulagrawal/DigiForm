*﻿Digiform is an application made with an objective to eliminate manual data entry and convert paper forms to digital*


# Workflow
## Android App:
Upload the image of the form to the Server

## Server:
1. Process the image
2. Initial calibration and create bounding boxes around labels and data fields
3. Extract coordinates
4. Feed the image in the OCR

## OCR:
1. Select image of one character from the coordinates of the bounding box
2. Text recognition using CNN
3. Send the output to the Database

## Database:
Store the converted digital forms


# Technology Used
* Android Studio
* OpenCV
* Python3
* PyTorch


# Software Requirements
## Phone
* Android Version 4.4 or higher
## PC
* Any OS with required softwares and dependencies installed(Has been tested on MS Windows 10 and Ubuntu 16.04)


# Dataset
MNIST Dataset has been used and the format has been modified according to the need.


# Contributors
* Akul Agrawal
* Deepak Kumar Gouda
* Rahul Kumar Gupta
* Yash Kothari
