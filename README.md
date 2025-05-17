# VLC-media-player-gesture-control

Real-time sign language recognition using computer vision and a trained machine learning model.

## Overview

This project aims to recognize hand gestures in real-time and associate them with VLC media player controls. Users can control VLC media player functions such as play/pause, volume up/down, forward, and backward using specific hand gestures.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
   - [Data Collection](#data-collection)
   - [Model Training](#model-training)
   - [Real-Time Detection](#real-time-detection)
3. [File Structure](#file-structure)
4. [Data Collection and Model Training](#data-collection-and-model-training)
5. [Real-Time Detection](#real-time-detection)
6. [VLC Media Player Control](#vlc-media-player-control)
7. [Predictions Photo](#Predictions-Photo)
8. [Future Improvements](#future-improvements)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/sign-language-recognition.git

2. Install Dependencies

   ```bash
   pip install -r requirements.txt

## Usage 

# Data Collection:

Run the data collection script to gather images for training:

   ```bash
   python data_collection.py
   ```

# Model Training:

Execute the model training script:
   ```bash
   python model_training.py
   ```
The trained model will be saved as hand_gesture_model.keras.

# Real-Time Detection:

Run the real-time detection script to identify hand gestures using your webcam:
   ```bash
   python real_time_detection.py
   ```
   
The trained model will be used to perform real-time hand gesture detection.

## File Structure
data_collection.py: Script for collecting hand gesture images.
model_training.py: Script for training the machine learning model.
real_time_detection.py: Script for real-time hand gesture detection.

## Data Collection and Model Training
The data_collection.py script captures hand gestures and saves images to train the model. The model_training.py script loads and preprocesses the collected data, trains a convolutional neural network (CNN), and saves the trained model.

## Real-Time Detection
The real_time_detection.py script utilizes the trained model to perform real-time hand gesture detection using your webcam.

## VLC Media Player Control
The hand gestures correspond to VLC media player controls:

1. Play/Pause: Spacebar
2. Volume Up: Ctrl + Up Arrow
3. Volume Down: Ctrl + Down Arrow
4. Forward: Right Arrow
5. Backward: Left Arrow


## Predictions Photo

<div align="center">
  <img src="https://github.com/paramsureliya/VLC-media-player-gesture-control/blob/main/play_pause.png" alt="Play/Pause" width="400" height="300">
  <br>
  <img src="https://github.com/paramsureliya/VLC-media-player-gesture-control/blob/main/mesh_diagram.png" alt="Mesh Diagram" width="400" height="300">
  <br>
   <img src="https://github.com/paramsureliya/VLC-media-player-gesture-control/blob/main/volume_up.png" alt="volume_up" width="400" height="300">
</div>


## Future Improvements

- Implement a more robust hand tracking algorithm.
- Expand the gesture set for additional controls.
- Feel free to contribute and improve this project!
- Deploy it in real time using.






