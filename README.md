# Offline Hand Gesture Based Laptop Control Using Image Processing

An offline laptop control system that uses hand gestures captured through a webcam.

## Features

- Hand detection using MediaPipe
- Finger counting
- Mouse movement using one finger
- Left click using two fingers
- Scroll control using three fingers
- Next/Previous track using four-finger swipe
- Volume control
- Play/Pause media control
- ESC control using fist
- Real-time on-screen dashboard

## Technologies Used

- Python
- OpenCV
- MediaPipe
- PyAutoGUI
- NumPy

## System Workflow

Webcam → Hand Detection → Hand Landmarks → Finger Counting → Gesture Recognition → Laptop Control

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
