# ME130 Human Robot Interaction Project

This repository contains all the code for the Human-Robot Interaction (HRI) project, including testing code, central PC processing code, and ESP-32 hardware control code. The project aims to design a robot capable of interacting with human users in the real world using computer vision, an IMU, and LiDAR. The initial concept is a delivery robot that identifies who has an item for delivery and who wants the item, delivering it to the desired person.

## File Overview

### Main Programs
- **`main.py`**
  - The primary Python program that manages the robot’s state machine and image processing.
  - Uses YOLO11 for object detection to control the robot’s behavior.

- **`HRI Platform IO/src/main.cpp`**
  - The main ESP-32 program that handles sensor and motor control.
  - Communicates with the Python program over MQTT to initiate state changes and update robot behavior.

### Development Programs
Located in the `development` directory:
- **`cv_testing.py`**
  - A program for developing and testing the computer vision algorithms used in `main.py`.
  - Includes image processing and annotations for navigation and object detection.

- **`event_sim_gui.py`**
  - Launches a simple GUI with buttons to simulate state change events for testing `main.py`.
  - Includes a readout for MQTT topics `bot/sensors` and `bot/state` to monitor messages and formatting.

- **`motor_testing.py`**
  - A program for testing robot motor control over MQTT.
  - Evaluates robot steering and motor responsiveness.

### Supporting Files and Models
- **`yolo_models/`**
  - Contains several pretrained PyTorch models for testing.
  - The final program uses `yolo11s.pt`, but `yolo11n.pt` offers similar functionality.

- **`classes.py`**
  - A dictionary mapping YOLO11 pretrained dataset class names to their indices in the results object.

This repository serves as a comprehensive resource for developing, testing, and deploying the HRI robot system.

