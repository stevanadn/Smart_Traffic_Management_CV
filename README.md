# Traffic Violation & Smart Traffic Light System 🚦

This repository contains a Computer Vision system implementation for detecting traffic violations (specifically helmet usage) and simulating a smart traffic light management system based on vehicle density. This project utilizes a custom-trained **YOLOv11** object detection model to identify vehicles, people, and helmet usage in real-time from an image.

## ✨ Key Features

* **Multi-Class Vehicle Detection:** Capable of identifying and distinguishing between motorcycles, cars, buses, and trucks.
* **Helmet Violation Detection:** Automatically identifies motorcyclists and checks whether they are wearing a helmet.
* **Rider-Motorcycle Association:** Utilizes *Intersection over Union* (IoU) to intelligently link a person with the nearest motorcycle for helmet analysis.
* **Traffic Light Duration Simulation:** Counts the total number of vehicles in the frame to recommend a dynamic green light duration, complete with minimum and maximum thresholds.
* **Clear Visualizations:** Provides visual output with clear bounding boxes and informational labels:
  * 🔴 **Violator (No Helmet):** Marked in red.
  * 🟢 **Safe Rider (With Helmet):** Marked in green.
  * 🔵/🟠 **Other Vehicles:** Marked in blue and orange.
  * ⏱️ **Traffic Light Information:** Displays the vehicle count and estimated light duration in the top right corner.

## 🛠️ Technologies Used

* **Python 3.12**
* **Ultralytics YOLOv11:** The main engine for object detection.
* **OpenCV:** For visual processing tasks (reading, writing, and drawing on images).
* **NumPy:** For efficient numerical operations, especially for matrix and IoU calculations.
* **Matplotlib:** For displaying the final results (*plotting*) within a notebook environment.
