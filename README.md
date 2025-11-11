# YOLOv8-Face-Blur
Real-Time Face Detection and Blur Application

This project automatically detects faces in videos using the Ultralytics YOLOv8 model and applies Gaussian blur to the detected areas. The goal is to anonymize videos while preserving individual privacy.

🚀 Features

🎥 Processes the video file (input.mp4) frame by frame.

🤖 Performs face detection with the YOLOv8 model.

🌀 Anonymizes detected faces with a high-order Gaussian blur.

⏹ Plays the video in real time (press q to exit).

💻 Can run on GPU (CUDA) or CPU.


🧩 Requirements

The project is compatible with Python 3.8+. You need to install the following libraries:

pip install ultralytics opencv-python

🧠 Usage

Download the YOLOv8 model (it will download automatically on the first run). Place the input.mp4 file in the project directory. Run the code:

python face_blur.py

Press q to exit.


📁 File Structure

📦 yolo-face-blur

┣ 📜 face_blur.py # Main code file

┣ 📜 README.md # Project description

┗ 🎞 input.mp4 # Video to be processed (example)
 
 ⚙️ Working Logic

For each frame, the YOLO model detects faces.
cv2.GaussianBlur is applied to each detection area (ROI).
The processed video frames are displayed on the screen.

🧑‍💻 Developer

Erkan Deveci
📍 GitHub: @cengerkan
