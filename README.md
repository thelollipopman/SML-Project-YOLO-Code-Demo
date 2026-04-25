# 🎯 Meme Pose Detector

A real-time pose detection application built using YOLO pose estimation.  
This project detects fun “meme poses” (e.g. **dab**, **T-pose**, and “6-7” motion**) from a webcam feed and overlays animated GIF effects when detected.

---

## 🚀 Features

- Real-time pose detection using YOLO
- Detects:
  - Dab pose
  - T-pose
  - “6-7” alternating arm motion
- GIF overlays triggered on detection
- Cooldown system to prevent repeated triggering
- Modular design for adding new poses easily

---

## 📦 Installation

1. Clone the repository or download the files.

2. (Optional but recommended) Create a virtual environment:

python -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows

3. Install dependencies from `requirements.txt`:

pip install -r requirements.txt

---

## ▶️ Usage

Run the main script:

python pose_detection.py

- Press **`q`** to quit the application.
- Make sure your webcam is enabled.

---

## 📁 Project Structure

.
├── pose_detection.py      # Main application
├── requirements.txt       # Python dependencies
└── gifs/
    ├── dab.gif
    ├── tpose.gif
    └── 67.gif

---

## ⚙️ Requirements

- Python 3.9+
- Webcam
- GPU (optional but recommended for better performance)

---

## 🧠 How It Works

- Uses YOLO pose model (`yolo26n-pose.pt`) to extract keypoints
- Detects poses using geometric rules (angles, distances, relative positions)
- Maintains a short history buffer for motion-based detection (e.g. “6-7”)
- Triggers visual effects with cooldown control

---

## 🛠️ Customisation

- Add new poses in the **pose registry**
- Adjust thresholds (angles, distances) in detection functions
- Add new GIFs in the `gifs/` folder and register them

---

## ⚠️ Notes

- Ensure `yolo26n-pose.pt` is available or downloadable by Ultralytics
- Lighting and camera angle affect detection accuracy
- Performance depends on your hardware
