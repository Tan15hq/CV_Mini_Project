📌 Blink Rate & Facial Dimension Analysis (Computer Vision)
📖 Overview

This project is a Computer Vision application that analyzes video recordings to:

(A) Estimate eye blinking rate (blinks per second/minute)
(B) Estimate facial dimensions such as face size, eye size, nose size, and mouth size

The system uses MediaPipe Face Landmarker (deep learning-based model) along with OpenCV to process video frames and extract meaningful physiological and geometric information.

🚀 Features
Detects blink rate using Eye Aspect Ratio (EAR)
Extracts 468 facial landmarks
Computes:
Face height & width
Eye dimensions
Nose dimensions
Mouth dimensions
Converts pixel measurements to real-world units (cm)
Supports multiple videos
Outputs results in:
.csv
.xlsx
🧠 Tech Stack
Python 3.10
OpenCV
MediaPipe (Tasks API)
NumPy
Pandas
OpenPyXL
📂 Project Structure
project/
│
├── main.py
├── videos/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
├── face_analysis_results.csv
├── face_analysis_results.xlsx
⚙️ Installation

Install dependencies:

pip install mediapipe opencv-python numpy pandas openpyxl
▶️ How to Run
Place all your videos inside a folder (e.g., videos/)
Update this line in main.py:
VIDEO_FOLDER = "videos"
Run the script:
python main.py
📊 Output

The program generates:

face_analysis_results.csv
face_analysis_results.xlsx

These files contain:

Blink count
Blink rate (per sec & per min)
Facial measurements (cm & pixels)
Average values across all videos
📈 Sample Results
Average Blink Rate: ~15 blinks/min
Face Dimensions:
Height: ~23 cm
Width: ~21 cm
⚠️ Limitations
Uses an assumed face height (23 cm) for scaling
No camera calibration → measurements are approximate
Performance depends on:
lighting
face visibility
camera angle
🔮 Future Improvements
Camera calibration for accurate measurements
Adaptive blink detection threshold
Improved landmark selection
Real-time processing
Integration with advanced deep learning models
🎤 Viva Explanation (Quick)

The system uses MediaPipe Face Landmarker to extract facial landmarks, computes Eye Aspect Ratio for blink detection, and estimates facial dimensions by converting pixel distances into real-world units using a reference scale.

👤 Author

Tanishq Reddy