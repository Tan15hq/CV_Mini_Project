"""
Video Analysis: Blink Rate + Facial Dimension Estimator
========================================================
Uses MediaPipe Tasks API (mediapipe >= 0.10.30)
Videos: GX010911.MP4 → GX010939.MP4

Requirements:
    pip3 install mediapipe opencv-python numpy pandas openpyxl requests
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
import numpy as np
import pandas as pd
from pathlib import Path
import urllib.request
import time

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
VIDEO_FOLDER            = "Tanishq_Subject6"
OUTPUT_FILE             = "face_analysis_results"
SAMPLE_EVERY_N_FRAMES   = 3
EAR_BLINK_THRESHOLD     = 0.21
EAR_CONSEC_FRAMES       = 2
REAL_FACE_HEIGHT_CM     = 23.0
MODEL_PATH              = "face_landmarker.task"
MODEL_URL               = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
# ─────────────────────────────────────────────────────────────

# MediaPipe FaceMesh landmark indices (same indices work with Tasks API)
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

FACE_TOP    = 10;  FACE_BOTTOM = 152
FACE_LEFT   = 234; FACE_RIGHT  = 454
NOSE_TOP    = 168; NOSE_BOTTOM = 2
NOSE_LEFT   = 129; NOSE_RIGHT  = 358
MOUTH_LEFT  = 61;  MOUTH_RIGHT = 291
MOUTH_TOP   = 13;  MOUTH_BOTTOM= 14
L_EYE_L = 33;  L_EYE_R = 133
R_EYE_L = 362; R_EYE_R = 263


def download_model():
    if not Path(MODEL_PATH).exists():
        print(f"  Downloading face landmarker model (~30 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"  Model saved to {MODEL_PATH}")
    else:
        print(f"  Model already exists: {MODEL_PATH}")


def calc_ear(lm, eye_idx, w, h):
    pts = [(lm[i].x * w, lm[i].y * h) for i in eye_idx]
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (A + B) / (2.0 * C) if C > 0 else 0


def pdist(lm, i1, i2, w, h):
    p1 = np.array([lm[i1].x * w, lm[i1].y * h])
    p2 = np.array([lm[i2].x * w, lm[i2].y * h])
    return float(np.linalg.norm(p1 - p2))


def analyze_video(vpath: Path, landmarker):
    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        print(f"  ⚠  Cannot open: {vpath.name}")
        return None

    fps      = cap.get(cv2.CAP_PROP_FPS) or 30
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = n_frames / fps

    blinks     = 0
    ear_consec = 0
    in_blink   = False
    processed  = 0

    acc = {k: [] for k in [
        "face_h","face_w",
        "leye_h","leye_w","reye_h","reye_w",
        "nose_h","nose_w","mouth_h","mouth_w",
    ]}

    fidx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fidx += 1
        if fidx % SAMPLE_EVERY_N_FRAMES != 0:
            continue

        h, w = frame.shape[:2]
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = landmarker.detect(mp_image)
        if not result.face_landmarks:
            continue

        lm = result.face_landmarks[0]
        processed += 1

        # ── Blink ────────────────────────────────────────────
        avg_ear = (calc_ear(lm, LEFT_EYE, w, h) + calc_ear(lm, RIGHT_EYE, w, h)) / 2
        if avg_ear < EAR_BLINK_THRESHOLD:
            ear_consec += 1
            in_blink = True
        else:
            if in_blink and ear_consec >= EAR_CONSEC_FRAMES:
                blinks += 1
            ear_consec = 0
            in_blink   = False

        # ── Dimensions ───────────────────────────────────────
        acc["face_h"].append(pdist(lm, FACE_TOP,    FACE_BOTTOM, w, h))
        acc["face_w"].append(pdist(lm, FACE_LEFT,   FACE_RIGHT,  w, h))

        le = [(lm[i].x*w, lm[i].y*h) for i in LEFT_EYE]
        re = [(lm[i].x*w, lm[i].y*h) for i in RIGHT_EYE]
        acc["leye_h"].append((np.linalg.norm(np.array(le[1])-np.array(le[5]))+
                              np.linalg.norm(np.array(le[2])-np.array(le[4])))/2)
        acc["leye_w"].append(pdist(lm, L_EYE_L, L_EYE_R, w, h))
        acc["reye_h"].append((np.linalg.norm(np.array(re[1])-np.array(re[5]))+
                              np.linalg.norm(np.array(re[2])-np.array(re[4])))/2)
        acc["reye_w"].append(pdist(lm, R_EYE_L, R_EYE_R, w, h))

        acc["nose_h"].append(pdist(lm, NOSE_TOP,     NOSE_BOTTOM,  w, h))
        acc["nose_w"].append(pdist(lm, NOSE_LEFT,    NOSE_RIGHT,   w, h))
        acc["mouth_h"].append(pdist(lm, MOUTH_TOP,   MOUTH_BOTTOM, w, h))
        acc["mouth_w"].append(pdist(lm, MOUTH_LEFT,  MOUTH_RIGHT,  w, h))

    cap.release()

    if processed == 0:
        print(f"  ⚠  No face detected: {vpath.name}")
        return None

    avg_face_h_px = float(np.mean(acc["face_h"]))
    px_per_cm     = avg_face_h_px / REAL_FACE_HEIGHT_CM if avg_face_h_px > 0 else 1

    def cm(key): return round(float(np.mean(acc[key])) / px_per_cm, 2)
    def px(key): return round(float(np.mean(acc[key])), 1)

    bps = round(blinks / duration, 4) if duration > 0 else 0
    bpm = round(bps * 60, 2)

    row = {
        "video":              vpath.name,
        "duration_sec":       round(duration, 1),
        "frames_processed":   processed,
        "blink_count":        blinks,
        "blinks_per_sec":     bps,
        "blinks_per_min":     bpm,
        "face_height_cm":     cm("face_h"),
        "face_width_cm":      cm("face_w"),
        "left_eye_height_cm": cm("leye_h"),
        "left_eye_width_cm":  cm("leye_w"),
        "right_eye_height_cm":cm("reye_h"),
        "right_eye_width_cm": cm("reye_w"),
        "nose_height_cm":     cm("nose_h"),
        "nose_width_cm":      cm("nose_w"),
        "mouth_height_cm":    cm("mouth_h"),
        "mouth_width_cm":     cm("mouth_w"),
        "face_height_px":     px("face_h"),
        "face_width_px":      px("face_w"),
        "px_per_cm":          round(px_per_cm, 2),
    }

    print(f"  ✓  {vpath.name:20s}  "
          f"{blinks:3d} blinks / {duration:.0f}s  "
          f"= {bps:.4f} blinks/sec  ({bpm:.1f}/min)")
    return row


def main():
    folder = Path(VIDEO_FOLDER).resolve()
    exts   = {".mp4", ".MP4", ".avi", ".AVI", ".mov", ".MOV", ".mkv", ".MKV"}
    videos = sorted(p for p in folder.iterdir() if p.suffix in exts)

    if not videos:
        print(f"No video files found in: {folder}")
        return

    print(f"\n{'='*65}")
    print(f"  cv.miniproject  —  {len(videos)} video(s) found")
    print(f"  {folder}")
    print(f"{'='*65}\n")

    # Download model if needed
    download_model()

    # Build landmarker (IMAGE mode = synchronous, one frame at a time)
    options = FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    rows = []
    t0   = time.time()

    with FaceLandmarker.create_from_options(options) as landmarker:
        for i, vp in enumerate(videos, 1):
            print(f"[{i:02d}/{len(videos)}] {vp.name}")
            row = analyze_video(vp, landmarker)
            if row:
                rows.append(row)
            print()

    if not rows:
        print("No results — no faces detected in any video.")
        return

    df = pd.DataFrame(rows)
    num_cols = df.select_dtypes(include="number").columns
    avg_row  = df[num_cols].mean().round(4).to_dict()
    avg_row["video"] = "── AVERAGE ──"
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

    csv_out  = folder / f"{OUTPUT_FILE}.csv"
    xlsx_out = folder / f"{OUTPUT_FILE}.xlsx"
    df.to_csv(csv_out,  index=False)
    df.to_excel(xlsx_out, index=False)

    elapsed = time.time() - t0
    print(f"{'='*65}")
    print(f"  Finished in {elapsed/60:.1f} min")
    print(f"  CSV  → {csv_out}")
    print(f"  XLSX → {xlsx_out}")
    print(f"{'='*65}\n")

    avg = df[df["video"] == "── AVERAGE ──"].iloc[0]
    print("📊  AVERAGES ACROSS ALL VIDEOS")
    print(f"\n  (A) Blink Rate")
    print(f"      {avg['blinks_per_sec']:.4f} blinks/sec   →   {avg['blinks_per_min']:.1f} blinks/min")
    print(f"\n  (B) Facial Dimensions  (face height reference = {REAL_FACE_HEIGHT_CM} cm)")
    print(f"      Face      {avg['face_height_cm']:5.2f} cm tall  x  {avg['face_width_cm']:5.2f} cm wide")
    print(f"      Left Eye  {avg['left_eye_height_cm']:5.2f} cm tall  x  {avg['left_eye_width_cm']:5.2f} cm wide")
    print(f"      Right Eye {avg['right_eye_height_cm']:5.2f} cm tall  x  {avg['right_eye_width_cm']:5.2f} cm wide")
    print(f"      Nose      {avg['nose_height_cm']:5.2f} cm tall  x  {avg['nose_width_cm']:5.2f} cm wide")
    print(f"      Mouth     {avg['mouth_height_cm']:5.2f} cm tall  x  {avg['mouth_width_cm']:5.2f} cm wide")
    print()


if __name__ == "__main__":
    main()
