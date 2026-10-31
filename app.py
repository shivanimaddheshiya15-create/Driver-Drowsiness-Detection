import os
import time
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib

from src.config import (
    FACE_MODEL_PATH,
    RF_MODEL_PATH,
    EAR_THRESHOLD,
    CLOSED_SECONDS,
    YAWN_THRESHOLD,
    YAWN_SECONDS,
    LEFT_EYE,
    RIGHT_EYE,
)
from src.eye_detection import calculate_ear
from src.face_detection import create_face_landmarker
from src.yawn_detection import calculate_mar
from src.alarm import trigger_alarm


# ==========================================
# PAGE + OPTIONAL MODEL
# ==========================================
st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="🚗",
    layout="wide",
)

rf_model = None
if os.path.isfile(RF_MODEL_PATH):
    try:
        rf_model = joblib.load(RF_MODEL_PATH)
    except Exception as exc:
        print(f"Optional Random Forest model could not be loaded: {exc}")

CLASS_NAMES = {0: "Closed", 1: "Open", 2: "Yawn", 3: "No Yawn"}


# ==========================================
# THEME
# ==========================================
if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

st.sidebar.title("⚙️ Settings")
st.session_state.theme = st.sidebar.radio(
    "🎨 Choose Dashboard Theme",
    ["Dark", "Light"],
    index=["Dark", "Light"].index(st.session_state.theme),
)

if st.session_state.theme == "Dark":
    bg_color, text_color, card_color, border_color = (
        "#0E1117", "#FAFAFA", "#1E293B", "#38BDF8"
    )
else:
    bg_color, text_color, card_color, border_color = (
        "#F8FAFC", "#1E293B", "#FFFFFF", "#0284C7"
    )

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {bg_color}; color: {text_color}; }}
    [data-testid="stSidebar"] {{ background-color: {card_color}; }}
    [data-testid="stMetric"] {{
        background-color: {card_color};
        padding: 15px;
        border-radius: 12px;
        border: 1px solid {border_color};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🚗 Driver Drowsiness Detection System")
st.markdown(
    "### Real-Time Driver Safety Monitoring\n"
    "Monitor eye closure, yawning, and signs of drowsiness using computer vision."
)
st.info(
    "Educational prototype only. Do not use this app as the sole safety system "
    "while driving."
)
st.divider()


# ==========================================
# DASHBOARD
# ==========================================

# ==========================================
st.subheader("📊 Live Driver Monitoring Dashboard")
col1, col2, col3 = st.columns(3)
with col1:
    eye_metric = st.empty()
with col2:
    yawn_metric = st.empty()
with col3:
    driver_metric = st.empty()

eye_metric.metric("👁️ Eye Status", "Waiting...")
yawn_metric.metric("🥱 Total Yawns", 0)
driver_metric.metric("🚗 Driver Status", "Ready")

m1, m2, m3 = st.columns(3)
m1.metric("EAR Threshold", EAR_THRESHOLD)
m2.metric("Alert Duration", f"{CLOSED_SECONDS:g} seconds")
m3.metric("Detection", "Live Webcam")
st.divider()


# ==========================================
# CONTROLS
# ==========================================
if "running" not in st.session_state:
    st.session_state.running = False

start_col, stop_col = st.columns(2)
with start_col:
    start_clicked = st.button(
        "▶️ Start Detection",
        type="primary",
        disabled=st.session_state.running,
        use_container_width=True,
    )
with stop_col:
    stop_clicked = st.button(
        "⏹️ Stop / Reset",
        disabled=not st.session_state.running,
        use_container_width=True,
    )

if start_clicked:
    st.session_state.running = True
if stop_clicked:
    st.session_state.running = False
    st.rerun()

frame_placeholder = st.empty()
status_placeholder = st.empty()
ear_placeholder = st.empty()

if not os.path.isfile(FACE_MODEL_PATH):
    st.error(f"MediaPipe face model not found: {FACE_MODEL_PATH}")
    st.stop()

if st.session_state.running:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap.release()
        st.session_state.running = False
        st.error("Webcam open nahi hua. Camera permission aur camera index check karo.")
        st.stop()

    closed_start = None
    yawn_start = None
    yawn_count = 0
    yawn_latched = False
    last_timestamp = 0
    alarm_latched = False

    try:
        with create_face_landmarker(FACE_MODEL_PATH) as landmarker:
            while st.session_state.running:
                success, frame = cap.read()
                if not success:
                    status_placeholder.error("Camera frame nahi mila.")
                    break

                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame,
                )

                timestamp = max(int(time.time() * 1000), last_timestamp + 1)
                last_timestamp = timestamp
                result = landmarker.detect_for_video(mp_image, timestamp)

                status = "No face detected"
                eye_status = "Unknown"
                color = (0, 200, 255)
                ear = None
                mar = 0.0
                alert_now = False

                if result.face_landmarks:
                    landmarks = result.face_landmarks[0]
                    left_ear = calculate_ear(landmarks, LEFT_EYE, width, height)
                    right_ear = calculate_ear(landmarks, RIGHT_EYE, width, height)
                    ear = (left_ear + right_ear) / 2.0
                    mar = calculate_mar(landmarks, width, height)

                    if ear < EAR_THRESHOLD:
                        eye_status = "Closed"
                        if closed_start is None:
                            closed_start = time.time()
                        elapsed = time.time() - closed_start
                        if elapsed >= CLOSED_SECONDS:
                            status = "DROWSINESS ALERT!"
                            color = (0, 0, 255)
                            if not alarm_latched:
                                trigger_alarm()
                                alarm_latched = True
                        else:
                            status = "Eyes closing..."
                            color = (0, 165, 255)
                            alarm_latched = False
                    else:
                        closed_start = None
                        alarm_latched = False
                        eye_status = "Open"
                        status = "Awake"
                        color = (0, 200, 0)

                    if mar > YAWN_THRESHOLD:
                        if yawn_start is None:
                            yawn_start = time.time()
                        if time.time() - yawn_start >= YAWN_SECONDS:
                            if not yawn_latched:
                                yawn_count += 1
                                yawn_latched = True
                    else:
                        yawn_start = None
                        yawn_latched = False

                    for idx in LEFT_EYE + RIGHT_EYE:
                        px = int(landmarks[idx].x * width)
                        py = int(landmarks[idx].y * height)
                        cv2.circle(frame, (px, py), 2, (0, 255, 255), -1)

                    if rf_model is not None:
                        # Optional prediction only; EAR/MAR remain the main signals.
                        try:
                            xs = [int(p.x * width) for p in landmarks]
                            ys = [int(p.y * height) for p in landmarks]
                            x1, x2 = max(0, min(xs)), min(width, max(xs))
                            y1, y2 = max(0, min(ys)), min(height, max(ys))
                            face_crop = frame[y1:y2, x1:x2]
                            if face_crop.size:
                                gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                                resized = cv2.resize(gray, (64, 64))
                                features = resized.flatten().reshape(1, -1)
                                if hasattr(rf_model, "n_features_in_") and (
                                    rf_model.n_features_in_ == features.shape[1]
                                ):
                                    prediction = int(rf_model.predict(features)[0])
                                    label = CLASS_NAMES.get(prediction, str(prediction))
                                    cv2.putText(
                                        frame, f"RF: {label}", (20, 180),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                        (255, 255, 0), 2,
                                    )
                        except Exception as exc:
                            print(f"Optional RF prediction skipped: {exc}")

                    cv2.putText(
                        frame, f"EAR: {ear:.3f}  MAR: {mar:.3f}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2,
                    )
                    cv2.putText(
                        frame, f"Total Yawns: {yawn_count}",
                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255), 2,
                    )
                    if yawn_latched:
                        cv2.putText(
                            frame, "YAWNING DETECTED!", (20, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
                        )

                cv2.putText(
                    frame, status, (20, height - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                )

                eye_metric.metric("👁️ Eye Status", eye_status)
                yawn_metric.metric("🥱 Total Yawns", yawn_count)
                driver_metric.metric("🚗 Driver Status", status)
                ear_placeholder.metric(
                    "Current EAR", f"{ear:.3f}" if ear is not None else "N/A"
                )
                frame_placeholder.image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                )
                if status == "DROWSINESS ALERT!":
                    status_placeholder.error("🚨 Drowsiness detected! Please take a break.")
                elif status == "Awake":
                    status_placeholder.success("✅ Driver appears awake.")
                else:
                    status_placeholder.warning(status)

                time.sleep(0.03)

    except Exception as exc:
        st.error(f"Detection error: {exc}")
    finally:
        cap.release()
        st.session_state.running = False
        cv2.destroyAllWindows()
else:
    status_placeholder.info("Click **Start Detection** to begin webcam monitoring.")