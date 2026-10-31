import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FACE_MODEL_PATH = os.path.join(BASE_DIR, "face_landmarker.task")
RF_MODEL_PATH = os.path.join(BASE_DIR, "models", "drowsiness_rf.joblib")

EAR_THRESHOLD = 0.21
CLOSED_SECONDS = 2.0

YAWN_THRESHOLD = 0.35
YAWN_SECONDS = 1.0

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]