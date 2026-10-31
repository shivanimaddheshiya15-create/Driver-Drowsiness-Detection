import cv2
import numpy as np


def preprocess_face(frame, size=(64, 64)):
    """
    Convert a face/frame to grayscale and resize it
    for Random Forest model prediction.
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    resized = cv2.resize(gray, size)

    features = resized.flatten().reshape(1, -1)

    return features


def extract_face_landmarks(landmarks, width, height):
    """
    Convert normalized MediaPipe landmarks
    into pixel coordinates.
    """

    points = []

    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)

        points.append((x, y))

    return np.array(points)