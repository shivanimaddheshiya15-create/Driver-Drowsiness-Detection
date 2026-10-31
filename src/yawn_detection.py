import time
import numpy as np


def calculate_mar(landmarks, width, height):
    upper = np.array([
        landmarks[13].x * width,
        landmarks[13].y * height
    ])

    lower = np.array([
        landmarks[14].x * width,
        landmarks[14].y * height
    ])

    left = np.array([
        landmarks[78].x * width,
        landmarks[78].y * height
    ])

    right = np.array([
        landmarks[308].x * width,
        landmarks[308].y * height
    ])

    mouth_height = np.linalg.norm(upper - lower)
    mouth_width = np.linalg.norm(left - right)

    if mouth_width == 0:
        return 0.0

    return float(mouth_height / mouth_width)


def update_yawn_state(
    mar,
    yawn_start,
    threshold=0.35,
    duration=1.0
):
    if mar > threshold:
        if yawn_start is None:
            yawn_start = time.time()

        if time.time() - yawn_start >= duration:
            return True, yawn_start

        return False, yawn_start

    return False, None