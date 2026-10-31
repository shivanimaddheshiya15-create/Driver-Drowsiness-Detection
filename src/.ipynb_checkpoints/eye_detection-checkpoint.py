import numpy as np


def calculate_ear(landmarks, eye_indices, width, height):
    """
    Calculate Eye Aspect Ratio (EAR) using MediaPipe landmarks.
    """

    points = []

    for index in eye_indices:
        x = landmarks[index].x * width
        y = landmarks[index].y * height
        points.append((x, y))

    points = np.array(points)

    vertical_1 = np.linalg.norm(points[1] - points[5])
    vertical_2 = np.linalg.norm(points[2] - points[4])
    horizontal = np.linalg.norm(points[0] - points[3])

    if horizontal == 0:
        return 0.0

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)

    return float(ear)


def get_eye_status(ear, threshold=0.21):
    """
    Return whether the eyes are Open or Closed.
    """

    if ear < threshold:
        return "Closed"

    return "Open"