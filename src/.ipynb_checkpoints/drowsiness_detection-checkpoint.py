import time


def check_drowsiness(ear, threshold=0.21):
    """
    Check whether the driver's eyes are closed.

    Parameters:
        ear: Eye Aspect Ratio
        threshold: EAR threshold for closed eyes

    Returns:
        True if eyes are closed, otherwise False
    """
    return ear < threshold


def check_drowsiness_duration(ear, closed_start, threshold=0.21, duration=2.0):
    """
    Detect prolonged eye closure.

    Returns:
        status, updated closed_start
    """
    if ear < threshold:
        if closed_start is None:
            closed_start = time.time()

        elapsed = time.time() - closed_start

        if elapsed >= duration:
            return "DROWSINESS ALERT!", closed_start

        return "Eyes closing...", closed_start

    return "Awake", None