import threading
import winsound


def play_alarm():
    """Play a short alarm sound on Windows."""
    try:
        winsound.Beep(1000, 500)
    except RuntimeError:
        print("Alarm could not be played.")


def trigger_alarm():
    """Play the alarm without blocking the webcam loop."""
    thread = threading.Thread(
        target=play_alarm,
        daemon=True
    )
    thread.start()