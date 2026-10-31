import mediapipe as mp
import os


def create_face_landmarker(model_path):
    """
    Initialize MediaPipe FaceLandmarker.
    """

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Face landmarker model not found: {model_path}"
        )

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=model_path
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1
    )

    return mp.tasks.vision.FaceLandmarker.create_from_options(
        options
    )