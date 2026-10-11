"""Vision: face detection adapters, tracking, target selection and active-speaker detection.

``tracker`` (T-025) is pure numpy; ``faces`` (T-027) wraps MediaPipe behind
``FaceBackend`` with an offline ``MockBackend`` and imports MediaPipe lazily.
Neither is an audio ``Stage``; they are deliberately not registered in
:mod:`hearing.registry`. Face names resolve lazily so that
``python -m hearing.vision.faces`` does not import (and register) it twice.
"""

from hearing.vision.tracker import Detection, IouTracker, iou_matrix

_FACES = (
    "LIP_INDICES",
    "FaceBackend",
    "FacePipeline",
    "MediaPipeBackend",
    "MockBackend",
    "get_backend",
    "lip_box",
    "mouth_open",
)

__all__ = [
    "LIP_INDICES",
    "Detection",
    "FaceBackend",
    "FacePipeline",
    "IouTracker",
    "MediaPipeBackend",
    "MockBackend",
    "get_backend",
    "iou_matrix",
    "lip_box",
    "mouth_open",
]


def __getattr__(name: str):
    if name in _FACES:
        from hearing.vision import faces

        return getattr(faces, name)
    raise AttributeError(f"module 'hearing.vision' has no attribute {name!r}")
