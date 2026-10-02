"""Keypoints from extract-keypoint-vslfullfront.ipynb.

34 body points (33 pose plus the shoulder midpoint as neck) and 42 hand
points, each xyz. Body and hand boxes are normalized per frame, then the
clip is cut or zero-padded to 80 frames and flattened to 228 values.
"""

from __future__ import annotations

import logging
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)

N_JOINTS = 76
FEATURE_DIM = N_JOINTS * 3
MAX_FRAMES = 80
FRAME_SIZE = 224
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

BODY_LANDMARKS = [
    "nose", "leftEyeInner", "leftEye", "leftEyeOuter", "rightEyeInner", "rightEye", "rightEyeOuter",
    "leftEar", "rightEar", "mouthLeft", "mouthRight", "leftShoulder", "rightShoulder",
    "leftElbow", "rightElbow", "leftWrist", "rightWrist", "leftPinky", "rightPinky",
    "leftIndex", "rightIndex", "leftThumb", "rightThumb",
    "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle",
    "leftHeel", "rightHeel", "leftFootIndex", "rightFootIndex", "neck",
]
HAND_LANDMARKS = [
    "wrist", "indexTip", "indexDIP", "indexPIP", "indexMCP",
    "middleTip", "middleDIP", "middlePIP", "middleMCP",
    "ringTip", "ringDIP", "ringPIP", "ringMCP",
    "littleTip", "littleDIP", "littlePIP", "littleMCP",
    "thumbTip", "thumbIP", "thumbMP", "thumbCMC",
]
HANDS_LANDMARKS = [name + suffix for name in HAND_LANDMARKS for suffix in ("_0", "_1")]
LANDMARKS = BODY_LANDMARKS + HANDS_LANDMARKS
ANCHOR_LANDMARKS = ["nose", "leftShoulder", "rightShoulder", "leftHip", "rightHip", "neck"]

POSE_INDEX = {
    "nose": 0, "leftEyeInner": 1, "leftEye": 2, "leftEyeOuter": 3,
    "rightEyeInner": 4, "rightEye": 5, "rightEyeOuter": 6,
    "leftEar": 7, "rightEar": 8, "mouthLeft": 9, "mouthRight": 10,
    "leftShoulder": 11, "rightShoulder": 12, "leftElbow": 13, "rightElbow": 14,
    "leftWrist": 15, "rightWrist": 16, "leftPinky": 17, "rightPinky": 18,
    "leftIndex": 19, "rightIndex": 20, "leftThumb": 21, "rightThumb": 22,
    "leftHip": 23, "rightHip": 24, "leftKnee": 25, "rightKnee": 26,
    "leftAnkle": 27, "rightAnkle": 28, "leftHeel": 29, "rightHeel": 30,
    "leftFootIndex": 31, "rightFootIndex": 32,
}
HAND_INDEX = {
    "wrist": 0, "thumbCMC": 1, "thumbMP": 2, "thumbIP": 3, "thumbTip": 4,
    "indexMCP": 5, "indexPIP": 6, "indexDIP": 7, "indexTip": 8,
    "middleMCP": 9, "middlePIP": 10, "middleDIP": 11, "middleTip": 12,
    "ringMCP": 13, "ringPIP": 14, "ringDIP": 15, "ringTip": 16,
    "littleMCP": 17, "littlePIP": 18, "littleDIP": 19, "littleTip": 20,
}

_ZERO = (0.0, 0.0, 0.0)
_EXTRACTOR: Optional["HolisticExtractor"] = None


class HolisticExtractor:
    """MediaPipe Holistic. One instance per process; reset between clips."""

    def __init__(self) -> None:
        import mediapipe as mp

        if not hasattr(mp, "solutions"):
            raise ImportError(
                "mp.solutions.holistic was removed in mediapipe 0.10.21. "
                'Install the last build that still has it: pip install "mediapipe==0.10.20"'
            )
        self._cls = mp.solutions.holistic.Holistic
        self._holistic = self._cls(static_image_mode=False, model_complexity=1)

    def close(self) -> None:
        self._holistic.close()

    def reset(self) -> None:
        self.close()
        self._holistic = self._cls(static_image_mode=False, model_complexity=1)

    def process(self, frame_bgr: np.ndarray):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        return self._holistic.process(rgb)


def normalize_body(row: dict) -> dict:
    """Scale each frame's body into a box around the torso anchors."""
    sequence_size = len(row["leftEar"])
    for index in range(sequence_size):
        x_coords = [row[name][index][0] for name in ANCHOR_LANDMARKS if row[name][index][0] != 0]
        y_coords = [row[name][index][1] for name in ANCHOR_LANDMARKS if row[name][index][1] != 0]
        if not x_coords or not y_coords:
            continue
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        dx = (max_x - min_x) * 1.6
        dy = (max_y - min_y) * 1.6
        if dx <= 0 or dy <= 0:
            continue
        center_x = (max_x + min_x) / 2
        center_y = (max_y + min_y) / 2
        box_min_x = center_x - dx / 2
        box_min_y = center_y - dy / 2
        for key in BODY_LANDMARKS:
            x, y, z = row[key][index]
            if x == 0 and y == 0:
                continue
            row[key][index] = ((x - box_min_x) / dx - 0.5, (y - box_min_y) / dy - 0.5, z)
    return row


def normalize_hands(row: dict) -> dict:
    """Scale each hand into its own xy box. z stays in MediaPipe units."""
    sequence_size = len(row["leftEar"])
    for suffix in ("_0", "_1"):
        for index in range(sequence_size):
            x_coords = [
                row[name + suffix][index][0]
                for name in HAND_LANDMARKS
                if row[name + suffix][index][0] != 0
            ]
            y_coords = [
                row[name + suffix][index][1]
                for name in HAND_LANDMARKS
                if row[name + suffix][index][1] != 0
            ]
            if not x_coords or not y_coords:
                continue
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            dx, dy = max_x - min_x, max_y - min_y
            if dx <= 0 or dy <= 0:
                continue
            for name in HAND_LANDMARKS:
                x, y, z = row[name + suffix][index]
                if x == 0 and y == 0:
                    continue
                row[name + suffix][index] = ((x - min_x) / dx - 0.5, (y - min_y) / dy - 0.5, z)
    return row


def _empty_sequence() -> dict:
    return {name: [] for name in LANDMARKS}


def _append_frame(sequence: dict, results) -> None:
    pose_data = {name: _ZERO for name in BODY_LANDMARKS}
    if results.pose_landmarks is not None:
        for name, index in POSE_INDEX.items():
            point = results.pose_landmarks.landmark[index]
            pose_data[name] = (point.x, point.y, point.z)
        left, right = pose_data["leftShoulder"], pose_data["rightShoulder"]
        pose_data["neck"] = (
            (left[0] + right[0]) / 2,
            (left[1] + right[1]) / 2,
            (left[2] + right[2]) / 2,
        )
    for name in BODY_LANDMARKS:
        sequence[name].append(pose_data[name])

    for suffix, hand in (("_0", results.left_hand_landmarks), ("_1", results.right_hand_landmarks)):
        hand_data = {name + suffix: _ZERO for name in HAND_LANDMARKS}
        if hand is not None:
            for name, index in HAND_INDEX.items():
                point = hand.landmark[index]
                hand_data[name + suffix] = (point.x, point.y, point.z)
        for name in HAND_LANDMARKS:
            sequence[name + suffix].append(hand_data[name + suffix])


def sequence_to_array(sequence: dict) -> np.ndarray:
    """Normalized landmark dict -> (T, 76, 3) in LANDMARKS order."""
    sequence = normalize_hands(normalize_body(sequence))
    length = len(sequence["neck"])
    if length == 0:
        return np.zeros((0, N_JOINTS, 3), dtype=np.float32)
    frames = [[sequence[name][index] for name in LANDMARKS] for index in range(length)]
    array = np.asarray(frames, dtype=np.float32)
    if array.shape[1:] != (N_JOINTS, 3):
        raise ValueError(f"expected {(N_JOINTS, 3)} joints, got {array.shape[1:]}")
    return array


def to_model_input(sequence: np.ndarray) -> np.ndarray:
    """(T, 76, 3) -> (80, 228), keeping the first frames and padding the tail with zeros."""
    if sequence.ndim != 3 or sequence.shape[1:] != (N_JOINTS, 3):
        raise ValueError(f"keypoints must be (T, {N_JOINTS}, 3), got {sequence.shape}")
    flat = np.ascontiguousarray(sequence.reshape(sequence.shape[0], FEATURE_DIM), dtype=np.float32)
    length = flat.shape[0]
    if length == 0:
        raise ValueError("video produced no keypoint frames")
    if length > MAX_FRAMES:
        flat = flat[:MAX_FRAMES]
    elif length < MAX_FRAMES:
        pad = np.zeros((MAX_FRAMES - length, FEATURE_DIM), dtype=np.float32)
        flat = np.concatenate([flat, pad], axis=0)
    return flat


def resize_square(frame_bgr: np.ndarray, size: int = FRAME_SIZE) -> np.ndarray:
    """Center-crop to a square, then resize to the 224 training frame."""
    height, width = frame_bgr.shape[:2]
    side = min(height, width)
    y0 = (height - side) // 2
    x0 = (width - side) // 2
    square = frame_bgr[y0:y0 + side, x0:x0 + side]
    if square.shape[0] != size or square.shape[1] != size:
        interpolation = cv2.INTER_AREA if side > size else cv2.INTER_LINEAR
        square = cv2.resize(square, (size, size), interpolation=interpolation)
    return square


def blank_sequence() -> dict:
    return _empty_sequence()


def append_holistic(sequence: dict, results) -> None:
    _append_frame(sequence, results)


def keypoints_from_video(video_path: Path, extractor: HolisticExtractor) -> np.ndarray:
    extractor.reset()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    sequence = _empty_sequence()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame = resize_square(frame)
            _append_frame(sequence, extractor.process(frame))
    finally:
        capture.release()
    array = sequence_to_array(sequence)
    logger.info("Keypoints %s %s", video_path, tuple(array.shape))
    return array


def _init_worker() -> None:
    global _EXTRACTOR
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(processName)s %(message)s")
    _EXTRACTOR = HolisticExtractor()


def _extract_job(video_path: str) -> tuple[str, np.ndarray]:
    if _EXTRACTOR is None:
        raise RuntimeError("Worker extractor was not initialized")
    sequence = keypoints_from_video(Path(video_path), _EXTRACTOR)
    return video_path, sequence


def iter_videos(source: Path) -> list[Path]:
    path = Path(source)
    if path.is_file():
        return [path]
    return [item for item in sorted(path.rglob("*")) if item.suffix.lower() in VIDEO_SUFFIXES]


def extract_sequences(videos: Sequence[Path], workers: int) -> list[tuple[str, np.ndarray]]:
    jobs = [str(video) for video in videos]
    worker_count = max(1, min(workers, len(jobs)))
    logger.info("Extracting keypoints for %d video(s) with %d worker(s)", len(jobs), worker_count)
    if worker_count == 1:
        _init_worker()
        try:
            return [_extract_job(job) for job in jobs]
        finally:
            if _EXTRACTOR is not None:
                _EXTRACTOR.close()
    with Pool(processes=worker_count, initializer=_init_worker) as pool:
        return pool.map(_extract_job, jobs)
