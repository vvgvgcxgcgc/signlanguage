"""Raw keypoints, matching extract-keypoint-vslfullfront-raw.ipynb.

Each frame is 76 joints of xyz in MediaPipe image coordinates: 34 body points
(33 pose plus the shoulder midpoint as neck) and 42 hand points, name-major,
left then right. Missing joints stay (0, 0, 0). No box normalization happens
here. preprocess.dataset.resample_clip is what turns a queue of these frames
into a model's (T, J, 4) input.

The training videos are already 224x224 squares at 25 fps, so the notebook
feeds them as they are. A camera frame has to go through `square_frame`
first, otherwise MediaPipe's normalized x and y are scaled by different
widths and the shoulder-width normalization no longer means the same thing.
"""

from __future__ import annotations

import logging
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, Sequence

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)

N_JOINTS = 76
FRAME_SIZE = 224
TRAIN_FPS = 25.0
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

_BODY_ROW = {name: index for index, name in enumerate(BODY_LANDMARKS)}
_NECK_ROW = _BODY_ROW["neck"]
_LEFT_SHOULDER_ROW = _BODY_ROW["leftShoulder"]
_RIGHT_SHOULDER_ROW = _BODY_ROW["rightShoulder"]
_HAND_ROW = {
    (name, slot): len(BODY_LANDMARKS) + name_index * 2 + slot
    for name_index, name in enumerate(HAND_LANDMARKS)
    for slot in (0, 1)
}

MODEL_DIR = Path(__file__).resolve().parents[1] / "mediapipe_models"
POSE_MODEL_PATH = MODEL_DIR / "pose_landmarker_full.task"
HAND_MODEL_PATH = MODEL_DIR / "hand_landmarker.task"
_MODEL_URLS = {
    POSE_MODEL_PATH: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    HAND_MODEL_PATH: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
}

_EXTRACTOR: Optional["KeypointExtractor"] = None


def ensure_models() -> None:
    """Download the pose and hand task files once. IMAGE mode has no tracker state."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for path, url in _MODEL_URLS.items():
        if path.is_file() and path.stat().st_size > 0:
            continue
        logger.info("Downloading %s", path.name)
        urllib.request.urlretrieve(url, path)


def _landmarker_options(running_mode):
    ensure_models()
    base = mp.tasks.BaseOptions
    vision = mp.tasks.vision
    pose = vision.PoseLandmarkerOptions(
        base_options=base(model_asset_path=str(POSE_MODEL_PATH)),
        running_mode=running_mode,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    hand = vision.HandLandmarkerOptions(
        base_options=base(model_asset_path=str(HAND_MODEL_PATH)),
        running_mode=running_mode,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return pose, hand


def _select_hands(hand_result) -> dict:
    chosen = {}
    landmarks_list = hand_result.hand_landmarks or []
    handedness_list = hand_result.handedness or []
    for landmarks, handedness in zip(landmarks_list, handedness_list):
        if not handedness:
            continue
        label = handedness[0].category_name
        score = handedness[0].score
        if label not in ("Left", "Right"):
            continue
        previous = chosen.get(label)
        if previous is None or score > previous[0]:
            chosen[label] = (score, landmarks)
    return chosen


def _fill_pose(row: np.ndarray, pose_result) -> None:
    if not pose_result.pose_landmarks:
        return
    landmarks = pose_result.pose_landmarks[0]
    for name, mp_index in POSE_INDEX.items():
        point = landmarks[mp_index]
        slot = _BODY_ROW[name]
        row[slot, 0] = point.x
        row[slot, 1] = point.y
        row[slot, 2] = point.z
    row[_NECK_ROW] = (row[_LEFT_SHOULDER_ROW] + row[_RIGHT_SHOULDER_ROW]) / 2


def _fill_hand(row: np.ndarray, hand_result) -> None:
    chosen = _select_hands(hand_result)
    for slot, label in ((0, "Left"), (1, "Right")):
        picked = chosen.get(label)
        if picked is None:
            continue
        landmarks = picked[1]
        for name, mp_index in HAND_INDEX.items():
            point = landmarks[mp_index]
            joint = _HAND_ROW[(name, slot)]
            row[joint, 0] = point.x
            row[joint, 1] = point.y
            row[joint, 2] = point.z


def square_frame(frame_bgr: np.ndarray, size: int = FRAME_SIZE) -> np.ndarray:
    """Center-crop to a square and resize to the 224 training frame."""
    height, width = frame_bgr.shape[:2]
    side = min(height, width)
    y0 = (height - side) // 2
    x0 = (width - side) // 2
    square = frame_bgr[y0:y0 + side, x0:x0 + side]
    if side == size:
        return np.ascontiguousarray(square)
    interpolation = cv2.INTER_AREA if side > size else cv2.INTER_LINEAR
    return cv2.resize(square, (size, size), interpolation=interpolation)


def landmarks_from_frame(frame_bgr: np.ndarray, pose_landmarker, hand_landmarker) -> np.ndarray:
    """One BGR frame to a raw (76, 3) row. The frame is not resized."""
    rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    row = np.zeros((N_JOINTS, 3), dtype=np.float32)
    _fill_pose(row, pose_landmarker.detect(image))
    _fill_hand(row, hand_landmarker.detect(image))
    return row


class KeypointExtractor:
    """One pose landmarker and one hand landmarker. Safe to reuse across frames.

    Offline clips stay in IMAGE mode: every frame is detected on its own, which
    is how the training keypoints were built. `live` switches both landmarkers
    to VIDEO mode so the camera tracks instead of detecting from scratch, and
    runs them one after the other. That stays inside the 25 fps budget on one
    core. `parallel` still overlaps the two detectors; the camera does not use
    it, because the overlap is what pins a second core.
    """

    def __init__(self, parallel: bool = False, live: bool = False) -> None:
        vision = mp.tasks.vision
        running = vision.RunningMode.VIDEO if live else vision.RunningMode.IMAGE
        pose_options, hand_options = _landmarker_options(running)
        self._pose = vision.PoseLandmarker.create_from_options(pose_options)
        self._hand = vision.HandLandmarker.create_from_options(hand_options)
        self._live = live
        self._stamp_ms = 0
        self._clock = time.monotonic()
        self._threads = ThreadPoolExecutor(max_workers=2) if parallel else None

    def _timestamp_ms(self) -> int:
        """Monotonic milliseconds. VIDEO mode rejects a repeated stamp."""
        elapsed = int((time.monotonic() - self._clock) * 1000)
        if elapsed <= self._stamp_ms:
            elapsed = self._stamp_ms + 1
        self._stamp_ms = elapsed
        return self._stamp_ms

    def close(self) -> None:
        if self._threads is not None:
            self._threads.shutdown(wait=True)
        self._pose.close()
        self._hand.close()

    def __enter__(self) -> "KeypointExtractor":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _image(self, frame_bgr: np.ndarray):
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    def process(self, frame_bgr: np.ndarray, hands: bool = True) -> np.ndarray:
        """Raw (76, 3). `hands=False` runs the pose landmarker only.

        The pose gate uses body joints, so a live camera can skip the hand
        landmarker while the signer is at rest and call `fill_hands` on the
        frames that actually enter a gloss clip.
        """
        if not self._live and self._threads is None:
            if hands:
                return landmarks_from_frame(frame_bgr, self._pose, self._hand)
            row = np.zeros((N_JOINTS, 3), dtype=np.float32)
            _fill_pose(row, self._pose.detect(self._image(frame_bgr)))
            return row
        image = self._image(frame_bgr)
        row = np.zeros((N_JOINTS, 3), dtype=np.float32)
        if self._live and self._threads is None:
            stamp = self._timestamp_ms()
            _fill_pose(row, self._pose.detect_for_video(image, stamp))
            if hands:
                _fill_hand(row, self._hand.detect_for_video(image, self._timestamp_ms()))
            return row
        if self._live:
            stamp = self._timestamp_ms()
            pose_job = self._threads.submit(self._pose.detect_for_video, image, stamp)
            hand_job = self._threads.submit(self._hand.detect_for_video, image, stamp) if hands else None
        else:
            pose_job = self._threads.submit(self._pose.detect, image)
            hand_job = self._threads.submit(self._hand.detect, image) if hands else None
        _fill_pose(row, pose_job.result())
        if hand_job is not None:
            _fill_hand(row, hand_job.result())
        return row

    def fill_hands(self, frame_bgr: np.ndarray, row: np.ndarray) -> None:
        """Write the 42 hand joints into an existing row. Pose joints stay."""
        image = self._image(frame_bgr)
        row[len(BODY_LANDMARKS):] = 0
        if self._live:
            _fill_hand(row, self._hand.detect_for_video(image, self._timestamp_ms()))
        else:
            _fill_hand(row, self._hand.detect(image))


def keypoints_from_video(video_path: Path, extractor: KeypointExtractor) -> np.ndarray:
    """Every frame of a video as raw float32 (T, 76, 3)."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(extractor.process(frame))
    finally:
        capture.release()
    if not frames:
        logger.error("No frames in %s", video_path)
        return np.zeros((0, N_JOINTS, 3), dtype=np.float32)
    array = np.stack(frames)
    logger.info("Keypoints %s %s", video_path, tuple(array.shape))
    return array


def _init_worker() -> None:
    global _EXTRACTOR
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(processName)s %(message)s")
    cv2.setNumThreads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    _EXTRACTOR = KeypointExtractor()


def _extract_job(video_path: str) -> tuple[str, np.ndarray]:
    if _EXTRACTOR is None:
        raise RuntimeError("Worker extractor was not initialized")
    return video_path, keypoints_from_video(Path(video_path), _EXTRACTOR)


def iter_videos(source: Path) -> list[Path]:
    path = Path(source)
    if path.is_file():
        return [path]
    return [item for item in sorted(path.rglob("*")) if item.suffix.lower() in VIDEO_SUFFIXES]


def extract_sequences(videos: Sequence[Path], workers: int) -> list[tuple[str, np.ndarray]]:
    jobs = [str(video) for video in videos]
    if not jobs:
        return []
    ensure_models()
    worker_count = max(1, min(workers, len(jobs)))
    logger.info("Extracting keypoints for %d video(s) with %d worker(s)", len(jobs), worker_count)
    cv2.setNumThreads(1)
    if worker_count == 1:
        _init_worker()
        try:
            return [_extract_job(job) for job in jobs]
        finally:
            if _EXTRACTOR is not None:
                _EXTRACTOR.close()
    with Pool(processes=worker_count, initializer=_init_worker) as pool:
        return pool.map(_extract_job, jobs)
