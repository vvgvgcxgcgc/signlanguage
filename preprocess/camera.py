"""Webcam: raise an open right hand to record, then run TCN.

An open right hand held up in front of the face starts a 5 second countdown.
Recording begins when that countdown ends and lasts 2 seconds. TCN then runs
once on that clip. Raising the open right hand again repeats the cycle.

    python -m preprocess.camera
"""

from __future__ import annotations

import argparse
import ast
import logging
import time
from pathlib import Path
from typing import Optional, Sequence

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from models.classifiers import load_models, pick_device, predict_gloss
from preprocess.keypoints import (
    FRAME_SIZE,
    HolisticExtractor,
    append_holistic,
    blank_sequence,
    resize_square,
    sequence_to_array,
    to_model_input,
)

logger = logging.getLogger(__name__)

_DRAW = mp.solutions.drawing_utils
_HOLISTIC = mp.solutions.holistic
_WINDOW = "holistic"
_DISPLAY_SCALE = 3

SIGN_DIR = Path(__file__).resolve().parents[1]
DEFAULT_GLOSSES = SIGN_DIR / "glosses.txt"
DEFAULT_CHECKPOINTS = SIGN_DIR / "checkpoints"

COUNTDOWN_SECONDS = 5.0
RECORD_SECONDS = 2.0
TRIGGER_FRAMES = 6
# Tip is farther from the wrist than this joint when the finger is extended.
_FINGER_JOINTS = ((4, 3), (8, 6), (12, 10), (16, 14), (20, 18))


def draw_keypoints(frame_bgr, results) -> None:
    """Draw pose and both hands on a BGR frame. Face mesh is not used."""
    if results.pose_landmarks:
        _DRAW.draw_landmarks(frame_bgr, results.pose_landmarks, _HOLISTIC.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        _DRAW.draw_landmarks(frame_bgr, results.left_hand_landmarks, _HOLISTIC.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        _DRAW.draw_landmarks(frame_bgr, results.right_hand_landmarks, _HOLISTIC.HAND_CONNECTIONS)


def _distance(a, b) -> float:
    return float(((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5)


def open_right_hand(hand) -> bool:
    """True when all five digits are extended and the fingertips are spread."""
    if hand is None:
        return False
    points = hand.landmark
    wrist = points[0]
    extended = 0
    for tip, joint in _FINGER_JOINTS:
        if _distance(points[tip], wrist) > _distance(points[joint], wrist) * 1.15:
            extended += 1
    span = _distance(points[8], points[20])
    palm = _distance(points[0], points[9]) + 1e-6
    return extended == 5 and span > palm * 0.45


def right_hand_raised(results) -> bool:
    """Open right hand held up in front of the face."""
    hand = results.right_hand_landmarks
    if not open_right_hand(hand):
        return False
    wrist = hand.landmark[0]
    if results.pose_landmarks is None:
        return wrist.y < 0.45
    pose = results.pose_landmarks.landmark
    nose = pose[0]
    shoulder_y = (pose[11].y + pose[12].y) * 0.5
    above_shoulders = wrist.y < shoulder_y - 0.02
    in_front = abs(wrist.x - nose.x) < 0.55
    return above_shoulders and in_front


def load_glosses(path: Path) -> list[str]:
    text = Path(path).read_text(encoding="utf-8").strip().rstrip(",")
    names = ast.literal_eval(f"[{text}]")
    glosses = sorted(str(name) for name in names)
    if len(glosses) != len(set(glosses)):
        raise ValueError(f"Duplicate gloss names in {path}")
    logger.info("Loaded %d glosses from %s", len(glosses), path)
    return glosses


class CaptureCycle:
    """Wait for the raised hand, count down 5s, then record 2s."""

    def __init__(self) -> None:
        self.phase = "wait"
        self.armed = True
        self.hold = 0
        self.deadline = 0.0
        self.sequence: Optional[dict] = None

    def step(self, results, now: float) -> Optional[dict]:
        """Return the recorded landmark dict when the 2 second clip finishes."""
        if self.phase == "wait":
            if right_hand_raised(results):
                self.hold += 1
                if self.armed and self.hold >= TRIGGER_FRAMES:
                    self.phase = "countdown"
                    self.deadline = now + COUNTDOWN_SECONDS
                    self.armed = False
                    self.hold = 0
                    logger.info("Countdown %ss", int(COUNTDOWN_SECONDS))
            else:
                self.hold = 0
                self.armed = True
            return None

        if self.phase == "countdown":
            if now < self.deadline:
                return None
            self.phase = "record"
            self.deadline = now + RECORD_SECONDS
            self.sequence = blank_sequence()
            append_holistic(self.sequence, results)
            logger.info("Recording %ss", int(RECORD_SECONDS))
            return None

        append_holistic(self.sequence, results)
        if now < self.deadline:
            return None
        finished = self.sequence
        count = len(finished["neck"]) if finished is not None else 0
        self.sequence = None
        self.phase = "wait"
        self.hold = 0
        logger.info("Recorded %d frames", count)
        return finished

    def banner(self, now: float) -> str:
        if self.phase == "countdown":
            remain = max(0.0, self.deadline - now)
            return str(max(1, int(np.ceil(remain))))
        if self.phase == "record":
            remain = max(0.0, self.deadline - now)
            return f"Thu {remain:.1f}s"
        return "Gio tay phai, xoe 5 ngon"


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def paint_status(frame_bgr, banner: str, gloss: str, countdown: bool) -> None:
    image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, frame_bgr.shape[1], 72), fill=(20, 20, 24))
    draw.text((12, 8), banner, font=_font(28), fill=(240, 200, 80) if countdown else (80, 200, 120))
    if gloss:
        draw.text((12, 40), gloss, font=_font(24), fill=(255, 255, 255))
    if countdown and banner.isdigit():
        size = _font(160)
        box = draw.textbbox((0, 0), banner, font=size)
        x = (frame_bgr.shape[1] - (box[2] - box[0])) // 2
        y = (frame_bgr.shape[0] - (box[3] - box[1])) // 2
        draw.text((x, y), banner, font=size, fill=(255, 220, 60))
    frame_bgr[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def run_camera(
    camera: int = 0,
    width: int = 0,
    height: int = 0,
    glosses_path: Path = DEFAULT_GLOSSES,
    checkpoint_dir: Path = DEFAULT_CHECKPOINTS,
) -> None:
    capture = cv2.VideoCapture(camera)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open camera {camera}")
    if width > 0:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height > 0:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(
        "Camera %s opened at %dx%d. Holistic runs on %d. Press q to quit.",
        camera,
        frame_width,
        frame_height,
        FRAME_SIZE,
    )

    glosses = load_glosses(glosses_path)
    device = pick_device()
    tcn = load_models(checkpoint_dir, num_classes=len(glosses), device=device, names=("TCN",))
    cycle = CaptureCycle()
    gloss = ""
    extractor = HolisticExtractor()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                logger.error("Camera %s returned no frame", camera)
                break
            frame = resize_square(frame)
            results = extractor.process(frame)
            now = time.monotonic()
            finished = cycle.step(results, now)
            if finished is not None and len(finished["neck"]) > 0:
                features = to_model_input(sequence_to_array(finished))
                prediction = predict_gloss(features, tcn, glosses, device)
                gloss = str(prediction["gloss"])
                logger.info("gloss: %s", gloss)
            draw_keypoints(frame, results)
            show = cv2.resize(
                frame,
                (FRAME_SIZE * _DISPLAY_SCALE, FRAME_SIZE * _DISPLAY_SCALE),
                interpolation=cv2.INTER_NEAREST,
            )
            paint_status(show, cycle.banner(now), gloss, cycle.phase == "countdown")
            cv2.imshow(_WINDOW, show)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        extractor.close()
        capture.release()
        cv2.destroyAllWindows()
        logger.info("Camera %s closed", camera)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Webcam Holistic preview with TCN on each finished gesture.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument("--width", type=int, default=0, help="Requested frame width. 0 keeps the camera default.")
    parser.add_argument("--height", type=int, default=0, help="Requested frame height. 0 keeps the camera default.")
    parser.add_argument("--glosses", type=Path, default=DEFAULT_GLOSSES)
    parser.add_argument("--checkpoints", type=Path, default=DEFAULT_CHECKPOINTS)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv)
    run_camera(
        camera=args.camera,
        width=args.width,
        height=args.height,
        glosses_path=args.glosses,
        checkpoint_dir=args.checkpoints,
    )


if __name__ == "__main__":
    main()
