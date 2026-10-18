"""Cut the first prepare-to-prepare clip out of each SignConnect video.

The pose-state model and the live gesture queue run the same way as the
camera. Only the first closed gloss is kept. Later gestures in the file are
ignored. A clip that the camera would ask the signer to repeat is not saved.

Raw (T, 76, 3) arrays go to data/signconnect/keypoints, one npy per video,
using the video stem as the file name. Scoring reads those files and does
not open the videos again.

    python extract_signconnect.py
    python extract_signconnect.py --workers 4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from collections import Counter
from multiprocessing import get_context
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from infer_camera import GestureQueue, action_probability, load_pose
from preprocess.dataset import accept_raw_frame, clip_needs_redo
from preprocess.keypoints import TRAIN_FPS, KeypointExtractor, square_frame
from preprocess.pose_state import WINDOW, latest_window

logger = logging.getLogger("extract_signconnect")

SIGN_DIR = Path(__file__).resolve().parent
DEFAULT_VIDEOS = SIGN_DIR / "data" / "signconnect" / "Videos"
DEFAULT_KEYPOINTS = SIGN_DIR / "data" / "signconnect" / "keypoints"
DEFAULT_EVAL_LABELS = SIGN_DIR / "labels" / "signconnect_j76_common.json"
DEFAULT_POSE = SIGN_DIR / "checkpoints" / "pose_state_mlp_best.pth"
_TRAILING_INDEX = re.compile(r"\(\d+\)$")
_WORKER: dict[str, object] = {}


class _FrameClock:
    """MediaPipe video stamps spaced at the training fps, one frame at a time."""

    def __init__(self, fps: float) -> None:
        self.step = 1000.0 / fps
        self.frame = 0
        self.sub = 0

    def stamp(self) -> int:
        value = int(self.frame * self.step) + self.sub
        self.sub += 1
        return value

    def advance(self) -> None:
        self.frame += 1
        self.sub = 0


def gloss_from_name(path: Path) -> str:
    """`Buổi sáng(2).mp4` is the gloss `Buổi sáng`. A single video has no index."""
    return _TRAILING_INDEX.sub("", path.stem)


def load_eval_glosses(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"{path} must be a non-empty gloss dict")
    names = [str(name) for name in payload]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate gloss names in {path}")
    return names


def _init_worker(pose_checkpoint: str) -> None:
    import cv2
    import torch

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    os.environ["OMP_NUM_THREADS"] = "1"
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    _WORKER["pose"] = load_pose(Path(pose_checkpoint), torch.device("cpu"))
    _WORKER["device"] = torch.device("cpu")


def _first_clip(
    video_path: Path,
    pose,
    device,
    threshold: float,
    min_shoulder: float,
) -> dict[str, object]:
    """Replay one file and return the first closed rest-action-rest clip."""
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    extractor = KeypointExtractor(live=True)
    clock = _FrameClock(TRAIN_FPS)
    extractor._timestamp_ms = clock.stamp
    gate = GestureQueue(threshold)
    recent: list[np.ndarray] = []
    held_square: Optional[np.ndarray] = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            square = square_frame(frame)
            row = extractor.process(square, hands=gate.in_action or gate.prepare_rejected)
            recent.append(row)
            if len(recent) > WINDOW:
                del recent[0]
            window = latest_window(np.stack(recent))
            probability = None if window is None else action_probability(pose, window, device)
            opening = (
                not gate.in_action
                and probability is not None
                and probability > gate.threshold
                and bool(gate.frames)
            )
            if opening:
                if held_square is not None:
                    extractor.fill_hands(held_square, gate.frames[0])
                extractor.fill_hands(square, row)
                if not accept_raw_frame(gate.frames[0], min_shoulder, check_hands=True):
                    gate.discard_prepare()
            check_hands = gate.in_action or gate.prepare_rejected or opening
            accept = accept_raw_frame(row, min_shoulder, check_hands=check_hands)
            finished = gate.step(row, probability, accept)
            if gate.in_action:
                held_square = None
            else:
                held_square = square.copy()
            clock.advance()
            if finished is None:
                continue
            kept = gate.last_kept
            dropped = gate.last_dropped
            if clip_needs_redo(kept, dropped):
                return {"status": "redo", "kept": kept, "dropped": dropped, "clip": None}
            return {
                "status": "ok",
                "kept": kept,
                "dropped": dropped,
                "clip": np.ascontiguousarray(finished, dtype=np.float32),
            }
    finally:
        capture.release()
        extractor.close()
    return {"status": "no_gesture", "kept": 0, "dropped": 0, "clip": None}


def _save_job(job: tuple[str, str, float, float]) -> dict[str, object]:
    """Cut one video and write its raw clip. The array stays in this process."""
    video, dest, threshold, min_shoulder = job
    try:
        result = _first_clip(
            Path(video),
            _WORKER["pose"],
            _WORKER["device"],
            threshold,
            min_shoulder,
        )
    except Exception:
        logger.exception("Failed %s", video)
        return {"video": video, "status": "error", "kept": 0, "dropped": 0}
    status = str(result["status"])
    if status == "ok" and result["clip"] is not None:
        np.save(dest, result["clip"])
    return {
        "video": video,
        "status": status,
        "kept": int(result["kept"]),
        "dropped": int(result["dropped"]),
    }


def extract(
    videos_dir: Path,
    keypoints_dir: Path,
    eval_labels: Path,
    pose_checkpoint: Path,
    threshold: float,
    min_shoulder: float,
    workers: int,
) -> None:
    eval_set = set(load_eval_glosses(eval_labels))
    keypoints_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    skipped_name = 0
    for path in sorted(videos_dir.iterdir()):
        if path.suffix.lower() != ".mp4":
            continue
        gloss = gloss_from_name(path)
        if gloss not in eval_set:
            skipped_name += 1
            logger.error("Skip %s: %r is not an eval gloss", path.name, gloss)
            continue
        jobs.append((str(path), str(keypoints_dir / f"{path.stem}.npy"), threshold, min_shoulder))
    if not jobs:
        raise FileNotFoundError(f"no eval videos in {videos_dir}")
    logger.info("Cutting %d videos into %s", len(jobs), keypoints_dir)

    if workers <= 1:
        _init_worker(str(pose_checkpoint))
        results = []
        for done, job in enumerate(jobs, start=1):
            results.append(_save_job(job))
            if done % 10 == 0 or done == len(jobs):
                logger.info("saved %d/%d", done, len(jobs))
    else:
        ctx = get_context("spawn")
        results = []
        with ctx.Pool(processes=workers, initializer=_init_worker, initargs=(str(pose_checkpoint),)) as pool:
            for done, result in enumerate(pool.imap_unordered(_save_job, jobs, chunksize=1), start=1):
                results.append(result)
                if done % 10 == 0 or done == len(jobs):
                    logger.info("saved %d/%d", done, len(jobs))

    counts = Counter(str(item["status"]) for item in results)
    logger.info("status %s skipped_names=%d", dict(counts), skipped_name)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save the first prepare-to-prepare clip of each video.")
    parser.add_argument("--videos", type=Path, default=DEFAULT_VIDEOS)
    parser.add_argument("--keypoints", type=Path, default=DEFAULT_KEYPOINTS)
    parser.add_argument("--eval-labels", type=Path, default=DEFAULT_EVAL_LABELS)
    parser.add_argument("--pose-checkpoint", type=Path, default=DEFAULT_POSE)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--min-shoulder", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)
    if not args.pose_checkpoint.is_file():
        logger.error("Missing pose checkpoint %s", args.pose_checkpoint)
        return 2
    extract(
        videos_dir=args.videos,
        keypoints_dir=args.keypoints,
        eval_labels=args.eval_labels,
        pose_checkpoint=args.pose_checkpoint,
        threshold=args.threshold,
        min_shoulder=args.min_shoulder,
        workers=max(1, args.workers),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
