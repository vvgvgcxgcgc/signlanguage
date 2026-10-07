"""Live gloss inference from a webcam.

Every frame is scored by the pose-state model. P(action) <= threshold is the
rest pose (label 0). The sign queue is raw (76, 3) frames and is not cut to a
fixed length: while the pose stays at rest the queue keeps only the frame just
scored, and once P(action) rises it accumulates every frame of the gesture.
When the pose returns to rest, that closing frame is appended and
preprocess.dataset.resample_clip turns the whole queue into the sign model's
own T.

    python infer_camera.py \\
        --checkpoint checkpoints/MS-TCN/j68_MS-TCN_best.pth \\
        --pose-checkpoint checkpoints/pose_state_mlp_best.pth

    python infer_camera.py \\
        --checkpoint checkpoints/ST-GCN/j68_ST-GCN_best.pth \\
        --pose-checkpoint checkpoints/pose_state_mlp_best.pth

One `--checkpoint` file. The architecture comes from `model_name` inside it,
so an ST-GCN file is the same call as an MS-TCN file. j76 keeps the leg joints
when that file was trained with `is_leg`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
import torch

from models.classifiers import LoadedModel, predict_gloss
from models.factory import build_model, pick_device
from models.pose_state import PoseStateMLP
from preprocess.camera import FrameGrabber, Pacer, display_frame, open_capture, paint_status, resolve_camera
from preprocess.dataset import resample_clip
from preprocess.keypoints import TRAIN_FPS, KeypointExtractor, square_frame
from preprocess.pose_state import ACTION, WINDOW, latest_window

logger = logging.getLogger("infer_camera")

_WINDOW_NAME = "sign"


class GestureQueue:
    """Raw-frame buffer for one gloss.

    Rest pops the queue and stores the frame just scored. Action appends, with
    no length cap. The next rest frame is appended too, then the caller
    resamples whatever was collected.
    """

    def __init__(self, threshold: float = 0.3) -> None:
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"threshold must be in (0, 1), got {threshold}")
        self.threshold = float(threshold)
        self.frames: list[np.ndarray] = []
        self.phase = "idle"
        self._in_action = False

    @property
    def in_action(self) -> bool:
        return self._in_action

    def step(self, frame: np.ndarray, action_prob: Optional[float]) -> Optional[np.ndarray]:
        """Return the raw (N, 76, 3) clip when a gesture closes."""
        row = np.asarray(frame, dtype=np.float32).copy()
        if action_prob is None or not np.isfinite(action_prob):
            if self._in_action:
                self.frames.append(row)
            return None
        if action_prob <= self.threshold:
            if self._in_action:
                self.frames.append(row)
                clip = np.stack(self.frames)
                self.frames.clear()
                self._in_action = False
                self.phase = "prepare"
                return clip
            self.frames.clear()
            self.frames.append(row)
            self.phase = "prepare"
            return None
        if not self.frames:
            self.phase = "idle"
            return None
        self.frames.append(row)
        self._in_action = True
        self.phase = "action"
        return None


def _read_labels(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    names = payload["labels"] if isinstance(payload, dict) else payload
    glosses = [str(name) for name in names]
    if len(glosses) != len(set(glosses)):
        raise ValueError(f"duplicate gloss names in {path}")
    return glosses


def load_sign(
    path: Path,
    device: torch.device,
    labels_path: Optional[Path] = None,
) -> tuple[LoadedModel, list[str]]:
    """Rebuild one gloss model. T and the joint set come from the checkpoint."""
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    name = str(payload["model_name"])
    num_classes = int(payload["num_classes"])
    is_leg = bool(payload["is_leg"])
    max_frames = int(payload["max_frames"])
    if labels_path is None:
        if "class_names" not in payload:
            raise ValueError(f"{path} has no class_names, pass --labels")
        glosses = [str(item) for item in payload["class_names"]]
    else:
        glosses = _read_labels(labels_path)
    if len(glosses) != num_classes:
        raise ValueError(f"{path.name} has {num_classes} classes but {len(glosses)} glosses were given")
    module = build_model(name, num_classes, is_leg=is_leg, max_frames=max_frames)
    state = {key.replace("module.", ""): value for key, value in payload["state_dict"].items()}
    module.load_state_dict(state)
    module.to(device).eval()
    loaded = LoadedModel(
        name=name,
        module=module,
        is_leg=is_leg,
        max_frames=max_frames,
        num_joints=module.embed.num_joints,
    )
    logger.info(
        "Sign model %s T=%d joints=%d classes=%d from %s",
        name,
        max_frames,
        loaded.num_joints,
        num_classes,
        path,
    )
    return loaded, glosses


def load_pose(path: Path, device: torch.device) -> PoseStateMLP:
    model, payload = PoseStateMLP.from_checkpoint(path, device)
    classes = int(model.config["num_classes"])
    if classes != 2:
        raise ValueError(f"pose model must have 2 classes (start, action), got {classes}")
    if int(model.window) != WINDOW:
        raise ValueError(f"pose model window is {model.window}, preprocess.pose_state uses {WINDOW}")
    logger.info("Pose model from %s, epoch=%s", path, payload.get("epoch"))
    return model


@torch.no_grad()
def action_probability(model: PoseStateMLP, window: np.ndarray, device: torch.device) -> float:
    """P(action) for one (W, J, 2) window. Class 1 is action."""
    batch = torch.from_numpy(np.ascontiguousarray(window, dtype=np.float32)).unsqueeze(0).to(device)
    return float(torch.softmax(model(batch), dim=1)[0, ACTION])


def run_camera(
    checkpoint: Path,
    pose_checkpoint: Path,
    camera: Optional[int] = None,
    width: int = 0,
    height: int = 0,
    threshold: float = 0.3,
    min_shoulder: float = 1e-3,
    labels_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
    fps: float = TRAIN_FPS,
) -> None:
    torch_device = device or pick_device()
    sign, glosses = load_sign(checkpoint, torch_device, labels_path)
    pose = load_pose(pose_checkpoint, torch_device)
    camera = resolve_camera(camera)
    extractor = KeypointExtractor(live=True)
    capture = open_capture(camera, width, height, fps)
    grabber = FrameGrabber(capture)
    pacer = Pacer(fps)
    gate = GestureQueue(threshold)
    recent: list[np.ndarray] = []
    held_square: Optional[np.ndarray] = None
    gloss = ""
    previous = ""
    stamps: list[float] = []
    logger.info("Threshold %.2f. Rest is P(action) <= %.2f. Press q to quit.", threshold, threshold)
    try:
        while True:
            pacer.wait()
            frame = grabber.read()
            if frame is None:
                logger.error("Camera %s stopped delivering frames", camera)
                break
            stamps.append(time.monotonic())
            if len(stamps) == 100:
                rate = (len(stamps) - 1) / (stamps[-1] - stamps[0])
                if abs(rate - TRAIN_FPS) > 3:
                    logger.warning("Loop runs at %.1f fps, training clips are %.0f fps", rate, TRAIN_FPS)
                else:
                    logger.info("Loop runs at %.1f fps", rate)
                stamps.clear()
            square = square_frame(frame)
            # Hands cost more than the body. The gate only reads body joints,
            # so they run while a gloss is open. The rest frame that opens the
            # next gloss is filled in when the pose crosses the threshold.
            row = extractor.process(square, hands=gate.in_action)
            recent.append(row)
            if len(recent) > WINDOW:
                del recent[0]
            window = latest_window(np.stack(recent))
            probability = None if window is None else action_probability(pose, window, torch_device)
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
            finished = gate.step(row, probability)
            if gate.in_action:
                held_square = None
            else:
                held_square = square.copy()
            if gate.phase != previous:
                shown = None if probability is None else round(probability, 3)
                logger.info("phase %s P(action)=%s queue=%d", gate.phase, shown, len(gate.frames))
                previous = gate.phase
            if finished is not None:
                features = resample_clip(finished, sign.max_frames, sign.is_leg, min_shoulder)
                if features is None:
                    logger.warning("Dropped a %d-frame clip, shoulders were too weak", finished.shape[0])
                else:
                    logger.info(
                        "Resampled %d raw frames to T=%d J=%d",
                        finished.shape[0],
                        features.shape[0],
                        features.shape[1],
                    )
                    prediction = predict_gloss(features, {sign.name: sign}, glosses, torch_device)
                    gloss = f"{prediction['gloss']}  {prediction['confidence']:.2f}"
                    logger.info("gloss: %s", gloss)
            detail = "no pose" if probability is None else f"p={probability:.2f}  n={len(gate.frames)}"
            shown = display_frame(square, row)
            paint_status(shown, gate.phase, detail, gloss)
            cv2.imshow(_WINDOW_NAME, shown)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        grabber.close()
        extractor.close()
        capture.release()
        cv2.destroyAllWindows()
        logger.info("Camera %s closed", camera)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer glosses from a webcam.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Sign model .pth. Its max_frames is T.")
    parser.add_argument("--pose-checkpoint", type=Path, required=True, help="PoseStateMLP .pth.")
    parser.add_argument("--labels", type=Path, default=None, help="JSON label list. Default is the checkpoint's class_names.")
    parser.add_argument("--camera", type=int, default=None, help="Index. Default is the built-in Mac camera.")
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--fps", type=float, default=TRAIN_FPS, help="Requested camera rate. Training videos are 25.")
    parser.add_argument("--threshold", type=float, default=0.3, help="P(action) at or below this is the rest pose.")
    parser.add_argument("--min-shoulder", type=float, default=1e-3)
    parser.add_argument("--device", default=None, help="cpu, cuda, or mps. Default picks one.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    if not args.checkpoint.is_file():
        logger.error("Missing sign checkpoint %s", args.checkpoint)
        return 2
    if not args.pose_checkpoint.is_file():
        logger.error("Missing pose checkpoint %s", args.pose_checkpoint)
        return 2
    device = None if args.device is None else torch.device(args.device)
    run_camera(
        checkpoint=args.checkpoint,
        pose_checkpoint=args.pose_checkpoint,
        camera=args.camera,
        width=args.width,
        height=args.height,
        threshold=args.threshold,
        min_shoulder=args.min_shoulder,
        labels_path=args.labels,
        device=device,
        fps=args.fps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
