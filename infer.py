"""Infer a VSL gloss name from a video.

Keypoint extraction lives in preprocess. The clip is resampled to each loaded
model's own temporal length before it is scored.
"""

from __future__ import annotations

import argparse
import ast
import logging
from pathlib import Path
from typing import Optional, Sequence

from models import load_models, pick_device, predict_gloss
from preprocess import extract_sequences, iter_videos, resample_clip

logger = logging.getLogger(__name__)

SIGN_DIR = Path(__file__).resolve().parent
DEFAULT_GLOSSES = SIGN_DIR / "glosses.txt"
DEFAULT_CHECKPOINTS = SIGN_DIR / "checkpoints"


def load_glosses(path: Path) -> list[str]:
    """Load gloss names and sort them the same way the training ids were built."""
    text = Path(path).read_text(encoding="utf-8").strip().rstrip(",")
    names = ast.literal_eval(f"[{text}]")
    glosses = sorted(str(name) for name in names)
    if len(glosses) != len(set(glosses)):
        raise ValueError(f"Duplicate gloss names in {path}")
    logger.info("Loaded %d glosses from %s", len(glosses), path)
    return glosses


def infer_videos(
    source: Path,
    glosses_path: Path = DEFAULT_GLOSSES,
    checkpoint_dir: Path = DEFAULT_CHECKPOINTS,
    workers: int = 1,
    device: Optional[object] = None,
) -> list[dict[str, object]]:
    videos = iter_videos(source)
    if not videos:
        raise FileNotFoundError(f"No videos under {source}")

    glosses = load_glosses(glosses_path)
    torch_device = device or pick_device()
    logger.info("Device: %s", torch_device)
    loaded = load_models(checkpoint_dir, num_classes=len(glosses), device=torch_device)
    extracted = extract_sequences(videos, workers=workers)

    results = []
    for video_path, sequence in extracted:
        if sequence.shape[0] == 0:
            logger.error("No frames in %s", video_path)
            results.append({"video": video_path, "gloss": None, "predictions": []})
            continue
        specs = {(model.is_leg, model.max_frames) for model in loaded.values()}
        if len(specs) != 1:
            raise ValueError(f"loaded models disagree on is_leg/max_frames: {specs}")
        is_leg, frames = next(iter(specs))
        features = resample_clip(sequence, frames, is_leg)
        if features is None:
            logger.error("Shoulders too weak to normalize %s", video_path)
            results.append({"video": video_path, "gloss": None, "predictions": []})
            continue
        logger.info("Resampled %s from %d frames to T=%d", video_path, sequence.shape[0], frames)
        prediction = predict_gloss(features, loaded, glosses, torch_device)
        prediction["video"] = video_path
        logger.info("gloss: %s", prediction["gloss"])
        results.append(prediction)
    return results


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer a VSL gloss name from a video.")
    parser.add_argument("source", type=Path, help="Video file or directory of videos.")
    parser.add_argument("--glosses", type=Path, default=DEFAULT_GLOSSES, help="Gloss label file, sorted A-Z.")
    parser.add_argument("--checkpoints", type=Path, default=DEFAULT_CHECKPOINTS, help="Directory with ST-GCN and TCN .pth files.")
    parser.add_argument("--workers", type=int, default=1, help="Process pool size for keypoint extraction.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv)
    infer_videos(args.source, glosses_path=args.glosses, checkpoint_dir=args.checkpoints, workers=args.workers)


if __name__ == "__main__":
    main()
