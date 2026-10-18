"""Score saved SignConnect clips and write metric files.

Clips come from extract_signconnect.py. Each npy is the first
prepare-to-prepare raw (T, 76, 3) clip. This script does not open videos.

Metrics use the glosses in labels/signconnect_j76_common.json. A gloss with
no scored clip is left out of the averages. A prediction outside that set is
a normal miss: it counts as wrong and does not add a class.

    python infer_signconnect.py
    python infer_signconnect.py --keypoints data/signconnect/keypoints
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from infer_camera import load_sign
from models.factory import count_parameters, pick_device
from preprocess.dataset import N_JOINTS, N_COORDS, PREPARE_PAD, pad_prepare, resample_clip

logger = logging.getLogger("infer_signconnect")

SIGN_DIR = Path(__file__).resolve().parent
DEFAULT_KEYPOINTS = SIGN_DIR / "data" / "signconnect" / "keypoints"
DEFAULT_EVAL_LABELS = SIGN_DIR / "labels" / "signconnect_j76_common.json"
DEFAULT_CHECKPOINTS = SIGN_DIR / "checkpoints"
DEFAULT_OUTPUT = SIGN_DIR / "reports" / "signconnect"
LABEL_SMOOTHING = 0.1
TOP_K = 5
F1_BINS = (
    ("<0.50", 0.0, 0.50),
    ("0.50-0.80", 0.50, 0.80),
    ("0.80-0.90", 0.80, 0.90),
    ("0.90-0.95", 0.90, 0.95),
    (">=0.95", 0.95, 1.01),
)
_TRAILING_INDEX = re.compile(r"\(\d+\)$")


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


def sign_checkpoints(root: Path) -> list[Path]:
    """Every sign `*_best.pth` under `root`. The pose-state file is not one of them."""
    if root.is_file():
        return [root]
    found = [
        path
        for path in sorted(root.rglob("*_best.pth"))
        if "pose_state" not in path.name
    ]
    if not found:
        raise FileNotFoundError(f"no sign checkpoint under {root}")
    return found


def model_id(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_best"):
        stem = stem[: -len("_best")]
    return stem


def _round(value: float) -> float:
    return round(float(value), 6)


def load_clips(keypoints_dir: Path, eval_names: Sequence[str]) -> tuple[list[dict[str, object]], int]:
    """Read raw (T, 76, 3) clips. The file stem is the gloss, with `(n)` stripped."""
    if not keypoints_dir.is_dir():
        raise FileNotFoundError(f"missing keypoints directory: {keypoints_dir}")
    eval_set = set(eval_names)
    rows: list[dict[str, object]] = []
    skipped = 0
    for path in sorted(keypoints_dir.glob("*.npy")):
        gloss = gloss_from_name(path)
        if gloss not in eval_set:
            skipped += 1
            logger.error("Skip %s: %r is not an eval gloss", path.name, gloss)
            continue
        clip = np.load(path)
        if clip.ndim != 3 or clip.shape[1:] != (N_JOINTS, N_COORDS) or clip.shape[0] == 0:
            skipped += 1
            logger.error("Skip %s: expected (T, %d, %d), got %s", path.name, N_JOINTS, N_COORDS, clip.shape)
            continue
        rows.append(
            {
                "video": str(path),
                "gloss": gloss,
                "status": "ok",
                "clip": np.ascontiguousarray(clip, dtype=np.float32),
            }
        )
    if not rows:
        raise FileNotFoundError(f"no eval clips in {keypoints_dir}")
    return rows, skipped


def _featurize(
    clips: Sequence[np.ndarray],
    length: int,
    is_leg: bool,
    min_shoulder: float,
) -> list[Optional[np.ndarray]]:
    features: list[Optional[np.ndarray]] = []
    for clip in clips:
        padded = pad_prepare(clip, PREPARE_PAD)
        features.append(resample_clip(padded, length, is_leg, min_shoulder))
    return features


@torch.no_grad()
def _logits(module: torch.nn.Module, features: Sequence[np.ndarray], device: torch.device) -> torch.Tensor:
    batch = torch.from_numpy(np.stack(features).astype(np.float32, copy=False)).to(device)
    return module(batch)


def _f1_bins(values: Sequence[float]) -> dict[str, int]:
    counts = {name: 0 for name, _low, _high in F1_BINS}
    for value in values:
        for name, low, high in F1_BINS:
            if low <= value < high:
                counts[name] += 1
                break
    return counts


def score_model(
    model_key: str,
    checkpoint: Path,
    loaded,
    glosses: Sequence[str],
    eval_names: Sequence[str],
    rows: Sequence[dict[str, object]],
    device: torch.device,
    min_shoulder: float,
) -> dict[str, object]:
    """Accuracy, top-5, loss, and per-gloss F1. Classes with no clip are omitted."""
    index_of = {name: index for index, name in enumerate(glosses)}
    missing = [name for name in eval_names if name not in index_of]
    if missing:
        raise ValueError(f"{model_key} is missing eval glosses: {', '.join(missing)}")
    eval_set = set(eval_names)

    usable = [row for row in rows if row["status"] == "ok" and row["clip"] is not None]
    features = _featurize(
        [row["clip"] for row in usable],
        loaded.max_frames,
        loaded.is_leg,
        min_shoulder,
    )
    scored_rows = []
    scored_features = []
    normalize_fail = 0
    for row, feature in zip(usable, features):
        if feature is None:
            normalize_fail += 1
            continue
        scored_rows.append(row)
        scored_features.append(feature)

    support = Counter()
    correct = Counter()
    top5_hits = Counter()
    loss_sum = defaultdict(float)
    predicted = Counter()
    errors: dict[str, Counter] = defaultdict(Counter)
    outside = 0
    if scored_features:
        logits = _logits(loaded.module, scored_features, device)
        targets = torch.tensor(
            [index_of[str(row["gloss"])] for row in scored_rows],
            dtype=torch.long,
            device=logits.device,
        )
        losses = F.cross_entropy(logits, targets, label_smoothing=LABEL_SMOOTHING, reduction="none")
        top_index = logits.argmax(dim=1)
        top_k = min(TOP_K, logits.shape[1])
        in_top = logits.topk(top_k, dim=1).indices.eq(targets.unsqueeze(1)).any(dim=1)
        loss_cpu = losses.detach().cpu().tolist()
        pred_cpu = top_index.detach().cpu().tolist()
        top_cpu = in_top.detach().cpu().tolist()
        for row, loss_value, pred_index, hit in zip(scored_rows, loss_cpu, pred_cpu, top_cpu):
            truth = str(row["gloss"])
            pred = glosses[int(pred_index)]
            support[truth] += 1
            loss_sum[truth] += float(loss_value)
            if hit:
                top5_hits[truth] += 1
            if pred == truth:
                correct[truth] += 1
                predicted[truth] += 1
            else:
                if pred in eval_set:
                    predicted[pred] += 1
                else:
                    outside += 1
                errors[truth][pred] += 1

    present = [name for name in eval_names if support[name] > 0]
    per_gloss: dict[str, dict[str, object]] = {}
    f1_values: list[float] = []
    for name in present:
        count = int(support[name])
        hits = int(correct[name])
        precision = hits / predicted[name] if predicted[name] else 0.0
        recall = hits / count
        f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
        f1_values.append(f1)
        pair_errors = [
            {"gloss": pred, "n": count_n}
            for pred, count_n in errors[name].most_common()
        ]
        per_gloss[name] = {
            "support": count,
            "correct": hits,
            "accuracy": _round(recall),
            "top5": _round(top5_hits[name] / count),
            "precision": _round(precision),
            "recall": _round(recall),
            "f1": _round(f1),
            "loss": _round(loss_sum[name] / count),
            "errors": pair_errors,
        }

    scored = int(sum(support.values()))
    hits = int(sum(correct.values()))
    top5_total = int(sum(top5_hits.values()))
    loss_total = float(sum(loss_sum.values()))
    pairs = []
    for name in present:
        total = support[name]
        for pred, count_n in errors[name].items():
            if count_n < 2:
                continue
            pairs.append(
                {
                    "true": name,
                    "pred": pred,
                    "n": int(count_n),
                    "support": int(total),
                }
            )
    pairs.sort(key=lambda item: (-int(item["n"]), str(item["true"]), str(item["pred"])))
    return {
        "model": loaded.name,
        "id": model_key,
        "checkpoint": str(checkpoint),
        "epoch": None,
        "params": count_parameters(loaded.module),
        "is_leg": loaded.is_leg,
        "max_frames": loaded.max_frames,
        "num_classes": len(glosses),
        "scored": scored,
        "correct": hits,
        "outside": outside,
        "normalize_fail": normalize_fail,
        "loss": _round(loss_total / scored) if scored else None,
        "accuracy": _round(hits / scored) if scored else None,
        "top5": _round(top5_total / scored) if scored else None,
        "macro_f1": _round(sum(f1_values) / len(f1_values)) if f1_values else None,
        "classes": len(present),
        "f1_one": sum(1 for value in f1_values if value >= 1.0 - 1e-9),
        "f1_bins": _f1_bins(f1_values),
        "confusion_pairs": pairs,
        "glosses": per_gloss,
    }


def predictions_of(
    loaded,
    glosses: Sequence[str],
    rows: Sequence[dict[str, object]],
    features: Sequence[Optional[np.ndarray]],
    device: torch.device,
) -> dict[str, str]:
    usable = [
        (row, feature)
        for row, feature in zip(rows, features)
        if row["status"] == "ok" and feature is not None
    ]
    if not usable:
        return {}
    logits = _logits(loaded.module, [feature for _row, feature in usable], device)
    chosen = logits.argmax(dim=1).detach().cpu().tolist()
    return {str(row["video"]): glosses[int(index)] for (row, _feature), index in zip(usable, chosen)}


def agreement_table(
    guesses: dict[str, dict[str, str]],
    truths: dict[str, str],
) -> list[dict[str, object]]:
    """Pairwise top-1 agreement on videos both models scored."""
    keys = list(guesses)
    rows = []
    for left_i, left in enumerate(keys):
        for right in keys[left_i + 1 :]:
            shared = sorted(set(guesses[left]) & set(guesses[right]))
            same = both_correct = only_left = only_right = both_wrong = 0
            for video in shared:
                left_ok = guesses[left][video] == truths[video]
                right_ok = guesses[right][video] == truths[video]
                if guesses[left][video] == guesses[right][video]:
                    same += 1
                if left_ok and right_ok:
                    both_correct += 1
                elif left_ok:
                    only_left += 1
                elif right_ok:
                    only_right += 1
                else:
                    both_wrong += 1
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "shared": len(shared),
                    "same_top1": same,
                    "same_top1_rate": _round(same / len(shared)) if shared else None,
                    "both_correct": both_correct,
                    "only_left": only_left,
                    "only_right": only_right,
                    "both_wrong": both_wrong,
                }
            )
    return rows


def save_metrics(output: Path, summary: dict[str, object], per_gloss: dict[str, object]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    gloss_path = output / "per_gloss.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    gloss_path.write_text(json.dumps(per_gloss, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s", summary_path)
    logger.info("Wrote %s", gloss_path)


def run(
    keypoints_dir: Path,
    eval_labels: Path,
    checkpoint_root: Path,
    output: Path,
    min_shoulder: float,
    device: Optional[torch.device],
) -> None:
    eval_names = load_eval_glosses(eval_labels)
    clips, skipped_name = load_clips(keypoints_dir, eval_names)
    logger.info("Clips %d, eval glosses %d, skipped %d", len(clips), len(eval_names), skipped_name)

    torch_device = device or pick_device()
    logger.info("Scoring on %s", torch_device)
    checkpoints = sign_checkpoints(checkpoint_root)
    payloads = []
    guesses: dict[str, dict[str, str]] = {}
    feature_cache: dict[tuple[int, bool], list[Optional[np.ndarray]]] = {}
    ok_rows = [row for row in clips if row["status"] == "ok" and row["clip"] is not None]
    for path in checkpoints:
        key = model_id(path)
        loaded, glosses = load_sign(path, torch_device)
        spec = (loaded.max_frames, loaded.is_leg)
        if spec not in feature_cache:
            feature_cache[spec] = _featurize(
                [row["clip"] for row in ok_rows],
                loaded.max_frames,
                loaded.is_leg,
                min_shoulder,
            )
        payload = score_model(
            key,
            path,
            loaded,
            glosses,
            eval_names,
            clips,
            torch_device,
            min_shoulder,
        )
        try:
            stored = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            stored = torch.load(path, map_location="cpu")
        epoch = stored.get("epoch")
        payload["epoch"] = None if epoch is None else int(epoch)
        payloads.append(payload)
        ok_features = feature_cache[spec]
        if len(ok_features) != len(ok_rows):
            raise RuntimeError(f"{key} feature count drifted")
        guesses[key] = predictions_of(loaded, glosses, ok_rows, ok_features, torch_device)
        logger.info(
            "%s scored=%s acc=%s top5=%s macro_f1=%s",
            key,
            payload["scored"],
            payload["accuracy"],
            payload["top5"],
            payload["macro_f1"],
        )

    summary_models = []
    per_gloss_models = {}
    for payload in payloads:
        glosses = payload.pop("glosses")
        per_gloss_models[str(payload["id"])] = glosses
        summary_models.append(payload)
    summary = {
        "clips": len(clips),
        "keypoints": str(keypoints_dir),
        "eval_labels": str(eval_labels),
        "eval_glosses": len(eval_names),
        "skipped": skipped_name,
        "label_smoothing": LABEL_SMOOTHING,
        "models": summary_models,
        "agreement": agreement_table(
            guesses,
            {str(row["video"]): str(row["gloss"]) for row in clips},
        ),
    }
    per_gloss = {
        "eval_labels": str(eval_labels),
        "note": "Classes with no scored clip are omitted. A prediction outside the eval set is a miss.",
        "models": per_gloss_models,
    }
    save_metrics(output, summary, per_gloss)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score saved SignConnect keypoint clips.")
    parser.add_argument("--keypoints", type=Path, default=DEFAULT_KEYPOINTS)
    parser.add_argument("--eval-labels", type=Path, default=DEFAULT_EVAL_LABELS)
    parser.add_argument("--checkpoints", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-shoulder", type=float, default=1e-3)
    parser.add_argument("--device", default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)
    device = None if args.device is None else torch.device(args.device)
    run(
        keypoints_dir=args.keypoints,
        eval_labels=args.eval_labels,
        checkpoint_root=args.checkpoints,
        output=args.output,
        min_shoulder=args.min_shoulder,
        device=device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
