"""Tải checkpoint, keypoint và video mẫu từ Kaggle.

Cả ba dataset đều public, nên tải bằng HTTP thường, không cần tài khoản hay token:

    huynguang/vsl-sign-checkpoints               checkpoint mọi thử nghiệm, nhãn, pose-state, keypoint QIPEDC
    huynguang/vsl400-keypoints-final             keypoint VSL400 (train/test), 1,2 GB
    nguyenanfms/vsl-vietnamese-sign-language-v2  video VSL400 gốc (chỉ lấy vài clip cho demo)

Dùng từ dòng lệnh, ở thư mục gốc của repo:

    python assets.py                 # mọi checkpoint (khoảng 670 MB)
    python assets.py demo            # chỉ những gì demo cần: CTR-GCN J76 + pose-state, khoảng 8 MB
    python assets.py videos          # 8 video mẫu cho slide/demo không có webcam
    python assets.py keypoints       # keypoint VSL400, cần để huấn luyện hoặc chấm lại
    python assets.py all

Trong code: `assets.checkpoint("j76_CTR-GCN")` trả về đường dẫn .pth và chỉ tải đúng tệp đó nếu chưa có.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
import unicodedata
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

logger = logging.getLogger("assets")

ROOT = Path(__file__).resolve().parent
API = "https://www.kaggle.com/api/v1/datasets/download"

CHECKPOINT_DATASET = "huynguang/vsl-sign-checkpoints"
CHECKPOINT_VERSION = 2   # ghim phiên bản để kết quả tái lập được
KEYPOINT_DATASET = "huynguang/vsl400-keypoints-final"
VIDEO_DATASET = "nguyenanfms/vsl-vietnamese-sign-language-v2"

ASSETS = ROOT / "assets"                 # nơi giải nén huynguang/vsl-sign-checkpoints
KEYPOINTS = ROOT / "vsl400-keypoint"     # train/ và test/
DEMO_VIDEOS = ROOT / "demo-video"        # <từ>/<id>.mp4

DEFAULT_MODEL = "j76_CTR-GCN"
# Một clip tập test cho mỗi từ, dùng khi trình chiếu không có webcam.
SAMPLE_CLIPS = {
    "Bánh bao": "002802", "Con gấu": "152727", "Hôi": "001414", "Kẹo": "003662",
    "Mỹ": "000648", "Ăn mừng": "158655", "Cắm trại": "002470", "Ngoan": "000835",
}


def _url(dataset: str, file: str | None = None, version: int | None = None) -> str:
    url = f"{API}/{dataset}"
    if file is not None:
        # Kaggle lưu tên tệp dạng NFC; macOS hay đưa ra NFD.
        url += "/" + urllib.parse.quote(unicodedata.normalize("NFC", file), safe="")
    return url if version is None else f"{url}?datasetVersionNumber={version}"


def _fetch(url: str, dest: Path) -> Path:
    """Tải `url` về `dest` qua một tệp tạm, để lần chạy bị ngắt không để lại tệp hỏng."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "vsl-sign-assets"})
    with urllib.request.urlopen(request, timeout=120) as response, \
            tempfile.NamedTemporaryFile(dir=dest.parent, delete=False, suffix=".part") as tmp:
        total = int(response.headers.get("Content-Length") or 0)
        done, step = 0, 0
        while chunk := response.read(1 << 20):
            tmp.write(chunk)
            done += len(chunk)
            if total > 20e6 and done * 10 // total > step:
                step = done * 10 // total
                logger.info("  %3d%%  %.0f/%.0f MB", step * 10, done / 1e6, total / 1e6)
    Path(tmp.name).replace(dest)
    return dest


def _download_dataset(dataset: str, target: Path) -> None:
    logger.info("Tải %s vào %s", dataset, target)
    target.mkdir(parents=True, exist_ok=True)
    archive = _fetch(_url(dataset), target / ".download.zip")
    try:
        with zipfile.ZipFile(archive) as zf:
            for member in zf.infolist():
                # Chặn đường dẫn thoát ra ngoài thư mục đích.
                path = (target / member.filename).resolve()
                if target.resolve() not in path.parents and path != target.resolve():
                    raise ValueError(f"đường dẫn lạ trong tệp nén: {member.filename}")
            zf.extractall(target)
    finally:
        archive.unlink(missing_ok=True)


# ---------------------------------------------------------------- checkpoint


def asset(relative: str) -> Path:
    """Một tệp của huynguang/vsl-sign-checkpoints trong assets/, chỉ tải tệp đó nếu chưa có."""
    path = ASSETS / relative
    if not path.is_file():
        logger.info("Tải %s/%s", CHECKPOINT_DATASET, relative)
        _fetch(_url(CHECKPOINT_DATASET, relative, CHECKPOINT_VERSION), path)
    return path


def ensure_checkpoints() -> Path:
    """Tải đủ mọi tệp của dataset checkpoint (khoảng 670 MB) vào assets/."""
    files = json.loads(asset("files.json").read_text(encoding="utf8"))
    for relative in files:
        asset(relative)
    return ASSETS


def manifest() -> dict:
    """Thông tin từng thử nghiệm: mô hình, số khớp, epoch, macro-F1, notebook Kaggle sinh ra nó."""
    return json.loads(asset("manifest.json").read_text(encoding="utf8"))


def checkpoint(experiment: str = DEFAULT_MODEL, kind: str = "best") -> Path:
    """Đường dẫn `<experiment>_<kind>.pth`. kind = best (suy luận) hoặc last (huấn luyện tiếp)."""
    info = manifest()
    if experiment not in info:
        raise KeyError(f"không có thử nghiệm {experiment}. Các thử nghiệm: {', '.join(sorted(info))}")
    if kind == "last" and "last_epoch" not in info[experiment]:
        raise FileNotFoundError(f"{experiment} không lưu _last.pth trên Kaggle")
    return asset(f"checkpoints/{experiment}_{kind}.pth")


def checkpoint_dir(kind: str = "best") -> Path:
    """assets/checkpoints với mọi tệp `_<kind>.pth` đã được tải."""
    for experiment in manifest():
        if kind == "best" or "last_epoch" in manifest()[experiment]:
            checkpoint(experiment, kind)
    return ASSETS / "checkpoints"


def labels(experiment: str = DEFAULT_MODEL) -> list[str]:
    data = json.loads(asset(f"labels/{experiment}.json").read_text(encoding="utf8"))
    return data["labels"] if isinstance(data, dict) else data


def label_file(name: str) -> Path:
    return asset(f"labels/{name}")


def pose_checkpoint() -> Path:
    return asset("pose_state_mlp_best.pth")


def signconnect_keypoints() -> Path:
    """Keypoint MediaPipe của các clip QIPEDC dùng cho phép thử khác miền."""
    files = json.loads(asset("files.json").read_text(encoding="utf8"))
    for relative in files:
        if relative.startswith("signconnect-keypoints/"):
            asset(relative)
    return ASSETS / "signconnect-keypoints"


# ---------------------------------------------------------------- dữ liệu


def ensure_keypoints() -> Path:
    """vsl400-keypoint/{train,test}/<từ>/<id>.npy, tải nếu chưa có (1,2 GB)."""
    # Dấu .complete: vsl400_keypoint() có thể đã tải lẻ vài tệp vào cùng thư mục.
    if not (KEYPOINTS / ".complete").is_file():
        _download_dataset(KEYPOINT_DATASET, ROOT)
        (KEYPOINTS / ".complete").touch()
    return KEYPOINTS


def _gloss_dir(gloss: str) -> Path:
    """Thư mục của từ, khớp cả tên NFC lẫn NFD (macOS hay lưu NFD)."""
    want = unicodedata.normalize("NFC", gloss)
    if DEMO_VIDEOS.is_dir():
        for d in DEMO_VIDEOS.iterdir():
            if unicodedata.normalize("NFC", d.name) == want:
                return d
    return DEMO_VIDEOS / want


def vsl400_video(gloss: str, clip: str, split: str = "test") -> Path:
    """Một video VSL400 gốc (224×224), tải riêng tệp đó vào data/vsl400-videos/ nếu chưa có."""
    gloss = unicodedata.normalize("NFC", gloss)
    path = ROOT / "data" / "vsl400-videos" / split / gloss / f"{clip}.mp4"
    if not path.is_file():
        logger.info("Tải video %s/%s/%s.mp4", split, gloss, clip)
        _fetch(_url(VIDEO_DATASET, f"processed/processed/frame_splited/{split}/{gloss}/{clip}.mp4"), path)
    return path


def vsl400_keypoint(split: str, gloss: str, clip: str) -> Path:
    """Một tệp keypoint vsl400-keypoint/<split>/<từ>/<id>.npy, chỉ tải tệp đó."""
    gloss = unicodedata.normalize("NFC", gloss)
    path = KEYPOINTS / split / gloss / f"{clip}.npy"
    if not path.is_file():
        _fetch(_url(KEYPOINT_DATASET, f"vsl400-keypoint/{split}/{gloss}/{clip}.npy"), path)
    return path


def ensure_demo_videos(clips: dict[str, str] | None = None) -> list[Path]:
    """Một video tập test cho mỗi từ trong `clips`, tải riêng từng tệp từ dataset video gốc.

    Video VSL400 không nằm trong repo: tải về máy để chạy demo, không phát tán lại.
    """
    paths = []
    for gloss, clip in (clips or SAMPLE_CLIPS).items():
        path = _gloss_dir(gloss) / f"{clip}.mp4"
        if not path.is_file():
            logger.info("Tải video mẫu %s/%s.mp4", gloss, clip)
            remote = f"processed/processed/frame_splited/test/{gloss}/{clip}.mp4"
            _fetch(_url(VIDEO_DATASET, remote), path)
        paths.append(path)
    return paths


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser(description="Tải dữ liệu public của dự án từ Kaggle.")
    parser.add_argument("what", nargs="*", default=["checkpoints"],
                        choices=["checkpoints", "demo", "videos", "keypoints", "all"])
    parser.add_argument("--force", action="store_true", help="Xoá assets/ và tải lại.")
    args = parser.parse_args(argv)
    what = {"checkpoints", "videos", "keypoints"} if "all" in args.what else set(args.what)
    if args.force and ASSETS.exists():
        shutil.rmtree(ASSETS)
    if "demo" in what:
        logger.info("Demo: %s, %s", checkpoint(), pose_checkpoint())
    if "checkpoints" in what:
        ensure_checkpoints()
        info = manifest()
        logger.info("%d thử nghiệm trong %s:", len(info), ASSETS)
        for name, entry in info.items():
            f1 = entry.get("test_macro_f1")
            logger.info("  %-26s %-22s epoch %-3s macro-F1 %s", name, entry["model"], entry["best_epoch"],
                        f"{f1:.4f}" if f1 is not None else "-")
    if "videos" in what:
        logger.info("%d video mẫu trong %s", len(ensure_demo_videos()), DEMO_VIDEOS)
    if "keypoints" in what:
        root = ensure_keypoints()
        logger.info("keypoint: %d từ (train), %d từ (test) trong %s",
                    sum(1 for _ in (root / "train").iterdir()), sum(1 for _ in (root / "test").iterdir()), root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
