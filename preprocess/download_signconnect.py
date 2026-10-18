"""Download the Lakeserl/SignConnect video repo into a local folder.

The Hub repo is raw mp4 files plus Text/label.csv, not a datasets table, so
load_dataset() has nothing to build. From the project root:

    python -m preprocess.download_signconnect
    python -m preprocess.download_signconnect --output data/signconnect
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from huggingface_hub import snapshot_download

logger = logging.getLogger("download_signconnect")

REPO_ID = "Lakeserl/SignConnect"
DEFAULT_OUTPUT = Path("data/signconnect")


def download(output: Path) -> Path:
    """Pull the dataset repo into `output` and return that path.

    Existing files are skipped, so a killed run can be started again.
    """
    output.mkdir(parents=True, exist_ok=True)
    logger.info("downloading %s into %s", REPO_ID, output)
    local_dir = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(output),
    )
    root = Path(local_dir)
    videos = list((root / "Videos").glob("*.mp4"))
    labels = root / "Text" / "label.csv"
    logger.info("videos=%d labels=%s", len(videos), labels.is_file())
    return root


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Download Lakeserl/SignConnect.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="folder that receives Videos/ and Text/",
    )
    args = parser.parse_args()
    download(args.output)


if __name__ == "__main__":
    main()
