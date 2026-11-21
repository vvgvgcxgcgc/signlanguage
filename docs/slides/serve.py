"""Mở slide trình bày và cho nút "Bắt đầu demo" chạy demo.py trên máy này.

    python docs/slides/serve.py            # mở http://127.0.0.1:8765/slides/

Server chỉ nghe ở 127.0.0.1. Thư mục gốc là docs/, nên slide dùng thẳng
hình trong docs/bao-cao-ket-qua/bieu-do mà không cần chép.

API:
    GET  /api/demo                 trạng thái {"running": bool, "mode": str}
    POST /api/demo/start?mode=...  camera | video
    POST /api/demo/stop
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

DOCS = Path(__file__).resolve().parents[1]
ROOT = DOCS.parent
PORT = 8765

def _python() -> str:
    venv = ROOT / ".venv" / "bin" / "python"
    return str(venv) if venv.is_file() else sys.executable


def _sample_videos() -> list[str]:
    """Video mẫu cho chế độ không có webcam, tải từ dataset VSL400 public trên Kaggle nếu chưa có."""
    sys.path.insert(0, str(ROOT))
    import assets

    return [str(path) for path in assets.ensure_demo_videos()]


class DemoProcess:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self.mode = ""

    def running(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    def start(self, mode: str) -> None:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return
            command = [_python(), str(ROOT / "demo.py")]
            if mode == "video":
                videos = _sample_videos()
                if not videos:
                    raise FileNotFoundError("Không tải được video mẫu (python assets.py videos)")
                command += ["--video", *videos]
            self._proc = subprocess.Popen(command, cwd=str(ROOT))
            self.mode = mode

    def stop(self) -> None:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
            self._proc = None


DEMO = DemoProcess()


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:  # noqa: A002 - tên của lớp cha
        if "/api/" in (args[0] if args else ""):
            super().log_message(format, *args)

    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if urlparse(self.path).path == "/api/demo":
            self._json({"running": DEMO.running(), "mode": DEMO.mode})
            return
        if self.path in ("/", ""):
            self.send_response(302)
            self.send_header("Location", "/slides/")
            self.end_headers()
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        try:
            if url.path == "/api/demo/start":
                mode = parse_qs(url.query).get("mode", ["camera"])[0]
                if mode not in ("camera", "video"):
                    self._json({"error": f"mode không hợp lệ: {mode}"}, 400)
                    return
                DEMO.start(mode)
            elif url.path == "/api/demo/stop":
                DEMO.stop()
            else:
                self._json({"error": "không có đường dẫn này"}, 404)
                return
        except Exception as error:  # noqa: BLE001 - trả lỗi cho giao diện
            self._json({"error": str(error)}, 500)
            return
        self._json({"running": DEMO.running(), "mode": DEMO.mode})


def main() -> int:
    handler = partial(Handler, directory=str(DOCS))
    server = ThreadingHTTPServer(("127.0.0.1", PORT), handler)
    url = f"http://127.0.0.1:{PORT}/slides/"
    print(f"Slide: {url}  (Ctrl+C để dừng)")
    if "--no-browser" not in sys.argv:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        DEMO.stop()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
