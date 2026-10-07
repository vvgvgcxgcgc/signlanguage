"""Draw raw keypoints on a webcam frame.

Each camera frame is center-cropped and resized to the 224x224 square the
training videos use before MediaPipe sees it. The window shows that square,
enlarged, so the framing on screen is the framing the model gets. Stand so
the view runs from the head to the hips, with the hands resting at the sides,
as in the training clips. The sign loop lives in infer_camera.

    python -m preprocess.camera
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from models.graph import BODY_BONES, CROSS_BONES, HAND_BONES, LEG_BONES
from preprocess.keypoints import FRAME_SIZE, LANDMARKS, N_JOINTS, TRAIN_FPS, KeypointExtractor, square_frame

logger = logging.getLogger(__name__)

_WINDOW = "sign"
DISPLAY_SCALE = 3
_NAME_INDEX = {name: index for index, name in enumerate(LANDMARKS)}
_EDGES: list[tuple[int, int, tuple[int, int, int]]] = []
for _start, _end in BODY_BONES + LEG_BONES:
    _EDGES.append((_NAME_INDEX[_start], _NAME_INDEX[_end], (80, 220, 120)))
for _suffix, _color in (("_0", (80, 180, 255)), ("_1", (255, 170, 70))):
    for _start, _end in HAND_BONES:
        _EDGES.append((_NAME_INDEX[_start + _suffix], _NAME_INDEX[_end + _suffix], _color))
for _start, _end in CROSS_BONES:
    _EDGES.append((_NAME_INDEX[_start], _NAME_INDEX[_end], (220, 220, 220)))

_PHASE_COLOR = {
    "idle": (160, 160, 170),
    "prepare": (240, 200, 80),
    "action": (80, 220, 120),
}


def draw_keypoints(frame_bgr: np.ndarray, row: np.ndarray) -> None:
    """Draw pose and both hands. A joint at (0, 0, 0) was not detected."""
    if row.shape != (N_JOINTS, 3):
        raise ValueError(f"expected ({N_JOINTS}, 3), got {row.shape}")
    height, width = frame_bgr.shape[:2]
    points = []
    for joint in row:
        if joint[0] == 0.0 and joint[1] == 0.0 and joint[2] == 0.0:
            points.append(None)
            continue
        x = int(np.clip(joint[0] * width, 0, width - 1))
        y = int(np.clip(joint[1] * height, 0, height - 1))
        points.append((x, y))
    for start, end, color in _EDGES:
        if points[start] is None or points[end] is None:
            continue
        cv2.line(frame_bgr, points[start], points[end], color, 2, cv2.LINE_AA)
    for point in points:
        if point is None:
            continue
        cv2.circle(frame_bgr, point, 3, (240, 240, 240), -1, cv2.LINE_AA)


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def paint_status(frame_bgr: np.ndarray, phase: str, detail: str, gloss: str) -> None:
    """Paint the phase banner. `phase` is idle, prepare, or action."""
    image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    width = frame_bgr.shape[1]
    draw.rectangle((0, 0, width, 78), fill=(20, 20, 24))
    color = _PHASE_COLOR.get(phase, (255, 255, 255))
    title = phase if not detail else f"{phase}  {detail}"
    draw.text((12, 8), title, font=_font(28), fill=color)
    if gloss:
        draw.text((12, 44), gloss, font=_font(24), fill=(255, 255, 255))
    frame_bgr[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


_BUILTIN = "AVCaptureDeviceTypeBuiltInWideAngleCamera"


def _capture_api() -> int:
    return cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY


def _objc_msg(restype, argtypes, *args):
    objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
    return ctypes.CFUNCTYPE(restype, *argtypes)(("objc_msgSend", objc))(*args)


def builtin_camera_index() -> int:
    """OpenCV index of the built-in Mac camera.

    OpenCV sorts AVFoundation devices by uniqueID, so an iPhone Continuity
    Camera sorts ahead of the Mac and steals index 0.
    """
    if sys.platform != "darwin":
        return 0
    objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
    objc.objc_getClass.restype = ctypes.c_void_p
    objc.objc_getClass.argtypes = [ctypes.c_char_p]
    objc.sel_registerName.restype = ctypes.c_void_p
    objc.sel_registerName.argtypes = [ctypes.c_char_p]

    def sel(name: str):
        return objc.sel_registerName(name.encode())

    def text(nsstring) -> str:
        raw = _objc_msg(ctypes.c_char_p, [ctypes.c_void_p, ctypes.c_void_p], nsstring, sel("UTF8String"))
        return raw.decode() if raw else ""

    avfoundation = ctypes.CDLL("/System/Library/Frameworks/AVFoundation.framework/AVFoundation")
    video = ctypes.c_void_p.in_dll(avfoundation, "AVMediaTypeVideo")
    muxed = ctypes.c_void_p.in_dll(avfoundation, "AVMediaTypeMuxed")
    device_cls = objc.objc_getClass(b"AVCaptureDevice")
    send3 = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    video_devices = _objc_msg(ctypes.c_void_p, send3, device_cls, sel("devicesWithMediaType:"), video)
    muxed_devices = _objc_msg(ctypes.c_void_p, send3, device_cls, sel("devicesWithMediaType:"), muxed)
    devices = _objc_msg(ctypes.c_void_p, send3, video_devices, sel("arrayByAddingObjectsFromArray:"), muxed_devices)
    count = _objc_msg(ctypes.c_uint64, [ctypes.c_void_p, ctypes.c_void_p], devices, sel("count"))
    rows = []
    for index in range(count):
        item = _objc_msg(
            ctypes.c_void_p,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64],
            devices,
            sel("objectAtIndex:"),
            index,
        )
        name = text(_objc_msg(ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_void_p], item, sel("localizedName")))
        uid = text(_objc_msg(ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_void_p], item, sel("uniqueID")))
        kind = text(_objc_msg(ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_void_p], item, sel("deviceType")))
        rows.append((uid, name, kind))
    for index, (_uid, name, kind) in enumerate(sorted(rows)):
        if kind == _BUILTIN:
            logger.info("Using built-in camera %s at index %s", name, index)
            return index
    raise RuntimeError("No built-in Mac camera was found")


def resolve_camera(camera: Optional[int]) -> int:
    """None selects the built-in Mac camera. An explicit index is used as given."""
    if camera is None:
        return builtin_camera_index()
    return int(camera)


def read_frame(
    capture: cv2.VideoCapture,
    camera: int,
    attempts: int = 45,
    quiet: bool = False,
) -> Optional[np.ndarray]:
    """Grab one frame, retrying while the camera is still starting."""
    for _ in range(attempts):
        ok, frame = capture.read()
        if ok and frame is not None and frame.size > 0:
            return frame
        time.sleep(0.03)
    if not quiet:
        logger.error(
            "Camera %s returned no frame. Grant Camera access to this terminal in "
            "System Settings > Privacy & Security > Camera, and close anything else using it.",
            camera,
        )
    return None


def open_capture(camera: int, width: int, height: int, fps: float = TRAIN_FPS) -> cv2.VideoCapture:
    """Open one camera index and wait until it delivers a frame.

    `fps` is a request. The camera may round it, so check the logged rate.
    """
    capture = cv2.VideoCapture(camera, _capture_api())
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open camera {camera}")
    if width > 0:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height > 0:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps > 0:
        capture.set(cv2.CAP_PROP_FPS, fps)
    frame = read_frame(capture, camera, attempts=45, quiet=True)
    if frame is None:
        capture.release()
        raise RuntimeError(f"Camera {camera} delivered no frame")
    logger.info(
        "Camera %s opened at %dx%d, %.1f fps requested, %.1f reported",
        camera,
        frame.shape[1],
        frame.shape[0],
        fps,
        capture.get(cv2.CAP_PROP_FPS),
    )
    return capture


class FrameGrabber:
    """Read the camera on a background thread and keep only the newest frame.

    A blocking read waits for the next exposure, about 33 ms at 30 fps, on top
    of the landmark time. Grabbing in the background removes that wait.
    """

    def __init__(self, capture: cv2.VideoCapture) -> None:
        self._capture = capture
        self._lock = threading.Condition()
        self._frame: Optional[np.ndarray] = None
        self._sequence = 0
        self._taken = 0
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            ok, frame = self._capture.read()
            if not ok or frame is None or frame.size == 0:
                time.sleep(0.005)
                continue
            with self._lock:
                self._frame = frame
                self._sequence += 1
                self._lock.notify_all()

    def read(self, timeout: float = 2.0) -> Optional[np.ndarray]:
        """Newest frame not returned before. None when nothing new arrives in `timeout`."""
        with self._lock:
            arrived = self._lock.wait_for(lambda: self._sequence > self._taken, timeout=timeout)
            if not arrived:
                return None
            self._taken = self._sequence
            return self._frame

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)


class Pacer:
    """Hold the loop at a fixed rate. A slow step is not made up later."""

    def __init__(self, fps: float) -> None:
        self.period = 1.0 / fps if fps > 0 else 0.0
        self._next = time.monotonic()

    def wait(self) -> None:
        if self.period <= 0:
            return
        now = time.monotonic()
        if now < self._next:
            time.sleep(self._next - now)
            self._next += self.period
        else:
            self._next = now + self.period


def display_frame(square_bgr: np.ndarray, row: np.ndarray) -> np.ndarray:
    """Enlarge the 224 model frame for the window and draw the skeleton on it."""
    size = FRAME_SIZE * DISPLAY_SCALE
    shown = cv2.resize(square_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    draw_keypoints(shown, row)
    return shown


def preview(camera: Optional[int] = None, width: int = 0, height: int = 0, fps: float = TRAIN_FPS) -> None:
    """Show the skeleton on the 224 square the model sees. No model runs."""
    camera = resolve_camera(camera)
    extractor = KeypointExtractor(live=True)
    capture = open_capture(camera, width, height, fps)
    grabber = FrameGrabber(capture)
    pacer = Pacer(fps)
    try:
        while True:
            pacer.wait()
            frame = grabber.read()
            if frame is None:
                logger.error("Camera %s stopped delivering frames", camera)
                break
            square = square_frame(frame)
            row = extractor.process(square)
            present = int(np.count_nonzero(np.any(row != 0, axis=1)))
            shown = display_frame(square, row)
            paint_status(shown, "idle", f"joints={present}", "")
            cv2.imshow(_WINDOW, shown)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        grabber.close()
        extractor.close()
        capture.release()
        cv2.destroyAllWindows()
        logger.info("Camera %s closed", camera)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview raw webcam keypoints.")
    parser.add_argument("--camera", type=int, default=None, help="Index. Default is the built-in Mac camera.")
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--fps", type=float, default=TRAIN_FPS, help="Requested camera rate. Training videos are 25.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv)
    preview(camera=args.camera, width=args.width, height=args.height, fps=args.fps)


if __name__ == "__main__":
    main()
