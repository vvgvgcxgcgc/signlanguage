"""Demo nhận diện ngôn ngữ ký hiệu tiếng Việt, theo các lựa chọn trong báo cáo.

Mô hình mặc định là CTR-GCN J76 (docs/bao-cao-ket-qua, mục D.II):
Top-1 94,3%, ECE thấp nhất trên video khác nguồn (8,4%), phủ 93,3% ở τ = 0,7,
khoảng 31 ms mỗi ký hiệu trên CPU. Phân loại chỉ chạy một lần mỗi ký hiệu,
nên chênh lệch vài chục ms giữa các mô hình không cảm nhận được.

Các ngưỡng:
    P(action) > 0,3                  đang ký (PoseStateMLP, mục C.V)
    n_loại / (n_giữ + n_loại) >= 0,5 khung hình lỗi, yêu cầu ký lại (công thức 5)
    độ tin cậy >= 0,8                chắc chắn, tự thêm vào câu
    0,7 <= độ tin cậy < 0,8          chấp nhận, tự thêm vào câu (τ = 0,7, Bảng 14)
    0,4 <= độ tin cậy < 0,7          lưỡng lự: hiện 5 gợi ý, ký lại hoặc chọn 1-5
                                     (Hình 30: Top-5 ~98,5%)
    độ tin cậy < 0,4                 không nhận ra (thường là từ chưa học), ký lại

Số liệu CTR-GCN theo τ (reports/ket-qua-2026-11-21/derived.json),
dạng phủ/đúng VSL400, phủ/đúng QIPEDC, chấp nhận nhầm từ chưa học:
    τ 0,4: 97,8/95,4  69,0/80,5  42,7%
    τ 0,7: 93,3/96,8  51,6/86,9  15,3%
    τ 0,8: 90,1/97,5  46,4/90,6   8,9%
    τ 0,9: 78,7/98,3  25,4/87,5   4,8%  (bỏ quá nhiều nên không dùng)

    python demo.py                                  # webcam, tự tải checkpoint từ Kaggle
    python demo.py --model j76_AAGCN                # mô hình khác trong assets/manifest.json
    python assets.py videos                         # tải 8 video mẫu vào demo-video/
    python demo.py --video "demo-video/Con gấu"     # chạy trên video mẫu
    python demo.py --video demo-video --record out.mp4 --no-window

Phím: 1-5 chọn gợi ý, R ký lại, Backspace xoá từ cuối, C xoá câu, Q/Esc thoát.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional, Sequence

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import assets
from infer_camera import GestureQueue, action_probability, load_pose, load_sign
from models.factory import pick_device
from preprocess.camera import FrameGrabber, Pacer, draw_keypoints, open_capture, resolve_camera
from preprocess.dataset import PREPARE_PAD, accept_raw_frame, clip_needs_redo, pad_prepare, resample_clip, select_sign_frames
from preprocess.keypoints import TRAIN_FPS, VIDEO_SUFFIXES, KeypointExtractor, square_frame
from preprocess.pose_state import WINDOW, latest_window

logger = logging.getLogger("demo")

ROOT = Path(__file__).resolve().parent

ACTION_THRESHOLD = 0.3
TAU_REJECT = 0.4
TAU_ACCEPT = 0.7
TAU_SURE = 0.8
TOP_K = 5

# ---------------------------------------------------------------- kết quả


@dataclass
class Result:
    kind: str  # sure | accept | unsure | unknown | redo
    gloss: str = ""
    confidence: float = 0.0
    topk: list[tuple[str, float]] = field(default_factory=list)
    latency_ms: float = 0.0
    reason: str = ""


class Recognizer:
    """Một mô hình phân loại, cộng kiểm soát chất lượng và ngưỡng từ chối."""

    def __init__(self, checkpoint: Path, device: torch.device, min_shoulder: float = 1e-3) -> None:
        self.model, self.glosses = load_sign(checkpoint, device)
        self.device = device
        self.min_shoulder = min_shoulder
        # Lượt chạy đầu chậm hơn vì khởi tạo kernel, chạy trước một lần.
        dummy = torch.zeros(1, self.model.max_frames, self.model.num_joints, 4, device=device)
        with torch.no_grad():
            self.model.module(dummy)

    @property
    def name(self) -> str:
        return self.model.name

    def classify(self, clip: np.ndarray, kept: int, dropped: int, pad: bool = True) -> Result:
        if clip_needs_redo(kept, dropped):
            total = max(kept + dropped, 1)
            return Result(
                "redo",
                reason=f"Mất tay hoặc vai ở {dropped}/{total} khung. Hãy đứng giữa khung hình, đủ ánh sáng.",
            )
        if pad:
            clip = pad_prepare(clip, PREPARE_PAD)
        features = resample_clip(clip, self.model.max_frames, self.model.is_leg, self.min_shoulder)
        if features is None:
            return Result("redo", reason="Không thấy rõ hai vai. Hãy lùi lại để thấy từ đầu đến hông.")
        batch = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32)).unsqueeze(0).to(self.device)
        start = time.perf_counter()
        with torch.no_grad():
            probs = torch.softmax(self.model.module(batch), dim=1)[0]
        if self.device.type == "mps":
            torch.mps.synchronize()
        latency = (time.perf_counter() - start) * 1000
        values, indices = torch.topk(probs, TOP_K)
        topk = [(self.glosses[int(i)], float(v)) for v, i in zip(values.cpu(), indices.cpu())]
        gloss, confidence = topk[0]
        if confidence >= TAU_SURE:
            kind = "sure"
        elif confidence >= TAU_ACCEPT:
            kind = "accept"
        elif confidence >= TAU_REJECT:
            kind = "unsure"
        else:
            kind = "unknown"
        logger.info("%s %.3f (%s) %.1f ms", gloss, confidence, kind, latency)
        return Result(kind, gloss, confidence, topk, latency)


# ---------------------------------------------------------------- giao diện

W, H = 1280, 720
PAD = 24
CAM = 672  # 224 x 3, đúng khung vuông mô hình nhìn thấy
PANEL_X = PAD + CAM + PAD
PANEL_W = W - PANEL_X - PAD

C_BG = (14, 16, 22)
C_CARD = (24, 27, 36)
C_CARD_HI = (32, 36, 48)
C_LINE = (44, 49, 64)
C_TEXT = (236, 239, 245)
C_MUTED = (140, 148, 168)
C_DIM = (92, 99, 118)
C_ACCENT = (99, 140, 255)
C_GREEN = (52, 211, 153)
C_TEAL = (45, 190, 205)
C_AMBER = (251, 191, 36)
C_RED = (248, 113, 113)

KIND_STYLE = {
    "sure": ("CHẮC CHẮN", C_GREEN, "Đã thêm vào câu."),
    "accept": ("CHẤP NHẬN", C_TEAL, "Đã thêm vào câu. Bấm R nếu chưa đúng."),
    "unsure": ("LƯỠNG LỰ", C_AMBER, "Chưa đủ tin cậy. Hãy ký lại, hoặc bấm 1–5 để chọn từ đúng."),
    "unknown": ("KHÔNG NHẬN RA", C_RED, "Độ tin cậy quá thấp, có thể là từ chưa học. Hãy ký lại rõ hơn."),
    "redo": ("KÝ LẠI", C_RED, ""),
}
PHASE_STYLE = {
    "idle": ("Chờ tư thế nghỉ", C_DIM),
    "prepare": ("Sẵn sàng", C_AMBER),
    "action": ("Đang ký…", C_GREEN),
}


def _font_file(bold: bool) -> Optional[str]:
    candidates = [
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
    ]
    return next((path for path in candidates if Path(path).is_file()), None)


_FONTS: dict[tuple[int, bool], ImageFont.ImageFont] = {}


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    key = (size, bold)
    if key not in _FONTS:
        path = _font_file(bold)
        if path is None:
            _FONTS[key] = ImageFont.load_default()
        else:
            face = ImageFont.truetype(path, size)
            if path.endswith("SFNS.ttf"):
                try:
                    face.set_variation_by_name("Bold" if bold else "Regular")
                except Exception:  # noqa: BLE001 - phông không có trục biến thể
                    pass
            _FONTS[key] = face
    return _FONTS[key]


def _text_w(draw: ImageDraw.ImageDraw, text: str, face) -> int:
    return int(draw.textlength(text, font=face))


def _fit(draw: ImageDraw.ImageDraw, text: str, width: int, sizes: Sequence[int], bold: bool = True):
    for size in sizes:
        face = font(size, bold)
        if _text_w(draw, text, face) <= width:
            return face
    return font(sizes[-1], bold)


def _wrap(draw: ImageDraw.ImageDraw, text: str, face, width: int) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if line and _text_w(draw, trial, face) > width:
            lines.append(line)
            line = word
        else:
            line = trial
    if line:
        lines.append(line)
    return lines


def _chip(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, color, face=None) -> int:
    face = face or font(13, True)
    width = _text_w(draw, text, face) + 22
    tint = tuple(int(c * 0.22 + b * 0.78) for c, b in zip(color, C_CARD))
    draw.rounded_rectangle((x, y, x + width, y + 26), radius=13, fill=tint)
    draw.text((x + 11, y + 13), text, font=face, fill=color, anchor="lm")
    return width


def _bar(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, value: float, color, track=C_LINE) -> None:
    draw.rounded_rectangle((x, y, x + w, y + h), radius=h // 2, fill=track)
    filled = int(round(w * max(0.0, min(1.0, value))))
    if filled >= h:
        draw.rounded_rectangle((x, y, x + filled, y + h), radius=h // 2, fill=color)


@dataclass
class UIState:
    model_name: str
    vocab: int
    device: str
    phase: str = "idle"
    result: Optional[Result] = None
    sentence: list[str] = field(default_factory=list)
    truth: str = ""
    source: str = "Webcam"
    fps: float = 0.0


class DemoUI:
    def __init__(self) -> None:
        self._panel: Optional[np.ndarray] = None
        self._panel_key: Optional[tuple] = None
        mask = np.zeros((CAM, CAM), np.uint8)
        cv2.rectangle(mask, (18, 0), (CAM - 19, CAM - 1), 255, -1)
        cv2.rectangle(mask, (0, 18), (CAM - 1, CAM - 19), 255, -1)
        for cx, cy in ((18, 18), (CAM - 19, 18), (18, CAM - 19), (CAM - 19, CAM - 19)):
            cv2.circle(mask, (cx, cy), 18, 255, -1, cv2.LINE_AA)
        self._mask = (mask.astype(np.float32) / 255.0)[..., None]
        self._badges: dict[tuple, np.ndarray] = {}

    # Ảnh PIL dùng RGB, OpenCV dùng BGR. Mọi màu ở trên là RGB.
    @staticmethod
    def _bgr(color) -> tuple[int, int, int]:
        return (int(color[2]), int(color[1]), int(color[0]))

    def _render_panel(self, state: UIState) -> np.ndarray:
        image = Image.new("RGB", (W, H), C_BG)
        d = ImageDraw.Draw(image)
        x0, right = PANEL_X, PANEL_X + PANEL_W

        # Tiêu đề
        d.text((x0, PAD + 2), "Nhận diện Ngôn ngữ Ký hiệu", font=font(26, True), fill=C_TEXT)
        cx = x0
        for text in (state.model_name, f"{state.vocab} từ", f"τ = {TAU_ACCEPT:.1f}".replace(".", ","), state.device.upper()):
            cx += _chip(d, cx, PAD + 42, text, C_MUTED, font(12, True)) + 8

        # Các bước
        y = PAD + 86
        d.rounded_rectangle((x0, y, right, y + 64), radius=14, fill=C_CARD)
        steps = [("1", "Hạ tay nghỉ", "prepare"), ("2", "Thực hiện ký hiệu", "action"), ("3", "Hạ tay để kết thúc", "done")]
        step_w = PANEL_W // 3
        for i, (num, label, key) in enumerate(steps):
            sx = x0 + i * step_w + 16
            active = state.phase == key or (key == "prepare" and state.phase == "idle")
            color = PHASE_STYLE.get(key, ("", C_ACCENT))[1] if active else C_DIM
            if key == "prepare" and state.phase == "idle":
                color = C_MUTED
            d.ellipse((sx, y + 20, sx + 24, y + 44), fill=color if active else C_LINE)
            d.text((sx + 12, y + 32), num, font=font(13, True), fill=C_BG if active else C_MUTED, anchor="mm")
            d.text((sx + 32, y + 32), label, font=font(14, active), fill=C_TEXT if active else C_MUTED, anchor="lm")

        # Thẻ kết quả
        y = PAD + 166
        card_h = 200
        result = state.result
        d.rounded_rectangle((x0, y, right, y + card_h), radius=16, fill=C_CARD)
        inner = PANEL_W - 48
        if result is None:
            d.text((x0 + 24, y + 26), "KẾT QUẢ", font=font(13, True), fill=C_DIM)
            d.text((x0 + 24, y + 64), "Chưa có ký hiệu", font=font(38, True), fill=C_DIM)
            lines = _wrap(d, "Đứng cách camera khoảng 1–1,5 m, thấy từ đầu đến hông, hai tay buông thả. "
                             "Thực hiện ký hiệu rồi hạ tay về tư thế nghỉ.", font(15), inner)
            for i, line in enumerate(lines[:3]):
                d.text((x0 + 24, y + 122 + i * 22), line, font=font(15), fill=C_MUTED)
        else:
            label, color, hint = KIND_STYLE[result.kind]
            d.rounded_rectangle((x0, y, x0 + 6, y + card_h), radius=3, fill=color)
            _chip(d, x0 + 24, y + 20, label, color)
            if result.kind != "redo":
                d.text((right - 24, y + 33), f"{result.latency_ms:.0f} ms", font=font(13, True), fill=C_DIM, anchor="rm")
                word = result.gloss
                face = _fit(d, word, inner, (56, 48, 40, 32, 26))
                word_color = {"unsure": C_AMBER, "unknown": C_DIM}.get(result.kind, C_TEXT)
                d.text((x0 + 24, y + 92), word, font=face, fill=word_color, anchor="lm")
                # Thanh độ tin cậy, có vạch τ = 0,7 và 0,9
                by = y + 132
                bw = inner - 70
                _bar(d, x0 + 24, by, bw, 12, result.confidence, color)
                for tau in (TAU_REJECT, TAU_ACCEPT, TAU_SURE):
                    tx = x0 + 24 + int(bw * tau)
                    d.line((tx, by - 4, tx, by + 16), fill=C_TEXT, width=2)
                    d.text((tx, by + 20), f"{tau:.1f}".replace(".", ","), font=font(11), fill=C_DIM, anchor="mt")
                d.text((right - 24, by + 6), f"{result.confidence * 100:.0f}%", font=font(20, True), fill=color, anchor="rm")
                message = hint
            else:
                d.text((x0 + 24, y + 92), "Hãy ký lại", font=font(48, True), fill=C_RED, anchor="lm")
                message = result.reason
            for i, line in enumerate(_wrap(d, message, font(15), inner)[:2]):
                d.text((x0 + 24, y + 158 + i * 20), line, font=font(15), fill=C_MUTED)

        # 5 gợi ý
        y = y + card_h + 14
        sug_h = 186
        d.rounded_rectangle((x0, y, right, y + sug_h), radius=16, fill=C_CARD)
        d.text((x0 + 24, y + 18), "5 TỪ GỢI Ý", font=font(13, True), fill=C_DIM)
        pickable = result is not None and result.kind == "unsure"
        if pickable:
            d.text((right - 24, y + 18), "bấm 1–5 để chọn", font=font(13, True), fill=C_AMBER, anchor="ra")
        rows = result.topk if result is not None else []
        for i in range(TOP_K):
            ry = y + 44 + i * 28
            if i < len(rows):
                gloss, prob = rows[i]
                top = i == 0
                if top:
                    d.rounded_rectangle((x0 + 12, ry - 4, right - 12, ry + 26), radius=8, fill=C_CARD_HI)
                key_color = C_AMBER if pickable else C_DIM
                d.rounded_rectangle((x0 + 22, ry + 1, x0 + 42, ry + 21), radius=5, outline=key_color, width=1)
                d.text((x0 + 32, ry + 11), str(i + 1), font=font(12, True), fill=key_color, anchor="mm")
                name = gloss if _text_w(d, gloss, font(16, top)) < 210 else gloss[:22] + "…"
                d.text((x0 + 54, ry + 11), name, font=font(16, top), fill=C_TEXT if top else C_MUTED, anchor="lm")
                bx = x0 + 280
                bw = PANEL_W - 280 - 80
                _bar(d, bx, ry + 6, bw, 10, prob, C_ACCENT if top else (70, 86, 130))
                d.text((right - 24, ry + 11), f"{prob * 100:.1f}%", font=font(14, top), fill=C_TEXT if top else C_MUTED, anchor="rm")
            else:
                d.rounded_rectangle((x0 + 22, ry + 1, x0 + 42, ry + 21), radius=5, outline=C_LINE, width=1)
                d.rounded_rectangle((x0 + 54, ry + 6, x0 + 54 + 120 - i * 14, ry + 16), radius=5, fill=C_LINE)

        # Câu đã ghép
        y = y + sug_h + 14
        sen_h = H - PAD - 34 - y
        d.rounded_rectangle((x0, y, right, y + sen_h), radius=16, fill=C_CARD)
        d.text((x0 + 24, y + 10), "CÂU", font=font(13, True), fill=C_DIM)
        text = " · ".join(state.sentence) if state.sentence else "Các từ được chấp nhận sẽ hiện ở đây"
        face = font(20, True) if state.sentence else font(15)
        while state.sentence and _text_w(d, text, face) > inner:
            text = "… " + text[text.find(" · ", 2) + 3 :] if " · " in text[2:] else text[1:]
        d.text((x0 + 24, y + 30), text, font=face, fill=C_TEXT if state.sentence else C_DIM)

        # Phím tắt
        keys = [("1–5", "chọn gợi ý"), ("R", "ký lại"), ("⌫", "xoá từ"), ("C", "xoá câu"), ("Q", "thoát")]
        kx, ky = x0, H - PAD - 8
        for key, label in keys:
            kw = _text_w(d, key, font(12, True)) + 12
            d.rounded_rectangle((kx, ky - 10, kx + kw, ky + 10), radius=5, outline=C_LINE, width=1)
            d.text((kx + kw / 2, ky), key, font=font(12, True), fill=C_MUTED, anchor="mm")
            kx += kw + 6
            d.text((kx, ky), label, font=font(12), fill=C_DIM, anchor="lm")
            kx += _text_w(d, label, font(12)) + 14
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

    def _badge(self, text: str, color, size: int = 15) -> np.ndarray:
        key = (text, color, size)
        if key not in self._badges:
            face = font(size, True)
            probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
            w = _text_w(probe, text, face) + 40
            image = Image.new("RGB", (w, 34), (0, 0, 0))
            d = ImageDraw.Draw(image)
            d.rounded_rectangle((0, 0, w - 1, 33), radius=17, fill=(16, 18, 24))
            d.ellipse((14, 12, 24, 22), fill=color)
            d.text((32, 17), text, font=face, fill=C_TEXT, anchor="lm")
            self._badges[key] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        return self._badges[key]

    def _label(self, text: str, color, size: int = 13) -> np.ndarray:
        key = ("label", text, color, size)
        if key not in self._badges:
            face = font(size, True)
            probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
            image = Image.new("RGB", (_text_w(probe, text, face) + 4, size + 8), (0, 0, 0))
            ImageDraw.Draw(image).text((2, (size + 8) // 2), text, font=face, fill=color, anchor="lm")
            self._badges[key] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        return self._badges[key]

    @staticmethod
    def _blit(canvas: np.ndarray, patch: np.ndarray, x: int, y: int, alpha: float = 0.88) -> None:
        h, w = patch.shape[:2]
        region = canvas[y : y + h, x : x + w]
        # Pixel đen thuần là nền ngoài góc bo, giữ nguyên ảnh camera.
        inside = np.any(patch > 0, axis=2, keepdims=True)
        blended = (patch * alpha + region * (1 - alpha)).astype(np.uint8)
        region[:] = np.where(inside, blended, region)

    def render(self, square_bgr: np.ndarray, row: np.ndarray, state: UIState, probability: Optional[float]) -> np.ndarray:
        key = (
            state.phase,
            id(state.result),
            tuple(state.sentence),
            state.model_name,
            state.device,
        )
        if key != self._panel_key:
            self._panel = self._render_panel(state)
            self._panel_key = key
        canvas = self._panel.copy()

        # Camera: đúng khung 224x224 mô hình nhìn thấy, phóng to 3 lần
        cam = cv2.resize(square_bgr, (CAM, CAM), interpolation=cv2.INTER_LINEAR)
        draw_keypoints(cam, row)
        region = canvas[PAD : PAD + CAM, PAD : PAD + CAM]
        region[:] = (cam * self._mask + region * (1 - self._mask)).astype(np.uint8)
        phase_label, phase_color = PHASE_STYLE.get(state.phase, ("", C_DIM))
        if probability is None:
            phase_label, phase_color = "Không thấy người", C_RED
        border = self._bgr(phase_color)
        cv2.rectangle(canvas, (PAD - 2, PAD - 2), (PAD + CAM + 1, PAD + CAM + 1), border, 2, cv2.LINE_AA)

        self._blit(canvas, self._badge(phase_label, phase_color), PAD + 16, PAD + 16)
        src = self._badge(state.source, C_ACCENT, 13)
        self._blit(canvas, src, PAD + CAM - 16 - src.shape[1], PAD + 16)
        if state.truth:
            truth = self._badge(f"Đáp án: {state.truth}", C_MUTED, 14)
            self._blit(canvas, truth, PAD + 16, PAD + 58)

        # Thanh P(action) ở đáy khung camera, vạch ngưỡng 0,3
        mx, my, mw = PAD + 20, PAD + CAM - 40, CAM - 40
        overlay = canvas.copy()
        cv2.rectangle(overlay, (PAD, my - 26), (PAD + CAM, PAD + CAM), (22, 18, 16), -1)
        cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)
        cv2.rectangle(canvas, (mx, my), (mx + mw, my + 8), self._bgr(C_LINE), -1)
        if probability is not None:
            fill = int(mw * probability)
            color = C_GREEN if probability > ACTION_THRESHOLD else C_AMBER
            cv2.rectangle(canvas, (mx, my), (mx + fill, my + 8), self._bgr(color), -1)
        tx = mx + int(mw * ACTION_THRESHOLD)
        cv2.line(canvas, (tx, my - 5), (tx, my + 13), self._bgr(C_TEXT), 2)
        label = self._label("Xác suất đang ký", C_MUTED)
        self._blit(canvas, label, mx, my - 26, alpha=1.0)
        if probability is not None:
            cv2.putText(canvas, f"{probability:.2f}", (mx + label.shape[1] + 6, my - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self._bgr(C_TEXT), 1, cv2.LINE_AA)
        self._blit(canvas, self._label("ngưỡng 0,3", C_DIM, 11), tx - 30, my + 12, alpha=1.0)
        if state.fps:
            fps = f"{state.fps:.0f} fps"
            cv2.putText(canvas, fps, (mx + mw - 52, my - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self._bgr(C_DIM), 1, cv2.LINE_AA)
        return canvas


# ---------------------------------------------------------------- nguồn hình


def _video_files(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted(p for p in source.rglob("*") if p.suffix.lower() in VIDEO_SUFFIXES)


def video_frames(paths: Sequence[Path], glosses: set[str]) -> Iterator[tuple[Optional[np.ndarray], str]]:
    """Từng khung của từng video. Trả (None, nhãn) khi một video kết thúc.

    Nhãn là tên thư mục chứa video khi đó là một gloss, như demo-video/<gloss>/.
    """
    for path in paths:
        capture = cv2.VideoCapture(str(path))
        folder = unicodedata.normalize("NFC", path.parent.name)
        truth = folder if folder in glosses else ""
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                yield frame, truth
        finally:
            capture.release()
        yield None, truth


# ---------------------------------------------------------------- vòng lặp chính


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device) if args.device else pick_device()
    recognizer = Recognizer(args.checkpoint, device, args.min_shoulder)
    pose = load_pose(args.pose_checkpoint, device)
    extractor = KeypointExtractor(live=True)
    ui = DemoUI()
    state = UIState(model_name=recognizer.name, vocab=len(recognizer.glosses), device=device.type)
    gate = GestureQueue(ACTION_THRESHOLD)

    capture = grabber = None
    videos: Optional[Iterator] = None
    if args.video is not None:
        files = [f for source in args.video for f in _video_files(source)]
        if not files:
            raise FileNotFoundError(f"Không có video trong {args.video}")
        videos = video_frames(files, set(recognizer.glosses))
        state.source = f"Video · {len(files)} clip"
    else:
        camera = resolve_camera(args.camera)
        capture = open_capture(camera, 0, 0, TRAIN_FPS)
        grabber = FrameGrabber(capture)
        state.source = "Webcam"

    writer = None
    if args.record is not None:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        # H.264 (avc1) mở được trong trình duyệt; mp4v là phương án dự phòng.
        for codec in ("avc1", "mp4v"):
            writer = cv2.VideoWriter(str(args.record), cv2.VideoWriter_fourcc(*codec), TRAIN_FPS, (W, H))
            if writer.isOpened():
                break

    pacer = Pacer(0 if args.no_window else TRAIN_FPS)
    recent: list[np.ndarray] = []
    held_square: Optional[np.ndarray] = None
    last_square = np.zeros((224, 224, 3), np.uint8)
    last_row = np.zeros((76, 3), np.float32)
    stamps: list[float] = []
    video_rows: list[np.ndarray] = []  # mọi khung của video đang chạy
    scored = False
    if not args.no_window:
        cv2.namedWindow("VSL demo", cv2.WINDOW_AUTOSIZE)

    def finish(clip: np.ndarray) -> None:
        nonlocal scored
        scored = True
        result = recognizer.classify(clip, gate.last_kept, gate.last_dropped)
        state.result = result
        if result.kind in ("sure", "accept"):
            state.sentence.append(result.gloss)

    last_probability: Optional[float] = None

    def show(probability: Optional[float], hold: int = 1) -> bool:
        canvas = ui.render(last_square, last_row, state, probability)
        for _ in range(hold):
            if writer is not None:
                writer.write(canvas)
        if args.no_window:
            return True
        cv2.imshow("VSL demo", canvas)
        key = cv2.waitKeyEx(1 if hold == 1 else int(hold * 1000 / TRAIN_FPS))
        return handle_key(key)

    def handle_key(key: int) -> bool:
        if key < 0:
            return True
        char = key & 0xFF
        if char in (ord("q"), ord("Q"), 27):
            return False
        result = state.result
        if ord("1") <= char <= ord("5") and result is not None and result.kind == "unsure":
            index = char - ord("1")
            if index < len(result.topk):
                gloss, prob = result.topk[index]
                state.sentence.append(gloss)
                state.result = Result("accept", gloss, prob, result.topk, result.latency_ms)
        elif char in (ord("r"), ord("R")) and result is not None:
            if result.kind in ("sure", "accept") and state.sentence and state.sentence[-1] == result.gloss:
                state.sentence.pop()
            state.result = None
        elif char in (8, 127) or key in (0x7F, 0x08):
            if state.sentence:
                state.sentence.pop()
        elif char in (ord("c"), ord("C")):
            state.sentence.clear()
            state.result = None
        return True

    try:
        while True:
            pacer.wait()
            if videos is not None:
                frame, truth = next(videos, (False, ""))
                if frame is False:
                    show(last_probability, hold=int(TRAIN_FPS * 2))
                    break
                if frame is None:
                    # Video kết thúc: coi như người ký hạ tay.
                    if gate.in_action and gate.frames:
                        clip = gate._close()  # noqa: SLF001
                        state.phase = "done"
                        finish(clip)
                    elif not scored and video_rows:
                        # Clip cắt sát, không có tư thế nghỉ ở đầu: chấm cả video như infer.py.
                        kept, dropped = select_sign_frames(np.stack(video_rows), args.min_shoulder)
                        state.result = recognizer.classify(kept, int(kept.shape[0]), dropped, pad=False)
                        if state.result.kind in ("sure", "accept"):
                            state.sentence.append(state.result.gloss)
                        state.phase = "done"
                    if not show(last_probability, hold=int(TRAIN_FPS * 1.6)):
                        break
                    gate = GestureQueue(ACTION_THRESHOLD)
                    recent.clear()
                    video_rows.clear()
                    scored = False
                    held_square = None
                    continue
                state.truth = truth
            else:
                frame = grabber.read()
                if frame is None:
                    logger.error("Camera ngừng gửi khung hình")
                    break

            stamps.append(time.monotonic())
            if len(stamps) > 30:
                del stamps[0]
                state.fps = (len(stamps) - 1) / (stamps[-1] - stamps[0])

            square = square_frame(frame)
            # Video: chạy tay ở mọi khung để còn chấm cả clip nếu không tách được.
            row = extractor.process(square, hands=videos is not None or gate.in_action or gate.prepare_rejected)
            if videos is not None:
                video_rows.append(row.copy())
            recent.append(row)
            if len(recent) > WINDOW:
                del recent[0]
            window = latest_window(np.stack(recent))
            probability = None if window is None else action_probability(pose, window, device)
            opening = not gate.in_action and probability is not None and probability > gate.threshold and bool(gate.frames)
            if opening:
                if held_square is not None:
                    extractor.fill_hands(held_square, gate.frames[0])
                extractor.fill_hands(square, row)
                if not accept_raw_frame(gate.frames[0], args.min_shoulder, check_hands=True):
                    gate.discard_prepare()
            check_hands = gate.in_action or gate.prepare_rejected or opening
            accept = accept_raw_frame(row, args.min_shoulder, check_hands=check_hands)
            finished = gate.step(row, probability, accept)
            held_square = None if gate.in_action else square.copy()
            state.phase = gate.phase
            if finished is not None:
                state.phase = "done"
                finish(finished)
            last_square, last_row, last_probability = square, row, probability
            if not show(probability):
                break
    finally:
        if grabber is not None:
            grabber.close()
        if capture is not None:
            capture.release()
        extractor.close()
        if writer is not None:
            writer.release()
            logger.info("Đã ghi %s", args.record)
        cv2.destroyAllWindows()


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo nhận diện ngôn ngữ ký hiệu tiếng Việt.")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Tệp .pth của mô hình phân loại. Mặc định tải j76_CTR-GCN từ Kaggle.")
    parser.add_argument("--model", default=assets.DEFAULT_MODEL,
                        help="Tên thử nghiệm trong huynguang/vsl-sign-checkpoints, dùng khi không có --checkpoint.")
    parser.add_argument("--pose-checkpoint", type=Path, default=None, help="Mặc định tải từ Kaggle.")
    parser.add_argument("--video", type=Path, nargs="+", default=None, help="Tệp hoặc thư mục video thay cho webcam.")
    parser.add_argument("--camera", type=int, default=None, help="Chỉ số camera. Mặc định camera tích hợp của Mac.")
    parser.add_argument("--record", type=Path, default=None, help="Ghi giao diện ra tệp .mp4.")
    parser.add_argument("--no-window", action="store_true", help="Không mở cửa sổ (dùng cùng --record).")
    parser.add_argument("--min-shoulder", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu", help="cpu, cuda hoặc mps. Báo cáo đo trên CPU.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    args = _parse_args(argv)
    if args.checkpoint is None:
        args.checkpoint = assets.checkpoint(args.model)
    if args.pose_checkpoint is None:
        args.pose_checkpoint = assets.pose_checkpoint()
    for path in (args.checkpoint, args.pose_checkpoint):
        if not path.is_file():
            logger.error("Thiếu checkpoint %s", path)
            return 2
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
