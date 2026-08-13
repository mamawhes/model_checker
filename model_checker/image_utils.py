"""图像工具：cv2 BGR ↔ QImage 转换、画检测框、缩略图生成。"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from PySide6.QtGui import QImage, QPixmap

from .state import Detection

# 固定颜色调色板（BGR），按类别索引取色
PALETTE = [
    (0, 0, 255),      # 红
    (0, 255, 0),      # 绿
    (255, 0, 0),      # 蓝
    (0, 255, 255),    # 黄
    (255, 0, 255),    # 品红
    (255, 255, 0),    # 青
    (0, 128, 255),    # 橙
    (128, 0, 255),    # 紫
]


def _color(class_id: int) -> tuple[int, int, int]:
    return PALETTE[class_id % len(PALETTE)]


def class_label(class_id: int, class_names: Optional[list[str]]) -> str:
    if class_names and 0 <= class_id < len(class_names):
        return class_names[class_id]
    return str(class_id)


def cv2_to_qimage(img_bgr: np.ndarray) -> QImage:
    """BGR ndarray → QImage (RGB888)。"""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()


def cv2_to_qpixmap(img_bgr: np.ndarray) -> QPixmap:
    return QPixmap.fromImage(cv2_to_qimage(img_bgr))


def draw_detections(
    img_bgr: np.ndarray,
    detections: list[Detection],
    class_names: Optional[list[str]] = None,
    thickness: int = 2,
) -> np.ndarray:
    """在图片副本上画检测框，返回新 ndarray。"""
    out = img_bgr.copy()
    for det in detections:
        color = _color(det.class_id)
        x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        label = f"{class_label(det.class_id, class_names)}:{det.score:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_text = max(0, y1 - th - baseline - 2)
        cv2.rectangle(
            out,
            (x1, y_text),
            (x1 + tw + 4, y_text + th + baseline + 2),
            color,
            -1,
        )
        cv2.putText(
            out,
            label,
            (x1 + 2, y_text + th),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def make_thumbnail(img_bgr: np.ndarray, size: int = 160) -> QPixmap:
    """等比缩放生成缩略图 QPixmap。"""
    h, w = img_bgr.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    thumb = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return cv2_to_qpixmap(thumb)


def load_image_bgr(path: str):
    """读图（支持中文路径），失败返回 None。"""
    try:
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None
