"""YOLOv8 / YOLOv11 ONNX 推理封装。

模型输出格式：[1, 4+nc, N]（nc 为类别数），无 objectness。
- 前 4 维为 cx, cy, w, h（输入图坐标）
- 后 nc 维为各类别分数
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort

from .state import Detection


def letterbox(
    img: np.ndarray, imgsz: int, color: tuple[int, int, int] = (114, 114, 114)
) -> tuple[np.ndarray, float, float, float]:
    """等比缩放 + 居中填充。返回 (out, ratio, pad_w, pad_h)。"""
    h, w = img.shape[:2]
    ratio = min(imgsz / h, imgsz / w)
    new_w, new_h = int(round(w * ratio)), int(round(h * ratio))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    out = np.full((imgsz, imgsz, 3), color, dtype=np.uint8)
    pad_w = (imgsz - new_w) // 2
    pad_h = (imgsz - new_h) // 2
    out[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized
    return out, ratio, float(pad_w), float(pad_h)


def _xywh2xyxy(xywh: np.ndarray) -> np.ndarray:
    x, y, w, h = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
    return np.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], axis=1)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    """单类别 NMS，返回保留索引。boxes 为 xyxy。"""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.clip((x2 - x1) * (y2 - y1), 0, None)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.clip(xx2 - xx1, 0, None)
        h = np.clip(yy2 - yy1, 0, None)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = np.where(union > 0, inter / union, 0.0)
        inds = np.where(iou <= iou_thr)[0]
        order = order[inds + 1]
    return keep


class YOLOv8Detector:
    """YOLOv8 ONNX 推理器。"""

    def __init__(
        self,
        model_path: str,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        providers: Optional[list[str]] = None,
    ) -> None:
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        providers = providers or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(
            model_path, providers=providers
        )
        self.input_name = self.session.get_inputs()[0].name
        in_shape = self.session.get_inputs()[0].shape  # [1, 3, H, W]
        # 以模型实际输入尺寸为准，覆盖传入的 imgsz
        if len(in_shape) == 4 and isinstance(in_shape[2], int):
            self.imgsz = int(in_shape[2])
        out_shape = self.session.get_outputs()[0].shape  # [1, 4+nc, N]
        self.nc = int(out_shape[1]) - 4

    def _preprocess(self, img_bgr: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        lb, ratio, pad_w, pad_h = letterbox(img_bgr, self.imgsz)
        rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
        chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = chw[np.newaxis, ...]  # [1,3,H,W]
        return blob, ratio, pad_w, pad_h

    def _postprocess(
        self,
        out: np.ndarray,
        ratio: float,
        pad_w: float,
        pad_h: float,
        orig_h: int,
        orig_w: int,
    ) -> list[Detection]:
        # out: [1, 4+nc, N] -> [N, 4+nc]
        pred = out[0].T
        if pred.shape[0] == 0:
            return []
        boxes_xywh = pred[:, :4]
        scores = pred[:, 4:]  # [N, nc]
        class_ids = scores.argmax(axis=1)
        max_scores = scores.max(axis=1)

        # 置信度过滤
        mask = max_scores >= self.conf
        boxes_xywh = boxes_xywh[mask]
        class_ids = class_ids[mask]
        max_scores = max_scores[mask]
        if len(boxes_xywh) == 0:
            return []

        boxes_xyxy = _xywh2xyxy(boxes_xywh)

        # 按类别分组 NMS
        keep_all: list[int] = []
        for c in np.unique(class_ids):
            idxs = np.where(class_ids == c)[0]
            k = _nms(boxes_xyxy[idxs], max_scores[idxs], self.iou)
            keep_all.extend(idxs[k].tolist())

        detections: list[Detection] = []
        for k in keep_all:
            x1, y1, x2, y2 = boxes_xyxy[k]
            # 反 letterbox：减 pad 后除 ratio，并裁剪到原图范围
            x1 = (x1 - pad_w) / ratio
            y1 = (y1 - pad_h) / ratio
            x2 = (x2 - pad_w) / ratio
            y2 = (y2 - pad_h) / ratio
            x1 = float(np.clip(x1, 0, orig_w - 1))
            y1 = float(np.clip(y1, 0, orig_h - 1))
            x2 = float(np.clip(x2, 0, orig_w - 1))
            y2 = float(np.clip(y2, 0, orig_h - 1))
            detections.append(
                Detection(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    class_id=int(class_ids[k]),
                    score=float(max_scores[k]),
                )
            )
        # 按分数降序
        detections.sort(key=lambda d: d.score, reverse=True)
        return detections

    def detect(self, img_bgr: np.ndarray) -> list[Detection]:
        h, w = img_bgr.shape[:2]
        blob, ratio, pad_w, pad_h = self._preprocess(img_bgr)
        out = self.session.run(None, {self.input_name: blob})[0]
        return self._postprocess(out, ratio, pad_w, pad_h, h, w)
