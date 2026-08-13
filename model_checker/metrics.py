"""精确率 (Precision) 与召回率 (Recall) 计算。

约定：
- 每帧模型画框数 B_i，用户填漏框 missed_i (FN)、框错 wrong_i (FP)
- TP_i = B_i - wrong_i
- Precision (micro) = Σ TP_i / Σ B_i
- Recall (micro)    = Σ TP_i / (Σ TP_i + Σ missed_i)
"""

from __future__ import annotations

from typing import Optional

from .state import FrameItem, SessionState


def _precision(tp: int, fp: int) -> Optional[float]:
    """TP/(TP+FP)，无预测时返回 None。"""
    denom = tp + fp
    if denom == 0:
        return None
    return tp / denom


def _recall(tp: int, fn: int) -> Optional[float]:
    """TP/(TP+FN)，无目标时返回 None。"""
    denom = tp + fn
    if denom == 0:
        return None
    return tp / denom


def compute_frame(frame: FrameItem) -> dict:
    """单帧指标。"""
    b = frame.box_count
    tp = frame.tp
    fp = frame.wrong
    fn = frame.missed
    return {
        "boxes": b,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": _precision(tp, fp),
        "recall": _recall(tp, fn),
    }


def compute_overall(state: SessionState) -> dict:
    """整体 micro 平均指标。"""
    tp = state.tp_total()
    fp = state.fp_total()
    fn = state.fn_total()
    boxes = state.total_boxes()
    return {
        "boxes": boxes,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": _precision(tp, fp),
        "recall": _recall(tp, fn),
    }


def fmt_ratio(v: Optional[float]) -> str:
    """格式化比例：None → 'N/A'，否则百分比保留 2 位。"""
    if v is None:
        return "N/A"
    return f"{v * 100:.2f}%"
