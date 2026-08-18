"""会话状态：贯穿向导五步的共享数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Detection:
    """单个检测框（原图坐标，像素）。"""

    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int
    score: float


@dataclass
class FrameItem:
    """一帧图片及其检测结果与人工标注。"""

    path: str
    detections: list[Detection] = field(default_factory=list)
    missed: int = 0  # 漏框数 (FN)
    wrong: int = 0  # 框错数 (FP)

    @property
    def box_count(self) -> int:
        return len(self.detections)

    @property
    def tp(self) -> int:
        """真正例 = 模型框数 - 框错数。"""
        return max(0, self.box_count - self.wrong)

    @property
    def reviewed(self) -> bool:
        # 进入审核页即视为已审；用 missed/wrong 字段是否被设置过判断不可靠，
        # 这里默认只要审核页遍历过即填值（0 也是合法值）。
        return True


@dataclass
class SessionState:
    """整个会话的共享状态。"""

    model_path: str = ""
    names_path: str = ""
    class_names: Optional[list[str]] = None
    video_path: str = ""
    # 图片目录模式：非空时跳过抽帧步骤，直接用目录内的图片作为帧来源
    image_dir: str = ""
    frame_count_target: int = 20
    filter_detected: bool = False  # 抽帧时仅保留检测到目标的帧
    conf: float = 0.25
    iou: float = 0.45
    imgsz: int = 640
    frames: list[FrameItem] = field(default_factory=list)
    # 标记推理是否已完成
    inference_done: bool = False
    # 标记审核页已访问的帧索引
    reviewed_indices: set[int] = field(default_factory=set)

    def reset(self) -> None:
        """重置会话（保留模型/参数选择以便快速重跑）。"""
        self.frames.clear()
        self.inference_done = False
        self.reviewed_indices.clear()

    def total_boxes(self) -> int:
        return sum(f.box_count for f in self.frames)

    def tp_total(self) -> int:
        return sum(f.tp for f in self.frames)

    def fp_total(self) -> int:
        return sum(f.wrong for f in self.frames)

    def fn_total(self) -> int:
        return sum(f.missed for f in self.frames)
