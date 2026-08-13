"""视频均匀抽帧。"""

from __future__ import annotations

import os
import tempfile
import time
from typing import Callable, Optional

import cv2
import numpy as np

ProgressCb = Callable[[int, int, str], None]


def get_video_info(path: str) -> dict:
    """获取视频基本信息。"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {path}")
    info = {
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS) or 0.0,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    if info["fps"] > 0:
        info["duration"] = info["frame_count"] / info["fps"]
    else:
        info["duration"] = 0.0
    return info


def make_session_dir() -> str:
    """创建会话临时目录，用于存放抽帧图片。"""
    ts = time.strftime("%Y%m%d_%H%M%S")
    d = os.path.join(tempfile.gettempdir(), f"model_checker_{ts}")
    os.makedirs(d, exist_ok=True)
    return d


def sample_frames(
    video_path: str,
    n: int,
    out_dir: str,
    progress_cb: Optional[ProgressCb] = None,
) -> list[str]:
    """从视频均匀抽 n 帧，保存为 PNG 到 out_dir，返回路径列表。

    若某帧读取失败则向后重试邻近帧。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError("无法获取视频总帧数")

    n = max(1, min(n, total))
    # 均匀索引
    indices = [round(i * (total - 1) / (n - 1)) if n > 1 else 0 for i in range(n)]
    # 去重保序
    seen = set()
    uniq = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            uniq.append(idx)

    paths: list[str] = []
    for i, idx in enumerate(uniq):
        ok = _read_and_save(cap, idx, out_dir, i, paths)
        if not ok:
            # 向后重试最多 5 帧
            for delta in range(1, 6):
                if idx + delta < total and _read_and_save(
                    cap, idx + delta, out_dir, i, paths
                ):
                    break
        if progress_cb:
            progress_cb(i + 1, len(uniq), f"抽帧 {i + 1}/{len(uniq)}")

    cap.release()
    return paths


def _read_and_save(
    cap: cv2.VideoCapture, idx: int, out_dir: str, i: int, paths: list[str]
) -> bool:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok or frame is None:
        return False
    path = os.path.join(out_dir, f"frame_{i:04d}.png")
    cv2.imwrite(path, frame)
    paths.append(path)
    return True
