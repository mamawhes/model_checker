"""步骤2：均匀抽帧，查看/删除/添加图片。

支持「仅保留检测到目标的帧」：勾选后抽帧阶段对每帧推理，丢弃无目标的帧，
保留帧的检测结果写入 FrameItem，供步骤3直接复用。
"""

from __future__ import annotations

import os

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..image_utils import draw_detections, load_image_bgr, make_thumbnail
from ..inference import YOLOv8Detector
from ..state import FrameItem
from ..video import make_session_dir, sample_frames
from .base_page import BasePage
from .workers import Worker

THUMB_SIZE = 160


class FramesPage(BasePage):
    def __init__(self, state, parent=None) -> None:
        super().__init__(state, parent)
        self._session_dir = ""
        self._worker: Worker | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.addWidget(self._title("步骤 2 / 5 — 抽帧结果与图片管理"))

        self.count_label = QLabel("准备抽帧…")
        root.addWidget(self.count_label)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        root.addWidget(self.progress)

        # 缩略图网格
        self.list = QListWidget()
        self.list.setViewMode(QListWidget.IconMode)
        self.list.setIconSize(QSize(THUMB_SIZE, THUMB_SIZE))
        self.list.setResizeMode(QListWidget.Adjust)
        self.list.setSelectionMode(QListWidget.ExtendedSelection)
        self.list.setSpacing(6)
        root.addWidget(self.list, 1)

        # 操作按钮（如需重新抽帧，请返回上一步修改视频/抽帧数量/过滤选项）
        bar = QHBoxLayout()
        self.btn_delete = QPushButton("删除选中")
        self.btn_delete.clicked.connect(self._delete_selected)
        self.btn_add = QPushButton("添加图片…")
        self.btn_add.clicked.connect(self._add_images)
        bar.addWidget(self.btn_delete)
        bar.addWidget(self.btn_add)
        bar.addStretch(1)
        root.addLayout(bar)

    def _title(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("font-size: 18px; font-weight: bold; padding: 8px 0;")
        return lbl

    # ---- 抽帧 ----
    def on_enter(self) -> None:
        # 抽帧进行中则等待，避免重入
        if self._worker is not None:
            return
        # 无帧则抽帧；返回上一步改参数后 state 已被清空，此处自动重新抽帧
        if not self.state.frames:
            self._run_sampling()
        else:
            self._refresh_list()
            self._update_count()

    def _run_sampling(self) -> None:
        self.progress.setVisible(True)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.count_label.setText("正在抽帧…")
        self._set_controls_enabled(False)

        n = self.state.frame_count_target
        self._session_dir = make_session_dir()
        video_path = self.state.video_path
        filter_detected = self.state.filter_detected
        model_path = self.state.model_path
        conf = self.state.conf
        iou = self.state.iou
        session_dir = self._session_dir

        def fn(progress_cb):
            # 阶段1：均匀抽帧
            paths = sample_frames(video_path, n, session_dir, progress_cb)
            if not filter_detected:
                return [(p, []) for p in paths]
            # 阶段2：逐帧推理，仅保留检测到目标的帧
            det = YOLOv8Detector(model_path, conf=conf, iou=iou)
            kept: list[tuple[str, list]] = []
            total = len(paths)
            for i, p in enumerate(paths):
                img = load_image_bgr(p)
                dets = det.detect(img) if img is not None else []
                if dets:
                    kept.append((p, dets))
                if progress_cb:
                    progress_cb(
                        i + 1,
                        total,
                        f"过滤推理 {i + 1}/{total}，已保留 {len(kept)}",
                    )
            return kept

        self._worker = Worker(fn)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_sample_done)
        self._worker.error.connect(self._on_sample_error)
        self._worker.start()

    def _on_progress(self, i: int, n: int, msg: str) -> None:
        if n > 0:
            self.progress.setValue(int(i / n * 100))
        self.count_label.setText(msg)

    def _on_sample_done(self, result) -> None:
        kept = result  # list[(path, detections)]
        self.state.frames = [FrameItem(path=p, detections=d) for p, d in kept]
        if self.state.filter_detected:
            # 过滤时已对保留帧推理，标记完成以供步骤3复用
            self.state.inference_done = True
        self.progress.setVisible(False)
        self._set_controls_enabled(True)
        self._refresh_list()
        self._update_count()
        self._cleanup_worker()

    def _on_sample_error(self, msg: str) -> None:
        self.progress.setVisible(False)
        self._set_controls_enabled(True)
        self.count_label.setText(f"抽帧失败：{msg}")
        QMessageBox.critical(self, "抽帧失败", msg)
        self._cleanup_worker()

    def _cleanup_worker(self) -> None:
        """清理后台线程，避免重复抽帧时旧 QThread 未释放导致闪退。"""
        if self._worker is not None:
            self._worker.stop()
            self._worker = None

    # ---- 列表操作 ----
    def _refresh_list(self) -> None:
        self.list.clear()
        for i, f in enumerate(self.state.frames):
            img = load_image_bgr(f.path)
            if img is None:
                continue
            # 若该帧已有检测结果（过滤保留），缩略图带框显示
            if f.detections:
                img = draw_detections(img, f.detections, self.state.class_names)
            pix = make_thumbnail(img, THUMB_SIZE)
            item = QListWidgetItem()
            item.setIcon(pix)
            item.setText(f"{i + 1}")
            item.setTextAlignment(Qt.AlignBottom | Qt.AlignHCenter)
            item.setData(Qt.UserRole, i)
            self.list.addItem(item)

    def _update_count(self) -> None:
        n = len(self.state.frames)
        suffix = "（已过滤保留）" if self.state.filter_detected else ""
        self.count_label.setText(f"当前共 {n} 张图片{suffix}")

    def _delete_selected(self) -> None:
        rows = sorted(
            {self.list.row(it) for it in self.list.selectedItems()},
            reverse=True,
        )
        if not rows:
            return
        # 删除临时文件（仅删除本次抽帧生成的）
        for r in rows:
            path = self.state.frames[r].path
            if self._session_dir and path.startswith(self._session_dir):
                try:
                    os.remove(path)
                except OSError:
                    pass
        for r in rows:
            del self.state.frames[r]
        self._refresh_list()
        self._update_count()

    def _add_images(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        paths, _ = QFileDialog.getOpenFileNames(
            self, "添加图片", "", "图片 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if not paths:
            return
        for p in paths:
            self.state.frames.append(FrameItem(path=p))
        # 新增帧无推理结果，作废旧推理标记
        self.state.inference_done = False
        self._refresh_list()
        self._update_count()

    def _set_controls_enabled(self, enabled: bool) -> None:
        self.btn_delete.setEnabled(enabled)
        self.btn_add.setEnabled(enabled)

    def can_next(self) -> tuple[bool, str]:
        if not self.state.frames:
            return False, "没有图片，请先抽帧或添加图片"
        return True, ""
