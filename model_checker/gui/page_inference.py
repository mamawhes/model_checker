"""步骤3：批量 ONNX 推理并预览带框结果。"""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QMessageBox,
)

from ..image_utils import draw_detections, load_image_bgr, make_thumbnail
from ..inference import YOLOv8Detector
from .base_page import BasePage
from .workers import Worker

THUMB_SIZE = 160


class InferencePage(BasePage):
    def __init__(self, state, parent=None) -> None:
        super().__init__(state, parent)
        self._worker: Worker | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.addWidget(self._title("步骤 3 / 5 — 模型推理"))

        self.summary = QLabel("点击「开始推理」运行 ONNX 检测")
        self.summary.setStyleSheet("font-size: 14px; padding: 4px 0;")
        root.addWidget(self.summary)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        root.addWidget(self.progress)

        self.btn_run = QPushButton("开始推理")
        self.btn_run.clicked.connect(self._run_inference)
        bar = QHBoxLayout()
        bar.addWidget(self.btn_run)
        bar.addStretch(1)
        root.addLayout(bar)

        self.list = QListWidget()
        self.list.setViewMode(QListWidget.IconMode)
        self.list.setIconSize(QSize(THUMB_SIZE, THUMB_SIZE))
        self.list.setResizeMode(QListWidget.Adjust)
        self.list.setSelectionMode(QListWidget.SingleSelection)
        root.addWidget(self.list, 1)

    def _title(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("font-size: 18px; font-weight: bold; padding: 8px 0;")
        return lbl

    def on_enter(self) -> None:
        if self.state.inference_done:
            total = sum(f.box_count for f in self.state.frames)
            self.summary.setText(
                f"已有推理结果：{len(self.state.frames)} 张图片，共 {total} 个目标"
                "（可点「开始推理」重跑）"
            )
            self._refresh_preview()
        elif not self._worker:
            self.summary.setText("点击「开始推理」运行 ONNX 检测")

    def _run_inference(self) -> None:
        self.btn_run.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.summary.setText("正在加载模型并推理…")
        self.list.clear()

        model_path = self.state.model_path
        conf = self.state.conf
        iou = self.state.iou
        frame_paths = [f.path for f in self.state.frames]

        def fn(progress_cb):
            det = YOLOv8Detector(model_path, conf=conf, iou=iou)
            results = []
            n = len(frame_paths)
            for i, p in enumerate(frame_paths):
                img = load_image_bgr(p)
                if img is None:
                    results.append([])
                else:
                    results.append(det.detect(img))
                if progress_cb:
                    progress_cb(i + 1, n, f"推理 {i + 1}/{n}")
            return results

        self._worker = Worker(fn)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, i: int, n: int, msg: str) -> None:
        if n > 0:
            self.progress.setValue(int(i / n * 100))
        self.summary.setText(msg)

    def _on_done(self, result) -> None:
        results = result
        for i, dets in enumerate(results):
            if i < len(self.state.frames):
                self.state.frames[i].detections = dets
        self.state.inference_done = True
        self.progress.setVisible(False)
        self.btn_run.setEnabled(True)
        total = sum(len(r) for r in results)
        self.summary.setText(
            f"推理完成：{len(results)} 张图片，共检测到 {total} 个目标"
        )
        self._refresh_preview()
        self._cleanup_worker()

    def _on_error(self, msg: str) -> None:
        self.progress.setVisible(False)
        self.btn_run.setEnabled(True)
        self.summary.setText(f"推理失败：{msg}")
        QMessageBox.critical(self, "推理失败", msg)
        self._cleanup_worker()

    def _cleanup_worker(self) -> None:
        """清理后台线程，避免重复推理时旧 QThread 未释放导致闪退。"""
        if self._worker is not None:
            self._worker.stop()
            self._worker = None

    def _refresh_preview(self) -> None:
        self.list.clear()
        for i, f in enumerate(self.state.frames):
            img = load_image_bgr(f.path)
            if img is None:
                continue
            drawn = draw_detections(img, f.detections, self.state.class_names)
            pix = make_thumbnail(drawn, THUMB_SIZE)
            item = QListWidgetItem()
            item.setIcon(pix)
            item.setText(f"{i + 1}  ({f.box_count})")
            item.setTextAlignment(Qt.AlignBottom | Qt.AlignHCenter)
            self.list.addItem(item)

    def can_next(self) -> tuple[bool, str]:
        if not self.state.inference_done:
            return False, "请先完成推理"
        return True, ""
