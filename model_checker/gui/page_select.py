"""步骤1：选择 ONNX 模型、视频与推理参数。"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from ..video import get_video_info
from .base_page import BasePage


class SelectPage(BasePage):
    def __init__(self, state, parent=None) -> None:
        super().__init__(state, parent)
        self._video_info: dict = {}
        # 记录上次离开时的参数，用于检测变化以决定是否重新抽帧/推理
        self._prev_video: str = ""
        self._prev_frame_count = None
        self._prev_filter = None
        self._prev_conf = None
        self._prev_iou = None
        self._prev_imgsz = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.addWidget(self._title("步骤 1 / 5 — 选择模型与视频"))

        # 模型组
        box_model = QGroupBox("ONNX 模型")
        mlay = QVBoxLayout(box_model)
        row1 = QHBoxLayout()
        self.model_edit = QLineEdit()
        self.model_edit.setReadOnly(True)
        self.model_edit.setPlaceholderText("选择 .onnx 模型文件")
        btn_model = QPushButton("选择模型…")
        btn_model.clicked.connect(self._pick_model)
        row1.addWidget(self.model_edit, 1)
        row1.addWidget(btn_model)
        mlay.addLayout(row1)

        row2 = QHBoxLayout()
        self.names_edit = QLineEdit()
        self.names_edit.setReadOnly(True)
        self.names_edit.setPlaceholderText("可选：类别名文件 (.names/.txt)，每行一类")
        btn_names = QPushButton("选择类别文件…")
        btn_names.clicked.connect(self._pick_names)
        row2.addWidget(self.names_edit, 1)
        row2.addWidget(btn_names)
        mlay.addLayout(row2)
        root.addWidget(box_model)

        # 视频组
        box_video = QGroupBox("待检测视频")
        vlay = QVBoxLayout(box_video)
        rowv = QHBoxLayout()
        self.video_edit = QLineEdit()
        self.video_edit.setReadOnly(True)
        self.video_edit.setPlaceholderText("选择视频文件 (mp4/avi/mov…)")
        btn_video = QPushButton("选择视频…")
        btn_video.clicked.connect(self._pick_video)
        rowv.addWidget(self.video_edit, 1)
        rowv.addWidget(btn_video)
        vlay.addLayout(rowv)
        self.video_info_label = QLabel("未选择视频")
        self.video_info_label.setStyleSheet("color: #888;")
        vlay.addWidget(self.video_info_label)
        root.addWidget(box_video)

        # 参数组
        box_param = QGroupBox("推理与抽帧参数")
        form = QFormLayout(box_param)
        self.spin_frames = QSpinBox()
        self.spin_frames.setRange(1, 1000)
        self.spin_frames.setValue(self.state.frame_count_target)
        form.addRow("抽帧数量：", self.spin_frames)

        self.chk_filter = QCheckBox("仅保留检测到目标的帧（抽帧时用模型过滤）")
        self.chk_filter.setChecked(self.state.filter_detected)
        self.chk_filter.setToolTip(
            "勾选后，抽帧阶段会对每帧运行模型推理，丢弃没有检测结果的帧。"
            "保留的帧已带检测结果，步骤3可直接复用。"
        )
        form.addRow(self.chk_filter)

        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.01, 0.99)
        self.spin_conf.setSingleStep(0.05)
        self.spin_conf.setValue(self.state.conf)
        form.addRow("置信度阈值 (conf)：", self.spin_conf)

        self.spin_iou = QDoubleSpinBox()
        self.spin_iou.setRange(0.01, 0.99)
        self.spin_iou.setSingleStep(0.05)
        self.spin_iou.setValue(self.state.iou)
        form.addRow("NMS IoU 阈值：", self.spin_iou)

        self.spin_imgsz = QSpinBox()
        self.spin_imgsz.setRange(320, 1280)
        self.spin_imgsz.setSingleStep(32)
        self.spin_imgsz.setValue(self.state.imgsz)
        form.addRow("输入尺寸 (imgsz)：", self.spin_imgsz)

        hint = QLabel(
            "提示：修改「视频」「抽帧数量」或勾选项后前进会自动重新抽帧；"
            "修改推理参数会作废已有推理结果。"
        )
        hint.setStyleSheet("color: #888; font-size: 12px;")
        form.addRow(hint)
        root.addWidget(box_param)
        root.addStretch(1)

    def _title(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("font-size: 18px; font-weight: bold; padding: 8px 0;")
        return lbl

    # ---- 选择器 ----
    def _pick_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 ONNX 模型", "", "ONNX 模型 (*.onnx)"
        )
        if path:
            self.model_edit.setText(path)

    def _pick_names(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择类别名文件", "", "文本文件 (*.names *.txt)"
        )
        if path:
            self.names_edit.setText(path)

    def _pick_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "", "视频 (*.mp4 *.avi *.mov *.mkv *.flv)"
        )
        if path:
            self.video_edit.setText(path)
            try:
                self._video_info = get_video_info(path)
                dur = self._video_info["duration"]
                self.video_info_label.setText(
                    f"分辨率 {self._video_info['width']}x{self._video_info['height']} · "
                    f"{self._video_info['fps']:.1f} fps · "
                    f"{self._video_info['frame_count']} 帧 · "
                    f"时长 {dur:.1f} s"
                )
            except Exception as e:  # noqa: BLE001
                self._video_info = {}
                self.video_info_label.setText(f"读取视频信息失败：{e}")

    # ---- 状态流转 ----
    def on_leave(self) -> None:
        new_video = self.video_edit.text().strip()
        new_frame_count = self.spin_frames.value()
        new_filter = self.chk_filter.isChecked()
        new_conf = self.spin_conf.value()
        new_iou = self.spin_iou.value()
        new_imgsz = self.spin_imgsz.value()

        # 检测参数变化（首次离开不触发）：
        #   视频/抽帧数/过滤勾选变化 → 清空全部，迫使步骤2重新抽帧
        #   仅推理参数变化 → 保留抽帧，作废推理与审核结果
        has_prev = self._prev_video != ""
        if has_prev and (
            self._prev_video != new_video
            or self._prev_frame_count != new_frame_count
            or self._prev_filter != new_filter
        ):
            self.state.reset()
        elif has_prev and (
            self._prev_conf != new_conf
            or self._prev_iou != new_iou
            or self._prev_imgsz != new_imgsz
        ):
            self.state.inference_done = False
            self.state.reviewed_indices.clear()
            for f in self.state.frames:
                f.detections.clear()
                f.missed = 0
                f.wrong = 0

        self._prev_video = new_video
        self._prev_frame_count = new_frame_count
        self._prev_filter = new_filter
        self._prev_conf = new_conf
        self._prev_iou = new_iou
        self._prev_imgsz = new_imgsz

        self.state.model_path = self.model_edit.text().strip()
        self.state.names_path = self.names_edit.text().strip()
        self.state.video_path = new_video
        self.state.frame_count_target = new_frame_count
        self.state.filter_detected = new_filter
        self.state.conf = new_conf
        self.state.iou = new_iou
        self.state.imgsz = new_imgsz
        # 加载类别名
        if self.state.names_path:
            try:
                with open(
                    self.state.names_path, "r", encoding="utf-8"
                ) as f:
                    self.state.class_names = [
                        line.strip() for line in f if line.strip()
                    ]
            except Exception:  # noqa: BLE001
                self.state.class_names = None
        else:
            self.state.class_names = None

    def can_next(self) -> tuple[bool, str]:
        if not self.model_edit.text().strip():
            return False, "请先选择 ONNX 模型"
        if not self.video_edit.text().strip():
            return False, "请先选择视频"
        if not self._video_info:
            return False, "视频信息读取失败，请重新选择视频"
        return True, ""
