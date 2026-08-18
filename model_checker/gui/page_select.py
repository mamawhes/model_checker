"""步骤1：选择 ONNX 模型、视频/图片目录与推理参数。"""

from __future__ import annotations

import os

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

from ..state import FrameItem
from ..video import get_video_info
from .base_page import BasePage

# 图片目录模式支持的图片扩展名
_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


class SelectPage(BasePage):
    def __init__(self, state, parent=None) -> None:
        super().__init__(state, parent)
        self._video_info: dict = {}
        # 记录上次离开时的参数，用于检测变化以决定是否重新抽帧/推理
        self._prev_video: str = ""
        self._prev_image_dir: str = ""
        self._prev_frame_count = None
        self._prev_filter = None
        self._prev_conf = None
        self._prev_iou = None
        self._prev_imgsz = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.addWidget(self._title("步骤 1 / 5 — 选择模型与数据来源"))

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

        # 图片目录组（与视频互斥：选其一即可，选另一个会清空当前）
        box_dir = QGroupBox("或 导入图片目录（跳过抽帧步骤）")
        dlay = QVBoxLayout(box_dir)
        rowd = QHBoxLayout()
        self.image_dir_edit = QLineEdit()
        self.image_dir_edit.setReadOnly(True)
        self.image_dir_edit.setPlaceholderText("选择包含图片的目录 (png/jpg/bmp/tif…)")
        btn_dir = QPushButton("选择目录…")
        btn_dir.clicked.connect(self._pick_image_dir)
        rowd.addWidget(self.image_dir_edit, 1)
        rowd.addWidget(btn_dir)
        dlay.addLayout(rowd)
        self.image_dir_info_label = QLabel("未选择图片目录")
        self.image_dir_info_label.setStyleSheet("color: #888;")
        dlay.addWidget(self.image_dir_info_label)
        root.addWidget(box_dir)

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
            "提示：视频与图片目录二选一。导入图片目录时可跳过抽帧步骤；"
            "修改「视频」「抽帧数量」或勾选项后前进会自动重新抽帧；"
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
            # 互斥：清空图片目录
            self.image_dir_edit.clear()
            self.image_dir_info_label.setText("未选择图片目录")
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
            self._update_param_enabled()

    def _pick_image_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择图片目录", "")
        if not path:
            return
        self.image_dir_edit.setText(path)
        # 互斥：清空视频
        self.video_edit.clear()
        self._video_info = {}
        self.video_info_label.setText("未选择视频")
        # 扫描统计
        paths = self._scan_image_dir(path)
        if paths:
            self.image_dir_info_label.setText(f"目录中共 {len(paths)} 张图片")
        else:
            self.image_dir_info_label.setText(
                "目录中没有支持的图片 (png/jpg/jpeg/bmp/tif/tiff)"
            )
        self._update_param_enabled()

    def _scan_image_dir(self, dir_path: str) -> list[str]:
        """扫描目录下所有支持的图片，按文件名排序返回绝对路径。"""
        try:
            names = [
                n
                for n in os.listdir(dir_path)
                if n.lower().endswith(_IMAGE_EXTS) and os.path.isfile(
                    os.path.join(dir_path, n)
                )
            ]
        except OSError:
            return []
        names.sort()
        return [os.path.join(dir_path, n) for n in names]

    def _update_param_enabled(self) -> None:
        """图片目录模式下「抽帧数量/过滤」无意义，禁用以示区分。"""
        image_dir_mode = bool(self.image_dir_edit.text().strip())
        self.spin_frames.setEnabled(not image_dir_mode)
        self.chk_filter.setEnabled(not image_dir_mode)

    # ---- 状态流转 ----
    def on_leave(self) -> None:
        new_video = self.video_edit.text().strip()
        new_image_dir = self.image_dir_edit.text().strip()
        new_frame_count = self.spin_frames.value()
        new_filter = self.chk_filter.isChecked()
        new_conf = self.spin_conf.value()
        new_iou = self.spin_iou.value()
        new_imgsz = self.spin_imgsz.value()

        # 检测参数变化（首次离开不触发）：
        #   来源(视频/图片目录)变化、或视频模式下抽帧数/过滤勾选变化 → 清空全部
        #   仅推理参数变化 → 保留抽帧，作废推理与审核结果
        has_prev = self._prev_video != "" or self._prev_image_dir != ""
        source_changed = (
            self._prev_video != new_video
            or self._prev_image_dir != new_image_dir
        )
        frame_param_changed = bool(new_video) and (
            self._prev_frame_count != new_frame_count
            or self._prev_filter != new_filter
        )
        if has_prev and (source_changed or frame_param_changed):
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
        self._prev_image_dir = new_image_dir
        self._prev_frame_count = new_frame_count
        self._prev_filter = new_filter
        self._prev_conf = new_conf
        self._prev_iou = new_iou
        self._prev_imgsz = new_imgsz

        self.state.model_path = self.model_edit.text().strip()
        self.state.names_path = self.names_edit.text().strip()
        self.state.video_path = new_video
        self.state.image_dir = new_image_dir
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

        # 图片目录模式：扫描目录填入 frames（首次进入或来源变化 reset 后）
        if new_image_dir and not self.state.frames:
            paths = self._scan_image_dir(new_image_dir)
            self.state.frames = [FrameItem(path=p) for p in paths]

    def can_next(self) -> tuple[bool, str]:
        if not self.model_edit.text().strip():
            return False, "请先选择 ONNX 模型"
        video = self.video_edit.text().strip()
        image_dir = self.image_dir_edit.text().strip()
        if not video and not image_dir:
            return False, "请先选择视频或图片目录"
        if video and not self._video_info:
            return False, "视频信息读取失败，请重新选择视频"
        if image_dir:
            if not os.path.isdir(image_dir):
                return False, "图片目录无效，请重新选择"
            if not self._scan_image_dir(image_dir):
                return False, "图片目录中没有支持的图片 (png/jpg/jpeg/bmp/tif/tiff)"
        return True, ""
