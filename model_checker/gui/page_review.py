"""步骤4：逐帧查看带框结果，填写漏框数与框错数。"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..image_utils import cv2_to_qpixmap, draw_detections, load_image_bgr
from ..metrics import compute_frame, fmt_ratio
from .image_viewer import ImageViewer
from .base_page import BasePage


class ReviewPage(BasePage):
    def __init__(self, state, parent=None) -> None:
        super().__init__(state, parent)
        self._cur = 0
        self._loading = False
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.addWidget(self._title("步骤 4 / 5 — 逐帧审核标注"))

        # 顶部导航
        nav = QHBoxLayout()
        self.btn_prev = QPushButton("◀ 上一帧")
        self.btn_prev.clicked.connect(self._prev)
        self.btn_next = QPushButton("下一帧 ▶")
        self.btn_next.clicked.connect(self._next)
        self.nav_label = QLabel("")
        self.nav_label.setStyleSheet("font-size: 14px;")
        nav.addWidget(self.btn_prev)
        nav.addWidget(self.nav_label)
        nav.addWidget(self.btn_next)
        nav.addStretch(1)
        self.hint_label = QLabel(
            "快捷键：A 上一帧 · D 下一帧 · Shift+左键拖拽画框测量目标尺寸（右键清除）"
        )
        self.hint_label.setStyleSheet("color: #888;")
        nav.addWidget(self.hint_label)
        root.addLayout(nav)

        # 主体：左图 + 右表单
        splitter = QSplitter(Qt.Horizontal)

        self.viewer = ImageViewer()
        splitter.addWidget(self.viewer)

        right = QWidget()
        rlay = QVBoxLayout(right)

        # 输入组
        box = QGroupBox("本帧标注")
        form = QFormLayout(box)
        self.lbl_boxes = QLabel("0")
        self.lbl_boxes.setStyleSheet("font-weight: bold;")
        form.addRow("模型框数 B：", self.lbl_boxes)

        self.spin_missed = QSpinBox()
        self.spin_missed.setRange(0, 9999)
        self.spin_missed.valueChanged.connect(self._on_input_changed)
        form.addRow("漏框数 (FN)：", self.spin_missed)

        self.spin_wrong = QSpinBox()
        self.spin_wrong.setRange(0, 9999)
        self.spin_wrong.valueChanged.connect(self._on_input_changed)
        form.addRow("框错数 (FP)：", self.spin_wrong)

        self.lbl_pr = QLabel("本帧 Precision: —  Recall: —")
        self.lbl_pr.setStyleSheet("color: #4a9eff; padding-top: 6px;")
        form.addRow(self.lbl_pr)
        rlay.addWidget(box)

        # 帧列表（快速跳转）
        rlay.addWidget(QLabel("帧列表（点击切换）："))
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self._on_list_changed)
        rlay.addWidget(self.list, 1)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, 1)

        # 快捷键：A 上一帧，D 下一帧（仅本页及子部件有焦点时生效）
        self._sc_prev = QShortcut(QKeySequence("A"), self)
        self._sc_prev.setContext(Qt.WidgetWithChildrenShortcut)
        self._sc_prev.activated.connect(self._prev)
        self._sc_next = QShortcut(QKeySequence("D"), self)
        self._sc_next.setContext(Qt.WidgetWithChildrenShortcut)
        self._sc_next.activated.connect(self._next)

    def _title(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("font-size: 18px; font-weight: bold; padding: 8px 0;")
        return lbl

    def on_enter(self) -> None:
        if not self.state.frames:
            return
        self._cur = 0
        self._build_frame_list()
        self._load_frame(0)

    def _build_frame_list(self) -> None:
        self.list.blockSignals(True)
        self.list.clear()
        for i, f in enumerate(self.state.frames):
            item = QListWidgetItem(f"第 {i + 1} 帧  ·  框 {f.box_count}")
            self.list.addItem(item)
        self.list.blockSignals(False)

    def _load_frame(self, idx: int) -> None:
        if idx < 0 or idx >= len(self.state.frames):
            return
        self._loading = True
        self._cur = idx
        f = self.state.frames[idx]

        # 显示带框图
        img = load_image_bgr(f.path)
        if img is not None:
            drawn = draw_detections(img, f.detections, self.state.class_names)
            self.viewer.set_pixmap(cv2_to_qpixmap(drawn))
        self.viewer.fit()

        # 同步输入
        self.lbl_boxes.setText(str(f.box_count))
        self.spin_wrong.setMaximum(max(0, f.box_count))
        self.spin_missed.setValue(f.missed)
        self.spin_wrong.setValue(f.wrong)

        # 同步列表选中
        self.list.blockSignals(True)
        self.list.setCurrentRow(idx)
        self.list.blockSignals(False)

        self.nav_label.setText(f"第 {idx + 1} / {len(self.state.frames)} 帧")
        self._update_pr()
        self.state.reviewed_indices.add(idx)
        self._loading = False

    def _on_list_changed(self, row: int) -> None:
        if self._loading or row < 0:
            return
        self._load_frame(row)

    def _on_input_changed(self) -> None:
        if self._loading or self._cur >= len(self.state.frames):
            return
        f = self.state.frames[self._cur]
        f.missed = self.spin_missed.value()
        f.wrong = self.spin_wrong.value()
        self._update_pr()
        # 更新列表项文案
        item = self.list.item(self._cur)
        if item is not None:
            item.setText(f"第 {self._cur + 1} 帧  ·  框 {f.box_count}")

    def _update_pr(self) -> None:
        if self._cur >= len(self.state.frames):
            return
        f = self.state.frames[self._cur]
        m = compute_frame(f)
        self.lbl_pr.setText(
            f"本帧 Precision: {fmt_ratio(m['precision'])}  ·  "
            f"Recall: {fmt_ratio(m['recall'])}  ·  "
            f"TP={m['tp']} FP={m['fp']} FN={m['fn']}"
        )

    def _prev(self) -> None:
        if self._cur > 0:
            self._load_frame(self._cur - 1)

    def _next(self) -> None:
        if self._cur < len(self.state.frames) - 1:
            self._load_frame(self._cur + 1)

    def can_next(self) -> tuple[bool, str]:
        if not self.state.frames:
            return False, "没有可审核的帧"
        n = len(self.state.frames)
        if len(self.state.reviewed_indices) < n:
            return (
                False,
                f"还有 {n - len(self.state.reviewed_indices)} 帧未审核，"
                "请逐帧确认（可填 0）。",
            )
        return True, ""
