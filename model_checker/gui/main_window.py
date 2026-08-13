"""主窗口：五步向导框架（侧边栏 + QStackedWidget + 导航按钮）。"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..state import SessionState
from .base_page import BasePage
from .page_frames import FramesPage
from .page_inference import InferencePage
from .page_report import ReportPage
from .page_review import ReviewPage
from .page_select import SelectPage

STEPS = [
    "1. 选择模型与视频",
    "2. 抽帧与图片管理",
    "3. 模型推理",
    "4. 逐帧审核标注",
    "5. 检测结果报告",
]


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.state = SessionState()
        self.setWindowTitle("YOLO 模型识别效果检测工具")
        self.resize(1280, 820)

        self._build_ui()
        self._goto(0)

    def _build_ui(self) -> None:
        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # 侧边栏
        self.sidebar = QListWidget()
        self.sidebar.setFixedWidth(200)
        self.sidebar.setStyleSheet(
            "QListWidget { background: #2b2b2b; color: #ddd; "
            "font-size: 14px; border: none; outline: none; }"
            "QListWidget::item { padding: 16px 12px; border-left: 3px solid "
            "transparent; }"
            "QListWidget::item:selected { background: #3a3a3a; "
            "border-left: 3px solid #4a9eff; color: #fff; }"
        )
        for s in STEPS:
            item = QListWidgetItem(s)
            self.sidebar.addItem(item)
        self.sidebar.currentRowChanged.connect(self._on_sidebar_changed)

        # 右侧：页面栈 + 底部导航
        right = QVBoxLayout()
        right.setContentsMargins(12, 12, 12, 12)
        right.setSpacing(10)

        self.stack = QStackedWidget()
        self.pages: list[BasePage] = [
            SelectPage(self.state, self),
            FramesPage(self.state, self),
            InferencePage(self.state, self),
            ReviewPage(self.state, self),
            ReportPage(self.state, self),
        ]
        for p in self.pages:
            self.stack.addWidget(p)

        # 底部导航
        bar = QHBoxLayout()
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #888;")
        self.prev_btn = QPushButton("上一步")
        self.next_btn = QPushButton("下一步")
        self.prev_btn.clicked.connect(self._prev)
        self.next_btn.clicked.connect(self._next)
        self.next_btn.setStyleSheet(
            "QPushButton { background: #4a9eff; color: white; "
            "padding: 6px 18px; border: none; }"
            "QPushButton:disabled { background: #555; }"
        )
        bar.addWidget(self.status_label)
        bar.addStretch(1)
        bar.addWidget(self.prev_btn)
        bar.addWidget(self.next_btn)

        right.addWidget(self.stack, 1)
        right.addLayout(bar)
        root.addWidget(self.sidebar)
        root.addLayout(right, 1)
        self.setCentralWidget(central)

    # ---- 导航 ----
    def _on_sidebar_changed(self, row: int) -> None:
        # 仅允许通过按钮导航，避免侧栏点击跳过校验
        # 但允许点击已解锁的步骤；此处简单禁用侧栏跳转
        pass

    def _goto(self, idx: int) -> None:
        if idx < 0 or idx >= len(self.pages):
            return
        cur = self.stack.currentIndex()
        if cur != idx:
            self.pages[cur].on_leave()
        self.stack.setCurrentIndex(idx)
        # 同步侧栏高亮（不触发递归）
        self.sidebar.blockSignals(True)
        self.sidebar.setCurrentRow(idx)
        self.sidebar.blockSignals(False)
        self.pages[idx].on_enter()
        self._update_nav(idx)

    def _update_nav(self, idx: int) -> None:
        self.prev_btn.setEnabled(idx > 0)
        is_last = idx == len(self.pages) - 1
        self.next_btn.setEnabled(not is_last)
        self.next_btn.setText("完成" if is_last else "下一步")

    def _prev(self) -> None:
        self._goto(self.stack.currentIndex() - 1)

    def _next(self) -> None:
        cur = self.stack.currentIndex()
        ok, reason = self.pages[cur].can_next()
        if not ok:
            QMessageBox.warning(self, "无法继续", reason)
            return
        if cur == len(self.pages) - 1:
            # 最后一页：重置回首页
            self.state.reset()
            for p in self.pages:
                p.on_leave()
            self._goto(0)
            return
        self._goto(cur + 1)

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)
