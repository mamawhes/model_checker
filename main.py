"""YOLO 模型识别效果检测工具 — 启动入口。

运行：uv run python main.py
"""

import sys

from PySide6.QtWidgets import QApplication

from model_checker.gui.main_window import MainWindow

STYLESHEET = (
    "QWidget { background: #1e1e1e; color: #e0e0e0; font-size: 13px; }"
    "QGroupBox { border: 1px solid #3a3a3a; border-radius: 4px; "
    "margin-top: 12px; padding-top: 10px; }"
    "QGroupBox::title { subcontrol-origin: margin; left: 10px; "
    "padding: 0 4px; color: #bbb; }"
    "QPushButton { background: #3a3a3a; border: 1px solid #4a4a4a; "
    "padding: 6px 14px; border-radius: 3px; }"
    "QPushButton:hover { background: #4a4a4a; }"
    "QPushButton:disabled { color: #666; }"
    "QLineEdit, QSpinBox, QDoubleSpinBox { background: #2b2b2b; "
    "border: 1px solid #4a4a4a; padding: 4px; border-radius: 2px; }"
    "QSpinBox::up-button, QDoubleSpinBox::up-button { subcontrol-origin: "
    "border; subcontrol-position: top right; width: 18px; "
    "border-left: 1px solid #4a4a4a; border-bottom: 1px solid #4a4a4a; "
    "background: #3a3a3a; }"
    "QSpinBox::down-button, QDoubleSpinBox::down-button { "
    "subcontrol-origin: border; subcontrol-position: bottom right; "
    "width: 18px; border-left: 1px solid #4a4a4a; background: #3a3a3a; }"
    "QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover, "
    "QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover "
    "{ background: #4a6a9e; }"
    "QSpinBox::up-arrow, QDoubleSpinBox::up-arrow { width: 0; height: 0; "
    "border-left: 4px solid none; border-right: 4px solid none; "
    "border-bottom: 5px solid #e0e0e0; }"
    "QSpinBox::down-arrow, QDoubleSpinBox::down-arrow { width: 0; "
    "height: 0; border-left: 4px solid none; border-right: 4px solid none; "
    "border-top: 5px solid #e0e0e0; }"
    "QListWidget { background: #252525; border: 1px solid #3a3a3a; }"
    "QTableWidget { background: #252525; gridline-color: #3a3a3a; "
    "selection-background-color: #4a9eff; }"
    "QHeaderView::section { background: #333; color: #ccc; "
    "padding: 6px; border: none; }"
    "QProgressBar { background: #2b2b2b; border: 1px solid #4a4a4a; "
    "text-align: center; border-radius: 2px; }"
    "QProgressBar::chunk { background: #4a9eff; }"
)


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("YOLO Model Checker")
    app.setStyleSheet(STYLESHEET)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
