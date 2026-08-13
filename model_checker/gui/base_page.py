"""页面基类，独立模块以避免与 main_window 循环导入。"""

from __future__ import annotations

from PySide6.QtWidgets import QWidget


class BasePage(QWidget):
    """向导页面基类。"""

    def __init__(self, state, parent=None) -> None:
        super().__init__(parent)
        self.state = state

    def on_enter(self) -> None:
        """进入页面时调用。"""
        pass

    def on_leave(self) -> None:
        """离开页面时调用。"""
        pass

    def can_next(self) -> tuple[bool, str]:
        """是否允许进入下一步，返回 (是否允许, 原因)。"""
        return True, ""
