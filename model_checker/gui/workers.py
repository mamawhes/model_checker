"""后台线程 Worker：避免抽帧/推理冻结 UI。"""

from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import QObject, QThread, Signal


class Worker(QObject):
    """通用 Worker：在独立 QThread 中执行一个 callable。

    信号：
        progress(int, int, str) — 当前/总数/描述
        finished(object)         — 成功结果
        error(str)               — 异常信息
    """

    progress = Signal(int, int, str)
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, fn: Callable, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._fn = fn
        self._thread: Optional[QThread] = None

    def _progress_cb(self, i: int, n: int, msg: str) -> None:
        self.progress.emit(i, n, msg)

    def run(self) -> None:
        try:
            result = self._fn(self._progress_cb)
            self.finished.emit(result)
        except Exception as e:  # noqa: BLE001
            self.error.emit(str(e))

    def start(self) -> None:
        self._thread = QThread()
        self.moveToThread(self._thread)
        self._thread.started.connect(self.run)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(5000)
            self._thread = None
