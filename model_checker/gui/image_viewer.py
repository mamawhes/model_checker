"""可缩放/平移的图片查看器，基于 QGraphicsView。

支持 Shift+左键拖拽画测量框（反转虚线 + 左下角尺寸标签），
用于判断疑似漏检目标的尺寸是否低于检测标准（小目标允许漏掉）。
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QMouseEvent,
    QPainter,
    QPen,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QMenu,
)


class MeasureItem(QGraphicsItem):
    """测量框：反转虚线矩形 + 左下角尺寸标签（像素）。"""

    def __init__(self, rect: QRectF) -> None:
        super().__init__()
        self._rect = rect.normalized()
        self.setZValue(1000)

    def set_rect(self, rect: QRectF) -> None:
        self.prepareGeometryChange()
        self._rect = rect.normalized()
        self.update()

    def _label_text(self) -> str:
        return f"{int(round(self._rect.width()))}×{int(round(self._rect.height()))}"

    def _label_rect(self) -> QRectF:
        fm = QFontMetrics(QFont())
        tw = fm.horizontalAdvance(self._label_text()) + 8
        th = fm.height() + 4
        return QRectF(self._rect.left(), self._rect.bottom(), tw, th)

    def boundingRect(self) -> QRectF:  # noqa: N802
        return self._rect.united(self._label_rect())

    def paint(self, painter: QPainter, option, widget=None) -> None:  # noqa: N802
        # 反转虚线矩形（Difference 合成，任意背景可见），需关闭抗锯齿
        pen = QPen(QColor(255, 255, 255), 1, Qt.DashLine)
        pen.setDashPattern([6, 4])
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.setCompositionMode(QPainter.CompositionMode_Difference)
        painter.drawRect(self._rect)
        painter.restore()
        # 左下角尺寸标签（黑底白字）
        lr = self._label_rect()
        painter.setBrush(QColor(0, 0, 0, 200))
        painter.setPen(Qt.NoPen)
        painter.drawRect(lr)
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(lr, Qt.AlignCenter, self._label_text())


class ImageViewer(QGraphicsView):
    """显示单张图片，支持滚轮缩放、拖拽平移、Shift+拖拽画测量框。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item: QGraphicsPixmapItem | None = None

        self.setRenderHints(self.renderHints())
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setBackgroundBrush(Qt.black)
        self.setFrameShape(QGraphicsView.NoFrame)

        # 测量状态
        self._measuring = False
        self._measure_start = QPointF()
        self._current: MeasureItem | None = None
        self._measure_items: list[MeasureItem] = []

    def set_pixmap(self, pixmap) -> None:
        self.clear_measures()
        self._scene.clear()
        if pixmap is None or pixmap.isNull():
            self._pixmap_item = None
            return
        self._pixmap_item = QGraphicsPixmapItem(pixmap)
        self._scene.addItem(self._pixmap_item)
        self._scene.setSceneRect(self._pixmap_item.boundingRect())
        self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)

    def clear_measures(self) -> None:
        """清除所有测量框（切帧时自动调用）。"""
        self._measuring = False
        self._current = None
        for it in self._measure_items:
            self._scene.removeItem(it)
        self._measure_items.clear()

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        if self._pixmap_item is None:
            return
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        self.scale(factor, factor)

    def fit(self) -> None:
        if self._pixmap_item is not None:
            self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)

    # ---- 测量框交互 ----
    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if (
            event.button() == Qt.LeftButton
            and (event.modifiers() & Qt.ShiftModifier)
        ):
            # 进入测量模式，禁用平移
            self.setDragMode(QGraphicsView.NoDrag)
            self._measuring = True
            self._measure_start = self.mapToScene(event.pos())
            self._current = MeasureItem(
                QRectF(self._measure_start, self._measure_start)
            )
            self._scene.addItem(self._current)
            self.setCursor(Qt.CrossCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._measuring and self._current is not None:
            end = self.mapToScene(event.pos())
            self._current.set_rect(QRectF(self._measure_start, end))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._measuring and self._current is not None:
            self._measuring = False
            item = self._current
            self._current = None
            # 太小的框视为误触丢弃
            if item._rect.width() < 3 or item._rect.height() < 3:
                self._scene.removeItem(item)
            else:
                self._measure_items.append(item)
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event) -> None:  # noqa: N802
        menu = QMenu(self)
        act_clear = menu.addAction("清除所有测量框")
        act = menu.exec(event.globalPos())
        if act is act_clear:
            self.clear_measures()
