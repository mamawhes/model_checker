"""步骤5：检测结果报告（Precision/Recall + 明细 + 导出报告目录）。"""

from __future__ import annotations

import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..export import export_report
from ..metrics import compute_frame, compute_overall, fmt_ratio
from .base_page import BasePage


class ReportPage(BasePage):
    def __init__(self, state, parent=None) -> None:
        super().__init__(state, parent)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.addWidget(self._title("步骤 5 / 5 — 检测结果报告"))

        # 总指标
        metric_box = QWidget()
        mlay = QHBoxLayout(metric_box)
        self.lbl_precision = self._big_metric("Precision (精确率)")
        self.lbl_recall = self._big_metric("Recall (召回率)")
        self.lbl_detail = QLabel("")
        self.lbl_detail.setStyleSheet("font-size: 13px; color: #aaa;")
        mlay.addWidget(self.lbl_precision)
        mlay.addWidget(self.lbl_recall)
        mlay.addWidget(self.lbl_detail, 1)
        root.addWidget(metric_box)

        # 明细表
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["帧", "图片", "模型框数 B", "框错 FP", "漏框 FN", "Precision", "Recall"]
        )
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        root.addWidget(self.table, 1)

        # 导出按钮
        bar = QHBoxLayout()
        self.btn_export = QPushButton("导出报告")
        self.btn_export.setToolTip(
            "导出一个目录，包含 PDF 报告、带框图片帧（右下角标注漏检/误检数）、CSV 明细"
        )
        self.btn_export.clicked.connect(self._export_report)
        bar.addStretch(1)
        bar.addWidget(self.btn_export)
        root.addLayout(bar)

    def _title(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("font-size: 18px; font-weight: bold; padding: 8px 0;")
        return lbl

    def _big_metric(self, name: str) -> QLabel:
        lbl = QLabel(f"{name}\n—")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #4a9eff;"
            "background: #2b2b2b; padding: 18px; border-radius: 6px;"
            "min-width: 180px;"
        )
        return lbl

    def on_enter(self) -> None:
        self._refresh()

    def _refresh(self) -> None:
        overall = compute_overall(self.state)
        self.lbl_precision.setText(
            f"Precision (精确率)\n{fmt_ratio(overall['precision'])}"
        )
        self.lbl_recall.setText(
            f"Recall (召回率)\n{fmt_ratio(overall['recall'])}"
        )
        self.lbl_detail.setText(
            f"总框数 B = {overall['boxes']}\n"
            f"TP = {overall['tp']}\n"
            f"FP (框错) = {overall['fp']}\n"
            f"FN (漏框) = {overall['fn']}\n"
            f"图片数 = {len(self.state.frames)}"
        )

        # 明细表
        self.table.setRowCount(len(self.state.frames))
        for i, f in enumerate(self.state.frames):
            m = compute_frame(f)
            cells = [
                str(i + 1),
                os.path.basename(f.path),
                str(m["boxes"]),
                str(m["fp"]),
                str(m["fn"]),
                fmt_ratio(m["precision"]),
                fmt_ratio(m["recall"]),
            ]
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if c in (2, 3, 4):
                    item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(i, c, item)
        self.table.resizeColumnsToContents()

    def _export_report(self) -> None:
        if not self.state.frames:
            QMessageBox.information(self, "提示", "没有可导出的数据")
            return
        parent_dir = QFileDialog.getExistingDirectory(self, "选择导出位置")
        if not parent_dir:
            return
        self.btn_export.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            out_dir = export_report(self.state, parent_dir)
            QMessageBox.information(
                self,
                "导出成功",
                f"报告已导出到：\n{out_dir}\n\n"
                f"包含：\n  report.pdf  （PDF 报告）\n"
                f"  detail.csv  （CSV 明细）\n"
                f"  frames/     （带框图片帧）",
            )
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "导出失败", str(e))
        finally:
            QApplication.restoreOverrideCursor()
            self.btn_export.setEnabled(True)
