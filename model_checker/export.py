"""导出检测报告到一个目录：PDF 报告 + 带框图片帧 + CSV 明细。

导出目录结构：
  model_checker_report_YYYYMMDD_HHMMSS/
    ├── report.pdf      # PDF 报告（总指标 + 逐帧明细表）
    ├── detail.csv      # CSV 明细表
    └── frames/         # 带检测框的图片帧
        ├── frame_0001.jpg   # 右下角标注漏检/误检数
        └── ...
"""

from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .image_utils import draw_detections, load_image_bgr
from .metrics import compute_frame, compute_overall, fmt_ratio
from .state import SessionState

_FONT_PATHS = [
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simsun.ttc",
]


def _cjk_font_path() -> str | None:
    for p in _FONT_PATHS:
        if os.path.exists(p):
            return p
    return None


def _pil_font(font_path: str | None, size: int):
    if font_path:
        try:
            return ImageFont.truetype(font_path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _annotate_corner(
    img_bgr: np.ndarray, missed: int, wrong: int, font_path: str | None
) -> np.ndarray:
    """在图片右下角画黑底白字标注：漏检:X  误检:Y。"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil)
    text = f"漏检:{missed}  误检:{wrong}"
    fs = max(16, pil.width // 40)
    font = _pil_font(font_path, fs)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = 6
    x = pil.width - tw - pad * 2 - 8
    y = pil.height - th - pad * 2 - 8
    draw.rectangle([x, y, x + tw + pad * 2, y + th + pad * 2], fill=(0, 0, 0))
    draw.text(
        (x + pad - bbox[0], y + pad - bbox[1]),
        text,
        fill=(255, 255, 255),
        font=font,
    )
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def _write_csv(path: Path, state: SessionState, overall: dict) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["帧", "图片", "模型框数B", "框错FP", "漏框FN", "TP", "Precision", "Recall"]
        )
        for i, f in enumerate(state.frames):
            m = compute_frame(f)
            w.writerow(
                [
                    i + 1,
                    f"frame_{i + 1:04d}.jpg",
                    m["boxes"],
                    m["fp"],
                    m["fn"],
                    m["tp"],
                    fmt_ratio(m["precision"]),
                    fmt_ratio(m["recall"]),
                ]
            )
        w.writerow([])
        w.writerow(
            [
                "总体",
                "",
                overall["boxes"],
                overall["fp"],
                overall["fn"],
                overall["tp"],
                fmt_ratio(overall["precision"]),
                fmt_ratio(overall["recall"]),
            ]
        )


def _register_pdf_font(font_path: str | None) -> str:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    if font_path is None:
        return "Helvetica"
    try:
        kwargs = {}
        if font_path.lower().endswith(".ttc"):
            kwargs["subfontIndex"] = 0
        pdfmetrics.registerFont(TTFont("CJK", font_path, **kwargs))
        return "CJK"
    except Exception:
        return "Helvetica"


def _write_pdf(
    path: Path, state: SessionState, overall: dict, font_path: str | None
) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    font_name = _register_pdf_font(font_path)
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
    )
    base = getSampleStyleSheet()["Normal"]
    title_style = ParagraphStyle(
        "t", parent=base, fontName=font_name, fontSize=20, leading=26
    )
    h2 = ParagraphStyle("h2", parent=base, fontName=font_name, fontSize=14, leading=18)
    normal = ParagraphStyle("n", parent=base, fontName=font_name, fontSize=11, leading=16)
    metric_style = ParagraphStyle(
        "m",
        parent=base,
        fontName=font_name,
        fontSize=16,
        leading=22,
        textColor=colors.HexColor("#1a73e8"),
    )

    story = []
    story.append(Paragraph("YOLO 模型检测效果报告", title_style))
    story.append(Spacer(1, 8))
    story.append(Paragraph(f"模型：{state.model_path}", normal))
    story.append(Paragraph(f"视频：{state.video_path}", normal))
    story.append(
        Paragraph(
            f"参数：conf={state.conf}  iou={state.iou}  imgsz={state.imgsz}"
            f"  抽帧数={state.frame_count_target}",
            normal,
        )
    )
    story.append(Paragraph(f"图片数：{len(state.frames)}", normal))
    story.append(Spacer(1, 12))

    story.append(Paragraph("总体指标", h2))
    story.append(
        Paragraph(f"Precision（精确率）：{fmt_ratio(overall['precision'])}", metric_style)
    )
    story.append(
        Paragraph(f"Recall（召回率）：{fmt_ratio(overall['recall'])}", metric_style)
    )
    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            f"总框数 B = {overall['boxes']}    TP = {overall['tp']}"
            f"    FP（框错）= {overall['fp']}    FN（漏框）= {overall['fn']}",
            normal,
        )
    )
    story.append(Spacer(1, 14))

    story.append(Paragraph("逐帧明细", h2))
    data = [["帧", "图片", "框数B", "框错FP", "漏框FN", "TP", "Precision", "Recall"]]
    for i, f in enumerate(state.frames):
        m = compute_frame(f)
        data.append(
            [
                str(i + 1),
                f"frame_{i + 1:04d}.jpg",
                str(m["boxes"]),
                str(m["fp"]),
                str(m["fn"]),
                str(m["tp"]),
                fmt_ratio(m["precision"]),
                fmt_ratio(m["recall"]),
            ]
        )
    tbl = Table(data, repeatRows=1, colWidths=[25, 110, 40, 40, 40, 30, 60, 60])
    tbl.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), font_name),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4a4a4a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ALIGN", (2, 1), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, colors.HexColor("#f5f5f5")],
                ),
            ]
        )
    )
    story.append(tbl)
    doc.build(story)


def export_report(state: SessionState, parent_dir: str) -> Path:
    """导出报告到 parent_dir 下的时间戳子目录，返回该目录路径。"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(parent_dir) / f"model_checker_report_{ts}"
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    font_path = _cjk_font_path()
    overall = compute_overall(state)

    # 1. 带框图片帧（右下角标注漏检/误检数）
    for i, f in enumerate(state.frames):
        img = load_image_bgr(f.path)
        if img is None:
            continue
        drawn = draw_detections(img, f.detections, state.class_names)
        drawn = _annotate_corner(drawn, f.missed, f.wrong, font_path)
        out_path = frames_dir / f"frame_{i + 1:04d}.jpg"
        ok, buf = cv2.imencode(".jpg", drawn)
        if ok:
            buf.tofile(str(out_path))

    # 2. CSV 明细
    _write_csv(out_dir / "detail.csv", state, overall)

    # 3. PDF 报告
    _write_pdf(out_dir / "report.pdf", state, overall, font_path)

    return out_dir
