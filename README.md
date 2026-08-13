# YOLO 模型识别效果检测工具

一个基于 PySide6 的桌面图形化软件，用于检测训练的 YOLO（v8/v11）ONNX 模型在视频上的目标识别效果，通过均匀抽帧 + 人工标注漏框/框错，最终计算模型的 **精确率 (Precision)** 与 **召回率 (Recall)**。

## 功能流程

1. **选择模型与视频**：选择 `.onnx` 模型文件、待检测视频，配置抽帧数量、置信度阈值、NMS IoU 阈值、输入尺寸。
2. **抽帧与图片管理**：对视频均匀抽帧，可查看缩略图，删除或添加图片。
3. **模型推理**：批量对图片运行 ONNX 推理，框选每个目标并预览。
4. **逐帧审核标注**：查看每张图片框选结果，判断漏框与框错，填写每帧漏框数 (FN) 与框错数 (FP)。
5. **检测结果报告**：输出整体与逐帧的 Precision / Recall，支持导出报告目录（PDF 报告 + CSV 明细 + 带框图片帧）。

## 指标定义

每帧模型画框数 `B`，人工填漏框数 `FN`、框错数 `FP`，则 `TP = B - FP`。

- **Precision (精确率)** = `Σ TP / Σ B`
- **Recall (召回率)** = `Σ TP / (Σ TP + Σ FN)`

无预测时 Precision 显示 N/A；无目标时 Recall 显示 N/A。

## 导出报告

步骤 5 可将检测结果导出到一个时间戳目录，结构如下：

```
model_checker_report_YYYYMMDD_HHMMSS/
├── report.pdf      # PDF 报告（总指标 + 逐帧明细表）
├── detail.csv      # CSV 明细表（含总体行）
└── frames/         # 带检测框的图片帧
    └── frame_0001.jpg   # 右下角标注漏检/误检数
```

- PDF/CSV 中文依赖系统字体（`simhei.ttf` / `msyh.ttc` / `simsun.ttc`），缺失时回退默认字体。
- 带框图片在右下角以黑底白字标注 `漏检:X  误检:Y`。

## 环境与运行

依赖 Python ≥ 3.12，使用 uv 管理环境。

```bash
# 安装依赖
uv sync

# 运行
uv run python main.py
```

## 技术栈

- **PySide6** — GUI 框架
- **onnxruntime** — ONNX 模型推理
- **opencv-python** — 视频抽帧与图像处理
- **numpy** — 数组与后处理（NMS）
- **reportlab** — PDF 报告生成
- **pillow** — 带框图片的中英文标注

## 项目结构

```
model_checker/
├── main.py                      # 启动入口（深色主题）
└── model_checker/
    ├── state.py                 # 会话状态与数据结构
    ├── inference.py             # YOLOv8 ONNX 推理（letterbox + NMS）
    ├── video.py                 # 视频均匀抽帧
    ├── image_utils.py           # 图像转换与画框
    ├── metrics.py               # Precision/Recall 计算
    ├── export.py                # 报告导出（PDF + CSV + 带框图片）
    └── gui/
        ├── base_page.py         # 向导页面基类（独立模块，避免循环导入）
        ├── main_window.py       # 主向导窗口
        ├── image_viewer.py      # 可缩放图片查看器
        ├── workers.py           # 后台线程 Worker
        ├── page_select.py       # 步骤1
        ├── page_frames.py       # 步骤2
        ├── page_inference.py    # 步骤3
        ├── page_review.py       # 步骤4
        └── page_report.py       # 步骤5
```
