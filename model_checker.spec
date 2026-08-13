# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置 — YOLO Model Checker (onedir, windowed)。

构建：uv run pyinstaller model_checker.spec --noconfirm
产物：dist/ModelChecker/ModelChecker.exe
"""

from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

# onnxruntime 的 C 扩展子模块需显式收集，避免运行时 ImportError
hiddenimports = []
hiddenimports += collect_submodules("onnxruntime")
hiddenimports += collect_submodules("onnxruntime.capi")
hiddenimports += collect_submodules("cv2")
hiddenimports += [
    "PIL._tkinter_finder",
    "reportlab.graphics.barcode",
    "reportlab.graphics.charts",
]

a = Analysis(
    ["main.py"],
    pathex=["."],
    binaries=[],
    datas=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # 未使用的重型模块，减小体积
        "tkinter",
        "matplotlib",
        "pytest",
        "IPython",
        "notebook",
        "jupyter",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ModelChecker",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # windowed：无控制台
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="ModelChecker",
)
