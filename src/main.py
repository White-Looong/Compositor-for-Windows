# -*- coding: utf-8 -*-
"""Compositor for Windows —— 启动入口。"""

from __future__ import annotations

import os
import sys

# 保证 `python -m src.main` 和 `python src/main.py` 都能导入 src 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    # Qt6 起高 DPI 缩放默认开启，这两个属性已弃用；仅在老版本 Qt 上才需要显式设置
    if hasattr(Qt, "AA_EnableHighDpiScaling") and not hasattr(Qt, "HighDpiScaleFactorRoundingPolicy"):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("Compositor for Windows")
    app.setOrganizationName("compositor-win")

    from .ui.main_window import MainWindow
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
