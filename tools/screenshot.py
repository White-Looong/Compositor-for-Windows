# -*- coding: utf-8 -*-
"""生成一张界面截图（离屏渲染），用于预览。

    python -m tools.screenshot [输出路径]
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF                 # noqa: E402
from PySide6.QtGui import QFontDatabase            # noqa: E402
from PySide6.QtWidgets import QApplication         # noqa: E402

from src.core.layer import Layer                   # noqa: E402
from src.core.selection import Selection           # noqa: E402
from src.ui.main_window import MainWindow          # noqa: E402


def _load_system_fonts():
    """离屏渲染时字体库是空的，手动挂载系统字体，否则中文会显示成方块。"""
    add = QFontDatabase.addApplicationFont
    for name in ("msyh.ttc", "msyhbd.ttc", "simhei.ttf", "segoeui.ttf",
                 "segoeuib.ttf"):
        p = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", name)
        if os.path.exists(p):
            add(p)


def _grad(w, h, c0, c1, horizontal=True):
    t = np.linspace(0, 1, w if horizontal else h, dtype=np.float32)
    a = np.zeros((h, w, 4), np.uint8)
    for i in range(3):
        v = c0[i] + (c1[i] - c0[i]) * t
        a[..., i] = (v[None, :] if horizontal else v[:, None]).astype(np.uint8)
    a[..., 3] = 255
    return a


def _disc(w, h, rgb, radius=0.42, feather=0.06):
    a = np.zeros((h, w, 4), np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2, h / 2
    r = np.sqrt(((xx - cx) / (w * radius)) ** 2 + ((yy - cy) / (h * radius)) ** 2)
    alpha = np.clip((1.0 - r) / feather, 0, 1)
    a[..., :3] = rgb
    a[..., 3] = (alpha * 255).astype(np.uint8)
    return a


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "docs/screenshot.png"
    tool = sys.argv[2] if len(sys.argv) > 2 else "move"
    app = QApplication(sys.argv)
    _load_system_fonts()
    win = MainWindow()
    win.resize(1440, 900)
    win.show()
    app.processEvents()

    doc = win.doc
    doc.name = "示例工程"
    doc.layers.clear()

    bg = Layer("背景", "image", _grad(1600, 1000, (245, 246, 250), (214, 220, 235)))
    bg.tx, bg.ty = 800.0, 500.0
    doc.layers.append(bg)

    rect = Layer("色块", "image", _grad(700, 460, (255, 168, 60), (232, 62, 140)))
    rect.tx, rect.ty = 620.0, 470.0
    rect.rot = -12.0
    rect.blend = "Multiply"
    rect.opacity = 0.9
    doc.layers.append(rect)

    disc = Layer("光斑", "image", _disc(620, 620, (70, 150, 255)))
    disc.tx, disc.ty = 1080.0, 640.0
    disc.sx = disc.sy = 0.85
    disc.blend = "Screen"
    doc.layers.append(disc)

    g = Layer("组 1", "group")
    g.blend = "Pass Through"
    inner = Layer("网格", "image", _grad(1200, 300, (30, 40, 60), (120, 140, 180)))
    inner.tx, inner.ty = 800.0, 830.0
    inner.opacity = 0.55
    inner.blend = "Overlay"
    g.children.append(inner)
    doc.layers.append(g)

    # 在"色块"图层上画两笔，展示画笔（受图层变换的逆变换换算）
    from src.core.paint import Stroke
    for pts in ([(480, 430), (600, 500), (730, 460)],
                [(520, 380), (660, 430), (780, 400)]):
        st = Stroke(doc=doc, layer=rect, tool="brush", size=34, hardness=0.35,
                    opacity=0.85, flow=0.6, smoothing=0.4, target="pixel",
                    color=(20, 25, 40))
        if st.begin(pts[0]):
            for p in pts[1:]:
                st.extend(p)
            st.end()

    # 一个羽化过的椭圆选区，展示行进蚁线
    doc.selection = None
    win.commit_selection(
        Selection.ellipse(doc.width, doc.height, 830, 400, 1340, 900))
    win.opts.feather = 0.0
    if doc.selection is not None:
        doc.selection.feather(18)

    if tool == "adjust":
        # 加一条曲线调整层，展示「调整层 + 曲线编辑器 + 直方图」
        win.add_adjustment_layer("curves")
        adj = win.selected_layer()
        adj.adjust["params"]["points"]["RGB"] = [
            [0.0, 0.02], [0.35, 0.28], [0.72, 0.82], [1.0, 1.0]]
        win.select_layer(adj.id)
        win.panel.rebuild()
        win.inspector.refresh()
    elif tool == "text":
        # 建一个文字图层，展示「文字工具 + 属性面板文字区 + 工具选项条字体设置」
        from src.core.text import ensure_fonts, sync_text_image
        ensure_fonts()
        tl = win.create_text_layer((800.0, 260.0), content="Compositor\n文字图层")
        tl.text["size"] = 130.0
        tl.text["color"] = [22, 26, 38]
        sync_text_image(tl)
        win.select_layer(tl.id)
        win.panel.rebuild()
        win.inspector.refresh()
    elif tool == "style":
        # 给"色块"加投影 + 描边，展示图层样式
        from src.core.effects import default_effects
        rect.effects = default_effects()
        rect.effects["drop_shadow"].update(
            enabled=True, distance=26.0, size=26.0, opacity=0.55,
            angle=45.0, color=[20, 24, 40])
        rect.effects["stroke"].update(
            enabled=True, size=7.0, position="外部", opacity=1.0,
            color=[255, 255, 255])
        disc.effects = default_effects()
        disc.effects["outer_glow"].update(
            enabled=True, size=34.0, opacity=0.85, color=[255, 240, 190])
        win.select_layer(rect.id)
        win.panel.rebuild()
        win.inspector.refresh()
    elif tool == "mix":
        # 新增的 4 种调整层：展示通道混合器面板
        win.add_adjustment_layer("channel_mixer")
        adj = win.selected_layer()
        adj.adjust["params"]["mix"]["红"] = {"red": 40.0, "green": 50.0,
                                             "blue": 20.0, "const": 0.0}
        win.select_layer(adj.id)
        win.panel.rebuild()
        win.inspector.refresh()
    elif tool == "smart":
        # 把"光斑"转成智能对象并加一条智能滤镜，展示缩略图角标与属性面板分组
        from src.core.smart import (convert_to_smart, new_filter,
                                    new_smart_instance)
        shell = convert_to_smart(doc, disc.id)
        shell.so_filters = [new_filter("motion", {"distance": 40, "angle": -20})]
        twin = new_smart_instance(doc, shell.id)
        twin.tx = 480.0
        twin.ty = 700.0
        twin.sx = twin.sy = 0.55
        win.select_layer(shell.id)
        win.panel.rebuild()
        win.inspector.refresh()
    elif tool == "filter":
        # 给"光斑"加一层高斯模糊，展示滤镜的破坏性结果
        from src.core.filters import apply_filter_array
        disc.image = apply_filter_array(disc.image, "gaussian",
                                        {"radius": 24.0})
        win.select_layer(disc.id)
        win.panel.rebuild()
    else:
        win.select_layer(disc.id)
        win.panel.rebuild()

    win._do_render()
    win.view.update_selection_overlay()
    win.view.fit()
    win.set_tool("move" if tool in ("adjust", "filter") else tool)
    if tool in ("brush", "eraser"):
        win.opts.size = 90
        win.view._update_brush_ring(QPointF(560.0, 300.0))
    app.processEvents()
    app.processEvents()

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    pm = win.grab()
    pm.save(out)
    print("截图已保存: %s (%dx%d)" % (os.path.abspath(out), pm.width(), pm.height()))


if __name__ == "__main__":
    main()
