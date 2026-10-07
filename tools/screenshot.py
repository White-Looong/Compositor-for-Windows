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

from PySide6.QtCore import QPoint, QPointF            # noqa: E402
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
    elif tool == "bevel":
        # 第十批：斜面浮雕 / 渐变叠加 / 图案叠加 / 内发光
        from src.core.effects import default_effects
        rect.effects = default_effects()
        rect.effects["gradient_overlay"].update(
            enabled=True, opacity=0.55, style="线性", angle=30.0,
            color=[255, 214, 90], color2=[190, 40, 120])
        rect.effects["bevel_emboss"].update(
            enabled=True, size=34.0, depth=1.8, soften=2.0,
            angle=45.0, altitude=32.0, style="内斜面")
        rect.effects["drop_shadow"].update(
            enabled=True, distance=24.0, size=24.0, opacity=0.5,
            angle=45.0, color=[20, 24, 40])
        disc.blend = "Normal"      # Screen 会把下面的图案吃掉
        disc.effects = default_effects()
        disc.effects["pattern_overlay"].update(
            enabled=True, opacity=0.6, blend="Multiply", pattern="斜纹",
            scale=22.0, color=[40, 60, 90])
        disc.effects["inner_glow"].update(
            enabled=True, opacity=0.85, size=44.0, blend="Screen",
            color=[255, 255, 235])
        disc.effects["stroke"].update(
            enabled=True, size=6.0, position="居中", opacity=0.9,
            color=[255, 255, 255])
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
    elif tool == "huesat":
        # 第十二批：色相/饱和度「分色彩范围」—— 只推移红色那一带
        win.add_adjustment_layer("hue_sat")
        adj = win.selected_layer()
        adj.adjust["params"].update(range="红色", hue=45, saturation=25,
                                    lightness=0)
        win.select_layer(adj.id)
        win.panel.rebuild()
        win.inspector.refresh()
    elif tool == "gradient":
        # 第十二批：渐变映射的任意控制点渐变编辑器
        win.add_adjustment_layer("gradient_map")
        adj = win.selected_layer()
        adj.adjust["params"]["stops"] = [
            [0.0, 26, 22, 66], [0.34, 214, 62, 120], [0.62, 255, 186, 90],
            [1.0, 250, 250, 245]]
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
    elif tool == "quick":
        # 第十一批：快速蒙版 —— 红罩盖住未选中的地方，选区是"涂"出来的
        from src.core import quick_mask
        quick_mask.enter(doc)
        doc.quick_mask[:] = 0
        doc.quick_mask[380:880, 700:1400] = 255
        doc.quick_mask[300:520, 180:520] = 255
        win._sync_quick_mask_ui()
        win.select_layer(rect.id)
        win.panel.rebuild()
    elif tool == "channels":
        # 第十七批：通道面板 —— R/G/B/Alpha 四路 + 一个把右半边打透明的通道，
        # 顺便展示"关掉红通道显示为纸白"（左半）和通道生效（右半透明）
        from src.core import channels as CH

        win.select_layer(rect.id)
        win.panel.rebuild()
        win._do_render()
        # 一个只盖住左半的通道：关红之后画面分成"白 / 棋盘格"两半
        c = CH.add_channel(win.doc, "打孔")
        m = np.zeros((doc.height, doc.width), np.uint8)
        m[:, :doc.width // 2] = 255
        win.doc.detach_channel(c)
        c.mask = m
        c2 = CH.add_channel(win.doc, "柔边")
        m2 = np.zeros((doc.height, doc.width), np.uint8)
        m2[:, :doc.width // 2] = np.linspace(255, 0, doc.width // 2,
                                             dtype=np.uint8)
        win.doc.detach_channel(c2)
        c2.mask = m2
        win.doc.channel_view["R"] = False
        win.commit("通道")
        win._do_render()
        win.channels.rebuild()
        win.inspector.refresh()
    elif tool == "psd":
        # 第十六批：PSD 导入 —— 文字层还原成可编辑文字 + 图层样式接上
        from src.core.psd_import import load_psd
        from src.core.text import ensure_fonts
        from tools.psdfixture import demo_effects, write_psd

        ensure_fonts()
        fams = list(QFontDatabase.families())
        fam = fams[0] if fams else ""
        tmp_psd = os.path.join(os.environ.get("TEMP", "."), "shot_psd.psd")
        write_psd(tmp_psd, doc.width, doc.height, [
            dict(name="PSD 文字", left=380, top=150, right=1180, bottom=290,
                 rgb=(255, 0, 0),
                 text=dict(content="PSD 文字层\n改字依然可编辑", size=104.0,
                           color=(24, 30, 48), family=fam, tracking=8,
                           justification=2, auto_leading=1.25)),
            dict(name="PSD 样式", left=470, top=430, right=1130, bottom=830,
                 rgb=(214, 218, 228),
                 # 只挂"能看出形状"的几项：渐变叠加会把整块盖住，
                 # 9 种全开的话截图上看不出投影 / 描边 / 发光
                 effects=[demo_effects(k)[0] for k in
                          ("drop_shadow", "stroke", "outer_glow",
                           "bevel_emboss", "color_overlay")]),
        ])
        win.set_document(load_psd(tmp_psd), reset_history=True)
        # set_document 换掉了整个工程，补一层浅底，不然投影 / 外发光看不出来
        from src.core.document import make_image_layer
        flat = make_image_layer("底", _grad(doc.width, doc.height,
                                             (246, 247, 250), (222, 227, 238)),
                                doc.width, doc.height)
        win.doc.layers.insert(0, flat)
        win.select_layer(win.doc.layers[1].id)
        win.panel.rebuild()
        win.inspector.refresh()
    elif tool == "dust":
        # 第十二批续：大半径中间值「蒙尘与划痕」—— 铺一张带灰尘与划痕的"扫描件"，
        # 右半边用滤镜修好，左右对照（一眼看出灰尘 / 划痕被吃掉、底色没被糊掉）
        from src.core.filters import apply_filter_array

        rng = np.random.default_rng(20261006)
        h, w = doc.height, doc.width
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        tone = 152.0 + 46.0 * np.sin(xs / 190.0) * np.cos(ys / 240.0)
        arr = np.zeros((h, w, 4), np.uint8)
        arr[..., :3] = np.clip(tone, 0, 255).astype(np.uint8)[..., None]
        arr[..., 3] = 255
        rows = rng.integers(0, h, 1100)
        cols = rng.integers(0, w, 1100)
        arr[rows, cols, :3] = 18                     # 灰尘
        arr[418:425, 150:1450, :3] = 26              # 划痕
        arr[703:709, 320:1300, :3] = 32
        clean = apply_filter_array(arr, "dust", {"radius": 6, "threshold": 36})
        half = w // 2
        arr[:, half:] = clean[:, half:]
        arr[:, half - 1:half + 1, :3] = 90           # 对照线
        doc.selection = None
        scan = Layer("扫描件（右半已修复）", "image", arr)
        scan.tx, scan.ty = w / 2.0, h / 2.0
        doc.layers.append(scan)
        win.select_layer(scan.id)
        win.panel.rebuild()
    elif tool in ("brushtips", "brushdlg"):
        # 第十四批：笔刷增强 —— 铺一张"纸"，每一行用不同笔尖 / 动态参数画一道
        import math

        from src.core.document import Document, make_image_layer
        from src.core.paint import Stroke

        bw, bh = 1200, 760
        bdoc = Document(bw, bh, "笔刷示例")
        paper_arr = np.zeros((bh, bw, 4), np.uint8)
        paper_arr[..., :3] = 248
        paper_arr[..., 3] = 255
        paper = make_image_layer("笔刷示例", paper_arr, bw, bh)
        paper.tx, paper.ty = bw / 2.0, bh / 2.0
        bdoc.layers.append(paper)
        win.set_document(bdoc, reset_history=True)
        win.select_layer(paper.id)

        rows = [
            dict(size=46, hardness=0.9, shape="圆形"),
            dict(size=46, hardness=0.9, shape="方形"),
            dict(size=46, hardness=0.9, shape="菱形"),
            dict(size=34, hardness=0.85, roundness=0.32, angle=-35),
            dict(size=24, hardness=0.9, scatter=2.2, count=6, size_jitter=0.55,
                 seed=5),
            dict(size=56, hardness=0.8, texture="斜纹", texture_scale=26,
                 texture_depth=0.85),
            dict(size=42, hardness=0.75, pressure="大小+不透明度",
                 pressure_amount=1.0),
        ]
        for i, row in enumerate(rows):
            row = dict(row)
            y = 96.0 + i * 94.0
            path = [(110.0 + t * 13.0, y + math.sin(t / 2.6) * 15.0)
                    for t in range(76)]
            st = Stroke(doc=bdoc, layer=paper, size=row.pop("size"),
                        hardness=row.pop("hardness"), color=(32, 36, 52), **row)
            if st.begin(path[0]):
                for q in path[1:]:
                    st.extend(q)
                st.end()

        if tool == "brushdlg":
            # 顺便把这套参数写进工具选项条，截图里能看到摘要
            win.opts.brush.update(
                shape="菱形", roundness=0.5, angle=-30, spacing=0.18,
                scatter=1.8, count=5, size_jitter=0.4, texture="斜纹",
                texture_scale=30, texture_depth=0.7,
                pressure="大小+不透明度", airbrush=True, air_rate=15)
            win.opts.refresh_brush_summary()
            win.opts.size = 42
        win.panel.rebuild()
    elif tool in ("textpara", "textedit", "textchars"):
        # 第十五批：文字增强 —— 段落缩进 / 两端对齐 / 逐字调整 / 画布内编辑
        from src.core.document import (Document, make_image_layer,
                                       make_text_layer)
        from src.core.text import ensure_fonts, sync_text_image

        ensure_fonts()
        tw, th = 1100, 640
        tdoc = Document(tw, th, "文字增强")
        paper = np.zeros((th, tw, 4), np.uint8)
        paper[..., :3] = 250
        paper[..., 3] = 255
        bg = make_image_layer("纸", paper, tw, th)
        bg.tx, bg.ty = tw / 2.0, th / 2.0
        tdoc.layers.append(bg)
        tdoc.layers.append(make_text_layer(
            "正文", tw, th,
            {"content": "把声音只送到你耳边\n这是参量阵超声扬声器\n最直观的效果演示",
             "size": 54.0, "color": [26, 32, 48], "line_height": 1.5,
             "align": "justify",
             "para": {"indent_first": 108.0, "indent_left": 10.0,
                      "indent_right": 10.0, "space_after": 26.0}},
            center=(tw / 2.0, th / 2.0)))
        tlay = tdoc.layers[-1]
        # 逐字调整：第 3 个字后加宽字距、第 14 个字压低基线并拉宽
        tlay.text["chars"] = {"2": {"dx": 30.0},
                              "13": {"dy": -16.0, "scale": 1.3}}
        sync_text_image(tlay)
        win.set_document(tdoc, reset_history=True)
        win.select_layer(tlay.id)
        win.panel.rebuild()
        win.inspector.refresh()
        if tool == "textedit":
            win.view.begin_text_edit(tlay)
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
    win.set_tool("brush" if tool in ("quick", "brushtips", "brushdlg")
                 else ("move" if tool in ("adjust", "filter", "huesat",
                                          "gradient", "dust", "textpara",
                                          "textedit", "textchars")
                       else tool))
    if tool in ("brush", "eraser", "quick"):
        win.opts.size = 90
        win.view._update_brush_ring(QPointF(560.0, 300.0))
    app.processEvents()
    app.processEvents()

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    if tool in ("textpara", "textedit"):
        # 段落那五行在属性面板下面，滚到刚好露出「左缩进」再截
        sb = win.inspector_scroll.verticalScrollBar()
        top = win.inspector.para_spins["indent_left"].mapTo(
            win.inspector, QPoint(0, 0)).y()
        sb.setValue(max(0, min(sb.maximum(), top - 24)))
        app.processEvents()
    if tool == "range":
        # 色彩范围是对话框，截它自己
        from src.ui.color_range_dialog import ColorRangeDialog
        dlg = ColorRangeDialog(win, win.composite_rgb())
        dlg.samples = [(255, 168, 60)]
        dlg.combo_preset.setCurrentIndex(0)
        dlg.slider.setValue(75)
        dlg.show()
        app.processEvents()
        app.processEvents()
        pm = dlg.grab()
    elif tool == "textchars":
        # 逐字调整也是对话框
        from src.ui.text_dialog import CharAdjustDialog
        tdlg = CharAdjustDialog(win, win)
        tdlg.select_index(2)
        tdlg.set_field("dx", 34)
        tdlg.show()
        app.processEvents()
        app.processEvents()
        pm = tdlg.grab()
    elif tool == "brushdlg":
        # 笔刷设置也是对话框
        from src.ui.brush_dialog import BrushDialog
        bdlg = BrushDialog(win, win.opts.brush, size=win.opts.size,
                           hardness=win.opts.hardness)
        bdlg.show()
        app.processEvents()
        app.processEvents()
        pm = bdlg.grab()
    else:
        pm = win.grab()
    pm.save(out)
    print("截图已保存: %s (%dx%d)" % (os.path.abspath(out), pm.width(), pm.height()))


if __name__ == "__main__":
    main()
