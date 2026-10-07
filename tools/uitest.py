# -*- coding: utf-8 -*-
"""界面冒烟测试（离屏运行，不弹窗）：把主要交互走一遍，捕捉异常。

    python -m tools.uitest
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import (QApplication, QComboBox,  # noqa: E402
                               QSlider)

from src.core.document import Document, make_image_layer   # noqa: E402
from src.core.layer import Layer                     # noqa: E402
from src.ui.main_window import MainWindow            # noqa: E402


def _img(w, h, rgb):
    a = np.zeros((h, w, 4), np.uint8)
    a[..., :3] = rgb
    a[..., 3] = 255
    return a


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    app.processEvents()

    doc = win.doc
    lay = Layer("测试图层", "image", _img(400, 300, (200, 40, 40)))
    lay.tx, lay.ty = doc.width / 2, doc.height / 2
    lay.sx = lay.sy = 0.8
    doc.layers.append(lay)
    win.select_layer(lay.id)
    win.commit("add")
    win.panel.rebuild()
    win._do_render()
    app.processEvents()
    print("  图层添加 / 选中 / 渲染 OK")

    win.panel.refresh_header()
    win.inspector.refresh()
    win.view._update_handles()
    app.processEvents()
    print("  面板 / 属性 / 手柄同步 OK")

    for kind in ("white", "black", "alpha", "luma", "invert"):
        win.add_mask(kind)
        win._do_render()
    win.toggle_mask()
    win.delete_mask()
    app.processEvents()
    print("  蒙版操作 OK")

    # 变换手柄交互（走几何计算路径）
    win.view._drag = {"role": "se", "start": win.view.mapToScene(win.rect().center()),
                      "layer": lay, "sx": lay.sx, "sy": lay.sy, "rot": lay.rot,
                      "tx": lay.tx, "ty": lay.ty, "a0": 0.0, "moved": True}
    from PySide6.QtCore import QPointF, Qt
    win.view._apply_drag(QPointF(lay.tx + 60, lay.ty + 40), Qt.NoModifier)
    win.view._drag = None
    win.view._update_handles()
    print("  缩放手柄 -> %.3f x %.3f" % (lay.sx, lay.sy))

    win.group_selection()
    win._do_render()
    print("  建组 OK，当前图层数 %d" % len(win.doc.layers))

    win.duplicate_layer()
    win._do_render()
    print("  复制图层 OK")

    win.undo()
    win._do_render()
    win.redo()
    win._do_render()
    print("  撤销 / 重做 OK")

    # 拖拽排序后的树重建
    win.panel.on_dropped()
    print("  图层树重建 OK，顶层 %d 个" % len(win.doc.layers))

    # ---------- 选区 ----------
    from PySide6.QtCore import QPointF
    from src.core.selection import Selection

    win.select_all()
    assert win.doc.selection is not None and not win.doc.selection.is_empty
    win.invert_selection()
    assert win.doc.selection.is_empty, "全选再反选应为空"
    win.select_all()
    win.deselect()
    assert win.doc.selection is None
    print("  全选 / 反选 / 取消 OK")

    win.commit_selection(Selection.rect(win.doc.width, win.doc.height,
                                        200, 200, 700, 600))
    assert win.doc.selection is not None
    box = win.doc.selection.bbox()
    assert box == (200, 200, 700, 600), box
    win.view.update_selection_overlay()
    print("  矩形选区 + 蚂蚁线 OK，bbox=%s" % (box,))

    # 魔棒（基于合成图）
    win.set_tool("magic")
    win.view._do_magic(QPointF(400.0, 400.0), Qt.NoModifier)
    assert win.doc.selection is not None and not win.doc.selection.is_empty
    print("  魔棒 OK，选中 %d px" % int(win.doc.selection.mask.sum() / 255))

    win.set_tool("rect")
    win.view.cancel_operation()
    win.set_tool("polygon")
    win.view._poly_click(QPointF(100.0, 100.0))
    win.view._poly_click(QPointF(500.0, 120.0))
    win.view._poly_click(QPointF(300.0, 500.0))
    win.view._commit_polygon()
    assert win.doc.selection is not None and not win.doc.selection.is_empty
    print("  多边形套索 OK，选中 %d px" % int(win.doc.selection.mask.sum() / 255))
    win.set_tool("move")

    # ---------- 画笔 ----------
    # 撤销/重做会整棵克隆图层树，所以要重新取回当前文档里的图层对象
    lay = win.doc.find(lay.id)
    assert lay is not None

    # 选区与图层完全不相交时应该画不上去（正确的限制行为）
    win.select_layer(lay.id)
    win.commit_selection(Selection.rect(win.doc.width, win.doc.height,
                                        0, 0, 60, 60))
    win.set_tool("brush")
    blocked = win.view._start_paint(QPointF(30.0, 30.0), Qt.NoModifier)
    assert not blocked, "选区与图层不相交时不应允许绘制"
    print("  选区不相交时拒绝绘制 OK")

    # 选区要真正盖住图层（图层经过前面的缩放拖拽，位置已经变了）
    cx, cy = lay.tx, lay.ty
    win.commit_selection(Selection.rect(win.doc.width, win.doc.height,
                                        cx - 200, cy - 200, cx + 200, cy + 200))
    win.set_tool("brush")
    win.opts.size = 40
    win.opts.hardness = 1.0
    win.opts.opacity = 1.0
    before = lay.image.copy()
    started = win.view._start_paint(QPointF(cx, cy), Qt.NoModifier)
    assert started, "画笔没能开始"
    win.view._stroke.extend((cx + 20, cy + 10))
    win.view._stroke.extend((cx + 40, cy + 20))
    win.view._stroke.end()
    win.view._stroke = None
    win.after_pixel_edit("画笔")
    assert not np.array_equal(lay.image, before), "画笔没有改动任何像素"
    print("  画笔描边 OK，改动像素 %d 个"
          % int((np.abs(lay.image.astype(int) - before.astype(int)).sum(axis=2) > 0).sum()))

    # 撤销必须还原像素（写时复制）
    win.undo()
    after_undo = win.doc.find(lay.id)
    assert after_undo is not None and np.array_equal(after_undo.image, before), \
        "撤销后像素没还原"
    win.redo()
    print("  画笔的撤销 / 重做 OK")

    # ---------- 笔刷增强（笔尖形状 / 散布 / 纹理 / 喷枪 / 压感） ----------
    from src.core.brush import BRUSH_DEFAULTS
    from src.ui.brush_dialog import BrushDialog, stroke_kwargs, summary_text

    dlg = BrushDialog(win, dict(BRUSH_DEFAULTS), size=30, hardness=0.8)
    n_slider = len(dlg.findChildren(QSlider))
    assert n_slider >= 8, n_slider
    assert dlg.preview._pm is not None, "预览应该画出来了"
    dlg.params["shape"] = "方形"
    dlg._sync_widgets()
    dlg.params["scatter"] = 2.5
    dlg.params["count"] = 4
    dlg.params["texture"] = "棋盘"
    dlg.params["airbrush"] = True
    dlg._sync_widgets()
    dlg._refresh_preview()
    got = dlg.result_params()
    assert got["shape"] == "方形" and got["airbrush"] is True
    assert dlg.original_params()["shape"] == "圆形", "取消时要能整份还原"
    kw = stroke_kwargs(got)
    assert kw["shape"] == "方形" and kw["count"] == 4 and kw["airbrush"] is True
    assert "方形" in summary_text(got) and "纹理 棋盘" in summary_text(got)
    # 复位
    dlg._reset()
    assert dlg.result_params()["shape"] == "圆形"
    assert dlg.result_params()["texture"] == "无"
    dlg.deleteLater()
    print("  笔刷设置对话框：%d 个滑块 + 预览 + 复位 / 取消还原 OK" % n_slider)

    # 工具选项条上的摘要与参数要跟着走
    win.opts.brush = dict(BRUSH_DEFAULTS)
    win.opts.brush.update(shape="菱形", scatter=1.5, count=3)
    win.opts.refresh_brush_summary()
    assert "菱形" in win.opts.lab_brush.text()
    assert "散布" in win.opts.lab_brush.text()

    # 画一笔：Stroke 要真的用上这些参数
    win.commit_selection(None)
    win.set_tool("brush")
    win.opts.size = 40
    win.opts.hardness = 1.0
    lay = win.doc.find(lay.id)
    win.select_layer(lay.id)
    started = win.view._start_paint(QPointF(lay.tx, lay.ty), Qt.NoModifier)
    assert started, "画笔没能开始"
    st = win.view._stroke
    assert st.shape == "菱形" and st.count == 3 and abs(st.scatter - 1.5) < 1e-6, \
        (st.shape, st.count, st.scatter)
    assert not win.view._air_timer.isActive(), "没开喷枪不该启动定时器"
    win.view._stroke.end()
    win.view._stroke = None

    # 喷枪：开始描边就要起定时器，松手要停
    win.opts.flow = 0.25          # 流量调低才看得出"持续加深"
    win.opts.opacity = 0.5
    win.opts.fg.set_color((255, 255, 255))   # 换成白色，画在已有的黑笔迹上才看得出来
    win.opts.brush.update(airbrush=True, air_rate=20)
    started = win.view._start_paint(QPointF(lay.tx, lay.ty), Qt.NoModifier)
    assert started
    assert win.view._stroke.airbrush, "Stroke 没拿到喷枪参数"
    assert win.view._air_timer.isActive(), "喷枪没启动定时器"
    assert win.view._air_timer.interval() == 50, win.view._air_timer.interval()
    before_air = int(lay.image[..., :3].sum())
    win.view._airbrush_tick()          # 手动补一笔（不用真的等 50ms）
    assert int(lay.image[..., :3].sum()) > before_air, "喷枪补笔没有改动像素"
    win.view._stroke.end()
    win.view._stroke = None
    win.view._air_timer.stop()
    win.opts.fg.set_color((0, 0, 0))
    win.opts.brush = dict(BRUSH_DEFAULTS)
    win.opts.refresh_brush_summary()
    win.after_pixel_edit("画笔")
    win.undo()
    print("  笔刷参数贯通到 Stroke + 喷枪定时器起停 OK")

    # ---------- 填充 / 清除 / 从选区建蒙版 ----------
    win.commit_selection(Selection.rect(win.doc.width, win.doc.height,
                                        cx - 150, cy - 150, cx + 150, cy + 150))
    win.fill_with_fg(None, use_fg=True)
    print("  用前景色填充 OK")

    win.delete_in_selection()
    print("  清除选区内容 OK")

    win.select_layer(lay.id)
    win.mask_from_selection()
    assert win.doc.find(lay.id).mask is not None
    print("  从选区建立蒙版 OK")

    # 在蒙版上画
    win.opts.target = "mask"
    win.set_tool("brush")
    st = win.view._start_paint(QPointF(cx, cy), Qt.NoModifier)
    assert st, "蒙版绘制没能开始"
    win.view._stroke.extend((cx + 20, cy + 10))
    win.view._stroke.end()
    win.view._stroke = None
    win.after_pixel_edit("画蒙版")
    print("  在蒙版上绘制 OK")
    win.opts.target = "pixel"
    win.set_tool("move")
    win.deselect()

    tmp = tempfile.mkdtemp()
    p = os.path.join(tmp, "ui.cwproj")
    win.doc.path = p
    win.save_project()
    assert os.path.getsize(p) > 0
    print("  保存工程 OK (%d bytes)" % os.path.getsize(p))

    # ---------- 调整层 ----------
    from src.core.adjust import ADJUST_ORDER

    win.deselect()
    win.new_document(800, 600)
    win._do_render()
    base = Layer("底", "image", _img(800, 600, (120, 120, 120)))
    base.tx, base.ty = 400.0, 300.0
    win.doc.layers.append(base)
    win.select_layer(base.id)
    win.commit("添加底图")
    win.panel.rebuild()
    win._do_render()

    win.add_adjustment_layer("invert")
    adj = win.selected_layer()
    assert adj is not None and adj.is_adjustment
    win._do_render()
    assert tuple(win._last_arr[300, 400, :3]) == (135, 135, 135), \
        win._last_arr[300, 400]
    print("  新建调整层 OK（120 反相 -> 135）")

    # 参数面板要能构建出控件
    win.inspector.refresh()
    assert win.inspector.adjust.isVisible()
    print("  调整面板显示 OK")

    # 改成另一种调整并拖动参数
    win.add_adjustment_layer("levels")
    adj = win.selected_layer()
    win.inspector.refresh()
    # 135 灰再过一道色阶，输入黑场抬到 200 应该被压到 0
    adj.adjust["params"]["in_black"] = 200
    win.request_render()
    win._do_render()
    assert int(win._last_arr[300, 400, 0]) == 0, win._last_arr[300, 400]
    print("  色阶参数改动生效 OK")

    # 撤销要能回到调整前的画面
    adj.adjust["params"]["in_black"] = 0
    win._do_render()
    snapshot = win._last_arr.copy()
    win.commit("改参数前")
    adj.adjust["params"]["in_black"] = 200
    win._do_render()
    assert not np.array_equal(win._last_arr, snapshot)
    win.commit("改参数后")
    win.undo()
    win._do_render()
    assert np.array_equal(win._last_arr, snapshot), "撤销后画面没还原"
    win.redo()
    win._do_render()
    assert not np.array_equal(win._last_arr, snapshot)
    win.undo()
    win._do_render()
    print("  调整参数的撤销 / 重做 OK")

    # 剪贴蒙版
    top = win.doc.find(adj.id)
    assert top is not None
    win.select_layer(top.id)
    win.toggle_clip()
    assert win.doc.find(top.id).clipped is True
    win._do_render()
    win.toggle_clip()
    assert win.doc.find(top.id).clipped is False
    print("  剪贴蒙版切换 OK")

    # 调整层蒙版
    win.select_layer(top.id)
    win.add_mask("black")
    assert win.doc.find(top.id).mask is not None
    win._do_render()
    win.delete_mask()
    print("  调整层蒙版 OK")

    # 每种调整都要能建出来、渲染、并生成面板控件
    for key in ADJUST_ORDER:
        win.add_adjustment_layer(key)
        cur = win.selected_layer()
        assert cur is not None and cur.is_adjustment and \
            cur.adjustment_key() == key, key
        win.inspector.refresh()
        win._do_render()
        assert win._last_arr.shape == (600, 800, 4), key
    print("  %d 种调整层全部可创建 / 渲染 / 生成控件" % len(ADJUST_ORDER))

    # ---------- 滤镜 ----------
    from src.core.filters import FILTER_ORDER, apply_filter_array
    from src.ui.filter_dialog import FilterDialog

    win.new_document(400, 300)
    win._do_render()
    # 用有纹理的图，纯色图模糊前后一样，测不出效果
    tex = np.zeros((300, 400, 4), np.uint8)
    ys, xs = np.mgrid[0:300, 0:400]
    tex[..., 0] = (xs % 32) * 8
    tex[..., 1] = (ys % 32) * 8
    tex[..., 2] = 128
    tex[..., 3] = 255
    fl = Layer("滤镜目标", "image", tex)
    fl.tx, fl.ty = 200.0, 150.0
    win.doc.layers.append(fl)
    win.select_layer(fl.id)
    win.commit("添加滤镜图层")
    win.panel.rebuild()
    win._do_render()

    for key in FILTER_ORDER:
        dlg = FilterDialog(win, key)
        dlg.preview.setChecked(False)
        out = apply_filter_array(fl.image, key, dlg.values(), None,
                                 dlg.strength())
        assert out.shape == fl.image.shape, key
        dlg.close()
    print("  %d 种滤镜对话框可构建，参数可执行" % len(FILTER_ORDER))

    # 走一遍主窗口的交互式应用（保留原图 + 取消还原）
    win._filter_ctx = {"layer": fl, "orig": fl.image.copy(), "sel": None,
                       "key": "gaussian"}
    orig = win._filter_ctx["orig"].copy()
    win._filter_preview({"radius": 10.0}, 1.0)
    assert not np.array_equal(fl.image, orig), "预览没有生效"
    fl.image = orig
    win._do_render()
    print("  滤镜实时预览 OK")

    win._filter_ctx = {"layer": fl, "orig": orig, "sel": None, "key": "gaussian"}
    fl.image = apply_filter_array(orig, "gaussian", {"radius": 10.0}, None, 1.0)
    win.after_pixel_edit("滤镜")
    assert not np.array_equal(win.doc.find(fl.id).image, orig)
    win.undo()
    win._do_render()
    assert np.array_equal(win.doc.find(fl.id).image, orig), "滤镜撤销没还原"
    print("  滤镜的撤销 OK")
    win._filter_ctx = None

    # 大半径中间值 / 蒙尘与划痕：半径滑块要能拉到 100，面板改参数能真的改动像素
    dlg = FilterDialog(win, "median")
    radii = dlg.findChildren(QSlider)
    assert radii and radii[0].maximum() == 100, [s.maximum() for s in radii]
    dlg.deleteLater()

    lay = win.doc.find(fl.id)          # 上面撤销过，旧引用可能已经不在文档里
    dusty = np.full((120, 120, 4), 200, np.uint8)
    dusty[..., 3] = 255
    dusty[58:63, 20:100, :3] = 20      # 一道划痕
    lay.image = dusty.copy()
    win._filter_ctx = {"layer": lay, "orig": dusty.copy(), "sel": None,
                       "key": "dust"}
    win._filter_preview({"radius": 6, "threshold": 40}, 1.0)
    assert int(lay.image[60, 60, 0]) == 200, "划痕没被消掉: %d" % lay.image[60, 60, 0]
    assert int(lay.image[10, 60, 0]) == 200, "划痕外不该被动"
    lay.image = dusty.copy()
    win._filter_ctx = None
    win._do_render()
    print("  蒙尘与划痕：半径可到 100，改参数消掉划痕且不碰别处")

    win.trim_to_content()
    win._do_render()
    print("  按内容裁剪 OK -> %dx%d" % (win.doc.width, win.doc.height))

    # ---------- 文字工具 ----------
    from src.core.text import ensure_fonts

    ensure_fonts()
    win.new_document(800, 600)
    win._do_render()
    win.opts.font_size = 72
    win.set_tool("text")
    tl = win.create_text_layer((400.0, 300.0), content="Compositor")
    assert tl is not None and tl.is_text
    win._do_render()
    print("  文字工具创建图层 OK，位图 %dx%d"
          % (tl.image.shape[1], tl.image.shape[0]))

    # 属性面板要显示「文字」那一组
    win.inspector.refresh()
    assert win.inspector.g_text.isVisible()
    assert win.inspector.txt_content.toPlainText() == "Compositor"
    print("  属性面板文字区 OK")

    # 改参数要重新栅格化
    w0 = tl.image.shape[1]
    win.set_text_param("size", 140.0)
    win.set_text_param("content", "Compositor 文字")
    win.set_text_param("bold", True)
    win.set_text_param("align", "left")
    win._do_render()
    cur = win.doc.find(tl.id)
    assert cur.image.shape[1] > w0, "改字号后位图该变宽"
    assert cur.text["bold"] is True and cur.text["align"] == "left"
    print("  改文字参数（字号 / 内容 / 粗体 / 对齐）OK")

    # 文字图层的像素由参数生成，直接改会被冲掉 —— 必须挡住
    win.set_tool("brush")
    win.select_layer(cur.id)
    assert not win.view._start_paint(QPointF(400.0, 300.0), Qt.NoModifier), \
        "文字图层不该允许直接绘制"
    win.fill_with_fg(None, use_fg=True)
    assert win.doc.find(cur.id).text is not None, "文字参数不该被填充破坏"
    print("  文字图层拒绝直接改像素 OK")

    # ---------- 文字增强：段落 / 逐字 / 画布内编辑（第十五批） ----------
    from src.ui.text_dialog import CharAdjustDialog

    cur = win.doc.find(cur.id)
    win.select_layer(cur.id)
    win.inspector.refresh()
    ins = win.inspector
    assert set(ins.para_spins) == {"indent_left", "indent_right",
                                   "indent_first", "space_before",
                                   "space_after"}, list(ins.para_spins)
    w_before = cur.image.shape[1]
    ins.para_spins["indent_left"].setValue(40)
    ins.para_spins["indent_right"].setValue(20)
    assert cur.text["para"]["indent_left"] == 40.0, cur.text["para"]
    assert cur.text["para"]["indent_right"] == 20.0
    assert cur.image.shape[1] > w_before, "加了缩进后位图该变宽"
    align_items = [ins.txt_align.itemText(i)
                   for i in range(ins.txt_align.count())]
    assert align_items[-1] == "两端对齐", align_items
    ins.txt_align.setCurrentIndex(len(align_items) - 1)
    assert cur.text["align"] == "justify"
    win.inspector.refresh()          # 刷新后 para 要能读回控件
    assert ins.para_spins["indent_left"].value() == 40
    ins.para_spins["indent_left"].setValue(0)
    ins.para_spins["indent_right"].setValue(0)
    ins.txt_align.setCurrentIndex(0)
    print("  段落控件：缩进写回参数并重栅格化，对齐多了「两端对齐」OK")

    # Ctrl+T：在画布上直接编辑
    assert win.edit_text_on_canvas()
    ed = win.view._text_edit
    assert ed is not None and win.view.edit_text_layer() is cur
    ed.setPlainText("画布内编辑")
    app.processEvents()
    assert cur.text["content"] == "画布内编辑", cur.text["content"]
    ed.setPlainText("画布内编辑\n第二行")
    app.processEvents()
    assert cur.text["content"].count("\n") == 1
    assert cur.image.shape[0] > 1
    geo = ed.geometry()
    win.view.set_zoom(win.view.zoom * 2)
    app.processEvents()
    assert ed.geometry() != geo, "缩放后输入框该跟着重定位"
    win.view.end_text_edit()
    assert win.view._text_edit is None and win.view.edit_text_layer() is None
    print("  在画布上编辑文字：打字实时改参数 / 缩放跟随 / Esc 关闭 OK")

    # 编辑期间换了选中图层 -> 输入框要自动收起（别把字写进别的图层）
    other = win.doc.layers[0]
    assert win.edit_text_on_canvas()
    win.select_layer(other.id)
    assert win.view.edit_text_layer() is None, "换图层后输入框该收起"
    win.select_layer(cur.id)
    print("  编辑期间换图层：输入框自动收起 OK")

    # 逐字调整对话框
    dlg = CharAdjustDialog(win, win)
    assert dlg.char_list.count() == len(cur.text["content"]), \
        (dlg.char_list.count(), repr(cur.text["content"]))
    assert dlg.select_index(1)
    dlg.set_field("dx", 60)
    assert cur.text["chars"].get("1", {}).get("dx") == 60.0, cur.text["chars"]
    dlg.set_field("dy", -30)
    dlg.set_field("scale", 150)
    got = cur.text["chars"]["1"]
    assert got["dy"] == -30.0 and abs(got["scale"] - 1.5) < 1e-6, got
    dlg.reset_current()
    assert "1" not in cur.text["chars"], "复位该字应该清掉这一项"
    dlg.set_field("dx", 55)
    assert cur.text["chars"], "刚设的值应该写回图层"
    dlg.clear_all()
    assert cur.text["chars"] == {}
    dlg.select_index(2)
    dlg.set_field("dx", 33)
    assert cur.text["chars"], "第 2 个字也该能设"
    dlg.reject()
    assert cur.text["chars"] == {}, "取消要把整份还原"
    dlg.deleteLater()
    print("  逐字调整对话框：选字 / 字距 / 基线 / 缩放 / 复位 / 清除 / 取消还原 OK")

    # 栅格化之后就是普通位图图层了
    win.rasterize_text()
    rt = win.doc.find(cur.id)
    assert not rt.is_text and rt.text is None
    win.inspector.refresh()
    assert not win.inspector.g_text.isVisible()
    print("  栅格化 OK -> 普通位图图层")
    win.set_tool("move")

    # ---------- PSD 导入 ----------
    from src.core.psd_import import is_available, load_psd

    if not is_available():
        print("  PSD 导入：未装 psd-tools，跳过")
    else:
        from tools.psdfixture import SECT_BOUND, SECT_CLOSED, write_psd

        psd_path = os.path.join(tmp, "ui.psd")
        write_psd(psd_path, 300, 200, [
            dict(name="GH", left=0, top=0, right=300, bottom=200,
                 rgb=(0, 0, 0), section=SECT_BOUND),
            dict(name="inner", left=20, top=20, right=120, bottom=120,
                 rgb=(0, 200, 0)),
            dict(name="G", left=0, top=0, right=0, bottom=0, rgb=(0, 0, 0),
                 section=SECT_CLOSED, section_blend=b"pass"),
            dict(name="top", left=100, top=60, right=280, bottom=180,
                 rgb=(30, 30, 220), opacity=200),
        ])
        doc = load_psd(psd_path)
        assert len(doc.layers) == 2, [l.name for l in doc.layers]
        assert doc.layers[0].is_group and len(doc.layers[0].children) == 1
        win.set_document(doc, reset_history=True)
        win._do_render()
        assert win._last_arr.shape == (200, 300, 4), win._last_arr.shape
        win.panel.rebuild()
        win.inspector.refresh()
        print("  PSD 导入 OK -> %d 个顶层图层（含 1 个组）" % len(doc.layers))

        # ---------- PSD 文字层 + 图层样式：走完整 UI 链路 ----------
        from src.core.text import ensure_fonts
        from PySide6.QtGui import QFontDatabase
        ensure_fonts()
        fams = list(QFontDatabase.families())
        assert fams, "离屏也该至少有 ensure_fonts 兜底的字体"
        fam = fams[0]
        from tools.psdfixture import demo_effects

        rich_path = os.path.join(tmp, "rich.psd")
        write_psd(rich_path, 300, 200, [
            dict(name="Txt", left=30, top=30, right=230, bottom=100,
                 rgb=(255, 0, 0),
                 text=dict(content="PSD", size=48.0, color=(20, 60, 200),
                           family=fam, underline=True, justification=2)),
            dict(name="Fx", left=40, top=120, right=180, bottom=180,
                 rgb=(210, 210, 210),
                 effects=list(demo_effects("all").values())),
        ])
        rich = load_psd(rich_path)
        win.set_document(rich, reset_history=True)
        win._do_render()
        win.panel.rebuild()
        win.inspector.refresh()
        assert win._last_arr.shape == (200, 300, 4), win._last_arr.shape

        txt = rich.layers[0]
        assert txt.is_text, "文字层没还原成可编辑文字（%s）" % txt.kind
        fxl = rich.layers[1]
        assert fxl.effects and fxl.has_effects, "图层样式没接上"
        assert fxl.has_effects, "has_effects 应为真"
        # 画布上要有实际像素（不是空白）
        assert int(win._last_arr[..., 3].max()) > 0, "合成结果全透明"
        print("  PSD 文字层 -> 界面：%d 个文字层还原、%d 个图层带样式"
              % (1, 1))

        # 选中文字层时属性面板要显示文字控件，并且能改字
        win.select_layer(txt.id)
        win.inspector.refresh()
        assert win.inspector.g_text.isVisible(), "文字层属性面板没显示"
        before = win._last_arr.copy()
        win.set_text_param("content", "ABC")
        win._do_render()
        assert win._last_arr.shape == before.shape
        assert txt.text["content"] == "ABC", txt.text["content"]
        assert txt.image is not None and txt.image.shape[2] == 4
        print("  PSD 文字层：属性面板可改字，画面实时更新（非破坏性）")

        # 选中带样式的图层：图层样式入口可用，effects 非空
        win.select_layer(fxl.id)
        win.inspector.refresh()
        on = [k for k, v in fxl.effects.items() if v.get("enabled")]
        # 9 种效果 + 被 use_global 的效果带出来的 global_light
        assert len(on) == 10, "应有 9 种效果 + 全局光，实际 %s" % on
        assert "global_light" in on, "global_light 应随被引用的效果一起导入"
        print("  PSD 图层样式：9 种效果 + 全局光挂上并参与界面刷新")

        # 撤销：改字要能撤销回 PSD 里的内容。
        # 两个坑一起踩：
        # ① `set_text_param` 走 `schedule_commit`，撤销点**延迟 500ms** 才落 ——
        #    不先 `_delayed_commit()` 冲掉它，undo 撤的是更早的状态
        # ② **撤销/重做会整棵克隆图层树**（§4.3 第 7 条），旧引用指向的是
        #    已被丢弃的副本，必须按 id 重新取
        win._delayed_commit()
        win.undo()
        assert win.doc.find(txt.id).text["content"] == "PSD", \
            win.doc.find(txt.id).text["content"]
        win.redo()
        assert win.doc.find(txt.id).text["content"] == "ABC", \
            win.doc.find(txt.id).text["content"]
        print("  PSD 文字层：改字可撤销 / 重做")

        # 工程往返：文字参数与样式都要保住
        p = os.path.join(tmp, "rich.cwproj")
        from src.core.project_io import load_project, save_project
        save_project(rich, p)
        back = load_project(p)
        assert back.layers[0].is_text, back.layers[0].kind
        assert back.layers[0].text["content"] == "ABC"
        assert back.layers[1].effects["drop_shadow"]["enabled"] is True
        win.set_document(back, reset_history=True)
        win._do_render()
        print("  PSD 文字 + 样式：工程往返后内容与样式都保住")

    # ---------- 通道面板（第十七批） ----------
    from PySide6.QtCore import Qt
    from src.core import channels as _ch
    from src.ui.channels_panel import ROLE_ID as _CH_ROLE_ID

    win.channels.rebuild()
    assert win.channels.list.count() == 4, win.channels.list.count()
    print("  通道面板：R / G / B / Alpha 四行就位")

    # 通道可见性只影响**显示**：_last_arr 始终是原始合成结果
    win._do_render()
    before = win._last_arr.copy()
    win.channels.list.item(0).setCheckState(Qt.Unchecked)   # 关红
    app.processEvents()
    win._do_render()
    disp = win.view._buf
    assert disp[..., 0].min() == 255, "关掉的红通道应显示为纸白"
    assert np.abs(win._last_arr.astype(np.int16)
                  - before.astype(np.int16)).max() == 0, \
        "_last_arr 不能被通道显示处理改掉（局部重渲染的缓存就是它）"
    print("  通道开关：关红显示为纸白，且不污染 _last_arr 缓存")
    win.channels.list.item(0).setCheckState(Qt.Checked)
    win._do_render()

    # 关 Alpha -> 画面全透明（看到棋盘格）
    win.channels.list.item(3).setCheckState(Qt.Unchecked)
    app.processEvents()
    win._do_render()
    assert win.view._buf[..., 3].max() == 0
    win.channels.list.item(3).setCheckState(Qt.Checked)
    win._do_render()
    assert win.view._buf[..., 3].max() > 0
    print("  通道开关：关 Alpha 画面全透明，再打开恢复")

    # 新建通道 -> 改遮罩 -> 影响显示。
    # 断言用"与加通道之前对比"而不是写死 255：这份文档来自 PSD 导入，
    # 局部本来就可能是透明的，写死数值会测到一个跟通道无关的东西。
    win._do_render()
    base = win.view._buf.copy()
    win.save_channel("__new__")
    app.processEvents()
    doc = win.doc
    assert len(doc.channels) == 1, [c.name for c in doc.channels]
    c = doc.channels[0]
    assert c.name == "Alpha 1", c.name
    m = np.zeros((doc.height, doc.width), np.uint8)
    m[:, :doc.width // 2] = 255
    doc.detach_channel(c)
    c.mask = m
    win.channels.rebuild()
    win._do_render()
    buf = win.view._buf
    assert np.array_equal(buf[:, :doc.width // 2, 3],
                          base[:, :doc.width // 2, 3]), \
        "遮罩内不该变透明"
    assert buf[:, doc.width - 10:, 3].max() == 0, "遮罩外该全透明"
    print("  附加通道：遮罩左半原样 / 右半全透明")

    # 载入选区：通道 -> 画布选区
    win.load_channel_selection(c.id)
    app.processEvents()
    assert doc.selection is not None, "载入选区后应有选区"
    bx = doc.selection.bbox()
    assert bx is not None and bx[2] - bx[0] == doc.width // 2, bx
    print("  载入选区：通道内容变成画布选区（宽 %d）"
          % (bx[2] - bx[0]))

    # 并入合并的 Alpha：只改显示，不动图层
    n_layers = len(doc.layers)
    win.merge_channel(c.id, "replace")
    app.processEvents()
    win._do_render()
    assert doc.composite_alpha is not None
    assert len(doc.layers) == n_layers, "并入合并 alpha 不该动图层"
    print("  并入合并的 Alpha：只改显示，%d 个图层原样不动" % n_layers)

    # 重命名：走面板那条真实路径（它会自己落撤销点）
    doc = win.doc
    c = _ch.find_channel(doc, c.id)
    assert c is not None, "通道在撤销栈换树之后应该还在"
    row = None
    for i in range(win.channels.list.count()):
        if win.channels.list.item(i).data(_CH_ROLE_ID) == c.id:
            row = win.channels.list.item(i)
            break
    assert row is not None, "面板里应该能找到这个通道"
    win.channels._on_rename(row, "打孔")
    app.processEvents()
    assert doc.channels[0].name == "打孔", doc.channels[0].name
    print("  通道重命名：Alpha 1 -> %s" % doc.channels[0].name)

    # 删除通道（可撤销）
    win.delete_channel(c.id)
    app.processEvents()
    assert len(doc.channels) == 0, [x.name for x in doc.channels]
    win.undo()
    app.processEvents()
    assert len(win.doc.channels) == 1, "删掉的通道应该能撤销回来"
    print("  删除通道：可撤销回来")

    # 工程往返
    win._delayed_commit()
    p = os.path.join(tmp, "ch.cwproj")
    from src.core.project_io import load_project, save_project
    save_project(win.doc, p)
    back = load_project(p)
    assert len(back.channels) == 1, [x.name for x in back.channels]
    assert back.channels[0].name == "打孔", back.channels[0].name
    win.set_document(back, reset_history=True)
    win._do_render()
    b2 = win.view._buf
    assert b2[:, back.width - 10:, 3].max() == 0,         "读回来的通道遮罩没起作用（遮罩外那半应该全透明）"
    print("  工程往返：通道名 / 遮罩 / 合并 Alpha 保住，画面一致")

    # 停靠区显隐
    win.toggle_channels_dock()
    assert not win.channels_dock.isVisible()
    win.toggle_channels_dock()
    assert win.channels_dock.isVisible()
    print("  F2 / 视图 → 通道面板：可显隐")

    # ---- 新增的 4 种调整层：面板要能建出来、能渲染 ----
    for key in ("channel_mixer", "gradient_map", "photo_filter",
                "selective_color"):
        win.add_adjustment_layer(key)
        app.processEvents()
        lay = win.selected_layer()
        assert lay is not None and lay.is_adjustment, key
        win.inspector.refresh()
        win._do_render()
        assert win._last_arr is not None
        win.delete_layer()
        app.processEvents()
    print("  4 种新调整层：建层 / 面板 / 渲染 OK")

    # 通道混合器的 keyed 控件（下拉框 + 一组滑块）要真的写回参数
    win.add_adjustment_layer("channel_mixer")
    app.processEvents()
    adj = win.selected_layer()
    win.inspector.refresh()
    combo = win.inspector.adjust.findChildren(QComboBox)[0]
    combo.setCurrentText("绿")
    sliders = win.inspector.adjust.findChildren(QSlider)
    sliders[0].setValue(50)          # 绿输出通道的「红」源权重
    app.processEvents()
    assert abs(adj.adjust["params"]["mix"]["绿"]["red"] - 50) < 1e-6, \
        adj.adjust["params"]["mix"]["绿"]
    win.delete_layer()
    app.processEvents()
    print("  通道混合器分组控件：写回 mix[绿][red] OK")

    # ---- 调整层增强：色相范围 / 黑白预设 / 渐变条控件 ----
    from src.core.adjust import BW_PRESETS
    from src.ui.adjust_panel import GradientEditor

    win.add_adjustment_layer("hue_sat")
    app.processEvents()
    hue = win.selected_layer()
    win.inspector.refresh()
    combo = win.inspector.adjust.findChildren(QComboBox)[0]
    assert combo.itemText(0) == "全图", combo.itemText(0)
    combo.setCurrentText("红色")
    app.processEvents()
    assert hue.adjust["params"]["range"] == "红色", hue.adjust["params"]
    win.delete_layer()
    app.processEvents()
    print("  色相/饱和度：色彩范围下拉框写回参数 OK")

    # 黑白预设：选一个预设把六个滑块一起写进去；手动拖滑块打回「自定义」
    win.add_adjustment_layer("black_white")
    app.processEvents()
    bw = win.selected_layer()
    win.inspector.refresh()
    preset_combo = win.inspector.adjust.findChildren(QComboBox)[0]
    assert preset_combo.itemText(0) == "自定义", preset_combo.itemText(0)
    preset_combo.setCurrentText("蓝色滤镜")
    app.processEvents()
    app.processEvents()          # 预设会延迟重建一次面板
    want = list(BW_PRESETS["蓝色滤镜"])
    got = [bw.adjust["params"][k] for k in
           ("reds", "yellows", "greens", "cyans", "blues", "magentas")]
    assert got == want, got
    assert bw.adjust["params"]["preset"] == "蓝色滤镜"
    win.inspector.adjust.findChildren(QSlider)[0].setValue(10)   # = 红色滑块
    app.processEvents()
    assert bw.adjust["params"]["reds"] == 10, bw.adjust["params"]["reds"]
    assert bw.adjust["params"]["preset"] == "自定义", \
        bw.adjust["params"]["preset"]
    assert win.inspector.adjust.findChildren(QComboBox)[0].currentText() == \
        "自定义", "手动改滑块后预设下拉框没打回「自定义」"
    win.delete_layer()
    app.processEvents()
    print("  黑白：预设写六个滑块 / 手动改滑块打回「自定义」OK")

    # 渐变映射：渐变条能改控制点并写回参数、参与渲染
    win.add_adjustment_layer("gradient_map")
    app.processEvents()
    gm = win.selected_layer()
    win.inspector.refresh()
    ed = win.inspector.adjust.findChildren(GradientEditor)[0]
    assert len(ed.stops()) == 3, "默认渐变应该是三个控制点"
    assert gm.adjust["params"].get("stops"), "面板没把控制点物化进参数"
    stops = [[0.0, 255, 0, 0], [0.5, 255, 255, 0], [1.0, 0, 0, 255]]
    ed.set_stops(stops)
    ed.gradientChanged.emit(ed.stops())
    app.processEvents()
    assert gm.adjust["params"]["stops"] == stops, gm.adjust["params"]["stops"]
    win._do_render()
    assert win._last_arr is not None
    win.delete_layer()
    app.processEvents()
    print("  渐变映射：渐变条控制点写回参数并参与渲染 OK")

    # ---- 图层样式 ----
    from src.core.effects import default_effects
    from src.ui.style_dialog import StyleDialog
    from src.core.document import make_image_layer
    doc = Document(400, 300, "样式测试")
    img = np.zeros((120, 120, 4), np.uint8)
    img[20:100, 20:100, :3] = (60, 180, 240)
    img[20:100, 20:100, 3] = 255
    sl = make_image_layer("方块", img, 400, 300)
    doc.layers.append(sl)
    win.set_document(doc, reset_history=True)
    win.select_layer(sl.id)
    app.processEvents()

    dlg = StyleDialog(sl, win, None)
    dlg.groups["stroke"][0].setChecked(True)          # 勾上描边
    dlg.groups["drop_shadow"][0].setChecked(True)
    app.processEvents()
    assert sl.effects["stroke"]["enabled"] is True
    win._do_render()
    assert win._last_arr.shape == (300, 400, 4)
    dlg.accept()
    app.processEvents()
    assert sl.effects is not None and sl.effects["stroke"]["enabled"]
    win.inspector.refresh()
    assert "描边" in win.inspector.lab_style.text(), \
        win.inspector.lab_style.text()
    print("  图层样式对话框：勾选 / 实时预览 / 落定 / 属性面板摘要 OK")

    # 取消要还原
    before = dict((k, dict(v)) for k, v in (sl.effects or {}).items())
    dlg2 = StyleDialog(sl, win, None)
    dlg2.groups["outer_glow"][0].setChecked(True)
    app.processEvents()
    dlg2.reject()
    app.processEvents()
    assert sl.effects["outer_glow"]["enabled"] is False, "取消没还原"
    assert sl.effects["stroke"]["enabled"] == before["stroke"]["enabled"]
    print("  图层样式取消：整份参数还原 OK")

    # 第十批：10 种效果 + 全局光 + 复制 / 粘贴 / 默认值
    from src.core import effects as EF

    keep = EF._SAVED_DEFAULT["v"]
    EF._SAVED_DEFAULT["v"] = None
    try:
        dlg3 = StyleDialog(sl, win, None)
        assert len(dlg3.groups) == len(EF.EFFECT_ORDER), "每种效果都要有面板"
        dlg3.groups["bevel_emboss"][0].setChecked(True)
        dlg3.groups["gradient_overlay"][0].setChecked(True)
        app.processEvents()
        win._do_render()
        mid = win._last_arr.copy()
        # 全局光：勾上之后跟着改角度，勾了「使用全局光」的斜面一起变
        dlg3._toggle_global(True)
        dlg3._set_global("angle", 225.0)
        win._do_render()
        assert np.abs(win._last_arr.astype(np.int16)
                      - mid.astype(np.int16)).max() > 0, "全局光角度没联动"
        # 关掉的效果有一部分变灰、groupbox 本身还能点（否则勾不回来）
        dlg3.groups["gradient_overlay"][0].setChecked(False)
        app.processEvents()
        assert dlg3.groups["gradient_overlay"][0].isEnabled()
        assert not dlg3.bodies["gradient_overlay"].isEnabled()
        win._do_render()
        assert np.abs(win._last_arr.astype(np.int16)
                      - mid.astype(np.int16)).max() > 0, "关掉效果画面没变"
        dlg3.reject()
        app.processEvents()
        assert sl.effects["bevel_emboss"]["enabled"] is False, "取消没还原"
        print("  图层样式：10 种面板 / 全局光联动 / 取消还原 OK")

        # 复制 / 粘贴：整份参数要能原样搬走（含新加的效果）
        sl.effects = EF.default_effects()
        sl.effects["bevel_emboss"]["enabled"] = True
        sl.effects["gradient_overlay"]["enabled"] = True
        win.select_layer(sl.id)
        assert win.copy_layer_style() is True, "应该能复制"
        sl.effects = None                      # 先清掉，确认是粘贴来的
        assert win.paste_layer_style() is True, "应该能粘贴"
        app.processEvents()
        win._do_render()
        win.inspector.refresh()
        assert sl.effects["gradient_overlay"]["enabled"] is True
        txt = win.inspector.lab_style.text()
        assert "斜面浮雕" in txt and "渐变叠加" in txt, txt
        print("  复制 / 粘贴图层样式：整份参数迁移 OK")

        assert win.save_style_default() is True
        fresh = EF.default_effects()
        assert fresh["gradient_overlay"]["enabled"] is True, "默认值没生效"
        assert win.reset_style_default() is True
        assert not EF.has_style_default()
        win._do_render()
        app.processEvents()
        print("  存为默认值 / 复位默认值 OK")
    finally:
        EF._SAVED_DEFAULT["v"] = keep
        if keep is not None:
            EF.save_style_default(keep)

    # ---- 浮动选区（拖动选中的像素）----
    from src.core.selection import Selection
    doc = Document(400, 300, "浮动测试")
    base = np.zeros((300, 400, 4), np.uint8)
    base[..., 0] = 255
    base[..., 3] = 255
    bl = make_image_layer("底", base, 400, 300)
    doc.layers.append(bl)
    doc.selection = Selection.rect(400, 300, 100, 100, 200, 200)
    win.set_document(doc, reset_history=True)
    win.select_layer(bl.id)
    app.processEvents()

    assert win.can_lift_selection(), "应该可以揭"
    assert win.lift_selection() is True
    app.processEvents()
    assert doc.float_layer is not None
    a0 = int(win._last_arr[150, 150, 3])
    doc.float_layer.tx += 150.0
    win._do_render()
    assert int(win._last_arr[150, 150, 3]) == 0, "移走后原处应该空"
    assert int(win._last_arr[150, 300, 3]) == 255, "新位置应该有内容"
    assert a0 == 255
    # 换工具自动落定
    win.set_tool("brush")
    app.processEvents()
    assert doc.float_layer is None, "换工具没落定"
    assert int(win._last_arr[150, 300, 3]) == 255, "落定后位置不对"
    print("  浮动选区：揭 / 拖动 / 换工具自动落定 OK")

    # Esc 丢弃（换一块还是有内容的地方，前一次已经把 100..200 挖空了）
    doc.selection = Selection.rect(400, 300, 20, 20, 80, 80)
    win.select_layer(bl.id)
    assert win.lift_selection() is True
    win.discard_float()
    app.processEvents()
    assert doc.float_layer is None
    win._do_render()
    assert int(win._last_arr[50, 50, 3]) == 0, "丢弃后原处仍是空的"
    print("  浮动选区 Esc 丢弃 OK")

    # ---- 浮动层自己的变换：手柄挂到浮动层上，Alt = 以中心缩放 ----
    from PySide6.QtCore import QPointF, Qt
    doc = Document(400, 300, "浮动缩放")
    base = np.zeros((300, 400, 4), np.uint8)
    base[..., :3] = (200, 60, 60)
    base[..., 3] = 255
    bl = make_image_layer("底", base, 400, 300)
    doc.layers.append(bl)
    doc.selection = Selection.rect(400, 300, 150, 100, 250, 200)
    win.set_document(doc, reset_history=True)
    win.set_tool("move")
    win.select_layer(bl.id)
    app.processEvents()
    assert win.lift_selection() is True
    fl = doc.float_layer
    win.view._update_handles()
    assert win.view._handle_layer() is fl, "有浮动层时手柄应该挂在它身上"
    print("  浮动层手柄：浮起来之后手柄直接挂在这块内容上")

    hw = fl.image.shape[1] / 2.0
    hh = fl.image.shape[0] / 2.0
    win.view._drag = {"role": "se", "start": QPointF(fl.tx + hw, fl.ty + hh),
                      "layer": fl, "sx": fl.sx, "sy": fl.sy, "rot": fl.rot,
                      "tx": fl.tx, "ty": fl.ty, "a0": 0.0, "moved": True}
    win.view._apply_drag(QPointF(fl.tx + hw + 40, fl.ty + hh + 30),
                         Qt.NoModifier)
    win.view._drag = None
    assert fl.sx > 1.2 and fl.sy > 1.2, "手柄没把浮动层放大：%s" % (
        (fl.sx, fl.sy),)
    print("  浮动层缩放：拖 se 手柄 -> %.2f x %.2f" % (fl.sx, fl.sy))

    def drag_se(mod, sx0=1.0, sy0=0.6, dx=40.0, dy=10.0):
        fl.sx, fl.sy = sx0, sy0
        ox, oy = fl.tx, fl.ty
        win.view._drag = {"role": "se", "start": QPointF(ox + hw, oy + hh),
                          "layer": fl, "sx": sx0, "sy": sy0, "rot": fl.rot,
                          "tx": ox, "ty": oy, "a0": 0.0, "moved": True}
        win.view._apply_drag(QPointF(ox + hw + dx, oy + hh + dy), mod)
        win.view._drag = None
        return ox, oy, fl.sx / sx0, fl.sy / sy0

    # Alt 拖 = 以中心为基准（中心不动），两个方向仍按各自位移走
    cx, cy, rx, ry = drag_se(Qt.AltModifier)
    assert abs(fl.tx - cx) < 1e-6 and abs(fl.ty - cy) < 1e-6, \
        "Alt 缩放应该以中心为基准，中心却动了"
    assert abs(rx - ry) > 0.05, "只按 Alt 时两个方向应该各走各的：%s" % (
        (round(rx, 3), round(ry, 3)),)
    print("  Alt 缩放：中心不动（%.0f, %.0f），x%.2f / y%.2f"
          % (fl.tx, fl.ty, rx, ry))

    # Alt+Shift = 以中心 + 等比（两个方向同一个倍率）
    cx, cy, rx, ry = drag_se(Qt.AltModifier | Qt.ShiftModifier)
    assert abs(rx - ry) < 1e-6, "Alt+Shift 应该是等比：%s" % (
        (round(rx, 3), round(ry, 3)),)
    assert abs(fl.tx - cx) < 1e-6 and abs(fl.ty - cy) < 1e-6
    print("  Alt+Shift 缩放：等比 x%.2f / y%.2f，中心不动" % (rx, ry))

    before = win._last_arr.copy() if win._last_arr is not None else None
    win._do_render()
    before = win._last_arr.copy()
    assert win.stamp_float() is True
    assert doc.float_layer is None
    win._do_render()
    d = np.abs(win._last_arr.astype(np.int16) - before.astype(np.int16))
    assert d.max() <= 2, "落定前后画面差太多：%d" % d.max()
    assert win.view._handle_layer() is bl, "落定之后手柄该回到选中图层"
    print("  浮动层带变换落定：画面一致（最大差 %d），手柄回到选中图层" % d.max())

    # ---- 色彩范围 ----
    from src.ui.color_range_dialog import ColorRangeDialog
    img = np.zeros((120, 200, 3), np.uint8)
    img[:, :100] = (220, 30, 30)
    img[:, 100:] = (30, 60, 220)
    dlg = ColorRangeDialog(win, img)
    dlg.samples = [(220, 30, 30)]
    dlg.combo_preset.setCurrentIndex(0)
    m = dlg.mask_small()
    assert int(m[60, 50]) == 255 and int(m[60, 150]) == 0, \
        "取样颜色应该只命中左半边的红"
    dlg.combo_preset.setCurrentIndex(5)      # 蓝色
    m = dlg.mask_small()
    assert int(m[60, 150]) > 200 and int(m[60, 50]) == 0, \
        "切到蓝色预置应该命中右半边"
    got = []
    dlg.selectionReady.connect(lambda mm, mode: got.append((mm, mode)))
    dlg.accept()
    assert len(got) == 1, "确定时应该把遮罩交出去"
    assert got[0][0].shape == (120, 200), "结果应该是全分辨率的"
    print("  色彩范围：预置切换 + 吸管取样 + 确定返回 (%d x %d)"
          % got[0][0].shape[::-1])

    # 走主窗口那条路：把遮罩合并进文档选区
    dlg = ColorRangeDialog(win, img)
    dlg.samples = [(220, 30, 30)]
    dlg.combo_preset.setCurrentIndex(0)
    win.set_document(Document(200, 120, "色彩范围"), reset_history=True)
    img_lay = make_image_layer("图", np.concatenate(
        [img, np.full((120, 200, 1), 255, np.uint8)], axis=2), 200, 120)
    win.doc.layers = [img_lay]
    win.select_layer(img_lay.id)
    win._do_render()
    win._apply_color_range(dlg.result_mask(), "replace")
    box = win.doc.selection.bbox()
    assert box is not None and box[2] <= 100, "选区应该只覆盖左半边：%s" % (box,)
    print("  色彩范围：合并进文档选区 bbox %s" % (box,))

    # ---- 快速蒙版 ----
    from src.core import quick_mask
    from src.core.paint import Stroke
    doc = Document(300, 200, "QM")
    bl = make_image_layer("底", _img(300, 200, (60, 90, 150)), 300, 200)
    doc.layers.append(bl)
    doc.selection = Selection.rect(300, 200, 40, 40, 140, 140)
    win.set_document(doc, reset_history=True)
    win.select_layer(bl.id)
    app.processEvents()

    win.toggle_quick_mask()
    assert quick_mask.is_on(doc), "Q 没进快速蒙版"
    assert int(doc.quick_mask[90, 90]) == 255 and int(doc.quick_mask[0, 0]) == 0
    assert "快速蒙版" in win.windowTitle(), "标题栏应该标出模式"
    win.view.update_selection_overlay()
    print("  快速蒙版：进入，遮罩按选区初始化，标题栏标注模式")

    # 清空后涂一笔白 —— 画笔在快速蒙版模式下涂的是遮罩，不是图层像素
    win.fill_quick_mask(0)
    assert not doc.quick_mask.any()
    layer_before = bl.image.copy()
    st = Stroke(doc=doc, layer=None, tool="brush", size=40, hardness=1.0,
                opacity=1.0, flow=1.0, smoothing=0.0, target="quick",
                color=(255, 255, 255))
    assert st.begin((100.0, 100.0)), "快速蒙版上应该能落笔"
    st.end()
    assert int(doc.quick_mask[100, 100]) == 255, "涂白应该把它纳入选区"
    assert np.array_equal(bl.image, layer_before), "不该动到图层像素"
    print("  快速蒙版：画笔涂的是遮罩，图层像素没被碰")

    win.toggle_quick_mask()
    assert not quick_mask.is_on(doc)
    assert "快速蒙版" not in win.windowTitle()
    box = doc.selection.bbox()
    assert box is not None and 78 <= box[0] and box[2] <= 122, \
        "退出后的选区不对：%s" % (box,)
    print("  快速蒙版：退出 -> 选区 bbox %s" % (box,))

    # 快速蒙版下按 Delete = 整片取消；画选框 = 退出快速蒙版
    win.toggle_quick_mask()
    win.deselect()
    assert not doc.quick_mask.any(), "快速蒙版下取消 = 遮罩清零"
    win.toggle_quick_mask()
    assert doc.selection is None
    print("  快速蒙版：取消选择 / 退出后选区为空")

    # ---- 渲染优化：脏矩形 / 交互期代理 ----
    from src.core.effects import default_effects
    from src.core.render import render_document

    doc = Document(600, 400, "渲染优化")
    bg = np.zeros((400, 600, 4), np.uint8)
    bg[..., :3] = 120
    bg[..., 3] = 255
    bl = make_image_layer("底", bg, 600, 400)
    doc.layers.append(bl)
    tl = Layer("块", "image", _img(200, 150, (240, 60, 60)))
    tl.tx, tl.ty = 200.0, 200.0
    doc.layers.append(tl)
    win.set_document(doc, reset_history=True)
    win.select_layer(tl.id)
    win._do_render()
    app.processEvents()
    assert win._last_arr is not None and win._arr_valid

    # 1) 只重算脏区，结果必须和整幅重算逐位一致（残影 bug 的守门员）
    before = win._last_arr.copy()
    # 脏区必须是 "旧位置 ∪ 新位置" —— 只标记新位置会把旧位置的像素留在画面上
    old_r = win.view._layer_dirty_rect(tl)
    tl.tx += 120.0
    win.view._render_moved(old_r, tl)
    win._do_render()
    assert win._last_pass == "partial", \
        "应该走局部重渲染，实际走了 %s" % win._last_pass
    assert win._last_arr.shape == before.shape
    full = render_document(doc)
    d = np.abs(win._last_arr.astype(np.int16) - full.astype(np.int16))
    assert d.max() == 0, "脏区重渲染与整幅不一致，最大差 %d" % d.max()
    print("  脏矩形重渲染：与整幅重算逐位一致（无残影）")

    # 1b) 画笔描边也走脏区，结果同样要和整幅一致
    from src.core.paint import Stroke
    st = Stroke(doc=doc, layer=tl, tool="brush", size=40, hardness=1.0,
                opacity=1.0, flow=1.0, smoothing=0.0, target="pixel",
                color=(20, 200, 60))
    assert st.begin((tl.tx, tl.ty))
    st.extend((tl.tx + 40.0, tl.ty + 20.0))
    win.request_render(st.take_dirty(pad=2))
    st.end()
    win._do_render()
    assert win._last_pass == "partial", "描边应走脏区，实际 %s" % win._last_pass
    d = np.abs(win._last_arr.astype(np.int16)
               - render_document(win.doc).astype(np.int16))
    assert d.max() == 0, "描边后画面与整幅不一致，最大差 %d" % d.max()
    print("  画笔描边：只重算脏区，结果与整幅一致")

    # 2) 带图层样式也要一致（模糊要卷积上下文 -> 主窗口会自动外扩）
    eff = default_effects()
    eff["drop_shadow"]["enabled"] = True
    eff["drop_shadow"]["distance"] = 10.0
    eff["drop_shadow"]["size"] = 12.0
    tl.effects = eff
    win.request_render()
    win._do_render()
    before = win._last_arr.copy()
    old_r = win.view._layer_dirty_rect(tl)
    tl.ty += 60.0
    win.view._render_moved(old_r, tl)
    win._do_render()
    full = render_document(doc)
    d = np.abs(win._last_arr.astype(np.int16) - full.astype(np.int16))
    assert d.max() == 0, "带投影时脏区重渲染不一致，最大差 %d" % d.max()
    print("  脏矩形重渲染：带投影同样逐位一致")

    # 3) 交互期代理：渲出来的位图比画布小，退出交互后恢复全分辨率
    win._render_ms = 900.0                 # 假装这张画布渲得很慢
    win.set_interactive(True)
    win.request_render()
    win._do_render()
    app.processEvents()
    assert win._last_pass == "proxy", "应该走代理渲染，实际 %s" % win._last_pass
    assert not win._arr_valid, "渲过代理之后缓存必须标记为过期"
    pm = win.view.pix_item.pixmap()
    assert pm.width() < doc.width, "交互期应该出低分辨率代理图"
    assert win.view.scene.doc_rect.width() == doc.width, "场景矩形仍按文档尺寸"
    win.set_interactive(False)
    app.processEvents()
    win._do_render()
    assert win._arr_valid, "退出交互后要补一张全分辨率"
    assert win._last_pass == "full", "退出交互后应整幅重算"
    pm2 = win.view.pix_item.pixmap()
    assert pm2.width() == doc.width, "退出交互后位图应恢复 1:1"
    print("  交互期代理：%dx%d 代理图，松手后恢复 %dx%d 全分辨率"
          % (pm.width(), pm.height(), pm2.width(), pm2.height()))

    # 4) 撤销之后缓存作废（整棵图层树被换掉了，不能拿旧缓存打补丁）
    win.commit("变换")
    win._do_render()
    assert win._arr_valid
    win.undo()
    assert not win._arr_valid, "撤销后渲染缓存必须作废"
    app.processEvents()
    win._do_render()
    assert win._arr_valid and win._last_arr is not None
    d = np.abs(win._last_arr.astype(np.int16)
               - render_document(win.doc).astype(np.int16))
    assert d.max() == 0, "撤销后重渲染与整幅不一致"
    print("  撤销后渲染缓存作废，重新整幅重算 OK")

    # 5) 真的走一遍手柄拖动（交互期出代理，松手补全分辨率且不留残影）
    from PySide6.QtCore import QPointF, Qt as _Qt

    win._render_ms = 900.0
    view = win.view
    win.set_interactive(True)              # 等价于 mousePressEvent 里按下手柄
    view._drag = {"role": "move", "start": QPointF(200.0, 200.0), "layer": tl,
                  "tx": tl.tx, "ty": tl.ty, "moved": False}
    view._apply_drag(QPointF(230.0, 240.0), _Qt.NoModifier)
    win._do_render()
    app.processEvents()
    assert view.pix_item.pixmap().width() < doc.width, "拖动中应出代理图"
    for i in range(8):                       # 连续拖 8 步，检查残影不会累积
        view._apply_drag(QPointF(240.0 + 10 * i, 250.0 + 8 * i), _Qt.NoModifier)
        win._do_render()
    view._drag = None
    win.set_interactive(False)
    app.processEvents()
    win._do_render()
    assert view.pix_item.pixmap().width() == win.doc.width
    d = np.abs(win._last_arr.astype(np.int16)
               - render_document(win.doc).astype(np.int16))
    assert d.max() == 0, "拖动结束后画面与整幅重算不一致（残影）"
    print("  手柄拖动：交互期代理 + 松手补全分辨率，无残影 OK")

    # 6) 拖调整层滑块：快照快路径（跳过所有下方图层的 warp + 混合）
    from src.core.adjust import default_params
    from src.core.document import make_adjustment_layer

    doc = Document(600, 400, "调整层快路径")
    bg = np.zeros((400, 600, 4), np.uint8)
    bg[..., :3] = 120
    bg[..., 3] = 255
    doc.layers.append(make_image_layer("底", bg, 600, 400))
    blk = Layer("块", "image", _img(200, 150, (240, 60, 60)))
    blk.tx, blk.ty = 200.0, 200.0
    doc.layers.append(blk)
    adj = make_adjustment_layer("色阶", "levels", 600, 400)
    doc.layers.append(adj)
    win.set_document(doc, reset_history=True)
    win.select_layer(adj.id)
    win.request_render(adjust=adj)
    win._do_render()
    assert win._last_pass == "full", "第一次要整幅渲染，顺带捕获快照"
    assert win._adj_snap is not None and win._adj_snap["arr"] is not None

    # 快照必须等于"只渲它下面那些图层"的结果
    below = Document(600, 400, "below")
    below.layers = list(doc.layers[:2])
    d = np.abs(win._adj_snap["arr"].astype(np.int16)
               - render_document(below).astype(np.int16))
    assert d.max() <= 1, "快照不等于调整层下方的合成结果，差 %d" % d.max()

    adj.adjust["params"]["gamma"] = 1.8
    win.request_render(adjust=adj)
    win._do_render()
    assert win._last_pass == "adjust", \
        "改调整参数应走快照快路径，实际 %s" % win._last_pass
    d = np.abs(win._last_arr.astype(np.int16)
               - render_document(doc).astype(np.int16))
    assert d.mean() < 0.5, "快路径与整幅偏离过大 mean=%.3f" % d.mean()
    print("  调整层快路径：改参数只重算调整层本身（与整幅平均差 %.3f）"
          % d.mean())

    # 有快照时即使在交互期也不降级成代理 —— 快路径本身就是全分辨率
    win._render_ms = 900.0
    win.set_interactive(True)
    adj.adjust["params"]["gamma"] = 2.2
    win.request_render(adjust=adj)
    win._do_render()
    app.processEvents()
    assert win._last_pass == "adjust", "有快照时不该再降级成代理"
    assert win.view.pix_item.pixmap().width() == doc.width, \
        "快路径出的就是全分辨率位图"
    win.set_interactive(False)
    app.processEvents()

    # 别的改动必须让快照作废（否则会拿过期的下方结果打补丁）
    adj.adjust["params"] = default_params("levels")
    win.request_render()
    assert win._adj_snap is None, "不带 adjust 的改动要作废快照"
    win._do_render()
    assert win._last_pass == "full", "快照作废后应整幅重算"
    d = np.abs(win._last_arr.astype(np.int16)
               - render_document(doc).astype(np.int16))
    assert d.max() == 0, "整幅重算与参考不一致"
    print("  调整层快路径：非调整改动后快照作废，回到整幅精确渲染")

    # 7) 滤镜预览降采样：大图层先在缩略图上算，半径按同一比例缩放
    from src.core.filters import apply_filter_array, preview_filter_scale

    doc = Document(600, 400, "滤镜预览")
    fl = Layer("大图", "image", _img(2000, 1400, (90, 140, 200)))
    fl.tx, fl.ty = 300.0, 200.0
    doc.layers.append(fl)
    win.set_document(doc, reset_history=True)
    win.select_layer(fl.id)
    win._do_render()
    s, small = preview_filter_scale(fl.image, "gaussian")
    assert s < 1.0, "2000 px 宽的图层预览应该降采样"
    win._filter_ctx = {
        "layer": fl, "orig": fl.image.copy(), "sel": None, "key": "gaussian",
        "scale": s, "small": small, "sel_small": None,
    }
    win._filter_preview({"radius": 40.0}, 1.0)
    assert fl.image.shape == (1400, 2000, 4), "预览完尺寸要还原成原图大小"
    ref = apply_filter_array(win._filter_ctx["orig"], "gaussian",
                             {"radius": 40.0})
    d = np.abs(fl.image.astype(np.int16) - ref.astype(np.int16))
    assert d.mean() < 6.0, "预览与全分辨率结果偏离过大 mean=%.2f" % d.mean()
    win._filter_ctx = None
    win._do_render()
    print("  滤镜预览：2000x1400 降到 %dx%d 再算，与全分辨率平均差 %.2f"
          % (small.shape[1], small.shape[0], d.mean()))

    # 8) 大画布走分块渲染：结果一致，且 `_last_pass` 标记为 tiled
    from src.core.render import render_tiled

    doc = Document(2600, 2400, "分块")
    doc.layers.append(make_image_layer("底", _img(2600, 2400, (150, 150, 150)),
                                       2600, 2400))
    blk = Layer("块", "image", _img(900, 700, (240, 80, 80)))
    blk.tx, blk.ty = 1200.0, 1200.0
    blk.sx, blk.sy = 1.5, 1.2
    blk.rot = 20.0
    doc.layers.append(blk)
    win.set_document(doc, reset_history=True)
    win._do_render()
    assert win._last_pass == "tiled", \
        "%.1f MP 的画布应该走分块渲染，实际 %s" % (2600 * 2400 / 1e6, win._last_pass)
    d = np.abs(win._last_arr.astype(np.int16)
               - render_document(doc).astype(np.int16))
    assert d.max() <= 1, "分块渲染与整幅不一致，最大差 %d" % d.max()
    assert win.view.pix_item.pixmap().width() == doc.width, "分块出的仍是全分辨率"
    print("  分块渲染：%.1f MP 画布走 tiled，与整幅最大差 %d/255"
          % (2600 * 2400 / 1e6, d.max()))

    # 小画布不该分块（省下的内存抵不上瓦片开销）
    small_doc = Document(800, 600, "小画布")
    small_doc.layers.append(make_image_layer("底", _img(800, 600, (90, 90, 90)),
                                             800, 600))
    win.set_document(small_doc, reset_history=True)
    win._do_render()
    assert win._last_pass == "full", \
        "小画布不该分块，实际 %s" % win._last_pass
    print("  分块渲染：0.5 MP 画布仍走整块（分块只对大画布划算）")

    # Dissolve 的噪声按缓冲尺寸生成，分块会错位 —— 必须自动退回整块
    doc.layers[1].blend = "Dissolve"
    assert render_tiled(doc) is None, "有 Dissolve 时 render_tiled 必须返回 None"
    win.request_render()
    win._do_render()
    # 接受 "full" 或 "proxy"：两者都是整幅渲染，噪声不会跨瓦片错位。
    # proxy 只在上一帧耗时超过阈值（机器负载抖动）时启用，属于合法兜底，
    # 不能因为它随机出现就判失败 —— 真正要排除的是 "tiled"。
    assert win._last_pass in ("full", "proxy"), \
        "有 Dissolve 时必须整幅渲染（full/proxy），实际 %s" % win._last_pass
    doc.layers[1].blend = "Normal"
    print("  分块渲染：有 Dissolve 时自动退回整块渲染（full/proxy）")

    win.view.fit()
    win.view.zoom_actual()
    app.processEvents()
    print("  视图缩放 OK")

    # ---- 智能对象：转换 / 多实例 / 编辑内容 / 智能滤镜 / 栅格化 ----
    from src.core.smart import content_instances

    doc = Document(600, 400, "智能对象")
    bg = np.zeros((400, 600, 4), np.uint8)
    bg[..., :3] = 120
    bg[..., 3] = 255
    doc.layers.append(make_image_layer("底", bg, 600, 400))
    blk = Layer("块", "image", _img(200, 150, (240, 60, 60)))
    blk.tx, blk.ty = 200.0, 200.0
    doc.layers.append(blk)
    win.set_document(doc, reset_history=True)
    win.select_layer(blk.id)
    win._do_render()
    app.processEvents()

    plain = win._last_arr.copy()
    win.convert_selection_to_smart()
    app.processEvents()
    shell = win.selected_layer()
    assert shell.kind == "smart", "菜单转换之后图层没变成智能对象"
    win._do_render()
    d = np.abs(win._last_arr.astype(np.int16) - plain.astype(np.int16))
    assert d.max() <= 1, "转换后画面不该变（实际最大差 %d）" % d.max()
    assert shell.image is not None, "智能对象的派生栅格没生成"
    print("  转换为智能对象：画面不变，图层列表出现 %s" % shell.name)

    # 属性面板要认得它
    win.inspector.refresh()
    assert win.inspector.g_smart.isVisible(), "属性面板没显示智能对象分组"
    win.panel.rebuild()
    app.processEvents()
    print("  属性面板 / 图层面板：智能对象分组与角标 OK")

    # 新建实例 -> 两个图层共用一份内容
    win.new_smart_copy()
    app.processEvents()
    twin = win.selected_layer()
    assert len(content_instances(doc, twin.so_id)) == 2, "应该有两个实例"
    twin.tx += 240.0
    win._do_render()
    two_up = win._last_arr.copy()
    assert int(two_up[200, 440, 3]) == 255, "第二个实例没画出来"
    print("  新建实例：共用同一份内容，各自有独立的变换")

    # 编辑内容：在编辑器里改子树，关掉窗口两个实例一起变
    win.edit_smart_content(shell)
    app.processEvents()
    assert len(win._so_editors) == 1, "编辑器没打开"
    ed = win._so_editors[0]
    content = doc.smart_contents[shell.so_id]
    for l in content.layers:
        l.visible = False
    ed.close()
    app.processEvents()
    assert len(win._so_editors) == 0, "编辑器关闭后没从列表里摘掉"
    win._do_render()
    app.processEvents()
    hidden = win._last_arr.copy()
    assert np.abs(hidden.astype(np.int16) - two_up.astype(np.int16)).max() > 0, \
        "内容改了画面却没变"
    for l in content.layers:
        l.visible = True
    content.touch()
    win._do_render()
    assert np.abs(win._last_arr.astype(np.int16)
                  - two_up.astype(np.int16)).max() == 0, "改回来应该完全还原"
    print("  编辑内容：编辑器回写 + touch()，两个实例联动且可完全还原")

    # 智能滤镜是参数化的：加完之后画面变了，删掉完全还原
    from src.core.smart import new_filter

    before_filters = win._last_arr.copy()
    # 用浮雕：内容正好是这块实心矩形的包围盒，模糊会被边缘复制吃掉看不出变化
    shell.so_filters = [new_filter("emboss", {"angle": 45, "height": 3,
                                              "amount": 100})]
    win._do_render()
    blurred = win._last_arr.copy()
    assert np.abs(blurred.astype(np.int16)
                  - before_filters.astype(np.int16)).max() > 0, "滤镜没生效"
    shell.so_filters = []
    win._do_render()
    assert np.abs(win._last_arr.astype(np.int16)
                  - before_filters.astype(np.int16)).max() == 0, \
        "删掉智能滤镜应该回到原样"
    print("  智能滤镜：非破坏性，删掉即还原")

    # 智能对象图层不许直接画 / 填充
    before_paint = win._last_arr.copy()
    assert win._reject_text_layer("绘制"), "智能对象应该被拦下来"
    win.select_layer(shell.id)
    win.fill_with_fg(use_fg=True)
    app.processEvents()
    assert np.abs(win._last_arr.astype(np.int16)
                  - before_paint.astype(np.int16)).max() == 0, \
        "智能对象不该被直接填充"
    print("  写保护：智能对象拒绝直接绘制 / 填充")

    # 栅格化 -> 变成普通位图，内容被回收
    win.select_layer(shell.id)
    win.rasterize_smart()
    app.processEvents()
    win._do_render()
    assert shell.kind == "image" and shell.image is not None
    assert shell.so_id is None
    assert np.abs(win._last_arr.astype(np.int16)
                  - before_paint.astype(np.int16)).max() == 0, \
        "栅格化不该改变画面"
    assert twin.so_id in doc.smart_contents, "另一个实例还在，内容不该被回收"
    print("  栅格化：画面不变，变回普通图层（内容仍在被别的实例引用）")

    # 工程往返
    p = os.path.join(tempfile.gettempdir(), "cw_uitest_smart.cwproj")
    from src.core.project_io import save_project, load_project
    save_project(doc, p)
    back = load_project(p)
    assert len(back.smart_contents) == 1, "读回来的嵌入内容数量不对"
    win.set_document(back, reset_history=True)
    win._do_render()
    assert np.abs(win._last_arr.astype(np.int16)
                  - before_paint.astype(np.int16)).max() <= 1, \
        "工程往返后画面不一致"
    print("  工程往返：智能对象能存能读")

    win.close()
    print("界面冒烟测试全部通过。")


if __name__ == "__main__":
    main()
