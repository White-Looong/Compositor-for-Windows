# -*- coding: utf-8 -*-
"""画布视图：缩放/平移、变换手柄、选区绘制、画笔描边。

工具分派：
    move     变换当前图层（8 缩放手柄 + 旋转手柄）
    rect/ellipse/lasso/polygon/magic   建立选区
    brush/eraser                       像素绘制
    fill                               填充
    text                               在点击处创建文字图层
"""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (QBrush, QColor, QImage, QPainter, QPainterPath,
                           QPen, QPixmap, QPolygonF, QTransform)
from PySide6.QtWidgets import (QGraphicsEllipseItem, QGraphicsPathItem,
                               QGraphicsPixmapItem, QGraphicsPolygonItem,
                               QGraphicsRectItem, QGraphicsScene,
                               QGraphicsView)

from ..core.effects import effect_padding
from ..core.paint import Stroke
from ..core.selection import (ADD, INTERSECT, REPLACE, SUBTRACT, Selection)
from .tool_options import PAINT_TOOLS, SELECT_TOOLS

# 手柄定义: 名称 -> (锚点在图像中心坐标中的符号, 固定点符号, 影响 x/y)
HANDLE_DEFS = {
    "nw": ((-1, -1), (1, 1), (True, True)),
    "ne": ((1, -1), (-1, 1), (True, True)),
    "se": ((1, 1), (-1, -1), (True, True)),
    "sw": ((-1, 1), (1, -1), (True, True)),
    "n": ((0, -1), (0, 1), (False, True)),
    "s": ((0, 1), (0, -1), (False, True)),
    "w": ((-1, 0), (1, 0), (True, False)),
    "e": ((1, 0), (-1, 0), (True, False)),
}

ROT_DIST = 24.0     # 旋转手柄到上边中点的距离（屏幕像素）
ANT_INTERVAL = 90   # 蚁线动画间隔（毫秒）


class CanvasScene(QGraphicsScene):
    """带棋盘格背景的场景。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.doc_rect = QRectF(0, 0, 0, 0)
        self.checker = True

    def drawBackground(self, painter: QPainter, rect):
        painter.fillRect(rect, QColor(43, 43, 46))
        if not self.checker or self.doc_rect.isEmpty():
            return
        r = self.doc_rect.intersected(rect)
        if r.isEmpty():
            return
        s = 16.0
        x0 = math.floor(r.left() / s) * s
        y0 = math.floor(r.top() / s) * s
        dark = QColor(150, 150, 150)
        light = QColor(200, 200, 200)
        x = x0
        while x < r.right():
            y = y0
            while y < r.bottom():
                i = int((x - x0) / s) + int((y - y0) / s)
                painter.fillRect(QRectF(x, y, s, s), light if i % 2 == 0 else dark)
                y += s
            x += s


class CanvasView(QGraphicsView):
    transformChanged = Signal()
    statusMessage = Signal(str)

    def __init__(self, main):
        super().__init__()
        self.main = main
        self.zoom = 1.0
        self.tool = "move"
        self._space_down = False
        self._panning = False
        self._pan_start = None
        self._drag = None
        self._buf = None

        # 选区相关
        self._preview_sel = None      # 拖拽中的临时选区
        self._sel_drag = None
        self._poly_pts = []
        self._dash = 0.0

        # 绘制相关
        self._stroke = None

        self.scene = CanvasScene(self)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.Antialiasing, False)
        self.setRenderHint(QPainter.SmoothPixmapTransform, False)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setMouseTracking(True)
        self.setBackgroundBrush(QColor(43, 43, 46))

        self.pix_item = self.scene.addPixmap(QPixmap())
        self.pix_item.setZValue(0)

        self.border_item = QGraphicsRectItem()
        self.border_item.setZValue(5)
        self.border_item.setPen(QPen(QColor(120, 120, 125), 0))
        self.scene.addItem(self.border_item)

        # 选区：半透明蓝底 + 黑白双虚线（在任何底上都看得见）
        self.sel_fill = self.scene.addPixmap(QPixmap())
        self.sel_fill.setZValue(6)
        self.sel_dark = QGraphicsPathItem()
        self.sel_dark.setZValue(11)
        self.scene.addItem(self.sel_dark)
        self.sel_light = QGraphicsPathItem()
        self.sel_light.setZValue(12)
        self.scene.addItem(self.sel_light)

        self.outline = QGraphicsPolygonItem()
        self.outline.setZValue(9)
        self.outline.setPen(QPen(QColor(30, 120, 220), 0))
        self.outline.setBrush(QBrush(Qt.NoBrush))
        self.scene.addItem(self.outline)

        self.handles = {}
        for name in HANDLE_DEFS:
            it = QGraphicsRectItem()
            it.setZValue(10)
            it.setPen(QPen(QColor(30, 120, 220), 0))
            it.setBrush(QBrush(QColor(255, 255, 255)))
            it.setData(0, name)
            self.scene.addItem(it)
            self.handles[name] = it
        self.rot_handle = QGraphicsEllipseItem()
        self.rot_handle.setZValue(10)
        self.rot_handle.setPen(QPen(QColor(30, 120, 220), 0))
        self.rot_handle.setBrush(QBrush(QColor(255, 255, 255)))
        self.rot_handle.setData(0, "rot")
        self.scene.addItem(self.rot_handle)

        self.brush_ring = QGraphicsEllipseItem()
        self.brush_ring.setZValue(20)
        self.brush_ring.setPen(QPen(QColor(255, 255, 255, 200), 0))
        self.brush_ring.setBrush(QBrush(Qt.NoBrush))
        self.scene.addItem(self.brush_ring)

        self._ant_timer = QTimer(self)
        self._ant_timer.setInterval(ANT_INTERVAL)
        self._ant_timer.timeout.connect(self._tick_ants)
        self._ant_timer.start()

    # ================= 显示 =================

    def set_pixmap(self, arr: np.ndarray, scale=None):
        """把合成结果显示到画布上。

        scale: (kx, ky) 表示 arr 是**按这个比例缩小的代理图**（交互期快预览用），
        位图会被放大回画布尺寸；None = 全分辨率，1:1 显示。
        """
        h, w = arr.shape[:2]
        self._buf = arr
        img = QImage(arr.data, w, h, 4 * w, QImage.Format_RGBA8888)
        self.pix_item.setPixmap(QPixmap.fromImage(img))
        if scale is None:
            self.pix_item.setTransform(QTransform())
        else:
            kx, ky = scale
            self.pix_item.setTransform(
                QTransform.fromScale(1.0 / max(kx, 1e-6), 1.0 / max(ky, 1e-6)))
        # 边框与场景矩形始终按**文档**尺寸算 —— 代理图比画布小，不能用它
        doc = self.main.doc
        dw = doc.width if doc is not None else w
        dh = doc.height if doc is not None else h
        self.border_item.setRect(QRectF(-0.5, -0.5, dw + 1, dh + 1))
        self.scene.doc_rect = QRectF(0, 0, dw, dh)
        self.scene.setSceneRect(QRectF(-2000, -2000, dw + 4000, dh + 4000))

    def set_zoom(self, z, anchor=None):
        z = max(0.02, min(64.0, z))
        self.zoom = z
        self.setTransform(self._make_transform())
        self._update_handles()
        self.update_selection_overlay()
        self.transformChanged.emit()

    def _make_transform(self):
        return QTransform.fromScale(self.zoom, self.zoom)

    def fit(self):
        doc = self.main.doc
        if not doc:
            return
        vp = self.viewport().rect()
        z = min((vp.width() - 40) / max(1, doc.width),
                (vp.height() - 40) / max(1, doc.height))
        self.set_zoom(z)
        self.centerOn(doc.width / 2.0, doc.height / 2.0)

    def zoom_actual(self):
        self.set_zoom(1.0)

    # ================= 选区显示 =================

    def active_selection(self):
        if self._preview_sel is not None:
            return self._preview_sel
        doc = self.main.doc
        return doc.selection if doc else None

    def update_selection_overlay(self):
        sel = self.active_selection()
        if sel is None or sel.is_empty:
            self.sel_fill.setPixmap(QPixmap())
            self.sel_dark.setPath(QPainterPath())
            self.sel_light.setPath(QPainterPath())
            return

        h, w = sel.mask.shape[:2]
        a = (sel.mask.astype(np.float32) * 0.35).astype(np.uint8)
        arr = np.zeros((h, w, 4), np.uint8)
        arr[..., 0] = 60
        arr[..., 1] = 140
        arr[..., 2] = 255
        arr[..., 3] = a
        arr = np.ascontiguousarray(arr)
        img = QImage(arr.data, w, h, 4 * w, QImage.Format_RGBA8888)
        self._sel_buf = arr
        self.sel_fill.setPixmap(QPixmap.fromImage(img))

        path = QPainterPath()
        for c in sel.contours():
            path.moveTo(float(c[0][0]), float(c[0][1]))
            for pt in c[1:]:
                path.lineTo(float(pt[0]), float(pt[1]))
            path.closeSubpath()
        self.sel_dark.setPath(path)
        self.sel_light.setPath(path)
        self._tick_ants()

    def _tick_ants(self):
        if self.sel_dark.path().isEmpty():
            return
        w = 1.0 / self.zoom
        self._dash = (self._dash + 1.0) % 8.0
        for item, color, off in ((self.sel_dark, QColor(20, 20, 20), self._dash),
                                 (self.sel_light, QColor(255, 255, 255),
                                  self._dash + 4.0)):
            pen = QPen(color, w)
            pen.setStyle(Qt.PenStyle.CustomDashLine)
            pen.setDashPattern([4.0, 4.0])
            pen.setDashOffset(off)
            item.setPen(pen)

    # ================= 变换几何 =================

    def _visual(self, layer, ux, uy):
        fx = -1.0 if layer.flip_h else 1.0
        fy = -1.0 if layer.flip_v else 1.0
        x = layer.sx * fx * ux
        y = layer.sy * fy * uy
        th = math.radians(layer.rot)
        c, s = math.cos(th), math.sin(th)
        return (layer.tx + x * c - y * s, layer.ty + x * s + y * c)

    def _local(self, layer, px, py):
        fx = -1.0 if layer.flip_h else 1.0
        fy = -1.0 if layer.flip_v else 1.0
        dx, dy = px - layer.tx, py - layer.ty
        th = math.radians(layer.rot)
        c, s = math.cos(th), math.sin(th)
        x = dx * c + dy * s
        y = -dx * s + dy * c
        sx = layer.sx * fx if abs(layer.sx) > 1e-9 else 1e-9
        sy = layer.sy * fy if abs(layer.sy) > 1e-9 else 1e-9
        return x / sx, y / sy

    def handle_points(self, layer):
        pts = {}
        src = layer.src_size
        if src is None:
            return None, None
        w, h = src
        hw, hh = w / 2.0, h / 2.0
        for name, (a, f, ax) in HANDLE_DEFS.items():
            x, y = self._visual(layer, a[0] * hw, a[1] * hh)
            pts[name] = QPointF(x, y)
        mx, my = self._visual(layer, 0.0, -hh)
        th = math.radians(layer.rot)
        ux, uy = -math.sin(th), math.cos(th)
        d = ROT_DIST / self.zoom
        return pts, QPointF(mx + ux * d, my + uy * d)

    def _update_handles(self):
        layer = self.main.selected_layer()
        show = layer is not None and not layer.locked and self.tool in (
            "move", "")
        for it in self.handles.values():
            it.setVisible(False)
        self.rot_handle.setVisible(False)
        self.outline.setVisible(False)
        self._update_brush_ring()
        if not show:
            return

        hs = 7.0 / self.zoom
        if layer.is_group:
            box = self.main.doc.layer_bbox(layer)
            if box is None:
                return
            r = QRectF(box[0], box[1], box[2] - box[0], box[3] - box[1])
            self.outline.setPolygon(QPolygonF([
                QPointF(r.left(), r.top()), QPointF(r.right(), r.top()),
                QPointF(r.right(), r.bottom()), QPointF(r.left(), r.bottom())]))
            self.outline.setVisible(True)
            for name, (px, py) in (("nw", (r.left(), r.top())),
                                   ("ne", (r.right(), r.top())),
                                   ("se", (r.right(), r.bottom())),
                                   ("sw", (r.left(), r.bottom()))):
                it = self.handles[name]
                it.setRect(QRectF(px - hs / 2, py - hs / 2, hs, hs))
                it.setVisible(True)
            return

        pts, rot_pt = self.handle_points(layer)
        if pts is None:
            return
        self.outline.setPolygon(QPolygonF(
            [pts["nw"], pts["ne"], pts["se"], pts["sw"]]))
        self.outline.setVisible(True)
        for name, p in pts.items():
            it = self.handles[name]
            it.setRect(QRectF(p.x() - hs / 2, p.y() - hs / 2, hs, hs))
            it.setVisible(True)
        self.rot_handle.setRect(
            QRectF(rot_pt.x() - hs / 2, rot_pt.y() - hs / 2, hs, hs))
        self.rot_handle.setVisible(True)

    def _update_brush_ring(self, pos=None):
        if self.tool in PAINT_TOOLS:
            r = self.main.opts.size / 2.0
            if pos is None:
                self.brush_ring.setVisible(False)
                return
            self.brush_ring.setRect(QRectF(pos.x() - r, pos.y() - r, 2 * r, 2 * r))
            self.brush_ring.setPen(QPen(QColor(255, 255, 255, 210), 1.0 / self.zoom))
            self.brush_ring.setVisible(True)
        else:
            self.brush_ring.setVisible(False)

    def _hit_handle(self, scene_pos):
        layer = self.main.selected_layer()
        if layer is None or self.tool != "move":
            return None
        tol = 7.0 / self.zoom
        if not layer.is_group and self.rot_handle.isVisible():
            r = self.rot_handle.rect()
            if (abs(scene_pos.x() - r.center().x()) <= tol * 1.4 and
                    abs(scene_pos.y() - r.center().y()) <= tol * 1.4):
                return "rot"
        for name, it in self.handles.items():
            if not it.isVisible():
                continue
            c = it.rect().center()
            if (abs(scene_pos.x() - c.x()) <= tol and
                    abs(scene_pos.y() - c.y()) <= tol):
                return name
        return None

    def _hit_box(self, layer, scene_pos):
        if layer.is_group:
            box = self.main.doc.layer_bbox(layer)
            if box is None:
                return False
            return (box[0] <= scene_pos.x() <= box[2] and
                    box[1] <= scene_pos.y() <= box[3])
        src = layer.src_size
        if src is None:
            return False
        w, h = src
        ux, uy = self._local(layer, scene_pos.x(), scene_pos.y())
        return abs(ux) <= w / 2.0 and abs(uy) <= h / 2.0

    # ================= 选区交互 =================

    def _modifier_mode(self, modifiers):
        shift = bool(modifiers & Qt.ShiftModifier)
        alt = bool(modifiers & Qt.AltModifier)
        if shift and alt:
            return INTERSECT
        if shift:
            return ADD
        if alt:
            return SUBTRACT
        return self.main.opts.sel_mode

    def _start_sel_drag(self, p, tool, modifiers):
        mode = self._modifier_mode(modifiers)
        sel = self.main.doc.selection
        inside = (sel is not None and not sel.is_empty and sel.contains(p.x(), p.y()))
        if tool in ("rect", "ellipse") and mode == REPLACE and inside:
            self._sel_drag = {"kind": "move", "start": p, "orig": sel.copy()}
            return True
        if tool == "lasso":
            self._sel_drag = {"kind": "lasso", "mode": mode, "pts": [p]}
            self._preview_sel = None
            return True
        self._sel_drag = {"kind": "shape", "tool": tool, "mode": mode,
                          "start": p, "cur": p}
        self._preview_sel = None
        return True

    def _build_shape_sel(self, d, modifiers):
        doc = self.main.doc
        a, b = d["start"], d["cur"]
        x0, y0 = a.x(), a.y()
        x1, y1 = b.x(), b.y()
        if modifiers & Qt.ShiftModifier:
            dx, dy = x1 - x0, y1 - y0
            m = max(abs(dx), abs(dy))
            x1 = x0 + (m if dx >= 0 else -m)
            y1 = y0 + (m if dy >= 0 else -m)
        if d["tool"] == "rect":
            return Selection.rect(doc.width, doc.height, x0, y0, x1, y1)
        return Selection.ellipse(doc.width, doc.height, x0, y0, x1, y1)

    def _poly_click(self, p):
        pts = self._poly_pts
        if len(pts) >= 3:
            d = math.hypot(p.x() - pts[0].x(), p.y() - pts[0].y())
            if d <= 8.0 / self.zoom:
                self._commit_polygon()
                return
        pts.append(QPointF(p))
        doc = self.main.doc
        arr = [(q.x(), q.y()) for q in pts]
        if len(arr) >= 3:
            self._preview_sel = Selection.polygon(doc.width, doc.height, arr)
        else:
            self._preview_sel = None
        self.update_selection_overlay()

    def _commit_polygon(self):
        doc = self.main.doc
        if len(self._poly_pts) >= 3:
            arr = [(q.x(), q.y()) for q in self._poly_pts]
            sel = Selection.polygon(doc.width, doc.height, arr)
            self.main.commit_selection(sel, self.main.opts.sel_mode)
        self._poly_pts = []
        self._preview_sel = None
        self.update_selection_overlay()

    def cancel_operation(self):
        """Esc：取消进行中的套索/多边形/选区。"""
        if self._drag is not None:
            self._drag = None
            self.main.set_interactive(False)
            self.main.request_render()
            return True
        if self._poly_pts:
            self._poly_pts = []
            self._preview_sel = None
            self.update_selection_overlay()
            return True
        if self._sel_drag is not None:
            self._sel_drag = None
            self._preview_sel = None
            self.update_selection_overlay()
            return True
        return False

    def _do_magic(self, p, modifiers):
        doc = self.main.doc
        rgb = self.main.composite_rgb()
        contiguous = not bool(modifiers & Qt.ControlModifier)
        sel = Selection.magic(doc.width, doc.height, rgb, p.x(), p.y(),
                              self.main.opts.tolerance, contiguous=contiguous)
        self.main.commit_selection(sel, self._modifier_mode(modifiers))

    # ================= 绘制 =================

    def _start_paint(self, p, modifiers):
        layer = self.main.selected_layer()
        if layer is None:
            self.statusMessage.emit("先选中一个位图图层再画")
            return False
        # 调整层没有像素，但它的蒙版可以画
        if layer.is_group or (layer.image is None and not layer.is_adjustment):
            self.statusMessage.emit("这个图层不能绘制")
            return False
        # 文字图层的像素是参数生成的，画上去下次改字就没了
        if layer.is_text:
            self.statusMessage.emit("文字图层不能直接绘制 —— 先在属性面板里栅格化")
            return False
        # 智能对象的像素是内容渲染出来的，下一帧就被内容覆盖回去了
        if layer.is_smart:
            self.statusMessage.emit(
                "智能对象不能直接绘制 —— 用「编辑内容」改内容、用「智能滤镜」加效果，"
                "或者先栅格化")
            return False
        if layer.locked:
            self.statusMessage.emit("图层已锁定")
            return False
        opts = self.main.opts
        target = opts.target if self.tool == "brush" else "pixel"
        if layer.is_adjustment:
            # 调整层没有像素，只能画它的蒙版
            if layer.mask is None:
                self.statusMessage.emit("调整层没有像素，先给它加个蒙版再画")
                return False
            target = "mask"
        elif target == "mask" and layer.mask is None:
            target = "pixel"
        st = Stroke(doc=self.main.doc, layer=layer, tool=self.tool,
                    size=opts.size, hardness=opts.hardness,
                    opacity=opts.opacity, flow=opts.flow,
                    smoothing=opts.smoothing, target=target,
                    color=opts.fg.rgb())
        if not st.begin((p.x(), p.y())):
            self.statusMessage.emit("无法在该图层绘制")
            return False
        self._stroke = st
        self.main.request_render()
        return True

    # ================= 局部重渲染（脏矩形） =================

    def _layer_dirty_rect(self, layer):
        """图层当前在画布上占的矩形（含图层样式的外扩），用于只重算这一块。

        返回 None 表示"算不出来"，调用方会退化成整幅重算 —— 宁可慢也不能错。
        """
        doc = self.main.doc
        if doc is None or layer is None:
            return None
        box = doc.layer_bbox(layer)
        if box is None:
            return None
        pad = int(effect_padding(layer.effects)) + 2
        x0 = int(np.floor(box[0])) - pad
        y0 = int(np.floor(box[1])) - pad
        x1 = int(np.ceil(box[2])) + pad
        y1 = int(np.ceil(box[3])) + pad
        x0 = max(0, min(doc.width, x0))
        y0 = max(0, min(doc.height, y0))
        x1 = max(0, min(doc.width, x1))
        y1 = max(0, min(doc.height, y1))
        if x1 <= x0 or y1 <= y0:
            return None
        return (x0, y0, x1, y1)

    def _render_moved(self, before, layer):
        """图层动过了：只重算 "旧位置 ∪ 新位置"。"""
        after = self._layer_dirty_rect(layer)
        if before is None or after is None:
            self.main.request_render()
            return
        self.main.request_render((min(before[0], after[0]),
                                  min(before[1], after[1]),
                                  max(before[2], after[2]),
                                  max(before[3], after[3])))

    # ================= 浮动选区 =================

    def _start_float_drag(self, p, modifiers):
        """移动工具 + 有选区 = 拖动选区里的像素。

        PS 里 Alt 拖动是"复制一份再挪"（原处保留），这里一样。
        返回 True 表示已经接管这次拖拽。
        """
        doc = self.main.doc
        if doc is None:
            return False
        sel = doc.selection
        if sel is None or sel.is_empty or not sel.contains(p.x(), p.y()):
            return False
        if doc.float_layer is None:
            # 先只记下意图，等真的拖动了再揭 —— 免得单击一下就把选区挖空
            self._drag = {"role": "float_pending", "start": p, "layer": None,
                          "tx": 0.0, "ty": 0.0, "moved": False,
                          "sel": sel.copy(),
                          "copy": bool(modifiers & Qt.AltModifier)}
            return True

        fl = doc.float_layer
        if fl is None:
            return False
        self._drag = {"role": "float", "start": p, "layer": fl,
                      "tx": fl.tx, "ty": fl.ty, "moved": False,
                      "sel": sel.copy()}
        return True

    def _apply_float_pending(self, d, p):
        """第一次真正移动时才把内容揭出来。"""
        dx = p.x() - d["start"].x()
        dy = p.y() - d["start"].y()
        if math.hypot(dx, dy) * self.zoom < 2.0:
            return
        if not self.main.lift_selection(copy_mode=d.get("copy", False)):
            self._drag = None
            return
        fl = self.main.doc.float_layer
        if fl is None:
            self._drag = None
            return
        d["role"] = "float"
        d["layer"] = fl
        d["tx"] = fl.tx
        d["ty"] = fl.ty
        self._apply_float_drag(d, p)

    def _apply_float_drag(self, d, p):
        fl = d["layer"]
        before = self._layer_dirty_rect(fl)
        dx = p.x() - d["start"].x()
        dy = p.y() - d["start"].y()
        fl.tx = d["tx"] + dx
        fl.ty = d["ty"] + dy
        d["moved"] = True
        # 选区轮廓跟着浮动内容一起走（蚂蚁线围住的是浮起来的像素）
        sel = d["sel"].copy()
        sel.translate(dx, dy)
        self.main.doc.selection = sel
        self._render_moved(before, fl)
        self.update_selection_overlay()
        self.statusMessage.emit("偏移: %.0f, %.0f" % (dx, dy))

    # ================= 鼠标 =================

    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y()
            f = 1.25 if delta > 0 else 1 / 1.25
            self.set_zoom(self.zoom * f)
            event.accept()
            return
        super().wheelEvent(event)

    def mousePressEvent(self, event):
        if (event.button() == Qt.MiddleButton or self._space_down
                or self.tool == "hand"):
            self._panning = True
            self._pan_start = event.position()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        if event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return

        p = self.mapToScene(event.position().toPoint())
        t = self.tool

        # 建新选区 = 放弃浮动状态，先把浮动内容盖回去
        if t in ("rect", "ellipse", "lasso", "polygon", "magic"):
            self.main.stamp_float(silent=True)

        if t in PAINT_TOOLS:
            self._start_paint(p, event.modifiers())
            event.accept()
            return
        if t == "fill":
            self.main.fill_with_fg(p)
            event.accept()
            return
        if t == "text":
            self.main.create_text_layer((p.x(), p.y()))
            event.accept()
            return
        if t == "magic":
            self._do_magic(p, event.modifiers())
            event.accept()
            return
        if t == "polygon":
            self._poly_click(p)
            event.accept()
            return
        if t in ("rect", "ellipse", "lasso"):
            if self._start_sel_drag(p, t, event.modifiers()):
                event.accept()
                return
            super().mousePressEvent(event)
            return

        # ---- 移动/变换工具 ----
        layer = self.main.selected_layer()
        if layer is None or layer.locked:
            super().mousePressEvent(event)
            return
        role = self._hit_handle(p)
        if role:
            self._drag = {"role": role, "start": p, "layer": layer,
                          "sx": layer.sx, "sy": layer.sy, "rot": layer.rot,
                          "tx": layer.tx, "ty": layer.ty,
                          "a0": math.atan2(p.y() - layer.ty, p.x() - layer.tx),
                          "moved": False}
            self.main.set_interactive(True)     # 大画布上改走低分辨率代理
            event.accept()
            return
        if self.tool == "move" and self._start_float_drag(p, event.modifiers()):
            self.main.set_interactive(True)
            event.accept()
            return
        if self._hit_box(layer, p):
            self._drag = {"role": "move", "start": p, "layer": layer,
                          "tx": layer.tx, "ty": layer.ty, "moved": False}
            self.main.set_interactive(True)
            event.accept()
            return
        self.main.select_layer(None)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning:
            d = event.position() - self._pan_start
            self._pan_start = event.position()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(d.x()))
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(d.y()))
            event.accept()
            return

        p = self.mapToScene(event.position().toPoint())

        if self._stroke is not None:
            self._stroke.extend((p.x(), p.y()))
            # 只重算这一笔抹过的那一小块（图层样式要往外画，留够外扩）
            layer = self.main.selected_layer()
            pad = int(effect_padding(layer.effects)) + 2 if layer else 2
            self.main.request_render(self._stroke.take_dirty(pad))
            self._update_brush_ring(p)
            event.accept()
            return

        if self._sel_drag is not None:
            d = self._sel_drag
            if d["kind"] == "shape":
                d["cur"] = p
                self._preview_sel = self._build_shape_sel(d, event.modifiers())
            elif d["kind"] == "lasso":
                last = d["pts"][-1]
                if math.hypot(p.x() - last.x(), p.y() - last.y()) > 2.0 / self.zoom:
                    d["pts"].append(QPointF(p))
                if len(d["pts"]) >= 3:
                    arr = [(q.x(), q.y()) for q in d["pts"]]
                    doc = self.main.doc
                    self._preview_sel = Selection.polygon(doc.width, doc.height, arr)
            elif d["kind"] == "move":
                tmp = d["orig"].copy()
                tmp.translate(p.x() - d["start"].x(), p.y() - d["start"].y())
                self._preview_sel = tmp
            self.update_selection_overlay()
            event.accept()
            return

        if self.tool == "polygon" and self._poly_pts:
            arr = [(q.x(), q.y()) for q in self._poly_pts] + [(p.x(), p.y())]
            if len(arr) >= 3:
                doc = self.main.doc
                self._preview_sel = Selection.polygon(doc.width, doc.height, arr)
                self.update_selection_overlay()
            event.accept()
            return

        if self._drag is not None:
            self._apply_drag(p, event.modifiers())
            event.accept()
            return

        if self.tool in PAINT_TOOLS:
            self.setCursor(Qt.CrossCursor)
            self._update_brush_ring(p)
            super().mouseMoveEvent(event)
            return

        if self.tool == "text":
            self.setCursor(Qt.CrossCursor)
            super().mouseMoveEvent(event)
            return

        # 仅更新光标
        layer = self.main.selected_layer()
        role = self._hit_handle(p) if layer else None
        cursors = {
            "nw": Qt.SizeFDiagCursor, "se": Qt.SizeFDiagCursor,
            "ne": Qt.SizeBDiagCursor, "sw": Qt.SizeBDiagCursor,
            "n": Qt.SizeVerCursor, "s": Qt.SizeVerCursor,
            "w": Qt.SizeHorCursor, "e": Qt.SizeHorCursor,
        }
        if role == "rot":
            self.setCursor(Qt.CrossCursor)
        elif role in cursors:
            self.setCursor(cursors[role])
        elif layer and self._hit_box(layer, p) and self.tool == "move":
            self.setCursor(Qt.SizeAllCursor)
        else:
            self.setCursor(Qt.ArrowCursor)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._panning:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return

        if self._stroke is not None:
            self._stroke.end()
            self._stroke = None
            self.main.after_pixel_edit()
            event.accept()
            return

        if self._sel_drag is not None:
            d = self._sel_drag
            sel = self._preview_sel
            mode = d.get("mode", REPLACE)
            self._sel_drag = None
            if d["kind"] == "move":
                self.main.doc.detach_selection()
                self.main.doc.selection = sel
                self.main.commit("移动选区")
            elif sel is not None and not sel.is_empty:
                self.main.commit_selection(sel, mode)
            else:
                self._preview_sel = None
                self.update_selection_overlay()
            event.accept()
            return

        if self._drag is not None:
            role = self._drag.get("role")
            moved = self._drag.get("moved")
            self._drag = None
            if moved:
                # 浮动层保持"浮着"，下次还能接着拖；回车 / 换工具才落定
                self.main.commit("移动选区内容" if role == "float" else "变换")
            self.main.set_interactive(False)   # 松手 -> 补一张全分辨率
            self.main.request_render()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self.tool == "polygon" and self._poly_pts:
            self._commit_polygon()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def _apply_drag(self, p, modifiers):
        d = self._drag
        layer = d["layer"]
        d["moved"] = True
        before = self._layer_dirty_rect(layer)   # 改动之前占的位置

        if d["role"] == "float_pending":
            self._apply_float_pending(d, p)
            return

        if d["role"] == "float":
            self._apply_float_drag(d, p)
            return

        if d["role"] == "move":
            dx = p.x() - d["start"].x()
            dy = p.y() - d["start"].y()
            if modifiers & Qt.ShiftModifier:
                if abs(dx) > abs(dy):
                    dy = 0.0
                else:
                    dx = 0.0
            layer.tx = d["tx"] + dx
            layer.ty = d["ty"] + dy
            self._render_moved(before, layer)
            self._update_handles()
            self.statusMessage.emit("位置: %.0f, %.0f" % (layer.tx, layer.ty))
            return

        if d["role"] == "rot":
            a1 = math.atan2(p.y() - layer.ty, p.x() - layer.tx)
            deg = d["rot"] + math.degrees(a1 - d["a0"])
            if modifiers & Qt.ShiftModifier:
                deg = round(deg / 15.0) * 15.0
            layer.rot = deg % 360.0
            self._render_moved(before, layer)
            self._update_handles()
            self.statusMessage.emit("旋转: %.1f°" % layer.rot)
            return

        src = layer.src_size
        if src is None:
            return
        w, h = src
        hw, hh = w / 2.0, h / 2.0
        a, f, axes = HANDLE_DEFS[d["role"]]

        # 必须用**拖拽开始时**记录的变换换算局部坐标，否则会自我反馈
        fx = -1.0 if layer.flip_h else 1.0
        fy = -1.0 if layer.flip_v else 1.0
        th = math.radians(d["rot"])
        c0, s0 = math.cos(th), math.sin(th)
        dx0, dy0 = p.x() - d["tx"], p.y() - d["ty"]
        lx = dx0 * c0 + dy0 * s0
        ly = -dx0 * s0 + dy0 * c0
        ux = lx / (d["sx"] * fx) if abs(d["sx"]) > 1e-9 else 0.0
        uy = ly / (d["sy"] * fy) if abs(d["sy"]) > 1e-9 else 0.0

        new_sx = d["sx"]
        new_sy = d["sy"]
        if axes[0]:
            new_sx = max(0.01, d["sx"] * abs(ux) / hw)
        if axes[1]:
            new_sy = max(0.01, d["sy"] * abs(uy) / hh)
        if modifiers & Qt.ShiftModifier:
            rx = new_sx / d["sx"] if axes[0] else 1.0
            ry = new_sy / d["sy"] if axes[1] else 1.0
            r = max(rx, ry)
            new_sx = d["sx"] * r
            new_sy = d["sy"] * r

        # 保持对角（固定点）不动
        c, s = c0, s0
        fix_x, fix_y = f[0] * hw, f[1] * hh
        ox = d["sx"] * fx * fix_x
        oy = d["sy"] * fy * fix_y
        px = d["tx"] + ox * c - oy * s
        py = d["ty"] + ox * s + oy * c

        nx = new_sx * fx * fix_x
        ny = new_sy * fy * fix_y
        layer.tx = px - (nx * c - ny * s)
        layer.ty = py - (nx * s + ny * c)
        layer.sx = new_sx
        layer.sy = new_sy

        self._render_moved(before, layer)
        self._update_handles()
        self.main.refresh_inspector()
        self.statusMessage.emit("缩放: %.1f%% x %.1f%%" %
                                (layer.sx * 100, layer.sy * 100))

    # ================= 键盘 =================

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if self.main.doc is not None and self.main.doc.float_layer is not None:
                self.main.stamp_float()
                event.accept()
                return
        if event.key() == Qt.Key_Escape:
            if self.main.doc is not None and self.main.doc.float_layer is not None:
                self.main.discard_float()
                event.accept()
                return
            if self.cancel_operation():
                event.accept()
                return
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._space_down = True
            self.setCursor(Qt.OpenHandCursor)
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Space:
            self._space_down = False
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return
        super().keyReleaseEvent(event)
