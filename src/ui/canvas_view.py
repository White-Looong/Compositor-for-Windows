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
import time

import numpy as np
from PySide6.QtCore import (QEvent, QLineF, QPointF, QRect, QRectF, Qt, QTimer,
                          Signal)
from PySide6.QtGui import (QBrush, QColor, QFont, QImage, QPainter,
                           QPainterPath, QPen, QPixmap, QPolygonF,
                           QTextBlockFormat, QTextCursor, QTransform)
from PySide6.QtWidgets import (QFrame, QGraphicsEllipseItem,
                               QGraphicsItem, QGraphicsLineItem,
                               QGraphicsPathItem, QGraphicsPixmapItem,
                               QGraphicsPolygonItem, QGraphicsRectItem,
                               QGraphicsScene, QGraphicsView, QPlainTextEdit)

from ..core import guides as _guides
from ..core import quick_mask
from ..core.effects import effect_padding
from ..core.paint import Stroke, _gray
from ..core.selection import (ADD, INTERSECT, REPLACE, SUBTRACT, Selection)
from . import brush_dialog
from .tool_options import (BLUR_TOOLS, CLONE_TOOLS, GRADIENT_TOOLS,
                          PAINT_TOOLS, PICK_TOOLS, SELECT_TOOLS,
                          SHAPE_TOOLS)

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
        self._qm_tick = 0.0           # 快速蒙版红罩的刷新节流时间戳

        # 绘制相关
        self._stroke = None
        # 模糊工具按住拖动时的轨迹（用来补样，快速拖动不漏成珠子）
        self._blur_drag = None
        # 污点修复 / 克隆图章的轨迹，形状工具的起点
        self._heal_drag = None
        self._clone_drag = None
        self._shape_start = None
        # 克隆图章的采样源（按住 Alt 时由 main_window 填进来）
        self.clone_source = None
        # 渐变工具那条预览线（场景在 setScene 之后才挂，得在那儿再建）
        self.grad_line = None
        # 向导类（第二十一批）：标尺 / 参考线 / 网格 / 拖动吸附
        self.show_rulers = True
        self.show_grid = False
        self.grid_spacing = 50.0
        self.grid_subdiv = 1
        self.snap_enabled = True
        self._snap_lines = []          # 拖动时命中的那些线（画高亮）
        self._drag_guides = None       # 从标尺拖参考线：{"kind":..., "cur":QPointF}
        self._snap_kind = None         # 移动工具拖动时命中的吸附源
        # 画布内编辑文字（Ctrl+T）：叠在 viewport 上的输入框
        self._text_edit = None
        self._text_edit_layer = None
        self._text_edit_style = None
        self._text_edit_busy = False     # 行高 merge 期间挡住重入
        # 喷枪：按住不动也要补笔，所以得有个自己的定时器
        self._air_timer = QTimer(self)
        self._air_timer.setSingleShot(False)
        self._air_timer.timeout.connect(self._airbrush_tick)

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

        # 快速蒙版：半透明红罩（盖在未选中 / 选中的那一侧，见 core/quick_mask.py）
        self.qm_fill = self.scene.addPixmap(QPixmap())
        self.qm_fill.setZValue(7)

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

        # 参考线：一堆细长矩形（竖线 = 窄而高，横线 = 宽而窄）
        # 用图元而不是 drawForeground 画，是为了让它们能接收鼠标事件
        # （拖动已有参考线 / 从标尺拖出新的一条）
        self._guide_items = []
        self._guide_dragging = False

        # 形状工具的预览框
        self.shape_preview = QGraphicsRectItem()
        self.shape_preview.setZValue(49)
        self.shape_preview.setPen(QPen(QColor(255, 255, 255, 220), 1.4))
        self.shape_preview.setBrush(QBrush(Qt.NoBrush))
        self.shape_preview.setVisible(False)
        self.scene.addItem(self.shape_preview)

        # 克隆图章的源位置标记（跟着 Alt 采样点走）
        self.clone_ring = QGraphicsRectItem()
        self.clone_ring.setZValue(48)
        self.clone_ring.setPen(QPen(QColor(255, 220, 120, 230), 1.4))
        self.clone_ring.setBrush(QBrush(Qt.NoBrush))
        self.clone_ring.setVisible(False)
        self.scene.addItem(self.clone_ring)

        # 渐变工具的预览线（场景就绪后才能 addItem）
        self.grad_line = QGraphicsLineItem()
        self.grad_line.setZValue(50)
        self.grad_line.setPen(QPen(QColor(255, 255, 255, 220), 1.4))
        self.grad_line.setVisible(False)
        self.scene.addItem(self.grad_line)

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
        self._place_text_edit()
        self.transformChanged.emit()

    def _make_transform(self):
        return QTransform.fromScale(self.zoom, self.zoom)

    def _ruler_inset(self):
        """标尺占掉的视口边距（四周内缩）。"""
        if not self.show_rulers:
            return 0, 0, 0, 0
        r = self.RULER
        return (r, r, 0, 0)

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
        if quick_mask.is_on(self.main.doc):
            # 快速蒙版模式下选区用红罩表示，蚂蚁线不画（PS 也是这样）
            self.sel_fill.setPixmap(QPixmap())
            self.sel_dark.setPath(QPainterPath())
            self.sel_light.setPath(QPainterPath())
            self._update_quick_mask_overlay()
            return
        self.qm_fill.setPixmap(QPixmap())
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

    def _update_quick_mask_overlay(self):
        """快速蒙版的红罩：alpha 由 core/quick_mask.overlay_alpha 算。"""
        doc = self.main.doc
        a = quick_mask.overlay_alpha(doc)
        if a is None:
            self.qm_fill.setPixmap(QPixmap())
            return
        h, w = a.shape[:2]
        arr = np.zeros((h, w, 4), np.uint8)
        arr[..., 0] = 255
        arr[..., 3] = a
        arr = np.ascontiguousarray(arr)
        self._qm_buf = arr
        img = QImage(arr.data, w, h, 4 * w, QImage.Format_RGBA8888)
        self.qm_fill.setPixmap(QPixmap.fromImage(img))

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

    def _handle_layer(self):
        """手柄挂在哪一层上。

        有浮动层时挂**浮动层** —— 浮起来的那块内容也可以直接拖手柄缩放 /
        旋转，不用先落定。没有浮动层时才是当前选中图层。
        """
        doc = self.main.doc
        if doc is not None and doc.float_layer is not None \
                and self.tool == "move":
            return doc.float_layer
        return self.main.selected_layer()

    def _update_handles(self):
        layer = self._handle_layer()
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
        if self.tool in PAINT_TOOLS or self.tool in BLUR_TOOLS \
                or self.tool in CLONE_TOOLS:
            r = self.main.opts.size / 2.0
            if pos is None:
                self.brush_ring.setVisible(False)
                return
            self.brush_ring.setRect(QRectF(pos.x() - r, pos.y() - r, 2 * r, 2 * r))
            self.brush_ring.setPen(QPen(QColor(255, 255, 255, 210), 1.0 / self.zoom))
            self.brush_ring.setVisible(True)
        else:
            self.brush_ring.setVisible(False)

    # ---------- 修饰类工具的预览 ----------

    def _show_gradient_line(self, start, end):
        self.grad_line.setLine(QLineF(start, end))
        self.grad_line.setPen(
            QPen(QColor(255, 255, 255, 220), 1.4 / max(0.05, self.zoom)))
        self.grad_line.setVisible(True)

    def _hide_gradient_line(self):
        self.grad_line.setVisible(False)

    # ---------- 向导：标尺 / 参考线 / 网格（第二十一批）----------

    RULER = 18              # 标尺厚度（视口像素，不随缩放变）

    def ruler_rect(self):
        """标尺占的两条带（左 + 上）。没开就返回 None。"""
        if not self.show_rulers:
            return None
        w = self.viewport().width()
        h = self.viewport().height()
        r = self.RULER
        return (0, 0, r, h), (r, 0, w, r)

    def in_ruler(self, x, y):
        """视口坐标是否落在标尺带里。"""
        rr = self.ruler_rect()
        if rr is None:
            return None
        (lx, ly, lw, lh), (tx, ty, tw, th) = rr
        if lx <= x < lx + lw and ly <= y < ly + lh:
            return _guides.H_GUIDE
        if tx <= x < tx + tw and ty <= y < ty + th:
            return _guides.V_GUIDE
        return None

    def guides(self):
        return self.main.doc.guides if self.main.doc is not None else None

    def rebuild_guide_items(self):
        """按doc.guides 重建参考线图元。只在参考线集合变了之后调。"""
        for it in self._guide_items:
            self.scene.removeItem(it)
        self._guide_items = []
        gs = self.guides()
        doc = self.main.doc
        if gs is None or doc is None:
            return
        w, h = float(doc.width), float(doc.height)
        for g in gs.guides:
            it = QGraphicsLineItem()
            it.setData(0, g.id)
            if g.is_horizontal:
                it.setLine(0.0, g.pos, w, g.pos)
            else:
                it.setLine(g.pos, 0.0, g.pos, h)
            col = QColor(*g.color)
            col.setAlpha(220)
            it.setPen(QPen(col, 1.0))
            it.setZValue(30)
            it.setFlag(QGraphicsItem.ItemIsSelectable, False)
            self.scene.addItem(it)
            self._guide_items.append(it)
        self.viewport().update()

    def sync_grid(self):
        """网格的显示开关跟着 doc 走（可能被工程文件带进来）。"""
        gs = self.guides()
        if gs is None:
            return
        self.show_grid = gs.grid.visible
        self.grid_spacing = gs.grid.spacing
        self.grid_subdiv = gs.grid.subdiv
        self.snap_enabled = (gs.snap_guides or gs.snap_grid
                             or gs.snap_layers or gs.snap_doc)
        self.viewport().update()

    def snap_tolerance(self):
        """吸附容差（**视口**像素）—— 屏幕上 7 px 换算成世界坐标。"""
        return 7.0 / max(self.zoom, 1e-6)

    def apply_snap(self, x, y, exclude_layer=None):
        """把画布坐标 (x,y) 吸到最近的吸附线上。

        返回 (sx, sy, hit)。`hit` 与 `guides.snap_point` 的第三项同构。
        吸附总开关关掉、或没开任何一种源时原样返回。
        """
        gs = self.guides()
        doc = self.main.doc
        if gs is None or doc is None:
            return x, y, []
        if not (gs.snap_guides or gs.snap_grid or gs.snap_layers or gs.snap_doc):
            return x, y, []
        layers = doc.layers if gs.snap_layers else ()
        return _guides.snap_point(x, y, gs, doc.width, doc.height,
                                 layers=layers, tol=self.snap_tolerance(),
                                 exclude=exclude_layer)

    # ---------- 标尺 / 网格 / 吸附高亮的绘制 ----------

    def drawForeground(self, painter, rect):
        # 参考线本身是图元（能被拖动），这里只画**标尺、网格、吸附高亮**
        super().drawForeground(painter, rect)
        if self.main.doc is None:
            return
        painter.save()
        try:
            if self.show_grid:
                self._paint_grid(painter, rect)
            self._paint_snap_lines(painter)
        finally:
            painter.restore()
        rr = self.ruler_rect()
        if rr is not None:
            self._paint_rulers(painter, rr)

    def _scene_rect(self, rect):
        """把视口矩形换成场景坐标（考虑滚动与缩放）。"""
        tl = self.mapToScene(0, 0)
        br = self.mapToScene(self.viewport().width(),
                             self.viewport().height())
        return QRectF(tl, br)

    def _paint_grid(self, painter, rect):
        doc = self.main.doc
        gs = self.guides()
        if doc is None or gs is None:
            return
        spec = _guides.GridSpec(self.show_grid, self.grid_spacing,
                               self.grid_subdiv)
        xs, ys = spec.lines(doc.width, doc.height)
        if not xs and not ys:
            return
        z = 1.0 / max(self.zoom, 1e-6)
        col = QColor(120, 140, 170, 90)
        col.setAlpha(90)
        painter.setPen(QPen(col, z))
        w, h = float(doc.width), float(doc.height)
        for x in xs:
            painter.drawLine(QPointF(x, 0.0), QPointF(x, h))
        for y in ys:
            painter.drawLine(QPointF(0.0, y), QPointF(w, y))

    def _paint_snap_lines(self, painter):
        """拖动时把命中的那些线画成醒目的洋红色（PS 的做法）。

        `self._snap_lines` 的元素是 (kind, axis, pos) —— **pos 是命中后
        那个坐标轴上的实际位置**。只给 (kind, axis) 是不够的：参考线可能
        有好几条，图层也有好几条边，不带坐标就只能瞎画一条（曾画出斜线）。
        """
        if not self._snap_lines:
            return
        doc = self.main.doc
        if doc is None:
            return
        z = 1.0 / max(self.zoom, 1e-6)
        painter.setPen(QPen(QColor(255, 60, 190), 1.6 * z))
        w, h = float(doc.width), float(doc.height)
        for item in self._snap_lines:
            if len(item) == 3:
                k, axis, pos = item
            else:
                k, axis = item
                pos = None
            if pos is None:
                continue
            if axis == "x":
                painter.drawLine(QPointF(pos, 0.0), QPointF(pos, h))
            else:
                painter.drawLine(QPointF(0.0, pos), QPointF(w, pos))

    def _paint_rulers(self, painter, rr):
        """画左上两条标尺带。刻度密度按当前缩放自适应（见 guides.ruler_ticks）。"""
        (lx, ly, lw, lh), (tx, ty, tw, th) = rr
        r = self.RULER
        painter.fillRect(QRect(lx, ly, lw, lh), QColor(58, 58, 62))
        painter.fillRect(QRect(tx, ty, tw, th), QColor(58, 58, 62))
        painter.setPen(QPen(QColor(90, 90, 96), 1.0))
        painter.drawLine(lx + lw - 1, ly, lx + lw - 1, ly + lh)
        painter.drawLine(tx, ty + th - 1, tx + tw, ty + th - 1)

        z = max(self.zoom, 1e-6)
        # 标尺只覆盖画布区域之外的部分，但刻度从 0 开始对齐画布
        scene = self._scene_rect(None)
        x0, y0 = scene.left(), scene.top()
        x1, y1 = scene.right(), scene.bottom()
        # 顶部标尺 -> 水平刻度；左侧标尺 -> 垂直刻度
        step_world = 1.0 / z                # 一个屏幕像素对应多少世界坐标
        ticks, major = _guides.ruler_ticks(x0, x1, step_world, 58.0)
        painter.setPen(QPen(QColor(150, 150, 156), 1.0))
        fm = painter.font()
        try:
            fm.setPointSize(8)
        except Exception:
            pass
        painter.setFont(fm)
        for i, v in enumerate(ticks):
            sx = int(self.mapFromScene(QPointF(v, 0.0)).x())
            if sx < tx or sx > tx + tw:
                continue
            ln = 8 if i in major else 4
            painter.drawLine(sx, ty + th - 1, sx, ty + th - 1 - ln)
            if i in major:
                painter.drawText(sx + 2, ty + 9, str(int(round(v))))
        ticks, major = _guides.ruler_ticks(y0, y1, step_world, 58.0)
        for i, v in enumerate(ticks):
            sy = int(self.mapFromScene(QPointF(0.0, v)).y())
            if sy < ly or sy > ly + lh:
                continue
            ln = 8 if i in major else 4
            painter.drawLine(lx + lw - 1, sy, lx + lw - 1 - ln, sy)
            if i in major:
                painter.save()
                painter.translate(lx + 9, sy - 2)
                painter.rotate(-90)
                painter.drawText(0, 0, str(int(round(v))))
                painter.restore()

    def _snap_layer_move(self, nx, ny, layer):
        """移动图层时的吸附：对**中心 + 四条边**各试一次，取命中里最近的那个。

        返回 (nx', ny', hits)。`hits` 里的元素是 (kind, axis)，直接给
        `_paint_snap_lines` 用。旋转过的图层用变换后的实际边位置（见下）。

        为什么不用旋转后的四角：转45° 的矩形四边不与坐标轴平行，
        「边上的点吸到竖线」在视觉上说不清是哪一条 —— PS 的做法也是
        只对**包围盒**（旋转后的 AABB）做吸附。所以这里按 AABB 算。
        """
        hits = []
        bx = by = 0.0
        found_x = False
        found_y = False
        best_x = None          # (kind, delta, dist)
        best_y = None
        vs, hs = _guides.layer_edges_and_centers(layer)
        # layer_edges 给的是**当前** tx/ty 下的位置；我们要的是**移动后**的位置，
        # 所以整体加上 (nx - layer.tx, ny - layer.ty)
        ox = nx - float(layer.tx)
        oy = ny - float(layer.ty)
        tol = self.snap_tolerance()

        def _better(cur, kind, delta, dist):
            """**先比优先级（kind 越小越高），同级再比距离**。

            只按距离挑会让「某个候选恰好也离另一条线很近」时输给了次优先的源
            —— 与 PS 不符。PS 是「参考线优先于网格优先于图层」。
            """
            if cur is None:
                return True
            if kind != cur[0]:
                return kind < cur[0]
            return dist < cur[2]

        if vs:
            for c in vs:
                cand = c + ox
                sx, _y, h = self.apply_snap(cand, 0.0, exclude_layer=layer)
                for k, axis in h:
                    if axis != "x":
                        continue
                    d = abs(sx - cand)
                    if d <= tol and _better(best_x, k, sx - cand, d):
                        best_x = (k, sx - cand, d, sx)
            if best_x is not None:
                bx = best_x[1]
                found_x = True
                hits = [(best_x[0], "x", best_x[3])]
            if found_x:
                nx += bx
        if hs:
            for c in hs:
                cand = c + oy
                _x, sy, h = self.apply_snap(0.0, cand, exclude_layer=layer)
                for k, axis in h:
                    if axis != "y":
                        continue
                    d = abs(sy - cand)
                    if d <= tol and _better(best_y, k, sy - cand, d):
                        best_y = (k, sy - cand, d, sy)
            if best_y is not None:
                by = best_y[1]
                found_y = True
                hits = hits + [(best_y[0], "y", best_y[3])]
            if found_y:
                ny += by
        return nx, ny, hits

    def guide_hit(self, scene_pt, tol_view=5.0):
        """场景坐标附近有没有参考线。返回 Guide 或 None。"""
        gs = self.guides()
        if gs is None:
            return None
        tol = tol_view / max(self.zoom, 1e-6)
        x, y = float(scene_pt.x()), float(scene_pt.y())
        best = None
        for g in gs.guides:
            d = abs(y - g.pos) if g.is_horizontal else abs(x - g.pos)
            if d <= tol and (best is None or d < abs(
                    (y - best.pos) if best.is_horizontal else (x - best.pos))):
                best = g
        return best

    def set_snap_lines(self, hits):
        """拖动过程中画高亮：把命中的那些线记下来，重绘时高亮显示。"""
        self._snap_lines = list(hits)
        self.viewport().update()

    def set_clone_source(self, snap):
        """main_window 在 Alt 采样后把 CloneSource 交给这里。"""
        self.clone_source = snap
        if snap is None:
            self.clone_ring.setVisible(False)
        else:
            self._clone_ring_rect()

    def _clone_ring_rect(self):
        """把源位置标记挪到当前采样点。"""
        s = self.clone_source
        if s is None:
            self.clone_ring.setVisible(False)
            return
        r = 6.0
        self.clone_ring.setRect(QRectF(s.ox - r, s.oy - r, r * 2, r * 2))
        self.clone_ring.setVisible(True)

    def _update_shape_preview(self, p):
        s = self._shape_start
        if s is None:
            return
        self.shape_preview.setRect(QRectF(
            min(s.x(), p.x()), min(s.y(), p.y()),
            abs(p.x() - s.x()), abs(p.y() - s.y())))
        self.shape_preview.setVisible(True)

    def _blur_hit(self, p):
        """模糊工具落一次。参数从工具选项条读，坐标是画布坐标。"""
        if self.main.blur_at(p.x(), p.y()):
            self._update_brush_ring(p)

    def _hit_handle(self, scene_pos):
        layer = self._handle_layer()
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

    # ================= 画布内编辑文字（Ctrl+T） =================

    # 字号必须写进**控件自己的样式表**：主窗口那份 `QWidget{font-size:12px}`
    # 优先级高于 setFont()，只用 setFont 的话字永远是最小号（踩过，见 §4.3 第 27 条）
    EDIT_STYLE = ("QPlainTextEdit{background: rgba(122,162,247,28);"
                  " border: 1px dashed #7aa2f7; color: #f0f0f0;"
                  " padding: 2px; font-family: %s; font-size: %dpx;"
                  " %s %s}")

    def begin_text_edit(self, layer):
        """在选中的文字图层上叠一个输入框，边打边看。

        只改 `layer.text["content"]`（非破坏性），文字依旧可以再改 / 再换字体。
        Esc、点到别处、换工具都会收起来。
        """
        if layer is None or not layer.is_text or layer.text is None:
            return False
        self.end_text_edit()
        ed = QPlainTextEdit(self.viewport())
        ed.setPlainText(str(layer.text.get("content", "")))
        ed.setFrameShape(QFrame.NoFrame)
        ed.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        ed.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        ed.textChanged.connect(self._on_text_edit_changed)
        ed.installEventFilter(self)
        self._text_edit = ed
        self._text_edit_layer = layer
        self._text_edit_style = None
        self._text_edit_busy = False
        self._place_text_edit()
        ed.show()
        ed.setFocus()
        ed.selectAll()
        return True

    def edit_text_layer(self):
        """正在画布上编辑的那个文字图层（没在编辑就返回 None）。"""
        return self._text_edit_layer if self._text_edit is not None else None

    def end_text_edit(self):
        """收起输入框（内容已经实时写进图层了，这里只收尾）。"""
        ed = self._text_edit
        self._text_edit = None
        self._text_edit_layer = None
        self._text_edit_style = None
        if ed is None:
            return False
        try:
            ed.textChanged.disconnect(self._on_text_edit_changed)
        except (RuntimeError, TypeError):
            pass
        ed.removeEventFilter(self)
        ed.hide()
        ed.deleteLater()
        self.setFocus()
        return True

    def _on_text_edit_changed(self):
        layer = self._text_edit_layer
        if layer is None or self._text_edit is None or self._text_edit_busy:
            return
        # 编辑期间换了选中图层就收起来 —— 否则输入的内容会被写进另一个图层
        if self.main.selected_layer() is not layer:
            self.end_text_edit()
            return
        self._apply_edit_line_height()
        self.main.set_text_param("content", self._text_edit.toPlainText())

    def _place_text_edit(self):
        """把输入框摆到文字图层在屏幕上的位置，字号也跟着缩放。"""
        ed = self._text_edit
        layer = self._text_edit_layer
        if ed is None or layer is None:
            return
        c = layer.corners()
        if c is None:
            return
        xs = [self.mapFromScene(QPointF(float(x), float(y))).x() for x, y in c]
        ys = [self.mapFromScene(QPointF(float(x), float(y))).y() for x, y in c]
        x0, y0 = min(xs) - 4.0, min(ys) - 4.0
        w = max(max(xs) - min(xs) + 8.0, 220.0)
        h = max(max(ys) - min(ys) + 8.0, 60.0)
        ed.setGeometry(int(round(x0)), int(round(y0)),
                       int(round(w)), int(round(h)))
        fam = str(layer.text.get("family") or "").strip()
        px = int(max(8, min(300, round(
            float(layer.text.get("size", 96.0)) * self.zoom))))
        css_fam = '"%s"' % fam if fam else '"Microsoft YaHei UI", "Segoe UI"'
        style = self.EDIT_STYLE % (
            css_fam, px,
            "font-weight:600;" if layer.text.get("bold") else "",
            "font-style:italic;" if layer.text.get("italic") else "")
        if style != self._text_edit_style:      # 一样就别重复设，省一次样式重算
            ed.setStyleSheet(style)
            self._text_edit_style = style
        f = QFont()
        if fam:
            f.setFamily(fam)
        f.setPixelSize(px)
        f.setBold(bool(layer.text.get("bold")))
        f.setItalic(bool(layer.text.get("italic")))
        ed.setFont(f)
        self._apply_edit_line_height()

    def _apply_edit_line_height(self):
        """让输入框的行距跟文字图层的 line_height 一致，才像"写在画布上"。

        行距不是 CSS 能设的东西，只能用块格式（比例行高）。每次改内容后
        新块会退回默认值，所以要在内容变化时重来一遍。

        注意：mergeBlockFormat() 自己会再发一次 textChanged，不挡住就会
        无限递归（踩过，见 §4.3 第 27 条）。
        """
        ed = self._text_edit
        if ed is None or self._text_edit_layer is None or self._text_edit_busy:
            return
        try:
            lh = float(self._text_edit_layer.text.get("line_height", 1.2))
        except (TypeError, ValueError):
            lh = 1.2
        fmt = QTextBlockFormat()
        # 注意：PySide6 这里要的是 int 而不是枚举对象本身
        fmt.setLineHeight(max(50.0, min(500.0, lh * 100.0)),
                          int(QTextBlockFormat.ProportionalHeight.value))
        self._text_edit_busy = True
        try:
            cur = ed.textCursor()
            cur.select(QTextCursor.Document)
            cur.mergeBlockFormat(fmt)
            cur.clearSelection()
            ed.setTextCursor(cur)
        finally:
            self._text_edit_busy = False

    def eventFilter(self, obj, event):
        """输入框里按 Esc / 失去焦点就收起来。"""
        if obj is self._text_edit:
            et = event.type()
            if et == QEvent.Type.KeyPress and event.key() == Qt.Key_Escape:
                self.end_text_edit()
                return True
            if et == QEvent.Type.FocusOut:
                self.end_text_edit()
                return False
        return super().eventFilter(obj, event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._place_text_edit()

    def scrollContentsBy(self, dx, dy):
        super().scrollContentsBy(dx, dy)
        self._place_text_edit()

    def cancel_operation(self):
        """Esc：取消进行中的套索/多边形/选区。"""
        if self.end_text_edit():
            return True
        if self._stroke is not None:
            self._air_timer.stop()
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
        doc = self.main.doc
        if doc is not None and quick_mask.is_on(doc):
            # 快速蒙版模式：画笔涂的就是那张遮罩，跟图层没关系。
            # 黑 = 排除出选区，白 = 纳入选区，灰 = 半选中
            opts = self.main.opts
            gray = int(round(_gray(opts.fg.rgb() if self.tool == "brush"
                                   else opts.bg.rgb())))
            st = Stroke(doc=doc, layer=None, tool="brush",
                        size=opts.size, hardness=opts.hardness,
                        opacity=opts.opacity, flow=opts.flow,
                        smoothing=opts.smoothing, target="quick",
                        color=(gray, gray, gray),
                        **brush_dialog.stroke_kwargs(opts.brush))
            if not st.begin((p.x(), p.y())):
                return False
            self._stroke = st
            self._start_airbrush(st)
            self.main.request_render()
            return True

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
                    color=opts.fg.rgb(),
                    **brush_dialog.stroke_kwargs(opts.brush))
        if not st.begin((p.x(), p.y())):
            self.statusMessage.emit("无法在该图层绘制")
            return False
        self._stroke = st
        self._start_airbrush(st)
        self.main.request_render()
        return True

    # ---------- 喷枪 ----------

    def _start_airbrush(self, st):
        """喷枪开启时，按住不动也要按固定频率补笔（PS 的喷枪就是这样）。"""
        self._air_timer.stop()
        if not getattr(st, "airbrush", False):
            return
        rate = float((self.main.opts.brush or {}).get("air_rate", 12.0))
        self._air_timer.start(int(max(16, round(1000.0 / max(1.0, rate)))))

    def _airbrush_tick(self):
        st = self._stroke
        if st is None:
            self._air_timer.stop()
            return
        d = st.airbrush_tick()
        if d is None:
            return
        if st.target == "quick":
            self.update_selection_overlay()
        else:
            layer = self.main.selected_layer()
            pad = int(effect_padding(layer.effects)) + 2 if layer else 2
            self.main.request_render(st.take_dirty(pad))

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

        vp = event.position().toPoint()
        # ---- 标尺带：拖出一条新参考线（移动工具之外都能拖）----
        rk = self.in_ruler(vp.x(), vp.y())
        if rk is not None:
            self._drag_guides = {"kind": rk, "start": vp}
            self.grad_line.setPen(QPen(QColor(255, 220, 120), 1.4))
            self.grad_line.setLine(QLineF(vp, vp))
            self.grad_line.setVisible(True)
            event.accept()
            return

        p = self.mapToScene(vp)
        t = self.tool

        # ---- 已有参考线：靠近就拖它（Alt 删）----
        gh = self.guide_hit(p)
        if gh is not None and self.tool == "move":
            if event.modifiers() & Qt.AltModifier:
                self.main.delete_guide(gh.id)
                event.accept()
                return
            self._drag = {"role": "guide", "start": p, "guide": gh,
                          "pos0": gh.pos, "moved": False}
            event.accept()
            return

        # 建新选区 = 放弃浮动状态，先把浮动内容盖回去；
        # 在快速蒙版模式下画选框等于"退出快速蒙版，重新选"
        if t in ("rect", "ellipse", "lasso", "polygon", "magic"):
            self.main.stamp_float(silent=True)
            if quick_mask.is_on(self.main.doc):
                self.main.toggle_quick_mask()

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
        if t == "picker":
            self.main.pick_at(p.x(), p.y())
            event.accept()
            return
        if t == "gradient":
            self._drag = {"role": "grad", "start": p}
            self._show_gradient_line(p, p)
            event.accept()
            return
        if t == "clone":
            if event.modifiers() & Qt.AltModifier:
                self.main.clone_sample(p.x(), p.y())
                event.accept()
                return
            if self.clone_source is None:
                self.statusMessage.emit("克隆图章：先按住 Alt 点一下采样")
                event.accept()
                return
            self._clone_drag = [QPointF(p)]
            self.main.clone_stamp(p.x(), p.y())
            event.accept()
            return
        if t == "heal":
            self._heal_drag = [QPointF(p)]
            self.main.heal_at(p.x(), p.y())
            event.accept()
            return
        if t == "shape":
            self._shape_start = QPointF(p)
            self._update_shape_preview(p)
            event.accept()
            return
        if t == "blurtool":
            # 按一下就是一次模糊；按住拖动一路抹过去（和画笔手感一致）
            self._blur_drag = [QPointF(p)]
            self._blur_hit(p)
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
        layer = self._handle_layer()      # 有浮动层时手柄是它的
        if layer is None or layer.locked:
            super().mousePressEvent(event)
            return
        role = self._hit_handle(p)
        if role:
            hl = self._handle_layer()      # 有浮动层时手柄是它的
            self._drag = {"role": role, "start": p, "layer": hl,
                          "sx": hl.sx, "sy": hl.sy, "rot": hl.rot,
                          "tx": hl.tx, "ty": hl.ty,
                          "a0": math.atan2(p.y() - hl.ty, p.x() - hl.tx),
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
        vp_move = lambda e: e.position().toPoint()
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
            if self._stroke.target == "quick":
                # 快速蒙版不是图层像素，不必重渲染，只刷红罩。
                # 限流：每帧重建整张 12 MP 的 RGBA 太贵，8 fps 够看了
                self._stroke.take_dirty(0)
                now = time.perf_counter()
                if now - self._qm_tick > 0.12:
                    self._qm_tick = now
                    self.update_selection_overlay()
            else:
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

        if self._drag_guides is not None:
            kind = self._drag_guides["kind"]
            sp = self.mapToScene(vp_move(event))
            cur = float(sp.y()) if kind == _guides.H_GUIDE else float(sp.x())
            # 新参考线本身也吸一下（吸到已有参考线 / 网格上，PS 就是这样）
            if kind == _guides.H_GUIDE:
                cur, _y, _h = self.apply_snap(0.0, cur)
            else:
                cur, _x, _h = self.apply_snap(cur, 0.0)
            self._drag_guides["cur"] = cur
            w = float(self.main.doc.width)
            h = float(self.main.doc.height)
            if kind == _guides.H_GUIDE:
                self.grad_line.setLine(QLineF(QPointF(0, cur),
                                              QPointF(w, cur)))
            else:
                self.grad_line.setLine(QLineF(QPointF(cur, 0),
                                              QPointF(cur, h)))
            event.accept()
            return

        if self._clone_drag is not None:
            last = self._clone_drag[-1]
            dist = math.hypot(p.x() - last.x(), p.y() - last.y())
            n = int(min(24, max(1, dist / max(1.0,
                                             self.main.opts.size / 3.0)) + 1))
            for k in range(1, n + 1):
                u = k / float(n)
                self.main.clone_stamp(last.x() + (p.x() - last.x()) * u,
                                      last.y() + (p.y() - last.y()) * u)
            self._clone_drag.append(QPointF(p))
            event.accept()
            return

        if self._heal_drag is not None:
            # 补样间距用 size/2（比克隆粗：修复本来就是一片一片的）
            last = self._heal_drag[-1]
            dist = math.hypot(p.x() - last.x(), p.y() - last.y())
            n = int(min(24, max(1, dist / max(1.0,
                                             self.main.opts.size / 2.0)) + 1))
            for k in range(1, n + 1):
                u = k / float(n)
                self.main.heal_at(last.x() + (p.x() - last.x()) * u,
                                  last.y() + (p.y() - last.y()) * u)
            self._heal_drag.append(QPointF(p))
            event.accept()
            return

        if self._shape_start is not None:
            self._update_shape_preview(p)
            event.accept()
            return

        if self._blur_drag is not None:
            # 落点之间补线，否则快速拖动会漏成一串珠子
            last = self._blur_drag[-1]
            dist = math.hypot(p.x() - last.x(), p.y() - last.y())
            n = int(min(24, max(1, dist / max(1.0,
                                             self.main.opts.size / 3.0)) + 1))
            for k in range(1, n + 1):
                u = k / float(n)
                self._blur_hit(QPointF(last.x() + (p.x() - last.x()) * u,
                                       last.y() + (p.y() - last.y()) * u))
            self._blur_drag.append(QPointF(p))
            event.accept()
            return

        if self._drag is not None:
            if self._drag.get("role") == "grad":
                self._show_gradient_line(self._drag["start"], p)
                event.accept()
                return
            self._apply_drag(p, event.modifiers())
            event.accept()
            return

        if self.tool in BLUR_TOOLS or self.tool in CLONE_TOOLS:
            self.setCursor(Qt.CrossCursor)
            self._update_brush_ring(p)
            if self.tool in CLONE_TOOLS:
                self._clone_ring_rect()
            super().mouseMoveEvent(event)
            return

        if self.tool in SHAPE_TOOLS:
            self.setCursor(Qt.CrossCursor)
            super().mouseMoveEvent(event)
            return

        if self.tool in PAINT_TOOLS:
            self.setCursor(Qt.CrossCursor)
            self._update_brush_ring(p)
            super().mouseMoveEvent(event)
            return

        if self.tool in ("text", "gradient", "picker"):
            self.setCursor(Qt.CrossCursor)
            super().mouseMoveEvent(event)
            return

        # 仅更新光标
        layer = self._handle_layer()
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
            quick = self._stroke.target == "quick"
            self._air_timer.stop()
            self._stroke.end()
            self._stroke = None
            if quick:
                self.main.commit("编辑快速蒙版")
                self.update_selection_overlay()
            else:
                self.main.after_pixel_edit()
            event.accept()
            return

        if self._drag_guides is not None:
            cur = self._drag_guides.get("cur")
            kind = self._drag_guides["kind"]
            self._drag_guides = None
            self.grad_line.setVisible(False)
            if cur is not None and self.main.doc is not None:
                self.main.add_guide(kind, cur)
            event.accept()
            return

        if self._drag is not None and self._drag.get("role") == "guide":
            g = self._drag["guide"]
            newpos = self._drag.get("newpos")
            self._drag = None
            if newpos is not None:
                self.main.move_guide(g.id, newpos)
            event.accept()
            return

        if self._clone_drag is not None:
            # clone_stamp 每次都已经 commit 过了，这里只清轨迹
            self._clone_drag = None
            event.accept()
            return

        if self._heal_drag is not None:
            # heal_at 每次都已经 commit 过了，这里只清轨迹
            self._heal_drag = None
            event.accept()
            return

        if self._shape_start is not None:
            st = self._shape_start
            self._shape_start = None
            self._hide_shape_preview()
            self.main.draw_shape((st.x(), st.y()), (p.x(), p.y()),
                                 modifiers=event.modifiers())
            event.accept()
            return

        if self._blur_drag is not None:
            # blur_at 每次都已经 commit 过了，这里只清轨迹
            self._blur_drag = None
            event.accept()
            return

        if self._drag is not None and self._drag.get("role") == "grad":
            st = self._drag["start"]
            self._drag = None
            self._hide_gradient_line()
            self.main.apply_gradient_drag((st.x(), st.y()), (p.x(), p.y()))
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
        d["moved"] = True

        if d["role"] == "guide":
            # 拖参考线：跟着鼠标走，松开才落（落点走吸附）
            g = d["guide"]
            raw = float(p.y()) if g.is_horizontal else float(p.x())
            if g.is_horizontal:
                np_, _y, hits = self.apply_snap(0.0, raw)
            else:
                np_, _x, hits = self.apply_snap(raw, 0.0)
            d["newpos"] = np_
            self.set_snap_lines(hits)
            g.pos = np_                    # 先动，松手由 main_window 落撤销点
            self.rebuild_guide_items()
            return

        layer = d["layer"]
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
            nx = d["tx"] + dx
            ny = d["ty"] + dy
            # 吸附（Ctrl 临时关闭，PS 就是这个键）。层自身不参与 ——
            # 否则它自己的边缘会把自己吸住不动
            if not (modifiers & Qt.ControlModifier):
                nx, ny, hits = self._snap_layer_move(nx, ny, layer)
                self.set_snap_lines(hits)
            layer.tx = nx
            layer.ty = ny
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

        if modifiers & Qt.AltModifier:
            # Alt = 以中心为基准缩放（PS 里 Alt 拖手柄就是两边一起长）
            layer.tx = d["tx"]
            layer.ty = d["ty"]
        else:
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
        if layer is not getattr(self.main.doc, "float_layer", None):
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
