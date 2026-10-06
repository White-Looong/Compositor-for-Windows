# -*- coding: utf-8 -*-
"""调整层的参数面板。

控件是**按参数表自动生成**的（core.adjust 里的 Param 描述），
所以新增一种调整只要写函数 + 参数表，不用碰这里的界面代码。

三块东西：
  * HistogramView —— 合成结果的直方图（色阶 / 曲线 / 曝光用它取参考）
  * CurveEditor   —— 可拖控制点的曲线编辑器，单调三次插值，不过冲
  * AdjustPanel   —— 把 Spec 渲染成一行行控件
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (QCheckBox, QColorDialog, QComboBox,
                               QFormLayout, QHBoxLayout, QLabel, QPushButton,
                               QSlider, QVBoxLayout, QWidget)

from ..core.adjust import ADJUSTMENTS, curve_lut, default_params

CHANNELS = ["RGB", "R", "G", "B"]


# ---------------------------------------------------------------- 直方图

class HistogramView(QWidget):
    """合成结果的 RGB + 明度直方图。大图按步长抽样，最多 25 万点。"""

    def __init__(self, main, height=64):
        super().__init__()
        self.main = main
        self.setFixedHeight(height)
        self.setMinimumWidth(120)
        self._bins = None

    def sizeHint(self):
        return QSize(200, 64)

    def refresh(self):
        arr = getattr(self.main, "_last_arr", None)
        if arr is None:
            self._bins = None
            self.update()
            return
        rgb = arr[..., :3]
        h, w = rgb.shape[:2]
        total = max(1, h * w)
        stride = int(np.ceil(np.sqrt(total / 250000.0)))
        flat = rgb[::stride, ::stride].reshape(-1, 3)
        bins = np.zeros((4, 256), np.int64)
        for c in range(3):
            bins[c] = np.bincount(flat[:, c], minlength=256)[:256]
        lum = (0.299 * flat[:, 0].astype(np.float32) +
               0.587 * flat[:, 1].astype(np.float32) +
               0.114 * flat[:, 2].astype(np.float32)).astype(np.uint8)
        bins[3] = np.bincount(lum, minlength=256)[:256]
        self._bins = bins
        self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(QRectF(0, 0, w, h), QColor("#191919"))
        p.setPen(QPen(QColor("#3a3a3d")))
        for i in range(1, 4):
            x = w * i / 4.0
            p.drawLine(QPointF(x, 0), QPointF(x, h))
        bins = self._bins
        if bins is None:
            p.setPen(QColor("#666"))
            p.drawText(QRectF(0, 0, w, h), Qt.AlignCenter, "无图像")
            p.end()
            return

        peak = max(1.0, float(bins[:, 1:255].max()))
        sx = w / 255.0
        sy = (h - 2.0) / peak

        def poly(row):
            pts = QPolygonF()
            b = bins[row]
            for i in range(256):
                pts.append(QPointF(i * sx, h - 1.0 - b[i] * sy))
            return pts

        p.setBrush(QBrush(QColor(150, 150, 150, 90)))
        p.setPen(Qt.NoPen)
        p.drawPolygon(poly(3))
        p.setBrush(Qt.NoBrush)
        for row, col in ((0, QColor(255, 70, 70, 200)),
                         (1, QColor(70, 255, 70, 200)),
                         (2, QColor(90, 130, 255, 200))):
            p.setPen(QPen(col, 1.2))
            p.drawPolyline(poly(row))
        p.end()


# ---------------------------------------------------------------- 曲线编辑器

class CurveEditor(QWidget):
    """可拖拽控制点的曲线编辑器。

    左键点空白处加点，拖点改形状，把点拖出边界删掉（两端点除外）。
    """

    pointsChanged = Signal(object)

    _PAD = 12
    _HIT = 9

    def __init__(self, points=None, size=176):
        super().__init__()
        self.setFixedSize(QSize(size, size))
        self.setMouseTracking(True)
        self.points = [list(q) for q in (points or [[0.0, 0.0], [1.0, 1.0]])]
        self._drag = -1
        self.channel = "RGB"

    def set_points(self, pts):
        self.points = [list(q) for q in pts] if pts else [[0.0, 0.0],
                                                          [1.0, 1.0]]
        self.update()

    # ---------- 坐标 ----------

    def _plot(self):
        s = self.width()
        pad = self._PAD
        return pad, pad, s - 2 * pad, s - 2 * pad

    def _to_px(self, q):
        x0, y0, w, h = self._plot()
        return (x0 + q[0] * w, y0 + (1.0 - q[1]) * h)

    def _from_px(self, pos):
        x0, y0, w, h = self._plot()
        return ((pos.x() - x0) / w, 1.0 - (pos.y() - y0) / h)

    # ---------- 绘制 ----------

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        s = self.width()
        p.fillRect(QRectF(0, 0, s, s), QColor("#191919"))

        x0, y0, w, h = self._plot()
        p.setPen(QPen(QColor("#333336"), 1))
        for i in range(5):
            t = i / 4.0
            p.drawLine(QPointF(x0 + t * w, y0), QPointF(x0 + t * w, y0 + h))
            p.drawLine(QPointF(x0, y0 + t * h), QPointF(x0 + w, y0 + t * h))
        p.setPen(QPen(QColor("#4a4a4e"), 1, Qt.DashLine))
        p.drawLine(QPointF(x0, y0 + h), QPointF(x0 + w, y0))

        lut = curve_lut(self.points)
        poly = QPolygonF()
        for i in range(64):
            v = lut[int(round(i * 255.0 / 63.0))]
            poly.append(QPointF(x0 + (i / 63.0) * w, y0 + (1.0 - v) * h))
        p.setPen(QPen(QColor("#e8e8ea"), 1.8))
        p.drawPolyline(poly)

        p.setPen(QPen(QColor("#2d6db5"), 1.4))
        p.setBrush(QBrush(QColor("#2d6db5")))
        for q in self.points:
            px, py = self._to_px(q)
            p.drawEllipse(QRectF(px - 3.5, py - 3.5, 7.0, 7.0))
        p.end()

    # ---------- 交互 ----------

    def _nearest(self, pos):
        best, bi = self._HIT * self._HIT, -1
        for i, q in enumerate(self.points):
            px, py = self._to_px(q)
            d = (px - pos.x()) ** 2 + (py - pos.y()) ** 2
            if d < best:
                best, bi = d, i
        return bi

    def _clamp_x(self, i, x):
        lo = 0.0 if i == 0 else self.points[i - 1][0] + 0.015
        n = len(self.points)
        hi = 1.0 if i == n - 1 else self.points[i + 1][0] - 0.015
        if i == 0:
            return 0.0
        if i == n - 1:
            return 1.0
        return max(lo, min(hi, x))

    def mousePressEvent(self, e):
        pos = e.position()
        i = self._nearest(pos)
        if i < 0:
            x, y = self._from_px(pos)
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            pts = sorted(self.points + [[x, y]], key=lambda q: q[0])
            self.points = pts
            i = pts.index([x, y])
        self._drag = i
        self._apply_move(pos)

    def mouseMoveEvent(self, e):
        if self._drag < 0:
            return
        self._apply_move(e.position())

    def mouseReleaseEvent(self, _e):
        self._drag = -1

    def _apply_move(self, pos):
        if self._drag < 0:
            return
        i = self._drag
        x, y = self._from_px(pos)
        n = len(self.points)
        # 拖出边界 = 删除该点（两端点保留）
        pad = 26.0
        if (0 < i < n - 1 and
                (pos.x() < -pad or pos.x() > self.width() + pad or
                 pos.y() < -pad or pos.y() > self.height() + pad)):
            self.points.pop(i)
            self._drag = -1
            self.update()
            self.pointsChanged.emit([list(q) for q in self.points])
            return
        self.points[i] = [self._clamp_x(i, x), max(0.0, min(1.0, y))]
        self.update()
        self.pointsChanged.emit([list(q) for q in self.points])


# ---------------------------------------------------------------- 面板

class AdjustPanel(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self._sync = False

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        self.title = QLabel("-")
        self.hist = HistogramView(main)
        self.body = QWidget()
        self.form = QFormLayout(self.body)
        self.form.setContentsMargins(0, 0, 0, 0)
        self.form.setLabelAlignment(Qt.AlignRight)
        self.form.setSpacing(5)
        self.reset_btn = QPushButton("复位为默认值")
        self.reset_btn.clicked.connect(self._reset)

        root.addWidget(self.title)
        root.addWidget(self.hist)
        root.addWidget(self.body)
        root.addWidget(self.reset_btn)

    # ---------- 同步 ----------

    def refresh(self):
        layer = self.main.selected_layer()
        on = bool(layer is not None and layer.is_adjustment)
        self.setVisible(on)
        if not on:
            return
        key = layer.adjustment_key()
        spec = ADJUSTMENTS.get(key)
        self.title.setText("调整层：%s" % (spec.name if spec else key))
        show_hist = bool(spec and spec.histogram)
        self.hist.setVisible(show_hist)
        if show_hist:
            self.hist.refresh()
        self._build(spec, layer)

    def refresh_hist(self):
        """渲染完成后只更新直方图，不重建控件。"""
        if self.isVisible() and self.hist.isVisible():
            self.hist.refresh()

    # ---------- 构建 ----------

    def _clear_form(self):
        while self.form.rowCount():
            self.form.removeRow(0)

    def _build(self, spec, layer):
        self._clear_form()
        if spec is None:
            return
        params = layer.adjust.setdefault("params", {})
        if not params:
            params.update(default_params(spec.key))
            layer.adjust["params"] = params

        self._sync = True
        for prm in spec.params:
            row = self._make_row(prm, params.get(prm.key, prm.default), layer)
            if row is not None:
                self.form.addRow(prm.label, row)
        if not spec.params:
            lab = QLabel("该调整没有可调参数")
            lab.setStyleSheet("color:#888;")
            self.form.addRow(lab)
        self._sync = False

    def _num_row(self, prm, value, cb):
        """滑块 + 数值标签。返回 (widget, set_value)，后者用于外部改值。"""
        row = QWidget()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        s = QSlider(Qt.Horizontal)
        s.setRange(int(round(prm.lo * prm.scale)),
                   int(round(prm.hi * prm.scale)))
        s.setSingleStep(max(1, int(round(prm.step * prm.scale))))
        s.setMinimumWidth(90)
        lab = QLabel()
        lab.setFixedWidth(44)
        lab.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        def fmt(v):
            if prm.kind == "int":
                return "%d" % int(round(v))
            return "%.*f" % (prm.decimals, v)

        def on_change(raw):
            v = raw / float(prm.scale)
            lab.setText(fmt(v))
            cb(int(round(v)) if prm.kind == "int" else v)

        def set_value(v):
            blocked = s.blockSignals(True)     # 外部改值不触发回调
            try:
                s.setValue(int(round(float(v) * prm.scale)))
            except (TypeError, ValueError):
                s.setValue(int(round(prm.default * prm.scale)))
            s.blockSignals(blocked)
            lab.setText(fmt(v))

        s.valueChanged.connect(on_change)
        try:
            set_value(float(value))
        except (TypeError, ValueError):
            set_value(prm.default)
        lay.addWidget(s, 1)
        lay.addWidget(lab)
        return row, set_value

    def _make_row(self, prm, value, layer):
        if prm.kind == "bool":
            cb = QCheckBox()
            cb.setChecked(bool(value))
            cb.stateChanged.connect(
                lambda _s, k=prm.key: self._set(k, bool(cb.isChecked())))
            return cb

        if prm.kind == "color":
            btn = QPushButton()
            btn.setFixedWidth(72)
            cur = [int(x) for x in (value or prm.default or [0, 0, 0])]

            def paint():
                btn.setStyleSheet(
                    "background:#%02x%02x%02x;border:1px solid #666;"
                    % (cur[0], cur[1], cur[2]))

            def pick():
                c = QColorDialog.getColor(QColor(*cur), self, "选择颜色")
                if not c.isValid():
                    return
                cur[:] = [c.red(), c.green(), c.blue()]
                paint()
                self._set(prm.key, list(cur))

            btn.clicked.connect(pick)
            paint()
            return btn

        if prm.kind == "keyed":
            return self._keyed_row(prm, value)

        if prm.kind == "choice":
            combo = QComboBox()
            combo.addItems(prm.choices)
            if value in prm.choices:
                combo.setCurrentText(value)
            combo.currentTextChanged.connect(
                lambda v, k=prm.key: self._set(k, v))
            return combo

        if prm.kind == "curve":
            box = QWidget()
            v = QVBoxLayout(box)
            v.setContentsMargins(0, 0, 0, 0)
            v.setSpacing(4)
            combo = QComboBox()
            combo.addItems(CHANNELS)
            editor = CurveEditor(size=176)
            v.addWidget(combo)
            v.addWidget(editor)

            pts = (value or {}).get("RGB") or []

            def load(ch):
                p = (layer.adjust.get("params", {}).get("points") or {}).get(ch)
                editor.set_points(p or [[0.0, 0.0], [1.0, 1.0]])

            def on_pts(newpts):
                d = layer.adjust["params"].setdefault("points", {})
                d[combo.currentText()] = [list(q) for q in newpts]
                self._changed()

            combo.currentTextChanged.connect(load)
            editor.pointsChanged.connect(on_pts)
            editor.set_points(pts)
            return box

        # int / double：滑块 + 数值
        row, _setter = self._num_row(
            prm, value,
            lambda v: self._set(prm.key, v))
        return row

    def _keyed_row(self, prm, value):
        """分组数值：下拉框选组 + 一组滑块。值形如 {组名: {子键: 值}}。"""
        box = QWidget()
        v = QVBoxLayout(box)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)

        combo = QComboBox()
        combo.addItems(prm.keys)
        v.addWidget(combo)

        setters = {}
        for sp in prm.sub:
            def make_cb(sp=sp):
                def cb(val):
                    if self._sync:
                        return
                    layer = self.main.selected_layer()
                    if layer is None or not layer.is_adjustment:
                        return
                    params = layer.adjust.setdefault("params", {})
                    group = params.setdefault(prm.key, {})
                    sub = group.setdefault(combo.currentText(), {})
                    sub[sp.key] = val
                    self._changed()
                return cb

            row, setter = self._num_row(sp, sp.default, make_cb())
            setters[sp.key] = setter
            cell = QWidget()
            h = QHBoxLayout(cell)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(4)
            lab = QLabel(sp.label)
            lab.setFixedWidth(26)
            h.addWidget(lab)
            h.addWidget(row, 1)
            v.addWidget(cell)

        def load(name):
            layer = self.main.selected_layer()
            d = {}
            if layer is not None and layer.is_adjustment:
                d = ((layer.adjust.get("params", {}).get(prm.key) or {})
                     .get(name) or {})
            self._sync = True
            for sp in prm.sub:
                setters[sp.key](d.get(sp.key, sp.default))
            self._sync = False

        combo.currentTextChanged.connect(load)
        load(prm.keys[0] if prm.keys else "")
        return box

    # ---------- 写回 ----------

    def _set(self, key, value):
        if self._sync:
            return
        layer = self.main.selected_layer()
        if layer is None or not layer.is_adjustment:
            return
        layer.adjust.setdefault("params", {})[key] = value
        self._changed()

    def _changed(self):
        layer = self.main.selected_layer()
        self.main.begin_interactive()      # 拖滑块期间走代理，停手后自动补全分辨率
        # 带上这个调整层：只有改调整参数能复用"它下方"的合成结果（快路径）
        self.main.request_render(adjust=layer)
        self.main.schedule_commit("调整参数")

    def _reset(self):
        layer = self.main.selected_layer()
        if layer is None or not layer.is_adjustment:
            return
        key = layer.adjustment_key()
        layer.adjust["params"] = default_params(key)
        self.main.commit("复位调整参数")
        self.main.request_render()
        self.refresh()
