# -*- coding: utf-8 -*-
"""笔刷设置对话框：笔尖形状 / 间隔 / 散布 / 纹理 / 喷枪 / 压感。

工具选项条上只放最常用的几项（大小、硬度、不透明度、流量、平滑、目标），
其余参数都收在这里 —— 和图层样式一个路子，按 `_SCHEMA` 表自动长控件。

右边的**笔迹预览**不是画出来的示意图，而是用一个真的 `Stroke` 在一张大白纸上
走一条正弦曲线，所以散布 / 纹理 / 间隔 / 抖动 长什么样就是实际画出来什么样。

「取消」会把打开前的整份参数还原（和图层样式一致）。
"""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPainter, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog,
                               QDialogButtonBox, QFormLayout, QGroupBox,
                               QHBoxLayout, QLabel, QPushButton, QSlider,
                               QVBoxLayout, QWidget)

from ..core.brush import BRUSH_DEFAULTS, PRESSURE_MODES, SHAPES, TEXTURES
from ..core.document import Document, make_image_layer
from ..core.paint import Stroke

# 参数表：k=参数名  lab=标签  t=控件类型
#   percent  滑块，值按 0~100 存成 0~1（div=100）
#   int      整数滑块
#   choice   下拉框
#   bool     复选框
_SCHEMA = [
    {"k": "shape", "lab": "笔尖形状", "t": "choice", "choices": SHAPES},
    {"k": "roundness", "lab": "圆度", "t": "percent", "lo": 5, "hi": 100,
     "tip": "越小笔尖越扁（椭圆）"},
    {"k": "angle", "lab": "角度", "t": "int", "lo": -180, "hi": 180,
     "tip": "笔尖自身的旋转角度"},
    {"k": "spacing", "lab": "间隔", "t": "percent", "lo": 1, "hi": 500,
     "tip": "相邻两个笔尖之间隔多远 —— 按笔尖直径的百分之几算"},
    {"k": "scatter", "lab": "散布", "t": "percent", "lo": 0, "hi": 1000,
     "tip": "每个笔尖随机偏离笔画中心多远 —— 按半径的百分之几算"},
    {"k": "count", "lab": "数量", "t": "int", "lo": 1, "hi": 10,
     "tip": "每步落几个笔尖（配合散布用）"},
    {"k": "size_jitter", "lab": "大小抖动", "t": "percent", "lo": 0, "hi": 100,
     "tip": "每个笔尖随机缩小多少"},
    {"k": "texture", "lab": "纹理", "t": "choice", "choices": TEXTURES,
     "tip": "图案锚在图像上，不随笔尖游动"},
    {"k": "texture_scale", "lab": "纹理大小", "t": "int", "lo": 4, "hi": 400,
     "tip": "纹理的周期，单位是图像像素"},
    {"k": "texture_depth", "lab": "纹理深浅", "t": "percent", "lo": 0, "hi": 100},
    {"k": "pressure", "lab": "压感", "t": "choice", "choices": PRESSURE_MODES,
     "tip": "鼠标没有笔压，这里是按运笔速度模拟：动得越快越细 / 越淡"},
    {"k": "pressure_amount", "lab": "压感强度", "t": "percent", "lo": 0,
     "hi": 100},
    {"k": "airbrush", "lab": "喷枪", "t": "bool",
     "tip": "按住不动会持续加深（此时不透明度不再是单次描边的上限）"},
    {"k": "air_rate", "lab": "喷枪速率", "t": "int", "lo": 2, "hi": 60,
     "tip": "每秒补几笔"},
]

# 组名 -> 这一组包含哪些参数（纯界面分组）
_GROUPS = [
    ("笔尖", ["shape", "roundness", "angle"]),
    ("笔划", ["spacing", "scatter", "count", "size_jitter"]),
    ("纹理", ["texture", "texture_scale", "texture_depth"]),
    ("动态", ["pressure", "pressure_amount"]),
    ("喷枪", ["airbrush", "air_rate"]),
]


class BrushPreview(QWidget):
    """用真 Stroke 在一个小白纸上走一条正弦曲线，所见即所得。"""

    W, H = 280, 96

    def __init__(self):
        super().__init__()
        self.setFixedHeight(self.H)
        self.setMinimumWidth(self.W)
        self._pm = None
        self.refresh({}, 24.0, 0.6)

    def refresh(self, params, size, hardness):
        p = dict(BRUSH_DEFAULTS)
        p.update(params or {})
        doc = Document(self.W, self.H, "brush-preview")
        base = np.zeros((self.H, self.W, 4), np.uint8)
        base[..., :3] = 250
        base[..., 3] = 255
        lay = make_image_layer("p", base, self.W, self.H)
        lay.tx, lay.ty = self.W / 2.0, self.H / 2.0
        doc.layers.append(lay)
        # 预览固定用固定种子，参数不动画面就不会跳
        st = Stroke(doc=doc, layer=lay, size=float(size),
                    hardness=float(hardness), color=(26, 26, 34),
                    shape=p["shape"], roundness=p["roundness"],
                    angle=p["angle"], spacing=p["spacing"],
                    scatter=p["scatter"], count=p["count"],
                    size_jitter=p["size_jitter"], texture=p["texture"],
                    texture_scale=p["texture_scale"],
                    texture_depth=p["texture_depth"], airbrush=False,
                    pressure=p["pressure"],
                    pressure_amount=p["pressure_amount"], seed=7)
        x0, x1, step = 16, self.W - 16, 5
        cy = self.H / 2.0
        amp = self.H * 0.22
        path = [(float(x), cy + math.sin(x / 26.0) * amp)
                for x in range(x0, x1 + 1, step)]
        if st.begin(path[0]):
            for q in path[1:]:
                st.extend(q)
            st.end()
        arr = np.ascontiguousarray(lay.image)
        img = QImage(arr.data, self.W, self.H, 4 * self.W,
                     QImage.Format_RGBA8888).copy()
        self._pm = QPixmap.fromImage(img)
        self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        if self._pm is not None:
            p.drawPixmap(0, 0, self.width(), self.height(), self._pm)
        else:
            p.fillRect(self.rect(), Qt.white)
        p.end()


class BrushDialog(QDialog):
    """笔刷设置。params 是打开前的副本，「取消」时整份还原。"""

    changed = Signal()

    def __init__(self, parent, params, size=24.0, hardness=0.6, live=None):
        super().__init__(parent)
        self.setWindowTitle("笔刷设置")
        self.setModal(True)
        self._orig = dict(BRUSH_DEFAULTS)
        self._orig.update(params or {})
        self.params = dict(self._orig)
        self.size = float(size)
        self.hardness = float(hardness)
        # live(params)：参数一改就回调（主窗口用它刷新状态栏摘要）
        self._live = live
        self._widgets = {}

        root = QVBoxLayout(self)
        body = QHBoxLayout()
        body.setSpacing(12)
        root.addLayout(body)

        left = QVBoxLayout()
        left.setSpacing(6)
        body.addLayout(left, 1)

        by_key = {}
        for title, keys in _GROUPS:
            g = QGroupBox(title)
            f = QFormLayout(g)
            f.setContentsMargins(8, 6, 8, 6)
            f.setSpacing(4)
            for spec in _SCHEMA:
                if spec["k"] in keys:
                    f.addRow(spec["lab"], self._make(spec))
                    by_key[spec["k"]] = spec
            left.addWidget(g)
        left.addStretch(1)

        right = QVBoxLayout()
        right.setSpacing(6)
        right.addWidget(QLabel("笔迹预览"))
        self.preview = BrushPreview()
        right.addWidget(self.preview)
        self.lab_summary = QLabel()
        self.lab_summary.setWordWrap(True)
        self.lab_summary.setStyleSheet("color:#8a8a8e;")
        right.addWidget(self.lab_summary)
        right.addStretch(1)
        body.addLayout(right)

        btns = QDialogButtonBox()
        self.btn_reset = QPushButton("复位为默认")
        btns.addButton(self.btn_reset, QDialogButtonBox.ResetRole)
        btns.addButton(QDialogButtonBox.Ok)
        btns.addButton(QDialogButtonBox.Cancel)
        self.btn_reset.clicked.connect(self._reset)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

        self._sync_widgets()
        self._refresh_preview()

    # ---------- 控件 ----------

    def _make(self, spec):
        k = spec["k"]
        t = spec["t"]
        tip = spec.get("tip", "")
        if t == "bool":
            w = QCheckBox()
            w.setChecked(bool(self.params.get(k)))
            w.toggled.connect(lambda v, key=k: self._set(key, bool(v)))
            self._widgets[k] = w
            return w
        if t == "choice":
            w = QComboBox()
            w.addItems(spec["choices"])
            if self.params.get(k) in spec["choices"]:
                w.setCurrentText(self.params[k])
            w.currentTextChanged.connect(lambda v, key=k: self._set(key, v))
            self._widgets[k] = w
            return w
        # 数值滑块
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        s = QSlider(Qt.Horizontal)
        s.setRange(spec["lo"], spec["hi"])
        s.setMinimumWidth(110)
        lab = QLabel()
        lab.setFixedWidth(46)
        lab.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        is_pct = t == "percent"
        cur = self.params.get(k, BRUSH_DEFAULTS.get(k, 0))
        raw = int(round(float(cur) * 100)) if is_pct else int(round(float(cur)))
        s.setValue(max(spec["lo"], min(spec["hi"], raw)))
        if tip:
            s.setToolTip(tip)

        def on_change(v, key=k, l=lab, pct=is_pct):
            val = v / 100.0 if pct else int(v)
            l.setText("%d%%" % v if pct else str(int(v)))
            self._set(key, val)

        def on_slide(v, l=lab, pct=is_pct):
            l.setText("%d%%" % v if pct else str(int(v)))

        s.valueChanged.connect(on_slide)
        s.valueChanged.connect(on_change)
        on_slide(s.value())
        lay.addWidget(s, 1)
        lay.addWidget(lab)
        self._widgets[k] = s
        return w

    def _set(self, key, value):
        self.params[key] = value
        self._refresh_preview()

    def _reset(self):
        self.params = dict(BRUSH_DEFAULTS)
        self._sync_widgets()
        self._refresh_preview()

    def _sync_widgets(self):
        """把 self.params 的值写回控件（blockSignals，避免回调打架）。"""
        for k, w in self._widgets.items():
            spec = next((q for q in _SCHEMA if q["k"] == k), None)
            if spec is None:
                continue
            v = self.params.get(k, BRUSH_DEFAULTS.get(k))
            blocked = w.blockSignals(True)
            try:
                if spec["t"] == "bool":
                    w.setChecked(bool(v))
                elif spec["t"] == "choice":
                    if v in spec["choices"]:
                        w.setCurrentText(v)
                else:
                    raw = (int(round(float(v) * 100)) if spec["t"] == "percent"
                           else int(round(float(v))))
                    w.setValue(max(spec["lo"], min(spec["hi"], raw)))
            finally:
                w.blockSignals(blocked)

    # ---------- 预览 / 摘要 ----------

    def _refresh_preview(self):
        self.preview.refresh(self.params, self.size, self.hardness)
        self.lab_summary.setText(summary_text(self.params))
        if self._live is not None:
            self._live(dict(self.params))

    def result_params(self):
        return dict(self.params)

    def original_params(self):
        return dict(self._orig)


def summary_text(params):
    """状态栏 / 提示一行摘要。"""
    p = dict(BRUSH_DEFAULTS)
    p.update(params or {})
    bits = [str(p["shape"])]
    if p["roundness"] < 0.999:
        bits.append("圆度 %d%%" % round(p["roundness"] * 100))
    if abs(p["angle"]) > 0.5:
        bits.append("%d°" % round(p["angle"]))
    if abs(p["spacing"] - BRUSH_DEFAULTS["spacing"]) > 1e-6:
        bits.append("间隔 %d%%" % round(p["spacing"] * 100))
    if p["scatter"] > 0:
        bits.append("散布 %d%%×%d" % (round(p["scatter"] * 100), p["count"]))
    if p["size_jitter"] > 0:
        bits.append("抖动 %d%%" % round(p["size_jitter"] * 100))
    if p["texture"] not in (None, "无"):
        bits.append("纹理 %s" % p["texture"])
    if p["pressure"] != "关":
        bits.append("压感 %s" % p["pressure"])
    if p["airbrush"]:
        bits.append("喷枪 %d/s" % round(p["air_rate"]))
    return " · ".join(bits)


def stroke_kwargs(params):
    """把笔刷参数字典翻译成 Stroke 的关键字参数。"""
    p = dict(BRUSH_DEFAULTS)
    p.update(params or {})
    return {
        "shape": p["shape"], "roundness": float(p["roundness"]),
        "angle": float(p["angle"]), "spacing": float(p["spacing"]),
        "scatter": float(p["scatter"]), "count": int(p["count"]),
        "size_jitter": float(p["size_jitter"]), "texture": p["texture"],
        "texture_scale": float(p["texture_scale"]),
        "texture_depth": float(p["texture_depth"]),
        "airbrush": bool(p["airbrush"]), "pressure": p["pressure"],
        "pressure_amount": float(p["pressure_amount"]),
    }
