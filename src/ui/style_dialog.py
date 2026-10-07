# -*- coding: utf-8 -*-
"""图层样式对话框：10 种效果 + 全局光。

改动是**实时预览**的：直接写回图层的 effects 并触发重渲染，
点「取消」或关窗时把打开前的参数整份还原。

参数表在 core.effects 里；这里只按 `_SCHEMA` 把它画成控件 ——
加新效果时在 core/effects.py 补一条默认参数和一个施加函数，
再在 `_SCHEMA` 里列一行即可，**不需要改界面逻辑**。

每个效果是一个可勾选的 QGroupBox。注意 groupbox 自己始终保持可点
（否则关掉之后就再也勾不回来了），变灰的是里面的 body。
"""

from __future__ import annotations

import copy

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QColorDialog, QComboBox,
                               QDialog, QDialogButtonBox, QFormLayout,
                               QGroupBox, QHBoxLayout, QLabel, QScrollArea,
                               QSlider, QVBoxLayout, QWidget)

from ..core.blend import BLEND_MODES
from ..core.effects import (BEVEL_DIRECTIONS, BEVEL_STYLES, EFFECT_NAMES,
                            EFFECT_ORDER, GLOBAL_LIGHT, GRADIENT_STYLES,
                            PATTERN_KINDS, STROKE_POSITIONS,
                            default_effects, factory_default_effects,
                            normalize_effects, save_style_default)
from .tool_options import ColorSwatch

# 每个效果的控件表。每项是一个 dict：
#   k: 参数名   lab: 标签   t: 控件类型
#   percent / int  -> 滑块，div 是换算倍率（写回时除以它）
#   bool / choice / blend / color
_SCHEMA = {
    "drop_shadow": [
        {"k": "opacity", "lab": "不透明度", "t": "percent", "lo": 0, "hi": 100},
        {"k": "use_global", "lab": "使用全局光", "t": "bool"},
        {"k": "angle", "lab": "角度", "t": "int", "lo": -180, "hi": 180},
        {"k": "distance", "lab": "距离", "t": "int", "lo": 0, "hi": 300},
        {"k": "spread", "lab": "扩展", "t": "int", "lo": 0, "hi": 100},
        {"k": "size", "lab": "大小", "t": "int", "lo": 0, "hi": 250},
        {"k": "color", "lab": "颜色", "t": "color"},
    ],
    "outer_glow": [
        {"k": "opacity", "lab": "不透明度", "t": "percent", "lo": 0, "hi": 100},
        {"k": "spread", "lab": "扩展", "t": "int", "lo": 0, "hi": 100},
        {"k": "size", "lab": "大小", "t": "int", "lo": 0, "hi": 250},
        {"k": "color", "lab": "颜色", "t": "color"},
    ],
    "inner_shadow": [
        {"k": "opacity", "lab": "不透明度", "t": "percent", "lo": 0, "hi": 100},
        {"k": "use_global", "lab": "使用全局光", "t": "bool"},
        {"k": "angle", "lab": "角度", "t": "int", "lo": -180, "hi": 180},
        {"k": "distance", "lab": "距离", "t": "int", "lo": 0, "hi": 300},
        {"k": "spread", "lab": "扩展", "t": "int", "lo": 0, "hi": 100},
        {"k": "size", "lab": "大小", "t": "int", "lo": 0, "hi": 250},
        {"k": "color", "lab": "颜色", "t": "color"},
    ],
    "inner_glow": [
        {"k": "opacity", "lab": "不透明度", "t": "percent", "lo": 0, "hi": 100},
        {"k": "blend", "lab": "混合模式", "t": "blend"},
        {"k": "spread", "lab": "扩展", "t": "int", "lo": 0, "hi": 100},
        {"k": "size", "lab": "大小", "t": "int", "lo": 0, "hi": 250},
        {"k": "color", "lab": "颜色", "t": "color"},
    ],
    "bevel_emboss": [
        {"k": "style", "lab": "样式", "t": "choice", "choices": BEVEL_STYLES},
        {"k": "direction", "lab": "方向", "t": "choice",
         "choices": BEVEL_DIRECTIONS},
        {"k": "depth", "lab": "深度", "t": "int", "lo": 0, "hi": 300,
         "div": 100},
        {"k": "size", "lab": "大小", "t": "int", "lo": 0, "hi": 250},
        {"k": "soften", "lab": "柔化", "t": "int", "lo": 0, "hi": 50},
        {"k": "use_global", "lab": "使用全局光", "t": "bool"},
        {"k": "angle", "lab": "角度", "t": "int", "lo": -180, "hi": 180},
        {"k": "altitude", "lab": "高度", "t": "int", "lo": 0, "hi": 90},
        {"k": "highlight_opacity", "lab": "高光不透明度", "t": "percent",
         "lo": 0, "hi": 100},
        {"k": "highlight_blend", "lab": "高光模式", "t": "blend"},
        {"k": "highlight_color", "lab": "高光颜色", "t": "color"},
        {"k": "shadow_opacity", "lab": "暗部不透明度", "t": "percent",
         "lo": 0, "hi": 100},
        {"k": "shadow_blend", "lab": "暗部模式", "t": "blend"},
        {"k": "shadow_color", "lab": "暗部颜色", "t": "color"},
    ],
    "satin": [
        {"k": "opacity", "lab": "不透明度", "t": "percent", "lo": 0, "hi": 100},
        {"k": "blend", "lab": "混合模式", "t": "blend"},
        {"k": "angle", "lab": "角度", "t": "int", "lo": -180, "hi": 180},
        {"k": "distance", "lab": "距离", "t": "int", "lo": 0, "hi": 300},
        {"k": "size", "lab": "大小", "t": "int", "lo": 0, "hi": 250},
        {"k": "invert", "lab": "反相", "t": "bool"},
        {"k": "color", "lab": "颜色", "t": "color"},
    ],
    "color_overlay": [
        {"k": "opacity", "lab": "不透明度", "t": "percent", "lo": 0, "hi": 100},
        {"k": "blend", "lab": "混合模式", "t": "blend"},
        {"k": "color", "lab": "颜色", "t": "color"},
    ],
    "gradient_overlay": [
        {"k": "opacity", "lab": "不透明度", "t": "percent", "lo": 0, "hi": 100},
        {"k": "blend", "lab": "混合模式", "t": "blend"},
        {"k": "style", "lab": "样式", "t": "choice",
         "choices": GRADIENT_STYLES},
        {"k": "angle", "lab": "角度", "t": "int", "lo": -180, "hi": 180},
        {"k": "color", "lab": "起始色", "t": "color"},
        {"k": "color2", "lab": "结束色", "t": "color"},
    ],
    "pattern_overlay": [
        {"k": "opacity", "lab": "不透明度", "t": "percent", "lo": 0, "hi": 100},
        {"k": "blend", "lab": "混合模式", "t": "blend"},
        {"k": "pattern", "lab": "图案", "t": "choice",
         "choices": PATTERN_KINDS},
        {"k": "scale", "lab": "缩放", "t": "int", "lo": 2, "hi": 400},
        {"k": "color", "lab": "颜色", "t": "color"},
    ],
    "stroke": [
        {"k": "opacity", "lab": "不透明度", "t": "percent", "lo": 0, "hi": 100},
        {"k": "size", "lab": "大小", "t": "int", "lo": 1, "hi": 100},
        {"k": "position", "lab": "位置", "t": "choice",
         "choices": STROKE_POSITIONS},
        {"k": "color", "lab": "颜色", "t": "color"},
    ],
}


class _Slider(QWidget):
    """滑块 + 数值标签。滑块走整数，写回时除以 `div`。"""

    def __init__(self, lo, hi, step, value, div, on_change):
        super().__init__()
        self._on_change = on_change
        self._div = div
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        s = QSlider(Qt.Horizontal)
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setMinimumWidth(120)
        self.lab = QLabel()
        self.lab.setFixedWidth(38)
        self.lab.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        s.valueChanged.connect(self._changed)
        self.slider = s
        lay.addWidget(s, 1)
        lay.addWidget(self.lab)
        self.set_value(value)

    def _changed(self, v):
        self.lab.setText("%d" % int(v))
        self._on_change(float(v) / self._div)

    def set_value(self, v):
        """v 是**参数值**（已被 div 换算过），这里显示成滑块上的整数。"""
        shown = int(round(float(v) * self._div))
        blocked = self.slider.blockSignals(True)
        self.slider.setValue(shown)
        self.slider.blockSignals(blocked)
        self.lab.setText("%d" % shown)


class StyleDialog(QDialog):
    def __init__(self, layer, main, parent=None):
        super().__init__(parent)
        self.layer = layer
        self.main = main
        self.setWindowTitle("图层样式 —— %s" % layer.name)
        self.resize(420, 660)

        self.orig = copy.deepcopy(layer.effects)
        base = default_effects()
        self.effects = normalize_effects(copy.deepcopy(layer.effects)
                                         if layer.effects else base)
        layer.effects = self.effects

        root = QVBoxLayout(self)
        root.setSpacing(8)
        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QScrollArea.NoFrame)
        self.host = QWidget()
        self.vbox = QVBoxLayout(self.host)
        self.vbox.setSpacing(6)
        self.scroll.setWidget(self.host)
        root.addWidget(self.scroll, 1)

        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel |
                               QDialogButtonBox.Reset)
        box.button(QDialogButtonBox.Reset).setText("复位")
        self.btn_save_default = box.addButton("存为默认样式",
                                              QDialogButtonBox.ActionRole)
        self.btn_save_default.clicked.connect(self._save_default)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        box.button(QDialogButtonBox.Reset).clicked.connect(self._reset)
        root.addWidget(box)

        self.groups = {}
        self.bodies = {}
        self._global_widgets = {}
        self._build()

    # ---------- 构建 ----------

    def _clear(self):
        while self.vbox.count():
            item = self.vbox.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

    def _build(self):
        """重建整个表单（`_reset` 也会走这里）。"""
        self._clear()
        self.groups = {}
        self.bodies = {}
        self._global_widgets = {}
        self.vbox.addWidget(self._make_global())
        for key in EFFECT_ORDER:
            self.vbox.addWidget(self._make_group(key))
        self.vbox.addStretch(1)
        self._sync_global()

    def _make_global(self):
        ge = self.effects.get(GLOBAL_LIGHT) or {}
        g = QGroupBox("全局光")
        g.setCheckable(True)
        g.setChecked(bool(ge.get("enabled")))
        g.toggled.connect(self._toggle_global)

        body = QWidget()
        form = QFormLayout(body)
        form.setContentsMargins(0, 0, 0, 0)
        form.setLabelAlignment(Qt.AlignRight)
        form.setSpacing(5)
        for fk, lab, lo, hi in (("angle", "角度", -180, 180),
                                ("altitude", "高度", 0, 90)):
            sl = _Slider(lo, hi, 1, float(ge.get(fk, 0.0)), 1,
                         lambda v, k=fk: self._set_global(k, v))
            form.addRow(lab, sl)
            self._global_widgets[fk] = sl
        tip = QLabel("勾了「使用全局光」的效果都跟着这里的角度走")
        tip.setWordWrap(True)
        tip.setEnabled(False)
        form.addRow("", tip)

        outer = QVBoxLayout(g)
        outer.setContentsMargins(8, 4, 8, 6)
        outer.addWidget(body)
        body.setEnabled(bool(ge.get("enabled")))
        self._global_box = g
        self._global_body = body
        return g

    def _make_group(self, key):
        e = self.effects[key]
        g = QGroupBox(EFFECT_NAMES[key])
        g.setCheckable(True)
        g.setChecked(bool(e.get("enabled")))
        g.toggled.connect(lambda on, k=key: self._toggle(k, on))

        # 注意：变灰的是 body，groupbox 本身要保持可点 ——
        # 否则关掉之后连勾选框都点不了，再也勾不回来。
        body = QWidget()
        form = QFormLayout(body)
        form.setContentsMargins(0, 0, 0, 0)
        form.setLabelAlignment(Qt.AlignRight)
        form.setSpacing(5)

        widgets = []
        for spec in _SCHEMA[key]:
            fk = spec["k"]
            kind = spec.get("t", "int")
            if kind == "bool":
                cb = QCheckBox(spec["lab"])
                cb.setChecked(bool(e.get(fk, False)))
                cb.toggled.connect(
                    lambda v, k=key, f=fk: self._set_raw(k, f, bool(v)))
                form.addRow("", cb)
                widgets.append((fk, cb))
            elif kind == "choice":
                combo = QComboBox()
                choices = spec["choices"]
                combo.addItems(choices)
                cur = str(e.get(fk, choices[0]))
                combo.setCurrentIndex(max(0, combo.findText(cur)))
                combo.currentTextChanged.connect(
                    lambda v, k=key, f=fk: self._set_raw(k, f, v))
                form.addRow(spec["lab"], combo)
                widgets.append((fk, combo))
            elif kind == "blend":
                combo = QComboBox()
                combo.addItems(BLEND_MODES)
                cur = str(e.get(fk, "Normal"))
                combo.setCurrentIndex(max(0, combo.findText(cur)))
                combo.currentTextChanged.connect(
                    lambda v, k=key, f=fk: self._set_raw(k, f, v))
                form.addRow(spec["lab"], combo)
                widgets.append((fk, combo))
            elif kind == "color":
                col = list(e.get(fk) or [0, 0, 0])
                sw = ColorSwatch((col[0], col[1], col[2]), spec["lab"])
                sw.clickedColor.connect(
                    lambda k=key, f=fk, s=sw: self._pick_color(k, f, s))
                row = QHBoxLayout()
                row.setContentsMargins(0, 0, 0, 0)
                row.addWidget(sw)
                row.addStretch(1)
                form.addRow(spec["lab"], row)
                widgets.append((fk, sw))
            else:
                div = spec.get("div", 1)
                if kind == "percent":
                    div = 100
                sl = _Slider(spec.get("lo", 0), spec.get("hi", 100),
                             spec.get("step", 1), float(e.get(fk, 0.0)),
                             div,
                             lambda v, k=key, f=fk: self._set_num(k, f, v))
                form.addRow(spec["lab"], sl)
                widgets.append((fk, sl))

        outer = QVBoxLayout(g)
        outer.setContentsMargins(8, 4, 8, 6)
        outer.addWidget(body)
        body.setEnabled(bool(e.get("enabled")))
        self.groups[key] = (g, widgets)
        self.bodies[key] = body
        return g

    # ---------- 写回 ----------

    def _toggle(self, key, on):
        self.effects[key]["enabled"] = bool(on)
        body = self.bodies.get(key)
        if body is not None:
            body.setEnabled(bool(on))
        self._sync_global()
        self._preview()

    def _toggle_global(self, on):
        self.effects[GLOBAL_LIGHT]["enabled"] = bool(on)
        self._global_body.setEnabled(bool(on))
        self._sync_global()
        self._preview()

    def _set_global(self, fk, v):
        self.effects[GLOBAL_LIGHT][fk] = float(v)
        self._sync_global()
        self._preview()

    def _global_on(self):
        return bool(self.effects.get(GLOBAL_LIGHT, {}).get("enabled"))

    def _sync_global(self):
        """勾了「使用全局光」的效果：自己的角度 / 高度输入框禁用并显示全局值。"""
        if not self.groups:
            return
        gvars = self.effects.get(GLOBAL_LIGHT) or {}
        for key in EFFECT_ORDER:
            e = self.effects.get(key) or {}
            if "use_global" not in e:
                continue
            entry = self.groups.get(key)
            if entry is None:
                continue
            _g, widgets = entry
            on = bool(e.get("use_global")) and self._global_on()
            enabled = bool(e.get("enabled"))
            for fk, w in widgets:
                if fk in ("angle", "altitude"):
                    w.setEnabled(enabled and not on)
                    if on and isinstance(w, _Slider):
                        w.set_value(float(gvars.get(fk, 0.0)))

    def _set_num(self, key, fk, v):
        self.effects[key][fk] = float(v)
        self._preview()

    def _set_raw(self, key, fk, v):
        self.effects[key][fk] = v
        self._preview()

    def _pick_color(self, key, fk, sw):
        c = QColorDialog.getColor(sw.color(), self, "效果颜色")
        if not c.isValid():
            return
        sw.set_color((c.red(), c.green(), c.blue()))
        self.effects[key][fk] = [c.red(), c.green(), c.blue()]
        self._preview()

    def _preview(self):
        self.layer.effects = self.effects
        self.main.request_render()

    def _reset(self):
        self.effects = factory_default_effects()
        self.layer.effects = self.effects
        self._build()
        self._preview()

    def _save_default(self):
        save_style_default(self.effects)
        self.main.status.showMessage("已把当前参数存为图层样式的默认值", 4000)

    # ---------- 关闭 ----------

    def reject(self):
        self.layer.effects = self.orig
        self.main.request_render()
        super().reject()

    def accept(self):
        if not any(self.effects[k].get("enabled") for k in EFFECT_ORDER):
            self.layer.effects = None      # 全关就清掉，保持工程文件干净
        else:
            self.layer.effects = self.effects
        self.main.commit("图层样式")
        self.main.request_render()
        super().accept()
