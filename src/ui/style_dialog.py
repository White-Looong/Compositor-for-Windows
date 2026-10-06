# -*- coding: utf-8 -*-
"""图层样式对话框：描边 / 投影 / 内阴影 / 外发光。

改动是**实时预览**的：直接写回图层的 effects 并触发重渲染，
点「取消」或关窗时把打开前的参数整份还原。

参数表在 core.effects 里，这里只负责把它画成控件。
"""

from __future__ import annotations

import copy

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QColorDialog, QComboBox,
                               QDialog, QDialogButtonBox, QFormLayout,
                               QGroupBox, QHBoxLayout, QLabel, QSlider,
                               QVBoxLayout, QWidget)

from ..core.effects import (EFFECT_NAMES, EFFECT_ORDER, STROKE_POSITIONS,
                            default_effects, normalize_effects)
from .tool_options import ColorSwatch

# 每个效果的 (键, 标签, 下限, 上限, 步进)
_FIELDS = {
    "drop_shadow": [("opacity", "不透明度", 0, 100, 1),
                    ("angle", "角度", -180, 180, 1),
                    ("distance", "距离", 0, 300, 1),
                    ("spread", "扩展", 0, 100, 1),
                    ("size", "大小", 0, 250, 1)],
    "outer_glow": [("opacity", "不透明度", 0, 100, 1),
                   ("spread", "扩展", 0, 100, 1),
                   ("size", "大小", 0, 250, 1)],
    "inner_shadow": [("opacity", "不透明度", 0, 100, 1),
                     ("angle", "角度", -180, 180, 1),
                     ("distance", "距离", 0, 300, 1),
                     ("spread", "扩展", 0, 100, 1),
                     ("size", "大小", 0, 250, 1)],
    "stroke": [("opacity", "不透明度", 0, 100, 1),
               ("size", "大小", 1, 100, 1)],
}


class _Slider(QWidget):
    """滑块 + 数值标签。"""

    def __init__(self, lo, hi, step, value, on_change):
        super().__init__()
        self._on_change = on_change
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
        self.lab.setText(str(int(v)))
        self._on_change(float(v))

    def set_value(self, v):
        blocked = self.slider.blockSignals(True)
        self.slider.setValue(int(round(v)))
        self.slider.blockSignals(blocked)
        self.lab.setText(str(int(round(v))))


class StyleDialog(QDialog):
    def __init__(self, layer, main, parent=None):
        super().__init__(parent)
        self.layer = layer
        self.main = main
        self.setWindowTitle("图层样式 —— %s" % layer.name)
        self.resize(340, 520)

        self.orig = copy.deepcopy(layer.effects)
        base = default_effects()
        self.effects = normalize_effects(copy.deepcopy(layer.effects)
                                         if layer.effects else base)
        layer.effects = self.effects

        root = QVBoxLayout(self)
        root.setSpacing(8)
        self.groups = {}
        for key in EFFECT_ORDER:
            root.addWidget(self._make_group(key))
        root.addStretch(1)

        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel |
                               QDialogButtonBox.Reset)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        box.button(QDialogButtonBox.Reset).clicked.connect(self._reset)
        root.addWidget(box)

    # ---------- 构建 ----------

    def _make_group(self, key):
        e = self.effects[key]
        g = QGroupBox(EFFECT_NAMES[key])
        g.setCheckable(True)
        g.setChecked(bool(e.get("enabled")))
        g.toggled.connect(lambda on, k=key: self._toggle(k, on))

        form = QFormLayout(g)
        form.setLabelAlignment(Qt.AlignRight)
        form.setSpacing(5)

        widgets = []
        for fkey, label, lo, hi, step in _FIELDS[key]:
            val = float(e.get(fkey, 0.0))
            if fkey == "opacity":
                val = val * 100.0
            sl = _Slider(lo, hi, step, val,
                         lambda v, k=key, f=fkey: self._set_num(k, f, v))
            form.addRow(label, sl)
            widgets.append((fkey, sl))

        if key == "stroke":
            combo = QComboBox()
            combo.addItems(STROKE_POSITIONS)
            combo.setCurrentText(e.get("position") or STROKE_POSITIONS[0])
            combo.currentTextChanged.connect(
                lambda v, k=key: self._set_raw(k, "position", v))
            form.addRow("位置", combo)

        col = list(e.get("color") or [0, 0, 0])
        sw = ColorSwatch((col[0], col[1], col[2]), "效果颜色")
        sw.clickedColor.connect(
            lambda k=key, s=sw: self._pick_color(k, s))
        row = QHBoxLayout()
        row.addWidget(sw)
        row.addStretch(1)
        form.addRow("颜色", row)

        g.setEnabled(bool(e.get("enabled")))
        self.groups[key] = (g, widgets)
        return g

    # ---------- 写回 ----------

    def _toggle(self, key, on):
        self.effects[key]["enabled"] = bool(on)
        g, _w = self.groups[key]
        g.setEnabled(bool(on))
        self._preview()

    def _set_num(self, key, fkey, v):
        # 不透明度在参数里是 0~1，滑块上按百分比显示
        self.effects[key][fkey] = v / 100.0 if fkey == "opacity" else v
        self._preview()

    def _set_raw(self, key, fkey, v):
        self.effects[key][fkey] = v
        self._preview()

    def _pick_color(self, key, sw):
        c = QColorDialog.getColor(sw.color(), self, "效果颜色")
        if not c.isValid():
            return
        sw.set_color((c.red(), c.green(), c.blue()))
        self.effects[key]["color"] = [c.red(), c.green(), c.blue()]
        self._preview()

    def _reset(self):
        self.effects = default_effects()
        self.layer.effects = self.effects
        for key in EFFECT_ORDER:
            g, widgets = self.groups[key]
            blocked = g.blockSignals(True)
            g.setChecked(False)
            g.setEnabled(False)
            g.blockSignals(blocked)
            for fkey, sl in widgets:
                v = float(self.effects[key].get(fkey, 0.0))
                if fkey == "opacity":
                    v *= 100.0
                sl.set_value(v)
        self._preview()

    def _preview(self):
        self.layer.effects = self.effects
        self.main.request_render()

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
