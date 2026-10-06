# -*- coding: utf-8 -*-
"""滤镜对话框：参数按 FSpec 自动生成，改动实时预览。

滤镜本身是破坏性的（改写像素），所以对话框打开期间主窗口会保留一份原图，
「取消」时整块还原 —— 用户看到的是无损的试错过程。
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog,
                               QDialogButtonBox, QFormLayout, QHBoxLayout,
                               QLabel, QSlider, QVBoxLayout, QWidget)

from ..core.filters import FILTERS


class FilterDialog(QDialog):
    """滤镜参数对话框。参数控件按 FSpec 自动生成。

    initial / initial_strength 用来**回显已有的值**：智能滤镜要把同一份参数
    再调一遍，打开对话框时必须显示原来的参数而不是默认值。
    """

    paramsChanged = Signal(dict, float)

    def __init__(self, parent, key, initial=None, initial_strength=1.0):
        super().__init__(parent)
        self.spec = FILTERS[key]
        self.setWindowTitle(self.spec.name.replace("…", ""))
        self._vals = self.spec.defaults()
        for k, v in (initial or {}).items():
            if k in self._vals:
                self._vals[k] = v
        self._strength = float(initial_strength)

        root = QVBoxLayout(self)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        form.setSpacing(6)
        self._widgets = {}
        for prm in self.spec.params:
            row = self._make_row(prm)
            self._widgets[prm.key] = row
            form.addRow(prm.label, row)
        root.addLayout(form)

        # 强度：相当于 Photoshop 的「渐隐」
        srow = QWidget()
        sl = QHBoxLayout(srow)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(6)
        self.str_slider = QSlider(Qt.Horizontal)
        self.str_slider.setRange(0, 100)
        self.str_slider.setValue(int(round(self._strength * 100)))
        self.str_slider.setMinimumWidth(110)
        self.str_label = QLabel("%d%%" % int(round(self._strength * 100)))
        self.str_label.setFixedWidth(44)
        self.str_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.str_slider.valueChanged.connect(self._on_strength)
        sl.addWidget(self.str_slider, 1)
        sl.addWidget(self.str_label)
        form.addRow("强度", srow)

        self.preview = QCheckBox("实时预览")
        self.preview.setChecked(True)
        self.preview.stateChanged.connect(self._emit)
        root.addWidget(self.preview)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        root.addWidget(bb)

        # 首帧先推一次，让预览显示默认参数的效果
        self._emit()

    # ---------- 控件 ----------

    def _make_row(self, prm):
        if prm.kind == "bool":
            cb = QCheckBox()
            cb.setChecked(bool(self._vals.get(prm.key, prm.default)))
            cb.stateChanged.connect(
                lambda _s: self._on_value(prm.key, cb.isChecked()))
            return cb

        if prm.kind == "choice":
            combo = QComboBox()
            combo.addItems(prm.choices)
            cur = str(self._vals.get(prm.key, prm.default))
            if cur in prm.choices:
                combo.setCurrentText(cur)
            combo.currentTextChanged.connect(
                lambda v: self._on_value(prm.key, v))
            return combo

        row = QWidget()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        s = QSlider(Qt.Horizontal)
        s.setRange(int(round(prm.lo * prm.scale)),
                   int(round(prm.hi * prm.scale)))
        s.setValue(int(round(float(self._vals.get(prm.key, prm.default))
                             * prm.scale)))
        s.setMinimumWidth(110)
        lab = QLabel()
        lab.setFixedWidth(48)
        lab.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        def fmt(v):
            if prm.kind == "int":
                return "%d" % int(round(v))
            return "%.*f" % (prm.decimals, v)

        def on_change(raw):
            v = raw / float(prm.scale)
            lab.setText(fmt(v))
            self._on_value(prm.key, int(round(v)) if prm.kind == "int" else v)

        s.valueChanged.connect(on_change)
        lab.setText(fmt(s.value() / float(prm.scale)))
        lay.addWidget(s, 1)
        lay.addWidget(lab)
        return row

    # ---------- 取值 ----------

    def _on_value(self, key, value):
        self._vals[key] = value
        self._emit()

    def _on_strength(self, v):
        self._strength = v / 100.0
        self.str_label.setText("%d%%" % v)
        self._emit()

    def _emit(self):
        if self.preview.isChecked():
            self.paramsChanged.emit(dict(self._vals), self._strength)

    def values(self):
        return dict(self._vals)

    def strength(self):
        return self._strength
