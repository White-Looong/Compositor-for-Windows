# -*- coding: utf-8 -*-
"""画布大小对话框：宽高 + 九宫格锚点 + 新边怎么填。

锚点用九个 radio 摆成 3x3 的格子（和 Photoshop 一样），而不是下拉 ——
下拉说不清"钉住哪一角"，而这正是这个对话框最容易搞错的地方。

「内容感知」那一条是本项目和 Photoshop 对齐的关键：它不是把新边填成
背景色，而是把现有边缘"长"出去（见 `core/content_aware.py`）。
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QDialog,
                               QDialogButtonBox, QFormLayout, QGridLayout,
                               QGroupBox, QHBoxLayout, QLabel, QSpinBox,
                               QVBoxLayout, QWidget)

from ..core.content_aware import FILL_MODES
from ..core.document import ANCHORS

# 边缘填充方式。前三个是纯色（PS 的 Background / White / Black 那一栏），
# 最后一个才是内容感知。
FILL_CHOICES = ("内容感知", "前景色", "白色", "黑色", "透明")


class CanvasSizeDialog(QDialog):
    def __init__(self, parent, doc, anchor="居中", fill="内容感知",
                 color=None, mode="邻近"):
        super().__init__(parent)
        self.setWindowTitle("画布大小")
        self._ow, self._oh = doc.width, doc.height
        self._color = tuple(color) if color is not None else None

        root = QVBoxLayout(self)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        form.setSpacing(6)

        self.width = QSpinBox()
        self.width.setRange(1, 40000)
        self.width.setValue(self._ow)
        self.height = QSpinBox()
        self.height.setRange(1, 40000)
        self.height.setValue(self._oh)
        form.addRow("宽度:", self.width)
        form.addRow("高度:", self.height)

        self.rel = QCheckBox("相对（填要扩出多少像素）")
        self.rel.stateChanged.connect(self._on_rel)
        form.addRow("", self.rel)

        # 九宫格锚点
        box = QGroupBox("定位（钉住哪一角）")
        grid = QGridLayout(box)
        grid.setSpacing(2)
        self.anchor_group = QButtonGroup(self)
        self.anchor_group.setExclusive(True)
        for i, name in enumerate(ANCHORS):
            rb = _rb(name, self)
            self.anchor_group.addButton(rb)
            grid.addWidget(rb, i // 3, i % 3)
            if name == anchor:
                rb.setChecked(True)
        if self.anchor_group.checkedId() < 0:
            self.anchor_group.buttons()[4].setChecked(True)   # 居中
        form.addRow(box)

        self.fill = QComboBox()
        self.fill.addItems(list(FILL_CHOICES))
        self.fill.setCurrentText(fill)
        self.fill.currentTextChanged.connect(self._on_fill)
        form.addRow("边缘填充", self.fill)

        self.mode = QComboBox()
        self.mode.addItems(list(FILL_MODES))
        self.mode.setCurrentText(mode)
        form.addRow("填充方式", self.mode)

        self.note = QLabel()
        self.note.setWordWrap(True)
        self.note.setStyleSheet("color:#888; font-size:11px;")
        self.form = form
        form.addRow("", self.note)

        root.addLayout(form)

        row = QHBoxLayout()
        row.addWidget(QLabel("提示：画布只能变大；变小请用「按内容裁剪」"))
        row.addStretch(1)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        row.addWidget(bb)
        root.addLayout(row)

        self._on_rel()
        self._on_fill(self.fill.currentText())

    # ---------- 交互 ----------

    def _on_rel(self):
        on = self.rel.isChecked()
        self.width.setMinimumWidth(0)
        self.width.setPrefix("+ " if on else "")
        self.height.setPrefix("+ " if on else "")
        if on:
            self.width.setRange(0, 40000)
            self.height.setRange(0, 40000)
            self.width.setValue(0)
            self.height.setValue(0)
        else:
            self.width.setRange(1, 40000)
            self.height.setRange(1, 40000)
            self.width.setValue(self._ow)
            self.height.setValue(self._oh)

    def _on_fill(self, name):
        self.mode.setEnabled(name == "内容感知")
        lab = self.form.labelForField(self.mode)
        if lab is not None:
            lab.setEnabled(name == "内容感知")
        texts = {
            "内容感知": "把每个位图图层的边缘向外延伸，边缘看起来是连续的",
            "前景色": "新边填当前前景色",
            "白色": "新边填白色",
            "黑色": "新边填黑色",
            "透明": "新边填透明（新增的 Alpha 通道）",
        }
        self.note.setText(texts.get(name, ""))

    # ---------- 取值 ----------

    def anchor(self):
        for rb in self.anchor_group.buttons():
            if rb.isChecked():
                return rb.text()
        return "居中"

    def values(self):
        if self.rel.isChecked():
            w = self._ow + self.width.value()
            h = self._oh + self.height.value()
        else:
            w, h = self.width.value(), self.height.value()
        name = self.fill.currentText()
        return {
            "width": max(1, int(w)),
            "height": max(1, int(h)),
            "anchor": self.anchor(),
            "fill": name,
            "color": (255, 255, 255) if name == "白色" else (
                (0, 0, 0) if name == "黑色" else self._color),
            "mode": self.mode.currentText(),
        }


def _rb(text, parent):
    from PySide6.QtWidgets import QRadioButton
    rb = QRadioButton(text, parent)
    rb.setMinimumWidth(56)
    return rb