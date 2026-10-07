# -*- coding: utf-8 -*-
"""内容感知填充对话框：模式 / 采样范围 / 羽化，带实时预览。

和 `filter_dialog.py` 一样是**破坏性试错**：对话框开着的时候主窗口
持有一份原图，切模式时直接拿原图重算，取消时整块还原 —— 用户看到的是
无损的试错过程（内容感知填充很慢，绝不能"点一下就改好了"）。
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog,
                               QDialogButtonBox, QFormLayout, QLabel,
                               QSpinBox, QVBoxLayout)

from ..core.content_aware import FILL_MODES

# 采样范围（邻近/纹理的搜索半径越大，能找到的参照越远，也越慢）
SEARCH_RADII = (16, 32, 48, 96, 192)


class ContentAwareDialog(QDialog):
    """`paramsChanged` 发出的字典键：mode / feather / patch / axis。"""

    paramsChanged = Signal(dict)

    def __init__(self, parent, initial=None):
        super().__init__(parent)
        self.setWindowTitle("内容感知填充")
        self._p = {"mode": "邻近", "feather": 2.0, "patch": 9,
                  "axis": "水平", "search": 48}
        for k, v in (initial or {}).items():
            if k in self._p:
                self._p[k] = v

        root = QVBoxLayout(self)

        self.mode = QComboBox()
        self.mode.addItems(list(FILL_MODES))
        self.mode.setCurrentText(self._p["mode"])
        self.mode.currentTextChanged.connect(self._on_mode)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        form.setSpacing(6)
        form.addRow("填充方式", self.mode)

        hint = QLabel("邻近：适合天空、墙面等平滑区域\n"
                      "镜像：适合左右对称的构图\n"
                      "纹理合成：适合草地、砖墙等有纹理的区域（慢）")
        hint.setStyleSheet("color:#888; font-size:11px;")
        root.addWidget(hint)

        self.axis = QComboBox()
        self.axis.addItems(["水平", "垂直"])
        self.axis.setCurrentText(self._p["axis"])
        self.axis.currentTextChanged.connect(
            lambda v: self._set("axis", v))
        form.addRow("镜像方向", self.axis)

        self.search = QComboBox()
        for r in SEARCH_RADII:
            self.search.addItem("±%d px" % r, r)
        cur = int(self._p["search"])
        if cur not in SEARCH_RADII:
            # 对话框被复用时可能带了别的值，取最接近的一档
            cur = min(SEARCH_RADII, key=lambda r: abs(r - cur))
            self._p["search"] = cur
        self.search.setCurrentIndex(SEARCH_RADII.index(cur))
        self.search.currentIndexChanged.connect(
            lambda i: self._set("search", int(self.search.itemData(i))))
        form.addRow("采样范围", self.search)

        self.patch = QSpinBox()
        self.patch.setRange(3, 31)
        self.patch.setSingleStep(2)
        self.patch.setValue(int(self._p["patch"]))
        self.patch.valueChanged.connect(
            lambda v: self._set("patch", int(v)))
        form.addRow("纹理块边长", self.patch)

        self.feather = QSpinBox()
        self.feather.setRange(0, 64)
        self.feather.setValue(int(self._p["feather"]))
        self.feather.valueChanged.connect(
            lambda v: self._set("feather", float(v)))
        self.form = form
        form.addRow("边缘羽化 (px)", self.feather)

        root.addLayout(form)

        self.preview = QCheckBox("实时预览")
        self.preview.setChecked(True)
        self.preview.stateChanged.connect(self._emit)
        root.addWidget(self.preview)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        root.addWidget(bb)

        self._on_mode(self._p["mode"])

    # ---------- 控件 ----------

    def _on_mode(self, mode):
        """只有「镜像」才需要方向，只有「纹理合成」才需要块边长 / 采样范围。

        QFormLayout 的行标签要一起藏掉 —— 只藏控件的话左边还挂着一个
        文字，行高也不收。
        """
        self._set_row_visible(self.axis, mode == "镜像")
        self._set_row_visible(self.patch, mode == "纹理合成")
        # 采样范围对「邻近」无效（inpaint 是全局边界扩散，没有搜索半径），
        # 但保留可见 + 灰掉，好让用户知道这个旋钮存在
        self.search.setEnabled(mode == "纹理合成")
        self._set("mode", mode)

    def _set_row_visible(self, widget, on):
        widget.setVisible(on)
        lab = self.form.labelForField(widget)
        if lab is not None:
            lab.setVisible(on)

    def _set(self, key, value):
        self._p[key] = value
        self._emit()

    def _emit(self):
        if self.preview.isChecked():
            self.paramsChanged.emit(dict(self._p))

    def values(self):
        return dict(self._p)