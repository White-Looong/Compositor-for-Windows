# -*- coding: utf-8 -*-
"""逐字调整对话框：给单个字加字距 / 抬基线 / 拉宽，边改边看。

和图层样式、笔刷设置一个路子：
  * 一张字符网格，点一个字就选中它
  * 下面三个滑块只作用于**选中的那个字**
  * 改动**立刻**写回图层并重渲染（画布上就能看到）
  * 「取消」把打开对话框时的整份参数还原回去

单位说明：字距 / 基线是**像素**（跟字号同尺度），水平缩放是百分比。
存进 `layer.text["chars"]` 的形式见 core/text.py 里的 `chars_to_json`。
"""

from __future__ import annotations

import copy

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QFormLayout,
                               QGroupBox, QHBoxLayout, QLabel, QListView,
                               QListWidget, QListWidgetItem, QPushButton,
                               QSlider, QVBoxLayout, QWidget)

from ..core.text import char_slots, chars_to_json, normalize_chars

# key, 标签, 滑块下限, 上限, 默认值（scale 用百分比表示）
FIELDS = [("dx", "字距", -200, 200, 0),
          ("dy", "基线", -200, 200, 0),
          ("scale", "水平缩放", 10, 300, 100)]
FIELD_KEYS = [k for k, _lb, _lo, _hi, _d in FIELDS]


class CharAdjustDialog(QDialog):
    def __init__(self, main, parent=None):
        super().__init__(parent)
        layer = main.selected_layer()
        if layer is None or not layer.is_text or layer.text is None:
            raise ValueError("逐字调整需要一个文字图层")
        self.main = main
        self.layer = layer
        self.params = copy.deepcopy(layer.text)
        # 打开时的现场，取消时整份还原
        self._orig_chars = copy.deepcopy(self.params.get("chars") or {})
        self.adj = normalize_chars(self.params.get("chars"))
        self._index = None
        self._sync = False

        self.setWindowTitle("逐字调整 —— %s" % layer.name)
        self.setMinimumWidth(500)
        root = QVBoxLayout(self)

        tip = QLabel("先点一个字，再拖下面的滑块。\n"
                     "字距是「这个字之后」多留的缝，基线是上下挪，"
                     "水平缩放是把这个字拉宽 / 压扁。")
        tip.setStyleSheet("color: #888;")
        root.addWidget(tip)

        self.char_list = QListWidget()
        self.char_list.setViewMode(QListWidget.IconMode)
        self.char_list.setFlow(QListView.LeftToRight)
        self.char_list.setWrapping(True)
        self.char_list.setResizeMode(QListWidget.Adjust)
        self.char_list.setMovement(QListView.Static)
        self.char_list.setSpacing(2)
        cf = self.char_list.font()
        cf.setPixelSize(16)
        self.char_list.setFont(cf)
        self.char_list.setMaximumHeight(150)
        self.char_list.currentItemChanged.connect(self._on_pick)
        root.addWidget(self.char_list)

        g = QGroupBox("选中的字")
        form = QFormLayout(g)
        self.sliders = {}
        self.value_labels = {}
        for key, label, lo, hi, dv in FIELDS:
            s = QSlider(Qt.Horizontal)
            s.setRange(lo, hi)
            s.setValue(dv)
            s.valueChanged.connect(lambda v, k=key: self._on_slide(k, v))
            lab = QLabel(str(dv))
            lab.setMinimumWidth(48)
            lab.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            row = QHBoxLayout()
            row.addWidget(s, 1)
            row.addWidget(lab)
            holder = QWidget()
            holder.setLayout(row)
            form.addRow(label, holder)
            self.sliders[key] = s
            self.value_labels[key] = lab
        root.addWidget(g)

        btns = QHBoxLayout()
        self.btn_reset = QPushButton("复位这个字")
        self.btn_reset.clicked.connect(self.reset_current)
        self.btn_clear = QPushButton("全部清除")
        self.btn_clear.clicked.connect(self.clear_all)
        btns.addWidget(self.btn_reset)
        btns.addWidget(self.btn_clear)
        btns.addStretch(1)
        root.addLayout(btns)

        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.button(QDialogButtonBox.Ok).setText("确定")
        box.button(QDialogButtonBox.Cancel).setText("取消")
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        root.addWidget(box)

        self.reload()

    # ---------- 字符网格 ----------

    def reload(self):
        """按当前内容重建字符网格（内容变了要重来，索引会整体错位）。"""
        self._sync = True
        self.char_list.clear()
        for idx, disp, is_nl in char_slots(self.params.get("content", "")):
            item = QListWidgetItem("%s" % disp)
            item.setData(Qt.UserRole, idx)
            item.setToolTip("第 %d 个字符" % (idx + 1))
            if idx in self.adj:
                item.setForeground(Qt.red)
            self.char_list.addItem(item)
        self._sync = False
        if self.char_list.count():
            self.char_list.setCurrentRow(0)
        else:
            self._index = None
            self._refresh_sliders()

    def select_index(self, index):
        """按字符下标选中（测试与外部调用用）。"""
        for row in range(self.char_list.count()):
            if self.char_list.item(row).data(Qt.UserRole) == index:
                self.char_list.setCurrentRow(row)
                return True
        return False

    def _on_pick(self, cur, _prev=None):
        if self._sync:
            return
        self._index = None if cur is None else int(cur.data(Qt.UserRole))
        self._refresh_sliders()

    def _refresh_sliders(self):
        """把选中那个字的当前值写回滑块（改过的话标红）。"""
        self._sync = True
        cur = self.adj.get(self._index) if self._index is not None else None
        for key, _lb, _lo, _hi, dv in FIELDS:
            if cur is None:
                v = dv
            elif key == "scale":
                v = int(round(cur[2] * 100.0))
            else:
                v = int(round(cur[0] if key == "dx" else cur[1]))
            self.sliders[key].setValue(max(self.sliders[key].minimum(),
                                           min(self.sliders[key].maximum(), v)))
            self.value_labels[key].setText(self._fmt(key, v))
        for key in FIELD_KEYS:
            self.sliders[key].setEnabled(self._index is not None)
        self.btn_reset.setEnabled(self._index is not None)
        self._sync = False

    def _fmt(self, key, v):
        return "%d%%" % v if key == "scale" else "%d px" % v

    # ---------- 改值 ----------

    def _on_slide(self, key, value):
        if self._sync or self._index is None:
            return
        self.value_labels[key].setText(self._fmt(key, value))
        cur = list(self.adj.get(self._index, (0.0, 0.0, 1.0)))
        if key == "scale":
            cur[2] = max(0.01, value / 100.0)
        elif key == "dx":
            cur[0] = float(value)
        else:
            cur[1] = float(value)
        self.adj[self._index] = tuple(cur)
        self._touch_current()

    def set_field(self, key, value):
        """给测试 / 外部用：直接设某个字段并立即生效。"""
        slider = self.sliders[key]
        slider.setValue(int(value))     # 触发 _on_slide

    def _touch_current(self):
        item = self.char_list.currentItem()
        if item is not None:
            item.setForeground(Qt.red)
        self.apply()

    def apply(self):
        """把逐字表写回图层（非破坏性：只改参数，不碰像素）。"""
        chars = chars_to_json(self.adj)
        self.params["chars"] = chars
        self.main.set_text_param("chars", chars)

    def reset_current(self):
        if self._index is None:
            return
        self.adj.pop(self._index, None)
        item = self.char_list.currentItem()
        if item is not None:
            item.setForeground(self.char_list.palette().text())
        self._refresh_sliders()
        self.apply()

    def clear_all(self):
        self.adj.clear()
        for row in range(self.char_list.count()):
            self.char_list.item(row).setForeground(
                self.char_list.palette().text())
        self._refresh_sliders()
        self.apply()

    # ---------- 收尾 ----------

    def _restore_orig(self):
        self.main.set_text_param("chars", copy.deepcopy(self._orig_chars))

    def reject(self):
        self._restore_orig()
        super().reject()

    def accept(self):
        super().accept()
