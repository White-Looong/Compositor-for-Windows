# -*- coding: utf-8 -*-
"""智能对象的「智能滤镜」管理对话框。

和普通滤镜的区别：普通滤镜写死在像素里，改一次就得 Ctrl+Z；
智能滤镜只是存在图层上的一串参数，每次渲染重新施加一遍 ——
所以这里可以随便改参数、调顺序、关掉、删掉，原始内容始终完好无损。
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox,
                               QHBoxLayout, QLabel, QListWidget,
                               QListWidgetItem, QMessageBox, QPushButton,
                               QVBoxLayout)

from ..core.filters import FILTERS, FILTER_ORDER
from ..core.smart import new_filter
from .filter_dialog import FilterDialog


class SmartFilterDialog(QDialog):
    def __init__(self, main, layer):
        super().__init__(main)
        self.main = main
        self.layer = layer
        self.setWindowTitle("智能滤镜 —— %s" % layer.name)
        self.resize(360, 380)
        # 进来的时候存一份：取消就把图层还原成这个样子
        self._base = [dict(f) for f in (layer.so_filters or [])]

        root = QVBoxLayout(self)
        root.addWidget(QLabel("下面的滤镜按顺序施加。取消可全部还原。"))

        self.list = QListWidget()
        self.list.itemChanged.connect(self._on_item_changed)
        self.list.currentRowChanged.connect(self._update_enabled)
        root.addWidget(self.list, 1)

        # ---- 添加 ----
        add_row = QHBoxLayout()
        self.combo = QComboBox()
        for key in FILTER_ORDER:
            self.combo.addItem(FILTERS[key].name.replace("…", ""), key)
        self.btn_add = QPushButton("添加")
        self.btn_add.clicked.connect(self._add)
        add_row.addWidget(self.combo, 1)
        add_row.addWidget(self.btn_add)
        root.addLayout(add_row)

        # ---- 编辑 / 顺序 / 删除 ----
        row = QHBoxLayout()
        self.btn_edit = QPushButton("参数…")
        self.btn_up = QPushButton("上移")
        self.btn_down = QPushButton("下移")
        self.btn_del = QPushButton("删除")
        self.btn_edit.clicked.connect(self._edit)
        self.btn_up.clicked.connect(lambda: self._move(-1))
        self.btn_down.clicked.connect(lambda: self._move(1))
        self.btn_del.clicked.connect(self._delete)
        for b in (self.btn_edit, self.btn_up, self.btn_down, self.btn_del):
            row.addWidget(b)
        root.addLayout(row)

        row2 = QHBoxLayout()
        self.btn_clear = QPushButton("全部清除")
        self.btn_clear.clicked.connect(self._clear)
        row2.addWidget(self.btn_clear)
        row2.addStretch(1)
        root.addLayout(row2)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        root.addWidget(bb)

        self._rebuild()

    # ---------- 列表 ----------

    def _filters(self):
        fl = self.layer.so_filters
        if fl is None:
            fl = self.layer.so_filters = []
        return fl

    def _label(self, f):
        name = FILTERS.get(f.get("key"))
        name = (name.name.replace("…", "") if name else f.get("key"))
        tail = ""
        if not f.get("enabled", True):
            tail += "（已关闭）"
        if abs(float(f.get("strength", 1.0)) - 1.0) > 1e-6:
            tail += "  %d%%" % int(round(float(f["strength"]) * 100))
        return name + tail

    def _rebuild(self, keep_index=0):
        self.list.blockSignals(True)
        self.list.clear()
        for f in self._filters():
            it = QListWidgetItem(self._label(f))
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if f.get("enabled", True)
                             else Qt.Unchecked)
            self.list.addItem(it)
        self.list.blockSignals(False)
        n = self.list.count()
        if n:
            self.list.setCurrentRow(max(0, min(n - 1, keep_index)))
        self._update_enabled()

    def _update_enabled(self):
        i = self.list.currentRow()
        n = self.list.count()
        has = n > 0
        self.btn_edit.setEnabled(has)
        self.btn_del.setEnabled(has)
        self.btn_clear.setEnabled(has)
        self.btn_up.setEnabled(has and i > 0)
        self.btn_down.setEnabled(has and i >= 0 and i < n - 1)

    # ---------- 操作 ----------

    def _apply_change(self):
        """滤镜列表变了：改的过程是可撤销的，所以不直接写历史栈。"""
        self.main.mark_dirty()
        self.main.request_render()
        self.main.status.showMessage("智能滤镜已更新（关掉对话框前随时可调）",
                                     3000)

    def _on_item_changed(self, item):
        self.list.blockSignals(True)
        try:
            i = self.list.row(item)
            fl = self._filters()
            if 0 <= i < len(fl):
                fl[i]["enabled"] = (item.checkState() == Qt.Checked)
                item.setText(self._label(fl[i]))
        finally:
            self.list.blockSignals(False)
        self._apply_change()

    def _add(self):
        key = self.combo.currentData()
        if not key:
            return
        dlg = FilterDialog(self, key)
        dlg.paramsChanged.connect(lambda _p, _s: None)
        if not dlg.exec():
            return
        self._filters().append(new_filter(key, dlg.values(), dlg.strength()))
        self._rebuild(self.list.count())
        self._apply_change()

    def _edit(self):
        i = self.list.currentRow()
        fl = self._filters()
        if not (0 <= i < len(fl)):
            return
        f = fl[i]
        snap = dict(f)
        dlg = FilterDialog(self, f.get("key"), initial=f.get("params"),
                           initial_strength=float(f.get("strength", 1.0)))

        def preview(params, strength):
            # 预览：临时改写这条滤镜（取消时会从 snap 整份还原）
            f["params"] = dict(params)
            f["strength"] = float(strength)
            self.main.begin_interactive()
            self.main.request_render()

        dlg.paramsChanged.connect(preview)
        if dlg.exec():
            f["params"] = dict(dlg.values())
            f["strength"] = float(dlg.strength())
            self._rebuild(i)
            self._apply_change()
        else:
            f.clear()
            f.update(snap)
            self.main.request_render()

    def _move(self, delta):
        i = self.list.currentRow()
        fl = self._filters()
        j = i + delta
        if not (0 <= i < len(fl)) or not (0 <= j < len(fl)):
            return
        fl[i], fl[j] = fl[j], fl[i]
        self._rebuild(j)
        self._apply_change()

    def _delete(self):
        i = self.list.currentRow()
        fl = self._filters()
        if not (0 <= i < len(fl)):
            return
        name = FILTERS.get(fl[i].get("key"))
        name = name.name.replace("…", "") if name else "这条滤镜"
        if QMessageBox.question(
                self, "删除智能滤镜", "删除「%s」？\n原始内容不会被改动。" % name,
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes) != QMessageBox.Yes:
            return
        fl.pop(i)
        self._rebuild(i)
        self._apply_change()

    def _clear(self):
        if QMessageBox.question(
                self, "清除智能滤镜", "删掉全部智能滤镜？\n原始内容不会被改动。",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes) != QMessageBox.Yes:
            return
        self._filters().clear()
        self._rebuild(0)
        self._apply_change()

    # ---------- 收尾 ----------

    def reject(self):
        self.layer.so_filters = [dict(f) for f in self._base]
        self.main.request_render()
        self.main.status.showMessage("已还原智能滤镜", 2500)
        super().reject()
