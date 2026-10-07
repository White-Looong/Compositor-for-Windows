# -*- coding: utf-8 -*-
"""通道面板：RGB / Alpha 四路 + 附加的 Alpha 通道列表。

与 Photoshop 的通道面板同构，但只做本项目支持得住的那部分（见 core/channels.py
文件头）：显示开关、载入选区、存为通道、重命名 / 删除 / 排序。

**通道开关只改显示，不改图层数据** —— 所以这里不用 `commit()`，
只要 `request_render()`；改的是 doc 里的显示状态，撤销时自然跟着回退。
"""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QListWidget,
                               QListWidgetItem, QMenu, QPushButton,
                               QVBoxLayout, QWidget)

from ..core import channels as CH

THUMB = 30
ROLE_ID = Qt.UserRole
ROLE_MASK = Qt.UserRole + 1          # 附加通道用：True = 这是可编辑的通道

# 画布自带的那几行（R/G/B/A + 合并 Alpha）没有 uuid，用 "@" 前缀区分。
RGB_ROW_PREFIX = CH.RGB_ROW_PREFIX
COMPOSITE_ROW = CH.COMPOSITE_ROW


def _gray_icon(gray, size=THUMB):
    """单通道灰度图 -> 图标（深色描边，缩略图看得清边界）。"""
    g = np.ascontiguousarray(gray, np.uint8)
    img = QImage(g.data, g.shape[1], g.shape[0], g.shape[1],
                 QImage.Format_Grayscale8)
    pm = QPixmap.fromImage(img).scaled(QSize(size, size), Qt.IgnoreAspectRatio,
                                       Qt.FastTransformation)
    out = QPixmap(QSize(size, size))
    out.fill(Qt.transparent)
    p = QPainter(out)
    p.drawPixmap(0, 0, pm)
    p.setPen(QPen(QColor(120, 120, 126), 1))
    p.drawRect(0, 0, size - 1, size - 1)
    p.end()
    return QIcon(out)


def _fit(gray, size=THUMB):
    if gray.shape[:2] == (size, size):
        return gray
    return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)


class ChannelsPanel(QWidget):
    """通道面板。挂进 QDockWidget，由 MainWindow 持有（见 main_window）。"""

    #: 面板按钮 / 右键菜单发的信号，MainWindow 接（"__new__" 表示新建）
    loadRequested = Signal(str)
    saveRequested = Signal(str)
    deleteRequested = Signal(str)
    mergeRequested = Signal(str, str)      # 通道 id, replace/add/subtract

    def __init__(self, main):
        super().__init__()
        self.main = main
        self._arr = None              # 最近一次渲染结果，用来生成 RGB 缩略图
        self._building = False

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        top = QHBoxLayout()
        top.setSpacing(4)
        top.addWidget(QLabel("通道："))
        self.kind = QComboBox()
        self.kind.addItems(["RGB", "灰度"])
        self.kind.setEnabled(False)   # 只读展示：底层还没有色彩模式切换
        self.kind.setToolTip("本项目统一按 RGB 处理，灰度只是显示上的近似")
        top.addWidget(self.kind, 1)
        root.addLayout(top)

        self.list = QListWidget()
        self.list.setIconSize(QSize(THUMB, THUMB))
        # 高度按行数自适应（每行 THUMB + padding），别写死 —— 写死的话
        # 通道一多就默默把后面的藏进滚动区，用户以为通道没建出来
        self._row_h = THUMB + 10
        self.list.setFixedHeight(self._row_h * 4 + 10)
        self.list.currentItemChanged.connect(self._on_current)
        self.list.itemDoubleClicked.connect(self._on_double)
        self.list.itemChanged.connect(self._on_checked)
        self.list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list.customContextMenuRequested.connect(self._menu)
        self.list.setStyleSheet(
            "QListWidget { background:#2b2b2e; color:#ddd;"
            " border:1px solid #3a3a3d; }"
            "QListWidget::item { padding:1px 2px; }"
            "QListWidget::item:selected { background:#3d5a8a; }")
        root.addWidget(self.list)

        row = QHBoxLayout()
        row.setSpacing(3)
        self.btn_load = QPushButton("载入")
        self.btn_load.setToolTip("把通道内容作为选区载入画布")
        self.btn_new = QPushButton("新建")
        self.btn_new.setToolTip("新建一个 Alpha 通道并存入当前选区")
        self.btn_del = QPushButton("删除")
        self.btn_del.setToolTip("删除选中的附加 Alpha 通道")
        for b in (self.btn_load, self.btn_new, self.btn_del):
            b.setFixedHeight(22)
            row.addWidget(b)
        root.addLayout(row)
        root.addStretch(1)

        self.btn_load.clicked.connect(
            lambda: self._fire(self.loadRequested, self.current_id()))
        self.btn_new.clicked.connect(lambda: self.saveRequested.emit("__new__"))
        self.btn_del.clicked.connect(
            lambda: self._fire(self.deleteRequested, self.current_id()))

        self.setStyleSheet(
            "QPushButton { background:#3a3a3e; color:#ddd;"
            " border:1px solid #4a4a4f; border-radius:2px; }"
            "QPushButton:hover { background:#45454a; }"
            "QPushButton:pressed { background:#2f2f33; }"
            "QPushButton:disabled { color:#777; }"
            "QLabel { color:#bbb; }")

    # ---------------------------------------------------------------- 数据

    def _fire(self, sig, value):
        if value:
            sig.emit(value)

    def set_render(self, arr):
        """喂一次渲染结果来刷新缩略图。传 None = 画布还没渲过。"""
        self._arr = arr
        self.rebuild()

    def current_id(self):
        it = self.list.currentItem()
        if it is None:
            return None
        cid = it.data(ROLE_ID)
        return None if cid in (None, "") else str(cid)

    def select_id(self, cid):
        for i in range(self.list.count()):
            if self.list.item(i).data(ROLE_ID) == cid:
                self.list.setCurrentRow(i)
                return True
        return False

    # ---------------------------------------------------------------- 构建

    def rebuild(self):
        doc = self.main.doc
        if doc is None:
            self._building = True
            self.list.clear()
            self._building = False
            return
        keep = self.current_id()
        self._building = True
        self.list.clear()
        arr = self._arr
        v = CH.rgb_visibility(doc)

        # RGB 三路 + Alpha
        for key in CH.RGB_VIEW_KEYS:
            label = ("红", "绿", "蓝", "Alpha")[CH.RGB_VIEW_KEYS.index(key)]
            if arr is None:
                gray = np.full((THUMB, THUMB), 235, np.uint8)
            else:
                gray = _fit(CH.channel_thumb(arr, key, THUMB))
            self._add_row(gray, label, RGB_ROW_PREFIX + key,
                          v.get(key, True), editable=False)

        # "合并的 Alpha 通道"（存在时才出现）
        ca = getattr(doc, "composite_alpha", None)
        if ca is not None:
            self._add_row(_fit(ca), "合并 Alpha", COMPOSITE_ROW, True,
                          editable=False)

        # 附加的 Alpha 通道
        for c in (getattr(doc, "channels", None) or []):
            self._add_row(_fit(c.mask), CH.truncate_name(c.name), c.id,
                          c.visible, editable=True, tooltip=c.name)

        self._building = False
        if keep and self.select_id(keep):
            pass
        elif self.list.count():
            self.list.setCurrentRow(0)
        self._update_buttons()
        self._fit_height()

    def _fit_height(self):
        """列表高度跟着通道条数走，最多 12 条（再高就把 dock 撑得没地方画布了）。"""
        n = max(4, min(12, self.list.count()))
        self.list.setFixedHeight(self._row_h * n + 10)

    def _add_row(self, gray, text, cid, visible, editable, tooltip=""):
        it = QListWidgetItem(_gray_icon(gray), text)
        it.setData(ROLE_ID, cid)
        it.setData(ROLE_MASK, bool(editable))
        if tooltip:
            it.setToolTip(tooltip)
        flags = it.flags() | Qt.ItemIsUserCheckable
        if editable:
            flags |= Qt.ItemIsEditable
        it.setFlags(flags)
        it.setCheckState(Qt.Checked if visible else Qt.Unchecked)
        self.list.addItem(it)
        return it

    def _update_buttons(self):
        it = self.list.currentItem()
        cid = self.current_id()
        if it is None:
            self.btn_load.setEnabled(False)
            self.btn_del.setEnabled(False)
            return
        editable = bool(it.data(ROLE_MASK))
        # 载入对 RGB / 合并 Alpha 也有意义（把那一路的明暗当选区）
        self.btn_load.setEnabled(bool(cid))
        self.btn_del.setEnabled(editable)

    # ---------------------------------------------------------------- 交互

    def _on_current(self, cur, _prev):
        if self._building:
            return
        self._update_buttons()

    def _on_checked(self, item):
        """勾选框 = 通道的显示开关。**只改显示**，不动图层数据。"""
        if self._building or item is None:
            return
        cid = str(item.data(ROLE_ID) or "")
        doc = self.main.doc
        if doc is None:
            return
        want = item.checkState() == Qt.Checked
        if cid.startswith(RGB_ROW_PREFIX):
            key = cid[1:]
            if key in CH.RGB_VIEW_KEYS:
                doc.channel_view[key] = want
        else:
            c = CH.find_channel(doc, cid)
            if c is not None:
                c.visible = want
            elif cid == COMPOSITE_ROW:
                if want and doc.composite_alpha is None:
                    arr = self._arr
                    if arr is not None:
                        doc.composite_alpha = CH.composite_alpha_of(arr)
        self.main.request_render()

    def _on_double(self, item):
        if self._building or item is None:
            return
        if item.flags() & Qt.ItemIsEditable:
            self.list.editItem(item)      # 附加通道：双击改名
            return
        cid = str(item.data(ROLE_ID) or "")
        if cid:
            self.loadRequested.emit(cid)

    def _on_rename(self, item, text):
        cid = str(item.data(ROLE_ID) or "")
        if not item.data(ROLE_MASK) or self.main.doc is None:
            return
        name = CH.rename_channel(self.main.doc, cid, text)
        if name is None:
            return
        if name.name != text:
            # 被去重 / 纠正过了，界面上的名字要跟着改回真实值
            self._building = True
            item.setText(CH.truncate_name(name.name))
            self._building = False
        # 重命名要落自己的撤销点 —— 否则下一次别的操作一撤销，
        # 改过的名字就被一起带回去了（用户会觉得"我刚改的名字怎么丢了"）
        self.main.commit("重命名通道")
        self.rebuild()
        self.select_id(cid)
        self.main.status.showMessage("通道已改名为「%s」" % name.name, 2500)

    def _menu(self, pos):
        item = self.list.itemAt(pos)
        if item is None:
            return
        cid = str(item.data(ROLE_ID) or "")
        m = QMenu(self)
        if cid:
            m.addAction("载入选区", lambda: self.loadRequested.emit(cid))
        if item.data(ROLE_MASK):
            m.addAction("重命名", lambda: self.list.editItem(item))
            m.addSeparator()
            a_up = m.addAction("上移")
            a_dn = m.addAction("下移")
            m.addSeparator()
            sub = m.addMenu("并入合并 Alpha")
            for key, label in (("replace", "替换"), ("add", "添加"),
                               ("subtract", "减去")):
                sub.addAction(label,
                              lambda _, k=key: self.mergeRequested.emit(cid, k))
            m.addSeparator()
            m.addAction("删除通道", lambda: self.deleteRequested.emit(cid))
            a_up.triggered.connect(lambda: self._move(cid, -1))
            a_dn.triggered.connect(lambda: self._move(cid, 1))
        else:
            m.addAction("新建通道…", lambda: self.saveRequested.emit("__new__"))
        m.exec(self.list.viewport().mapToGlobal(pos))

    def _move(self, cid, delta):
        if CH.reorder_channel(self.main.doc, cid, delta):
            self.rebuild()
            self.select_id(cid)
