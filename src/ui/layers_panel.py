# -*- coding: utf-8 -*-
"""图层面板：树形图层列表 + 混合模式 / 不透明度。"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QSize, Qt
from PySide6.QtGui import (QBrush, QColor, QIcon, QImage, QPainter, QPen,
                           QPixmap, QPolygonF)
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QHBoxLayout,
                               QHeaderView, QLabel, QMenu, QPushButton,
                               QSlider, QTreeWidget, QTreeWidgetItem,
                               QVBoxLayout, QWidget)

from ..core.adjust import ADJUSTMENTS, ADJUST_ORDER
from ..core.blend import BLEND_MODES, PASS_THROUGH
from ..core.render import render_layer_thumb


def _arr_to_icon(arr, size=32):
    h, w = arr.shape[:2]
    img = QImage(arr.data, w, h, 4 * w, QImage.Format_RGBA8888)
    pm = QPixmap.fromImage(img)
    if pm.size() != QSize(size, size):
        pm = pm.scaled(QSize(size, size), Qt.KeepAspectRatio,
                       Qt.SmoothTransformation)
    return QIcon(pm)


def _folder_icon(size=32):
    pm = QPixmap(QSize(size, size))
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(QColor(210, 175, 90))
    p.setPen(Qt.NoPen)
    p.drawRoundedRect(2, 8, size - 4, size - 11, 2, 2)
    p.setBrush(QColor(235, 205, 130))
    p.drawRoundedRect(2, 6, 13, 5, 1, 1)
    p.end()
    return QIcon(pm)


def _adjust_icon(size=32):
    """调整层图标：左黑右白的圆（Photoshop 用的就是这个符号）。"""
    pm = QPixmap(QSize(size, size))
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    r = size - 6
    p.setPen(QPen(QColor(150, 150, 155), 1.2))
    p.setBrush(QBrush(QColor(235, 235, 240)))
    p.drawEllipse(3, 3, r, r)
    p.setBrush(QBrush(QColor(40, 40, 45)))
    p.drawPie(3, 3, r, r, 90 * 16, 180 * 16)
    p.end()
    return QIcon(pm)


def _text_icon(size=32):
    """文字图层图标：一个 T（Photoshop 用的就是这个符号）。"""
    pm = QPixmap(QSize(size, size))
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(QPen(QColor(235, 235, 240), 2.4))
    p.drawLine(5, 8, size - 5, 8)
    p.drawLine(size // 2, 8, size // 2, size - 6)
    p.end()
    return QIcon(pm)


def _shape_icon(size=32):
    """矢量形状图层图标：方框 + 圆 + 锚点（PS 用类似的小方块加锚点）。"""
    pm = QPixmap(QSize(size, size))
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(QPen(QColor(150, 150, 158), 1.2))
    p.setBrush(QBrush(QColor(226, 92, 84)))
    p.drawEllipse(4, 4, size - 15, size - 15)
    p.setBrush(Qt.NoBrush)
    p.drawRect(11, 11, size - 15, size - 15)
    p.setBrush(QBrush(QColor(235, 235, 240)))
    p.setPen(Qt.NoPen)
    for x, y in ((4, 4), (size - 11, 4), (4, size - 11),
                 (size - 11, size - 11)):
        p.drawEllipse(x - 1, y - 1, 3, 3)
    p.end()
    return QIcon(pm)


def _smart_icon(size=32):
    """智能对象图标：右下角带折角标记的方块（和 Photoshop 的角标一个意思）。"""
    pm = QPixmap(QSize(size, size))
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(QPen(QColor(150, 150, 158), 1.3))
    p.setBrush(QBrush(QColor(58, 58, 64)))
    p.drawRoundedRect(3, 3, size - 6, size - 6, 2, 2)
    # 右下角折角
    p.setBrush(QBrush(QColor(205, 205, 215)))
    p.drawPolygon(QPolygonF([QPointF(size - 4, size - 14),
                             QPointF(size - 14, size - 4),
                             QPointF(size - 4, size - 4)]))
    p.end()
    return QIcon(pm)


class _Tree(QTreeWidget):
    def __init__(self, panel):
        super().__init__()
        self.panel = panel

    def dropEvent(self, event):
        super().dropEvent(event)
        self.panel.on_dropped()


class LayersPanel(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self._building = False
        self._folder_icon = _folder_icon()
        self._adjust_icon = _adjust_icon()
        self._text_icon = _text_icon()
        self._shape_icon = _shape_icon()
        self._smart_icon = _smart_icon()
        self._smart_badge = _smart_icon(14)
        self._shape_badge = _shape_icon(14)

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # --- 混合模式 / 不透明度（与 Photoshop 一样放在图层面板顶部）---
        row1 = QHBoxLayout()
        row1.setSpacing(4)
        self.blend = QComboBox()
        self.blend.addItems(BLEND_MODES)
        self.blend.setEditable(False)
        self.blend.currentTextChanged.connect(self._on_blend)
        row1.addWidget(self.blend, 1)
        self.opacity = QSlider(Qt.Horizontal)
        self.opacity.setRange(0, 100)
        self.opacity.setValue(100)
        self.opacity.setFixedWidth(90)
        self.opacity.valueChanged.connect(self._on_opacity)
        row1.addWidget(self.opacity)
        self.opacity_label = QLabel("100%")
        self.opacity_label.setFixedWidth(38)
        self.opacity_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row1.addWidget(self.opacity_label)
        root.addLayout(row1)

        # --- 图层树 ---
        self.tree = _Tree(self)
        self.tree.setHeaderHidden(True)
        self.tree.setColumnCount(2)
        self.tree.setRootIsDecorated(True)
        self.tree.setIndentation(12)
        self.tree.setIconSize(QSize(28, 28))
        self.tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tree.setDragDropMode(QAbstractItemView.InternalMove)
        self.tree.setDefaultDropAction(Qt.MoveAction)
        self.tree.setEditTriggers(QAbstractItemView.DoubleClicked |
                                  QAbstractItemView.EditKeyPressed)
        self.tree.setAlternatingRowColors(False)
        self.tree.setStyleSheet(
            "QTreeWidget { background:#2b2b2e; color:#ddd; border:1px solid #3a3a3d; }"
            "QTreeWidget::item { padding:2px 0; }"
            "QTreeWidget::item:selected { background:#2d6db5; color:#fff; }")
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self.tree.setColumnWidth(1, 26)
        self.tree.itemChanged.connect(self._on_item_changed)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._on_tree_menu)
        self.tree.itemSelectionChanged.connect(self._on_sel_changed)
        root.addWidget(self.tree, 1)

        # --- 底部按钮 ---
        row2 = QHBoxLayout()
        row2.setSpacing(3)
        self.btn_add = QPushButton("导入")
        self.btn_add.setToolTip("把图片作为新图层导入")
        self.btn_add.clicked.connect(self.main.import_image_as_layer)
        self.btn_group = QPushButton("建组")
        self.btn_group.clicked.connect(self.main.group_selection)
        self.btn_dup = QPushButton("复制")
        self.btn_dup.clicked.connect(self.main.duplicate_layer)
        self.btn_adj = QPushButton("调整")
        self.btn_adj.setToolTip("新建调整图层（非破坏性色彩调整）")
        self.btn_adj.setMenu(self._adjust_menu())
        self.btn_mask = QPushButton("蒙版")
        self.btn_mask.setMenu(self._mask_menu())
        self.btn_del = QPushButton("删除")
        self.btn_del.clicked.connect(self.main.delete_layer)
        for b in (self.btn_add, self.btn_group, self.btn_dup,
                  self.btn_adj, self.btn_mask, self.btn_del):
            b.setFixedHeight(26)
            row2.addWidget(b)
        root.addLayout(row2)

        self.setStyleSheet(
            "QComboBox, QSlider, QLabel { color:#ddd; }"
            "QPushButton { background:#3a3a3d; color:#ddd; border:1px solid #4a4a4e;"
            " border-radius:3px; padding:3px 6px; }"
            "QPushButton:hover { background:#46464a; }"
            "QPushButton:pressed { background:#2d6db5; }")

    # ---------- 调整层菜单 ----------

    def _adjust_menu(self):
        m = QMenu(self)
        for key in ADJUST_ORDER:
            spec = ADJUSTMENTS[key]
            m.addAction(spec.name,
                        lambda _c=False, k=key: self.main.add_adjustment_layer(k))
        return m

    # ---------- 蒙版菜单 ----------

    def _mask_menu(self):
        m = QMenu(self)
        m.addAction("添加白色蒙版（显示全部）", lambda: self.main.add_mask("white"))
        m.addAction("添加黑色蒙版（隐藏全部）", lambda: self.main.add_mask("black"))
        m.addAction("从图层透明度建立蒙版", lambda: self.main.add_mask("alpha"))
        m.addAction("从图层明度建立蒙版", lambda: self.main.add_mask("luma"))
        m.addSeparator()
        m.addAction("反相蒙版", lambda: self.main.add_mask("invert"))
        m.addAction("停用/启用蒙版", lambda: self.main.toggle_mask())
        m.addAction("删除蒙版", lambda: self.main.delete_mask())
        return m

    # ---------- 重建 ----------

    def rebuild(self):
        doc = self.main.doc
        if doc is None:
            return
        sel = self.main.selected_id
        expanded = set()
        for it in self._iter_items():
            if it.isExpanded():
                expanded.add(it.data(0, Qt.UserRole))

        self._building = True
        self.tree.clear()
        for layer in reversed(doc.layers):      # 面板中顶层在最上面
            self._add(layer, None, expanded)
        self._building = False

        target = None
        for it in self._iter_items():
            if it.data(0, Qt.UserRole) == sel:
                target = it
                break
        if target is not None:
            self.tree.blockSignals(True)
            self.tree.setCurrentItem(target)
            self.tree.blockSignals(False)
        else:
            self.main.selected_id = None
        self.refresh_header()

    def _add(self, layer, parent, expanded):
        if parent is None:
            it = QTreeWidgetItem(self.tree)
        else:
            it = QTreeWidgetItem(parent)
        it.setText(0, layer.name)
        it.setData(0, Qt.UserRole, layer.id)
        flags = (Qt.ItemIsSelectable | Qt.ItemIsEnabled |
                 Qt.ItemIsEditable | Qt.ItemIsDragEnabled |
                 Qt.ItemIsUserCheckable)
        if layer.is_group:
            flags |= Qt.ItemIsDropEnabled
        it.setFlags(flags)
        it.setCheckState(1, Qt.Checked if layer.visible else Qt.Unchecked)

        if layer.is_group:
            it.setIcon(0, self._folder_icon)
        elif layer.is_adjustment:
            it.setIcon(0, self._adjust_icon)
        elif layer.is_text:
            it.setIcon(0, self._text_icon)
        elif layer.is_shape:
            # 形状的位图是派生出来的，一样能出缩略图；右下角叠个形状角标
            thumb = render_layer_thumb(layer, 28)
            if thumb is not None:
                base = _arr_to_icon(np.ascontiguousarray(thumb), 28)
                pm = base.pixmap(QSize(28, 28))
                p = QPainter(pm)
                p.setRenderHint(QPainter.Antialiasing)
                p.drawPixmap(14, 14, self._shape_badge.pixmap(QSize(14, 14)))
                p.end()
                it.setIcon(0, QIcon(pm))
            else:
                it.setIcon(0, self._shape_icon)
        elif layer.is_smart:
            # 缩略图上面叠一层智能对象角标：既能预览内容，又看得出它是智能对象
            thumb = render_layer_thumb(layer, 28)
            if thumb is not None:
                base = _arr_to_icon(np.ascontiguousarray(thumb), 28)
                pm = base.pixmap(QSize(28, 28))
                p = QPainter(pm)
                p.setRenderHint(QPainter.Antialiasing)
                p.drawPixmap(14, 14, self._smart_badge.pixmap(QSize(14, 14)))
                p.end()
                it.setIcon(0, QIcon(pm))
            else:
                it.setIcon(0, self._smart_icon)
        else:
            thumb = render_layer_thumb(layer, 28)
            if thumb is not None:
                it.setIcon(0, _arr_to_icon(np.ascontiguousarray(thumb), 28))
        tags = []
        if layer.mask is not None:
            tags.append("蒙版")
        if layer.clipped:
            tags.append("裁剪")
        if layer.is_smart:
            n = len(layer.so_filters or [])
            tags.append("智能滤镜 x%d" % n if n else "智能对象")
        if layer.is_shape:
            tags.append("形状")
        if tags:
            it.setText(0, "%s   [%s]" % (layer.name, "/".join(tags)))
        it.setSizeHint(0, QSize(0, 32))

        for ch in layer.children:
            self._add(ch, it, expanded)
        if layer.id in expanded:
            it.setExpanded(True)
        return it

    def _iter_items(self):
        stack = [self.tree.topLevelItem(i)
                 for i in range(self.tree.topLevelItemCount())]
        while stack:
            it = stack.pop()
            yield it
            for i in range(it.childCount()):
                stack.append(it.child(i))

    def _on_tree_menu(self, pos):
        """图层树上右键：智能对象才有菜单（其他的还用主菜单）。"""
        it = self.tree.itemAt(pos)
        if it is None or self.main.doc is None:
            return
        lid = it.data(0, Qt.UserRole)
        layer = self.main.doc.find(lid)
        if layer is None or not layer.is_smart:
            return
        main = self.main
        m = QMenu(self)

        def pick_and(fn):
            def go():
                main.select_layer(lid)
                fn()
            return go

        m.addAction("编辑内容…", pick_and(main.edit_smart_content))
        m.addAction("智能滤镜…", pick_and(main.edit_smart_filters))
        m.addAction("新建实例（共用内容）", pick_and(main.new_smart_copy))
        m.addSeparator()
        m.addAction("栅格化", pick_and(main.rasterize_smart))
        m.exec(self.tree.viewport().mapToGlobal(pos))

    def refresh_header(self):
        """同步顶部混合模式 / 不透明度控件。"""
        layer = self.main.selected_layer()
        enable = layer is not None
        self.blend.setEnabled(enable)
        self.opacity.setEnabled(enable)
        if not enable:
            return
        self._building = True
        items = BLEND_MODES
        if layer.is_group and self.blend.count() == len(BLEND_MODES):
            self.blend.insertItem(0, PASS_THROUGH)
        elif not layer.is_group and self.blend.count() != len(BLEND_MODES):
            self.blend.removeItem(0)
        self.blend.blockSignals(True)
        idx = self.blend.findText(layer.blend)
        self.blend.setCurrentIndex(max(0, idx))
        self.blend.blockSignals(False)
        self.opacity.blockSignals(True)
        self.opacity.setValue(int(round(layer.opacity * 100)))
        self.opacity.blockSignals(False)
        self.opacity_label.setText("%d%%" % int(round(layer.opacity * 100)))
        self._building = False

    # ---------- 事件 ----------

    def _on_sel_changed(self):
        if self._building:
            return
        it = self.tree.currentItem()
        if it is None:
            return
        lid = it.data(0, Qt.UserRole)
        if lid != self.main.selected_id:
            self.main.selected_id = lid
            self.main.on_select_changed(from_panel=True)

    def _on_item_changed(self, item, column):
        if self._building:
            return
        lid = item.data(0, Qt.UserRole)
        layer = self.main.doc.find(lid) if self.main.doc else None
        if layer is None:
            return
        if column == 0:
            new_name = item.text(0).split("   [")[0].strip()
            if new_name and new_name != layer.name:
                layer.name = new_name
                self.main.commit("重命名图层")
        elif column == 1:
            vis = item.checkState(1) == Qt.Checked
            if vis != layer.visible:
                layer.visible = vis
                self.main.commit("切换可见性")
                self.main.request_render(self.main.layer_dirty_rect(layer))

    def _on_blend(self, text):
        if self._building:
            return
        layer = self.main.selected_layer()
        if layer is None or not text:
            return
        layer.blend = text
        self.main.commit("混合模式")
        self.main.request_render(self.main.layer_dirty_rect(layer))

    def _on_opacity(self, value):
        if self._building:
            return
        layer = self.main.selected_layer()
        if layer is None:
            return
        self.opacity_label.setText("%d%%" % value)
        layer.opacity = value / 100.0
        self.main.request_render()

    def on_dropped(self):
        """拖拽结束后按树结构重建文档的图层树。"""
        doc = self.main.doc
        if doc is None:
            return

        def build(parent_item, ancestors):
            out = []
            count = (self.tree.topLevelItemCount() if parent_item is None
                     else parent_item.childCount())
            for i in range(count):
                it = (self.tree.topLevelItem(i) if parent_item is None
                      else parent_item.child(i))
                lid = it.data(0, Qt.UserRole)
                if lid in ancestors:
                    continue
                layer = doc.find(lid)
                if layer is None:
                    continue
                if layer.is_group:
                    layer.children = build(it, ancestors | {lid})
                out.append(layer)
            return out

        doc.layers = list(reversed(build(None, set())))
        self.main.commit("调整图层顺序")
        self.main.request_render()
        self.main.refresh_inspector()
