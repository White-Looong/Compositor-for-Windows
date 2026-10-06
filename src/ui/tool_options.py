# -*- coding: utf-8 -*-
"""顶部工具选项条：随当前工具切换显示的参数。"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QColorDialog,
                               QComboBox, QDoubleSpinBox, QFontComboBox,
                               QHBoxLayout, QLabel, QPushButton, QSlider,
                               QSpinBox, QToolBar, QWidget)

from ..core.selection import ADD, INTERSECT, REPLACE, SUBTRACT

SEL_MODES = [
    ("新选区", REPLACE),
    ("添加到", ADD),
    ("从减去", SUBTRACT),
    ("与交叉", INTERSECT),
]

SELECT_TOOLS = {"rect", "ellipse", "lasso", "polygon", "magic"}
PAINT_TOOLS = {"brush", "eraser"}
TEXT_TOOLS = {"text"}


def _slider(maximum, value, width=92):
    s = QSlider(Qt.Horizontal)
    s.setRange(0, maximum)
    s.setValue(value)
    s.setFixedWidth(width)
    return s


class ColorSwatch(QPushButton):
    clickedColor = Signal()

    def __init__(self, color, tooltip):
        super().__init__()
        self.setFixedSize(30, 26)
        self.setToolTip(tooltip)
        self.set_color(color)
        self.clicked.connect(self.clickedColor.emit)

    def set_color(self, color):
        self._color = QColor(*color)
        self.setStyleSheet(
            "background:%s; border:1px solid #777; border-radius:2px;"
            % self._color.name())

    def color(self):
        return self._color

    def rgb(self):
        return (self._color.red(), self._color.green(), self._color.blue())


class ToolOptions(QToolBar):
    """工具参数。数值由本对象持有，画布与菜单从这里读。"""

    def __init__(self, main):
        super().__init__("工具选项")
        self.main = main
        self.setMovable(False)
        self.setFloatable(False)
        self.setStyleSheet(
            "QToolBar { background:#323235; border-bottom:1px solid #3a3a3d; "
            "padding:3px; spacing:6px; }")

        self.sel_mode = REPLACE
        self.feather = 0.0
        self.tolerance = 32
        self.size = 60
        self.hardness = 0.55
        self.opacity = 1.0
        self.flow = 1.0
        self.smoothing = 0.25
        self.target = "pixel"

        # 新建文字图层时的默认字体设置
        self.font_family = ""
        self.font_size = 96
        self.font_bold = False

        self._build_hint()
        self._build_select()
        self.act_sep1 = self._add_sep()
        self._build_brush()
        self._build_text()
        self.act_sep2 = self._add_sep()
        self._build_colors()

        self.current_tool = "move"
        self.set_tool("move")

    # ---------- 构建 ----------

    def _add_sep(self):
        lab = QLabel("│")
        lab.setStyleSheet("color:#555;")
        return self.addWidget(lab)

    def _build_hint(self):
        """移动工具没有参数，给一行操作提示，免得选项条看起来是空的。"""
        self.lab_hint = QLabel("拖动图层移动 · 拖方形手柄缩放 · 拖圆点旋转 · "
                               "Shift 等比 / 吸附 15° · 空格或中键拖动平移画布")
        self.lab_hint.setStyleSheet("color:#9a9a9e;")
        self.act_hint = self.addWidget(self.lab_hint)

    def _build_select(self):
        self.grp_sel = QWidget()
        from PySide6.QtWidgets import QHBoxLayout
        lay = QHBoxLayout(self.grp_sel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self._mode_btns = []
        for i, (text, mode) in enumerate(SEL_MODES):
            b = QPushButton(text)
            b.setCheckable(True)
            b.setFixedHeight(24)
            b.setStyleSheet(
                "QPushButton { background:#3a3a3d; color:#ccc; border:1px solid "
                "#4a4a4e; border-radius:3px; padding:2px 7px; }"
                "QPushButton:checked { background:#2d6db5; color:#fff; }")
            self.mode_group.addButton(b, i)
            b.clicked.connect(lambda _c=False, m=mode: self._set_mode(m))
            lay.addWidget(b)
            self._mode_btns.append(b)
        self._mode_btns[0].setChecked(True)

        lay.addWidget(QLabel("羽化"))
        self.spin_feather = QDoubleSpinBox()
        self.spin_feather.setRange(0, 500)
        self.spin_feather.setDecimals(1)
        self.spin_feather.setSingleStep(1)
        self.spin_feather.setMaximumWidth(72)
        self.spin_feather.valueChanged.connect(
            lambda v: setattr(self, "feather", float(v)))
        lay.addWidget(self.spin_feather)

        self.lab_tol = QLabel("容差")
        self.spin_tol = QSpinBox()
        self.spin_tol.setRange(0, 255)
        self.spin_tol.setValue(self.tolerance)
        self.spin_tol.setMaximumWidth(66)
        self.spin_tol.valueChanged.connect(
            lambda v: setattr(self, "tolerance", int(v)))
        lay.addWidget(self.lab_tol)
        lay.addWidget(self.spin_tol)
        self.act_sel = self.addWidget(self.grp_sel)

    def _build_brush(self):
        self.grp_brush = QWidget()
        from PySide6.QtWidgets import QHBoxLayout
        lay = QHBoxLayout(self.grp_brush)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        self.spin_size = QSpinBox()
        self.spin_size.setRange(1, 4000)
        self.spin_size.setValue(self.size)
        self.spin_size.setMaximumWidth(70)
        self.spin_size.valueChanged.connect(
            lambda v: setattr(self, "size", int(v)))
        lay.addWidget(QLabel("大小"))
        lay.addWidget(self.spin_size)

        def add_slider(label, attr, value, tip=""):
            lab = QLabel(label)
            s = _slider(100, int(round(value * 100)))
            val = QLabel("%d%%" % int(round(value * 100)))
            val.setFixedWidth(38)
            val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            if tip:
                lab.setToolTip(tip)
                s.setToolTip(tip)

            def on_change(v, a=attr, l=val):
                setattr(self, a, v / 100.0)
                l.setText("%d%%" % v)

            s.valueChanged.connect(on_change)
            lay.addWidget(lab)
            lay.addWidget(s)
            lay.addWidget(val)
            return lab, s, val

        add_slider("硬度", "hardness", self.hardness, "0=极软边，100=硬边")
        add_slider("不透明度", "opacity", self.opacity, "单次描边累计的最大浓度")
        add_slider("流量", "flow", self.flow, "每次落笔的量")
        add_slider("平滑", "smoothing", self.smoothing, "笔画抖动抑制")

        self.lab_target = QLabel("目标")
        self.combo_target = QComboBox()
        self.combo_target.addItems(["像素", "蒙版"])
        self.combo_target.setMaximumWidth(78)
        self.combo_target.currentIndexChanged.connect(
            lambda i: setattr(self, "target", "mask" if i == 1 else "pixel"))
        lay.addWidget(self.lab_target)
        lay.addWidget(self.combo_target)
        self.act_brush = self.addWidget(self.grp_brush)

    def _build_text(self):
        """文字工具的默认字体设置（新建一个文字图层时用这里的字）。"""
        self.grp_text = QWidget()
        lay = QHBoxLayout(self.grp_text)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        self.font_combo = QFontComboBox()
        self.font_combo.setMaximumWidth(190)
        self.font_combo.currentFontChanged.connect(
            lambda f: setattr(self, "font_family", f.family()))
        lay.addWidget(QLabel("字体"))
        lay.addWidget(self.font_combo)

        self.spin_font_size = QSpinBox()
        self.spin_font_size.setRange(1, 2000)
        self.spin_font_size.setValue(self.font_size)
        self.spin_font_size.setMaximumWidth(70)
        self.spin_font_size.valueChanged.connect(
            lambda v: setattr(self, "font_size", int(v)))
        lay.addWidget(QLabel("字号"))
        lay.addWidget(self.spin_font_size)

        cb = QCheckBox("粗体")
        cb.setChecked(self.font_bold)
        cb.toggled.connect(lambda v: setattr(self, "font_bold", bool(v)))
        lay.addWidget(cb)

        self.text_swatch = ColorSwatch((0, 0, 0), "文字颜色")
        self.text_swatch.clickedColor.connect(
            lambda: self._pick(self.text_swatch))
        lay.addWidget(QLabel("颜色"))
        lay.addWidget(self.text_swatch)
        self.act_text = self.addWidget(self.grp_text)

    def _build_colors(self):
        self.fg = ColorSwatch((0, 0, 0), "前景色")
        self.bg = ColorSwatch((255, 255, 255), "背景色")
        self.fg.clickedColor.connect(lambda: self._pick(self.fg))
        self.bg.clickedColor.connect(lambda: self._pick(self.bg))
        self.act_fg = self.addWidget(self.fg)
        self.act_bg = self.addWidget(self.bg)
        btn = QPushButton("交换 (X)")
        btn.setFixedHeight(26)
        btn.clicked.connect(self.swap_colors)
        self.act_swap = self.addWidget(btn)

    # ---------- 行为 ----------

    def _set_mode(self, mode):
        self.sel_mode = mode

    def _pick(self, swatch):
        c = QColorDialog.getColor(swatch.color(), self, "选择颜色")
        if c.isValid():
            swatch.set_color((c.red(), c.green(), c.blue()))

    def swap_colors(self):
        a, b = self.fg.rgb(), self.bg.rgb()
        self.fg.set_color(b)
        self.bg.set_color(a)

    def text_params(self):
        """新建文字图层用的参数：工具选项条上的字体 / 字号 / 粗体 / 颜色。"""
        return {
            "family": self.font_family,
            "size": float(self.font_size),
            "bold": bool(self.font_bold),
            "color": list(self.text_swatch.rgb()),
        }

    def set_tool(self, tool):
        """注意：QToolBar 里必须隐藏 action，直接 setVisible 在小组件上会被工具栏覆盖。"""
        self.current_tool = tool
        is_sel = tool in SELECT_TOOLS
        is_paint = tool in PAINT_TOOLS
        is_text = tool in TEXT_TOOLS
        self.act_sel.setVisible(is_sel)
        self.act_brush.setVisible(is_paint)
        self.act_text.setVisible(is_text)
        self.lab_tol.setVisible(tool == "magic")
        self.spin_tol.setVisible(tool == "magic")
        self.act_sep1.setVisible(is_sel)
        self.act_sep2.setVisible(is_paint or is_text)
        need_color = is_paint or tool == "fill"
        for a in (self.act_fg, self.act_bg, self.act_swap):
            a.setVisible(need_color)
        self.act_hint.setVisible(tool == "move")

    def sync_target_availability(self, has_mask, only_mask=False):
        """当前图层没有蒙版时，禁止把画笔目标设成蒙版。

        only_mask 用于调整层：它没有像素，只能画蒙版。
        """
        self.combo_target.setEnabled(has_mask and not only_mask)
        if only_mask:
            self.combo_target.setCurrentIndex(1)   # 0=像素 1=蒙版
            self.target = "mask"
            return
        if not has_mask:
            self.combo_target.setCurrentIndex(0)
            self.target = "pixel"
