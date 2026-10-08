# -*- coding: utf-8 -*-
"""顶部工具选项条：随当前工具切换显示的参数。"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QColorDialog,
                               QComboBox, QDialog, QDoubleSpinBox,
                               QFontComboBox, QHBoxLayout, QLabel,
                               QPushButton, QSlider, QSpinBox, QToolBar,
                               QWidget)

from ..core.selection import ADD, INTERSECT, REPLACE, SUBTRACT
from ..core.brush import BRUSH_DEFAULTS

SEL_MODES = [
    ("新选区", REPLACE),
    ("添加到", ADD),
    ("从减去", SUBTRACT),
    ("与交叉", INTERSECT),
]

SELECT_TOOLS = {"rect", "ellipse", "lasso", "polygon", "magic"}
PAINT_TOOLS = {"brush", "eraser"}
TEXT_TOOLS = {"text"}
# 修饰类工具：参数各不相同，各自有独立的参数组
GRADIENT_TOOLS = {"gradient"}
BLUR_TOOLS = {"blurtool"}
PICK_TOOLS = {"picker"}
# 第二十批：克隆图章（Alt 采样）、污点修复、形状工具
CLONE_TOOLS = {"clone", "heal"}
SHAPE_TOOLS = {"shape"}

# 形状工具的下拉项：矢量图层模式多两种（多边形 / 星形），
# 它们要"顶点参数"才能画，像素模式（往位图上涂）给不了
SHAPE_VECTOR_ITEMS = ["矩形", "椭圆", "直线", "多边形", "星形"]
SHAPE_PIXEL_ITEMS = ["矩形", "椭圆", "直线"]
_SHAPE_KEYS = {"矩形": "rect", "椭圆": "ellipse", "直线": "line",
               "多边形": "polygon", "星形": "polygon"}


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
        # 进阶笔刷参数（形状/间隔/散布/纹理/喷枪/压感），在「笔刷设置」对话框里改
        self.brush = dict(BRUSH_DEFAULTS)

        # 新建文字图层时的默认字体设置
        self.font_family = ""
        self.font_size = 96
        self.font_bold = False

        # 渐变工具：样式 + 两个色标（前景色 → 背景色，由 fg/bg 同步过来）
        self.grad_style = "线性"
        self.grad_reverse = False

        # 模糊工具：复用画笔的「大小 / 硬度」，另加每次落笔的量
        self.blur_strength = 0.5
        # 吸管：取样半径（0 = 单点），是否从合成结果取
        self.pick_radius = 0
        self.pick_composite = True

        # 克隆图章：Alt 采样的偏移（源坐标 - 当前落点），按住 Alt 重新采样
        self.clone_offset = None
        # 形状工具：绘制模式 + 形状，填充用前景色、描边用背景色
        # mode = "shape" 建**矢量形状图层**；"pixel" 直接画在当前图层的像素上
        self.shape_mode = "shape"
        self.shape_kind = "矩形"
        self.shape_outline = False
        self.shape_fill = True
        self.shape_stroke_w = 4.0

        self._build_hint()
        self._build_select()
        self.act_sep1 = self._add_sep()
        self._build_brush()
        self._build_text()
        self._build_gradient()
        self._build_blur()
        self._build_pick()
        self._build_clone()
        self._build_shape()
        self.act_sep2 = self._add_sep()
        self._build_colors()

        self.current_tool = "move"
        self.refresh_brush_summary()
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

        # 进阶参数（形状 / 间隔 / 散布 / 纹理 / 喷枪 / 压感）收进对话框
        self.btn_brush = QPushButton("笔刷…")
        self.btn_brush.setFixedHeight(26)
        self.btn_brush.setToolTip("笔尖形状、间隔、散布、纹理、喷枪、压感")
        self.btn_brush.clicked.connect(self.open_brush_dialog)
        lay.addWidget(self.btn_brush)
        self.lab_brush = QLabel()
        self.lab_brush.setStyleSheet("color:#9a9a9e;")
        lay.addWidget(self.lab_brush)

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

    def _build_gradient(self):
        """渐变工具：样式 + 反向。颜色用通用的前景/背景色（fg → bg）。"""
        from ..core.retouch import GRADIENT_STYLES
        self.grp_gradient = QWidget()
        lay = QHBoxLayout(self.grp_gradient)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        self.combo_grad = QComboBox()
        self.combo_grad.addItems(list(GRADIENT_STYLES))
        self.combo_grad.setMaximumWidth(80)
        self.combo_grad.setToolTip("渐变样式：线性 / 径向 / 角度 / 对称 / 菱形")
        self.combo_grad.currentTextChanged.connect(
            lambda s: setattr(self, "grad_style", s))
        lay.addWidget(QLabel("样式"))
        lay.addWidget(self.combo_grad)

        cb = QCheckBox("反向")
        cb.setToolTip("交换两个色标：背景色 → 前景色")
        cb.toggled.connect(lambda v: setattr(self, "grad_reverse", bool(v)))
        self.chk_grad_rev = cb
        lay.addWidget(cb)

        self.lab_grad_hint = QLabel("拖出一条线确定渐变方向与范围")
        self.lab_grad_hint.setStyleSheet("color:#9a9a9e;")
        lay.addWidget(self.lab_grad_hint)
        self.act_gradient = self.addWidget(self.grp_gradient)

    def _build_blur(self):
        """模糊工具：大小 / 硬度复用画笔那套，另加「强度」。"""
        self.grp_blur = QWidget()
        lay = QHBoxLayout(self.grp_blur)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        self.spin_blur_size = QSpinBox()
        self.spin_blur_size.setRange(1, 2000)
        self.spin_blur_size.setValue(self.size)
        self.spin_blur_size.setMaximumWidth(70)
        self.spin_blur_size.valueChanged.connect(
            lambda v: setattr(self, "size", int(v)))
        lay.addWidget(QLabel("大小"))
        lay.addWidget(self.spin_blur_size)

        lay.addWidget(QLabel("硬度"))
        self.spin_blur_hard = _slider(100, int(round(self.hardness * 100)))
        self._val_blur_hard = QLabel("%d%%" % int(round(self.hardness * 100)))
        self._val_blur_hard.setFixedWidth(38)
        self._val_blur_hard.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.spin_blur_hard.valueChanged.connect(self._on_blur_hard)
        lay.addWidget(self.spin_blur_hard)
        lay.addWidget(self._val_blur_hard)

        lay.addWidget(QLabel("强度"))
        self.spin_blur_str = _slider(100, int(round(self.blur_strength * 100)))
        self._val_blur_str = QLabel("%d%%" % int(round(self.blur_strength * 100)))
        self._val_blur_str.setFixedWidth(38)
        self._val_blur_str.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.spin_blur_str.valueChanged.connect(self._on_blur_str)
        lay.addWidget(self.spin_blur_str)
        lay.addWidget(self._val_blur_str)
        self.act_blur = self.addWidget(self.grp_blur)

    def _on_blur_hard(self, v):
        self.hardness = v / 100.0
        self._val_blur_hard.setText("%d%%" % v)

    def _on_blur_str(self, v):
        self.blur_strength = v / 100.0
        self._val_blur_str.setText("%d%%" % v)

    def _build_pick(self):
        """吸管：取样半径 + 取样来源。"""
        self.grp_pick = QWidget()
        lay = QHBoxLayout(self.grp_pick)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        lay.addWidget(QLabel("取样"))
        self.spin_pick_rad = QSpinBox()
        self.spin_pick_rad.setRange(0, 64)
        self.spin_pick_rad.setSuffix(" px")
        self.spin_pick_rad.setMaximumWidth(72)
        self.spin_pick_rad.setToolTip("0 = 只取点击的那一个像素；>0 取邻域均值，能避开杂点")
        self.spin_pick_rad.valueChanged.connect(
            lambda v: setattr(self, "pick_radius", int(v)))
        lay.addWidget(self.spin_pick_rad)

        cb = QCheckBox("所有图层")
        cb.setChecked(True)
        cb.setToolTip("勾 = 从合成结果取色；不勾 = 只从当前图层取")
        cb.toggled.connect(lambda v: setattr(self, "pick_composite", bool(v)))
        self.chk_pick_all = cb
        lay.addWidget(cb)
        self.act_pick = self.addWidget(self.grp_pick)

    def _build_clone(self):
        """克隆图章 / 污点修复：共用「大小 / 硬度 / 强度」。"""
        self.grp_clone = QWidget()
        lay = QHBoxLayout(self.grp_clone)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        self.spin_clone_size = QSpinBox()
        self.spin_clone_size.setRange(1, 2000)
        self.spin_clone_size.setValue(self.size)
        self.spin_clone_size.setMaximumWidth(70)
        self.spin_clone_size.valueChanged.connect(
            lambda v: setattr(self, "size", int(v)))
        lay.addWidget(QLabel("大小"))
        lay.addWidget(self.spin_clone_size)

        lay.addWidget(QLabel("硬度"))
        self.spin_clone_hard = _slider(100, int(round(self.hardness * 100)))
        v1 = QLabel("%d%%" % int(round(self.hardness * 100)))
        v1.setFixedWidth(38)
        v1.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.spin_clone_hard.valueChanged.connect(
            lambda v: self._set_pair("hardness", v, v1))
        lay.addWidget(self.spin_clone_hard)
        lay.addWidget(v1)

        lay.addWidget(QLabel("强度"))
        self.spin_clone_str = _slider(100, int(round(self.blur_strength * 100)))
        v2 = QLabel("%d%%" % int(round(self.blur_strength * 100)))
        v2.setFixedWidth(38)
        v2.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.spin_clone_str.valueChanged.connect(
            lambda v: self._set_pair("strength", v, v2))
        lay.addWidget(self.spin_clone_str)
        lay.addWidget(v2)

        self.lab_clone_hint = QLabel("按住 Alt 采样，松开拖动盖章")
        self.lab_clone_hint.setStyleSheet("color:#9a9a9e;")
        lay.addWidget(self.lab_clone_hint)
        self.act_clone = self.addWidget(self.grp_clone)

    def _set_pair(self, attr, v, label):
        if attr == "strength":
            self.blur_strength = v / 100.0
        else:
            self.hardness = v / 100.0
        label.setText("%d%%" % v)

    def _build_shape(self):
        """形状工具：绘制模式（矢量图层 / 像素）+ 形状 + 填充 · 描边。"""
        self.grp_shape = QWidget()
        lay = QHBoxLayout(self.grp_shape)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        self.combo_shape_mode = QComboBox()
        self.combo_shape_mode.addItems(["矢量图层", "像素"])
        self.combo_shape_mode.setMaximumWidth(90)
        self.combo_shape_mode.setToolTip(
            "矢量图层 —— 建一个可编辑的形状图层：放大缩小不糊，"
            "颜色 / 圆角 / 边数随时改\n"
            "像素 —— 直接画在当前图层的像素上（画完不能再改）")
        self.combo_shape_mode.currentTextChanged.connect(self._on_shape_mode)
        lay.addWidget(QLabel("绘制为"))
        lay.addWidget(self.combo_shape_mode)

        self.combo_shape = QComboBox()
        self.combo_shape.setMaximumWidth(80)
        self.combo_shape.currentTextChanged.connect(
            lambda s: setattr(self, "shape_kind", s))
        lay.addWidget(self.combo_shape)

        cb = QCheckBox("描边")
        cb.setToolTip("矢量图层：画一圈边框（用背景色）；像素：只画边框，内部不动")
        cb.toggled.connect(lambda v: setattr(self, "shape_outline", bool(v)))
        self.chk_shape_outline = cb
        lay.addWidget(cb)

        cbf = QCheckBox("填充")
        cbf.setChecked(True)
        cbf.setToolTip("矢量图层：形状内部填前景色")
        cbf.toggled.connect(lambda v: setattr(self, "shape_fill", bool(v)))
        self.chk_shape_fill = cbf
        lay.addWidget(cbf)

        self.spin_shape_stroke = QSpinBox()
        self.spin_shape_stroke.setRange(1, 400)
        self.spin_shape_stroke.setValue(int(self.shape_stroke_w))
        self.spin_shape_stroke.setMaximumWidth(58)
        self.spin_shape_stroke.setToolTip("矢量图层的描边宽度（像素）")
        self.spin_shape_stroke.valueChanged.connect(
            lambda v: setattr(self, "shape_stroke_w", float(v)))
        lay.addWidget(QLabel("描边宽"))
        lay.addWidget(self.spin_shape_stroke)

        self.lab_shape_hint = QLabel("拖出形状，Shift 等比")
        self.lab_shape_hint.setStyleSheet("color:#9a9a9e;")
        lay.addWidget(self.lab_shape_hint)
        self.act_shape = self.addWidget(self.grp_shape)

        self._on_shape_mode(self.combo_shape_mode.currentText())

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

    # ---------- 修饰类工具 ----------

    def gradient_params(self):
        """渐变工具参数。两个色标直接取前景色 / 背景色（PS 的语义）。"""
        a, b = self.fg.rgb(), self.bg.rgb()
        if self.grad_reverse:
            a, b = b, a
        return {"style": self.grad_style,
                "color": list(a), "color2": list(b)}

    def sync_gradient_hint(self):
        """切到渐变工具时刷新那句提示（颜色变了要重画色标）。"""
        self.lab_grad_hint.setText(
            "拖出一条线确定渐变方向与范围 · %s → %s"
            % (self.gradient_params()["color"], self.gradient_params()["color2"]))

    def blur_params(self):
        return {"size": float(self.size),
                "hardness": float(self.hardness),
                "strength": float(self.blur_strength)}

    def pick_params(self):
        return {"radius": int(self.pick_radius),
                "composite": bool(self.pick_composite)}

    # ---------- 第二十批 ----------

    def clone_params(self):
        """克隆图章 / 污点修复的共用参数。"""
        return {"size": float(self.size),
                "hardness": float(self.hardness),
                "strength": float(self.blur_strength)}

    def shape_params(self):
        p = {"kind": self.shape_kind, "outline": bool(self.shape_outline)}
        return p

    # ---------- 矢量形状图层（第二十六批）----------

    def _on_shape_mode(self, text):
        """切换「矢量图层 / 像素」：可选形状不同，给矢量模式补上专属控件。"""
        self.shape_mode = "pixel" if text == "像素" else "shape"
        vec = self.shape_mode == "shape"
        items = SHAPE_VECTOR_ITEMS if vec else SHAPE_PIXEL_ITEMS
        cur = self.shape_kind if self.shape_kind in items else items[0]
        self.combo_shape.blockSignals(True)
        self.combo_shape.clear()
        self.combo_shape.addItems(items)
        self.combo_shape.setCurrentText(cur)
        self.combo_shape.blockSignals(False)
        self.shape_kind = cur
        self.chk_shape_fill.setEnabled(vec)
        self.spin_shape_stroke.setEnabled(vec)
        self.lab_shape_hint.setText(
            "拖出形状，Shift 等比 · 填充=前景色，描边=背景色"
            if vec else "拖出形状，Shift 等比")

    def shape_layer_params(self, box):
        """矢量形状图层的参数：形状 + 填充（前景色）+ 描边（背景色）。

        `box` 是画布坐标的 (x0, y0, x1, y1)。注意形状参数里的几何量是
        **局部坐标**，渲染时会先平移到原点再画（见 core.shape）。
        """
        lab = self.shape_kind
        kind = _SHAPE_KEYS.get(lab, "rect")
        p = {
            "kind": kind,
            "box": [float(box[0]), float(box[1]),
                    float(box[2]), float(box[3])],
            "fill": {"on": bool(self.shape_fill),
                     "color": list(self.fg.rgb()), "opacity": 1.0},
            "stroke": {"on": bool(self.shape_outline),
                       "color": list(self.bg.rgb()),
                       "width": float(self.shape_stroke_w),
                       "opacity": 1.0},
        }
        if lab == "星形":
            p["star"] = 0.45
        if kind == "line":
            # 直线没有"内部"，描边就是它本身 —— 强制开描边、关填充
            p["stroke"]["on"] = True
            p["fill"]["on"] = False
        return p

    # ---------- 笔刷 ----------

    def brush_params(self):
        """进阶笔刷参数（给 canvas_view 建 Stroke 用）。"""
        return dict(self.brush)

    def refresh_brush_summary(self):
        from .brush_dialog import summary_text
        self.lab_brush.setText(summary_text(self.brush))

    def open_brush_dialog(self):
        from .brush_dialog import BrushDialog
        dlg = BrushDialog(self, self.brush, size=self.size,
                          hardness=self.hardness)
        if dlg.exec() != QDialog.Accepted:
            return False
        self.brush = dlg.result_params()
        self.refresh_brush_summary()
        return True

    def set_tool(self, tool):
        """注意：QToolBar 里必须隐藏 action，直接 setVisible 在小组件上会被工具栏覆盖。"""
        self.current_tool = tool
        is_sel = tool in SELECT_TOOLS
        is_paint = tool in PAINT_TOOLS
        is_text = tool in TEXT_TOOLS
        is_grad = tool in GRADIENT_TOOLS
        is_blur = tool in BLUR_TOOLS
        is_pick = tool in PICK_TOOLS
        is_clone = tool in CLONE_TOOLS
        is_shape = tool in SHAPE_TOOLS
        self.act_sel.setVisible(is_sel)
        self.act_brush.setVisible(is_paint)
        self.act_text.setVisible(is_text)
        self.act_gradient.setVisible(is_grad)
        self.act_blur.setVisible(is_blur)
        self.act_pick.setVisible(is_pick)
        self.act_clone.setVisible(is_clone)
        self.act_shape.setVisible(is_shape)
        self.lab_tol.setVisible(tool == "magic")
        self.spin_tol.setVisible(tool == "magic")
        self.act_sep1.setVisible(is_sel)
        self.act_sep2.setVisible(is_paint or is_text)
        # 渐变/模糊要前景色背景色（模糊只要前景色作提示，渐变两个都要）
        need_color = (is_paint or tool == "fill" or is_grad or is_blur
                      or is_shape)
        for a in (self.act_fg, self.act_bg, self.act_swap):
            a.setVisible(need_color)
        if is_grad:
            self.sync_gradient_hint()
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
