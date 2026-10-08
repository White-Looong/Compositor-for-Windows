# -*- coding: utf-8 -*-
"""属性面板：变换数值、文字参数、画布尺寸、图层信息。"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox,
                               QFontComboBox, QFormLayout, QGroupBox,
                               QHBoxLayout, QLabel, QPlainTextEdit,
                               QPushButton, QSpinBox, QVBoxLayout, QWidget)

from ..core.shape import SHAPE_ITEMS, SHAPE_LABELS
from ..core.text import ALIGN_CENTER, ALIGN_ITEMS, PARA_ITEMS
from .adjust_panel import AdjustPanel
from .tool_options import ColorSwatch


class Inspector(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self._syncing = False

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # ---- 变换 ----
        self.g1 = g1 = QGroupBox("变换")
        f1 = QFormLayout(g1)
        f1.setLabelAlignment(Qt.AlignRight)
        self.x = self._dbl(-100000, 100000, 1)
        self.y = self._dbl(-100000, 100000, 1)
        self.w = self._dbl(0.01, 100000, 1)
        self.h = self._dbl(0.01, 100000, 1)
        self.rot = self._dbl(-360, 360, 1)
        for widget, label in ((self.x, "X"), (self.y, "Y"),
                              (self.w, "宽"), (self.h, "高"),
                              (self.rot, "角度")):
            f1.addRow(label, widget)
        for widget in (self.x, self.y, self.w, self.h, self.rot):
            widget.valueChanged.connect(self._on_transform)
            widget.editingFinished.connect(self._on_transform_done)

        btns = QHBoxLayout()
        self.btn_fh = QPushButton("水平翻转")
        self.btn_fv = QPushButton("垂直翻转")
        self.btn_reset = QPushButton("复位")
        self.btn_fh.clicked.connect(self._flip_h)
        self.btn_fv.clicked.connect(self._flip_v)
        self.btn_reset.clicked.connect(self._reset)
        for b in (self.btn_fh, self.btn_fv, self.btn_reset):
            btns.addWidget(b)
        f1.addRow(btns)
        root.addWidget(g1)

        # ---- 智能对象（只有选中智能对象图层时才显示）----
        self.g_smart = QGroupBox("智能对象")
        fs = QFormLayout(self.g_smart)
        fs.setLabelAlignment(Qt.AlignRight)
        self.smart_info = QLabel("-")
        self.smart_info.setWordWrap(True)
        self.btn_smart_edit = QPushButton("编辑内容…")
        self.btn_smart_edit.setToolTip(
            "在独立窗口里改这段内容 —— 所有共用它的实例会一起更新")
        self.btn_smart_filters = QPushButton("智能滤镜…")
        self.btn_smart_filters.setToolTip(
            "参数化的滤镜：随时改参数、调顺序、删掉，原始内容不受影响")
        self.btn_smart_raster = QPushButton("栅格化")
        self.btn_smart_raster.setToolTip(
            "烧成普通位图图层 —— 内容与智能滤镜从此不可再改")
        self.btn_smart_edit.clicked.connect(self.main.edit_smart_content)
        self.btn_smart_filters.clicked.connect(self.main.edit_smart_filters)
        self.btn_smart_raster.clicked.connect(self.main.rasterize_smart)
        fs.addRow(self.smart_info)
        fs.addRow(self.btn_smart_edit)
        fs.addRow(self.btn_smart_filters)
        fs.addRow(self.btn_smart_raster)
        root.addWidget(self.g_smart)

        # ---- 图层样式 ----
        self.g_style = QGroupBox("图层样式")
        fst = QFormLayout(self.g_style)
        fst.setLabelAlignment(Qt.AlignRight)
        self.lab_style = QLabel("无")
        self.lab_style.setWordWrap(True)
        self.btn_style = QPushButton("图层样式…")
        self.btn_style.setToolTip(
            "描边 / 投影 / 内阴影 / 外发光 —— 参数存在图层上，随时可改")
        self.btn_style.clicked.connect(self._open_styles)
        fst.addRow(self.lab_style)
        fst.addRow(self.btn_style)
        root.addWidget(self.g_style)

        # ---- 调整层参数（只有选中调整层时才显示）----
        self.adjust = AdjustPanel(main)
        root.addWidget(self.adjust)

        # ---- 矢量形状（只有选中形状图层时才显示）----
        self.g_shape = QGroupBox("形状")
        fsh = QFormLayout(self.g_shape)
        fsh.setLabelAlignment(Qt.AlignRight)

        self.combo_shape = QComboBox()
        self.combo_shape.addItems([lb for _k, lb in SHAPE_ITEMS])
        self.combo_shape.setMaximumWidth(118)
        self.combo_shape.setToolTip("改形状种类：矩形的圆角、多边形的边数"
                                    "在下面")
        self.combo_shape.currentIndexChanged.connect(self._on_shape_kind)
        fsh.addRow("种类", self.combo_shape)

        self.chk_fill = QCheckBox("填充")
        self.chk_fill.toggled.connect(self._on_shape_fill)
        self.sw_fill = ColorSwatch((217, 83, 79), "填充颜色")
        self.sw_fill.clickedColor.connect(self._pick_shape_fill)
        row_fill = QHBoxLayout()
        row_fill.addWidget(self.chk_fill)
        row_fill.addWidget(self.sw_fill)
        fsh.addRow("填充", row_fill)

        self.chk_stroke = QCheckBox("描边")
        self.chk_stroke.toggled.connect(self._on_shape_stroke)
        self.sw_stroke = ColorSwatch((38, 38, 38), "描边颜色")
        self.sw_stroke.clickedColor.connect(self._pick_shape_stroke)
        row_stk = QHBoxLayout()
        row_stk.addWidget(self.chk_stroke)
        row_stk.addWidget(self.sw_stroke)
        fsh.addRow("描边", row_stk)

        self.spin_stroke_w = self._dbl(0.0, 400.0, 1.0)
        self.spin_stroke_w.valueChanged.connect(
            lambda v: self._set_shape_style("stroke", "width", float(v)))
        fsh.addRow("描边宽", self.spin_stroke_w)

        self.spin_radius = self._dbl(0.0, 2000.0, 2.0)
        self.spin_radius.setToolTip("矩形的圆角半径（像素）")
        self.spin_radius.valueChanged.connect(
            lambda v: self._set_shape_param("radius", float(v)))
        fsh.addRow("圆角", self.spin_radius)

        self.spin_sides = self._dbl(3, 60, 1)
        self.spin_sides.setDecimals(0)
        self.spin_sides.setToolTip("多边形的边数")
        self.spin_sides.valueChanged.connect(
            lambda v: self._set_shape_param("sides", int(round(v))))
        fsh.addRow("边数", self.spin_sides)

        self.spin_star = self._dbl(0, 90, 5)
        self.spin_star.setDecimals(0)
        self.spin_star.setToolTip("星形的内缩比：0 = 普通多边形，"
                                  "越大角越尖（%）")
        self.spin_star.valueChanged.connect(
            lambda v: self._set_shape_param("star",
                                            max(0.0, min(0.9, v / 100.0))))
        fsh.addRow("星形", self.spin_star)

        self.btn_raster_shape = QPushButton("栅格化")
        self.btn_raster_shape.setToolTip(
            "转成普通位图图层 —— 之后才能用画笔 / 滤镜改像素，但形状不能再改")
        self.btn_raster_shape.clicked.connect(self.main.rasterize_shape)
        fsh.addRow(self.btn_raster_shape)
        root.addWidget(self.g_shape)

        # ---- 文字（只有选中文字图层时才显示）----
        self.g_text = QGroupBox("文字")
        ft = QFormLayout(self.g_text)
        ft.setLabelAlignment(Qt.AlignRight)

        self.txt_content = QPlainTextEdit()
        self.txt_content.setFixedHeight(70)
        self.txt_content.textChanged.connect(self._on_text_content)
        ft.addRow("内容", self.txt_content)

        self.txt_font = QFontComboBox()
        self.txt_font.setMaximumWidth(150)
        self.txt_font.currentFontChanged.connect(self._on_text_font)
        ft.addRow("字体", self.txt_font)

        self.txt_size = QSpinBox()
        self.txt_size.setRange(1, 2000)
        self.txt_size.setSingleStep(1)
        self.txt_size.setMaximumWidth(118)
        self.txt_size.valueChanged.connect(
            lambda v: self._set_text("size", float(v)))
        ft.addRow("字号", self.txt_size)

        self.txt_color = ColorSwatch((0, 0, 0), "文字颜色")
        self.txt_color.clickedColor.connect(self._pick_text_color)
        ft.addRow("颜色", self.txt_color)

        self.txt_lh = self._dbl(0.5, 5.0, 0.05)
        self.txt_ls = self._dbl(-50.0, 200.0, 0.5)
        self.txt_lh.valueChanged.connect(
            lambda v: self._set_text("line_height", float(v)))
        self.txt_ls.valueChanged.connect(
            lambda v: self._set_text("letter_spacing", float(v)))
        ft.addRow("行距", self.txt_lh)
        ft.addRow("字距", self.txt_ls)

        self.txt_align = QComboBox()
        self.txt_align.addItems([lb for _k, lb in ALIGN_ITEMS])
        self.txt_align.setMaximumWidth(118)
        self.txt_align.currentIndexChanged.connect(self._on_text_align)
        ft.addRow("对齐", self.txt_align)

        # ---- 段落（第十五批）：缩进与段间距，单位是像素 ----
        self.para_spins = {}
        for key, label in PARA_ITEMS:
            sp = QSpinBox()
            sp.setRange(-2000, 2000)
            sp.setSingleStep(2)
            sp.setMaximumWidth(118)
            sp.setToolTip("像素。段前 / 段后距只作用于段落之间"
                          "（首段之前、末段之后不留）")
            sp.valueChanged.connect(
                lambda v, k=key: self._set_para(k, float(v)))
            self.para_spins[key] = sp
            ft.addRow(label, sp)

        style_row = QHBoxLayout()
        self.cb_bold = QCheckBox("粗体")
        self.cb_italic = QCheckBox("斜体")
        self.cb_under = QCheckBox("下划线")
        for cb, key in ((self.cb_bold, "bold"),
                        (self.cb_italic, "italic"),
                        (self.cb_under, "underline")):
            cb.toggled.connect(lambda v, k=key: self._set_text(k, bool(v)))
            style_row.addWidget(cb)
        ft.addRow(style_row)

        edit_row = QHBoxLayout()
        self.btn_canvas_edit = QPushButton("在画布上编辑")
        self.btn_canvas_edit.setToolTip("Ctrl+T —— 直接在画布上改字，边改边看")
        self.btn_canvas_edit.clicked.connect(self.main.edit_text_on_canvas)
        self.btn_perchar = QPushButton("逐字调整…")
        self.btn_perchar.setToolTip("给单个字加字距 / 抬基线 / 拉宽")
        self.btn_perchar.clicked.connect(self.main.edit_text_chars)
        edit_row.addWidget(self.btn_canvas_edit)
        edit_row.addWidget(self.btn_perchar)
        ft.addRow(edit_row)

        self.btn_raster = QPushButton("栅格化")
        self.btn_raster.setToolTip(
            "转成普通位图图层 —— 之后才能用画笔 / 滤镜改像素，但文字不能再改")
        self.btn_raster.clicked.connect(self.main.rasterize_text)
        ft.addRow(self.btn_raster)
        root.addWidget(self.g_text)

        # ---- 画布 ----
        g2 = QGroupBox("画布")
        f2 = QFormLayout(g2)
        f2.setLabelAlignment(Qt.AlignRight)
        self.cw = QSpinBox()
        self.ch = QSpinBox()
        for s in (self.cw, self.ch):
            s.setRange(1, 30000)
            s.setSingleStep(1)
            s.setMaximumWidth(118)
        self.cw.editingFinished.connect(self._on_canvas)
        self.ch.editingFinished.connect(self._on_canvas)
        f2.addRow("宽", self.cw)
        f2.addRow("高", self.ch)
        btn_fit = QPushButton("按内容裁剪")
        btn_fit.clicked.connect(self._trim_to_content)
        f2.addRow(btn_fit)
        root.addWidget(g2)

        # ---- 信息 ----
        g3 = QGroupBox("图层信息")
        f3 = QFormLayout(g3)
        f3.setLabelAlignment(Qt.AlignRight)
        self.info = QLabel("-")
        self.info.setWordWrap(True)
        f3.addRow(self.info)
        root.addWidget(g3)

        root.addStretch(1)

    def _dbl(self, lo, hi, step):
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(2)
        s.setSingleStep(step)
        s.setKeyboardTracking(True)
        s.setMaximumWidth(118)
        return s

    # ---------- 同步 ----------

    def refresh(self):
        self._syncing = True
        layer = self.main.selected_layer()
        is_adj = bool(layer is not None and layer.is_adjustment)
        # 调整层没有像素，变换面板对它没意义
        self.g1.setVisible(not is_adj)
        enable = layer is not None and not layer.is_group and not is_adj
        for widget in (self.x, self.y, self.w, self.h, self.rot,
                       self.btn_fh, self.btn_fv, self.btn_reset):
            widget.setEnabled(enable)
        self.adjust.refresh()
        is_text = bool(layer is not None and layer.is_text)
        self.g_text.setVisible(is_text)
        if is_text:
            self._fill_text(layer)
        is_shape = bool(layer is not None and layer.is_shape)
        self.g_shape.setVisible(is_shape)
        if is_shape:
            self._fill_shape(layer)
        self._fill_smart(layer)
        self._fill_style(layer)
        if is_adj:
            self.info.setText(
                "名称: %s\n类型: 调整层\n调整: %s\n蒙版: %s"
                % (layer.name, layer.adjustment_key(),
                   "有" if layer.mask is not None else "无"))
            doc = self.main.doc
            if doc:
                self.cw.setValue(doc.width)
                self.ch.setValue(doc.height)
            self._syncing = False
            return
        if layer is not None and not layer.is_group:
            src = layer.src_size or (0, 0)
            self.x.setValue(layer.tx)
            self.y.setValue(layer.ty)
            self.w.setValue(abs(src[0] * layer.sx))
            self.h.setValue(abs(src[1] * layer.sy))
            self.rot.setValue(layer.rot)
            p = layer.text or {}
            extra = ""
            if layer.is_text:
                extra = "\n字体: %s  %.0f px" % (
                    p.get("family") or "默认", float(p.get("size", 0.0)))
            elif layer.is_shape:
                sp = layer.shape or {}
                extra = "\n形状: %s（矢量，缩放不失真）" % (
                    SHAPE_LABELS.get(sp.get("kind"), sp.get("kind")),)
            self.info.setText(
                "名称: %s\n类型: %s\n源尺寸: %d x %d\n缩放: %.1f%% x %.1f%%\n"
                "蒙版: %s%s" % (layer.name,
                                "文字" if layer.is_text
                                else ("智能对象" if layer.is_smart
                                      else ("形状" if layer.is_shape
                                            else "位图")),
                                src[0], src[1],
                                layer.sx * 100, layer.sy * 100,
                                "有" if layer.mask is not None else "无",
                                extra))
        elif layer is not None:
            self.info.setText("名称: %s\n类型: 图层组\n子图层: %d"
                              % (layer.name, len(layer.children)))
        else:
            for widget in (self.x, self.y, self.w, self.h, self.rot):
                widget.setValue(0)
            self.info.setText("未选中图层")

        doc = self.main.doc
        if doc:
            self.cw.setValue(doc.width)
            self.ch.setValue(doc.height)
        self._syncing = False

    def _fill_smart(self, layer):
        is_smart = bool(layer is not None and layer.is_smart)
        self.g_smart.setVisible(is_smart)
        if not is_smart:
            return
        doc = self.main.doc
        content = (doc.smart_contents or {}).get(layer.so_id) if doc else None
        if content is None:
            self.smart_info.setText("内容已丢失")
            return
        try:
            from ..core.smart import content_instances
            n = len(content_instances(doc, layer.so_id))
        except Exception:
            n = 1
        nf = len(layer.so_filters or [])
        self.smart_info.setText(
            "内容: %d x %d\n实例: %d 个（共用同一份内容）\n智能滤镜: %d 条"
            % (content.width, content.height, n, nf))

    # ---------- 图层样式 ----------

    def _fill_style(self, layer):
        from ..core.effects import EFFECT_NAMES, EFFECT_ORDER
        on = bool(layer is not None and not layer.is_group
                  and not layer.is_adjustment)
        self.g_style.setVisible(on)
        if not on:
            return
        names = []
        eff = layer.effects or {}
        for key in EFFECT_ORDER:
            d = eff.get(key)
            if isinstance(d, dict) and d.get("enabled"):
                names.append(EFFECT_NAMES[key])
        self.lab_style.setText("、".join(names) if names else "无")

    def _open_styles(self):
        from .style_dialog import StyleDialog
        layer = self.main.selected_layer()
        if layer is None or layer.is_group or layer.is_adjustment:
            return
        dlg = StyleDialog(layer, self.main, self)
        dlg.exec()
        self._fill_style(layer)
        self.refresh()

    # ---------- 文字 ----------

    def _fill_text(self, layer):
        p = layer.text or {}
        self.txt_content.setPlainText(str(p.get("content", "")))
        fam = p.get("family") or ""
        if fam:
            idx = self.txt_font.findText(fam)
            if idx >= 0:
                self.txt_font.setCurrentIndex(idx)
        self.txt_size.setValue(int(round(float(p.get("size", 96.0)))))
        c = p.get("color") or (0, 0, 0)
        self.txt_color.set_color(
            (int(c[0]), int(c[1]), int(c[2])) if len(c) >= 3 else (0, 0, 0))
        self.txt_lh.setValue(float(p.get("line_height", 1.2)))
        self.txt_ls.setValue(float(p.get("letter_spacing", 0.0)))
        keys = [k for k, _lb in ALIGN_ITEMS]
        a = p.get("align", ALIGN_CENTER)
        self.txt_align.setCurrentIndex(
            keys.index(a) if a in keys else keys.index(ALIGN_CENTER))
        self.cb_bold.setChecked(bool(p.get("bold", False)))
        self.cb_italic.setChecked(bool(p.get("italic", False)))
        self.cb_under.setChecked(bool(p.get("underline", False)))
        para = p.get("para") or {}
        for key, sp in self.para_spins.items():
            try:
                v = int(round(float(para.get(key, 0.0) or 0.0)))
            except (TypeError, ValueError):
                v = 0
            sp.setValue(max(sp.minimum(), min(sp.maximum(), v)))

    def _set_text(self, key, value):
        if self._syncing:
            return
        self.main.set_text_param(key, value)

    # ---------- 矢量形状（第二十六批）----------

    def _fill_shape(self, layer):
        p = layer.shape or {}
        keys = [k for k, _lb in SHAPE_ITEMS]
        kind = p.get("kind", keys[0])
        self.combo_shape.setCurrentIndex(
            keys.index(kind) if kind in keys else 0)
        fl = p.get("fill") or {}
        st = p.get("stroke") or {}
        self.chk_fill.setChecked(bool(fl.get("on", False)))
        self.chk_stroke.setChecked(bool(st.get("on", False)))
        c = fl.get("color") or (217, 83, 79)
        self.sw_fill.set_color(
            (int(c[0]), int(c[1]), int(c[2])) if len(c) >= 3 else (0, 0, 0))
        c = st.get("color") or (38, 38, 38)
        self.sw_stroke.set_color(
            (int(c[0]), int(c[1]), int(c[2])) if len(c) >= 3 else (0, 0, 0))
        self.spin_stroke_w.setValue(float(st.get("width", 4.0) or 0.0))
        self.spin_radius.setValue(float(p.get("radius", 0.0) or 0.0))
        self.spin_sides.setValue(float(p.get("sides", 6) or 6))
        self.spin_star.setValue(
            round(float(p.get("star", 0.0) or 0.0) * 100.0))
        # 圆角只管矩形；边数 / 星形只管多边形（路径暂时不在面板里改）
        is_rect = kind == "rect"
        is_poly = kind == "polygon"
        self.spin_radius.setEnabled(is_rect)
        self.spin_sides.setEnabled(is_poly)
        self.spin_star.setEnabled(is_poly)
        self.chk_fill.setEnabled(kind != "line")

    def _set_shape_param(self, key, value):
        if self._syncing:
            return
        self.main.set_shape_param(key, value)

    def _set_shape_style(self, which, key, value):
        if self._syncing:
            return
        self.main.set_shape_style(which, key, value)

    def _on_shape_kind(self, index):
        if self._syncing:
            return
        if 0 <= index < len(SHAPE_ITEMS):
            self.main.set_shape_param("kind", SHAPE_ITEMS[index][0])
            self.refresh()

    def _on_shape_fill(self, v):
        if self._syncing:
            return
        self.main.set_shape_style("fill", "on", bool(v))

    def _on_shape_stroke(self, v):
        if self._syncing:
            return
        self.main.set_shape_style("stroke", "on", bool(v))

    def _pick_shape_fill(self):
        from PySide6.QtWidgets import QColorDialog
        c = QColorDialog.getColor(self.sw_fill.color(), self, "填充颜色")
        if c.isValid():
            self.sw_fill.set_color((c.red(), c.green(), c.blue()))
            self.main.set_shape_style(
                "fill", "color", [c.red(), c.green(), c.blue()])

    def _pick_shape_stroke(self):
        from PySide6.QtWidgets import QColorDialog
        c = QColorDialog.getColor(self.sw_stroke.color(), self, "描边颜色")
        if c.isValid():
            self.sw_stroke.set_color((c.red(), c.green(), c.blue()))
            self.main.set_shape_style(
                "stroke", "color", [c.red(), c.green(), c.blue()])

    def _set_para(self, key, value):
        """段落属性：一次写整份 dict（渲染那边按段落取值）。"""
        if self._syncing:
            return
        layer = self.main.selected_layer()
        if layer is None or not layer.is_text or layer.text is None:
            return
        para = dict(layer.text.get("para") or {})
        if float(para.get(key, 0.0) or 0.0) == float(value):
            return
        para[key] = float(value)
        self.main.set_text_param("para", para)

    def _on_text_content(self):
        if self._syncing:
            return
        self.main.set_text_param("content", self.txt_content.toPlainText())

    def _on_text_font(self, font):
        if self._syncing:
            return
        self.main.set_text_param("family", font.family())

    def _on_text_align(self, index):
        if self._syncing:
            return
        if 0 <= index < len(ALIGN_ITEMS):
            self.main.set_text_param("align", ALIGN_ITEMS[index][0])

    def _pick_text_color(self):
        from PySide6.QtWidgets import QColorDialog
        c = QColorDialog.getColor(self.txt_color.color(), self, "文字颜色")
        if c.isValid():
            self.txt_color.set_color((c.red(), c.green(), c.blue()))
            self.main.set_text_param(
                "color", [c.red(), c.green(), c.blue()])

    # ---------- 编辑 ----------

    def _on_transform(self, _v=None):
        if self._syncing:
            return
        layer = self.main.selected_layer()
        if layer is None or layer.is_group:
            return
        src = layer.src_size
        if not src:
            return
        sw, sh = src
        layer.tx = self.x.value()
        layer.ty = self.y.value()
        sx = self.w.value() / sw if sw else 1.0
        sy = self.h.value() / sh if sh else 1.0
        layer.sx = sx if abs(sx) > 1e-4 else 0.01
        layer.sy = sy if abs(sy) > 1e-4 else 0.01
        layer.rot = self.rot.value()
        self.main.request_render()
        self.main.view._update_handles()

    def _on_transform_done(self):
        if self._syncing:
            return
        self.main.commit("变换数值")

    def _flip_h(self):
        layer = self.main.selected_layer()
        if layer is None or layer.is_group:
            return
        layer.flip_h = not layer.flip_h
        self.main.commit("水平翻转")
        self.main.request_render(self.main.layer_dirty_rect(layer))

    def _flip_v(self):
        layer = self.main.selected_layer()
        if layer is None or layer.is_group:
            return
        layer.flip_v = not layer.flip_v
        self.main.commit("垂直翻转")
        self.main.request_render(self.main.layer_dirty_rect(layer))

    def _reset(self):
        layer = self.main.selected_layer()
        doc = self.main.doc
        if layer is None or layer.is_group or not doc:
            return
        layer.sx = 1.0
        layer.sy = 1.0
        layer.rot = 0.0
        layer.flip_h = False
        layer.flip_v = False
        layer.tx = doc.width / 2.0
        layer.ty = doc.height / 2.0
        self.main.commit("复位变换")
        self.main.request_render()
        self.refresh()

    def _on_canvas(self):
        if self._syncing:
            return
        doc = self.main.doc
        if not doc:
            return
        w, h = self.cw.value(), self.ch.value()
        if (w, h) != (doc.width, doc.height):
            doc.resize(w, h)
            self.main.commit("画布大小")
            self.main.request_render()
            self.main.view.update_selection_overlay()

    def _trim_to_content(self):
        self.main.trim_to_content()

    def on_rendered(self):
        """渲染完成后只刷新直方图，不重建控件。"""
        self.adjust.refresh_hist()
