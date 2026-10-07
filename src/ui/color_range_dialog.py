# -*- coding: utf-8 -*-
"""色彩范围对话框（PS 的「选择 > 色彩范围」）。

用法与滤镜对话框一样：**预览走缩略图，确定时才算全分辨率**。
大画布上每拖一次容差滑块就跑一遍全图的 HSV 变换会卡住手，所以缩略图
（长边 900 px）负责预览，「确定」时用原图重算一遍 —— 和滤镜预览降采样
（§5.23）是同一套路子。

吸管直接点在预览图上：点一下 = 重新取样，Shift = 加一个取样点，
Alt = 去掉一个取样点。这样对话框不用跟画布抢鼠标事件。
"""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog,
                               QDialogButtonBox, QFormLayout, QHBoxLayout,
                               QLabel, QPushButton, QSlider, QVBoxLayout,
                               QWidget)

from ..core.color_range import PRESETS, NEEDS_SAMPLE, range_mask
from ..core.selection import ADD, INTERSECT, REPLACE, SUBTRACT

PREVIEW_EDGE = 900          # 预览缩略图长边
VIEW_W, VIEW_H = 460, 340   # 预览控件尺寸

_MODES = [("新选区", REPLACE), ("添加到选区", ADD),
          ("从选区减去", SUBTRACT), ("与选区交叉", INTERSECT)]
_SHOWS = ["原图", "灰度", "黑色杂边", "白色杂边", "快速蒙版"]
_SHOW_KEYS = ["none", "gray", "black", "white", "quick"]


class _PreviewLabel(QLabel):
    """可点击取样的预览图。点下去报的是**原图坐标**。"""

    picked = Signal(object, object)     # (x, y) 原图坐标, modifiers

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setCursor(Qt.CrossCursor)
        self.setMinimumSize(VIEW_W, VIEW_H)
        self.setStyleSheet("background:#1c1c1e; border:1px solid #4a4a4e;")
        self._iw = 0
        self._ih = 0

    def set_image(self, arr):
        h, w = arr.shape[:2]
        self._iw, self._ih = w, h
        img = QImage(np.ascontiguousarray(arr).data, w, h, 4 * w,
                     QImage.Format_RGBA8888)
        self._buf = arr           # QImage 直接用 arr 的内存，得留着引用
        self.setPixmap(QPixmap.fromImage(img).scaled(
            max(1, self.width()), max(1, self.height()), Qt.KeepAspectRatio,
            Qt.SmoothTransformation))

    def _to_image(self, pos):
        if self._iw <= 0:
            return None
        lw, lh = max(1, self.width()), max(1, self.height())
        s = min(lw / self._iw, lh / self._ih)
        dw, dh = self._iw * s, self._ih * s
        x0 = (lw - dw) / 2.0
        y0 = (lh - dh) / 2.0
        x = (pos.x() - x0) / s
        y = (pos.y() - y0) / s
        if not (0 <= x < self._iw and 0 <= y < self._ih):
            return None
        return int(x), int(y)

    def mousePressEvent(self, event):
        pt = self._to_image(event.position())
        if pt is not None:
            self.picked.emit(pt, event.modifiers())
        super().mousePressEvent(event)


class ColorRangeDialog(QDialog):
    selectionReady = Signal(object, str)    # (uint8 遮罩, 选区运算 mode)

    def __init__(self, parent, rgb, current_sel=None):
        super().__init__(parent)
        self.setWindowTitle("色彩范围")
        self.full = np.ascontiguousarray(rgb)
        H, W = self.full.shape[:2]
        s = PREVIEW_EDGE / float(max(H, W))
        if s < 1.0:
            self.small = cv2.resize(self.full, (max(1, int(round(W * s))),
                                                max(1, int(round(H * s)))),
                                    interpolation=cv2.INTER_AREA)
        else:
            self.small = self.full
        self.scale = s
        # 默认取样点用画面中心的颜色，打开就有东西可看
        cy, cx = self.small.shape[0] // 2, self.small.shape[1] // 2
        self.samples = [tuple(int(v) for v in self.small[cy, cx])]
        self.invert = False
        self.show_key = "quick"

        root = QVBoxLayout(self)

        self.view = _PreviewLabel()
        self.view.picked.connect(self._on_pick)
        root.addWidget(self.view, 0, Qt.AlignCenter)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        form.setSpacing(6)

        self.combo_preset = QComboBox()
        for pid, name, _ in PRESETS:
            self.combo_preset.addItem(name, pid)
        self.combo_preset.currentIndexChanged.connect(self._on_preset)
        form.addRow("选择", self.combo_preset)

        frow = QWidget()
        fl = QHBoxLayout(frow)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.setSpacing(6)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 200)
        self.slider.setValue(40)
        self.slider.setMinimumWidth(160)
        self.lab_fuzz = QLabel("40")
        self.lab_fuzz.setFixedWidth(36)
        self.lab_fuzz.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.slider.valueChanged.connect(self._on_fuzz)
        fl.addWidget(self.slider, 1)
        fl.addWidget(self.lab_fuzz)
        form.addRow("颜色容差", frow)

        srow = QWidget()
        sl = QHBoxLayout(srow)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(6)
        self.btn_pick = QPushButton("取样")
        self.btn_add = QPushButton("+ 添加")
        self.btn_sub = QPushButton("- 减去")
        self.btn_clear = QPushButton("清空取样")
        for b, tip in ((self.btn_pick, "点预览图取一个颜色（替换原有取样）"),
                       (self.btn_add, "按住 Shift 点预览图 = 再加一个取样点"),
                       (self.btn_sub, "按住 Alt 点预览图 = 去掉一个取样点"),
                       (self.btn_clear, "清空所有取样点")):
            b.setToolTip(tip)
            sl.addWidget(b)
        self.btn_pick.clicked.connect(lambda: self._hint("点预览图取样"))
        self.btn_add.clicked.connect(lambda: self._hint("按住 Shift 点预览图"))
        self.btn_sub.clicked.connect(lambda: self._hint("按住 Alt 点预览图"))
        self.btn_clear.clicked.connect(self._clear_samples)
        form.addRow("吸管", srow)

        self.lab_samples = QLabel("")
        self.lab_samples.setStyleSheet("color:#9a9a9e;")
        form.addRow("", self.lab_samples)

        self.combo_show = QComboBox()
        self.combo_show.addItems(_SHOWS)
        self.combo_show.setCurrentIndex(_SHOW_KEYS.index(self.show_key))
        self.combo_show.currentIndexChanged.connect(self._refresh)
        form.addRow("选区预览", self.combo_show)

        self.combo_mode = QComboBox()
        for name, _ in _MODES:
            self.combo_mode.addItem(name)
        self.combo_mode.currentIndexChanged.connect(self._refresh)
        form.addRow("选区运算", self.combo_mode)

        self.chk_invert = QCheckBox("反相")
        self.chk_invert.stateChanged.connect(self._on_invert)
        form.addRow("", self.chk_invert)

        root.addLayout(form)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        root.addWidget(bb)

        self._on_preset()
        self._refresh()

    # ---------- 交互 ----------

    def _hint(self, text):
        self.lab_samples.setText(text)

    def _preset_id(self):
        return self.combo_preset.currentData()

    def _on_preset(self):
        pid = self._preset_id()
        need = pid in NEEDS_SAMPLE
        for b in (self.btn_pick, self.btn_add, self.btn_sub, self.btn_clear):
            b.setEnabled(need)
        self._refresh()

    def _on_fuzz(self, v):
        self.lab_fuzz.setText("%d" % v)
        self._refresh()

    def _on_invert(self, _s):
        self.invert = self.chk_invert.isChecked()
        self._refresh()

    def _clear_samples(self):
        self.samples = []
        self._refresh()

    def _on_pick(self, pt, modifiers):
        """预览图上点一下 -> 取样。Shift 加、Alt 减。"""
        x, y = pt
        col = tuple(int(v) for v in self.small[y, x])
        if modifiers & Qt.AltModifier:
            # 去掉最接近的那个取样点
            if self.samples:
                d = [max(abs(a - b) for a, b in zip(col, s))
                     for s in self.samples]
                self.samples.pop(int(np.argmin(d)))
        elif modifiers & Qt.ShiftModifier:
            if col not in self.samples:
                self.samples.append(col)
        else:
            self.samples = [col]
        self.combo_preset.setCurrentIndex(0)      # 取样 => 切到"取样颜色"
        self._refresh()

    # ---------- 预览 ----------

    def mask_small(self):
        return range_mask(self.small, self._preset_id(), self.samples,
                          float(self.slider.value()), invert=self.invert)

    def _refresh(self):
        m = self.mask_small()
        h, w = m.shape[:2]
        a = m.astype(np.float32) / 255.0
        key = _SHOW_KEYS[self.combo_show.currentIndex()]
        small = self.small
        if key == "none":
            out = np.concatenate([small, np.full((h, w, 1), 255, np.uint8)],
                                 axis=2)
        elif key == "gray":
            out = np.concatenate([m[..., None], m[..., None], m[..., None],
                                  np.full((h, w, 1), 255, np.uint8)], axis=2)
        elif key in ("black", "white"):
            base = np.full((h, w, 3), 0 if key == "black" else 255, np.uint8)
            mix = (small.astype(np.float32) * a[..., None]
                   + base.astype(np.float32) * (1.0 - a[..., None]))
            out = np.concatenate([np.clip(mix, 0, 255).astype(np.uint8),
                                  np.full((h, w, 1), 255, np.uint8)], axis=2)
        else:   # quick：红色盖住未选中的地方
            arr = np.zeros((h, w, 4), np.uint8)
            arr[..., 0] = 255
            arr[..., 3] = np.clip((255.0 - m) * 0.5, 0, 255).astype(np.uint8)
            over = arr[..., 3:4].astype(np.float32) / 255.0
            rgb = (np.array([255, 0, 0], np.float32) * over
                   + small.astype(np.float32) * (1.0 - over))
            out = np.concatenate([np.clip(rgb, 0, 255).astype(np.uint8),
                                  np.full((h, w, 1), 255, np.uint8)], axis=2)
        self.view.set_image(np.ascontiguousarray(out))

        pid = self._preset_id()
        if pid in NEEDS_SAMPLE:
            txt = "取样点 %d 个：%s" % (
                len(self.samples),
                " ".join("(%d,%d,%d)" % s for s in self.samples[:4]))
            if len(self.samples) > 4:
                txt += " …"
            if not self.samples:
                txt = "还没有取样点 —— 点预览图取一个"
            self.lab_samples.setText(txt)
        else:
            self.lab_samples.setText("")

    # ---------- 结果 ----------

    def result_mask(self):
        """全分辨率重算一遍（预览用的是缩略图，不能拿来当结果）。"""
        return range_mask(self.full, self._preset_id(), self.samples,
                          float(self.slider.value()), invert=self.invert)

    def result_mode(self):
        return _MODES[self.combo_mode.currentIndex()][1]

    def accept(self):
        m = self.result_mask()
        if not m.any():
            self.lab_samples.setText("没有命中任何像素 —— 调大容差或换个取样点")
            return
        self.selectionReady.emit(m, self.result_mode())
        super().accept()
