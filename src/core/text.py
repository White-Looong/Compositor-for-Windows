# -*- coding: utf-8 -*-
"""文字图层：把文字栅格化成 RGBA 位图。

设计要点（跟 Photoshop 一致，也跟本项目"非破坏性"的调性一致）：

  * 文字图层的像素由**参数**生成（存在 layer.text 里），改参数就重新栅格化，
    所以随时能改字 / 字体 / 字号，不损失清晰度
  * 栅格化结果放在 layer.image，于是变换、蒙版、混合模式、不透明度这些
    既有机制全部自动生效 —— core/render.py 一行都不用改
  * 想用画笔 / 滤镜直接改像素，必须先「栅格化」把它转成普通位图图层，
    否则下次改文字会把改过的像素冲掉

QPainter 相关的坑：
  * offscreen QPA 下 QFontDatabase.families() 返回 0，文字会渲成空白，
    所以 ensure_fonts() 会手动挂一个系统字体（真实桌面运行下是 no-op）
  * QImage.bits() 返回的是 memoryview（PySide6 6.11 没有 setsize），
    直接 np.frombuffer 即可，但要按 bytesPerLine 裁掉行对齐的填充字节
"""

from __future__ import annotations

import copy
import math
import os

import numpy as np
from PySide6.QtCore import QPointF
from PySide6.QtGui import (QColor, QFont, QFontDatabase, QFontMetricsF,
                           QImage, QPainter)

# 对齐方式：存英文 key，界面显示中文 label
ALIGN_LEFT = "left"
ALIGN_CENTER = "center"
ALIGN_RIGHT = "right"
ALIGN_ITEMS = [(ALIGN_LEFT, "左对齐"), (ALIGN_CENTER, "居中"),
               (ALIGN_RIGHT, "右对齐")]
ALIGN_LABELS = dict(ALIGN_ITEMS)

DEFAULT_FAMILY = ""      # 空 = 系统默认字体
DEFAULT_SIZE = 96.0      # 像素

_TEXT_DEFAULTS = {
    "content": "文字",
    "family": DEFAULT_FAMILY,
    "size": DEFAULT_SIZE,
    "bold": False,
    "italic": False,
    "underline": False,
    "color": [0, 0, 0],
    "line_height": 1.2,       # 相对 ascent+descent 的倍数
    "letter_spacing": 0.0,    # 每个字符额外加的像素
    "align": ALIGN_CENTER,
}

# 离屏兜底时按这个顺序找字体
_FALLBACK_FONTS = ("C:/Windows/Fonts/msyh.ttc",
                   "C:/Windows/Fonts/msyh.ttf",
                   "C:/Windows/Fonts/simhei.ttf",
                   "C:/Windows/Fonts/arial.ttf")

_FONTS_READY = False


def default_text_params(**overrides):
    """一份新的文字参数（深拷贝，避免多人共用同一个 list/dict）。"""
    p = copy.deepcopy(_TEXT_DEFAULTS)
    p.update(overrides)
    return p


def normalize_text_params(p):
    """补全缺失字段并做类型规整，读工程 / 读 PSD 后用它兜底。"""
    out = default_text_params()
    if not p:
        return out
    for k in out:
        if k in p and p[k] is not None:
            out[k] = p[k]
    out["content"] = _normalize_content(out["content"])
    try:
        out["size"] = float(out["size"])
    except (TypeError, ValueError):
        out["size"] = DEFAULT_SIZE
    try:
        out["line_height"] = float(out["line_height"])
    except (TypeError, ValueError):
        out["line_height"] = 1.2
    try:
        out["letter_spacing"] = float(out["letter_spacing"])
    except (TypeError, ValueError):
        out["letter_spacing"] = 0.0
    c = out.get("color") or [0, 0, 0]
    out["color"] = [int(max(0, min(255, int(v)))) for v in list(c)[:3]]
    if len(out["color"]) < 3:
        out["color"] = (out["color"] + [0, 0, 0])[:3]
    out["bold"] = bool(out["bold"])
    out["italic"] = bool(out["italic"])
    out["underline"] = bool(out["underline"])
    out["family"] = str(out.get("family") or DEFAULT_FAMILY)
    if out["align"] not in ALIGN_LABELS:
        out["align"] = ALIGN_CENTER
    return out


def _normalize_content(s):
    if s is None:
        return ""
    return str(s).replace("\r\n", "\n").replace("\r", "\n")


# ---------------------------------------------------------------- 字体

def ensure_fonts():
    """离屏 / 无字体环境的兜底：拿不到字体时手动挂一个系统字体。

    真实桌面运行下字体是齐的，这里直接返回。
    """
    global _FONTS_READY
    if _FONTS_READY:
        return
    _FONTS_READY = True
    try:
        if QFontDatabase.families():
            return
    except Exception:
        return
    for path in _FALLBACK_FONTS:
        if os.path.exists(path):
            try:
                QFontDatabase.addApplicationFont(path)
            except Exception:
                pass
            return


def _make_font(p):
    f = QFont()
    fam = str(p.get("family") or "").strip()
    if fam:
        f.setFamily(fam)
    f.setPixelSize(max(1, int(round(float(p.get("size", DEFAULT_SIZE))))))
    f.setBold(bool(p.get("bold", False)))
    f.setItalic(bool(p.get("italic", False)))
    f.setUnderline(bool(p.get("underline", False)))
    try:
        f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing,
                           float(p.get("letter_spacing", 0.0)))
    except Exception:
        pass
    return f


# ---------------------------------------------------------------- 栅格化

def _to_rgba(img):
    """QImage -> (h,w,4) uint8 RGBA。按 bytesPerLine 裁掉行对齐填充。"""
    img = img.convertToFormat(QImage.Format_RGBA8888)
    h, w = img.height(), img.width()
    stride = int(img.bytesPerLine())
    if h <= 0 or w <= 0 or stride < w * 4:
        return np.zeros((max(1, h), max(1, w), 4), np.uint8)
    buf = img.bits()
    arr = np.frombuffer(buf, np.uint8, count=stride * h)
    arr = arr.reshape(h, stride)[:, :w * 4].reshape(h, w, 4)
    return np.ascontiguousarray(arr.copy())


def render_text(params):
    """把文字参数栅格化成 (h,w,4) uint8 RGBA（直线色）。

    四周留 padding，给斜体的倾斜、下划线的下探和抗锯齿留地方，
    否则字形会被裁掉一点边。
    """
    ensure_fonts()
    p = normalize_text_params(params)
    lines = p["content"].split("\n")
    if not lines:
        lines = [""]

    font = _make_font(p)
    fm = QFontMetricsF(font)
    ascent = float(fm.ascent())
    descent = float(fm.descent())
    if ascent <= 0.0:
        ascent = float(fm.height()) if fm.height() > 0 else DEFAULT_SIZE
        descent = 0.0
    size = max(1, int(round(float(p["size"]))))
    lh = max(1.0, (ascent + descent) * float(p["line_height"]))
    pad = max(6, int(round(size * 0.35)))

    widths = [float(fm.horizontalAdvance(ln)) for ln in lines]
    maxw = max(widths) if widths else 0.0
    w = int(math.ceil(maxw)) + pad * 2
    h = int(math.ceil(ascent + descent + lh * (len(lines) - 1))) + pad * 2
    w = max(w, 1)
    h = max(h, 1)

    img = QImage(w, h, QImage.Format_ARGB32)
    img.fill(0)
    pn = QPainter(img)
    pn.setRenderHint(QPainter.TextAntialiasing, True)
    pn.setRenderHint(QPainter.Antialiasing, True)
    pn.setFont(font)
    c = p["color"]
    pn.setPen(QColor(int(c[0]), int(c[1]), int(c[2])))

    align = p["align"]
    y = pad + ascent
    for ln, lw in zip(lines, widths):
        if align == ALIGN_LEFT:
            x = float(pad)
        elif align == ALIGN_RIGHT:
            x = w - pad - lw
        else:
            x = (w - lw) / 2.0
        pn.drawText(QPointF(x, y), ln)
        y += lh
    pn.end()
    return _to_rgba(img)


def measure_text(params):
    """返回 (宽, 高)，含 padding。给 UI 预估尺寸用。"""
    arr = render_text(params)
    return arr.shape[1], arr.shape[0]


def sync_text_image(layer):
    """按当前参数重新栅格化文字图层的位图。

    保持 tx/ty（图层中心）不动，所以改字的时候文字是在原地"长大/缩小"。
    返回新的位图；不是文字图层或缺参数时返回 None。
    """
    if layer is None or getattr(layer, "text", None) is None:
        return None
    layer.image = render_text(layer.text)
    return layer.image
