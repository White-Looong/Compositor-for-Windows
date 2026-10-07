# -*- coding: utf-8 -*-
"""文字图层：把文字栅格化成 RGBA 位图。

设计要点（跟 Photoshop 一致，也跟本项目"非破坏性"的调性一致）：

  * 文字图层的像素由**参数**生成（存在 layer.text 里），改参数就重新栅格化，
    所以随时能改字 / 字体 / 字号，不损失清晰度
  * 栅格化结果放在 layer.image，于是变换、蒙版、混合模式、不透明度这些
    既有机制全部自动生效 —— core/render.py 一行都不用改
  * 想用画笔 / 滤镜直接改像素，必须先「栅格化」把它转成普通位图图层，
    否则下次改文字会把改过的像素冲掉

两层"更细"的参数（第十五批加）：

  * `chars` —— **逐字调整**：`{"3": {"dx": 4.0, "dy": -8.0, "scale": 1.2}}`，
    键是字符在 `content` 里的位置（**含换行符**，所以唯一且稳定）。
    dx 是该字之后追加的字距（像素）、dy 是基线偏移、scale 是水平缩放
  * `para`  —— **段落属性**：左/右缩进、首行缩进、段前/段后距（都是像素）

关键实现约定：一行里**只要有**逐字调整就要逐字符画，否则走整行 `drawText`
的快路径 —— 这样没用到新参数的旧工程渲染结果与早期版本**逐位一致**。
逐字推进量用"前缀差"（`advance(line[:i+1]) - advance(line[:i])`）算，
不能直接用单字宽度：前缀差里带字偶间距（kerning），单字宽度里没有。

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
ALIGN_JUSTIFY = "justify"
ALIGN_ITEMS = [(ALIGN_LEFT, "左对齐"), (ALIGN_CENTER, "居中"),
               (ALIGN_RIGHT, "右对齐"), (ALIGN_JUSTIFY, "两端对齐")]
ALIGN_LABELS = dict(ALIGN_ITEMS)

DEFAULT_FAMILY = ""      # 空 = 系统默认字体
DEFAULT_SIZE = 96.0      # 像素

# 段落属性（都是像素）。本项目里每个硬换行是一段，所以"首行缩进"只作用于
# 整块文字的第一行 —— 单行文字用起来正是"首行缩进两个字"的效果。
_PARA_DEFAULTS = {
    "indent_left": 0.0,
    "indent_right": 0.0,
    "indent_first": 0.0,
    "space_before": 0.0,
    "space_after": 0.0,
}
PARA_ITEMS = [("indent_left", "左缩进"), ("indent_right", "右缩进"),
              ("indent_first", "首行缩进"), ("space_before", "段前距"),
              ("space_after", "段后距")]
PARA_LABELS = dict(PARA_ITEMS)

CHAR_ITEMS = [("dx", "字距"), ("dy", "基线"), ("scale", "水平缩放")]

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
    "chars": {},              # 逐字调整（见文件头说明）
    "para": dict(_PARA_DEFAULTS),
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


def default_para():
    """一份新的段落属性。"""
    return dict(_PARA_DEFAULTS)


def normalize_chars(c):
    """逐字调整表 -> {索引: (dx, dy, scale)}。

    入参是 JSON 形态 `{"3": {"dx": 4, "dy": -8, "scale": 1.2}}`（也接受列表形态
    `{"3": [4, -8, 1.2]}`）。只保留**有效且非默认**的项：全 0 / 1 的项直接丢掉，
    否则拖动滑块来回一圈就会在工程文件里攒一堆空条目。
    """
    out = {}
    if not c:
        return out
    try:
        items = list(c.items())
    except AttributeError:
        return out
    for k, v in items:
        try:
            i = int(k)
        except (TypeError, ValueError):
            continue
        if i < 0:
            continue
        if isinstance(v, dict):
            raw = (v.get("dx", 0.0), v.get("dy", 0.0), v.get("scale", 1.0))
        elif isinstance(v, (list, tuple)):
            raw = (list(v) + [0.0, 0.0, 1.0])[:3]
        else:
            continue
        try:
            dx, dy, sc = (float(x) for x in raw)
        except (TypeError, ValueError):
            continue
        if not (sc > 0.0):
            sc = 1.0
        if abs(dx) < 1e-6 and abs(dy) < 1e-6 and abs(sc - 1.0) < 1e-6:
            continue
        out[i] = (round(dx, 3), round(dy, 3), round(sc, 4))
    return out


def chars_to_json(adj):
    """{索引: (dx, dy, scale)} -> 可进 JSON 的形态（存回 layer.text 用）。"""
    out = {}
    for i in sorted(adj):
        dx, dy, sc = adj[i]
        if abs(dx) < 1e-6 and abs(dy) < 1e-6 and abs(sc - 1.0) < 1e-6:
            continue
        out[str(int(i))] = {"dx": round(float(dx), 3),
                            "dy": round(float(dy), 3),
                            "scale": round(float(sc), 4)}
    return out


def normalize_para(p):
    """段落属性：补全缺失字段、类型规整。"""
    out = dict(_PARA_DEFAULTS)
    if isinstance(p, dict):
        for k in out:
            try:
                out[k] = float(p.get(k, 0.0) or 0.0)
            except (TypeError, ValueError):
                out[k] = 0.0
    return out


def char_slots(content):
    """把文字内容拆成可逐字调整的槽位。

    返回 [(索引, 显示字符, 是否换行)]；索引与 `chars` 的键一一对应。
    换行也占一个槽位（它也能调间距），界面上显示成 `\\n` —— 用 "⏎" 这种符号
    在有些字体里是缺字形的，会渲染成一个方块，反而看不懂。
    """
    s = _normalize_content(content)
    out = []
    for i, ch in enumerate(s):
        if ch == "\n":
            out.append((i, "\\n", True))
        elif ch == " ":
            out.append((i, "·", False))
        else:
            out.append((i, ch, False))
    return out


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
    # 逐字调整统一存成 JSON 形态（键是字符下标）并丢掉全默认的项；
    # 下标越界的项留着也无害（渲染时查不到），但界面会把它清掉
    out["chars"] = chars_to_json(normalize_chars(out.get("chars")))
    out["para"] = normalize_para(out.get("para"))
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


def _line_advances(fm, line):
    """前缀推进量：adv[i] = 前 i 个字符的宽度（含字距与字偶间距）。"""
    n = len(line)
    adv = [0.0] * (n + 1)
    for i in range(n):
        adv[i + 1] = float(fm.horizontalAdvance(line[:i + 1]))
    return adv


def layout_line(fm, line, base_index, chars, justify=0.0):
    """排一行字，返回 (逐字位置表, 行宽)。

    逐字位置表是 [(相对行首的 x, 基线偏移 dy, 水平缩放 scale), ...]；
    **没有逐字调整、也不用两端对齐时返回 (None, 自然宽度)** —— 渲染那边
    收到 None 就走整行 drawText 的快路径，保证旧工程逐位不变。

    调整量对**后续**字符是累积的：dx 是"这个字后面多留几个像素"，
    横向缩放把字变宽之后，右边的字也要跟着让位。
    """
    n = len(line)
    if n == 0:
        return None, 0.0
    adv = _line_advances(fm, line)
    if not justify and not any((base_index + i) in chars for i in range(n)):
        return None, adv[n]

    out = []
    shift = 0.0
    for i in range(n):
        dx, dy, sc = chars.get(base_index + i, (0.0, 0.0, 1.0))
        # 这个字的 dx **不推自己**（字距 = 该字之后多留的缝），
        # 但横向缩放要把自己变宽，右边的字要跟着让位，所以两个量分开累
        out.append((adv[i] + shift + justify * i, dy, sc))
        shift += dx + (adv[i + 1] - adv[i]) * (sc - 1.0)
    return out, adv[n] + shift + justify * (n - 1)


def render_text(params):
    """把文字参数栅格化成 (h,w,4) uint8 RGBA（直线色）。

    四周留 padding，给斜体的倾斜、下划线的下探和抗锯齿留地方，
    否则字形会被裁掉一点边。逐字基线偏移（dy）会把上下也要留的余量算进去。
    """
    ensure_fonts()
    p = normalize_text_params(params)
    lines = p["content"].split("\n")
    if not lines:
        lines = [""]
    chars = normalize_chars(p.get("chars"))
    para = normalize_para(p.get("para"))

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

    # 每行的字符起始下标（换行符也占一位，才能和 chars 的键对上）
    starts, idx = [], 0
    for ln in lines:
        starts.append(idx)
        idx += len(ln) + 1

    il, ir = float(para["indent_left"]), float(para["indent_right"])
    i_first = float(para["indent_first"])

    # 第一遍：自然行宽 -> 画布尺寸
    natural = []
    for k, ln in enumerate(lines):
        _lay, wd = layout_line(fm, ln, starts[k], chars)
        natural.append(wd)
    maxw = 0.0
    for k, wd in enumerate(natural):
        first = i_first if k == 0 else 0.0
        maxw = max(maxw, il + first + wd + ir)
    w = max(1, int(math.ceil(maxw)) + pad * 2)

    # 逐字基线偏移会让字顶出去。**上下对称**地留出 max|dy| 的余量：
    # 这样既不会裁掉字形，文字块的行位置也一点都不动（图层按中心定位，
    # 单边扩高会把整块字推走，改一个字的基线结果整句跳一下）。
    dys = [abs(v[1]) for v in chars.values()]
    marg = max(dys) if dys else 0.0

    before = [float(para["space_before"]) if k else 0.0
              for k in range(len(lines))]
    after = [float(para["space_after"]) for _ in lines]
    ys, y = [], pad + ascent + marg
    for k in range(len(lines)):
        if k:
            y += lh + after[k - 1] + before[k]
        ys.append(y)
    # 段前 / 段后都只作用于段落**之间**（首段之前、末段之后不留）
    content_h = (ascent + descent + lh * (len(lines) - 1)
                 + sum(after[:-1]) + sum(before[1:]) + marg * 2)
    h = max(1, int(math.ceil(content_h)) + pad * 2)

    img = QImage(w, h, QImage.Format_ARGB32)
    img.fill(0)
    pn = QPainter(img)
    pn.setRenderHint(QPainter.TextAntialiasing, True)
    pn.setRenderHint(QPainter.Antialiasing, True)
    pn.setFont(font)
    c = p["color"]
    pn.setPen(QColor(int(c[0]), int(c[1]), int(c[2])))

    align = p["align"]
    last = len(lines) - 1
    for k, ln in enumerate(lines):
        first = i_first if k == 0 else 0.0
        avail = w - pad * 2 - il - first - ir
        lay, wd = layout_line(fm, ln, starts[k], chars)
        if align == ALIGN_JUSTIFY and k < last and len(ln) > 1:
            # 两端对齐：把"差多少到满行"平摊到字缝里（末行不拉，按左对齐）
            just = max(0.0, (avail - wd) / (len(ln) - 1))
            if just:
                lay, wd = layout_line(fm, ln, starts[k], chars, just)
        if align == ALIGN_RIGHT:
            x0 = w - pad - ir - wd
        elif align == ALIGN_CENTER:
            x0 = (w - wd) / 2.0 + (il + first - ir) / 2.0
        else:                      # 左对齐 / 两端对齐的末行
            x0 = pad + il + first
        y = ys[k]
        if lay is None:
            pn.drawText(QPointF(x0, y), ln)
            continue
        for i, (cx, dy, sc) in enumerate(lay):
            if abs(sc - 1.0) < 1e-6:
                pn.drawText(QPointF(x0 + cx, y + dy), ln[i])
            else:
                # 横向缩放：以字的左基线点为原点拉伸，右边的字已经让过位了
                pn.save()
                pn.translate(x0 + cx, y + dy)
                pn.scale(sc, 1.0)
                pn.drawText(QPointF(0.0, 0.0), ln[i])
                pn.restore()
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
