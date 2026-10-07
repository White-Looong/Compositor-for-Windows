# -*- coding: utf-8 -*-
"""SVG 导入（**子集**解析 + Qt 栅格化）→ (h, w, 4) uint8 RGBA。

为什么自己解析而不是调 QtSvg / cairosvg：
    * PySide6-Essentials **不含 QtSvg 模块**，再装一个 pyside6-addons 只为导入
      不划算；cairosvg 在 Windows 上没有官方 wheel（要自己编 cairo），装不上。
    * 自己做解析反而能只支持"画得出来"的那部分语法，缺的部分给明确的降级，
      不至于像通用库那样掉进一堆我们看不懂的边角。

**支持的语法**
    结构：svg / g / defs / use / symbol
    图形：path / rect / circle / ellipse / line / polyline / polygon / text / image
    定义：linearGradient / radialGradient / clipPath
    外观：fill / fill-opacity / stroke / stroke-width / stroke-opacity /
          stroke-linecap / stroke-linejoin / stroke-dasharray / fill-rule /
          opacity / transform / display
    变换：translate / scale / rotate / skewX / skewY / matrix

**不支持的**（画出来就是透明的，不是报错 —— 见 HANDOFF §5.37）
    pattern 填充、mask、marker、filter、script、嵌入字体、external href、
    gradientTransform、嵌套 <svg> 的视口裁切。定长 stroke-width 的百分比写法
    （相对 viewport 宽度）也按绝对值处理。

坐标约定：SVG 的 y 轴向下、原点在左上，和 QPainter 一致，所以 **不做 y 翻转**。
根元素上的 viewBox 通过一次根变换映射到输出尺寸；之后所有几何都按用户坐标写。
"""

from __future__ import annotations

import base64
import io
import math
import os
import re

import numpy as np
from xml.etree import ElementTree as ET

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (QColor, QFont, QFontDatabase, QImage,
                           QLinearGradient, QPainter, QPainterPath, QPen,
                           QRadialGradient, QTransform)

# 输出上限：矢量图没有固有尺寸，随便一个 10000×10000 的 viewBox 就能吃 400 MB
MAX_DIM = 8192

_NUM_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")
# 命令名是**整个单词**（translate / skewY），写成单字符会把 translate 截成 t
_TF_RE = re.compile(r"([A-Za-z]+)\s*\(([^)]*)\)")


# ---------------------------------------------------------------- 小工具

def _tag(el):
    """去掉 XML 命名空间，`{...}path` → `path`。"""
    t = el.tag
    if isinstance(t, str) and t.startswith("{"):
        return t.split("}", 1)[-1]
    return t


def _attr(el, name):
    """取属性，兼容 `xlink:href` 这种带前缀的写法。

    ElementTree 解析后命名空间属性的真实键是 `{uri}local`，
    `el.get("xlink:href")` 恒为 None —— 只能再退一步比**本地名**。
    """
    v = el.get(name)
    if v is not None:
        return v
    for k, val in el.attrib.items():
        if k.rpartition("}")[2].split(":")[-1] == name.split(":")[-1]:
            return val
    return None


def _prop(el, name):
    """先查内联 style、再查呈现属性（CSS 优先级更高，SVG 也如此）。"""
    v = el.get(name)
    style = el.get("style")
    if style:
        for decl in style.split(";"):
            k, _, val = decl.partition(":")
            if k.strip() == name and val.strip():
                return val.strip()
    return v


def _f(el, name, default=0.0):
    try:
        return float(_prop(el, name))
    except (TypeError, ValueError):
        return default


def _nums(text):
    """解析一串数字（逗号 / 空格随便混）。"""
    if not text:
        return []
    return [float(t) for t in _NUM_RE.findall(text)]


def _has(el, name, value):
    return (_prop(el, name) or "").strip() == value


# ---------------------------------------------------------------- 颜色 / 画笔

def _color(text, default=None):
    """SVG 颜色字符串 → QColor。解析不出来返回 default。"""
    if text is None:
        return default
    s = text.strip().lower()
    if not s or s in ("none", "transparent", "currentcolor"):
        return default
    if s.startswith("#") and len(s) in (4, 5):
        # #rgb / #rgba → #rrggbb / #rrggbbaa。**只展开一遍** —— 展开两次会
        # 得到 12 位字符串，QColor 直接认不出来（描边静默消失，很难查）
        s = "#" + "".join(c * 2 for c in s[1:])
    m = re.match(r"^rgba?\(([^)]*)\)$", s)
    if m:
        parts = _nums(m.group(1))
        if len(parts) >= 3:
            r, g, b = parts[0], parts[1], parts[2]
            a = parts[3] if len(parts) > 3 else 1.0
            r, g, b = [int(round(c * (100.0 if "%" in m.group(1) else 1.0)))
                       for c in (r, g, b)]
            return QColor(int(np.clip(r, 0, 255)), int(np.clip(g, 0, 255)),
                          int(np.clip(b, 0, 255)),
                          int(np.clip(a * 255, 0, 255)))
    if s.startswith("#") and len(s) == 9:                            # #rrggbbaa
        return QColor(int(s[1:3], 16), int(s[3:5], 16), int(s[5:7], 16),
                      int(s[7:9], 16))
    c = QColor(s)
    return c if c.isValid() else default


def _cap(text):
    # PySide6 只把枚举挂在 Qt.PenCapStyle 上，Qt.PenFlatCap 这种简写不可用
    # PySide6 只把枚举挂在 Qt.PenCapStyle 上，Qt.PenFlatCap 这种简写不可用
    return {"butt": Qt.PenCapStyle.FlatCap, "round": Qt.PenCapStyle.RoundCap,
            "square": Qt.PenCapStyle.SquareCap}.get(
                (text or "butt").strip(), Qt.PenCapStyle.FlatCap)


def _join(text):
    return {"miter": Qt.PenJoinStyle.MiterJoin,
            "round": Qt.PenJoinStyle.RoundJoin,
            "bevel": Qt.PenJoinStyle.BevelJoin}.get(
                (text or "miter").strip(), Qt.PenJoinStyle.MiterJoin)


def _dash(text):
    v = _nums(text)
    if not v:
        return []
    if len(v) % 2:
        v = v + v[:1]                     # 奇数个就重复一次凑齐 奇/偶
    return v


# ---------------------------------------------------------------- transform

def parse_transform(text):
    """SVG transform 串 → QTransform。认不出的部分跳过，不报错。"""
    t = QTransform()
    if not text:
        return t
    for name, inner in _TF_RE.findall(text):
        nums = _nums(inner)
        cmd = name.lower()
        if cmd == "matrix" and len(nums) == 6:
            t = QTransform(*nums) * t
        elif cmd == "translate" and nums:
            tx = nums[0]
            ty = nums[1] if len(nums) > 1 else 0.0
            t = QTransform().translate(tx, ty) * t
        elif cmd == "scale" and nums:
            sx = nums[0]
            sy = nums[1] if len(nums) > 1 else sx
            t = QTransform().scale(sx, sy) * t
        elif cmd == "rotate" and nums:
            deg = nums[0]
            if len(nums) >= 3:                 # rotate(deg cx cy)：绕点转
                t = QTransform().translate(nums[1], nums[2]) * t
                t = QTransform().rotate(deg) * t
                t = QTransform().translate(-nums[1], -nums[2]) * t
            else:
                t = QTransform().rotate(deg) * t
        elif cmd == "skewx" and nums:
            t = QTransform().shear(math.tan(math.radians(nums[0])), 0.0) * t
        elif cmd == "skewy" and nums:
            t = QTransform().shear(0.0, math.tan(math.radians(nums[0]))) * t
    return t


# ---------------------------------------------------------------- path 数据

def _arc_center(x1, y1, rx, ry, phi_deg, fa, fs, x2, y2):
    """SVG F.6.5：由两端点与半径反推椭圆中心（弧心角度也要用它算）。"""
    phi = math.radians(phi_deg)
    cos_p, sin_p = math.cos(phi), math.sin(phi)
    rx, ry = abs(rx), abs(ry)
    dx2, dy2 = (x1 - x2) / 2.0, (y1 - y2) / 2.0
    x1p = cos_p * dx2 + sin_p * dy2
    y1p = -sin_p * dx2 + cos_p * dy2
    # 半径包不住两端点就**等比**放大到刚好（规范如此，不是两轴各自缩放）
    lam = (x1p * x1p) / (rx * rx) + (y1p * y1p) / (ry * ry)
    if lam > 1.0:
        s = math.sqrt(lam)
        rx, ry = rx * s, ry * s
    denom = (x1p * x1p) / (rx * rx) + (y1p * y1p) / (ry * ry)
    coef = 0.0 if denom == 0 else math.sqrt(max(0.0, 1.0 - denom) / denom)
    if fa == fs:
        coef = -coef
    cxp = coef * rx * y1p / ry
    cyp = -coef * ry * x1p / rx
    return (cos_p * cxp - sin_p * cyp + (x1 + x2) / 2.0,
            sin_p * cxp + cos_p * cyp + (y1 + y2) / 2.0,
            rx, ry, phi)


def _arc_to_cubic(path, x1, y1, rx, ry, phi_deg, fa, fs, x2, y2):
    """SVG 的 `A` 弧 → 接到 path 尾部（arcTo 会先 lineTo 弧的起点）。"""
    if rx == 0 or ry == 0:
        path.lineTo(QPointF(x2, y2))
        return
    if phi_deg == 0:
        cx, cy, rx2, ry2, _ = _arc_center(x1, y1, rx, ry, phi_deg, fa, fs, x2, y2)
        # 从 t1 沿 sweep-flag 指向的方向到 t2 跨了多少度（0~360）
        sweep = math.degrees(math.atan2((y2 - cy) / ry2, (x2 - cx) / rx2)) \
            - math.degrees(math.atan2((y1 - cy) / ry2, (x1 - cx) / rx2))
        forward = sweep % 360.0                  # 0..360，正向推进的角度
        if bool(fa) != (forward > 180.0):        # 大弧 / 小弧选一边
            forward -= 360.0
        if not fs:
            forward = -forward                   # sweep-flag=0：反方向绕
        # arcTo 的角度是数学约定（0° 在 3 点、朝上为正），而 SVG 在
        # y 向下的屏幕坐标里 atan2 方向相反 —— 起点取负 y、扫掠翻号。
        path.arcTo(QRectF(cx - rx2, cy - ry2, 2 * rx2, 2 * ry2),
                   -math.degrees(math.atan2((y1 - cy) / ry2, (x1 - cx) / rx2)),
                   -forward)
        return
    # 带旋转的弧：两段的三次贝塞尔近似（phi≠0 在真实素材里极少，精度够用）
    cx, cy, rx, ry, phi = _arc_center(x1, y1, rx, ry, phi_deg, fa, fs, x2, y2)
    if rx == 0 or ry == 0:
        path.lineTo(QPointF(x2, y2))
        return
    cos_p, sin_p = math.cos(phi), math.sin(phi)

    def angle(ux, uy, vx, vy):
        return math.atan2(uy * vx - ux * vy, ux * vx + uy * vy)

    th1 = angle(1.0, 0.0, (x1 - cx) / rx, (y1 - cy) / ry)
    dth = angle((x1 - cx) / rx, (y1 - cy) / ry,
                (x2 - cx) / rx, (y2 - cy) / ry)
    dth = (dth + 2.0 * math.pi) % (2.0 * math.pi)
    if fs and dth < 1e-12:
        dth = 2.0 * math.pi
    elif not fs and abs(dth - 2.0 * math.pi) < 1e-12:
        dth = 1e-12
    if not fs:
        dth = 2.0 * math.pi - dth
    if abs(dth) < 1e-12:
        path.lineTo(QPointF(x2, y2))
        return

    def point(t):
        xa, ya = rx * math.cos(t), ry * math.sin(t)
        return (cx + cos_p * xa - sin_p * ya, cy + sin_p * xa + cos_p * ya)

    k = 4.0 / 3.0 * math.tan(dth / 4.0 / 2.0)      # d = 4/3·tan(Δ/4)·r
    span = dth / 2.0
    for t0 in (0.0, span):
        t1 = t0 + span
        pa, pb = point(th1 + t0), point(th1 + t1)
        c1 = (pa[0] + rx * (-cos_p * math.sin(th1 + t0)
                            - sin_p * math.cos(th1 + t0)) * k,
              pa[1] + ry * (cos_p * math.cos(th1 + t0)
                            - sin_p * math.sin(th1 + t0)) * k)
        c2 = (pb[0] + rx * (-cos_p * math.sin(th1 + t1)
                            - sin_p * math.cos(th1 + t1)) * k,
              pb[1] + ry * (cos_p * math.cos(th1 + t1)
                            - sin_p * math.sin(th1 + t1)) * k)
        path.cubicTo(QPointF(c1[0], c1[1]), QPointF(c2[0], c2[1]),
                     QPointF(pb[0], pb[1]))


def _path_data(d):
    """SVG `d` 属性 → QPainterPath。

    token 化之后是一个简单的状态机：命令字母后面跟固定个数的数字
    （M/L/T 2 个、H/V 1 个、C 6、S/Q 4、A 7），参数不够就停；认不出的
    命令直接跳过，绝不抛异常 —— 线上随便一个网页导出的 SVG 都可能有
    我们不支持的语法，annotate 崩掉比画错更难查。

    **tokenizer 必须用 token 自己的下标**，不能拿 `s[i]` 去 match 原始串：
    两者错位时正则会一直匹配同一处，`i` 不前进 → 整个解析死循环。
    """
    p = QPainterPath()
    if not d:
        return p
    toks = []
    for m in re.finditer(r"[A-Za-z]|[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", d):
        s = m.group(0)
        toks.append(float(s) if not s.isalpha() else s)

    # S 的规范写法是 `x2 y2 x y`（4 个），这里多读两个当扩展端点用，
    # 不足 6 个就补 0 —— 反正只有 C 会去读 a[4] / a[5]。
    ARGS = {"m": 2, "l": 2, "t": 2, "h": 1, "v": 1, "c": 6, "s": 6,
            "q": 4, "a": 7}
    cx = cy = 0.0          # 当前点
    sx = sy = 0.0          # 当前子路径起点
    px = py = 0.0          # 上一个控制点（S / T 的反射基准）
    idx, ntok = 0, len(toks)
    first_move = True

    def take(k):
        """读 k 个数字；不够就返回 None（并且把 idx 停在命令字母上）。"""
        nonlocal idx
        vals = []
        for j in range(k):
            v = toks[idx + j] if idx + j < ntok else None
            if not isinstance(v, float):
                return None
            vals.append(v)
        idx += k
        return vals

    def upd(a):
        """绝对坐标直接用；相对坐标先加上当前点。"""
        return a

    while idx < ntok:
        tok = toks[idx]
        if isinstance(tok, float):            # 裸数字：沿用它前面的命令
            idx += 1
            continue
        cmd = tok
        idx += 1
        low = cmd.lower()
        # 相对指令一律小写；写成 `cmd != low` 会把大写（绝对）命令
        # 也判成相对，端点全被多加一遍当前点。
        rel = cmd.islower()
        if low == "z":
            p.closeSubpath()
            cx, cy = sx, sy
            first_move = True
            continue

        if low in ("m", "l", "t"):
            while idx < ntok and isinstance(toks[idx], float):
                vals = take(2)
                if vals is None:
                    break
                x, y = vals
                if rel:
                    x += cx
                    y += cy
                if low == "m":
                    if first_move:
                        p.moveTo(QPointF(x, y))
                        sx, sy = x, y
                        first_move = False
                    else:
                        p.lineTo(QPointF(x, y))
                elif low == "l":
                    p.lineTo(QPointF(x, y))
                else:                         # T：控制点由上一个 Q/S 反射
                    qx, qy = 2 * cx - px, 2 * cy - py
                    p.quadTo(QPointF(qx, qy), QPointF(x, y))
                    px, py = qx, qy
                cx, cy = x, y
        elif low in ("h", "v"):
            while idx < ntok and isinstance(toks[idx], float):
                vals = take(1)
                if vals is None:
                    break
                v = vals[0] + (cy if rel else 0.0) if low == "v" else \
                    vals[0] + (cx if rel else 0.0)
                if low == "h":
                    cx = v
                    p.lineTo(QPointF(cx, cy))
                else:
                    cy = v
                    p.lineTo(QPointF(cx, cy))
        elif low in ("c", "s", "q", "a"):
            k = ARGS[low]
            while idx < ntok and isinstance(toks[idx], float):
                vals = take(k)
                if vals is None:
                    break
                if low == "a":
                    x2, y2 = vals[5], vals[6]
                    if rel:
                        x2 += cx
                        y2 += cy
                    _arc_to_cubic(p, cx, cy, vals[0], vals[1], vals[2],
                                  vals[3], vals[4], x2, y2)
                    px, py = cx, cy
                    cx, cy = x2, y2
                else:
                    a = list(vals)
                    if low == "q":                       # 4 个参数，别按 6 个取
                        if rel:
                            a[0] += cx; a[1] += cy
                            a[2] += cx; a[3] += cy
                        p.quadTo(QPointF(a[0], a[1]), QPointF(a[2], a[3]))
                        px, py = a[0], a[1]
                        cx, cy = a[2], a[3]
                    else:                                # c / s
                        a += [0.0] * (6 - len(a))       # s 只有 4 个参数，补齐
                        if rel:
                            for q in range(6):           # 相对指令：全部相对当前点
                                a[q] += cx if q % 2 == 0 else cy
                        if low == "c":
                            p.cubicTo(QPointF(a[0], a[1]), QPointF(a[2], a[3]),
                                      QPointF(a[4], a[5]))
                            px, py = a[2], a[3]
                            cx, cy = a[4], a[5]
                        else:                            # s：控制点由上一个 C 反射
                            qx, qy = 2 * cx - px, 2 * cy - py
                            p.cubicTo(QPointF(qx, qy), QPointF(a[0], a[1]),
                                      QPointF(a[2], a[3]))
                            px, py = a[0], a[1]
                            cx, cy = a[2], a[3]
        else:                                  # 认不出的命令：吃掉所有数字
            while idx < ntok and isinstance(toks[idx], float):
                idx += 1
    return p


# ---------------------------------------------------------------- 定义区

def _collect_defs(root):
    """把 defs 里能用的定义挑出来：渐变与 clipPath。"""
    out = {}

    def grab(el):
        tag = _tag(el)
        id_ = el.get("id")
        if not id_:
            return
        if tag == "linearGradient" or tag == "radialGradient":
            out["g:" + id_] = el
        elif tag == "clipPath":
            out["c:" + id_] = el

    for el in root.iter():
        if _tag(el) == "defs":
            for sub in el.iter():
                grab(sub)
    return out


def _clip_path(el, defs):
    """clipPath → QPainterPath（用户坐标系）。"""
    p = QPainterPath()
    for sub in el:
        tag = _tag(sub)
        if tag == "path":
            p.addPath(_path_data(_prop(sub, "d") or ""))
        elif tag == "rect":
            p.addRect(_rect_box(sub))
        elif tag == "circle":
            r = _f(sub, "r")
            p.addEllipse(QRectF(_f(sub, "cx") - r, _f(sub, "cy") - r,
                                2 * r, 2 * r))
        elif tag == "ellipse":
            rx, ry = _f(sub, "rx"), _f(sub, "ry")
            p.addEllipse(QRectF(_f(sub, "cx") - rx, _f(sub, "cy") - ry,
                                2 * rx, 2 * ry))
        elif tag == "polygon" or tag == "polyline":
            pts = _nums(_prop(sub, "points") or "")
            pp = QPainterPath()
            for i in range(0, len(pts) - 1, 2):
                if i == 0:
                    pp.moveTo(QPointF(pts[0], pts[1]))
                else:
                    pp.lineTo(QPointF(pts[i], pts[i + 1]))
            if tag == "polygon":
                pp.closeSubpath()
            p.addPath(pp)
    return p


def _rect_box(el):
    x, y = _f(el, "x"), _f(el, "y")
    w, h = _f(el, "width"), _f(el, "height")
    rx, ry = _f(el, "rx", -1.0), _f(el, "ry", -1.0)
    if rx <= 0 and ry <= 0:
        return QRectF(x, y, w, h)
    if rx < 0 or ry < 0:
        s = min(w, h) / 2.0
        if rx < 0:
            rx = s
        if ry < 0:
            ry = s
    return QRectF(x + rx, y + ry, max(0.0, w - 2 * rx), max(0.0, h - 2 * ry))


def _bbox_of(path_or_box):
    if isinstance(path_or_box, QPainterPath):
        return path_or_box.boundingRect()
    return path_or_box


# ---------------------------------------------------------------- 渐变

def _gradient(el, defs, bbox):
    """渐变元素 → QGradient，坐标已经换算到**用户空间**（调用方乘元素变换）。

    这里刻意不用 `QGradient.setTransform()` —— PySide6 6.11 没绑定它。
    objectBoundingBox（SVG 默认）把 0~1 的归一化坐标按 bbox 展开成绝对坐标；
    objectBoundingBox 下的半径按 bbox 对角线归一化（SVG 1.1 的规定），
    这样非等比缩放下圆渐变会跟着一起变椭圆，和 SVG 的观感一致。
    """
    tag = _tag(el)
    user_space = (_prop(el, "gradientUnits") or "").strip() == "userSpaceOnUse"
    stops = []
    for sub in el:
        if _tag(sub) != "stop":
            continue
        off = float(np.clip(_f(sub, "offset", 0.0), 0.0, 1.0))
        col = _color(_prop(sub, "stop-color"), QColor(0, 0, 0))
        if col is None:
            continue
        op = float(np.clip(_f(sub, "stop-opacity", 1.0), 0.0, 1.0))
        col.setAlphaF(col.alphaF() * op)
        stops.append((off, col))
    if not stops:
        return None

    bx, by = bbox.x(), bbox.y()
    bw, bh = max(bbox.width(), 1e-9), max(bbox.height(), 1e-9)

    if tag == "linearGradient":
        if user_space:
            g = QLinearGradient(QPointF(_f(el, "x1", 0.0), _f(el, "y1", 0.0)),
                                QPointF(_f(el, "x2", 1.0), _f(el, "y2", 1.0)))
        else:
            g = QLinearGradient(QPointF(bx + _f(el, "x1", 0.0) * bw,
                                        by + _f(el, "y1", 0.0) * bh),
                                QPointF(bx + _f(el, "x2", 1.0) * bw,
                                        by + _f(el, "y2", 1.0) * bh))
    else:
        if user_space:
            cx, cy = _f(el, "cx", 0.0), _f(el, "cy", 0.0)
            r = _f(el, "r", 0.0)
            fx, fy = _f(el, "fx", cx), _f(el, "fy", cy)
        else:
            cx = bx + _f(el, "cx", 0.5) * bw
            cy = by + _f(el, "cy", 0.5) * bh
            r = _f(el, "r", 0.5) * math.sqrt((bw * bw + bh * bh) / 2.0)
            fx = bx + _f(el, "fx", 0.5) * bw
            fy = by + _f(el, "fy", 0.5) * bh
        g = QRadialGradient(QPointF(cx, cy), max(r, 1e-6), QPointF(fx, fy))
    # PySide6 把 addColorStop 绑成了 setColorAt（Qt 6 命名），且没有 setTransform
    for off, col in stops:
        g.setColorAt(off, col)
    return g


def _fill_brush(el, defs, bbox):
    """元素的 fill 属性 → QBrush／None（None = 不填充）。

    **fill 缺省是黑色**（SVG 规范），不是"不填充"。漏了这条，只写 stroke
    不写 fill 的图形（描边图、线稿图标）会整体消失 —— 描边还在但看不清。
    """
    text = _prop(el, "fill")
    if text is None:
        return QColor(0, 0, 0)
    text = text.strip()
    if not text or text == "none":
        return None
    if text.startswith("url("):
        key = text[text.find("#") + 1:].rstrip(")").strip()
        src = defs.get("g:" + key)
        if src is None:
            return QColor(0, 0, 0)
        g = _gradient(src, defs, bbox)
        return g or QColor(0, 0, 0)
    col = _color(text, QColor(0, 0, 0))
    fo = _f(el, "fill-opacity", 1.0)          # 渐变不乘：透明度在 stop 里
    if 0.0 < fo < 1.0:
        col.setAlphaF(float(np.clip(col.alphaF() * fo, 0.0, 1.0)))
    return col


# ---------------------------------------------------------------- 绘制

class _Renderer:
    def __init__(self, img, defs, roots):
        self.img = img
        self.defs = defs
        self.roots = roots           # 顶层 svg 树，供 use 反查 id
        self.base_dir = os.getcwd()  # <image href> 的基准目录
        self.pn = QPainter(img)
        self.pn.setRenderHint(QPainter.Antialiasing, True)
        self.pn.setRenderHint(QPainter.TextAntialiasing, True)
        self.pn.setCompositionMode(QPainter.CompositionMode_SourceOver)

    def _push_clip(self, el):
        """把 clip-path 设成 painter 当前的裁剪区（调用方已设好变换）。

        只管 setClipPath，save / restore 由调用方配对，避免嵌套 save 时
        一次 restore 把裁剪区本身也撤销掉。
        """
        clip = _prop(el, "clip-path") or ""
        if not clip.startswith("url("):
            return
        key = clip[clip.find("#") + 1:].rstrip(")").strip()
        cel = self.defs.get("c:" + key)
        if cel is None:
            return
        cp = _clip_path(cel, self.defs)
        if not cp.isEmpty():
            self.pn.setClipPath(cp)

    # -- 主递归 ----------------------------------------------------
    def walk(self, el, ctm, opacity):
        tag = _tag(el)
        if tag in ("defs", "symbol", "clipPath", "mask", "pattern", "marker",
                   "style", "filter", "title", "desc"):
            return
        if (_prop(el, "display") or "").strip() == "none":
            return
        # use 要拿到**父矩阵**自己拼变换：walk 这行已经把元素自己的
        # transform 乘进 ctm 了，_use 再乘一次就翻两倍（scale(1.5) → 2.25）。
        if tag == "use":
            self._use(el, ctm, opacity)
            return
        # 子变换**先于**父矩阵作用，所以新矩阵左乘（QTransform 的 `A*B` 是先 A 后 B）
        ctm = parse_transform(_prop(el, "transform")) * ctm
        if tag == "svg":
            # 嵌套 svg 的视口裁切：只当容器，不裁切
            for sub in el:
                self.walk(sub, ctm, opacity)
            return
        if tag == "g" or tag == "a" or tag == "switch":
            # 组的 clip-path 和 opacity 都要往下传：裁剪要包住整组，
            # 不透明度是相乘的（PS 里组也是乘，不是取最大）。
            gop = opacity * (_f(el, "opacity", 1.0))
            if gop <= 0.001:
                return
            self.pn.save()
            self.pn.setTransform(ctm)
            self._push_clip(el)
            for sub in el:
                self.walk(sub, ctm, gop)
            self.pn.restore()
            return

        op = opacity * (_f(el, "opacity", 1.0))
        if op <= 0.001:
            return
        if tag == "use":
            self._use(el, ctm, op)
            return

        self.pn.save()
        self.pn.setTransform(ctm)
        self.pn.setOpacity(float(np.clip(op, 0.0, 1.0)))

        self._push_clip(el)

        if tag == "text":
            self._text(el)
        elif tag == "image":
            self._image(el)
        else:
            path = self._shape(el)
            if path is None:
                self.pn.restore()
                return
            self._paint(el, path)
        self.pn.restore()

    # -- 形状 ------------------------------------------------------
    def _shape(self, el):
        tag = _tag(el)
        try:
            if tag == "rect":
                return _rect_path(el)
            # 注意 addEllipse / addRect / addRoundedRect 都是 **void**，
            # 不能写成 `return QPainterPath().addEllipse(...)` —— 那会返回 None
            if tag == "circle":
                pp = QPainterPath()
                r = _f(el, "r")
                pp.addEllipse(QRectF(_f(el, "cx") - r, _f(el, "cy") - r,
                                     2 * r, 2 * r))
                return pp
            if tag == "ellipse":
                pp = QPainterPath()
                rx, ry = _f(el, "rx"), _f(el, "ry")
                pp.addEllipse(QRectF(_f(el, "cx") - rx, _f(el, "cy") - ry,
                                     2 * rx, 2 * ry))
                return pp
            if tag == "line":
                p = QPainterPath()
                p.moveTo(QPointF(_f(el, "x1"), _f(el, "y1")))
                p.lineTo(QPointF(_f(el, "x2"), _f(el, "y2")))
                return p
            if tag == "path":
                return _path_data(_prop(el, "d") or "")
            if tag in ("polygon", "polyline"):
                pts = _nums(_prop(el, "points") or "")
                p = QPainterPath()
                for i in range(0, len(pts) - 1, 2):
                    if i == 0:
                        p.moveTo(QPointF(pts[0], pts[1]))
                    else:
                        p.lineTo(QPointF(pts[i], pts[i + 1]))
                if tag == "polygon":
                    p.closeSubpath()
                return p
        except (TypeError, ValueError):
            return None
        return None

    def _paint(self, el, path):
        bbox = path.boundingRect()
        brush = _fill_brush(el, self.defs, bbox)
        sweight = _f(el, "stroke-width", 1.0)
        stroke = _prop(el, "stroke")
        if stroke and stroke.strip() not in ("none", ""):
            sw = sweight
            if str(sweight).endswith("%"):          # 相对 viewport：近似成绝对值
                sw = float(str(sweight).rstrip("%")) / 100.0 * self.img.width()
            col = _color(stroke, QColor(0, 0, 0))
            if col is not None:
                col.setAlphaF(float(np.clip(col.alphaF() *
                                             _f(el, "stroke-opacity", 1.0), 0, 1)))
                pen = QPen(col, sw)
                pen.setCapStyle(_cap(_prop(el, "stroke-linecap")))
                pen.setJoinStyle(_join(_prop(el, "stroke-linejoin")))
                dash = _dash(_prop(el, "stroke-dasharray"))
                if dash:
                    pen.setDashPattern([max(0.0, d) for d in dash])
                self.pn.setPen(pen)
        else:
            self.pn.setPen(Qt.NoPen)

        self.pn.setBrush(Qt.NoBrush if brush is None else brush)
        if brush is not None:
            self.pn.drawPath(path)

    # -- use ------------------------------------------------------
    def _use(self, el, ctm, opacity):
        href = _attr(el, "href") or _attr(el, "xlink:href") or ""
        if not href.startswith("#"):
            return
        key = href[1:]
        src = None
        for root in self.roots:
            for cand in root.iter():
                if cand.get("id") == key:
                    src = cand
                    break
            if src is not None:
                break
        if src is None:
            return
        # `A*B` 是先 A 后 B：子元素变换先作用、父矩阵后作用 → `parse * ctm`；
        # use 的 x / y 按规范是在 transform **之后**才平移，所以再 `t * Tr`。
        t = parse_transform(_prop(el, "transform")) * ctm
        x, y = _f(el, "x"), _f(el, "y")
        if x or y:
            t = t * QTransform().translate(x, y)
        self.walk(src, t, opacity)

    # -- text / image ---------------------------------------------
    def _text(self, el):
        txt = "".join(el.itertext())
        if not txt:
            return
        f = QFont()
        fam = (_prop(el, "font-family") or "").strip()
        f.setFamily(fam if fam else _first_family())
        size = _f(el, "font-size", 12.0)
        f.setPointSizeF(size)
        if (_prop(el, "font-weight") or "").strip() in ("bold", "bolder", "700",
                                                        "800", "900"):
            f.setBold(True)
        if (_prop(el, "font-style") or "").strip() in ("italic", "oblique"):
            f.setItalic(True)
        sp = _f(el, "letter-spacing", 0.0)
        if sp:
            f.setLetterSpacing(QFont.AbsoluteSpacing, sp)
        self.pn.setFont(f)
        col = _fill_brush(el, self.defs, QRectF()) or QColor(0, 0, 0)
        self.pn.setBrush(Qt.NoBrush)
        self.pn.setPen(QPen(col, 1.0))
        anchor = (_prop(el, "text-anchor") or "start").strip()
        x, y = _f(el, "x"), _f(el, "y")
        fm = self.pn.fontMetrics()
        for line in txt.split("\n"):
            if anchor == "middle":
                dx = -fm.horizontalAdvance(line) / 2.0
            elif anchor == "end":
                dx = -fm.horizontalAdvance(line)
            else:
                dx = 0.0
            self.pn.drawText(QPointF(x + dx, y), line)
            y += fm.height()

    def _image(self, el):
        href = _attr(el, "href") or _attr(el, "xlink:href") or ""
        data = None
        if href.startswith("data:"):
            head, _, b64 = href.partition(",")
            try:
                if ";base64" in head:
                    data = base64.b64decode(b64)
                else:
                    data = b64.encode("utf-8")
            except Exception:
                return
        else:
            if re.match(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", href):
                return                      # 外链不拉
            try:
                with open(os.path.join(os.path.dirname(self.base_dir),
                                       href), "rb") as fh:
                    data = fh.read()
            except OSError:
                return
        if not data:
            return
        try:
            from PIL import Image
            im = Image.open(io.BytesIO(data))
            if im.mode != "RGBA":
                im = im.convert("RGBA")
            arr = np.array(im)
            h, w = arr.shape[:2]
            qimg = QImage(arr.data, w, h, QImage.Format_RGBA8888)
        except Exception:
            return
        x, y = _f(el, "x"), _f(el, "y")
        w = _f(el, "width", qimg.width())
        h = _f(el, "height", qimg.height())
        if qimg.width() and qimg.height():
            qimg = qimg.scaled(int(w), int(h))
        self.pn.drawImage(QPointF(x, y), qimg)

    # -- 收尾 ------------------------------------------------------
    def end(self):
        self.pn.end()


def _first_family():
    """离屏 Qt 常常只有一两个字体，给不出就留空（Qt 用默认字体）。"""
    fams = QFontDatabase.families()
    return fams[0] if fams else ""


def _rect_path(el):
    box = _rect_box(el)
    p = QPainterPath()
    rx, ry = _f(el, "rx", -1.0), _f(el, "ry", -1.0)
    if rx <= 0 and ry <= 0:
        p.addRect(box)
        return p
    p.addRoundedRect(box, rx, ry)
    return p


# ---------------------------------------------------------------- 对外接口

def parse_svg(source):
    """解析 SVG 文本 → (root, 内部尺寸或 None)。source 可以是路径或字符串。"""
    if os.path.isfile(source) and source.lower().endswith(".svg"):
        with open(source, "rb") as f:
            text = f.read().decode("utf-8", "replace")
        base = os.path.dirname(os.path.abspath(source))
    elif "\n<" in source[:512] or source.lstrip().startswith("<"):
        text = source
        base = os.getcwd()
    else:
        with open(source, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        base = os.path.dirname(os.path.abspath(source))
    root = ET.fromstring(text)
    return root, base


def intrinsic_size(root):
    """(宽, 高, viewBox) —— 都可能是 None。"""
    vb = _prop(root, "viewBox")
    box = _nums(vb)
    w = _prop(root, "width")
    h = _prop(root, "height")
    wf = _f(root, "width", None)
    hf = _f(root, "height", None)
    if w and str(w).strip().endswith(("px", "pt", "pc", "in", "cm", "mm")):
        wf = float(str(w).strip()[:-2]) if str(w).strip()[:-2] else wf
    if h and str(h).strip().endswith(("px", "pt", "pc", "in", "cm", "mm")):
        hf = float(str(h).strip()[:-2]) if str(h).strip()[:-2] else hf
    return wf, hf, (box if len(box) == 4 else None)


def render_svg(source, width=None, height=None):
    """把 SVG 栅格化成 (h, w, 4) uint8 RGBA。

    width / height 都给 None 时按 SVG 自身尺寸（没有尺寸就按 viewBox 比例，
    还没有就 300×150 —— SVG 规范里的默认值）。
    """
    root, base = parse_svg(source)
    iw, ih, vb = intrinsic_size(root)

    if width is None and height is None:
        if iw and ih:
            width, height = int(round(iw)), int(round(ih))
        elif vb:
            width, height = int(round(vb[2])), int(round(vb[3]))
        else:
            width, height = 300, 150
    if width is None:
        width = int(round(height * (vb[2] / vb[3])) if vb else height)
    if height is None:
        height = int(round(width * (vb[3] / vb[2])) if vb else width)

    width = int(np.clip(width, 1, MAX_DIM))
    height = int(np.clip(height, 1, MAX_DIM))

    img = QImage(width, height, QImage.Format_ARGB32)
    img.fill(0)                                   # 透明底，不是黑底
    defs = _collect_defs(root)

    r = _Renderer(img, defs, [root])
    r.base_dir = base
    ctm = QTransform()
    if vb:
        sx = width / vb[2]
        sy = height / vb[3]
        par = (_prop(root, "preserveAspectRatio") or "").strip()
        if "none" in par:
            ox = oy = 0.0                       # 拉伸填满：两轴各自缩放
        else:
            # 默认是 xMidYMid meet：等比 + 居中（不是写成 none 的才等比）
            s = min(sx, sy)
            sx = sy = s
            ox, oy = (width - vb[2] * s) / 2.0, (height - vb[3] * s) / 2.0
        # 顺序是 `A*B` 先 A 后 B：先减 viewBox 原点，再缩放平移到画布
        ctm = QTransform(sx, 0, 0, sy, ox, oy) \
            * QTransform().translate(-vb[0], -vb[1])
    for el in root:
        r.walk(el, ctm, 1.0)
    r.end()

    img = img.convertToFormat(QImage.Format_RGBA8888)
    h, w = img.height(), img.width()
    stride = img.bytesPerLine()
    arr = np.frombuffer(img.bits(), np.uint8, count=stride * h)
    arr = arr.reshape(h, stride)[:, :w * 4].reshape(h, w, 4).copy()
    return np.ascontiguousarray(arr)


def svg_size(source):
    """SVG 的固有尺寸（都不给时按 300×150）。"""
    root, _ = parse_svg(source)
    iw, ih, vb = intrinsic_size(root)
    if iw and ih:
        return int(round(iw)), int(round(ih))
    if vb:
        return int(round(vb[2])), int(round(vb[3]))
    return 300, 150


def svg_available():
    return True
