# -*- coding: utf-8 -*-
"""矢量形状图层：把路径参数栅格化成 RGBA 位图。

跟 `core/text.py` 是同一套路子 —— 像素由**参数**生成（存在 `layer.shape` 里），
改参数就重新栅格化，所以改颜色 / 圆角 / 边数 / 缩放都不损失清晰度。
但形状比文字多一条关键性质：

  * **位图是按当前缩放现画的**：`sx/sy` 直接决定栅格的尺寸，画完就把
    `sx/sy` 归 1（尺寸已经"烤进"位图里了）。所以放大缩小**不会糊** ——
    这正是"矢量"和"位图"的区别。文字图层做不到这点（字号是参数，
    缩放是变换，放大只能把已有栅格拉大）。

于是 `core/render.py` 一行都不用改：形状图层渲染出来就是一张普通 RGBA，
变换 / 蒙版 / 混合模式 / 不透明度 / 图层样式全部自动生效。

坐标系：参数的几何量（`box` / `points` / `segs`）是**形状局部坐标**，
单位就是画布像素；栅格化时先平移到原点、再按 (sx, sy) 缩放、外加描边
padding，所以缩放只是"换一个分辨率重画一次"。

五种形状：

  * `rect`     —— 矩形（带圆角 `radius`）
  * `ellipse`  —— 椭圆（内切于 `box`）
  * `line`     —— 直线（`points` 两个端点，只有描边，宽度就是 `stroke.width`）
  * `polygon`  —— 正多边形 / 星形（`box` + `sides` + `star`）
  * `path`     —— 贝塞尔路径（`segs`，每段是 [起点, 控制点1, 控制点2, 终点]）
"""

from __future__ import annotations

import copy
import math

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen

from .layer import LAYER_SHAPE

SHAPE_RECT = "rect"
SHAPE_ELLIPSE = "ellipse"
SHAPE_LINE = "line"
SHAPE_POLYGON = "polygon"
SHAPE_PATH = "path"

# 界面下拉框用的顺序（key, 中文名）
SHAPE_ITEMS = [(SHAPE_RECT, "矩形"), (SHAPE_ELLIPSE, "椭圆"),
               (SHAPE_LINE, "直线"), (SHAPE_POLYGON, "多边形"),
               (SHAPE_PATH, "路径")]
SHAPE_LABELS = dict(SHAPE_ITEMS)
SHAPE_KINDS = [k for k, _lb in SHAPE_ITEMS]

DEFAULT_FILL = [217, 83, 79]
DEFAULT_STROKE = [38, 38, 38]
DEFAULT_STROKE_WIDTH = 4.0

_SHAPE_DEFAULTS = {
    "kind": SHAPE_RECT,
    "box": [0.0, 0.0, 200.0, 160.0],   # (x0, y0, x1, y1) 局部坐标
    "radius": 0.0,                      # 圆角（矩形）
    "points": [],                       # 直线 / 路径的控制点（局部坐标）
    "segs": [],                         # 贝塞尔段，见文件头
    "closed": True,                     # 路径是否闭合
    "sides": 6,                         # 多边形边数
    "star": 0.0,                        # 星形内缩比（0 = 普通多边形）
    "fill": {"on": True, "color": list(DEFAULT_FILL), "opacity": 1.0},
    "stroke": {"on": False, "color": list(DEFAULT_STROKE),
               "width": DEFAULT_STROKE_WIDTH, "opacity": 1.0},
}


def default_shape_params(**overrides):
    """一份新的形状参数（深拷贝，别让多个图层共用同一个 list/dict）。"""
    p = copy.deepcopy(_SHAPE_DEFAULTS)
    p.update(overrides)
    return normalize_shape_params(p)


# ---------------------------------------------------------------- 规整

def _norm_box(b):
    """box -> (x0, y0, x1, y1) 浮点，且保证 x0<=x1 / y0<=y1。"""
    try:
        v = [float(x) for x in list(b)[:4]]
    except (TypeError, ValueError):
        return 0.0, 0.0, 1.0, 1.0
    if len(v) < 4:
        v = (v + [0.0, 0.0, 1.0, 1.0])[:4]
    x0, y0, x1, y1 = v
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    # 完全扁掉的形状渲出来是空的，给它 1px
    if x1 - x0 < 1e-6:
        x1 = x0 + 1.0
    if y1 - y0 < 1e-6:
        y1 = y0 + 1.0
    return x0, y0, x1, y1


def _norm_pts(pts):
    out = []
    if not pts:
        return out
    for it in pts:
        try:
            v = list(it)
            out.append([float(v[0]), float(v[1])])
        except (TypeError, ValueError, IndexError):
            continue
    return out


def _norm_segs(segs):
    """贝塞尔段：每段 4 个点 [p0, c1, c2, p1]。缺控制点就退化成直线段。"""
    out = []
    if not segs:
        return out
    for s in segs:
        try:
            pts = _norm_pts(list(s)[:4])
        except (TypeError, ValueError):
            continue
        if len(pts) < 2:
            continue
        while len(pts) < 4:
            pts.append(list(pts[-1]))
        out.append(pts)
    return out


def _norm_color(c, default):
    try:
        v = [int(max(0, min(255, int(round(float(x)))))) for x in list(c)[:3]]
    except (TypeError, ValueError):
        v = list(default)
    if len(v) < 3:
        v = (v + list(default))[:3]
    return v


def _norm_style(d, default_color, default_width):
    """填充 / 描边这两个 dict 的规整。"""
    out = {"on": False, "color": list(default_color), "opacity": 1.0}
    if default_width is not None:
        out["width"] = float(default_width)
    if not isinstance(d, dict):
        return out
    out["on"] = bool(d.get("on", False))
    out["color"] = _norm_color(d.get("color"), default_color)
    try:
        out["opacity"] = max(0.0, min(1.0, float(d.get("opacity", 1.0))))
    except (TypeError, ValueError):
        out["opacity"] = 1.0
    if default_width is not None:
        try:
            out["width"] = max(0.0, min(2000.0,
                                        float(d.get("width", default_width))))
        except (TypeError, ValueError):
            out["width"] = float(default_width)
    return out


def normalize_shape_params(p):
    """补全缺失字段、做类型规整。读工程 / 建图层后都用它兜底。"""
    out = copy.deepcopy(_SHAPE_DEFAULTS)
    if isinstance(p, dict):
        for k in out:
            if p.get(k) is not None:
                out[k] = p[k]
    kind = out.get("kind")
    if kind not in SHAPE_LABELS:
        kind = SHAPE_RECT
    out["kind"] = kind
    out["box"] = list(_norm_box(out.get("box")))
    try:
        out["radius"] = max(0.0, float(out.get("radius", 0.0) or 0.0))
    except (TypeError, ValueError):
        out["radius"] = 0.0
    try:
        out["sides"] = int(max(3, min(60, int(out.get("sides", 6) or 6))))
    except (TypeError, ValueError):
        out["sides"] = 6
    try:
        out["star"] = max(0.0, min(0.9, float(out.get("star", 0.0) or 0.0)))
    except (TypeError, ValueError):
        out["star"] = 0.0
    out["closed"] = bool(out.get("closed", True))
    out["points"] = _norm_pts(out.get("points"))
    out["segs"] = _norm_segs(out.get("segs"))
    out["fill"] = _norm_style(out.get("fill"), DEFAULT_FILL, None)
    out["stroke"] = _norm_style(out.get("stroke"), DEFAULT_STROKE,
                                DEFAULT_STROKE_WIDTH)
    return out


# ---------------------------------------------------------------- 几何

def polygon_points(box, sides=6, star=0.0, rot_deg=0.0):
    """内切于 box 的正多边形 / 星形的顶点。

    `star > 0` 时顶点数翻倍（外顶点 / 内顶点交替），内顶点按 `star` 往里缩。
    第一个顶点在正上方（-90°），跟 PS 的多边形工具一致。
    """
    x0, y0, x1, y1 = _norm_box(box)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    rx, ry = (x1 - x0) / 2.0, (y1 - y0) / 2.0
    n = max(3, int(sides))
    m = n * 2 if star > 0.0 else n
    base = math.radians(-90.0 + float(rot_deg or 0.0))
    pts = []
    for i in range(m):
        k = 1.0 if (star <= 0.0 or i % 2 == 0) else (1.0 - star)
        a = base + 2.0 * math.pi * i / m
        pts.append([cx + rx * k * math.cos(a), cy + ry * k * math.sin(a)])
    return pts


def shape_bbox(p):
    """形状几何的包围盒 (x0, y0, x1, y1)（局部坐标，不含描边）。"""
    p = normalize_shape_params(p)
    kind = p["kind"]
    if kind in (SHAPE_RECT, SHAPE_ELLIPSE, SHAPE_POLYGON):
        return tuple(p["box"])
    pts = p["points"] if kind == SHAPE_LINE else [
        pt for seg in p["segs"] for pt in seg]
    if not pts:
        return tuple(p["box"])
    xs = [q[0] for q in pts]
    ys = [q[1] for q in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def _build_path(p, x0, y0, ax, ay, pad):
    """构造位图坐标里的 QPainterPath。

    局部坐标 (x, y) -> 位图 (pad + (x-x0)*ax, pad + (y-y0)*ay)。
    """
    kind = p["kind"]

    def tp(q):
        return QPointF(pad + (q[0] - x0) * ax, pad + (q[1] - y0) * ay)

    path = QPainterPath()
    if kind == SHAPE_RECT:
        bx0, by0, bx1, by1 = p["box"]
        w = (bx1 - bx0) * ax
        h = (by1 - by0) * ay
        r = float(p["radius"]) * min(ax, ay)
        r = max(0.0, min(r, min(w, h) / 2.0))
        if r <= 0.0:
            path.addRect(pad, pad, w, h)
        else:
            path.addRoundedRect(pad, pad, w, h, r, r)
        return path
    if kind == SHAPE_ELLIPSE:
        bx0, by0, bx1, by1 = p["box"]
        path.addEllipse(pad, pad, (bx1 - bx0) * ax, (by1 - by0) * ay)
        return path
    if kind == SHAPE_LINE:
        pts = p["points"][:2]
        if len(pts) < 2:
            return None
        path.moveTo(tp(pts[0]))
        path.lineTo(tp(pts[1]))
        return path
    if kind == SHAPE_POLYGON:
        pts = polygon_points(p["box"], p["sides"], p["star"])
        path.moveTo(tp(pts[0]))
        for q in pts[1:]:
            path.lineTo(tp(q))
        path.closeSubpath()
        return path
    # SHAPE_PATH：一段段三次贝塞尔接起来
    segs = p["segs"]
    if not segs:
        return None
    path.moveTo(tp(segs[0][0]))
    for s in segs:
        path.cubicTo(tp(s[1]), tp(s[2]), tp(s[3]))
    if p["closed"]:
        path.closeSubpath()
    return path


def _to_rgba(img):
    """QImage -> (h,w,4) uint8 RGBA。按 bytesPerLine 裁掉行对齐填充。"""
    img = img.convertToFormat(QImage.Format_RGBA8888)
    h, w = img.height(), img.width()
    stride = int(img.bytesPerLine())
    if h <= 0 or w <= 0 or stride < w * 4:
        return np.zeros((max(1, h), max(1, w), 4), np.uint8)
    arr = np.frombuffer(img.bits(), np.uint8, count=stride * h)
    arr = arr.reshape(h, stride)[:, :w * 4].reshape(h, w, 4)
    return np.ascontiguousarray(arr.copy())


def render_shape(params, sx=1.0, sy=1.0):
    """把形状参数栅格化成 (h,w,4) uint8 RGBA（直线色）。

    (sx, sy) 是图层当前的缩放 —— **它决定栅格分辨率**，不是画完再拉伸。
    四周留描边一半宽度的 padding，否则描边的外半边会被裁掉。
    """
    p = normalize_shape_params(params)
    x0, y0, x1, y1 = shape_bbox(p)
    cw = max(1.0, x1 - x0)
    ch = max(1.0, y1 - y0)

    ax = abs(float(sx)) if abs(float(sx)) > 1e-4 else 1.0
    ay = abs(float(sy)) if abs(float(sy)) > 1e-4 else 1.0
    s = (ax + ay) / 2.0

    st = p["stroke"]
    # 直线没有填充：描边就是它本身，所以即便 stroke.off 也要给一个最小线宽
    if p["kind"] == SHAPE_LINE:
        lw = float(st["width"]) if st["on"] else max(1.0, DEFAULT_STROKE_WIDTH)
    else:
        lw = float(st["width"]) if st["on"] else 0.0
    pad = int(math.ceil(lw * s / 2.0)) + 2
    w = max(1, int(math.ceil(cw * ax)) + pad * 2)
    h = max(1, int(math.ceil(ch * ay)) + pad * 2)

    img = QImage(w, h, QImage.Format_ARGB32)
    img.fill(0)
    pn = QPainter(img)
    pn.setRenderHint(QPainter.Antialiasing, True)
    path = _build_path(p, x0, y0, ax, ay, pad)
    if path is not None:
        fl = p["fill"]
        if fl["on"] and p["kind"] != SHAPE_LINE:
            c = fl["color"]
            pn.fillPath(path, QColor(int(c[0]), int(c[1]), int(c[2]),
                                     int(round(fl["opacity"] * 255))))
        if p["kind"] == SHAPE_LINE or st["on"]:
            c = st["color"]
            pen = QPen(QColor(int(c[0]), int(c[1]), int(c[2]),
                              int(round(st["opacity"] * 255))),
                       max(0.05, lw * s))
            pen.setJoinStyle(Qt.RoundJoin)
            pen.setCapStyle(Qt.RoundCap)
            pn.setPen(pen)
            pn.strokePath(path, pen)
    pn.end()
    return _to_rgba(img)


# ---------------------------------------------------------------- 与图层对接

def _cache_key(p, sx, sy):
    """参数 + 缩放 -> 可比较的 key（决定栅格要不要重画）。"""
    return (p["kind"],
            tuple(round(v, 3) for v in p["box"]),
            round(float(p["radius"]), 3),
            int(p["sides"]), round(float(p["star"]), 3),
            bool(p["closed"]),
            tuple((round(a, 3), round(b, 3)) for a, b in p["points"]),
            tuple(tuple((round(q[0], 3), round(q[1], 3)) for q in seg)
                  for seg in p["segs"]),
            (bool(p["fill"]["on"]),
             tuple(p["fill"]["color"]), round(p["fill"]["opacity"], 3)),
            (bool(p["stroke"]["on"]),
             tuple(p["stroke"]["color"]),
             round(p["stroke"]["width"], 3),
             round(p["stroke"]["opacity"], 3)),
            round(float(sx), 4), round(float(sy), 4))


def sync_shape_image(layer):
    """按当前参数与缩放重新栅格化形状图层的位图。

    两点与 `core.text.sync_text_image` 不同：

    * 栅格尺寸里**含**当前的 sx/sy，画完把 sx/sy 归 1 —— 所以缩放是
      「换分辨率重画」而不是「把旧栅格拉大」，放大也不糊
    * tx/ty（图层中心）不变，所以形状是原地变大变小，位置不跳

    只在参数或缩放真的变了时才重画（靠 `_cache_key`），否则每次渲染都要
    重新画一遍。返回新的位图；不是形状图层时返回 None。
    """
    if layer is None or layer.kind != LAYER_SHAPE:
        return None
    if layer.shape is None:
        return None
    p = normalize_shape_params(layer.shape)
    sx = abs(float(layer.sx)) if abs(float(layer.sx)) > 1e-4 else 1.0
    sy = abs(float(layer.sy)) if abs(float(layer.sy)) > 1e-4 else 1.0
    key = _cache_key(p, sx, sy)
    if layer.image is not None and layer.__dict__.get("_shape_key") == key:
        return layer.image
    layer.image = render_shape(p, sx, sy)
    layer.__dict__["_shape_key"] = key
    # 尺寸已经烤进位图：sx/sy 归 1（保留符号，负缩放 = 翻转）
    layer.sx = math.copysign(1.0, layer.sx) if layer.sx else 1.0
    layer.sy = math.copysign(1.0, layer.sy) if layer.sy else 1.0
    return layer.image


def sync_shapes(doc):
    """把文档里所有形状图层的派生位图填好（渲染管线的入口调用）。"""
    if doc is None:
        return 0
    n = 0
    stack = list(getattr(doc, "layers", None) or [])
    while stack:
        l = stack.pop()
        if l.kind == LAYER_SHAPE:
            if sync_shape_image(l) is not None:
                n += 1
        if l.children:
            stack.extend(l.children)
    fl = getattr(doc, "float_layer", None)
    if fl is not None and fl.kind == LAYER_SHAPE:
        if sync_shape_image(fl) is not None:
            n += 1
    return n
