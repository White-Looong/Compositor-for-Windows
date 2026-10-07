# -*- coding: utf-8 -*-
"""向导类工具的数据模型：参考线、网格、标尺、吸附。

全部是**纯几何计算 + 纯数据**，不碰 Qt 也不碰图层对象 —— 这样吸附判定
能被 selftest 直接测（拖动逻辑很难靠界面测准）。

坐标系一律是**画布坐标**（图层的 tx/ty 也在这套坐标里）。

三层结构，和 Photoshop 一致：
- **参考线** `Guide`：用户拖出来的线，分水平 / 垂直，可以从标尺拖出来
- **网格** `GridSpec`：间距 + 细分数（每大格再切几等分），只影响显示
- **智能参考线**（拖动时动态出现的那种）不算在内 —— 那是拖动过程里
  临时算出来给 `snap()` 用的候选，不存进doc

吸附 `snap()` 的优先级（**先到先得**，和 PS 一致）：
文档边界 > 参考线 > 网格 > 图层边缘 / 中心
"""

from __future__ import annotations

import math

import uuid

# 吸附目标的来源，按优先级从高到低。数值小的先赢。
SNAP_DOC = 0
SNAP_GUIDE = 1
SNAP_GRID = 2
SNAP_LAYER = 3

SNAP_SOURCE_NAME = {
    SNAP_DOC: "文档边界",
    SNAP_GUIDE: "参考线",
    SNAP_GRID: "网格",
    SNAP_LAYER: "图层",
}

# 方向
H_GUIDE = "h"      # 水平线（y = 常数）
V_GUIDE = "v"      # 垂直线（x = 常数）


def _new_id():
    return uuid.uuid4().hex[:12]


class Guide:
    """一条参考线。

    `pos` 是画布坐标：水平线是 y，垂直线是 x。
    `kind` 区分是**普通参考线**（用户拖出来的）还是**切片线**
    （从切片工具来的）—— 显示样式不同，但吸附行为一样。
    """

    __slots__ = ("id", "kind", "pos", "color")

    def __init__(self, kind, pos, color=(90, 160, 240)):
        self.id = _new_id()
        self.kind = kind                 # H_GUIDE / V_GUIDE
        self.pos = float(pos)
        self.color = tuple(color)

    @property
    def is_horizontal(self):
        return self.kind == H_GUIDE

    def to_dict(self):
        return {"id": self.id, "kind": self.kind, "pos": self.pos,
                "color": list(self.color)}

    @staticmethod
    def from_dict(d):
        g = Guide(d["kind"], d["pos"])
        g.id = d.get("id") or g.id
        if d.get("color"):
            g.color = tuple(d["color"])
        return g

    def copy(self):
        g = Guide(self.kind, self.pos, self.color)
        g.id = self.id
        return g

    def __repr__(self):
        return "Guide(%s %.1f)" % (self.kind, self.pos)


class GridSpec:
    """网格参数。间距以**画布像素**计，不随缩放变化（PS 就是这样）。

    `subdiv` 是每个大格再切几等分：subdiv=1 表示不细分（只有大格线），
    subdiv=4 表示每个大格里再画 3 条细线。

    `spacing` / `subdiv` 用 property 包了一层：改它们会把 `lines()` 的缓存
    打掉。**直接把 spacing 写进 __slots__ 会让缓存不失效**（改完间距
    吸附还按老间距走，这种 bug 很难查）。
    """

    __slots__ = ("visible", "_spacing", "_subdiv", "_cache")

    def __init__(self, visible=False, spacing=50.0, subdiv=1):
        self.visible = bool(visible)
        self._spacing = max(1.0, float(spacing))
        self._subdiv = max(1, int(subdiv))
        self._cache = None          # (w, h, xs, ys)

    @property
    def spacing(self):
        return self._spacing

    @spacing.setter
    def spacing(self, v):
        v = max(1.0, float(v))
        if v != self._spacing:
            self._spacing = v
            self._cache = None

    @property
    def subdiv(self):
        return self._subdiv

    @subdiv.setter
    def subdiv(self, v):
        v = max(1, int(v))
        if v != self._subdiv:
            self._subdiv = v
            self._cache = None

    def to_dict(self):
        return {"visible": self.visible, "spacing": self.spacing,
                "subdiv": self.subdiv}

    @staticmethod
    def from_dict(d):
        return GridSpec(d.get("visible", False),
                        d.get("spacing", 50.0), d.get("subdiv", 1))

    def copy(self):
        return GridSpec(self.visible, self.spacing, self.subdiv)

    def lines(self, w, h):
        """-> (竖线x列表, 横线y列表)，只给落在画布内的。

        细线在大格内部等距分布，**不含大格线本身**（免得画两遍）。
        结果按 (w, h) 缓存 —— 吸附每帧都要调，12 MP 上重算太贵。
        """
        if not self.visible or self.spacing <= 0:
            return [], []
        w = int(w)
        h = int(h)
        if self._cache is not None and self._cache[0] == (w, h):
            return self._cache[1], self._cache[2]
        xs, ys = [], []
        s = self.spacing
        x = s
        while x < w:
            xs.append(x)
            x += s
        y = s
        while y < h:
            ys.append(y)
            y += s
        if self.subdiv > 1:
            step = s / float(self.subdiv)
            fx, fy = [], []
            for k in range(1, self.subdiv):
                off = step * k
                x = s + off
                while x < w:
                    fx.append(x)
                    x += s
                y = s + off
                while y < h:
                    fy.append(y)
                    y += s
            xs.extend(fx)
            ys.extend(fy)
        xs.sort()
        ys.sort()
        self._cache = ((w, h), xs, ys)
        return xs, ys


class GuideSet:
    """文档级的参考线集合 + 网格 + 吸附开关。

    参考线**存进工程文件**（`.cwproj` 的 manifest），网格设置也存 ——
    都是用户摆好的布局，重开工程不该丢。
    """

    def __init__(self):
        self.guides = []
        self.grid = GridSpec()
        self.snap_guides = True       # 吸附到参考线
        self.snap_grid = True         # 吸附到网格
        self.snap_layers = True       # 吸附到图层边缘 / 中心
        self.snap_doc = True          # 吸附到文档边界

    # ---------- 参考线增删 ----------

    def add(self, kind, pos):
        g = Guide(kind, pos)
        self.guides.append(g)
        return g

    def remove(self, gid):
        for i, g in enumerate(self.guides):
            if g.id == gid:
                del self.guides[i]
                return True
        return False

    def clear(self):
        self.guides = []

    def find(self, gid):
        for g in self.guides:
            if g.id == gid:
                return g
        return None

    def nearest(self, kind, pos, tol):
        """离 pos 最近的一条同类参考线（|d| <= tol），返回 (guide, d) 或 None。

        `d` 是 pos - guide.pos，调用方拿它做偏移。
        """
        best = None
        for g in self.guides:
            if g.kind != kind:
                continue
            d = pos - g.pos
            if abs(d) <= tol and (best is None or abs(d) < abs(best[1])):
                best = (g, d)
        return best

    def sorted_lines(self, w, h):
        """-> (竖线列表, 横线列表)，带颜色，给绘制用。"""
        vs = [g for g in self.guides if g.kind == V_GUIDE
              and -w <= g.pos <= 2 * w]
        hs = [g for g in self.guides if g.kind == H_GUIDE
              and -h <= g.pos <= 2 * h]
        vs.sort(key=lambda g: g.pos)
        hs.sort(key=lambda g: g.pos)
        return vs, hs

    # ---------- 存取 ----------

    def to_dict(self):
        return {
            "guides": [g.to_dict() for g in self.guides],
            "grid": self.grid.to_dict(),
            "snap": {"guides": self.snap_guides, "grid": self.snap_grid,
                     "layers": self.snap_layers, "doc": self.snap_doc},
        }

    @staticmethod
    def from_dict(d):
        gs = GuideSet()
        for gd in (d or {}).get("guides") or []:
            try:
                gs.guides.append(Guide.from_dict(gd))
            except Exception:
                continue                     # 坏条目安静跳过
        gs.grid = GridSpec.from_dict((d or {}).get("grid") or {})
        sn = (d or {}).get("snap") or {}
        gs.snap_guides = bool(sn.get("guides", True))
        gs.snap_grid = bool(sn.get("grid", True))
        gs.snap_layers = bool(sn.get("layers", True))
        gs.snap_doc = bool(sn.get("doc", True))
        return gs


# =====================================================================
# 吸附判定
# =====================================================================

def layer_edges_and_centers(layer):
    """-> (竖线列表, 横线列表)：一个图层的可吸附位置。

    竖线 = 左右边界 + 垂直中心；横线 = 上下边界 + 水平中心。
    变换（含旋转与翻转）算进来 —— 旋转 45° 的矩形吸附到的是它变换后的边。

    尺寸取 `layer.src_size`（Layer 没有 w/h 属性，只有这个）。
    """
    import cv2
    import numpy as np

    src = layer.src_size
    if src is None:
        return [], []
    w, h = float(src[0]), float(src[1])
    M = layer.matrix(0.0, 0.0)
    if M is None:
        return [], []
    # **别用 cv2.transform**：它要 3x3 矩阵（OpenCV 5 明确
    # `scn == m.cols || scn + 1 == m.cols`，2x3 配 (N,2) 也会报断言失败）。
    # 这里只要四个角的变换，numpy 手算两行就够，也少一个依赖
    corners = np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]], np.float64)
    pts = corners @ M[:, :2].T + M[:, 2]
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    return ([min(xs), max(xs), (min(xs) + max(xs)) / 2.0],
            [min(ys), max(ys), (min(ys) + max(ys)) / 2.0])


def snap_point(x, y, gs, doc_w, doc_h, layers=(), tol=6.0, exclude=None):
    """把 (x, y) 吸到最近的候选线上。

    返回 (sx, sy, hit) ；`hit` 是 (kind, axis) 的列表，
    kind 取 SNAP_DOC / SNAP_GUIDE / SNAP_GRID / SNAP_LAYER，
    axis 是 "x" 或 "y"。都没吸到时 hit 为空。

    **x 与 y 各自独立判定**（PS 就是这样：可以只吸住水平方向）。
    优先级：文档边界 > 参考线 > 网格 > 图层。
    """
    sx, sy = float(x), float(y)
    hits = []

    def _try_axis(val, axis):
        """返回 (吸附后的值, kind 或 None)

        判定顺序就是优先级：**参考线 > 网格 > 图层 > 文档边界**。
        文档边界排最后 —— 它是「兜底」性质的（画布边缘），用户特意拉的
        参考线应该赢（PS 也是这样）。
        """
        # 1) 参考线
        if gs.snap_guides and gs.guides:
            kind = V_GUIDE if axis == "x" else H_GUIDE
            near = gs.nearest(kind, val, tol)
            if near is not None:
                return near[0].pos, SNAP_GUIDE
        # 2) 网格
        if gs.snap_grid and gs.grid.visible and gs.grid.spacing > 0:
            # **直接用 GridSpec.lines() 的结果**，别在这里另算一遍 ——
            # 另算的那版在大格与细线之间会跳（实测 35 吸到 40），
            # 而 lines() 已经保证「大格线 + 细线」严格等距递增
            xs, _ys = gs.grid.lines(max(doc_w, val + tol + 1.0),
                                    max(doc_h, 1.0))
            best = None
            for gx in xs:
                d = abs(val - gx)
                if d <= tol and (best is None or d < best[0]):
                    best = (d, gx)
            if best is not None:
                return best[1], SNAP_GRID
        # 3) 图层
        if gs.snap_layers and layers:
            best = None
            best_d = tol
            for lay in layers:
                if lay is exclude or lay.is_group or not lay.visible:
                    continue
                vs, hs = layer_edges_and_centers(lay)
                for c in (vs if axis == "x" else hs):
                    d = abs(val - c)
                    if d <= best_d:
                        best_d = d
                        best = c
            if best is not None:
                return best, SNAP_LAYER
        # 4) 文档边界。x 轴的候选是 0 与 doc_w；y 轴是 0 与 doc_h。
        #    **别把 doc_h 塞进 x 的候选里**（写岔了就是 x 方向会吸到下边界）
        if gs.snap_doc:
            span = float(doc_w) if axis == "x" else float(doc_h)
            for c in (0.0, span):
                if abs(val - c) <= tol:
                    return c, SNAP_DOC
        return val, None

    nx, kx = _try_axis(sx, "x")
    if kx is not None:
        sx = nx
        hits.append((kx, "x"))
    ny, ky = _try_axis(sy, "y")
    if ky is not None:
        sy = ny
        hits.append((ky, "y"))
    return sx, sy, hits


def snap_label(hits):
    """把命中的吸附源拼成状态栏提示，如「已吸附到：参考线 · 网格」。"""
    names = []
    for k, _axis in hits:
        nm = SNAP_SOURCE_NAME.get(k)
        if nm and nm not in names:
            names.append(nm)
    return "已吸附到：" + " · ".join(names) if names else ""


def ruler_ticks(lo, hi, spacing=10.0, step_px=80.0):
    """-> (位置列表, 主刻度下标集合)：标尺画哪几个刻度。

    `spacing` 是**世界坐标**上一个小格对应多少像素，`step_px` 是屏幕上
    至少隔多少像素才画一个 —— 缩得太小时自动跳到 5 / 10 / 50 的整数档，
    免得标尺挤成一团。
    """
    if spacing <= 0 or hi <= lo:
        return [], set()
    if step_px <= 0:
        step_px = 50.0
    # 找最小的整数档 mult，使 (spacing*mult) 折算成屏幕像素 >= step_px。
    # 换算比 = spacing / step_px（每世界像素占多少屏幕像素）
    ratio = spacing / float(step_px)
    mult = 1
    for cand in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000):
        if cand * ratio < 1.0:
            mult = cand
        else:
            break
    else:
        mult = 10000
    gap = spacing * mult
    first = math.floor(lo / gap) * gap
    out = []
    i = 0
    v = first
    while v <= hi + 1e-6:
        out.append(v)
        i += 1
        v += gap
    return out, set(range(0, len(out), 5))