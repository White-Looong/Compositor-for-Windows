# -*- coding: utf-8 -*-
"""图层样式：描边 / 投影 / 内阴影 / 外发光。

Photoshop 的图层样式是**非破坏性**的：参数存在图层上，渲染时才生成。
这里一样——`layer.effects` 只存参数，`render.py` 在把图层变换进缓冲之后、
合成到画布之前调用 `apply_effects()`。

两个重要的语义约定（跟 PS 一致）：

1. **效果尺寸按画布像素算，不随图层变换缩放。** 把图层放大 4 倍，投影的
   模糊半径不会跟着变大 —— 所以效果必须在**变换之后**施加。
2. **效果需要的外扩空间由 `effect_padding()` 返回**，`render` 据此把图层
   的变换区域向外扩一圈，否则投影/发光会被裁掉。

所有函数操作的都是变换后的图层 patch：
    c —— (h,w,3) float32 0~1 直线色（未预乘）
    a —— (h,w,1) float32 0~1 不透明度
patch 四周已经有 pad 像素的空白，效果可以放心往外扩。
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .blend import composite

# 效果 id -> 中文名，UI 与测试都从这里取顺序
EFFECT_ORDER = ["drop_shadow", "outer_glow", "inner_shadow", "stroke"]
EFFECT_NAMES = {
    "drop_shadow": "投影",
    "outer_glow": "外发光",
    "inner_shadow": "内阴影",
    "stroke": "描边",
}

STROKE_POSITIONS = ["外部", "内部", "居中"]


def default_effects():
    """四组效果的默认参数（全部关闭）。"""
    return {
        "drop_shadow": {
            "enabled": False,
            "opacity": 0.6,
            "angle": 45.0,
            "distance": 8.0,
            "spread": 0.0,
            "size": 8.0,
            "color": [0, 0, 0],
        },
        "outer_glow": {
            "enabled": False,
            "opacity": 0.6,
            "spread": 0.0,
            "size": 12.0,
            "color": [255, 255, 190],
        },
        "inner_shadow": {
            "enabled": False,
            "opacity": 0.5,
            "angle": 45.0,
            "distance": 5.0,
            "spread": 0.0,
            "size": 6.0,
            "color": [0, 0, 0],
        },
        "stroke": {
            "enabled": False,
            "opacity": 1.0,
            "size": 3.0,
            "position": "外部",
            "color": [255, 255, 255],
        },
    }


def normalize_effects(effects):
    """补齐缺失的键，让旧工程 / 手写的参数也能安全使用。"""
    base = default_effects()
    if not isinstance(effects, dict):
        return base
    for key, dflt in base.items():
        cur = effects.get(key)
        if not isinstance(cur, dict):
            effects[key] = dict(dflt)
            continue
        for k, v in dflt.items():
            cur.setdefault(k, v)
    return effects


def has_effects(effects):
    if not effects:
        return False
    for key in EFFECT_ORDER:
        e = effects.get(key)
        if isinstance(e, dict) and e.get("enabled"):
            return True
    return False


_SCALE_KEYS = ("size", "distance", "spread")


def scale_effects(effects, s):
    """把效果里的长度参数按 s 缩放，返回新的 dict（不动原对象）。

    代理渲染（低分辨率预览）时用：效果的尺寸是按画布像素算的，画布整体
    缩小 s 倍之后，这些数值也必须跟着缩小，投影 / 发光的观感才对得上。
    """
    if not effects:
        return None
    s = float(s)
    if abs(s - 1.0) < 1e-6:
        return effects
    out = {}
    for key, cur in effects.items():
        if not isinstance(cur, dict):
            out[key] = cur
            continue
        d = dict(cur)
        for k in _SCALE_KEYS:
            if k in d and isinstance(d[k], (int, float)):
                d[k] = float(d[k]) * s
        out[key] = d
    return out


def effect_padding(effects):
    """渲染区域要向外扩多少像素，效果才不会被裁掉。"""
    if not has_effects(effects):
        return 0
    pad = 0
    for key in EFFECT_ORDER:
        e = effects.get(key)
        if not isinstance(e, dict) or not e.get("enabled"):
            continue
        if key == "stroke":
            size = abs(float(e.get("size", 0.0)))
            if e.get("position") == "内部":
                continue
            if e.get("position") == "居中":
                size = size / 2.0
            pad = max(pad, int(math.ceil(size)) + 1)
            continue
        spread = abs(float(e.get("spread", 0.0)))
        reach = _blur_reach(abs(float(e.get("size", 0.0))))
        dist = abs(float(e.get("distance", 0.0))) if "distance" in e else 0.0
        # 内阴影虽然不往外画（m ≤ a2），但它是"扩张 -> 模糊 -> 平移"算出来的，
        # 贴着 patch 边界算会缺上下文 —— 照样要外扩。
        pad = max(pad, int(math.ceil(spread + reach + dist)) + 1)
    return min(pad, 512)


def _blur_reach(size):
    """高斯模糊实际会波及的半径（像素）。

    别想当然写 3σ：`cv2.GaussianBlur(a, (0,0), sigma)` 在 ksize=0 时按
    `round(sigma * (depth==CV_8U ? 3 : 4) * 2 + 1)` 取核 —— **float32 走的是 4σ 那一档**，
    而这里的 alpha 就是 float32，所以半径 ≈ 4σ = 2*size。
    算小了会让 `effect_padding()` 不够大，投影 / 发光被 patch 边界截掉一截
    （局部重渲染时更明显：贴着区域边界算出来的结果和整幅对不上）。
    """
    return 4.0 * _sigma(size) + 1.0


def _sigma(size):
    return max(size, 0.0) / 2.0


def _dilate(a, r):
    """对 (h,w) float32 alpha 做圆形膨胀，r 为像素半径。"""
    ri = int(round(r))
    if ri <= 0:
        return a
    d = ri * 2 + 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))
    return cv2.dilate(a, k)


def _erode(a, r):
    ri = int(round(r))
    if ri <= 0:
        return a
    d = ri * 2 + 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))
    return cv2.erode(a, k)


def _blur(a, size):
    s = _sigma(size)
    if s <= 0.0:
        return a
    return cv2.GaussianBlur(a, (0, 0), s)


def _shift(a, dx, dy):
    """整数平移，移出的部分丢弃、空出来的部分补 0（不是环绕）。"""
    h, w = a.shape[:2]
    ix, iy = int(round(dx)), int(round(dy))
    out = np.zeros_like(a)
    xs0 = max(0, -ix)
    xs1 = w - max(0, ix)
    ys0 = max(0, -iy)
    ys1 = h - max(0, iy)
    xd0 = max(0, ix)
    xd1 = w - max(0, -ix)
    yd0 = max(0, iy)
    yd1 = h - max(0, -iy)
    if xs1 > xs0 and ys1 > ys0:
        out[yd0:yd1, xd0:xd1] = a[ys0:ys1, xs0:xs1]
    return out


def _offset(angle_deg, distance):
    """角度 -> 位移。屏幕坐标 y 向下，45° 指向右下（默认光来自左上）。"""
    r = math.radians(float(angle_deg))
    return distance * math.cos(r), distance * math.sin(r)


def _color_f32(e):
    col = e.get("color") or [0, 0, 0]
    return (np.array([col[0], col[1], col[2]], np.float32) / 255.0)


def _under(bc, ba, alpha2, color, mode="Normal"):
    """把一层纯色按 alpha 合成到 (bc, ba) 上（原地修改）。"""
    if alpha2.max() <= 0.0:
        return
    cc = np.zeros_like(bc)
    cc[..., :] = color
    composite(bc, ba, cc, alpha2[..., None].astype(np.float32), mode)


# ---------------------------------------------------------------- 各效果

def _shadow_alpha(a2, e):
    """投影 / 内阴影共用的「阴影形状」：扩张 -> 模糊 -> 平移。"""
    al = _dilate(a2, abs(float(e.get("spread", 0.0))))
    al = _blur(al, abs(float(e.get("size", 0.0))))
    dx, dy = _offset(e.get("angle", 45.0), float(e.get("distance", 0.0)))
    return _shift(al, dx, dy), dx, dy


def _apply_drop_shadow(bc, ba, a2, e):
    sh, _dx, _dy = _shadow_alpha(a2, e)
    # 抠掉内容本身占的地方：PS 的投影在图层下方，被不透明内容完全遮住
    sh = np.clip(sh * float(e.get("opacity", 1.0)), 0.0, 1.0)
    if sh.max() <= 0.0:
        return
    _under(bc, ba, sh, _color_f32(e), "Normal")


def _apply_outer_glow(bc, ba, a2, e):
    g = _dilate(a2, abs(float(e.get("spread", 0.0))))
    g = _blur(g, abs(float(e.get("size", 0.0))))
    g = np.clip(g * float(e.get("opacity", 1.0)), 0.0, 1.0)
    if g.max() <= 0.0:
        return
    _under(bc, ba, g, _color_f32(e), "Screen")


def _apply_inner_shadow(bc, ba, a2, e):
    sh, _dx, _dy = _shadow_alpha(a2, e)
    # 只在形状内部、且没被平移后的副本盖住的地方 → 就是光源对面的内边缘
    m = np.clip(a2 * (1.0 - sh), 0.0, 1.0) * float(e.get("opacity", 1.0))
    if m.max() <= 0.0:
        return
    # 内阴影不改 alpha，只压颜色
    col = _color_f32(e)
    bc[..., :] = bc * (1.0 - m[..., None]) + col * m[..., None]


def _apply_stroke(bc, ba, a2, e):
    size = abs(float(e.get("size", 0.0)))
    if size <= 0.0:
        return
    pos = e.get("position") or "外部"
    if pos == "外部":
        ring = np.clip(_dilate(a2, size) - a2, 0.0, 1.0)
    elif pos == "内部":
        ring = np.clip(a2 - _erode(a2, size), 0.0, 1.0)
    else:
        half = max(size / 2.0, 0.5)
        ring = np.clip(_dilate(a2, half) - _erode(a2, half), 0.0, 1.0)
    ring = ring * float(e.get("opacity", 1.0))
    if ring.max() <= 0.0:
        return
    col = _color_f32(e)
    cc = np.zeros_like(bc)
    cc[..., :] = col
    composite(bc, ba, cc, ring[..., None].astype(np.float32), "Normal")


# ---------------------------------------------------------------- 入口

def apply_effects(c, a, effects):
    """把图层样式作用到已变换的 patch 上，返回新的 (c, a)。

    c: (h,w,3) float32 直线色；a: (h,w,1) float32 0~1。
    没有启用任何效果时原样返回（不拷贝），避免每帧白白复制大数组。
    """
    if not has_effects(effects):
        return c, a
    a2 = a[..., 0]
    if float(a2.max()) <= 0.0:
        return c, a      # 全透明，效果无处附着

    bc = np.zeros_like(c)
    ba = np.zeros_like(a)

    e = effects.get("drop_shadow")
    if isinstance(e, dict) and e.get("enabled"):
        _apply_drop_shadow(bc, ba, a2, e)
    e = effects.get("outer_glow")
    if isinstance(e, dict) and e.get("enabled"):
        _apply_outer_glow(bc, ba, a2, e)

    # 内容本体压在阴影 / 发光之上
    composite(bc, ba, c, a, "Normal")

    e = effects.get("inner_shadow")
    if isinstance(e, dict) and e.get("enabled"):
        _apply_inner_shadow(bc, ba, a2, e)
    e = effects.get("stroke")
    if isinstance(e, dict) and e.get("enabled"):
        _apply_stroke(bc, ba, a2, e)
    return bc, np.clip(ba, 0.0, 1.0)
