# -*- coding: utf-8 -*-
"""图层样式：投影 / 外发光 / 内阴影 / 内发光 / 斜面浮雕 / 光泽 /
颜色叠加 / 渐变叠加 / 图案叠加 / 描边，外加一个「全局光」。

Photoshop 的图层样式是**非破坏性**的：参数存在图层上，渲染时才生成。
这里一样——`layer.effects` 只存参数，`render.py` 在把图层变换进缓冲之后、
合成到画布之前调用 `apply_effects()`。

两个重要的语义约定（跟 PS 一致）：

1. **效果尺寸按画布像素算，不随图层变换缩放。** 把图层放大 4 倍，投影的
   模糊半径不会跟着变大 —— 所以效果必须在**变换之后**施加。
2. **效果需要的外扩空间由 `effect_padding()` 返回**，`render` 据此把图层
   的变换区域向外扩一圈，否则投影/发光会被裁掉。

第三个容易踩的：**坐标原点不能用 patch 左上角**。
渐变叠加 / 图案叠加如果按 patch 坐标画，局部重渲染和分块渲染时每块的
"第 0 行"位置不一样，同一条渐变会被渲到不同位置 —— 画面直接错。
所以这两种效果走 `frame` 参数：它是"图层内容包围盒"坐标系，由
`render.py` 按图层自己的 bbox 算出来传进来，与渲染区域无关。

所有函数操作的都是变换后的图层 patch：
    c —— (h,w,3) float32 0~1 直线色（未预乘）
    a —— (h,w,1) float32 0~1 不透明度
patch 四周已经有 pad 像素的空白，效果可以放心往外扩。

合成顺序（`apply_effects` 里）：

    投影 / 外发光                画在内容下面的 bc/ba 上
    内容本体                     composite 到 bc/ba
    内阴影 / 内发光
    斜面浮雕 / 光泽
    颜色 / 渐变 / 图案叠加
    描边                         最上面

其中"内阴影 / 内发光 / 斜面 / 光泽 / 三种叠加"**只改颜色不动 alpha**
（`_tint()` 会把 alpha 盖回去），投影 / 外发光 / 描边则可以扩大形状的 alpha。
"""

from __future__ import annotations

import copy
import json
import math
import os

import cv2
import numpy as np

from .blend import composite

# 效果 id -> 中文名，UI 与测试都从这里取顺序（顺序 = PS 面板从上到下）
EFFECT_ORDER = [
    "drop_shadow", "outer_glow", "inner_shadow", "inner_glow",
    "bevel_emboss", "satin",
    "color_overlay", "gradient_overlay", "pattern_overlay",
    "stroke",
]
EFFECT_NAMES = {
    "drop_shadow": "投影",
    "outer_glow": "外发光",
    "inner_shadow": "内阴影",
    "inner_glow": "内发光",
    "bevel_emboss": "斜面浮雕",
    "satin": "光泽",
    "color_overlay": "颜色叠加",
    "gradient_overlay": "渐变叠加",
    "pattern_overlay": "图案叠加",
    "stroke": "描边",
}

# 不是"效果"，而是一组被若干效果共用的光照参数。放在同一个 dict 里既不用
# 改数据模型，也能跟着图层存进工程文件（不在 EFFECT_ORDER 里，不影响
# has_effects / effect_padding）。
GLOBAL_LIGHT = "global_light"

STROKE_POSITIONS = ["外部", "内部", "居中"]
BEVEL_STYLES = ["内斜面", "外斜面", "浮雕", "枕状浮雕"]
BEVEL_DIRECTIONS = ["上", "下"]
GRADIENT_STYLES = ["线性", "径向", "角度", "对称", "菱形"]
PATTERN_KINDS = ["点阵", "网格", "斜纹", "棋盘", "噪声"]


def _factory_defaults():
    """出厂默认：全部关闭。用户「存为默认值」之后不再是这个值。"""
    return {
        "drop_shadow": {
            "enabled": False,
            "opacity": 0.6,
            "use_global": True,
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
            "use_global": True,
            "angle": 45.0,
            "distance": 5.0,
            "spread": 0.0,
            "size": 6.0,
            "color": [0, 0, 0],
        },
        "inner_glow": {
            "enabled": False,
            "opacity": 0.6,
            "blend": "Screen",
            "spread": 0.0,
            "size": 10.0,
            "color": [255, 255, 220],
        },
        "bevel_emboss": {
            "enabled": False,
            "style": "内斜面",
            "direction": "上",
            "size": 6.0,
            "soften": 0.0,
            "depth": 1.0,
            "use_global": True,
            "angle": 45.0,
            "altitude": 30.0,
            "highlight_opacity": 0.7,
            "highlight_color": [255, 255, 255],
            "highlight_blend": "Screen",
            "shadow_opacity": 0.7,
            "shadow_color": [0, 0, 0],
            "shadow_blend": "Multiply",
        },
        "satin": {
            "enabled": False,
            "opacity": 0.4,
            "blend": "Multiply",
            "angle": 90.0,
            "distance": 20.0,
            "size": 12.0,
            "invert": False,
            "color": [30, 30, 90],
        },
        "color_overlay": {
            "enabled": False,
            "opacity": 0.5,
            "blend": "Normal",
            "color": [255, 0, 0],
        },
        "gradient_overlay": {
            "enabled": False,
            "opacity": 0.5,
            "blend": "Normal",
            "style": "线性",
            "angle": 90.0,
            "color": [255, 240, 0],
            "color2": [0, 90, 255],
        },
        "pattern_overlay": {
            "enabled": False,
            "opacity": 0.5,
            "blend": "Multiply",
            "pattern": "点阵",
            "scale": 24.0,
            "color": [30, 30, 30],
        },
        "stroke": {
            "enabled": False,
            "opacity": 1.0,
            "size": 3.0,
            "position": "外部",
            "color": [255, 255, 255],
        },
        GLOBAL_LIGHT: {
            "enabled": False,
            "angle": 45.0,
            "altitude": 30.0,
        },
    }


# 「存为默认值」的参数（None = 还没存过）
_SAVED_DEFAULT = {"v": None}

# 样式剪贴板（进程内，够用；不做跨实例）
_STYLE_CLIPBOARD = {"v": None}


def _defaults_path():
    d = os.path.join(os.path.expanduser("~"), ".compositor_win")
    return os.path.join(d, "style_defaults.json")


# ------------------------------------------------------ 默认值的存取

def factory_default_effects():
    """出厂默认（忽略用户存过的默认值）—— 对话框的「复位」用它。"""
    return copy.deepcopy(_factory_defaults())


def default_effects():
    """新建图层样式时的初始参数：用户存过默认值就从那份出发。"""
    saved = _SAVED_DEFAULT["v"]
    if saved is None:
        return copy.deepcopy(_factory_defaults())
    return copy.deepcopy(saved)


def has_style_default():
    """有没有存过自定义默认值。"""
    return _SAVED_DEFAULT["v"] is not None


def save_style_default(effects):
    """把这份参数存成默认值，之后新建的样式都从它出发。

    持久化到用户目录的 style_defaults.json；写不进去（家目录只读等）就
    只留在内存里，不影响当前这次使用。
    """
    base = normalize_effects(copy.deepcopy(effects) if effects else None)
    _SAVED_DEFAULT["v"] = copy.deepcopy(base)
    try:
        p = _defaults_path()
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(base, f, ensure_ascii=False, indent=1)
    except Exception:
        pass
    return _SAVED_DEFAULT["v"]


def reset_style_default():
    """清掉用户存的默认值，回到出厂状态。"""
    _SAVED_DEFAULT["v"] = None
    try:
        p = _defaults_path()
        if os.path.exists(p):
            os.remove(p)
    except Exception:
        pass


def _load_style_default():
    """进程启动时读一次用户存的默认值（`default_effects()` 会用）。"""
    try:
        p = _defaults_path()
        if not os.path.exists(p):
            return
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        if isinstance(d, dict):
            _SAVED_DEFAULT["v"] = normalize_effects(d)
    except Exception:
        pass


_load_style_default()


# ------------------------------------------------------ 样式剪贴板

def copy_style(effects):
    """复制图层样式到剪贴板（深拷贝，之后改图层不会再跟着变）。"""
    if not effects:
        _STYLE_CLIPBOARD["v"] = None
        return None
    _STYLE_CLIPBOARD["v"] = copy.deepcopy(effects)
    return _STYLE_CLIPBOARD["v"]


def style_clipboard():
    return _STYLE_CLIPBOARD["v"]


def has_style_clipboard():
    return _STYLE_CLIPBOARD["v"] is not None


def paste_style():
    """-> 剪贴板里的样式副本（深拷贝），没有则返回 None。"""
    v = _STYLE_CLIPBOARD["v"]
    if v is None:
        return None
    return normalize_effects(copy.deepcopy(v))


# ------------------------------------------------------ 归一化 / 查询

def normalize_effects(effects):
    """补齐缺失的键，让旧工程 / 手写的参数也能安全使用。

    老工程里只有 4 种效果，读回来缺的那几种由这里补成默认（全是关闭的），
    不会让画面莫名多出斜面浮雕来。
    """
    base = _factory_defaults()
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


def enabled_effects(effects):
    """启用的效果 id 列表（按 EFFECT_ORDER）。"""
    if not effects:
        return []
    return [k for k in EFFECT_ORDER
            if isinstance(effects.get(k), dict) and effects[k].get("enabled")]


_SCALE_KEYS = ("size", "distance", "spread")


def scale_effects(effects, s):
    """把效果里的长度参数按 s 缩放，返回新的 dict（不动原对象）。

    代理渲染（低分辨率预览）时用：效果的尺寸是按画布像素算的，画布整体
    缩小 s 倍之后，这些数值也必须跟着缩小，投影 / 发光的观感才对得上。

    图案叠加的 `scale`（图案周期）**故意不缩**：它的坐标基准是图层内容
    包围盒，代理文档里那个盒已经缩了 s 倍，再缩一次就是双重缩放。
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


# 纯叠加类效果完全画在图层内部，不需要往外扩
_INNER_ONLY = ("color_overlay", "gradient_overlay", "pattern_overlay")


def effect_padding(effects):
    """渲染区域要向外扩多少像素，效果才不会被裁掉。"""
    if not has_effects(effects):
        return 0
    pad = 0
    for key in EFFECT_ORDER:
        e = effects.get(key)
        if not isinstance(e, dict) or not e.get("enabled"):
            continue
        if key in _INNER_ONLY:
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
        if key == "bevel_emboss":
            # 斜面是"平移叠加"算出来的，平移量最大就是 size + soften
            reach += abs(float(e.get("size", 0.0))) + \
                abs(float(e.get("soften", 0.0)))
        dist = abs(float(e.get("distance", 0.0))) if "distance" in e else 0.0
        # 内阴影虽然不往外画（m ≤ a2），但它是"扩张 -> 模糊 -> 平移"算出来的，
        # 贴着 patch 边界算会缺上下文 —— 照样要外扩。
        pad = max(pad, int(math.ceil(spread + reach + dist)) + 1)
    return min(pad, 512)


# ---------------------------------------------------------------- 工具

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


def _clamp(v, lo, hi):
    return max(lo, min(hi, float(v)))


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


def _pix_angle(angle_deg, distance, altitude=90.0):
    """带高度角的位移：altitude 越接近 90°（光源在正上方），位移越短。"""
    dx, dy = _offset(angle_deg, distance)
    k = math.cos(math.radians(_clamp(altitude, 0.0, 90.0)))
    return dx * k, dy * k


def _color_f32(e, key="color"):
    col = e.get(key) if isinstance(e, dict) else None
    col = list(col) if col is not None else [0, 0, 0]
    while len(col) < 3:
        col.append(0)
    return (np.array([col[0], col[1], col[2]], np.float32) / 255.0)


def _color_array(shape, col):
    """把 0~255 的 RGB 铺成 (h,w,3) float32。"""
    c = _color_f32({"color": col})
    out = np.empty((shape[0], shape[1], 3), np.float32)
    out[..., :] = c
    return out


def _under(bc, ba, alpha2, color, mode="Normal"):
    """把一层纯色按 alpha 合成到 (bc, ba) 上（原地修改）。

    投影 / 外发光用：它们画在内容**下面**，会撑大形状的 alpha。
    """
    if alpha2.max() <= 0.0:
        return
    cc = np.zeros_like(bc)
    cc[..., :] = color
    composite(bc, ba, cc, alpha2[..., None].astype(np.float32), mode)


def _tint(bc, ba, m, cc, mode):
    """按 mask m 叠一层颜色到 bc 上，**alpha 保持不变**。

    所有"只改颜色不改形状"的效果（内阴影 / 内发光 / 斜面 / 光泽 / 叠加类）
    都走这里 —— 组合出来的 alpha 会动（composite 是完整的混合公式），
    算完把原来的 alpha 盖回去就行。
    """
    m = np.clip(m, 0.0, 1.0)
    if m.max() <= 0.0:
        return
    keep = ba.copy()
    composite(bc, ba, cc, m[..., None].astype(np.float32), mode)
    ba[...] = keep


# ---------------------------------------------------------------- 全局光

def global_light(effects):
    """-> 全局光参数 dict；没启用返回 None。"""
    if not isinstance(effects, dict):
        return None
    g = effects.get(GLOBAL_LIGHT)
    if not isinstance(g, dict) or not g.get("enabled"):
        return None
    return g


def effective_angle(effects, e):
    """某个效果实际用的光照角度（勾了「使用全局光」就跟着全局走）。"""
    if isinstance(e, dict) and e.get("use_global"):
        g = global_light(effects)
        if g is not None:
            return float(g.get("angle", 45.0))
    return float(e.get("angle", 45.0)) if isinstance(e, dict) else 45.0


def effective_altitude(effects, e):
    """同上，取高度角（目前只有斜面浮雕用）。"""
    if isinstance(e, dict) and e.get("use_global"):
        g = global_light(effects)
        if g is not None:
            return float(g.get("altitude", 30.0))
    return float(e.get("altitude", 30.0)) if isinstance(e, dict) else 30.0


# ---------------------------------------------------------------- 坐标

def frame_grid(shape, frame):
    """patch 内每像素在"内容框坐标系"里的 X / Y（float32）。

    frame = (ox, oy, fw, fh)：ox/oy 是 patch 左上角相对内容框左上角的偏移，
    fw/fh 是内容框的尺寸。没有 frame 时退化成 patch 自己的坐标。
    """
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    xx = xx.astype(np.float32)
    yy = yy.astype(np.float32)
    if frame is None:
        return xx, yy
    return xx + float(frame[0]), yy + float(frame[1])


# ---------------------------------------------------------------- 渐变 / 图案

def _gradient_colors(shape, frame, e):
    """-> (h,w,3) float32：渐变叠加在每个像素上的颜色。"""
    X, Y = frame_grid(shape, frame)
    fw = float(frame[2]) if frame else float(shape[1])
    fh = float(frame[3]) if frame else float(shape[0])
    fw, fh = max(fw, 1.0), max(fh, 1.0)
    cx, cy = fw * 0.5, fh * 0.5
    ang = math.radians(float(e.get("angle", 90.0)))
    style = e.get("style") or "线性"

    def _linear():
        ux, uy = math.cos(ang), math.sin(ang)
        half = max(0.5 * (abs(ux) * fw + abs(uy) * fh), 1.0)
        return ((X - cx) * ux + (Y - cy) * uy) / half * 0.5 + 0.5

    if style == "径向":
        rad = max(0.5 * math.hypot(fw, fh), 1.0)
        t = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) / rad
    elif style == "角度":
        t = (np.arctan2(Y - cy, X - cx) - ang) / (2.0 * math.pi)
        t = t - np.floor(t)
    elif style == "菱形":
        rad = max(0.25 * (fw + fh), 1.0)
        t = (np.abs(X - cx) + np.abs(Y - cy)) / rad
    elif style == "对称":
        t = np.abs(2.0 * _linear() - 1.0)
    else:
        t = _linear()
    t = np.clip(t, 0.0, 1.0).astype(np.float32)

    c0 = _color_f32(e, "color")
    c1 = _color_f32(e, "color2")
    return c0 + (c1 - c0) * t[..., None]


_NOISE_TILE = {}


def _noise_tile(n=256):
    """固定 seed 的噪声底片（不能每次随变，否则渲染结果不稳定）。"""
    t = _NOISE_TILE.get(n)
    if t is None:
        rng = np.random.default_rng(20261006)
        g = rng.random((n, n)).astype(np.float32)
        g = cv2.GaussianBlur(g, (0, 0), 1.0)
        lo, hi = float(g.min()), float(g.max())
        t = np.clip((g - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        _NOISE_TILE[n] = t
    return t


def _pattern_mask(shape, frame, e):
    """-> (h,w) float32 0~1：图案在每个像素上的强度。"""
    X, Y = frame_grid(shape, frame)
    sc = max(float(e.get("scale", 24.0)), 2.0)
    u = X / sc
    v = Y / sc
    kind = e.get("pattern") or PATTERN_KINDS[0]
    if kind == "棋盘":
        iu = np.floor(u).astype(np.int64)
        iv = np.floor(v).astype(np.int64)
        return np.where(((iu + iv) % 2) == 0, 1.0, 0.0).astype(np.float32)
    if kind == "噪声":
        tile = _noise_tile()
        n = tile.shape[0]
        iu = np.floor(u).astype(np.int64) % n
        iv = np.floor(v).astype(np.int64) % n
        return tile[iv, iu]
    fu = u - np.floor(u)
    fv = v - np.floor(v)
    if kind == "点阵":
        d = np.sqrt((fu - 0.5) ** 2 + (fv - 0.5) ** 2)
        return np.clip(1.0 - d / 0.34, 0.0, 1.0).astype(np.float32)
    if kind == "网格":
        d = np.minimum(np.minimum(fu, 1.0 - fu), np.minimum(fv, 1.0 - fv))
        return np.clip(1.0 - d / 0.10, 0.0, 1.0).astype(np.float32)
    # 斜纹：沿对角线的一道软边条
    s = (u + v)
    s = s - np.floor(s)
    band = (np.clip((s - 0.22) / 0.06, 0.0, 1.0)
            * np.clip((0.78 - s) / 0.06, 0.0, 1.0))
    return band.astype(np.float32)


def pattern_mask(shape, frame, pattern, scale):
    """给画笔纹理用的公开入口：按内容框坐标生成 (h,w) 的图案强度 0~1。

    和图层样式的图案叠加共用同一套图案，所以两边看起来是一致的。
    """
    return _pattern_mask(shape, frame, {"pattern": pattern, "scale": scale})


# ---------------------------------------------------------------- 各效果

def _shadow_alpha(a2, e, effects=None):
    """投影 / 内阴影共用的「阴影形状」：扩张 -> 模糊 -> 平移。"""
    al = _dilate(a2, abs(float(e.get("spread", 0.0))))
    al = _blur(al, abs(float(e.get("size", 0.0))))
    ang = effective_angle(effects, e)
    dx, dy = _offset(ang, float(e.get("distance", 0.0)))
    return _shift(al, dx, dy), dx, dy


def _apply_drop_shadow(bc, ba, a2, e, effects=None):
    sh, _dx, _dy = _shadow_alpha(a2, e, effects)
    # 抠掉内容本身占的地方：PS 的投影在图层下方，被不透明内容完全遮住
    sh = np.clip(sh * float(e.get("opacity", 1.0)), 0.0, 1.0)
    if sh.max() <= 0.0:
        return
    _under(bc, ba, sh, _color_f32(e), "Normal")


def _apply_outer_glow(bc, ba, a2, e, effects=None):
    g = _dilate(a2, abs(float(e.get("spread", 0.0))))
    g = _blur(g, abs(float(e.get("size", 0.0))))
    g = np.clip(g * float(e.get("opacity", 1.0)), 0.0, 1.0)
    if g.max() <= 0.0:
        return
    _under(bc, ba, g, _color_f32(e), "Screen")


def _apply_inner_shadow(bc, ba, a2, e, effects=None):
    sh, _dx, _dy = _shadow_alpha(a2, e, effects)
    # 只在形状内部、且没被平移后的副本盖住的地方 → 就是光源对面的内边缘
    m = np.clip(a2 * (1.0 - sh), 0.0, 1.0) * float(e.get("opacity", 1.0))
    if m.max() <= 0.0:
        return
    # 内阴影不改 alpha，只压颜色
    col = _color_f32(e)
    bc[..., :] = bc * (1.0 - m[..., None]) + col * m[..., None]


def _apply_inner_glow(bc, ba, a2, e, effects=None):
    """内发光：把形状往里收一圈再模糊，与原形状作差 → 留在边缘内侧。"""
    spread = abs(float(e.get("spread", 0.0)))
    size = abs(float(e.get("size", 0.0)))
    inner = _erode(a2, max(spread, size * 0.5))
    glow = _blur(inner, size) if size > 0.0 else inner
    m = np.clip(a2 - glow, 0.0, 1.0) * float(e.get("opacity", 1.0))
    if m.max() <= 0.0:
        return
    _tint(bc, ba, m, _color_array(a2.shape, e.get("color")),
          e.get("blend") or "Screen")


def _bevel_masks(a2, e, effects=None):
    """斜面浮雕的受光 / 背光遮罩 -> (lit, dark)。

    做法：沿光照方向做 n 次逐步平移再取平均，得到一个"斜坡"；
    形状减去斜坡 = 受光面（投影方向上那儿先空出来），反向就是背光面。
    """
    if float(a2.max()) <= 0.0:
        return None, None
    size = max(float(e.get("size", 0.0)), 1.0)
    style = e.get("style") or BEVEL_STYLES[0]
    dist = size * 2.0 if style == "浮雕" else size
    dx, dy = _pix_angle(effective_angle(effects, e), dist,
                        effective_altitude(effects, e))

    if style == "外斜面":
        work = _dilate(a2, max(size * 0.5, 1.0))
    elif style == "枕状浮雕":
        work = _erode(a2, max(size * 0.5, 1.0))
    else:
        work = a2
    soften = max(float(e.get("soften", 0.0)), 0.0)
    if soften > 0.0:
        work = _blur(work, soften)

    n = max(2, int(round(dist)))
    acc_f = np.zeros_like(work)
    acc_b = np.zeros_like(work)
    for i in range(1, n + 1):
        t = i / float(n)
        acc_f += _shift(work, dx * t, dy * t)
        acc_b += _shift(work, -dx * t, -dy * t)
    lit = np.clip(work - acc_f / float(n), 0.0, 1.0)
    dark = np.clip(work - acc_b / float(n), 0.0, 1.0)
    if style == "枕状浮雕":
        lit, dark = dark, lit
    if (e.get("direction") or "上") == "下":
        lit, dark = dark, lit
    # 效果只能画在图层自己占的地方，不许改变形状
    return np.minimum(lit, a2), np.minimum(dark, a2)


def _apply_bevel_emboss(bc, ba, a2, e, effects=None):
    lit, dark = _bevel_masks(a2, e, effects)
    depth = _clamp(float(e.get("depth", 1.0)), 0.0, 3.0)
    ho = _clamp(float(e.get("highlight_opacity", 0.7)) * depth, 0.0, 1.0)
    so = _clamp(float(e.get("shadow_opacity", 0.7)) * depth, 0.0, 1.0)
    if lit is not None and ho > 0.0:
        _tint(bc, ba, lit * ho,
              _color_array(a2.shape, e.get("highlight_color")),
              e.get("highlight_blend") or "Screen")
    if dark is not None and so > 0.0:
        _tint(bc, ba, dark * so,
              _color_array(a2.shape, e.get("shadow_color")),
              e.get("shadow_blend") or "Multiply")


def _apply_satin(bc, ba, a2, e, effects=None):
    """光泽：形状向光源方向与其反向各平移一次，两者的差就是"绸缎"明暗带。"""
    size = abs(float(e.get("size", 0.0)))
    dx, dy = _offset(effective_angle(effects, e),
                     float(e.get("distance", 0.0)))
    base = _blur(a2, size * 0.5) if size > 0.0 else a2
    fwd = _shift(base, dx, dy)
    bck = _shift(base, -dx, -dy)
    if size > 0.0:
        fwd = _blur(fwd, size)
        bck = _blur(bck, size)
    m = np.abs(fwd - bck)
    if e.get("invert"):
        m = 1.0 - m
    m = np.clip(m * float(e.get("opacity", 1.0)), 0.0, 1.0) * a2
    if m.max() <= 0.0:
        return
    _tint(bc, ba, m, _color_array(a2.shape, e.get("color")),
          e.get("blend") or "Multiply")


def _apply_color_overlay(bc, ba, a2, e, effects=None, frame=None):
    m = np.clip(a2 * float(e.get("opacity", 1.0)), 0.0, 1.0)
    if m.max() <= 0.0:
        return
    _tint(bc, ba, m, _color_array(a2.shape, e.get("color")),
          e.get("blend") or "Normal")


def _apply_gradient_overlay(bc, ba, a2, e, effects=None, frame=None):
    m = np.clip(a2 * float(e.get("opacity", 1.0)), 0.0, 1.0)
    if m.max() <= 0.0:
        return
    _tint(bc, ba, m, _gradient_colors(a2.shape, frame, e),
          e.get("blend") or "Normal")


def _apply_pattern_overlay(bc, ba, a2, e, effects=None, frame=None):
    m = np.clip(a2 * float(e.get("opacity", 1.0)), 0.0, 1.0)
    if m.max() <= 0.0:
        return
    m = m * _pattern_mask(a2.shape, frame, e)
    if m.max() <= 0.0:
        return
    _tint(bc, ba, m, _color_array(a2.shape, e.get("color")),
          e.get("blend") or "Multiply")


def _apply_stroke(bc, ba, a2, e, effects=None):
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

def apply_effects(c, a, effects, frame=None):
    """把图层样式作用到已变换的 patch 上，返回新的 (c, a)。

    c: (h,w,3) float32 直线色；a: (h,w,1) float32 0~1。
    frame: (ox, oy, fw, fh) 内容框坐标系，渐变 / 图案叠加要靠它保证
           局部渲染与整幅一致，见模块 docstring。
    没有启用任何效果时原样返回（不拷贝），避免每帧白白复制大数组。
    """
    if not has_effects(effects):
        return c, a
    a2 = a[..., 0]
    if float(a2.max()) <= 0.0:
        return c, a      # 全透明，效果无处附着

    bc = np.zeros_like(c)
    ba = np.zeros_like(a)

    def _on(d):
        e = effects.get(d)
        return isinstance(e, dict) and e.get("enabled")

    # 画在内容下面的
    if _on("drop_shadow"):
        _apply_drop_shadow(bc, ba, a2, effects["drop_shadow"], effects)
    if _on("outer_glow"):
        _apply_outer_glow(bc, ba, a2, effects["outer_glow"], effects)

    # 内容本体压在阴影 / 发光之上
    composite(bc, ba, c, a, "Normal")

    # 内部效果：只改颜色，不改 alpha
    if _on("inner_shadow"):
        _apply_inner_shadow(bc, ba, a2, effects["inner_shadow"], effects)
    if _on("inner_glow"):
        _apply_inner_glow(bc, ba, a2, effects["inner_glow"], effects)
    if _on("bevel_emboss"):
        _apply_bevel_emboss(bc, ba, a2, effects["bevel_emboss"], effects)
    if _on("satin"):
        _apply_satin(bc, ba, a2, effects["satin"], effects)
    if _on("color_overlay"):
        _apply_color_overlay(bc, ba, a2, effects["color_overlay"], effects,
                             frame)
    if _on("gradient_overlay"):
        _apply_gradient_overlay(bc, ba, a2, effects["gradient_overlay"],
                                effects, frame)
    if _on("pattern_overlay"):
        _apply_pattern_overlay(bc, ba, a2, effects["pattern_overlay"],
                               effects, frame)
    if _on("stroke"):
        _apply_stroke(bc, ba, a2, effects["stroke"], effects)
    return bc, np.clip(ba, 0.0, 1.0)
