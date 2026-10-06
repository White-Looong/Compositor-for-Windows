# -*- coding: utf-8 -*-
"""调整层：非破坏性色彩调整。

每个调整项是一个**纯函数**

    func(rgb, params) -> rgb

rgb 是 (h, w, 3) float32、0~1 的**直线色**（non-premultiplied）；
只改颜色，不动 alpha —— alpha 由渲染管线处理。

调整层在渲染时作用于「它下面所有图层的合成结果」，所以图层顺序、
不透明度、蒙版、混合模式对调整层都照常生效（跟 Photoshop 一致）。
参数改动不碰像素，随时可以改回去，这就是"非破坏性"。

Param / Spec 只描述参数长什么样，UI 据此自动生成控件，
所以新增一个调整项只要写函数 + 参数表，不用动界面代码。
"""

from __future__ import annotations

import copy

import cv2
import numpy as np


class Param:
    """一个可调参数的描述。kind 决定 UI 生成什么控件。

    kind:
        int / double   滑块 + 数值
        bool           复选框
        choice         下拉框
        curve          曲线编辑器
        color          颜色按钮（值为 [r,g,b] 0~255）
        keyed          分组数值：值形如 {分组名: {子键: 数值}}，
                       面板上用下拉框选分组、下面挂一组滑块
    """

    def __init__(self, key, label, kind="double", lo=0.0, hi=1.0,
                 default=0.0, step=1.0, choices=None, decimals=2,
                 keys=None, sub=None):
        self.key = key
        self.label = label
        self.kind = kind
        self.lo = lo
        self.hi = hi
        self.default = default
        self.step = step
        self.choices = list(choices or [])
        self.decimals = decimals
        # keyed 专用：分组名列表 + 每组的子参数
        self.keys = list(keys or [])
        self.sub = list(sub or [])
        # 滑块用整数走，double 乘这个系数后再除回去
        self.scale = 1000 if kind == "double" else 1


def keyed_default(keys, sub):
    """keyed 参数的默认值：{分组名: {子键: 子默认值}}。"""
    return dict((k, dict((s.key, s.default) for s in sub)) for k in keys)


class Spec:
    def __init__(self, key, name, params, func, histogram=False):
        self.key = key
        self.name = name
        self.params = list(params)
        self.func = func
        self.histogram = histogram  # 是否需要在面板上显示直方图

    def defaults(self):
        # dict 型默认值（keyed / color）必须深拷贝，否则每个新层共享同一个对象
        out = {}
        for p in self.params:
            v = p.default
            out[p.key] = copy.deepcopy(v) if isinstance(v, (dict, list)) else v
        return out


# ---------------------------------------------------------------- 工具

_CH_IDX = {"RGB": [0, 1, 2], "R": [0], "G": [1], "B": [2]}


def _chan_idx(name):
    return _CH_IDX.get(name, _CH_IDX["RGB"])


def _srgb_to_linear(v):
    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.04045, v / 12.92,
                    np.power((v + 0.055) / 1.055, 2.4))


def _linear_to_srgb(v):
    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.0031308, v * 12.92,
                    1.055 * np.power(v, 1.0 / 2.4) - 0.055)


def _luma(rgb):
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def _hsv(rgb):
    """RGB(0~1) -> HSV，H 在 0~360，S/V 在 0~1。"""
    src = np.ascontiguousarray(rgb, dtype=np.float32)
    return cv2.cvtColor(src, cv2.COLOR_RGB2HSV)


def _rgb(hsv):
    src = np.ascontiguousarray(hsv, dtype=np.float32)
    return cv2.cvtColor(src, cv2.COLOR_HSV2RGB)


# ---------------------------------------------------------------- 色阶

def _levels(rgb, p):
    ib = p["in_black"] / 255.0
    iw = max(p["in_white"] / 255.0, ib + 1.0 / 255.0)
    g = max(p["gamma"], 0.01)
    ob = p["out_black"] / 255.0
    ow = p["out_white"] / 255.0
    idx = _chan_idx(p["channel"])

    out = rgb.copy()
    x = np.clip((out[..., idx] - ib) / (iw - ib), 0.0, 1.0)
    if abs(g - 1.0) > 1e-4:
        x = np.power(x, 1.0 / g)
    out[..., idx] = ob + x * (ow - ob)
    return out


# ---------------------------------------------------------------- 曲线

def _monotone_lut(pts, n=256):
    """单调三次 Hermite 插值（Fritsch–Carlson），不会像 Catmull-Rom 那样过冲。"""
    xs = np.array([q[0] for q in pts], dtype=np.float64)
    ys = np.array([q[1] for q in pts], dtype=np.float64)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    m = len(xs)
    t = np.arange(n, dtype=np.float64) / (n - 1.0)
    if m == 1:
        return np.full(n, float(np.clip(ys[0], 0.0, 1.0)))

    h = np.diff(xs)
    h = np.where(np.abs(h) < 1e-9, 1e-9, h)
    delta = np.diff(ys) / h

    tang = np.zeros(m, dtype=np.float64)
    tang[0] = delta[0]
    tang[-1] = delta[-1]
    for i in range(1, m - 1):
        if delta[i - 1] * delta[i] <= 0.0:
            tang[i] = 0.0
        else:
            w1 = 2.0 * h[i] + h[i - 1]
            w2 = h[i] + 2.0 * h[i - 1]
            tang[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])

    seg = np.clip(np.searchsorted(xs, t, side="right") - 1, 0, m - 2)
    hs = xs[seg + 1] - xs[seg]
    hs = np.where(np.abs(hs) < 1e-9, 1e-9, hs)
    s = np.clip((t - xs[seg]) / hs, 0.0, 1.0)
    s2 = s * s
    s3 = s2 * s
    h00 = 2.0 * s3 - 3.0 * s2 + 1.0
    h10 = s3 - 2.0 * s2 + s
    h01 = -2.0 * s3 + 3.0 * s2
    h11 = s3 - s2
    y = (h00 * ys[seg] + h10 * hs * tang[seg] +
         h01 * ys[seg + 1] + h11 * hs * tang[seg + 1])
    return np.clip(y, 0.0, 1.0)


def curve_lut(pts):
    """给 UI 画曲线用。pts 为空/少于两点时返回恒等 LUT。"""
    if not pts or len(pts) < 2:
        return np.arange(256, dtype=np.float64) / 255.0
    return _monotone_lut(pts, 256)


def _curves(rgb, p):
    allpts = p.get("points") or {}
    out = rgb
    for ch, ci in (("RGB", [0, 1, 2]), ("R", [0]), ("G", [1]), ("B", [2])):
        pts = allpts.get(ch) or []
        if len(pts) < 2:
            continue
        lut = _monotone_lut(pts, 256)
        idx = np.clip((out[..., ci] * 255.0 + 0.5).astype(np.int32), 0, 255)
        v = lut[idx]
        if out is rgb:
            out = out.copy()
        out[..., ci] = v
    return out


# ---------------------------------------------------------------- 曝光

def _exposure(rgb, p):
    e = float(p["exposure"])
    off = float(p["offset"])
    g = max(float(p["gamma"]), 0.01)
    lin = _srgb_to_linear(rgb)
    lin = lin * (2.0 ** e)          # 曝光在线性光里做，符合真实相机行为
    lin = lin + off                 # 位移主要抬黑场
    v = _linear_to_srgb(lin)
    if abs(g - 1.0) > 1e-4:
        v = np.power(np.clip(v, 0.0, 1.0), 1.0 / g)
    return v


# ---------------------------------------------------------------- 自然饱和度

def _vibrance(rgb, p):
    vib = float(p["vibrance"]) / 100.0
    sat = float(p["saturation"]) / 100.0
    lum = _luma(rgb)[..., None]
    chroma = rgb - lum
    s = _hsv(rgb)[..., 1]           # 当前饱和度，用来决定"该不该再提"
    if vib >= 0.0:
        k = 1.0 + sat + vib * (1.0 - s)
    else:
        k = 1.0 + sat + vib * s
    k = np.clip(k, 0.0, 5.0)[..., None]
    # 用亮度轴缩放色度而不是缩放 HSV 的 S，灰度像素天然不受影响
    return lum + chroma * k


# ---------------------------------------------------------------- 色相 / 饱和度

def _hue_sat(rgb, p):
    hsv = _hsv(rgb)
    hue = float(p["hue"])
    sat = float(p["saturation"]) / 100.0
    light = float(p["lightness"]) / 100.0
    if p.get("colorize"):
        # 着色：色相滑块直接当目标色相，饱和度滑块当着色浓度
        hsv[..., 0] = hue % 360.0
        hsv[..., 1] = np.clip(abs(sat), 0.0, 1.0)
    else:
        hsv[..., 0] = (hsv[..., 0] + hue) % 360.0
        hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + sat), 0.0, 1.0)
    hsv[..., 2] = np.clip(hsv[..., 2] * (1.0 + light), 0.0, 1.0)
    return _rgb(hsv)


# ---------------------------------------------------------------- 黑白

_BW_KEYS = ["reds", "yellows", "greens", "cyans", "blues", "magentas"]
_BW_ANCHORS = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]


def _black_white(rgb, p):
    """六段色相滑块。

    映射方式：先用 Rec.601 算出基础灰度，再按该像素色相附近的滑块值
    对灰度做 gamma 式推移 —— 这样纯黑和纯白不会被推动，只动中间调，
    不会出现"提高红色结果整张发灰"的问题。
    """
    base = np.clip(_luma(rgb), 0.0, 1.0)
    hsv = _hsv(rgb)
    h = hsv[..., 0]
    s = hsv[..., 1]

    num = np.zeros(base.shape, dtype=np.float64)
    den = np.zeros(base.shape, dtype=np.float64)
    for key, anchor in zip(_BW_KEYS, _BW_ANCHORS):
        dh = np.abs(((h - anchor + 180.0) % 360.0) - 180.0)   # 0~180
        w = np.clip(1.0 - dh / 60.0, 0.0, 1.0)                # 三角权重
        num += w * (float(p.get(key, 0.0)) / 100.0)
        den += w
    den = np.where(den < 1e-6, 1.0, den)
    weighted = num / den

    # 接近灰色的像素色相不可靠，用六个滑块的平均值
    mean = sum(float(p.get(k, 0.0)) for k in _BW_KEYS) / 600.0
    amt = np.where(s < 0.05, mean, weighted)
    amt = np.clip(amt, -0.9, 9.0)

    gray = np.power(base, 1.0 / (1.0 + amt))
    return np.repeat(gray[..., None], 3, axis=-1)


# ---------------------------------------------------------------- 色彩平衡

def _color_balance(rgb, p):
    v = np.clip(_luma(rgb), 0.0, 1.0)
    tone = p.get("tone", "中间调")
    if tone == "阴影":
        w = np.power(1.0 - v, 1.5)
    elif tone == "高光":
        w = np.power(v, 1.5)
    else:
        w = np.power(1.0 - np.abs(2.0 * v - 1.0), 1.5)

    out = rgb.copy()
    out[..., 0] += w * (float(p["cyan_red"]) / 100.0) * 0.5
    out[..., 1] -= w * (float(p["magenta_green"]) / 100.0) * 0.5
    out[..., 2] -= w * (float(p["yellow_blue"]) / 100.0) * 0.5
    out = np.clip(out, 0.0, 1.0)

    if p.get("preserve_luma"):
        old = np.maximum(_luma(rgb), 1e-6)
        new = np.maximum(_luma(out), 1e-6)
        f = np.clip(old / new, 0.0, 5.0)[..., None]
        out = np.clip(out * f, 0.0, 1.0)
    return out


# ---------------------------------------------------------------- 亮度 / 对比度

def _bright_contrast(rgb, p):
    """Photoshop 的"旧版"亮度/对比度公式。"""
    b = float(p["brightness"])
    c = float(p["contrast"])
    num = 259.0 * (c + 255.0)
    den = 255.0 * (259.0 - c)
    f = num / den if abs(den) > 1e-6 else 1.0
    return (f * (rgb * 255.0 - 128.0) + 128.0 + b) / 255.0


# ---------------------------------------------------------------- 其它

def _invert(rgb, p):
    return 1.0 - rgb


def _posterize(rgb, p):
    levels = int(max(2, min(255, p["levels"])))
    x = np.clip(rgb, 0.0, 1.0) * (levels - 1)
    return np.round(x) / (levels - 1)


def _threshold(rgb, p):
    lv = float(p["level"]) / 255.0
    g = _luma(rgb)
    v = np.where(g > lv, 1.0, 0.0)
    return np.repeat(v[..., None], 3, axis=-1)


# ---------------------------------------------------------------- 通道混合器

_MIX_ROWS = ["红", "绿", "蓝"]
_MIX_SRC = ["red", "green", "blue"]


def _channel_mixer(rgb, p):
    """通道混合器：输出通道 = 三个源通道的加权和 + 常数。

    默认权重是单位矩阵（红=红、绿=绿、蓝=蓝），所以不做任何事。
    勾了「单色」时按 Photoshop 的习惯只出一个灰度通道 —— 权重取
    「红」那一行（面板上默认停在这一行），结果复制到三个通道。
    """
    mix = p.get("mix") or {}
    rows = []
    consts = []
    for i, name in enumerate(_MIX_ROWS):
        d = mix.get(name) or {}
        ident = [100.0 if j == i else 0.0 for j in range(3)]
        rows.append([float(d.get(k, ident[j])) / 100.0
                     for j, k in enumerate(_MIX_SRC)])
        consts.append(float(d.get("const", 0.0)) / 100.0)

    m = np.array(rows, dtype=np.float32)          # (3,3) 行=输出通道
    c = np.array(consts, dtype=np.float32)        # (3,)
    out = rgb @ m.T + c
    if p.get("mono"):
        out = np.repeat(out[..., 0:1], 3, axis=-1)
    return out


# ---------------------------------------------------------------- 渐变映射

def _gradient_map(rgb, p):
    """按亮度把图像映射到一条三段渐变（暗部 / 中间 / 亮部）上。

    用 Rec.601 亮度取 t，暗部->中间段走 t*2，中间->亮部段走 (t-0.5)*2。
    比直接按通道插值更稳：不会把彩色高光压成纯色块。
    """
    lo = np.asarray(p.get("low") or [0, 0, 0], np.float32) / 255.0
    mid = np.asarray(p.get("mid") or [128, 128, 128], np.float32) / 255.0
    hi = np.asarray(p.get("high") or [255, 255, 255], np.float32) / 255.0
    t = np.clip(_luma(rgb), 0.0, 1.0)[..., None] * 2.0
    return np.where(t < 1.0, lo + (mid - lo) * t,
                    mid + (hi - mid) * (t - 1.0))


# ---------------------------------------------------------------- 照片滤镜

def _photo_filter(rgb, p):
    """照片滤镜：镜头前加一片有色滤镜的模型。

    滤镜色先按自身亮度归一化（scale = color / luma(color)），这样中性灰的
    滤镜不会改变画面亮度；再按浓度在「原样」和「乘上滤镜」之间插值。
    橙片（默认色）会把蓝通道压下去，就是加温效果。
    """
    col = np.asarray(p.get("color") or [255, 165, 0], np.float32) / 255.0
    d = max(0.0, min(1.0, float(p.get("density", 25.0)) / 100.0))
    lc = max(float(0.299 * col[0] + 0.587 * col[1] + 0.114 * col[2]), 1e-4)
    scale = col / lc
    out = np.clip(rgb * (1.0 + d * (scale - 1.0)), 0.0, 1.0)
    if p.get("preserve_luma"):
        old = np.maximum(_luma(rgb), 1e-6)
        new = np.maximum(_luma(out), 1e-6)
        out = out * np.clip(old / new, 0.0, 5.0)[..., None]
    return out


# ---------------------------------------------------------------- 可选颜色

_SC_FAMILIES = ["红", "黄", "绿", "青", "蓝", "洋红", "白", "中性色", "黑"]
_SC_ANCHORS = {"红": 0.0, "黄": 60.0, "绿": 120.0, "青": 180.0,
               "蓝": 240.0, "洋红": 300.0}


def _sc_delta(x, amt):
    """相对量：加补色按现有分量比例削减，减补色按剩余空间补足。"""
    return np.where(amt >= 0.0, -amt * x, -amt * (1.0 - x))


def _selective_color(rgb, p):
    """可选颜色：分 9 个色彩族，各自调 C/M/Y/K。

    每个像素对每个族算一个权重（色相三角权重 x 饱和度 x 明度区间），
    权重之间基本互斥，所以可以简单地把各族的改变量加权叠加起来。
    """
    fams = p.get("families") or {}
    if not any(fams.get(k) for k in _SC_FAMILIES):
        return rgb

    hsv = _hsv(rgb)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    soft = np.clip(s / 0.25, 0.0, 1.0)
    white_w = np.clip((v - 0.75) / 0.25, 0.0, 1.0) * (1.0 - soft)
    black_w = np.clip((0.25 - v) / 0.25, 0.0, 1.0)
    mid = np.clip(1.0 - white_w - black_w, 0.0, 1.0)

    weights = {"白": white_w,
               "黑": black_w,
               "中性色": mid * (1.0 - soft)}
    for name, anchor in _SC_ANCHORS.items():
        dh = np.abs(((h - anchor + 180.0) % 360.0) - 180.0)
        weights[name] = (np.clip(1.0 - dh / 60.0, 0.0, 1.0) *
                         np.clip(s / 0.2, 0.0, 1.0) * mid)

    out = rgb.copy()
    for name in _SC_FAMILIES:
        d = fams.get(name) or {}
        cy = float(d.get("cyan", 0.0)) / 100.0
        mg = float(d.get("magenta", 0.0)) / 100.0
        ye = float(d.get("yellow", 0.0)) / 100.0
        bk = float(d.get("black", 0.0)) / 100.0
        if cy == 0.0 and mg == 0.0 and ye == 0.0 and bk == 0.0:
            continue
        w = weights[name]
        out[..., 0] += w * _sc_delta(out[..., 0], cy)
        out[..., 1] += w * _sc_delta(out[..., 1], mg)
        out[..., 2] += w * _sc_delta(out[..., 2], ye)
        if bk != 0.0:
            for ci in range(3):
                out[..., ci] += w * _sc_delta(out[..., ci], bk)
    return out


# ---------------------------------------------------------------- 注册表

def _spec(key, name, params, func, histogram=False):
    return Spec(key, name, params, func, histogram)


ADJUSTMENTS = {}


def _register(spec):
    ADJUSTMENTS[spec.key] = spec
    return spec


_register(_spec("levels", "色阶", [
    Param("channel", "通道", "choice", choices=["RGB", "R", "G", "B"],
          default="RGB"),
    Param("in_black", "输入黑场", "int", 0, 254, 0),
    Param("in_white", "输入白场", "int", 1, 255, 255),
    Param("gamma", "中间调", "double", 0.10, 9.99, 1.0, decimals=2),
    Param("out_black", "输出黑场", "int", 0, 254, 0),
    Param("out_white", "输出白场", "int", 1, 255, 255),
], _levels, histogram=True))

_register(_spec("curves", "曲线", [
    Param("points", "曲线", "curve"),
], _curves, histogram=True))

_register(_spec("exposure", "曝光度", [
    Param("exposure", "曝光", "double", -5.0, 5.0, 0.0, decimals=2),
    Param("offset", "位移", "double", -0.5, 0.5, 0.0, decimals=3),
    Param("gamma", "灰度系数", "double", 0.01, 10.0, 1.0, decimals=2),
], _exposure, histogram=True))

_register(_spec("vibrance", "自然饱和度", [
    Param("vibrance", "自然饱和度", "int", -100, 100, 0),
    Param("saturation", "饱和度", "int", -100, 100, 0),
], _vibrance))

_register(_spec("hue_sat", "色相/饱和度", [
    Param("hue", "色相", "int", -180, 180, 0),
    Param("saturation", "饱和度", "int", -100, 100, 0),
    Param("lightness", "明度", "int", -100, 100, 0),
    Param("colorize", "着色", "bool", default=False),
], _hue_sat))

_register(_spec("black_white", "黑白", [
    Param("reds", "红色", "int", -200, 200, 0),
    Param("yellows", "黄色", "int", -200, 200, 0),
    Param("greens", "绿色", "int", -200, 200, 0),
    Param("cyans", "青色", "int", -200, 200, 0),
    Param("blues", "蓝色", "int", -200, 200, 0),
    Param("magentas", "洋红", "int", -200, 200, 0),
], _black_white))

_register(_spec("color_balance", "色彩平衡", [
    Param("tone", "色调范围", "choice",
          choices=["阴影", "中间调", "高光"], default="中间调"),
    Param("cyan_red", "青 → 红", "int", -100, 100, 0),
    Param("magenta_green", "洋红 → 绿", "int", -100, 100, 0),
    Param("yellow_blue", "黄 → 蓝", "int", -100, 100, 0),
    Param("preserve_luma", "保留明度", "bool", default=False),
], _color_balance))

_register(_spec("bright_contrast", "亮度/对比度", [
    Param("brightness", "亮度", "int", -150, 150, 0),
    Param("contrast", "对比度", "int", -50, 100, 0),
], _bright_contrast, histogram=True))

_register(_spec("invert", "反相", [], _invert))
_register(_spec("posterize", "色调分离", [
    Param("levels", "色阶数", "int", 2, 255, 4),
], _posterize))
_register(_spec("threshold", "阈值", [
    Param("level", "阈值色阶", "int", 1, 255, 128),
], _threshold, histogram=True))

_MIX_SUB = [Param("red", "红", "int", -200, 200, 0),
            Param("green", "绿", "int", -200, 200, 0),
            Param("blue", "蓝", "int", -200, 200, 0),
            Param("const", "常数", "int", -200, 200, 0)]

# 默认是单位矩阵（红=红、绿=绿、蓝=蓝），所以新建的通道混合器不改画面
_MIX_DEFAULT = keyed_default(_MIX_ROWS, _MIX_SUB)
for _i, _name in enumerate(_MIX_ROWS):
    _MIX_DEFAULT[_name][_MIX_SRC[_i]] = 100

_register(_spec("channel_mixer", "通道混合器", [
    Param("mono", "单色", "bool", default=False),
    Param("mix", "输出通道", "keyed",
          keys=_MIX_ROWS, sub=_MIX_SUB, default=_MIX_DEFAULT),
], _channel_mixer, histogram=True))

_register(_spec("gradient_map", "渐变映射", [
    Param("low", "暗部", "color", default=[0, 0, 0]),
    Param("mid", "中间", "color", default=[128, 128, 128]),
    Param("high", "亮部", "color", default=[255, 255, 255]),
], _gradient_map, histogram=True))

_register(_spec("photo_filter", "照片滤镜", [
    Param("color", "滤镜颜色", "color", default=[255, 165, 0]),
    Param("density", "浓度", "int", 0, 100, 25),
    Param("preserve_luma", "保留明度", "bool", default=True),
], _photo_filter))

_SC_SUB = [Param("cyan", "青", "int", -100, 100, 0),
           Param("magenta", "洋红", "int", -100, 100, 0),
           Param("yellow", "黄", "int", -100, 100, 0),
           Param("black", "黑", "int", -100, 100, 0)]

_register(_spec("selective_color", "可选颜色", [
    Param("families", "色彩族", "keyed",
          keys=_SC_FAMILIES, sub=_SC_SUB,
          default=keyed_default(_SC_FAMILIES, _SC_SUB)),
], _selective_color))

# 面板里的显示顺序
ADJUST_ORDER = ["levels", "curves", "exposure", "vibrance", "hue_sat",
                "black_white", "color_balance", "bright_contrast",
                "channel_mixer", "gradient_map", "photo_filter",
                "selective_color",
                "invert", "posterize", "threshold"]


def default_params(key):
    spec = ADJUSTMENTS.get(key)
    if spec is None:
        return {}
    p = spec.defaults()
    if key == "curves":
        p["points"] = {"RGB": [[0.0, 0.0], [1.0, 1.0]],
                       "R": [], "G": [], "B": []}
    return p


def apply_adjustment(key, rgb, params):
    """应用调整。rgb 为 (h,w,3) float32 0~1，返回同形状的新数组。"""
    spec = ADJUSTMENTS.get(key)
    if spec is None:
        return rgb
    p = spec.defaults()
    if params:
        p.update(params)
    out = spec.func(rgb, p)
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)
