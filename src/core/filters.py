# -*- coding: utf-8 -*-
"""滤镜：对位图图层做**破坏性**像素处理（可撤销，受选区限制）。

每个滤镜是
    func(rgb, alpha, params) -> (rgb, alpha)
rgb/alpha 都是 float32、0~1。

两条重要约定：

1. **先预乘 alpha 再卷积**。直接在直线色上做模糊，透明区域周围的黑色会被
   卷进来，边缘出现脏黑边；预乘之后透明像素贡献为 0，除法还原，边缘干净。

2. **在图层源坐标系上执行**。滤镜不受图层缩放/旋转影响，半径的单位是
   「图层原始像素」——这和 Photoshop 一致。

滤镜本身是破坏性的（改写像素），但走撤销栈 + 写时复制，所以随时能撤回。
想要非破坏性请用调整层。
"""

from __future__ import annotations

import math

import cv2
import numpy as np


class FParam:
    def __init__(self, key, label, kind="int", lo=0, hi=100, default=0,
                 step=1, choices=None, decimals=2, spatial=False):
        self.key = key
        self.label = label
        self.kind = kind            # int / double / bool / choice
        self.lo = lo
        self.hi = hi
        self.default = default
        self.step = step
        self.choices = list(choices or [])
        self.decimals = decimals
        self.scale = 1000 if kind == "double" else 1
        # spatial=True：单位是**源图像素**，预览降采样时要按同一比例缩小
        # （半径 / 距离 / 高度这类），否则缩略图上跑出来的效果和实际差很远
        self.spatial = spatial


class FSpec:
    def __init__(self, key, name, params, func, preview_scale=True):
        self.key = key
        self.name = name
        self.params = list(params)
        self.func = func
        # 预览能不能"先降采样再算"。杂色是按像素生成的、查找边缘只是 3x3
        # Sobel，降采样反而会让预览失真，这两个关掉。
        self.preview_scale = preview_scale

    def defaults(self):
        return dict((p.key, p.default) for p in self.params)


# ---------------------------------------------------------------- 工具

def _unpremul(pm, a):
    return pm / np.maximum(a, 1e-6)


def _keep_ch(x):
    """OpenCV 对 (h,w,1) 的输入经常把通道维丢掉，这里补回来。"""
    return x[..., None] if x.ndim == 2 else x


def _blur_premul(rgb, a, blur_fn):
    """预乘 -> 模糊 -> 还原。blur_fn 接受并返回一个 float32 数组。"""
    pm = rgb * a
    pm2 = _keep_ch(blur_fn(pm))
    a2 = _keep_ch(blur_fn(a))
    return _unpremul(pm2, np.maximum(a2, 0.0)), np.clip(a2, 0.0, 1.0)


def _gauss(x, sigma):
    if sigma <= 0.02:
        return x
    return cv2.GaussianBlur(x, (0, 0), sigmaX=sigma, sigmaY=sigma,
                            borderType=cv2.BORDER_REPLICATE)


def _luma(rgb):
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] +
            0.114 * rgb[..., 2]).astype(np.float32)


def _as_gray(v):
    return np.repeat(np.clip(v, 0.0, 1.0)[..., None], 3, axis=-1)


# ---------------------------------------------------------------- 模糊

def f_gaussian(rgb, a, p):
    r = float(p["radius"])
    return _blur_premul(rgb, a, lambda x: _gauss(x, r))


def _motion_kernel(dist, angle_deg):
    dist = int(max(1, round(dist)))
    n = 2 * dist + 1
    c = dist
    ang = math.radians(angle_deg)
    dx, dy = math.cos(ang), math.sin(ang)
    k = np.zeros((n, n), np.float32)
    for i in range(-dist, dist + 1):
        x = c + dx * i
        y = c + dy * i
        x0, y0 = int(math.floor(x)), int(math.floor(y))
        fx, fy = x - x0, y - y0
        for xx, yy, w in ((x0, y0, (1 - fx) * (1 - fy)),
                          (x0 + 1, y0, fx * (1 - fy)),
                          (x0, y0 + 1, (1 - fx) * fy),
                          (x0 + 1, y0 + 1, fx * fy)):
            if 0 <= xx < n and 0 <= yy < n and w > 0:
                k[yy, xx] += w
    s = k.sum()
    return k / s if s > 0 else k


def f_motion(rgb, a, p):
    k = _motion_kernel(p["distance"], p["angle"])
    return _blur_premul(rgb, a,
                        lambda x: cv2.filter2D(x, -1, k,
                                               borderType=cv2.BORDER_REPLICATE))


def f_radial(rgb, a, p):
    """径向（缩放式）模糊：把若干层逐渐缩小的副本叠加求平均。"""
    amount = float(p["amount"]) / 100.0
    steps = int(max(1, min(24, round(amount * 20.0))))
    bh, bw = rgb.shape[:2]
    cx, cy = bw / 2.0, bh / 2.0
    pm = rgb * a
    acc_c = pm.copy()
    acc_a = a.copy()
    cnt = 1.0
    for i in range(1, steps + 1):
        s = 1.0 - (float(i) / steps) * amount * 0.4
        M = np.array([[s, 0.0, cx * (1.0 - s)],
                      [0.0, s, cy * (1.0 - s)]], np.float64)
        acc_c += cv2.warpAffine(pm, M, (bw, bh), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(0, 0, 0))
        acc_a += _keep_ch(cv2.warpAffine(a, M, (bw, bh),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=(0,)))
        cnt += 1.0
    pm2 = acc_c / cnt
    a2 = np.clip(acc_a / cnt, 0.0, 1.0)
    return _unpremul(pm2, a2), a2


# ---------------------------------------------------------------- 中间值 / 蒙尘与划痕

# 半径不超过这个值时走 float32 的 medianBlur（精确、无量化）
_MEDIAN_FLOAT_R = 2
# 半径上限 100 -> 核 201。实测 8-bit 路径在 k <= 201 时对任意图像尺寸都安全
# （再大 OpenCV 的 AVX2 快速路径会断言失败，所以别把上限提到 100 以上）
_MEDIAN_MAX_R = 100


def _median_rgb(rgb, r):
    """任意半径的 3 通道中间值（半径 1~_MEDIAN_MAX_R）。"""
    r = int(max(1, min(_MEDIAN_MAX_R, r)))
    k = 2 * r + 1
    if r <= _MEDIAN_FLOAT_R:
        return np.stack([cv2.medianBlur(rgb[..., i], k) for i in range(3)],
                        axis=-1)
    u = np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return cv2.medianBlur(u, k).astype(np.float32) / 255.0


def f_median(rgb, a, p):
    """中间值。半径 1~100。

    OpenCV 的 `medianBlur` 对 **float32** 只支持 3x3 / 5x5（半径 1~2），
    但对 **uint8 支持任意奇数核**（内部是 Huang 的滑动直方图，核越大几乎不变慢：
    12 MP 上半径 100 也只要 ~200 ms）。所以半径 > 2 时把三个通道量化到 8-bit 再算。
    管线本来就在 8-bit 上收尾，量化误差最多 1/255，肉眼不可见。
    小半径仍走 float32 路径 —— 那两个尺寸 OpenCV 算得又准又快，没必要量化。
    """
    r = int(max(1, p["radius"]))
    return np.clip(_median_rgb(rgb, r), 0.0, 1.0), a


def f_dust(rgb, a, p):
    """蒙尘与划痕：只有偏离中间值超过阈值的像素才被替换，其余原样保留。

    阈值 = 0 时退化成纯粹的中间值滤镜。判定**按通道**做（和 Photoshop 一致）。
    适合去扫描件的灰尘、划痕、孤立噪点 —— 半径给大也不会把细节糊掉，
    因为没超阈值的像素根本不动。
    """
    r = int(max(1, p["radius"]))
    thr = float(p["threshold"]) / 255.0
    med = _median_rgb(rgb, r)
    out = np.where(np.abs(rgb - med) > thr, med, rgb)
    return np.clip(out, 0.0, 1.0), a


# ---------------------------------------------------------------- 锐化 / 高反差

def f_sharpen(rgb, a, p):
    """USM 锐化：原图 + amount * (原图 - 模糊图)，只在超过阈值的区域生效。"""
    amt = float(p["amount"]) / 100.0
    rad = max(float(p["radius"]), 0.05)
    thr = float(p["threshold"]) / 255.0
    pm = rgb * a
    low = _gauss(pm, rad)
    high = pm - low
    m = (np.abs(high) >= thr).astype(np.float32)
    pm2 = pm + amt * high * m
    return _unpremul(pm2, np.maximum(a, 1e-6)), a


def f_highpass(rgb, a, p):
    rad = max(float(p["radius"]), 0.05)
    g = _luma(rgb)
    hp = g - _gauss(g, rad) + 0.5
    return _as_gray(hp), a


# ---------------------------------------------------------------- 形态学

def f_minmax(rgb, a, p):
    r = int(max(1, p["radius"]))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    op = cv2.dilate if p["mode"] == "最大值" else cv2.erode
    out = np.stack([op(rgb[..., i], k) for i in range(3)], axis=-1)
    a2 = np.clip(_keep_ch(op(a, k)), 0.0, 1.0)
    return np.clip(out, 0.0, 1.0), a2


# ---------------------------------------------------------------- 杂色 / 风格化

def f_noise(rgb, a, p):
    amt = float(p["amount"]) / 100.0
    if amt <= 0.0:
        return rgb, a
    rng = np.random.default_rng()
    if p["distribution"] == "高斯":
        n = rng.normal(0.0, amt * 0.5, rgb.shape).astype(np.float32)
    else:
        n = rng.uniform(-amt, amt, rgb.shape).astype(np.float32)
    if p["monochrome"]:
        n = np.repeat(n[..., 0:1], 3, axis=-1)
    return np.clip(rgb + n * a, 0.0, 1.0), a


def _emboss_kernel(height, angle_deg):
    L = max(1, int(height))
    n = 2 * L + 1
    c = L
    ang = math.radians(angle_deg)
    dx, dy = math.cos(ang), -math.sin(ang)
    k = np.zeros((n, n), np.float32)
    for i in range(-L, L + 1):
        x = int(round(c + dx * i))
        y = int(round(c + dy * i))
        if 0 <= x < n and 0 <= y < n:
            k[y, x] += float(i) / L
    s = np.abs(k).sum()
    return k / s if s > 0 else k


def f_emboss(rgb, a, p):
    k = _emboss_kernel(p["height"], p["angle"])
    g = _luma(rgb)
    d = cv2.filter2D(g, -1, k, borderType=cv2.BORDER_REPLICATE)
    emb = _as_gray(0.5 + d * 2.0)
    t = float(np.clip(float(p["amount"]) / 100.0, 0.0, 1.0))
    return np.clip(rgb * (1.0 - t) + emb * t, 0.0, 1.0), a


def f_edges(rgb, a, p):
    g = _luma(rgb)
    gx = cv2.Sobel(g, -1, 1, 0, ksize=3)
    gy = cv2.Sobel(g, -1, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy) * float(p["strength"])
    return _as_gray(mag), a


# ---------------------------------------------------------------- 注册表

FILTERS = {}
FILTER_ORDER = []


def _register(spec):
    FILTERS[spec.key] = spec
    FILTER_ORDER.append(spec.key)
    return spec


_register(FSpec("gaussian", "高斯模糊…", [
    FParam("radius", "半径", "double", 0.1, 500.0, 5.0, decimals=1,
           spatial=True),
], f_gaussian))

_register(FSpec("motion", "动感模糊…", [
    FParam("distance", "距离", "int", 1, 300, 30, spatial=True),
    FParam("angle", "角度", "int", -180, 180, 0),
], f_motion))

_register(FSpec("radial", "径向模糊…", [
    FParam("amount", "数量", "int", 1, 100, 20),
], f_radial))

_register(FSpec("sharpen", "USM 锐化…", [
    FParam("amount", "数量", "int", 0, 500, 80),
    FParam("radius", "半径", "double", 0.1, 50.0, 1.0, decimals=1,
           spatial=True),
    FParam("threshold", "阈值", "int", 0, 255, 2),
], f_sharpen))

_register(FSpec("median", "中间值…", [
    FParam("radius", "半径", "int", 1, _MEDIAN_MAX_R, 1, spatial=True),
], f_median))

_register(FSpec("dust", "蒙尘与划痕…", [
    FParam("radius", "半径", "int", 1, _MEDIAN_MAX_R, 3, spatial=True),
    FParam("threshold", "阈值", "int", 0, 255, 20),
], f_dust))

_register(FSpec("noise", "添加杂色…", [
    FParam("amount", "数量", "int", 0, 100, 12),
    FParam("distribution", "分布", "choice", choices=["高斯", "均匀"],
           default="高斯"),
    FParam("monochrome", "单色", "bool", default=False),
], f_noise, preview_scale=False))

_register(FSpec("emboss", "浮雕…", [
    FParam("angle", "角度", "int", -180, 180, 45),
    FParam("height", "高度", "int", 1, 10, 3, spatial=True),
    FParam("amount", "数量", "int", 0, 200, 100),
], f_emboss))

_register(FSpec("edges", "查找边缘…", [
    FParam("strength", "强度", "double", 0.1, 5.0, 2.0, decimals=1),
], f_edges, preview_scale=False))

_register(FSpec("highpass", "高反差保留…", [
    FParam("radius", "半径", "double", 0.1, 200.0, 10.0, decimals=1,
           spatial=True),
], f_highpass))

_register(FSpec("minmax", "最小值 / 最大值…", [
    FParam("mode", "模式", "choice", choices=["最大值", "最小值"],
           default="最大值"),
    FParam("radius", "半径", "int", 1, 20, 2, spatial=True),
], f_minmax))


def default_filter_params(key):
    spec = FILTERS.get(key)
    return spec.defaults() if spec else {}


# ------------------------------------------------- 预览降采样（拖参数时用）

# 超过这个最长边，预览就先在缩略图上算（滤镜对话框每改一次参数都要跑一遍整层）
PREVIEW_MAX_EDGE = 1600
PREVIEW_MIN_SCALE = 0.25
# 尺度参数缩放后小于这个像素数就不值得降采样了 —— 半径本来只有 1~2 px 的
# 滤镜既不慢，缩放后又会明显失真
PREVIEW_MIN_RADIUS = 1.5


def spatial_param_keys(key):
    """这个滤镜里哪些参数是"以源像素为单位"的。"""
    spec = FILTERS.get(key)
    if spec is None:
        return []
    return [p.key for p in spec.params if p.spatial]


def scale_filter_params(key, params, scale):
    """按预览缩放比例缩放尺度参数（半径 / 距离 / 高度）。

    其余参数（数量、角度、阈值…）是无量纲的，原样保留。
    """
    out = dict(params or {})
    if scale >= 0.999:
        return out
    spec = FILTERS.get(key)
    if spec is None:
        return out
    for p in spec.params:
        if not p.spatial or p.key not in out:
            continue
        v = float(out[p.key]) * scale
        if p.kind == "int":
            out[p.key] = int(max(1, round(v)))
        else:
            out[p.key] = max(0.05, v)
    return out


def preview_filter_scale(img, key):
    """预览用的降采样比例 —— 只由**位图尺寸**决定，与当前参数无关。

    返回 (scale, 缩略图)；scale == 1.0 时返回的就是原图本身。
    拿到之后还要用 `preview_can_downscale()` 看当前参数允许不允许（半径太小
    的滤镜缩放后会失真），允许才在这张缩略图上算。
    缩略图只做一次、缓在对话框的上下文里，拖参数时不会反复降采样。
    """
    spec = FILTERS.get(key)
    if spec is None or not spec.preview_scale or img is None:
        return 1.0, img
    h, w = img.shape[:2]
    edge = max(h, w)
    if edge <= PREVIEW_MAX_EDGE:
        return 1.0, img
    scale = max(PREVIEW_MIN_SCALE, PREVIEW_MAX_EDGE / float(edge))
    small = cv2.resize(img, (max(1, int(round(w * scale))),
                             max(1, int(round(h * scale)))),
                       interpolation=cv2.INTER_AREA)
    return small.shape[1] / float(w), small


def preview_can_downscale(key, params, scale):
    """当前参数能不能在缩略图上预览。

    尺度参数（半径 / 距离 / 高度）缩放后小于 PREVIEW_MIN_RADIUS 就不行 ——
    半径本来就只有 1~2 px 的滤镜既不慢，缩放后又会明显失真，不如老实全分辨率算。
    """
    if scale >= 0.999:
        return False
    keys = spatial_param_keys(key)
    if not keys:
        return True                 # 没有尺度参数（径向模糊等），缩放不影响语义
    d = FILTERS[key].defaults()
    d.update(params or {})
    mx = max(abs(float(d.get(k, 0.0))) for k in keys)
    return mx * scale >= PREVIEW_MIN_RADIUS


def apply_filter_array(img, key, params, sel=None, strength=1.0):
    """对 (h,w,4) uint8 RGBA 应用滤镜，返回**新的** uint8 数组。

    sel:      (h,w) float32 0~1，图层源坐标系的选区；None 表示全图层
    strength: 整体强度 0~1，相当于 Photoshop 的「渐隐」
    """
    spec = FILTERS.get(key)
    if spec is None:
        return img.copy()
    p = spec.defaults()
    if params:
        p.update(params)

    rgb = img[..., :3].astype(np.float32) / 255.0
    a = img[..., 3:4].astype(np.float32) / 255.0
    nrgb, na = spec.func(rgb, a, p)
    nrgb = np.clip(nrgb, 0.0, 1.0)
    na = np.clip(na, 0.0, 1.0)

    # 选区与强度是同一个「混合权重」，叠在一起算一次就行
    w = float(strength)
    if sel is not None:
        w = w * sel              # 变成 (h,w) 数组
    if not (isinstance(w, float) and w >= 0.999):
        ww = w[..., None] if np.ndim(w) == 2 else w
        nrgb = rgb * (1.0 - ww) + nrgb * ww
        na = a * (1.0 - ww) + na * ww
    if na.ndim == 3 and na.shape[2] == 1:
        na = na[..., 0]

    out = np.empty_like(img)
    out[..., :3] = np.clip(nrgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
    out[..., 3] = np.clip(na * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return out
