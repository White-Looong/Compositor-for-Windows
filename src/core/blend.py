# -*- coding: utf-8 -*-
"""混合模式与图层合成。

色彩公式遵循 W3C Compositing and Blending Level 1，与 Photoshop 基本一致。
所有运算在 float32、0~1 区间的**直线（non-premultiplied）**色彩空间中进行。

合成公式：
    Co = as*(1-ab)*Cs + as*ab*B(Cb, Cs) + (1-as)*ab*Cb
    ao = as + ab*(1-as)
其中 Cb/ab 是下方（背景），Cs/as 是上方（当前图层）。
"""

from __future__ import annotations

import numpy as np

PASS_THROUGH = "Pass Through"

# 按 Photoshop 面板中的顺序排列
BLEND_MODES = [
    "Normal", "Dissolve",
    "Darken", "Multiply", "Color Burn", "Linear Burn", "Darker Color",
    "Lighten", "Screen", "Color Dodge", "Linear Dodge (Add)", "Lighter Color",
    "Overlay", "Soft Light", "Hard Light", "Vivid Light", "Linear Light",
    "Pin Light", "Hard Mix",
    "Difference", "Exclusion", "Subtract", "Divide",
    "Hue", "Saturation", "Color", "Luminosity",
]

# 这些模式不能逐通道独立计算
NON_SEPARABLE = {"Hue", "Saturation", "Color", "Luminosity",
                 "Darker Color", "Lighter Color"}


def _div(a, b):
    """带保护的除法，分母为 0 时返回 0 而不是 inf/nan。"""
    with np.errstate(divide="ignore", invalid="ignore"):
        return a / np.where(np.abs(b) < 1e-12, 1.0, b)


def _lum(c):
    return 0.3 * c[..., 0] + 0.59 * c[..., 1] + 0.11 * c[..., 2]


def _sat(c):
    return c.max(axis=-1) - c.min(axis=-1)


def _clip_color(c):
    """把可能越界的颜色拉回 [0,1] 并保持色相与亮度（W3C ClipColor）。"""
    L = _lum(c)[..., None]
    n = c.min(axis=-1)[..., None]
    x = c.max(axis=-1)[..., None]

    out = c
    dn = L - n
    t = L + _div((c - L) * L, dn)
    out = np.where(n < 0, t, out)

    dx = x - L
    t2 = L + _div((c - L) * (1.0 - L), dx)
    out = np.where(x > 1.0, t2, out)
    return np.clip(out, 0.0, 1.0)


def _set_lum(c, l):
    """保持色相/饱和度，把亮度改成 l。"""
    d = l - _lum(c)
    return _clip_color(c + d[..., None])


def _set_sat(c, s):
    """保持色相/亮度，把饱和度改成 s。"""
    mn = c.min(axis=-1)[..., None]
    mx = c.max(axis=-1)[..., None]
    d = mx - mn
    t = (c - mn) * s[..., None] / np.where(np.abs(d) < 1e-12, 1.0, d)
    return np.where(np.abs(d) < 1e-12, 0.0, t)


def blend_colors(cb, cs, mode):
    """计算混合色 B(Cb, Cs)。cb/cs 为 (...,3) float32，范围 0~1。"""
    if mode == "Normal":
        return cs
    if mode == "Multiply":
        return cb * cs
    if mode == "Screen":
        return cb + cs - cb * cs
    if mode == "Darken":
        return np.minimum(cb, cs)
    if mode == "Lighten":
        return np.maximum(cb, cs)
    if mode == "Difference":
        return np.abs(cb - cs)
    if mode == "Exclusion":
        return cb + cs - 2.0 * cb * cs
    if mode == "Linear Burn":
        return np.clip(cb + cs - 1.0, 0.0, 1.0)
    if mode == "Linear Dodge (Add)":
        return np.clip(cb + cs, 0.0, 1.0)
    if mode == "Subtract":
        return np.clip(cb - cs, 0.0, 1.0)
    if mode == "Divide":
        return np.clip(_div(cb, cs), 0.0, 1.0)
    if mode == "Color Burn":
        b = _div(1.0 - cb, cs)
        return np.where(cs <= 1e-6, 0.0, 1.0 - np.clip(b, 0.0, 1.0))
    if mode == "Color Dodge":
        b = _div(cb, 1.0 - cs)
        return np.where(cs >= 1.0 - 1e-6, 1.0, np.clip(b, 0.0, 1.0))
    if mode == "Overlay":
        return np.where(cb <= 0.5, 2.0 * cb * cs,
                        1.0 - 2.0 * (1.0 - cb) * (1.0 - cs))
    if mode == "Hard Light":
        return np.where(cs <= 0.5, 2.0 * cs * cb,
                        1.0 - 2.0 * (1.0 - cs) * (1.0 - cb))
    if mode == "Soft Light":
        d = np.where(cb <= 0.25,
                     ((16.0 * cb - 12.0) * cb + 4.0) * cb,
                     np.sqrt(np.clip(cb, 0.0, 1.0)))
        return np.where(cs <= 0.5,
                        cb - (1.0 - 2.0 * cs) * cb * (1.0 - cb),
                        cb + (2.0 * cs - 1.0) * (d - cb))
    if mode == "Vivid Light":
        return np.where(cs <= 0.5,
                        np.clip(1.0 - _div(1.0 - cb, 2.0 * cs), 0.0, 1.0),
                        np.clip(_div(cb, 2.0 * (1.0 - cs)), 0.0, 1.0))
    if mode == "Linear Light":
        return np.where(cs <= 0.5,
                        np.clip(cb + 2.0 * cs - 1.0, 0.0, 1.0),
                        np.clip(cb + 2.0 * (cs - 0.5), 0.0, 1.0))
    if mode == "Pin Light":
        return np.where(cs <= 0.5,
                        np.minimum(cb, 2.0 * cs),
                        np.maximum(cb, 2.0 * (cs - 0.5)))
    if mode == "Hard Mix":
        return np.where(cs + cb >= 1.0, 1.0, 0.0)
    if mode == "Darker Color":
        m = (_lum(cs) < _lum(cb))[..., None]
        return np.where(m, cs, cb)
    if mode == "Lighter Color":
        m = (_lum(cs) > _lum(cb))[..., None]
        return np.where(m, cs, cb)
    if mode == "Hue":
        return _set_lum(_set_sat(cs, _sat(cb)), _lum(cb))
    if mode == "Saturation":
        return _set_lum(_set_sat(cb, _sat(cs)), _lum(cb))
    if mode == "Color":
        return _set_lum(cs, _lum(cb))
    if mode == "Luminosity":
        return _set_lum(cb, _lum(cs))
    return cs


_noise_cache = {}


def dissolve_noise(h, w):
    """溶解用的静态噪声，按尺寸缓存，保证每次渲染结果稳定。"""
    n = _noise_cache.get((h, w))
    if n is None:
        rng = np.random.default_rng(20240917)
        n = rng.random((h, w, 1)).astype(np.float32)
        _noise_cache[(h, w)] = n
    return n


def composite(dst_c, dst_a, src_c, src_a, mode):
    """把 src 合成到 dst 上（原地修改 dst_c / dst_a）。

    dst_c: (h,w,3) float32 直线色
    dst_a: (h,w,1) float32 0~1
    src_c/src_a: 与 dst 同形状，或更小（此时按左上角对齐）——调用方保证形状一致。
    """
    as_ = src_a
    if mode == "Dissolve":
        h, w = as_.shape[:2]
        as_ = (dissolve_noise(h, w) < as_).astype(np.float32)
        mode = "Normal"

    ab = dst_a
    if mode == "Normal":
        b = src_c
    else:
        b = blend_colors(dst_c, src_c, mode)

    co = as_ * (1.0 - ab) * src_c + as_ * ab * b + (1.0 - as_) * ab * dst_c
    ao = as_ + ab * (1.0 - as_)

    safe = np.where(np.abs(ao) < 1e-6, 1.0, ao)
    dst_c[...] = np.clip(_div(co, safe), 0.0, 1.0)
    dst_a[...] = np.clip(ao, 0.0, 1.0)
