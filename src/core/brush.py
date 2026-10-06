# -*- coding: utf-8 -*-
"""笔刷图章与像素合成原语。

这里只做"把一个已知强度阵列 a 盖到目标数组上"的活儿；
强度怎么算（流量、不透明度上限、选区限制）由 paint.py 负责。
"""

from __future__ import annotations

import numpy as np

_stamp_cache = {}


def make_stamp(radius, hardness):
    """生成一个 (n,n) 的笔刷图章，值域 0~1。

    radius:   半径（像素），小于 0.5 时按 0.5 处理
    hardness: 0=极软，1=硬边
    """
    r = max(0.5, float(radius))
    h = float(np.clip(hardness, 0.0, 1.0))
    n = int(max(1, round(r * 2)))
    key = (n, round(h, 3))
    s = _stamp_cache.get(key)
    if s is not None:
        return s

    ax = np.arange(n, dtype=np.float32) - (n - 1) / 2.0
    yy, xx = np.meshgrid(ax, ax, indexing="ij")
    d = np.sqrt(xx * xx + yy * yy) / r          # 0 中心，1 边缘

    if h >= 0.995:
        a = (d <= 1.0).astype(np.float32)
    else:
        a = np.clip((1.0 - d) / (1.0 - h + 1e-6), 0.0, 1.0)
        a = a * a * (3.0 - 2.0 * a)             # smoothstep，边缘过渡更自然
    a[d > 1.0] = 0.0
    _stamp_cache[key] = a
    return a


def _overlap(img_shape, cx, cy, n):
    """计算图章落在图像上的重叠区域。

    返回 (dy0, dy1, dx0, dx1, sy0, sy1, sx0, sx1)；无重叠返回 None。
    """
    half = n // 2
    x0 = int(round(cx)) - half
    y0 = int(round(cy)) - half
    H, W = img_shape[:2]
    sx0 = max(0, -x0)
    sy0 = max(0, -y0)
    dx0 = max(0, x0)
    dy0 = max(0, y0)
    dx1 = min(W, x0 + n)
    dy1 = min(H, y0 + n)
    if dx1 <= dx0 or dy1 <= dy0:
        return None
    return (dy0, dy1, dx0, dx1,
            sy0, sy0 + (dy1 - dy0), sx0, sx0 + (dx1 - dx0))


def composite_rgba(img, cx, cy, a, rgb):
    """把强度阵列 a 以 source-over 方式画到 RGBA 图像上（原地修改）。"""
    ov = _overlap(img.shape, cx, cy, a.shape[0])
    if ov is None:
        return
    dy0, dy1, dx0, dx1, sy0, sy1, sx0, sx1 = ov
    sub = a[sy0:sy1, sx0:sx1]
    if not sub.any():
        return
    aa = sub[..., None]

    dst = img[dy0:dy1, dx0:dx1].astype(np.float32)
    da = dst[..., 3:4] / 255.0
    dc = dst[..., :3]

    oa = aa + da * (1.0 - aa)
    oc = (np.asarray(rgb, np.float32) * aa + dc * da * (1.0 - aa)) / np.maximum(oa, 1e-6)
    out = np.concatenate([oc, oa * 255.0], axis=2)
    img[dy0:dy1, dx0:dx1] = np.clip(out, 0, 255).astype(np.uint8)


def erase_rgba(img, cx, cy, a):
    """橡皮擦：只削 alpha，颜色保留（与 Photoshop 普通图层行为一致）。"""
    ov = _overlap(img.shape, cx, cy, a.shape[0])
    if ov is None:
        return
    dy0, dy1, dx0, dx1, sy0, sy1, sx0, sx1 = ov
    sub = a[sy0:sy1, sx0:sx1]
    if not sub.any():
        return
    dst = img[dy0:dy1, dx0:dx1]
    alpha = dst[..., 3].astype(np.float32) * (1.0 - sub)
    dst[..., 3] = np.clip(alpha, 0, 255).astype(np.uint8)


def composite_mask(mask, cx, cy, a, value):
    """在蒙版上涂。value: 0~255 的目标灰度。"""
    ov = _overlap(mask.shape, cx, cy, a.shape[0])
    if ov is None:
        return
    dy0, dy1, dx0, dx1, sy0, sy1, sx0, sx1 = ov
    sub = a[sy0:sy1, sx0:sx1]
    if not sub.any():
        return
    dst = mask[dy0:dy1, dx0:dx1].astype(np.float32)
    out = dst * (1.0 - sub) + float(value) * sub
    mask[dy0:dy1, dx0:dx1] = np.clip(out, 0, 255).astype(np.uint8)


def fill_rgba(img, sel, rgb, alpha=1.0):
    """用颜色填充整张图（受 sel 限制）。sel 是与 img 同尺寸的 float32 0~1，可为 None。"""
    h, w = img.shape[:2]
    a = np.ones((h, w), np.float32) * float(np.clip(alpha, 0.0, 1.0))
    if sel is not None:
        a = a * sel
    aa = a[..., None]
    dst = img.astype(np.float32)
    da = dst[..., 3:4] / 255.0
    dc = dst[..., :3]
    oa = aa + da * (1.0 - aa)
    oc = (np.asarray(rgb, np.float32) * aa + dc * da * (1.0 - aa)) / np.maximum(oa, 1e-6)
    out = np.concatenate([oc, oa * 255.0], axis=2)
    img[...] = np.clip(out, 0, 255).astype(np.uint8)


def clear_rgba(img, sel):
    """清除像素（变透明），受 sel 限制。"""
    if sel is None:
        img[...] = 0
        return
    a = (1.0 - np.clip(sel, 0.0, 1.0))[..., None]
    img[...] = (img.astype(np.float32) * a).astype(np.uint8)
