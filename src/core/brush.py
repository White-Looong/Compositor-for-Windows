# -*- coding: utf-8 -*-
"""笔刷图章与像素合成原语。

这里只做"把一个已知强度阵列 a 盖到目标数组上"的活儿；
强度怎么算（流量、不透明度上限、选区限制）由 paint.py 负责。

笔尖的形状由 `make_stamp()` 生成：圆 / 方 / 菱形，可以再叠「圆度」拉成椭圆、
按「角度」旋转；`texture_patch()` 按**源坐标**取一张图案给画笔当纹理用。
"""

from __future__ import annotations

import math

import numpy as np

# 笔尖形状
SHAPES = ["圆形", "方形", "菱形"]
# 纹理（无 + 复用图层样式那套内置图案，见 core/effects.py）
TEXTURES = ["无", "点阵", "网格", "斜纹", "棋盘", "噪声"]

# 笔刷设置默认值（工具选项条 / 笔刷对话框读写的就是这份）
BRUSH_DEFAULTS = {
    "shape": "圆形",
    "roundness": 1.0,          # 5%~100%，越小笔尖越扁（椭圆）
    "angle": 0.0,              # -180~180 度，笔尖旋转
    "spacing": 0.25,           # 1%~500%，**笔尖直径**的百分之几（PS 语义）
    "scatter": 0.0,            # 0%~1000%，以半径为单位的随机偏移
    "count": 1,                # 1~10，每步落几个点
    "size_jitter": 0.0,        # 0%~100%，大小随机抖动
    "texture": "无",
    "texture_scale": 24.0,     # 纹理周期（源像素）
    "texture_depth": 0.5,      # 0%~100%，纹理的深浅
    "airbrush": False,         # 喷枪：按住不动会持续加深
    "air_rate": 12.0,          # 喷枪每秒补几笔
    "pressure": "关",           # 关 / 大小 / 不透明度 / 大小+不透明度
    "pressure_amount": 1.0,    # 0%~100%，模拟压感的强度
}
PRESSURE_MODES = ["关", "大小", "不透明度", "大小+不透明度"]

_stamp_cache = {}


def make_stamp(radius, hardness, shape="圆形", roundness=1.0, angle=0.0):
    """生成一个 (n,n) 的笔刷图章，值域 0~1。

    radius:    半径（像素），小于 0.5 时按 0.5 处理
    hardness:  0=极软，1=硬边
    shape:     圆形 / 方形 / 菱形
    roundness: 0.05~1，把笔尖沿旋转后的一个轴压扁 -> 椭圆笔尖
    angle:     笔尖旋转角度（度）
    """
    r = max(0.5, float(radius))
    h = float(np.clip(hardness, 0.0, 1.0))
    rn = float(np.clip(roundness, 0.05, 1.0))
    ang = float(angle) % 360.0
    n = int(max(1, round(r * 2)))
    key = (n, round(h, 3), shape, round(rn, 3), round(ang, 1))
    s = _stamp_cache.get(key)
    if s is not None:
        return s

    ax = np.arange(n, dtype=np.float32) - (n - 1) / 2.0
    yy, xx = np.meshgrid(ax, ax, indexing="ij")
    if ang:
        t = math.radians(ang)
        ct, st = math.cos(t), math.sin(t)
        xr = xx * ct + yy * st
        yr = -xx * st + yy * ct
    else:
        xr, yr = xx, yy
    xr = xr / r
    yr = yr / max(r * rn, 1e-6)

    if shape == "方形":                       # 切比雪夫距离 = 正方形
        d = np.maximum(np.abs(xr), np.abs(yr))
    elif shape == "菱形":                     # 曼哈顿距离 = 菱形
        d = np.abs(xr) + np.abs(yr)
    else:
        d = np.sqrt(xr * xr + yr * yr)

    if h >= 0.995:
        a = (d <= 1.0).astype(np.float32)
    else:
        a = np.clip((1.0 - d) / (1.0 - h + 1e-6), 0.0, 1.0)
        a = a * a * (3.0 - 2.0 * a)             # smoothstep，边缘过渡更自然
    a[d > 1.0] = 0.0
    _stamp_cache[key] = a
    return a


def stamp_cache_size():
    """给测试用：看看图章缓存有没有无限膨胀。"""
    return len(_stamp_cache)


def texture_patch(shape, origin, kind, scale):
    """按**源坐标**取一张图案，作为笔刷纹理。

    图案锚在图像上而不是跟着笔尖走 —— 和 Photoshop 的「纹理」一样，
    同一笔划过去图案是连续的，不会"游动"。
    `origin` 是这块 patch 左上角在源坐标里的位置。
    """
    from .effects import pattern_mask
    return pattern_mask(shape, (float(origin[0]), float(origin[1]), 0.0, 0.0),
                        kind, float(scale))


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
