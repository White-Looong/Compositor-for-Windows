# -*- coding: utf-8 -*-
"""色彩范围：按颜色 / 明暗挑出一片区域（PS 的「选择 > 色彩范围」）。

返回值一律是 (h,w) uint8 的**匹配度遮罩**（0 = 完全不匹配，255 = 完全匹配），
中间值就是"半选中"。调用方拿它当 Selection 的 mask 用 —— 容差越大过渡越软，
这和 PS 里"容差决定选区边缘的软硬"是一致的。

判定基准有两类：

1. **取样颜色**：吸管点出来的具体颜色，按各通道最大差值（切比雪夫距离）判定。
   可以存多个取样点，命中任一即选中。
2. **预置范围**：六色相 / 高光 / 中间调 / 阴影 / 肤色，用 HSB 或亮度判定。

**这是近似实现**，不是 Photoshop 的逐位复刻：PS 内部的色相分界与"溢色"
判定依赖它自己的色彩引擎和 CMYK 配置文件，这里用的是 OpenCV 的 HSV 与
Rec.601 亮度，数值上会差几个百分点。日常挑红 / 挑高光这类需求完全够用。
"""

from __future__ import annotations

import cv2
import numpy as np

# (id, 显示名, 是否需要吸管取样)
PRESETS = [
    ("sampled", "取样颜色", True),
    ("reds", "红色", False),
    ("yellows", "黄色", False),
    ("greens", "绿色", False),
    ("cyans", "青色", False),
    ("blues", "蓝色", False),
    ("magentas", "洋红", False),
    ("highlights", "高光", False),
    ("midtones", "中间调", False),
    ("shadows", "阴影", False),
    ("skintones", "肤色", False),
]

PRESET_NAMES = dict((p[0], p[1]) for p in PRESETS)
NEEDS_SAMPLE = set(p[0] for p in PRESETS if p[2])

# 色相分界（OpenCV 的 H 是 0~179，等于 0~358 度）
# (lo, hi, wrap)：wrap=True 表示区间跨过 0 点（如红色 344°~16°）
_HUE_BANDS = {
    "reds": (172.0, 8.0, True),
    "yellows": (20.0, 36.0, False),
    "greens": (38.0, 80.0, False),
    "cyans": (80.0, 100.0, False),
    "blues": (100.0, 133.0, False),
    "magentas": (133.0, 172.0, False),
    "skintones": (0.0, 25.0, False),
}


def _luminance(rgb):
    """Rec.601 亮度，返回 0~1 的 float32。"""
    f = rgb.astype(np.float32) / 255.0
    return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]


def _band_match(h, lo, hi, wrap):
    """h（0~179 环形）落在 [lo,hi] 内的程度：区间内 1，区间外按距离衰减到 0。

    返回 (inside 布尔, 到区间的距离)。
    """
    if wrap:
        inside = (h >= lo) | (h <= hi)
    else:
        inside = (h >= lo) & (h <= hi)
    # 到两个端点的**环形**距离，取小的那个
    d1 = np.abs(h - lo)
    d1 = np.minimum(d1, 180.0 - d1)
    d2 = np.abs(h - hi)
    d2 = np.minimum(d2, 180.0 - d2)
    d = np.where(inside, 0.0, np.minimum(d1, d2))
    return inside, d


def _ramp(t, soft):
    """t >= 0 时 1，t <= -soft 时 0，中间线性。soft 必须 > 0。"""
    return np.clip(t / soft + 1.0, 0.0, 1.0)


def range_mask(rgb, preset="sampled", samples=(), fuzziness=40.0,
               invert=False):
    """算出匹配度遮罩 (h,w) uint8。

    rgb        (h,w,3) uint8 的合成图（不含 alpha）
    samples    取样颜色 [ (r,g,b), ... ]，只在 preset == "sampled" 时用到
    fuzziness  容差 0~200
    invert     反相
    """
    f = float(np.clip(fuzziness, 0.0, 200.0))
    k = f / 200.0                       # 归一化容差 0~1
    h, w = rgb.shape[:2]

    if preset == "sampled":
        cols = [tuple(int(c) for c in s) for s in samples]
        if not cols:
            return np.zeros((h, w), np.uint8)
        # 容差 = 各通道允许的最大差值；软边取容差的 25%
        tol = max(1.0, f)
        soft = max(2.0, tol * 0.25)
        best = np.zeros((h, w), np.float32)
        arr = rgb.astype(np.int16)
        for (sr, sg, sb) in cols:
            d = np.abs(arr - np.array([sr, sg, sb], np.int16)).max(axis=2)
            np.maximum(best, _ramp(tol - d.astype(np.float32), soft),
                       out=best)
        out = best
    elif preset in ("highlights", "midtones", "shadows"):
        L = _luminance(rgb)
        if preset == "shadows":
            hi = 0.25 + k * 0.35
            out = _ramp(hi - L, 0.12)
        elif preset == "highlights":
            lo = 0.75 - k * 0.35
            out = _ramp(L - lo, 0.12)
        else:
            half = 0.25 + k * 0.45
            out = _ramp(half - np.abs(L - 0.5), 0.12)
    else:
        band = _HUE_BANDS.get(preset)
        if band is None:
            return np.zeros((h, w), np.uint8)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        H = hsv[..., 0].astype(np.float32)
        S = hsv[..., 1].astype(np.float32) / 255.0
        V = hsv[..., 2].astype(np.float32) / 255.0
        lo, hi, wrap = band
        # 容差把色相带向两边放宽（最多 ±15 个 OpenCV 单位 = ±30 度）
        pad = k * 15.0
        _, d = _band_match(H, lo - pad, hi + pad, wrap)
        out = _ramp(-d, 6.0)
        # 灰色不算"有颜色"：容差越大越能容忍低饱和
        smin = max(0.0, 0.15 - k * 0.15)
        out = out * _ramp(S - smin, 0.08)
        if preset == "skintones":
            # 肤色还要排除掉过暗和过饱和的部分
            out = out * _ramp(V - 0.20, 0.10) * _ramp(0.85 - S, 0.15)
        else:
            # 太暗的像素 HSV 色相本身就不稳，一律不算"有颜色"
            out = out * _ramp(V - 0.20, 0.12)

    if invert:
        out = 1.0 - out
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)
