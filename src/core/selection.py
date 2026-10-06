# -*- coding: utf-8 -*-
"""选区。

选区是**画布坐标**下的一张单通道遮罩（uint8，0~255，允许羽化后的中间值）。
所有绘制、填充、蒙版操作都受它限制。
"""

from __future__ import annotations

import cv2
import numpy as np

ADD = "add"
SUBTRACT = "subtract"
INTERSECT = "intersect"
REPLACE = "replace"


class Selection:
    def __init__(self, width, height, mask=None):
        self.w = int(width)
        self.h = int(height)
        if mask is None:
            self.mask = np.zeros((self.h, self.w), np.uint8)
        else:
            self.mask = mask.astype(np.uint8, copy=False)
            self.h, self.w = self.mask.shape[:2]

    # ---------- 基本 ----------

    @property
    def is_empty(self):
        return not bool(self.mask.any())

    def copy(self):
        return Selection(self.w, self.h, self.mask.copy())

    def bbox(self):
        ys, xs = np.nonzero(self.mask)
        if len(xs) == 0:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

    def float_mask(self):
        return self.mask.astype(np.float32) / 255.0

    def contains(self, x, y):
        xi, yi = int(x), int(y)
        if not (0 <= xi < self.w and 0 <= yi < self.h):
            return False
        return self.mask[yi, xi] > 127

    def contours(self, threshold=127):
        """返回用于绘制"行进蚁线"的轮廓点列表。"""
        _, bin_ = cv2.threshold(self.mask, threshold, 255, cv2.THRESH_BINARY)
        res = cv2.findContours(bin_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cs = res[0] if len(res) == 2 else res[1]
        return [c[:, 0, :] for c in cs if len(c) >= 3]

    # ---------- 几何构造 ----------

    @staticmethod
    def _canvas(w, h):
        return np.zeros((h, w), np.uint8)

    @classmethod
    def rect(cls, w, h, x0, y0, x1, y1):
        m = cls._canvas(w, h)
        x0, x1 = sorted((int(x0), int(x1)))
        y0, y1 = sorted((int(y0), int(y1)))
        x0 = max(0, min(w, x0)); x1 = max(0, min(w, x1))
        y0 = max(0, min(h, y0)); y1 = max(0, min(h, y1))
        if x1 > x0 and y1 > y0:
            m[y0:y1, x0:x1] = 255
        return cls(w, h, m)

    @classmethod
    def ellipse(cls, w, h, x0, y0, x1, y1):
        m = cls._canvas(w, h)
        x0, x1 = sorted((int(x0), int(x1)))
        y0, y1 = sorted((int(y0), int(y1)))
        if x1 <= x0 or y1 <= y0:
            return cls(w, h, m)
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        rx = max(0.5, (x1 - x0) / 2.0)
        ry = max(0.5, (y1 - y0) / 2.0)
        m = cls._canvas(w, h)
        cv2.ellipse(m, (int(round(cx)), int(round(cy))),
                    (int(round(rx)), int(round(ry))),
                    0, 0, 360, 255, -1, cv2.LINE_8)
        return cls(w, h, m)

    @classmethod
    def polygon(cls, w, h, pts, close=True):
        m = cls._canvas(w, h)
        if len(pts) < 2:
            return cls(w, h, m)
        arr = np.array(pts, np.int32).reshape(-1, 1, 2)
        if len(pts) == 2:
            cv2.line(m, tuple(arr[0, 0]), tuple(arr[1, 0]), 255, 1)
        elif close:
            cv2.fillPoly(m, [arr], 255, cv2.LINE_AA)
        else:
            cv2.polylines(m, [arr], False, 255, 1)
        return cls(w, h, m)

    @classmethod
    def magic(cls, w, h, rgb, x, y, tolerance, contiguous=True):
        """魔棒。rgb 是 (h,w,3) uint8 的合成图。"""
        m = cls._canvas(w, h)
        x = int(np.clip(x, 0, w - 1))
        y = int(np.clip(y, 0, h - 1))
        t = float(np.clip(tolerance, 0, 255))
        if not contiguous:
            ref = rgb[y, x].astype(np.int16)
            diff = np.abs(rgb.astype(np.int16) - ref).max(axis=2)
            m[(diff <= t)] = 255
            return cls(w, h, m)
        src = np.ascontiguousarray(rgb)
        mask = np.zeros((h + 2, w + 2), np.uint8)
        flags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
        tol = (t, t, t)
        try:
            cv2.floodFill(src, mask, (x, y), (0, 0, 0), tol, tol, flags)
        except cv2.error:
            return cls(w, h, m)
        m = mask[1:-1, 1:-1]
        return cls(w, h, m)

    @classmethod
    def all(cls, w, h):
        m = np.full((h, w), 255, np.uint8)
        return cls(w, h, m)

    # ---------- 运算 ----------

    def combine(self, other, mode):
        a = self.mask.astype(np.int16)
        b = other.mask.astype(np.int16)
        if mode == REPLACE:
            out = b
        elif mode == ADD:
            out = np.clip(a + b, 0, 255)
        elif mode == SUBTRACT:
            out = np.clip(a - b, 0, 255)
        elif mode == INTERSECT:
            out = np.minimum(a, b)
        else:
            out = b
        return Selection(self.w, self.h, out.astype(np.uint8))

    def feather(self, radius):
        """羽化：对遮罩做高斯模糊，边缘渐变到 0。"""
        r = float(max(0.0, radius))
        if r <= 0.01:
            return
        sigma = max(0.5, r / 2.0)
        k = int(max(3, round(sigma * 3))) | 1
        blurred = cv2.GaussianBlur(self.mask, (k, k), sigma)
        # 模糊会让峰值下降，按原始峰值重新归一化，中心保持全选
        peak = float(blurred.max())
        if peak > 1e-3:
            blurred = np.clip(blurred * (255.0 / peak), 0, 255)
        self.mask = blurred.astype(np.uint8)

    def expand(self, px):
        px = int(max(0, round(px)))
        if px == 0:
            return
        inv = (255 - self.mask).astype(np.uint8)
        dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
        self.mask[self.mask == 0] = 0
        self.mask[(self.mask == 0) & (dist <= px)] = 255

    def contract(self, px):
        px = int(max(0, round(px)))
        if px == 0:
            return
        dist = cv2.distanceTransform(self.mask, cv2.DIST_L2, 3)
        self.mask[dist <= px] = 0

    def invert(self):
        self.mask = (255 - self.mask).astype(np.uint8)

    def translate(self, dx, dy):
        """平移选区轮廓（内容跟着走，不做裁剪）。"""
        dx, dy = int(round(dx)), int(round(dy))
        if dx == 0 and dy == 0:
            return
        M = np.array([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]])
        self.mask = cv2.warpAffine(self.mask, M, (self.w, self.h),
                                   flags=cv2.INTER_NEAREST,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(0,))

    def resize_to(self, w, h):
        """画布尺寸变化时同步选区尺寸。"""
        if (w, h) == (self.w, self.h):
            return
        self.mask = cv2.resize(self.mask, (w, h), interpolation=cv2.INTER_NEAREST)
        self.w, self.h = w, h
