# -*- coding: utf-8 -*-
"""描边引擎：把画布上的鼠标轨迹变成图层像素的改变。

关键点是坐标换算——图层可能被缩放/旋转过，而像素必须写在**图层源图像坐标系**里，
所以每条描边开始时先求一次 canvas->src 的逆仿射变换，之后每个点都走这个矩阵。

不透明度/流量按 Photoshop 的语义实现：
  * flow    每次落笔的量
  * opacity 单次描边累计的上限（不开启"喷枪"时不会无限叠加）
用一个累计缓冲 acc 来实现这个上限。

顺带记录**脏矩形**：每一笔只改动源图上一小块，渲染时没必要整幅重算。
`take_dirty()` 把累积的源坐标脏区换算成画布坐标还给调用方，取走即清零。
"""

from __future__ import annotations

import cv2
import numpy as np

from .brush import (composite_mask, composite_rgba, erase_rgba, make_stamp,
                    _overlap)


def _gray(rgb):
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


class Stroke:
    """一次按下-拖动-松开的完整笔画。"""

    def __init__(self, doc, layer, tool="brush", size=30.0, hardness=0.6,
                 opacity=1.0, flow=1.0, smoothing=0.3, target="pixel",
                 color=(0, 0, 0)):
        self.doc = doc
        self.layer = layer
        self.tool = tool                 # brush / eraser
        self.size = float(size)
        self.hardness = float(hardness)
        self.opacity = float(np.clip(opacity, 0.0, 1.0))
        self.flow = float(np.clip(flow, 0.01, 1.0))
        self.smoothing = float(np.clip(smoothing, 0.0, 1.0))
        self.target = target             # pixel / mask
        self.color = tuple(int(c) for c in color)

        self.ok = False
        self._src = None                 # 目标数组（image 或 mask）
        self._is_mask = False
        self._Minv = None
        self._sel = None                 # 源坐标下的选区 float32
        self._stamp = None
        self._acc = None
        self._last = None
        self._smooth = None
        self._Mfwd = None                # src -> canvas（Minv 的逆）
        self._dirty = None               # 源坐标累积脏区 [x0,y0,x1,y1]

    # ---------- 生命周期 ----------

    def begin(self, canvas_pt):
        layer = self.layer
        if layer is None or layer.is_group or layer.locked:
            return False
        if self.target == "mask":
            if layer.mask is None:
                return False
            self._src = layer.mask
            self._is_mask = True
        else:
            if layer.image is None:
                return False
            self._src = layer.image
            self._is_mask = False

        # 写时复制：先把要改的数组复制一份，历史快照里的旧数据才不会被污染
        self.doc.detach_pixels(layer)
        if self.target == "mask":
            self._src = layer.mask
        else:
            self._src = layer.image

        if layer.is_adjustment:
            # 调整层没有像素，它的蒙版就是画布坐标系，逆变换是恒等
            self._Minv = np.array([[1.0, 0.0, 0.0],
                                   [0.0, 1.0, 0.0]], np.float64)
        else:
            M = layer.matrix()
            if M is None:
                return False
            self._Minv = cv2.invertAffineTransform(M)   # canvas -> src
        self._Mfwd = cv2.invertAffineTransform(self._Minv)   # src -> canvas
        self._dirty = None

        src_h, src_w = self._src.shape[:2]
        sel = self.doc.selection
        if sel is not None and not sel.is_empty:
            sm = sel.float_mask()
            self._sel = cv2.warpAffine(sm, self._Minv, (src_w, src_h),
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=(0,))
            np.clip(self._sel, 0.0, 1.0, out=self._sel)
            if not self._sel.any():
                return False

        scale = (abs(layer.sx) + abs(layer.sy)) / 2.0
        scale = max(scale, 1e-3)
        self._r = max(0.5, (self.size / 2.0) / scale)
        self._stamp = make_stamp(self._r, self.hardness)
        self._buf = np.zeros_like(self._stamp)
        self._full = np.zeros_like(self._stamp)
        self._spacing = max(0.75, self._r * 0.25)

        if self.opacity < 0.999:
            self._acc = np.zeros((src_h, src_w), np.float32)

        self.ok = True
        p = self._to_src(canvas_pt)
        self._smooth = p
        self._last = p
        self._dab(p)
        return True

    def extend(self, canvas_pt):
        if not self.ok:
            return
        p = self._to_src(canvas_pt)
        k = 1.0 - 0.95 * self.smoothing
        self._smooth = (self._smooth[0] + (p[0] - self._smooth[0]) * k,
                        self._smooth[1] + (p[1] - self._smooth[1]) * k)
        self._line_to(self._smooth)

    def end(self):
        self.ok = False
        self._acc = None
        self._sel = None

    # ---------- 脏矩形 ----------

    def take_dirty(self, pad=0):
        """取走自上次调用以来画过的源坐标区域，换算成**画布坐标**矩形。

        返回 (x0, y0, x1, y1) 整数、已按 pad 外扩；没有任何改动时返回 None。
        pad 用来给图层样式（投影 / 发光会往外扩）留地方。
        """
        d = self._dirty
        self._dirty = None
        if d is None or self._Mfwd is None:
            return None
        m = self._Mfwd
        xs = []
        ys = []
        for px, py in ((d[0], d[1]), (d[2], d[1]), (d[0], d[3]), (d[2], d[3])):
            xs.append(m[0, 0] * px + m[0, 1] * py + m[0, 2])
            ys.append(m[1, 0] * px + m[1, 1] * py + m[1, 2])
        x0 = int(np.floor(min(xs))) - pad
        y0 = int(np.floor(min(ys))) - pad
        x1 = int(np.ceil(max(xs))) + pad
        y1 = int(np.ceil(max(ys))) + pad
        if x1 <= x0 or y1 <= y0:
            return None
        return (x0, y0, x1, y1)

    # ---------- 内部 ----------

    def _to_src(self, pt):
        x, y = float(pt[0]), float(pt[1])
        m = self._Minv
        return (m[0, 0] * x + m[0, 1] * y + m[0, 2],
                m[1, 0] * x + m[1, 1] * y + m[1, 2])

    def _line_to(self, p):
        x0, y0 = self._last
        x1, y1 = p
        dx, dy = x1 - x0, y1 - y0
        dist = (dx * dx + dy * dy) ** 0.5
        n = int(dist / self._spacing)
        if n <= 0:
            return
        for i in range(1, n + 1):
            t = i / n
            self._dab((x0 + dx * t, y0 + dy * t))
        self._last = p

    def _dab(self, p):
        np.multiply(self._stamp, self.flow, out=self._buf)
        ov = _overlap(self._src.shape, p[0], p[1], self._buf.shape[0])
        if ov is None:
            return
        dy0, dy1, dx0, dx1, sy0, sy1, sx0, sx1 = ov
        sub = self._buf[sy0:sy1, sx0:sx1]

        if self._acc is not None:
            acc = self._acc[dy0:dy1, dx0:dx1]
            room = np.maximum(0.0, self.opacity - acc)
            sub = np.minimum(sub, room)
            acc += sub
        if self._sel is not None:
            sub = sub * self._sel[dy0:dy1, dx0:dx1]

        # 记脏区：图章是以 p 为中心、n×n 的方块（源坐标）
        n = float(self._full.shape[0])
        r = n / 2.0 + 1.0
        bx0, by0 = p[0] - r, p[1] - r
        bx1, by1 = p[0] + r, p[1] + r
        if self._dirty is None:
            self._dirty = [bx0, by0, bx1, by1]
        else:
            d = self._dirty
            d[0] = min(d[0], bx0)
            d[1] = min(d[1], by0)
            d[2] = max(d[2], bx1)
            d[3] = max(d[3], by1)

        # 合成函数需要完整的图章尺寸来自己算重叠区，所以回填到全尺寸缓冲里
        self._full[:] = 0.0
        self._full[sy0:sy1, sx0:sx1] = sub

        if self._is_mask:
            composite_mask(self._src, p[0], p[1], self._full, _gray(self.color))
        elif self.tool == "eraser":
            erase_rgba(self._src, p[0], p[1], self._full)
        else:
            composite_rgba(self._src, p[0], p[1], self._full, self.color)
