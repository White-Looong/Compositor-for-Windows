# -*- coding: utf-8 -*-
"""浮动选区：把选中的像素从图层里"揭"出来单独移动。

Photoshop 用移动工具拖动选区内容时做的三步：

    1. lift   —— 选区里的像素被复制成一个独立的浮动层，原处被挖空
                 （Alt 拖动时原处保留，等于"复制一份再挪"）
    2. 拖动   —— 只移动这个浮动层，原图层不动
    3. stamp  —— 回车 / 切换工具 / 取消选区时把浮动层盖回原图层

浮动层**不在 doc.layers 里**，由 render 单独画在最上面，所以它不参与图层
面板、不参与排序，也不会被误当成普通图层编辑。

坐标：浮动层的位图就是画布坐标系下的一块（tx/ty 是它的中心），
sx=sy=1、rot=0。stamp 时要把它重采样回目标图层的**源坐标**再合成，
这一步用 cv2.remap 做，避免"先渲染到画布再反变换"那种两次插值。
"""

from __future__ import annotations

import cv2
import numpy as np

from .layer import LAYER_IMAGE, Layer
from .render import _warp_layer, _warp_straight


def can_lift(doc, layer):
    """能不能把当前选区的内容揭出来。"""
    if doc is None or layer is None:
        return False
    sel = doc.selection
    if sel is None or sel.is_empty:
        return False
    # 组 / 调整层 / 文字层没有可直接改的像素
    if layer.is_group or layer.is_adjustment or layer.is_text:
        return False
    if layer.locked or layer.image is None:
        return False
    return doc.float_layer is None


def lift_selection(doc, layer, copy_mode=False):
    """把选区内容揭成浮动层，返回它；条件不满足返回 None。

    copy_mode=False 时原图层上被选中的像素会被清成透明（PS 的默认行为）。
    调用方负责 detach_pixels + commit。
    """
    if not can_lift(doc, layer):
        return None
    sel = doc.selection
    box = sel.bbox()
    if box is None:
        return None
    x0, y0, x1, y1 = box
    w = max(1, x1 - x0)
    h = max(1, y1 - y0)

    got = _warp_layer(layer, doc, x0, y0, w, h)
    if got is None:
        return None
    pc, pa, lx0, ly0, lx1, ly1 = got
    # `_warp_layer` 只返回"图层包围盒 ∩ 选区框"那一块；图层比选区小的时候
    # （比如旋转过的图层、或者选区拖到空白处）patch 会比选区框小，
    # 直接和选区蒙版相乘会广播出错 —— 先摆回选区框大小的缓冲里。
    if (lx1 - lx0, ly1 - ly0) != (w, h):
        c = np.zeros((h, w, 3), np.float32)
        a = np.zeros((h, w, 1), np.float32)
        c[ly0:ly1, lx0:lx1] = pc
        a[ly0:ly1, lx0:lx1] = pa
    else:
        c, a = pc, pa
    mask = sel.float_mask()[y0:y1, x0:x1][..., None]
    aa = np.clip(a * mask, 0.0, 1.0)
    if float(aa.max()) <= 0.0:
        return None        # 选区里根本没有这个图层的内容

    rgb = np.clip(c * 255.0, 0, 255).astype(np.uint8)
    alpha = np.clip(aa * 255.0, 0, 255).astype(np.uint8)
    img = np.ascontiguousarray(np.concatenate([rgb, alpha], axis=2))

    fl = Layer("浮动选区", LAYER_IMAGE, image=img)
    fl.tx = (x0 + x1) / 2.0
    fl.ty = (y0 + y1) / 2.0
    doc.float_layer = fl

    if not copy_mode:
        erase_selection(doc, layer)
    return fl


def erase_selection(doc, layer):
    """把选区覆盖到的地方在图层源图上清成透明（原地修改）。"""
    sel = doc.selection
    if sel is None or sel.is_empty or layer.image is None:
        return
    M = layer.matrix()
    if M is None:
        return
    Minv = cv2.invertAffineTransform(M)
    h, w = layer.image.shape[:2]
    m = cv2.warpAffine(sel.float_mask(), Minv, (w, h),
                       flags=cv2.INTER_LINEAR)
    a = layer.image[..., 3:4].astype(np.float32)
    layer.image[..., 3:4] = np.clip(
        a * (1.0 - np.clip(m, 0.0, 1.0)[..., None]), 0, 255).astype(np.uint8)


def float_bbox(fl):
    """浮动层在画布坐标下的矩形 (x0, y0, x1, y1)。

    走 `Layer.bbox()` 而不是手算，因为浮动层**可以带变换** —— 拖手柄缩放
    / 旋转过之后，包围盒要按变换后的四角算。不裁到画布内：浮到画布外的
    内容也是内容，落定时照样要盖回图层。
    """
    if fl is None or fl.image is None:
        return None
    return fl.bbox()


def stamp_float(doc, layer=None):
    """把浮动层盖回目标图层（原地改像素），返回是否真的落定了。

    layer 为 None 时用当前选中图层；目标不可写时浮动层直接丢弃。
    调用方负责 detach_pixels + commit。
    """
    fl = doc.float_layer
    if fl is None or fl.image is None:
        doc.float_layer = None
        return False
    if layer is None or layer.image is None or layer.locked or layer.is_text:
        doc.float_layer = None
        return False
    M = layer.matrix()
    if M is None:
        doc.float_layer = None
        return False

    # 1) 浮动层（含它自己的变换）先渲成画布坐标系里的一块 RGBA
    box = float_bbox(fl)
    if box is None:
        doc.float_layer = None
        return False
    x0 = int(np.floor(box[0]))
    y0 = int(np.floor(box[1]))
    x1 = int(np.ceil(box[2]))
    y1 = int(np.ceil(box[3]))
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        doc.float_layer = None
        return False
    Mf = fl.matrix(ox=x0, oy=y0)        # 浮动位图 -> 这块 patch
    # 必须**先预乘 alpha 再插值**（§3.2）：直接对直线色做双线性，边缘会和
    # 外面的透明黑（RGB=0）混出一条脏边。这里走渲染用的同一个函数
    pc, pa = _warp_straight(fl.image, Mf, (w, h), cv2.INTER_LINEAR)
    patch = np.concatenate([pc, pa], axis=2)

    # 2) 这块 patch 覆盖到源图的哪个区域？四角反变换回去求包围盒
    Minv = cv2.invertAffineTransform(M)
    pts = np.array([[x0, y0, 1.0], [x1, y0, 1.0],
                    [x1, y1, 1.0], [x0, y1, 1.0]], np.float64)
    sp = pts @ Minv.T                       # 浮动层四角 -> 源坐标
    H, W = layer.image.shape[:2]
    sx0 = max(0, int(np.floor(sp[:, 0].min())) - 2)
    sy0 = max(0, int(np.floor(sp[:, 1].min())) - 2)
    sx1 = min(W, int(np.ceil(sp[:, 0].max())) + 2)
    sy1 = min(H, int(np.ceil(sp[:, 1].max())) + 2)
    if sx1 <= sx0 or sy1 <= sy0:
        doc.float_layer = None
        return False
    sw, sh = sx1 - sx0, sy1 - sy0

    # 3) patch -> 源坐标。
    #    注意 warpAffine 不加 WARP_INVERSE_MAP 时算的是 dst(p) = src(M⁻¹·p)
    #    （§4.3-13），也就是"传进去的矩阵被当作 dst->src"。所以先拼出
    #    **源图坐标 -> patch 坐标** 的正向矩阵再求逆，别直接把正向的传进去
    a, c, tx = M[0, 0], M[0, 1], M[0, 2]
    b, d, ty = M[1, 0], M[1, 1], M[1, 2]
    fwd = np.array([[a, c, a * sx0 + c * sy0 + tx - x0],
                    [b, d, b * sx0 + d * sy0 + ty - y0]], np.float64)
    sc, sa_ = _warp_straight(patch * 255.0, cv2.invertAffineTransform(fwd),
                             (sw, sh), cv2.INTER_LINEAR)
    s = np.concatenate([sc, sa_], axis=2)

    dst = layer.image[sy0:sy1, sx0:sx1].astype(np.float32) / 255.0
    sa = s[..., 3:4]
    da = dst[..., 3:4]
    oa = np.clip(sa + da * (1.0 - sa), 0.0, 1.0)
    safe = np.where(np.abs(oa) < 1e-6, 1.0, oa)
    # 直线色的 source-over：两边都按 alpha 加权
    oc = (s[..., :3] * sa + dst[..., :3] * da * (1.0 - sa)) / safe
    out = np.concatenate([np.clip(oc, 0.0, 1.0), oa], axis=2)
    layer.image[sy0:sy1, sx0:sx1] = np.clip(out * 255.0, 0, 255).astype(np.uint8)

    doc.float_layer = None
    return True


def discard_float(doc):
    """丢掉浮动内容（不盖回去）。原处已经挖空，想找回来用撤销。"""
    if doc is not None and doc.float_layer is not None:
        doc.float_layer = None
        return True
    return False
