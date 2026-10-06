# -*- coding: utf-8 -*-
"""PSD 导入：把 Photoshop 的图层树映射到本项目的 Layer。

用 psd-tools 解析（纯 Python，比自己写 PSD 解析划算得多）。

能直接映射的：
  * 图层组        -> LAYER_GROUP，PSD 的 'pass' 对应 PASS_THROUGH
  * 像素图层      -> LAYER_IMAGE
  * 图层蒙版      -> layer.mask（灰度，缩放到图层位图尺寸）
  * 剪贴蒙版      -> layer.clipped
  * 混合模式      -> 按 PSD 的 4 字节 key 映射，对不上的退回 Normal

降级处理的（保住画面，丢掉可编辑性）：
  * 文字 / 形状 / 智能对象 / 调整层 -> 合成成一张位图
    PSD 里文字的字体、字号、字距散落在 TySh 标记块里，
    psd-tools 只稳定暴露文本内容，硬还原会渲出一堆错位的东西，
    不如直接用 PSD 内嵌的合成图，画面是准的。

坐标换算：PSD 的 bbox 是 (left, top, right, bottom)，画布像素坐标；
本项目用 tx/ty 表示图层**中心**，位图取原始尺寸，所以 sx=sy=1。
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from .blend import BLEND_MODES, PASS_THROUGH
from .document import Document, make_image_layer
from .layer import LAYER_GROUP, LAYER_IMAGE, Layer

PSD_EXT = ".psd"

# PSD 的 4 字节混合模式 key -> 本项目 BLEND_MODES 里的名字
# 注意 Overlay 在 PSD 规范里是 'over'，不是 'ovrl'
BLEND_MAP = {
    "norm": "Normal",
    "diss": "Dissolve",
    "dark": "Darken",
    "mul ": "Multiply",
    "idiv": "Color Burn",
    "lbrn": "Linear Burn",
    "dkCl": "Darker Color",
    "lite": "Lighten",
    "scrn": "Screen",
    "div ": "Color Dodge",
    "lddg": "Linear Dodge (Add)",
    "lgCl": "Lighter Color",
    "over": "Overlay",
    "sLit": "Soft Light",
    "hLit": "Hard Light",
    "vLit": "Vivid Light",
    "lLit": "Linear Light",
    "pLit": "Pin Light",
    "hMix": "Hard Mix",
    "diff": "Difference",
    "smud": "Exclusion",
    "fsub": "Subtract",
    "fdiv": "Divide",
    "hue ": "Hue",
    "sat ": "Saturation",
    "colr": "Color",
    "lum ": "Luminosity",
}


def is_available():
    """psd-tools 是否装了。没装就给个友好提示，别在使用时才炸。"""
    try:
        import psd_tools  # noqa: F401
        return True
    except Exception:
        return False


def blend_name(bm):
    """PSD 混合模式 -> 本项目名字。对不上退回 Normal。"""
    key = bm
    if isinstance(key, (bytes, bytearray)):
        key = bytes(key).decode("ascii", "ignore")
    name = BLEND_MAP.get(str(key))
    if name is None or name not in BLEND_MODES:
        return "Normal"
    return name


def _is_group(layer):
    kind = getattr(layer, "kind", "") or ""
    if kind == "group":
        return True
    try:
        return bool(layer.is_group())
    except Exception:
        return False


def _children(layer):
    """子图层（底 -> 顶）。非组图层返回空表。"""
    try:
        return list(iter(layer))
    except TypeError:
        return []
    except Exception:
        return []


def _as_rgba(im):
    """psd-tools 给的 PIL Image / ndarray -> (h,w,4) uint8 RGBA。"""
    if im is None:
        return None
    if hasattr(im, "convert"):          # PIL.Image
        arr = np.array(im.convert("RGBA"), np.uint8)
    else:                                # numpy
        arr = np.asarray(im)
    if arr.ndim == 2:
        arr = np.dstack([arr, arr, arr, np.full(arr.shape, 255, np.uint8)])
    elif arr.ndim == 3:
        if arr.shape[2] == 4:
            pass
        elif arr.shape[2] == 3:
            arr = np.concatenate(
                [arr, np.full(arr.shape[:2] + (1,), 255, np.uint8)], axis=2)
        else:
            arr = arr[..., :3]
            arr = np.concatenate(
                [arr, np.full(arr.shape[:2] + (1,), 255, np.uint8)], axis=2)
    else:
        return None
    return np.ascontiguousarray(arr.astype(np.uint8))


def _layer_bitmap(layer):
    """取图层位图。像素层优先 topil()，其余（组/智能对象/形状/调整）走 composite()。"""
    kind = getattr(layer, "kind", "") or ""

    def try_composite():
        try:
            return layer.composite()
        except Exception:
            return None

    def try_topil():
        try:
            return layer.topil()
        except Exception:
            return None

    if kind == "pixel":
        im = try_topil() or try_composite()
    else:
        im = try_composite() or try_topil()
    return _as_rgba(im)


def _layer_mask(layer, shape):
    """取图层蒙版，缩放到位图尺寸 (h,w)，灰度 uint8。"""
    try:
        m = layer.mask
    except Exception:
        return None
    if m is None:
        return None
    try:
        im = m.topil()
    except Exception:
        return None
    if im is None:
        return None
    arr = np.array(im.convert("L"), np.uint8)
    h, w = shape[:2]
    if arr.shape[:2] != (h, w) and h > 0 and w > 0:
        arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(arr)


def _bbox(layer):
    try:
        l, t, r, b = layer.bbox
    except Exception:
        return None
    l, t, r, b = int(l), int(t), int(r), int(b)
    if r <= l or b <= t:
        return None
    return l, t, r, b


def _convert(layer):
    """把一个 psd-tools 图层（含组）转成 Layer；没法转换时返回 None。"""
    name = (getattr(layer, "name", "") or "").strip() or "图层"

    if _is_group(layer):
        g = Layer(name, LAYER_GROUP)
        try:
            bm = layer.blend_mode
        except Exception:
            bm = None
        key = bm
        if isinstance(key, (bytes, bytearray)):
            key = bytes(key).decode("ascii", "ignore")
        g.blend = PASS_THROUGH if str(key) == "pass" else blend_name(bm)
        try:
            g.opacity = max(0.0, min(1.0, int(layer.opacity) / 255.0))
        except Exception:
            g.opacity = 1.0
        try:
            g.visible = bool(layer.visible)
        except Exception:
            g.visible = True
        for child in _children(layer):
            c = _convert(child)
            if c is not None:
                g.children.append(c)
        return g if g.children else None

    arr = _layer_bitmap(layer)
    if arr is None or arr.size == 0 or arr.shape[0] == 0 or arr.shape[1] == 0:
        return None
    h, w = arr.shape[:2]

    box = _bbox(layer)
    if box is not None:
        l, t, r, b = box
        if (r - l, b - t) != (w, h):
            # 位图尺寸和 bbox 不一致（有 padding 或缩放过的智能对象），拉回 bbox
            arr = cv2.resize(arr, (r - l, b - t), interpolation=cv2.INTER_AREA)
            h, w = arr.shape[:2]
        cx, cy = (l + r) / 2.0, (t + b) / 2.0
    else:
        cx, cy = w / 2.0, h / 2.0

    lay = Layer(name, LAYER_IMAGE, image=arr)
    lay.tx = cx
    lay.ty = cy
    try:
        lay.opacity = max(0.0, min(1.0, int(layer.opacity) / 255.0))
    except Exception:
        lay.opacity = 1.0
    try:
        lay.visible = bool(layer.visible)
    except Exception:
        lay.visible = True
    try:
        lay.blend = blend_name(layer.blend_mode)
    except Exception:
        lay.blend = "Normal"
    try:
        lay.clipped = bool(layer.clipping)
    except Exception:
        lay.clipped = False

    m = _layer_mask(layer, (h, w))
    if m is not None:
        lay.mask = m
        lay.mask_enabled = True
    return lay


def load_psd(path):
    """读一个 .psd，返回 Document。

    注意不写 doc.path —— PSD 不是 .cwproj，不能让 Ctrl+S 直接覆盖原文件。
    """
    if not is_available():
        raise RuntimeError(
            "需要 psd-tools 才能导入 PSD：\n"
            "  pip install psd-tools -i https://mirrors.aliyun.com/pypi/simple")
    from psd_tools import PSDImage

    psd = PSDImage.open(path)
    doc = Document(int(psd.width), int(psd.height),
                   os.path.splitext(os.path.basename(path))[0])
    for layer in psd:
        lay = _convert(layer)
        if lay is not None:
            doc.layers.append(lay)

    if not doc.layers:
        # 没有图层信息（扁平 PSD）：用整幅合成图兜底
        try:
            arr = _as_rgba(psd.composite())
        except Exception:
            arr = None
        if arr is not None and arr.size:
            doc.layers.append(
                make_image_layer("背景", arr, doc.width, doc.height))
        else:
            bg = np.zeros((doc.height, doc.width, 4), np.uint8)
            bg[..., :3] = 255
            bg[..., 3] = 255
            doc.layers.append(
                make_image_layer("背景", bg, doc.width, doc.height))
    return doc
