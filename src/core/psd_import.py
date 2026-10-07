# -*- coding: utf-8 -*-
"""PSD 导入：把 Photoshop 的图层树映射到本项目的 Layer。

用 psd-tools 解析（纯 Python，比自己写 PSD 解析划算得多）。

能直接映射的：
  * 图层组        -> LAYER_GROUP，PSD 的 'pass' 对应 PASS_THROUGH
  * 像素图层      -> LAYER_IMAGE
  * 图层蒙版      -> layer.mask（灰度，缩放到图层位图尺寸）
  * 剪贴蒙版      -> layer.clipped
  * 混合模式      -> 按 PSD 的 4 字节 key 映射，对不上的退回 Normal
  * 图层样式      -> layer.effects（描边 / 投影 / 内阴影 / 外发光 / 内发光 /
    斜面浮雕 / 光泽 / 颜色叠加 / 渐变叠加），见 core/psd_effects.py
  * 文字图层      -> LAYER_TEXT，参数从 TySh 还原（见 core/psd_text.py），
    导入后仍能改字 / 换字体 / 调字号

降级处理的（保住画面，丢掉可编辑性）：
  * 形状 / 智能对象 / 调整层 -> 合成成一张位图
  * 文字层在**还原不出来**时也合成成位图：字体不在系统里、同层混排、
    带描边、变形文字 —— 这些情况下用 PSD 内嵌的合成图，画面是准的

坐标换算：PSD 的 bbox 是 (left, top, right, bottom)，画布像素坐标；
本项目用 tx/ty 表示图层**中心**，位图取原始尺寸，所以 sx=sy=1。
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from .blend import BLEND_MODES, PASS_THROUGH
from .document import Document, make_image_layer
from .layer import LAYER_GROUP, LAYER_IMAGE, LAYER_TEXT, Layer
from . import psd_effects, psd_text

PSD_EXT = ".psd"

# 导入结果的统计，导入完由 main_window 报给用户（"3 个文字层还原成了可编辑文字"）
STATS = {"text": 0, "text_fallback": 0, "effects": 0, "layers": 0}


def reset_stats():
    STATS.update({"text": 0, "text_fallback": 0, "effects": 0, "layers": 0})

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


def _common_attrs(lay, layer):
    """不透明度 / 可见性 / 混合模式 / 剪贴蒙版 —— 两种图层都一样。"""
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
    return lay


def _apply_effects(lay, layer):
    """把 PSD 的图层样式接到 lay.effects 上（转不出来就当没有）。"""
    eff = psd_effects.effects_from_layer(layer)
    if eff:
        lay.effects = eff
        STATS["effects"] += 1
    return lay


def _convert_text(layer, name):
    """文字图层 -> LAYER_TEXT。还原不出来返回 None（调用方降级成位图）。"""
    try:
        ts = layer.typesetting
    except Exception:
        return None
    if ts is None:
        return None
    try:
        engine = layer.engine_dict
    except Exception:
        engine = None
    params = psd_text.text_params_from_typesetting(ts, engine_dict=engine)
    if not params:
        return None
    arr = psd_text.render_text(params)
    if arr is None or arr.size == 0:
        return None

    lay = Layer(name, LAYER_TEXT, image=arr)
    lay.text = params
    lay.blend = "Normal"
    # PSD 的 bbox 是文字在画布上的实际占位，而 render_text 四周留了 padding，
    # 所以按"墨迹范围 vs bbox"校正缩放与中心（换算偏差全被 sx/sy 吸收）
    sx, sy, cx, cy = psd_text.ink_scale(arr, _bbox(layer))
    lay.sx, lay.sy = sx, sy
    lay.tx, lay.ty = cx, cy
    _common_attrs(lay, layer)
    m = _layer_mask(layer, arr.shape[:2])
    if m is not None:
        lay.mask = m
        lay.mask_enabled = True
    return lay


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

    # 文字层：先试还原成可编辑文字层，不行再走位图
    if (getattr(layer, "kind", "") or "") == "type":
        lay = _convert_text(layer, name)
        if lay is not None:
            STATS["text"] += 1
            _apply_effects(lay, layer)
            return lay
        STATS["text_fallback"] += 1

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
    _common_attrs(lay, layer)

    m = _layer_mask(layer, (h, w))
    if m is not None:
        lay.mask = m
        lay.mask_enabled = True
    _apply_effects(lay, layer)
    return lay


def load_psd(path):
    """读一个 .psd，返回 Document。

    注意不写 doc.path —— PSD 不是 .cwproj，不能让 Ctrl+S 直接覆盖原文件。
    导入完可以读 `STATS` 知道还原成了几个文字层 / 带上了几处图层样式。
    """
    if not is_available():
        raise RuntimeError(
            "需要 psd-tools 才能导入 PSD：\n"
            "  pip install psd-tools -i https://mirrors.aliyun.com/pypi/simple")
    from psd_tools import PSDImage

    reset_stats()
    psd = PSDImage.open(path)
    doc = Document(int(psd.width), int(psd.height),
                   os.path.splitext(os.path.basename(path))[0])
    for layer in psd:
        lay = _convert(layer)
        if lay is not None:
            doc.layers.append(lay)
    STATS["layers"] = len(doc.layers)

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
