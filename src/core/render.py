# -*- coding: utf-8 -*-
"""文档渲染：把图层树合成为一张 RGBA 位图。

要点：
  * 位图始终以原始分辨率保存，渲染时才做仿射变换 —— 缩放是无损的
  * 变换时先预乘 alpha 再插值，避免透明黑边产生的"脏边"
  * 只在图层包围盒范围内做运算，其他区域直接跳过
  * 图层组默认 Pass Through（子图层直接与下方背景混合），
    改混合模式或不透明度 < 100% 时自动隔离成独立缓冲
  * **调整层**作用于「它下面所有图层的合成结果」，不透明度 / 蒙版 / 混合模式
    决定调整作用的强弱，alpha 不受影响
  * **剪贴蒙版**（调整层的 clipped=True）：把基底图层先渲进独立缓冲，
    套用调整后再按基底自身的混合模式与不透明度合成下去，所以只影响基底

性能相关（见 §5.21）：
  * `render_document(doc, region=...)` 只渲一块，`_warp_layer` 会先把源图裁成
    需要的那一小块再转 float32 —— 不裁的话，12 MP 的背景层渲 500x500 也要
    先把整张图转成 192 MB 的 float32，慢得离谱
  * `render_proxy(doc, scale)` 渲低分辨率代理图，交互期用它顶替全分辨率
  * **拖调整层滑块**走 `render_from_snapshot()`：调整层下方的合成结果在
    一次整幅渲染时顺手存成快照，之后改参数只重算调整层本身 —— 见 §5.23
  * **分块渲染** `render_tiled()` / `render_region_tiled()`：把画布切成
    tile² 的瓦片逐块算，所有临时缓冲复用同一块，峰值内存不再随画布面积
    线性增长 —— 见 §5.24
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .adjust import apply_adjustment
from .blend import PASS_THROUGH, blend_colors, composite
from .document import Document
from .effects import apply_effects, effect_padding, scale_effects
from .smart import sync_smart


def _layer_pad(layer):
    """图层样式需要的外扩像素数；没样式就是 0。"""
    if not layer.effects:
        return 0
    return effect_padding(layer.effects)


def _interp_flags(sx, sy):
    s = (abs(sx) + abs(sy)) / 2.0
    return cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR


# 源图裁剪开关：关掉之后走"整张源图转 float32"的老路径，只给测试对比用
SRC_CROP = True

# 调整层快照的精度（见 _store_snap）。默认 uint8 省内存；测试会临时换成
# float32，用来验证"快路径的逻辑本身是精确的"。
SNAP_DTYPE = np.uint8


def _src_window(src, M, dw, dh):
    """目标矩形 (0,0)-(dw,dh) 在源图里对应的窗口，已按插值半径外扩。

    返回 (x0,y0,x1,y1)（已裁剪到源图范围内）；算不出来时返回 None。
    外扩量取决于 src->dst 的缩放比：缩得越狠，INTER_AREA 的核越大。
    """
    if not SRC_CROP:
        return None
    sh, sw = src.shape[:2]
    if sw <= 0 or sh <= 0 or dw <= 0 or dh <= 0:
        return None
    det = abs(M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0])
    if det < 1e-9:
        return None                 # 矩阵不可逆（缩放被压成 0），老实整张算
    try:
        Minv = cv2.invertAffineTransform(M)
    except cv2.error:
        return None
    px = []
    py = []
    for x, y in ((0.0, 0.0), (float(dw), 0.0), (0.0, float(dh)),
                 (float(dw), float(dh))):
        px.append(Minv[0, 0] * x + Minv[0, 1] * y + Minv[0, 2])
        py.append(Minv[1, 0] * x + Minv[1, 1] * y + Minv[1, 2])

    # 外扩量 = 目标像素在源图上的"足迹"半径：A⁻¹ 的两列是单位正方形的像，
    # 取包围半径 0.5*(|u|+|v|)。各向异性缩放（比如 sx=0.1 而 sy=3）时必须这么算，
    # 只看行列式会低估很多，裁剪出来的窗口不够插值用，边缘就会缺内容。
    ux = math.hypot(Minv[0, 0], Minv[1, 0])
    vy = math.hypot(Minv[0, 1], Minv[1, 1])
    margin = min(int(math.ceil(0.5 * (ux + vy))) + 3, 128)

    x0 = int(math.floor(min(px))) - margin
    y0 = int(math.floor(min(py))) - margin
    x1 = int(math.ceil(max(px))) + margin
    y1 = int(math.ceil(max(py))) + margin
    x0 = max(0, min(sw, x0))
    y0 = max(0, min(sh, y0))
    x1 = max(0, min(sw, x1))
    y1 = max(0, min(sh, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    # 整张都在窗口里，裁了也没省，直接返回 None 走原路径（少一次切片）
    if x0 == 0 and y0 == 0 and x1 == sw and y1 == sh:
        return None
    return (x0, y0, x1, y1)


def _crop_matrix(M, x0, y0):
    """源图裁掉左上角 (x0,y0) 之后，对应的仿射矩阵。"""
    M2 = M.copy()
    M2[0, 2] = M[0, 0] * x0 + M[0, 1] * y0 + M[0, 2]
    M2[1, 2] = M[1, 0] * x0 + M[1, 1] * y0 + M[1, 2]
    return M2


def _warp_straight(src, M, dsize, flags):
    """变换 RGBA 位图，返回 (color float32 0..1, alpha float32 0..1)。"""
    a_src = src[..., 3:4].astype(np.float32) / 255.0
    rgb = (src[..., :3].astype(np.float32) / 255.0) * a_src   # 预乘
    a = cv2.warpAffine(a_src, M, dsize, flags=flags,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=(0,))
    pm = cv2.warpAffine(rgb, M, dsize, flags=flags,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    if a.ndim == 2:                      # 单通道输入时 OpenCV 会丢掉通道维
        a = a[..., None]
    a = np.clip(a, 0.0, 1.0)
    c = np.clip(pm / np.maximum(a, 1e-6), 0.0, 1.0)
    return c, a


def _clip(v, lo, hi):
    return max(lo, min(hi, v))


def _warp_layer(layer, doc, org_x, org_y, bw, bh, pad=0):
    """把位图图层变换到缓冲坐标。

    pad: 区域向外扩的像素数（图层样式要往外画投影/发光，不扩会被裁掉）。
    返回 (c, a, lx0, ly0, lx1, ly1)，后四个是相对缓冲左上角的区域；
    图层完全在区域外时返回 None。
    """
    if layer.image is None:
        return None
    box = layer.bbox(doc.width, doc.height)
    if box is None:
        return None
    x0 = int(np.floor(box[0])) - pad
    y0 = int(np.floor(box[1])) - pad
    x1 = int(np.ceil(box[2])) + pad
    y1 = int(np.ceil(box[3])) + pad
    # 与缓冲矩形求交
    x0 = _clip(x0, org_x, org_x + bw)
    y0 = _clip(y0, org_y, org_y + bh)
    x1 = _clip(x1, org_x, org_x + bw)
    y1 = _clip(y1, org_y, org_y + bh)
    if x1 <= x0 or y1 <= y0:
        return None

    M = layer.matrix(ox=x0, oy=y0)
    if M is None:
        return None
    dsize = (x1 - x0, y1 - y0)
    flags = _interp_flags(layer.sx, layer.sy)

    # 只把目标区域需要的那块源像素取出来 —— 大图层渲一小块时这是关键优化
    win = _src_window(layer.image, M, dsize[0], dsize[1])
    if win is None:
        src, Mw = layer.image, M
    else:
        wx0, wy0, wx1, wy1 = win
        src = layer.image[wy0:wy1, wx0:wx1]
        Mw = _crop_matrix(M, wx0, wy0)
    c, a = _warp_straight(src, Mw, dsize, flags)

    if layer.mask is not None and layer.mask_enabled:
        m = layer.mask
        if m.shape[:2] != layer.image.shape[:2]:
            m = cv2.resize(m, (layer.image.shape[1], layer.image.shape[0]),
                           interpolation=cv2.INTER_AREA)
        if win is not None:
            m = m[wy0:wy1, wx0:wx1]
        mw = cv2.warpAffine(m.astype(np.float32) / 255.0, Mw, dsize,
                            flags=flags, borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(0,))
        a = a * np.clip(mw, 0.0, 1.0)[..., None]

    return c, a, x0 - org_x, y0 - org_y, x1 - org_x, y1 - org_y


def _canvas_mask(mask, doc, org_x, org_y, bw, bh):
    """调整层的蒙版是画布尺寸的，取出缓冲对应的那一块，返回 (bh,bw,1) 0~1。"""
    if mask.shape[:2] != (doc.height, doc.width):
        m = cv2.resize(mask, (doc.width, doc.height),
                       interpolation=cv2.INTER_AREA)
    else:
        m = mask
    sub = m[org_y:org_y + bh, org_x:org_x + bw]
    if sub.shape[:2] != (bh, bw):
        pad = np.zeros((bh, bw), np.uint8)
        pad[:sub.shape[0], :sub.shape[1]] = sub
        sub = pad
    return (sub.astype(np.float32) / 255.0)[..., None]


def _apply_adjust(acc_c, acc_a, layer, doc, org_x, org_y):
    """把一个调整层作用到 acc 上（原地修改 acc_c）。

    alpha 不变 —— 调整层只改颜色。不透明度与蒙版决定调整的强度，
    混合模式决定调整结果如何与原有颜色混合（Normal 就是直接替换）。
    """
    key = layer.adjustment_key()
    if not key:
        return
    new_c = apply_adjustment(key, acc_c, (layer.adjust or {}).get("params"))

    w = float(layer.opacity)
    if layer.mask is not None and layer.mask_enabled:
        w = w * _canvas_mask(layer.mask, doc, org_x, org_y,
                             acc_c.shape[1], acc_c.shape[0])

    mode = "Normal" if layer.blend == PASS_THROUGH else layer.blend
    if mode == "Normal":
        target = new_c
    else:
        target = blend_colors(acc_c, new_c, mode)
    acc_c[...] = acc_c * (1.0 - w) + target * w


def _composite_base(layer, doc, acc_c, acc_a, org_x, org_y, chain,
                    snap_layer=None, snap=None):
    """剪贴蒙版：基底 + 一串裁剪到它的调整层。

    先把基底渲进独立缓冲（不带它自己的混合模式与不透明度），套完调整，
    再按基底的混合模式与不透明度合成下去 —— 所以调整只影响基底这一层。
    """
    if layer.is_group:
        bh, bw = acc_c.shape[:2]
        tc = np.zeros((bh, bw, 3), np.float32)
        ta = np.zeros((bh, bw, 1), np.float32)
        _composite_list(layer.children, doc, tc, ta, org_x, org_y,
                        snap_layer, snap)
        for adj in chain:
            _apply_adjust(tc, ta, adj, doc, org_x, org_y)
        ta = ta * float(layer.opacity)
        mode = "Normal" if layer.blend == PASS_THROUGH else layer.blend
        composite(acc_c, acc_a, tc, ta, mode)
        return

    got = _warp_layer(layer, doc, org_x, org_y, acc_c.shape[1], acc_c.shape[0],
                      _layer_pad(layer))
    if got is None:
        return
    c, a, lx0, ly0, lx1, ly1 = got
    if layer.effects:
        c, a = apply_effects(c, a, layer.effects)   # 效果尺寸按画布像素，变换后再加
    h = ly1 - ly0
    w = lx1 - lx0
    tc = np.zeros((h, w, 3), np.float32)
    ta = np.zeros((h, w, 1), np.float32)
    composite(tc, ta, c, a, "Normal")
    for adj in chain:
        _apply_adjust(tc, ta, adj, doc, org_x + lx0, org_y + ly0)
    ta = ta * float(layer.opacity)
    composite(acc_c[ly0:ly1, lx0:lx1], acc_a[ly0:ly1, lx0:lx1], tc, ta,
              "Normal" if layer.blend == PASS_THROUGH else layer.blend)


def _composite_one(layer, doc, acc_c, acc_a, org_x, org_y,
                   snap_layer=None, snap=None):
    """把单个位图 / 组图层按自身的混合模式与不透明度合成进 acc。"""
    bw = acc_c.shape[1]
    bh = acc_c.shape[0]

    if layer.is_group:
        if layer.blend == PASS_THROUGH and layer.opacity >= 1.0:
            # 不隔离：子图层直接与下方背景混合
            _composite_list(layer.children, doc, acc_c, acc_a, org_x, org_y,
                            snap_layer, snap)
            return
        gc = np.zeros((bh, bw, 3), np.float32)
        ga = np.zeros((bh, bw, 1), np.float32)
        _composite_list(layer.children, doc, gc, ga, org_x, org_y,
                        snap_layer, snap)
        ga *= float(layer.opacity)
        mode = "Normal" if layer.blend == PASS_THROUGH else layer.blend
        composite(acc_c, acc_a, gc, ga, mode)
        return

    got = _warp_layer(layer, doc, org_x, org_y, bw, bh, _layer_pad(layer))
    if got is None:
        return
    c, a, lx0, ly0, lx1, ly1 = got
    if layer.effects:
        c, a = apply_effects(c, a, layer.effects)
    a = a * float(layer.opacity)
    composite(acc_c[ly0:ly1, lx0:lx1], acc_a[ly0:ly1, lx0:lx1], c, a,
              layer.blend)


def _composite_list(layers, doc, acc_c, acc_a, org_x, org_y,
                    snap_layer=None, snap=None):
    """把 layers（底->顶）合成进累加缓冲。

    acc_c/acc_a 覆盖画布区域 [org_x, org_x+bw) x [org_y, org_y+bh)。

    snap_layer / snap：给"拖调整层滑块"的快路径服务 —— 走到 snap_layer
    之前时，把当前的 acc 存进 snap（见 render_from_snapshot）。
    """
    n = len(layers)
    i = 0
    while i < n:
        layer = layers[i]
        if not layer.visible or layer.opacity <= 0.0:
            i += 1
            continue

        if layer.is_adjustment:
            if snap is not None and layer is snap_layer:
                _store_snap(snap, acc_c, acc_a)
            _apply_adjust(acc_c, acc_a, layer, doc, org_x, org_y)
            i += 1
            continue

        # 收集紧跟在后面、裁剪到本层的调整层
        chain = []
        j = i + 1
        while j < n and layers[j].is_adjustment and layers[j].clipped:
            if layers[j].visible and layers[j].opacity > 0.0:
                chain.append(layers[j])
            j += 1

        if chain:
            _composite_base(layer, doc, acc_c, acc_a, org_x, org_y, chain,
                            snap_layer, snap)
        else:
            _composite_one(layer, doc, acc_c, acc_a, org_x, org_y,
                           snap_layer, snap)
        i = j


def _write_snap(dst, acc_c, acc_a):
    """把 acc 按 SNAP_DTYPE 写进 dst（(h,w,4)）。"""
    if SNAP_DTYPE == np.float32:
        dst[...] = np.concatenate([acc_c, acc_a], axis=2)
        return
    dst[..., :3] = np.clip(acc_c * 255.0 + 0.5, 0, 255)
    dst[..., 3] = np.clip(acc_a[..., 0] * 255.0 + 0.5, 0, 255)


def _store_snap(snap, acc_c, acc_a):
    """把某个调整层**作用之前**的 acc 存成快照。

    默认存 uint8：12 MP 的 float32 acc 要 192 MB，而 uint8 只要 48 MB
    （和 `_last_arr` 一个量级）。代价是暗部最多几个灰阶的舍入 —— 深色像素
    经大 gamma 放大后个别点能差到 5/255（实测 43 k 像素里 2 个）。
    所以快照只用于拖动期间的实时预览，松手后的那次整幅重算是精确的。

    `SNAP_DTYPE` 换成 float32 可以存成无损的（测试拿它验"逻辑本身没错"）。

    分块渲染（`render_tiled`）走 `dst` / `box` 两支：瓦片只有 tile² 那么大，
    快照得按瓦片核心区**分批写进**整幅数组，不能每块都新建一份。
    """
    dst = snap.get("dst")
    if dst is not None:
        cy, cx, ch, cw = snap["box"]
        _write_snap(dst, acc_c[cy:cy + ch, cx:cx + cw],
                    acc_a[cy:cy + ch, cx:cx + cw])
        return
    if SNAP_DTYPE == np.float32:
        snap["arr"] = np.ascontiguousarray(
            np.concatenate([acc_c, acc_a], axis=2))
        return
    rgb = np.clip(acc_c * 255.0 + 0.5, 0, 255).astype(np.uint8)
    a = np.clip(acc_a * 255.0 + 0.5, 0, 255).astype(np.uint8)
    snap["arr"] = np.ascontiguousarray(np.concatenate([rgb, a], axis=2))


def render_document(doc, region=None, snap_layer=None, snap=None):
    """合成整个文档。

    region: (x0, y0, x1, y1) 画布坐标；None 表示全画布。
    snap_layer / snap: 全幅渲染时顺带把 snap_layer 下方的合成结果存进 snap
    （dict），供 `render_from_snapshot()` 复用 —— 见 §5.23。
    返回 (h, w, 4) uint8 RGBA，直线色。
    """
    if region is None:
        region = (0, 0, doc.width, doc.height)
    sync_smart(doc)                 # 智能对象：把派生的 image 填好
    x0 = _clip(int(region[0]), 0, doc.width)
    y0 = _clip(int(region[1]), 0, doc.height)
    x1 = _clip(int(region[2]), 0, doc.width)
    y1 = _clip(int(region[3]), 0, doc.height)
    w = max(1, x1 - x0)
    h = max(1, y1 - y0)

    # 快照只在**整幅**渲染时捕获：局部渲染的 acc 只覆盖一小块，存下来没法复用
    cap_layer = None
    if snap is not None and snap_layer is not None:
        if x0 == 0 and y0 == 0 and x1 == doc.width and y1 == doc.height:
            cap_layer = snap_layer

    acc_c = np.zeros((h, w, 3), np.float32)
    acc_a = np.zeros((h, w, 1), np.float32)
    _composite_list(doc.layers, doc, acc_c, acc_a, x0, y0,
                    cap_layer, snap if cap_layer is not None else None)

    # 浮动选区画在最上面（它不在 doc.layers 里）
    fl = getattr(doc, "float_layer", None)
    if fl is not None and fl.visible and fl.opacity > 0.0:
        _composite_one(fl, doc, acc_c, acc_a, x0, y0)

    rgb = np.clip(acc_c * 255.0, 0, 255).astype(np.uint8)
    alpha = np.clip(acc_a * 255.0, 0, 255).astype(np.uint8)
    out = np.concatenate([rgb, alpha], axis=2)
    return np.ascontiguousarray(out)


def render_from_snapshot(doc, layer, snap):
    """从"调整层下方的合成结果"出发，只重算它和它上面的图层。

    拖调整层滑块时，它**下方**所有图层的合成结果是不变的 —— 全幅重算里
    最贵的那部分（warp + 混合）可以整个跳掉，只留一次调整运算。

    snap 里没有可用的快照、或这个调整层不在顶层 / 是剪贴蒙版时返回 None，
    调用方退回 `render_document()`。
    """
    if snap is None or layer is None:
        return None
    if not layer.is_adjustment or layer.clipped:
        return None                 # 剪贴蒙版作用于基底的独立缓冲，没有"下方 acc"
    arr = snap.get("arr")
    if arr is None:
        return None
    if snap.get("w") != doc.width or snap.get("h") != doc.height:
        return None
    if arr.shape[0] != doc.height or arr.shape[1] != doc.width:
        return None
    idx = None
    for i, l in enumerate(doc.layers):
        if l is layer:
            idx = i
            break
    if idx is None:
        return None                 # 嵌在组里的调整层：组的隔离缓冲没法复用
    if has_dissolve(doc):
        return None                 # Dissolve 噪声按缓冲尺寸生成

    sync_smart(doc)
    if arr.dtype == np.float32:
        acc_c = np.ascontiguousarray(arr[..., :3])
        acc_a = np.ascontiguousarray(arr[..., 3:4])
    else:
        acc_c = arr[..., :3].astype(np.float32) / 255.0
        acc_a = arr[..., 3:4].astype(np.float32) / 255.0
    _composite_list(doc.layers[idx:], doc, acc_c, acc_a, 0, 0)

    fl = getattr(doc, "float_layer", None)
    if fl is not None and fl.visible and fl.opacity > 0.0:
        _composite_one(fl, doc, acc_c, acc_a, 0, 0)

    rgb = np.clip(acc_c * 255.0, 0, 255).astype(np.uint8)
    alpha = np.clip(acc_a * 255.0, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(np.concatenate([rgb, alpha], axis=2))


def max_effect_padding(doc):
    """文档里所有图层样式需要的最大外扩像素数。

    局部重渲染时要把区域按这个数向外撑大再算 —— 投影 / 发光是模糊出来的，
    贴着区域边界算会缺卷积上下文，边缘会和整幅渲染的结果对不上。
    """
    pad = 0
    for layer in doc.all_layers():
        if layer.effects:
            pad = max(pad, effect_padding(layer.effects))
    fl = getattr(doc, "float_layer", None)
    if fl is not None and fl.effects:
        pad = max(pad, effect_padding(fl.effects))
    return min(pad, 512)


def has_dissolve(doc):
    """文档里有没有可见图层用 Dissolve 混合模式。

    Dissolve 的噪声是按缓冲尺寸生成的（`blend.dissolve_noise(h, w)`），
    局部渲染时缓冲变小、噪声图案会错位，所以只要有 Dissolve 就必须整幅重算。
    """
    for layer in doc.all_layers():
        if not layer.visible or layer.opacity <= 0.0:
            continue
        if layer.blend == "Dissolve":
            return True
    fl = getattr(doc, "float_layer", None)
    if fl is not None and fl.blend == "Dissolve":
        return True
    return False


# ---------------- 代理渲染（交互期的低分辨率预览） ----------------

_PROXY_CACHE = {}       # (id(arr), nw, nh) -> (缩小后的数组, 原数组的引用)
_PROXY_CACHE_MAX = 32


def clear_proxy_cache():
    """清空代理位图缓存（源图被就地改过时调用）。"""
    _PROXY_CACHE.clear()


def _proxy_image(arr, nw, nh):
    """把位图缩到 (nw,nh)，带缓存。

    代理渲染每帧都要用同一批缩小后的源图，不缓存的话光降采样就比渲染还慢。
    缓存值里存一份原数组引用 —— 只要缓存还在，原数组就不会被回收，
    `id()` 也就不会被别的对象复用。
    """
    if arr is None:
        return None
    h, w = arr.shape[:2]
    nw = max(1, min(w, int(nw)))
    nh = max(1, min(h, int(nh)))
    if nw == w and nh == h:
        return arr
    key = (id(arr), nw, nh)
    hit = _PROXY_CACHE.get(key)
    if hit is not None:
        return hit[0]
    small = cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_AREA)
    if len(_PROXY_CACHE) >= _PROXY_CACHE_MAX:
        _PROXY_CACHE.clear()        # 满了就全清，比维护 LRU 简单得多
    _PROXY_CACHE[key] = (small, arr)
    return small


def _proxy_layer(layer, kx, ky, s):
    """把图层（含整棵子树）按比例缩到代理画布的坐标系里。

    两件事：
    1. tx/ty 按 k 缩放（中心在画布坐标里的位置），sx/sy 按 "k ÷ 源图缩小倍率"
       缩放 —— 这样 `Layer.matrix()` 正好等于原矩阵整体缩小 k 倍。
    2. 源位图与蒙版先降采样到代理分辨率（带缓存）。这是代理渲染能快起来的
       关键：12 MP 的背景层，全分辨率渲染时要把整张图转成 144 MB 的 float32，
       而 1/4 代理只需要 1/16 的像素。
    """
    n = layer.clone(deep=False)
    n.tx = layer.tx * kx
    n.ty = layer.ty * ky

    fx = fy = 1.0
    if layer.image is not None:
        h, w = layer.image.shape[:2]
        # 源图 1 px 在代理画布上占多少 px：比 s 还小就不用再降采样了
        fx = min(1.0, s, max(abs(layer.sx) * kx, 1e-3))
        fy = min(1.0, s, max(abs(layer.sy) * ky, 1e-3))
        small = _proxy_image(layer.image, round(w * fx), round(h * fy))
        if small is not None:
            n.image = small
            fx = small.shape[1] / max(1.0, float(w))
            fy = small.shape[0] / max(1.0, float(h))
    if layer.mask is not None:
        mh, mw = layer.mask.shape[:2]
        n.mask = _proxy_image(layer.mask, round(mw * fx), round(mh * fy))

    n.sx = layer.sx * kx / max(fx, 1e-6)
    n.sy = layer.sy * ky / max(fy, 1e-6)
    if layer.effects:
        n.effects = scale_effects(layer.effects, s)
    return n


def make_proxy_doc(doc, scale):
    """造一个"整体缩小"的临时文档，用于交互期快速预览。

    返回的 Document 与 doc 共享所有 numpy 数组（只读渲染，不写回），
    图层对象是浅克隆，改它不会影响原文档。
    调整层的蒙版是画布尺寸的，`_canvas_mask()` 会自动把它缩到代理画布尺寸。
    """
    s = max(0.05, min(1.0, float(scale)))
    sync_smart(doc)                 # 智能对象：代理同样要用派生出来的 image
    pw = max(1, int(round(doc.width * s)))
    ph = max(1, int(round(doc.height * s)))
    kx = pw / max(1.0, float(doc.width))
    ky = ph / max(1.0, float(doc.height))
    p = Document(pw, ph, doc.name)
    p.dpi = doc.dpi
    p.layers = [_proxy_layer(l, kx, ky, s) for l in doc.layers]
    fl = getattr(doc, "float_layer", None)
    if fl is not None:
        p.float_layer = _proxy_layer(fl, kx, ky, s)
    return p


def render_proxy(doc, scale):
    """低分辨率渲染一张代理图。返回 (arr, kx, ky)。

    arr 是 (ph,pw,4) uint8，kx/ky 是它相对原画布的实际缩放比例
    （canvas 尺寸 × k = 代理图尺寸），调用方据此把位图放大回去显示。
    """
    p = make_proxy_doc(doc, scale)
    arr = render_document(p)
    kx = p.width / max(1.0, float(doc.width))
    ky = p.height / max(1.0, float(doc.height))
    return arr, kx, ky


# ---------------- 分块渲染（大画布省峰值内存） ----------------

TILE_DEFAULT = 512       # 瓦片边长（画布像素）
TILE_MIN = 128           # 再小下去，瓦片间的重复计算就压过省下的内存了


def render_region_tiled(doc, region, out=None, tile=None,
                        snap_layer=None, snap=None):
    """把 region 分块渲进 out（out 为 None 时新建一块 (h,w,4) uint8）。

    结果和 `render_document(doc, region=...)` 一致，但**峰值内存只和瓦片
    大小有关**：12 MP 的画布不再一次性开 144 MB 的 float32 acc，而是所有
    瓦片复用同一块 tile² 的缓冲（acc / 组隔离缓冲 / 变换缓冲全都 tile²）。

    注意两点：
      * 每块都按 `max_effect_padding()` 向外扩 pad 再算，算完只取核心区 ——
        投影 / 外发光是模糊出来的，贴着瓦片边界算会缺卷积上下文，接缝处
        就会和整幅渲染对不上。这和脏区重渲染用的是同一个套路。
      * 有 Dissolve 时返回 None（它的噪声按缓冲尺寸生成，分块会错位），
        调用方要退回 `render_document()`。
    """
    if has_dissolve(doc):
        return None
    sync_smart(doc)
    x0 = _clip(int(region[0]), 0, doc.width)
    y0 = _clip(int(region[1]), 0, doc.height)
    x1 = _clip(int(region[2]), 0, doc.width)
    y1 = _clip(int(region[3]), 0, doc.height)
    w = max(1, x1 - x0)
    h = max(1, y1 - y0)
    if out is None:
        out = np.empty((h, w, 4), np.uint8)
    elif out.shape[0] < h or out.shape[1] < w:
        raise ValueError("out 放不下 region：%r vs %dx%d"
                         % (out.shape, w, h))

    t = max(TILE_MIN, int(tile or TILE_DEFAULT))
    if t >= max(w, h) and snap_layer is None:
        # 区域还没一块瓦片大，分了也没意义
        out[...] = render_document(doc, region=(x0, y0, x1, y1))
        return out

    pad = max_effect_padding(doc)
    snap_arr = None
    if snap is not None and snap_layer is not None:
        snap_arr = np.empty((h, w, 4), SNAP_DTYPE)

    fl = getattr(doc, "float_layer", None)
    show_fl = fl is not None and fl.visible and fl.opacity > 0.0

    # 瓦片缓冲按尺寸缓存复用：绝大多数瓦片尺寸一样，只有贴边的那几块不同，
    # 所以实际只需要 2~3 份缓冲，却能让所有中间数组都只有 tile² 那么大。
    bufs = {}

    def _buf(bh, bw):
        v = bufs.get((bh, bw))
        if v is None:
            v = (np.zeros((bh, bw, 3), np.float32),
                 np.zeros((bh, bw, 1), np.float32))
            bufs[(bh, bw)] = v
        v[0][...] = 0.0
        v[1][...] = 0.0
        return v

    for ty in range(0, h, t):
        ch = min(t, h - ty)
        for tx in range(0, w, t):
            cw = min(t, w - tx)
            # 向外扩 pad 再算，但**夹在 region 里面**：整块渲染时缓冲区就是
            # region 那么大，扩出去的部分它根本没算，分块也跟着不算，两边
            # 才对得上（否则瓦片会额外吃到 region 外的上下文，边缘反而不同）。
            # 另外 org 绝不能为负 —— `_canvas_mask()` 用
            # `m[org_y:org_y+bh, org_x:org_x+bw]` 取蒙版，负索引会被 numpy
            # 当成"从末尾数"，直接取错。
            ex = max(x0, x0 + tx - pad)
            ey = max(y0, y0 + ty - pad)
            eex = min(x1, x0 + tx + cw + pad)
            eey = min(y1, y0 + ty + ch + pad)
            cx = x0 + tx - ex           # 核心区在缓冲里的偏移
            cy = y0 + ty - ey
            acc_c, acc_a = _buf(eey - ey, eex - ex)
            if snap_arr is not None:
                snap["dst"] = snap_arr[ty:ty + ch, tx:tx + cw]
                snap["box"] = (cy, cx, ch, cw)
            _composite_list(doc.layers, doc, acc_c, acc_a, ex, ey,
                            snap_layer, snap)
            if show_fl:
                _composite_one(fl, doc, acc_c, acc_a, ex, ey)
            dst = out[ty:ty + ch, tx:tx + cw]
            dst[..., :3] = np.clip(
                acc_c[cy:cy + ch, cx:cx + cw] * 255.0, 0, 255)
            dst[..., 3] = np.clip(
                acc_a[cy:cy + ch, cx:cx + cw, 0] * 255.0, 0, 255)

    if snap_arr is not None:
        snap.pop("dst", None)
        snap.pop("box", None)
        snap["arr"] = snap_arr
    return out


def render_tiled(doc, tile=None, snap_layer=None, snap=None):
    """分块渲染整幅画布，返回 (h,w,4) uint8；有 Dissolve 时返回 None。"""
    return render_region_tiled(doc, (0, 0, doc.width, doc.height),
                               tile=tile, snap_layer=snap_layer, snap=snap)


def render_layer_thumb(layer, size=40):
    """图层缩略图（不应用变换，只显示源图内容），返回 (n,n,4) uint8。"""
    if layer.is_group:
        return None
    if layer.image is None:
        return None
    img = layer.image
    h, w = img.shape[:2]
    s = max(h, w)
    if s == 0:
        return None
    n = size
    if s <= n:
        canvas = np.zeros((n, n, 4), np.uint8)
        y = (n - h) // 2
        x = (n - w) // 2
        canvas[y:y + h, x:x + w] = img
        return canvas
    scale = n / s
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((n, n, 4), np.uint8)
    y = (n - nh) // 2
    x = (n - nw) // 2
    canvas[y:y + nh, x:x + nw] = small
    return canvas
