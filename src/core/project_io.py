# -*- coding: utf-8 -*-
"""工程文件读写与导入导出。

.cwproj 是一个 zip 包：
    manifest.json            工程结构与所有图层参数
    layers/<id>.png          各图层原始分辨率位图（RGBA）
    masks/<id>.png           各图层蒙版（灰度）
    manifest.json#smart      智能对象的嵌入内容（尺寸 + 图层树 + 修订号）

位图永远存原始分辨率，变换只存参数，所以是非破坏性的。
智能对象图层**不存**栅格（那张位图是内容渲染出来的派生数据），
存的是内容本身的图层树 —— 读回来之后由 core.smart 重新合成。
"""

from __future__ import annotations

import io
import json
import os
import zipfile

import numpy as np
from PIL import Image

from .layer import DERIVED_KINDS

PROJECT_EXT = ".cwproj"


SVG_EXT = ".svg"
HEIF_EXT = (".heic", ".heif")

_HEIF_OK = None


def _require_heif():
    """HEIC / HEIF 交给 pillow-heif 解码，注册一次即全局生效。"""
    global _HEIF_OK
    if _HEIF_OK is None:
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
            _HEIF_OK = True
        except Exception:
            _HEIF_OK = False
    if not _HEIF_OK:
        raise RuntimeError(
            "读取 HEIC 需要 pillow-heif：\n\n"
            "  pip install pillow-heif "
            "-i https://mirrors.aliyun.com/pypi/simple\n\n"
            "装完重启程序即可。")
    return True


def _to_rgba(im):
    """PIL 图 → RGBA。HEIC 常见 I;16 / CMYK / 灰度，直接 convert 会崩。"""
    if im.mode == "RGBA":
        return im
    arr = np.array(im)
    if arr.dtype == np.uint8:
        return im.convert("RGBA")
    if arr.dtype == np.uint16:                     # 16-bit（HEIC / TIFF）
        arr = (arr >> 8).astype(np.uint8)
    else:                                          # I / F 这类浮点整数模式
        arr = np.clip(np.nan_to_num(arr.astype(np.float32)) * 255.0,
                      0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr, "L").convert("RGBA")
    if arr.shape[2] == 4:
        return Image.fromarray(arr, "RGBA")
    return Image.fromarray(arr[:, :, :3], "RGB").convert("RGBA")


def imread_rgba(path, target=None):
    """读取图片为 (h,w,4) uint8 RGBA。支持中文路径。

    `target=(w, h)` 是**输出尺寸**，只对 SVG 有意义 —— 矢量图没有固有分辨率，
    target 缺省时按 SVG 自身的 width/height（没有就按 viewBox 比例）。位图
    永远按原始分辨率读，target 会被忽略。
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == SVG_EXT:
        from .svg_import import render_svg, svg_size
        w = h = None
        if target:
            w, h = target[0], target[1]
        if not w or not h:
            sw, sh = svg_size(path)
            w = w or sw
            h = h or sh
        return render_svg(path, w, h)
    if ext in HEIF_EXT:
        _require_heif()
    with open(path, "rb") as f:
        data = f.read()
    im = Image.open(io.BytesIO(data))
    im = _apply_exif_orientation(im)
    im = _to_rgba(im)
    return np.array(im)


def _apply_exif_orientation(im):
    try:
        exif = im.getexif()
        code = exif.get(274, 1)
    except Exception:
        return im
    mapping = {3: 180, 6: 270, 8: 90}
    deg = mapping.get(code)
    if deg:
        im = im.rotate(deg, expand=True)
    return im


def imwrite(path, arr, quality=95):
    """写出图片。arr 为 (h,w,4) 或 (h,w,3) uint8。支持中文路径。"""
    ext = os.path.splitext(path)[1].lower()
    if arr.shape[2] == 4 and ext in (".jpg", ".jpeg"):
        im = Image.fromarray(arr, "RGBA").convert("RGB")
    else:
        im = Image.fromarray(arr, "RGBA" if arr.shape[2] == 4 else "RGB")
    if ext in (".jpg", ".jpeg"):
        im.save(path, quality=quality, subsampling=0)
    else:
        im.save(path)
    return path


def _png_bytes(arr):
    if arr.ndim == 2:
        mode = "L"
    else:
        mode = "RGBA" if arr.shape[2] == 4 else "RGB"
    im = Image.fromarray(arr, mode)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def save_project(doc, path):
    """保存工程。返回实际写入路径。"""
    images = {}
    manifest = doc.to_dict()

    def walk(layers):
        for l in layers:
            # 智能对象 / 矢量形状的栅格是派生出来的，绝不写进工程
            # （体积大且必然过期；读回来由 core.smart / core.shape 重算）
            if l.kind in DERIVED_KINDS:
                l.__dict__.pop("_img_key", None)
            elif l.image is not None:
                key = "layers/%s.png" % l.id
                images[key] = l.image
                l.__dict__["_img_key"] = key
            if l.mask is not None:
                key = "masks/%s.png" % l.id
                images[key] = l.mask
                l.__dict__["_mask_key"] = key
            if l.is_group:
                walk(l.children)

    # 先把 key 写进 dict
    def to_dict_with_keys(l):
        d = l.to_dict()
        if l.__dict__.get("_img_key"):
            d["image"] = l.__dict__["_img_key"]
        if l.__dict__.get("_mask_key"):
            d["mask"] = l.__dict__["_mask_key"]
        if l.is_group:
            d["children"] = [to_dict_with_keys(c) for c in l.children]
        return d

    walk(doc.layers)
    manifest["layers"] = [to_dict_with_keys(l) for l in doc.layers]
    if doc.float_layer is not None:
        walk([doc.float_layer])
        manifest["float"] = to_dict_with_keys(doc.float_layer)

    # 智能对象的嵌入内容：内容的图层和顶层图层共用同一个 "layers/<id>.png"
    # 命名空间（图层 id 本身就是全局唯一的），不用另起一套
    contents = doc.setdefault_smart_contents()
    if contents:
        manifest["smart"] = {}
        for cid, c in contents.items():
            walk(c.layers)
            manifest["smart"][cid] = dict(c.to_dict(),
                                          layers=[to_dict_with_keys(l)
                                                  for l in c.layers])

    # 通道面板的 Alpha 通道遮罩：和图层蒙版一样存成单通道 PNG。
    # 注意 key 要在 to_dict 之前写好 —— 通道的 dict 里带的是这个相对路径。
    if getattr(doc, "channels", None):
        chd = dict(manifest.get("channels") or {})
        entries = []
        for c in doc.channels:
            key = "channels/%s.png" % c.id
            images[key] = c.mask
            entries.append({"id": c.id, "key": key})
        chd["masks"] = entries
        manifest["channels"] = chd
    # "合并的 Alpha 通道"也是一张遮罩，一起存
    ca = getattr(doc, "composite_alpha", None)
    if ca is not None:
        key = "channels/_composite.png"
        images[key] = ca
        chd = dict(manifest.get("channels") or {})
        chd["composite"] = key
        manifest["channels"] = chd

    tmp = path + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json",
                   json.dumps(manifest, ensure_ascii=False, indent=2))
        for key, arr in images.items():
            z.writestr(key, _png_bytes(np.ascontiguousarray(arr)))
    if os.path.exists(path):
        os.remove(path)
    os.replace(tmp, path)
    doc.path = path
    return path


def load_project(path):
    """读取工程，返回 Document。"""
    images = {}
    with zipfile.ZipFile(path, "r") as z:
        manifest = json.loads(z.read("manifest.json").decode("utf-8"))
        for name in z.namelist():
            if name == "manifest.json" or name.endswith("/"):
                continue
            data = z.read(name)
            im = Image.open(io.BytesIO(data))
            images[name] = np.array(im)
    from .document import Document
    doc = Document.from_dict(manifest, images)
    # from_dict 已经按 id 把通道遮罩换回数组了；这里只补"合并的 Alpha 通道"
    # （它不是 Channel 对象，所以不走 channels.from_dict）
    ckey = (manifest.get("channels") or {}).get("composite")
    if ckey and ckey in images:
        arr = np.asarray(images[ckey])
        if arr.ndim == 3:
            arr = arr[..., 3] if arr.shape[2] == 4 else arr[..., 0]
        doc.composite_alpha = np.ascontiguousarray(arr.astype(np.uint8))
    doc.path = path
    return doc


def export_flat(arr, path, quality=95):
    imwrite(path, arr, quality=quality)
    return path
