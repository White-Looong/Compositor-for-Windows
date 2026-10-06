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

from .layer import LAYER_SMART

PROJECT_EXT = ".cwproj"


def imread_rgba(path):
    """读取图片为 (h,w,4) uint8 RGBA。支持中文路径。"""
    with open(path, "rb") as f:
        data = f.read()
    im = Image.open(io.BytesIO(data))
    im = _apply_exif_orientation(im)
    if im.mode != "RGBA":
        im = im.convert("RGBA")
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
            # 智能对象的栅格是派生出来的，绝不写进工程（体积大且必然过期）
            if l.kind == LAYER_SMART:
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
    doc.path = path
    return doc


def export_flat(arr, path, quality=95):
    imwrite(path, arr, quality=quality)
    return path
