# -*- coding: utf-8 -*-
"""智能对象（Smart Object）。

一个智能对象图层自己**不存 pixel**，它只引用一段「嵌入内容」
（`SmartContent`：一份独立的画布尺寸 + 图层树）。渲染时才把这段内容合成成
一张位图，再当成普通位图图层参与合成。

由此得到三件普通图层做不到的事：

1. **内容可独立编辑** —— 改内容，所有引用它的实例一起更新
2. **多实例共享** —— 若干图层指向同一个 content id，编辑一次全部联动
3. **智能滤镜** —— 滤镜**参数**存在图层上，每次渲染重新施加一遍，
   所以随时能改参数、调顺序、删掉，不像普通滤镜那样写死在像素里

和普通图层一样，变换也是非破坏性的（位图始终按内容原始分辨率生成，
只在参与父文档合成时做仿射变换），所以缩小再放大不丢细节。

## 渲染接入

`layer.image` 是**派生**的：`render.render_document()` 与
`render.make_proxy_doc()` 在开工前会调用 `sync_smart(doc)`，
把每个智能对象图层的 `image` 填成「内容合成结果 + 智能滤镜」。
填好之后渲染管线完全不用知道智能对象的存在 —— 变换、蒙版、图层样式、
混合模式全部照旧。

## 缓存

`layer.__dict__["_so_key"] = (content.id, content.rev, 滤镜签名)`。
`SmartContent.rev` 是内容修订号，任何内容改动（编辑器回写）都要 `touch()`。
签名变了或者 key 不匹配就重算。

**注意**：`Layer.clone()` 不复制 `_so_key`，所以撤销/重做之后一定重算，
宁可慢也不能拿旧栅格顶替。

## 写时复制

历史的坑和普通图层一样：内容里的子图层数组是**共享**的
（`Document.clone()` 会连带浅克隆 `smart_contents`）。
在内容里做任何像素级修改之前，必须对那个子图层 `detach_pixels()`。
"""

from __future__ import annotations

import json
import math
import uuid

import numpy as np

from .filters import apply_filter_array
from .layer import LAYER_ADJUSTMENT, LAYER_SMART, Layer


class SmartContent:
    """嵌入的内容：一份独立的小画布 + 图层树（底 -> 顶）。"""

    def __init__(self, name="智能对象", width=1, height=1, layers=None,
                 cid=None):
        self.id = cid or uuid.uuid4().hex
        self.name = name
        self.width = int(max(1, width))
        self.height = int(max(1, height))
        self.layers = list(layers or [])
        self.rev = 1                 # 内容修订号；改动内容必须 touch()

    # ---------- 修订 ----------

    def touch(self):
        """内容被改动了：修订号 +1，所有实例的栅格缓存作废。"""
        self.rev += 1

    # ---------- 和 Layer 同构的小工具 ----------

    @property
    def is_group(self):
        return False

    def all_layers(self):
        for l in self.layers:
            yield from l.walk()

    def find(self, lid):
        for l in self.all_layers():
            if l.id == lid:
                return l
        return None

    # ---------- 复制 / 序列化 ----------

    def clone(self):
        """浅克隆：content 对象与子图层对象新建，numpy 数组共享。

        和 `Document.clone()` 同构 —— 撤销栈靠这个既省内存又保证隔离，
        前提是像素级改动之前记得 `detach_pixels()`。
        """
        c = SmartContent(self.name, self.width, self.height,
                         [l.clone() for l in self.layers], cid=self.id)
        c.rev = self.rev
        return c

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "rev": self.rev,
            "layers": [l.to_dict() for l in self.layers],
        }

    @staticmethod
    def from_dict(d, images):
        layers = [Layer.from_dict(ld, images)
                  for ld in d.get("layers", [])]
        c = SmartContent(d.get("name", "智能对象"),
                         d.get("width", 1), d.get("height", 1),
                         layers, cid=d.get("id"))
        c.rev = int(d.get("rev", 1) or 1)
        return c


# ---------------------------------------------------------------- 栅格

_RENDERING = set()      # 正在合成中的 content id —— 放自引用死循环


def _signature(filters):
    """智能滤镜列表的签名。参数和顺序变了都要重算。"""
    if not filters:
        return ""
    try:
        return json.dumps(filters, sort_keys=True, ensure_ascii=False,
                          default=str)
    except Exception:
        return repr(filters)


def _content_key(content, layer):
    return (content.id, content.rev, _signature(layer.so_filters))


def smart_raster(content, filters=None, contents=None, busy=None):
    """把内容合成为一张 RGBA 位图，按顺序施加智能滤镜。

    filters 里每一项是 {"key", "params", "strength", "enabled"}。
    返回新的 (h,w,4) uint8 数组（不复用任何共享缓冲）。
    """
    busy = _RENDERING if busy is None else busy
    if content.id in busy:
        # 自引用（内容里套了自己）：退回空白，别把栈撑爆
        return np.zeros((content.height, content.width, 4), np.uint8)

    from .document import Document
    from .render import render_document

    busy.add(content.id)
    try:
        d = Document(content.width, content.height, content.name)
        d.smart_contents = contents if contents is not None else {}
        d.layers = content.layers
        arr = render_document(d)
    finally:
        busy.discard(content.id)

    for f in (filters or []):
        if not f or not f.get("enabled", True):
            continue
        key = f.get("key")
        if not key:
            continue
        # 不传选区：智能滤镜作用于整个内容，选区语义由蒙版负责
        arr = apply_filter_array(arr, key, f.get("params"), None,
                                 float(f.get("strength", 1.0)))
    return np.ascontiguousarray(arr)


def _sync_layer(layer, contents, busy):
    if layer.kind != LAYER_SMART:
        return
    cid = layer.so_id
    content = contents.get(cid) if cid else None
    if content is None:
        if layer.image is not None:
            layer.image = None
        layer.__dict__.pop("_so_key", None)
        return
    # 内容尺寸先记在图层上：栅格没生成时，属性面板 / 包围盒 / 手柄要靠它
    layer.so_size = (content.width, content.height)
    key = _content_key(content, layer)
    if layer.image is not None and layer.__dict__.get("_so_key") == key:
        return
    layer.image = smart_raster(content, layer.so_filters, contents, busy)
    layer.__dict__["_so_key"] = key


def sync_smart(doc):
    """渲染前把所有智能对象图层的 `image` 填好。

    没有智能对象时几乎是空转（一次 getattr + len 判断），每帧调也没关系。
    """
    contents = getattr(doc, "smart_contents", None)
    if not contents:
        return
    busy = _RENDERING
    try:
        for layer in doc.all_layers():
            _sync_layer(layer, contents, busy)
        fl = getattr(doc, "float_layer", None)
        if fl is not None:
            _sync_layer(fl, contents, busy)
    finally:
        busy.clear()


def refresh_sizes(doc):
    """只把内容尺寸同步到图层上（不渲栅格）。读工程之后调一次即可。"""
    contents = getattr(doc, "smart_contents", None)
    if not contents:
        return
    for l in doc.all_layers():
        if l.kind == LAYER_SMART and l.so_id:
            c = contents.get(l.so_id)
            if c is not None:
                l.so_size = (c.width, c.height)


def ensure_smart_image(doc, layer):
    """取某个智能对象图层当前的栅格（必要时才重算）。"""
    contents = getattr(doc, "smart_contents", None) or {}
    _sync_layer(layer, contents, set())
    return layer.image


# ---------------------------------------------------------------- 转换

def raw_bbox(layer):
    """未经画布裁剪的包围盒；组取子树并集。返回 (x0,y0,x1,y1) 浮点。"""
    if layer.is_group:
        boxes = [raw_bbox(c) for c in layer.children if c.visible]
        boxes = [b for b in boxes if b is not None]
        if not boxes:
            return None
        return (min(b[0] for b in boxes), min(b[1] for b in boxes),
                max(b[2] for b in boxes), max(b[3] for b in boxes))
    return layer.bbox()


def _subtree_pad(layer):
    """子树里图层样式需要的外扩像素数（投影/发光不能贴边裁掉）。"""
    from .effects import effect_padding
    pad = 0
    for l in layer.walk():
        if l.effects:
            pad = max(pad, effect_padding(l.effects))
    return min(pad, 512)


def _crop_mask(m, x0, y0, w, h):
    """把画布尺寸的蒙版裁到内容画布上，落在范围外的地方补 0（全遮）。

    x0/y0 可能是小数（内容画布是按中心对齐撑出来的），这里四舍五入到整像素 ——
    蒙版本来就是按整像素采样、没有插值，半像素的偏差不会影响外观。
    """
    out = np.zeros((h, w), np.uint8)
    mh, mw = m.shape[:2]
    sx0, sy0 = int(round(x0)), int(round(y0))
    ix0 = max(0, sx0)
    iy0 = max(0, sy0)
    ix1 = min(mw, sx0 + w)
    iy1 = min(mh, sy0 + h)
    if ix1 > ix0 and iy1 > iy0:
        out[iy0 - sy0:iy1 - sy0, ix0 - sx0:ix1 - sx0] = m[iy0:iy1, ix0:ix1]
    return out


def convert_to_smart(doc, lid, standalone_content=True):
    """把一个图层（或图层组）连同它上面的剪贴蒙版调整层一起转成智能对象。

    返回新建的智能对象图层；不能转时返回 None。
    原图层子树被搬进内容里（按包围盒平移），混合模式/蒙版/样式都跟着走，
    外层只留一张 Normal / 100% 的壳。
    """
    layer, lst, idx = doc.find_with_parent(lid)
    if layer is None or lst is None:
        return None
    if layer.is_adjustment or layer.kind == LAYER_SMART:
        return None
    box = raw_bbox(layer)
    if box is None:
        return None

    pad = _subtree_pad(layer)
    bw = box[2] - box[0]
    bh = box[3] - box[1]
    cw = max(1, int(math.ceil(bw + 2.0 * pad)))
    ch = max(1, int(math.ceil(bh + 2.0 * pad)))
    # 内容画布左上角对齐到**整像素**：之后外层壳把它贴回画布时是整数平移，
    # cv2 的双线性插值退化成原样拷贝，不会再做一遍重采样把边缘揉糊。
    # 代价是整块内容最多偏移半个像素 —— 比二次重采样划算得多。
    x0 = float(math.floor((box[0] + box[2]) / 2.0 - cw / 2.0))
    y0 = float(math.floor((box[1] + box[3]) / 2.0 - ch / 2.0))
    ctr_x = x0 + cw / 2.0
    ctr_y = y0 + ch / 2.0

    # 收集紧跟在上面、裁剪到它的调整层 —— 它们只影响这一层，得一起搬进去
    moved = [layer]
    j = idx + 1
    while j < len(lst) and lst[j].kind == LAYER_ADJUSTMENT and lst[j].clipped:
        moved.append(lst[j])
        j += 1
    for l in moved:
        lst.remove(l)

    for l in moved:
        for sub in l.walk():
            sub.tx -= x0
            sub.ty -= y0
        if l.kind == LAYER_ADJUSTMENT:
            # 调整层没有自己的几何，摆到内容正中即可；蒙版是画布尺寸的，要裁
            l.tx = cw / 2.0
            l.ty = ch / 2.0
            if l.mask is not None:
                l.mask = _crop_mask(l.mask, x0, y0, cw, ch)

    cid = uuid.uuid4().hex
    content = SmartContent(layer.name, cw, ch, moved, cid=cid)
    doc.setdefault_smart_contents()[cid] = content

    shell = Layer(layer.name, LAYER_SMART)
    shell.blend = "Normal"
    shell.opacity = 1.0
    shell.so_id = cid
    shell.so_filters = []
    shell.so_size = (cw, ch)
    shell.tx = ctr_x
    shell.ty = ctr_y
    if standalone_content:
        shell.name = layer.name
    lst.insert(idx, shell)
    return shell


def new_smart_instance(doc, lid):
    """复制一个智能对象图层，但**共用同一份内容**（PS 的「通过拷贝新建」）。

    改内容会联动所有实例；各自的变换/蒙版/滤镜是独立的。
    """
    layer = doc.find(lid)
    if layer is None or layer.kind != LAYER_SMART:
        return None
    n = layer.clone()
    n.id = uuid.uuid4().hex
    n.name = layer.name + " 副本"
    n.so_filters = [dict(f) for f in (layer.so_filters or [])]
    n.so_size = layer.so_size
    # 栅格缓存不能共用：滤镜不同就要重算，而 _so_key 里已经带了滤镜签名
    n.__dict__.pop("_so_key", None)
    lst, idx = doc.parent_list(layer.id)
    if lst is None:
        doc.layers.append(n)
    else:
        lst.insert(idx + 1, n)
    return n


def content_instances(doc, cid):
    """引用这份内容的所有智能对象图层（实例可能不止一个）。"""
    out = []
    for l in doc.all_layers():
        if l.kind == LAYER_SMART and l.so_id == cid:
            out.append(l)
    fl = getattr(doc, "float_layer", None)
    if fl is not None and fl.kind == LAYER_SMART and fl.so_id == cid:
        out.append(fl)
    return out


def rasterize_smart(doc, layer):
    """把智能对象烧成普通位图图层。

    栅格按内容原始分辨率生成，变换不动，所以画面看不出变化；
    代价是内容与智能滤镜从此不可再改。
    """
    if layer is None or layer.kind != LAYER_SMART:
        return False
    cid = layer.so_id
    arr = ensure_smart_image(doc, layer)
    if arr is None:
        return False
    # image 可能被别处共享（比如另一个实例刚好同 key），这里必须独占一份
    layer.image = arr.copy()
    layer.kind = "image"
    layer.so_id = None
    layer.so_filters = None
    layer.so_size = None
    layer.__dict__.pop("_so_key", None)
    _drop_unused_content(doc, cid)
    return True


def _drop_unused_content(doc, cid):
    """没人引用了就把内容回收（智能对象栅格化 / 图层删除之后）。"""
    if not cid:
        return
    contents = getattr(doc, "smart_contents", None)
    if not contents:
        return
    if content_instances(doc, cid):
        return
    contents.pop(cid, None)


def prune_contents(doc):
    """扫一遍，清掉已经没有任何实例引用的内容。返回清理条数。"""
    contents = getattr(doc, "smart_contents", None)
    if not contents:
        return 0
    used = set()
    for l in doc.all_layers():
        if l.kind == LAYER_SMART and l.so_id:
            used.add(l.so_id)
    fl = getattr(doc, "float_layer", None)
    if fl is not None and fl.kind == LAYER_SMART and fl.so_id:
        used.add(fl.so_id)
    dead = [k for k in contents if k not in used]
    for k in dead:
        contents.pop(k, None)
    return len(dead)


def new_filter(key, params=None, strength=1.0, enabled=True):
    """一条智能滤镜记录。"""
    return {
        "key": key,
        "params": dict(params or {}),
        "strength": float(strength),
        "enabled": bool(enabled),
    }
