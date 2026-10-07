# -*- coding: utf-8 -*-
"""工程（文档）模型：画布尺寸 + 图层树。"""

from __future__ import annotations

import cv2

from .adjust import default_params
from .blend import PASS_THROUGH
from .layer import (LAYER_ADJUSTMENT, LAYER_GROUP, LAYER_IMAGE, LAYER_SMART,
                    LAYER_TEXT, Layer)
from .text import default_text_params, render_text


# 通道显示开关的初值：R/G/B/A 全开。放在这里是为了让 `Document.__init__`
# 里能直接赋一个真 dict，调用方不必记得先调 channels.rgb_visibility()。
def _ch_visible_default():
    return {"R": True, "G": True, "B": True, "A": True}


# 通道：写 manifest 的那一小段（显示开关 + 通道列表）。
# 遮罩本体由 project_io 像图层那样存成 PNG，并把 id -> zip 内路径写进
# 这个 dict 的 "masks" 字段，见 core/project_io.py::save_project
def _ch_to_dict(doc):
    from . import channels as ch
    return ch.to_dict(doc)


def _ch_from_dict(doc, d, images):
    from . import channels as ch
    if not d:
        return doc
    masks = {}
    for entry in (d.get("masks") or []):
        img = images.get(entry.get("key"))
        if img is None:
            continue
        import numpy as _np
        a = _np.asarray(img)
        if a.ndim == 3:
            a = a[..., 3] if a.shape[2] == 4 else a[..., 0]
        masks[entry.get("id")] = _np.ascontiguousarray(a.astype(_np.uint8))
    return ch.from_dict(doc, d, masks)


class Document:
    def __init__(self, width=1280, height=800, name="Untitled"):
        self.width = int(width)
        self.height = int(height)
        self.name = name
        self.path = None          # .cwproj 路径
        self.layers = []          # 底 -> 顶
        self.dpi = 72.0
        self.selection = None     # core.selection.Selection，画布坐标
        self.float_layer = None   # 浮动选区（core.float_sel），不在 layers 里
        # 快速蒙版：uint8 (h,w)，255 = 会被选中的区域（语义与显示方式无关）
        # None 表示不在快速蒙版模式。见 core/quick_mask.py
        self.quick_mask = None
        self.quick_mask_mode = "masked"   # "masked" 红盖未选中 / "selected" 红盖选中
        # 智能对象的嵌入内容：{content_id: core.smart.SmartContent}
        # 多个图层可以指向同一个 id —— 那就是"多实例共享"
        self.smart_contents = {}
        # 通道面板（core.channels）：
        #   channels        —— 附加的 Alpha 通道列表（每项一张 uint8 遮罩）
        #   channel_view    —— R/G/B/A 四路的显示开关
        #   composite_alpha —— "合并的 Alpha 通道"（None = 用合成结果本身的 alpha）
        # 这三样都只影响**显示**与选区，不动图层数据（见 core/channels.py 文件头）
        self.channels = []
        self.channel_view = _ch_visible_default()
        self.composite_alpha = None

    # ---------- 结构 ----------

    @property
    def floating(self):
        return self.float_layer is not None

    def setdefault_smart_contents(self):
        """老工程读进来时可能没有这个字段，用到的时候现补。"""
        if getattr(self, "smart_contents", None) is None:
            self.smart_contents = {}
        return self.smart_contents

    def clone(self):
        """浅克隆：图层对象新建，numpy 数组共享。见 Layer.clone 说明。"""
        d = Document(self.width, self.height, self.name)
        d.path = self.path
        d.dpi = self.dpi
        d.layers = [l.clone() for l in self.layers]
        d.selection = self.selection
        d.float_layer = (self.float_layer.clone()
                         if self.float_layer is not None else None)
        # 同 selection：先共享，改之前由 detach_quick_mask() 复制
        d.quick_mask = self.quick_mask
        d.quick_mask_mode = self.quick_mask_mode
        # 通道遮罩同理：numpy 数组共享，改之前 detach_channel()
        d.channels = [c.copy() for c in (getattr(self, "channels", None) or [])]
        v = getattr(self, "channel_view", None)
        d.channel_view = dict(v) if v else None
        ca = getattr(self, "composite_alpha", None)
        d.composite_alpha = (ca.copy() if ca is not None else None)
        # 内容也要克隆：在编辑器里改子图层时，历史快照不能被跟着改
        d.smart_contents = dict(
            (k, c.clone()) for k, c in (self.smart_contents or {}).items())
        return d

    def smart_content(self, cid):
        return (self.smart_contents or {}).get(cid)

    def detach_pixels(self, layer):
        """像素级修改前调用，见 Layer.detach_pixels。"""
        layer.detach_pixels()

    def detach_selection(self):
        """修改选区前调用：换成本文档独占的副本，避免污染撤销栈。"""
        if self.selection is not None:
            self.selection = self.selection.copy()

    def detach_quick_mask(self):
        """修改快速蒙版前调用，理由同 detach_selection。"""
        if self.quick_mask is not None:
            self.quick_mask = self.quick_mask.copy()

    def detach_channel(self, channel):
        """修改某个 Alpha 通道的遮罩前调用，理由同 detach_selection。

        只在**真的要改内容**时调；加 / 删通道、重命名那些改的是列表结构，
        clone 时列表本身已经新建了，不用 detach。
        """
        if channel is not None:
            channel.mask = channel.mask.copy()

    def detach_composite_alpha(self):
        """修改「合并的 Alpha 通道」前调用，理由同上。"""
        if self.composite_alpha is not None:
            self.composite_alpha = self.composite_alpha.copy()

    def resize(self, w, h):
        self.width = int(w)
        self.height = int(h)
        if self.selection is not None:
            self.selection.resize_to(self.width, self.height)
        if self.quick_mask is not None:
            self.quick_mask = cv2.resize(
                self.quick_mask, (self.width, self.height),
                interpolation=cv2.INTER_NEAREST)
        from . import channels as _ch
        _ch.resize_channels(self, self.width, self.height)

    def all_layers(self):
        for l in self.layers:
            yield from l.walk()

    def find(self, lid):
        for l in self.all_layers():
            if l.id == lid:
                return l
        return None

    def find_with_parent(self, lid):
        """返回 (layer, parent_list, index)；顶层图层 parent_list 为 self.layers。"""
        stack = [(self.layers, None)]
        while stack:
            lst, _ = stack.pop()
            for i, l in enumerate(lst):
                if l.id == lid:
                    return l, lst, i
                if l.is_group:
                    stack.append((l.children, l))
        return None, None, -1

    def parent_list(self, lid):
        _, lst, idx = self.find_with_parent(lid)
        return lst, idx

    def remove(self, lid):
        l, lst, idx = self.find_with_parent(lid)
        if l is None:
            return None
        lst.pop(idx)
        return l

    def insert_above(self, lid, layer):
        l, lst, idx = self.find_with_parent(lid)
        if l is None:
            self.layers.append(layer)
        else:
            lst.insert(idx + 1, layer)
        return layer

    def flatten_order(self):
        return list(self.all_layers())

    # ---------- 包围盒 ----------

    def layer_bbox(self, layer):
        """图层（含组）在画布坐标下的包围盒。"""
        if layer.is_group:
            boxes = [self.layer_bbox(c) for c in layer.children if c.visible]
            boxes = [b for b in boxes if b is not None]
            if not boxes:
                return None
            return (min(b[0] for b in boxes), min(b[1] for b in boxes),
                    max(b[2] for b in boxes), max(b[3] for b in boxes))
        return layer.bbox(self.width, self.height)

    def content_bbox(self):
        boxes = [self.layer_bbox(l) for l in self.layers if l.visible]
        boxes = [b for b in boxes if b is not None]
        if not boxes:
            return None
        return (min(b[0] for b in boxes), min(b[1] for b in boxes),
                max(b[2] for b in boxes), max(b[3] for b in boxes))

    def to_dict(self):
        return {
            "format": "compositor-win-project",
            "version": 1,
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "dpi": self.dpi,
            "layers": [l.to_dict() for l in self.layers],
            "smart": dict((k, c.to_dict())
                          for k, c in (self.smart_contents or {}).items()),
            "channels": _ch_to_dict(self),
        }
        return d

    @staticmethod
    def from_dict(d, images):
        doc = Document(d.get("width", 1280), d.get("height", 800),
                       d.get("name", "Untitled"))
        doc.dpi = d.get("dpi", 72.0)
        for ld in d.get("layers", []):
            doc.layers.append(Layer.from_dict(ld, images))
        if d.get("float"):
            doc.float_layer = Layer.from_dict(d["float"], images)
        from .smart import SmartContent, refresh_sizes
        doc.smart_contents = {}
        for cid, cd in (d.get("smart") or {}).items():
            c = SmartContent.from_dict(cd, images)
            doc.smart_contents[c.id or cid] = c
        # 只把内容尺寸填到图层上；栅格留到渲染时按需生成，读大工程不会卡
        refresh_sizes(doc)
        from . import channels as _ch
        _ch_from_dict(doc, d.get("channels"), images or {})
        return doc


def make_smart_layer(name="智能对象", content=None, canvas_w=0, canvas_h=0):
    """新建一个引用已有内容的智能对象图层。

    content 为 None 时会先建一份 1x1 的空内容（调用方自己填充画布尺寸与图层）。
    """
    from .smart import SmartContent
    if content is None:
        content = SmartContent(name, 1, 1, [])
    l = Layer(name, LAYER_SMART, image=None)
    l.blend = "Normal"
    l.so_id = content.id
    l.so_filters = []
    l.so_size = (content.width, content.height)
    l.tx = canvas_w / 2.0
    l.ty = canvas_h / 2.0
    return l


def make_group(name="Group", children=None):
    g = Layer(name, LAYER_GROUP)
    g.blend = PASS_THROUGH
    if children:
        g.children = list(children)
    return g


def make_image_layer(name="Layer", image=None, canvas_w=0, canvas_h=0):
    l = Layer(name, LAYER_IMAGE, image=image)
    l.tx = canvas_w / 2.0
    l.ty = canvas_h / 2.0
    return l


def make_adjustment_layer(name="调整", key="levels", canvas_w=0, canvas_h=0):
    """调整层：没有像素，只有参数。蒙版按画布尺寸建立。"""
    l = Layer(name, LAYER_ADJUSTMENT)
    l.blend = "Normal"
    l.adjust = {"type": key, "params": default_params(key)}
    l.tx = canvas_w / 2.0
    l.ty = canvas_h / 2.0
    return l


def make_text_layer(name="文字", canvas_w=0, canvas_h=0, params=None,
                    center=None):
    """文字层：像素由参数栅格化生成，参数留在 layer.text 里可以再改。

    center 给 (x, y) 时用它作为图层中心，否则放到画布中心。
    """
    p = default_text_params(**(params or {}))
    l = Layer(name, LAYER_TEXT, image=render_text(p))
    l.blend = "Normal"
    l.text = p
    if center is None:
        l.tx = canvas_w / 2.0
        l.ty = canvas_h / 2.0
    else:
        l.tx = float(center[0])
        l.ty = float(center[1])
    return l
