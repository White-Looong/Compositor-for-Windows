# -*- coding: utf-8 -*-
"""工程（文档）模型：画布尺寸 + 图层树。"""

from __future__ import annotations

from .adjust import default_params
from .blend import PASS_THROUGH
from .layer import (LAYER_ADJUSTMENT, LAYER_GROUP, LAYER_IMAGE, LAYER_SMART,
                    LAYER_TEXT, Layer)
from .text import default_text_params, render_text


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
        # 智能对象的嵌入内容：{content_id: core.smart.SmartContent}
        # 多个图层可以指向同一个 id —— 那就是"多实例共享"
        self.smart_contents = {}

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

    def resize(self, w, h):
        self.width = int(w)
        self.height = int(h)
        if self.selection is not None:
            self.selection.resize_to(self.width, self.height)

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
