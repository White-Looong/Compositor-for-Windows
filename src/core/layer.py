# -*- coding: utf-8 -*-
"""图层数据模型。

图层分四种：
  * image      —— 位图图层，持有一张 RGBA 位图 + 可选蒙版 + 变换
  * group      —— 图层组（文件夹），持有子图层，自身也有不透明度与混合模式
  * adjustment —— 调整层，没有像素，只有一组参数（core.adjust），
                  渲染时作用于它下面所有图层的合成结果
  * text       —— 文字图层，像素由参数（core.text）栅格化生成，
                  改参数就重新生成，所以是非破坏性的；想改像素得先栅格化
  * smart      —— 智能对象，自身不存像素，引用一段嵌入内容（core.smart），
                  渲染时才合成；内容可以独立编辑，滤镜是参数化的（智能滤镜）

变换（tx/ty/sx/sy/rot/flip）是**非破坏性**的：位图始终以原始分辨率保存，
只在渲染时做仿射变换，所以缩放再放大不会丢细节。
"""

from __future__ import annotations

import copy
import math
import uuid

import numpy as np

from .blend import PASS_THROUGH

LAYER_IMAGE = "image"
LAYER_GROUP = "group"
LAYER_ADJUSTMENT = "adjustment"
LAYER_TEXT = "text"
LAYER_SMART = "smart"

# 这几种图层有像素（或像素是派生出来的），可以按位图图层参与合成
PIXEL_KINDS = (LAYER_IMAGE, LAYER_TEXT, LAYER_SMART)


class Layer:
    def __init__(self, name="Layer", kind=LAYER_IMAGE, image=None,
                 width=0, height=0):
        self.id = uuid.uuid4().hex
        self.name = name
        self.kind = kind
        self.visible = True
        self.locked = False
        self.opacity = 1.0
        self.blend = ("Normal" if kind in PIXEL_KINDS else PASS_THROUGH)

        self.children = []          # 仅 group 使用，底 -> 顶
        self.image = image          # (h,w,4) uint8 RGBA，直线色
        self.mask = None            # (h,w) uint8，可选
        self.adjust = None          # 调整层：{"type": key, "params": {...}}
        self.text = None            # 文字层：参数 dict，见 core.text
        self.effects = None         # 图层样式：{效果 id: {enabled, ...}}，见 core.effects
        self.so_id = None           # 智能对象：嵌入内容的 id，见 core.smart
        self.so_filters = None      # 智能对象：智能滤镜列表 [{"key","params",...}]
        self.so_size = None         # 智能对象：内容尺寸 (w,h)，栅格没生成时兜底用
        self.clipped = False        # 裁剪到下方图层（剪贴蒙版）
        self.mask_enabled = True
        self.mask_linked = True

        # 变换：tx/ty 是图层**中心**在画布坐标中的位置
        self.tx = width / 2.0
        self.ty = height / 2.0
        self.sx = 1.0
        self.sy = 1.0
        self.rot = 0.0
        self.flip_h = False
        self.flip_v = False

        self._noise_cache_key = None

    # ---------- 基本属性 ----------

    @property
    def is_group(self):
        return self.kind == LAYER_GROUP

    @property
    def is_adjustment(self):
        return self.kind == LAYER_ADJUSTMENT

    @property
    def is_text(self):
        return self.kind == LAYER_TEXT

    @property
    def is_smart(self):
        return self.kind == LAYER_SMART

    @property
    def has_effects(self):
        """是否启用了至少一个图层样式。"""
        if not self.effects:
            return False
        from .effects import has_effects
        return has_effects(self.effects)

    def adjustment_key(self):
        return (self.adjust or {}).get("type")

    @property
    def src_size(self):
        """源图像尺寸 (w, h)。组图层返回 None。

        智能对象的栅格是渲染时才生成的，还没生成时退回到内容尺寸 ——
        属性面板、包围盒、手柄都靠它，不能是 None。
        """
        if self.image is None:
            return self.so_size if self.kind == LAYER_SMART else None
        h, w = self.image.shape[:2]
        return w, h

    def source_extent(self):
        """变换所基于的源尺寸；组用其子图层包围盒推导，见 document。"""
        return self.src_size

    # ---------- 变换矩阵 ----------

    def matrix(self, ox=0.0, oy=0.0):
        """返回 2x3 仿射矩阵，把**源图像坐标**映射到（画布坐标 - (ox,oy)）。

        cv2.warpAffine 需要的是 src->dst 的 2x3 矩阵。
        """
        src = self.src_size
        if src is None:
            return None
        w, h = src
        cx, cy = w / 2.0, h / 2.0
        fx = -1.0 if self.flip_h else 1.0
        fy = -1.0 if self.flip_v else 1.0
        th = math.radians(self.rot)
        cos, sin = math.cos(th), math.sin(th)

        # 组合顺序：先平移到中心为原点 -> 翻转 -> 缩放 -> 旋转 -> 平移到目标
        a = self.sx * fx * cos
        b = self.sx * fx * sin
        c = -self.sy * fy * sin
        d = self.sy * fy * cos
        # 手工展开 T(tx-ox, ty-oy) @ R @ S @ F @ T(-cx,-cy)
        # 结果: [a c tx-ox - (a*cx + c*cy) ; b d ty-oy - (b*cx + d*cy)]
        tx = self.tx - ox - (a * cx + c * cy)
        ty = self.ty - oy - (b * cx + d * cy)
        return np.array([[a, c, tx],
                         [b, d, ty]], dtype=np.float64)

    def corners(self):
        """源图像四角经变换后的画布坐标，(4,2)。"""
        src = self.src_size
        if src is None:
            return None
        w, h = src
        M = self.matrix()
        pts = np.array([[0.0, 0.0, 1.0],
                        [w, 0.0, 1.0],
                        [w, h, 1.0],
                        [0.0, h, 1.0]])
        return pts @ M.T

    def bbox(self, canvas_w=0, canvas_h=0):
        """变换后的轴对齐包围盒 (x0, y0, x1, y1)，浮点。"""
        c = self.corners()
        if c is None:
            return None
        x0 = float(c[:, 0].min())
        y0 = float(c[:, 1].min())
        x1 = float(c[:, 0].max())
        y1 = float(c[:, 1].max())
        if canvas_w:
            x0 = max(x0, 0.0)
            y0 = max(y0, 0.0)
            x1 = min(x1, float(canvas_w))
            y1 = min(y1, float(canvas_h))
        return (x0, y0, x1, y1)

    def set_scale_keep_center(self, sx, sy):
        self.sx = sx
        self.sy = sy

    def reset_transform(self, canvas_w=0, canvas_h=0):
        self.sx = 1.0
        self.sy = 1.0
        self.rot = 0.0
        self.tx = canvas_w / 2.0
        self.ty = canvas_h / 2.0

    # ---------- 复制 ----------

    def clone(self, deep=False):
        """浅克隆：新建 Layer 对象但**共享** numpy 数组。

        核心骨架阶段没有任何像素级编辑（没有画笔/选区），所以浅克隆是安全的；
        撤销栈因此非常省内存。将来加入绘制类工具时，改动像素前必须
        先 `layer.image = layer.image.copy()`。
        """
        n = Layer(self.name, self.kind, image=self.image)
        n.id = self.id
        n.visible = self.visible
        n.locked = self.locked
        n.opacity = self.opacity
        n.blend = self.blend
        n.mask = self.mask
        n.mask_enabled = self.mask_enabled
        n.mask_linked = self.mask_linked
        # 调整层 / 文字层参数要深拷贝：历史快照不能被后续的参数改动带跑
        n.adjust = copy.deepcopy(self.adjust) if self.adjust else None
        n.text = copy.deepcopy(self.text) if self.text else None
        n.effects = copy.deepcopy(self.effects) if self.effects else None
        n.so_id = self.so_id
        n.so_filters = (copy.deepcopy(self.so_filters)
                        if self.so_filters is not None else None)
        n.so_size = self.so_size
        n.clipped = self.clipped
        n.tx, n.ty = self.tx, self.ty
        n.sx, n.sy = self.sx, self.sy
        n.rot = self.rot
        n.flip_h, n.flip_v = self.flip_h, self.flip_v
        for ch in self.children:
            n.children.append(ch.clone(deep=deep))
        return n

    def detach_pixels(self):
        """写时复制：把本图层的像素数组换成本层独占的副本。

        历史快照与当前图层原本共享同一个 numpy 数组。任何像素级修改（画笔、
        橡皮、填充、清除）之前必须先调用它，否则会把撤销栈里的旧状态一起改掉。
        """
        if self.image is not None:
            self.image = self.image.copy()
        if self.mask is not None:
            self.mask = self.mask.copy()

    def walk(self):
        """深度优先遍历自身及所有后代。"""
        yield self
        for ch in self.children:
            yield from ch.walk()

    def to_dict(self):
        d = {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "visible": self.visible,
            "locked": self.locked,
            "opacity": self.opacity,
            "blend": self.blend,
            "tx": self.tx, "ty": self.ty,
            "sx": self.sx, "sy": self.sy,
            "rot": self.rot,
            "flip_h": self.flip_h, "flip_v": self.flip_v,
            "mask_enabled": self.mask_enabled,
            "mask_linked": self.mask_linked,
            "clipped": self.clipped,
        }
        if self.adjust:
            d["adjust"] = self.adjust
        if self.text:
            d["text"] = self.text
        if self.effects:
            d["effects"] = self.effects
        if self.kind == LAYER_SMART:
            # 注意：智能对象的 image 是派生的，不进工程文件 —— 存的是内容本身
            if self.so_id:
                d["so_id"] = self.so_id
            if self.so_filters is not None:
                d["so_filters"] = self.so_filters
        if self.is_group:
            d["children"] = [c.to_dict() for c in self.children]
        return d

    @staticmethod
    def from_dict(d, images):
        lay = Layer(d.get("name", "Layer"), d.get("kind", LAYER_IMAGE))
        lay.id = d.get("id", lay.id)
        lay.visible = d.get("visible", True)
        lay.locked = d.get("locked", False)
        lay.opacity = d.get("opacity", 1.0)
        lay.blend = d.get("blend", "Normal")
        lay.tx = d.get("tx", 0.0)
        lay.ty = d.get("ty", 0.0)
        lay.sx = d.get("sx", 1.0)
        lay.sy = d.get("sy", 1.0)
        lay.rot = d.get("rot", 0.0)
        lay.flip_h = d.get("flip_h", False)
        lay.flip_v = d.get("flip_v", False)
        lay.mask_enabled = d.get("mask_enabled", True)
        lay.mask_linked = d.get("mask_linked", True)
        lay.clipped = bool(d.get("clipped", False))
        if d.get("adjust"):
            lay.adjust = d["adjust"]
        if d.get("text"):
            try:
                from .text import normalize_text_params
                lay.text = normalize_text_params(d["text"])
            except Exception:
                lay.text = d["text"]
        if d.get("effects"):
            try:
                from .effects import normalize_effects
                lay.effects = normalize_effects(d["effects"])
            except Exception:
                lay.effects = d["effects"]
        key = d.get("image")
        if key and key in images:
            lay.image = images[key]
        if lay.kind == LAYER_SMART:
            # 栅格是内容渲染出来的，读回来由 core.smart 重算，不进工程文件
            lay.image = None
            lay.so_id = d.get("so_id")
            lay.so_filters = d.get("so_filters") or []
        mkey = d.get("mask")
        if mkey and mkey in images:
            m = images[mkey]
            lay.mask = m if m.ndim == 2 else m[..., 0]
        for cd in d.get("children", []):
            lay.children.append(Layer.from_dict(cd, images))
        if lay.is_group and not lay.blend:
            lay.blend = PASS_THROUGH
        return lay
