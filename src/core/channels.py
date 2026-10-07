# -*- coding: utf-8 -*-
"""通道：Alpha 通道 + RGB 通道的显示开关。

与 Photoshop 的语义一致，两条原则：

1. **通道可见性只影响"显示"，不动合成数据**。关掉红通道是把屏幕上那一路
   填成白色，不是把 R 乘 0。所以它必须作用在 `render_document()` 的**输出**上，
   而不是 acc 缓冲里 —— 一旦乘进 acc，撤销/重做与局部重渲染就要额外考虑
   "当时哪些通道是开的"，而且和 Photoshop 的行为对不上。
2. **RGB 三个通道是画布自带的**，没有存成通道数据；只有 **Alpha 通道**是
   真正的附加数据（一张 (h,w) uint8 遮罩），要存进工程文件。

典型用法：
    ch = add_channel(doc, "蒙版 A")          # 存当前选区为通道
    ch.mask[...] = 0                          # 改内容（先 detach_channel）
    sel = channel_to_selection(doc, "蒙版 A") # 载入选区
"""

from __future__ import annotations

import re
import unicodedata

import cv2
import numpy as np

RGB_KEYS = ("R", "G", "B")
RGB_LABELS = {"R": "红", "G": "绿", "B": "蓝"}
ALPHA_KEY = "A"
RGB_VIEW_KEYS = RGB_KEYS + (ALPHA_KEY,)

# 通道面板里前几行是画布自带的（RGB 三路 + Alpha + 合并 Alpha），它们没有 uuid。
# 用这个前缀和真正的通道 id 区分开 —— UI 与 MainWindow 都要判它，统一放这里。
RGB_ROW_PREFIX = "@"
COMPOSITE_ROW = "@composite"

MAX_CHANNELS = 16           # Photoshop 的通道上限（不含 RGB 与合并 alpha）


def _visible_default():
    """RGB + alpha 全开的显示状态。"""
    return dict((k, True) for k in RGB_VIEW_KEYS)


class Channel:
    """一个 Alpha 通道：名字 + 一张 (h,w) uint8 遮罩 + 自己的可见性。"""

    def __init__(self, name, width, height, mask=None, visible=True):
        self.id = None
        self.name = str(name)
        self.visible = bool(visible)
        if mask is None:
            self.mask = np.zeros((int(height), int(width)), np.uint8)
        else:
            self.mask = np.ascontiguousarray(mask.astype(np.uint8, copy=False))

    @property
    def width(self):
        return int(self.mask.shape[1])

    @property
    def height(self):
        return int(self.mask.shape[0])

    def is_empty(self):
        return not bool(self.mask.any())

    def copy(self):
        """浅拷贝：新 Channel 对象，但**遮罩数组共享**。

        必须共享 —— 撤销栈整个建立在"浅克隆 + 写时复制"上（见 §3.1）。
        这里要是顺手深拷贝，每次 commit 都会把每张通道遮罩复制一遍，
        撤销栈 40 步 × 几十张通道遮罩的内存直接爆掉。
        真正要改内容前调 `Document.detach_channel()`。
        """
        c = Channel(self.name, 1, 1, self.mask, self.visible)
        c.id = self.id
        return c

    def to_dict(self):
        return {"id": self.id, "name": self.name,
                "visible": bool(self.visible)}

    @staticmethod
    def from_dict(d, mask):
        c = Channel(d.get("name", "通道"), 1, 1, mask,
                    d.get("visible", True))
        c.id = d.get("id")
        return c


# ---------------------------------------------------------------- 名字规范化

def unique_name(doc, base, taken=None):
    """生成不重名的通道名："蒙版" -> "蒙版 2"。

    `taken` 传进来时用它判重（传通道 id 集合）；否则按 doc 现有的通道名判。
    """
    used = set(taken or ())
    if not taken:
        used = set(c.name for c in (getattr(doc, "channels", None) or []))
    base = str(base).strip() or "通道"
    if base not in used:
        return base
    i = 2
    while "%s %d" % (base, i) in used:
        i += 1
    return "%s %d" % (base, i)


# 显示名的宽度：中日韩字符占两列，不算的话 "通道 10" 会比 "通道 9" 短一格
def _display_width(s):
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1
               for ch in str(s))


def truncate_name(s, limit=18):
    """按显示宽度截断（超出加省略号）。界面上用，别让长名把布局挤歪。"""
    if _display_width(s) <= limit:
        return s
    out, w = "", 0
    for ch in str(s):
        cw = 2 if unicodedata.east_asian_width(ch) in "WF" else 1
        if w + cw > limit - 1:
            break
        out += ch
        w += cw
    return out + "…"


# ---------------------------------------------------------------- 增删改

def add_channel(doc, name, mask=None):
    """新建一个 Alpha 通道。返回 Channel；超过 MAX_CHANNELS 返回 None。

    `mask` 省略时用当前选区（没选区就是全 0）。
    """
    chans = getattr(doc, "channels", None)
    if chans is None:
        chans = doc.channels = []
    if len(chans) >= MAX_CHANNELS:
        return None
    src = mask
    if src is None and doc.selection is not None:
        src = doc.selection.mask
    if src is not None:
        src = np.ascontiguousarray(np.asarray(src, np.uint8))
    ch = Channel(unique_name(doc, name), doc.width, doc.height, src)
    import uuid
    ch.id = uuid.uuid4().hex[:12]
    chans.append(ch)
    return ch


def remove_channel(doc, cid):
    """按 id 删一个 Alpha 通道。删的是附加数据，永远不删画布自带的 RGB。"""
    chans = getattr(doc, "channels", None) or []
    for i, c in enumerate(chans):
        if c.id == cid:
            return chans.pop(i)
    return None


def find_channel(doc, cid):
    for c in (getattr(doc, "channels", None) or []):
        if c.id == cid:
            return c
    return None


def rename_channel(doc, cid, name):
    """重命名。空名 / 超长名 / 与别的通道重名都会被纠正（不报错）。

    算出来的名字**不会等于它自己原来的名字** —— 把「蒙版 3」改成「蒙版」，
    蒙版 / 蒙版 2 都被占了，那就该给「蒙版 4」，而不是原地不动让人以为没生效。
    """
    ch = find_channel(doc, cid)
    if ch is None:
        return None
    name = str(name or "").strip()
    if not name:
        return ch
    # taken 里**含**它自己的旧名：见上面那句注释
    taken = set(x.name for x in (getattr(doc, "channels", None) or []))
    ch.name = unique_name(doc, name, taken=taken)
    return ch


def reorder_channel(doc, cid, delta):
    """上移 / 下移一位（PS 里拖通道的顺序）。返回是否真的动了。"""
    chans = getattr(doc, "channels", None) or []
    idx = next((i for i, c in enumerate(chans) if c.id == cid), None)
    if idx is None:
        return False
    j = idx + int(delta)
    if not (0 <= j < len(chans)):
        return False
    chans[idx], chans[j] = chans[j], chans[idx]
    return True


# ---------------------------------------------------------------- 与选区互转

def channel_to_selection(doc, cid):
    """载入选区：通道遮罩 -> doc.selection。通道为空时得到全不选的选区。"""
    ch = find_channel(doc, cid)
    if ch is None:
        return None
    from .selection import Selection
    doc.detach_selection()
    if ch.is_empty():
        doc.selection = None       # PS：空通道载入选区 = 取消选区
    else:
        doc.selection = Selection(doc.width, doc.height, ch.mask.copy())
    return doc.selection


def selection_to_channel(doc, name="Alpha 1"):
    """新建通道并把当前选区存进去。没有选区时存一张全 0（= 不选中任何处）。"""
    if doc.selection is None:
        mask = np.zeros((doc.height, doc.width), np.uint8)
    else:
        mask = doc.selection.mask.copy()
    return add_channel(doc, name, mask)


def merge_channel_into_alpha(doc, cid, mode="replace"):
    """把通道并进「合并的 Alpha」（也就是所有图层合成后的不透明度）。

    mode: replace（替掉）/ add（逐像素取 max，即并集）/ subtract（相减）
    只能改**显示**用的合成 alpha，不动图层数据 —— 这和「图层 → 拼合」是两件事。

    「合并的 Alpha」永远是**画布尺寸**；通道尺寸对不上时把通道缩过去
    （而不是把合并 alpha 改成通道的尺寸 —— 那会让画布整个错位）。
    """
    ch = find_channel(doc, cid)
    if ch is None:
        return False
    cur = getattr(doc, "composite_alpha", None)
    if cur is None:
        return False
    target = (doc.height, doc.width)
    if cur.shape[:2] != target:
        cur = cv2.resize(cur, (target[1], target[0]),
                         interpolation=cv2.INTER_LINEAR)
    m = ch.mask
    if m.shape[:2] != target:
        m = cv2.resize(m, (target[1], target[0]),
                       interpolation=cv2.INTER_NEAREST)
    if mode == "add":
        out = np.maximum(cur, m)
    elif mode == "subtract":
        out = np.clip(cur.astype(np.int32) - m.astype(np.int32), 0, 255)
    else:
        out = m.copy()
    doc.composite_alpha = np.ascontiguousarray(out.astype(np.uint8))
    return True


def reset_composite_alpha(doc):
    doc.composite_alpha = None


# ---------------------------------------------------------------- 显示

def rgb_visibility(doc):
    v = getattr(doc, "channel_view", None)
    if not isinstance(v, dict) or not v:
        v = doc.channel_view = _visible_default()
    for k in RGB_VIEW_KEYS:
        v.setdefault(k, True)
    return v


def any_hidden(doc):
    """有没有通道被关掉。全开着时渲染可以走快速路径（原样返回入参）。"""
    v = rgb_visibility(doc)
    if not all(v.get(k, True) for k in RGB_VIEW_KEYS):
        return True
    for c in (getattr(doc, "channels", None) or []):
        if not c.visible and not c.is_empty():
            return True
        # 开着且非空的额外通道也要乘上去 —— 别漏判，
        # 否则会走"什么都不用改"的快速路径，通道的遮罩就不生效了
        if c.visible and not c.is_empty():
            return True
    if getattr(doc, "composite_alpha", None) is not None:
        return True
    return False


def apply_channel_view(arr, doc):
    """把渲染结果按通道可见性处理一遍，返回新的 (h,w,4) uint8。

    关掉的 RGB 通道**填白**（PS 的行为：显示为纸白，不是黑色）；
    关掉的 alpha 通道整张填 0（全透明 → 看到棋盘格）。
    额外 alpha 通道开着时按它的遮罩**再乘一遍**不透明度。

    全通道都开着、也没有额外通道与合并 alpha 时，**原样返回入参同一个对象**
    —— 局部重渲染每帧都会调这里，不想每次都白拷一张。
    """
    if not any_hidden(doc):
        return arr
    v = rgb_visibility(doc)
    out = arr.copy()
    for i, k in enumerate(RGB_KEYS):
        if not v.get(k, True):
            out[..., i] = 255
    if not v.get(ALPHA_KEY, True):
        out[..., 3] = 0
    a = out[..., 3]
    for c in (getattr(doc, "channels", None) or []):
        if not c.visible or c.is_empty():
            continue
        if c.mask.shape[:2] != a.shape[:2]:
            m = cv2.resize(c.mask, (a.shape[1], a.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
        else:
            m = c.mask
        # 通道是"乘上去"：0 = 该处全透明。uint16 中间量避免 uint8 溢出。
        out[..., 3] = ((a.astype(np.uint16) * m.astype(np.uint16) + 127)
                       // 255).astype(np.uint8)
        a = out[..., 3]
    # 画布级的"合并 alpha 通道"（通道面板里改的那个）
    ca = getattr(doc, "composite_alpha", None)
    if ca is not None and ca.shape[:2] == a.shape[:2]:
        out[..., 3] = ((a.astype(np.uint16) * ca.astype(np.uint16) + 127)
                       // 255).astype(np.uint8)
    return out


# ---------------------------------------------------------------- 缩略图

def channel_thumb(arr, key, size=34):
    """通道缩略图：(h,w,4) uint8 的合成结果 + 通道 key -> (n,n) uint8 灰度图。

    RGB 三路取对应分量，Alpha 取第 3 路。缩到 size 见方（不变形比例的
    情况由调用方决定；这里跟图层面板的 `render_layer_thumb` 保持一致）。
    """
    h, w = arr.shape[:2]
    if key in RGB_KEYS:
        g = arr[..., RGB_KEYS.index(key)]
    elif key == ALPHA_KEY:
        g = arr[..., 3]
    else:
        g = np.full((h, w), 255, np.uint8)
    im = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
    return im


def composite_alpha_of(arr):
    """从渲染结果里取合成后的 alpha（供"载入选区"用）。"""
    return np.ascontiguousarray(arr[..., 3])


# ---------------------------------------------------------------- 画布尺寸变化

def resize_channels(doc, w, h):
    """画布尺寸变了，所有通道遮罩跟着缩（最近邻，保住硬边的选区形状）。"""
    for c in (getattr(doc, "channels", None) or []):
        if c.mask.shape[:2] != (int(h), int(w)):
            c.mask = np.ascontiguousarray(
                cv2.resize(c.mask, (int(w), int(h)),
                           interpolation=cv2.INTER_NEAREST))
    ca = getattr(doc, "composite_alpha", None)
    if ca is not None and ca.shape[:2] != (int(h), int(w)):
        doc.composite_alpha = np.ascontiguousarray(
            cv2.resize(ca, (int(w), int(h)), interpolation=cv2.INTER_NEAREST))
    return doc


# ---------------------------------------------------------------- 存取

def to_dict(doc):
    """写进工程 manifest 的部分。遮罩本体由 project_io 像图层那样存成 PNG。"""
    return {
        "view": dict((k, bool(v)) for k, v in rgb_visibility(doc).items()),
        "alpha": [c.to_dict() for c in (getattr(doc, "channels", None) or [])],
        # 注意不能写 bool(doc.composite_alpha) —— 那是 ndarray，
        # 超过一个元素时真值判断会直接抛 ValueError（要用 is not None）
        "composite_alpha": getattr(doc, "composite_alpha", None) is not None,
    }


def from_dict(doc, d, masks):
    """读回来。`masks` 是 {channel_id: (h,w) uint8}。"""
    d = d or {}
    doc.channel_view = _visible_default()
    for k, v in (d.get("view") or {}).items():
        if k in RGB_VIEW_KEYS:
            doc.channel_view[k] = bool(v)
    doc.channels = []
    for cd in (d.get("alpha") or []):
        m = masks.get(cd.get("id"))
        if m is None:
            continue
        doc.channels.append(Channel.from_dict(cd, m))
    doc.composite_alpha = None
    if d.get("composite_alpha"):
        doc.composite_alpha = np.zeros((doc.height, doc.width), np.uint8)
    return doc
