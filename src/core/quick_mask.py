# -*- coding: utf-8 -*-
"""快速蒙版（PS 的 Q 键）。

进到这个模式之后，选区变成一张可以直接用画笔涂的灰度遮罩：

    * 涂**黑** -> 该处被挡住（退出后不在选区里）
    * 涂**白** -> 该处被放行（退出后在选区里）
    * 涂灰     -> 半选中，退出后是羽化的边

遮罩存在 `doc.quick_mask`，语义**一律是"会被选中的程度"**（255 = 选中），
跟显示方式无关。`doc.quick_mask_mode` 只决定红色盖在哪一边：

    * "masked"   红色盖住**未选中**区域（PS 默认，红的地方动不了）
    * "selected" 红色盖住**选中**区域（PS 的第二个选项）

把"数据语义"和"显示语义"分开，切换显示方式就只是换一次叠加图，不用改数据。
"""

from __future__ import annotations

import cv2
import numpy as np

from .selection import Selection

MASKED = "masked"
SELECTED = "selected"

# 红色的不透明度（PS 默认 50%）
OVERLAY_ALPHA = 0.5


def is_on(doc):
    return doc is not None and getattr(doc, "quick_mask", None) is not None


def enter(doc):
    """进入快速蒙版：遮罩由当前选区初始化。返回 True 表示状态真的变了。"""
    if doc is None or is_on(doc):
        return False
    sel = doc.selection
    if sel is not None and not sel.is_empty:
        doc.quick_mask = _fit_mask(sel.mask, doc.width, doc.height)
    else:
        # 没有选区时整幅都是"没选中"，所以遮罩全 0（画面全是红的）
        doc.quick_mask = np.zeros((doc.height, doc.width), np.uint8)
    return True


def exit_to_selection(doc):
    """退出：遮罩变成选区。返回 True 表示状态真的变了。"""
    if not is_on(doc):
        return False
    m = doc.quick_mask
    if m is None or not m.any():
        doc.selection = None
    else:
        doc.selection = Selection(doc.width, doc.height, m.copy())
    doc.quick_mask = None
    return True


def cancel(doc):
    """放弃这次快速蒙版的改动，恢复到进入前的选区。"""
    if not is_on(doc):
        return False
    doc.quick_mask = None
    return True


def overlay_alpha(doc):
    """红色叠加层的 alpha（0~255），画布尺寸。不在快速蒙版模式返回 None。"""
    if not is_on(doc):
        return None
    m = doc.quick_mask.astype(np.float32)
    if doc.quick_mask_mode == SELECTED:
        a = m
    else:
        a = 255.0 - m
    return np.clip(a * OVERLAY_ALPHA, 0, 255).astype(np.uint8)


def _fit_mask(mask, w, h):
    """遮罩尺寸对不上画布时兜底拉一下（工程被改过尺寸之类的极端情况）。"""
    if mask is None:
        return np.zeros((h, w), np.uint8)
    if mask.shape == (h, w):
        return mask.astype(np.uint8, copy=True)
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
