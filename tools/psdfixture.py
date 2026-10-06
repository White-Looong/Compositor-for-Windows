# -*- coding: utf-8 -*-
"""生成一个最小合法 PSD，仅供测试 PSD 导入用。

Photoshop 的 PSD 格式非常长，这里只实现测试需要的那一小块：
8-bit RGB、若干实心色块图层、可选图层组、混合模式 / 不透明度 / 剪贴 / 可见性。
够验证 core/psd_import.py 的图层树映射、坐标换算和属性还原。

    from tools.psdfixture import write_psd

注意：这不是通用 PSD 写入器，别拿去干别的。
"""

from __future__ import annotations

import struct

# 图层记录里的 flags：bit1 置位 = 隐藏，bit3 = Photoshop 5.0 及以后
FLAG_VISIBLE = 0x08
FLAG_HIDDEN = 0x0A

# lsct（图层组区段标记）的 type —— 见 psd_tools.constants.SectionDivider
# 注意顺序：组头（BOUNDING=3）在前，子图层在中间，组尾（1/2）在后
SECT_OPEN = 1       # 组尾标记：展开的组
SECT_CLOSED = 2     # 组尾标记：折叠的组
SECT_BOUND = 3      # 组头
SECT_END = SECT_BOUND   # 兼容旧名


def _be(fmt, *vals):
    return struct.pack(">" + fmt, *vals)


def _name_block(name):
    """图层名：Pascal 字符串，整体补齐到 4 的倍数。"""
    raw = str(name).encode("latin-1", "ignore")[:255]
    block = bytes([len(raw)]) + raw
    return block + b"\x00" * ((-len(block)) % 4)


def _tagged_block(key, data):
    """一个额外数据块：'8BIM' + key + 长度 + 数据（长度补齐到 4 的倍数）。"""
    pad = (-len(data)) % 4
    return b"8BIM" + key + _be("I", len(data)) + data + b"\x00" * pad


def _sect_block(kind, blend=b"norm"):
    """图层组的 lsct 块。组头（3）只有 4 字节 type；组尾（1/2）还要带混合模式。

    组的名字 / 混合模式 / 不透明度都是从**组尾**那条记录读的
    （psd-tools 把组尾记录设成 Group 的 _record），所以要设在 section 上。
    """
    if kind == SECT_BOUND:
        return _tagged_block(b"lsct", _be("I", SECT_BOUND))
    return _tagged_block(b"lsct", _be("I", kind) + b"8BIM" + blend + _be("I", 0))


def _mask_block(lay):
    """图层蒙版数据块。长度固定 20 字节（4x4 坐标 + 默认色 + 标志 + 2 填充）。"""
    m = lay["mask"]
    l, t, r, b = m["left"], m["top"], m["right"], m["bottom"]
    return b"".join([
        _be("i", t), _be("i", l), _be("i", b), _be("i", r),
        _be("B", 255),        # 默认色：255 = 白（显示）
        _be("B", 0),
        b"\x00\x00",
    ])


def _layer_record(lay):
    l, t, r, b = lay["left"], lay["top"], lay["right"], lay["bottom"]
    n = (r - l) * (b - t)
    # 组结束标记层没有像素，通道数写 0（Photoshop 也是这么写的）
    is_end = lay.get("section") == SECT_END
    mask = lay.get("mask")
    chans = b""
    if not is_end:
        if mask is not None:
            mn = (mask["right"] - mask["left"]) * (mask["bottom"] - mask["top"])
            chans += _be("h", -2) + _be("i", mn + 2)
        chans += (_be("h", 0) + _be("i", n + 2) +
                  _be("h", 1) + _be("i", n + 2) +
                  _be("h", 2) + _be("i", n + 2))
    nch = 0 if is_end else (4 if mask is not None else 3)
    rec = b"".join([
        _be("i", t), _be("i", l), _be("i", b), _be("i", r),   # top left bottom right
        _be("H", nch),                                        # 通道数
        chans,
        b"8BIM",
        lay.get("blend", b"norm"),
        _be("B", int(lay.get("opacity", 255))),
        _be("B", int(lay.get("clipping", 0))),
        _be("B", FLAG_HIDDEN if not lay.get("visible", True) else FLAG_VISIBLE),
        b"\x00",                                              # filler
    ])
    if mask is not None:
        extra = _be("I", 20) + _mask_block(lay)
    else:
        extra = _be("I", 0)
    extra += _be("I", 0) + _name_block(lay.get("name", "layer"))
    sect = lay.get("section")
    if sect is not None:
        extra += _sect_block(sect, lay.get("section_blend", b"norm"))
    return rec + _be("I", len(extra)) + extra


def _channel_data(lay):
    l, t, r, b = lay["left"], lay["top"], lay["right"], lay["bottom"]
    n = (r - l) * (b - t)
    out = b""
    mask = lay.get("mask")
    if mask is not None:
        mn = (mask["right"] - mask["left"]) * (mask["bottom"] - mask["top"])
        out += _be("H", 0) + bytes([int(mask["gray"])]) * mn
    for ch in range(3):
        out += _be("H", 0) + bytes([int(lay["rgb"][ch])]) * n
    return out


def write_psd(path, width, height, layers, background=(255, 255, 255)):
    """写一个 PSD。

    layers: 底 -> 顶。每项是一个 dict：
        name / left / top / right / bottom / rgb=(r,g,b)
        blend=b'norm' / opacity=255 / clipping=0 / visible=True
        mask: {"left","top","right","bottom","gray"} —— 可选图层蒙版
        section: SECT_BOUND 表示组头（在子图层之前），
                 SECT_CLOSED / SECT_OPEN 表示组尾标记
    """
    records = b"".join(_layer_record(l) for l in layers)
    chandata = b"".join(_channel_data(l) for l in layers
                        if l.get("section") != SECT_END)

    layer_info = _be("H", len(layers)) + records + chandata
    layer_info += b"\x00" * ((-len(layer_info)) % 2)

    # LayerAndMaskInfo = 长度(4) + 图层信息长度(4) + 图层信息 + 全局蒙版长度(4) + 数据
    layer_and_mask = (_be("I", 4 + len(layer_info) + 4) +
                      _be("I", len(layer_info)) + layer_info + _be("I", 0))

    header = (b"8BPS" + _be("H", 1) + b"\x00" * 6 + _be("H", 3) +
              _be("I", height) + _be("I", width) + _be("H", 8) + _be("H", 3))

    n = width * height
    imgdata = _be("H", 0)
    for ch in range(3):
        imgdata += bytes([int(background[ch])]) * n

    with open(path, "wb") as f:
        f.write(header)
        f.write(_be("I", 0))        # color mode data
        f.write(_be("I", 0))        # image resources
        f.write(layer_and_mask)
        f.write(imgdata)
    return path
