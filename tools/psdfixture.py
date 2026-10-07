# -*- coding: utf-8 -*-
"""生成一个最小合法 PSD，仅供测试 PSD 导入用。

Photoshop 的 PSD 格式非常长，这里只实现测试需要的那一小块：
8-bit RGB、若干实心色块图层、可选图层组、混合模式 / 不透明度 / 剪贴 / 可见性、
可选图层蒙版、**可选文字图层（TySh）**、**可选图层样式（lfxr）**。
够验证 core/psd_import.py 的图层树映射、坐标换算、属性还原，
以及 core/psd_text.py / core/psd_effects.py 的两条新链路。

    from tools.psdfixture import write_psd

文字层和样式块用 psd_tools 自己的 descriptor 序列化器生成 —— 自己手写
那段二进制不现实，而且写出来的格式还得能被 psd_tools 读回去才说明测试有效。

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
    extra += _extra_blocks(lay)
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


# ---------------------------------------------------------------- 文字层 / 图层样式

def _ed():
    """psd_tools 的 EngineData 导入（延迟到真正要写文字层时才需要 psd-tools）。"""
    from psd_tools.psd import engine_data as ed
    return ed


def _d():
    from psd_tools.psd import descriptor as dsc
    return dsc


def _sheet(dd, ed):
    """{键: 值} -> EngineData。键要包成 Property（序列化成 /Key）。"""
    out = ed.EngineData()
    for k, v in dd.items():
        out[ed.Property(k)] = v
    return out


def _color(r, g, b, ed):
    c = ed.EngineData()
    c[ed.Property("Rd  ")] = ed.Float(float(r))
    c[ed.Property("Grn ")] = ed.Float(float(g))
    c[ed.Property("Bl  ")] = ed.Float(float(b))
    return c


def _color_list(rgb, ed):
    """FillColor 的结构：`{Clrs: [{Clr : {Rd, Grn, Bl}}]}`。

    **别写成 `Values`** —— psd-tools 的 `CharacterStyle.fill_color` 读的是
    `Values` 里一串平铺数字，跟 Photoshop 实际写出来的结构对不上，会安静地
    返回 None（`core/psd_text.color_from_engine` 走的就是 `Clrs` 这条路）。
    离屏 Qt 只有 2 个字体，测试要用 `fonts` 显式指定系统里真有的字体名。
    """
    entry = ed.EngineData()
    entry[ed.Property("Clr ")] = _color(rgb[0], rgb[1], rgb[2], ed)
    lst = ed.List()
    lst.append(entry)
    box = ed.EngineData()
    box[ed.Property("Clrs")] = lst
    return box


def _char_style(spec, ed):
    """一个字符样式表（StyleRun 里的 StyleSheetData）。

    spec 的键与 core/psd_text.py 关心的字段一一对应；没给的用合理默认。
    """
    size = float(spec.get("size", 24.0))
    d = {
        "Font": ed.Integer(int(spec.get("font_index", 0))),
        "FontSize": ed.Float(size),
        "FauxBold": ed.Bool(bool(spec.get("faux_bold", False))),
        "FauxItalic": ed.Bool(bool(spec.get("faux_italic", False))),
        "Underline": ed.Bool(bool(spec.get("underline", False))),
        "Tracking": ed.Integer(int(spec.get("tracking", 0))),
        "Kerning": ed.Integer(int(spec.get("kerning", 0))),
        "AutoKern": ed.Bool(True),
        "Leading": ed.Float(size * 1.2),
        "AutoLeading": ed.Bool(True),
        "FillFlag": ed.Bool(bool(spec.get("fill_flag", True))),
        "StrokeFlag": ed.Bool(bool(spec.get("stroke_flag", False))),
        "BaselineShift": ed.Float(float(spec.get("baseline_shift", 0.0))),
        "HorizontalScale": ed.Float(float(spec.get("hscale", 100.0))),
        "VerticalScale": ed.Float(float(spec.get("vscale", 100.0))),
        "FillColor": _color_list(spec.get("color", (0, 0, 0)), ed),
    }
    return _sheet(d, ed)


def _style_run(runs, ed):
    """RunArray：每项是 {StyleSheet: {StyleSheetData: {...}}}，少一层就读不到。

    （层数是踩出来的：psd-tools 的 typesetting 先取 `StyleSheet` 再取
    `StyleSheetData`，直接塞在 RunArray 项里会被当成空样式。）
    """
    lens = ed.List()
    arr = ed.List()
    for text, spec in runs:
        lens.append(ed.Integer(len(text)))
        data = ed.EngineData()
        data[ed.Property("StyleSheetData")] = _char_style(spec, ed)
        item = ed.EngineData()
        item[ed.Property("StyleSheet")] = data
        arr.append(item)
    out = ed.EngineData()
    out[ed.Property("RunLengthArray")] = lens
    out[ed.Property("RunArray")] = arr
    return out


def _para_run(paras, ed):
    """ParagraphRun：RunArray 项是 {ParagraphSheet: {Properties: {...}}}。"""
    lens = ed.List()
    arr = ed.List()
    for text, spec in paras:
        lens.append(ed.Integer(len(text)))
        d = {
            "Justification": ed.Integer(int(spec.get("justification", 0))),
            "StartIndent": ed.Float(float(spec.get("start_indent", 0.0))),
            "EndIndent": ed.Float(float(spec.get("end_indent", 0.0))),
            "FirstLineIndent": ed.Float(float(spec.get("first_indent", 0.0))),
            "SpaceBefore": ed.Float(float(spec.get("space_before", 0.0))),
            "SpaceAfter": ed.Float(float(spec.get("space_after", 0.0))),
            "AutoLeading": ed.Float(float(spec.get("auto_leading", 1.2))),
        }
        props = ed.EngineData()
        props[ed.Property("Properties")] = _sheet(d, ed)
        item = ed.EngineData()
        item[ed.Property("ParagraphSheet")] = props
        arr.append(item)
    out = ed.EngineData()
    out[ed.Property("RunLengthArray")] = lens
    out[ed.Property("RunArray")] = arr
    return out


def _tysh_block(lay):
    """图层的 TySh（TypeToolObjectSetting）数据块内容。

    结构（见 psd_tools.psd.tagged_blocks.TypeToolObjectSetting）：
        version(H) + transform(6d) + text_version(H) + DescriptorBlock
        + warp_version(H) + DescriptorBlock + 4×i 的 bbox
    DescriptorBlock 里 `Txt ` 是纯文本、`EngineData` 是排版引擎数据。
    """
    import io
    ed, dsc = _ed(), _d()
    from psd_tools.psd.tagged_blocks import TypeToolObjectSetting

    text = lay["text"]
    content = text.get("content", "")
    # PSD 内部用 \r 换行（真实 Photoshop 也这么存）
    psd_text = content.replace("\r\n", "\r").replace("\n", "\r")

    eng = ed.EngineData()
    editor = ed.EngineData()
    editor[ed.Property("Text")] = ed.String(psd_text)
    eng[ed.Property("Editor")] = editor
    # 单段时把 text 本身当样式表用（"size"/"color"/"tracking" 直接写在 text 里）；
    # 写了 "runs" / "paras" 才会用它们分段 —— 两者都不写就用 font 默认值。
    eng[ed.Property("StyleRun")] = _style_run(
        text.get("runs") or [(content, text)], ed)
    eng[ed.Property("ParagraphRun")] = _para_run(
        text.get("paras") or [(content, text)], ed)

    res = ed.EngineData()
    fonts = ed.List()
    # 不写 "fonts" 时，用 text 里的 family / font_name 组一个 FontSet 条目 ——
    # 否则 run 的 Font 索引指向的那个条目是空的，字体名会退成默认的 Arial。
    default_font = {
        "name": text.get("font_name", text.get("family", "Arial")),
        "family": text.get("family", "Arial"),
        "style": text.get("font_style", "Regular"),
    }
    for f in text.get("fonts", [default_font]):
        item = ed.EngineData()
        item[ed.Property("Name")] = ed.String(f.get("name", "Arial"))
        item[ed.Property("FontFamily")] = ed.String(f.get("family", "Arial"))
        item[ed.Property("FontStyle")] = ed.String(f.get("style", "Regular"))
        item[ed.Property("FontType")] = ed.Integer(1)
        fonts.append(item)
    res[ed.Property("FontSet")] = fonts

    top = ed.EngineData()
    top[ed.Property("EngineDict")] = eng
    top[ed.Property("ResourceDict")] = res

    td = dsc.DescriptorBlock(classID=b"EngineData")
    td[dsc.String(b"Txt ")] = dsc.String(psd_text)
    td[dsc.String(b"EngineData")] = dsc.RawData(top.tobytes())
    warp = dsc.DescriptorBlock(classID=b"warp")

    l, t, r, b = lay["left"], lay["top"], lay["right"], lay["bottom"]
    obj = TypeToolObjectSetting(
        version=1, transform=(1.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        text_version=50, text_data=td, warp_version=1, warp=warp,
        left=l, top=t, right=r, bottom=b)
    fp = io.BytesIO()
    obj.write(fp)
    return fp.getvalue()


def _dkey(s):
    return s.encode("ascii") if isinstance(s, str) else bytes(s)


def _desc(props, dsc, classID=b"null"):
    """{键: 值} -> **Descriptor 对象**（不是 bytes）。

    键统一包成 String（descriptor 的键就是 4 字节 key，不是 /Property）；
    非 4 字节的键会被 psd-tools 拒掉。

    返回对象而不是字节：效果项要放进 `List` 里，而 `List.write` 要求每一项
    都是带 `ostype` 的元素对象（提前序列化成 bytes 会报
    "'bytes' object has no attribute 'ostype'"）。要字节用 `_desc_bytes`。
    """
    blk = dsc.Descriptor(classID=classID)
    for k, v in props.items():
        blk[dsc.String(_dkey(k))] = v
    return blk


def _desc_bytes(props, dsc, classID=b"null"):
    """同 `_desc`，但直接给字节（顶层块用）。"""
    import io
    fp = io.BytesIO()
    _desc(props, dsc, classID).write(fp)
    return fp.getvalue()


def _fx_color(rgb, dsc):
    c = dsc.Descriptor()
    for k, v in zip((b"Rd  ", b"Grn ", b"Bl  "),
                    (float(rgb[0]), float(rgb[1]), float(rgb[2]))):
        c[dsc.String(k)] = dsc.Double(v)
    return c


def _fx_solid(rgb, dsc):
    """纯色填充。

    注意  存的就是**颜色描述符本身**（{Rd, Grn, Bl}），不要再包一层
    {Clr : ...} —— 包了 psd-tools 的  就取不到 RGB，
    颜色会静默退回默认值。
    """
    return _fx_color(rgb, dsc)


def _fx_gradient(colors, dsc, style=b"Lnr ", angle=90.0):
    """渐变：{Clrs: [{Clr, Midpoint}], Type, Angle, Reverse, Dither}。

    本项目的渐变叠加只有首尾两色，所以这里也只放首尾两个色标。
    """
    lst = dsc.List()
    mids = ([0.0, 1.0] if len(colors) < 2 else
            [i / float(len(colors) - 1) for i in range(len(colors))])
    for rgb, mid in zip(colors, mids):
        e = dsc.Descriptor()
        e[dsc.String(b"Clr ")] = _fx_color(rgb, dsc)
        e[dsc.String(b"Midpoint")] = dsc.Double(float(mid))
        lst.append(e)
    g = dsc.Descriptor()
    g[dsc.String(b"Clrs")] = lst
    g[dsc.String(b"Type")] = dsc.Enumerated(b"GrdT", style)
    g[dsc.String(b"Angle")] = dsc.UnitFloat(angle, b"#Ang")
    g[dsc.String(b"Reverse")] = dsc.Bool(False)
    g[dsc.String(b"Dither")] = dsc.Bool(False)
    return g


def _fx_item(dsc, extra):
    """一个 effect 项 = Descriptor，classID 取 `_class`。

    `enab` / `present` / `showInDialog` / `Opct` 是每个效果都有的公共键。
    psd-tools 过滤时要求 `present` 为真，并且**靠 classID 认出效果种类**，
    所以 `_class` 一定要给对，否则这一条会被静默丢掉。

    `Opct` 是百分数（#Prc 单位浮点，100 = 全不透明），本项目的 opacity 是
    0~1 —— 换算在 core/psd_effects.py 里做，这里保持 PSD 原样。
    """
    e = dict(extra)
    classID = e.pop("_class", b"null")
    opacity = e.pop("_opacity", 100.0)
    props = {
        b"enab": dsc.Bool(bool(e.pop("enabled", True))),
        b"present": dsc.Bool(True),
        b"showInDialog": dsc.Bool(bool(e.pop("shown", True))),
        b"Opct": dsc.UnitFloat(float(opacity), b"#Prc"),
    }
    for k, v in e.items():
        props[_dkey(k)] = v
    return _desc(props, dsc, classID=classID)


def _lfxr_block(lay):
    """图层的图层样式块（lfx2 = OBJECT_BASED_EFFECTS_LAYER_INFO）。

    注意 `lfx2` 在 psd-tools 里注册成 **DescriptorBlock2**，也就是
    `version(H) + data_version(H=16) + Descriptor` —— 比普通 DescriptorBlock
    多一个 16，而且**结尾没有那 2 字节填充**。写错了会报
    "Invalid data section size"（后面全是垃圾字节）。

    （顺带一提：真正的 key 是 `lfx2`，不是 `lfxr` —— 后者不是合法 tag，
    psd-tools 会打一条 "Unknown key" 然后把整个块丢掉。）
    Descriptor 里 `masterFXSwitch` 是总开关，`Scl ` 是缩放 100%。
    """
    import io
    dsc = _d()
    props = {
        b"masterFXSwitch": dsc.Bool(True),
        b"Scl ": dsc.UnitFloat(100.0, b"#Prc"),
    }
    lst = dsc.List()
    for e in lay.get("effects") or []:
        e = dict(e)
        lst.append(_fx_item(dsc, e))
    props[b"objectBasedEffects"] = lst
    blk = dsc.DescriptorBlock2(version=0, data_version=16,
                               classID=b"null")
    for k, v in props.items():
        blk[dsc.String(k)] = v
    fp = io.BytesIO()
    blk.write(fp, padding=1)
    return fp.getvalue()


def _extra_blocks(lay):
    """图层的附加数据块（除 lsct 外）。"""
    out = b""
    if lay.get("text"):
        out += _tagged_block(b"TySh", _tysh_block(lay))
    if lay.get("effects"):
        out += _tagged_block(b"lfx2", _lfxr_block(lay))
    return out


def demo_effects(which="all"):
    """一组写好的 lfxr effect 片段（键名已经是 PSD 的）。

    `which` 可以是单个名字或 "all"。返回的每一项直接塞进图层的
    "effects": [...] 里。`Opct` 用百分数（100 = 全不透明）。

    **每项必须带 `_class`**：psd-tools 是靠 Descriptor 的 classID 判断这是
    哪个效果的，对不上就整条被静默跳过，`layer.effects` 返回空列表。

    ## 键名对照（全部来自 psd_tools.terminology.Key / Klass，别自己猜）

    | 效果 | classID | 大小 | 距离/位移 | 扩散 | 颜色 | 混合 |
    | --- | --- | --- | --- | --- | --- | --- |
    | 投影 DrSh | `DrSh` | `blur` | `Dstn` | `Ckmt` | `Clr ` | `Md  ` |
    | 内阴影 IrSh | `IrSh` | `blur` | `Dstn` | `Ckmt` | `Clr ` | `Md  ` |
    | 外发光 OrGl | `OrGl` | `blur` | — | `Ckmt` | `Clr ` | `Md  ` |
    | 内发光 IrGl | `IrGl` | `blur` | — | `Ckmt` | `Clr ` | `Md  ` |
    | 斜面浮雕 ebbl | `ebbl` | **`blur`** | — | — | `hglC`/`sdwC` | `hglM`/`sdwM` |
    | 光泽 ChFX | `ChFX` | `blur` | `Dstn` | `Ckmt` | `Clr ` | `Md  ` |
    | 颜色叠加 SoFi | `SoFi` | — | — | — | `Clr ` | `Md  ` |
    | 渐变叠加 GrFl | `GrFl` | — | — | — | `Grad` | `Md  ` |
    | 描边 FrFX | `FrFX` | `Sz  ` | — | — | `Clr ` | — |

    几个反直觉的地方（都是实测踩出来的）：

    * **斜面的大小读 `blur` 不是 `Sz  `，而描边的大小读 `Sz  `**。
      两个键不能互换，填错就静默变 0。
    * 斜面的高度角是 `Lald`、强度是 `srgR`（不是 `Dpth`）、方向是 `bvlD`
      （不是 `Drct`）、样式是 `bvlS`。`bevel_type` 读的是 technique
      （SfBL/PrBL 那套"怎么算"），**不是**"内斜面/浮雕"这种样式。
    * 光泽的角度也是 `lagl`（不是 `Angl`）。
    * **`Md  ` 里是全名**（`b'screen'` / `b'multiply'` / `b'normal'`），
      和图层记录里那套 4 字节 key（`b'scrn'` / `b'mul '`）不是一回事 ——
      psd-tools 的 `DESCRIPTOR_BLEND_MODES` 只认全名。
    * `Enumerated(typeID, enum)` 的**两个参数别搞反**：前者是 4 字节类型名
      （`BlnM` / `GrdT` / `bvlD`），后者才是取值。
    * `Clr ` 存的就是颜色描述符 `{Rd, Grn, Bl}` 本身，**不要再包一层
      `{Clr : ...}`** —— 包了颜色取不到，会静默退回默认值。
    """
    dsc = _d()

    def solid(rgb):
        return _fx_solid(rgb, dsc)

    def grad(colors, style=b"Lnr ", angle=90.0):
        return _fx_gradient(colors, dsc, style, angle)

    out = {
        # 投影：uglg=0 用局部角 lagl；Dstn=距离，blur=模糊，Ckmt=扩散
        "drop_shadow": {
            "_class": b"DrSh",
            "Md  ": dsc.Enumerated(b"BlnM", b"multiply"),
            "Clr ": solid((0, 0, 0)),
            "uglg": dsc.Bool(False),
            "lagl": dsc.UnitFloat(120.0, b"#Ang"),
            "Dstn": dsc.UnitFloat(9.0, b"#Pxl"),
            "blur": dsc.UnitFloat(7.0, b"#Pxl"),
            "Ckmt": dsc.UnitFloat(3.0, b"#Pxl"),
            "Nose": dsc.UnitFloat(0.0, b"#Prc"),
            "_opacity": 75.0,
        },
        # 内阴影
        "inner_shadow": {
            "_class": b"IrSh",
            "Md  ": dsc.Enumerated(b"BlnM", b"multiply"),
            "Clr ": solid((10, 20, 30)),
            "uglg": dsc.Bool(True),
            "lagl": dsc.UnitFloat(30.0, b"#Ang"),
            "Dstn": dsc.UnitFloat(4.0, b"#Pxl"),
            "blur": dsc.UnitFloat(5.0, b"#Pxl"),
            "Ckmt": dsc.UnitFloat(0.0, b"#Pxl"),
            "_opacity": 60.0,
        },
        # 外发光：solid 表示发光颜色来源是纯色（GlowColor）
        "outer_glow": {
            "_class": b"OrGl",
            "Md  ": dsc.Enumerated(b"BlnM", b"screen"),
            "Clr ": solid((255, 230, 120)),
            "blur": dsc.UnitFloat(14.0, b"#Pxl"),
            "Ckmt": dsc.UnitFloat(2.0, b"#Pxl"),
            "Nose": dsc.UnitFloat(0.0, b"#Prc"),
            "_opacity": 80.0,
        },
        "inner_glow": {
            "_class": b"IrGl",
            "Md  ": dsc.Enumerated(b"BlnM", b"screen"),
            "Clr ": solid((255, 255, 200)),
            "glwS": dsc.Enumerated(b"GlwS", b"SrcE"),
            "blur": dsc.UnitFloat(9.0, b"#Pxl"),
            "Ckmt": dsc.UnitFloat(1.0, b"#Pxl"),
            "_opacity": 70.0,
        },
        # 斜面浮雕：bvlS=样式、Drct=方向、alt =高度角、Dpth=强度、Sftn=柔化
        "bevel_emboss": {
            "_class": b"ebbl",
            "bvlS": dsc.Enumerated(b"bvlS", b"InrB"),
            "bvlD": dsc.Enumerated(b"bvlD", b"In  "),
            "bvlT": dsc.Enumerated(b"bvlT", b"SfBL"),
            "blur": dsc.UnitFloat(6.0, b"#Pxl"),
            "Lald": dsc.UnitFloat(45.0, b"#Ang"),
            "srgR": dsc.UnitFloat(150.0, b"#Prc"),
            "Sftn": dsc.UnitFloat(4.0, b"#Pxl"),
            "uglg": dsc.Bool(True),
            "lagl": dsc.UnitFloat(90.0, b"#Ang"),
            "hglM": dsc.Enumerated(b"BlnM", b"screen"),
            "hglC": solid((255, 255, 255)),
            "hglO": dsc.UnitFloat(80.0, b"#Prc"),
            "sdwM": dsc.Enumerated(b"BlnM", b"multiply"),
            "sdwC": solid((0, 0, 0)),
            "sdwO": dsc.UnitFloat(65.0, b"#Prc"),
            "AntA": dsc.Bool(True),
            "_opacity": 100.0,
        },
        # 光泽：Dstn=距离、blur=大小、Invr=反相、Ckmt=扩散
        "satin": {
            "_class": b"ChFX",
            "Md  ": dsc.Enumerated(b"BlnM", b"multiply"),
            "Clr ": solid((40, 40, 90)),
            "lagl": dsc.UnitFloat(90.0, b"#Ang"),
            "Dstn": dsc.UnitFloat(15.0, b"#Pxl"),
            "blur": dsc.UnitFloat(11.0, b"#Pxl"),
            "Ckmt": dsc.UnitFloat(0.0, b"#Pxl"),
            "Invr": dsc.Bool(True),
            "AntA": dsc.Bool(True),
            "_opacity": 55.0,
        },
        "color_overlay": {
            "_class": b"SoFi",
            "Md  ": dsc.Enumerated(b"BlnM", b"normal"),
            "Clr ": solid((200, 40, 90)),
            "_opacity": 50.0,
        },
        "gradient_overlay": {
            "_class": b"GrFl",
            "Md  ": dsc.Enumerated(b"BlnM", b"normal"),
            "Grad": grad([(255, 240, 0), (0, 80, 255)], b"Lnr ", 45.0),
            "Angl": dsc.UnitFloat(45.0, b"#Ang"),
            "Type": dsc.Enumerated(b"Lnr ", b"GrdT"),
            "Rvrs": dsc.Bool(False),
            "Dthr": dsc.Bool(False),
            "Algn": dsc.Bool(True),
            "Scl ": dsc.UnitFloat(100.0, b"#Prc"),
            "_opacity": 60.0,
        },
        # 描边：Styl=位置(InsF 内 / OutF 外 / CtrF 居中)、PntT=填充类型(SClr)
        "stroke": {
            "_class": b"FrFX",
            "Styl": dsc.Enumerated(b"FStl", b"OutF"),
            "PntT": dsc.Enumerated(b"FrFl", b"SClr"),
            "Clr ": solid((255, 255, 255)),
            "Sz  ": dsc.UnitFloat(4.0, b"#Pxl"),
            "_opacity": 90.0,
        },
    }
    if which == "all":
        return out
    return [out[which]]


def write_psd(path, width, height, layers, background=(255, 255, 255)):
    """写一个 PSD。

    layers: 底 -> 顶。每项是一个 dict：
        name / left / top / right / bottom / rgb=(r,g,b)
        blend=b'norm' / opacity=255 / clipping=0 / visible=True
        mask: {"left","top","right","bottom","gray"} —— 可选图层蒙版
        section: SECT_BOUND 表示组头（在子图层之前），
                 SECT_CLOSED / SECT_OPEN 表示组尾标记

        text: 文字层（写 TySh 块）——
            {"content": "Hi", "size": 48, "color": (255,0,0),
             "family": "Arial", "tracking": 20, "justification": 2,
             "start_indent": 10, "auto_leading": 1.2, ...}
            段落用 "paras": [(文本, {段落属性})]、字符用
            "runs": [(文本, {字符属性})] 覆盖单段的写法；
            "fonts": [{"name","family","style"}] 决定 FontSet 条目。

        effects: 图层样式（写 lfxr 块）—— 一组**已经转成 PSD 键名**的
            descriptor 片段，Opct 用百分数。见 `demo_effects()`。
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
