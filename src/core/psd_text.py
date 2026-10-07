# -*- coding: utf-8 -*-
"""PSD 文字层（TySh）-> 本项目文字参数。

单独一个模块，而不是塞进 psd_import：这里的逻辑全是"读 psd-tools 的
TypeSetting 对象 + 换算单位"，与"图层树怎么映射"是两件事，好单测。

能还原的（core.text 的参数模型覆盖得到）：
  字体名 / 字号 / 粗 / 斜 / 下划线 / 填充色 / 字距 / 行距 / 对齐 /
  左·右·首行缩进 / 段前·段后距

**刻意不还原的**（core.text 没有对应参数，硬塞会渲出错误结果）：
  * 描边（PSD 的 stroke_flag）——文字描边是另一套渲染
  * 同一层里多样式混排（本项目的字体/字号/颜色是整层一套的）
  * 变形文字（warp）、沿路径排版
遇到上面这些就返回 None，由调用方降级成位图 —— 画面是准的。

单位换算（这是最容易出错的地方）：
  * 字号：PSD 存**点**（pt），本项目存**像素**。按 72dpi 文档算 1pt = 1px，
    所以先按 1:1 渲染，再由 `ink_scale()` 按实际墨迹范围校正 —— 见那里
  * 字距 tracking：PSD 存**千分之一 em**（1.0 = 1/1000 em），
    换像素要乘字号：`px = tracking / 1000 * size_px`
  * 缩进 / 段距：也是 pt，同样 1:1
"""

from __future__ import annotations

import numpy as np
from PySide6.QtGui import QFontDatabase

from .text import (ALIGN_CENTER, ALIGN_JUSTIFY, ALIGN_LEFT, ALIGN_RIGHT,
                   default_text_params, ensure_fonts, render_text)

# PostScript 名里的常见修饰后缀。剥掉它们才拿得到"字体家族名"。
# 例："Arial-BoldItalicMT" -> "Arial"、"SourceHanSansSC-Bold" -> "SourceHanSansSC"
#
# **别把 "Roman" / "Book" 放进来**：它们是 TimesNew**Roman**、BookmanOldStyle
# 这类**家族名的一部分**，剥掉就变成 "TimesNew"。判据是"只出现在尾部、
# 且剥完还剩下像家族名的部分"，不是"看起来像字体风格"。
_PS_SUFFIXES = ("BoldItalic", "BoldOblique", "SemiBold", "ExtraBold",
                "LightItalic", "BlackItalic", "ThinItalic", "Light",
                "Italic", "Oblique", "Bold", "Regular", "Medium",
                "Black", "Heavy", "Thin", "UltraLight", "Demi", "PS")

_PS_TAIL = ("MT", "PS", "Std", "DF", "WGL", "BT", "PSMT")


def _norm(s):
    return "".join(ch for ch in str(s or "") if ch.isalnum()).lower()


def _strip_style_suffix(s):
    """剥掉尾部的样式后缀与老式缩写。

    "SourceHanSansSCBold" -> "SourceHanSansSC"
    "BoldItalicMT"        -> ""（整段都是样式，直接空掉）

    **要循环剥**：一个名字里可能叠着好几个（"BoldItalic" + "MT"），
    只剥一层的话 "Arial-BoldItalicMT" 会剩下 "Bold"，
    而 "Bold" 比 "Arial" 短一点点才没被选中 —— 换一批字体就会翻车。
    """
    changed = True
    while changed and s:
        changed = False
        for suf in _PS_SUFFIXES:
            if len(s) > len(suf) and s.endswith(suf):
                s = s[:-len(suf)]
                changed = True
                break
        for tail in _PS_TAIL:
            if len(s) > len(tail) and s.endswith(tail):
                s = s[:-len(tail)]
                changed = True
                break
    return s


def _strip_ps(name):
    """从 PostScript 名里剥出尽可能干净的字体家族名。

    三步：
    1. 按 `-` 切开，**逐段**先剥样式后缀，再取最长的一段当家族名
       （"Arial-BoldMT" 的第二段会先变成 "Bold" → 认出它纯是样式 → 只剩
       第一段 "Arial" 可选）
    2. 认不出的段（比如 "MyriadPro-Regular" 的 "Regular"）直接忽略
    3. 全部不像样式但仍然取最长 —— 宁可长一点，也别把家族名截断

    别按长度直接排序原始分段：那样 "Arial-BoldMT" 会选中更长的
    "BoldMT"，结果是"字体叫 Bold"。**先剥后缀再比长度**。
    """
    s = str(name or "").strip()
    if not s:
        return ""
    parts = [p for p in s.split("-") if p]
    if len(parts) > 1:
        cands = []
        for p in parts:
            q = _strip_style_suffix(p)
            if len(q) >= 2:
                cands.append(q)
        if cands:
            cands.sort(key=len, reverse=True)
            s = cands[0]
    return _strip_style_suffix(s)


def _style_flags(style, postscript):
    """从字体样式名里判断粗 / 斜。"""
    t = ("%s %s" % (style or "", postscript or "")).lower()
    bold = ("bold" in t) or ("black" in t) or ("heavy" in t) or ("semibold" in t)
    italic = ("italic" in t) or ("oblique" in t)
    return bold, italic


def match_family(candidates):
    """在系统字体里找一个能用的家族名。找不到返回 ""。

    `candidates` 是按"可信度从高到低"排好的候选名（PostScript 名、family 字段、
    剥掉样式后缀的名字……）。两级匹配：

    1. **归一化精确匹配** —— 忽略大小写与非字母数字，因为 PSD 里的名字和
       Windows 字体名常常差一个连字符或大小写（"SourceHanSansSC" vs
       "Source Han Sans SC"）
    2. **包含匹配** —— 一方是另一方的子串就认。跨语言字体名只能靠这一步
       （"思源黑体" vs "SourceHanSansSC" 匹配不上，但 "Myriad Pro" 能
       命中 "MyriadPro-Regular"）

    实在找不到就返回 ""，由调用方决定降级 —— 宁可退回 PSD 的合成位图，
    也不要拿一个不相干的字体重排，画面会明显跑偏。
    """
    ensure_fonts()
    try:
        fams = list(QFontDatabase.families())
    except Exception:
        fams = []
    if not fams:
        return ""
    by_norm = {}
    for f in fams:
        by_norm.setdefault(_norm(f), f)
    cands = [c for c in (_norm(c) for c in candidates) if c]
    for c in cands:
        hit = by_norm.get(c)
        if hit:
            return hit
    # 第二级：包含。取最长命中的那个，避免 "Arial" 抢走 "Arial Narrow"
    best = None
    for c in cands:
        for f in fams:
            nf = _norm(f)
            if len(nf) < 3 or len(c) < 3:
                continue
            if c in nf or nf in c:
                if best is None or len(nf) > len(_norm(best)):
                    best = f
    return best or ""


def _color_rgb(argb):
    """PSD 的 ARGB（各分量 0~1 的 float 元组）-> [r,g,b] 0~255。"""
    if not argb:
        return None
    vals = [float(v) for v in argb]
    if len(vals) < 3:
        return None

    def q(v):
        return int(max(0, min(255, round(v * 255.0))))

    return [q(vals[1]), q(vals[2]), q(vals[3])]


def color_from_engine(engine_dict):
    """从原始 EngineDict 里取第一个 run 的填充色 -> [r,g,b]。

    **为什么不能只靠 `CharacterStyle.fill_color`**：psd-tools 那条路径读的是
    `FillColor/Values` 里一串平铺的 ARGB 数字，而 **Photoshop 实际写的是
    `FillColor/Clrs` —— 一个 [{Clr: {Rd, Grn, Bl}}] 的列表**。两者对不上时
    `fill_color` 安静地返回 None，文字就变成默认黑字，看不出是哪里错了。

    真实 PSD 一律走这个函数；`Values` 形态当作兼容（万一有别的导出器那么写）。
    """
    if not engine_dict:
        return None
    try:
        runs = engine_dict["StyleRun"]["RunArray"]
        sheet = runs[0]["StyleSheet"]["StyleSheetData"]
        fill = sheet["FillColor"]
    except Exception:
        return None
    if fill is None:
        return None
    # 形态 A（Photoshop 实际）：Clrs -> [{Clr : {Rd, Grn, Bl}}]
    try:
        clrs = fill["Clrs"]
        first = list(clrs)[0]
        col = first["Clr"]
        return [int(max(0, min(255, round(float(col["Rd"]))))),
                int(max(0, min(255, round(float(col["Grn"]))))),
                int(max(0, min(255, round(float(col["Bl"])))))]
    except Exception:
        pass
    # 形态 B：Values -> [a, r, g, b]
    try:
        v = list(fill["Values"])
        if len(v) >= 4:
            return [int(max(0, min(255, round(float(v[1]) * 255)))),
                    int(max(0, min(255, round(float(v[2]) * 255)))),
                    int(max(0, min(255, round(float(v[3]) * 255))))]
    except Exception:
        pass
    return None


def _justification(j):
    """psd_tools 的 Justification 枚举 -> core.text 的 align key。"""
    name = getattr(j, "name", "") or str(j)
    n = name.upper()
    if n.startswith("JUSTIFY_ALL"):
        return ALIGN_JUSTIFY
    if n.startswith("JUSTIFY_LAST_CENTER") or n == "CENTER":
        return ALIGN_CENTER
    if n.startswith("JUSTIFY_LAST_RIGHT") or n == "RIGHT":
        return ALIGN_RIGHT
    if n == "LEFT":
        return ALIGN_LEFT
    # JUSTIFY_LAST_LEFT：末行左对齐，语义上就是普通左对齐
    return ALIGN_LEFT


def _run_key(style):
    """判断两个 run 的"整层级"样式是否一致。

    只有整层级属性全一致的层才敢还原成单个文字图层；这些属性不一致就
    说明 PSD 里是混排，我们一个全局字体模型表达不了 -> 降级位图。
    """
    font = style.font
    return (
        _norm(font.postscript_name if font else ""),
        _norm(font.family if font else ""),
        round(float(style.font_size), 3),
        bool(style.faux_bold),
        bool(style.faux_italic),
        bool(style.underline),
        tuple(style.fill_color or ()),
    )


def text_params_from_typesetting(ts, engine_dict=None, strict_font=True):
    """psd-tools 的 TypeSetting -> 本项目的文字参数 dict。

    ts             `layer.typesetting`（有 typesetting 的 psd-tools 版本）
    engine_dict    `layer.engine_dict`，**只用来取填充色**（见 `color_from_engine`）；
                   没有也能跑，只是颜色退回黑
    strict_font=True 时要求字体能在系统里找到，否则返回 None（降级位图）。
    返回的 dict 已经过 `default_text_params` 补全，可以直接喂 render_text。
    """
    if ts is None:
        return None
    try:
        runs = list(ts.runs)
        paras = list(ts.paragraphs)
    except Exception:
        return None
    if not runs:
        return None

    content = str(getattr(ts, "text", "") or "")
    # PSD 用 \r 换行，本项目用 \n（core.text._normalize_content 也做，但这里
    # 提前处理能让 char_slots 的下标语义一致）
    content = content.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n")
    if not content.strip():
        return None

    # 混排检测：整层级样式不一致就别硬还原
    keys = {_run_key(r.style) for r in runs if r.text}
    if len(keys) > 1:
        return None

    st = runs[0].style
    font = st.font
    ps_name = st.font_name
    fam_field = (font.family if font else "") or ""
    style_field = (font.style if font else "") or ""
    b_style, i_style = _style_flags(style_field, ps_name)
    # 字体自带的粗斜（"Arial-Bold"）与 faux（描粗）合成一个开关
    bold = bool(st.faux_bold) or b_style
    italic = bool(st.faux_italic) or i_style

    family = match_family([fam_field, _strip_ps(ps_name), ps_name,
                           _strip_ps(fam_field)])
    if strict_font and not family:
        return None      # 字体不在系统里 -> 画面必然不对，降级位图

    try:
        size = float(st.font_size)
    except (TypeError, ValueError):
        size = 0.0
    if size <= 0.0:
        size = 1.0

    # 字距：PSD 是千分之一 em
    try:
        tracking = float(st.tracking)
    except (TypeError, ValueError):
        tracking = 0.0
    letter_spacing = tracking / 1000.0 * size

    # 行距：auto_leading 是相对倍数，直接就是 core.text 的 line_height
    p0 = paras[0].style if paras else None
    if p0 is not None:
        try:
            auto = float(p0.auto_leading)
        except (TypeError, ValueError):
            auto = 0.0
        if 0.05 <= auto <= 5.0:
            line_height = auto
        else:
            # 固定行距：leading 是点，转成相对 ascent+descent 的倍数很难
            # （本项目的分母是 QFontMetricsF 的 ascent+descent，PSD 不给）。
            # 保守取 1.2，视觉上最接近且不会错得离谱。
            line_height = 1.2
    else:
        line_height = 1.2

    para = {}
    if p0 is not None:
        def _g(name):
            try:
                return float(getattr(p0, name))
            except (TypeError, ValueError):
                return 0.0
        para = {
            "indent_left": _g("start_indent"),
            "indent_right": _g("end_indent"),
            "indent_first": _g("first_line_indent"),
            "space_before": _g("space_before"),
            "space_after": _g("space_after"),
        }

    color = (color_from_engine(engine_dict) or _color_rgb(st.fill_color)
             or [0, 0, 0])
    p = default_text_params(
        content=content,
        family=family,
        size=size,
        bold=bold,
        italic=italic,
        underline=bool(st.underline),
        color=color,
        line_height=line_height,
        letter_spacing=letter_spacing,
        align=_justification(p0.justification) if p0 is not None else ALIGN_LEFT,
        para=para,
    )
    return p


def ink_rect(arr, thresh=8):
    """位图里**实际有内容**的包围盒 (x0, y0, x1, y1)；空图返回 None。

    render_text 在四周留了 padding（给斜体 / 下划线 / 抗锯齿），位图尺寸
    不等于字的实际占位。跟 PSD 的 bbox 对齐必须用墨迹范围，
    否则字会整体偏小一圈。
    """
    if arr is None or arr.size == 0:
        return None
    a = arr[..., 3]
    ys, xs = np.nonzero(a > thresh)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def ink_scale(arr, box):
    """按 PSD 的 bbox 校正文字位图的缩放与偏移。

    `box` 是 PSD 图层的 (l, t, r, b)。返回 (sx, sy, cx, cy)：
    前两个是相对当前渲染结果的缩放（写进 layer.sx/sy），
    后两个是**画布坐标**下的图层中心（写进 layer.tx/ty）。

    为什么这样最稳：字号从 pt 换算成 px 一定有偏差（不同机器的字体度量、
    PSD 里存的是 72dpi 而显示分辨率未必如此），但只要"量出来的墨迹范围"
    对上 bbox，视觉位置和大小就是准的 —— 换算偏差全被 sx/sy 吸收掉了。
    """
    ink = ink_rect(arr)
    if box is None or ink is None:
        h, w = arr.shape[:2]
        return 1.0, 1.0, w / 2.0, h / 2.0
    l, t, r, b = box
    bw, bh = float(r - l), float(b - t)
    iw = float(ink[2] - ink[0])
    ih = float(ink[3] - ink[1])
    if iw <= 0.0 or ih <= 0.0 or bw <= 0.0 or bh <= 0.0:
        h, w = arr.shape[:2]
        return 1.0, 1.0, w / 2.0, h / 2.0
    sx = bw / iw
    sy = bh / ih
    # 位图几何中心 -> 墨迹中心的像素偏移。
    # 渲染时"位图中心落在图层中心 (tx,ty) 上"，所以位图里的墨迹中心
    # 落点是 `tx + dcx * sx`。要它正好等于 bbox 中心，就得**减掉**这个偏移：
    #     tx = bbox中心 - dcx * sx
    # 写成加号会把偏移叠加两次，文字会平移两倍距离（实测能偏 15 px）。
    h, w = arr.shape[:2]
    dcx = ((ink[0] + ink[2]) / 2.0) - (w / 2.0)
    dcy = ((ink[1] + ink[3]) / 2.0) - (h / 2.0)
    cx = (l + r) / 2.0 - dcx * sx
    cy = (t + b) / 2.0 - dcy * sy
    return sx, sy, cx, cy
