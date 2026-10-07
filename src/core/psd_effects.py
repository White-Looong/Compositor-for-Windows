# -*- coding: utf-8 -*-
"""PSD 图层样式（lfxr / 旧式 lrFX）-> 本项目的 effects dict。

psd-tools 1.23 已经把 `layer.effects` 解析成带属性的对象（`api/effects.py`），
所以这里只做**语义映射 + 单位换算**，不碰二进制。

映射与换算
  不透明度   PSD 是百分数（0~100），本项目是 0~1
  模糊 size  PSD 是像素，本项目也是像素（`_blur_reach` 按 4σ 处理）
  距离 distance / 描边宽度 同为像素
  角度      PSD 用"0 = 正右、顺时针"，本项目 `effects._offset` 也是
            `cos/sin` 同一套约定，所以直接给，不用翻转
  全局光    `use_global_light` 映射到本项目的 `use_global`

**接不进来的**（本项目没有对应参数，硬塞会渲错）：图案填充的位图本身
（`Ptrn` 图案 ID）、渐变的中间色标与抖动、自定义轮廓、`Satin` 之外的
contour。这些都只取"能表达的那部分"，取不到就退到相近的近似值。
"""

from __future__ import annotations

from .blend import BLEND_MODES
from .effects import GLOBAL_LIGHT, default_effects, has_effects

# psd_tools 的 BlendMode 名 -> 本项目 BLEND_MODES 里的名字
_BLEND_MAP = {
    "NORMAL": "Normal",
    "DISSOLVE": "Dissolve",
    "DARKEN": "Darken",
    "MULTIPLY": "Multiply",
    "COLOR_BURN": "Color Burn",
    "LINEAR_BURN": "Linear Burn",
    "DARKER_COLOR": "Darker Color",
    "LIGHTEN": "Lighten",
    "SCREEN": "Screen",
    "COLOR_DODGE": "Color Dodge",
    "LINEAR_DODGE": "Linear Dodge (Add)",
    "LIGHTER_COLOR": "Lighter Color",
    "OVERLAY": "Overlay",
    "SOFT_LIGHT": "Soft Light",
    "HARD_LIGHT": "Hard Light",
    "VIVID_LIGHT": "Vivid Light",
    "LINEAR_LIGHT": "Linear Light",
    "PIN_LIGHT": "Pin Light",
    "HARD_MIX": "Hard Mix",
    "DIFFERENCE": "Difference",
    "EXCLUSION": "Exclusion",
    "SUBTRACT": "Subtract",
    "DIVIDE": "Divide",
    "HUE": "Hue",
    "SATURATION": "Saturation",
    "COLOR": "Color",
    "LUMINOSITY": "Luminosity",
}

_GRADIENT_MAP = {
    "LINEAR": "线性",
    "RADIAL": "径向",
    "ANGLE": "角度",
    "REFLECTED": "对称",
    "DIAMOND": "菱形",
}

_BEVEL_STYLE_MAP = {
    "OUTER_BEVEL": "外斜面",
    "INNER_BEVEL": "内斜面",
    "EMBOSS": "浮雕",
    "PILLOW_EMBOSS": "枕状浮雕",
}

_STROKE_POS_MAP = {
    "OUTSIDE": "外部",
    "INSIDE": "内部",
    "CENTER": "居中",
}


def _f(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _blend(mode, default="Normal"):
    name = _BLEND_MAP.get(str(getattr(mode, "name", mode) or "").upper())
    if name and name in BLEND_MODES:
        return name
    return default


def _rgb(color):
    """PSD 的颜色 descriptor -> [r,g,b] 0~255。

    descriptor 里的 'Rd  ' / 'Grn ' / 'Bl  ' 是 0~255 的双精度。
    也接受 psd-tools 的 `_ColorMixin.color` 返回的那种 Descriptor。
    """
    if color is None:
        return None
    try:
        rd = _f(color.get(b"Rd  "), -1.0)
        gr = _f(color.get(b"Grn "), -1.0)
        bl = _f(color.get(b"Bl  "), -1.0)
    except Exception:
        return None
    if rd < 0 or gr < 0 or bl < 0:
        return None
    return [int(_clamp(rd, 0, 255)), int(_clamp(gr, 0, 255)),
            int(_clamp(bl, 0, 255))]


def _on(e):
    """效果是不是"开着"。

    PSD 有两个坑：`present` 说这个效果在列表里，`showInDialog` 说它在
    面板里可见（用户临时关掉时 `enabled` 仍为 True）。要两个都真开。
    """
    try:
        if not e.present:
            return False
    except Exception:
        pass
    try:
        if not e.enabled:
            return False
    except Exception:
        return False
    try:
        if not e.shown:
            return False
    except Exception:
        pass
    return True


def _angle(e, default=45.0):
    """角度。use_global_light 时 psd-tools 已给出全局角，直接用。"""
    try:
        if e.use_global_light:
            return _f(e.angle, default) % 360.0
    except Exception:
        pass
    try:
        return _f(e.angle, default) % 360.0
    except Exception:
        return default


def _shadow_distance(e):
    """阴影的"距离"（位移），PSD 存在阴影专属的 `Dstn` 键里。

    psd-tools 的 `_ShadowEffect` 没有把它提成属性（`_ChokeNoiseMixin` 的
    `choke` 是扩散量，不是位移），所以直接读原始 descriptor。
    """
    try:
        return _f(e.descriptor.get(b"Dstn"), 0.0)
    except Exception:
        return 0.0


def _choke(e):
    """本项目的 `spread` = PSD 的 **choke**（`Ckmt`，单位像素）。

    别用 psd-tools 的 `OuterGlow.spread` 那个属性：它读的是 `ShdN`
    （ShadingNoise，杂色），跟"扩散"完全不是一回事。
    语义对照见 core/effects.py：`_dilate(a2, spread)`。
    """
    try:
        return _f(e.choke, 0.0)
    except Exception:
        return 0.0


def _gradient_stops(grad):
    """从渐变 descriptor 里取首尾两个色标 -> ([r,g,b], [r,g,b])。

    本项目的渐变只有 color / color2 两端，PSD 的中间色标和抖动接不进来，
    所以取 Colors 列表的第一和最后。取不到就返回 (None, None)。
    """
    if grad is None:
        return (None, None)
    try:
        colors = grad.get(b"Clrs")
    except Exception:
        return (None, None)
    if colors is None:
        return (None, None)
    try:
        items = list(colors)
    except Exception:
        return (None, None)
    picks = []
    for c in (items[0] if items else None,
              items[-1] if len(items) > 1 else None):
        if c is None:
            continue
        try:
            loc = c.get(b"Clr ")
        except Exception:
            loc = None
        rgb = _rgb(loc)
        if rgb:
            picks.append(rgb)
    if not picks:
        return (None, None)
    if len(picks) == 1:
        return (picks[0], picks[0])
    return (picks[0], picks[1])


def _iter_effects(layer):
    """遍历图层的效果列表，坏数据直接跳过而不是炸掉整个导入。"""
    try:
        fx = layer.effects
    except Exception:
        return []
    try:
        return list(fx)
    except Exception:
        return []


def effects_from_layer(layer):
    """一个 psd-tools 图层 -> effects dict（core.effects 的格式）。

    没有任何可识别的、且开启着的效果时返回 None（调用方就不设 effects）。
    """
    out = default_effects()
    out[GLOBAL_LIGHT]["enabled"] = False
    touched = set()
    gl_angle = None
    gl_alt = None

    for e in _iter_effects(layer):
        try:
            name = type(e).__name__
        except Exception:
            continue
        if not _on(e):
            continue
        op = _clamp(_f(e.opacity, 100.0) / 100.0, 0.0, 1.0)

        try:
            if name == "DropShadow":
                d = out["drop_shadow"]
                d.update({
                    "enabled": True, "opacity": op,
                    "use_global": bool(e.use_global_light),
                    "angle": _angle(e, 45.0),
                    "distance": _shadow_distance(e),
                    "size": _f(e.size, 0.0),
                    "spread": _choke(e),
                })
                c = _rgb(getattr(e, "color", None))
                if c:
                    d["color"] = c
                touched.add("drop_shadow")

            elif name == "InnerShadow":
                d = out["inner_shadow"]
                d.update({
                    "enabled": True, "opacity": op,
                    "use_global": bool(e.use_global_light),
                    "angle": _angle(e, 45.0),
                    "distance": _shadow_distance(e),
                    "size": _f(e.size, 0.0),
                    "spread": _choke(e),
                })
                c = _rgb(getattr(e, "color", None))
                if c:
                    d["color"] = c
                touched.add("inner_shadow")

            elif name == "OuterGlow":
                d = out["outer_glow"]
                d.update({"enabled": True, "opacity": op,
                          "size": _f(e.size, 0.0),
                          "spread": _choke(e)})
                c = _rgb(getattr(e, "color", None))
                if c:
                    d["color"] = c
                touched.add("outer_glow")

            elif name == "InnerGlow":
                d = out["inner_glow"]
                d.update({"enabled": True, "opacity": op,
                          "blend": _blend(e.blend_mode, "Screen"),
                          "size": _f(e.size, 0.0),
                          "spread": _choke(e)})
                c = _rgb(getattr(e, "color", None))
                if c:
                    d["color"] = c
                touched.add("inner_glow")

            elif name == "BevelEmboss":
                d = out["bevel_emboss"]
                d.update({
                    "enabled": True,
                    # 注意：psd-tools 的 `bevel_type` 返回的是 BevelTechnique
                    # （SfBL/PrBL 那套"怎么算"），"内斜面/浮雕"这种**样式**
                    # 才是 `bevel_style`。认错字段会永远落到默认值。
                    "style": _BEVEL_STYLE_MAP.get(
                        getattr(e.bevel_style, "name", ""), "内斜面"),
                    "direction": ("上" if getattr(e.direction, "name", "") ==
                                  "STAMP_IN" else "下"),
                    "size": _f(e.size, 0.0),
                    "soften": _f(e.soften, 0.0),
                    "depth": _clamp(_f(e.depth, 0.0) / 100.0, 0.0, 10.0),
                    "use_global": bool(e.use_global_light),
                    "angle": _angle(e, 45.0),
                    "altitude": _clamp(_f(e.altitude, 30.0), 0.0, 90.0),
                    "highlight_opacity": _clamp(
                        _f(e.highlight_opacity, 100.0) / 100.0, 0.0, 1.0),
                    "shadow_opacity": _clamp(
                        _f(e.shadow_opacity, 100.0) / 100.0, 0.0, 1.0),
                })
                c = _rgb(e.highlight_color)
                if c:
                    d["highlight_color"] = c
                c = _rgb(e.shadow_color)
                if c:
                    d["shadow_color"] = c
                try:
                    if e.use_global_light:
                        gl_angle = _f(e.angle, 45.0) % 360.0
                        gl_alt = _clamp(_f(e.altitude, 30.0), 0.0, 90.0)
                except Exception:
                    pass
                touched.add("bevel_emboss")

            elif name == "Satin":
                d = out["satin"]
                d.update({
                    "enabled": True, "opacity": op,
                    "blend": _blend(e.blend_mode, "Multiply"),
                    "angle": _clamp(_f(e.angle, 90.0) % 360.0, 0.0, 360.0),
                    "distance": _f(e.distance, 0.0),
                    "size": _f(e.size, 0.0),
                    "invert": bool(e.inverted),
                })
                touched.add("satin")

            elif name == "ColorOverlay":
                d = out["color_overlay"]
                c = _rgb(getattr(e, "color", None))
                if not c:
                    continue
                d.update({"enabled": True, "opacity": op,
                          "blend": _blend(e.blend_mode, "Normal"),
                          "color": c})
                touched.add("color_overlay")

            elif name == "GradientOverlay":
                d = out["gradient_overlay"]
                c0, c1 = _gradient_stops(getattr(e, "gradient", None))
                if not c0:
                    continue
                d.update({
                    "enabled": True, "opacity": op,
                    "blend": _blend(e.blend_mode, "Normal"),
                    "style": _GRADIENT_MAP.get(
                        getattr(e.type, "name", ""), "线性"),
                    "angle": _clamp(_f(e.angle, 90.0) % 360.0, 0.0, 360.0),
                    "color": c0, "color2": c1 or c0,
                })
                touched.add("gradient_overlay")

            elif name == "Stroke":
                d = out["stroke"]
                pos = _STROKE_POS_MAP.get(getattr(e.position, "name", ""),
                                          "外部")
                fill = getattr(e.fill_type, "name", "")
                c = None
                if fill == "SOLID_COLOR":
                    c = _rgb(getattr(e, "color", None))
                else:
                    # 渐变 / 图案填充的描边：取渐变首色当纯色，视觉上最接近
                    c0, _c1 = _gradient_stops(getattr(e, "gradient", None))
                    c = c0
                if not c:
                    continue
                d.update({"enabled": True, "opacity": op,
                          "size": _f(e.size, 0.0), "position": pos,
                          "color": c})
                touched.add("stroke")
        except Exception:
            # 单个效果读不出来不能连累整层，跳过它
            continue

    if not touched:
        return None

    # 全局光：把用到 use_global 的那个角度记进全局光设置
    if gl_angle is not None:
        out[GLOBAL_LIGHT].update({
            "enabled": True, "angle": gl_angle,
            "altitude": gl_alt if gl_alt is not None else 30.0,
        })

    # 把没碰到的效果关掉（default_effects 里就是 enabled=False，
    # 这里再显式确认一次，免得以后改了默认值这里跟着出错）
    for k, v in out.items():
        if k == GLOBAL_LIGHT:
            continue
        v["enabled"] = k in touched

    return out if has_effects(out) else None
