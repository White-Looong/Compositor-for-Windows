# -*- coding: utf-8 -*-
"""生成一组覆盖各种语法的 SVG，供 selftest / uitest / screenshot 用。

SVG 的合法文本到处都是，但"覆盖到项目解析器分支"的那些语法并不常见 ——
这里把 `core/svg_import.py` 每条支路都喂一份：基本形状、transform、渐变、
描边、弧、文本、clip、use、内联 style、透明度、class/id 引用。

    from tools.svgfixture import ALL_SVG, write_svgs

样例都是**小图**（200~300 px），别拿去做大图性能基准。
"""

from __future__ import annotations

import os

NS = 'xmlns="http://www.w3.org/2000/svg"'

SHAPES = '''<svg %(ns)s width="200" height="200" viewBox="0 0 200 200">
  <rect x="10" y="10" width="60" height="40" fill="#e74c3c"/>
  <rect x="80" y="10" width="60" height="40" fill="#3498db" rx="10" ry="6"/>
  <circle cx="40" cy="100" r="28" fill="#2ecc71"/>
  <ellipse cx="120" cy="100" rx="45" ry="22" fill="#f1c40f" fill-opacity="0.8"/>
  <line x1="10" y1="150" x2="190" y2="150" stroke="#8e44ad" stroke-width="6"/>
  <polyline points="10,180 40,165 70,185 100,160" fill="none" stroke="#16a085" stroke-width="3"/>
  <polygon points="130,155 190,155 190,190 130,190" fill="#e67e22" fill-opacity="0.6"/>
  <path d="M10 190 L100 190 L100 200 L10 200 Z" fill="#34495e"/>
</svg>'''

TRANSFORMS = '''<svg %(ns)s width="200" height="200" viewBox="0 0 200 200">
  <g transform="translate(20,20)">
    <rect x="0" y="0" width="40" height="40" fill="#2980b9"/>
  </g>
  <g transform="translate(90,20) scale(2)">
    <circle cx="0" cy="0" r="20" fill="#c0392b"/>
  </g>
  <g transform="rotate(30 100 100)">
    <rect x="80" y="90" width="40" height="40" fill="#27ae60"/>
  </g>
  <g transform="translate(20,120) skewX(20)">
    <rect x="0" y="0" width="60" height="25" fill="#8e44ad"/>
  </g>
  <g transform="translate(90,130) skewY(-15)">
    <polygon points="0,0 50,0 25,40" fill="#f39c12"/>
  </g>
  <g transform="matrix(1 0 0 1 0 60)">
    <circle cx="30" cy="20" r="15" fill="#7f8c8d"/>
  </g>
</svg>'''

GRADIENTS = '''<svg %(ns)s width="240" height="120" viewBox="0 0 240 120">
  <defs>
    <linearGradient id="lg">
      <stop offset="0" stop-color="#0000ff"/>
      <stop offset="0.5" stop-color="#00ff00"/>
      <stop offset="1" stop-color="#ff0000"/>
    </linearGradient>
    <linearGradient id="lg2" gradientUnits="userSpaceOnUse" x1="10" y1="10"
                    x2="110" y2="10">
      <stop offset="0" stop-color="#ffffff" stop-opacity="0"/>
      <stop offset="1" stop-color="#0000ff" stop-opacity="1"/>
    </linearGradient>
    <radialGradient id="rg">
      <stop offset="0" stop-color="#ffffff"/>
      <stop offset="1" stop-color="#e91e63"/>
    </radialGradient>
    <radialGradient id="rg2" gradientUnits="userSpaceOnUse" cx="70" cy="60"
                    r="50" fx="55" fy="45">
      <stop offset="0" stop-color="#ffeb3b"/>
      <stop offset="1" stop-color="#3f51b5"/>
    </radialGradient>
  </defs>
  <rect x="10" y="10" width="100" height="100" fill="url(#lg)"/>
  <rect x="130" y="10" width="100" height="100" fill="url(#lg2)"/>
  <circle cx="60" cy="60" r="50" fill="url(#rg)"/>
  <circle cx="190" cy="60" r="50" fill="url(#rg2)"/>
</svg>'''

STROKES = '''<svg %(ns)s width="240" height="120" viewBox="0 0 240 120">
  <path d="M10 20 L110 20" stroke="#000" stroke-width="10" stroke-linecap="butt"/>
  <path d="M10 45 L110 45" stroke="#000" stroke-width="10" stroke-linecap="round"/>
  <path d="M10 70 L110 70" stroke="#000" stroke-width="10" stroke-linecap="square"/>
  <polyline points="130,10 180,60 130,110" fill="none" stroke="#c0392b"
            stroke-width="8" stroke-linejoin="round"/>
  <polyline points="180,10 230,60 180,110" fill="none" stroke="#2980b9"
            stroke-width="8" stroke-linejoin="bevel"/>
  <path d="M10 100 L230 100" stroke="#16a085" stroke-width="4"
        stroke-dasharray="12 6 3 3"/>
  <path d="M130 100 L230 100" stroke="#8e44ad" stroke-width="4"
        stroke-dasharray="8 4"/>
</svg>'''

ARCS = '''<svg %(ns)s width="240" height="140" viewBox="0 0 240 140">
  <path d="M10 120 A40 40 0 0 1 90 120 Z" fill="#e74c3c"/>
  <path d="M110 120 A45 25 0 1 0 200 120 Z" fill="#3498db" fill-opacity="0.7"/>
  <path d="M10 30 A30 30 0 0 0 70 30" fill="none" stroke="#2ecc71" stroke-width="5"/>
  <path d="M100 30 L100 30 A25 25 0 1 1 150 80 L100 80 Z" fill="#f1c40f"/>
</svg>'''

# 离屏 Qt 只有两三个字体，字体名故意写系统里一定有的（或干脆不写）
TEXT = '''<svg %(ns)s width="240" height="140" viewBox="0 0 240 140">
  <text x="20" y="40" font-size="28" fill="#c0392b">Compositor</text>
  <text x="20" y="80" font-size="24" fill="#2980b9" text-anchor="middle"
        font-weight="bold">ABC 123</text>
  <text x="220" y="120" font-size="22" fill="#16a085" text-anchor="end"
        style="fill:#8e44ad;letter-spacing:2">End</text>
</svg>'''

CLIP = '''<svg %(ns)s width="200" height="200" viewBox="0 0 200 200">
  <defs>
    <clipPath id="circleClip">
      <circle cx="100" cy="100" r="80"/>
    </clipPath>
    <clipPath id="rectClip">
      <rect x="40" y="40" width="120" height="120"/>
    </clipPath>
  </defs>
  <g clip-path="url(#circleClip)">
    <rect x="0" y="0" width="200" height="200" fill="#9b59b6"/>
    <circle cx="60" cy="150" r="50" fill="#f1c40f"/>
  </g>
  <g clip-path="url(#rectClip)">
    <circle cx="150" cy="60" r="40" fill="#2ecc71"/>
  </g>
</svg>'''

USE = '''<svg %(ns)s width="200" height="200" viewBox="0 0 200 200">
  <defs>
    <circle id="dot" cx="0" cy="0" r="12" fill="#e74c3c"/>
    <rect id="sq" x="-10" y="-10" width="20" height="20" fill="#3498db"/>
  </defs>
  <use href="#dot" x="40" y="40"/>
  <use href="#dot" x="100" y="40" transform="translate(0,0) scale(1.5)"
       style="fill:#2ecc71"/>
  <use href="#sq" x="40" y="100"/>
  <use xlink:href="#dot" x="100" y="100"
       xmlns:xlink="http://www.w3.org/1999/xlink"/>
</svg>'''

STYLE = '''<svg %(ns)s width="200" height="120" viewBox="0 0 200 120">
  <rect x="10" y="10" width="80" height="40" style="fill:#e91e63;stroke:#000;stroke-width:3"/>
  <rect x="110" y="10" width="80" height="40" fill="#e91e63"
        style="fill-opacity:0.4;stroke:#000;stroke-width:2"/>
  <g opacity="0.5">
    <circle cx="50" cy="90" r="25" fill="#8e44ad"/>
    <circle cx="150" cy="90" r="25" fill="#16a085"/>
  </g>
  <path d="M10 100 L190 100" stroke="#000" stroke-width="2" opacity="0.3"/>
</svg>'''

EMPTY = '''<svg %(ns)s viewBox="0 0 100 60"/>'''

BASIC_PATH = '''<svg %(ns)s width="120" height="120" viewBox="0 0 120 120">
  <path d="M10 10 H110 V110 H10 Z" fill="#34495e"/>
  <path d="M60 20 C20 60 100 60 60 100" fill="none" stroke="#e74c3c" stroke-width="4"/>
  <path d="M60 10 Q30 60 60 110 T60 120" fill="none" stroke="#2980b9" stroke-width="3"/>
  <path d="M20 60 S60 20 100 60" fill="none" stroke="#27ae60" stroke-width="3"/>
</svg>'''

ALL_SVG = {
    "shapes": SHAPES % {"ns": NS},
    "transforms": TRANSFORMS % {"ns": NS},
    "gradients": GRADIENTS % {"ns": NS},
    "strokes": STROKES % {"ns": NS},
    "arcs": ARCS % {"ns": NS},
    "text": TEXT % {"ns": NS},
    "clip": CLIP % {"ns": NS},
    "use": USE % {"ns": NS},
    "style": STYLE % {"ns": NS},
    "empty": EMPTY % {"ns": NS},
    "path": BASIC_PATH % {"ns": NS},
}


def write_svgs(directory, only=None):
    """把样例落盘，返回 [(文件名, 内容)]。"""
    os.makedirs(directory, exist_ok=True)
    out = []
    for name, xml in ALL_SVG.items():
        if only and name not in only:
            continue
        p = os.path.join(directory, name + ".svg")
        with open(p, "w", encoding="utf-8") as f:
            f.write(xml)
        out.append((p, xml))
    return out


def save_as_ppm(name, arr, directory):
    """测试里写图用不出图，这里只是留个口子给截图脚本。"""
    path = os.path.join(directory, name + ".png")
    try:
        from PIL import Image
        Image.fromarray(arr, "RGBA").save(path)
    except Exception:
        return None
    return path


if __name__ == "__main__":
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")
    for p, _ in write_svgs(d):
        print(p)
