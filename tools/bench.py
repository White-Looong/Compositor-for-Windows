# -*- coding: utf-8 -*-
"""渲染性能基准。

    python -m tools.bench            # 全部场景
    python -m tools.bench composite  # 只跑某一个

优化前后跑同一个脚本对比数字，别靠感觉。**基线数字记在 docstring 里**，
改动渲染管线时顺手更新。

场景刻意分成两类：
  * `flat`  —— 一堆不透明大色块图层，轴对齐整数位置。测的是 composite 与
               免变换快路径（真实工程里最常见的一类：抠图 / 拼图）
  * `rot`   —— 带旋转与缩放、非整数位置、混合模式各异。测的是 warpAffine
  * `adj`   —— 调整层 + 组内调整层，测 render_from_snapshot 的覆盖面
"""

from __future__ import annotations

import argparse
import cProfile
import io
import os
import pstats
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np                                              # noqa: E402
from PySide6.QtWidgets import QApplication                      # noqa: E402

from src.core.adjust import default_params                       # noqa: E402
from src.core.document import (Document, LAYER_ADJUSTMENT,       # noqa: E402
                               make_image_layer)
from src.core.layer import Layer                                 # noqa: E402

# 基线是第十八批动手**之前**的实测值（12 MP = 4000x3000，
# Python 3.13 / numpy 2.5 / OpenCV 5.0）。第十八批做完后的目标值记在下面。
#   flat      10733 -> 4293   (-60%)   composite 不透明快路径 + 整数平移免 warp
#   rot        3007 -> 2048   (-32%)   整数平移那部分
#   adj-top   11722 -> 6101   (-48%)
#   adj-nested 10905 -> 6141  (-44%)
BASELINE = {
    "flat x8": 10733,
    "rot x8": 3007,
    "adj-top": 11722,
    "adj-nested": 10905,
}
AFTER_18 = {
    "flat x8": 4293,
    "rot x8": 2048,
    "adj-top": 6101,
    "adj-nested": 6141,
}

def _app():
    return QApplication.instance() or QApplication([])


def _layer(arr, name="L"):
    h, w = arr.shape[:2]
    return make_image_layer(name, arr, w, h)


def build_flat(n=8, w=4000, h=3000):
    """不透明色块、轴对齐整数位置 —— composite 的纯开销。"""
    d = Document(w, h, "flat")
    for i in range(n):
        a = np.zeros((h, w, 4), np.uint8)
        a[..., 0] = 40 + i * 9
        a[..., 1] = 90
        a[..., 2] = 140
        a[..., 3] = 255
        d.layers.append(_layer(a, "L%d" % i))
    return d


def build_rot(n=8, w=4000, h=3000):
    """带旋转缩放 + 各种混合模式 —— warpAffine 的开销。"""
    from src.core.blend import BLEND_MODES
    d = build_flat(1, w, h)
    for i in range(n):
        a = np.zeros((900, 1200, 4), np.uint8)
        a[..., 0] = 30 + i * 20
        a[..., 2] = 200 - i * 15
        a[..., 3] = 235
        l = _layer(a, "R%d" % i)
        l.rot = 7.0 * (i % 5) - 14.0
        l.sx = l.sy = 0.8 + 0.07 * i
        l.tx = 600.0 + i * 210.3
        l.ty = 500.0 + i * 133.7
        l.blend = BLEND_MODES[1 + (i % 8)]
        l.opacity = 0.9
        d.layers.append(l)
    return d


def build_adj(w=4000, h=3000, nested=False):
    """调整层。nested=True 时套在组里（模拟"组内调整层"）。"""
    from src.core.adjust import ADJUSTMENTS
    d = build_flat(6, w, h)
    keys = list(ADJUSTMENTS.keys())[:4]
    if not nested:
        for k in keys:
            a = Layer("adj " + k, LAYER_ADJUSTMENT)
            a.adjust = {"type": k, "params": default_params(k)}
            a.tx, a.ty = w / 2.0, h / 2.0
            d.layers.append(a)
        return d
    g = Layer("组", "group")
    for k in keys:
        a = Layer("adj " + k, LAYER_ADJUSTMENT)
        a.adjust = {"type": k, "params": default_params(k)}
        a.tx, a.ty = w / 2.0, h / 2.0
        g.children.append(a)
    d.layers.append(g)
    return d


def timeit(fn, repeat=1):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, (time.perf_counter() - t0) * 1000.0)
    return best, out


def run(name, fn, repeat=1, profile=False):
    from src.core.render import render_document
    t, arr = timeit(lambda: render_document(fn()), repeat)
    mp = arr.shape[0] * arr.shape[1] / 1e6
    base = BASELINE.get(name)
    delta = ""
    if base:
        d = (t - base) / base * 100.0
        delta = "  (%+.0f%% vs 基线)" % d
    print("  %-12s %8.0f ms   %5.1f MP%s" % (name, t, mp, delta))
    if profile:
        pr = cProfile.Profile()
        pr.enable()
        render_document(fn())
        pr.disable()
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(10)
        for line in s.getvalue().splitlines()[4:18]:
            print("      ", line)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", nargs="?", default="all")
    ap.add_argument("-n", "--repeat", type=int, default=1)
    ap.add_argument("-p", "--profile", action="store_true")
    args = ap.parse_args()
    _app()

    scenes = {
        "flat": ("flat x8", lambda: build_flat(8)),
        "rot": ("rot x8", lambda: build_rot(8)),
        "adj": ("adj-top", lambda: build_adj(nested=False)),
        "adj-nested": ("adj-nested", lambda: build_adj(nested=True)),
    }
    todo = scenes if args.scene == "all" else (
        {args.scene: scenes[args.scene]} if args.scene in scenes else {})
    if not todo:
        sys.exit("未知场景 %r，可选：%s 或 all"
                 % (args.scene, " / ".join(scenes)))

    print("渲染基准（%d MP，%d 次取最优）" % (12, args.repeat))
    for key in ("flat", "rot", "adj", "adj-nested"):
        if key in todo:
            name, fn = todo[key]
            run(name, fn, args.repeat, args.profile)


if __name__ == "__main__":
    main()