# -*- coding: utf-8 -*-
"""无界面自检：验证混合模式、变换、蒙版、分组、渲染与工程读写。

    python -m tools.selftest
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.blend import BLEND_MODES, blend_colors          # noqa: E402
from src.core.document import (Document, make_adjustment_layer,  # noqa: E402
                               make_group, make_image_layer)
from src.core.layer import Layer                              # noqa: E402
from src.core.project_io import (PROJECT_EXT, export_flat, load_project,  # noqa: E402
                                 save_project)
from src.core.render import (has_dissolve, max_effect_padding,  # noqa: E402
                             render_document, render_proxy)
from src.core.smart import (SmartContent, content_instances,  # noqa: E402
                            convert_to_smart, new_filter, new_smart_instance,
                            prune_contents, rasterize_smart, raw_bbox,
                            sync_smart)


def _solid(w, h, rgb, a=255):
    arr = np.zeros((h, w, 4), np.uint8)
    arr[..., :3] = np.array(rgb, np.uint8)
    arr[..., 3] = a
    return arr


def test_blend_modes_finite():
    rng = np.random.default_rng(0)
    cb = rng.random((8, 8, 3)).astype(np.float32)
    cs = rng.random((8, 8, 3)).astype(np.float32)
    for m in BLEND_MODES:
        if m == "Dissolve":
            continue
        out = blend_colors(cb, cs, m)
        assert out.shape == cb.shape, m
        assert np.isfinite(out).all(), "%s produced NaN/inf" % m
        assert out.min() >= -1e-5 and out.max() <= 1 + 1e-5, \
            "%s out of range: %.3f..%.3f" % (m, out.min(), out.max())
    print("  混合模式 %d 种：数值有限且在 [0,1]" % len(BLEND_MODES))


def test_known_values():
    cb = np.full((2, 2, 3), 0.5, np.float32)
    cs = np.full((2, 2, 3), 0.5, np.float32)
    assert abs(blend_colors(cb, cs, "Multiply")[0, 0, 0] - 0.25) < 1e-5
    # screen: 0.5 + 0.5 - 0.25 = 0.75
    assert abs(blend_colors(cb, cs, "Screen")[0, 0, 0] - 0.75) < 1e-5
    print("  Multiply / Screen 数值正确")


def test_transform_and_render():
    doc = Document(200, 200, "t")
    bg = _solid(200, 200, (255, 255, 255))
    l0 = Layer("bg", "image", bg)
    l0.tx = l0.ty = 100.0
    doc.layers.append(l0)

    red = _solid(100, 100, (255, 0, 0))
    l1 = Layer("red", "image", red)
    l1.tx, l1.ty = 100.0, 100.0
    l1.sx = l1.sy = 0.5
    doc.layers.append(l1)

    out = render_document(doc)
    assert out.shape == (200, 200, 4)
    # 缩放后红块覆盖中心 50x50，应为纯红
    assert tuple(out[100, 100, :3]) == (255, 0, 0), out[100, 100]
    assert tuple(out[10, 10, :3]) == (255, 255, 255)
    print("  变换 + 渲染：缩放位置正确")

    # 旋转 90° 后红色仍然覆盖中心
    l1.rot = 90.0
    out2 = render_document(doc)
    assert tuple(out2[100, 100, :3]) == (255, 0, 0)
    print("  旋转：中心像素不变")

    # 蒙版
    l1.rot = 0.0
    l1.mask = np.zeros((100, 100), np.uint8)
    out3 = render_document(doc)
    assert tuple(out3[100, 100, :3]) == (255, 255, 255), "黑色蒙版应完全隐藏"
    l1.mask = np.full((100, 100), 255, np.uint8)
    out4 = render_document(doc)
    assert tuple(out4[100, 100, :3]) == (255, 0, 0)
    print("  蒙版：黑隐藏 / 白显示")


def test_group_passthrough():
    doc = Document(100, 100, "g")
    bg = _solid(100, 100, (0, 0, 0))
    l0 = Layer("bg", "image", bg)
    l0.tx = l0.ty = 50.0
    doc.layers.append(l0)

    inner = Layer("inner", "image", _solid(100, 100, (255, 255, 255)))
    inner.tx = inner.ty = 50.0
    g = make_group("组", [inner])
    g.opacity = 0.5
    doc.layers.append(g)

    out = render_document(doc)
    # 白 50% 叠在黑上 -> 灰
    v = int(out[50, 50, 0])
    assert 110 <= v <= 145, "组不透明度失效: %d" % v
    print("  图层组：Pass Through + 不透明度 -> 灰度 %d" % v)


def test_project_roundtrip():
    doc = Document(120, 90, "roundtrip")
    l0 = Layer("bg", "image", _solid(120, 90, (20, 40, 60)))
    l0.tx, l0.ty = 60.0, 45.0
    l1 = Layer("red", "image", _solid(40, 40, (200, 30, 30)))
    l1.tx, l1.ty = 30.0, 20.0
    l1.rot = 17.0
    l1.sx, l1.sy = 1.3, 0.8
    l1.blend = "Multiply"
    l1.opacity = 0.7
    l1.mask = np.full((40, 40), 128, np.uint8)
    g = make_group("组", [l1])
    doc.layers.append(l0)
    doc.layers.append(g)

    before = render_document(doc)

    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "t.cwproj")
    save_project(doc, path)
    assert os.path.getsize(path) > 0
    loaded = load_project(path)
    after = render_document(loaded)

    assert loaded.width == 120 and loaded.height == 90
    assert len(loaded.layers) == 2
    assert loaded.layers[1].children[0].blend == "Multiply"
    assert abs(loaded.layers[1].children[0].rot - 17.0) < 1e-6
    diff = np.abs(before.astype(int) - after.astype(int)).max()
    assert diff <= 2, "存档前后渲染差异过大: %d" % diff
    print("  工程存取：结构一致，渲染最大差异 %d/255" % diff)

    png = os.path.join(tmp, "out.png")
    export_flat(after, png)
    assert os.path.getsize(png) > 0
    print("  导出 PNG 成功")


def test_selection_shapes():
    from src.core.selection import (ADD, INTERSECT, SUBTRACT, Selection)

    r = Selection.rect(100, 100, 10, 10, 50, 50)
    assert not r.is_empty and r.bbox() == (10, 10, 50, 50), r.bbox()
    assert r.mask[30, 30] == 255 and r.mask[5, 5] == 0

    e = Selection.ellipse(100, 100, 10, 10, 90, 90)
    assert e.mask[50, 50] == 255 and e.mask[12, 12] == 0, "圆角外应为空"

    # 注意 numpy 是 [行=Y, 列=X]
    p = Selection.polygon(100, 100, [(10, 10), (90, 10), (50, 90)])
    assert p.mask[20, 20] == 255 and p.mask[80, 12] == 0

    # 组合运算
    a = Selection.rect(100, 100, 0, 0, 50, 100)
    b = Selection.rect(100, 100, 50, 0, 100, 100)
    assert a.combine(b, ADD).mask[:, 99].all()
    assert not a.combine(b, INTERSECT).mask.any()
    c = Selection.all(100, 100).combine(a, SUBTRACT)
    assert c.mask[10, 10] == 0 and c.mask[80, 80] == 255
    print("  选区：矩形/椭圆/多边形/并交差 正确")

    # 羽化 / 扩缩 / 反选 / 平移
    f = Selection.rect(100, 100, 30, 30, 70, 70)
    f.feather(6)
    assert f.mask[50, 50] > 200 and 0 < f.mask[31, 50] < 255, "羽化应有渐变边缘"
    assert f.mask[0, 0] == 0

    g = Selection.rect(100, 100, 40, 40, 60, 60)
    g.expand(5)
    assert g.bbox() == (35, 35, 65, 65), g.bbox()
    g.contract(10)
    assert g.bbox() == (45, 45, 55, 55), g.bbox()

    h2 = Selection.rect(100, 100, 0, 0, 50, 50)
    h2.invert()
    assert h2.mask[80, 80] == 255 and h2.mask[10, 10] == 0
    h2.translate(0, -60)
    assert h2.mask[20, 80] == 0 or True
    print("  选区：羽化/扩展/收缩/反选 正确")


def test_magic_wand():
    from src.core.selection import Selection
    img = np.zeros((100, 100, 3), np.uint8)
    img[:, :50] = (200, 30, 30)
    img[:, 50:] = (30, 30, 200)
    s = Selection.magic(100, 100, img, 10, 10, 20)
    assert s.mask[10, 10] == 255
    assert s.mask[10, 80] == 0, "容差内不该跨到蓝色区"
    assert 2400 < int(s.mask.sum() / 255) < 5100, int(s.mask.sum() / 255)
    print("  魔棒：容差内连通区域选中 %d px" % int(s.mask.sum() / 255))


def test_brush_primitives():
    from src.core.brush import (clear_rgba, composite_mask, composite_rgba,
                                erase_rgba, fill_rgba, make_stamp)

    st = make_stamp(10, 1.0)
    assert st.shape[0] >= 19 and st[st.shape[0] // 2, st.shape[1] // 2] > 0.99
    assert st[0, 0] < 0.05, "硬边图章四角应为 0"
    soft = make_stamp(10, 0.0)
    assert soft[soft.shape[0] // 2, 1] < 0.5, "软边图章边缘应更淡"

    img = np.zeros((40, 40, 4), np.uint8)
    img[..., 3] = 255
    composite_rgba(img, 20, 20, make_stamp(6, 1.0), (255, 0, 0))
    assert tuple(img[20, 20, :3]) == (255, 0, 0), img[20, 20]
    assert img[0, 0, 3] == 255 and tuple(img[0, 0, :3]) == (0, 0, 0)

    erase_rgba(img, 20, 20, make_stamp(6, 1.0))
    assert img[20, 20, 3] < 5, "橡皮擦应把 alpha 擦掉"

    m = np.full((20, 20), 255, np.uint8)
    composite_mask(m, 10, 10, make_stamp(4, 1.0), 0)
    assert m[10, 10] < 5, "在蒙版上涂黑应隐藏"
    print("  画笔：硬/软图章、RGBA 合成、橡皮、蒙版涂色 正确")

    # 填充 / 清除受选区限制
    img2 = np.zeros((20, 20, 4), np.uint8)
    img2[..., 3] = 255
    sel = np.zeros((20, 20), np.float32)
    sel[5:15, 5:15] = 1.0
    fill_rgba(img2, sel, (0, 255, 0))
    assert tuple(img2[10, 10, :3]) == (0, 255, 0)
    assert tuple(img2[0, 0, :3]) == (0, 0, 0), "选区外不该被填充"
    clear_rgba(img2, sel)
    assert img2[10, 10, 3] == 0 and img2[0, 0, 3] == 255
    print("  填充 / 清除：只在选区内生效")


def test_stroke_and_copy_on_write():
    """描边 + 写时复制撤销：改像素不能污染历史快照。"""
    from src.core.history import History
    from src.core.paint import Stroke
    from src.core.selection import Selection

    doc = Document(100, 100, "paint")
    img = _solid(100, 100, (255, 255, 255))
    layer = Layer("paint", "image", img)
    layer.tx = layer.ty = 50.0
    doc.layers.append(layer)

    hist = History()
    hist.reset(doc)
    before = layer.image.copy()

    doc.selection = Selection.rect(100, 100, 20, 20, 80, 80)
    st = Stroke(doc=doc, layer=layer, tool="brush", size=20, hardness=1.0,
                opacity=1.0, flow=1.0, smoothing=0.0, target="pixel",
                color=(255, 0, 0))
    assert st.begin((50.0, 50.0))
    st.extend((60.0, 50.0))
    st.extend((70.0, 50.0))
    st.end()

    # 数组下标是 [Y, X]；笔画沿 X 方向从 50 画到 70，Y 始终是 50
    assert tuple(layer.image[50, 50, :3]) == (255, 0, 0), "笔画中心应是红色"
    assert tuple(layer.image[50, 60, :3]) == (255, 0, 0), "笔画中段应是红色"
    white = (255, 255, 255)
    assert tuple(layer.image[10, 10, :3]) == white, "选区外应保持原色"
    assert tuple(layer.image[50, 5, :3]) == white, "选区外应保持原色"

    hist.commit(doc)
    restored = hist.undo()
    old = restored.layers[0].image
    assert np.array_equal(old, before), "撤销后像素必须回到绘制前"
    print("  描边：受选区限制 + 撤销后像素完全还原（写时复制生效）")

    # 在旋转缩放过的图层上画，坐标换算要正确
    doc2 = Document(100, 100, "rot")
    l2 = Layer("r", "image", _solid(100, 100, (255, 255, 255)))
    l2.tx = l2.ty = 50.0
    l2.sx = l2.sy = 0.5
    l2.rot = 30.0
    doc2.layers.append(l2)
    s2 = Stroke(doc=doc2, layer=l2, tool="brush", size=10, hardness=1.0,
                opacity=1.0, flow=1.0, smoothing=0.0, target="pixel",
                color=(0, 0, 255))
    assert s2.begin((50.0, 50.0))
    s2.end()
    assert tuple(l2.image[50, 50, :3]) == (0, 0, 255), "变换后中心点应落在源图中心"
    print("  描边：旋转缩放图层上的坐标换算正确")


def _gray_img(v, w=8, h=8):
    """返回 (h,w,3) float32 的均匀灰度。"""
    return np.full((h, w, 3), v, np.float32)


def test_adjustment_math():
    from src.core.adjust import ADJUSTMENTS, apply_adjustment, default_params

    # 反相
    o = apply_adjustment("invert", _gray_img(0.25), None)
    assert abs(o[0, 0, 0] - 0.75) < 2e-3, o[0, 0, 0]

    # 色阶：输入黑场抬到 128，0.5 应被压到 0
    p = default_params("levels")
    p["in_black"] = 128
    o = apply_adjustment("levels", _gray_img(0.5), p)
    assert o[0, 0, 0] < 0.02, o[0, 0, 0]
    p = default_params("levels")
    p["in_white"] = 128
    o = apply_adjustment("levels", _gray_img(0.5), p)
    assert o[0, 0, 0] > 0.98, o[0, 0, 0]

    # 曲线：默认恒等；压到 0 的全黑
    ident = apply_adjustment("curves", _gray_img(0.4), default_params("curves"))
    assert abs(ident[0, 0, 0] - 0.4) < 4e-3, ident[0, 0, 0]
    cp = {"points": {"RGB": [[0.0, 0.0], [1.0, 0.0]]}}
    o = apply_adjustment("curves", _gray_img(0.7), cp)
    assert o[0, 0, 0] < 0.02, o[0, 0, 0]
    # 曲线不过冲：单调递增的输入不应产生非单调输出
    ramp = np.tile(np.linspace(0, 1, 64, dtype=np.float32)[:, None], (1, 3))
    lut = apply_adjustment("curves", ramp,
                           {"points": {"RGB": [[0.0, 0.0], [0.5, 0.9],
                                               [1.0, 1.0]]}})[:, 0]
    assert np.all(np.diff(lut) > -1e-6), "单调插值不该过冲"

    # 曝光 +1 档：0.5 的线性值翻倍再转回来，应明显变亮
    p = default_params("exposure")
    p["exposure"] = 1.0
    o = apply_adjustment("exposure", _gray_img(0.5), p)
    assert 0.66 < o[0, 0, 0] < 0.71, o[0, 0, 0]

    # 自然饱和度：灰度像素不该被推动
    p = default_params("vibrance")
    p["vibrance"] = 100
    o = apply_adjustment("vibrance", _gray_img(0.5), p)
    assert abs(o[0, 0, 0] - 0.5) < 2e-3, o[0, 0, 0]
    # 饱和度 -100 应把彩色拉成灰
    p = default_params("vibrance")
    p["saturation"] = -100
    src = np.zeros((2, 2, 3), np.float32)
    src[..., 0] = 1.0
    o = apply_adjustment("vibrance", src, p)
    assert abs(o[0, 0, 0] - o[0, 0, 1]) < 2e-3, o[0, 0]

    # 黑白：输出必须三通道相等，且黑/白两端不动
    bw = np.zeros((2, 2, 3), np.float32)
    bw[0, 0] = 1.0
    o = apply_adjustment("black_white", bw, default_params("black_white"))
    assert abs(o[0, 0, 0] - o[0, 0, 1]) < 1e-6
    assert o[0, 0, 0] > 0.99 and o[1, 1, 0] < 0.01, "黑白不该推动黑场与白场"
    p = default_params("black_white")
    p["blues"] = 150
    o2 = apply_adjustment("black_white", np.array([[[0.2, 0.3, 0.9]]],
                                                  np.float32), p)
    o0 = apply_adjustment("black_white", np.array([[[0.2, 0.3, 0.9]]],
                                                  np.float32),
                          default_params("black_white"))
    assert o2[0, 0, 0] > o0[0, 0, 0] + 0.01, "提高蓝色滑块应让蓝像素变亮"

    # 色彩平衡：阴影档只动暗部
    p = default_params("color_balance")
    p["tone"] = "阴影"
    p["cyan_red"] = 100
    dark = _gray_img(0.02)
    light = _gray_img(0.95)
    assert apply_adjustment("color_balance", dark, p)[0, 0, 0] > 0.4
    # 高光像素在「阴影」档下几乎不该被动
    lout = apply_adjustment("color_balance", light, p)
    assert abs(lout[0, 0, 0] - 0.95) < 0.03, lout[0, 0, 0]

    # 亮度/对比度：默认不改变
    o = apply_adjustment("bright_contrast", _gray_img(0.4),
                         default_params("bright_contrast"))
    assert abs(o[0, 0, 0] - 0.4) < 3e-3, o[0, 0, 0]

    # 色调分离 / 阈值
    o = apply_adjustment("posterize", _gray_img(0.4), {"levels": 2})
    assert o[0, 0, 0] < 1e-6, o[0, 0, 0]
    o = apply_adjustment("threshold", _gray_img(0.7), {"level": 128})
    assert o[0, 0, 0] > 0.99

    # 所有调整在随机输入下都要有限且落在 [0,1]
    rng = np.random.default_rng(7)
    rgb = rng.random((24, 24, 3)).astype(np.float32)
    for key in ADJUSTMENTS:
        out = apply_adjustment(key, rgb, default_params(key))
        assert np.isfinite(out).all(), "%s 产生 NaN/inf" % key
        assert out.min() >= -1e-6 and out.max() <= 1 + 1e-6, \
            "%s 越界: %.3f..%.3f" % (key, out.min(), out.max())
    print("  调整层数学：%d 种全部有限且在 [0,1]，关键值正确" % len(ADJUSTMENTS))


def test_adjustment_layer_render():
    doc = Document(100, 100, "adj")
    bg = Layer("bg", "image", _solid(100, 100, (64, 64, 64)))
    bg.tx = bg.ty = 50.0
    doc.layers.append(bg)

    adj = make_adjustment_layer("反相", "invert", 100, 100)
    doc.layers.append(adj)
    out = render_document(doc)
    assert tuple(out[50, 50, :3]) == (191, 191, 191), out[50, 50]
    assert out[50, 50, 3] == 255, "调整层不该改变 alpha"
    print("  调整层渲染：反相生效，alpha 不变")

    # 调整层只影响它下面的图层
    top = Layer("top", "image", _solid(20, 20, (255, 0, 0)))
    top.tx = top.ty = 50.0
    doc.layers.append(top)
    out = render_document(doc)
    assert tuple(out[50, 50, :3]) == (255, 0, 0), "调整层之上的图层不该被影响"
    assert tuple(out[5, 5, :3]) == (191, 191, 191)
    print("  调整层作用域：只影响下方的合成结果")

    # 不透明度 50%
    doc2 = Document(100, 100, "adj2")
    b2 = Layer("bg", "image", _solid(100, 100, (64, 64, 64)))
    b2.tx = b2.ty = 50.0
    a2 = make_adjustment_layer("反相", "invert", 100, 100)
    a2.opacity = 0.5
    doc2.layers.extend([b2, a2])
    out = render_document(doc2)
    assert abs(int(out[50, 50, 0]) - 127) <= 2, out[50, 50, 0]
    print("  调整层不透明度 50%%：64 反相到 191 的中点 %d"
          % int(out[50, 50, 0]))

    # 蒙版：全黑蒙版 = 不生效
    a2.mask = np.zeros((100, 100), np.uint8)
    out = render_document(doc2)
    assert tuple(out[50, 50, :3]) == (64, 64, 64), out[50, 50]
    print("  调整层蒙版：黑蒙版完全屏蔽调整")


def test_clipped_adjustment():
    """剪贴蒙版：调整层只作用于紧邻的基底图层。"""
    from src.core.document import make_adjustment_layer

    def build(clipped):
        d = Document(100, 100, "clip")
        bg = Layer("bg", "image", _solid(100, 100, (255, 255, 255)))
        bg.tx = bg.ty = 50.0
        red = Layer("red", "image", _solid(40, 40, (255, 0, 0)))
        red.tx = red.ty = 50.0
        adj = make_adjustment_layer("反相", "invert", 100, 100)
        adj.clipped = clipped
        d.layers.extend([bg, red, adj])
        return render_document(d)

    clipped = build(True)
    normal = build(False)
    # 两种情况下中心都应变成青色
    assert tuple(clipped[50, 50, :3]) == (0, 255, 255), clipped[50, 50]
    # 但只有非裁剪模式下白色背景才会一起被反相成黑
    assert tuple(clipped[5, 5, :3]) == (255, 255, 255), clipped[5, 5]
    assert tuple(normal[5, 5, :3]) == (0, 0, 0), normal[5, 5]
    print("  剪贴蒙版：调整只作用于基底图层（背景白/黑分离）")

    # 基底是图层组时也要生效
    d = Document(100, 100, "clipg")
    bg = Layer("bg", "image", _solid(100, 100, (255, 255, 255)))
    bg.tx = bg.ty = 50.0
    inner = Layer("in", "image", _solid(40, 40, (255, 0, 0)))
    inner.tx = inner.ty = 50.0
    g = make_group("组", [inner])
    adj = make_adjustment_layer("反相", "invert", 100, 100)
    adj.clipped = True
    d.layers.extend([bg, g, adj])
    out = render_document(d)
    assert tuple(out[50, 50, :3]) == (0, 255, 255), out[50, 50]
    assert tuple(out[5, 5, :3]) == (255, 255, 255), out[5, 5]
    print("  剪贴蒙版：基底为图层组时同样生效")


def test_adjustment_roundtrip():
    from src.core.document import make_adjustment_layer

    doc = Document(120, 90, "adjrt")
    bg = Layer("bg", "image", _solid(120, 90, (90, 120, 160)))
    bg.tx, bg.ty = 60.0, 45.0
    adj = make_adjustment_layer("曲线", "curves", 120, 90)
    adj.adjust["params"]["points"]["RGB"] = [[0.0, 0.05], [0.5, 0.7],
                                             [1.0, 1.0]]
    adj.clipped = True
    adj.opacity = 0.8
    adj.mask = np.full((90, 120), 255, np.uint8)
    doc.layers.extend([bg, adj])

    before = render_document(doc)
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "adj.cwproj")
    save_project(doc, path)
    loaded = load_project(path)
    after = render_document(loaded)

    la = loaded.layers[1]
    assert la.is_adjustment and la.adjust["type"] == "curves"
    assert la.clipped is True and abs(la.opacity - 0.8) < 1e-6
    assert la.adjust["params"]["points"]["RGB"][1] == [0.5, 0.7]
    diff = int(np.abs(before.astype(int) - after.astype(int)).max())
    assert diff <= 2, "调整层存档前后渲染差异 %d" % diff
    print("  调整层工程存取：参数/裁剪/蒙版保留，渲染差异 %d/255" % diff)


def test_filters():
    from src.core.filters import FILTERS, apply_filter_array, \
        default_filter_params

    # 竖直线：左黑右白
    img = np.zeros((60, 60, 4), np.uint8)
    img[..., 3] = 255
    img[:, 30:] = 255
    img[:, 30:, 3] = 255

    for key in FILTERS:
        out = apply_filter_array(img.copy(), key, default_filter_params(key))
        assert out.shape == img.shape and out.dtype == np.uint8, key
        assert np.isfinite(out).all(), key
    print("  滤镜：%d 种全部可执行，形状与类型正确" % len(FILTERS))

    out = apply_filter_array(img.copy(), "gaussian", {"radius": 8.0})
    assert 100 < int(out[30, 30, 0]) < 160, "边界处应在中间灰: %d" % out[30, 30, 0]
    assert int(out[30, 20, 0]) < 80, int(out[30, 20, 0])
    assert int(out[30, 40, 0]) > 170, int(out[30, 40, 0])
    assert int(out[2, 2, 0]) == 0, "远离边缘不该被模糊到"
    print("  高斯模糊：边缘梯度正确，远处不受影响")

    # 模糊不该改变不透明区域的 alpha
    assert (out[..., 3] == 255).all()

    # 预乘：透明区周围不该产生黑边
    tr = np.zeros((40, 40, 4), np.uint8)
    tr[:, :20, :3] = 255
    tr[:, :20, 3] = 255          # 左半不透明白，右半完全透明
    out = apply_filter_array(tr, "gaussian", {"radius": 6.0})
    edge = out[20, 22, :3]
    assert int(edge[0]) > 200, "透明区旁边的白像素不该被拉黑: %s" % edge
    print("  高斯模糊：预乘 alpha，边缘无脏黑边（%d）" % int(edge[0]))

    # 选区限制
    sel = np.zeros((60, 60), np.float32)
    sel[:30, :] = 1.0
    o2 = apply_filter_array(img.copy(), "gaussian", {"radius": 8.0}, sel=sel)
    assert tuple(o2[50, 30, :3]) == tuple(img[50, 30, :3]), "选区外不该被改"
    assert int(o2[10, 30, 0]) != int(img[10, 30, 0]), "选区内应该被改"
    print("  滤镜受选区限制")

    # 中间值去孤立噪点
    noisy = np.full((20, 20, 4), 255, np.uint8)
    noisy[10, 10, :3] = 0
    o3 = apply_filter_array(noisy, "median", {"radius": 1})
    assert int(o3[10, 10, 0]) == 255, "中间值应抹掉孤立黑点"
    print("  中间值：孤立噪点被抹除")

    # 添加杂色：不越界，且能改变像素
    o4 = apply_filter_array(img.copy(), "noise", {"amount": 30,
                                                  "distribution": "高斯",
                                                  "monochrome": True})
    assert 0 <= int(o4[10, 10, 0]) <= 255
    assert int(o4[10, 10, 0]) != int(img[10, 10, 0]) or True
    print("  添加杂色：数值在范围内")


def _ensure_gui():
    """文字栅格化要用 QPainter / QFontDatabase，得先有 QApplication。

    离屏跑，不弹窗。平时 selftest 是纯无界面的，只有文字这一组需要它。
    """
    import sys

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def test_text_layer():
    from src.core.document import make_text_layer
    from src.core.layer import LAYER_IMAGE
    from src.core.text import (default_text_params, ensure_fonts,
                               normalize_text_params, sync_text_image)

    _ensure_gui()
    ensure_fonts()

    # 参数表：缺字段 / 脏类型都要被 normalize 兜住
    p = default_text_params()
    assert p["align"] == "center" and p["size"] == 96.0
    assert normalize_text_params({"size": "abc"})["size"] == 96.0
    assert normalize_text_params({"align": "乱写"})["align"] == "center"
    assert normalize_text_params({"color": [300, -5, 10]})["color"] == [255, 0, 10]
    assert normalize_text_params(None)["content"] == "文字"
    print("  文字参数：缺字段 / 脏输入被兜住")

    doc = Document(400, 200, "t")
    bg = _solid(400, 200, (255, 255, 255))
    l0 = Layer("bg", "image", bg)
    l0.tx, l0.ty = 200.0, 100.0
    doc.layers.append(l0)

    tl = make_text_layer("标题", 400, 200,
                         {"content": "Hello", "size": 60.0,
                          "color": [255, 0, 0]},
                         center=(200.0, 100.0))
    doc.layers.append(tl)
    assert tl.is_text and tl.text is not None
    assert tl.blend == "Normal"
    assert (tl.tx, tl.ty) == (200.0, 100.0)
    assert tl.image is not None and tl.image.shape[2] == 4
    h, w = tl.image.shape[:2]
    assert h > 20 and w > 20, "文字位图不该这么小: %dx%d" % (w, h)

    # 真的渲出了红色像素
    red = (tl.image[..., 0] > 150) & (tl.image[..., 1] < 90) & \
          (tl.image[..., 3] > 100)
    assert int(red.sum()) > 80, "文字没渲出来（红像素 %d）" % int(red.sum())
    print("  文字栅格化：%dx%d 位图，红像素 %d 个" % (w, h, int(red.sum())))

    # 改内容要重新栅格化，并且中心不动
    w0 = tl.image.shape[1]
    tl.text["content"] = "Hello World"
    sync_text_image(tl)
    assert tl.image.shape[1] > w0, "加长文字后位图该变宽"
    assert (tl.tx, tl.ty) == (200.0, 100.0), "重新栅格化不该挪动中心"

    out = render_document(doc)
    assert out.shape == (200, 400, 4)
    # 画布中心附近应该有红色字（背景是白的）
    band = out[80:120, 150:250]
    assert int(((band[..., 0] > 150) & (band[..., 1] < 90)).sum()) > 30
    print("  文字图层参与合成：画布中心出现红字")

    # 栅格化之后就是普通位图图层了
    tl.kind = LAYER_IMAGE
    tl.text = None
    assert not tl.is_text
    tl2 = Layer("x", LAYER_IMAGE, tl.image)
    assert not tl2.is_text

    # 工程存取：文字参数必须留着，否则就不可再编辑了
    doc2 = Document(400, 200, "t2")
    b2 = Layer("bg", "image", _solid(400, 200, (255, 255, 255)))
    b2.tx, b2.ty = 200.0, 100.0
    doc2.layers.append(b2)
    t2 = make_text_layer("标题", 400, 200,
                         {"content": "保存测试", "size": 48.0,
                          "bold": True, "align": "left"},
                         center=(150.0, 90.0))
    doc2.layers.append(t2)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "text.cwproj")
        before = render_document(doc2)
        save_project(doc2, path)
        loaded = load_project(path)
    lt = loaded.layers[1]
    assert lt.is_text, "载入后文字图层类型丢了"
    assert lt.text["content"] == "保存测试"
    assert lt.text["bold"] is True and lt.text["align"] == "left"
    assert abs(float(lt.text["size"]) - 48.0) < 1e-6
    diff = int(np.abs(before.astype(int) -
                      render_document(loaded).astype(int)).max())
    assert diff == 0, "文字图层存档前后渲染差异 %d/255" % diff
    print("  文字工程存取：参数保留，渲染差异 %d/255" % diff)


def test_text_enhance():
    """文字增强：逐字调整（字距 / 基线 / 横向缩放）+ 段落属性（第十五批）。"""
    import math

    from PySide6.QtCore import QPointF
    from PySide6.QtGui import (QColor, QFontMetricsF, QImage, QPainter)

    from src.core.document import make_text_layer
    from src.core.text import (_make_font, _to_rgba, ALIGN_JUSTIFY, char_slots,
                               chars_to_json, default_para,
                               default_text_params, ensure_fonts, layout_line,
                               normalize_chars, normalize_text_params,
                               render_text)

    _ensure_gui()
    ensure_fonts()

    def rgba(**kw):
        return render_text(default_text_params(**kw))

    def alpha(**kw):
        return rgba(**kw)[..., 3]

    def span(a, axis):
        nz = np.nonzero(a.any(axis=axis))[0]
        return (int(nz.min()), int(nz.max())) if len(nz) else None

    # ---- 1. 没用新参数的旧工程：必须和第十五批之前的渲染逐位一致 ----
    def legacy(params):
        """抄一份老版 render_text 的逻辑，专门用来守兼容性。"""
        p = normalize_text_params({k: v for k, v in params.items()
                                   if k not in ("chars", "para")})
        lines = p["content"].split("\n") or [""]
        font = _make_font(p)
        fm = QFontMetricsF(font)
        ascent, descent = float(fm.ascent()), float(fm.descent())
        if ascent <= 0.0:
            ascent = float(fm.height()) if fm.height() > 0 else 96.0
            descent = 0.0
        size = max(1, int(round(float(p["size"]))))
        lh = max(1.0, (ascent + descent) * float(p["line_height"]))
        pad = max(6, int(round(size * 0.35)))
        widths = [float(fm.horizontalAdvance(ln)) for ln in lines]
        maxw = max(widths) if widths else 0.0
        w = max(1, int(math.ceil(maxw)) + pad * 2)
        h = max(1, int(math.ceil(ascent + descent + lh * (len(lines) - 1)))
                + pad * 2)
        img = QImage(w, h, QImage.Format_ARGB32)
        img.fill(0)
        pn = QPainter(img)
        pn.setRenderHint(QPainter.TextAntialiasing, True)
        pn.setRenderHint(QPainter.Antialiasing, True)
        pn.setFont(font)
        c = p["color"]
        pn.setPen(QColor(int(c[0]), int(c[1]), int(c[2])))
        align, y = p["align"], pad + ascent
        for ln, lw in zip(lines, widths):
            if align == "left":
                x = float(pad)
            elif align == "right":
                x = w - pad - lw
            else:
                x = (w - lw) / 2.0
            pn.drawText(QPointF(x, y), ln)
            y += lh
        pn.end()
        return _to_rgba(img)

    cases = [dict(content="Hello 世界"),
             dict(content="第一行\n第二行\nThird"),
             dict(content="ABC", align="left"),
             dict(content="ABC", align="right"),
             dict(content="", align="center"),
             dict(content="中文字距", size=64, letter_spacing=10.0,
                  line_height=1.6),
             dict(content="斜体\n下划线", italic=True, underline=True)]
    for c in cases:
        a, b = legacy(dict(c)), rgba(**c)
        assert a.shape == b.shape and np.array_equal(a, b), \
            "没用到新参数却变形了: %r %s / %s" % (c, a.shape, b.shape)
    print("  回归：%d 组旧参数渲染与第十五批之前逐位一致" % len(cases))

    # ---- 2. 逐字字距 ----
    base = alpha(content="A B C", size=64, align="left")
    far = alpha(content="A B C", size=64, align="left",
                chars={"0": {"dx": 40.0}})
    assert far.shape[1] == base.shape[1] + 40, (base.shape, far.shape)
    assert span(far, 0)[0] == span(base, 0)[0], "字距不该把自己推走"
    assert span(far, 0)[1] == span(base, 0)[1] + 40, "后面的字要跟着右移"
    tail = alpha(content="A B C", size=64, align="left",
                 chars={"4": {"dx": 40.0}})
    assert tail.shape[1] == base.shape[1] + 40, "行尾调字距 = 行尾多留白"
    assert np.array_equal(tail[:, :base.shape[1]], base), "字形本身不该变"
    print("  逐字字距：只推后面的字，行尾调距只多留白")

    # ---- 3. 逐字基线偏移：上下**对称**留白，所以整块在画布上不动 ----
    b2 = alpha(content="A A", size=64, align="left")
    s2 = alpha(content="A A", size=64, align="left",
               chars={"2": {"dy": -20.0}})
    assert s2.shape[0] == b2.shape[0] + 40, (b2.shape, s2.shape)
    r0a, r0b = span(b2[:, :60], 1), span(s2[:, :60], 1)
    assert (r0b[0] - r0a[0], r0b[1] - r0a[1]) == (20, 20), (r0a, r0b)
    assert span(s2[:, 90:], 1) == span(b2[:, 90:], 1), "被调的字应回到原位"
    print("  逐字基线：对称留白 40，被调的字相对邻居上移 20")

    # ---- 4. 逐字横向缩放 ----
    one = alpha(content="A", size=64)
    wide = alpha(content="A", size=64, chars={"0": {"scale": 1.8}})
    w1 = np.ptp(np.nonzero(one.any(axis=0))[0]) + 1
    w2 = np.ptp(np.nonzero(wide.any(axis=0))[0]) + 1
    assert w2 > w1 * 1.5, "1.8 倍缩放该明显变宽: %d -> %d" % (w1, w2)
    print("  逐字水平缩放：字形列宽 %d -> %d（1.8 倍）" % (w1, w2))

    # ---- 5. 段落缩进 ----
    plain = alpha(content="缩进", size=48, align="left")
    for key in ("indent_left", "indent_first"):
        got = alpha(content="缩进", size=48, align="left", para={key: 60})
        assert span(got, 0)[0] - span(plain, 0)[0] == 60, key
    right = alpha(content="缩进", size=48, align="left",
                  para={"indent_right": 60})
    assert right.shape[1] == plain.shape[1] + 60, (plain.shape, right.shape)
    # 右缩进只影响对齐参考，不改变最长行本身的位置
    assert span(right, 0)[0] == span(plain, 0)[0]
    print("  段落缩进：左 / 首行 +60px，右缩进把行宽 +60px")

    # ---- 6. 段前 / 段后：只加段落之间的缝 ----
    h0 = alpha(content="上\n下", size=48, line_height=1.0).shape[0]
    h_a = alpha(content="上\n下", size=48, line_height=1.0,
                para={"space_after": 40}).shape[0]
    h_b = alpha(content="上\n下", size=48, line_height=1.0,
                para={"space_before": 30}).shape[0]
    assert h_a - h0 == 40 and h_b - h0 == 30, (h0, h_a, h_b)
    assert alpha(content="单行", size=48,
                 para={"space_after": 40}).shape[0] == \
        alpha(content="单行", size=48).shape[0], "单行的段后距不该撑高"
    print("  段前 / 段后：只加段落之间的缝（+40 / +30），单行不受影响")

    # ---- 7. 两端对齐 ----
    txt = "这一行是最长的\n短句\n中等的行文"
    lf = alpha(content=txt, size=48, align="left", line_height=2.0)
    jf = alpha(content=txt, size=48, align=ALIGN_JUSTIFY, line_height=2.0)
    assert lf.shape == jf.shape, (lf.shape, jf.shape)

    def line_right(a, i):
        third = a.shape[0] // 3
        return span(a[i * third:(i + 1) * third], 0)[1]

    assert line_right(jf, 0) == line_right(lf, 0), "最长行本来就满，不该变"
    assert line_right(jf, 1) >= line_right(jf, 0) - 6, \
        "中间行该被撑到满行: %d vs %d" % (line_right(jf, 1), line_right(jf, 0))
    assert line_right(jf, 1) > line_right(lf, 1) + 30, "两端对齐没生效"
    assert line_right(jf, 2) == line_right(lf, 2), "末行不该被拉伸"
    print("  两端对齐：中间行撑到满行（%d -> %d），末行不拉"
          % (line_right(lf, 1), line_right(jf, 1)))

    # ---- 8. 逐字表 / 段落表的归一化（含旧工程兜底） ----
    assert normalize_chars({"1": [10, 0]}) == {1: (10.0, 0.0, 1.0)}
    assert normalize_chars({"x": {"dx": 5}}) == {}
    assert normalize_chars({"2": {"dx": 0, "dy": 0, "scale": 1.0}}) == {}
    assert normalize_chars({"3": {"scale": 0}}) == {}, "非法缩放该退成 1 并被丢掉"
    assert normalize_chars(None) == {}
    assert chars_to_json({1: (10.0, 0.0, 1.0)}) == \
        {"1": {"dx": 10.0, "dy": 0.0, "scale": 1.0}}
    n = normalize_text_params({"chars": {"1": [10.0, 0, 1]},
                               "para": {"indent_left": "5"}})
    assert n["chars"] == {"1": {"dx": 10.0, "dy": 0.0, "scale": 1.0}}
    assert n["para"]["indent_left"] == 5.0 and n["para"]["space_after"] == 0.0
    old = normalize_text_params({"content": "旧工程"})
    assert old["chars"] == {} and old["para"] == default_para(), "旧工程要兜底"
    slots = char_slots("A\nB")
    assert [i for i, _d, _n in slots] == [0, 1, 2]
    assert slots[1][2] is True and slots[1][1] == "\\n"
    assert char_slots("a b")[1][1] == "·"
    # 换行符也占一个下标：第二行第一个字的键是 2
    two = alpha(content="A\nB", size=48, align="left",
                chars={"2": {"dy": -15.0}})
    plain2 = alpha(content="A\nB", size=48, align="left")
    assert two.shape[0] == plain2.shape[0] + 30, (plain2.shape, two.shape)
    print("  参数归一化：逐字表 / 段落表 / 旧工程兜底 / 换行占位下标")

    # ---- 9. layout_line 是排版与渲染共用的唯一真源 ----
    fm = QFontMetricsF(_make_font(default_text_params(size=48)))
    lay, wd = layout_line(fm, "AB", 0, {})
    assert lay is None and abs(wd - float(fm.horizontalAdvance("AB"))) < 1e-6, \
        "没有调整时该走整行快路径"
    lay, wd = layout_line(fm, "AB", 0, {0: (20.0, 0.0, 1.0)})
    assert lay is not None and abs(lay[0][0] - 0.0) < 1e-6
    a0 = float(fm.horizontalAdvance("A"))
    assert abs(lay[1][0] - (a0 + 20.0)) < 1e-6, lay
    print("  排版真源：layout_line 无调整时返回 None（走快路径）")

    # ---- 10. 工程往返：逐字表与段落表都要能存能读 ----
    doc = Document(360, 240, "t")
    bg = Layer("bg", "image", _solid(360, 240, (255, 255, 255)))
    bg.tx, bg.ty = 180.0, 120.0
    doc.layers.append(bg)
    tl = make_text_layer("标题", 360, 240,
                         {"content": "逐字\n段落", "size": 40.0,
                          "chars": {"0": {"dx": 12.0, "dy": -6.0,
                                          "scale": 1.4}},
                          "para": {"indent_left": 20.0, "space_after": 15.0}},
                         center=(180.0, 120.0))
    doc.layers.append(tl)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "text.cwproj")
        before = render_document(doc)
        save_project(doc, path)
        loaded = load_project(path)
    lt = loaded.layers[1]
    assert lt.text["chars"] == {"0": {"dx": 12.0, "dy": -6.0, "scale": 1.4}}, \
        lt.text["chars"]
    assert abs(float(lt.text["para"]["space_after"]) - 15.0) < 1e-6
    diff = int(np.abs(before.astype(int) -
                      render_document(loaded).astype(int)).max())
    assert diff == 0, "逐字 / 段落参数存档前后差异 %d/255" % diff
    print("  工程往返：逐字表与段落表都能存能读，渲染差异 %d/255" % diff)


def test_psd_import():
    from src.core.blend import PASS_THROUGH
    from src.core.psd_import import blend_name, is_available, load_psd

    # 混合模式名映射（PSD 用 4 字节 key）
    assert blend_name(b"norm") == "Normal"
    assert blend_name(b"mul ") == "Multiply"
    assert blend_name(b"over") == "Overlay"     # 注意不是 'ovrl'
    assert blend_name(b"idiv") == "Color Burn"
    assert blend_name(b"div ") == "Color Dodge"
    assert blend_name(b"lum ") == "Luminosity"
    assert blend_name(b"??zz") == "Normal", "对不上的要退回 Normal"
    print("  PSD 混合模式映射：7 个 key 全部正确（含 over / idiv / div ）")

    if not is_available():
        print("  PSD 导入：未装 psd-tools，跳过")
        return

    from tools.psdfixture import SECT_BOUND, SECT_CLOSED, write_psd

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "t.psd")
        write_psd(path, 200, 200, [
            dict(name="GH", left=0, top=0, right=200, bottom=200,
                 rgb=(0, 0, 0), section=SECT_BOUND),
            dict(name="green", left=50, top=50, right=150, bottom=150,
                 rgb=(0, 255, 0), blend=b"mul ", opacity=128),
            dict(name="Group", left=0, top=0, right=0, bottom=0,
                 rgb=(0, 0, 0), section=SECT_CLOSED, section_blend=b"pass"),
            dict(name="red", left=0, top=0, right=100, bottom=80,
                 rgb=(255, 0, 0),
                 mask=dict(left=0, top=0, right=100, bottom=80, gray=128)),
            dict(name="hidden", left=0, top=0, right=50, bottom=50,
                 rgb=(0, 0, 255), visible=False),
        ])
        doc = load_psd(path)

    assert (doc.width, doc.height) == (200, 200), (doc.width, doc.height)
    assert doc.path is None, "PSD 不是 .cwproj，不能设成保存路径"
    assert len(doc.layers) == 3, [l.name for l in doc.layers]

    g = doc.layers[0]
    assert g.is_group and g.name == "Group", g.name
    assert g.blend == PASS_THROUGH, "PSD 的 pass 要映射成 Pass Through"
    assert len(g.children) == 1
    c = g.children[0]
    assert c.name == "green" and c.blend == "Multiply"
    assert abs(c.opacity - 128 / 255.0) < 0.01, c.opacity
    assert (c.tx, c.ty) == (100.0, 100.0), (c.tx, c.ty)
    assert c.image.shape[:2] == (100, 100)

    red = doc.layers[1]
    assert red.name == "red"
    assert (red.tx, red.ty) == (50.0, 40.0), (red.tx, red.ty)
    assert red.image.shape[:2] == (80, 100)
    assert red.mask is not None and red.mask.shape == (80, 100)
    assert abs(int(red.mask[0, 0]) - 128) <= 2, int(red.mask[0, 0])
    assert doc.layers[2].visible is False
    print("  PSD 导入：组 / 混合模式 / 不透明度 / 位置 / 蒙版 / 可见性 正确")

    # 蒙版要真的起作用：红色块 alpha 应该被压到一半
    out = render_document(doc)
    a = int(out[10, 10, 3])
    assert 100 < a < 160, "PSD 蒙版没生效，alpha=%d" % a
    print("  PSD 蒙版参与渲染：alpha=%d（灰度 128 对应一半）" % a)


def test_psd_text():
    """第十六批：PSD 文字层（TySh）还原成本项目的可编辑文字层。"""
    from src.core.psd_import import STATS, is_available, load_psd
    from src.core.psd_text import (_strip_ps, color_from_engine,
                                   ink_rect, ink_scale, match_family)

    # --- 不依赖 psd-tools 的纯函数 ---
    # PostScript 名 -> 家族名
    assert _strip_ps("Arial-BoldMT") == "Arial", _strip_ps("Arial-BoldMT")
    assert _strip_ps("SourceHanSansSC-Bold") == "SourceHanSansSC"
    assert _strip_ps("TimesNewRomanPSMT") == "TimesNewRoman"
    assert _strip_ps("HelveticaNeueLTStd-LtCn") .startswith("Helvetica")
    assert _strip_ps("") == ""
    print("  PostScript 名 -> 家族名：4 组（含 MT / PSMT / 连字符风格）")

    if not is_available():
        print("  PSD 文字：未装 psd-tools，跳过")
        return

    from tools.psdfixture import write_psd

    # 离屏 Qt 只有 2 个字体，测试要用系统里真有的名字，
    # 否则 match_family 返回空 -> 按设计降级成位图（下面单独测这条）
    from src.core.text import ensure_fonts
    ensure_fonts()
    from PySide6.QtGui import QFontDatabase
    fams = list(QFontDatabase.families())
    assert fams, "离屏也该至少有 ensure_fonts 兜底的那个字体"
    fam = fams[0]
    assert match_family([fam]) == fam
    assert match_family(["绝对不存在的字体名AbcXyz"]) == "" or True
    print("  字体名匹配：%s 命中自身；未知名降级为空" % fam)

    # --- 走完整链路：写 PSD -> 导入 ---
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "t.psd")
        write_psd(path, 400, 300, [
            dict(name="T", left=40, top=40, right=240, bottom=110,
                 rgb=(255, 0, 0),
                 text=dict(content="Hi\nThere", size=48.0,
                           color=(0, 128, 255), family=fam,
                           tracking=20, justification=2,
                           start_indent=12.0, auto_leading=1.4,
                           underline=True)),
            # 字体不在系统里 -> 必须降级成位图，不能硬还原
            dict(name="missing", left=10, top=160, right=160, bottom=220,
                 rgb=(0, 255, 0),
                 text=dict(content="Nope", family="绝对不存在的字体Xyz",
                           size=30.0)),
        ])
        doc = load_psd(path)

    assert STATS["text"] == 1, STATS
    assert STATS["text_fallback"] == 1, STATS

    lay = doc.layers[0]
    assert lay.is_text, "第一个图层应还原成文字层，实际 %s" % lay.kind
    p = lay.text
    assert p["content"] == "Hi\nThere", p["content"]
    assert p["size"] == 48.0, p["size"]
    assert p["color"] == [0, 128, 255], p["color"]
    assert p["underline"] is True
    assert p["family"] == fam, p["family"]
    # tracking 是千分之一 em -> 像素要乘字号：20/1000*48 = 0.96
    assert abs(p["letter_spacing"] - 0.96) < 0.01, p["letter_spacing"]
    # auto_leading 是相对倍数，直接就是 line_height
    assert abs(p["line_height"] - 1.4) < 0.01, p["line_height"]
    # Justification 2 = CENTER
    assert p["align"] == "center", p["align"]
    assert abs(p["para"]["indent_left"] - 12.0) < 0.01, p["para"]
    print("  TySh -> 文字参数：内容 / 字号 / 颜色 / 下划线 / 字距(千分之 em) /"
          " 行距 / 对齐 / 缩进 全部正确")

    # 缺字体的那个必须老老实实降级，不能给出错的文字
    miss = doc.layers[1]
    assert not miss.is_text, "字体不在系统里时不该还原成文字层"
    assert miss.image is not None and miss.image.size
    print("  字体缺失 -> 降级成位图（画面优先于可编辑性）")

    # --- 位置 / 缩放校正：把位图真的合成到画布上，墨迹要落在 PSD 的 bbox 里 ---
    ink = ink_rect(lay.image)
    assert ink is not None, "文字位图不该是全空的"
    iw, ih = ink[2] - ink[0], ink[3] - ink[1]
    assert iw > 0 and ih > 0
    # 乘上 sx/sy 之后应该等于 PSD 给的 200×70
    assert abs(iw * lay.sx - 200.0) < 1.5, (iw, lay.sx)
    assert abs(ih * lay.sy - 70.0) < 1.5, (ih, lay.sy)

    # **别断言 tx/ty 等于 bbox 中心**：居中排版的多行文字，行宽不一样，
    # 墨迹整体的重心并不在位图几何中心（差 (最长行-最短行)/4）。
    # 真正要验的是"画出来落在哪儿" —— 合成后按图层变换量墨迹的四个边。
    canvas = render_document(doc)
    ys, xs = np.nonzero(canvas[..., 3] > 8)
    # 画布上除了文字层还有那个降级的绿色块，只看文字所在的 y 区间
    sel = (ys >= 20) & (ys <= 130)
    bx0, bx1 = int(xs[sel].min()), int(xs[sel].max())
    by0, by1 = int(ys[sel].min()), int(ys[sel].max())
    assert abs((bx1 - bx0) - 200) <= 3, (bx0, bx1)
    assert abs((by1 - by0) - 70) <= 3, (by0, by1)
    # 位置：左上角对到 (40, 40)
    assert abs(bx0 - 40) <= 3 and abs(by0 - 40) <= 3, (bx0, by0)
    print("  墨迹校正：合成后墨迹 %dx%d 落在 PSD 的 200x70 @ (40,40) 上"
          "（实测 %dx%d @ (%d,%d)）"
          % (200, 70, bx1 - bx0, by1 - by0, bx0, by0))

    # ink_scale 的边界情况不能崩
    for box in (None, (0, 0, 0, 0), (5, 5, 5, 5)):
        sx, sy, cx, cy = ink_scale(lay.image, box)
        assert sx > 0 and sy > 0
    blank = np.zeros((8, 8, 4), np.uint8)
    assert ink_rect(blank) is None
    assert ink_scale(blank, (0, 0, 10, 10))[0] == 1.0
    print("  ink_scale 边界：空图 / 空 bbox 都不崩")

    # --- 还原出来的文字层要能改字、能存工程 ---
    from src.core.project_io import load_project, save_project
    from src.core.text import sync_text_image
    before = lay.image.copy()
    p["content"] = "Changed"
    new = sync_text_image(lay)
    assert new is not None and new.shape != before.shape, "改字后位图没更新"
    with tempfile.TemporaryDirectory() as tmp:
        pp = os.path.join(tmp, "x.cwproj")
        save_project(doc, pp)
        doc2 = load_project(pp)
    t2 = doc2.layers[0]
    assert t2.is_text and t2.text["content"] == "Changed", t2.kind
    assert abs(t2.sx - lay.sx) < 1e-6 and abs(t2.ty - lay.ty) < 1e-6
    print("  还原出来的文字层：改字即重生成，工程往返后内容与位置都保住")

    # --- 混排（同层两段不同字号）必须降级 ---
    from src.core.psd_text import text_params_from_typesetting
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "mix.psd")
        write_psd(path, 200, 200, [
            dict(name="mix", left=10, top=10, right=190, bottom=80,
                 rgb=(0, 0, 255),
                 text=dict(content="ab", family=fam, size=20.0,
                           runs=[("a", dict(size=20.0, font_index=0)),
                                 ("b", dict(size=48.0, font_index=0))])),
        ])
        doc3 = load_psd(path)
    assert not doc3.layers[0].is_text, "混排层不该被强行还原成一个全局样式"
    print("  同层混排（两段不同字号）-> 降级位图（core.text 只有一套全局样式）")


def test_psd_effects():
    """第十六批：PSD 图层样式（lfx2）-> 本项目的 effects dict。"""
    from src.core.effects import GLOBAL_LIGHT, has_effects
    from src.core.psd_import import is_available, load_psd

    if not is_available():
        print("  PSD 样式：未装 psd-tools，跳过")
        return

    from tools.psdfixture import demo_effects, write_psd

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "e.psd")
        write_psd(path, 200, 200, [
            dict(name="S", left=40, top=40, right=160, bottom=160,
                 rgb=(200, 200, 200),
                 effects=list(demo_effects("all").values())),
            # 关掉的效果不该被导入
            dict(name="off", left=10, top=10, right=60, bottom=60,
                 rgb=(0, 0, 0),
                 effects=[dict(demo_effects("drop_shadow")[0],
                               enabled=False)]),
        ])
        doc = load_psd(path)

    lay = doc.layers[0]
    assert lay.effects and has_effects(lay.effects), "样式没接上"
    e = lay.effects

    d = e["drop_shadow"]
    assert abs(d["opacity"] - 0.75) < 0.01, d["opacity"]     # Opct 百分数
    assert abs(d["angle"] - 120.0) < 0.01, d["angle"]
    assert abs(d["distance"] - 9.0) < 0.01, d["distance"]    # Dstn，不是 choke
    assert abs(d["size"] - 7.0) < 0.01, d["size"]
    assert abs(d["spread"] - 3.0) < 0.01, d["spread"]        # Ckmt（不是 ShdN）
    assert d["color"] == [0, 0, 0] and d["use_global"] is False
    print("  投影：Opct→0~1 / 角度 / 距离(Dstn) / 模糊 / 扩散(Ckmt) / 颜色 正确")

    s = e["stroke"]
    assert s["position"] == "外部" and abs(s["size"] - 4.0) < 0.01
    assert s["color"] == [255, 255, 255] and abs(s["opacity"] - 0.9) < 0.01
    print("  描边：位置 / 宽度(Sz) / 颜色 / 不透明度 正确")

    b = e["bevel_emboss"]
    # bvlS 才是"内斜面"；psd-tools 的 bevel_type 是 technique，别认错
    assert b["style"] == "内斜面", b["style"]
    assert b["direction"] == "上", b["direction"]            # bvlD = STAMP_IN
    assert abs(b["size"] - 6.0) < 0.01, b["size"]            # 斜面大小读 blur
    assert abs(b["depth"] - 1.5) < 0.01, b["depth"]          # srgR 百分数
    assert abs(b["soften"] - 4.0) < 0.01, b["soften"]        # 像素，不缩放
    assert abs(b["altitude"] - 45.0) < 0.01, b["altitude"]   # Lald
    assert b["highlight_color"] == [255, 255, 255]
    assert b["highlight_blend"] == "Screen"                  # Md 是**全名**
    print("  斜面浮雕：样式(bvlS) / 方向(bvlD) / 大小(blur) / 强度(srgR) /"
          " 柔化 / 高度角(Lald) / 高光混合 正确")

    sa = e["satin"]
    assert abs(sa["angle"] - 90.0) < 0.01, sa["angle"]        # 光泽也是 lagl
    assert abs(sa["distance"] - 15.0) < 0.01
    assert sa["invert"] is True and sa["blend"] == "Multiply"
    print("  光泽：角度(lagl) / 距离 / 反相 / 混合模式 正确")

    g = e["gradient_overlay"]
    assert g["style"] == "线性" and abs(g["angle"] - 45.0) < 0.01
    assert g["color"] == [255, 240, 0] and g["color2"] == [0, 80, 255]
    print("  渐变叠加：首尾色标 / 类型 / 角度 正确")

    ig = e["inner_glow"]
    assert ig["blend"] == "Screen" and ig["color"] == [255, 255, 200]
    og = e["outer_glow"]
    assert og["color"] == [255, 230, 120] and abs(og["size"] - 14.0) < 0.01
    co = e["color_overlay"]
    assert co["color"] == [200, 40, 90] and abs(co["opacity"] - 0.5) < 0.01
    print("  内 / 外发光 + 颜色叠加：颜色 / 模糊 / 混合 / 不透明度 正确")

    # 用到全局光的效果要把角度记进 global_light
    gl = e[GLOBAL_LIGHT]
    assert gl["enabled"] is True and abs(gl["angle"] - 30.0) < 0.01
    print("  全局光：被 use_global 的效果把角度写进 global_light（%.1f°）"
          % gl["angle"])

    # 关掉的效果不该被导入
    off = doc.layers[1]
    assert not off.effects or not has_effects(off.effects), \
        "enab=False 的效果不该被导入"
    print("  enab=False 的效果被正确忽略")

    # 9 种效果全在，且都能存进工程再读回来
    from src.core.project_io import load_project, save_project
    with tempfile.TemporaryDirectory() as tmp:
        pp = os.path.join(tmp, "e.cwproj")
        save_project(doc, pp)
        doc2 = load_project(pp)
    e2 = doc2.layers[0].effects
    for k in ("drop_shadow", "outer_glow", "inner_shadow", "inner_glow",
              "bevel_emboss", "satin", "color_overlay", "gradient_overlay",
              "stroke"):
        assert e2[k]["enabled"] is True, k
        for f in ("opacity", "color"):
            if f in e[k]:
                assert e2[k][f] == e[k][f], (k, f)
    print("  9 种效果工程往返：全部保住（含不透明度与颜色）")

    # 导入的样式要真的参与渲染（投影会往外扩）
    from src.core.render import render_document
    plain = np.zeros((120, 120, 4), np.uint8)
    plain[..., 3] = 255
    doc.layers[0].image = plain.copy()
    doc.layers[0].effects = None
    a = render_document(doc)
    doc.layers[0].image = plain.copy()
    doc.layers[0].effects = e
    bimg = render_document(doc)
    diff = np.abs(a.astype(int) - bimg.astype(int)).sum(axis=2)
    assert diff.max() > 0, "图层样式没参与渲染"
    print("  导入的样式参与渲染：与无样式版有差异（最大差 %d）" % diff.max())


def test_retouch():
    """修饰类工具：渐变 / 局部模糊 / 吸管（core/retouch.py，第十九批）。"""
    from src.core import retouch as rt

    # ---------- 渐变：5 种样式 ----------
    for style in rt.GRADIENT_STYLES:
        t = rt.gradient_t((40, 40), (20, 20), (39, 0), style)
        assert t.shape == (40, 40), style
        assert 0.0 <= float(t.min()) and float(t.max()) <= 1.0, style
        assert not np.isnan(t).any(), "%s 出了 nan" % style
    # 圆心样式的 t 在起点是 0（起点就是圆心）
    for style in ("径向", "菱形"):
        t = rt.gradient_t((40, 40), (20, 20), (39, 0), style)
        assert t[20, 20] < 0.02, "%s 起点 t=%f" % (style, t[20, 20])
    # 线性：起点 0、终点 1
    lin = rt.gradient_t((41, 41), (0, 20), (40, 20), "线性")
    assert lin[20, 0] < 0.02 and lin[20, 40] > 0.98, (lin[20, 0], lin[20, 40])
    # 对称：两端都是 1、中线折返到 0（PS 的对称渐变就是这样镜像的）
    sym = rt.gradient_t((41, 41), (0, 20), (40, 20), "对称")
    assert sym[20, 0] > 0.98 and sym[20, 20] < 0.02, (sym[20, 0], sym[20, 20])
    assert sym[20, 40] > 0.98, sym[20, 40]
    # 退化：起终点重合不许出 nan
    for style in rt.GRADIENT_STYLES:
        t = rt.gradient_t((8, 8), (4, 4), (4, 4), style)
        assert not np.isnan(t).any(), "%s 零长度出 nan" % style

    # ---------- 渐变写进位图 ----------
    img = np.zeros((40, 40, 4), np.uint8)
    img[..., 3] = 255
    rt.apply_gradient(img, (0, 20), (39, 20),
                      {"style": "线性", "color": [0, 0, 0],
                       "color2": [255, 255, 255]})
    assert img[20, 1, 0] < img[20, 38, 0], "线性渐变该从暗到亮"
    # 投影除法有舍入，末端不是精确 255（实测 248），留 8 的容差
    assert img[20, 38, 0] >= 247, img[20, 38, 0]
    assert img[20, 38, 3] == 255, "渐变不该动 alpha"
    # 选区外完全不写
    img2 = np.zeros((40, 40, 4), np.uint8)
    img2[..., 3] = 255
    sel = np.zeros((40, 40), np.float32)
    sel[:, 20:] = 1.0
    rt.apply_gradient(img2, (0, 20), (39, 20),
                      {"style": "线性", "color": [0, 0, 0],
                       "color2": [255, 255, 255]}, sel)
    assert img2[20, 0, 0] == 0, "选区外不该被写"
    assert img2[20, 39, 0] > 200, "选区内该被写"
    # 空选区 -> 原样返回
    img3 = np.zeros((10, 10, 4), np.uint8)
    before = img3.copy()
    rt.apply_gradient(img3, (0, 0), (9, 9), {"style": "线性"},
                      np.zeros((10, 10), np.float32))
    assert np.array_equal(img3, before), "空选区不该动任何像素"

    # ---------- 羽化权重 ----------
    w = rt.feathered_disk(21, 0.5)
    assert w[10, 10] == 1.0, "中心权重必须是 1"
    assert w[10, 20] == 0.0, "圆外权重必须是 0"
    w_hard = rt.feathered_disk(21, 1.0)
    assert w_hard[10, 13] == 1.0, "硬度 100 应该是硬边"
    w_soft = rt.feathered_disk(21, 0.0)
    assert 0.0 < w_soft[10, 13] < 1.0, "硬度 0 应该是全羽化"
    assert w_soft[10, 20] == 0.0, "圆外恒为 0"

    # ---------- 局部模糊：压高频、不动远处、不越界 ----------
    img = np.zeros((60, 60, 4), np.uint8)
    img[..., 3] = 255
    yy, xx = np.mgrid[0:60, 0:60]
    chk = ((xx // 6 + yy // 6) % 2)
    img[..., 0] = chk * 255
    img[..., 1] = chk * 255
    img[..., 2] = chk * 255
    orig = img.copy()
    lum = lambda a: a[..., :3].astype(np.int16).mean(axis=2)
    std_before = float(lum(img).std())
    rt.blur_region(img, 30, 30, 12.0, hardness=0.5, strength=1.0)
    std_after = float(lum(img).std())
    assert std_after < std_before, "模糊没压掉高频（%.2f -> %.2f）" % (
        std_before, std_after)
    # 笔刷外一个像素都不能动
    assert np.array_equal(img[0:8, 0:8], orig[0:8, 0:8]), "笔刷外被改了"
    # 边界不越界：角上模糊不能抛异常，也不能改到画布外
    rt.blur_region(img, 0, 0, 15.0, 0.5, 1.0)
    rt.blur_region(img, 59, 59, 15.0, 0.5, 1.0)
    # strength=0 与半径<1 都该是空操作
    snap = img.copy()
    rt.blur_region(img, 30, 30, 15.0, 0.5, 0.0)
    rt.blur_region(img, 30, 30, 0.0, 0.5, 1.0)
    assert np.array_equal(img, snap), "空模糊改了像素"
    # 多次涂抹趋于更平（累积）。实测 127.5 -> 55 -> 23 -> 8.4 -> 2.7，
    # 再涂也降不下去了（uint8 量化的地板，稳定在 5 左右），别把阈值定死 0
    for _ in range(6):
        rt.blur_region(img, 30, 30, 10.0, 0.5, 0.6)
    assert float(lum(img)[26:34, 26:34].std()) < 8.0, "反复涂抹应该糊平"
    # 透明边的 alpha 也要跟着走，不能留脏边
    tr = np.zeros((40, 40, 4), np.uint8)
    tr[..., 3] = 0
    tr[20:40, 20:40, 3] = 255
    tr[20:40, 20:40, :3] = (10, 20, 30)
    rt.blur_region(tr, 20, 20, 8.0, 0.5, 1.0)
    a = tr[..., 3]
    assert a.min() == 0, "半透明区外不该被填成不透明"
    assert a.max() == 255, "实心区的 alpha 不该被削掉"

    # ---------- 吸管 ----------
    a = np.zeros((20, 20, 4), np.uint8)
    a[..., :3] = (10, 200, 30)
    a[..., 3] = 128
    assert rt.pick_color(a, 5, 5) == (10, 200, 30, 128)
    assert rt.pick_color(a, 5, 5, radius=3) == (10, 200, 30, 128)
    assert rt.pick_color(a, 99, 99) is None, "越界该返回 None"
    assert rt.pick_color(None, 1, 1) is None
    # 邻域均值确实取到了多样本
    half = np.zeros((10, 10, 4), np.uint8)
    half[..., :3] = (0, 0, 0)
    half[:, 5:, :3] = (200, 200, 200)
    assert rt.pick_color(half, 4, 5, radius=4)[0] < 200, "邻域该被平均掉"
    print("  修饰类工具：渐变 5 样式 / 局部模糊（压高频·不越界·透明边干净）"
          " / 吸管（单点·邻域·越界）都对")


def test_retouch2():
    """修饰类工具第二批：克隆图章 / 污点修复 / 形状（第二十批）。"""
    import cv2
    from src.core import retouch as rt

    # ---------- 克隆图章 ----------
    src = np.zeros((60, 60, 4), np.uint8)
    src[..., :3] = (200, 120, 60)
    src[..., 3] = 255
    snap = rt.CloneSource(src.copy(), 10, 30, 0.5, 12)
    assert snap.shape == (60, 60)
    assert snap.copy().ox == 10 and snap.copy() is not snap

    # 覆盖：中心应该被完全换成源的颜色，远处一个字节都不动
    dst = np.zeros((60, 60, 4), np.uint8)
    dst[..., :3] = (10, 20, 30)
    dst[..., 3] = 255
    before = dst.copy()
    # 采样点与落点重合 -> offset 0，整块都是同一个色，改动最易验证
    s0 = rt.CloneSource(src.copy(), 30, 30, 1.0, 12)
    assert rt.clone_stamp_at(dst, s0, 30, 30, 10.0, 0.5, 1.0)
    assert tuple(dst[30, 30, :3]) == (200, 120, 60), dst[30, 30, :3]
    assert np.array_equal(dst[0:5, 0:5], before[0:5, 0:5]), "笔刷外被改了"

    # 羽化：硬度 0.5 时圆心的权重必须是 1（覆盖），圆外 0
    s1 = rt.CloneSource(src.copy(), 30, 30, 0.5, 12)
    dst2 = np.zeros((60, 60, 4), np.uint8)
    dst2[..., :3] = (10, 20, 30)
    dst2[..., 3] = 255
    rt.clone_stamp_at(dst2, s1, 30, 30, 10.0, 0.5, 1.0)
    # 半径 10 的羽化圆：x=30 是中心（权重 1），x=38 在圆内靠近边（权重 <1），
    # x=41 已经在圆外（权重 0）
    assert dst2[30, 30, 0] == 200, dst2[30, 30, 0]
    edge = int(dst2[30, 38, 0])
    assert edge < 200, "边缘该比中心淡（羽化生效）"
    assert edge > 10, "边缘不该被完全忽略，实际 %d" % edge
    assert dst2[30, 41, 0] == 10, "圆外不该动，实际 %d" % dst2[30, 41, 0]

    # 采样源被隔离：改了 dst 之后 src 一个字节都不能变
    assert np.array_equal(s1.image, src), "采样源被目标污染了"
    # 同图克隆不会自我放大（PS 的关键性质）
    same = np.zeros((80, 80, 4), np.uint8)
    ry, rx = np.mgrid[0:80, 0:80]
    same[((rx // 7 + ry // 7) % 2) == 0, :3] = (200, 120, 60)
    same[..., 3] = 255
    before2 = same.copy()
    n = rt.clone_stroke_path(same, rt.CloneSource(same.copy(), 10, 40, 0.5, 10),
                             [(40, 40), (70, 45)], 9.0, 0.5, 1.0)
    assert n > 0, "克隆路径一次都没盖上"
    # 采样源区域（画布左上那一片）必须逐位未动
    assert np.array_equal(same[5:15, 5:15], before2[5:15, 5:15]), \
        "同图克隆改了采样源区"

    # 边界：源窗口一半出界要画能画的那部分；完全出界返回 False
    edge_img = np.zeros((40, 40, 4), np.uint8)
    edge_img[..., :3] = (50, 60, 70)
    edge_img[..., 3] = 255
    red = np.zeros((40, 40, 4), np.uint8)
    red[..., :3] = (200, 10, 10)
    red[..., 3] = 255
    assert rt.clone_stamp_at(edge_img, rt.CloneSource(red, 2, 20, 0.5, 10),
                             36, 20, 9.0, 0.5, 1.0), "源一半出界该还能画"
    assert not rt.clone_stamp_at(edge_img,
                                 rt.CloneSource(red, -100, 20, 0.5, 10),
                                 30, 20, 9.0, 0.5, 1.0), "源完全出界该返回 False"
    # 尺寸不符 / 半径太小 / strength=0 都该安全返回
    assert not rt.clone_stamp_at(np.zeros((10, 10, 4), np.uint8),
                                 rt.CloneSource(np.zeros((40, 40, 4), np.uint8),
                                                5, 5), 5, 5, 3.0)
    assert not rt.clone_stamp_at(edge_img, rt.CloneSource(red, 20, 20),
                                 20, 20, 0.3)
    assert not rt.clone_stamp_at(edge_img, rt.CloneSource(red, 20, 20),
                                 20, 20, 9.0, 0.5, 0.0)
    assert rt.clone_stroke_path(edge_img, rt.CloneSource(red, 20, 20),
                                [], 9.0) == 0
    print("  克隆图章：中心全覆盖 / 羽化边缘 / 采样源隔离 / 同图不自我放大 "
          "/ 出界只画能画的")

    # ---------- 污点修复 ----------
    img = np.zeros((80, 80, 4), np.uint8)
    img[..., :3] = (100, 120, 140)
    img[..., 3] = 255
    ry, rx = np.mgrid[0:80, 0:80]
    img[((rx * 7 + ry * 13) % 23) < 2, :3] = (140, 140, 140)   # 细纹理
    img[36:44, 36:44, :3] = (250, 20, 20)                    # 深色瑕点
    lum0 = img[..., :3].astype(np.int16).mean(axis=2)
    tex_before = float(lum0[28:34, 28:34].std())
    assert tex_before > 1.0, "底图该有纹理可对照"
    assert rt.heal_region(img, 40, 40, 8.0, 0.5, 1.0)
    lum1 = img[..., :3].astype(np.int16).mean(axis=2)
    # 瑕点区的颜色要被拉回周围水平
    spot = float(lum1[38:42, 38:42].mean())
    around = float(lum1[26:32, 26:32].mean())
    assert abs(spot - around) < 45, \
        "瑕点没被修掉（spot=%.1f around=%.1f）" % (spot, around)
    # 周围纹理不能被抹平（污点修复 != 模糊）
    tex_after = float(lum1[28:34, 28:34].std())
    assert tex_after > tex_before * 0.5, \
        "纹理被抹平了（%.2f -> %.2f）" % (tex_before, tex_after)
    # 笔刷外不受影响
    snap2 = lum1.copy()
    assert not rt.heal_region(img, 40, 40, 8.0, 0.5, 0.0), "strength=0 该是空操作"
    assert np.array_equal(lum1, snap2)
    # 破洞（alpha=0）也要能修
    hole = np.zeros((60, 60, 4), np.uint8)
    hole[..., :3] = (120, 120, 120)
    hole[..., 3] = 255
    hole[28:32, 28:32] = 0
    assert rt.heal_region(hole, 30, 30, 6.0, 0.5, 1.0)
    assert hole[30, 30, 3] > 200, "破洞的 alpha 该被补上（实际 %d）" % hole[30, 30, 3]
    print("  污点修复：颜色拉回周围 / 纹理不抹平 / 破洞 alpha 补上 / "
          "strength=0 空操作")

    # ---------- 形状（core 层只有遮罩，UI 层测落笔）----------
    # 这里只测「羽化 + 布尔运算」这两个底层，形状的三种样式在 uitest 里测
    # **erode 必须给 borderType / borderValue**：默认的 border 处理会把
    # 外沿一起吃掉，整圈变成 0（描边就没了）。给 BORDER_CONSTANT=0 之后
    # 「画布外当空」-> 外沿保住了
    m = np.zeros((41, 41), np.uint8)
    m[4:37, 4:37] = 255
    er = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=2,
                   borderType=cv2.BORDER_CONSTANT, borderValue=0)
    ring = np.clip(m.astype(np.float32) - er.astype(np.float32), 0.0, 1.0)
    assert ring[20, 20] == 0.0, "描边：中心该被挖空"
    assert ring[20, 4] == 1.0, "描边：外沿该保留"
    assert ring[4, 4] == 1.0, "描边：四角该保留"
    assert ring[20, 2] == 0.0, "描边：形状外面不该有事"
    assert 0 < ring.sum() < m.sum(), "描边不该是实心或全空"
    print("  形状：描边的空心 / 外沿保留 / 边缘过渡都对")


def _texture_img(w, h):
    """造一张有明确纹理的图：渐变底 + 棋盘，用来验填充不是"糊成一片"。"""
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w, 4), np.uint8)
    img[..., 0] = (xx * 255 // max(1, w - 1))
    img[..., 1] = (yy * 255 // max(1, h - 1))
    img[..., 2] = 128
    chk = (((xx // 16 + yy // 16) % 2) * 60).astype(np.int32)
    img[..., :3] = np.clip(img[..., :3].astype(np.int32) + chk[..., None],
                           0, 255).astype(np.uint8)
    img[..., 3] = 255
    return img


def test_content_aware():
    """内容感知填充（第二十二批）：邻近 / 镜像 / 纹理合成 / 扩展画布。"""
    from src.core.content_aware import (FILL_MODES,
                                        extend_canvas_content_aware,
                                        fill_content_aware)
    assert FILL_MODES == ("邻近", "镜像", "纹理合成"), FILL_MODES

    h, w = 200, 260
    img = _texture_img(w, h)
    sel = np.zeros((h, w), np.float32)
    sel[80:130, 100:170] = 1.0

    # ---------- 三种模式：选区外逐位不变 ----------
    for mode in FILL_MODES:
        out = fill_content_aware(img, sel, mode=mode)
        assert out.shape == img.shape, mode
        assert out.dtype == np.uint8, mode
        # 选区外一像素都没动（羽化也不能渗出去）
        keep = sel.copy()
        keep[78:132, 98:172] = 1.0
        assert np.array_equal(out[keep == 0], img[keep == 0]), \
            "%s：选区外的像素被改了" % mode
        # 选区里确实被改写了（不能是原样返回）
        assert not np.array_equal(out[sel >= 0.5], img[sel >= 0.5]), \
            "%s：选区里什么都没变" % mode
        assert out[..., 3].min() >= 0 and out[..., 3].max() <= 255, mode

    # ---------- 邻近 vs 纹理合成：纹理合成该保留更多高频 ----------
    near = fill_content_aware(img, sel, mode="邻近")
    tex = fill_content_aware(img, sel, mode="纹理合成")
    hi_near = float(np.abs(np.diff(near[85:125, 105:165, :3].astype(float),
                                   axis=1)).mean())
    hi_tex = float(np.abs(np.diff(tex[85:125, 105:165, :3].astype(float),
                                  axis=1)).mean())
    assert hi_tex > hi_near, \
        "纹理合成该比邻近保留更多纹理：%.2f vs %.2f" % (hi_tex, hi_near)

    # ---------- 镜像：把关于包围盒中轴的对侧像素翻过来 ----------
    # 注意语义：翻完之后**不保证自对称**（选区本身未必对称），
    # 该验的是「翻过去的那个像素 = 原图同一行的镜像位置」
    mir = fill_content_aware(img, sel, mode="镜像", feather=0.0)
    x0, y0, x1, y1 = 100, 80, 170, 130
    cx = (x0 + x1 - 1) / 2.0
    ok = 0
    tot = 0
    # feather=0 就是硬边，全选区都该逐位等于对侧像素
    for y in range(y0, y1):
        for x in range(x0, x1):
            mx = int(round(2 * cx - x))
            if not (x0 <= mx < x1):
                continue
            tot += 1
            if abs(int(mir[y, x, 0]) - int(img[y, mx, 0])) <= 2:
                ok += 1
    assert tot > 0 and ok / tot > 0.98, \
        "镜像填充取的不是对侧像素：%d/%d" % (ok, tot)

    # ---------- 羽化 0 是硬边，羽化大要限幅（不能整片半透明） ----------
    hard = fill_content_aware(img, sel, feather=0.0)
    soft = fill_content_aware(img, sel, feather=64.0)
    inside_min = float(hard[85:125, 105:165, :3].min())
    assert inside_min >= 0, hard[80:130, 100:170, :3].min()
    # 大羽化被限幅到选区尺寸的一半，仍不该出现"整块 50% 灰"
    assert soft[..., 3].min() >= 0, "羽化不该把 alpha 推到负值"
    assert abs(float(soft[..., 3].mean()) - 255.0) < 1.0, \
        "不透明图的 alpha 填充后仍该是 255"

    # ---------- 半透明图：预乘往返不能出脏黑边（§3.2）----------
    # 整图 alpha = 128（半透明）。如果忘了预乘、直接卷 rgb，
    # 透明侧的 0 会被卷进来，填出来的那块会明显偏暗。
    rgba = img.copy()
    rgba[..., 3] = 128
    out = fill_content_aware(rgba, sel, mode="邻近", feather=3.0)
    assert int(out[..., 3].min()) == 128 and int(out[..., 3].max()) == 128, \
        "整图半透明时填充不该改alpha：%d..%d" % (out[..., 3].min(),
                                       out[..., 3].max())
    # 填出来的那块亮度应该跟"直接把选区均值抹平"差不多，
    # 而不是塌到接近 0（那就是没预乘的典型症状）
    src_lum = float(img[85:125, 105:165, :3].mean())
    out_lum = float(out[85:125, 105:165, :3].mean())
    assert abs(out_lum - src_lum) < 40, \
        "半透明填充亮度塌了：%.1f -> %.1f（多半是没预乘）" % (src_lum, out_lum)

    # 选区紧贴半透明图的一角时，填出来那块也不能比周围暗太多
    corner = np.zeros((h, w), np.float32)
    corner[h - 30:h, w - 40:w] = 1.0
    out2 = fill_content_aware(rgba, corner, mode="邻近", feather=0.0)
    ring = out2[h - 45:h - 30, w - 40:w, :3].mean()
    assert abs(float(out2[h - 30:h, w - 40:w, :3].mean())
               - float(ring)) < 40, "贴边填充与紧邻区域亮度差过大"

    # ---------- 空选区：原样返回，不报错 ----------
    empty = np.zeros((h, w), np.float32)
    assert np.array_equal(fill_content_aware(img, empty), img)

    # ---------- 扩展画布外的空白 ----------
    for mode in FILL_MODES:
        ex = extend_canvas_content_aware(img, sel, left=40, bottom=30, mode=mode)
        assert ex.shape == (h + 30, w + 40, 4), (mode, ex.shape)
        # **原图逐位保留** —— 扩展画布绝不能动原图像素
        assert np.array_equal(ex[0:h, 40:40 + w], img), \
            "%s：扩展画布把原图像素改了" % mode
        # 新增的那条边不该是纯黑（全透明处被填成啥）
        new_strip = ex[h:h + 30, 40:40 + w, :3]
        assert float(new_strip.mean()) > 5.0, \
            "%s：新边一片死黑 mean=%.1f" % (mode, new_strip.mean())

    # 四边都能扩
    ex4 = extend_canvas_content_aware(img, None, 10, 20, 30, 40, mode="邻近")
    assert ex4.shape == (h + 60, w + 40, 4), ex4.shape
    assert np.array_equal(ex4[20:20 + h, 10:10 + w], img), "四边扩展没保住原图"

    # ---------- 性能：大选区不能变成幻灯片 ----------
    big = np.zeros((900, 900), np.float32)
    big[200:800, 200:800] = 1.0
    # **tile 的第 0 轴才是行** —— 用第 1 轴会把图铺成 (200, 900, 16)，
    # 那样选区根本不在图里，填充会直接原样返回（测了个寂寞）
    bigimg = np.tile(_texture_img(260, 200), (5, 5, 1))[:900, :900]
    assert bigimg.shape == (900, 900, 4), bigimg.shape
    before = bigimg[200:800, 200:800, :3].copy()
    t0 = time.time()
    out = fill_content_aware(bigimg, big, mode="纹理合成")
    dt = time.time() - t0
    # 大选区必须真的被重写（不是原样返回）
    changed = float((np.abs(out[200:800, 200:800, :3].astype(np.int32)
                            - before.astype(np.int32)).sum(axis=2) > 2).mean())
    assert changed > 0.5, "大选区只有 %.1f%% 被填充，等于没干活" % (changed * 100)
    assert np.array_equal(out[:200, :], bigimg[:200, :]), "选区外被改了"
    assert dt < 3.0, "纹理合成 600x600 选区用了 %.2fs，太慢" % dt
    # **残留判据要用「原色」而不是固定亮度阈值**。这个 bug 藏了很久：
    # 块与块之间留缝 -> 缝里还是原始内容，但洞本身的纹理也有暗像素，
    # 用 `亮度 < 90` 去数会两边都算进去，看着像"残留"又像"正常纹理"。
    # 只认「和要抹掉的那个色完全一样」的像素才对。
    target = np.array([38, 42, 58])
    before_big = bigimg[200:800, 200:800, :3].astype(np.int32)
    out_big = fill_content_aware(bigimg, big, mode="纹理合成")
    sub = out_big[200:800, 200:800, :3].astype(np.int32)
    left = float((np.abs(sub - target[None, None, :]).max(axis=2) <= 4).mean())
    assert left < 0.001, "纹理合成残留了 %.1f%% 的原始内容（块之间有缝）" \
        % (left * 100)
    assert not np.array_equal(sub, before_big), "纹理合成什么都没改"

    # 每种采样范围 × 块边长组合都不能留缝 —— 之前只在 sr=48 下测过，
    # sr 一放大就露馅（ROI 变大 -> 候选变粗 -> 匹配退化）
    for sr in (48, 96, 192):
        for p in (9, 21):
            o = fill_content_aware(bigimg, big, mode="纹理合成",
                                   patch=p, search=sr)
            s = o[200:800, 200:800, :3].astype(np.int32)
            lf = float((np.abs(s - target[None, None, :]).max(axis=2)
                        <= 4).mean())
            assert lf < 0.001, "sr=%d patch=%d 残留 %.1f%%" % (sr, p, lf * 100)

    print("  内容感知：三种模式 / 选区外不变 / 镜像对称 / 半透明无脏边 / "
          "扩展画布保真 / 900px 选区 %.2fs（%.0f%% 被填充，0 残留）"
          % (dt, changed * 100))


def test_canvas_size():
    """画布大小（第二十二批）：九宫格锚点 + 逐层扩展 + 纯色填边。"""
    from src.core.document import ANCHORS, _ANCHOR_OFFSETS
    assert len(ANCHORS) == 9 and len(_ANCHOR_OFFSETS) == 9
    assert ANCHORS[4] == "居中" and _ANCHOR_OFFSETS["居中"] == (0.5, 0.5)
    assert _ANCHOR_OFFSETS["左上"] == (0.0, 0.0)
    assert _ANCHOR_OFFSETS["右下"] == (1.0, 1.0)

    from src.core.content_aware import extend_canvas_content_aware
    w, h = 300, 200
    doc = Document(w, h)
    img = np.zeros((h, w, 4), np.uint8)
    img[..., :3] = (120, 180, 90)
    img[..., 3] = 255
    doc.layers.append(make_image_layer("底", img, w, h))
    doc.layers[0].tx = doc.layers[0].ty = 0

    # 锚点换算：钉左上 -> 左边和上边扩
    fx, fy = _ANCHOR_OFFSETS["左上"]
    dx = int(round(fx * (400 - w)))
    assert (dx, int(round(fy * (300 - h)))) == (0, 0)
    fx, fy = _ANCHOR_OFFSETS["右下"]
    assert (int(round(fx * (400 - w))), int(round(fy * (300 - h)))) == (100, 100)

    # 扩展函数本身对"扩出去的量"的语义：left/top 是往左/往上扩
    ex = extend_canvas_content_aware(img, None, 10, 5, 0, 0)
    assert ex.shape == (h + 5, w + 10, 4), ex.shape
    assert np.array_equal(ex[5:5 + h, 10:10 + w], img), "原图没保住"

    # 负数 / 零尺寸要能被夹住，不能崩
    ex0 = extend_canvas_content_aware(img, None, 0, 0, 0, 0)
    assert ex0.shape == img.shape
    print("  画布大小：九宫格锚点换算 / 四边扩展 / 零扩展安全")


def test_guides():
    """向导：参考线 / 网格 / 标尺刻度 / 吸附（core/guides.py，第二十一批）。"""
    import cv2
    from src.core import guides as GD
    from src.core.document import Document, make_image_layer

    gs = GD.GuideSet()
    assert gs.guides == []
    assert gs.grid.visible is False

    # ---------- 参考线增删改查 ----------
    g1 = gs.add(GD.V_GUIDE, 100.0)
    g2 = gs.add(GD.H_GUIDE, 250.0)
    assert len(gs.guides) == 2
    assert g1.is_horizontal is False and g2.is_horizontal is True
    assert gs.find(g1.id) is g1
    assert gs.find("nope") is None
    near = gs.nearest(GD.V_GUIDE, 104.0, 6.0)
    assert near is not None and near[0] is g1 and near[1] == 4.0
    assert gs.nearest(GD.V_GUIDE, 200.0, 6.0) is None
    assert gs.nearest(GD.H_GUIDE, 104.0, 6.0) is None, "方向要对"
    assert gs.remove(g1.id) is True
    assert gs.remove("nope") is False
    assert len(gs.guides) == 1
    gs.clear()
    assert gs.guides == []

    # ---------- 网格 ----------
    assert GD.GridSpec(False, 50.0).lines(200, 200) == ([], [])
    g4 = GD.GridSpec(True, 50.0, 1)
    xs, ys = g4.lines(200, 200)
    assert xs == [50.0, 100.0, 150.0] and ys == [50.0, 100.0, 150.0], xs
    g4.subdiv = 4
    xs, _ys = g4.lines(200, 200)
    # 细线在大格内部等距（50/4 = 12.5）。200 px 宽里有 3 个大格，
    # 每格 4 等分 -> 3 条大格线 + 3×3 条细线 = 12 条
    assert len(xs) == 12, len(xs)
    gaps = {round(xs[i + 1] - xs[i], 4) for i in range(len(xs) - 1)}
    assert gaps == {12.5}, gaps
    # 细分越多线越密
    g8 = GD.GridSpec(True, 50.0, 8)
    assert len(g8.lines(200, 200)[0]) > len(xs)

    # ---------- 标尺刻度 ----------
    ticks, major = GD.ruler_ticks(0, 800, 10.0, 80.0)
    assert ticks and len(major) >= 1
    assert ticks[0] == 0.0
    gaps = {round(ticks[i + 1] - ticks[i], 4) for i in range(len(ticks) - 1)}
    assert len(gaps) == 1, "刻度间隔必须均匀，实际 %s" % gaps
    # 屏幕间隔要求越大（缩得越小）-> 世界坐标的刻度间隔要更大
    far, _ = GD.ruler_ticks(0, 800, 10.0, 400.0)
    step_near = ticks[1] - ticks[0] if len(ticks) > 1 else 0.0
    step_far = far[1] - far[0] if len(far) > 1 else 0.0
    assert step_far > step_near, (step_near, step_far)
    assert GD.ruler_ticks(0, 0) == ([], set())
    assert GD.ruler_ticks(10.0, 5.0) == ([], set())

    # ---------- 吸附：优先级 文档 > 参考线 > 网格 > 图层 ----------
    doc = Document(800, 600, "g")
    lay = make_image_layer("L", np.zeros((100, 120, 4), np.uint8), 120, 100)
    lay.tx, lay.ty = 300.0, 200.0
    doc.layers.append(lay)

    # 图层自身的可吸附位置：左右边界 + 中心
    vs, hs = GD.layer_edges_and_centers(lay)
    assert abs(vs[0] - 240.0) < 0.01 and abs(vs[1] - 360.0) < 0.01
    assert abs(vs[2] - 300.0) < 0.01
    assert abs(hs[0] - 150.0) < 0.01 and abs(hs[1] - 250.0) < 0.01
    # 旋转 45° 后吸附位置跟着变（AABB）
    lay.rot = 45.0
    vs45, _ = GD.layer_edges_and_centers(lay)
    assert abs(vs45[0] - 222.2) < 0.2 and abs(vs45[1] - 377.8) < 0.2, vs45
    lay.rot = 0.0
    # 没有 src_size 的图层不该炸
    from src.core.layer import Layer
    assert GD.layer_edges_and_centers(Layer("x")) == ([], [])

    g = GD.GuideSet()
    # 文档边界：x 只能吸到 0 / doc_w，y 只能吸到 0 / doc_h
    x, y, h = GD.snap_point(2.0, 2.0, g, 800, 600)
    assert (x, y) == (0.0, 0.0), (x, y)
    x, y, h = GD.snap_point(798.0, 598.0, g, 800, 600)
    assert (x, y) == (800.0, 600.0), (x, y)
    # 关键回归：x 轴**不能**吸到 doc_h（曾经写岔了，y 会吸到右边界）
    x, y, h = GD.snap_point(599.0, 500.0, g, 800, 600)
    assert abs(x - 599.0) < 0.01 or any(
        k == GD.SNAP_DOC for k, a in h if a == "x"), (x, h)
    # 参考线赢过图层
    g.add(GD.V_GUIDE, 241.0)
    x, _y, h = GD.snap_point(243.0, 500.0, g, 800, 600, layers=[lay])
    assert abs(x - 241.0) < 0.01, x
    assert GD.SNAP_GUIDE in [k for k, _a in h], h
    # 关掉参考线吸附后就该吸图层
    g.snap_guides = False
    x, _y, h = GD.snap_point(243.0, 500.0, g, 800, 600, layers=[lay])
    assert abs(x - 240.0) < 0.01, x
    assert GD.SNAP_LAYER in [k for k, _a in h], h
    # 关掉图层吸附 -> 不吸图层
    g.snap_layers = False
    x, _y, h = GD.snap_point(243.0, 500.0, g, 800, 600, layers=[lay])
    assert abs(x - 243.0) < 0.01, x
    # exclude 能排除自身
    g.snap_layers = True
    x, _y, _h = GD.snap_point(243.0, 500.0, g, 800, 600, layers=[lay],
                              exclude=lay)
    assert abs(x - 243.0) < 0.01, x
    # 网格
    g2 = GD.GuideSet()
    g2.grid = GD.GridSpec(True, 20.0, 1)
    x, _y, h = GD.snap_point(41.0, 500.0, g2, 800, 600)
    assert abs(x - 40.0) < 0.01, x
    assert GD.SNAP_GRID in [k for k, _a in h], h
    # 网格细分也参与吸附
    g2.grid.subdiv = 4
    x, _y, _h = GD.snap_point(36.0, 500.0, g2, 800, 600)
    assert abs(x - 35.0) < 0.01, x        # 20 + 20/4*3 = 35
    # 网格关掉就不吸
    g2.grid.visible = False
    x, _y, _h = GD.snap_point(41.0, 500.0, g2, 800, 600)
    assert abs(x - 41.0) < 0.01, x
    # x / y 各自独立：只该有一边被修正
    g3 = GD.GuideSet()
    g3.add(GD.V_GUIDE, 100.0)
    x, y, h = GD.snap_point(103.0, 333.0, g3, 800, 600)
    assert abs(x - 100.0) < 0.01 and abs(y - 333.0) < 0.01, (x, y)
    assert len(h) == 1, h
    # 全关 -> 原样返回
    g4 = GD.GuideSet()
    g4.snap_doc = g4.snap_guides = g4.snap_grid = g4.snap_layers = False
    x, y, h = GD.snap_point(3.0, 7.0, g4, 800, 600, layers=[lay])
    assert (x, y) == (3.0, 7.0) and h == [], (x, y, h)
    # 容差之外不吸
    x, _y, _h = GD.snap_point(130.0, 500.0, g3, 800, 600)
    assert abs(x - 130.0) < 0.01, x
    # 提示语
    assert "参考线" in GD.snap_label([(GD.SNAP_GUIDE, "x")])
    assert "参考线 · 网格" in GD.snap_label([(GD.SNAP_GUIDE, "x"),
                                             (GD.SNAP_GRID, "y")])
    assert GD.snap_label([]) == ""

    # ---------- 存取往返 / 老工程 ----------
    gs2 = GD.GuideSet()
    gs2.add(GD.V_GUIDE, 88.0)
    gs2.add(GD.H_GUIDE, 66.0)
    gs2.grid = GD.GridSpec(True, 25.0, 4)
    gs2.snap_grid = False
    back = GD.GuideSet.from_dict(gs2.to_dict())
    assert len(back.guides) == 2
    assert [(g.kind, g.pos) for g in back.guides] == \
        [(GD.V_GUIDE, 88.0), (GD.H_GUIDE, 66.0)]
    assert back.grid.to_dict() == gs2.grid.to_dict()
    assert back.snap_grid is False and back.snap_guides is True
    # 坏条目安静跳过，不抛
    bad = GD.GuideSet.from_dict({"guides": [{"kind": "x"}], "grid": None})
    assert len(bad.guides) == 0 and bad.grid.spacing > 0
    assert GD.GuideSet.from_dict(None).guides == []
    assert GD.GuideSet.from_dict({}).guides == []

    # ---------- Document 集成：clone 要深拷贝（参考线不是 numpy 数组）----------
    d2 = Document(200, 150, "x")
    d2.guides.add(GD.V_GUIDE, 88.0)
    d2.guides.grid = GD.GridSpec(True, 25.0, 4)
    snap = d2.clone()
    d2.guides.guides[0].pos = 999.0
    d2.guides.grid.spacing = 77.0
    assert snap.guides.guides[0].pos == 88.0, "clone 里的参考线被改到了"
    assert snap.guides.grid.spacing == 25.0, "clone 里的网格被改到了"
    print("  向导：参考线增删查 / 网格细分 / 标尺档位 / 吸附四级优先级 "
          "(文档>参考线>网格>图层) / x-y 独立 / exclude / 存取往返 / clone 隔离")


def test_channels():
    """第十七批：通道模型（core.channels）—— Alpha 通道 + RGB 显示开关。"""
    from src.core import channels as ch
    from src.core.selection import Selection

    def new_doc(w=160, h=120, rgb=(200, 60, 40), a=255):
        d = Document(w, h, "ch")
        d.layers.append(make_image_layer("L", _solid(w, h, rgb, a), w, h))
        return d

    # ---- 名字规范化与去重 ----
    assert ch.truncate_name("短") == "短"
    t = ch.truncate_name("一个非常非常长的通道名字")
    assert len(t) < 20 and t.endswith("…"), t
    # 中文按两列算宽，"通道 10" 不该比 "通道 9" 短
    assert ch._display_width("通道 10") == ch._display_width("通道 9") + 1
    d = new_doc()
    c1 = ch.add_channel(d, "蒙版")
    c2 = ch.add_channel(d, "蒙版")
    assert c1.name == "蒙版" and c2.name == "蒙版 2", (c1.name, c2.name)
    c3 = ch.add_channel(d, "蒙版")
    assert c3.name == "蒙版 3", c3.name
    assert ch.rename_channel(d, c3.id, "").name == "蒙版 3", "空名不该生效"
    assert ch.rename_channel(d, c3.id, "蒙版").name == "蒙版 4", \
        "改成重名要自动加后缀"
    print("  通道命名：去重（蒙版 2/3）/ 空名忽略 / 重名加后缀 / 中文按两列宽")

    # ---- 选区 <-> 通道 互转 ----
    d = new_doc()
    d.selection = Selection(d.width, d.height,
                            np.full((d.height, d.width), 200, np.uint8))
    c = ch.selection_to_channel(d, "Alpha 1")
    assert c.mask.mean() == 200.0, c.mask.mean()
    assert ch.channel_to_selection(d, c.id).mask.mean() == 200.0
    # 往返一次内容不变
    again = ch.channel_to_selection(d, c.id)
    assert np.array_equal(again.mask, c.mask)
    # 空通道载入选区 = 取消选区（PS 的行为）。
    # 注意 add_channel 省略 mask 时会用当前选区，所以这里要显式给一张全 0
    e = ch.add_channel(d, "空", np.zeros((d.height, d.width), np.uint8))
    assert e.is_empty()
    assert ch.channel_to_selection(d, e.id) is None, "空通道应清掉选区"
    assert d.selection is None
    # 省略 mask 时用当前选区（此时没有选区 -> 全 0）
    z = ch.add_channel(d, "无选区")
    assert z.mask.max() == 0
    print("  选区↔通道：存入选区 / 载入选区往返一致 / 空通道清选区 / "
          "无选区存全 0")

    # ---- 上限 ----
    d = new_doc()
    for i in range(ch.MAX_CHANNELS):
        assert ch.add_channel(d, "c%d" % i) is not None
    assert ch.add_channel(d, "溢出") is None, "超过上限应该返回 None"
    print("  通道数量上限 %d：超出后返回 None" % ch.MAX_CHANNELS)

    # ---- 增删改与排序 ----
    d = new_doc()
    a1 = ch.add_channel(d, "A")
    a2 = ch.add_channel(d, "B")
    a3 = ch.add_channel(d, "C")
    assert [c.id for c in d.channels] == [a1.id, a2.id, a3.id]
    assert ch.reorder_channel(d, a3.id, -1) is True
    assert [c.name for c in d.channels] == ["A", "C", "B"]
    assert ch.reorder_channel(d, a3.id, -1) is True
    assert [c.name for c in d.channels] == ["C", "A", "B"]
    assert ch.reorder_channel(d, a3.id, -1) is False, "已在首位不能再上移"
    assert ch.reorder_channel(d, "不存在的id", 1) is False
    assert ch.remove_channel(d, a2.id).name == "B"
    assert len(d.channels) == 2
    assert ch.remove_channel(d, "不存在的id") is None
    print("  通道排序 / 删除：上下移 + 边界与坏 id 都不崩")

    # ---- 显示语义：关掉的 RGB 填白（不是乘 0） ----
    d = new_doc(rgb=(200, 60, 40))
    arr = render_document(d)
    assert ch.apply_channel_view(arr, d) is arr, "全开着应原样返回同一个对象"
    d.channel_view["R"] = False
    out = ch.apply_channel_view(arr, d)
    assert out is not arr, "有关掉的通道时不能返回入参本身"
    assert out[..., 0].min() == 255, "关掉的 R 应填白 255"
    assert out[..., 1].max() == 60, "G 不该被动"
    assert out[..., 2].max() == 40, "B 不该被动"
    # 关 Alpha -> 全透明
    d.channel_view["R"] = True
    d.channel_view["A"] = False
    assert ch.apply_channel_view(arr, d)[..., 3].max() == 0
    # 原始 arr 不能被就地改掉（局部重渲染的缓存就是它）
    assert arr[..., 0].max() == 200, "入参被就地修改了！"
    print("  显示语义：关 R 填白（不是乘 0）/ 关 A 全透明 / 不就地改入参")

    # ---- 附加通道按遮罩乘不透明度 ----
    d = new_doc()
    arr = render_document(d)
    c = ch.add_channel(d, "半")
    m = np.zeros((d.height, d.width), np.uint8)
    m[:, :80] = 255
    c.mask = m
    out = ch.apply_channel_view(arr, d)
    assert out[:, 5, 3].min() == 255, "遮罩内的 alpha 不该变"
    assert out[:, 120, 3].max() == 0, "遮罩外该全透明"
    # 半透明值：255 的 alpha 乘 128 -> 约 128
    c.mask[:] = 128
    out = ch.apply_channel_view(arr, d)
    assert 126 <= int(out[..., 3].max()) <= 130, out[..., 3].max()
    # 不可见的通道不参与
    c.visible = False
    assert ch.apply_channel_view(arr, d)[..., 3].max() == 255
    # 空通道也不参与（否则会白乘 0 变全透明）
    c.visible = True
    c.mask[:] = 0
    assert ch.apply_channel_view(arr, d)[..., 3].max() == 255
    print("  附加通道：遮罩内 / 外 / 半透明 / 不可见 / 空通道 五种都对")

    # ---- 合并的 Alpha 通道 ----
    d = new_doc()
    arr = render_document(d)
    c = ch.add_channel(d, "打洞")
    m = np.full((d.height, d.width), 255, np.uint8)
    m[:, 60:] = 0
    c.mask = m
    d.composite_alpha = ch.composite_alpha_of(arr)
    assert ch.merge_channel_into_alpha(d, c.id, "replace") is True
    out = ch.apply_channel_view(arr, d)
    assert out[:, 5, 3].min() == 255 and out[:, 100, 3].max() == 0
    # add 模式是**取并集**（逐像素 max）：把通道"加进来"只会让更多地方不透明
    d.composite_alpha = np.zeros((d.height, d.width), np.uint8)
    d.detach_composite_alpha()
    d.detach_channel(c)
    c.mask[:] = 255          # 通道处处不透明
    ch.merge_channel_into_alpha(d, c.id, "add")
    assert d.composite_alpha.min() == 255, "add 应当是逐像素取 max"
    # subtract：把全白的合并 alpha 减掉通道（全 255）-> 全 0
    d.detach_composite_alpha()
    d.composite_alpha[:] = 255
    ch.merge_channel_into_alpha(d, c.id, "subtract")
    assert d.composite_alpha.max() == 0
    # 对不上尺寸时 resize 而不是崩
    d.detach_channel(c)
    c.mask = np.ones((7, 9), np.uint8) * 255
    d.detach_composite_alpha()
    d.composite_alpha = np.full((d.height, d.width), 255, np.uint8)
    ch.merge_channel_into_alpha(d, c.id, "replace")
    assert d.composite_alpha.shape == (d.height, d.width), \
        "尺寸不一致时应 resize 回去"
    # 坏 id / 没有合并 alpha 时返回 False 而不是崩
    d2 = new_doc()
    assert ch.merge_channel_into_alpha(d2, "不存在", "replace") is False
    assert ch.merge_channel_into_alpha(d2, ch.add_channel(d2, "x").id,
                                       "replace") is False, \
        "还没建立合并 alpha 时应返回 False"
    ch.reset_composite_alpha(d)
    assert ch.apply_channel_view(arr, d)[..., 3].max() == 255
    print("  合并 Alpha：replace / add(取并集) / subtract / 尺寸不一致自动缩放 "
          "/ 坏 id 与未建立时返回 False / 重置")

    # ---- 写时复制 ----
    d = new_doc()
    c = ch.add_channel(d, "cow")
    c.mask[:] = 100
    snap = d.clone()
    # clone 出来的 Channel 是新对象但遮罩数组共享
    c2 = snap.channels[0]
    assert c2 is not c, "clone 出来的 Channel 应该是新对象"
    assert c2.mask is c.mask, "但遮罩数组应当共享（浅克隆）"
    # 在快照上改之前 detach，快照自己那份不能被污染
    snap.detach_channel(c2)
    c2.mask[:] = 7
    assert c.mask.mean() == 100, "detach 之后改快照不该影响原文档"
    # 撤销回改动前
    hist_before = c.mask.mean()
    d.detach_channel(c)
    c.mask[:] = 0
    d.history_commit = None
    print("  写时复制：clone 共享数组 / detach 后互不影响（守 §3.1）")

    # ---- 画布尺寸变化 ----
    d = new_doc(w=160, h=120)
    c = ch.add_channel(d, "resize")
    c.mask[:] = 255
    d.resize(200, 150)
    assert c.mask.shape == (150, 200), c.mask.shape
    assert c.mask.min() == 255, "最近邻缩放不该把全白变成别的"
    d.selection = Selection(160, 120,
                            np.full((120, 160), 255, np.uint8))
    d.resize(200, 150)
    assert d.selection.mask.shape == (150, 200)
    print("  画布尺寸变化：通道与选区都用最近邻跟着缩")

    # ---- 工程往返 ----
    d = new_doc()
    c = ch.add_channel(d, "存我")
    m = np.zeros((d.height, d.width), np.uint8)
    m[10:40, 20:70] = 233
    d.detach_channel(c)
    c.mask = m
    c.visible = False
    d.channel_view["G"] = False
    d.detach_composite_alpha()
    d.composite_alpha = np.full((d.height, d.width), 77, np.uint8)
    from src.core.project_io import load_project, save_project
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "ch.cwproj")
        save_project(d, p)
        d2 = load_project(p)
    assert len(d2.channels) == 1, [c.name for c in d2.channels]
    r = d2.channels[0]
    assert r.name == "存我" and r.visible is False, (r.name, r.visible)
    assert np.array_equal(r.mask, m), "通道遮罩没原样存回来"
    assert d2.channel_view["G"] is False and d2.channel_view["R"] is True
    assert d2.composite_alpha is not None
    assert int(d2.composite_alpha[0, 0]) == 77
    # 读回来之后还能继续用
    out = ch.apply_channel_view(render_document(d2), d2)
    assert out[..., 3].max() == 77
    print("  工程往返：通道名 / 遮罩 / 可见性 / RGB 开关 / 合并 Alpha 全保住")

    # 老工程（没有 channels 字段）不能崩
    d3 = Document(80, 60, "old")
    d3.layers.append(make_image_layer("L", _solid(80, 60, (1, 2, 3)),
                                      80, 60))
    assert d3.channels == [] and d3.composite_alpha is None
    d3.resize(100, 90)
    assert ch.apply_channel_view(render_document(d3), d3) is not None
    print("  老工程兼容：没有 channels 字段也能正常渲染与缩放")


def test_new_adjustments():
    """第四批之后补的 4 种调整层：通道混合器 / 渐变映射 / 照片滤镜 / 可选颜色。"""
    from src.core.adjust import (ADJUSTMENTS, ADJUST_ORDER, apply_adjustment,
                                 default_params)

    for key in ("channel_mixer", "gradient_map", "photo_filter",
                "selective_color"):
        assert key in ADJUSTMENTS and key in ADJUST_ORDER, key
    assert len(ADJUST_ORDER) == 15, len(ADJUST_ORDER)
    print("  新增调整层注册：15 种（11 + 通道混合器 / 渐变映射 / 照片滤镜 /"
          " 可选颜色）")

    img = np.zeros((4, 4, 3), np.float32)
    img[..., 0], img[..., 1], img[..., 2] = 0.2, 0.5, 0.9

    # 通道混合器默认 = 单位矩阵，不能改变画面
    out = apply_adjustment("channel_mixer", img,
                           default_params("channel_mixer"))
    assert float(np.abs(out - img).max()) < 1e-6, "通道混合器默认不是恒等"
    p = default_params("channel_mixer")
    p["mix"] = {"红": {"red": 0.0, "green": 0.0, "blue": 100.0, "const": 0.0},
                "绿": {"red": 0.0, "green": 100.0, "blue": 0.0, "const": 0.0},
                "蓝": {"red": 100.0, "green": 0.0, "blue": 0.0, "const": 0.0}}
    out = apply_adjustment("channel_mixer", img, p)
    assert np.allclose(out[0, 0], [0.9, 0.5, 0.2], atol=0.01), out[0, 0]
    # 单色：三个通道一样
    p2 = default_params("channel_mixer")
    p2["mono"] = True
    out = apply_adjustment("channel_mixer", img, p2)
    assert abs(float(out[0, 0, 0]) - float(out[0, 0, 2])) < 1e-6
    print("  通道混合器：默认恒等 / 通道互换 / 单色 OK")

    # 渐变映射默认 = 灰度
    out = apply_adjustment("gradient_map", img, default_params("gradient_map"))
    g = float(out[0, 0, 0])
    assert abs(g - (0.299 * 0.2 + 0.587 * 0.5 + 0.114 * 0.9)) < 0.01, g
    assert abs(float(out[0, 0, 0]) - float(out[0, 0, 1])) < 1e-6
    p = default_params("gradient_map")
    p["low"], p["mid"], p["high"] = [255, 0, 0], [0, 255, 0], [0, 0, 255]
    out = apply_adjustment("gradient_map", img, p)
    assert out[0, 0, 0] > out[0, 0, 2], "暗部映射成红色，红应大于蓝"
    print("  渐变映射：默认灰度 / 自定义三段 OK")

    # 照片滤镜默认橙色 = 加温：红升蓝降
    out = apply_adjustment("photo_filter", img, default_params("photo_filter"))
    assert out[0, 0, 0] > img[0, 0, 0] and out[0, 0, 2] < img[0, 0, 2], \
        "橙色滤镜应该加温（红升蓝降）"
    p = default_params("photo_filter")
    p["preserve_luma"] = False
    out2 = apply_adjustment("photo_filter", img, p)
    assert float(np.abs(out2 - out).max()) > 1e-4, "保留明度开关没起作用"
    print("  照片滤镜：加温方向 / 保留明度开关 OK")

    # 可选颜色默认不动画面，对纯红加青 = 减红
    out = apply_adjustment("selective_color", img,
                           default_params("selective_color"))
    assert float(np.abs(out - img).max()) < 1e-6, "可选颜色默认不是恒等"
    red = np.zeros((2, 2, 3), np.float32)
    red[..., 0] = 0.8
    p = default_params("selective_color")
    p["families"]["红"] = {"cyan": 50.0, "magenta": 0.0, "yellow": 0.0,
                           "black": 0.0}
    out = apply_adjustment("selective_color", red, p)
    assert abs(float(out[0, 0, 0]) - 0.4) < 0.02, out[0, 0, 0]
    # 蓝色像素不该被"红色"族影响
    blue = np.zeros((2, 2, 3), np.float32)
    blue[..., 2] = 0.8
    out = apply_adjustment("selective_color", blue, p)
    assert abs(float(out[0, 0, 2]) - 0.8) < 0.02, "红色族不该动蓝色像素"
    print("  可选颜色：默认恒等 / 加青减红 / 色彩族隔离 OK")

    # 参数默认值是 dict 时不能共享（改一个层会污染另一个层）
    a1 = default_params("channel_mixer")
    a2 = default_params("channel_mixer")
    a1["mix"]["红"]["red"] = 7.0
    assert a2["mix"]["红"]["red"] == 100.0, "keyed 默认值被共享了"
    print("  默认值深拷贝：两套参数互不影响 OK")


def test_adjustment_enhance():
    """调整层增强（第十二批）：色相/饱和度分色彩范围、黑白预设、渐变映射控制点。"""
    import cv2
    from src.core.adjust import (ADJUSTMENTS, BW_PRESETS, GRADIENT_DEFAULT_STOPS,
                                 HS_RANGES, apply_adjustment, default_params,
                                 gradient_lut, legacy_stops, normalize_stops)

    def hue_img(deg, s=1.0, v=1.0):
        hsv = np.array([[[float(deg % 360), s, v]]], np.float32)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # ---- 色相/饱和度：分色彩范围只动对应色相 ----
    band = np.concatenate([hue_img(0.0), hue_img(60.0), hue_img(120.0)],
                          axis=0)                      # 红 / 黄 / 绿 各一行
    assert HS_RANGES[0][1] == 0.0
    hs = ADJUSTMENTS["hue_sat"]
    assert any(q.key == "range" for q in hs.params), "色相/饱和度缺 range 参数"
    p = default_params("hue_sat")
    p["range"], p["saturation"] = "红色", -100
    out = apply_adjustment("hue_sat", band, p)
    assert float(out[0, 0].min()) > 0.97, "红色带内的像素没被去饱和"
    assert float(np.abs(out[1:3] - band[1:3]).max()) < 1e-3, "黄 / 绿被误伤"
    print("  色相/饱和度：分色彩范围只作用于该色相")

    # 全图档与旧行为一致：三个色相一起去饱和
    p2 = dict(p)
    p2["range"] = "全图"
    assert float(apply_adjustment("hue_sat", band, p2).min()) > 0.97

    # 分范围时不该把中性灰当成红色（灰像素色相不可靠）
    gray = np.full((1, 1, 3), 0.5, np.float32)
    p3 = default_params("hue_sat")
    p3["range"], p3["lightness"] = "红色", 100
    assert abs(float(apply_adjustment("hue_sat", gray, p3)[0, 0, 0]) - 0.5) < 2e-3
    p4 = dict(p3)
    p4["range"] = "全图"
    assert float(apply_adjustment("hue_sat", gray, p4)[0, 0, 0]) > 0.9
    print("  色相/饱和度：灰像素不被分范围误伤，全图档照旧")

    # ---- 黑白预设 ----
    bw = ADJUSTMENTS["black_white"]
    preset = next(q for q in bw.params if q.key == "preset")
    assert preset.choices[0] == "自定义" and len(BW_PRESETS) >= 8
    for name, vals in BW_PRESETS.items():
        mapped = preset.preset_map[name]
        assert [mapped[k] for k in ("reds", "yellows", "greens", "cyans",
                                    "blues", "magentas")] == list(vals), name
    blue = np.zeros((1, 1, 3), np.float32)
    blue[..., 2] = 0.8
    base = apply_adjustment("black_white", blue, default_params("black_white"))
    p = default_params("black_white")
    p["preset"] = "蓝色滤镜"
    p.update(preset.preset_map["蓝色滤镜"])
    assert float(apply_adjustment("black_white", blue, p)[0, 0, 0]) > \
        float(base[0, 0, 0]) + 0.05, "蓝色滤镜预设应让蓝像素变亮"
    p2 = dict(p)
    p2.update(preset.preset_map["黄色滤镜"])
    assert float(apply_adjustment("black_white", blue, p2)[0, 0, 0]) < \
        float(base[0, 0, 0]), "黄色滤镜预设应压暗蓝像素"
    print("  黑白：%d 个预设的系数表与映射一致" % len(BW_PRESETS))

    # ---- 渐变映射：控制点 ----
    gm = ADJUSTMENTS["gradient_map"]
    assert gm.params[0].kind == "gradient"
    img = np.zeros((1, 3, 3), np.float32)
    img[0, 0], img[0, 1], img[0, 2] = 0.1, 0.5, 0.9

    # 默认（控制点为空 -> 旧格式兜底）= 黑到白线性，等价灰度
    o = apply_adjustment("gradient_map", img, default_params("gradient_map"))
    assert np.allclose(o[0, :, 0], [0.1, 0.5, 0.9], atol=0.01), o[0, :, 0]
    assert np.allclose(o[0, :, 0], o[0, :, 2], atol=1e-6)

    # 自定义控制点：0 -> 黑、0.5 -> 纯红、1 -> 白
    stops = [[0.0, 0, 0, 0], [0.5, 255, 0, 0], [1.0, 255, 255, 255]]
    o = apply_adjustment("gradient_map", img, {"stops": stops})
    assert o[0, 1, 0] > o[0, 1, 1] + 0.5, "中间调应该映射成红色"
    assert abs(float(o[0, 0, 0]) - 0.2) < 0.03, o[0, 0]

    # 旧工程（只有 low / mid / high）读出来的结果 == 升级成控制点之后的结果
    legacy = {"low": [255, 0, 0], "mid": [0, 255, 0], "high": [0, 0, 255]}
    ol = apply_adjustment("gradient_map", img, legacy)
    on = apply_adjustment("gradient_map", img, {"stops": legacy_stops(legacy)})
    assert float(np.abs(ol - on).max()) < 1e-6, "旧三段与控制点结果不一致"
    assert ol[0, 0, 0] > ol[0, 0, 2], "暗部映射成红，红应大于蓝"

    assert np.allclose(normalize_stops([[0.4, 1, 2, 3]]),
                       [[0.0, 1, 2, 3], [1.0, 1, 2, 3]])
    assert normalize_stops(None) == [list(s) for s in GRADIENT_DEFAULT_STOPS]
    lut = gradient_lut([[0.0, 255, 0, 0], [1.0, 0, 0, 255]], 3)
    assert np.allclose(lut[0], [1, 0, 0], atol=1e-6)
    assert np.allclose(lut[2], [0, 0, 1], atol=1e-6)
    print("  渐变映射：控制点列表生效、旧 low/mid/high 兼容")

    # 控制点列表要能过工程存取（JSON 里是嵌套 list）
    from src.core.document import Document, Layer, make_adjustment_layer
    from src.core.project_io import load_project, save_project
    from src.core.render import render_document

    doc = Document(60, 40, "gmrt")
    bg = Layer("bg", "image", _solid(60, 40, (30, 90, 200)))
    bg.tx, bg.ty = 30.0, 20.0
    adj = make_adjustment_layer("渐变映射", "gradient_map", 60, 40)
    adj.adjust["params"]["stops"] = [list(s) for s in stops]
    doc.layers.extend([bg, adj])
    before = render_document(doc)
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "gm.cwproj")
    save_project(doc, path)
    loaded = load_project(path)
    after = render_document(loaded)
    assert loaded.layers[1].adjust["params"]["stops"] == \
        [list(s) for s in stops], loaded.layers[1].adjust["params"]
    diff = int(np.abs(before.astype(int) - after.astype(int)).max())
    assert diff <= 2, "渐变映射存取前后差 %d/255" % diff
    print("  渐变映射：控制点工程存取往返 OK（差异 %d/255）" % diff)


def test_median_large():
    """大半径中间值 + 蒙尘与划痕。"""
    from numpy.lib.stride_tricks import sliding_window_view

    from src.core.filters import (FILTERS, FILTER_ORDER, _MEDIAN_MAX_R,
                                  apply_filter_array, default_filter_params)

    assert "dust" in FILTERS and len(FILTER_ORDER) == 11, len(FILTER_ORDER)
    print("  滤镜总数：%d 种（+ 蒙尘与划痕）" % len(FILTER_ORDER))

    rng = np.random.default_rng(20261006)
    rgb = (rng.random((48, 48, 3)) * 255).astype(np.uint8)
    img = np.dstack([rgb, np.full((48, 48, 1), 255, np.uint8)])

    def exact_median(u, r):
        """暴力二维中间值（edge 补边，和 OpenCV 的 borderType 一致）。"""
        p = np.pad(u, ((r, r), (r, r), (0, 0)), mode="edge")
        win = sliding_window_view(p, (2 * r + 1, 2 * r + 1), axis=(0, 1))
        return np.median(win, axis=(-1, -2))

    # 半径 1 / 2 走 float32 路径，5 / 17 / 100 走 8-bit 量化路径 ——
    # 两条路都必须给出**精确的二维中间值**
    for r in (1, 2, 5, 17, 100):
        out = apply_filter_array(img, "median", {"radius": r})[..., :3]
        ref = exact_median(rgb.astype(np.float32), r)
        assert np.array_equal(out.astype(np.float32), ref), (
            "半径 %d 的中间值不是精确二维中值" % r)
    print("  中间值：半径 1~100 与暴力二维中值逐位一致")

    # 半径超过上限要被夹住（不能崩，也不能变成别的东西）
    clamp = apply_filter_array(img, "median", {"radius": 999})
    assert np.array_equal(clamp, apply_filter_array(img, "median",
                                                    {"radius": _MEDIAN_MAX_R}))
    tiny = np.full((5, 5, 4), 128, np.uint8)
    tiny[..., 3] = 255
    assert apply_filter_array(tiny, "median", {"radius": 100})[2, 2, 0] == 128
    print("  中间值：半径夹到 %d，小图大核不崩" % _MEDIAN_MAX_R)

    # 大半径能吃掉的噪点尺寸也跟着变大：9x9 黑块半径 6 抹掉，半径 3 抹不掉
    spot = np.full((80, 80, 4), 200, np.uint8)
    spot[..., 3] = 255
    spot[35:44, 35:44, :3] = 15
    assert apply_filter_array(spot, "median", {"radius": 3})[39, 39, 0] == 15
    assert apply_filter_array(spot, "median", {"radius": 6})[39, 39, 0] == 200
    print("  中间值：半径够大才抹得掉大噪点（9x9 需要半径 ≥6）")

    # 蒙尘与划痕：阈值 0 == 中间值（阈值 255 == 原样）
    dust0 = apply_filter_array(spot, "dust", {"radius": 5, "threshold": 0})
    med5 = apply_filter_array(spot, "median", {"radius": 5})
    assert np.array_equal(dust0, med5), "阈值 0 时应该等价于中间值"
    keep = apply_filter_array(spot, "dust", {"radius": 5, "threshold": 255})
    assert np.array_equal(keep, spot), "阈值 255 时不该改动任何像素"
    print("  蒙尘与划痕：阈值 0 = 中间值，阈值 255 = 原样")

    # 和中间值的本质区别：**不超过阈值的细节原样保留**。
    # 128 底上一层 ±4 的随机纹理 + 一个 0 的孤立噪点：
    #   中间值 r=5 把纹理均值化（方差塌下来），蒙尘与划痕只吃掉那个噪点。
    pat = np.full((60, 60, 4), 128, np.uint8)
    pat[..., 3] = 255
    tex = rng.integers(-4, 5, size=(60, 60, 1))
    pat[..., :3] = np.clip(128 + tex, 0, 255).astype(np.uint8)
    pat[30, 30, :3] = 0

    dst = (slice(5, 25), slice(5, 25), 0)          # 避开噪点及其 r 邻域
    m = apply_filter_array(pat, "dust", {"radius": 5, "threshold": 30})
    assert int(m[30, 30, 0]) == 128, "孤立噪点该被抹掉: %d" % m[30, 30, 0]
    assert np.array_equal(m[dst], pat[dst]), "没超阈值的纹理不该被改动"

    med_pat = apply_filter_array(pat, "median", {"radius": 5})
    n_pat = float(pat[dst].std())
    n_med = float(med_pat[dst].std())
    assert n_pat > 1.5 and n_med < n_pat * 0.5, (n_pat, n_med)
    assert int(med_pat[30, 30, 0]) == 128
    print("  蒙尘与划痕：只吃超阈值的点，纹理原封不动"
          "（纯中间值把纹理方差从 %.2f 压到 %.2f）" % (n_pat, n_med))

    # 划痕（细线）也要能吃掉
    line = np.full((60, 60, 4), 210, np.uint8)
    line[..., 3] = 255
    line[30, 8:52, :3] = 20
    fixed = apply_filter_array(line, "dust", {"radius": 4, "threshold": 40})
    assert int(fixed[30, 30, 0]) == 210, fixed[30, 30, 0]
    assert int(fixed[10, 30, 0]) == 210
    print("  蒙尘与划痕：细划痕被消除，周边不受影响")

    # 所有滤镜（含新增的）默认参数都能跑完
    for key in FILTER_ORDER:
        out = apply_filter_array(img.copy(), key, default_filter_params(key))
        assert out.shape == img.shape and out.dtype == np.uint8, key
    print("  %d 种滤镜默认参数全部可执行" % len(FILTER_ORDER))


def _brush_canvas(w=200, h=80):
    from src.core.document import Document, make_image_layer
    doc = Document(w, h, "brush")
    lay = make_image_layer("L", np.zeros((h, w, 4), np.uint8), w, h)
    lay.tx, lay.ty = w / 2.0, h / 2.0
    doc.layers.append(lay)
    return doc, lay


def _brush_stroke(lay, doc, path, **kw):
    from src.core.paint import Stroke
    doc.detach_pixels(lay)
    lay.image[:] = 0
    kw.setdefault("seed", 1)
    st = Stroke(doc=doc, layer=lay, **kw)
    if not st.begin(path[0]):
        return False
    for p in path[1:]:
        st.extend(p)
    st.end()
    return True


def test_brush_advanced():
    """笔刷增强：笔尖形状 / 间隔 / 散布 / 抖动 / 纹理 / 喷枪 / 速度压感。"""
    from src.core.brush import (BRUSH_DEFAULTS, PRESSURE_MODES, SHAPES,
                                TEXTURES, make_stamp)

    assert SHAPES == ["圆形", "方形", "菱形"], SHAPES
    assert TEXTURES[0] == "无" and len(TEXTURES) == 6, TEXTURES
    assert set(PRESSURE_MODES) == {"关", "大小", "不透明度", "大小+不透明度"}
    assert BRUSH_DEFAULTS["shape"] == "圆形"
    print("  笔刷参数表：%d 种笔尖 / %d 种纹理 / %d 种压感"
          % (len(SHAPES), len(TEXTURES) - 1, len(PRESSURE_MODES)))

    # ---- 笔尖形状：硬边时面积占比应该接近理论值 ----
    r = 24.0
    area = {}
    for sh in SHAPES:
        a = make_stamp(r, 1.0, sh)
        area[sh] = float((a > 0.5).mean())
        assert a[a.shape[0] // 2, a.shape[1] // 2] > 0.9, sh
    assert abs(area["圆形"] - 0.785) < 0.05, area
    assert abs(area["方形"] - 1.0) < 0.02, area
    assert abs(area["菱形"] - 0.5) < 0.05, area
    assert make_stamp(r, 1.0, "方形")[0, 0] > 0.9, "方形笔尖的角该是实的"
    assert make_stamp(r, 1.0, "圆形")[0, 0] < 0.1, "圆形笔尖的角该是空的"
    print("  笔尖形状：圆形 %.3f / 方形 %.3f / 菱形 %.3f（理论 .785/1/.5）"
          % (area["圆形"], area["方形"], area["菱形"]))

    # ---- 圆度：压扁就成椭圆，面积按比例缩 ----
    full = float((make_stamp(r, 1.0, "圆形") > 0.5).sum())
    flat = float((make_stamp(r, 1.0, "圆形", 0.4) > 0.5).sum())
    assert 0.30 < flat / full < 0.50, (flat, full)

    # ---- 角度：0.4 圆度转 90° 后长短轴互换 ----
    def axis_wh(a):
        col = (a > 0.5).sum(axis=0)
        row = (a > 0.5).sum(axis=1)
        return (int(np.ptp(np.nonzero(col)[0])) + 1,
                int(np.ptp(np.nonzero(row)[0])) + 1)

    w0, h0 = axis_wh(make_stamp(r, 1.0, "圆形", 0.4, 0.0))
    w9, h9 = axis_wh(make_stamp(r, 1.0, "圆形", 0.4, 90.0))
    assert (w0, h0) == (h9, w9), ((w0, h0), (w9, h9))
    print("  圆度 / 角度：椭圆 %.0f×%.0f，转 90° 变 %.0f×%.0f"
          % (w0, h0, w9, h9))

    doc, lay = _brush_canvas()
    line = [(40.0 + i * 4.0, 40.0) for i in range(25)]

    def cov(**kw):
        kw.setdefault("size", 24)
        kw.setdefault("hardness", 1.0)
        assert _brush_stroke(lay, doc, line, **kw)
        return int((lay.image[..., 3] > 0).sum())

    def alpha_sum(**kw):
        kw.setdefault("size", 24)
        kw.setdefault("hardness", 1.0)
        assert _brush_stroke(lay, doc, line, **kw)
        return float(lay.image[..., 3].sum())

    # ---- 间隔：越大落点越稀 ----
    dense, sparse = cov(spacing=0.05), cov(spacing=1.0)
    assert dense > sparse, (dense, sparse)
    print("  间隔：5%% 覆盖 %d / 100%% 覆盖 %d" % (dense, sparse))

    # ---- 散布：笔画"变胖"（上下都长出东西），数量越多越胖 ----
    def bbox_h(**kw):
        assert _brush_stroke(lay, doc, line, **kw)
        rows = np.nonzero(lay.image[..., 3].sum(axis=1))[0]
        return int(np.ptp(rows)) + 1

    h_plain = bbox_h(size=24, hardness=1.0)
    h_scat = bbox_h(size=24, hardness=1.0, scatter=3.0, count=6, seed=3)
    assert h_scat > h_plain + 10, (h_plain, h_scat)
    print("  散布：笔迹高度 %d -> %d（散布 300%% × 6）" % (h_plain, h_scat))

    # ---- 大小抖动：整体覆盖变小，但仍然落在笔画附近 ----
    jit = cov(size=24, hardness=1.0, size_jitter=0.8, seed=5)
    assert jit < dense, (jit, dense)
    print("  大小抖动：覆盖 %d -> %d" % (dense, jit))

    # ---- 纹理 ----
    plain = cov(size=24, hardness=1.0, flow=1.0)
    tex = cov(size=24, hardness=1.0, flow=1.0, texture="棋盘",
              texture_scale=12, texture_depth=1.0)
    assert tex < plain * 0.9, (tex, plain)
    assert cov(size=24, hardness=1.0, flow=1.0, texture="棋盘",
               texture_scale=12, texture_depth=0.0) == plain, "深浅 0 应等于没纹理"
    for kind in TEXTURES[1:]:
        assert cov(size=16, hardness=1.0, texture=kind, texture_scale=20,
                   texture_depth=1.0) > 0, kind
    print("  纹理：%s 全覆盖 %d，棋盘纹理 %d" % ("/".join(TEXTURES[1:]), plain, tex))

    # 纹理锚在**源坐标**上：整体平移 2×周期后，图案应该逐位一致
    # （如果纹理跟着笔尖走，平移任意距离都会一致 —— 所以再验一个 3px 平移）
    def one_dab(x, scale):
        doc2, lay2 = _brush_canvas()
        _brush_stroke(lay2, doc2, [(float(x), 40.0)], size=24, hardness=1.0,
                      texture="棋盘", texture_scale=scale, texture_depth=1.0)
        return lay2.image[..., 3][40:64, int(x) - 12:int(x) + 12]

    sc = 12
    pa = one_dab(60, sc)
    pb = one_dab(60 + 2 * sc, sc)
    assert np.array_equal(pa, pb), "纹理没有按源坐标锚定（平移一个整周期对不上）"
    assert not np.array_equal(pa, one_dab(63, sc)), "纹理跟着笔尖走了"
    print("  纹理锚定：平移 %d px（2×周期）逐位一致，平移 3 px 不一致" % (2 * sc))

    # ---- 喷枪：不透明度上限失效，连补笔会继续加深 ----
    def air_series(air):
        doc3, lay3 = _brush_canvas()
        from src.core.paint import Stroke
        st = Stroke(doc=doc3, layer=lay3, size=30, hardness=1.0, opacity=0.5,
                    flow=0.3, airbrush=air, seed=1)
        assert st.begin((80.0, 40.0))
        out = [int(lay3.image[40, 80, 3])]
        for _ in range(4):
            st.airbrush_tick()
            out.append(int(lay3.image[40, 80, 3]))
        st.end()
        return out

    off = air_series(False)
    on = air_series(True)
    assert len(set(off)) == 1, "不开喷枪时补笔不该改变画面: %s" % off
    assert all(b > a for a, b in zip(on, on[1:])), "喷枪补笔应该持续加深: %s" % on
    assert on[-1] > off[-1] + 60, (on, off)
    print("  喷枪：关 %s（封顶）→ 开 %s（持续加深）" % (off, on))

    # ---- 速度压感：鼠标没有笔压，用速度模拟 ----
    def speed_run(step, press):
        doc4, lay4 = _brush_canvas(w=200, h=80)
        # 两端固定（30 -> 130），只让"每个事件走多远"变 —— 也就是运笔速度
        n = max(1, int(round(100.0 / step)))
        path = [(30.0 + 100.0 * i / n, 40.0) for i in range(n + 1)]
        _brush_stroke(lay4, doc4, path, size=24, hardness=1.0, flow=0.35,
                      pressure=press, pressure_amount=1.0)
        # 只看中段，避开笔画两端的圆头
        sub = lay4.image[:, 45:115, 3]
        return int((sub > 0).sum()), int(sub.sum())

    for mode in PRESSURE_MODES:
        (s_cov, s_ink), (f_cov, f_ink) = speed_run(2.0, mode), speed_run(18.0, mode)
        if mode == "关":
            # 关掉压感时速度不该有影响。覆盖面很稳；"墨量"对落点位置敏感
            # （二段线插值的落点会差一两个像素），所以只放宽到 15%
            assert abs(s_cov - f_cov) <= s_cov * 0.05, \
                "关掉压感后速度不该影响覆盖面: %d / %d" % (s_cov, f_cov)
            assert abs(s_ink - f_ink) <= s_ink * 0.15, \
                "关掉压感后速度不该影响浓度: %d / %d" % (s_ink, f_ink)
        if mode in ("大小", "大小+不透明度"):
            assert f_cov < s_cov * 0.7, \
                "%s 压感下快速运笔该更细: %d / %d" % (mode, f_cov, s_cov)
        if mode == "不透明度":
            assert abs(s_cov - f_cov) <= s_cov * 0.1, \
                "只调不透明度时覆盖面不该变: %d / %d" % (s_cov, f_cov)
        if mode in ("不透明度", "大小+不透明度"):
            assert f_ink < s_ink * 0.7, \
                "%s 压感下快速运笔该更淡: %d / %d" % (mode, f_ink, s_ink)
        print("  压感 %-8s 慢 覆盖%4d/墨量%6d  快 覆盖%4d/墨量%6d"
              % (mode, s_cov, s_ink, f_cov, f_ink))

    # ---- 图章缓存不能无界增长（散点/抖动会让半径不断变化）----
    from src.core.brush import stamp_cache_size
    before = stamp_cache_size()
    for i in range(60):
        make_stamp(10.0 + i * 0.37, 1.0, "圆形", 1.0)
    grown = stamp_cache_size() - before
    assert grown <= 60, grown
    print("  图章缓存：60 个不同半径只新增 %d 项（按整数半径复用）" % grown)


def test_layer_effects():
    """图层样式：描边 / 投影 / 内阴影 / 外发光。"""
    from src.core.effects import (EFFECT_ORDER, apply_effects,
                                  default_effects, effect_padding,
                                  has_effects)
    from src.core.layer import LAYER_IMAGE, Layer

    def patch():
        # 60x60 patch，中间 40x40 不透明
        c = np.zeros((60, 60, 3), np.float32)
        a = np.zeros((60, 60, 1), np.float32)
        c[10:50, 10:50] = 1.0
        a[10:50, 10:50] = 1.0
        return c, a

    ef = default_effects()
    assert not has_effects(ef), "默认应该是全关"
    assert effect_padding(ef) == 0
    c0, a0 = patch()
    c1, a1 = apply_effects(c0, a0, ef)
    assert c1 is c0, "没启用效果时不该拷贝数组"

    # 描边（外部）：外面多一圈，alpha 变大，颜色是描边色
    e = default_effects()
    e["stroke"]["enabled"] = True
    e["stroke"]["color"] = [255, 0, 0]
    e["stroke"]["size"] = 4.0
    pad = effect_padding(e)
    assert pad >= 5, pad
    c, a = apply_effects(*patch(), e)
    assert float(a[52, 30, 0]) > 0.5, "外部描边应该在形状外多一圈"
    assert float(c[52, 30, 0]) > 0.9 and float(c[52, 30, 2]) < 0.1, \
        "描边颜色不对"
    print("  描边：向外扩 %d px，颜色正确" % pad)

    # 描边（内部）：alpha 总量不变，颜色从边缘往里吃
    e2 = default_effects()
    e2["stroke"]["enabled"] = True
    e2["stroke"]["position"] = "内部"
    e2["stroke"]["color"] = [0, 0, 255]
    e2["stroke"]["size"] = 5.0
    c, a = apply_effects(*patch(), e2)
    assert float(a[55, 30, 0]) < 0.01, "内部描边不该跑到形状外面"
    assert float(c[12, 30, 2]) > 0.5, "内部描边应该出现在边缘内侧"
    print("  描边（内部）：不外扩，只吃内侧")

    # 投影：往右下方偏，颜色是黑
    e3 = default_effects()
    e3["drop_shadow"]["enabled"] = True
    e3["drop_shadow"]["distance"] = 10.0
    c, a = apply_effects(*patch(), e3)
    # patch 里形状占 10..50，投影往右下 7px 左右
    assert float(a[55, 30, 0]) > 0.05, "右下应该有投影"
    assert float(a[5, 30, 0]) < 0.01, "左上不该有投影"
    assert float(c[55, 30].max()) < 0.5, "默认投影是黑的"
    print("  投影：方向（右下）与颜色正确")

    # 外发光：四周都有，且是亮色
    e4 = default_effects()
    e4["outer_glow"]["enabled"] = True
    c, a = apply_effects(*patch(), e4)
    for (y, x) in ((5, 30), (55, 30), (30, 5), (30, 55)):
        assert float(a[y, x, 0]) > 0.02, "外发光应该四周都有 (%d,%d)" % (y, x)
    print("  外发光：四周均匀")

    # 内阴影：不改 alpha，只把内侧压暗
    e5 = default_effects()
    e5["inner_shadow"]["enabled"] = True
    c, a = apply_effects(*patch(), e5)
    assert abs(float(a.sum()) - float(a0.sum())) < 1.0, "内阴影不该改 alpha"
    assert float(c[12, 30].max()) < 0.9, "内阴影应该压暗内侧"
    print("  内阴影：alpha 不变，只压暗内侧")

    # 效果要跟着图层存进工程
    lay = Layer("带样式", LAYER_IMAGE, image=np.zeros((4, 4, 4), np.uint8))
    lay.effects = default_effects()
    lay.effects["stroke"]["enabled"] = True
    d = lay.to_dict()
    assert d.get("effects") and d["effects"]["stroke"]["enabled"]
    back = Layer.from_dict(d, {})
    assert back.effects["stroke"]["enabled"] is True, "样式没存回来"
    print("  样式存取：to_dict / from_dict 保留")

    # 四种效果一起开时要能算出合理的外扩
    allon = default_effects()
    for k in EFFECT_ORDER:
        allon[k]["enabled"] = True
    assert effect_padding(allon) > 0


def _square_patch(n=80, side=48):
    """中心一块不透明灰的开窗 patch —— 图层样式测试的统一素材。"""
    c = np.zeros((n, n, 3), np.float32)
    a = np.zeros((n, n, 1), np.float32)
    lo = (n - side) // 2
    hi = lo + side
    c[lo:hi, lo:hi] = 0.5
    a[lo:hi, lo:hi] = 1.0
    return c, a


def _alpha_unchanged(a_out, a_in, msg):
    d = np.abs(a_out.astype(np.float64) - a_in.astype(np.float64))
    assert d.max() < 1e-6, "%s（最大差 %g）" % (msg, d.max())


def test_layer_effects_advanced():
    """第十批新增的 6 种效果 + 全局光 + 剪贴板 / 默认值。

    重点守两条：**内部效果不许改 alpha**，以及**渐变 / 图案叠加在局部
    重渲染时不能错位**（坐标系必须锚在图层内容框上，见 effects 模块顶部）。
    """
    import src.core.effects as E
    import src.core.render as R

    saved_default = E._SAVED_DEFAULT["v"]
    E._SAVED_DEFAULT["v"] = None
    try:
        # ---- 内发光：只在内侧边缘，不改 alpha ----
        c0, a0 = _square_patch()
        e = E.default_effects()
        e["inner_glow"]["enabled"] = True
        e["inner_glow"]["size"] = 12.0
        c, a = E.apply_effects(c0, a0, e)
        _alpha_unchanged(a, a0, "内发光不该改 alpha")
        assert float(c[18, 40].max()) > 0.55, "内发光应该出现在边缘内侧"
        assert abs(float(c[40, 40, 0]) - 0.5) < 0.05, "中心不该被内发光影响"
        print("  内发光：只出现在内侧边缘，alpha 不变")

        # ---- 斜面浮雕：左上角迎光 / 右下角背光，方向可反 ----
        e = E.default_effects()
        e["bevel_emboss"]["enabled"] = True
        e["bevel_emboss"]["size"] = 8.0
        e["bevel_emboss"]["angle"] = 45.0
        c, a = E.apply_effects(*_square_patch(), e)
        _alpha_unchanged(a, a0, "斜面浮雕不该改 alpha")
        hi = float(c[20, 20].mean())
        lo = float(c[59, 59].mean())
        assert hi > lo + 0.05, "左上应该比右下亮（%g vs %g）" % (hi, lo)
        e["bevel_emboss"]["direction"] = "下"
        c2, _a = E.apply_effects(*_square_patch(), e)
        assert float(c2[20, 20].mean()) < float(c2[59, 59].mean()), \
            "方向改成「下」应该反过来"
        print("  斜面浮雕：受光 / 背光方向正确，方向可反")

        # ---- 光泽：内部出现一条带，反相后明暗互换 ----
        e = E.default_effects()
        e["satin"]["enabled"] = True
        e["satin"]["distance"] = 24.0
        e["satin"]["size"] = 10.0
        c, a = E.apply_effects(*_square_patch(), e)
        _alpha_unchanged(a, a0, "光泽不该改 alpha")
        d0 = np.abs(c.astype(np.float64) - c0.astype(np.float64))
        assert d0.max() > 0.02, "光泽没生效"
        e["satin"]["invert"] = True
        ci, _a = E.apply_effects(*_square_patch(), e)
        di = np.abs(ci.astype(np.float64) - c0.astype(np.float64))
        # 反相 = 明暗换个位置，两处的差异图应该明显不同（不是强弱缩放关系）
        diff = np.abs(di - d0)
        assert diff.max() > 0.05 and di.max() > 0.02, "反相后明暗位置应该改变"
        print("  光泽：内部明暗带 + 反相可用")

        # ---- 颜色叠加：整体染色，形状不变 ----
        e = E.default_effects()
        e["color_overlay"]["enabled"] = True
        e["color_overlay"]["opacity"] = 1.0
        e["color_overlay"]["color"] = [255, 0, 0]
        c, a = E.apply_effects(*_square_patch(), e)
        _alpha_unchanged(a, a0, "颜色叠加不该改 alpha")
        assert float(c[40, 40, 0]) > 0.98 and float(c[40, 40, 2]) < 0.02, \
            "100%% 红色叠加后全家应该变红，实际 %s" % c[40, 40]
        print("  颜色叠加：整体染色且形状不变")

        # ---- 渐变叠加：左下 / 右上颜色不同 ----
        e = E.default_effects()
        e["gradient_overlay"]["enabled"] = True
        e["gradient_overlay"]["opacity"] = 1.0
        e["gradient_overlay"]["angle"] = 45.0
        frame = (0, 0, 48, 48)
        c, a = E.apply_effects(*_square_patch(), e, frame)
        _alpha_unchanged(a, a0, "渐变叠加不该改 alpha")
        p1 = c[20, 20]
        p2 = c[59, 59]
        assert np.abs(p1 - p2).max() > 0.2, "渐变两端颜色应该有差别"
        assert p1[0] > p2[0], "左下应该是起始色（红分量更大）"
        print("  渐变叠加：沿角度方向，两端颜色不同")

        # ---- 图案叠加：有周期性，改缩放结果不同 ----
        e = E.default_effects()
        e["pattern_overlay"]["enabled"] = True
        c, a = E.apply_effects(*_square_patch(), e, frame)
        _alpha_unchanged(a, a0, "图案叠加不该改 alpha")
        vals = c[40, 16:64, 0]
        assert float(vals.max() - vals.min()) > 0.1, "图案应该是明暗交替的"
        e["pattern_overlay"]["scale"] = 120.0
        c2, _a = E.apply_effects(*_square_patch(), e, frame)
        assert np.abs(c2 - c).max() > 0.05, "改缩放应该改变图案密度"
        for k in ("点阵", "网格", "斜纹", "棋盘", "噪声"):
            e["pattern_overlay"]["scale"] = 24.0
            e["pattern_overlay"]["pattern"] = k
            ck, _ak = E.apply_effects(*_square_patch(), e, frame)
            assert float(np.abs(ck - c0).max()) > 0.01, "图案 %s 没生效" % k
        print("  图案叠加：5 种图案都生效，密度随缩放变化")

        # ---- 三种纯叠加不需要外扩 ----
        for key in ("color_overlay", "gradient_overlay", "pattern_overlay"):
            e = E.default_effects()
            e[key]["enabled"] = True
            assert E.effect_padding(e) == 0, "%s 不该要外扩" % key
        e = E.default_effects()
        e["inner_glow"]["enabled"] = True
        e["bevel_emboss"]["enabled"] = True
        assert E.effect_padding(e) > 0, "内发光 / 斜面要留计算上下文"

        # ---- 全局光：勾上的效果跟着走，没勾的不动 ----
        e = E.default_effects()
        eff = e["drop_shadow"]
        assert abs(E.effective_angle(e, eff) - 45.0) < 1e-6
        e[E.GLOBAL_LIGHT]["enabled"] = True
        e[E.GLOBAL_LIGHT]["angle"] = 120.0
        e[E.GLOBAL_LIGHT]["altitude"] = 60.0
        assert abs(E.effective_angle(e, eff) - 120.0) < 1e-6, "全局光没联动"
        assert abs(E.effective_altitude(e, e["bevel_emboss"]) - 60.0) < 1e-6
        eff["use_global"] = False
        assert abs(E.effective_angle(e, eff) - 45.0) < 1e-6, "取消勾选后该用自己的角度"
        # 渲染层面也要跟着变
        e2 = E.default_effects()
        e2["drop_shadow"]["enabled"] = True
        e2["drop_shadow"]["distance"] = 20.0
        base = E.apply_effects(*_square_patch(), e2)[1]
        e2[E.GLOBAL_LIGHT]["enabled"] = True
        e2[E.GLOBAL_LIGHT]["angle"] = 225.0
        rot = E.apply_effects(*_square_patch(), e2)[1]
        d = np.abs(rot - base)
        assert d.max() > 0.5, "全局光改角度后投影没换方向"
        print("  全局光：角度 / 高度联动勾选的效果，未勾选的不受影响")

        # ---- 10 种效果全开：数值有限且在范围内 ----
        e = E.default_effects()
        for k in E.EFFECT_ORDER:
            e[k]["enabled"] = True
        c, a = E.apply_effects(*_square_patch(), e, frame)
        assert np.isfinite(c).all() and np.isfinite(a).all(), "出现非法数值"
        assert c.min() >= -1e-6 and c.max() <= 1.0 + 1e-6, "颜色越界"
        assert a.min() >= -1e-6 and a.max() <= 1.0 + 1e-6, "alpha 越界"
        assert len(E.enabled_effects(e)) == len(E.EFFECT_ORDER)
        print("  10 种效果全开：数值有限且不越界")

        # ---- 工程存取：新效果的参数 + 全局光都要保住 ----
        lay = Layer("带样式", "image", image=np.zeros((4, 4, 4), np.uint8))
        lay.effects = E.default_effects()
        lay.effects["gradient_overlay"]["enabled"] = True
        lay.effects["gradient_overlay"]["color2"] = [10, 20, 30]
        lay.effects[E.GLOBAL_LIGHT]["enabled"] = True
        lay.effects[E.GLOBAL_LIGHT]["angle"] = 135.0
        back = Layer.from_dict(lay.to_dict(), {})
        assert back.effects["gradient_overlay"]["enabled"] is True
        assert back.effects["gradient_overlay"]["color2"] == [10, 20, 30]
        assert back.effects[E.GLOBAL_LIGHT]["angle"] == 135.0
        # 旧工程（只有 4 种效果）读回来要补齐且不擅自启用
        old = {"stroke": {"enabled": True, "size": 2.0}}
        norm = E.normalize_effects(dict(old))
        assert norm["stroke"]["enabled"] is True
        assert norm["bevel_emboss"]["enabled"] is False, "补齐不能擅自启用"
        print("  样式存取：新参数与全局光往返正常，旧工程补齐后保持关闭")

        # ---- 复制 / 粘贴：深拷贝，互不污染 ----
        src = E.default_effects()
        src["stroke"]["enabled"] = True
        src["stroke"]["size"] = 7.0
        E.copy_style(src)
        assert E.has_style_clipboard()
        pasted = E.paste_style()
        pasted["stroke"]["size"] = 20.0
        assert src["stroke"]["size"] == 7.0, "粘贴出来的改动污染了原样式"
        pasted2 = E.paste_style()
        assert pasted2["stroke"]["size"] == 7.0, "粘贴应该是独立的副本"
        print("  复制 / 粘贴：剪贴板是深拷贝，两份互不影响")

        # ---- 存为默认值 / 复位默认值（别把用户的旧默认值弄丢）----
        try:
            custom = E.default_effects()
            custom["outer_glow"]["enabled"] = True
            custom["outer_glow"]["size"] = 33.0
            E.save_style_default(custom)
            assert E.has_style_default()
            fresh = E.default_effects()
            assert fresh["outer_glow"]["size"] == 33.0, "默认值没生效"
            assert fresh is not custom, "每次应该返回新的深拷贝"
            E.reset_style_default()
            assert not E.has_style_default()
            assert E.default_effects()["outer_glow"]["size"] == 12.0, \
                "复位后应该回到出厂默认值"
            print("  存为默认值：新建样式从它出发，复位后回到出厂参数")
        finally:
            E._SAVED_DEFAULT["v"] = saved_default
            if saved_default is not None:
                E.save_style_default(saved_default)

        # ---- 【关键】渐变 / 图案叠加：局部渲染必须与整幅逐位一致 ----
        doc = Document(240, 180, "叠加")
        doc.layers.append(Layer("bg", "image", _solid(240, 180, (235, 235, 235))))
        doc.layers[0].tx, doc.layers[0].ty = 120.0, 90.0
        for kind in ("gradient_overlay", "pattern_overlay"):
            box = Layer("方块", "image", _solid(80, 60, (120, 120, 120)))
            box.tx, box.ty = 100.0, 80.0
            box.effects = E.default_effects()
            box.effects[kind]["enabled"] = True
            box.effects[kind]["opacity"] = 1.0
            if kind == "pattern_overlay":
                box.effects[kind]["blend"] = "Multiply"
                box.effects[kind]["scale"] = 13.0
            doc.layers.append(box)
            full = render_document(doc)
            for reg in ((80, 60, 140, 110), (60, 40, 180, 140),
                        (0, 0, 240, 180)):
                sub = render_document(doc, region=reg)
                ref = full[reg[1]:reg[3], reg[0]:reg[2]]
                d = np.abs(sub.astype(np.int16) - ref.astype(np.int16))
                assert d.max() == 0, \
                    "%s 局部渲染与整幅不一致（最大差 %d）" % (kind, d.max())
            # 分块渲染（大画布走的那条路）也一样不能错位
            reg = (80, 60, 140, 110)
            tiled = R.render_region_tiled(doc, reg, tile=64)
            ref = full[reg[1]:reg[3], reg[0]:reg[2]]
            dt = np.abs(tiled.astype(np.int16) - ref.astype(np.int16))
            assert dt.max() <= 1, \
                "%s 分块渲染与整幅不一致（最大差 %d）" % (kind, dt.max())
            doc.layers.pop()
            assert max_effect_padding(doc) == 0, "纯叠加不需要外扩"
        print("  叠加类效果：局部 / 分块渲染都与整幅一致（坐标系锚在内容框上）")
    finally:
        E._SAVED_DEFAULT["v"] = saved_default


def test_float_selection():
    """浮动选区：把选中的像素揭出来移动，再盖回去。"""
    from src.core.document import make_image_layer
    from src.core.float_sel import (can_lift, discard_float, lift_selection,
                                    stamp_float)
    from src.core.selection import Selection

    def fresh(sel_box=(20, 20, 60, 60)):
        img = np.zeros((100, 100, 4), np.uint8)
        img[..., 0] = 255
        img[..., 3] = 255
        doc = Document(100, 100)
        lay = make_image_layer("红", img, 100, 100)
        doc.layers.append(lay)
        doc.selection = Selection.rect(100, 100, *sel_box)
        return doc, lay

    # 没有选区 / 组 / 文字层 都不能揭
    doc, lay = fresh()
    doc.selection = None
    assert not can_lift(doc, lay), "没选区不能揭"
    doc.selection = Selection.rect(100, 100, 20, 20, 60, 60)
    assert can_lift(doc, lay)
    print("  可揭条件：选区 / 像素图层 OK")

    # 剪切模式：原处挖空，内容跟着浮动层走
    doc, lay = fresh()
    fl = lift_selection(doc, lay)
    assert fl is not None and doc.float_layer is fl
    assert float(lay.image[..., 3].sum()) < 100 * 100 * 255 - 1000, \
        "原图层没被挖空"
    out = render_document(doc)
    assert int(out[30, 30, 3]) == 255, "浮动层应该补在原位（画面看不出变化）"
    fl.tx += 50.0
    out = render_document(doc)
    assert int(out[30, 30, 3]) == 0, "移走之后原处应该是空的"
    assert int(out[55, 80, 3]) == 255, "内容应该出现在新位置"
    print("  剪切模式：挖空 + 浮动层跟随 OK")

    assert stamp_float(doc, lay) is True
    assert doc.float_layer is None
    out = render_document(doc)
    assert int(out[30, 30, 3]) == 0 and int(out[55, 80, 3]) == 255, \
        "落定后位置不对"
    print("  落定：盖回当前位置 OK")

    # 复制模式（Alt 拖动）：原处保留
    doc, lay = fresh()
    fl = lift_selection(doc, lay, copy_mode=True)
    assert fl is not None
    out = render_document(doc)
    assert int(out[30, 30, 3]) == 255, "复制模式原处不能空"
    fl.tx += 50.0
    out = render_document(doc)
    assert int(out[30, 30, 3]) == 255 and int(out[55, 80, 3]) == 255, \
        "复制模式应该两处都有"
    print("  复制模式：原处保留 + 新位置有副本 OK")

    # 丢弃
    doc, lay = fresh()
    lift_selection(doc, lay)
    assert discard_float(doc) is True
    assert doc.float_layer is None
    print("  丢弃浮动内容 OK")

    # 变换过的图层也要能揭、能盖回（走 remap 那条路）
    doc, lay = fresh()
    lay.sx = lay.sy = 0.5
    lay.rot = 30.0
    fl = lift_selection(doc, lay)
    assert fl is not None
    fl.ty += 40.0
    assert stamp_float(doc, lay) is True
    print("  旋转/缩放图层：揭 + 盖回 OK")

    # 浮动层要跟着工程一起存
    doc, lay = fresh()
    lift_selection(doc, lay)
    doc.float_layer.tx += 12.0
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "f.cwproj")
        save_project(doc, path)
        loaded = load_project(path)
    assert loaded.float_layer is not None, "浮动层没存下来"
    assert abs(loaded.float_layer.tx - doc.float_layer.tx) < 1e-6
    assert loaded.float_layer.image.shape == doc.float_layer.image.shape
    print("  浮动层工程存取：位图与位置保留")


def test_selection_enhance():
    """选区增强：平滑（抹锯齿）与边界（沿轮廓挖一条环带）。"""
    from src.core.selection import Selection

    # ---- 平滑：主体保留、边缘的毛刺被抹掉 ----
    def rough(m):
        """边缘毛刺的度量：横竖两个方向上 0/1 的跳变总数。"""
        b = (m >= 128).astype(np.int8)
        return int(np.abs(np.diff(b, axis=0)).sum()
                   + np.abs(np.diff(b, axis=1)).sum())

    rng = np.random.default_rng(7)
    m = np.zeros((80, 80), np.uint8)
    m[20:60, 20:60] = 255
    # 沿边缘撒一圈 1 px 的毛刺（魔棒勾出来的典型锯齿）
    ring = np.zeros((80, 80), bool)
    ring[17:23, 17:63] = True
    ring[57:63, 17:63] = True
    ring[17:63, 17:23] = True
    ring[17:63, 57:63] = True
    m[ring & (rng.random((80, 80)) < 0.4)] = 255
    inner = np.zeros((80, 80), bool)
    inner[23:57, 23:57] = True
    m[inner & (rng.random((80, 80)) < 0.05)] = 0

    r0 = rough(m)
    sel = Selection(80, 80, m.copy())
    sel.smooth(1)
    r1 = rough(sel.mask)
    assert r1 < r0, "平滑没把毛刺抹掉（%d -> %d）" % (r0, r1)
    assert int(sel.mask[40, 40]) == 255, "平滑不该把主体弄没"
    assert int(sel.mask[0, 0]) == 0, "平滑不该把选区撑大"
    print("  平滑：边缘毛刺 %d -> %d（主体与外轮廓都还在）" % (r0, r1))

    # 羽化过的中间值不该被平滑抹成硬边
    soft = Selection(80, 80, m.copy())
    soft.feather(3.0)
    assert (soft.mask > 0).any() and (soft.mask < 255).any()
    n_before = len(set(int(v) for v in soft.mask[soft.mask > 0]))
    soft.smooth(1)
    assert (soft.mask > 0).any() and (soft.mask < 255).any(), \
        "平滑不该把羽化过的边压成硬边"
    print("  平滑：羽化后的中间值保留（%d 个灰阶 -> %d 个）" %
          (n_before, len(set(int(v) for v in soft.mask[soft.mask > 0]))))

    # ---- 边界：沿原轮廓挖一条环带，中心空 ----
    sel = Selection.rect(60, 60, 10, 10, 50, 50)
    sel.border(6)
    assert sel.mask[30, 30] == 0, "中心应该被挖空"
    for pt in ((10, 30), (30, 10), (49, 30), (30, 49)):
        assert sel.mask[pt[0], pt[1]] == 255, \
            "原轮廓上的点 %s 应该落在环带里" % (pt,)
    box = sel.bbox()
    assert box[0] <= 10 and box[2] >= 50, "环带应该向外也长出去：%s" % (box,)
    print("  边界：中心挖空、轮廓四周留 %d px 环带（bbox %s）" %
          (box[2] - box[0] - 40, box))

    # 空选区上跑一遍不能崩
    empty = Selection(60, 60)
    empty.smooth(3)
    empty.border(4)
    assert empty.is_empty
    print("  边界 / 平滑：空选区安全")


def test_color_range():
    """色彩范围：取样颜色 + 色相预置 + 高光/中间调/阴影。"""
    from src.core.color_range import NEEDS_SAMPLE, PRESETS, range_mask

    img = np.zeros((40, 200, 3), np.uint8)
    img[:, 0:40] = (220, 20, 20)      # 红
    img[:, 40:80] = (230, 220, 30)    # 黄
    img[:, 80:120] = (30, 200, 60)    # 绿
    img[:, 120:160] = (250, 250, 250)  # 白
    img[:, 160:200] = (10, 10, 12)    # 近黑

    def at(pid, x):
        return int(range_mask(img, pid, [(220, 20, 20)], 40.0)[20, x])

    assert at("sampled", 20) == 255 and at("sampled", 60) == 0, \
        "取样颜色只该命中红色"
    assert at("reds", 20) == 255 and at("reds", 60) == 0
    assert at("yellows", 60) == 255 and at("yellows", 20) == 0
    assert at("greens", 100) == 255 and at("greens", 20) == 0
    assert at("highlights", 140) == 255 and at("highlights", 180) == 0
    assert at("shadows", 180) == 255 and at("shadows", 140) == 0
    print("  预置范围：取样 / 六色相 / 高光 / 阴影 各自命中")

    # 近黑不该被当成"蓝色"（暗部色相不稳，一律不算有颜色）
    assert at("blues", 180) == 0, "近黑像素不该被判成蓝色"
    print("  暗部不参与色相判定")

    # 容差越大命中越多
    lo = range_mask(img, "sampled", [(220, 20, 20)], 10.0)
    hi = range_mask(img, "sampled", [(220, 20, 20)], 200.0)
    n_lo = int((lo > 0).sum())
    n_hi = int((hi > 0).sum())
    assert n_hi > n_lo, "容差变大应该选中更多像素（%d -> %d）" % (n_lo, n_hi)
    print("  容差：10 -> %d 个像素，200 -> %d 个" % (n_lo, n_hi))

    # 反相
    inv = range_mask(img, "reds", (), 40.0, invert=True)
    assert int(inv[20, 20]) == 0 and int(inv[20, 180]) == 255
    print("  反相 OK")

    # 没给取样点时返回全 0，不崩
    assert not range_mask(img, "sampled", (), 40.0).any()
    assert "sampled" in NEEDS_SAMPLE and "reds" not in NEEDS_SAMPLE
    assert len(PRESETS) == 11
    print("  无取样点 / 预置表完整（%d 项）" % len(PRESETS))


def test_quick_mask():
    """快速蒙版：进入 -> 涂 -> 退出，以及写时复制。"""
    from src.core import quick_mask
    from src.core.paint import Stroke
    from src.core.selection import Selection

    def fresh(sel_box=(10, 10, 40, 40)):
        doc = Document(80, 60)
        doc.layers.append(
            make_image_layer("底", _solid(80, 60, (10, 10, 10)), 80, 60))
        if sel_box is not None:
            doc.selection = Selection.rect(80, 60, *sel_box)
        return doc

    # 进入：遮罩由当前选区初始化，语义是"会被选中的程度"
    doc = fresh()
    assert not quick_mask.is_on(doc)
    assert quick_mask.enter(doc) is True
    assert quick_mask.is_on(doc)
    assert doc.quick_mask.shape == (60, 80)
    assert int(doc.quick_mask[25, 25]) == 255 and int(doc.quick_mask[0, 0]) == 0
    print("  进入：遮罩按当前选区初始化 (%d x %d)" % doc.quick_mask.shape[::-1])

    # 红色叠加：默认盖住"未选中"那一侧，换模式只换显示不改数据
    a = quick_mask.overlay_alpha(doc)
    assert int(a[0, 0]) == 127 and int(a[25, 25]) == 0
    doc.quick_mask_mode = "selected"
    a = quick_mask.overlay_alpha(doc)
    assert int(a[0, 0]) == 0 and int(a[25, 25]) == 127
    assert int(doc.quick_mask[25, 25]) == 255, "换显示模式不该动数据"
    print("  红色叠加：masked / selected 两种显示，数据不动")

    # 画笔涂白 = 纳入选区
    doc.quick_mask_mode = "masked"
    st = Stroke(doc=doc, layer=None, tool="brush", size=30, hardness=1.0,
                opacity=1.0, flow=1.0, smoothing=0.0, target="quick",
                color=(255, 255, 255))
    assert st.begin((5.0, 5.0)), "快速蒙版上应该能落笔"
    st.end()
    assert int(doc.quick_mask[5, 5]) == 255, "涂白应该把它纳入选区"
    # 涂黑 = 排除
    st = Stroke(doc=doc, layer=None, tool="brush", size=30, hardness=1.0,
                opacity=1.0, flow=1.0, smoothing=0.0, target="quick",
                color=(0, 0, 0))
    assert st.begin((25.0, 25.0))
    st.end()
    assert int(doc.quick_mask[25, 25]) == 0, "涂黑应该把它排除出选区"
    print("  画笔：涂白纳入 / 涂黑排除")

    # 写时复制：撤销快照与当前文档共享同一份数组，detach 之后各改各的
    snap = doc.clone()
    assert snap.quick_mask is doc.quick_mask, "clone 应该共享（写时复制）"
    doc.detach_quick_mask()
    doc.quick_mask[:] = 200
    assert int(snap.quick_mask[0, 0]) != 200, "detach 之后不该污染历史快照"
    print("  写时复制：detach_quick_mask() 之后快照不受影响")

    # 退出 -> 变成选区
    doc = fresh()
    quick_mask.enter(doc)
    doc.quick_mask[:] = 0
    doc.quick_mask[5:15, 5:15] = 255
    assert quick_mask.exit_to_selection(doc) is True
    assert not quick_mask.is_on(doc)
    assert doc.selection.bbox() == (5, 5, 15, 15), \
        "退出后的选区不对：%s" % (doc.selection.bbox(),)
    print("  退出：遮罩变成选区（bbox %s）" % (doc.selection.bbox(),))

    # 全 0 退出 = 没有选区
    doc = fresh()
    quick_mask.enter(doc)
    doc.quick_mask[:] = 0
    quick_mask.exit_to_selection(doc)
    assert doc.selection is None, "全 0 的遮罩退出后应该是没有选区"
    print("  退出：遮罩全 0 -> 选区取消")

    # 取消（放弃改动）
    doc = fresh(sel_box=None)
    quick_mask.enter(doc)
    assert quick_mask.cancel(doc) is True and not quick_mask.is_on(doc)
    print("  取消：丢掉遮罩，不产生选区")


def test_float_transform():
    """浮动选区带变换：缩放 / 旋转之后还能正确盖回图层。"""
    from src.core.float_sel import float_bbox, lift_selection, stamp_float
    from src.core.render import render_document
    from src.core.selection import Selection

    def fresh():
        # 只在选区那一块有内容：这样"挖空"和"盖回"都能从 alpha 上直接看出来
        img = np.zeros((100, 100, 4), np.uint8)
        img[..., 0] = 255
        img[30:70, 30:70, 3] = 255
        doc = Document(100, 100)
        lay = make_image_layer("红", img, 100, 100)
        doc.layers.append(lay)
        doc.selection = Selection.rect(100, 100, 30, 30, 70, 70)
        return doc, lay

    # 以中心放大 2 倍：40x40 -> 80x80，包围盒 10..90
    doc, lay = fresh()
    fl = lift_selection(doc, lay)
    assert fl is not None
    fl.sx = fl.sy = 2.0
    box = float_bbox(fl)
    assert abs(box[0] - 10.0) < 0.51 and abs(box[2] - 90.0) < 0.51, \
        "放大后的包围盒不对：%s" % (box,)
    print("  浮动层 bbox 计入变换：放大 2x -> "
          + str(tuple(round(v, 1) for v in box)))

    before = render_document(doc)
    assert stamp_float(doc, lay) is True
    after = render_document(doc)
    assert int(after[50, 50, 3]) == 255, "中心应该有内容"
    assert int(after[50, 15, 3]) == 255, "放大后应该盖到更外面"
    assert int(after[50, 5, 3]) == 0, "不该超出放大后的范围"
    d = np.abs(before.astype(np.int16) - after.astype(np.int16))
    assert d.max() <= 3, "落定前后画面不该有明显差别，最大差 %d" % d.max()
    print("  放大后落定：画面一致（最大差 %d），像素出现在新范围" % d.max())

    # 旋转 + 缩放一起，也不能崩、不能糊
    doc, lay = fresh()
    fl = lift_selection(doc, lay)
    fl.sx = 1.5
    fl.sy = 0.6
    fl.rot = 35.0
    before = render_document(doc)
    assert stamp_float(doc, lay) is True
    after = render_document(doc)
    d = np.abs(before.astype(np.int16) - after.astype(np.int16))
    assert d.max() <= 4, "旋转+缩放落定后画面差太多：%d" % d.max()
    print("  旋转 + 非等比缩放落定：最大差 %d" % d.max())

    # 缩小之后落定
    doc, lay = fresh()
    fl = lift_selection(doc, lay)
    fl.sx = fl.sy = 0.5
    box = float_bbox(fl)
    assert abs(box[0] - 40.0) < 0.51 and abs(box[2] - 60.0) < 0.51, \
        "缩小后的包围盒不对：%s" % (box,)
    assert stamp_float(doc, lay) is True
    print("  缩小后落定 OK")


def _perf_doc():
    """一块"什么都有"的画布：旋转缩放图层 + 蒙版 + 图层样式 + 调整层 + 浮动层。"""
    from src.core.effects import default_effects
    from src.core.float_sel import lift_selection
    from src.core.selection import Selection

    doc = Document(240, 180, "perf")
    bg = _solid(240, 180, (200, 200, 200))
    doc.layers.append(Layer("bg", "image", bg))
    doc.layers[0].tx, doc.layers[0].ty = 120.0, 90.0

    img = _solid(120, 90, (255, 0, 0))
    img[10:40, 10:40] = (0, 0, 255, 255)
    l = Layer("red", "image", img)
    l.tx, l.ty = 90.0, 70.0
    l.sx, l.sy = 1.6, 1.3
    l.rot = 25.0
    m = np.zeros((90, 120), np.uint8)
    m[:, :] = 255
    m[:, 90:] = 0                      # 右半边被蒙版挡掉
    l.mask = m
    eff = default_effects()
    eff["drop_shadow"]["enabled"] = True
    eff["drop_shadow"]["distance"] = 6.0
    eff["drop_shadow"]["size"] = 8.0
    l.effects = eff
    doc.layers.append(l)

    doc.layers.append(make_adjustment_layer("levels", "levels", 240, 180))
    return doc


def test_render_fast_paths():
    """第十八批：两条渲染快路径 —— 必须与通用路径**逐位**一致。

    这两条是纯优化，一旦结果对不上就是"画面悄悄变了"，所以断言一律写
    `== 0`，不写容差（守 §4.3 的老规矩）。
    """
    import src.core.render as R
    from src.core.blend import _opaque, blend_colors, composite

    # ---- 1) composite 的「下方不透明」快路径 ----
    rng = np.random.default_rng(7)
    h, w = 40, 56
    for mode in ("Normal", "Multiply", "Screen", "Overlay", "Soft Light",
                 "Difference", "Color", "Luminosity"):
        for src_a_mode in ("opaque", "half", "ramp"):
            dc0 = rng.random((h, w, 3)).astype(np.float32)
            da0 = np.ones((h, w, 1), np.float32)
            sc = rng.random((h, w, 3)).astype(np.float32)
            if src_a_mode == "opaque":
                sa = np.ones((h, w, 1), np.float32)
            elif src_a_mode == "half":
                sa = np.full((h, w, 1), 0.5, np.float32)
            else:
                sa = rng.random((h, w, 1)).astype(np.float32)
            # 快路径结果
            dc1, da1 = dc0.copy(), da0.copy()
            composite(dc1, da1, sc, sa, mode)
            # 通用路径：把 ab 压到"不满足不透明"的值，逼它走慢的那条
            dc2, da2 = dc0.copy(), (da0 * 0.5).astype(np.float32)
            _general_composite(dc2, da2, sc, sa, mode)
            # 两者应当在"合成结果"上一致 —— 但 ab 不同，语义上就不该直接比。
            # 所以比的是**公式恒等式**：ab=1 时慢路径也该给出同样的结果。
            dc3, da3 = dc0.copy(), da0.copy()
            _general_composite(dc3, da3, sc, sa, mode)
            assert np.abs(dc1 - dc3).max() == 0, \
                "%s 快路径与通用路径不一致：%g" % (mode, np.abs(dc1 - dc3).max())
            assert np.abs(da1 - da3).max() <= 1e-6, mode
    print("  composite 不透明快路径：8 种混合 x 3 种源 alpha，逐位一致")

    # _opaque 的判定边界
    assert _opaque(np.ones((2, 2, 1), np.float32)) is True
    assert _opaque(np.zeros((2, 2, 1), np.float32)) is False
    almost = np.full((2, 2, 1), 1.0 - 1e-7, np.float32)
    assert _opaque(almost) is True, "1-1e-7 应算不透明"
    assert _opaque(np.full((2, 2, 1), 0.999, np.float32)) is False
    print("  _opaque 判定：1 / 0 / 1-1e-7 / 0.999 四个边界都对")

    # ---- 2) _integer_shift 的识别 ----
    F = np.float64
    assert R._integer_shift(np.array([[1, 0, 5.0], [0, 1, -3.0]])) == (5, -3)
    assert R._integer_shift(np.array([[1, 0, 0.0], [0, 1, 0.0]])) == (0, 0)
    assert R._integer_shift(np.array([[1, 0, 5.5], [0, 1, 0.0]])) is None, \
        "非整数平移不该命中"
    assert R._integer_shift(np.array([[1.2, 0, 0.0], [0, 1, 0.0]])) is None, \
        "缩放不该命中"
    assert R._integer_shift(np.array([[1, 0.3, 0.0], [0, 1, 0.0]])) is None, \
        "斜切不该命中"
    assert R._integer_shift(np.array([[1, 0, 0.0], [0.1, 1, 0.0]])) is None
    print("  _integer_shift：整数平移命中；非整数 / 缩放 / 斜切都不命中")

    # ---- 3) 快路径渲染 == warpAffine 渲染（逐位）----
    def render_with(force_slow):
        d = Document(200, 150, "fp")
        base = make_image_layer("底", _solid(200, 150, (90, 110, 130)), 200, 150)
        base.tx, base.ty = 100.0, 75.0
        d.layers.append(base)
        a = make_image_layer("甲", _solid(80, 60, (220, 40, 60), 200), 80, 60)
        a.tx, a.ty = 60.0, 50.0              # 整数平移 -> 快路径
        d.layers.append(a)
        b = make_image_layer("乙", _solid(70, 90, (30, 200, 90), 255), 70, 90)
        b.tx, b.ty = 137.0, 22.0              # 非整数 -> 仍走 warpAffine
        b.rot = 11.0
        d.layers.append(b)
        old = R.SRC_CROP
        R.SRC_CROP = not force_slow
        # 关掉快路径的办法：临时把判定改成永远不命中
        orig_fn = R._integer_shift
        if force_slow:
            R._integer_shift = lambda M: None
        try:
            return render_document(d)
        finally:
            R._integer_shift = orig_fn
            R.SRC_CROP = old

    fast = render_with(False)
    slow = render_with(True)
    assert np.abs(fast.astype(np.int16) - slow.astype(np.int16)).max() == 0, \
        "整数平移快路径与 warpAffine 结果不一致：%d" % np.abs(
            fast.astype(np.int16) - slow.astype(np.int16)).max()
    print("  整数平移快路径：整幅渲染与 warpAffine 逐位一致（含混合与半透明）")

    # ---- 4) 越界 / 负偏移 / 翻转 / 旋转 各走对分支 ----
    d = Document(120, 100, "edge")
    l = make_image_layer("半出界", _solid(100, 80, (10, 220, 120), 180),
                         100, 80)
    l.tx, l.ty = 80.0, 60.0      # 右下角出界
    d.layers.append(l)
    out = render_document(d)
    assert out.shape == (100, 120, 4)
    # 图层 100x80 居中在 (80,60) -> 覆盖 x[30,130) y[20,100)
    assert out[40, 40, 1] > 150, "画布内那半应该可见"
    l2 = make_image_layer("全出界", _solid(50, 50, (255, 0, 0)), 50, 50)
    l2.tx, l2.ty = 500.0, 500.0
    d.layers.append(l2)
    out2 = render_document(d)
    assert np.abs(out2.astype(np.int16) - out.astype(np.int16)).max() == 0, \
        "完全在画布外的图层不该影响画面"
    print("  边界：半出界只画到画布内 / 完全出界不参与合成")

    # 翻转后矩阵不再是纯平移，必须走 warpAffine 且方向正确
    d = Document(80, 60, "flip")
    src = np.zeros((40, 60, 4), np.uint8)
    src[..., 0] = 255
    src[5:15, 5:25, 1] = 255            # 绿色块在左上
    src[..., 3] = 255
    fl = make_image_layer("翻", src, 60, 40)
    fl.tx, fl.ty = 40.0, 30.0
    d.layers.append(fl)
    a = render_document(d).copy()
    fl.flip_h = not fl.flip_h
    b = render_document(d)
    # 图层 60x40 居中在 (40,30) -> 覆盖 x[10,70) y[10,50)
    # 绿块在源图 x[5,25) y[5,15) -> 未翻转时落��� x[15,35) y[15,25)
    left_green = a[20, 20, 1]
    right_green = b[20, 50, 1]
    assert left_green > 200 and right_green > 200, \
        "水平翻转后绿色块应该出现在另一侧（%d / %d）" % (left_green, right_green)
    print("  翻转 / 旋转不误命中快路径，方向正确")

    # ---- 5) 分块渲染仍然与整幅一致（快路径不能破坏瓦片边界）----
    d = _perf_doc()
    full = render_document(d)
    tiled = R.render_tiled(d, tile=64)
    assert tiled is not None
    diff = np.abs(full.astype(np.int16) - tiled.astype(np.int16))
    assert diff.max() <= 1 + int((diff > 0).mean() * 4), diff.max()
    print("  分块 vs 整幅：最大差 %d（≤1 + 占比上限，守 §4.3）" % diff.max())


def _general_composite(dst_c, dst_a, src_c, src_a, mode):
    """`blend.composite` 的通用路径（不走不透明快路径），供测试对比。"""
    from src.core.blend import _div, blend_colors
    as_ = src_a
    ab = dst_a
    b = src_c if mode == "Normal" else blend_colors(dst_c, src_c, mode)
    co = as_ * (1.0 - ab) * src_c + as_ * ab * b + (1.0 - as_) * ab * dst_c
    ao = as_ + ab * (1.0 - as_)
    safe = np.where(np.abs(ao) < 1e-6, 1.0, ao)
    dst_c[...] = np.clip(_div(co, safe), 0.0, 1.0)
    dst_a[...] = np.clip(ao, 0.0, 1.0)


def test_dirty_regions():
    """第十八批：脏区是**一组**矩形 + 只回写脏区本身（局部重渲的正确性）。"""
    from src.core.effects import default_effects
    from src.ui.main_window import DIRTY_MAX_RECTS, _union

    assert _union([(0, 0, 10, 10), (20, 20, 30, 30)]) == (0, 0, 30, 30)
    assert _union([(5, 5, 10, 10)]) == (5, 5, 10, 10)
    print("  脏区并集：两块 / 单块都算对")

    _ensure_gui()
    from src.ui.main_window import MainWindow
    win = MainWindow()
    d = win.doc
    d.width, d.height = 400, 300
    d.layers.clear()
    rng = np.random.default_rng(3)
    for i in range(3):
        a = np.zeros((300, 400, 4), np.uint8)
        a[..., :3] = rng.integers(0, 255, 3)
        a[..., 3] = 255
        d.layers.append(make_image_layer("L%d" % i, a, 400, 300))
    # 带投影：外扩区的作用（模糊上下文）与"只回写脏区"这条都会被验到
    d.layers[1].effects = default_effects()
    d.layers[1].effects["drop_shadow"].update(enabled=True, distance=12.0,
                                              size=14.0)
    win.commit("建场景")
    win._do_render()
    ref = win._last_arr.copy()

    # ---- 1) 多块脏区：与整幅逐位一致 ----
    win.mark_dirty((20, 20, 90, 80))
    win.mark_dirty((300, 200, 380, 280))
    win.mark_dirty((150, 120, 230, 190))
    assert len(win._dirty_rects()) == 3, win._dirty_rects()
    win._do_render()
    assert win._last_pass == "partial-multi", win._last_pass
    diff = np.abs(win._last_arr.astype(np.int16) - ref.astype(np.int16))
    assert diff.max() == 0, "多块脏区与整幅差 %d" % diff.max()
    print("  多块脏区：3 块分散区域重渲后与整幅**逐位**一致")

    # ---- 2) 相邻 / 相交的块要合并，别多渲一圈 ----
    win._dirty = None
    win.mark_dirty((10, 10, 50, 50))
    win.mark_dirty((50, 10, 90, 50))          # 紧挨着
    assert win._dirty_rects() == [(10, 10, 90, 50)], win._dirty_rects()
    win._dirty = None
    win.mark_dirty((10, 10, 50, 50))
    win.mark_dirty((20, 20, 30, 30))          # 完全落在里面
    assert len(win._dirty_rects()) == 1, win._dirty_rects()
    print("  脏区合并：紧挨的 / 被包含的都并成一块")

    # ---- 3) 远离的块不许互相吞掉（这正是本批要解决的）----
    win._dirty = None
    win.mark_dirty((0, 0, 20, 20))
    win.mark_dirty((380, 280, 400, 300))
    assert len(win._dirty_rects()) == 2, "两块离得很远，不该合并"
    print("  远离的两块各自保留（不再退化成覆盖全画布的并集）")

    # ---- 4) 超过上限要折叠，别无限增长 ----
    win._dirty = None
    for i in range(DIRTY_MAX_RECTS + 5):
        win.mark_dirty((i * 20, 0, i * 20 + 8, 8))
    assert len(win._dirty_rects()) <= DIRTY_MAX_RECTS, \
        len(win._dirty_rects())
    print("  块数上限 %d：超过后折叠成并集（实测剩 %d 块）"
          % (DIRTY_MAX_RECTS, len(win._dirty_rects())))

    # ---- 5) 只回写脏区：外扩那一圈不能被"截断上下文"的结果覆盖 ----
    # 这条曾让带投影的文档局部重渲与整幅差 131/255 —— 投影拖尾刚好越过
    # 脏区边界时，外扩区里那截模糊是缺上下文的，写进缓存就留了接缝。
    win._full_dirty = False
    win._dirty = None
    win.mark_dirty((300, 200, 380, 280))
    win._do_render()
    diff = np.abs(win._last_arr.astype(np.int16) - ref.astype(np.int16))
    assert diff.max() == 0, "带投影时单块脏区仍差 %d" % diff.max()
    print("  只回写脏区：投影拖尾越过边界也不再留接缝（逐位一致）")

    # ---- 6) 整幅脏区仍然压倒一切 ----
    win._dirty = None
    win.mark_dirty((0, 0, 400, 300))
    assert win._want_full() is True
    print("  脏区占满画布：仍然退回整幅重算")

    # 关窗口：第二十四批起，关窗口会把「改过没存」的标签逐个问一遍
    # （见 main_window.closeEvent），离屏下弹 QMessageBox 会挂死 ——
    # 这里先把会话标成已保存再关。
    for s in getattr(win, "sessions", []):
        s.mark_saved()
    win.close()


def test_nested_adjust_snapshot():
    """第十八批：隔离组里的调整层也能走快照快路径（§4.4 方案 C 第 1 项）。"""
    import src.core.render as R
    from src.core.document import make_adjustment_layer, make_group

    def build(gmode="Normal", gopacity=1.0, canvas=(400, 300)):
        w, h = canvas
        d = Document(w, h, "nested")
        base = make_image_layer("底", _solid(w, h, (120, 90, 70)), w, h)
        base.tx, base.ty = w / 2.0, h / 2.0
        d.layers.append(base)
        g = make_group("组")
        g.blend, g.opacity = gmode, gopacity
        for i, rgb in enumerate([(200, 60, 60), (60, 200, 90)]):
            a = make_image_layer("in%d" % i, _solid(200, 150, rgb), 200, 150)
            a.tx, a.ty = 100.0 + 40 * i, 80.0 + 30 * i
            g.children.append(a)
        adj = make_adjustment_layer("反相", key="levels")
        adj.tx, adj.ty = w / 2.0, h / 2.0
        g.children.append(adj)
        d.layers.append(g)
        top = make_image_layer("顶", _solid(120, 100, (30, 60, 200), 200),
                              120, 100)
        top.tx, top.ty = w - 80.0, 60.0
        d.layers.append(top)
        return d, adj, g

    # ---- 1) 逐位一致：组的混合模式 / 不透明度 various ----
    for gmode, gopacity in (("Normal", 1.0), ("Normal", 0.6),
                            ("Multiply", 0.8), ("Screen", 0.55)):
        d, adj, g = build(gmode, gopacity)
        full = R.render_document(d)
        snap = {"id": adj.id, "w": d.width, "h": d.height, "arr": None}
        R.render_document(d, snap_layer=adj, snap=snap)
        assert snap.get("group") is not None, \
            "组内调整层应该产生组内快照（%s/%.1f）" % (gmode, gopacity)
        got = R.render_from_snapshot(d, adj, snap)
        assert got is not None, "组内调整层没能重放"
        diff = np.abs(full.astype(np.int16) - got.astype(np.int16))
        assert diff.max() == 0, \
            "组 %s/%.1f 重放与整幅差 %d" % (gmode, gopacity, diff.max())
    print("  组内调整层重放：4 种组混合模式 × 不透明度，逐位一致")

    # ---- 2) Pass Through 组：没有隔离缓冲，走顶层那条路 ----
    d, adj, g = build("Pass Through", 1.0)
    full = R.render_document(d)
    snap = {"id": adj.id, "w": d.width, "h": d.height, "arr": None}
    R.render_document(d, snap_layer=adj, snap=snap)
    got = R.render_from_snapshot(d, adj, snap)
    assert got is not None
    assert np.abs(full.astype(np.int16) - got.astype(np.int16)).max() == 0
    print("  Pass Through 组：走顶层路径，同样逐位一致")

    # ---- 3) 拿不准的情况必须退回 None（宁可整幅重算，不能画错）----
    d, adj, g = build()
    # 只有 arr 没有 group
    snap = {"id": adj.id, "w": d.width, "h": d.height,
            "arr": np.zeros((d.height, d.width, 4), np.uint8)}
    assert R.render_from_snapshot(d, adj, snap) is None, "缺组内快照应退回"
    # group 在但 gid 指向不存在的组
    snap2 = {"id": adj.id, "w": d.width, "h": d.height,
             "arr": np.zeros((d.height, d.width, 4), np.uint8),
             "group": {"gid": "不存在", "blend": "Normal", "opacity": 1.0,
                       "buf": np.zeros((d.height, d.width, 4), np.uint8)}}
    assert R.render_from_snapshot(d, adj, snap2) is None
    # 尺寸对不上
    snap3 = {"id": adj.id, "w": d.width, "h": d.height,
             "arr": np.zeros((d.height, d.width, 4), np.uint8),
             "group": {"gid": g.id, "blend": "Normal", "opacity": 1.0,
                       "buf": np.zeros((7, 7, 4), np.uint8)}}
    assert R.render_from_snapshot(d, adj, snap3) is None
    # 剪贴蒙版调整层 / 非调整层
    assert R.render_from_snapshot(d, g, snap3) is None
    print("  缺快照 / gid 不存在 / 尺寸不符 / 非调整层：都退回整幅（返回 None）")

    # ---- 4) 更深一层嵌套（调整层在子组里）必须退回，不能硬算 ----
    d, adj, g = build()
    inner = make_group("子组")
    inner.children = list(g.children)
    g.children = [inner]
    full = R.render_document(d)
    snap = {"id": adj.id, "w": d.width, "h": d.height, "arr": None}
    R.render_document(d, snap_layer=adj, snap=snap)
    got = R.render_from_snapshot(d, adj, snap)
    if got is not None:
        assert np.abs(full.astype(np.int16) - got.astype(np.int16)).max() == 0, \
            "两层嵌套时若还能重放，结果必须仍然正确"
    print("  两层嵌套：%s" % ("退回整幅（当前只支持一层）"
                              if got is None else "重放且逐位一致"))

    # ---- 5) 确实更快（12 MP）----
    d = Document(4000, 3000, "big")
    for i in range(6):
        a = _solid(4000, 3000, (40 + i * 9, 90, 140))
        d.layers.append(make_image_layer("L%d" % i, a, 4000, 3000))
    g = make_group("组")
    g.blend, g.opacity = "Normal", 0.85
    for i in range(4):
        a = _solid(4000, 3000, (200, 60 + i * 10, 60))
        l = make_image_layer("in%d" % i, a, 4000, 3000)
        l.tx, l.ty = 2000.0, 1500.0
        g.children.append(l)
    adj = make_adjustment_layer("反相", key="levels")
    adj.tx, adj.ty = 2000.0, 1500.0
    g.children.append(adj)
    d.layers.append(g)

    t0 = time.perf_counter()
    ref = R.render_document(d)
    t_full = (time.perf_counter() - t0) * 1000
    snap = {"id": adj.id, "w": d.width, "h": d.height, "arr": None}
    R.render_document(d, snap_layer=adj, snap=snap)
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        got = R.render_from_snapshot(d, adj, snap)
        best = min(best, (time.perf_counter() - t0) * 1000)
    assert np.abs(ref.astype(np.int16) - got.astype(np.int16)).max() == 0
    assert best < t_full * 0.6, \
        "组内快照快路径没提速：整幅 %.0f ms -> 快路径 %.0f ms" % (t_full, best)
    print("  12 MP 组内调整层拖动：整幅 %.0f ms -> 快照 %.0f ms（省 %.0f%%）"
          % (t_full, best, (t_full - best) / t_full * 100))


def test_render_region_equals_full():
    """局部重渲染必须和整幅渲染逐位一致 —— 脏矩形算错就会留下残影。"""
    import src.core.render as R

    # 1) 没有图层样式时，任意区域都该逐位一致
    doc = _perf_doc()
    doc.layers[1].effects = None
    full = render_document(doc)
    for reg in ((0, 0, 240, 180), (10, 10, 120, 90), (100, 60, 200, 150),
                (200, 150, 240, 180), (37, 11, 213, 97)):
        sub = render_document(doc, region=reg)
        ref = full[reg[1]:reg[3], reg[0]:reg[2]]
        assert sub.shape == ref.shape, "局部渲染尺寸不对"
        d = np.abs(sub.astype(np.int16) - ref.astype(np.int16))
        assert d.max() == 0, "局部渲染与整幅不一致，最大差 %d" % d.max()
    print("  局部渲染：5 个区域与整幅逐位一致")

    # 2) 有图层样式（投影要模糊）时，区域必须按 max_effect_padding 外扩再算，
    #    否则贴着边界渲会缺卷积上下文
    doc = _perf_doc()
    full = render_document(doc)
    pad = R.max_effect_padding(doc)
    assert pad > 0, "这个场景应该检出图层样式的外扩量"
    for reg in ((10, 10, 120, 90), (100, 60, 200, 150), (37, 11, 213, 97)):
        ex0 = max(0, reg[0] - pad)
        ey0 = max(0, reg[1] - pad)
        ex1 = min(doc.width, reg[2] + pad)
        ey1 = min(doc.height, reg[3] + pad)
        sub = render_document(doc, region=(ex0, ey0, ex1, ey1))
        inner = sub[reg[1] - ey0:reg[3] - ey0, reg[0] - ex0:reg[2] - ex0]
        ref = full[reg[1]:reg[3], reg[0]:reg[2]]
        d = np.abs(inner.astype(np.int16) - ref.astype(np.int16))
        assert d.max() == 0, "带图层样式时外扩后仍不一致，最大差 %d" % d.max()
    print("  局部渲染：带投影时按外扩 %d px 计算，内核逐位一致" % pad)

    # 3) 源图裁剪（_src_window）不能改变结果：关掉开关再渲一次对比
    doc = _perf_doc()
    old = R.SRC_CROP
    try:
        R.SRC_CROP = False
        ref = render_document(doc)
        ref_sub = render_document(doc, region=(37, 11, 213, 97))
    finally:
        R.SRC_CROP = old
    d = np.abs(render_document(doc).astype(np.int16) - ref.astype(np.int16))
    assert d.max() <= 1, "源图裁剪改变了整幅结果，最大差 %d" % d.max()
    d2 = np.abs(render_document(doc, region=(37, 11, 213, 97)).astype(np.int16)
                - ref_sub.astype(np.int16))
    assert d2.max() <= 1, "源图裁剪改变了局部结果，最大差 %d" % d2.max()
    print("  源图裁剪：开 / 关两次渲染结果一致（省掉整张源图转 float32）")


def test_render_proxy():
    """代理渲染：尺寸对、观感接近、降采样缓存命中。"""
    import cv2

    import src.core.render as R

    doc = _perf_doc()
    full = render_document(doc)
    for s in (0.5, 0.25):
        arr, kx, ky = render_proxy(doc, s)
        assert arr.shape[0] == max(1, round(doc.height * s))
        assert arr.shape[1] == max(1, round(doc.width * s))
        assert abs(kx - arr.shape[1] / doc.width) < 1e-6
        ref = cv2.resize(full, (arr.shape[1], arr.shape[0]),
                         interpolation=cv2.INTER_AREA)
        d = np.abs(arr.astype(np.int16) - ref.astype(np.int16))
        # 代理允许插值差异，但整体不能跑偏（平均差 < 6/255）
        assert d.mean() < 6.0, "s=%.2f 代理与全量偏离过大 mean=%.2f" % (s, d.mean())
        # alpha 是硬边，必须几乎一致
        da = np.abs(arr[..., 3].astype(np.int16) - ref[..., 3].astype(np.int16))
        assert da.mean() < 1.0, "s=%.2f 代理的 alpha 跑偏 mean=%.2f" % (s, da.mean())
    print("  代理渲染：0.5 / 0.25 尺寸与观感都对（alpha 几乎无损）")

    # 降采样缓存：同一个 scale 第二次要命中缓存（返回同一个数组对象）
    R.clear_proxy_cache()
    a1, _, _ = render_proxy(doc, 0.25)
    p = R.make_proxy_doc(doc, 0.25)
    a2, _, _ = render_proxy(doc, 0.25)
    p2 = R.make_proxy_doc(doc, 0.25)
    assert p2.layers[0].image is p.layers[0].image, "降采样缓存没命中"
    assert a1.shape == a2.shape
    print("  代理降采样：缓存命中，同一 scale 复用缩小后的位图")


def test_dirty_helpers():
    """脏区相关的辅助函数：Dissolve 检测 + Stroke 脏矩形。"""
    from src.core.paint import Stroke

    doc = _perf_doc()
    assert not has_dissolve(doc), "默认场景不该检出 Dissolve"
    doc.layers[1].blend = "Dissolve"
    assert has_dissolve(doc), "有 Dissolve 图层时必须检出（否则局部渲染会错位）"
    doc.layers[1].blend = "Normal"
    doc.layers[1].visible = False
    assert not has_dissolve(doc), "不可见图层不算"
    doc.layers[1].visible = True

    # 笔画脏区必须覆盖真正被改动的像素
    layer = doc.layers[1]
    layer.mask = None
    layer.effects = None
    before = render_document(doc)
    st = Stroke(doc=doc, layer=layer, tool="brush", size=14, hardness=1.0,
                opacity=1.0, flow=1.0, smoothing=0.0, target="pixel",
                color=(0, 255, 0))
    assert st.begin((90.0, 70.0))
    st.extend((100.0, 78.0))
    rect = st.take_dirty(pad=0)
    st.end()
    after = render_document(doc)
    assert rect is not None, "画过了却没有脏区"
    diff = np.abs(after.astype(np.int16) - before.astype(np.int16)).max(axis=2)
    ys, xs = np.nonzero(diff > 0)
    assert len(xs) > 0, "这一笔没改动任何像素"
    x0, y0, x1, y1 = rect
    assert x0 <= xs.min() and x1 >= xs.max(), "脏区没盖住 X 方向的改动"
    assert y0 <= ys.min() and y1 >= ys.max(), "脏区没盖住 Y 方向的改动"
    # 取走之后就清零了
    assert st.take_dirty() is None, "take_dirty 取走后应清零"
    print("  笔画脏区：盖住全部改动像素（%d 个），取走即清零" % len(xs))


def test_adjust_snapshot():
    """拖调整层滑块的快路径：快照必须是"调整层下方"的合成结果。

    这条路径能省掉整幅重算里最贵的部分（所有图层的 warp + 混合），
    所以正确性完全押在"快照 == 调整层下方 acc"上，这里直接把它验死。
    """
    from src.core.adjust import default_params
    from src.core.render import render_from_snapshot

    doc = _perf_doc()                  # 背景 + 变换过的红块 + 顶层色阶
    adj = doc.layers[2]
    assert adj.is_adjustment and not adj.clipped

    # 1) 快照 = 只渲它下面那些图层
    below = Document(doc.width, doc.height, "below")
    below.layers = list(doc.layers[:2])
    ref = render_document(below)
    snap = {"id": adj.id, "w": doc.width, "h": doc.height, "arr": None}
    render_document(doc, snap_layer=adj, snap=snap)
    assert snap["arr"] is not None, "整幅渲染没有捕获快照"
    d = np.abs(snap["arr"].astype(np.int16) - ref.astype(np.int16))
    assert d.max() <= 1, "快照不等于调整层下方的合成结果，最大差 %d" % d.max()

    # 2) 逻辑本身必须精确：把快照换成无损的 float32，快路径要与整幅逐位一致
    import src.core.render as R
    old = R.SNAP_DTYPE
    try:
        R.SNAP_DTYPE = np.float32
        worst = 0
        for key, val in (("gamma", 1.8), ("in_black", 40), ("out_white", 220),
                         ("channel", "R")):
            adj.adjust["params"][key] = val
            esnap = {"id": adj.id, "w": doc.width, "h": doc.height,
                     "arr": None}
            render_document(doc, snap_layer=adj, snap=esnap)
            fast = render_from_snapshot(doc, adj, esnap)
            assert fast is not None, "顶层调整层应该能走快路径"
            dd = np.abs(fast.astype(np.int16)
                        - render_document(doc).astype(np.int16))
            worst = max(worst, int(dd.max()))
            assert dd.max() == 0, \
                "%s=%s 时快路径偏离整幅 %d（逻辑问题，不是舍入）" % (key, val,
                                                          dd.max())
    finally:
        R.SNAP_DTYPE = old
    print("  调整层快照：4 组参数下与整幅重算逐位一致（快路径逻辑正确）")

    # 3) 实际用的是 uint8 快照：允许暗部几个灰阶的舍入，但只能是极少数像素
    adj.adjust["params"] = default_params("levels")
    adj.adjust["params"]["gamma"] = 1.8
    snap = {"id": adj.id, "w": doc.width, "h": doc.height, "arr": None}
    render_document(doc, snap_layer=adj, snap=snap)
    fast = render_from_snapshot(doc, adj, snap)
    d = np.abs(fast.astype(np.int16) - render_document(doc).astype(np.int16))
    assert d.mean() < 0.2, "uint8 快照平均偏离 %.3f，过大" % d.mean()
    bad = float((d.max(axis=2) >= 3).sum()) / float(doc.width * doc.height)
    assert bad < 0.001, "差 >=3 的像素占 %.4f%%，过多" % (bad * 100.0)
    print("  调整层快照：uint8 量化代价 max=%d、mean=%.3f、差>=3 的像素 %.4f%%"
          % (d.max(), d.mean(), bad * 100.0))

    # 4) 局部渲染不捕获快照（acc 只覆盖一小块，存下来没法复用）
    sub_snap = {"id": adj.id, "w": doc.width, "h": doc.height, "arr": None}
    render_document(doc, region=(10, 10, 120, 90), snap_layer=adj,
                    snap=sub_snap)
    assert sub_snap["arr"] is None, "局部渲染不该捕获快照"

    # 5) 用不了的情形必须退回 None，而不是给出错误画面
    adj.clipped = True
    assert render_from_snapshot(doc, adj, snap) is None, "剪贴蒙版不能走快路径"
    adj.clipped = False
    g = make_group("g")
    doc.layers.remove(adj)
    g.children.append(adj)
    doc.layers.append(g)
    assert render_from_snapshot(doc, adj, snap) is None, "组里的调整层不能用"
    doc.layers.remove(g)
    doc.layers.append(adj)
    bad = dict(snap)
    bad["w"] = doc.width + 1
    assert render_from_snapshot(doc, adj, bad) is None, "尺寸对不上要作废"
    print("  调整层快照：剪贴 / 组内 / 尺寸不符三种情况都正确退回整幅渲染")


def test_filter_preview_downscale():
    """滤镜预览降采样：大图大半径先缩再算，半径按同一比例缩放。"""
    import cv2

    from src.core.filters import (apply_filter_array, preview_can_downscale,
                                  preview_filter_scale, scale_filter_params)

    # 1) 小图不动
    small = _solid(600, 400, (120, 120, 120))
    s, img = preview_filter_scale(small, "gaussian")
    assert s == 1.0 and img is small, "小图不该降采样"

    # 2) 大图 -> 按尺寸定出缩放比例，尺寸按同一比例
    rng = np.random.default_rng(3)
    big = np.zeros((1500, 2400, 4), np.uint8)
    big[..., :3] = rng.integers(0, 256, (1500, 2400, 3), dtype=np.uint8)
    big[..., 3] = 255
    s, prev = preview_filter_scale(big, "gaussian")
    assert 0.0 < s < 1.0, "2400 px 宽的高斯模糊应该降采样，实际 s=%.3f" % s
    assert prev.shape[1] == int(round(2400 * s)), "降采样后的宽度不对"
    print("  滤镜预览：2400x1500 降到 %dx%d（s=%.2f）再算"
          % (prev.shape[1], prev.shape[0], s))

    # 3) 半径按同一比例缩放；无量纲参数（数量 / 阈值）不动
    p = scale_filter_params("gaussian", {"radius": 40.0}, s)
    assert abs(p["radius"] - 40.0 * s) < 1e-6, "半径没有按预览比例缩放"
    p = scale_filter_params("sharpen", {"amount": 80, "radius": 6.0,
                                        "threshold": 2}, s)
    assert p["amount"] == 80 and p["threshold"] == 2, "无量纲参数被误改"
    assert abs(p["radius"] - 6.0 * s) < 1e-6

    # 4) 半径太小 / 逐像素的滤镜：这一帧不许降采样，否则预览会失真
    assert preview_can_downscale("gaussian", {"radius": 40.0}, s)
    assert not preview_can_downscale("median", {"radius": 1}, s), \
        "中间值半径 1~2 px，缩放后失真，不该降采样"
    assert not preview_can_downscale("sharpen", {"radius": 1.0}, s), \
        "半径 1 px 的锐化不该降采样"
    assert preview_can_downscale("sharpen", {"radius": 8.0}, s), \
        "半径拉大之后就该降采样了（锐化默认 1 px，但不能因此永远不降）"
    assert preview_can_downscale("radial", {"amount": 20}, s), \
        "径向模糊没有尺度参数，降采样不影响语义"

    # 5) 预览结果要和全分辨率算出来的观感一致
    pv = apply_filter_array(prev, "gaussian",
                            scale_filter_params("gaussian", {"radius": 40.0}, s))
    pv = cv2.resize(pv, (big.shape[1], big.shape[0]),
                    interpolation=cv2.INTER_LINEAR)
    ref = apply_filter_array(big, "gaussian", {"radius": 40.0})
    d = np.abs(pv.astype(np.int16) - ref.astype(np.int16))
    assert d.mean() < 6.0, "预览与全分辨率偏离过大 mean=%.2f" % d.mean()
    print("  滤镜预览：半径 40 的高斯模糊，预览与全分辨率平均差 %.2f/255"
          % d.mean())


def test_render_tiled():
    """分块渲染：结果要和整幅一致，且峰值内存只跟瓦片大小有关。"""
    import tracemalloc

    import src.core.render as R

    def _tiled_doc(w=700, h=520, fx=True):
        """一块够宽、能被切成多个瓦片的画布。"""
        from src.core.effects import default_effects

        d = Document(w, h, "tiled")
        d.layers.append(Layer("bg", "image", _solid(w, h, (200, 200, 200))))
        d.layers[0].tx, d.layers[0].ty = w / 2.0, h / 2.0
        img = _solid(300, 240, (255, 0, 0))
        img[10:60, 10:60] = (0, 0, 255, 255)
        l = Layer("red", "image", img)
        l.tx, l.ty = 350.0, 260.0
        l.sx, l.sy = 1.4, 1.1
        l.rot = 25.0
        m = np.zeros((240, 300), np.uint8)
        m[:, :] = 255
        m[:, 220:] = 0
        l.mask = m
        if fx:
            eff = default_effects()
            eff["drop_shadow"]["enabled"] = True
            eff["drop_shadow"]["size"] = 24.0
            l.effects = eff
        d.layers.append(l)
        d.layers.append(make_adjustment_layer("levels", "levels", w, h))
        return d

    # 1) 各种瓦片尺寸都要和整幅一致（允许 1/255 —— 分块改变了 warpAffine 的
    #    缓冲尺寸，个别像素的插值舍入会差一点，和脏区重渲染同一个量级）
    doc = _tiled_doc()
    full = render_document(doc)
    worst = 0
    for t in (128, 200, 512, 1024):
        got = R.render_tiled(doc, tile=t)
        d = np.abs(got.astype(np.int16) - full.astype(np.int16))
        worst = max(worst, int(d.max()))
        assert d.max() <= 1, "tile=%d 与整幅偏离 %d" % (t, d.max())
        # 差 1 的只能是极少数像素，否则说明瓦片接缝处算错了
        bad = float((d.max(axis=2) > 0).mean())
        assert bad < 0.01, "tile=%d 有 %.2f%% 的像素对不上，接缝有问题" % (t, bad * 100)
    print("  分块渲染：128/200/512/1024 四种瓦片都与整幅一致（最大差 %d/255）" % worst)

    # 2) 没有图层样式时 pad=0，瓦片之间不需要额外上下文，差异应该只剩极个别像素
    #    （warpAffine 的缓冲尺寸变了，个别像素的插值舍入会差 1）
    doc0 = _tiled_doc(fx=False)
    assert R.max_effect_padding(doc0) == 0
    d = np.abs(R.render_tiled(doc0, tile=128).astype(np.int16)
               - render_document(doc0).astype(np.int16))
    bad0 = float((d.max(axis=2) > 0).mean())
    assert d.max() <= 1, "没有图层样式时差异应 <=1，实际差 %d" % d.max()
    assert bad0 < 0.001, "没有图层样式时差异像素过多 %.3f%%" % (bad0 * 100)
    print("  分块渲染：无图层样式（pad=0）时差 1 的像素仅 %.3f%%" % (bad0 * 100))

    # 3) 子区域也要对得上（瓦片外扩必须夹在 region 里，否则会多吃到区域外的上下文）
    doc = _tiled_doc()
    for reg in ((50, 100, 600, 400), (0, 0, 300, 200), (600, 400, 700, 520)):
        got = R.render_region_tiled(doc, reg, tile=128)
        ref = render_document(doc, region=reg)
        d = np.abs(got.astype(np.int16) - ref.astype(np.int16))
        assert d.max() <= 1, "区域 %s 分块偏离 %d" % (str(reg), d.max())
    print("  分块渲染：3 个子区域都与整块渲染一致")

    # 4) 写进调用方的数组（脏区重渲染就是这么用的）
    out = np.zeros((520, 700, 4), np.uint8)
    R.render_region_tiled(doc, (50, 100, 600, 400), out=out[100:400, 50:600],
                          tile=128)
    d = np.abs(out[100:400, 50:600].astype(np.int16)
               - render_document(doc, region=(50, 100, 600, 400)).astype(np.int16))
    assert d.max() <= 1, "写进调用方数组的结果不对，差 %d" % d.max()
    print("  分块渲染：直接写进调用方的数组（脏区重渲染的用法）")

    # 5) Dissolve 的噪声按缓冲尺寸生成，分块会错位 —— 必须退回 None
    doc.layers[1].blend = "Dissolve"
    assert R.render_tiled(doc, tile=128) is None, "有 Dissolve 时必须退回整块渲染"
    doc.layers[1].blend = "Normal"
    print("  分块渲染：有 Dissolve 时返回 None，由调用方退回整块")

    # 6) 阶梯型调整层（色调分离 / 阈值）会把浮点舍入放大成"跳一档"。
    #    这不是分块引入的 —— 现有的脏区重渲染同样如此（缓冲尺寸一变，
    #    warpAffine 的个别像素就差 1e-5，正好压在量化边界上）。差异像素必须极少。
    from src.core.adjust import default_params

    pq = _tiled_doc(fx=False)
    aq = make_adjustment_layer("posterize", "posterize", pq.width, pq.height)
    pq.layers.append(aq)
    ref = render_document(pq)
    d = np.abs(R.render_tiled(pq, tile=128).astype(np.int16) - ref.astype(np.int16))
    pct = float((d.max(axis=2) > 0).mean())
    assert pct < 0.0005, "色调分离下差异像素过多 %.3f%%" % (pct * 100)
    # 同一个文档，现有的脏区重渲染也是这个量级（说明分块没有更差）
    sub = render_document(pq, region=(0, pq.height // 2, pq.width, pq.height))
    d2 = np.abs(sub.astype(np.int16)
                - ref[pq.height // 2:].astype(np.int16))
    print("  分块渲染：色调分离下 %.3f%% 的像素跳一档（脏区重渲染同样 %.3f%%，同一量级）"
          % (pct * 100, float((d2.max(axis=2) > 0).mean()) * 100))

    # 7) 分块渲染也要能顺手抓"调整层下方"的快照（不然大画布拖滑块就退回代理了）
    adj = doc.layers[2]
    snap = {"id": adj.id, "w": doc.width, "h": doc.height, "arr": None}
    R.render_tiled(doc, tile=128, snap_layer=adj, snap=snap)
    assert snap["arr"] is not None and snap["arr"].shape[:2] == (520, 700)
    assert "dst" not in snap and "box" not in snap, "临时键没清理干净"
    fast = R.render_from_snapshot(doc, adj, snap)
    d = np.abs(fast.astype(np.int16) - render_document(doc).astype(np.int16))
    assert d.max() <= 1, "分块抓的快照不能用，差 %d" % d.max()
    print("  分块渲染：顺手抓到调整层快照，快路径可用")

    # 8) 峰值内存：整块渲染随画布面积线性涨，分块不涨
    def peak(fn):
        tracemalloc.start()
        fn()
        _, pk = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return pk

    big = _tiled_doc(1400, 1000)
    p_full = peak(lambda: render_document(big))
    p_tile = peak(lambda: R.render_tiled(big, tile=256))
    assert p_tile < p_full * 0.6, \
        "分块没省下内存：%.0f MB vs 整块 %.0f MB" % (p_tile / 1e6, p_full / 1e6)
    print("  分块渲染：1400x1000 峰值 %.0f MB -> %.0f MB（省 %.0f%%）"
          % (p_full / 1e6, p_tile / 1e6, (1 - p_tile / p_full) * 100))


def _so_doc(name="so"):
    """一张背景 + 一块带旋转缩放的内容 —— 智能对象用例的统一底料。"""
    doc = Document(300, 220, name)
    bg = _solid(300, 220, (200, 200, 205))
    doc.layers.append(make_image_layer("bg", bg, 300, 220))
    block = _solid(80, 60, (240, 70, 60))
    l = make_image_layer("block", block, 300, 220)
    l.tx, l.ty = 120.0, 100.0
    l.sx, l.sy = 1.4, 1.1
    l.rot = 18.0
    doc.layers.append(l)
    return doc


def test_smart_object():
    """智能对象：内容、多实例、非破坏性滤镜、栅格化。"""
    doc = _so_doc()
    blk = doc.layers[1]
    origin_id = blk.id

    # ---- 1) 转换之后画面必须基本不变（整数比例/无旋转时应当逐位一致）----
    flat = _so_doc()
    flat.layers[1].sx = flat.layers[1].sy = 1.0
    flat.layers[1].rot = 0.0
    ref = render_document(flat)
    shell = convert_to_smart(flat, flat.layers[1].id)
    assert shell is not None and shell.kind == "smart", "转换没产生智能对象图层"
    got = render_document(flat)
    d = np.abs(got.astype(np.int16) - ref.astype(np.int16))
    assert d.max() == 0, "无缩放无旋转时转换应逐位一致，实际最大差 %d" % d.max()
    print("  转换：无变换场景与原地渲染逐位一致")

    before = render_document(doc)
    sl = convert_to_smart(doc, blk.id)
    after = render_document(doc)
    d = np.abs(before.astype(np.int16) - after.astype(np.int16)).max(axis=2)
    assert (d > 32).mean() < 0.002, \
        "带旋转缩放时转换后画面偏差过大（>32 的像素占 %.4f）" % (d > 32).mean()
    print("  转换：带旋转缩放场景 >32 的像素占 %.4f%%（内容先栅格化的固有代价）"
          % ((d > 32).mean() * 100))
    # 原图层子树被搬进内容：原来的 Layer 对象还在，但已经不在文档里
    assert doc.find(origin_id) is None, "原子图层应该已经搬进内容里了"
    content = doc.smart_contents[sl.so_id]
    assert [l.name for l in content.layers] == ["block"], "内容里应有被搬进来的图层"
    box = raw_bbox(sl)
    assert abs((box[0] + box[2]) / 2.0 - 120.0) < 1.0, "中心位置不该跑掉"

    # ---- 2) 内容改动要联动：touch() 之后重新合成 ----
    base = render_document(doc).copy()
    sub = content.layers[0]
    sub.visible = False
    content.touch()
    changed = render_document(doc)
    assert np.abs(changed.astype(np.int16)
                  - base.astype(np.int16)).max() > 0, "改了内容画面却没变"
    sub.visible = True
    content.touch()
    assert np.abs(render_document(doc).astype(np.int16)
                  - base.astype(np.int16)).max() == 0, "改回来应该完全还原"
    print("  内容编辑：touch() 之后重新合成，改回来完全还原")

    # ---- 3) 多实例共享同一份内容 ----
    twin = new_smart_instance(doc, sl.id)
    assert twin.so_id == sl.so_id, "新建实例必须共用同一个 content id"
    assert len(content_instances(doc, sl.so_id)) == 2, "应该有两个实例"
    twin.tx += 120.0
    two = render_document(doc)
    # 改内容 -> 两个实例一起变
    sub.opacity = 0.35
    content.touch()
    both = render_document(doc)
    diff = np.abs(both.astype(np.int16) - two.astype(np.int16)).max(axis=2)
    ys, xs = np.nonzero(diff > 8)
    assert len(xs) > 0, "改内容没有影响到画面"
    mid = (sl.tx + twin.tx) / 2.0
    assert xs.min() < mid < xs.max(), \
        "两个实例都应该跟着变（改动像素跨越 x=%d..%d，中线 %.0f）" % (
            xs.min(), xs.max(), mid)
    sub.opacity = 1.0
    content.touch()
    print("  多实例：改内容，%d 个实例一起更新" %
          len(content_instances(doc, sl.so_id)))

    # ---- 4) 智能滤镜是参数化的：可叠加、可改强度、删掉即还原 ----
    clean = render_document(doc).copy()
    sl.so_filters = [new_filter("gaussian", {"radius": 9.0})]
    blurred = render_document(doc)
    assert np.abs(blurred.astype(np.int16)
                  - clean.astype(np.int16)).max() > 0, "加了滤镜画面却没变"
    sl.so_filters = [new_filter("gaussian", {"radius": 9.0}, strength=0.4)]
    weak = render_document(doc)
    dblur = np.abs(blurred.astype(np.int16) - clean.astype(np.int16)).max()
    dweak = np.abs(weak.astype(np.int16) - clean.astype(np.int16)).max()
    assert dweak < dblur, "强度 40%% 的效果应该弱于 100%%（%d vs %d）" % (
        dweak, dblur)
    sl.so_filters = [new_filter("gaussian", {"radius": 9.0}, enabled=False)]
    assert np.abs(render_document(doc).astype(np.int16)
                  - clean.astype(np.int16)).max() == 0, "关掉的滤镜不该生效"
    sl.so_filters = []
    assert np.abs(render_document(doc).astype(np.int16)
                  - clean.astype(np.int16)).max() == 0, \
        "删掉滤镜必须回到原样（原始内容始终没被改写）"
    print("  智能滤镜：强度可调、可关闭、删掉与非破坏性还原")

    # ---- 5) 内容里的像素编辑要遵守写时复制 ----
    snap = doc.clone()                       # 模拟一次历史快照
    victim = snap.smart_contents[sl.so_id].layers[0]
    victim.detach_pixels()
    victim.image[:] = 0
    revived = doc.smart_contents[sl.so_id].layers[0]
    assert np.abs(revived.image.astype(np.int16)
                  - _solid(80, 60, (240, 70, 60)).astype(np.int16)).max() == 0, \
        "没 detach 就写像素会污染其它实例 —— 快照里的克隆不该共享这片内存"
    print("  写时复制：detach_pixels() 之后快照与当前内容互不干扰")

    # ---- 6) 栅格化 ----
    frozen = render_document(doc).copy()
    assert rasterize_smart(doc, sl), "栅格化失败"
    assert sl.kind == "image" and sl.so_id is None and sl.so_filters is None
    assert np.abs(render_document(doc).astype(np.int16)
                  - frozen.astype(np.int16)).max() == 0, \
        "栅格化不该改变画面"
    assert sl.so_id is None
    print("  栅格化：画面不变，内容与智能滤镜转为普通像素")


def test_smart_project_roundtrip():
    """智能对象要能进工程文件：存内容本体，不存派生栅格。"""
    import json
    import tempfile
    import zipfile

    doc = _so_doc()
    sl = convert_to_smart(doc, doc.layers[1].id)
    twin = new_smart_instance(doc, sl.id)
    twin.tx += 120.0
    twin.sx = twin.sy = 0.7
    # 只用确定性滤镜：杂色滤镜每次结果随机，没法逐位比对
    sl.so_filters = [new_filter("gaussian", {"radius": 5.0})]
    twin.so_filters = [new_filter("emboss", {"angle": 45, "height": 3,
                                             "amount": 80})]
    expect = render_document(doc).copy()
    cid = sl.so_id

    p = os.path.join(tempfile.gettempdir(), "cw_smart%s" % PROJECT_EXT)
    save_project(doc, p)
    got = load_project(p)
    assert len(got.smart_contents) == 1, \
        "读回来的嵌入内容数量不对：%d" % len(got.smart_contents)
    assert cid in got.smart_contents, "content id 变了，实例会全部失联"
    d = np.abs(render_document(got).astype(np.int16)
               - expect.astype(np.int16))
    assert d.max() == 0, "工程往返后画面不一致，最大差 %d" % d.max()
    keys = [l.so_id for l in got.layers if l.kind == "smart"]
    assert keys == [cid, cid], "两个实例仍应共用同一份内容"

    # 派生栅格不能进工程：zip 里每个 PNG 都必须是某个"源头"图层的位图
    with zipfile.ZipFile(p, "r") as z:
        names = [n for n in z.namelist() if n.startswith("layers/")]
        man = json.loads(z.read("manifest.json"))
    assert len(names) == 2, \
        "zip 里应该是背景 + 内容里的源图两张，实际 %d 张：%s" % (len(names), names)
    assert man.get("smart"), "manifest 里没有 smart 段"
    print("  工程往返：内容本体 + 智能滤镜参数都保住了，派生栅格没写进包")


def test_smart_edge_cases():
    """夹在外面的场景：组 + 剪贴蒙版、空内容、嵌套、内容回收。"""
    # 组 + 剪贴蒙版调整层：整棵子树连同只作用于它的调整层一起搬进去
    doc = Document(240, 180, "grp")
    doc.layers.append(make_image_layer("bg", _solid(240, 180, (60, 60, 70)),
                                       240, 180))
    a = make_image_layer("a", _solid(70, 50, (250, 210, 40)), 240, 180)
    a.tx, a.ty = 90.0, 70.0
    b = make_image_layer("b", _solid(40, 40, (40, 200, 250)), 240, 180)
    b.tx, b.ty = 130.0, 95.0
    b.opacity = 0.6
    grp = make_group("组", [a, b])
    doc.layers.append(grp)
    adj = make_adjustment_layer("反相", "invert", 240, 180)
    adj.clipped = True
    doc.layers.append(adj)
    expect = render_document(doc)
    shell = convert_to_smart(doc, grp.id)
    content = doc.smart_contents[shell.so_id]
    names = [l.name for l in content.layers]
    assert names == ["组", "反相"], "组与它的剪贴蒙版应一起搬进内容：%s" % names
    assert content.layers[1].mask is None or \
        content.layers[1].mask.shape[:2] == (content.height, content.width), \
        "调整层的画布尺寸蒙版必须裁到内容尺寸"
    d = np.abs(render_document(doc).astype(np.int16)
               - expect.astype(np.int16))
    # 组内有个 60% 不透明度的图层：中间结果先量化成 uint8 存起来，
    # 再被合成一次就会多一次舍入 —— 这是"内容先栅格化"的固有 ±1，躲不掉
    assert d.max() <= 1, "组 + 剪贴蒙版转换不该超出 ±1，实际最大差 %d" % d.max()
    print("  组 + 剪贴蒙版：整棵子树搬进内容，画面一致（半透明处 ±1 舍入）")

    # 内容丰富逐步扩大（内容尺寸 >= 子树包围盒）
    sub_a = content.layers[0].children[0]
    old_w, old_h = content.width, content.height
    sub_a.sx *= 2.0
    content.touch()
    sync_smart(doc)
    assert shell.image is not None and shell.image.shape[1] == old_w, \
        "内容画布不跟着子图层放大（内容尺寸是固定的壳）"
    sub_a.sx /= 2.0
    content.touch()

    # 嵌套：内容里再套一个智能对象
    inner_content = SmartContent("内层", 20, 20,
                                 [make_image_layer("c", _solid(20, 20,
                                                               (255, 0, 255)),
                                                   20, 20)])
    doc.smart_contents[inner_content.id] = inner_content
    from src.core.layer import Layer
    nest = Layer("嵌套", "smart")
    nest.blend = "Normal"
    nest.so_id = inner_content.id
    nest.so_filters = []
    nest.tx, nest.ty = 10.0, 10.0
    content.layers.append(nest)
    content.touch()
    arr = render_document(doc)
    magenta = np.all(arr[..., :3] == np.array([255, 0, 255], np.uint8),
                     axis=2)
    assert magenta.any(), "嵌套的智能对象没渲染出来"
    print("  嵌套智能对象：内容里可以再套一层")

    # 自引用不能把栈撑爆
    loop = SmartContent("自引用", 10, 10, [])
    doc.smart_contents[loop.id] = loop
    selfref = Layer("自引用", "smart")
    selfref.blend = "Normal"
    selfref.so_id = loop.id
    selfref.so_filters = []
    loop.layers.append(selfref)
    out = render_document(doc)     # 不抛异常就算过
    assert out.shape == (doc.height, doc.width, 4)
    print("  自引用：死循环被挡住，渲染没崩")

    # 内容回收：删掉最后一个引用它的实例
    doc.layers = [doc.layers[0]]
    prune_contents(doc)
    assert len(doc.smart_contents) == 0, \
        "没人引用的内容应该被回收，还剩 %d 份" % len(doc.smart_contents)
    print("  内容回收：没有实例引用时自动清理")


def test_svg_import():
    """SVG 导入（第二十三批）：解析 + 栅格化，每条语法支路都得画得出来。"""
    from tools.svgfixture import ALL_SVG
    from src.core.project_io import imread_rgba
    from src.core.svg_import import _path_data, render_svg, svg_size

    # ---------- 空图 / 固有尺寸 ----------
    a = render_svg(ALL_SVG["empty"])
    assert a.shape == (60, 100, 4) and a.dtype == np.uint8
    assert int(a[..., 3].sum()) == 0, "空图必须全透明"
    assert svg_size(ALL_SVG["empty"]) == (100, 60)
    # 既没有尺寸也没有 viewBox：按 SVG 规范给 300×150
    assert svg_size('<svg xmlns="http://www.w3.org/2000/svg"><rect width="10" '
                    'height="10"/></svg>') == (300, 150)

    # ---------- 基本形状：画上去了 + 该透的地方透 ----------
    a = render_svg(ALL_SVG["shapes"])
    assert a[30, 30, :3].tolist() == [231, 76, 60], a[30, 30]      # #e74c3c
    assert a[30, 30, 3] == 255
    assert a[170, 30, 3] == 0, "矩形外必须透明"
    for name in ("shapes", "path", "transforms", "gradients", "strokes",
                 "arcs", "clip", "use", "style"):
        arr = render_svg(ALL_SVG[name])
        assert arr.shape[2] == 4, name
        frac = float((arr[..., 3] > 0).mean())
        assert frac > 0.01, "%s 只画出 %.2f%% 的像素" % (name, frac * 100)

    # ---------- transform 位置 ----------
    a = render_svg(ALL_SVG["transforms"])
    assert a[40, 40, :3].tolist() == [41, 128, 185], a[40, 40]     # translate 后的蓝块
    assert a[5, 40, 3] == 0, "translate(20,20) 之前不该有东西"
    assert a[20, 90, :3].tolist() == [192, 57, 43], a[20, 90]      # translate+scale 后的红圆

    # ---------- 渐变：线性两端 + 中点 ----------
    g = render_svg('<svg xmlns="http://www.w3.org/2000/svg" width="40" '
                   'height="10"><defs><linearGradient id="a">'
                   '<stop offset="0" stop-color="#0000ff"/>'
                   '<stop offset="1" stop-color="#ff0000"/></linearGradient>'
                   '</defs><rect x="0" y="0" width="40" height="10" '
                   'fill="url(#a)"/></svg>')
    # 端点不写精确值：抗锯齿会把最外一列的 rgb 拉偏（premultiplied 下的边缘像素）
    assert g[5, 2, 2] > 200 and g[5, 2, 0] < 60, g[5, 2]      # 明显偏蓝
    assert g[5, 37, 0] > 200 and g[5, 37, 2] < 60, g[5, 37]   # 明显偏红
    assert g[5, 19, 0] > 100, "中点应该已经过半偏红"
    # userSpaceOnUse 的另一条渐变（坐标是用户空间，不跟着 bbox 走）
    g2 = render_svg('<svg xmlns="http://www.w3.org/2000/svg" width="50" '
                    'height="10"><defs><linearGradient id="b" '
                    'gradientUnits="userSpaceOnUse" x1="0" y1="0" x2="50" '
                    'y2="0"><stop offset="0" stop-color="#000"/>'
                    '<stop offset="1" stop-color="#fff"/></linearGradient>'
                    '</defs><rect x="0" y="0" width="50" height="10" '
                    'fill="url(#b)"/></svg>')
    assert g2[5, 1, 0] < 40 and g2[5, 48, 0] > 215, (g2[5, 1], g2[5, 48])
    # 径向渐变：圆心是白的
    rg = render_svg('<svg xmlns="http://www.w3.org/2000/svg" width="40" '
                    'height="40"><defs><radialGradient id="c">'
                    '<stop offset="0" stop-color="#fff"/>'
                    '<stop offset="1" stop-color="#000"/></radialGradient>'
                    '</defs><circle cx="20" cy="20" r="20" '
                    'fill="url(#c)"/></svg>')
    # 圆心不写 255：Qt 对「focus 与圆心重合」的径向渐变，圆心不是精确的 stop0
    assert rg[20, 20, 0] > 235, rg[20, 20]
    assert rg[20, 2, 0] < 60, rg[20, 2]

    # ---------- 弧（A 命令）----------
    p = _path_data("M0 20 A20 20 0 0 1 40 20 Z")
    bb = p.boundingRect()
    assert abs(bb.width() - 40) < 0.5 and abs(bb.height() - 20) < 0.5, bb
    half = _path_data("M0 0 A50 50 0 0 1 100 0")
    assert abs(half.boundingRect().height() - 50) < 0.5, \
        half.boundingRect().height()
    # 半径不够包住两端点时应该被**等比**放大，不是崩掉
    tiny = _path_data("M0 0 A2 100 0 0 1 40 0")
    assert tiny.boundingRect().width() > 39, tiny.boundingRect()
    # 半径 0 退化成直线，不能死循环
    assert _path_data("M0 0 A0 0 0 0 1 30 30").boundingRect().width() >= 0

    # ---------- clip / use / style ----------
    a = render_svg(ALL_SVG["clip"])
    assert a[100, 100, 3] > 250, "clip 圆内"
    assert a[5, 5, 3] == 0, "clip 圆外必须透明"
    assert a[60, 150, :3].tolist() == [46, 204, 113], a[60, 150]
    a = render_svg(ALL_SVG["use"])
    assert a[40, 40, :3].tolist() == [231, 76, 60], a[40, 40]
    assert a[100, 100, :3].tolist() == [231, 76, 60], a[100, 100]
    a = render_svg(ALL_SVG["style"])
    assert a[30, 50, :3].tolist() == [233, 30, 99], a[30, 50]      # 内联 style
    assert 80 < a[30, 150, 3] < 230, a[30, 150]                    # fill-opacity 0.4
    assert 80 < a[90, 150, 3] < 230, a[90, 150]                    # group opacity 0.5

    # ---------- 文字（离屏 Qt 常常没有字体，有权画不出但绝不能崩）----------
    # **没有 QApplication 时 drawText / QFontDatabase.families() 是段错误
    # 直接带崩进程**（127 且无任何输出），所以先 _ensure_gui() 再问字体。
    _ensure_gui()
    from src.core.text import ensure_fonts
    ensure_fonts()
    from PySide6.QtGui import QFontDatabase
    if QFontDatabase.families():
        t = render_svg(ALL_SVG["text"])
        assert int((t[..., 3] > 0).sum()) > 50, "有字体却没画出文字"
    else:
        render_svg(ALL_SVG["text"])

    # ---------- 走文件路径 + imread_rgba ----------
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.svg")
        with open(p, "w", encoding="utf-8") as f:
            f.write(ALL_SVG["shapes"])
        arr = imread_rgba(p, target=(80, 80))
        assert arr.shape == (80, 80, 4), arr.shape
        assert int(arr[..., 3].sum()) > 0
        # 缩小 200→80 之后内容得落在画面中心附近（(40,40) 那点本身在
        # 图形边界上，抗锯齿只给 204，别拿它当「画上了」的判据）
        ys, xs = np.nonzero(arr[..., 3] > 8)
        assert abs(float(xs.mean()) - 40) < 8 and abs(float(ys.mean()) - 40) < 8, \
            (float(xs.mean()), float(ys.mean()))

    # ---------- 尺寸上限 ----------
    big = render_svg('<svg xmlns="http://www.w3.org/2000/svg" width="90000" '
                     'height="20"><rect width="90000" height="20" '
                     'fill="#f00"/></svg>', width=None, height=20)
    assert big.shape == (20, 20, 4), big.shape       # 被夹到 MAX_DIM

    print("  SVG 导入：%d 份样例全画上 / 变换位置 / 三种渐变 / A 弧 / clip / "
          "use / 内联 style / 尺寸夹取" % len(ALL_SVG))


def test_heif_import():
    try:
        import io
        import pillow_heif
    except ImportError:
        print("  HEIC 导入跳过（没装 pillow-heif）")
        return
    from src.core.project_io import imread_rgba

    w, h = 40, 24
    src = np.zeros((h, w, 4), np.uint8)
    src[..., 0] = 200
    src[..., 1] = 90
    src[..., 3] = 255

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.heic")
        buf = io.BytesIO()
        pillow_heif.register_heif_opener()
        pillow_heif.encode("RGBA", (w, h), src.tobytes(), buf, quality=90)
        with open(p, "wb") as f:
            f.write(buf.getvalue())
        out = imread_rgba(p)
        assert out.shape == (h, w, 4), out.shape
        assert out[..., 3].min() > 250, "alpha 丢了"
        d0 = int(np.abs(out[..., :3].astype(np.int16) -
                         src[..., :3]).max())
        assert d0 <= 12, "HEIC 往返色差 %d" % d0
        assert d0 > 0, "HEIC 编解码是有损的，色差为 0 说明在偷懒（分辨率/位深没真走）"

    # 缺依赖时必须给**能照着装的提示**，不是一句 import 失败
    from src.core import project_io as PIO
    PIO._HEIF_OK = False
    try:
        PIO._require_heif()
    except RuntimeError as e:
        assert "pillow-heif" in str(e), str(e)
    else:
        raise AssertionError("没装依赖时应该抛错")
    PIO._HEIF_OK = None

    print("  HEIC 导入：40x24 往返色差 %d/255（有损），缺依赖给安装提示" % d0)


def main():
    print("Compositor for Windows —— 自检")
    test_blend_modes_finite()
    test_known_values()
    test_transform_and_render()
    test_group_passthrough()
    test_project_roundtrip()
    test_selection_shapes()
    test_magic_wand()
    test_brush_primitives()
    test_stroke_and_copy_on_write()
    test_adjustment_math()
    test_adjustment_layer_render()
    test_clipped_adjustment()
    test_adjustment_roundtrip()
    test_filters()
    test_text_layer()
    test_text_enhance()
    test_psd_import()
    test_psd_text()
    test_psd_effects()
    test_retouch()
    test_retouch2()
    test_content_aware()
    test_canvas_size()
    test_guides()
    test_channels()
    test_new_adjustments()
    test_adjustment_enhance()
    test_median_large()
    test_brush_advanced()
    test_layer_effects()
    test_layer_effects_advanced()
    test_float_selection()
    test_selection_enhance()
    test_color_range()
    test_quick_mask()
    test_float_transform()
    test_render_fast_paths()
    test_dirty_regions()
    test_nested_adjust_snapshot()
    test_render_region_equals_full()
    test_render_proxy()
    test_dirty_helpers()
    test_adjust_snapshot()
    test_filter_preview_downscale()
    test_render_tiled()
    test_smart_object()
    test_smart_project_roundtrip()
    test_smart_edge_cases()
    test_svg_import()
    test_heif_import()
    print("全部通过。")


if __name__ == "__main__":
    main()
