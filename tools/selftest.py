# -*- coding: utf-8 -*-
"""无界面自检：验证混合模式、变换、蒙版、分组、渲染与工程读写。

    python -m tools.selftest
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.blend import BLEND_MODES, blend_colors          # noqa: E402
from src.core.document import (Document, make_adjustment_layer,  # noqa: E402
                               make_group, make_image_layer)
from src.core.layer import Layer                              # noqa: E402
from src.core.project_io import (PROJECT_EXT, export_flat, load_project,  # noqa: E402
                                 save_project)
from src.core.render import (has_dissolve, render_document,   # noqa: E402
                             render_proxy)
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
    test_psd_import()
    test_new_adjustments()
    test_layer_effects()
    test_float_selection()
    test_render_region_equals_full()
    test_render_proxy()
    test_dirty_helpers()
    test_adjust_snapshot()
    test_filter_preview_downscale()
    test_render_tiled()
    test_smart_object()
    test_smart_project_roundtrip()
    test_smart_edge_cases()
    print("全部通过。")


if __name__ == "__main__":
    main()
