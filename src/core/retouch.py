# -*- coding: utf-8 -*-
"""修饰类工具的纯计算部分：渐变、局部模糊、吸管取色。

这三个都不碰 UI，也不碰图层对象 —— 输入 ndarray，输出 ndarray。
`canvas_view.py` 只负责把鼠标坐标转成调用参数。

**渐变**支持 Photoshop 的 5 种样式：线性 / 径向 / 角度 / 对称 / 菱形。
数学和 `effects.py::_gradient_colors` 同一套（那边是图层内容框当坐标系，
这边是用户拖出来的起止点），所以同一个渐变在「图层样式 → 渐变叠加」和
「渐变工具」里看起来一致。

**局部模糊**是 Photoshop 的模糊工具：按住拖动，笔刷覆盖的那块被高斯模糊，
边缘按到笔刷中心的距离羽化，不是一次性给整层套滤镜。所以它必须**增量**做 ——
每帧只处理新扫过的那一块，否则大图层上会卡死。
"""

from __future__ import annotations

import math

import cv2
import numpy as np

# 渐变样式（显示名与 effects.py 保持一致）
GRADIENT_STYLES = ("线性", "径向", "角度", "对称", "菱形")

# 吸管取样的默认取样半径（方形邻域边长的一半）
PICK_RADIUS_DEFAULT = 0


def gradient_params():
    """渐变工具的默认参数。样式名与 effects.py 共用同一套字符串。"""
    return {"style": "线性", "color": [0, 0, 0], "color2": [255, 255, 255]}


def gradient_t(shape, start, end, style="线性"):
    """-> (h,w) float32，0~1：每个像素在渐变上的位置。

    `start` / `end` 是画布坐标里的两个点。径向与角度样式只看 end，
    当成圆心与半径 / 起始角。
    """
    h, w = shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    X = xx.astype(np.float32)
    Y = yy.astype(np.float32)
    x0, y0 = float(start[0]), float(start[1])
    x1, y1 = float(end[0]), float(end[1])

    style = style if style in GRADIENT_STYLES else "线性"
    if style == "径向":
        # start 是圆心，拖多远就是半径（PS 的径向渐变：圆心处是第一个色标）
        r = math.hypot(x1 - x0, y1 - y0)
        if r < 1.0:
            r = 1.0
        t = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2) / r
        return np.clip(t, 0.0, 1.0).astype(np.float32)

    if style == "角度":
        # start 是圆心，end 的方向是起始角，扫一整圈
        ang = math.atan2(y1 - y0, x1 - x0)
        t = (np.arctan2(Y - y0, X - x0) - ang) / (2.0 * math.pi)
        return np.clip(t - np.floor(t), 0.0, 1.0).astype(np.float32)

    if style == "菱形":
        # start 是菱形中心，拖到 end 的距离是半对角线
        r = abs(x1 - x0) + abs(y1 - y0)
        if r < 1.0:
            r = 1.0
        t = (np.abs(X - x0) + np.abs(Y - y0)) / r
        return np.clip(t, 0.0, 1.0).astype(np.float32)

    # 线性 / 对称：start 到 end 的投影
    dx, dy = x1 - x0, y1 - y0
    len2 = dx * dx + dy * dy
    if len2 < 0.25:
        # 起终点几乎重合：给一条极窄的斜坡，避免除以 0
        return np.full(shape[:2], 0.5, np.float32)
    lin = np.clip(((X - x0) * dx + (Y - y0) * dy) / len2, 0.0, 1.0)
    if style == "对称":
        return np.abs(lin * 2.0 - 1.0).astype(np.float32)
    return lin.astype(np.float32)


def gradient_colors(shape, start, end, params):
    """-> (h,w,3) float32 0~1：渐变在每个像素上的颜色。"""
    t = gradient_t(shape, start, end, params.get("style", "线性"))
    c0 = np.array(params.get("color", (0, 0, 0)), np.float32) / 255.0
    c1 = np.array(params.get("color2", (255, 255, 255)), np.float32) / 255.0
    return c0 + (c1 - c0) * t[..., None]


def apply_gradient(img, start, end, params, sel=None):
    """把渐变涂进 (h,w,4) uint8 RGBA，原地修改并返回同一个数组。

    `sel` 是 (h,w) float32 0~1 的图层源坐标选区；None 表示全图层。
    选区之外的渐变**完全不写**（连 alpha 都不碰）。

    前景色/背景色的语义跟 Photoshop 一致：渐变从前景色走到背景色，
    所以 `params["color"]` 取 fg、`color2` 取 bg，由调用方填好。
    """
    spec = dict(gradient_params())
    spec.update(params or {})
    t = gradient_t(img.shape, start, end, spec["style"])
    c0 = np.array(spec["color"], np.float32) / 255.0
    c1 = np.array(spec["color2"], np.float32) / 255.0
    col = c0 + (c1 - c0) * t[..., None]

    if sel is not None:
        w = np.clip(np.asarray(sel, np.float32), 0.0, 1.0)
        if not w.any():
            return img
        ww = w[..., None]
    else:
        ww = None

    dst = img[..., :3].astype(np.float32) / 255.0
    if ww is None:
        img[..., :3] = np.clip(col * 255.0 + 0.5, 0, 255).astype(np.uint8)
        return img
    img[..., :3] = np.clip((dst * (1.0 - ww) + col * ww) * 255.0 + 0.5,
                           0, 255).astype(np.uint8)
    return img


def feathered_disk(size, hardness=0.5):
    """-> (size,size) float32：笔刷覆盖权重（中心 1，边缘羽化到 0）。

    和 `brush.py` 的硬度语义一致：hardness=1 是硬边圆盘，0 是全羽化。
    """
    r = max(1, int(size)) // 2
    d = float(max(0.0, min(1.0, hardness)))
    inner = r * d
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    dist = np.sqrt(xx.astype(np.float32) ** 2 + yy.astype(np.float32) ** 2)
    if inner <= 0.5:
        w = np.clip(1.0 - dist / max(r, 1), 0.0, 1.0)
    else:
        # 中心圆盘是 1，从 inner 到 r 之间 smoothstep 落到 0
        t = np.clip((dist - inner) / max(r - inner, 1e-3), 0.0, 1.0)
        w = 1.0 - (t * t * (3.0 - 2.0 * t))
        w = np.where(dist <= inner, 1.0, w)
    w[dist > r] = 0.0
    return w.astype(np.float32)


def blur_region(img, cx, cy, radius, hardness=0.5, strength=1.0):
    """把 (h,w,4) uint8 的一个圆形区域做高斯模糊，原地修改。

    这是 Photoshop 的**模糊工具**语义：只模糊笔刷盖住的那块，边缘按笔刷
    形状羽化，不是给整层套一次高斯。

    `strength` 0~1 是这次落笔的量（多次涂抹会累积到完全模糊）。
    半径很大时只取 ROI 卷积，不整幅做。
    """
    if radius < 0.5 or strength <= 0.0:
        return img
    h, w = img.shape[:2]
    x0, y0 = int(math.floor(cx - radius)), int(math.floor(cy - radius))
    x1 = int(math.ceil(cx + radius)) + 1
    y1 = int(math.ceil(cy + radius)) + 1
    # 核是 sigma = radius/3（PS 的经验值），卷积要往外再扩 3 sigma
    sigma = max(radius / 3.0, 0.5)
    pad = int(math.ceil(sigma * 3.0))
    rx0, ry0 = x0 - pad, y0 - pad
    rx1, ry1 = x1 + pad, y1 + pad
    sx0, sy0 = max(0, rx0), max(0, ry0)
    sx1, sy1 = min(w, rx1), min(h, ry1)
    if sx1 <= sx0 or sy1 <= sy0:
        return img

    roi = img[sy0:sy1, sx0:sx1]
    src = roi.astype(np.float32) / 255.0
    a = src[..., 3:4]
    pm = src[..., :3] * a                      # 预乘，和 filters.py 一致
    k = int(max(3, 2 * int(sigma * 2.0) + 1))
    k |= 1                                        # 必须是奇数
    bpm = cv2.GaussianBlur(pm, (k, k), sigma)
    ba = cv2.GaussianBlur(a, (k, k), sigma)
    if ba.ndim == 2:
        ba = ba[..., None]
    ba = np.clip(ba, 0.0, 1.0)
    nrgb = np.clip(bpm / np.maximum(ba, 1e-6), 0.0, 1.0)

    # 笔刷权重落在原 ROI 上（不是卷积扩出去的那圈）
    side = int(math.ceil(radius)) * 2 + 1
    wgt = feathered_disk(side, hardness)
    bx0 = int(math.floor(cx - radius)) - sx0
    by0 = int(math.floor(cy - radius)) - sy0
    th, tw = roi.shape[:2]
    wfull = np.zeros((th, tw), np.float32)
    gy0, gx0 = max(0, by0), max(0, bx0)
    gy1 = min(th, by0 + wgt.shape[0])
    gx1 = min(tw, bx0 + wgt.shape[1])
    if gy1 > gy0 and gx1 > gx0:
        wfull[gy0:gy1, gx0:gx1] = wgt[gy0 - by0:gy1 - by0,
                                       gx0 - bx0:gx1 - bx0]
    wfull = np.clip(wfull * float(strength), 0.0, 1.0)[..., None]
    if not wfull.any():
        return img

    o_rgb = roi[..., :3].astype(np.float32) / 255.0
    o_a = roi[..., 3:4].astype(np.float32) / 255.0
    out_rgb = o_rgb * (1.0 - wfull) + nrgb * wfull
    out_a = o_a * (1.0 - wfull) + ba * wfull
    roi[..., :3] = np.clip(out_rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
    roi[..., 3:4] = np.clip(out_a * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return img


def pick_color(arr, x, y, radius=PICK_RADIUS_DEFAULT, composite=None):
    """吸管：返回 (r, g, b, a)，越界或空图返回 None。

    `composite` 非 None 时从**合成结果**取色（PS 的「取样：所有图层」）；
    否则从传入的那张数组取（当前图层）。
    `radius` > 0 时取邻域均值，能避开单个杂点。
    """
    src = composite if composite is not None else arr
    if src is None or src.size == 0:
        return None
    h, w = src.shape[:2]
    xi, yi = int(round(x)), int(round(y))
    if radius <= 0:
        if not (0 <= xi < w and 0 <= yi < h):
            return None
        px = src[yi, xi]
        return (int(px[0]), int(px[1]), int(px[2]), int(px[3])
                if px.shape[0] > 3 else 255)
    r = int(radius)
    x0, y0 = max(0, xi - r), max(0, yi - r)
    x1, y1 = min(w, xi + r + 1), min(h, yi + r + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    patch = src[y0:y1, x0:x1]
    m = patch.reshape(-1, patch.shape[-1])
    return tuple(int(v) for v in m.mean(axis=0))


# =====================================================================
# 克隆图章
# =====================================================================

class CloneSource:
    """克隆图章的采样源：按下 Alt 时对目标图层拍一张快照，之后一直用它。

    **为什么要快照而不是实时读源图层**：PS 的克隆图章就是这样 —— 采样之后
    源图层再改，已采的那份不变；而且源和目标可以是同一张图，不快照就会
    边涂边采到自己的新颜料（无限自我复制，越涂越花）。

    `mask` 是羽化权重（同尺寸 float32 0~1），按住 Shift 拖动可以只重新采样
    而不落笔。
    """

    __slots__ = ("image", "ox", "oy", "hardness", "size")

    def __init__(self, image, ox, oy, hardness=0.5, size=60):
        self.image = image
        self.ox = int(ox)
        self.oy = int(oy)
        self.hardness = float(hardness)
        self.size = int(size)

    def copy(self):
        c = CloneSource(self.image.copy(), self.ox, self.oy,
                        self.hardness, self.size)
        return c

    @property
    def shape(self):
        return self.image.shape[:2]


def clone_stamp_at(dst, src_snap, cx, cy, radius, hardness=0.5,
                   strength=1.0, src_offset=None):
    """把采样源里对应位置的一块按羽化覆盖到 dst（原地）。

    `dst` / `src_snap.image` 都是 (h,w,4) uint8 RGBA，尺寸**必须一致**。
    `src_offset` 是 (dx, dy)：采样源相对目标的位移。默认按 `CloneSource`
    记录的 (ox, oy) 与当前落点算 —— 即「按下 Alt 的那点」与「现在这点」
    的差，PS 就是这样保持图案随笔尖一起走的。

    采样落到画布外的部分**跳过**（不做边缘拉伸，PS 也是直接不画）。
    """
    if dst is None or src_snap is None or radius < 0.5 or strength <= 0.0:
        return False
    if dst.shape[:2] != src_snap.shape[:2]:
        return False
    h, w = dst.shape[:2]
    if src_offset is None:
        src_offset = (src_snap.ox - int(round(cx)), src_snap.oy - int(round(cy)))
    dx, dy = int(src_offset[0]), int(src_offset[1])

    x0 = int(round(cx - radius))
    y0 = int(round(cy - radius))
    x1 = int(round(cx + radius)) + 1
    y1 = int(round(cy + radius)) + 1
    if x1 <= 0 or y1 <= 0 or x0 >= w or y0 >= h:
        return False                      # 整块都在画布外
    wgt = feathered_disk(int(round(radius)) * 2 + 1, hardness)

    # 目标与源各自的合法窗口。x0/y0 **就是**权重窗口的左上角
    # （权重边长正好 2r+1，中心索引就是 r），不要再减一次 off ——
    # 减了会让笔刷整体偏移 r 个像素
    dx0, dy0 = x0, y0
    dx1, dy1 = x0 + wgt.shape[1], y0 + wgt.shape[0]
    sx0, sy0 = dx0 + dx, dy0 + dy
    sx1, sy1 = sx0 + wgt.shape[1], sy0 + wgt.shape[0]
    # 源与目标各自与画布求交集，再取**两者共同的**矩形。
    # 不能要求「裁剪后两者尺寸相同」—— 源窗口一半出界时尺寸必然不同，
    # 那样写等于「稍微出界就一像素都不画」（PS 是画能画的那部分）
    # dst 里 (X,Y) 对应的源坐标是 (X+dx, Y+dy)，所以源不越界的上界是 w-dx / h-dy
    gx0 = max(dx0, 0, -dx)
    gy0 = max(dy0, 0, -dy)
    gx1 = min(dx1, w, w - dx)
    gy1 = min(dy1, h, h - dy)
    if gx1 <= gx0 or gy1 <= gy0:
        return False
    jx0 = gx0 + dx
    jy0 = gy0 + dy

    th, tw = gy1 - gy0, gx1 - gx0
    wy = max(0, gy0 - dy0)
    wx = max(0, gx0 - dx0)
    wgt = wgt[wy:wy + th, wx:wx + tw]
    if wgt.shape != (th, tw) or not wgt.any():
        return False
    ww = (np.clip(wgt * float(strength), 0.0, 1.0))[..., None]

    dr = dst[gy0:gy1, gx0:gx1]
    sl = src_snap.image[jy0:jy0 + th, jx0:jx0 + tw]
    d_rgb = dr[..., :3].astype(np.float32)
    d_a = dr[..., 3:4].astype(np.float32)
    s_rgb = sl[..., :3].astype(np.float32)
    s_a = sl[..., 3:4].astype(np.float32)
    dr[..., :3] = np.clip(d_rgb * (1.0 - ww) + s_rgb * ww, 0, 255).astype(np.uint8)
    dr[..., 3:4] = np.clip(d_a * (1.0 - ww) + s_a * ww, 0, 255).astype(np.uint8)
    return True


def clone_stroke_path(dst, src_snap, path, radius, hardness=0.5,
                      strength=1.0):
    """沿 path（源坐标的连续点）盖一串克隆印章，返回实际盖了几次。

    `path` 是 [(x, y), ...]。落点之间按半径/3 补样，和模糊工具同理 ——
    否则快速拖动会漏成一串珠子。

    注意 `strength` 是**每次盖章**的量，补样也吃这个值，所以间隔越小累积越快。
    这与 Photoshop 的画笔一致（间隔小 = 同一处叠更多次）。
    """
    if not path:
        return 0
    pts = [(float(p[0]), float(p[1])) for p in path]
    n = 0
    step = max(1.0, radius / 3.0)
    for i in range(len(pts) - 1):
        ax, ay = pts[i]
        bx, by = pts[i + 1]
        # 含两端：j 从 0 到 k，每段都盖到端点（相邻段会重叠一次，
        # 但那正是「上一段的终点 = 下一段的起点」本该重叠的地方）
        k = max(1, int(math.hypot(bx - ax, by - ay) / step))
        for j in range(k + 1):
            u = j / float(k)
            if clone_stamp_at(dst, src_snap, ax + (bx - ax) * u,
                              ay + (by - ay) * u, radius, hardness,
                              strength):
                n += 1
    return n


# =====================================================================
# 污点修复（healing brush）
# =====================================================================

def heal_region(img, cx, cy, radius, hardness=0.5, strength=1.0):
    """把 (h,w,4) uint8 里以 (cx,cy) 为中心的**瑕点**用周围像素推断掉。

    这是 Photoshop 污点修复画笔的语义，与「模糊」不同：模糊是抹平，
    污点修复是**让那块的纹理与质感跟周围接得上**。

    做法：取中心那一小块当"待修补区域"，再从它周围取一圈同样大小的
    参照块，按当前噪声水平（`noise`）在两者之间做 `inpaint`：
      - `noise` 小 -> 直接用周围的平均色（瑕点很浅，或者只是颜色不对）
      - `noise` 大 -> 用 `cv2.inpaint` 按边界往里扩散纹理（瑕点很深，
        比如划痕、洞）
    之后按笔刷羽化权重混合回去，中心那块与周围无缝。
    """
    if radius < 1.0 or strength <= 0.0:
        return False
    h, w = img.shape[:2]
    x0, y0 = int(round(cx - radius)), int(round(cy - radius))
    x1, y1 = int(round(cx + radius)) + 1, int(round(cy + radius)) + 1
    pad = int(math.ceil(radius / 3.0 * 3.0))
    rx0, ry0 = x0 - pad, y0 - pad
    rx1, ry1 = x1 + pad, y1 + pad
    sx0, sy0 = max(0, rx0), max(0, ry0)
    sx1, sy1 = min(w, rx1), min(h, ry1)
    if sx1 <= sx0 or sy1 <= sy0:
        return False
    roi = img[sy0:sy1, sx0:sx1]
    rh, rw = roi.shape[:2]

    # 中心瑕点区（在 roi 里的坐标）
    cx0, cy0 = x0 - sx0, y0 - sy0
    cx1, cy1 = x1 - sx0, y1 - sy0
    if cx1 <= cx0 or cy1 <= cy0:
        return False
    ch, cw = cy1 - cy0, cx1 - cx0

    # 参照块：取中心块**右侧**同样大小的一块（瑕点左边往往就是要参照的
    # 干净区域，但右边更安全 —— 右边坏了还能往更右找，左边会连带进瑕点）
    if rw < cw or rh < ch:
        return False
    ref_x0 = min(cx1, rw - cw)
    ref_y0 = min(max(cy0, 0), rh - ch)

    src_rgb = roi[..., :3].astype(np.float32) / 255.0
    src_a = roi[..., 3:4].astype(np.float32) / 255.0
    bad = (slice(cy0, cy1), slice(cx0, cx1))
    ref = (slice(ref_y0, ref_y0 + ch), slice(ref_x0, ref_x0 + cw))

    # 当前噪声水平：中心区 vs 参照区的亮度差
    def _lum(rgb):
        return (rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587
                + rgb[..., 2] * 0.114)

    diff = float(np.abs(_lum(src_rgb[bad]) - _lum(src_rgb[ref])).mean())
    noise = min(max(diff * 3.0, 0.0), 1.0)

    # 轻度：直接换成参照块的内容（瑕点只是颜色不对，纹理本身没问题）
    # 重度：按边界往里扩散（cv2.inpaint，遮罩就是中心区）
    if noise < 0.18:
        fixed_rgb = src_rgb[ref].copy()
        fixed_a = src_a[ref].copy()
    else:
        mask = np.zeros((rh, rw), np.uint8)
        mask[bad] = 255
        bgr8 = np.clip(src_rgb[..., ::-1] * 255.0, 0, 255).astype(np.uint8)
        fixed_bgr = cv2.inpaint(bgr8, mask, 3.0, cv2.INPAINT_TELEA)
        # **只要中心区那一块** —— inpaint 返回的是整幅 ROI，尺寸是 (rh, rw)，
        # 直接拿去跟 (ch, cw) 的目标混会 broadcasting 报错
        fixed_rgb = (fixed_bgr[..., ::-1].astype(np.float32) / 255.0)[bad]
        # 让 alpha 跟周围一致（瑕点常是"破洞"，alpha 也不对）
        ring = np.ones((rh, rw), np.float32)
        ring[bad] = 0.0
        n_ring = float(ring.sum())
        fixed_a = (np.full((ch, cw, 1), float(src_a[..., 0].sum()) / max(n_ring, 1.0),
                         np.float32) if n_ring > 0 else src_a[bad].copy())

    # 按笔刷羽化权重把修好的那一块混回去。中心区的边长就是
    # int(round(radius))*2+1，与 feathered_disk 同尺寸，直接对齐即可
    wgt = feathered_disk(int(round(radius)) * 2 + 1, hardness)
    if wgt.shape != (ch, cw):
        wgt = np.ones((ch, cw), np.float32)
    ww = np.clip(wgt * float(strength), 0.0, 1.0)[..., None]

    tgt_rgb = src_rgb[bad]
    tgt_a = src_a[bad]
    out_rgb = tgt_rgb * (1.0 - ww) + fixed_rgb * ww
    out_a = tgt_a * (1.0 - ww) + fixed_a * ww
    roi[..., :3][bad] = np.clip(out_rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
    roi[..., 3:4][bad] = np.clip(out_a * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return True
    return True
