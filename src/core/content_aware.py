# -*- coding: utf-8 -*-
"""内容感知填充：用选区**周围**的像素推断选区里该是什么。

和「污点修复」（retouch.heal_region）的区别：污点修复是"一个小点补一下"，
笔刷多大就修多大；内容感知填充是"整块选区都要凭空长出来"，而且可以
长到**画布外面去**（Photoshop 的「内容感知填充 → 扩展画布外的空白」）。

三种取样策略（对应 Photoshop 里那三个单选项）：

- **邻近**：以选区边缘一圈像素为边界条件，往里扩散（`cv2.inpaint`）。
  最快，适合天空、墙面这类平滑区域；遇到纹理就糊。
- **镜像**：把选区一侧的像素沿选区中心对称翻过去。
  适合对称构图；选区贴边时退化成邻近。
- **纹理合成**：真正的按块匹配（PatchMatch 的一阶版本）——从周围找
  **每一块**的最相似邻块贴进去，能骗过眼睛，但慢，而且要占内存。

**必须预乘 alpha**（同 §3.2）：选区边缘常常半透明，直接卷进黑边会出脏边。
所以整个流程先把 rgba 拆成预乘的 pm 与 a，算完再反预乘。

所有函数都**只吃 ndarray、只吐 ndarray**，不碰 UI 和图层对象。
"""

from __future__ import annotations

import math

import cv2
import numpy as np

# 采样策略
FILL_MODES = ("邻近", "镜像", "纹理合成")
# 纹理合成默认块边长（像素）。块越大越慢，但对大尺度纹理越像。
TEXTURE_PATCH_DEFAULT = 9


# ---------- 预乘 / 反预乘 ----------

def _premul(img):
    """(h,w,4) uint8 -> (pm (h,w,3) float32 0~1, a (h,w,1) float32 0~1)。

    整幅 alpha 都 1 的图层（即普通不透明位图）走快路径：直接返回
    `img/255` 与全 1，省掉一次乘法。
    """
    f = img.astype(np.float32) / 255.0
    a = f[..., 3:4]
    if a.min() >= 1.0:
        return f[..., :3].copy(), np.ones_like(a)
    return f[..., :3] * a, a


def _unpremul(pm, a):
    """(pm, a) float32 -> (h,w,4) uint8，alpha 原样带上。"""
    out_a = np.clip(a[..., 0], 0.0, 1.0)
    safe = np.maximum(out_a, 1e-6)[..., None]
    rgb = np.clip(np.where(out_a[..., None] > 1e-6, pm / safe, 0.0), 0.0, 1.0)
    out = np.empty(pm.shape[:2] + (4,), np.uint8)
    out[..., :3] = np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
    out[..., 3] = np.clip(out_a * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return out


def _keep_ch(x):
    """OpenCV 某些操作会把 (h,w,1) 压成 (h,w)，这里补回来（§3.3）。"""
    return x[..., None] if x.ndim == 2 else x


def _mask_to_u8(mask):
    """0~1 float -> 0/255 uint8（阈值 0.5，与 Photoshop 的抗锯齿选区一致）。"""
    return (np.clip(mask, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _bbox(mask):
    """选区的包围盒 (x0, y0, x1, y1)，空选区返回 None。"""
    ys, xs = np.nonzero(mask >= 0.5)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


# ---------- 三种取样 ----------

def _fill_nearest(rgb, a, mask_u8):
    """邻近：以边缘为条件往里扩散。

    `radius` 是扩散半径。选区很大时要**先降采样再升回来**，
    否则 1000px 的选区会跑很久（cv2.inpaint 是 O(选区面积) 且常数不小）。
    """
    h, w = mask_u8.shape[:2]
    box = _bbox(mask_u8.astype(np.float32) / 255.0)
    if box is None:
        return rgb, a
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0

    # 目标块大于 1 MP 就走降采样路径
    scale = 1.0
    if bw * bh > 1_000_000:
        scale = math.sqrt(bw * bh / 1_000_000.0)

    if scale > 1.0:
        sw = max(8, int(round(bw / scale)))
        sh = max(8, int(round(bh / scale)))
        # 多带一圈上下文（选区外 25%），inpaint 的边界条件靠它
        pad = int(math.ceil(max(sw, sh) * 0.25)) + 4
        rx0, ry0 = max(0, x0 - pad), max(0, y0 - pad)
        rx1 = min(w, x1 + pad)
        ry1 = min(h, y1 + pad)
        sub_rgb = rgb[ry0:ry1, rx0:rx1]
        sub_a = _keep_ch(a[ry0:ry1, rx0:rx1])
        sub_m = mask_u8[ry0:ry1, rx0:rx1]
        interp = cv2.INTER_AREA if scale > 1.0 else cv2.INTER_LINEAR
        small_rgb = cv2.resize(sub_rgb, (sw, sh), interpolation=interp)
        small_a = cv2.resize(sub_a[..., 0], (sw, sh),
                             interpolation=interp)[..., None]
        small_m = cv2.resize(sub_m, (sw, sh), interpolation=cv2.INTER_NEAREST)

        bgr = np.clip(small_rgb[..., ::-1] * 255.0, 0, 255).astype(np.uint8)
        done = cv2.inpaint(bgr, small_m, 3.0, cv2.INPAINT_TELEA)
        small_rgb = done[..., ::-1].astype(np.float32) / 255.0
        small_a = cv2.inpaint((small_a[..., 0] * 255).astype(np.uint8),
                              small_m, 3.0, cv2.INPAINT_TELEA
                              ).astype(np.float32)[..., None] / 255.0

        # 升采样回原分辨率，再按权重混（只用选区内的部分）
        up_rgb = cv2.resize(small_rgb, (rx1 - rx0, ry1 - ry0),
                            interpolation=cv2.INTER_LINEAR)
        up_a = cv2.resize(small_a[..., 0], (rx1 - rx0, ry1 - ry0),
                          interpolation=cv2.INTER_LINEAR)[..., None]
        wsel = (sub_m.astype(np.float32) / 255.0)[..., None]
        rgb = rgb.copy()
        a = a.copy()
        rgb[ry0:ry1, rx0:rx1] = rgb[ry0:ry1, rx0:rx1] * (1 - wsel) \
            + up_rgb * wsel
        a[ry0:ry1, rx0:rx1] = a[ry0:ry1, rx0:rx1] * (1 - wsel) \
            + up_a * wsel
        return rgb, a

    bgr = np.clip(rgb[..., ::-1] * 255.0, 0, 255).astype(np.uint8)
    done = cv2.inpaint(bgr, mask_u8, 3.0, cv2.INPAINT_TELEA)
    out_rgb = done[..., ::-1].astype(np.float32) / 255.0
    # alpha 同样走 uint8：**cv2.inpaint 即使收float32 也是按 0~255 的量纲
    # 处理的**（传 1.0 回来是 1，除 255 才是 0.0039 —— 看着像"全透明"）
    a8 = cv2.inpaint(np.clip(_keep_ch(a)[..., 0] * 255.0, 0, 255).astype(np.uint8),
                     mask_u8, 3.0, cv2.INPAINT_TELEA)
    out_a = a8.astype(np.float32)[..., None] / 255.0
    return out_rgb, out_a


def _fill_mirror(rgb, a, mask_u8, axis="水平"):
    """镜像：以选区包围盒的中心对称轴取像素。

    水平 = 左右翻（沿竖直中轴），垂直 = 上下翻（沿水平中轴）。
    翻过来的来源落在选区外（没被选中的部分）时用原像素；
    仍然落在选区内（选区比镜像还大）时退化成邻近。
    """
    h, w = mask_u8.shape[:2]
    m = mask_u8 >= 128
    box = _bbox(m.astype(np.float32))
    if box is None:
        return rgb, a
    x0, y0, x1, y1 = box
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5

    # 全幅的镜像映射：每个目标像素对应的源坐标
    yy, xx = np.mgrid[0:h, 0:w]
    sx = np.rint(2 * cx - 1 - xx).astype(np.int32) if axis == "水平" \
        else xx.astype(np.int32)
    sy = np.rint(2 * cy - 1 - yy).astype(np.int32) if axis == "垂直" \
        else yy.astype(np.int32)
    valid = (sx >= 0) & (sx < w) & (sy >= 0) & (sy < h)
    sx_c = np.clip(sx, 0, w - 1)
    sy_c = np.clip(sy, 0, h - 1)
    out_rgb = rgb[sy_c, sx_c]
    out_a = a[sy_c, sx_c]
    # 来源出画布 -> 该处退回原像素，后面交给 inpaint 兜底
    out_rgb = np.where(valid[..., None], out_rgb, rgb)
    out_a = np.where(valid[..., None], out_a, a)

    got = (m & valid)
    if got.all() or not m.any():
        return out_rgb, out_a
    # 还有补不上的（来源也在选区里 / 出画布）-> 邻近兜底
    fill_mask = np.where(got, np.uint8(0), mask_u8).astype(np.uint8)
    return _fill_nearest(out_rgb, out_a, fill_mask)


def _lo3(block):
    """(patch,patch,3) -> 模糊后的 (patch*patch,3) 低频结构。

    匹配时除了逐像素 SAD，再比一份模糊过的 —— 纯高频会被噪点主导，
    挑出来的块跟周围脉络接不上。
    """
    b = cv2.GaussianBlur(block.reshape(-1, 1, 3), (0, 0), 1.2)
    return b.reshape(-1, 3)


def _pick_blocks(work_m, half, patch, wh, ww, step, limit):
    """挑一批待补块的位置 [(y, x)]，y/x 是**块心**。

    **相邻块必须重叠**（步长 ≤ patch），否则块与块之间会留下没盖住的缝
    ——那些像素原封不动，看着像"填充失败了一半"。所以步长的上限是
    `patch`，不是 `limit`。

    做法：先用较粗的步长试，太密就逐步放粗，但**粗到 patch 就停**；
    仍然超`limit` 才在两轴抽稀（这时缝已经不可避免，靠后面的邻近兜底补）。
    """
    s = max(1, min(step, patch))
    while True:
        ys = list(range(half, max(half + 1, wh - half), s))
        xs = list(range(half, max(half + 1, ww - half), s))
        total = len(ys) * len(xs)
        if total <= limit or s >= patch or s <= 1:
            break
        s = min(patch, s + max(1, s))
    if len(ys) * len(xs) > limit:
        # 到patch 步长还超（洞特别大）：两轴一起抽稀，别只抽一轴，
        # 否则二维密度不均，填出来是一片一片的方块
        k = max(1, int(math.sqrt(len(ys) * len(xs) / float(limit))))
        ys = ys[::k]
        xs = xs[::k]
    out = [(y, x) for y in ys for x in xs if work_m[y, x]]
    # 抽稀后一个都没选中的极端情况：从洞里直接取像素当块心
    if not out:
        hole = np.argwhere(work_m[half:wh - half, half:ww - half])
        for y, x in hole[:limit]:
            out.append((int(y) + half, int(x) + half))
    return out


def _fill_texture(rgb, a, mask_u8, patch=9, search_radius=48):
    """纹理合成：按块从周围找最相似的邻块贴进去。

    实现是 PatchMatch 的简化版 + **coherence 填充**：
    1. 把待补区域按 `patch` 大小切块；
    2. 每块算它和周围候选块的距离（SAD），取最小的那个；
    3. 候选块必须**完全不含待补像素**，否则就是把洞复制到洞里；
    4. 一轮贴完重新算一遍，直到整块填满。

    为了能在 1~2 秒内返回，选区超过一定面积就**只在降采样图上匹配**，
    再把结果升采样贴回去（跟邻近一样的思路）。
    """
    h, w = mask_u8.shape[:2]
    m = mask_u8 >= 128
    if not m.any():
        return rgb, a
    box = _bbox(m.astype(np.float32))
    x0, y0, x1, y1 = box
    pad = int(math.ceil(patch * 2))
    rx0, ry0 = max(0, x0 - search_radius - pad), max(0, y0 - search_radius - pad)
    rx1, ry1 = min(w, x1 + search_radius + pad), min(h, y1 + search_radius + pad)
    if rx1 - rx0 < patch + 1 or ry1 - ry0 < patch + 1:
        return _fill_nearest(rgb, a, mask_u8)

    patch = max(3, int(patch) | 1)          # 强制奇数，中心才是块心
    step = patch                              # 块网格步长（不重叠）

    # 降采样：选区太大时在小的图上匹配
    scale = 1.0
    bw, bh = x1 - x0, y1 - y0
    if bw * bh > 400_000:
        scale = math.sqrt(bw * bh / 400_000.0)
    if scale > 1.0:
        sw, sh = rx1 - rx0, ry1 - ry0
        tw = max(patch + 2, int(round(sw / scale)))
        th = max(patch + 2, int(round(sh / scale)))
        work_rgb = cv2.resize(rgb[ry0:ry1, rx0:rx1], (tw, th),
                              interpolation=cv2.INTER_AREA)
        work_m = cv2.resize(m[ry0:ry1, rx0:rx1].astype(np.uint8), (tw, th),
                            interpolation=cv2.INTER_NEAREST) >= 128
        work_a = cv2.resize(_keep_ch(a[ry0:ry1, rx0:rx1])[..., 0], (tw, th),
                             interpolation=cv2.INTER_AREA)[..., None]
    else:
        work_rgb = rgb[ry0:ry1, rx0:rx1]
        work_a = a[ry0:ry1, rx0:rx1]
        work_m = m[ry0:ry1, rx0:rx1]

    wh, ww = work_m.shape[:2]
    small = min(patch, max(3, min(wh, ww) - 1))
    if small % 2 == 0:
        small -= 1
    patch = max(3, small)

    src_ok = ~work_m
    # 积分图：一次查询「这块里有没有待补像素」
    ii = cv2.integral(src_ok.astype(np.uint8))

    def _all_known(y, x):
        y0b, x0b = max(0, y), max(0, x)
        y1b, x1b = min(wh, y + patch), min(ww, x + patch)
        if y1b - y0b < 1 or x1b - x0b < 1:
            return False
        return int(ii[y1b, x1b] - ii[y0b, x1b]
                   - ii[y1b, x0b] + ii[y0b, x0b]) == (y1b - y0b) * (x1b - x0b)

    # 候选源位置：网格采样。**数量要有上限** —— 每个待补块都要和
    # 所有候选算一次 SAD（ncand × patch² × 3），候选上千就是秒级。
    #
    # 上限按 ROI 面积给（每 ~2000 px 一个候选），而不是固定 600：
    # 固定值碰上大 ROI 时 cstep 会涨到几十，候选变得很粗，
    # 能挑到的纹理块就少了，填出来一片平滑（实测采样范围一放大就出现）。
    MAX_CAND = int(np.clip((wh * ww) // 2000, 256, 3000))
    cstep = max(1, patch // 3)
    while True:
        cand = [(y, x)
                for y in range(0, wh - patch + 1, cstep)
                for x in range(0, ww - patch + 1, cstep)
                if _all_known(y, x)]
        if len(cand) <= MAX_CAND or cstep >= min(wh, ww):
            break
        # **步长只加1**（原来翻倍）：翻倍会让 cstep 几步就冲到几十，
        # 候选随之变粗，纹理匹配质量断崖式下降
        cstep += 1
    if not cand:
        return _fill_nearest(rgb, a, mask_u8)

    half = patch // 2
    # 主色用的候选（不含 alpha，alpha 由周围插值给）
    cand_rgb = np.stack([work_rgb[y:y + patch, x:x + patch].reshape(-1, 3)
                         for y, x in cand])
    cand_lo = np.stack([_lo3(work_rgb[y:y + patch, x:x + patch])
                        for y, x in cand])
    work_out = work_rgb.copy()

    # 待补块列表：块心落在待补区里的网格点。
    #
    # 块数上限要**跟着洞的面积走**：固定 400 对一个小洞绰绰有余，
    # 对一个 600x600 的洞却只能盖住零头，剩下的全靠兜底 ——
    # 结果就是「纹理合成」在中间留一大块平滑区。
    # 这里按「块边长刚好铺满洞」估算需要多少块，再放宽一倍。
    hole_ys, hole_xs = np.nonzero(work_m)
    if len(hole_ys) == 0:
        return rgb, a
    need = int(len(hole_ys) / float(patch * patch)) + 8
    MAX_BLOCK = int(np.clip(need * 2, 64, 2500))
    block_step = max(1, patch // 2)
    todo = _pick_blocks(work_m, half, patch, wh, ww, block_step, MAX_BLOCK)

    # 由外向内填（coherence）：每轮挑**离已知区最近**的块先贴，
    # 纹理才能从边界往里长。
    # depth = distanceTransform(filled) 在待补区里给的是「到已知区的距离」，
    # 所以要**升序**（小的先填）。写成降序就是从最深的洞开始填，
    # 它会把旁边的块一起盖成"已填"，靠边那些块随即被跳过，
    # 最后留下一块三角形没补上。
    filled = work_m.copy()
    depth = cv2.distanceTransform(filled.astype(np.uint8), cv2.DIST_L2, 3)
    rounds = 0
    while todo and rounds < 96:
        rounds += 1
        todo.sort(key=lambda t: depth[t[0], t[1]])
        for (y, x) in todo:
            blk = filled[y - half:y + half + 1, x - half:x + half + 1]
            if not blk.any():
                continue                       # 已被别的块盖住
            tgt = work_out[y - half:y + half + 1,
                           x - half:x + half + 1].reshape(-1, 3)
            # **目标块取自work_out（已合成的结果），不是 work_rgb**。
            # 这是纹理能"长进去"的关键：每一轮都用上一轮的结果当目标，
            # 纹理就从边界一路往里传。一直拿原始像素当目标的话，
            # 每个块都在跟"那个要被抹掉的暗块"比，
            # 谁都不像 -> argmin 退化成随便挑，填出来一片平坦。
            tgt_lo = _lo3(tgt.reshape(patch, patch, 3))
            # (ncand, patch^2, 3) -> 每个候选的总SAD -> (ncand,)
            # **不能直接对二维矩阵 argmin**：那取的是扁平下标，会越界
            d = np.abs(cand_rgb - tgt[None, :]).sum(axis=2).sum(axis=1)
            # 再加一项低频（结构）距离。纯高频 SAD 会被噪点主导，
            # 挑出来的块跟周围"脉络"接不上；压一层模糊后再比，
            # 相当于优先选走向一致的块。
            d = d + np.abs(cand_lo - tgt_lo[None, :]).sum(axis=2).sum(axis=1) \
                * patch
            sy, sx = cand[int(np.argmin(d))]
            work_out[y - half:y + half + 1, x - half:x + half + 1] = \
                work_rgb[sy:sy + patch, sx:sx + patch]
            filled[y - half:y + half + 1, x - half:x + half + 1] = False
        # 剩下的块（没被任何一轮盖住的）留在 todo 里继续试，
        # 但每轮重算深度，让「贴着已知区」的先走
        todo = [(y, x) for (y, x) in todo
                if filled[y - half:y + half + 1, x - half:x + half + 1].any()]
        depth = cv2.distanceTransform(filled.astype(np.uint8), cv2.DIST_L2, 3)

    # alpha：待补区的 alpha 按"到已知区的距离"做扩散（inpaint 只为省事）
    out_a = work_a.copy()
    if work_m.any():
        a8 = cv2.inpaint((work_a[..., 0] * 255).astype(np.uint8),
                         (work_m.astype(np.uint8)) * 255, 3.0,
                         cv2.INPAINT_TELEA)
        out_a[..., 0] = a8.astype(np.float32) / 255.0

    # 块数有上限（MAX_BLOCK），大洞一定填不满。
    # 剩下的**必须交给邻近兜底**，否则会出现"只填了几个补丁块、
    # 其余还是原来的内容" —— 块数上限是为了速度，不是为了允许留空。
    if filled.any():
        leftover = (filled.astype(np.uint8) * 255)
        # **源必须是 work_out（已经贴过块的），不是 work_rgb**。
        # 传 work_rgb 的话，块与块之间的缝是夹在「原始暗块」中间的，
        # inpaint 拿到的边界条件全是暗色 -> 缝被填成暗色，
        # 整片看着像没填（实测残留能到 70%）。
        fb_rgb, fb_a = _fill_nearest(work_out, out_a, leftover)
        take = filled[..., None]
        work_out = np.where(take, fb_rgb, work_out)
        out_a = np.where(take, fb_a, out_a)
        # **不要动 work_m**。它在这之后还要当写回遮罩用，而 `filled`
        # 已经被块填成了全False —— 拿它写回等于什么都不写，
        # 纹理合成就变成了"算完但什么都不显示"。

    if scale > 1.0:
        up_rgb = cv2.resize(work_out, (rx1 - rx0, ry1 - ry0),
                            interpolation=cv2.INTER_LINEAR)
        up_a = cv2.resize(out_a[..., 0], (rx1 - rx0, ry1 - ry0),
                          interpolation=cv2.INTER_LINEAR)[..., None]
        res_rgb = rgb.copy()
        res_a = a.copy()
        res_rgb[ry0:ry1, rx0:rx1] = up_rgb
        res_a[ry0:ry1, rx0:rx1] = up_a
    else:
        # **work_* 是 ROI 裁剪（rx0..rx1），rgb/a 是全幅** —— 尺寸不一样，
        # 直接 np.where 会广播失败。必须先写回 ROI 再拼。
        roi_rgb = rgb[ry0:ry1, rx0:rx1].copy()
        roi_a = a[ry0:ry1, rx0:rx1].copy()
        roi_rgb[work_m] = work_out[work_m]
        roi_a[work_m] = out_a[work_m]
        res_rgb = rgb.copy()
        res_a = a.copy()
        res_rgb[ry0:ry1, rx0:rx1] = roi_rgb
        res_a[ry0:ry1, rx0:rx1] = roi_a
    return res_rgb, res_a


# ---------- 对外主入口 ----------

def fill_content_aware(img, sel, mode="邻近", feather=2.0,
                       patch=TEXTURE_PATCH_DEFAULT, axis="水平",
                       search=48):
    """内容感知填充。

    img: (h,w,4) uint8，**不会被修改**
    sel: (h,w) float32 0~1 的选区遮罩，与 img 同尺寸
    mode: 邻近 / 镜像 / 纹理合成
    feather: 选区边缘羽化（像素），0 = 硬边。默认 2，
             太大（> 选区尺寸）会整片变成半透明，交给调用方限幅。
    patch: 纹理合成时的块边长
    axis: 镜像的方向，水平 / 垂直
    search: 采样范围（半径，像素）。邻近用不上；纹理的候选源只在
            选区外这么远的地方找 —— 大了更不容易"贴出重复的块"，
            也更慢。

    -> 新的 (h,w,4) uint8
    """
    if img is None or img.ndim != 3 or img.shape[2] < 4:
        return img
    if sel is None:
        return img.copy()
    mask = np.clip(np.asarray(sel, np.float32), 0.0, 1.0)
    if mask.ndim == 1:                # 传进来一维 = 没有选区
        return img.copy()
    if mask.shape[:2] != img.shape[:2]:
        # 尺寸对不上就缩放。**只缩放，不按比例再乘一次权重** ——
        # 那样等于把选区整体调暗，是错的
        h, w = img.shape[:2]
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    if not (mask >= 0.5).any():
        return img.copy()

    mode = mode if mode in FILL_MODES else FILL_MODES[0]
    # 羽化半径不许超过选区自身大小，否则整块都成了半透明
    box = _bbox(mask)
    if box is not None:
        x0, y0, x1, y1 = box
        lim = max(1.0, min(x1 - x0, y1 - y0) * 0.5)
        feather = float(np.clip(feather, 0.0, lim))
    # 二值 mask 给 inpaint / 块匹配用；羽化权重单独留一份给混合
    mask_u8 = _mask_to_u8(mask)
    soft = mask.copy()
    if feather > 0.5:
        soft = cv2.GaussianBlur(soft, (0, 0), feather / 2.0)
        # 羽化会把选区外的像素也带进来一点点，按原选区裁回去
        soft = np.clip(soft, 0.0, 1.0) * (mask > 0.0)

    rgb, a = _premul(img)
    if mode == "邻近":
        out_rgb, out_a = _fill_nearest(rgb, a, mask_u8)
    elif mode == "镜像":
        out_rgb, out_a = _fill_mirror(rgb, a, mask_u8, axis=axis)
    else:
        out_rgb, out_a = _fill_texture(rgb, a, mask_u8, patch=patch,
                                       search_radius=max(8, int(search)))

    ww = soft[..., None]
    return _unpremul(rgb * (1 - ww) + out_rgb * ww, a * (1 - ww) + out_a * ww)


def extend_canvas_content_aware(img, sel, left=0, top=0, right=0, bottom=0,
                                mode="邻近", patch=TEXTURE_PATCH_DEFAULT,
                                axis="水平", search=48):
    """画布外面那一圈用内容感知的方式长出来（PS：扩展画布外的空白）。

    做法：把原图按 (left, top) 贴进新画布，原位置之外的部分算「待补」，
    再交给同一套填充逻辑补齐。返回的图一定和目标尺寸一样大，
    **原图所占区域逐位不变**。

    left/top/right/bottom 都是**扩出来的像素数**（≥0）。
    """
    if img is None:
        return img
    h, w = img.shape[:2]
    nw = w + int(left) + int(right)
    nh = h + int(top) + int(bottom)
    if nw <= 0 or nh <= 0:
        return img
    canvas = None
    l, t = int(left), int(top)
    # 贴图要裁在画布内（扩出来的部分留给填充）
    sx0, sy0 = max(0, -l), max(0, -t)
    dx0, dy0 = max(0, l), max(0, t)
    cw = min(w - sx0, nw - dx0)
    chh = min(h - sy0, nh - dy0)

    if cw > 0 and chh > 0:
        core = img[sy0:sy0 + chh, sx0:sx0 + cw]
        # **必须先用 BORDER_REPLICATE 把边缘复制出去当初值**。
        # 只贴核心再inpaint 是不行的：新区域 alpha 是 0，而本模块是
        # 预乘处理的（§3.2），0 alpha 会把颜色一起乘成 0 ——
        # inpaint 拿到的边界条件全是黑的，填出来就是一条死黑边。
        canvas = cv2.copyMakeBorder(core, dy0, nh - dy0 - core.shape[0],
                                    dx0, nw - dx0 - core.shape[1],
                                    cv2.BORDER_REPLICATE)
    else:
        # 原图整个被移出画布（理论上不会发生），退化成纯透明底
        canvas = np.zeros((nh, nw, 4), np.uint8)

    # 待补遮罩：不在原图范围内的部分
    hole = np.ones((nh, nw), np.float32)
    hole[dy0:dy0 + chh, dx0:dx0 + cw] = 0.0
    if sel is not None:
        s = np.clip(np.asarray(sel, np.float32), 0.0, 1.0)
        s = cv2.resize(s, (nw, nh), interpolation=cv2.INTER_LINEAR)
        hole = np.maximum(hole, s)
    if not (hole >= 0.5).any():
        return canvas

    # 边缘 2px 用 0~1 渐变，让填充结果和原图接得上。
    # **只在原图之外的那圈做羽化** —— 羽化会往外扩 2~3px，渗回原图区域
    # 就等于把原图像素也blend 掉，扩展画布必须原图**逐位不变**。
    keep = hole.copy()                # 1 = 原图像素区
    hole_s = cv2.GaussianBlur(hole, (0, 0), 1.2)
    hole_s = np.clip(hole_s * 1.6, 0.0, 1.0) * (1.0 - keep)
    out = fill_content_aware(canvas, hole_s, mode=mode, feather=0.0,
                             patch=patch, axis=axis, search=search)
    # 双保险：原图区域直接写回，任何混合误差都不留
    if cw > 0 and chh > 0:
        out[dy0:dy0 + chh, dx0:dx0 + cw] = core
    return out