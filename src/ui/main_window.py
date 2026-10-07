# -*- coding: utf-8 -*-
"""主窗口：菜单、面板装配、文档操作。"""

from __future__ import annotations

import os
import time
import uuid

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QDockWidget,
                               QFileDialog, QFormLayout, QInputDialog, QLabel,
                               QMainWindow, QMessageBox, QScrollArea, QSpinBox,
                               QToolBar)

from ..core.adjust import ADJUSTMENTS, ADJUST_ORDER
from ..core.brush import clear_rgba, fill_rgba
from ..core.document import (Document, make_adjustment_layer, make_group,
                             make_image_layer, make_text_layer)
from ..core.content_aware import (extend_canvas_content_aware,
                                  fill_content_aware)
from ..core.filters import (FILTERS, FILTER_ORDER, apply_filter_array,
                            preview_can_downscale, preview_filter_scale,
                            scale_filter_params)
from ..core.history import History
from ..core.layer import LAYER_IMAGE
from ..core.paint import _gray
from ..core.project_io import (PROJECT_EXT, export_flat, imread_rgba,
                               load_project, save_project)
from ..core import guides as _guides
from ..core import quick_mask
from ..core import retouch
from ..core import channels as _ch
from ..core.psd_import import (PSD_EXT, STATS, is_available as psd_available,
                               load_psd)
from ..core.render import (clear_proxy_cache, has_dissolve, max_effect_padding,
                           render_document, render_from_snapshot,
                           render_proxy, render_region_tiled, render_tiled)
from ..core.selection import (ADD, INTERSECT, REPLACE, SUBTRACT, Selection)
from ..core.smart import (content_instances, convert_to_smart, new_filter,
                          new_smart_instance, prune_contents,
                          rasterize_smart, refresh_sizes, sync_smart)
from ..core.text import sync_text_image
from .canvas_view import CanvasView
from .content_aware_dialog import ContentAwareDialog
from .filter_dialog import FilterDialog
from .channels_panel import ROLE_MASK as _CH_ROLE_MASK, ChannelsPanel
from .inspector import Inspector
from .layers_panel import LayersPanel
from .tool_options import ToolOptions

# 工具 id -> (按钮文字, 快捷键, 状态栏提示)
TOOLS = [
    ("move", "移动", "V", "移动 / 缩放 / 旋转图层"),
    ("rect", "矩形", "M", "矩形选框（Shift 正方形，在选区内拖动可移动选区）"),
    ("ellipse", "椭圆", "Shift+M", "椭圆选框（Shift 正圆）"),
    ("lasso", "套索", "L", "自由套索：按住拖动勾轮廓"),
    ("polygon", "多边", "Shift+L", "多边形套索：逐点点击，双击或点回起点闭合"),
    ("magic", "魔棒", "W", "按颜色选择（Ctrl 点击 = 全图不连续）"),
    ("brush", "画笔", "B", "画笔（可画在像素或蒙版上）"),
    ("eraser", "橡皮", "E", "橡皮擦：擦成透明"),
    ("fill", "填充", "G", "油漆桶：用前景色填充"),
    ("gradient", "渐变", "Shift+G", "拖出一条线画渐变（5 种样式，前景色 → 背景色）"),
    ("blurtool", "模糊", "Shift+B", "局部模糊：按住拖动，笔刷盖住的地方被高斯模糊"),
    ("picker", "吸管", "Shift+I", "吸管：点一下取色到前景色（Alt 点取背景色）"),
    ("clone", "克隆", "Shift+C", "克隆图章：按住 Alt 采样，松开拖动盖章"),
    ("heal", "修复", "Shift+H", "污点修复：点/拖过瑕点，用周围纹理补上"),
    ("shape", "形状", "Shift+U", "形状工具：拖出矩形 / 椭圆 / 直线"),
    ("text", "文字", "T", "点击画布创建文字图层（之后在属性面板里改）"),
    ("hand", "抓手", "H", "平移画布（按住空格可临时切换）"),
]

STYLE = """
QWidget { background-color:#2b2b2e; color:#ddd;
          font-family:"Microsoft YaHei UI","Segoe UI",sans-serif; font-size:12px; }
QMainWindow::separator { background:#3a3a3d; width:2px; height:2px; }
QMenuBar { background:#323235; border-bottom:1px solid #3a3a3d; }
QMenuBar::item:selected { background:#2d6db5; }
QMenu { background:#323235; border:1px solid #4a4a4e; }
QMenu::item:selected { background:#2d6db5; }
QStatusBar { background:#323235; }
QToolBar { background:#323235; border:none; spacing:2px; padding:2px; }
QToolButton { color:#ddd; padding:3px 6px; }
QGroupBox { border:1px solid #4a4a4e; border-radius:4px; margin-top:8px;
            padding-top:6px; }
QGroupBox::title { subcontrol-origin:margin; left:8px; padding:0 3px; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background:#222224; border:1px solid #4a4a4e; border-radius:3px;
    padding:2px 4px; }
QSlider::groove:horizontal { background:#222224; height:4px; border-radius:2px; }
QSlider::handle:horizontal { background:#8a8a8e; width:12px; margin:-4px 0;
                             border-radius:6px; }
"""

# ---- 渲染调度（见 §5.21）----
PROXY_BUDGET_MS = 70.0     # 全量渲染超过这个毫秒数，交互期就改走代理
PROXY_TARGET_MS = 25.0     # 代理渲染想压到的耗时
PROXY_MIN_SCALE = 0.15     # 代理最小缩放（再小就糊得看不出在拖什么了）
PROXY_IDLE_MS = 350        # 交互停手多久之后补一张全分辨率
PARTIAL_AREA_MAX = 0.6     # 脏区超过画布这个比例，直接整幅重算更划算
TILE_MIN_AREA = 6_000_000  # 画布超过 6 MP 才分块：小画布省下的内存抵不上瓦片开销
DIRTY_MAX_RECTS = 8        # 脏区最多记这么多块，超了就折叠成并集（见 mark_dirty）


def _union(rects):
    """一组矩形的并集包围盒。"""
    return (min(r[0] for r in rects), min(r[1] for r in rects),
            max(r[2] for r in rects), max(r[3] for r in rects))
TILE_EDGE = 1024           # 瓦片边长（画布像素）。实测 24 MP：1920 MB -> 357 MB，+30% 耗时

IMG_FILTER = ("图片文件 (*.png *.jpg *.jpeg *.bmp *.gif *.webp *.tif *.tiff "
              "*.heic *.heif *.svg);;所有文件 (*.*)")
PROJ_FILTER = "Compositor 工程 (*%s);;所有文件 (*.*)" % PROJECT_EXT
PSD_FILTER = "Photoshop 文档 (*%s);;所有文件 (*.*)" % PSD_EXT


class NewDocDialog(QDialog):
    def __init__(self, parent, w=1920, h=1080):
        super().__init__(parent)
        self.setWindowTitle("新建工程")
        form = QFormLayout(self)
        self.w = QSpinBox()
        self.h = QSpinBox()
        for s in (self.w, self.h):
            s.setRange(1, 30000)
            s.setSingleStep(10)
        self.w.setValue(w)
        self.h.setValue(h)
        form.addRow("宽度 (px)", self.w)
        form.addRow("高度 (px)", self.h)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Compositor for Windows")
        self.resize(1400, 900)
        self.setStyleSheet(STYLE)

        self.doc = None
        self.history = History()
        self.selected_id = None
        self._render_pending = False
        self._filter_ctx = None
        self._so_editors = []       # 打开的智能对象编辑窗口（要留住引用防被回收）
        # 渲染缓存：_last_arr 是最近一次**全分辨率**整幅合成结果，
        # _arr_valid 为 False 时它已经过期（比如刚渲过代理图），必须整幅重算。
        self._last_arr = None
        self._arr_valid = False
        self._full_dirty = True       # 整幅都要重算
        self._dirty = None            # 否则：待重算的区域，单个矩形或矩形列表
        self._last_pass = ""          # 上次走的哪条路：full / partial / proxy
        self._render_ms = 0.0         # 上次全量渲染耗时，用来决定要不要走代理
        self._adjust_ms = 0.0         # 上次快路径的耗时（决定它够不够代替代理）
        self._interactive = False     # 正在拖手柄 / 拖滑块
        # 拖调整层滑块的快路径：_adj_snap 存着"这个调整层下方"的合成结果，
        # 它不随调整参数变化，所以改参数时只需重算调整层本身（见 §5.23）。
        # 任何**别的**改动都会让它作废 —— 见 request_render()。
        self._adj_layer = None
        self._adj_snap = None
        self._idle_timer = QTimer(self)
        self._idle_timer.setSingleShot(True)
        self._idle_timer.setInterval(PROXY_IDLE_MS)
        self._idle_timer.timeout.connect(self._end_interactive)
        # 拖滑块会连发几十次改动，延迟一下再落撤销点，避免把历史栈冲爆
        self._commit_timer = QTimer(self)
        self._commit_timer.setSingleShot(True)
        self._commit_timer.setInterval(500)
        self._commit_timer.timeout.connect(self._delayed_commit)
        self._commit_label = ""

        self.view = CanvasView(self)
        self.setCentralWidget(self.view)

        self.panel = LayersPanel(self)
        dock = QDockWidget("图层", self)
        dock.setWidget(self.panel)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        self.panel_dock = dock

        self.panel.setMinimumWidth(250)

        self.inspector = Inspector(self)
        scroll = QScrollArea()
        scroll.setWidget(self.inspector)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.inspector.setMinimumWidth(248)
        self.inspector_scroll = scroll
        dock2 = QDockWidget("属性", self)
        dock2.setWidget(scroll)
        dock2.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        dock2.setMinimumWidth(286)
        self.addDockWidget(Qt.RightDockWidgetArea, dock2)
        self.inspector_dock = dock2
        # 属性面板里要放得下曲线编辑器 + 直方图，所以下半区留宽一点
        self.resizeDocks([dock, dock2], [420, 500], Qt.Vertical)

        # 通道面板：与图层面板、属性面板同一个停靠区（Photoshop 也是三栏）
        self.channels = ChannelsPanel(self)
        self.channels.loadRequested.connect(self.load_channel_selection)
        self.channels.saveRequested.connect(self.save_channel)
        self.channels.deleteRequested.connect(self.delete_channel)
        self.channels.mergeRequested.connect(self.merge_channel)
        self.channels.list.itemChanged.connect(self._on_channel_renamed)
        dock3 = QDockWidget("通道", self)
        dock3.setWidget(self.channels)
        dock3.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.RightDockWidgetArea, dock3)
        self.channels_dock = dock3

        self.zoom_label = QLabel("100%")
        self.status = self.statusBar()
        self.status.addPermanentWidget(self.zoom_label)
        self.view.transformChanged.connect(self._on_zoom_changed)
        self.view.statusMessage.connect(self.status.showMessage)

        self._make_menus()
        self._make_toolbox()
        # 工具选项条单独占一行，放在主工具栏下面
        self.opts = ToolOptions(self)
        self.addToolBarBreak()
        self.addToolBar(self.opts)
        self.new_document(1600, 1000)
        self.set_tool("move")

    # ---------- 菜单 ----------

    def _act(self, text, shortcut, slot, tip=""):
        a = QAction(text, self)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        a.triggered.connect(slot)
        if tip:
            a.setStatusTip(tip)
        return a

    def _make_menus(self):
        m_file = self.menuBar().addMenu("文件")
        m_file.addAction(self._act("新建工程", "Ctrl+N", self.ask_new_document))
        m_file.addAction(self._act("打开工程…", "Ctrl+O", self.open_project))
        m_file.addAction(self._act("保存工程", "Ctrl+S", self.save_project))
        m_file.addAction(self._act("工程另存为…", "Ctrl+Shift+S", self.save_project_as))
        m_file.addSeparator()
        m_file.addAction(self._act("导入图片为图层…", "Ctrl+I",
                                   self.import_image_as_layer))
        m_file.addAction(self._act("导入 PSD…", "Ctrl+Shift+O", self.import_psd,
                                   "把 Photoshop 文档的图层树导进新工程"))
        m_file.addSeparator()
        m_file.addAction(self._act("导出为 PNG…", "Ctrl+Shift+E",
                                   lambda: self.export("png")))
        m_file.addAction(self._act("导出为 JPEG…", "", lambda: self.export("jpg")))
        m_file.addSeparator()
        m_file.addAction(self._act("退出", "Ctrl+Q", self.close))

        m_edit = self.menuBar().addMenu("编辑")
        m_edit.addAction(self._act("撤销", "Ctrl+Z", self.undo))
        m_edit.addAction(self._act("重做", "Ctrl+Shift+Z", self.redo))

        m_layer = self.menuBar().addMenu("图层")
        m_layer.addAction(self._act("新建图层组", "Ctrl+G", self.group_selection))
        m_layer.addAction(self._act("复制图层", "Ctrl+J", self.duplicate_layer))
        m_layer.addAction(self._act("删除图层", "", self.delete_layer))
        m_layer.addSeparator()
        m_adj = m_layer.addMenu("新建调整图层")
        for key in ADJUST_ORDER:
            nm = ADJUSTMENTS[key].name
            m_adj.addAction(self._act(nm, "",
                                      lambda _c=False, k=key:
                                      self.add_adjustment_layer(k)))
        m_layer.addAction(self._act("创建/释放剪贴蒙版", "Ctrl+Alt+G",
                                    self.toggle_clip,
                                    "让调整层只作用于紧邻的下方图层"))
        m_so = m_layer.addMenu("智能对象")
        m_so.addAction(self._act("转换为智能对象", "Ctrl+Alt+S",
                                 self.convert_selection_to_smart,
                                 "把图层/组连同它的剪贴蒙版装进一段可独立编辑的内容"))
        m_so.addAction(self._act("新建实例（共用内容）", "", self.new_smart_copy,
                                 "复制图层但共用同一份内容 —— 改内容会联动所有实例"))
        m_so.addAction(self._act("编辑内容…", "", self.edit_smart_content,
                                 "在独立窗口里改这段内容，所有实例一起更新"))
        m_so.addAction(self._act("智能滤镜…", "", self.edit_smart_filters,
                                 "参数化的滤镜：随时改参数、调顺序、删掉"))
        m_so.addSeparator()
        m_so.addAction(self._act("栅格化智能对象", "", self.rasterize_smart,
                                 "烧成普通位图图层，内容从此不可再改"))
        m_layer.addAction(self._act("栅格化图层", "", self.rasterize_selected,
                                    "文字图层 / 智能对象转成普通位图图层"))
        m_sty = m_layer.addMenu("图层样式")
        m_sty.addAction(self._act("图层样式…", "", self.edit_layer_style,
                                  "10 种效果 + 全局光，参数化、随时可改"))
        m_sty.addSeparator()
        m_sty.addAction(self._act("复制图层样式", "Ctrl+Alt+C",
                                  self.copy_layer_style,
                                  "把选中图层的整套样式存进样式剪贴板"))
        m_sty.addAction(self._act("粘贴图层样式", "Ctrl+Alt+V",
                                  self.paste_layer_style,
                                  "把剪贴板里的样式整份贴到选中图层"))
        m_sty.addSeparator()
        m_sty.addAction(self._act("存为默认值", "", self.save_style_default,
                                  "之后新建的图层样式都从这份参数出发"))
        m_sty.addAction(self._act("复位默认值", "", self.reset_style_default,
                                  "回到出厂的默认样式参数"))
        m_layer.addSeparator()
        m_layer.addAction(self._act("置于顶层", "Ctrl+Shift+]", self.raise_to_top))
        m_layer.addAction(self._act("置于底层", "Ctrl+Shift+[", self.lower_to_bottom))

        m_text = self.menuBar().addMenu("文字")
        m_text.addAction(self._act("在画布上编辑文字", "Ctrl+T",
                                   self.edit_text_on_canvas,
                                   "直接在画布上打字，边改边看"))
        m_text.addAction(self._act("逐字调整…", "", self.edit_text_chars,
                                   "给单个字加字距 / 抬基线 / 拉宽"))
        m_text.addSeparator()
        m_text.addAction(self._act("栅格化文字", "", self.rasterize_text,
                                   "转成普通位图图层，之后才能用画笔和滤镜"))

        m_filter = self.menuBar().addMenu("滤镜")
        for key in FILTER_ORDER:
            nm = FILTERS[key].name
            m_filter.addAction(self._act(nm, "",
                                         lambda _c=False, k=key:
                                         self.ask_filter(k)))

        m_retool = self.menuBar().addMenu("修饰")
        m_retool.addAction(self._act("渐变…", "", lambda: self.set_tool("gradient"),
                                     "拖一条线画渐变（Shift+G）"))
        m_retool.addAction(self._act("局部模糊", "Shift+B",
                                     lambda: self.set_tool("blurtool"),
                                     "按住拖动，只糊笔刷盖住的地方"))
        m_retool.addSeparator()
        m_retool.addAction(self._act("前景色 ← 吸管", "Shift+I",
                                     lambda: self.set_tool("picker"),
                                     "点一下取色到前景色"))
        m_retool.addAction(self._act(
            "克隆图章", "Shift+C", lambda: self.set_tool("clone"),
            "按住 Alt 采样，松开拖动盖章"))
        m_retool.addAction(self._act(
            "污点修复", "Shift+H", lambda: self.set_tool("heal"),
            "点过瑕点，用周围的纹理补上"))
        m_retool.addAction(self._act(
            "形状…", "Shift+U", lambda: self.set_tool("shape"),
            "拖出矩形 / 椭圆 / 直线"))
        m_retool.addSeparator()
        m_retool.addAction(self._act(
            "前景色 ← 背景色", "X",
            # lambda 包一层：opts 是菜单建好之后才挂上来的
            lambda: self.opts.swap_colors()))

        m_img = self.menuBar().addMenu("图像")
        m_img.addAction(self._act(
            "画布大小…", "Ctrl+Alt+C", self.ask_canvas_size,
            "给画布加边：底色、透明，或者用内容感知的方式把边缘长出去"))
        m_img.addAction(self._act(
            "按内容裁剪", "", self.trim_to_content,
            "把四周全空的区域裁掉"))

        m_view = self.menuBar().addMenu("视图")
        m_view.addAction(self._act("适应窗口", "Ctrl+0", self.view.fit))
        m_view.addAction(self._act("实际像素", "Ctrl+1", self.view.zoom_actual))
        m_view.addAction(self._act("放大", "Ctrl++", lambda: self.view.set_zoom(self.view.zoom * 1.25)))
        m_view.addAction(self._act("缩小", "Ctrl+-", lambda: self.view.set_zoom(self.view.zoom / 1.25)))
        m_view.addSeparator()
        m_guide = self.menuBar().addMenu("向导")
        m_guide.addAction(self._act(
            "标尺", "Ctrl+Shift+R", lambda: self.set_rulers(not self.view.show_rulers),
            "显示 / 隐藏左上两条标尺"))
        m_guide.addAction(self._act(
            "网格", "Ctrl+Shift+'", lambda: self.set_grid(on=not self.doc.guides.grid.visible),
            "显示 / 隐藏布局网格"))
        m_guide.addAction(self._act(
            "吸附", "", lambda: self.set_snap(
                "all", not self.doc.guides.snap_guides),
            "总开关（按住 Ctrl 临时切换）"))
        m_guide.addSeparator()
        for nm, key in (("吸附到文档边界", "doc"), ("吸附到参考线", "guides"),
                        ("吸附到网格", "grid"), ("吸附到图层", "layers")):
            m_guide.addAction(self._act(
                nm, "", (lambda k: lambda: self.set_snap(k, not getattr(
                    self.doc.guides, {"doc": "snap_doc", "guides": "snap_guides",
                                      "grid": "snap_grid",
                                      "layers": "snap_layers"}[k])))(key)))
        m_guide.addSeparator()
        m_guide.addAction(self._act("清除全部参考线", "", self.clear_guides,
                                    "Alt + 点击某条参考线也能单独删"))

        m_view.addAction(self._act("通道面板", "F2", self.toggle_channels_dock,
                                   "显示 / 隐藏通道面板"))
        m_view.addAction(self._act("图层面板", "", self.panel_dock.setVisible,
                                   "显示 / 隐藏图层面板"))
        m_view.addAction(self._act("属性面板", "", self.inspector_dock.setVisible,
                                   "显示 / 隐藏属性面板"))

        # 通道菜单（Photoshop 放在「选择」旁边，这里同理）
        m_ch = self.menuBar().addMenu("通道")
        m_ch.addAction(self._act("新建通道…", "Ctrl+Alt+N",
                                 lambda: self.save_channel("__new__"),
                                 "把当前选区存成一个 Alpha 通道"))
        m_ch.addAction(self._act("载入选区", "", self._load_selected_channel,
                                 "把选中的通道内容作为选区载入"))
        m_ch.addAction(self._act("删除通道", "", self._delete_selected_channel,
                                 "删除选中的附加 Alpha 通道"))
        m_ch.addSeparator()
        m_ch.addAction(self._act("并入合并的 Alpha（替换）", "", self._merge_replace,
                                 "用选中通道替换合并的 Alpha 通道"))
        m_ch.addAction(self._act("并入合并的 Alpha（添加）", "", self._merge_add))
        m_ch.addAction(self._act("并入合并的 Alpha（减去）", "", self._merge_subtract))
        m_ch.addSeparator()
        m_ch.addAction(self._act("重置合并的 Alpha", "", self._reset_composite_alpha,
                                 "恢复成所有图层合成后的原始不透明度"))
        m_ch.addAction(self._act("RGB 全部显示", "", self._show_all_rgb,
                                 "把红 / 绿 / 蓝三路都打开"))

        tb = self.addToolBar("主工具栏")
        for text, slot in (("新建", self.ask_new_document),
                           ("打开", self.open_project),
                           ("保存", self.save_project),
                           ("导入图片", self.import_image_as_layer),
                           ("导入PSD", self.import_psd),
                           ("撤销", self.undo),
                           ("重做", self.redo)):
            tb.addAction(self._act(text, "", slot))
        tb.addSeparator()

        m_sel = self.menuBar().addMenu("选择")
        m_sel.addAction(self._act("全部", "Ctrl+A", self.select_all))
        m_sel.addAction(self._act("取消选择", "Ctrl+D", self.deselect))
        m_sel.addAction(self._act("反选", "Ctrl+Shift+I", self.invert_selection))
        m_sel.addSeparator()
        m_sel.addAction(self._act("羽化…", "", self.ask_feather))
        m_sel.addAction(self._act("扩展…", "", lambda: self.ask_grow("expand")))
        m_sel.addAction(self._act("收缩…", "", lambda: self.ask_grow("contract")))
        m_sel.addAction(self._act("平滑…", "", self.ask_smooth))
        m_sel.addAction(self._act("边界…", "", self.ask_border))
        m_sel.addSeparator()
        m_sel.addAction(self._act("色彩范围…", "", self.ask_color_range))
        m_sel.addAction(self._act("快速蒙版", "Q", self.toggle_quick_mask))
        m_sel.addSeparator()
        m_sel.addAction(self._act("用前景色填充", "Alt+Delete",
                                  lambda: self.fill_with_fg(None, use_fg=True)))
        m_sel.addAction(self._act("用背景色填充", "Ctrl+Delete",
                                  lambda: self.fill_with_fg(None, use_fg=False)))
        m_sel.addAction(self._act(
            "内容感知填充…", "Shift+Ctrl+F", self.ask_content_aware_fill,
            "用选区周围的像素推断出选区里该是什么；可实时预览"))
        m_sel.addAction(self._act("清除选区内容", "Delete", self.delete_in_selection))
        m_sel.addSeparator()
        m_sel.addAction(self._act("存储为蒙版", "", self.mask_from_selection))

    # ---------- 工具箱 ----------

    def _make_toolbox(self):
        box = QToolBar("工具箱")
        box.setMovable(False)
        box.setFloatable(False)
        box.setStyleSheet(
            "QToolBar { background:#323235; spacing:1px; padding:2px; }"
            "QToolButton { color:#ddd; min-width:42px; padding:5px 2px; }"
            "QToolButton:checked { background:#2d6db5; color:#fff; }")
        self.addToolBar(Qt.LeftToolBarArea, box)
        self.tool_group = QActionGroup(self)
        self.tool_group.setExclusive(True)
        for tid, text, key, tip in TOOLS:
            a = QAction(text, self)
            a.setCheckable(True)
            a.setShortcut(QKeySequence(key))
            a.setStatusTip("%s  [%s]" % (tip, key))
            a.triggered.connect(lambda _c=False, t=tid: self.set_tool(t))
            a.setData(tid)
            box.addAction(a)
            self.tool_group.addAction(a)
            if tid == "move":
                self.tool_actions = {}
            self.tool_actions[tid] = a
        self.tool_actions["move"].setChecked(True)

    def set_tool(self, tool):
        self.stamp_float(silent=True)      # 换工具前先把浮动内容落定
        self.set_interactive(False)        # 拖到一半换工具：退出代理，补全分辨率
        self.view.tool = tool
        self.opts.set_tool(tool)
        a = self.tool_actions.get(tool)
        if a is not None and not a.isChecked():
            a.setChecked(True)
        self.view._update_handles()
        self.view.setCursor(Qt.ArrowCursor)
        tips = dict((t[0], t[3]) for t in TOOLS)
        if tool in tips:
            self.status.showMessage(tips[tool], 2500)

    # ---------- 文档 ----------

    def new_document(self, w, h, name="未命名"):
        doc = Document(w, h, name)
        bg = np.zeros((h, w, 4), np.uint8)
        bg[..., :3] = 255
        bg[..., 3] = 255
        layer = make_image_layer("背景", bg, w, h)
        doc.layers.append(layer)
        self.set_document(doc, reset_history=True)
        self.view.fit()

    def ask_new_document(self):
        dlg = NewDocDialog(self, self.doc.width if self.doc else 1600,
                           self.doc.height if self.doc else 1000)
        if dlg.exec():
            self.new_document(dlg.w.value(), dlg.h.value())

    def set_document(self, doc, reset_history=False):
        self.doc = doc
        self.selected_id = None
        self._last_arr = None           # 换了文档，渲染缓存作废
        self._arr_valid = False
        self._dirty = None
        self._render_ms = 0.0
        if reset_history:
            self.history.reset(doc)
        self.panel.rebuild()
        self.inspector.refresh()
        self.request_render()
        self.view.update_selection_overlay()
        self._update_title()

    def _update_title(self):
        name = self.doc.name if self.doc else ""
        path = self.doc.path if self.doc and self.doc.path else "未保存"
        tag = "  [快速蒙版]" if quick_mask.is_on(self.doc) else ""
        self.setWindowTitle("Compositor for Windows — %s  [%s]%s"
                            % (name, path, tag))

    # ---------- 选择 ----------

    def selected_layer(self):
        if self.doc is None or self.selected_id is None:
            return None
        return self.doc.find(self.selected_id)

    def select_layer(self, lid):
        if lid != self.selected_id:
            self.stamp_float(silent=True)  # 换图层前先把浮动内容盖回去
            # 正在画布上编辑文字时换了图层 -> 先把输入框收起来，
            # 否则接着敲的字会被写进新选中的图层（见 §4.3 第 26 条）
            editing = self.view.edit_text_layer()
            if editing is not None and editing.id != lid:
                self.view.end_text_edit()
        self.selected_id = lid
        self.on_select_changed(from_panel=False)

    def on_select_changed(self, from_panel=False):
        if not from_panel:
            self.panel.tree.blockSignals(True)
            target = None
            if self.selected_id is not None:
                for it in self.panel._iter_items():
                    if it.data(0, Qt.UserRole) == self.selected_id:
                        target = it
                        break
            self.panel.tree.setCurrentItem(target)
            self.panel.tree.blockSignals(False)
        self.panel.refresh_header()
        self.inspector.refresh()
        self.view._update_handles()
        # 选中调整层时把属性面板滚到顶部，参数才看得见
        layer = self.selected_layer()
        if layer is not None and layer.is_adjustment:
            self.inspector_scroll.verticalScrollBar().setValue(0)
        self.opts.sync_target_availability(
            bool(layer and layer.mask is not None),
            only_mask=bool(layer and layer.is_adjustment))

    # ---------- 渲染 ----------

    def mark_dirty(self, rect=None):
        """标记需要重算的区域（画布坐标 (x0,y0,x1,y1)）。

        None 表示"整幅都要重算"。只有明确知道改动只影响一小块时才传矩形 ——
        传错了画面就会残留旧像素，宁可保守。

        **脏区是一组矩形，不是一个**：一次拖拽可能在上百处落笔，逐次取并集
        会很快涨成覆盖大半个画布的一个矩形，那些互不相邻的小块就被一起重算了。
        所以这里保留成列表，只在「块数超上限」或「并集已占大半画布」时才
        折叠成一个（见 `_dirty_rects`）。
        """
        if rect is None:
            self._full_dirty = True
            self._dirty = None
            return
        if self._full_dirty:
            return                      # 已经是整幅，不用再并
        x0, y0, x1, y1 = [int(v) for v in rect]
        if x1 <= x0 or y1 <= y0:
            return
        cur = self._dirty
        if cur is None:
            self._dirty = (x0, y0, x1, y1)
            return
        if isinstance(cur, tuple):
            cur = [cur]
        # 和已有块相交或紧挨着就并进去（省掉一条缝，避免多渲一次）
        merged = False
        for i, r in enumerate(cur):
            if x0 <= r[2] and r[0] <= x1 and y0 <= r[3] and r[1] <= y1:
                cur[i] = (min(r[0], x0), min(r[1], y0),
                          max(r[2], x1), max(r[3], y1))
                merged = True
                break
        if not merged:
            cur.append((x0, y0, x1, y1))
        if len(cur) > DIRTY_MAX_RECTS:
            cur = [_union(cur)]
        self._dirty = cur[0] if len(cur) == 1 else cur

    def _dirty_rects(self):
        """当前的脏区，统一成"矩形列表"。空表示什么都不用重算。"""
        d = self._dirty
        if d is None:
            return []
        return [d] if isinstance(d, tuple) else list(d)

    def request_render(self, rect=None, adjust=None):
        """请求一次渲染。rect 非空时只重算那一块（见 mark_dirty）。

        adjust: 传正在改参数的那个调整层 —— 只有"改调整参数"这一种改动
        能复用它下方的合成结果，所以这也是唯一不清快路径缓存的情形。
        """
        if self.doc is None:
            return
        self._adj_layer = adjust
        if adjust is None:
            self._adj_snap = None       # 别处动了 -> "调整层下方"的结果已过期
        elif (self._adj_snap is None or self._adj_snap.get("id") != adjust.id
              or self._adj_snap.get("w") != self.doc.width
              or self._adj_snap.get("h") != self.doc.height):
            self._adj_snap = {"id": adjust.id, "w": self.doc.width,
                              "h": self.doc.height, "arr": None}
        if rect is not None:
            self.mark_dirty(rect)
        else:
            self.mark_dirty(None)
        if self._interactive:
            self._idle_timer.start()    # 停手一会儿就补一张全分辨率
        if not self._render_pending:
            self._render_pending = True
            QTimer.singleShot(0, self._do_render)

    # ---- 交互期代理渲染 ----

    def set_interactive(self, on):
        """进入 / 退出"连续改动"状态（拖手柄、拖滑块）。

        进入后渲染会先走低分辨率代理，松手（或停手 PROXY_IDLE_MS 毫秒）
        自动补一张全分辨率。只有大画布才真的会降级 —— 小图渲得够快就
        一直用全分辨率。
        """
        on = bool(on)
        if on:
            self._idle_timer.start()
        elif self._interactive:
            self._idle_timer.stop()
        if on == self._interactive:
            return
        self._interactive = on
        if not on:
            self.request_render()       # 补一张清晰的

    def begin_interactive(self):
        """没有明确"松手"事件的连续改动（滑块）用这个：靠空闲定时器收尾。"""
        self.set_interactive(True)

    def _end_interactive(self):
        if self._interactive:
            self.set_interactive(False)

    def _proxy_scale(self):
        """按上次全量渲染的耗时挑一个代理缩放。开销大致与 scale² 成正比。"""
        if self._render_ms <= PROXY_BUDGET_MS:
            return 0.0                  # 0 = 不用代理
        s = (PROXY_TARGET_MS / self._render_ms) ** 0.5
        return max(PROXY_MIN_SCALE, min(1.0, s))

    def _adj_snap_ok(self, layer):
        """快路径的快照能不能用。

        顶层调整层看 `arr`；隔离组里的调整层要 `arr`（组外画布）与
        `group`（组内缓冲）两份都在，缺一份就退回整幅。
        """
        s = self._adj_snap
        if s is None or layer is None or self.doc is None:
            return False
        if s.get("id") != layer.id:
            return False
        arr = s.get("arr")
        if arr is None:
            return False
        if arr.shape[0] != self.doc.height or arr.shape[1] != self.doc.width:
            return False
        if self.doc.find(layer.id) is not None:
            return True               # 顶层
        grp = s.get("group")
        if not isinstance(grp, dict) or grp.get("buf") is None:
            return False
        gb = grp["buf"]
        return gb.shape[0] == self.doc.height and gb.shape[1] == self.doc.width

    def _use_tiles(self):
        """这次全量渲染要不要分块。

        只有大画布才值得：小画布分块的收益（省峰值内存）抵不上开销
        （瓦片重叠的重复计算 + 每块的固定启动成本）。见 §5.24。
        """
        doc = self.doc
        return doc.width * doc.height >= TILE_MIN_AREA

    def _want_full(self):
        """这次是不是必须整幅重算。"""
        doc = self.doc
        arr = self._last_arr
        if arr is None or not self._arr_valid:
            return True
        if arr.shape[0] != doc.height or arr.shape[1] != doc.width:
            return True
        if self._full_dirty or self._dirty is None:
            return True
        if has_dissolve(doc):
            return True                 # Dissolve 的噪声按缓冲尺寸生成，局部渲会错位
        # 脏区可能是一组矩形（见 mark_dirty）：逐块判断，任一块超阈值就整幅
        for x0, y0, x1, y1 in self._dirty_rects():
            area = float(max(0, x1 - x0) * max(0, y1 - y0))
            whole = float(max(1, doc.width * doc.height))
            if area > whole * PARTIAL_AREA_MAX:
                return True
        return False

    def _show(self, arr, scale=None):
        """把渲染结果按通道可见性处理后送进画布。

        **通道可见性只在这里生效** —— `_last_arr` 存的始终是**未经通道处理**
        的合成结果。这样局部重渲染往 `_last_arr` 的脏区里补的那一块，
        和周围的原始数据是同一套坐标 / 取值，不用额外考虑"当时哪些通道开着"，
        也不会和 Photoshop 的行为对不上（PS 关红通道是显示为纸白，不是把 R 乘 0）。
        """
        arr = _ch.apply_channel_view(arr, self.doc) if self.doc is not None else arr
        if scale is None:
            self.view.set_pixmap(arr)
        else:
            self.view.set_pixmap(arr, scale=scale)

    def _do_render(self):
        self._render_pending = False
        doc = self.doc
        if doc is None:
            return
        t0 = time.perf_counter()
        # 0) 拖调整层滑块 + 有"它下方"的快照 -> 只重算调整层本身。
        #    大画布上这一步可能仍要几百毫秒（调整本身要碰每一个像素），
        #    那时拖动期间还是交给代理更跟手，停手后再用它出全分辨率。
        adj = self._adj_layer
        if (adj is not None and self._adj_snap_ok(adj)
                and not (self._interactive
                         and self._adjust_ms > PROXY_BUDGET_MS)):
            arr = render_from_snapshot(doc, adj, self._adj_snap)
            if arr is not None:
                self._adjust_ms = (time.perf_counter() - t0) * 1000.0
                self._last_pass = "adjust"
                self._arr_valid = True
                self._last_arr = arr
                self._full_dirty = False
                self._dirty = None
                self._show(arr)
                self.view._update_handles()
                self.inspector.on_rendered()
                return
            # 用不了（剪贴蒙版 / 嵌在组里）就别每帧白试一次
            self._adj_layer = None
            adj = None

        # 1) 交互期 + 大画布 -> 低分辨率代理
        if self._interactive:
            s = self._proxy_scale()
            if s > 0.0:
                arr, kx, ky = render_proxy(doc, s)
                self._arr_valid = False      # 代理图不能当缓存，之后要补全分辨率
                self._last_pass = "proxy"
                self._show(arr, scale=(kx, ky))
                self.view._update_handles()
                return

        # 2) 整幅重算（整幅渲染时顺手把调整层下方的 acc 存成快照，
        #    下一次拖滑块就能走上面的快路径）。大画布走分块：峰值内存只和
        #    瓦片大小有关。Dissolve 时 render_tiled 返回 None，退回整块算。
        if self._want_full():
            arr = None
            if self._use_tiles():
                arr = render_tiled(doc, tile=TILE_EDGE,
                                   snap_layer=adj, snap=self._adj_snap)
                self._last_pass = "tiled"
            if arr is None:
                arr = render_document(doc, snap_layer=adj, snap=self._adj_snap)
                self._last_pass = "full"
            self._render_ms = (time.perf_counter() - t0) * 1000.0
            self._arr_valid = True
        else:
            # 3) 只重算脏区，补回缓存的整幅结果里。脏区可能是**一组**矩形
            #    （一次拖拽在多处落笔），逐块渲、逐块写回。
            #    先按图层样式的外扩量把每块撑大（模糊要卷积上下文），算完裁回内核
            pad = max_effect_padding(doc)
            arr = self._last_arr
            for x0, y0, x1, y1 in self._dirty_rects():
                ex0 = max(0, x0 - pad)
                ey0 = max(0, y0 - pad)
                ex1 = min(doc.width, x1 + pad)
                ey1 = min(doc.height, y1 + pad)
                if ex1 <= ex0 or ey1 <= ey0:
                    continue
                # **只回写脏区本身，不回写外扩的那一圈**。外扩区只是给模糊
                # 提供卷积上下文用的，那里的像素是"在截断的上下文里算出来的"，
                # 写进缓存会覆盖掉本來正确的值 —— 投影拖尾刚好越过脏区边界时
                # 就能看到一条接缝。实测这条曾让局部重渲染与整幅差 131/255。
                sub = None
                if self._use_tiles():
                    # 分块渲染的 out= 只能写进一块连续缓冲，不能直接指向
                    # 缓存里的那一段（否则外扩区也被覆盖，见上面的注释），
                    # 所以照样先落到临时缓冲再回写脏区。
                    tmp = np.empty((ey1 - ey0, ex1 - ex0, 4), np.uint8)
                    sub = render_region_tiled(doc, (ex0, ey0, ex1, ey1),
                                               out=tmp, tile=TILE_EDGE)
                if sub is None:
                    sub = render_document(
                        doc, region=(ex0, ey0, ex1, ey1))
                arr[y0:y1, x0:x1] = sub[y0 - ey0:y1 - ey0, x0 - ex0:x1 - ex0]
            self._last_pass = "partial" if len(self._dirty_rects()) == 1 \
                else "partial-multi"
        self._full_dirty = False
        self._dirty = None
        self._last_arr = arr
        self._show(arr)
        self._show_channels_panel(arr)
        self.view._update_handles()
        self.inspector.on_rendered()

    def refresh_inspector(self):
        self.inspector.refresh()

    def _on_zoom_changed(self):
        self.zoom_label.setText("%d%%" % int(round(self.view.zoom * 100)))

    # ---------- 撤销 ----------

    def commit(self, label=""):
        self.history.commit(self.doc)
        # 落撤销点 = 一次改动的结束：源图可能被就地改过，代理的降采样缓存作废。
        # 拖拽过程中不会 commit，所以缓存能一直热到松手。
        clear_proxy_cache()
        if label:
            self.status.showMessage(label, 3000)

    def schedule_commit(self, label=""):
        """连续改动（拖滑块）时用：延迟落一个撤销点，只记最终状态。"""
        self._commit_label = label
        self._commit_timer.start()

    def _delayed_commit(self):
        self.commit(self._commit_label)

    def undo(self):
        d = self.history.undo()
        if d is None:
            return
        keep = self.selected_id
        self.doc = d
        self.selected_id = keep
        self._last_arr = None           # 整棵图层树被换掉了，缓存作废
        self._arr_valid = False
        self.panel.rebuild()
        self.inspector.refresh()
        self.request_render()
        self.view.update_selection_overlay()
        self.status.showMessage("撤销", 2000)

    def redo(self):
        d = self.history.redo()
        if d is None:
            return
        keep = self.selected_id
        self.doc = d
        self.selected_id = keep
        self._last_arr = None
        self._arr_valid = False
        self.panel.rebuild()
        self.inspector.refresh()
        self.request_render()
        self.view.update_selection_overlay()
        self.status.showMessage("重做", 2000)

    # ---------- 文件操作 ----------

    def open_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开工程", "", PROJ_FILTER)
        if not path:
            return
        try:
            doc = load_project(path)
        except Exception as e:
            QMessageBox.critical(self, "打开失败", str(e))
            return
        self.set_document(doc, reset_history=True)
        self.view.fit()

    def save_project(self):
        if self.doc is None:
            return
        if not self.doc.path:
            self.save_project_as()
            return
        try:
            save_project(self.doc, self.doc.path)
        except Exception as e:
            QMessageBox.critical(self, "保存失败", str(e))
            return
        self._update_title()
        self.status.showMessage("已保存 %s" % self.doc.path, 3000)

    def save_project_as(self):
        if self.doc is None:
            return
        default = os.path.join(
            os.path.expanduser("~"), "Documents",
            (self.doc.name or "untitled") + PROJECT_EXT)
        path, _ = QFileDialog.getSaveFileName(self, "保存工程", default,
                                              PROJ_FILTER)
        if not path:
            return
        if not path.lower().endswith(PROJECT_EXT):
            path += PROJECT_EXT
        try:
            save_project(self.doc, path)
        except Exception as e:
            QMessageBox.critical(self, "保存失败", str(e))
            return
        self._update_title()
        self.status.showMessage("已保存 %s" % path, 3000)

    def export(self, kind):
        if self.doc is None:
            return
        ext = "." + kind
        filt = "PNG 图片 (*.png)" if kind == "png" else "JPEG 图片 (*.jpg *.jpeg)"
        path, _ = QFileDialog.getSaveFileName(
            self, "导出", (self.doc.name or "export") + ext, filt)
        if not path:
            return
        try:
            export_flat(render_document(self.doc), path)
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))
            return
        self.status.showMessage("已导出 %s" % path, 3000)

    def import_image_as_layer(self):
        if self.doc is None:
            return
        paths, _ = QFileDialog.getOpenFileNames(self, "导入图片", "", IMG_FILTER)
        if not paths:
            return
        for p in paths:
            try:
                # SVG 是矢量，尺寸按画布走（不然一张 16×16 的图标会挤在角落里）
                arr = imread_rgba(p, target=(self.doc.width, self.doc.height))
            except Exception as e:
                QMessageBox.warning(self, "导入失败", "%s\n%s" % (p, e))
                continue
            name = os.path.splitext(os.path.basename(p))[0]
            layer = make_image_layer(name, arr, self.doc.width, self.doc.height)
            # 比画布大就缩放到能放进去
            h, w = arr.shape[:2]
            if w > self.doc.width or h > self.doc.height:
                s = min(self.doc.width / w, self.doc.height / h)
                layer.sx = s
                layer.sy = s
            self.doc.layers.append(layer)
            self.selected_id = layer.id
        self.commit("导入图层")
        self.panel.rebuild()
        self.request_render()

    def import_psd(self):
        """导入一个 Photoshop 文档，整棵图层树搬进来。"""
        if not psd_available():
            QMessageBox.information(
                self, "缺少依赖",
                "导入 PSD 需要 psd-tools：\n\n"
                "  pip install psd-tools "
                "-i https://mirrors.aliyun.com/pypi/simple\n\n"
                "装完重启程序即可。")
            return
        path, _ = QFileDialog.getOpenFileName(self, "导入 PSD", "",
                                              PSD_FILTER)
        if not path:
            return
        try:
            doc = load_psd(path)
        except Exception as e:
            QMessageBox.critical(self, "导入失败", "%s\n%s" % (path, e))
            return
        self.set_document(doc, reset_history=True)
        self.view.fit()
        bits = ["已导入 %s —— %d 个顶层图层"
                % (os.path.basename(path), len(doc.layers))]
        if STATS["text"]:
            bits.append("%d 个文字层已还原成可编辑文字" % STATS["text"])
        if STATS["text_fallback"]:
            bits.append("%d 个文字层因字体缺失或混排转为位图"
                        % STATS["text_fallback"])
        if STATS["effects"]:
            bits.append("%d 个图层带上图层样式" % STATS["effects"])
        bits.append("智能对象 / 调整层仍为合并位图")
        self.status.showMessage("；".join(bits), 8000)

    # ---------- 文字 ----------

    def create_text_layer(self, pt=None, content=None):
        """建一个文字图层。

        pt      画布坐标 (x, y)，None 表示画布中心
        content None 时弹输入框让用户输入（工具箱点击走这条）
        """
        if self.doc is None:
            return None
        if content is None:
            content, ok = QInputDialog.getMultiLineText(
                self, "文字图层", "输入文字：", "文字")
            if not ok:
                return None
        params = self.opts.text_params()
        params["content"] = content
        layer = make_text_layer("文字", self.doc.width, self.doc.height,
                                params, center=pt)
        if self.selected_id and self.doc.find(self.selected_id) is not None:
            self.doc.insert_above(self.selected_id, layer)
        else:
            self.doc.layers.append(layer)
        self.selected_id = layer.id
        self.commit("新建文字图层")
        self.panel.rebuild()
        self.inspector.refresh()
        self.request_render()
        self.status.showMessage(
            "文字已创建 —— 在右侧「属性」面板里改字 / 字体 / 字号，随时可改", 5000)
        return layer

    def edit_text_on_canvas(self):
        """Ctrl+T：在画布上直接改文字图层的字。"""
        layer = self.selected_layer()
        if layer is None or not layer.is_text or layer.text is None:
            self.status.showMessage(
                "先选中一个文字图层（用文字工具点画布可以新建）", 4000)
            return False
        if not self.view.begin_text_edit(layer):
            return False
        self.status.showMessage(
            "直接在画布上打字，Esc 结束 —— 改的是文字参数，之后随时还能再改", 6000)
        return True

    def edit_text_chars(self):
        """逐字调整对话框（字距 / 基线 / 横向缩放）。"""
        layer = self.selected_layer()
        if layer is None or not layer.is_text or layer.text is None:
            self.status.showMessage("先选中一个文字图层", 4000)
            return None
        from .text_dialog import CharAdjustDialog
        dlg = CharAdjustDialog(self, self)
        dlg.exec()
        self.request_render()
        return dlg

    def set_text_param(self, key, value):
        """改文字图层的一个参数并重新栅格化（非破坏性：参数还在）。"""
        layer = self.selected_layer()
        if layer is None or not layer.is_text or layer.text is None:
            return
        if layer.text.get(key) == value:
            return
        layer.text[key] = value
        sync_text_image(layer)
        self.panel.rebuild()
        self.request_render()
        self.schedule_commit("文字参数")

    def rasterize_selected(self):
        """栅格化当前图层：文字图层 / 智能对象 -> 普通位图图层。"""
        layer = self.selected_layer()
        if layer is None:
            return
        if layer.is_text:
            self.rasterize_text()
        elif layer.is_smart:
            self.rasterize_smart()
        else:
            self.status.showMessage("这个图层本来就是普通位图图层", 2500)

    def rasterize_text(self):
        """文字图层 -> 普通位图图层。栅格化之后才能用画笔 / 滤镜改像素。"""
        layer = self.selected_layer()
        if layer is None or not layer.is_text:
            return
        layer.kind = LAYER_IMAGE
        layer.text = None
        self.commit("栅格化文字")
        self.panel.rebuild()
        self.inspector.refresh()
        self.request_render()
        self.status.showMessage(
            "已栅格化成普通位图图层 —— 现在可以用画笔和滤镜了（文字不能再改）", 5000)

    # ---------- 智能对象 ----------

    def _require_smart(self, what):
        """取当前选中的智能对象图层；不是就给句提示并返回 None。"""
        layer = self.selected_layer()
        if layer is None:
            self.status.showMessage("请先选中一个图层")
            return None
        if not layer.is_smart:
            self.status.showMessage(
                "要先%s，得先选中智能对象图层 —— 用「图层 > 智能对象 > 转换为智能对象」" % what,
                5000)
            return None
        content = (self.doc.smart_contents or {}).get(layer.so_id)
        if content is None:
            self.status.showMessage("这个智能对象的内容丢了 —— 请重新载入工程", 5000)
            return None
        return layer

    def convert_selection_to_smart(self):
        """把当前图层（或图层组）连同它的剪贴蒙版装进一段独立内容。"""
        layer = self.selected_layer()
        if layer is None:
            self.status.showMessage("请先选中一个图层或图层组")
            return
        if layer.is_adjustment:
            self.status.showMessage("调整层不能单独转成智能对象")
            return
        if layer.is_smart:
            self.status.showMessage("这个图层已经是智能对象了")
            return
        sync_smart(self.doc)            # 视觉上讲：先把派生栅格刷新到最新
        shell = convert_to_smart(self.doc, layer.id)
        if shell is None:
            self.status.showMessage("转换失败：这个图层没有可渲染的内容", 4000)
            return
        sync_smart(self.doc)         # 立刻生成栅格，缩略图/属性面板才拿得到尺寸
        self.selected_id = shell.id
        self.commit("转换为智能对象")
        self.panel.rebuild()
        self.inspector.refresh()
        self.mark_dirty()
        self.request_render()
        self.status.showMessage(
            "已转成智能对象 —— 双击缩略图或用「编辑内容」打开 sub-document 改它，"
            "滤镜会变成随时可调的智能滤镜", 6000)

    def new_smart_copy(self):
        """复制一个共用同一份内容的实例（改内容会联动）。"""
        layer = self.selected_layer()
        if layer is None or not layer.is_smart:
            self.status.showMessage("请先选中一个智能对象图层")
            return
        n = new_smart_instance(self.doc, layer.id)
        if n is None:
            return
        self.selected_id = n.id
        self.commit("新建智能对象实例")
        self.panel.rebuild()
        self.mark_dirty()
        self.request_render()
        self.status.showMessage(
            "新建了一个共用内容的实例 —— 各自的变换 / 蒙版 / 智能滤镜是独立的", 5000)

    def edit_smart_content(self, layer=None):
        """打开独立窗口编辑智能对象的嵌入内容。"""
        if layer is None:
            layer = self._require_smart("编辑内容")
            if layer is None:
                return
        elif not layer.is_smart:
            return
        content = self.doc.smart_contents[layer.so_id]
        # 已经开着同一个内容的编辑器了就复用（避免两个窗口互相覆盖）
        for ed in list(self._so_editors):
            try:
                if ed.content_id == content.id:
                    ed.raise_()
                    ed.activateWindow()
                    return
            except RuntimeError:
                self._so_editors.remove(ed)
            except AttributeError:
                pass

        sub = Document(content.width, content.height, content.name)
        sub.layers = content.layers         # 直接编辑内容本身（不是副本）
        sub.smart_contents = self.doc.smart_contents
        ed = SmartObjectEditor(self, self, content, sub)
        self._so_editors.append(ed)
        ed.show()

    def _on_smart_edited(self, content):
        """内容编辑器关闭：内容已就地改过，作废栅格缓存重新渲染。"""
        content.touch()
        refresh_sizes(self.doc)
        clear_proxy_cache()
        self.mark_dirty()
        self.request_render()
        self.panel.rebuild()
        self.inspector.refresh()
        self.status.showMessage(
            "智能对象内容已更新 —— 共用这份内容的 %d 个实例一起变了"
            % len(content_instances(self.doc, content.id)), 5000)

    def edit_smart_filters(self):
        """智能滤镜管理对话框。"""
        layer = self._require_smart("管理智能滤镜")
        if layer is None:
            return
        from .smart_dialog import SmartFilterDialog
        dlg = SmartFilterDialog(self, layer)
        dlg.exec()
        self.commit("智能滤镜")
        self.inspector.refresh()
        self.panel.rebuild()

    def rasterize_smart(self):
        """智能对象 -> 普通位图图层。"""
        layer = self._require_smart("栅格化")
        if layer is None:
            return
        if not rasterize_smart(self.doc, layer):
            self.status.showMessage("栅格化失败：内容渲染不出来", 4000)
            return
        self.commit("栅格化智能对象")
        self.panel.rebuild()
        self.inspector.refresh()
        self.mark_dirty()
        self.request_render()
        self.status.showMessage(
            "已烧成普通位图图层 —— 内容与智能滤镜从此不可再改", 5000)

    # ---------- 浮动选区（移动选中的像素）----------

    def can_lift_selection(self):
        from ..core.float_sel import can_lift
        return can_lift(self.doc, self.selected_layer())

    def lift_selection(self, copy_mode=False):
        """把选区内容揭成浮动层，之后拖动只移动它。"""
        from ..core.float_sel import lift_selection
        doc, layer = self.doc, self.selected_layer()
        if not self.can_lift_selection():
            return False
        doc.detach_pixels(layer)        # 写时复制：挖空会改像素
        fl = lift_selection(doc, layer, copy_mode=copy_mode)
        if fl is None:
            return False
        self.commit("移动选区内容")
        self.request_render()
        self.status.showMessage(
            "已把选区内容揭成浮动层 —— 拖动移动，拖手柄缩放"
            "（Alt 以中心、Shift 等比），回车落定，Esc 丢弃", 6000)
        return True

    def stamp_float(self, silent=False):
        """把浮动层盖回当前图层。没有浮动层时什么都不做。"""
        from ..core.float_sel import stamp_float
        doc = self.doc
        if doc is None or doc.float_layer is None:
            return False
        layer = self.selected_layer()
        if layer is not None:
            doc.detach_pixels(layer)
        ok = stamp_float(doc, layer)
        if ok:
            self.commit("落定浮动选区")
            if not silent:
                self.status.showMessage("浮动选区已盖回图层", 3000)
        self.request_render()
        return ok

    def discard_float(self):
        """Esc：丢掉浮动内容（原处已经挖空，用 Ctrl+Z 撤销）。"""
        from ..core.float_sel import discard_float
        if self.doc is None or self.doc.float_layer is None:
            return False
        discard_float(self.doc)
        self.request_render()
        self.status.showMessage(
            "已丢弃浮动内容（原处已挖空，Ctrl+Z 可撤销）", 5000)
        return True

    def edit_layer_style(self):
        """打开图层样式对话框（10 种效果 + 全局光）。"""
        layer = self.selected_layer()
        if layer is None:
            self.status.showMessage("请先选中一个图层")
            return
        if layer.is_group or layer.is_adjustment:
            self.status.showMessage("图层组和调整层不支持图层样式")
            return
        from .style_dialog import StyleDialog
        dlg = StyleDialog(layer, self, self)
        dlg.exec()
        self.inspector.refresh()
        self.panel.rebuild()

    # ---------- 图层样式的复制 / 粘贴 / 默认值 ----------

    def _style_target(self, need_copy=False):
        """-> 图层；不能用时提示并返回 None。"""
        layer = self.selected_layer()
        if layer is None:
            self.status.showMessage("请先选中一个图层")
            return None
        if layer.is_group or layer.is_adjustment:
            self.status.showMessage("图层组和调整层不支持图层样式")
            return None
        if need_copy and not layer.effects:
            self.status.showMessage("这个图层没有图层样式")
            return None
        return layer

    def copy_layer_style(self):
        """把选中图层的整套样式存进剪贴板（含关闭的效果，方便整份迁移）。"""
        from ..core.effects import EFFECT_NAMES, copy_style, enabled_effects
        layer = self._style_target(need_copy=True)
        if layer is None:
            return False
        copy_style(layer.effects)
        names = [EFFECT_NAMES[k]
                 for k in enabled_effects(layer.effects)]
        self.status.showMessage(
            "已复制图层样式：%s" % ("、".join(names) if names else "（没有启用的效果）"),
            4000)
        return True

    def paste_layer_style(self):
        """整份覆盖当前图层的样式。"""
        from ..core.effects import has_style_clipboard, paste_style
        layer = self._style_target()
        if layer is None:
            return False
        if not has_style_clipboard():
            self.status.showMessage("剪贴板里没有图层样式")
            return False
        layer.effects = paste_style()
        self.commit("粘贴图层样式")
        self.request_render()
        self.inspector.refresh()
        self.panel.rebuild()
        self.status.showMessage("已粘贴图层样式", 4000)
        return True

    def save_style_default(self):
        """把选中图层的当前参数存成默认值（下次新建样式从它出发）。"""
        from ..core.effects import save_style_default as _save
        layer = self._style_target(need_copy=True)
        if layer is None:
            return False
        _save(layer.effects)
        self.status.showMessage("已把当前参数存为图层样式的默认值", 4000)
        return True

    def reset_style_default(self):
        """清掉用户存的默认值，回到出厂参数。"""
        from ..core.effects import reset_style_default as _reset
        _reset()
        self.status.showMessage("已复位图层样式的默认值", 4000)
        return True

    def _reject_text_layer(self, what):
        """像素是"算出来"的图层不能直接改像素 —— 先挡住。

        文字图层的像素由参数生成，智能对象的像素由嵌入内容渲染而来，
        它们都会在下次重算时被冲掉，所以要改就得走各自的门：
        文字 -> 属性面板改参数；智能对象 -> 「编辑内容」/「智能滤镜」；
        真想直接画 -> 先栅格化。
        """
        layer = self.selected_layer()
        if layer is None:
            return False
        if layer.is_text:
            self.status.showMessage(
                "文字图层不能直接%s —— 想改像素请先在属性面板里「栅格化」" % what,
                5000)
            return True
        if layer.is_smart:
            self.status.showMessage(
                "智能对象的像素是内容渲染出来的，不能直接%s —— "
                "改内容用「图层 > 智能对象 > 编辑内容」，加滤镜用「智能滤镜」，"
                "想随笔画就先「栅格化智能对象」" % what, 7000)
            return True
        return False

    # ---------- 图层操作 ----------

    def duplicate_layer(self):
        layer = self.selected_layer()
        if layer is None:
            return
        clone = layer.clone()
        clone.id = uuid.uuid4().hex
        clone.name = layer.name + " 副本"
        lst, idx = self.doc.parent_list(layer.id)
        if lst is None:
            self.doc.layers.append(clone)
        else:
            lst.insert(idx + 1, clone)
        self.selected_id = clone.id
        self.commit("复制图层")
        self.panel.rebuild()
        self.request_render()

    def delete_layer(self):
        layer = self.selected_layer()
        if layer is None:
            return
        self.stamp_float(silent=True)
        if len(self.doc.layers) == 1 and not layer.is_group:
            QMessageBox.information(self, "提示", "至少要保留一个图层。")
            return
        self.doc.remove(layer.id)
        self.selected_id = None
        prune_contents(self.doc)    # 智能对象删掉之后，没人引用的内容要回收
        self.commit("删除图层")
        self.panel.rebuild()
        self.inspector.refresh()
        self.request_render()

    def group_selection(self):
        layer = self.selected_layer()
        if layer is None:
            return
        lst, idx = self.doc.parent_list(layer.id)
        if lst is None:
            return
        g = make_group("组 %d" % (len(list(self.doc.all_layers())) + 1),
                       [layer])
        lst[idx] = g
        self.selected_id = g.id
        self.commit("新建图层组")
        self.panel.rebuild()
        self.request_render()

    # ---------- 调整层 ----------

    def add_adjustment_layer(self, key):
        """在当前图层上方插入一个调整层。作用对象是它下面的全部内容。"""
        if self.doc is None:
            return
        spec = ADJUSTMENTS.get(key)
        if spec is None:
            return
        layer = make_adjustment_layer(spec.name, key,
                                      self.doc.width, self.doc.height)
        if self.selected_id and self.doc.find(self.selected_id) is not None:
            self.doc.insert_above(self.selected_id, layer)
        else:
            self.doc.layers.append(layer)
        self.selected_id = layer.id
        self.commit("新建调整图层：%s" % spec.name)
        self.panel.rebuild()
        self.inspector.refresh()
        self.request_render()
        self.status.showMessage(
            "已添加「%s」调整层 —— 拖下面板里的滑块改参数，随时可改"
            % spec.name, 5000)

    def toggle_clip(self):
        """剪贴蒙版：让调整层只作用于紧邻的下方图层。"""
        layer = self.selected_layer()
        if layer is None:
            return
        layer.clipped = not layer.clipped
        if layer.clipped and not layer.is_adjustment:
            self.status.showMessage(
                "已标记裁剪，但它下面需要紧跟一个调整层才会生效", 4000)
        self.commit("剪贴蒙版")
        self.panel.rebuild()
        self.request_render()

    # ---------- 滤镜 ----------

    def ask_filter(self, key):
        """打开滤镜对话框。

        普通图层：预览期间改的是副本，取消就整块还原（滤镜本身是破坏性的）。
        智能对象图层：加的是**智能滤镜**，参数存下来，随时能改能删。
        """
        if self.doc is None:
            return
        spec = FILTERS.get(key)
        layer = self.selected_layer()
        if spec is None:
            return
        if layer is not None and layer.is_smart:
            self._ask_smart_filter(key, spec)
            return
        if layer is None or layer.is_group or layer.image is None:
            QMessageBox.information(
                self, "无法应用", "请先选中一个位图图层。\n"
                "调整层与图层组没有像素，滤镜对它们不适用。")
            return
        if self._reject_text_layer("应用滤镜"):
            return
        if layer.locked:
            self.status.showMessage("图层已锁定")
            return

        # 大图层预览时先在缩略图上算（半径按同一比例缩小），确定时才全分辨率
        scale, small = preview_filter_scale(layer.image, key)
        sel = self._src_selection(layer)
        self._filter_ctx = {
            "layer": layer,
            "orig": layer.image.copy(),
            "sel": sel,
            "key": key,
            "scale": scale,
            "small": small if scale < 1.0 else None,
            "sel_small": (cv2.resize(sel, (small.shape[1], small.shape[0]),
                                     interpolation=cv2.INTER_AREA)
                          if (scale < 1.0 and sel is not None) else None),
        }
        dlg = FilterDialog(self, key)
        dlg.paramsChanged.connect(self._filter_preview)
        ok = dlg.exec()
        ctx = self._filter_ctx
        self._filter_ctx = None
        if ok:
            out = apply_filter_array(ctx["orig"], key, dlg.values(),
                                     ctx["sel"], dlg.strength())
            # apply_filter_array 总是返回新数组，不会动历史快照，无需 detach
            layer.image = out
            self.after_pixel_edit("滤镜：%s" % spec.name.replace("…", ""))
        else:
            layer.image = ctx["orig"]
            self.request_render()
            self.status.showMessage("已取消滤镜", 2500)

    def _ask_smart_filter(self, key, spec):
        """给智能对象追加一条智能滤镜（内容保持不变，随时可以改）。"""
        layer = self.selected_layer()
        base = [dict(f) for f in (layer.so_filters or [])]
        dlg = FilterDialog(self, key)

        def preview(params, strength):
            layer.so_filters = base + [
                new_filter(key, params, strength)]
            self.begin_interactive()    # 拖滑块期间走代理，停手后自动补全分辨率
            self.request_render()

        dlg.paramsChanged.connect(preview)
        ok = dlg.exec()
        if ok:
            layer.so_filters = base + [
                new_filter(key, dlg.values(), dlg.strength())]
            self.commit("智能滤镜：%s" % spec.name.replace("…", ""))
            self.inspector.refresh()
            self.status.showMessage(
                "已加成智能滤镜 —— 原始内容没动，「图层 > 智能对象 > 智能滤镜」里随时可改可删",
                6000)
        else:
            layer.so_filters = base
            self.request_render()
            self.status.showMessage("已取消智能滤镜", 2500)

    def _filter_preview(self, params, strength):
        ctx = self._filter_ctx
        if ctx is None:
            return
        s = ctx.get("scale", 1.0)
        if s < 1.0 and preview_can_downscale(ctx["key"], params, s):
            # 在缩略图上跑，半径按同一比例缩小，算完放大回原尺寸 —— 预览而已，
            # 「确定」时走的是下面的全分辨率精确路径
            out = apply_filter_array(ctx["small"], ctx["key"],
                                     scale_filter_params(ctx["key"], params, s),
                                     ctx["sel_small"], strength)
            out = cv2.resize(out, (ctx["orig"].shape[1], ctx["orig"].shape[0]),
                             interpolation=cv2.INTER_LINEAR)
        else:
            out = apply_filter_array(ctx["orig"], ctx["key"], params,
                                     ctx["sel"], strength)
        ctx["layer"].image = out
        self.begin_interactive()        # 拖滑块期间走代理，停手后自动补全分辨率
        self.request_render()

    def raise_to_top(self):
        layer = self.selected_layer()
        if layer is None:
            return
        self.doc.remove(layer.id)
        self.doc.layers.append(layer)
        self.commit("置于顶层")
        self.panel.rebuild()
        self.request_render()

    def lower_to_bottom(self):
        layer = self.selected_layer()
        if layer is None:
            return
        self.doc.remove(layer.id)
        self.doc.layers.insert(0, layer)
        self.commit("置于底层")
        self.panel.rebuild()
        self.request_render()

    # ---------- 选区 ----------

    def _sel(self):
        return self.doc.selection if self.doc else None

    def _require_doc(self):
        return self.doc is not None

    def commit_selection(self, sel, mode=REPLACE):
        """把一个新建的选区按 mode 合并进文档，并写入撤销栈。"""
        if not self._require_doc() or sel is None:
            return
        if self.opts.feather > 0:
            sel = sel.copy()
            sel.feather(self.opts.feather)
        prev = self.doc.selection
        if prev is not None and not prev.is_empty and mode != REPLACE:
            self.doc.detach_selection()
            self.doc.selection = self.doc.selection.combine(sel, mode)
        else:
            self.doc.selection = sel.copy()
        self.commit("选区")
        self.view.update_selection_overlay()

    def select_all(self):
        if not self._require_doc():
            return
        if quick_mask.is_on(self.doc):
            self.fill_quick_mask(255)
            return
        self.doc.detach_selection()
        self.doc.selection = Selection.all(self.doc.width, self.doc.height)
        self.commit("全选")
        self.view.update_selection_overlay()

    def deselect(self):
        if not self._require_doc():
            return
        if quick_mask.is_on(self.doc):
            self.fill_quick_mask(0)
            return
        self.stamp_float(silent=True)
        self.doc.detach_selection()
        self.doc.selection = None
        self.commit("取消选择")
        self.view.update_selection_overlay()

    # ---------- 通道（core.channels） ----------

    def _show_channels_panel(self, arr):
        """把刚渲出来的**原始**合成结果喂给通道面板生成缩略图。

        传原始 arr（不是 `_show()` 处理过的）：PS 的通道缩略图始终显示
        通道的**真实内容**，不受"这一路当前可不可见"影响。
        """
        if hasattr(self, "channels"):
            self.channels.set_render(arr)

    def load_channel_selection(self, cid):
        """把通道内容作为选区载入画布。"""
        doc = self.doc
        if doc is None or not cid:
            return
        if str(cid) == _ch.COMPOSITE_ROW:
            if doc.composite_alpha is None:
                self.status.showMessage("还没有合并的 Alpha 通道", 2500)
                return
            doc.detach_selection()
            from .selection import Selection
            if not doc.composite_alpha.any():
                doc.selection = None
            else:
                doc.selection = Selection(doc.width, doc.height,
                                          doc.composite_alpha.copy())
        elif str(cid).startswith(_ch.RGB_ROW_PREFIX):
            # RGB / Alpha 那一路：按当前渲染结果的对应分量当灰度选区
            arr = self._last_arr
            if arr is None:
                return
            doc.detach_selection()
            from .selection import Selection
            g = _ch.channel_thumb(arr, str(cid)[1:], doc.width)
            if not g.any():
                doc.selection = None
            else:
                doc.selection = Selection(doc.width, doc.height, g)
        else:
            if _ch.channel_to_selection(doc, cid) is None and \
                    _ch.find_channel(doc, cid) is None:
                return
        self.commit("载入选区")
        self.view.update_selection_overlay()
        self.channels.rebuild()

    def save_channel(self, cid):
        """新建 Alpha 通道并存入当前选区。"""
        doc = self.doc
        if doc is None:
            return
        if cid != "__new__":
            return
        ch = _ch.selection_to_channel(doc, "Alpha %d" % (
            len(doc.channels) + 1))
        if ch is None:
            self.status.showMessage(
                "通道已达上限（%d 个）" % _ch.MAX_CHANNELS, 3000)
            return
        self.commit("新建通道")
        self.channels.rebuild()
        self.channels.select_id(ch.id)
        self.status.showMessage("已新建通道「%s」" % ch.name, 2500)

    def delete_channel(self, cid):
        doc = self.doc
        if doc is None or not cid:
            return
        ch = _ch.remove_channel(doc, cid)
        if ch is None:
            return
        self.commit("删除通道")
        self.channels.rebuild()
        self.status.showMessage("已删除通道「%s」" % ch.name, 2500)

    def merge_channel(self, cid, mode="replace"):
        """把通道并进「合并的 Alpha 通道」（只改显示，不动图层数据）。"""
        doc = self.doc
        if doc is None or not cid:
            return
        if doc.composite_alpha is None:
            arr = self._last_arr
            if arr is None:
                return
            doc.composite_alpha = _ch.composite_alpha_of(arr)
        if _ch.merge_channel_into_alpha(doc, cid, mode):
            self.commit("并入合并 Alpha")
            self.channels.rebuild()
            self.request_render()

    def _on_channel_renamed(self, item):
        """QListWidget 的行内编辑结束 -> 落到 Channel.name 上。"""
        if item is None or not item.data(_CH_ROLE_MASK):
            return
        self.channels._on_rename(item, item.text())

    def toggle_channels_dock(self):
        v = not self.channels_dock.isVisible()
        self.channels_dock.setVisible(v)

    def _load_selected_channel(self):
        cid = self.channels.current_id()
        if cid:
            self.load_channel_selection(cid)

    def _delete_selected_channel(self):
        cid = self.channels.current_id()
        if cid and not cid.startswith(_ch.RGB_ROW_PREFIX) \
                and cid != _ch.COMPOSITE_ROW:
            self.delete_channel(cid)
        elif cid:
            self.status.showMessage("画布自带的通道不能删除", 2500)

    def _merge_replace(self):
        self._merge_selected("replace")

    def _merge_add(self):
        self._merge_selected("add")

    def _merge_subtract(self):
        self._merge_selected("subtract")

    def _merge_selected(self, mode):
        cid = self.channels.current_id()
        if not cid or cid.startswith(_ch.RGB_ROW_PREFIX) \
                or cid == _ch.COMPOSITE_ROW:
            self.status.showMessage("请先选中一个附加的 Alpha 通道", 2500)
            return
        self.merge_channel(cid, mode)

    def _reset_composite_alpha(self):
        if self.doc is None or self.doc.composite_alpha is None:
            return
        _ch.reset_composite_alpha(self.doc)
        self.commit("重置合并 Alpha")
        self.channels.rebuild()
        self.request_render()

    def _show_all_rgb(self):
        if self.doc is None:
            return
        v = _ch.rgb_visibility(self.doc)
        for k in _ch.RGB_KEYS:
            v[k] = True
        self.channels.rebuild()
        self.request_render()

    def invert_selection(self):
        if quick_mask.is_on(self.doc):
            self.invert_quick_mask()
            return
        sel = self._sel()
        if sel is None or sel.is_empty:
            self.select_all()
            return
        self.doc.detach_selection()
        self.doc.selection.invert()
        self.commit("反选")
        self.view.update_selection_overlay()

    def ask_feather(self):
        sel = self._sel()
        if sel is None or sel.is_empty:
            return
        v, ok = QInputDialog.getDouble(self, "羽化", "羽化半径 (px):",
                                       self.opts.feather, 0, 500, 1)
        if not ok:
            return
        self.doc.detach_selection()
        self.doc.selection.feather(v)
        self.commit("羽化")
        self.view.update_selection_overlay()

    def ask_grow(self, kind):
        sel = self._sel()
        if sel is None or sel.is_empty:
            return
        title = "扩展选区" if kind == "expand" else "收缩选区"
        v, ok = QInputDialog.getInt(self, title, "像素:", 4, 0, 2000, 1)
        if not ok:
            return
        self.doc.detach_selection()
        if kind == "expand":
            self.doc.selection.expand(v)
        else:
            self.doc.selection.contract(v)
        self.commit(title)
        self.view.update_selection_overlay()

    def ask_smooth(self):
        sel = self._sel()
        if sel is None or sel.is_empty:
            return
        v, ok = QInputDialog.getInt(self, "平滑", "取样半径 (px):",
                                    3, 1, 100, 1)
        if not ok:
            return
        self.doc.detach_selection()
        self.doc.selection.smooth(v)
        self.commit("平滑选区")
        self.view.update_selection_overlay()

    def ask_border(self):
        sel = self._sel()
        if sel is None or sel.is_empty:
            return
        v, ok = QInputDialog.getInt(self, "边界", "边界宽度 (px):",
                                    8, 1, 500, 1)
        if not ok:
            return
        self.doc.detach_selection()
        self.doc.selection.border(v)
        self.commit("边界选区")
        self.view.update_selection_overlay()

    def ask_color_range(self):
        """色彩范围对话框。"""
        if not self._require_doc():
            return
        from .color_range_dialog import ColorRangeDialog
        rgb = self.composite_rgb()
        dlg = ColorRangeDialog(self, rgb, self.doc.selection)
        dlg.selectionReady.connect(self._apply_color_range)
        dlg.exec()

    def _apply_color_range(self, mask, mode):
        """把色彩范围算出的遮罩按 mode 合并进当前选区。"""
        sel = Selection(self.doc.width, self.doc.height, mask)
        if sel.is_empty:
            self.status.showMessage("没有命中任何像素 —— 调大容差试试", 4000)
            return
        self.commit_selection(sel, mode)

    # ---------- 快速蒙版 ----------

    def toggle_quick_mask(self):
        """Q：在「选区」和「可以直接涂的遮罩」之间来回切。"""
        if not self._require_doc():
            return
        if quick_mask.is_on(self.doc):
            self.stamp_float(silent=True)
            if not quick_mask.exit_to_selection(self.doc):
                return
            self.commit("退出快速蒙版")
            self.status.showMessage("已退出快速蒙版 —— 遮罩变成了选区", 4000)
        else:
            self.stamp_float(silent=True)
            if not quick_mask.enter(self.doc):
                return
            self.commit("进入快速蒙版")
            self.status.showMessage(
                "快速蒙版：涂黑 = 排除出选区，涂白 = 纳入选区，再按 Q 退出", 6000)
        self.view.update_selection_overlay()
        self._sync_quick_mask_ui()

    def set_quick_mask_mode(self, mode):
        """红色盖在哪一边：masked = 盖未选中区（默认）。"""
        if self.doc is None or not quick_mask.is_on(self.doc):
            return
        self.doc.quick_mask_mode = mode
        self.view.update_selection_overlay()

    def fill_quick_mask(self, value):
        """整片遮罩填成某个灰度（0~255）。"""
        if self.doc is None or not quick_mask.is_on(self.doc):
            return False
        self.doc.detach_quick_mask()
        self.doc.quick_mask = np.full(
            (self.doc.height, self.doc.width), int(np.clip(value, 0, 255)),
            np.uint8)
        self.commit("填充快速蒙版")
        self.view.update_selection_overlay()
        return True

    def invert_quick_mask(self):
        if self.doc is None or not quick_mask.is_on(self.doc):
            return False
        self.doc.detach_quick_mask()
        self.doc.quick_mask = (255 - self.doc.quick_mask).astype(np.uint8)
        self.commit("反相快速蒙版")
        self.view.update_selection_overlay()
        return True

    def _sync_quick_mask_ui(self):
        """进/出快速蒙版时把模式写进标题栏（常驻提示，不会被临时消息冲掉）。"""
        self._update_title()
        self.inspector.refresh()

    def mask_from_selection(self):
        """把当前选区转成当前图层的蒙版。"""
        sel = self._sel()
        layer = self.selected_layer()
        if layer is None or layer.is_group:
            QMessageBox.information(self, "提示", "请先选中一个图层。")
            return
        if layer.image is None and not layer.is_adjustment:
            QMessageBox.information(self, "提示", "请先选中一个位图图层。")
            return
        if sel is None or sel.is_empty:
            QMessageBox.information(self, "提示", "当前没有选区。")
            return
        if layer.is_adjustment:
            # 调整层的蒙版就在画布坐标系里
            if sel.mask.shape != (self.doc.height, self.doc.width):
                m = cv2.resize(sel.mask, (self.doc.width, self.doc.height),
                               interpolation=cv2.INTER_AREA)
            else:
                m = sel.mask.copy()
        else:
            Minv = cv2.invertAffineTransform(layer.matrix())
            h, w = layer.image.shape[:2]
            m = cv2.warpAffine(sel.mask, Minv, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(0,))
        self.doc.detach_pixels(layer)
        layer.mask = np.clip(m, 0, 255).astype(np.uint8)
        layer.mask_enabled = True
        self.commit("从选区建立蒙版")
        self.panel.rebuild()
        self.request_render()

    # ---------- 像素编辑 ----------

    def composite_rgb(self):
        """合成图的 RGB（供魔棒取色）。"""
        arr = getattr(self, "_last_arr", None)
        if arr is None:
            arr = render_document(self.doc)
            self._last_arr = arr
        return np.ascontiguousarray(arr[..., :3])

    def _src_selection(self, layer):
        """把画布坐标的选区换算到图层源坐标，返回 float32 0~1 或 None。

        调整层没有源图像，它的蒙版就是画布坐标系，直接返回原选区。
        """
        sel = self._sel()
        if sel is None or sel.is_empty:
            return None
        if layer.is_adjustment:
            return sel.float_mask()
        Minv = cv2.invertAffineTransform(layer.matrix())
        h, w = layer.image.shape[:2]
        m = cv2.warpAffine(sel.float_mask(), Minv, (w, h),
                           flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(0,))
        return np.clip(m, 0.0, 1.0)

    def fill_with_fg(self, pt=None, use_fg=True):
        """油漆桶 / 填充。pt 为 None 时按当前选区填充整层。"""
        if quick_mask.is_on(self.doc):
            # 快速蒙版模式下"填充"填的是那张遮罩：前景色黑的涂满 = 整片取消
            color = self.opts.fg.rgb() if use_fg else self.opts.bg.rgb()
            self.fill_quick_mask(_gray(color))
            return
        layer = self.selected_layer()
        if layer is None or layer.is_group:
            self.status.showMessage("请先选中一个图层")
            return
        if layer.image is None and not layer.is_adjustment:
            self.status.showMessage("请先选中一个位图图层")
            return
        if self._reject_text_layer("填充"):
            return
        if layer.locked:
            self.status.showMessage("图层已锁定")
            return
        color = self.opts.fg.rgb() if use_fg else self.opts.bg.rgb()
        target = self.opts.target
        sel = self._src_selection(layer)
        self.doc.detach_pixels(layer)
        if target == "mask" and layer.mask is not None:
            v = _gray(color)
            if sel is None:
                layer.mask = np.full(layer.mask.shape, int(round(v)), np.uint8)
            else:
                m = layer.mask.astype(np.float32)
                layer.mask = np.clip(m * (1 - sel) + v * sel, 0, 255).astype(np.uint8)
        else:
            fill_rgba(layer.image, sel, color)
        self.after_pixel_edit("填充")

    def delete_in_selection(self):
        """清除选区内的像素（Delete）。"""
        if quick_mask.is_on(self.doc):
            # 快速蒙版模式下 Delete = 把遮罩清成 0（整片取消选中）
            self.fill_quick_mask(0)
            return
        sel = self._sel()
        layer = self.selected_layer()
        if sel is None or sel.is_empty:
            self.delete_layer()
            return
        if layer is None or layer.is_group or layer.image is None:
            return
        if self._reject_text_layer("清除"):
            return
        if layer.locked:
            self.status.showMessage("图层已锁定")
            return
        s = self._src_selection(layer)
        self.doc.detach_pixels(layer)
        clear_rgba(layer.image, s)
        self.after_pixel_edit("清除")

    def after_pixel_edit(self, label="绘制"):
        """像素被改动后的统一收尾：撤销点 + 重渲染 + 刷新缩略图。"""
        self.commit(label)
        self.panel.rebuild()
        self.request_render()
        self.inspector.refresh()

    # ---------- 修饰类工具（core/retouch.py）----------

    def _retouch_target_layer(self, what):
        """修饰类工具要改像素时的统一前置检查。返回图层或 None。"""
        if quick_mask.is_on(self.doc):
            self.status.showMessage(
                "快速蒙版模式下「%s」不可用，先退出快速蒙版" % what, 4000)
            return None
        layer = self.selected_layer()
        if layer is None or layer.is_group:
            self.status.showMessage("先选中一个图层")
            return None
        if layer.image is None and not layer.is_adjustment:
            self.status.showMessage("这个图层没有像素可改")
            return None
        if self._reject_text_layer(what):
            return None
        if layer.locked:
            self.status.showMessage("图层已锁定")
            return None
        return layer

    def apply_gradient_drag(self, start, end, layer=None):
        """渐变工具：把 start→end 的渐变涂进图层。

        拖动坐标是画布坐标，要逆变换到图层源坐标 —— 渐变得跟着图层走，
        图层缩放过之后渐变范围也要跟着缩。选区存在时只涂选区内。
        一次拖动 = 一个撤销点。
        """
        layer = layer or self._retouch_target_layer("渐变")
        if layer is None:
            return False
        params = self.opts.gradient_params()
        Minv = cv2.invertAffineTransform(layer.matrix())
        p0 = cv2.transform(np.array([[[start[0], start[1]]]], np.float32),
                           Minv)[0, 0]
        p1 = cv2.transform(np.array([[[end[0], end[1]]]], np.float32),
                           Minv)[0, 0]
        sel = self._src_selection(layer)
        on_mask = (self.opts.target == "mask" and layer.mask is not None)
        self.doc.detach_pixels(layer)
        if on_mask:
            col = retouch.gradient_colors(layer.mask.shape, p0, p1, params)
            v = col.mean(axis=2) * 255.0
            if sel is None:
                layer.mask = np.clip(v, 0, 255).astype(np.uint8)
            else:
                m = layer.mask.astype(np.float32)
                layer.mask = np.clip(m * (1 - sel) + v * sel,
                                     0, 255).astype(np.uint8)
        else:
            retouch.apply_gradient(layer.image, p0, p1, params, sel)
        self.after_pixel_edit("渐变")
        return True

    def blur_at(self, x, y, params=None, layer=None):
        """模糊工具：在 (x,y)（画布坐标）落一次模糊。

        和画笔一样是**增量**的 —— 每次只处理落点附近那一块，
        大图层上按住拖动才不会卡。
        """
        layer = layer or self._retouch_target_layer("模糊")
        if layer is None:
            return False
        p = params or self.opts.blur_params()
        rad = p["size"] / 2.0
        if layer.is_adjustment:
            if layer.mask is None:
                self.status.showMessage("调整层的蒙版不存在，先加个蒙版")
                return False
            self.doc.detach_pixels(layer)
            # 蒙版是灰度：造一份 RGBA 副本去卷积，取回灰度那一路
            h, w = layer.mask.shape
            tmp = np.zeros((h, w, 4), np.uint8)
            tmp[..., 0] = tmp[..., 1] = tmp[..., 2] = layer.mask
            tmp[..., 3] = 255
            retouch.blur_region(tmp, x, y, rad, p["hardness"], p["strength"])
            newm = tmp[..., 0].astype(np.float32)
            sel = self._sel()
            if sel is not None and not sel.is_empty:
                m = layer.mask.astype(np.float32)
                newm = m * (1 - sel.float_mask()) + newm * sel.float_mask()
            layer.mask = np.clip(newm, 0, 255).astype(np.uint8)
        else:
            if layer.is_smart:
                self.status.showMessage(
                    "智能对象不能直接模糊 —— 用「编辑内容」或先栅格化", 5000)
                return False
            self.doc.detach_pixels(layer)
            Minv = cv2.invertAffineTransform(layer.matrix())
            src = cv2.transform(np.array([[[x, y]]], np.float32), Minv)[0, 0]
            # 半径以画布像素计；图层缩放过，源坐标里的半径要一起缩
            sc = float(layer.sx or 1.0)
            retouch.blur_region(layer.image, src[0], src[1], rad * sc,
                                p["hardness"], p["strength"])
        self.after_pixel_edit("模糊")
        return True

    def _retouch_src_layer(self, what, layer):
        """把画布坐标换算到图层源坐标，顺带给一个"源坐标里的半径换算系数"。

        返回 (Minv, scale) 或 None。`scale` 是画布像素 -> 源像素的缩放，
        模糊 / 克隆 / 修复的半径都要乘它 —— 图层缩放过之后笔刷半径得跟着缩，
        否则视觉上的笔触大小就不对。
        """
        Minv = cv2.invertAffineTransform(layer.matrix())
        m = layer.matrix()
        det = abs(m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0])
        return Minv, float(np.sqrt(max(det, 1e-9)))

    def clone_sample(self, x, y, params=None):
        """克隆图章：按住 Alt 时对当前图层拍一张快照。返回 CloneSource 或 None。

        拍的是**图层源图像**，并且是副本 —— 之后无论源图层怎么改，这份快照不变。
        否则在同一张图上克隆会「边涂边采到自己的新颜料」，越涂越花。
        """
        layer = self._retouch_target_layer("克隆图章")
        if layer is None or layer.is_adjustment:
            return None
        if layer.is_smart:
            self.status.showMessage("智能对象不能直接克隆 —— 先栅格化", 4000)
            return None
        p = params or self.opts.clone_params()
        Minv = cv2.invertAffineTransform(layer.matrix())
        src = cv2.transform(np.array([[[x, y]]], np.float32), Minv)[0, 0]
        h, w = layer.image.shape[:2]
        if not (0 <= src[0] < w and 0 <= src[1] < h):
            self.status.showMessage("Alt 采样要点在图层有像素的地方", 4000)
            return None
        snap = retouch.CloneSource(layer.image.copy(), int(round(src[0])),
                                  int(round(src[1])), p["hardness"],
                                  int(p["size"]))
        self.view.set_clone_source(snap)
        self.status.showMessage(
            "已采样源坐标 (%d, %d) —— 移动鼠标后左键盖章，Alt 重新采样"
            % (snap.ox, snap.oy), 4000)
        return snap

    def clone_stamp(self, x, y, params=None):
        """克隆图章：落一次章（拖动时 canvas_view 会补样成 path）。"""
        snap = self.view.clone_source
        if snap is None:
            self.status.showMessage("克隆图章：先按住 Alt 点一下采样", 4000)
            return False
        layer = self._retouch_target_layer("克隆图章")
        if layer is None or layer.is_adjustment:
            return False
        if layer.is_smart:
            self.status.showMessage("智能对象不能直接克隆 —— 先栅格化", 4000)
            return False
        p = params or self.opts.clone_params()
        got = self._retouch_src_layer("克隆图章", layer)
        if got is None:
            return False
        Minv, scale = got
        src = cv2.transform(np.array([[[x, y]]], np.float32), Minv)[0, 0]
        if snap.shape != layer.image.shape[:2]:
            self.status.showMessage("采样时的图层尺寸与现在不一致（撤销过？）", 4000)
            return False
        self.doc.detach_pixels(layer)
        ok = retouch.clone_stamp_at(layer.image, snap, src[0], src[1],
                                    p["size"] / 2.0 * scale, p["hardness"],
                                    p["strength"])
        if not ok:
            return False
        self.after_pixel_edit("克隆图章")
        return True

    def heal_at(self, x, y, params=None):
        """污点修复：把 (x,y) 附近的瑕点用周围纹理补上。"""
        layer = self._retouch_target_layer("污点修复")
        if layer is None:
            return False
        p = params or self.opts.clone_params()
        if layer.is_adjustment:
            if layer.mask is None:
                self.status.showMessage("调整层的蒙版不存在，先加个蒙版")
                return False
            self.doc.detach_pixels(layer)
            h, w = layer.mask.shape
            tmp = np.zeros((h, w, 4), np.uint8)
            tmp[..., 0] = tmp[..., 1] = tmp[..., 2] = layer.mask
            tmp[..., 3] = 255
            if not retouch.heal_region(tmp, x, y, p["size"] / 2.0,
                                      p["hardness"], p["strength"]):
                return False
            layer.mask = tmp[..., 0]
        else:
            if layer.is_smart:
                self.status.showMessage(
                    "智能对象不能直接修复 —— 用「编辑内容」或先栅格化", 5000)
                return False
            self.doc.detach_pixels(layer)
            Minv, scale = self._retouch_src_layer("污点修复", layer)
            src = cv2.transform(np.array([[[x, y]]], np.float32), Minv)[0, 0]
            if not retouch.heal_region(layer.image, src[0], src[1],
                                       p["size"] / 2.0 * scale,
                                       p["hardness"], p["strength"]):
                return False
        self.after_pixel_edit("污点修复")
        return True

    def draw_shape(self, p0, p1, params=None, modifiers=None):
        """形状工具：在 p0→p1 之间落一个矩形 / 椭圆 / 直线。

        `modifiers` 里 Shift 表示等比 / 正圆 / 45° 直线（和选区工具一致）。
        """
        layer = self._retouch_target_layer("形状")
        if layer is None:
            return False
        p = params or self.opts.shape_params()
        x0, y0 = float(p0[0]), float(p0[1])
        x1, y1 = float(p1[0]), float(p1[1])
        shift = bool(modifiers and modifiers & Qt.ShiftModifier)
        kind = p["kind"]
        Minv, scale = self._retouch_src_layer("形状", layer)
        a = cv2.transform(np.array([[[x0, y0]]], np.float32), Minv)[0, 0]
        b = cv2.transform(np.array([[[x1, y1]]], np.float32), Minv)[0, 0]
        ax, ay = float(a[0]), float(a[1])
        bx, by = float(b[0]), float(b[1])
        if shift:
            if kind == "直线":
                # 45° 吸附：取 dx / dy 里绝对值大的那个作为主轴
                if abs(bx - ax) >= abs(by - ay):
                    by = ay
                else:
                    bx = ax
            else:
                # 等比：取较大的边长当边长
                s = max(abs(bx - ax), abs(by - ay))
                bx = ax + (s if bx >= ax else -s)
                by = ay + (s if by >= ay else -s)

        h, w = layer.image.shape[:2]
        ix0, iy0 = int(round(min(ax, bx))), int(round(min(ay, by)))
        ix1, iy1 = int(round(max(ax, bx))), int(round(max(ay, by)))
        ix0, iy0 = max(0, ix0), max(0, iy0)
        ix1, iy1 = min(w, ix1), min(h, iy1)
        if ix1 <= ix0 or iy1 <= iy0:
            self.status.showMessage("形状太小了")
            return False
        bw = max(1, int(round(abs(bx - ax) * max(scale, 0.5))))
        bh = max(1, int(round(abs(by - ay) * max(scale, 0.5))))
        sel = np.zeros((iy1 - iy0, ix1 - ix0), np.float32)
        yy, xx = np.mgrid[iy0:iy1, ix0:ix1]
        cx = (ax + bx) / 2.0
        cy = (ay + by) / 2.0
        # 先建一个全 False 的：Python 判定局部变量看整个函数体，
        # 分支里才赋值的话另一个分支读它会 UnboundLocalError
        inside = np.zeros(sel.shape, bool)
        if kind == "椭圆":
            inside = (((xx + 0.5 - cx) / (bw / 2.0 + 1e-6)) ** 2 +
                      ((yy + 0.5 - cy) / (bh / 2.0 + 1e-6)) ** 2) <= 1.0
        elif kind == "直线":
            # 直线：沿方向的矩形带，宽度就是笔刷大小
            if abs(by - ay) <= 1e-6:
                inside = np.abs(yy + 0.5 - ay) <= bh / 2.0
            else:
                t = (yy + 0.5 - ay) / (by - ay)
                cx_line = ax + (bx - ax) * t
                inside = np.abs(xx + 0.5 - cx_line) <= bw / 2.0
        else:
            inside = np.ones(sel.shape, bool)
        # fill_rgba 要求 sel 与 image 同尺寸，把形状框铺回整幅
        # （用 numpy 而不是循环，避免一张 12 MP 的临时数组）
        full = np.zeros((h, w), np.float32)
        full[iy0:iy1, ix0:ix1] = inside.astype(np.float32)
        if p["outline"]:
            # **必须给 borderType / borderValue**：默认的 border 处理会把
            # 外沿也吃掉，描边就没了（实测整圈全 0）。给 BORDER_CONSTANT=0
            # 之后「画布外当作空」-> 外沿保住了，这正是描边要的
            er = cv2.erode(full, np.ones((3, 3), np.uint8),
                           iterations=max(1, int(round(min(bw, bh) * 0.06))),
                           borderType=cv2.BORDER_CONSTANT, borderValue=0)
            full = np.clip(full - er.astype(np.float32), 0.0, 1.0)
        sel = full
        # 选区约束（画布坐标的选区换算过来）
        user_sel = self._src_selection(layer)
        if user_sel is not None:
            sel = sel * user_sel
        color = self.opts.fg.rgb()
        self.doc.detach_pixels(layer)
        fill_rgba(layer.image, sel, color)
        self.after_pixel_edit("形状")
        return True

    # ---------- 向导：参考线 / 网格 / 吸附（第二十一批）----------

    def add_guide(self, kind, pos):
        """新增一条参考线。返回 Guide 或 None（没文档时）。"""
        if self.doc is None:
            return None
        g = self.doc.guides.add(kind, float(pos))
        self.commit("添加参考线")
        self.view.rebuild_guide_items()
        self.status.showMessage(
            "已添加%s参考线 @ %.0f（拖动可移动，Alt+点击可删除）"
            % ("水平" if kind == _guides.H_GUIDE else "垂直", g.pos), 3000)
        return g

    def move_guide(self, gid, pos):
        """移动一条参考线。拖动过程中已经直接改过pos，这里只落撤销点。"""
        g = self.doc.guides.find(gid) if self.doc is not None else None
        if g is None:
            return False
        g.pos = float(pos)
        self.commit("移动参考线")
        self.view.rebuild_guide_items()
        return True

    def delete_guide(self, gid):
        if self.doc is None or not self.doc.guides.remove(gid):
            return False
        self.commit("删除参考线")
        self.view.rebuild_guide_items()
        self.status.showMessage("已删除参考线", 2000)
        return True

    def clear_guides(self):
        if self.doc is None or not self.doc.guides.guides:
            return False
        self.doc.guides.clear()
        self.commit("清除全部参考线")
        self.view.rebuild_guide_items()
        self.status.showMessage("已清除全部参考线", 2000)
        return True

    def set_rulers(self, on):
        """标尺显隐。开了之后画布可视区会内缩标尺的宽度。"""
        self.view.show_rulers = bool(on)
        self.view.update()
        self.status.showMessage("标尺：%s" % ("显示" if on else "隐藏"), 2000)

    def set_grid(self, on=None, spacing=None, subdiv=None):
        """网格开关 / 间距 / 细分。参数为 None 表示不改那项。"""
        gs = self.doc.guides if self.doc is not None else None
        if gs is None:
            return False
        if on is not None:
            gs.grid.visible = bool(on)
        if spacing is not None:
            gs.grid.spacing = max(1.0, float(spacing))
        if subdiv is not None:
            gs.grid.subdiv = max(1, int(subdiv))
        self.view.sync_grid()
        self.commit("网格设置")
        self.status.showMessage(
            "网格：%s · 间距 %.0f px · %d 等分"
            % ("显示" if gs.grid.visible else "隐藏", gs.grid.spacing,
               gs.grid.subdiv), 3000)
        return True

    def set_snap(self, which=None, on=None):
        """吸附开关。which 取 doc / guides / grid / layers / all。"""
        gs = self.doc.guides if self.doc is not None else None
        if gs is None:
            return False
        names = {"doc": "snap_doc", "guides": "snap_guides",
                 "grid": "snap_grid", "layers": "snap_layers"}
        if which is None or which == "all":
            for attr in names.values():
                setattr(gs, attr, bool(on))
        else:
            attr = names.get(which)
            if attr is None:
                return False
            setattr(gs, attr, bool(on))
        self.view.sync_grid()
        self.view.viewport().update()
        return True

    def pick_at(self, x, y, params=None):
        """吸管：取色到前景色。返回 (r,g,b) 或 None。"""
        p = params or self.opts.pick_params()
        if quick_mask.is_on(self.doc):
            qm = self.doc.quick_mask
            src = np.dstack([qm, np.full(qm.shape, 255, np.uint8)])
            got = retouch.pick_color(src, x, y, p["radius"])
            if got is None:
                self.status.showMessage("取样失败：点在画布外", 3000)
                return None
            g = int(got[0])
            self.opts.fg.set_color((g, g, g))
            self.status.showMessage("已取样 灰度 %d（快速蒙版遮罩）" % g, 3000)
            return (g, g, g)
        from_composite = bool(p["composite"]) and self._last_arr is not None
        arr = self._last_arr if from_composite else (
            self.selected_layer().image if self.selected_layer() else None)
        got = retouch.pick_color(arr, x, y, p["radius"])
        if got is None:
            self.status.showMessage("取样失败：点在画布外或这里没有内容", 3000)
            return None
        r, g, b = int(got[0]), int(got[1]), int(got[2])
        self.opts.fg.set_color((r, g, b))
        where = "合成结果" if from_composite else "当前图层"
        self.status.showMessage("已取样 %s RGB(%d, %d, %d)" % (where, r, g, b),
                                3000)
        return (r, g, b)

    # ---------- 蒙版 ----------

    def add_mask(self, kind):
        layer = self.selected_layer()
        if layer is None or layer.is_group:
            QMessageBox.information(self, "提示",
                                    "请选择一个图层。图层组暂不支持蒙版。")
            return
        if layer.is_adjustment:
            # 调整层的蒙版按画布尺寸建立（它没有源图像）
            if kind in ("alpha", "luma"):
                QMessageBox.information(
                    self, "提示", "调整层没有像素，无法从透明度或明度建立蒙版。")
                return
            shape = (self.doc.height, self.doc.width)
            if kind == "invert":
                if layer.mask is None:
                    return
                layer.mask = (255 - layer.mask).astype(np.uint8)
                self.commit("反相蒙版")
            elif kind == "white":
                layer.mask = np.full(shape, 255, np.uint8)
                layer.mask_enabled = True
                self.commit("添加蒙版")
            elif kind == "black":
                layer.mask = np.zeros(shape, np.uint8)
                layer.mask_enabled = True
                self.commit("添加蒙版")
            self.panel.rebuild()
            self.request_render()
            return

        if layer.image is None:
            QMessageBox.information(self, "提示", "请选择一个位图图层。")
            return
        img = layer.image
        if kind == "invert":
            if layer.mask is None:
                return
            layer.mask = (255 - layer.mask).astype(np.uint8)
            self.commit("反相蒙版")
        elif kind == "white":
            layer.mask = np.full(img.shape[:2], 255, np.uint8)
            layer.mask_enabled = True
            self.commit("添加蒙版")
        elif kind == "black":
            layer.mask = np.zeros(img.shape[:2], np.uint8)
            layer.mask_enabled = True
            self.commit("添加蒙版")
        elif kind == "alpha":
            layer.mask = img[..., 3].copy()
            layer.mask_enabled = True
            self.commit("从透明度建立蒙版")
        elif kind == "luma":
            rgb = img[..., :3].astype(np.float32)
            m = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])
            layer.mask = np.clip(m, 0, 255).astype(np.uint8)
            layer.mask_enabled = True
            self.commit("从明度建立蒙版")
        self.panel.rebuild()
        self.request_render()

    def toggle_mask(self):
        layer = self.selected_layer()
        if layer is None or layer.mask is None:
            return
        layer.mask_enabled = not layer.mask_enabled
        self.commit("切换蒙版")
        self.panel.rebuild()
        self.request_render()

    def delete_mask(self):
        layer = self.selected_layer()
        if layer is None or layer.mask is None:
            return
        layer.mask = None
        self.commit("删除蒙版")
        self.panel.rebuild()
        self.request_render()

    # ---------- 画布 ----------

    def trim_to_content(self):
        if self.doc is None:
            return
        box = self.doc.content_bbox()
        if box is None:
            return
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        w = max(1, x1 - x0)
        h = max(1, y1 - y0)
        if (w, h) == (self.doc.width, self.doc.height) and x0 == 0 and y0 == 0:
            return

        def shift(layer, dx, dy):
            if layer.is_group:
                for c in layer.children:
                    shift(c, dx, dy)
            else:
                layer.tx += dx
                layer.ty += dy

        for l in self.doc.layers:
            shift(l, -x0, -y0)
        if self.doc.selection is not None:
            self.doc.detach_selection()
            self.doc.selection.translate(-x0, -y0)
        self.doc.resize(w, h)
        self.commit("按内容裁剪")
        self.inspector.refresh()
        self.request_render()
        self.view.update_selection_overlay()

    # ---------- 内容感知填充（core/content_aware.py）----------

    def ask_content_aware_fill(self):
        """选择 → 内容感知填充…：用选区周围的像素推断选区里该是什么。

        和 `fill_with_fg` 的区别：不填颜色，**填内容** —— 挖掉一块电线杆，
        填回来的是它背后的天空/墙面纹理。

        对话框开着的时候反复重算（纹理合成模式下一次要几百 ms，所以做在
        `_on_caf_preview` 里带 try/except，失败就跳过这一帧）；
        「确定」才落撤销点，「取消」把原图整块写回去。
        """
        if not self._require_doc():
            return
        layer = self._retouch_target_layer("内容感知填充")
        if layer is None:
            return
        sel = self._sel()
        if sel is None or sel.is_empty:
            self.status.showMessage("先做一个选区 —— 内容感知填充靠选区定位", 5000)
            return
        src = layer.image.copy()          # 破坏性试错的底稿
        lid = layer.id
        src_sel = self._src_selection(layer)

        dlg = ContentAwareDialog(self)
        dlg.paramsChanged.connect(
            lambda p: self._on_caf_preview(lid, src, src_sel, p))
        if dlg.exec() != QDialog.DialogCode.Accepted:
            # 取消：把底稿写回去。**必须 doc.find(id) 重取** ——
            # 预览期间 commit 过的话，layer 已经是新对象了
            cur = self.doc.find(lid)
            if cur is not None:
                cur.image = src
                self.request_render()
            return
        p = dlg.values()
        cur = self.doc.find(lid)
        if cur is None or cur.image is None:
            return
        self.doc.detach_pixels(cur)
        cur.image = fill_content_aware(cur.image, src_sel, **p)
        self.after_pixel_edit("内容感知填充")

    def _on_caf_preview(self, lid, src, src_sel, params):
        """对话框的实时预览。失败（超时/超大选区）就跳过这一帧。"""
        if self.doc is None:
            return
        cur = self.doc.find(lid)
        if cur is None or cur.image is None:
            return
        self.doc.detach_pixels(cur)
        try:
            out = fill_content_aware(src, src_sel, **params)
        except (cv2.error, MemoryError, ValueError):
            return
        cur.image = out
        self.request_render()

    def ask_canvas_size(self):
        """图像 → 画布大小…：给画布加边。

        新加出来的边有三种填法：
        - 前景色 / 透明 / 白色
        - **内容感知**：把边缘"长"出去 —— 这是 Photoshop 的
          「扩展画布外的空白」，本项目对**所有位图图层**都做一遍，
          所以边缘看起来是连续的，而不是突然多一圈。
        """
        if not self._require_doc():
            return
        from .canvas_size_dialog import CanvasSizeDialog
        dlg = CanvasSizeDialog(self, self.doc,
                               color=tuple(self.opts.fg.rgb()))
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        vals = dlg.values()
        self.resize_canvas(**vals)
        self.commit("画布大小")
        self.inspector.refresh()
        self.request_render()
        self.view.update_selection_overlay()

    def resize_canvas(self, width, height, anchor="居中", fill="内容感知",
                      color=None, mode="邻近"):
        """真正干活的画布大小。anchor 是 3x3 锚点（九宫格）。

        内容感知那一条路：先算出各边**扩出多少像素**，逐个位图图层
        各自 `extend_canvas_content_aware` 一次。之所以要逐层做而不是
        只在合成结果上做 —— 图层是可以分开编辑的，合成图上长出来的
        边缘没法分配回各层。
        """
        from ..core.document import _ANCHOR_OFFSETS
        ow, oh = self.doc.width, self.doc.height
        nw, nh = max(1, int(width)), max(1, int(height))
        # 锚点是 0~1 的比例，要按**新旧尺寸之差**换算成整数像素。
        # 直接拿比例乘新尺寸会在「只扩 10px」时算出 0.5px，取整就丢了一边。
        #
        # dx 是**原图整体右移多少**：fx=0（左上）不移，fx=1（右下）全移。
        # 所以左边扩出来的是 dx，右边扩出来的是差值剩下的那部分 ——
        # 两个都要独立算，只看其中一个会在 fx=0 或 1 时漏掉一整边。
        fx, fy = _ANCHOR_OFFSETS[anchor]
        grow_x, grow_y = nw - ow, nh - oh
        dx = int(round(fx * grow_x))
        dy = int(round(fy * grow_y))
        left, top = max(0, dx), max(0, dy)
        right = max(0, grow_x - dx)
        bottom = max(0, grow_y - dy)

        sel = self.doc.selection.float_mask() if self.doc.selection is not None else None

        if fill == "内容感知" and (left or top or right or bottom):
            for layer in list(self.doc.all_layers()):
                if layer.is_group or layer.image is None:
                    continue
                self.doc.detach_pixels(layer)
                h, w = layer.image.shape[:2]
                layer.image = extend_canvas_content_aware(
                    layer.image, sel, left, top, right, bottom,
                    mode=mode, axis="水平")
                # **tx/ty 是「源图中心在画布里的位置」，不是左上角**
                # （见 layer.matrix()：tx - a*cx）。所以要跟着新尺寸重算：
                # 原来 tx = w/2 表示左边贴画布左边，扩完左边就变成 left
                layer.tx = layer.image.shape[1] / 2.0 + left
                layer.ty = layer.image.shape[0] / 2.0 + top

        elif fill != "内容感知":
            col = np.array(color if color is not None else (255, 255, 255),
                           np.uint8)
            pad_val = (int(col[0]), int(col[1]), int(col[2]),
                       0 if fill == "透明" else 255)
            for layer in list(self.doc.all_layers()):
                if layer.is_group or layer.image is None:
                    continue
                self.doc.detach_pixels(layer)
                h, w = layer.image.shape[:2]
                # **tx/ty 是源图中心在画布里的位置，不是左上角**
                # （见 layer.matrix()：tx - a*cx，cx = w/2）。
                # 所以画布往左扩 left 个像素 = 位图整体右移 left，
                # 位图本身要**左侧补 left 列**。补完再裁掉落到画布外的部分。
                lx0 = int(round(layer.tx - w / 2.0)) + left
                ly0 = int(round(layer.ty - h / 2.0)) + top
                pl = max(0, -lx0)          # 左侧要补多少列
                pt = max(0, -ly0)
                pr = max(0, lx0 + w - nw)  # 右侧溢出多少就裁/补多少
                pb = max(0, ly0 + h - nh)
                # 溢出量也可能是负的（图层比新画布小）——那就补边
                pw = max(pr, 0) + max(0, nw - (lx0 + max(w, 0))) - pl
                ph = max(pb, 0) + max(0, nh - (ly0 + max(h, 0))) - pt
                if pl or pt or pw > 0 or ph > 0:
                    layer.image = cv2.copyMakeBorder(
                        layer.image, pt, max(0, ph), pl, max(0, pw),
                        cv2.BORDER_CONSTANT, value=pad_val)
                # 裁掉落在新画布外的部分（画布变小的情况）
                nh_, nw_ = layer.image.shape[:2]
                cx0 = max(0, -lx0)
                cy0 = max(0, -ly0)
                cx1 = int(max(cx0, min(nw_, nw - lx0)))
                cy1 = int(max(cy0, min(nh_, nh - ly0)))
                if cx1 <= cx0 or cy1 <= cy0:
                    continue        # 整层都在画布外，交给图层变换自己处理
                if (cx0, cy0, cx1, cy1) != (0, 0, nw_, nh_):
                    layer.image = layer.image[cy0:cy1, cx0:cx1].copy()
                # 新的左上角在画布里的位置 -> 换算回中心
                ox = lx0 + cx0
                oy = ly0 + cy0
                layer.tx = ox + layer.image.shape[1] / 2.0
                layer.ty = oy + layer.image.shape[0] / 2.0

        self.doc.resize(nw, nh)


class SmartObjectEditor(MainWindow):
    """智能对象内容的独立编辑窗口。

    直接复用整主窗口 —— 图层面板、工具栏、画布全是同一套实现，
    用户不需要学第二套操作。窗口里编辑的就是内容本身（`sub.layers` 与
    `content.layers` 是同一批对象），所以关闭之后主窗口只要 `touch()` 一下
    让栅格缓存失效即可。

    这个窗口不是模态的：可以同时开着几个智能对象互相对照。
    """

    def __init__(self, parent, owner, content, doc):
        super().__init__()
        self._owner = owner
        self._content = content
        self.content_id = content.id
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.setWindowTitle("智能对象：%s  (%d x %d)"
                            % (content.name, content.width, content.height))
        self.resize(900, 700)
        self.set_document(doc, reset_history=True)
        self.set_tool("move")
        if doc.layers:
            self.select_layer(doc.layers[-1].id)
        self.view.fit()
        self.status.showMessage(
            "编辑智能对象内容 —— 关闭窗口即保存，所有引用它的实例会一起更新。\n"
            "注意：画布尺寸是内容尺寸，改不了（要改请回主画布缩放图层）", 8000)

    def closeEvent(self, event):
        # 内容可能被整个换掉了（比如用户在里面删光图层又新建）
        self._content.layers = list(self.doc.layers)
        try:
            self._owner._on_smart_edited(self._content)
        except Exception:
            pass
        try:
            self._owner._so_editors.remove(self)
        except ValueError:
            pass
        super().closeEvent(event)
