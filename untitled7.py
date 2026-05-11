# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 20:14:05 2025

@author: RDH
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 11:13:22 2025

@author: RDH
"""

# -*- coding: utf-8 -*-
"""
Asphalt Infrared Recognition App - PyQt5 Version
Refactored from Tkinter by Gemini, inspired by SlopeWarningApp style
Further enhancements based on user feedback (UI layout and terminology).
CO2 peak removal (2000-2500 cm-1) functionality added.

2025-10-10 更新（盲样线性系列 -> 混掺）：
- 局部混掺识别遵循“三步”：
  1) 对“选中组”的质心(身份位置1/2)做线性性检测（PCA第一主成分解释率 + 归一化正交残差/RMS）。
  2) 若线性明显，则以主方向上两端组为端元（来源1/来源2）。
  3) 仅对两端之间的中间组做投影配比；端点不判混掺。

2025-10-13 更新（原油及沥青标号预测模块）：
- 新增 Tab「原油及沥青标号预测」：
  - 复选数据库（标样），自动端元识别（不锚定具体名称）
  - 按钮“加载针入度值”读取 CSV（Group,Penetration[0.1mm]）
  - 复选盲样，选择回归器（线性/二次/单调），点击“匹配”
  - 输出：来源1/来源2 + 百分比、预测针入度、预测标号（内置规则）
  - 自动导出：原油及沥青标号预测.csv

2025-10-29 更新（三性能映射与预测模块 + 标签页切换修复）：
- 新增 Tab「三性能映射与预测」：
  - 加载沥青性能表 CSV（智能识别列名，如 针入度/软化点/延度 或 Penetration/SofteningPoint/Ductility）
  - 复选数据库（标样）与复选盲样
  - 选择回归器（线性 / 二次 / 三次 多项式 + Ridge），分别拟合三指标
  - 输出预测针入度、软化点、延度及标号，导出 三性能预测.csv
- 仅在“局部分析完成”时自动切到对应标签；其它任务保持当前标签页不变

2025-10-29 晚（本次更新，满足用户五点需求）：
- 修复：任何任务完成后不再“自动切回主页面”，统一恢复到任务启动前的标签页（局部分析除外，仍自动切换到局部标签）。
- 恢复：2D「三性能映射与预测」页新增“模型与训练”区，显示训练集真值/拟合/误差表与RMSE汇总。
- 新增：Tab「3D PCA交互图」（可旋转/缩放/复选样品 + 自动缩放 + 每个样品文本标签）。
- 新增：Tab「3D 相似度检测」（PC1-3三维质心距离 + 平均谱PCA距离 融合，导出矩阵&对照表）。
- 新增：Tab「3D 原油及沥青标号预测」（与2D一致的线性系列端元自动识别与t→针入度回归，但在3D空间）。
- 新增：Tab「3D 三性能映射与预测」（[PC1,2,3]→三性能，多项式Ridge；含训练表与RMSE）。
"""

import sys
import os
import time
import threading
import queue

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QTextEdit, QAction,
                             QStatusBar, QGridLayout, QGroupBox, QListWidget, QAbstractItemView,
                             QTabWidget, QMessageBox, QProgressBar, QTreeWidget, QTreeWidgetItem,
                             QHeaderView, QSizePolicy, QRadioButton, QButtonGroup, QCheckBox)
from PyQt5.QtGui import QFont, QColor, QTextCursor, QFontMetrics
from PyQt5.QtCore import Qt, QTimer, QDateTime

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # needed for 3D

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 供标号预测与三性能映射用到的模型
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

# --- Constants for queue message types ---
MSG_LOG = "log_message"
MSG_PROGRESS_START = "progress_start"
MSG_PROGRESS_STOP = "progress_stop"
MSG_SHOW_MESSAGEBOX = "show_messagebox"
MSG_BUTTON_STATE = "button_state"
MSG_CLEAR_UI_BEFORE_PROCESSING = "clear_ui_before_processing"
MSG_POPULATE_CATEGORY_LIST = "populate_category_list"
MSG_DISPLAY_GLOBAL_IDENTITY_POSITIONS_DATA = "display_global_identity_positions_data"
MSG_DISPLAY_LOCAL_IDENTITY_POSITIONS_DATA = "display_local_identity_positions_data"
MSG_UPDATE_MIXTURE_TABLE_LOCAL = "update_mixture_table_local"
MSG_UPDATE_SIMILARITY_TABLE = "update_similarity_table"
MSG_PROCESSING_COMPLETE = "processing_complete"
MSG_LOCAL_PROCESSING_COMPLETE = "local_processing_complete"

# 新增：原油及沥青标号预测页 & 三性能映射与预测页的结果刷新
MSG_UPDATE_OIL_GRADE_RESULTS = "update_oil_grade_results"
MSG_UPDATE_TRIPLE_RESULTS = "update_triple_results"
# 新增：2D 三性能训练表
MSG_UPDATE_TRIPLE_TRAIN_TABLE = "update_triple_train_table"

# ===== 新增：3D 模块 =====
MSG_UPDATE_SIMILARITY3D_TABLE = "update_similarity3d_table"
MSG_UPDATE_OIL_GRADE3D_RESULTS = "update_oil_grade3d_results"
MSG_UPDATE_TRIPLE3D_RESULTS = "update_triple3d_results"
MSG_UPDATE_TRIPLE3D_TRAIN_TABLE = "update_triple3d_train_table"

RESIZE_DEBOUNCE_MS_PYQT = 250

# --- Column Names for Identity Positions ---
IDENTITY_POSITION_COL_1 = "身份位置1"
IDENTITY_POSITION_COL_2 = "身份位置2"

# ---------- 盲样线性系列 & 混掺识别 阈值参数 ----------
MIN_GROUPS_FOR_SERIES = 3            # 至少3个组才有意义做线性检测
LINEAR_PCA_RATIO_MIN = 0.96          # PCA第一主成分解释率阈值（线性显著性）
LINEAR_RMS_PERP_RATIO_MAX = 0.08     # 质心到拟合直线RMS正交残差 / 端元距离 的上限
LINEAR_MAX_PERP_RATIO_MAX = 0.12     # 单组最大正交残差比例上限
ENDPOINT_TOL = 0.08                  # 两端判定容差
TOP_LINES_TO_PLOT = 1                # GUI最多画几条参考线
EPS_AB_MIN = 1e-6                    # 端元距离下限
# -------------------------------------------------------

# ---------- 内置标号规则（单位：0.1 mm） ----------
BUILTIN_GRADE_RULES = [
    {"Label": "190#", "Low": 180, "High": 200},
    {"Label": "170#", "Low": 160, "High": 180},
    {"Label": "130#", "Low": 120, "High": 140},
    {"Label": "110#", "Low": 100, "High": 120},
    {"Label": "90#",  "Low":  80, "High": 100},
    {"Label": "70#",  "Low":  60, "High":  80},
    {"Label": "50#",  "Low":  40, "High":  60},
    {"Label": "30#",  "Low":  20, "High":  40},
    {"Label": "10#",  "Low":   0, "High":  20},
]
# -------------------------------------------------------

# --------- 工具函数：鲁棒列名匹配 ---------
def _find_col(df_cols, candidates):
    cols_lower_map = {str(c).strip().lower(): c for c in df_cols}
    for cand in candidates:
        key = str(cand).strip().lower()
        if key in cols_lower_map:
            return cols_lower_map[key]
    def _norm(s):
        s = str(s)
        for ch in "（）()[]{}":
            s = s.replace(ch, " ")
        return "".join(s.split()).lower()
    norm_map = {_norm(c): c for c in df_cols}
    for cand in candidates:
        kk = _norm(cand)
        if kk in norm_map:
            return norm_map[kk]
    return None


class MatplotlibCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100, is3d=False):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        if is3d:
            self.axes = self.fig.add_subplot(111, projection='3d')
        else:
            self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()

    def clear_axes(self):
        self.axes.clear()
        self.draw_idle()

    def redraw_plot(self):
        try:
            self.fig.tight_layout(pad=0.5)
            self.draw_idle()
        except Exception as e:
            print(f"Error during MatplotlibCanvas redraw: {e}")


class PlotContainerWidget(QWidget):
    def __init__(self, parent=None, is3d=False, with_toolbar=False):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        self.canvas = MatplotlibCanvas(self, is3d=is3d)
        if with_toolbar:
            self.toolbar = NavigationToolbar(self.canvas, self)
            layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._perform_debounced_redraw)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._resize_timer.start(RESIZE_DEBOUNCE_MS_PYQT)

    def _perform_debounced_redraw(self):
        if self.canvas and self.canvas.fig:
            if self.width() < 10 or self.height() < 10 : return
            self.canvas.redraw_plot()

    def get_figure(self): return self.canvas.fig
    def get_axes(self): return self.canvas.axes
    def get_canvas(self): return self.canvas


class AsphaltInfraredAppPyQt(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("沥青身份来源智能识别技术 (PyQt5版)")
        self.setGeometry(100, 100, 1600, 980)

        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        self.file_path = None
        self.output_directory = os.getcwd()
        self.data_processing_active = False
        self._current_worker_thread = None
        self.task_queue = queue.Queue()

        # 记忆：任务启动前的标签页索引（用于“完成后返回”）
        self._last_tab_index = 0

        # 針入度标注表（用于标号预测）
        self.penetration_map = {}  # dict: Group -> Penetration(0.1mm)

        # 三性能表（Group -> dict{Penetration, SofteningPoint, Ductility}）
        self.performance_map = {}

        self.init_ui()
        self.create_menu_bar()
        self.create_status_bar()

        self.queue_poll_timer = QTimer(self)
        self.queue_poll_timer.timeout.connect(self._poll_queue)
        self.queue_poll_timer.start(100)

        self._add_log_message("系统已启动。请选择数据文件并开始计算。", "INFO")

    # ---------------- UI ----------------
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- Left Panel ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setFixedWidth(460)

        # File Operations
        file_ops_group = QGroupBox("数据文件")
        file_ops_layout = QGridLayout(file_ops_group)
        self.btn_open = QPushButton("📁 选择数据文件")
        self.btn_open.clicked.connect(self.open_file)
        self.lbl_file_path_status = QLabel("未选择文件")
        self.lbl_file_path_status.setWordWrap(True)
        file_ops_layout.addWidget(self.btn_open, 0, 0)
        file_ops_layout.addWidget(self.lbl_file_path_status, 0, 1, 1, 2)
        left_layout.addWidget(file_ops_group)

        # Processing
        processing_group = QGroupBox("分析与计算")
        processing_layout = QGridLayout(processing_group)
        self.btn_process = QPushButton("⚙️ 开始识别计算")
        self.btn_process.clicked.connect(self.start_global_processing_thread)
        processing_layout.addWidget(self.btn_process, 0, 0, 1, 3)

        self.btn_show_identity_plot = QPushButton("📊 身份位置图(导出)")
        self.btn_show_identity_plot.clicked.connect(self.show_exported_identity_positions_plot)
        processing_layout.addWidget(self.btn_show_identity_plot, 1, 0)

        self.btn_update_local_identity_positions = QPushButton("🔄 更新局部分析")
        self.btn_update_local_identity_positions.clicked.connect(self.start_local_identity_positions_update_thread)
        processing_layout.addWidget(self.btn_update_local_identity_positions, 1, 1)

        self.btn_show_matrix = QPushButton("🔢 相似度矩阵(导出)")
        self.btn_show_matrix.clicked.connect(self.show_similarity_matrix_file)
        processing_layout.addWidget(self.btn_show_matrix, 2, 0)

        self.btn_show_standard = QPushButton("📋 对照表(导出)")
        self.btn_show_standard.clicked.connect(self.show_similarity_standard_file)
        processing_layout.addWidget(self.btn_show_standard, 2, 1)

        left_layout.addWidget(processing_group)

        # Category Selection（用于旧的“局部分析”）
        category_group = QGroupBox("沥青类别选择 (用于局部分析)")
        category_layout = QVBoxLayout(category_group)
        self.category_list_widget = QListWidget()
        self.category_list_widget.setSelectionMode(QAbstractItemView.MultiSelection)
        self.category_list_widget.setFixedHeight(240)
        category_layout.addWidget(self.category_list_widget)
        cat_btn_layout = QHBoxLayout()
        self.btn_select_all = QPushButton("全选")
        self.btn_select_all.clicked.connect(self.category_list_widget.selectAll)
        cat_btn_layout.addWidget(self.btn_select_all)
        self.btn_deselect_all = QPushButton("全不选")
        self.btn_deselect_all.clicked.connect(self.category_list_widget.clearSelection)
        cat_btn_layout.addWidget(self.btn_deselect_all)
        category_layout.addLayout(cat_btn_layout)
        left_layout.addWidget(category_group, 2)

        # 状态区
        status_log_group = QGroupBox("运行状态")
        status_log_layout = QVBoxLayout(status_log_group)
        status_log_layout.setContentsMargins(5,5,5,5)
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(15)
        status_log_layout.addWidget(self.progress_bar)
        self.log_output_area = QTextEdit()
        self.log_output_area.setReadOnly(True)
        font_metrics = QFontMetrics(self.log_output_area.font())
        line_height = font_metrics.height()
        self.log_output_area.setFixedHeight(line_height * 3 + 10)
        status_log_layout.addWidget(self.log_output_area)
        left_layout.addWidget(status_log_group, 0)
        left_layout.addStretch(1)
        left_panel.setLayout(left_layout)

        # --- Right Panel ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0,0,0,0)
        self.notebook = QTabWidget()
        right_layout.addWidget(self.notebook, 1)
        right_panel.setLayout(right_layout)

        # Tab 1: Global 2D
        self.tab_global_identity_positions = QWidget()
        self.notebook.addTab(self.tab_global_identity_positions, "全局身份位置分布图")
        global_id_pos_layout = QVBoxLayout(self.tab_global_identity_positions)
        self.global_identity_positions_plot_container = PlotContainerWidget(self.tab_global_identity_positions, is3d=False, with_toolbar=True)
        self.global_identity_positions_plot_container.setObjectName("GlobalIdentityPositionsPlotContainer")
        global_id_pos_layout.addWidget(self.global_identity_positions_plot_container)

        # Tab 2: Local + Mixture
        self.tab_local_identity_positions = QWidget()
        self.notebook.addTab(self.tab_local_identity_positions, "局部身份位置与混掺分析")
        local_id_pos_tab_layout = QVBoxLayout(self.tab_local_identity_positions)
        self.local_identity_positions_plot_container = PlotContainerWidget(self.tab_local_identity_positions, is3d=False, with_toolbar=True)
        self.local_identity_positions_plot_container.setObjectName("LocalIdentityPositionsPlotContainer")
        local_id_pos_tab_layout.addWidget(self.local_identity_positions_plot_container, 3)
        mixture_group_in_tab = QGroupBox("混掺识别结果 (基于当前局部分析)")
        mixture_group_layout = QVBoxLayout(mixture_group_in_tab)
        self.mixture_tree_local = QTreeWidget()
        self._setup_qtree_widget_columns(self.mixture_tree_local, "mixture")
        mixture_group_layout.addWidget(self.mixture_tree_local)
        local_id_pos_tab_layout.addWidget(mixture_group_in_tab, 2)

        # Tab 3: Similarity (2D同原逻辑)
        self.tab_similarity = QWidget()
        self.notebook.addTab(self.tab_similarity, "相似度对照表 (2D)")
        similarity_layout = QVBoxLayout(self.tab_similarity)
        self.similarity_tree = QTreeWidget()
        self._setup_qtree_widget_columns(self.similarity_tree, "similarity")
        similarity_layout.addWidget(self.similarity_tree)

        # === Tab 4: 原油及沥青标号预测 (2D) ===
        self.tab_oil_grade = QWidget()
        self.notebook.addTab(self.tab_oil_grade, "原油及沥青标号预测 (2D)")
        oil_layout = QVBoxLayout(self.tab_oil_grade)

        db_box = QGroupBox("选择数据库（标样，用于自动识别端元与训练 t→针入度）")
        db_v = QVBoxLayout(db_box)
        self.db_list_widget = QListWidget()
        self.db_list_widget.setSelectionMode(QAbstractItemView.MultiSelection)
        db_v.addWidget(self.db_list_widget)
        db_btn_row = QHBoxLayout()
        self.btn_db_select_all = QPushButton("全选")
        self.btn_db_select_all.clicked.connect(self.db_list_widget.selectAll)
        db_btn_row.addWidget(self.btn_db_select_all)
        self.btn_db_clear = QPushButton("全不选")
        self.btn_db_clear.clicked.connect(self.db_list_widget.clearSelection)
        db_btn_row.addWidget(self.btn_db_clear)
        self.btn_load_pen_csv = QPushButton("📥 加载针入度值(CSV)")
        self.btn_load_pen_csv.clicked.connect(self.load_penetration_csv_clicked)
        db_btn_row.addWidget(self.btn_load_pen_csv)
        db_v.addLayout(db_btn_row)
        self.lbl_pen_loaded = QLabel("未加载针入度表")
        db_v.addWidget(self.lbl_pen_loaded)
        oil_layout.addWidget(db_box)

        blind_box = QGroupBox("选择盲样（将基于上方数据库的端元线做投影匹配）")
        blind_v = QVBoxLayout(blind_box)
        self.blind_list_widget = QListWidget()
        self.blind_list_widget.setSelectionMode(QAbstractItemView.MultiSelection)
        blind_v.addWidget(self.blind_list_widget)
        blind_btn_row = QHBoxLayout()
        self.btn_blind_select_all = QPushButton("全选")
        self.btn_blind_select_all.clicked.connect(self.blind_list_widget.selectAll)
        blind_btn_row.addWidget(self.btn_blind_select_all)
        self.btn_blind_clear = QPushButton("全不选")
        self.btn_blind_clear.clicked.connect(self.blind_list_widget.clearSelection)
        blind_btn_row.addWidget(self.btn_blind_clear)
        blind_v.addLayout(blind_btn_row)
        oil_layout.addWidget(blind_box)

        reg_box = QGroupBox("回归器（t → 针入度）")
        reg_h = QHBoxLayout(reg_box)
        self.radio_lin = QRadioButton("线性回归（样本＜5）")
        self.radio_quad = QRadioButton("二次多项式回归（默认）")
        self.radio_iso = QRadioButton("单调回归(样本噪声大)")
        self.radio_quad.setChecked(True)
        self.reg_group = QButtonGroup(self)
        self.reg_group.addButton(self.radio_lin, 0)
        self.reg_group.addButton(self.radio_quad, 1)
        self.reg_group.addButton(self.radio_iso, 2)
        reg_h.addWidget(self.radio_lin)
        reg_h.addWidget(self.radio_quad)
        reg_h.addWidget(self.radio_iso)
        oil_layout.addWidget(reg_box)

        action_row = QHBoxLayout()
        self.btn_match_oil_grade = QPushButton("🔍 匹配并预测标号")
        self.btn_match_oil_grade.clicked.connect(self.start_match_oil_grade_thread)
        action_row.addWidget(self.btn_match_oil_grade)
        oil_layout.addLayout(action_row)

        self.oil_grade_tree = QTreeWidget()
        self.oil_grade_tree.setColumnCount(10)
        self.oil_grade_tree.setHeaderLabels([
            "样品", "来源1", "来源2", "来源1比例(%)", "来源2比例(%)",
            "t", "d_perp", "预测针入度(0.1mm)", "标号", "备注"
        ])
        self.oil_grade_tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        for ci in range(1,10):
            self.oil_grade_tree.header().setSectionResizeMode(ci, QHeaderView.ResizeToContents)
        oil_layout.addWidget(self.oil_grade_tree)

        # === Tab 5: 三性能映射与预测 (2D) ===
        self.tab_triple = QWidget()
        self.notebook.addTab(self.tab_triple, "三性能映射与预测 (2D)")
        triple_layout = QVBoxLayout(self.tab_triple)

        perf_box = QGroupBox("加载沥青性能表（CSV）")
        perf_v = QVBoxLayout(perf_box)
        perf_btn_row = QHBoxLayout()
        self.btn_load_perf_csv = QPushButton("📥 加载性能表(CSV)")
        self.btn_load_perf_csv.clicked.connect(self.load_performance_csv_clicked)
        perf_btn_row.addWidget(self.btn_load_perf_csv)
        self.lbl_perf_loaded = QLabel("未加载性能表")
        perf_v.addLayout(perf_btn_row)
        perf_v.addWidget(self.lbl_perf_loaded)
        triple_layout.addWidget(perf_box)

        triple_sel_box = QGroupBox("选择数据库（标样）与盲样")
        triple_sel_v = QVBoxLayout(triple_sel_box)

        subrow_db = QHBoxLayout()
        col_db = QVBoxLayout()
        col_db_label = QLabel("数据库（标样）")
        self.triple_db_list = QListWidget()
        self.triple_db_list.setSelectionMode(QAbstractItemView.MultiSelection)
        col_db.addWidget(col_db_label)
        col_db.addWidget(self.triple_db_list)
        row_db_btns = QHBoxLayout()
        self.btn_triple_db_all = QPushButton("全选")
        self.btn_triple_db_all.clicked.connect(self.triple_db_list.selectAll)
        row_db_btns.addWidget(self.btn_triple_db_all)
        self.btn_triple_db_clear = QPushButton("全不选")
        self.btn_triple_db_clear.clicked.connect(self.triple_db_list.clearSelection)
        row_db_btns.addWidget(self.btn_triple_db_clear)
        col_db.addLayout(row_db_btns)
        subrow_db.addLayout(col_db)

        col_blind = QVBoxLayout()
        col_blind_label = QLabel("盲样")
        self.triple_blind_list = QListWidget()
        self.triple_blind_list.setSelectionMode(QAbstractItemView.MultiSelection)
        col_blind.addWidget(col_blind_label)
        col_blind.addWidget(self.triple_blind_list)
        row_blind_btns = QHBoxLayout()
        self.btn_triple_blind_all = QPushButton("全选")
        self.btn_triple_blind_all.clicked.connect(self.triple_blind_list.selectAll)
        row_blind_btns.addWidget(self.btn_triple_blind_all)
        self.btn_triple_blind_clear = QPushButton("全不选")
        self.btn_triple_blind_clear.clicked.connect(self.triple_blind_list.clearSelection)
        row_blind_btns.addWidget(self.btn_triple_blind_clear)
        col_blind.addLayout(row_blind_btns)
        subrow_db.addLayout(col_blind)
        triple_sel_v.addLayout(subrow_db)

        triple_layout.addWidget(triple_sel_box)

        triple_reg_box = QGroupBox("回归器（[身份位置1, 身份位置2] → 三性能）")
        triple_reg_h = QHBoxLayout(triple_reg_box)
        self.triple_radio_lin = QRadioButton("线性回归")
        self.triple_radio_quad = QRadioButton("二次多项式（默认）")
        self.triple_radio_cubic = QRadioButton("三次多项式")
        self.triple_radio_quad.setChecked(True)
        self.triple_reg_group = QButtonGroup(self)
        self.triple_reg_group.addButton(self.triple_radio_lin, 1)
        self.triple_reg_group.addButton(self.triple_radio_quad, 2)
        self.triple_reg_group.addButton(self.triple_radio_cubic, 3)
        triple_reg_h.addWidget(self.triple_radio_lin)
        triple_reg_h.addWidget(self.triple_radio_quad)
        triple_reg_h.addWidget(self.triple_radio_cubic)
        triple_layout.addWidget(triple_reg_box)

        triple_run_row = QHBoxLayout()
        self.btn_run_triple = QPushButton("🔍 拟合并预测三性能")
        self.btn_run_triple.clicked.connect(self.start_triple_mapping_thread)
        triple_run_row.addWidget(self.btn_run_triple)
        triple_layout.addLayout(triple_run_row)

        # —— 恢复并增强：模型与训练（2D）——
        group_train2d = QGroupBox("模型与训练（训练集真值/预测/误差）")
        vt2d = QVBoxLayout(group_train2d)
        self.triple_train_tree = QTreeWidget()
        self.triple_train_tree.setColumnCount(10)
        self.triple_train_tree.setHeaderLabels([
            "Group",
            "Pen(True)", "Pen(Pred)", "ΔPen",
            "SP(True)", "SP(Pred)", "ΔSP",
            "Duc(True)", "Duc(Pred)", "ΔDuc"
        ])
        for ci in range(10):
            mode = QHeaderView.Stretch if ci == 0 else QHeaderView.ResizeToContents
            self.triple_train_tree.header().setSectionResizeMode(ci, mode)
        vt2d.addWidget(self.triple_train_tree)
        self.lbl_triple_rmse = QLabel("RMSE(P/S/D)：- / - / -")
        vt2d.addWidget(self.lbl_triple_rmse)
        triple_layout.addWidget(group_train2d)

        self.triple_tree = QTreeWidget()
        self.triple_tree.setColumnCount(7)
        self.triple_tree.setHeaderLabels([
            "样品", "预测针入度(0.1mm)", "预测软化点(℃)", "预测延度(cm)",
            "标号", "训练RMSE(Pen/SP/Duc)", "备注"
        ])
        self.triple_tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        for ci in range(1,7):
            self.triple_tree.header().setSectionResizeMode(ci, QHeaderView.ResizeToContents)
        triple_layout.addWidget(self.triple_tree)

        # === 新增 Tab 6：3D PCA交互图 ===
        self.tab_pca3d = QWidget()
        self.notebook.addTab(self.tab_pca3d, "PCA得分图 (3D 交互)")
        p3_layout = QVBoxLayout(self.tab_pca3d)

        choose_box = QGroupBox("选择显示的 Group（自动缩放到所选范围）")
        cb_v = QVBoxLayout(choose_box)
        self.list_pca3d_groups = QListWidget()
        self.list_pca3d_groups.setSelectionMode(QAbstractItemView.MultiSelection)
        cb_v.addWidget(self.list_pca3d_groups)
        row_btns = QHBoxLayout()
        btn_all = QPushButton("全选")
        btn_all.clicked.connect(self.list_pca3d_groups.selectAll)
        row_btns.addWidget(btn_all)
        btn_none = QPushButton("全不选")
        btn_none.clicked.connect(self.list_pca3d_groups.clearSelection)
        row_btns.addWidget(btn_none)
        self.chk3d_labels = QCheckBox("显示样品标签")
        self.chk3d_labels.setChecked(True)
        row_btns.addWidget(self.chk3d_labels)
        cb_v.addLayout(row_btns)
        p3_layout.addWidget(choose_box)

        self.plot3d_container = PlotContainerWidget(self.tab_pca3d, is3d=True, with_toolbar=True)
        p3_layout.addWidget(self.plot3d_container)
        row_refresh = QHBoxLayout()
        self.btn_refresh_pca3d = QPushButton("🔄 刷新3D图（需先完成一次全局计算）")
        self.btn_refresh_pca3d.clicked.connect(self.update_pca3d_plot)
        row_refresh.addWidget(self.btn_refresh_pca3d)
        p3_layout.addLayout(row_refresh)

        # === 新增 Tab 7：3D 相似度检测 ===
        self.tab_similarity3d = QWidget()
        self.notebook.addTab(self.tab_similarity3d, "相似度对照表 (3D)")
        sim3_layout = QVBoxLayout(self.tab_similarity3d)
        row_sim3 = QHBoxLayout()
        self.btn_calc_sim3d = QPushButton("🧮 计算3D相似度（需先全局计算）")
        self.btn_calc_sim3d.clicked.connect(self.start_similarity3d_thread)
        row_sim3.addWidget(self.btn_calc_sim3d)
        sim3_layout.addLayout(row_sim3)
        self.similarity3d_tree = QTreeWidget()
        self.similarity3d_tree.setColumnCount(3)
        self.similarity3d_tree.setHeaderLabels(["类别组合", "相似度", "判定结果"])
        self.similarity3d_tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.similarity3d_tree.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.similarity3d_tree.header().setSectionResizeMode(2, QHeaderView.Stretch)
        sim3_layout.addWidget(self.similarity3d_tree)

        # === 新增 Tab 8：3D 原油及沥青标号预测 ===
        self.tab_oil_grade3d = QWidget()
        self.notebook.addTab(self.tab_oil_grade3d, "原油及沥青标号预测 (3D)")
        og3_layout = QVBoxLayout(self.tab_oil_grade3d)

        db3_box = QGroupBox("选择数据库（标样，自动端元识别于3D）")
        db3_v = QVBoxLayout(db3_box)
        self.db3_list = QListWidget()
        self.db3_list.setSelectionMode(QAbstractItemView.MultiSelection)
        db3_v.addWidget(self.db3_list)
        row_db3 = QHBoxLayout()
        btn_db3_all = QPushButton("全选")
        btn_db3_all.clicked.connect(self.db3_list.selectAll)
        row_db3.addWidget(btn_db3_all)
        btn_db3_none = QPushButton("全不选")
        btn_db3_none.clicked.connect(self.db3_list.clearSelection)
        row_db3.addWidget(btn_db3_none)
        self.btn_load_pen_csv2 = QPushButton("📥 加载针入度值(CSV)")
        self.btn_load_pen_csv2.clicked.connect(self.load_penetration_csv_clicked)
        row_db3.addWidget(self.btn_load_pen_csv2)
        db3_v.addLayout(row_db3)
        self.lbl_pen_loaded2 = QLabel("当前针入度记录数：" + str(len(self.penetration_map)))
        db3_v.addWidget(self.lbl_pen_loaded2)
        og3_layout.addWidget(db3_box)

        blind3_box = QGroupBox("选择盲样")
        blind3_v = QVBoxLayout(blind3_box)
        self.blind3_list = QListWidget()
        self.blind3_list.setSelectionMode(QAbstractItemView.MultiSelection)
        blind3_v.addWidget(self.blind3_list)
        row_b3 = QHBoxLayout()
        btn_b3_all = QPushButton("全选")
        btn_b3_all.clicked.connect(self.blind3_list.selectAll)
        row_b3.addWidget(btn_b3_all)
        btn_b3_none = QPushButton("全不选")
        btn_b3_none.clicked.connect(self.blind3_list.clearSelection)
        row_b3.addWidget(btn_b3_none)
        blind3_v.addLayout(row_b3)
        og3_layout.addWidget(blind3_box)

        reg3_box = QGroupBox("回归器（t → 针入度）")
        r3_h = QHBoxLayout(reg3_box)
        self.radio3_lin = QRadioButton("线性回归（样本＜5）")
        self.radio3_quad = QRadioButton("二次多项式回归（默认）")
        self.radio3_iso = QRadioButton("单调回归(样本噪声大)")
        self.radio3_quad.setChecked(True)
        self.reg3_group = QButtonGroup(self)
        self.reg3_group.addButton(self.radio3_lin, 0)
        self.reg3_group.addButton(self.radio3_quad, 1)
        self.reg3_group.addButton(self.radio3_iso, 2)
        r3_h.addWidget(self.radio3_lin)
        r3_h.addWidget(self.radio3_quad)
        r3_h.addWidget(self.radio3_iso)
        og3_layout.addWidget(reg3_box)

        row_og3 = QHBoxLayout()
        self.btn_match_oil_grade3d = QPushButton("🔍 匹配并预测标号 (3D)")
        self.btn_match_oil_grade3d.clicked.connect(self.start_match_oil_grade3d_thread)
        row_og3.addWidget(self.btn_match_oil_grade3d)
        og3_layout.addLayout(row_og3)

        self.oil_grade3d_tree = QTreeWidget()
        self.oil_grade3d_tree.setColumnCount(10)
        self.oil_grade3d_tree.setHeaderLabels([
            "样品", "来源1", "来源2", "来源1比例(%)", "来源2比例(%)",
            "t", "d_perp", "预测针入度(0.1mm)", "标号", "备注"
        ])
        self.oil_grade3d_tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        for ci in range(1,10):
            self.oil_grade3d_tree.header().setSectionResizeMode(ci, QHeaderView.ResizeToContents)
        og3_layout.addWidget(self.oil_grade3d_tree)

        # === 新增 Tab 9：3D 三性能映射与预测 ===
        self.tab_triple3d = QWidget()
        self.notebook.addTab(self.tab_triple3d, "三性能映射与预测 (3D)")
        t3_layout = QVBoxLayout(self.tab_triple3d)

        sel3_box = QGroupBox("选择数据库（标样）与盲样（基于 PC1-3 质心）")
        sel3_v = QVBoxLayout(sel3_box)
        row3 = QHBoxLayout()
        vdb3 = QVBoxLayout()
        vdb3.addWidget(QLabel("数据库（标样）"))
        self.triple3_db_list = QListWidget()
        self.triple3_db_list.setSelectionMode(QAbstractItemView.MultiSelection)
        vdb3.addWidget(self.triple3_db_list)
        rdb3 = QHBoxLayout()
        bdb3_all = QPushButton("全选"); bdb3_all.clicked.connect(self.triple3_db_list.selectAll); rdb3.addWidget(bdb3_all)
        bdb3_none = QPushButton("全不选"); bdb3_none.clicked.connect(self.triple3_db_list.clearSelection); rdb3.addWidget(bdb3_none)
        vdb3.addLayout(rdb3)
        row3.addLayout(vdb3)

        vbl3 = QVBoxLayout()
        vbl3.addWidget(QLabel("盲样"))
        self.triple3_blind_list = QListWidget()
        self.triple3_blind_list.setSelectionMode(QAbstractItemView.MultiSelection)
        vbl3.addWidget(self.triple3_blind_list)
        rbl3 = QHBoxLayout()
        bbl3_all = QPushButton("全选"); bbl3_all.clicked.connect(self.triple3_blind_list.selectAll); rbl3.addWidget(bbl3_all)
        bbl3_none = QPushButton("全不选"); bbl3_none.clicked.connect(self.triple3_blind_list.clearSelection); rbl3.addWidget(bbl3_none)
        vbl3.addLayout(rbl3)
        row3.addLayout(vbl3)
        sel3_v.addLayout(row3)
        t3_layout.addWidget(sel3_box)

        reg3d_box = QGroupBox("回归器（[PC1, PC2, PC3] → 三性能）")
        r3d_h = QHBoxLayout(reg3d_box)
        self.triple3_radio_lin = QRadioButton("线性回归")
        self.triple3_radio_quad = QRadioButton("二次多项式（默认）")
        self.triple3_radio_cubic = QRadioButton("三次多项式")
        self.triple3_radio_quad.setChecked(True)
        self.triple3_reg_group = QButtonGroup(self)
        self.triple3_reg_group.addButton(self.triple3_radio_lin, 1)
        self.triple3_reg_group.addButton(self.triple3_radio_quad, 2)
        self.triple3_reg_group.addButton(self.triple3_radio_cubic, 3)
        r3d_h.addWidget(self.triple3_radio_lin)
        r3d_h.addWidget(self.triple3_radio_quad)
        r3d_h.addWidget(self.triple3_radio_cubic)
        t3_layout.addWidget(reg3d_box)

        run3d_row = QHBoxLayout()
        self.btn_run_triple3d = QPushButton("🔍 拟合并预测三性能 (3D)")
        self.btn_run_triple3d.clicked.connect(self.start_triple3d_mapping_thread)
        run3d_row.addWidget(self.btn_run_triple3d)
        t3_layout.addLayout(run3d_row)

        group_train3d = QGroupBox("模型与训练（训练集真值/预测/误差，3D）")
        vt3d = QVBoxLayout(group_train3d)
        self.triple3_train_tree = QTreeWidget()
        self.triple3_train_tree.setColumnCount(10)
        self.triple3_train_tree.setHeaderLabels([
            "Group",
            "Pen(True)", "Pen(Pred)", "ΔPen",
            "SP(True)", "SP(Pred)", "ΔSP",
            "Duc(True)", "Duc(Pred)", "ΔDuc"
        ])
        for ci in range(10):
            mode = QHeaderView.Stretch if ci == 0 else QHeaderView.ResizeToContents
            self.triple3_train_tree.header().setSectionResizeMode(ci, mode)
        vt3d.addWidget(self.triple3_train_tree)
        self.lbl_triple3_rmse = QLabel("RMSE(P/S/D)：- / - / -")
        vt3d.addWidget(self.lbl_triple3_rmse)
        t3_layout.addWidget(group_train3d)

        self.triple3_tree = QTreeWidget()
        self.triple3_tree.setColumnCount(7)
        self.triple3_tree.setHeaderLabels([
            "样品", "预测针入度(0.1mm)", "预测软化点(℃)", "预测延度(cm)",
            "标号", "训练RMSE(Pen/SP/Duc)", "备注"
        ])
        self.triple3_tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        for ci in range(1,7):
            self.triple3_tree.header().setSectionResizeMode(ci, QHeaderView.ResizeToContents)
        t3_layout.addWidget(self.triple3_tree)

        # 加入到布局
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, 1)

    def _setup_qtree_widget_columns(self, tree_widget, tree_type):
        if tree_type == "mixture":
            tree_widget.setColumnCount(5); tree_widget.setHeaderLabels(["沥青编号", "来源1", "来源2", "来源1比例(%)", "来源2比例(%)"])
            tree_widget.header().setSectionResizeMode(0, QHeaderView.Interactive); tree_widget.setColumnWidth(0,120)
            tree_widget.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
            tree_widget.header().setSectionResizeMode(2, QHeaderView.ResizeToContents)
            tree_widget.header().setSectionResizeMode(3, QHeaderView.Stretch)
            tree_widget.header().setSectionResizeMode(4, QHeaderView.Stretch)
        elif tree_type == "similarity":
            tree_widget.setColumnCount(3); tree_widget.setHeaderLabels(["类别组合", "相似度", "判定结果"])
            tree_widget.header().setSectionResizeMode(0, QHeaderView.Stretch)
            tree_widget.header().setSectionResizeMode(1, QHeaderView.ResizeToContents); tree_widget.setColumnWidth(1,100)
            tree_widget.header().setSectionResizeMode(2, QHeaderView.Stretch)

    def create_menu_bar(self):
        menu_bar = self.menuBar(); file_menu = menu_bar.addMenu("文件")
        open_action = QAction("📁 打开数据文件...", self); open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action); file_menu.addSeparator()
        exit_action = QAction("退出", self); exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def create_status_bar(self):
        self.status_bar = QStatusBar(); self.setStatusBar(self.status_bar); self.status_bar.showMessage("系统准备就绪")

    # ---------------- 日志/状态/消息 ----------------
    def _add_log_message(self, message, level="INFO"):
        timestamp = QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss"); formatted_message = f"[{timestamp}][{level}] {message}"
        original_color = self.log_output_area.textColor(); original_weight = self.log_output_area.fontWeight()
        if level == "ERROR": self.log_output_area.setTextColor(QColor("red")); self.log_output_area.setFontWeight(QFont.Bold)
        elif level == "WARNING": self.log_output_area.setTextColor(QColor("orange"))
        elif level == "DEBUG": self.log_output_area.setTextColor(QColor("gray"))
        self.log_output_area.append(formatted_message)
        self.log_output_area.setTextColor(original_color); self.log_output_area.setFontWeight(original_weight)
        self.log_output_area.moveCursor(QTextCursor.End)

    def _set_button_states(self, processing_active):
        self.data_processing_active = processing_active; state = not processing_active
        self.btn_open.setEnabled(state); self.btn_process.setEnabled(state); self.btn_update_local_identity_positions.setEnabled(state)
        # 2D页按钮
        self.btn_load_pen_csv.setEnabled(state)
        self.btn_match_oil_grade.setEnabled(state)
        self.btn_load_perf_csv.setEnabled(state)
        self.btn_run_triple.setEnabled(state)
        # 3D页按钮
        self.btn_refresh_pca3d.setEnabled(state)
        self.btn_calc_sim3d.setEnabled(state)
        self.btn_match_oil_grade3d.setEnabled(state)
        self.btn_run_triple3d.setEnabled(state)
        self.btn_load_pen_csv2.setEnabled(state)

    def _remember_current_tab(self):
        self._last_tab_index = self.notebook.currentIndex()

    def _poll_queue(self):
        try:
            while True:
                msg = self.task_queue.get_nowait(); msg_type = msg.get("type")
                if msg_type == MSG_LOG: self._add_log_message(msg["message"], msg.get("level", "INFO"))
                elif msg_type == MSG_PROGRESS_START:
                    if msg.get("mode") == "indeterminate": self.progress_bar.setRange(0,0)
                    else: self.progress_bar.setRange(0,100); self.progress_bar.setValue(0)
                elif msg_type == MSG_PROGRESS_STOP: self.progress_bar.setRange(0,100); self.progress_bar.setValue(0)
                elif msg_type == MSG_SHOW_MESSAGEBOX:
                    style = msg.get("style", "info").lower(); title = msg.get("title", "信息"); message_text = msg.get("message", "")
                    if style == "info": QMessageBox.information(self, title, message_text)
                    elif style == "warning": QMessageBox.warning(self, title, message_text)
                    elif style == "error": QMessageBox.critical(self, title, message_text)
                elif msg_type == MSG_BUTTON_STATE: self._set_button_states(msg["processing_active"])
                elif msg_type == MSG_CLEAR_UI_BEFORE_PROCESSING:
                    self.global_identity_positions_plot_container.canvas.clear_axes()
                    self.local_identity_positions_plot_container.canvas.clear_axes()
                    self.mixture_tree_local.clear()
                    self.similarity_tree.clear()
                    self.category_list_widget.clear()
                    # 新页也清空
                    self.db_list_widget.clear()
                    self.blind_list_widget.clear()
                    self.oil_grade_tree.clear()
                    # 三性能页清空
                    self.triple_db_list.clear()
                    self.triple_blind_list.clear()
                    self.triple_tree.clear()
                    self.triple_train_tree.clear()
                    # 3D页清空
                    self.list_pca3d_groups.clear()
                    self.similarity3d_tree.clear()
                    self.db3_list.clear(); self.blind3_list.clear(); self.oil_grade3d_tree.clear()
                    self.triple3_db_list.clear(); self.triple3_blind_list.clear(); self.triple3_tree.clear(); self.triple3_train_tree.clear()
                elif msg_type == MSG_POPULATE_CATEGORY_LIST:
                    labels = msg["data"]
                    # 旧模块：局部分析
                    self.category_list_widget.clear(); self.category_list_widget.addItems(labels)
                    # 标号预测：数据库 + 盲样
                    self.db_list_widget.clear(); self.db_list_widget.addItems(labels)
                    self.blind_list_widget.clear(); self.blind_list_widget.addItems(labels)
                    # 三性能映射：数据库 + 盲样
                    self.triple_db_list.clear(); self.triple_db_list.addItems(labels)
                    self.triple_blind_list.clear(); self.triple_blind_list.addItems(labels)
                    # 3D 图/相似度/预测：同样填充
                    self.list_pca3d_groups.clear(); self.list_pca3d_groups.addItems(labels)
                    self.db3_list.clear(); self.db3_list.addItems(labels)
                    self.blind3_list.clear(); self.blind3_list.addItems(labels)
                    self.triple3_db_list.clear(); self.triple3_db_list.addItems(labels)
                    self.triple3_blind_list.clear(); self.triple3_blind_list.addItems(labels)
                elif msg_type == MSG_DISPLAY_GLOBAL_IDENTITY_POSITIONS_DATA:
                    self._display_identity_positions_on_canvas(
                        self.global_identity_positions_plot_container,
                        msg["scores"], msg["major_labels"], msg["groups_for_scores"],
                        "全局身份位置分布图 (GUI)"
                    )
                elif msg_type == MSG_DISPLAY_LOCAL_IDENTITY_POSITIONS_DATA:
                    self._display_identity_positions_on_canvas(
                        self.local_identity_positions_plot_container,
                        msg["selected_data_dict"],
                        msg["selected_labels"],
                        None,
                        "局部身份位置分布图 (GUI)",
                        is_local=True,
                        mixture_lines_data=msg.get("mixture_lines_data", [])
                    )
                elif msg_type == MSG_UPDATE_MIXTURE_TABLE_LOCAL:
                    self.mixture_tree_local.clear()
                    for row_data in msg["data"]:
                        item = QTreeWidgetItem([str(x) for x in row_data]); self.mixture_tree_local.addTopLevelItem(item)
                elif msg_type == MSG_UPDATE_SIMILARITY_TABLE:
                    self.similarity_tree.clear()
                    for row_data in msg["data"]:
                        item = QTreeWidgetItem([row_data[0], f"{row_data[1]:.3f}", row_data[2]]); self.similarity_tree.addTopLevelItem(item)
                elif msg_type == MSG_UPDATE_OIL_GRADE_RESULTS:
                    self.oil_grade_tree.clear()
                    for row in msg["rows"]:
                        self.oil_grade_tree.addTopLevelItem(QTreeWidgetItem([str(x) for x in row]))
                elif msg_type == MSG_UPDATE_TRIPLE_RESULTS:
                    self.triple_tree.clear()
                    for row in msg["rows"]:
                        self.triple_tree.addTopLevelItem(QTreeWidgetItem([str(x) for x in row]))
                elif msg_type == MSG_UPDATE_TRIPLE_TRAIN_TABLE:
                    # 2D 训练表 & RMSE
                    self.triple_train_tree.clear()
                    for row in msg["rows"]:
                        self.triple_train_tree.addTopLevelItem(QTreeWidgetItem([str(x) for x in row]))
                    self.lbl_triple_rmse.setText(msg.get("rmse_text", "RMSE(P/S/D)：- / - / -"))
                # ===== 3D =====
                elif msg_type == MSG_UPDATE_SIMILARITY3D_TABLE:
                    self.similarity3d_tree.clear()
                    for row_data in msg["data"]:
                        self.similarity3d_tree.addTopLevelItem(QTreeWidgetItem([row_data[0], f"{row_data[1]:.3f}", row_data[2]]))
                elif msg_type == MSG_UPDATE_OIL_GRADE3D_RESULTS:
                    self.oil_grade3d_tree.clear()
                    for row in msg["rows"]:
                        self.oil_grade3d_tree.addTopLevelItem(QTreeWidgetItem([str(x) for x in row]))
                elif msg_type == MSG_UPDATE_TRIPLE3D_RESULTS:
                    self.triple3_tree.clear()
                    for row in msg["rows"]:
                        self.triple3_tree.addTopLevelItem(QTreeWidgetItem([str(x) for x in row]))
                elif msg_type == MSG_UPDATE_TRIPLE3D_TRAIN_TABLE:
                    self.triple3_train_tree.clear()
                    for row in msg["rows"]:
                        self.triple3_train_tree.addTopLevelItem(QTreeWidgetItem([str(x) for x in row]))
                    self.lbl_triple3_rmse.setText(msg.get("rmse_text", "RMSE(P/S/D)：- / - / -"))

                elif msg_type in (MSG_PROCESSING_COMPLETE, MSG_LOCAL_PROCESSING_COMPLETE):
                    self._set_button_states(False)
                    self.progress_bar.setRange(0, 100)
                    self.progress_bar.setValue(100 if msg.get("success") else 0)

                    title = "完成" if msg.get("success") else "错误"
                    message_text = msg.get("message", "")

                    if msg.get("success"):
                        QMessageBox.information(self, title, message_text)
                        # 仅“局部分析完成”时自动切到“局部身份位置与混掺分析”页
                        if msg_type == MSG_LOCAL_PROCESSING_COMPLETE:
                            self.notebook.setCurrentWidget(self.tab_local_identity_positions)
                        else:
                            # 修复：完成后恢复到任务启动前的标签页
                            self.notebook.setCurrentIndex(self._last_tab_index)
                    else:
                        QMessageBox.critical(self, title, message_text)
        except queue.Empty:
            pass
        finally:
            self.queue_poll_timer.start(100)

    # ---------------- 文件与导出预览 ----------------
    def open_file(self):
        if self.data_processing_active: return
        options = QFileDialog.Options(); path, _ = QFileDialog.getOpenFileName(self, "选择数据文件", "", "CSV 文件 (*.csv);;所有文件 (*)", options=options)
        if path: self.file_path = path; self.lbl_file_path_status.setText(os.path.basename(path)); self._add_log_message(f"已选择数据文件: {self.file_path}", "INFO")
        else: self.lbl_file_path_status.setText("未选择文件"); self._add_log_message("未选择数据文件。", "WARNING")

    def _show_exported_file(self, filename_key, warning_message):
        if not self.file_path: QMessageBox.warning(self, "警告", "数据文件路径未设置！"); self._add_log_message(f"警告: 打开{filename_key}失败，路径未设置。", "WARNING"); return
        file_to_show_path = os.path.join(os.path.dirname(self.file_path), filename_key)
        if os.path.exists(file_to_show_path):
            try:
                import webbrowser; webbrowser.open(f"file:///{os.path.realpath(file_to_show_path)}"); self._add_log_message(f"{filename_key} 已尝试打开。", "INFO")
            except Exception as e:
                self._add_log_message(f"无法打开文件 {file_to_show_path}: {e}", "ERROR"); QMessageBox.critical(self, "错误", f"无法打开文件: {e}")
        else:
            QMessageBox.warning(self, "警告", warning_message); self._add_log_message(f"警告: {warning_message} ({filename_key}不存在)", "WARNING")

    def show_exported_identity_positions_plot(self): self._show_exported_file("全局身份位置分布图.png", "全局身份位置分布图不存在，请先进行计算！")
    def show_similarity_matrix_file(self): self._show_exported_file("相似度矩阵.csv", "相似度矩阵不存在，请先进行计算！")
    def show_similarity_standard_file(self): self._show_exported_file("相似度对照表.csv", "相似度对照表不存在，请先进行计算！")

    # ---------------- 全局处理（必须先跑一次，供3D使用） ----------------
    def start_global_processing_thread(self):
        if not self.file_path: QMessageBox.warning(self, "警告", "请先选择数据文件！"); self._add_log_message("警告: 未选择数据文件进行处理。", "WARNING"); return
        if self.data_processing_active: QMessageBox.information(self, "处理中", "当前已有数据处理任务在运行，请稍候。"); return
        self._remember_current_tab()
        self._set_button_states(True); self.task_queue.put({"type": MSG_LOG, "message": "全局身份位置识别计算任务开始...", "level": "INFO"})
        self.task_queue.put({"type": MSG_PROGRESS_START, "mode": "indeterminate"})
        self.task_queue.put({"type": MSG_CLEAR_UI_BEFORE_PROCESSING})
        self._current_worker_thread = threading.Thread(target=self._worker_process_data_global, name="GlobalProcessingThread"); self._current_worker_thread.daemon = True; self._current_worker_thread.start()

    def start_local_identity_positions_update_thread(self):
        if not self.file_path: QMessageBox.warning(self, "警告", "请先选择数据文件并完成全局计算！"); return
        if self.data_processing_active: QMessageBox.information(self, "处理中", "当前已有数据处理任务在运行，请稍候。"); return
        selected_items = self.category_list_widget.selectedItems()
        if not selected_items: QMessageBox.warning(self, "警告", "请先在左侧列表中选择至少一个沥青类别！"); return
        selected_labels = [item.text() for item in selected_items]
        self._remember_current_tab()
        self._set_button_states(True); self.task_queue.put({"type": MSG_LOG, "message": "局部身份位置分析任务开始（线性性判定 + 端元-中间混掺）...", "level": "INFO"})
        self.task_queue.put({"type": MSG_PROGRESS_START, "mode": "indeterminate"})
        self.local_identity_positions_plot_container.canvas.clear_axes()
        self.task_queue.put({"type": MSG_UPDATE_MIXTURE_TABLE_LOCAL, "data": []})
        self._current_worker_thread = threading.Thread(target=self._worker_update_local_identity_positions, args=(selected_labels,), name="LocalIdentityUpdateThread"); self._current_worker_thread.daemon = True; self._current_worker_thread.start()

    def _display_identity_positions_on_canvas(self, plot_container, data,
                               major_or_selected_labels, groups_for_scores,
                               title_text, is_local=False, mixture_lines_data=None):
        ax = plot_container.get_axes()
        ax.clear()
        num_labels = len(major_or_selected_labels)
        colors = plt.cm.get_cmap('tab10', num_labels if num_labels > 0 else 1)

        col_id_pos1 = IDENTITY_POSITION_COL_1
        col_id_pos2 = IDENTITY_POSITION_COL_2

        if is_local:
            selected_data_dict = data
            for idx, label in enumerate(major_or_selected_labels):
                df_subset = selected_data_dict.get(label)
                if df_subset is None or df_subset.empty: continue
                if col_id_pos1 not in df_subset.columns or col_id_pos2 not in df_subset.columns:
                     self.task_queue.put({"type": MSG_LOG, "message": f"错误: 局部数据缺少'{col_id_pos1}'或'{col_id_pos2}' for {label}", "level":"ERROR"}); continue
                points = df_subset[[col_id_pos1, col_id_pos2]].values
                ax.scatter(points[:, 0], points[:, 1], s=30, color=colors(idx % num_labels), label=label)
                if points.shape[0] > 0: center = points.mean(axis=0); ax.text(center[0], center[1], label, fontsize=8)

            if mixture_lines_data:
                self.task_queue.put({"type": MSG_LOG, "message": f"GUI绘图：开始绘制混掺指示线，共{len(mixture_lines_data)}条。", "level":"DEBUG"})
                for line_data in mixture_lines_data:
                    ax.plot(line_data['x'], line_data['y'], linestyle='--', color='gray', alpha=0.7, zorder=1)
                    if 'points_to_highlight' in line_data:
                        for p_series_df in line_data['points_to_highlight']:
                            if not p_series_df.empty and col_id_pos1 in p_series_df.columns and col_id_pos2 in p_series_df.columns:
                                ax.scatter(p_series_df[col_id_pos1], p_series_df[col_id_pos2], s=80, facecolors='none', edgecolors='black', linewidths=1.5, zorder=3)
                    if 'A_point_text' in line_data and 'A_point' in line_data and line_data['A_point'] is not None:
                        ax.text(line_data['A_point'][0], line_data['A_point'][1], line_data['A_point_text'], fontsize=8, color='blue', zorder=4, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="blue", alpha=0.7))
                    if 'B_point_text' in line_data and 'B_point' in line_data and line_data['B_point'] is not None:
                        ax.text(line_data['B_point'][0], line_data['B_point'][1], line_data['B_point_text'], fontsize=8, color='red', zorder=4, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="red", alpha=0.7))
        else:
            scores_data = data
            for idx, label_prefix in enumerate(major_or_selected_labels):
                relevant_indices = [i for i, grp_label in enumerate(groups_for_scores) if str(grp_label).startswith(label_prefix)]
                if not relevant_indices: continue
                subset_scores = scores_data[relevant_indices, :]
                if subset_scores.shape[1] < 2: self.task_queue.put({"type": MSG_LOG, "message": f"类别 {label_prefix} 的身份位置得分少于2维。", "level": "WARNING"}); continue
                ax.scatter(subset_scores[:, 0], subset_scores[:, 1], s=30, color=colors(idx % num_labels), label=label_prefix)
                if subset_scores.shape[0] > 0: center = subset_scores[:, :2].mean(axis=0); ax.text(center[0], center[1], label_prefix, fontsize=8)

        ax.set_xlabel(IDENTITY_POSITION_COL_1, fontsize=9)
        ax.set_ylabel(IDENTITY_POSITION_COL_2, fontsize=9)
        ax.set_title(title_text, fontsize=10)
        ax.grid(True)
        plot_container._perform_debounced_redraw()
        self.task_queue.put({"type": MSG_LOG, "message": f"GUI内 {title_text} 已请求更新。", "level":"DEBUG"})

    # ---------------- 预处理/全局PCA ----------------
    def _baseline_als_corrected_worker(self, y, lam=1e5, p=0.05, niter=10):
        y_float = y.astype(np.float64); L = len(y_float)
        if L < 3: self.task_queue.put({"type": MSG_LOG, "message": f"光谱太短 ({L} 点)无法进行基线校正，跳过。", "level":"WARNING"}); return y_float
        D_sparse = diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2), format='csc'); D_dt_sparse = D_sparse.dot(D_sparse.transpose())
        w = np.ones(L, dtype=np.float64)
        for i in range(niter): W_sparse = diags([w], [0], shape=(L,L), format='csc'); Z_sparse = W_sparse + lam * D_dt_sparse; z = spsolve(Z_sparse, w * y_float); w = p * (y_float > z) + (1 - p) * (y_float < z)
        return y_float - z

    def _worker_process_data_global(self):
        try:
            q_put = self.task_queue.put
            q_put({"type": MSG_LOG, "message": "线程：开始加载数据...", "level": "INFO"})
            data_df = pd.read_csv(self.file_path)
            q_put({"type": MSG_LOG, "message": f"线程：数据文件 '{os.path.basename(self.file_path)}' 加载成功。", "level": "INFO"})
            if "Group" not in data_df.columns: q_put({"type": MSG_PROCESSING_COMPLETE, "success": False, "message": "数据文件中缺少 'Group' 列。"}); return
            data_df["Group"] = data_df["Group"].astype(str); unique_labels_list = sorted(list(data_df["Group"].unique()))
            q_put({"type": MSG_POPULATE_CATEGORY_LIST, "data": unique_labels_list}); q_put({"type": MSG_LOG, "message": "线程：沥青类别已加载。", "level": "INFO"})
            base_output_dir = os.path.dirname(self.file_path)
            wavenumbers = data_df.columns[2:]

            # ====== CO2 区域（2330-2380 cm⁻¹）删除 ======
            q_put({"type": MSG_LOG, "message": "线程：正在检查并排除 2000-2500 cm⁻¹ 范围的光谱区域...", "level": "INFO"})
            cols_to_drop = []
            for wn_str in wavenumbers:
                try:
                    wn_float = float(wn_str)
                    if 2330 <= wn_float <= 2380:
                        cols_to_drop.append(wn_str)
                except ValueError:
                    continue
            if cols_to_drop:
                data_df.drop(columns=cols_to_drop, inplace=True)
                q_put({"type": MSG_LOG, "message": f"线程：成功排除了 {len(cols_to_drop)} 个在 CO₂ 区域的数据点。", "level": "INFO"})
            else:
                q_put({"type": MSG_LOG, "message": "线程：未在数据中发现 2000-2500 cm⁻¹ 范围的数据点。", "level": "INFO"})
            wavenumbers = data_df.columns[2:]

            current_data_for_processing = data_df.copy()

            # Baseline
            q_put({"type": MSG_LOG, "message": "线程：正在进行基线校正...", "level": "INFO"})
            spectra_values = current_data_for_processing[wavenumbers].values
            corrected_spectra_list = [self._baseline_als_corrected_worker(spectra_values[i, :]) for i in range(spectra_values.shape[0])]
            current_data_for_processing[wavenumbers] = np.array(corrected_spectra_list)
            current_data_for_processing.to_csv(os.path.join(base_output_dir, "基线校正_data.csv"), index=False, encoding="utf-8-sig")
            q_put({"type": MSG_LOG, "message": f"线程：基线校正完成", "level": "INFO"})

            # MinMax
            q_put({"type": MSG_LOG, "message": "线程：正在进行归一化...", "level": "INFO"})
            scaler_minmax = MinMaxScaler(); current_data_for_processing[wavenumbers] = scaler_minmax.fit_transform(current_data_for_processing[wavenumbers].T).T
            current_data_for_processing.to_csv(os.path.join(base_output_dir, "归一化_data.csv"), index=False, encoding="utf-8-sig"); q_put({"type": MSG_LOG, "message": f"线程：归一化完成", "level": "INFO"})

            # Standardize
            q_put({"type": MSG_LOG, "message": "线程：正在进行标准化...", "level": "INFO"})
            scaler_std = StandardScaler(); current_data_for_processing[wavenumbers] = scaler_std.fit_transform(current_data_for_processing[wavenumbers].T).T
            current_data_for_processing.to_csv(os.path.join(base_output_dir, "标准化_data.csv"), index=False, encoding="utf-8-sig"); q_put({"type": MSG_LOG, "message": f"线程：标准化完成", "level": "INFO"})

            # Autoscale (log sigma)
            q_put({"type": MSG_LOG, "message": "线程：正在进行数值缩放...", "level": "INFO"})
            X_mean_overall = current_data_for_processing[wavenumbers].mean(axis=0); S_std_overall = current_data_for_processing[wavenumbers].std(axis=0)
            with np.errstate(divide='ignore', invalid='ignore'): Y_log_scaling_factors = np.log10(S_std_overall)
            autoscaled_spectra_values_raw = (current_data_for_processing[wavenumbers] - X_mean_overall) / Y_log_scaling_factors
            autoscaled_spectra_values_cleaned = np.nan_to_num(autoscaled_spectra_values_raw, nan=0.0, posinf=0.0, neginf=0.0)
            current_data_for_processing[wavenumbers] = autoscaled_spectra_values_cleaned
            path_scaled_data = os.path.join(base_output_dir, "数值缩放后_data.csv")
            current_data_for_processing.to_csv(path_scaled_data, index=False, encoding="utf-8-sig")
            q_put({"type": MSG_LOG, "message": f"线程：数值缩放完成，保存至 {os.path.basename(path_scaled_data)}", "level": "INFO"})

            # PCA -> Identity positions
            q_put({"type": MSG_LOG, "message": "线程：正在进行内部成分提取...", "level": "INFO"})
            spectra_for_component_extraction = current_data_for_processing[wavenumbers].values
            if np.any(np.isnan(spectra_for_component_extraction)) or np.any(np.isinf(spectra_for_component_extraction)):
                spectra_for_component_extraction = np.nan_to_num(spectra_for_component_extraction, nan=0.0, posinf=0.0, neginf=0.0)
            if spectra_for_component_extraction.shape[0] == 0: q_put({"type": MSG_PROCESSING_COMPLETE, "success": False, "message": "没有有效的光谱数据进行内部成分提取。"}); return

            internal_pca_model = PCA(); internal_pca_model.fit(spectra_for_component_extraction)
            all_component_scores = internal_pca_model.transform(spectra_for_component_extraction)
            major_labels_for_plot = sorted(list(set([label.split("-")[0] for label in unique_labels_list])))
            q_put({"type": MSG_DISPLAY_GLOBAL_IDENTITY_POSITIONS_DATA, "scores": all_component_scores[:, :2], "major_labels": major_labels_for_plot, "groups_for_scores": current_data_for_processing["Group"].tolist()})

            fig_export_global = plt.figure(figsize=(12, 8)); ax_export = fig_export_global.add_subplot(111)
            colors_export = plt.cm.get_cmap('tab20', len(major_labels_for_plot))
            for idx, label_prefix in enumerate(major_labels_for_plot):
                relevant_indices = [i for i, grp_label in enumerate(current_data_for_processing["Group"]) if str(grp_label).startswith(label_prefix)]
                if not relevant_indices: continue
                subset_scores_export = all_component_scores[relevant_indices, :2]
                ax_export.scatter(subset_scores_export[:, 0], subset_scores_export[:, 1], s=50, color=colors_export(idx), label=label_prefix)
                if subset_scores_export.shape[0] > 0: center = subset_scores_export.mean(axis=0); ax_export.text(center[0], center[1], label_prefix, fontsize=9)
            ax_export.set_xlabel(IDENTITY_POSITION_COL_1); ax_export.set_ylabel(IDENTITY_POSITION_COL_2); ax_export.set_title("全局身份位置分布图 (所有类别 - 导出)")
            ax_export.legend(loc="best", prop={'size': 8}); ax_export.grid(True); fig_export_global.tight_layout()
            path_id_pos_export = os.path.join(base_output_dir, "全局身份位置分布图.png")
            fig_export_global.savefig(path_id_pos_export, dpi=300); plt.close(fig_export_global)
            q_put({"type": MSG_LOG, "message": f"线程：高分辨率全局身份位置图已保存。", "level": "INFO"})

            # 保存全部分量得分
            scores_df_to_save = pd.DataFrame(all_component_scores, columns=[f"内部成分{i}" for i in range(1, all_component_scores.shape[1] + 1)])
            scores_df_to_save.insert(0, "Group", current_data_for_processing["Group"])
            scores_df_to_save.to_csv(os.path.join(base_output_dir, "身份位置得分数据.csv"), index=False, encoding="utf-8-sig")

            # 将PC1/PC2写回便于2D
            current_data_for_processing[IDENTITY_POSITION_COL_1] = all_component_scores[:, 0]
            current_data_for_processing[IDENTITY_POSITION_COL_2] = all_component_scores[:, 1]
            current_data_for_processing.to_csv(path_scaled_data, index=False, encoding="utf-8-sig")
            q_put({"type": MSG_LOG, "message": f"线程：{os.path.basename(path_scaled_data)} 已更新并包含 {IDENTITY_POSITION_COL_1}, {IDENTITY_POSITION_COL_2}。", "level": "INFO"})

            # Similarity（2D，与旧逻辑一致）
            q_put({"type": MSG_LOG, "message": "线程：正在计算相似度矩阵 (2D)...", "level": "INFO"})
            major_labels_for_plot_unique = major_labels_for_plot
            similarity_matrix = pd.DataFrame(index=major_labels_for_plot_unique, columns=major_labels_for_plot_unique, dtype=float)
            scale_factor_sim_thread = 0.2
            n_comp_avg_spec_fixed = 2; internal_pca_avg_spec = None
            spectra_vals_for_avg_fit = current_data_for_processing[wavenumbers].values
            num_samples_fit_avg = spectra_vals_for_avg_fit.shape[0]; num_features_fit_avg = spectra_vals_for_avg_fit.shape[1]
            if num_samples_fit_avg >= n_comp_avg_spec_fixed and num_features_fit_avg >= n_comp_avg_spec_fixed:
                internal_pca_avg_spec = PCA(n_components=n_comp_avg_spec_fixed)
                try: internal_pca_avg_spec.fit(np.nan_to_num(spectra_vals_for_avg_fit))
                except Exception as fit_err: q_put({"type": MSG_LOG, "message": f"线程错误：拟合平均光谱内部模型(n=2)失败: {fit_err}", "level": "ERROR"}); internal_pca_avg_spec = None

            for label1 in major_labels_for_plot_unique:
                for label2 in major_labels_for_plot_unique:
                    sim_combined_val = 0.0
                    scores1_id_pos = current_data_for_processing[current_data_for_processing["Group"].str.startswith(label1)][[IDENTITY_POSITION_COL_1, IDENTITY_POSITION_COL_2]]
                    scores2_id_pos = current_data_for_processing[current_data_for_processing["Group"].str.startswith(label2)][[IDENTITY_POSITION_COL_1, IDENTITY_POSITION_COL_2]]
                    sim_euclidean_val = 0.0
                    if not scores1_id_pos.empty and not scores2_id_pos.empty:
                        c1_vals = scores1_id_pos.mean().values
                        c2_vals = scores2_id_pos.mean().values
                        if not (np.isnan(c1_vals).any() or np.isnan(c2_vals).any()):
                            dist_euc = np.linalg.norm(c1_vals - c2_vals)
                            sim_euclidean_val = np.exp(-dist_euc ** 2 / (2 * scale_factor_sim_thread ** 2))
                            if np.isnan(sim_euclidean_val): sim_euclidean_val = 0.0
                    sim_avg_val = 0.0
                    if internal_pca_avg_spec is not None:
                        avg_spec1_s = current_data_for_processing[wavenumbers][current_data_for_processing["Group"].str.startswith(label1)].mean()
                        avg_spec2_s = current_data_for_processing[wavenumbers][current_data_for_processing["Group"].str.startswith(label2)].mean()
                        if not (avg_spec1_s.isnull().all() or avg_spec2_s.isnull().all()):
                            avg1_for_transform = np.nan_to_num(avg_spec1_s.values.reshape(1, -1)); avg2_for_transform = np.nan_to_num(avg_spec2_s.values.reshape(1, -1))
                            try:
                                scores_avg1_transformed = internal_pca_avg_spec.transform(avg1_for_transform); scores_avg2_transformed = internal_pca_avg_spec.transform(avg2_for_transform)
                                dist_avg_internal = np.linalg.norm(scores_avg1_transformed - scores_avg2_transformed)
                                sim_avg_val = 1 / (1 + dist_avg_internal);
                                if np.isnan(sim_avg_val): sim_avg_val = 0.0
                            except Exception: sim_avg_val = 0.0
                    sim_combined_val = 0.2 * sim_euclidean_val + 0.8 * sim_avg_val
                    if np.isnan(sim_combined_val): sim_combined_val = 0.0
                    similarity_matrix.loc[label1, label2] = sim_combined_val

            similarity_matrix.to_csv(os.path.join(base_output_dir, "相似度矩阵.csv"), index=True, encoding="utf-8-sig")
            q_put({"type": MSG_LOG, "message": "线程：相似度矩阵已计算并保存。", "level": "INFO"})
            labelled_data_for_q = []
            processed_pairs_set = set()
            for r_label in major_labels_for_plot_unique:
                for c_label in major_labels_for_plot_unique:
                    pair_key = tuple(sorted((r_label, c_label)))
                    if pair_key not in processed_pairs_set:
                        sim_val = float(similarity_matrix.loc[r_label, c_label]); std_text = ""
                        if r_label == c_label: std_text = "同一品牌相同批次沥青 (自身对比)"
                        elif sim_val <= 0.80: std_text = "不同品牌沥青"
                        elif 0.80 < sim_val <= 0.90: std_text = "同一品牌不同批次沥青"
                        else: std_text = "同一品牌相同批次沥青"
                        labelled_data_for_q.append((f"{r_label}-{c_label}", sim_val, std_text)); processed_pairs_set.add(pair_key)
            labelled_data_for_q.sort(key=lambda x: x[0])
            pd.DataFrame(labelled_data_for_q, columns=["Category Pair", "Similarity", "Standard"]).to_csv(os.path.join(base_output_dir, "相似度对照表.csv"), index=False, encoding="utf-8-sig")
            q_put({"type": MSG_LOG, "message": "线程：相似度对照表CSV已保存。", "level": "INFO"})
            q_put({"type": MSG_UPDATE_SIMILARITY_TABLE, "data": labelled_data_for_q})
            q_put({"type": MSG_PROCESSING_COMPLETE, "success": True, "message": "全局身份位置识别计算成功完成！"})
        except Exception as e:
            import traceback; tb_str = traceback.format_exc(); self.task_queue.put({"type": MSG_LOG, "message": f"线程错误 (全局处理): {e}\n{tb_str}", "level": "ERROR"})
            self.task_queue.put({"type": MSG_PROCESSING_COMPLETE, "success": False, "message": f"全局身份位置识别计算失败: {e}"})

    # ---------------- 局部线性与混掺 ----------------
    def _worker_update_local_identity_positions(self, selected_labels_list):
        try:
            q_put = self.task_queue.put; base_output_dir = os.path.dirname(self.file_path)
            processed_data_path = os.path.join(base_output_dir, "数值缩放后_data.csv")

            if not os.path.exists(processed_data_path): q_put({"type": MSG_LOCAL_PROCESSING_COMPLETE, "success": False, "message": f"预处理数据 '{os.path.basename(processed_data_path)}' 不存在。"}); return
            data_with_id_pos = pd.read_csv(processed_data_path); data_with_id_pos["Group"] = data_with_id_pos["Group"].astype(str)

            if IDENTITY_POSITION_COL_1 not in data_with_id_pos.columns or IDENTITY_POSITION_COL_2 not in data_with_id_pos.columns:
                q_put({"type": MSG_LOCAL_PROCESSING_COMPLETE, "success": False, "message": f"数据缺少'{IDENTITY_POSITION_COL_1}'或'{IDENTITY_POSITION_COL_2}'列。"}); return

            df_selected = data_with_id_pos[data_with_id_pos["Group"].isin(selected_labels_list)].copy()
            if df_selected.empty: q_put({"type": MSG_LOCAL_PROCESSING_COMPLETE, "success": False, "message": "所选类别在数据中没有匹配项。"}); return

            local_id_pos_data_for_gui = {}
            for label in selected_labels_list:
                sub = df_selected[df_selected["Group"] == label]
                if not sub.empty:
                    local_id_pos_data_for_gui[label] = sub[[IDENTITY_POSITION_COL_1, IDENTITY_POSITION_COL_2]]

            fig_export_local = plt.figure(figsize=(12, 8)); ax_export = fig_export_local.add_subplot(111)
            colors_local_export = plt.cm.get_cmap('tab20', len(selected_labels_list))
            for idx, label in enumerate(selected_labels_list):
                df_subset_export = local_id_pos_data_for_gui.get(label)
                if df_subset_export is None or df_subset_export.empty: continue
                pts = df_subset_export.values
                ax_export.scatter(pts[:, 0], pts[:, 1], s=50, color=colors_local_export(idx), label=label)
                if pts.shape[0] > 0: c = pts.mean(axis=0); ax_export.text(c[0], c[1], label, fontsize=9)
            ax_export.set_xlabel(IDENTITY_POSITION_COL_1); ax_export.set_ylabel(IDENTITY_POSITION_COL_2); ax_export.set_title("局部身份位置分布图 (导出)")
            ax_export.legend(loc="best", prop={'size': 8}); ax_export.grid(True); fig_export_local.tight_layout()
            path_local_id_pos_export = os.path.join(base_output_dir, "局部身份位置分布图.png")
            fig_export_local.savefig(path_local_id_pos_export, dpi=300); plt.close(fig_export_local)
            q_put({"type": MSG_LOG, "message": f"线程：局部身份位置分布图已保存：{os.path.basename(path_local_id_pos_export)}", "level": "INFO"})

            group_means = (df_selected
                           .groupby("Group")[[IDENTITY_POSITION_COL_1, IDENTITY_POSITION_COL_2]]
                           .mean()
                           .reset_index())
            unique_groups = group_means["Group"].tolist()
            if len(unique_groups) < MIN_GROUPS_FOR_SERIES:
                q_put({"type": MSG_DISPLAY_LOCAL_IDENTITY_POSITIONS_DATA, "selected_data_dict": local_id_pos_data_for_gui, "selected_labels": selected_labels_list, "mixture_lines_data": []})
                q_put({"type": MSG_UPDATE_MIXTURE_TABLE_LOCAL, "data": []})
                q_put({"type": MSG_LOCAL_PROCESSING_COMPLETE, "success": True, "message": f"有效组数不足（< {MIN_GROUPS_FOR_SERIES}），未进行混掺分析。"})
                return

            P1, P2 = IDENTITY_POSITION_COL_1, IDENTITY_POSITION_COL_2
            Xc = group_means[[P1, P2]].values.astype(float)

            pca = PCA(n_components=2).fit(Xc)
            ratio1 = float(pca.explained_variance_ratio_[0])
            dir_u = pca.components_[0]
            s = (Xc - Xc.mean(axis=0)) @ dir_u
            min_idx = int(np.argmin(s)); max_idx = int(np.argmax(s))
            A_label = group_means.iloc[min_idx]["Group"]; B_label = group_means.iloc[max_idx]["Group"]
            A = Xc[min_idx]; B = Xc[max_idx]
            AB = B - A; AB_len = float(np.linalg.norm(AB))
            if not np.isfinite(AB_len) or AB_len < EPS_AB_MIN:
                q_put({"type": MSG_DISPLAY_LOCAL_IDENTITY_POSITIONS_DATA, "selected_data_dict": local_id_pos_data_for_gui, "selected_labels": selected_labels_list, "mixture_lines_data": []})
                q_put({"type": MSG_UPDATE_MIXTURE_TABLE_LOCAL, "data": []})
                q_put({"type": MSG_LOCAL_PROCESSING_COMPLETE, "success": True, "message": "端元距离过小，未进行混掺分析。"})
                return

            def perp_dist_ratio(point, A, B):
                AB = B - A; AB_len = np.linalg.norm(AB)
                if AB_len < EPS_AB_MIN: return np.inf
                t = np.dot(point - A, AB) / (AB_len*AB_len)
                proj = A + t * AB
                d = np.linalg.norm(point - proj)
                return float(d / AB_len), float(np.clip(t, 0.0, 1.0))

            perp_ratios, t_all = [], []
            for pt in Xc:
                pr, t = perp_dist_ratio(pt, A, B)
                perp_ratios.append(pr); t_all.append(t)
            rms_perp = float(np.sqrt(np.mean(np.square(perp_ratios))))
            max_perp = float(np.max(perp_ratios))

            q_put({"type": MSG_LOG, "message": f"线性检测：PC1解释率={ratio1:.3f}，RMS⊥残差={rms_perp:.3f}，MAX⊥残差={max_perp:.3f}", "level":"INFO"})

            if not (ratio1 >= LINEAR_PCA_RATIO_MIN and rms_perp <= LINEAR_RMS_PERP_RATIO_MAX and max_perp <= LINEAR_MAX_PERP_RATIO_MAX):
                q_put({"type": MSG_LOG, "message": "线性关系不显著或残差过大：未进行混掺判定。", "level":"WARNING"})
                q_put({"type": MSG_DISPLAY_LOCAL_IDENTITY_POSITIONS_DATA, "selected_data_dict": local_id_pos_data_for_gui, "selected_labels": selected_labels_list, "mixture_lines_data": []})
                q_put({"type": MSG_UPDATE_MIXTURE_TABLE_LOCAL, "data": []})
                q_put({"type": MSG_LOCAL_PROCESSING_COMPLETE, "success": True, "message": "线性性不足，未进行混掺分析。"})
                return

            mixture_results_for_table = []
            points_to_highlight = []
            grp_points_map = {g: df_selected[df_selected["Group"] == g][[P1, P2]].copy()
                              for g in unique_groups}

            t_by_group = {}
            for idx_row, row in group_means.iterrows():
                g = row["Group"]
                pt = np.array([row[P1], row[P2]], dtype=float)
                t_raw = float(np.dot(pt - A, AB) / (AB_len*AB_len))
                t = float(np.clip(t_raw, 0.0, 1.0))
                t_by_group[g] = t

            for g in unique_groups:
                t = t_by_group[g]
                if (t > ENDPOINT_TOL) and (t < 1.0 - ENDPOINT_TOL):
                    base1_pct = round((1.0 - t) * 100.0, 2)
                    base2_pct = round(t * 100.0, 2)
                    mixture_results_for_table.append([g, A_label, B_label, base1_pct, base2_pct])
                    if g in grp_points_map:
                        points_to_highlight.append(grp_points_map[g])

            mixture_lines_for_plot = [{
                'x': [A[0], B[0]],
                'y': [A[1], B[1]],
                'points_to_highlight': points_to_highlight,
                'A_point': A,
                'A_point_text': f"{A_label}(端元)",
                'B_point': B,
                'B_point_text': f"{B_label}(端元)"
            }]

            q_put({"type": MSG_DISPLAY_LOCAL_IDENTITY_POSITIONS_DATA, "selected_data_dict": local_id_pos_data_for_gui, "selected_labels": selected_labels_list, "mixture_lines_data": mixture_lines_for_plot})
            q_put({"type": MSG_UPDATE_MIXTURE_TABLE_LOCAL, "data": mixture_results_for_table})

            if mixture_results_for_table:
                mixture_df_to_save = pd.DataFrame(mixture_results_for_table, columns=["Group", "Base1", "Base2", "Base1(%)", "Base2(%)"])
                path_mixture_export = os.path.join(base_output_dir, "混掺识别结果.csv")
                mixture_df_to_save.to_csv(path_mixture_export, index=False, encoding="utf-8-sig")
                q_put({"type": MSG_LOG, "message": f"线程：混掺识别结果已保存：{os.path.basename(path_mixture_export)}", "level": "INFO"})
            else:
                q_put({"type": MSG_LOG, "message": "未发现位于两端之间的“中间样品”，无混掺结果输出。", "level":"INFO"})

            q_put({"type": MSG_LOCAL_PROCESSING_COMPLETE, "success": True, "message": "局部线性检测与端元-中间混掺识别完成！"})
        except Exception as e:
            import traceback; tb_str = traceback.format_exc(); q_put({"type": MSG_LOG, "message": f"线程错误 (局部分析): {e}\n{tb_str}", "level": "ERROR"}); q_put({"type": MSG_LOCAL_PROCESSING_COMPLETE, "success": False, "message": f"局部分析失败: {e}"})

    # ====================== 原油及沥青标号预测 (2D) ======================
    def load_penetration_csv_clicked(self):
        if not self.file_path:
            QMessageBox.warning(self, "警告", "请先选择并进行一次全局计算。")
            return
        options = QFileDialog.Options()
        path, _ = QFileDialog.getOpenFileName(self, "选择针入度标注CSV", "", "CSV 文件 (*.csv);;所有文件 (*)", options=options)
        if not path:
            return
        try:
            df = pd.read_csv(path)
            if "Group" not in df.columns or "Penetration" not in df.columns:
                raise ValueError("CSV 必须包含列：Group, Penetration")
            new_map = {}
            for _, r in df.iterrows():
                g = str(r["Group"])
                try:
                    p = float(r["Penetration"])
                except:
                    continue
                new_map[g] = p
            self.penetration_map = new_map
            self.lbl_pen_loaded.setText(f"已加载 {len(self.penetration_map)} 条针入度记录")
            self.lbl_pen_loaded2.setText("当前针入度记录数：" + str(len(self.penetration_map)))
            self._add_log_message(f"针入度标注加载完成：{os.path.basename(path)}，条数={len(self.penetration_map)}", "INFO")
            QMessageBox.information(self, "完成", f"针入度标注加载完成（{len(self.penetration_map)} 条）。")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载失败：{e}")
            self._add_log_message(f"加载针入度CSV失败：{e}", "ERROR")

    def start_match_oil_grade_thread(self):
        if not self.file_path:
            QMessageBox.warning(self, "警告", "请先完成一次全局预处理（⚙️ 开始识别计算）。")
            return
        if self.data_processing_active:
            QMessageBox.information(self, "处理中", "当前已有数据处理任务在运行。")
            return
        db_items = self.db_list_widget.selectedItems()
        blind_items = self.blind_list_widget.selectedItems()
        if not db_items:
            QMessageBox.warning(self, "警告", "请先在“数据库（标样）”中选择至少 3 个组。")
            return
        if len(db_items) < MIN_GROUPS_FOR_SERIES:
            QMessageBox.warning(self, "警告", f"所选标样不足 {MIN_GROUPS_FOR_SERIES} 组，无法进行线性检测与端元识别。")
            return
        if not blind_items:
            QMessageBox.warning(self, "警告", "请在“盲样”中选择至少 1 个组。")
            return

        db_labels = [i.text() for i in db_items]
        blind_labels = [i.text() for i in blind_items]
        reg_id = self.reg_group.checkedId()  # 0:线性 1:二次 2:单调

        self._remember_current_tab()
        self._set_button_states(True)
        self.task_queue.put({"type": MSG_PROGRESS_START, "mode": "indeterminate"})
        th = threading.Thread(target=self._worker_match_oil_grade, args=(db_labels, blind_labels, reg_id), name="MatchOilGradeThread")
        th.daemon = True
        th.start()
        self._current_worker_thread = th

    def _assign_grade_builtin(self, penetration):
        p = float(penetration)
        hits = [r for r in BUILTIN_GRADE_RULES if r["Low"] <= p <= r["High"]]
        if len(hits) == 0:
            return "未匹配", "超出规则范围"
        labels = [r["Label"] for r in hits]
        if len(labels) == 1:
            return labels[0], ""
        return "/".join(labels), "临界区间(多档)"

    def _worker_match_oil_grade(self, db_labels, blind_labels, reg_id):
        q_put = self.task_queue.put
        try:
            base_dir = os.path.dirname(self.file_path)
            data_path = os.path.join(base_dir, "数值缩放后_data.csv")
            if not os.path.exists(data_path):
                raise FileNotFoundError("缺少 数值缩放后_data.csv，请先执行一次全局计算。")
            df = pd.read_csv(data_path)
            df["Group"] = df["Group"].astype(str)
            P1, P2 = IDENTITY_POSITION_COL_1, IDENTITY_POSITION_COL_2
            if P1 not in df.columns or P2 not in df.columns:
                raise ValueError(f"数据缺少列：{P1}/{P2}（请先执行全局计算）")

            # === Step 1: 标样线性检测 + 端元识别（2D） ===
            df_db = df[df["Group"].isin(db_labels)].copy()
            if df_db.empty:
                raise ValueError("标样在数据中没有找到匹配项。")

            group_means = df_db.groupby("Group")[[P1, P2]].mean().reset_index()
            if group_means.shape[0] < MIN_GROUPS_FOR_SERIES:
                raise ValueError(f"标样有效组数不足 {MIN_GROUPS_FOR_SERIES}。")

            Xc = group_means[[P1, P2]].values.astype(float)
            pca = PCA(n_components=2).fit(Xc)
            ratio1 = float(pca.explained_variance_ratio_[0])
            dir_u = pca.components_[0]
            s = (Xc - Xc.mean(axis=0)) @ dir_u
            min_idx = int(np.argmin(s)); max_idx = int(np.argmax(s))
            A_label = group_means.iloc[min_idx]["Group"]; B_label = group_means.iloc[max_idx]["Group"]
            A = Xc[min_idx]; B = Xc[max_idx]
            AB = B - A; AB_len = float(np.linalg.norm(AB))

            if not np.isfinite(AB_len) or AB_len < EPS_AB_MIN:
                raise ValueError("端元距离过小，无法进行投影。")

            def perp_dist_ratio(point, A, B):
                AB = B - A; L2 = float(np.dot(AB, AB))
                if L2 < EPS_AB_MIN: return np.inf, 0.0, 0.0
                t_raw = float(np.dot(point - A, AB) / L2)
                t = float(np.clip(t_raw, 0.0, 1.0))
                proj = A + t * AB
                d = float(np.linalg.norm(point - proj) / np.sqrt(L2))
                return d, t_raw, t

            d_list = []
            for pt in Xc:
                d, _, _ = perp_dist_ratio(pt, A, B)
                d_list.append(d)
            rms_perp = float(np.sqrt(np.mean(np.square(d_list))))
            max_perp = float(np.max(d_list))

            q_put({"type": MSG_LOG, "message": f"[标样线性检测(2D)] 端元={A_label}/{B_label}，PC1={ratio1:.3f}，RMS⊥={rms_perp:.3f}，MAX⊥={max_perp:.3f}", "level":"INFO"})

            if not (ratio1 >= LINEAR_PCA_RATIO_MIN and rms_perp <= LINEAR_RMS_PERP_RATIO_MAX and max_perp <= LINEAR_MAX_PERP_RATIO_MAX):
                raise ValueError("标样未呈现明显线性系列（PC1解释率/正交残差未过阈）。")

            # === Step 2: 回归器 t→Penetration（仅用具备针入度的标样） ===
            if not self.penetration_map:
                raise ValueError("尚未加载针入度表。请点击“加载针入度值(CSV)”。")

            t_train, y_train, g_train = [], [], []
            for _, r in group_means.iterrows():
                g = r["Group"]
                if g not in self.penetration_map:
                    continue
                pt = np.array([r[P1], r[P2]], dtype=float)
                d, t_raw, t = perp_dist_ratio(pt, A, B)
                t_train.append(t); y_train.append(float(self.penetration_map[g])); g_train.append(g)

            if len(t_train) < 2:
                raise ValueError("用于训练的标样（含针入度）数量不足 2，无法拟合回归器。")

            X = np.array(t_train).reshape(-1, 1)
            y = np.array(y_train, dtype=float)

            if self.reg_group.checkedId() == 0:
                model = LinearRegression()
            elif self.reg_group.checkedId() == 2:
                model = IsotonicRegression(out_of_bounds="clip")
            else:
                model = make_pipeline(PolynomialFeatures(2, include_bias=False), Ridge(alpha=1e-3))

            if isinstance(model, IsotonicRegression):
                model.fit(X.ravel(), y); yhat = model.predict(X.ravel())
            else:
                model.fit(X, y); yhat = model.predict(X)

            rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
            q_put({"type": MSG_LOG, "message": f"[回归器(2D)] 训练样本={len(y)}，RMSE={rmse:.2f} (0.1mm)", "level":"INFO"})

            # === Step 3: 盲样预测 ===
            rows_out = []
            df_blind = df[df["Group"].isin(blind_labels)].copy()
            if df_blind.empty:
                raise ValueError("盲样在数据中没有找到匹配项。")

            for g, sub in df_blind.groupby("Group"):
                P = sub[[P1, P2]].mean().values.astype(float)
                d, t_raw, t = perp_dist_ratio(P, A, B)
                base1_pct = round((1.0 - t) * 100.0, 2)
                base2_pct = round(t * 100.0, 2)
                if isinstance(model, IsotonicRegression):
                    pen_pred = float(model.predict(np.array([t]))[0])
                else:
                    pen_pred = float(model.predict(np.array([[t]]))[0])

                grade, note = self._assign_grade_builtin(pen_pred)
                note_extra = []
                if t_raw < -0.05 or t_raw > 1.05:
                    note_extra.append("t外推已裁剪")
                note_extra.append(f"d_perp={d:.3f}")
                note_all = "; ".join(([note] if note else []) + note_extra)

                rows_out.append([
                    g, str(A_label), str(B_label),
                    f"{base1_pct:.2f}", f"{base2_pct:.2f}",
                    f"{t:.3f}", f"{d:.3f}", f"{pen_pred:.2f}", grade, note_all
                ])
            q_put({"type": MSG_UPDATE_OIL_GRADE_RESULTS, "rows": rows_out})
            out_csv = os.path.join(base_dir, "原油及沥青标号预测.csv")
            pd.DataFrame(rows_out, columns=[
                "Group","Base1","Base2","Base1(%)","Base2(%)",
                "t","d_perp","Penetration_Pred","Grade","Notes"
            ]).to_csv(out_csv, index=False, encoding="utf-8-sig")
            q_put({"type": MSG_LOG, "message": f"原油及沥青标号预测已导出：{os.path.basename(out_csv)}", "level":"INFO"})
            q_put({"type": MSG_PROCESSING_COMPLETE, "success": True, "message": "原油及沥青标号预测完成 (2D)"})
        except Exception as e:
            import traceback
            q_put({"type": MSG_LOG, "message": f"标号预测失败(2D)：{e}\n{traceback.format_exc()}", "level": "ERROR"})
            q_put({"type": MSG_PROCESSING_COMPLETE, "success": False, "message": f"标号预测失败(2D)：{e}"})

    # ====================== 三性能映射与预测 (2D) ======================
    def load_performance_csv_clicked(self):
        if not self.file_path:
            QMessageBox.warning(self, "警告", "请先选择并进行一次全局计算。")
            return
        options = QFileDialog.Options()
        path, _ = QFileDialog.getOpenFileName(self, "选择沥青性能表CSV", "", "CSV 文件 (*.csv);;所有文件 (*)", options=options)
        if not path:
            return
        try:
            df = pd.read_csv(path)
            cols = df.columns
            col_group = _find_col(cols, ["Group"])
            if not col_group:
                raise ValueError("CSV 必须包含列：Group")
            col_pen = _find_col(cols, ["Penetration", "针入度", "针入度(0.1mm)"])
            col_sp  = _find_col(cols, ["SofteningPoint", "软化点", "软化点(℃)"])
            col_duc = _find_col(cols, ["Ductility", "延度", "延度(cm)"])
            if not (col_pen and col_sp and col_duc):
                raise ValueError("CSV 必须包含针入度、软化点、延度三列（支持中英文常见列名）")
            perf_map = {}
            cnt = 0
            for _, r in df.iterrows():
                g = str(r[col_group])
                try:
                    pen = float(r[col_pen]); sp  = float(r[col_sp]); duc = float(r[col_duc])
                except:
                    continue
                perf_map[g] = {"Penetration": pen, "SofteningPoint": sp, "Ductility": duc}
                cnt += 1
            self.performance_map = perf_map
            self.lbl_perf_loaded.setText(f"已加载 {cnt} 条性能记录")
            self._add_log_message(f"性能表加载完成：{os.path.basename(path)}，条数={cnt}", "INFO")
            QMessageBox.information(self, "完成", f"性能表加载完成（{cnt} 条）。")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载失败：{e}")
            self._add_log_message(f"加载性能表失败：{e}", "ERROR")

    def start_triple_mapping_thread(self):
        if not self.file_path:
            QMessageBox.warning(self, "警告", "请先完成一次全局预处理（⚙️ 开始识别计算）。")
            return
        if self.data_processing_active:
            QMessageBox.information(self, "处理中", "当前已有数据处理任务在运行。")
            return
        if not self.performance_map:
            QMessageBox.warning(self, "警告", "尚未加载性能表。请点击“加载性能表(CSV)”。")
            return

        db_items = self.triple_db_list.selectedItems()
        blind_items = self.triple_blind_list.selectedItems()
        if not db_items or not blind_items:
            QMessageBox.warning(self, "警告", "请至少选择 1 个标样与 1 个盲样。")
            return

        db_labels = [i.text() for i in db_items]
        blind_labels = [i.text() for i in blind_items]
        degree = self.triple_reg_group.checkedId()  # 1/2/3

        self._remember_current_tab()
        self._set_button_states(True)
        self.task_queue.put({"type": MSG_PROGRESS_START, "mode": "indeterminate"})
        th = threading.Thread(target=self._worker_triple_mapping, args=(db_labels, blind_labels, degree), name="TripleMappingThread")
        th.daemon = True
        th.start()
        self._current_worker_thread = th

    def _fit_poly_ridge(self, X, y, degree):
        degree = int(max(1, min(int(degree), 5)))
        if degree == 1:
            model = make_pipeline(PolynomialFeatures(1, include_bias=False), Ridge(alpha=1e-3))
        else:
            model = make_pipeline(PolynomialFeatures(degree, include_bias=False), Ridge(alpha=1e-3))
        model.fit(X, y)
        yhat = model.predict(X)
        rmse = float(np.sqrt(np.mean((y - yhat) ** 2))) if len(y) > 0 else float("nan")
        return model, rmse

    def _worker_triple_mapping(self, db_labels, blind_labels, degree):
        q_put = self.task_queue.put
        try:
            base_dir = os.path.dirname(self.file_path)
            data_path = os.path.join(base_dir, "数值缩放后_data.csv")
            if not os.path.exists(data_path):
                raise FileNotFoundError("缺少 数值缩放后_data.csv，请先执行一次全局计算。")
            df = pd.read_csv(data_path)
            df["Group"] = df["Group"].astype(str)
            P1, P2 = IDENTITY_POSITION_COL_1, IDENTITY_POSITION_COL_2
            if P1 not in df.columns or P2 not in df.columns:
                raise ValueError(f"数据缺少列：{P1}/{P2}（请先执行全局计算）")

            df_db = df[df["Group"].isin(db_labels)].copy()
            if df_db.empty:
                raise ValueError("标样在数据中没有找到匹配项。")

            group_means = (df_db.groupby("Group")[[P1, P2]].mean().reset_index())
            X_train_list, y_pen_list, y_sp_list, y_duc_list = [], [], [], []
            valid_groups = []
            for _, r in group_means.iterrows():
                g = str(r["Group"])
                if g not in self.performance_map:
                    continue
                X_train_list.append([float(r[P1]), float(r[P2])])
                y_pen_list.append(float(self.performance_map[g]["Penetration"]))
                y_sp_list.append(float(self.performance_map[g]["SofteningPoint"]))
                y_duc_list.append(float(self.performance_map[g]["Ductility"]))
                valid_groups.append(g)

            if len(X_train_list) < 2:
                raise ValueError("用于训练的标样（含完整三性能）数量不足 2，无法拟合回归器。")

            X_train = np.array(X_train_list, dtype=float)
            y_pen = np.array(y_pen_list, dtype=float)
            y_sp  = np.array(y_sp_list, dtype=float)
            y_duc = np.array(y_duc_list, dtype=float)

            mdl_pen, rmse_pen = self._fit_poly_ridge(X_train, y_pen, degree)
            mdl_sp,  rmse_sp  = self._fit_poly_ridge(X_train, y_sp,  degree)
            mdl_duc, rmse_duc = self._fit_poly_ridge(X_train, y_duc, degree)

            q_put({"type": MSG_LOG, "message": f"[三性能映射训练(2D)] 样本={len(X_train)}, RMSE(P/S/D)={rmse_pen:.2f}/{rmse_sp:.2f}/{rmse_duc:.2f}", "level":"INFO"})

            # —— 训练表（真值/拟合/误差）——
            train_rows = []
            y_pen_hat = mdl_pen.predict(X_train)
            y_sp_hat  = mdl_sp.predict(X_train)
            y_duc_hat = mdl_duc.predict(X_train)
            for i, g in enumerate(valid_groups):
                train_rows.append([
                    g,
                    f"{y_pen[i]:.2f}", f"{y_pen_hat[i]:.2f}", f"{(y_pen_hat[i]-y_pen[i]):.2f}",
                    f"{y_sp[i]:.2f}",  f"{y_sp_hat[i]:.2f}",  f"{(y_sp_hat[i]-y_sp[i]):.2f}",
                    f"{y_duc[i]:.2f}", f"{y_duc_hat[i]:.2f}", f"{(y_duc_hat[i]-y_duc[i]):.2f}",
                ])
            q_put({"type": MSG_UPDATE_TRIPLE_TRAIN_TABLE, "rows": train_rows, "rmse_text": f"RMSE(P/S/D)：{rmse_pen:.2f} / {rmse_sp:.2f} / {rmse_duc:.2f}"})

            # 预测：盲样组质心
            df_blind = df[df["Group"].isin(blind_labels)].copy()
            if df_blind.empty:
                raise ValueError("盲样在数据中没有找到匹配项。")

            rows_out = []
            for g, sub in df_blind.groupby("Group"):
                P = sub[[P1, P2]].mean().values.astype(float).reshape(1, -1)
                pen_pred = float(mdl_pen.predict(P)[0])
                sp_pred  = float(mdl_sp.predict(P)[0])
                duc_pred = float(mdl_duc.predict(P)[0])
                grade, note = self._assign_grade_builtin(pen_pred)
                train_rmse_text = f"{rmse_pen:.2f}/{rmse_sp:.2f}/{rmse_duc:.2f}"
                rows_out.append([
                    str(g), f"{pen_pred:.2f}", f"{sp_pred:.2f}", f"{duc_pred:.2f}",
                    grade, train_rmse_text, ""
                ])

            q_put({"type": MSG_UPDATE_TRIPLE_RESULTS, "rows": rows_out})
            out_csv = os.path.join(base_dir, "三性能预测.csv")
            pd.DataFrame(rows_out, columns=[
                "Group","Penetration_Pred(0.1mm)","SofteningPoint_Pred(℃)","Ductility_Pred(cm)",
                "Grade","Train_RMSE(P/S/D)","Notes"
            ]).to_csv(out_csv, index=False, encoding="utf-8-sig")
            q_put({"type": MSG_LOG, "message": f"三性能预测已导出：{os.path.basename(out_csv)}", "level":"INFO"})
            q_put({"type": MSG_PROCESSING_COMPLETE, "success": True, "message": "三性能映射与预测完成 (2D)"})
        except Exception as e:
            import traceback
            q_put({"type": MSG_LOG, "message": f"三性能映射与预测失败(2D)：{e}\n{traceback.format_exc()}", "level": "ERROR"})
            q_put({"type": MSG_PROCESSING_COMPLETE, "success": False, "message": f"三性能映射与预测失败(2D)：{e}"})

    # ====================== 3D PCA 交互图 ======================
    def _load_scores_csv_for_3d(self):
        if not self.file_path:
            raise FileNotFoundError("未选择数据文件。")
        base_dir = os.path.dirname(self.file_path)
        path_scores = os.path.join(base_dir, "身份位置得分数据.csv")
        if not os.path.exists(path_scores):
            raise FileNotFoundError("缺少 身份位置得分数据.csv，请先执行一次全局计算。")
        df = pd.read_csv(path_scores)
        cols = df.columns.tolist()
        need_cols = ["Group", "内部成分1", "内部成分2", "内部成分3"]
        for c in need_cols:
            if c not in cols:
                raise ValueError("得分表缺少必要列：" + c)
        return df

    def update_pca3d_plot(self):
        try:
            df = self._load_scores_csv_for_3d()
        except Exception as e:
            QMessageBox.warning(self, "提示", f"无法加载3D PCA 数据：{e}")
            return

        selected = self.list_pca3d_groups.selectedItems()
        if not selected:
            # 默认全选一次
            self.list_pca3d_groups.selectAll()
            selected = self.list_pca3d_groups.selectedItems()
        chosen = set([it.text() for it in selected])

        ax = self.plot3d_container.get_axes()
        ax.clear()

        # 颜色映射按 group 名称离散
        all_groups = sorted(df["Group"].unique().tolist())
        cmap = plt.cm.get_cmap('tab20', len(all_groups) if len(all_groups)>0 else 1)
        color_map = {g: cmap(i % 20) for i, g in enumerate(all_groups)}

        # 绘制点与标签
        shown_pts = []
        for idx, row in df.iterrows():
            g = str(row["Group"])
            if g not in chosen:
                continue
            x = float(row["内部成分1"]); y = float(row["内部成分2"]); z = float(row["内部成分3"])
            ax.scatter(x, y, z, s=18, color=color_map[g], alpha=0.9)
            shown_pts.append([x, y, z, g])

        # 标签：Group-序号（同Group内的计数）
        if self.chk3d_labels.isChecked() and shown_pts:
            # 为每个 group 计数
            counter = {}
            for x, y, z, g in shown_pts:
                counter[g] = counter.get(g, 0) + 1
                ax.text(x, y, z, f"{g}-{counter[g]}", fontsize=7)

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title("PCA 得分图 (3D 交互, 可拖动/滚轮缩放)")

        # 自动缩放：根据所选点的边界设置范围
        if shown_pts:
            pts = np.array([[x,y,z] for x,y,z,_ in shown_pts], dtype=float)
            mins = pts.min(axis=0); maxs = pts.max(axis=0)
            spans = np.maximum(maxs - mins, 1e-6)
            pad = spans * 0.1
            ax.set_xlim(mins[0]-pad[0], maxs[0]+pad[0])
            ax.set_ylim(mins[1]-pad[1], maxs[1]+pad[1])
            ax.set_zlim(mins[2]-pad[2], maxs[2]+pad[2])

        self.plot3d_container._perform_debounced_redraw()
        self._add_log_message("3D PCA 图已更新。", "INFO")

    # ====================== 3D 相似度检测 ======================
    def start_similarity3d_thread(self):
        if not self.file_path:
            QMessageBox.warning(self, "警告", "请先完成一次全局预处理（⚙️ 开始识别计算）。")
            return
        if self.data_processing_active:
            QMessageBox.information(self, "处理中", "当前已有数据处理任务在运行。")
            return
        self._remember_current_tab()
        self._set_button_states(True)
        self.task_queue.put({"type": MSG_PROGRESS_START, "mode": "indeterminate"})
        th = threading.Thread(target=self._worker_similarity3d, name="Similarity3DThread")
        th.daemon = True
        th.start()
        self._current_worker_thread = th

    def _worker_similarity3d(self):
        q_put = self.task_queue.put
        try:
            base_dir = os.path.dirname(self.file_path)
            score_path = os.path.join(base_dir, "身份位置得分数据.csv")
            data_path = os.path.join(base_dir, "数值缩放后_data.csv")
            if not os.path.exists(score_path):
                raise FileNotFoundError("缺少 身份位置得分数据.csv，请先执行一次全局计算。")
            if not os.path.exists(data_path):
                raise FileNotFoundError("缺少 数值缩放后_data.csv，请先执行一次全局计算。")

            df_score = pd.read_csv(score_path)
            if not set(["Group","内部成分1","内部成分2","内部成分3"]).issubset(df_score.columns):
                raise ValueError("得分表缺少 PC1/PC2/PC3 所需列。")
            df_score["Group"] = df_score["Group"].astype(str)

            df_data = pd.read_csv(data_path)
            wavenumbers = df_data.columns[2:]
            spectra = df_data[wavenumbers].values
            spectra = np.nan_to_num(spectra)
            pca_avg = PCA(n_components=2).fit(spectra)

            # major label（与2D一致：按“-”前缀聚合）
            unique_labels = sorted(set([g.split("-")[0] for g in df_score["Group"].unique().tolist()]))
            sim_mat = pd.DataFrame(index=unique_labels, columns=unique_labels, dtype=float)

            sigma = 0.25  # 3D 距离的尺度参数，可按需要调整

            for a in unique_labels:
                for b in unique_labels:
                    # 三维质心相似度
                    A_pts = df_score[df_score["Group"].str.startswith(a)][["内部成分1","内部成分2","内部成分3"]].values
                    B_pts = df_score[df_score["Group"].str.startswith(b)][["内部成分1","内部成分2","内部成分3"]].values
                    sim3d = 0.0
                    if len(A_pts)>0 and len(B_pts)>0:
                        cA = A_pts.mean(axis=0); cB = B_pts.mean(axis=0)
                        dist3 = float(np.linalg.norm(cA - cB))
                        sim3d = np.exp(-(dist3**2)/(2*sigma**2))

                    # 平均谱 PCA 距离（与2D相同思路）
                    avgA = df_data[df_data["Group"].str.startswith(a)][wavenumbers].mean().values.reshape(1,-1)
                    avgB = df_data[df_data["Group"].str.startswith(b)][wavenumbers].mean().values.reshape(1,-1)
                    if np.isnan(avgA).all() or np.isnan(avgB).all():
                        sim_avg = 0.0
                    else:
                        ta = pca_avg.transform(np.nan_to_num(avgA)); tb = pca_avg.transform(np.nan_to_num(avgB))
                        dist_avg = float(np.linalg.norm(ta - tb))
                        sim_avg = 1/(1+dist_avg)

                    sim = 0.2*sim3d + 0.8*sim_avg
                    sim_mat.loc[a,b] = sim

            sim_mat.to_csv(os.path.join(base_dir,"相似度矩阵_3D.csv"), encoding="utf-8-sig")
            # 生成对照表
            labelled = []
            used = set()
            for a in unique_labels:
                for b in unique_labels:
                    k = tuple(sorted((a,b)))
                    if k in used: continue
                    used.add(k)
                    val = float(sim_mat.loc[a,b])
                    if a == b: std = "同一品牌相同批次沥青 (自身对比)"
                    elif val <= 0.80: std = "不同品牌沥青"
                    elif 0.80 < val <= 0.90: std = "同一品牌不同批次沥青"
                    else: std = "同一品牌相同批次沥青"
                    labelled.append((f"{a}-{b}", val, std))
            labelled.sort(key=lambda x: x[0])
            pd.DataFrame(labelled, columns=["Category Pair","Similarity","Standard"]).to_csv(os.path.join(base_dir,"相似度对照表_3D.csv"), index=False, encoding="utf-8-sig")

            q_put({"type": MSG_UPDATE_SIMILARITY3D_TABLE, "data": labelled})
            q_put({"type": MSG_PROCESSING_COMPLETE, "success": True, "message": "3D 相似度检测完成"})
        except Exception as e:
            import traceback
            q_put({"type": MSG_LOG, "message": f"3D 相似度检测失败：{e}\n{traceback.format_exc()}", "level": "ERROR"})
            q_put({"type": MSG_PROCESSING_COMPLETE, "success": False, "message": f"3D 相似度检测失败：{e}"})

    # ====================== 3D 原油及沥青标号预测 ======================
    def start_match_oil_grade3d_thread(self):
        if not self.file_path:
            QMessageBox.warning(self, "警告", "请先完成一次全局预处理（⚙️ 开始识别计算）。")
            return
        if self.data_processing_active:
            QMessageBox.information(self, "处理中", "当前已有数据处理任务在运行。")
            return
        db_items = self.db3_list.selectedItems()
        blind_items = self.blind3_list.selectedItems()
        if not db_items or not blind_items:
            QMessageBox.warning(self, "警告", "请在3D页选择标样与盲样。")
            return
        if len(db_items) < MIN_GROUPS_FOR_SERIES:
            QMessageBox.warning(self, "警告", f"所选标样不足 {MIN_GROUPS_FOR_SERIES} 组，无法进行线性检测与端元识别。")
            return

        db_labels = [i.text() for i in db_items]
        blind_labels = [i.text() for i in blind_items]
        reg_id = self.reg3_group.checkedId()

        self._remember_current_tab()
        self._set_button_states(True)
        self.task_queue.put({"type": MSG_PROGRESS_START, "mode": "indeterminate"})
        th = threading.Thread(target=self._worker_match_oil_grade_3d, args=(db_labels, blind_labels, reg_id), name="MatchOilGrade3DThread")
        th.daemon = True
        th.start()
        self._current_worker_thread = th

    def _worker_match_oil_grade_3d(self, db_labels, blind_labels, reg_id):
        q_put = self.task_queue.put
        try:
            base_dir = os.path.dirname(self.file_path)
            score_path = os.path.join(base_dir, "身份位置得分数据.csv")
            if not os.path.exists(score_path):
                raise FileNotFoundError("缺少 身份位置得分数据.csv，请先执行一次全局计算。")
            df = pd.read_csv(score_path)
            if not set(["Group","内部成分1","内部成分2","内部成分3"]).issubset(df.columns):
                raise ValueError("得分表缺少 PC1/PC2/PC3 所需列。")
            df["Group"] = df["Group"].astype(str)

            df_db = df[df["Group"].isin(db_labels)].copy()
            if df_db.empty:
                raise ValueError("标样在得分表中没有找到匹配项。")

            # 3D 质心
            gmean = df_db.groupby("Group")[["内部成分1","内部成分2","内部成分3"]].mean().reset_index()
            if gmean.shape[0] < MIN_GROUPS_FOR_SERIES:
                raise ValueError(f"标样有效组数不足 {MIN_GROUPS_FOR_SERIES}。")
            Xc = gmean[["内部成分1","内部成分2","内部成分3"]].values.astype(float)

            pca = PCA(n_components=3).fit(Xc)
            ratio1 = float(pca.explained_variance_ratio_[0])
            u = pca.components_[0]  # 主方向
            s = (Xc - Xc.mean(axis=0)) @ u
            iA = int(np.argmin(s)); iB = int(np.argmax(s))
            A_label = gmean.iloc[iA]["Group"]; B_label = gmean.iloc[iB]["Group"]
            A = Xc[iA]; B = Xc[iB]
            AB = B - A; L = float(np.linalg.norm(AB))
            if not np.isfinite(L) or L < EPS_AB_MIN:
                raise ValueError("端元距离过小，无法进行投影。")

            def perp3(point, A, B):
                AB = B - A; L2 = float(np.dot(AB, AB))
                if L2 < EPS_AB_MIN: return np.inf, 0.0, 0.0
                t_raw = float(np.dot(point - A, AB) / L2)
                t = float(np.clip(t_raw, 0.0, 1.0))
                proj = A + t*AB
                d = float(np.linalg.norm(point - proj) / np.sqrt(L2))
                return d, t_raw, t

            # 线性性指标（3D）
            d_list = []
            for pt in Xc:
                d,_,_ = perp3(pt, A, B)
                d_list.append(d)
            rms_perp = float(np.sqrt(np.mean(np.square(d_list))))
            max_perp = float(np.max(d_list))
            q_put({"type": MSG_LOG, "message": f"[标样线性检测(3D)] 端元={A_label}/{B_label}，PC1={ratio1:.3f}，RMS⊥={rms_perp:.3f}，MAX⊥={max_perp:.3f}", "level":"INFO"})
            if not (ratio1 >= LINEAR_PCA_RATIO_MIN and rms_perp <= LINEAR_RMS_PERP_RATIO_MAX and max_perp <= LINEAR_MAX_PERP_RATIO_MAX):
                raise ValueError("标样未呈现明显线性系列（3D PC1解释率/正交残差未过阈）。")

            if not self.penetration_map:
                raise ValueError("尚未加载针入度表。请点击“加载针入度值(CSV)”。")

            # 训练集（t→Pen）
            t_train, y_train = [], []
            for _, r in gmean.iterrows():
                g = r["Group"]
                if g not in self.penetration_map:
                    continue
                P = np.array([r["内部成分1"],r["内部成分2"],r["内部成分3"]], dtype=float)
                d, t_raw, t = perp3(P, A, B)
                t_train.append(t); y_train.append(float(self.penetration_map[g]))
            if len(t_train) < 2:
                raise ValueError("用于训练的标样（含针入度）数量不足 2，无法拟合回归器。")
            X = np.array(t_train).reshape(-1,1); y = np.array(y_train, dtype=float)
            if reg_id == 0: model = LinearRegression()
            elif reg_id == 2: model = IsotonicRegression(out_of_bounds="clip")
            else: model = make_pipeline(PolynomialFeatures(2, include_bias=False), Ridge(alpha=1e-3))
            if isinstance(model, IsotonicRegression): model.fit(X.ravel(), y); yhat = model.predict(X.ravel())
            else: model.fit(X, y); yhat = model.predict(X)
            rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
            q_put({"type": MSG_LOG, "message": f"[回归器(3D)] 训练样本={len(y)}，RMSE={rmse:.2f} (0.1mm)", "level":"INFO"})

            # 盲样预测
            rows_out = []
            df_blind = df[df["Group"].isin(blind_labels)].copy()
            if df_blind.empty:
                raise ValueError("盲样在得分表中没有找到匹配项。")
            for g, sub in df_blind.groupby("Group"):
                P = sub[["内部成分1","内部成分2","内部成分3"]].mean().values.astype(float)
                d, t_raw, t = perp3(P, A, B)
                base1_pct = round((1.0 - t) * 100.0, 2)
                base2_pct = round(t * 100.0, 2)
                if isinstance(model, IsotonicRegression):
                    pen_pred = float(model.predict(np.array([t]))[0])
                else:
                    pen_pred = float(model.predict(np.array([[t]]))[0])
                grade, note = self._assign_grade_builtin(pen_pred)
                note_extra = []
                if t_raw < -0.05 or t_raw > 1.05:
                    note_extra.append("t外推已裁剪")
                note_extra.append(f"d_perp={d:.3f}")
                note_all = "; ".join(([note] if note else []) + note_extra)
                rows_out.append([
                    g, str(A_label), str(B_label),
                    f"{base1_pct:.2f}", f"{base2_pct:.2f}",
                    f"{t:.3f}", f"{d:.3f}", f"{pen_pred:.2f}", grade, note_all
                ])
            q_put({"type": MSG_UPDATE_OIL_GRADE3D_RESULTS, "rows": rows_out})
            out_csv = os.path.join(base_dir, "原油及沥青标号预测_3D.csv")
            pd.DataFrame(rows_out, columns=[
                "Group","Base1","Base2","Base1(%)","Base2(%)",
                "t","d_perp","Penetration_Pred","Grade","Notes"
            ]).to_csv(out_csv, index=False, encoding="utf-8-sig")
            q_put({"type": MSG_LOG, "message": f"3D 原油及沥青标号预测已导出：{os.path.basename(out_csv)}", "level":"INFO"})
            q_put({"type": MSG_PROCESSING_COMPLETE, "success": True, "message": "3D 原油及沥青标号预测完成"})
        except Exception as e:
            import traceback
            q_put({"type": MSG_LOG, "message": f"3D 原油及沥青标号预测失败：{e}\n{traceback.format_exc()}", "level": "ERROR"})
            q_put({"type": MSG_PROCESSING_COMPLETE, "success": False, "message": f"3D 原油及沥青标号预测失败：{e}"})

    # ====================== 3D 三性能映射与预测 ======================
    def start_triple3d_mapping_thread(self):
        if not self.file_path:
            QMessageBox.warning(self, "警告", "请先完成一次全局预处理（⚙️ 开始识别计算）。")
            return
        if self.data_processing_active:
            QMessageBox.information(self, "处理中", "当前已有数据处理任务在运行。")
            return
        if not self.performance_map:
            QMessageBox.warning(self, "警告", "尚未加载性能表（与2D共用）。")
            return
        db_items = self.triple3_db_list.selectedItems()
        blind_items = self.triple3_blind_list.selectedItems()
        if not db_items or not blind_items:
            QMessageBox.warning(self, "警告", "请在3D页至少选择 1 个标样与 1 个盲样。")
            return
        db_labels = [i.text() for i in db_items]
        blind_labels = [i.text() for i in blind_items]
        degree = self.triple3_reg_group.checkedId()

        self._remember_current_tab()
        self._set_button_states(True)
        self.task_queue.put({"type": MSG_PROGRESS_START, "mode": "indeterminate"})
        th = threading.Thread(target=self._worker_triple3d_mapping, args=(db_labels, blind_labels, degree), name="Triple3DMappingThread")
        th.daemon = True
        th.start()
        self._current_worker_thread = th

    def _worker_triple3d_mapping(self, db_labels, blind_labels, degree):
        q_put = self.task_queue.put
        try:
            base_dir = os.path.dirname(self.file_path)
            score_path = os.path.join(base_dir, "身份位置得分数据.csv")
            if not os.path.exists(score_path):
                raise FileNotFoundError("缺少 身份位置得分数据.csv，请先执行一次全局计算。")
            df = pd.read_csv(score_path)
            if not set(["Group","内部成分1","内部成分2","内部成分3"]).issubset(df.columns):
                raise ValueError("得分表缺少 PC1/PC2/PC3 所需列。")
            df["Group"] = df["Group"].astype(str)

            df_db = df[df["Group"].isin(db_labels)].copy()
            if df_db.empty:
                raise ValueError("标样在得分表中没有找到匹配项。")

            gm = df_db.groupby("Group")[["内部成分1","内部成分2","内部成分3"]].mean().reset_index()
            X_train_list, y_pen_list, y_sp_list, y_duc_list = [], [], [], []
            valid_groups = []
            for _, r in gm.iterrows():
                g = str(r["Group"])
                if g not in self.performance_map:
                    continue
                X_train_list.append([float(r["内部成分1"]), float(r["内部成分2"]), float(r["内部成分3"])])
                y_pen_list.append(float(self.performance_map[g]["Penetration"]))
                y_sp_list.append(float(self.performance_map[g]["SofteningPoint"]))
                y_duc_list.append(float(self.performance_map[g]["Ductility"]))
                valid_groups.append(g)

            if len(X_train_list) < 2:
                raise ValueError("用于训练的标样（含完整三性能）数量不足 2，无法拟合回归器。")

            X_train = np.array(X_train_list, dtype=float)
            y_pen = np.array(y_pen_list, dtype=float)
            y_sp  = np.array(y_sp_list, dtype=float)
            y_duc = np.array(y_duc_list, dtype=float)

            def fit_poly(X, y, degree):
                degree = int(max(1, min(int(degree), 5)))
                if degree == 1:
                    model = make_pipeline(PolynomialFeatures(1, include_bias=False), Ridge(alpha=1e-3))
                else:
                    model = make_pipeline(PolynomialFeatures(degree, include_bias=False), Ridge(alpha=1e-3))
                model.fit(X, y); pred = model.predict(X)
                rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
                return model, pred, rmse

            mdl_pen, pen_hat, rmse_pen = fit_poly(X_train, y_pen, degree)
            mdl_sp,  sp_hat,  rmse_sp  = fit_poly(X_train, y_sp,  degree)
            mdl_duc, duc_hat, rmse_duc = fit_poly(X_train, y_duc, degree)

            q_put({"type": MSG_LOG, "message": f"[三性能映射训练(3D)] 样本={len(X_train)}, RMSE(P/S/D)={rmse_pen:.2f}/{rmse_sp:.2f}/{rmse_duc:.2f}", "level":"INFO"})

            # 训练表
            rows_train = []
            for i, g in enumerate(valid_groups):
                rows_train.append([
                    g,
                    f"{y_pen[i]:.2f}", f"{pen_hat[i]:.2f}", f"{(pen_hat[i]-y_pen[i]):.2f}",
                    f"{y_sp[i]:.2f}",  f"{sp_hat[i]:.2f}",  f"{(sp_hat[i]-y_sp[i]):.2f}",
                    f"{y_duc[i]:.2f}", f"{duc_hat[i]:.2f}", f"{(duc_hat[i]-y_duc[i]):.2f}",
                ])
            q_put({"type": MSG_UPDATE_TRIPLE3D_TRAIN_TABLE, "rows": rows_train, "rmse_text": f"RMSE(P/S/D)：{rmse_pen:.2f} / {rmse_sp:.2f} / {rmse_duc:.2f}"})

            # 盲样预测
            df_blind = df[df["Group"].isin(blind_labels)].copy()
            if df_blind.empty:
                raise ValueError("盲样在得分表中没有找到匹配项。")
            rows_out = []
            for g, sub in df_blind.groupby("Group"):
                P = sub[["内部成分1","内部成分2","内部成分3"]].mean().values.astype(float).reshape(1,-1)
                pen_pred = float(mdl_pen.predict(P)[0])
                sp_pred  = float(mdl_sp.predict(P)[0])
                duc_pred = float(mdl_duc.predict(P)[0])
                grade, note = self._assign_grade_builtin(pen_pred)
                train_rmse_text = f"{rmse_pen:.2f}/{rmse_sp:.2f}/{rmse_duc:.2f}"
                rows_out.append([
                    str(g), f"{pen_pred:.2f}", f"{sp_pred:.2f}", f"{duc_pred:.2f}",
                    grade, train_rmse_text, ""
                ])
            q_put({"type": MSG_UPDATE_TRIPLE3D_RESULTS, "rows": rows_out})
            out_csv = os.path.join(base_dir, "三性能预测_3D.csv")
            pd.DataFrame(rows_out, columns=[
                "Group","Penetration_Pred(0.1mm)","SofteningPoint_Pred(℃)","Ductility_Pred(cm)",
                "Grade","Train_RMSE(P/S/D)","Notes"
            ]).to_csv(out_csv, index=False, encoding="utf-8-sig")
            q_put({"type": MSG_LOG, "message": f"3D 三性能预测已导出：{os.path.basename(out_csv)}", "level":"INFO"})
            q_put({"type": MSG_PROCESSING_COMPLETE, "success": True, "message": "三性能映射与预测完成 (3D)"})
        except Exception as e:
            import traceback
            q_put({"type": MSG_LOG, "message": f"三性能映射与预测失败(3D)：{e}\n{traceback.format_exc()}", "level": "ERROR"})
            q_put({"type": MSG_PROCESSING_COMPLETE, "success": False, "message": f"三性能映射与预测失败(3D)：{e}"})

    # ---------------- 关闭事件 ----------------
    def closeEvent(self, event):
        reply = QMessageBox.question(self, '确认退出', "您确定要退出系统吗?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self._add_log_message("系统正在关闭...", "INFO")
            if self._current_worker_thread and self._current_worker_thread.is_alive(): self._add_log_message("警告：后台任务仍在运行...", "WARNING")
            self.queue_poll_timer.stop(); event.accept()
        else: event.ignore()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = AsphaltInfraredAppPyQt()
    main_win.show()
    sys.exit(app.exec_())
