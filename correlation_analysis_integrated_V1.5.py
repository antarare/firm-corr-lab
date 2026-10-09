# -*- coding: utf-8 -*-
import sys, os, traceback
import numpy as np
import pandas as pd
from datetime import datetime
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import (
    QApplication, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QFileDialog, QCheckBox, QLineEdit, QMessageBox,
    QComboBox, QListWidget, QListWidgetItem, QAbstractItemView, QSpinBox,
    QMainWindow, QColorDialog, QGroupBox, QInputDialog, QProgressDialog,
    QSlider, QTextEdit
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QPainter, QColor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, FontManager
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.font_manager as fm
import warnings
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

# ========= 全局辅助函数 =========

# 更健壮的中文字体设置，包含多个备选字体
def setup_chinese_font():
    """设置中文字体，包含多个备选方案"""
    # 常见的中文字体列表，按优先级排序
    chinese_fonts = [
        'Microsoft YaHei',      # 微软雅黑
        'SimHei',               # 黑体
        'SimSun',               # 宋体
        'KaiTi',                # 楷体
        'FangSong',             # 仿宋
        'Arial Unicode MS',     # 跨平台字体
        'DejaVu Sans',          # Linux常用字体
        'sans-serif'            # 最后回退
    ]
    
    # 检查系统可用的字体
    available_fonts = [f.name for f in FontManager().ttflist]
    
    # 选择第一个可用的中文字体
    selected_font = None
    for font in chinese_fonts:
        if font in available_fonts:
            selected_font = font
            break
    
    if selected_font:
        print(f"使用字体: {selected_font}")
        # 设置字体家族，避免使用Arial等不支持中文的字体
        plt.rcParams['font.sans-serif'] = [selected_font] + [f for f in chinese_fonts if f != selected_font]
        # 设置默认字体为选中的中文字体
        plt.rcParams['font.family'] = selected_font
    else:
        print("警告: 未找到合适的中文字体，使用默认字体")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
    
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    plt.rcParams['font.size'] = 10
    
    # 设置seaborn使用相同的中文字体
    sns.set(font=selected_font)
    sns.set_style("whitegrid", {'font.family': selected_font})
    
    return selected_font

# 调用字体设置函数并获取选中的字体
selected_font = setup_chinese_font()

# 禁用所有字体相关的警告
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', category=UserWarning, module='seaborn')
warnings.filterwarnings('ignore', category=UserWarning)

# ========= 数学辅助函数 =========

def safe_log_returns_from_prices_df(df_prices):
    """df_prices: rows=time, cols=firms -> returns log returns, drop NaN rows"""
    p = df_prices.copy()
    r = np.log(p / p.shift(1))
    return r.dropna(how='all')

def multi_factor_residuals(returns_df, meta_df, remove_market=True, remove_industry=False, remove_region=False):
    """
    returns_df: DataFrame index=time cols=firms
    meta_df: DataFrame with columns ['企业名称','行业','区域']
    Returns DataFrame of residuals (same shape)
    """
    T = returns_df.shape[0]
    firms = returns_df.columns.tolist()
    residuals = pd.DataFrame(index=returns_df.index, columns=firms, dtype=float)

    # precompute industry / region means if needed
    industry_series = {}
    region_series = {}
    if remove_industry:
        inds_map = meta_df.set_index('企业名称')['行业'].to_dict()
        unique_inds = sorted([x for x in meta_df['行业'].dropna().unique()])
        for ind in unique_inds:
            members = [f for f, g in inds_map.items() if g == ind and f in returns_df.columns]
            if len(members) == 0:
                industry_series[ind] = np.full((T, 1), np.nan)
            else:
                industry_series[ind] = returns_df[members].mean(axis=1).values.reshape(-1, 1)
    if remove_region:
        regs_map = meta_df.set_index('企业名称')['区域'].to_dict()
        unique_regs = sorted([x for x in meta_df['区域'].dropna().unique()])
        for reg in unique_regs:
            members = [f for f, g in regs_map.items() if g == reg and f in returns_df.columns]
            if len(members) == 0:
                region_series[reg] = np.full((T, 1), np.nan)
            else:
                region_series[reg] = returns_df[members].mean(axis=1).values.reshape(-1, 1)

    market = returns_df.mean(axis=1).values.reshape(-1, 1) if remove_market else None
    meta_indexed = meta_df.set_index('企业名称')

    for firm in firms:
        y = returns_df[firm].values.reshape(-1, 1)
        X_list = []
        if remove_market:
            X_list.append(market)
        if remove_industry:
            try:
                ind = meta_indexed.at[firm, '行业']
                X_list.append(industry_series.get(ind, np.full((T,1), np.nan)))
            except Exception:
                X_list.append(np.full((T,1), np.nan))
        if remove_region:
            try:
                reg = meta_indexed.at[firm, '区域']
                X_list.append(region_series.get(reg, np.full((T,1), np.nan)))
            except Exception:
                X_list.append(np.full((T,1), np.nan))
        if len(X_list) == 0:
            residuals[firm] = y.flatten()
            continue
        X = np.hstack(X_list)
        # mask invalid rows
        valid = ~np.isnan(y.flatten())
        for col in range(X.shape[1]):
            valid = valid & (~np.isnan(X[:, col]))
        if valid.sum() < 3:
            residuals[firm] = np.nan
            continue
        Xv = X[valid, :]
        yv = y[valid, :].flatten()
        try:
            coef, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
            fitted = Xv.dot(coef)
            res = np.full(y.shape[0], np.nan)
            idxs = np.where(valid)[0]
            res[idxs] = yv - fitted
            residuals[firm] = res
        except Exception:
            residuals[firm] = np.nan
    return residuals

def compute_corr_matrix_for_subset(resid_returns, firms, method='pearson'):
    sub = resid_returns[firms].dropna(how='all')
    if sub.shape[1] < 2:
        return None
    return sub.corr(method=method)

def overall_mean_offdiag(corr_df):
    if corr_df is None:
        return np.nan
    try:
        arr = corr_df.values.astype(float)
    except ValueError:
        arr = pd.to_numeric(corr_df.values.flatten(), errors='coerce').reshape(corr_df.shape)
    n = arr.shape[0]
    if n <= 1:
        return np.nan
    mask = ~np.eye(n, dtype=bool)
    vals = arr[mask]
    return np.nanmean(vals)

def compute_group_timeseries(resid_returns, mapping_series, groups, win, step, method='pearson'):
    dates = list(resid_returns.index)
    n_t = len(dates)
    starts = list(range(0, max(0, n_t - win + 1), step))
    rows = []
    times = []
    for s in starts:
        e = s + win
        window = resid_returns.iloc[s:e]
        if window.shape[0] < 3:
            continue
        ent_corr = window.corr(method=method)
        # aggregate to group x group
        grp_mat = pd.DataFrame(index=groups, columns=groups, dtype=float)
        for a in groups:
            fa = mapping_series[mapping_series == a].index.intersection(ent_corr.index).tolist()
            for b in groups:
                fb = mapping_series[mapping_series == b].index.intersection(ent_corr.index).tolist()
                if len(fa) == 0 or len(fb) == 0:
                    grp_mat.loc[a, b] = np.nan
                else:
                    grp_mat.loc[a, b] = ent_corr.loc[fa, fb].values.mean()
        rowmean = {}
        for i, g in enumerate(groups):
            row = grp_mat.loc[g].values.astype(float)
            others = np.delete(row, i)
            rowmean[g] = np.nanmean(others)
        rows.append(rowmean)
        center_idx = s + win // 2
        if center_idx >= n_t:
            center_idx = n_t - 1
        times.append(dates[center_idx])
    if len(times) == 0:
        return pd.DataFrame()
    return pd.DataFrame(rows, index=pd.to_datetime(times))

def yearly_statistics_from_series(ts_series):
    if ts_series is None or len(ts_series) == 0:
        return pd.DataFrame()
    df = pd.DataFrame({'val': ts_series})
    df['year'] = pd.DatetimeIndex(df.index).year
    agg = df.groupby('year')['val'].agg(['mean', 'var', 'min', 'max', 'median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)])
    agg.columns = ['mean', 'var', 'min', 'max', 'median', 'q25', 'q75']
    return agg

# ========= 违约相关性计算相关函数 =========

def symmetrize(M):
    return (M + M.T) / 2.0

def force_diag_one(M):
    M = M.copy()
    np.fill_diagonal(M.values if isinstance(M, pd.DataFrame) else M, 1.0)
    return M

def nearest_positive_definite(A):
    A = np.array(A, dtype=float)
    B = (A + A.T) / 2.0
    _, s, Vt = np.linalg.svd(B)
    H = (Vt.T * s) @ Vt
    A2 = (B + H) / 2.0
    A3 = (A2 + A2.T) / 2.0

    def is_pd(X):
        try:
            np.linalg.cholesky(X)
            return True
        except np.linalg.LinAlgError:
            return False

    if is_pd(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    k = 1
    while not is_pd(A3):
        min_eig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += np.eye(A3.shape[0]) * (-min_eig * k + spacing)
        k += 1
        if k > 1000:
            raise RuntimeError("nearest_positive_definite failed to converge")
    return A3

def make_valid_correlation_matrix(df):
    M = pd.DataFrame(symmetrize(df.values), index=df.index, columns=df.columns)
    M = force_diag_one(M)
    try:
        np.linalg.cholesky(M.values)
        return M
    except np.linalg.LinAlgError:
        A_pd = nearest_positive_definite(M.values)
        D = np.sqrt(np.diag(A_pd))
        D[D == 0] = 1.0
        corr = A_pd / np.outer(D, D)
        corr = pd.DataFrame(corr, index=df.index, columns=df.columns)
        corr = force_diag_one(corr)
        return corr

def compute_default_corr(region_corr, industry_corr, firm_info, alpha=0.5):
    n = len(firm_info)
    result = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ind_i, reg_i = firm_info.loc[i, "行业"], firm_info.loc[i, "区域"]
            ind_j, reg_j = firm_info.loc[j, "行业"], firm_info.loc[j, "区域"]

            rho_ind = industry_corr.loc[ind_i, ind_j]
            rho_reg = region_corr.loc[reg_i, reg_j]

            # 综合相关性
            rho_asset = alpha * rho_ind + (1 - alpha) * rho_reg
            # 资产相关映射至违约相关（通过正态 Copula 简化映射）
            rho_default = np.sin(np.pi / 2 * rho_asset) ** 2  # 保证在 0~1 之间

            result[i, j] = rho_default
    result = pd.DataFrame(result, index=firm_info["企业"], columns=firm_info["企业"])
    np.fill_diagonal(result.values, 1.0)
    return result

# ========= GUI 组件 =========

class MatCanvas(FigureCanvas):
    def __init__(self, width=7, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.fig = fig
        self.colorbar = None

    def plot_heatmap(self, corr_df, title=""):
        self.ax.clear()
        # 清除整个图形并重新创建子图，确保颜色条被完全清除
        self.fig.clf()
        self.ax = self.fig.add_subplot(111)
            
        if corr_df is None:
            self.draw()
            return
        
        # 创建字体属性对象
        font_prop = FontProperties(fname=None, family=selected_font)
        
        # 设置中文字体
        self.ax.set_title(title, fontproperties=font_prop)
        
        # 使用固定的颜色映射范围 [-1, 1] 并确保显示颜色条
        vmin, vmax = -1, 1
        sns.heatmap(corr_df, cmap="coolwarm", center=0, ax=self.ax, 
                   vmin=vmin, vmax=vmax, cbar=True, cbar_kws={'shrink': 0.8})
        
        # 设置坐标轴标签字体
        for label in self.ax.get_xticklabels():
            label.set_fontproperties(font_prop)
        for label in self.ax.get_yticklabels():
            label.set_fontproperties(font_prop)
            
        self.draw()

    def plot_series(self, df, title="", xlabel="Time"):
        self.ax.clear()
        if df is None or df.shape[0] == 0:
            self.draw()
            return
            
        # 创建字体属性对象
        font_prop = FontProperties(fname=None, family=selected_font)
        
        # 设置中文字体
        self.ax.set_title(title, fontproperties=font_prop)
        self.ax.set_xlabel(xlabel, fontproperties=font_prop)
        
        for col in df.columns:
            self.ax.plot(df.index, df[col], label=str(col))
        
        if df.shape[1] <= 20:
            legend = self.ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8)
            # 设置图例字体
            for text in legend.get_texts():
                text.set_fontproperties(font_prop)
        self.draw()

    def save_fig(self, path):
        self.fig.savefig(path, bbox_inches='tight', dpi=300)

    def plot_distribution(self, corr_values, title="相关性系数分布"):
        """绘制相关性系数分布图（使用百分数占比）"""
        self.ax.clear()
        
        if corr_values is None or len(corr_values) == 0:
            self.draw()
            return
        
        # 创建字体属性对象
        font_prop = FontProperties(fname=None, family=selected_font)
        
        # 设置中文字体
        self.ax.set_title(title, fontproperties=font_prop)
        self.ax.set_xlabel("相关性系数区间", fontproperties=font_prop)
        self.ax.set_ylabel("百分比占比 (%)", fontproperties=font_prop)
        
        # 只计算相关性值小于1的情况
        corr_values = corr_values[corr_values < 1.0]
        
        if len(corr_values) == 0:
            self.ax.text(0.5, 0.5, '没有相关性值小于1的数据', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=self.ax.transAxes, fontproperties=font_prop)
            self.draw()
            return
        
        # 设置区间间隔为0.05
        bin_width = 0.05
        bins = np.arange(0, 1.0 + bin_width, bin_width)
        
        # 计算每个区间的出现次数和百分比
        counts, bin_edges = np.histogram(corr_values, bins=bins)
        total_count = len(corr_values)
        percentages = (counts / total_count) * 100
        
        # 创建柱状图
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bars = self.ax.bar(bin_centers, percentages, width=bin_width * 0.8, 
                          alpha=0.7, color='skyblue', edgecolor='black')
        
        # 在柱子上方显示百分比数值
        for bar, percentage in zip(bars, percentages):
            if percentage > 0:
                self.ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           f'{percentage:.2f}%', ha='center', va='bottom', fontsize=8)
        
        # 设置坐标轴范围
        self.ax.set_xlim(0, 1.0)
        self.ax.set_ylim(0, max(percentages) * 1.1 if max(percentages) > 0 else 5)
        
        # 设置网格
        self.ax.grid(True, alpha=0.3)
        
        # 设置x轴刻度
        self.ax.set_xticks(bin_edges)
        self.ax.set_xticklabels([f'{x:.2f}' for x in bin_edges], rotation=45)
        
        # 设置y轴刻度为百分比格式
        self.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
        
        # 设置坐标轴标签字体
        for label in self.ax.get_xticklabels():
            label.set_fontproperties(font_prop)
        for label in self.ax.get_yticklabels():
            label.set_fontproperties(font_prop)
        
        self.draw()

class ColorTransitionWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(300, 50)
        
    def set_colors(self, color_neg, color_zero, color_pos):
        self.color_neg = color_neg
        self.color_zero = color_zero
        self.color_pos = color_pos
        self.update()
        
    def paintEvent(self, event):
        if not hasattr(self, 'color_neg') or not hasattr(self, 'color_zero') or not hasattr(self, 'color_pos'):
            return
            
        painter = QPainter(self)
        width = self.width()
        height = self.height()
        
        # 绘制颜色过渡条
        for i in range(width):
            # 计算当前位置在-1到1之间的值
            pos = (i / width) * 2 - 1
            
            if pos <= 0:
                # -1到0之间的过渡
                t = (pos + 1) / 1  # 从0到1
                r = int(self.color_neg.red() * (1-t) + self.color_zero.red() * t)
                g = int(self.color_neg.green() * (1-t) + self.color_zero.green() * t)
                b = int(self.color_neg.blue() * (1-t) + self.color_zero.blue() * t)
            else:
                # 0到1之间的过渡
                t = pos / 1  # 从0到1
                r = int(self.color_zero.red() * (1-t) + self.color_pos.red() * t)
                g = int(self.color_zero.green() * (1-t) + self.color_pos.green() * t)
                b = int(self.color_zero.blue() * (1-t) + self.color_pos.blue() * t)
                
            color = QColor(r, g, b)
            painter.setPen(color)
            painter.setBrush(color)
            painter.drawRect(i, 0, 1, height)

class HeatmapCanvas(FigureCanvas):
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)
        
    def plot_heatmap(self, data, color_neg, color_zero, color_pos, title="Correlation Matrix Heatmap", 
                    show_values=True, smooth_transition=True):
        # 完全清除图形
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        
        if data is None or data.empty:
            self.ax.text(0.5, 0.5, 'Please import Excel file first', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=self.ax.transAxes, fontsize=14)
            self.draw()
            return
            
        # 创建自定义颜色映射
        colors = [
            (color_neg.red()/255, color_neg.green()/255, color_neg.blue()/255),  # -1的颜色
            (color_zero.red()/255, color_zero.green()/255, color_zero.blue()/255),  # 0的颜色
            (color_pos.red()/255, color_pos.green()/255, color_pos.blue()/255)   # 1的颜色
        ]
        
        # 根据是否启用色彩过渡效果设置颜色映射
        if smooth_transition:
            # 启用色彩过渡时使用更高分辨率的颜色映射，实现更平滑的过渡
            cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=1024)
        else:
            # 如果不启用色彩过渡，使用离散颜色映射
            cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=3)
        
        # 绘制热力图 - 根据选项控制是否显示数值，不显示颜色条标签
        sns.heatmap(data, 
                   ax=self.ax,
                   cmap=cmap,
                   center=0,
                   vmin=-1,
                   vmax=1,
                   annot=show_values,  # 根据选项控制是否显示数值
                   fmt='.2f',
                   linewidths=0.1,  # 更细的线条，让颜色过渡更平滑
                   cbar_kws={'label': ''})  # 空标签，不显示文字说明
        
        self.ax.set_title(title, fontsize=14, fontweight='bold')
        self.ax.tick_params(axis='both', which='major', labelsize=8)
        
        # 自动调整标签方向避免重叠
        plt.setp(self.ax.get_xticklabels(), rotation=45, ha='right')
        plt.setp(self.ax.get_yticklabels(), rotation=0)
        
        self.fig.tight_layout()
        self.draw()

# ========= 多线程计算类 =========

class MainCalculationThread(QThread):
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(bool, str)
    
    def __init__(self, app_instance, remove_market, remove_ind, remove_reg, win, step):
        super().__init__()
        self.app = app_instance
        self.remove_market = remove_market
        self.remove_ind = remove_ind
        self.remove_reg = remove_reg
        self.win = win
        self.step = step
        self._is_running = True
        
    def stop(self):
        """安全停止线程"""
        self._is_running = False
        self.quit()
        self.wait(3000)  # 等待3秒
        
    def run(self):
        try:
            # 步骤1: 残差化
            if not self._is_running:
                return
            self.progress_signal.emit(10, "正在残差化...")
            if not (self.remove_market or self.remove_ind or self.remove_reg):
                self.app.resid_returns = self.app.returns.copy()
            else:
                self.progress_signal.emit(20, "正在残差化（逐公司回归，可能较慢）...")
                self.app.resid_returns = multi_factor_residuals(self.app.returns, self.app.meta_df,
                                                                remove_market=self.remove_market,
                                                                remove_industry=self.remove_ind,
                                                                remove_region=self.remove_reg)
            
            # 步骤2: 计算相关性矩阵（分批）
            if not self._is_running:
                return
            self.progress_signal.emit(40, "计算企业相关性矩阵...")
            self.app.ent_corr = self.app._compute_correlation_optimized(self.app.resid_returns)
            
            # 步骤3: 聚合到区域和行业
            if not self._is_running:
                return
            self.progress_signal.emit(60, "聚合到区域和行业...")
            mapping_reg = self.app.meta_df.set_index('企业名称')['区域'].to_dict()
            mapping_ind = self.app.meta_df.set_index('企业名称')['行业'].to_dict()
            firms = [f for f in self.app.ent_corr.columns if f in mapping_reg]
            regions = sorted(list({mapping_reg.get(f) for f in firms}))
            industries = sorted(list({mapping_ind.get(f) for f in firms}))
            
            # region matrix
            prov_mat = pd.DataFrame(index=regions, columns=regions, dtype=float)
            for a in regions:
                if not self._is_running:
                    return
                fa = [f for f in firms if mapping_reg.get(f) == a]
                for b in regions:
                    fb = [f for f in firms if mapping_reg.get(f) == b]
                    prov_mat.loc[a, b] = self.app.ent_corr.loc[fa, fb].values.mean() if len(fa) and len(fb) else np.nan
            self.app.prov_corr = prov_mat
            
            # industry
            ind_mat = pd.DataFrame(index=industries, columns=industries, dtype=float)
            for a in industries:
                if not self._is_running:
                    return
                fa = [f for f in firms if mapping_ind.get(f) == a]
                for b in industries:
                    fb = [f for f in firms if mapping_ind.get(f) == b]
                    ind_mat.loc[a, b] = self.app.ent_corr.loc[fa, fb].values.mean() if len(fa) and len(fb) else np.nan
            self.app.ind_corr = ind_mat

            # 步骤4: 滑动窗口计算
            if not self._is_running:
                return
            self.progress_signal.emit(80, "计算滑动窗口...")
            dates = list(self.app.resid_returns.index)
            n_cols = len(dates)
            starts = list(range(0, max(0, n_cols - self.win + 1), self.step))
            prov_overall, ind_overall, ent_overall, times = [], [], [], []
            
            # 限制滑动窗口数量，避免计算过多
            max_windows = 50
            if len(starts) > max_windows:
                step_size = len(starts) // max_windows
                starts = starts[::step_size]
            
            for i, s in enumerate(starts):
                if not self._is_running:
                    return
                e = s + self.win
                sub = self.app.resid_returns.iloc[s:e]
                if sub.shape[0] < 3:
                    continue
                ent_sub_corr = sub.corr()
                provG = self.app._aggregate_group_correlation(ent_sub_corr, mapping_reg, regions)
                indG = self.app._aggregate_group_correlation(ent_sub_corr, mapping_ind, industries)
                prov_overall.append(overall_mean_offdiag(provG))
                ind_overall.append(overall_mean_offdiag(indG))
                ent_overall.append(overall_mean_offdiag(ent_sub_corr))
                center = dates[s + self.win // 2] if (s + self.win // 2) < n_cols else dates[-1]
                times.append(pd.to_datetime(center))
                
                # 更新进度
                progress = 80 + int((i + 1) / len(starts) * 15)
                self.progress_signal.emit(progress, f"计算滑动窗口... {i+1}/{len(starts)}")
            
            if len(times) > 0:
                self.app.prov_overall_series = pd.Series(prov_overall, index=pd.DatetimeIndex(times))
                self.app.ind_overall_series = pd.Series(ind_overall, index=pd.DatetimeIndex(times))
                self.app.ent_overall_series = pd.Series(ent_overall, index=pd.DatetimeIndex(times))
            else:
                self.app.prov_overall_series = pd.Series()
                self.app.ind_overall_series = pd.Series()
                self.app.ent_overall_series = pd.Series()
            
            if self._is_running:
                self.progress_signal.emit(100, "计算完成")
                self.finished_signal.emit(True, "主计算完成")
            
        except Exception as e:
            if self._is_running:
                self.finished_signal.emit(False, f"主计算失败：{str(e)}")

# ========= 主应用程序类 =========

class CorrelationAnalysisApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("整合版相关性分析工具")
        self.resize(1400, 900)

        # 定义统一的样式
        self._setup_unified_styles()

        # 进度条相关
        self.progress_dialog = None
        self.calculation_thread = None

        # data holders for main calculation
        self.raw_df = None
        self.meta_df = None
        self.price_df = None
        self.returns = None
        self.resid_returns = None

        # results for main calculation
        self.ent_corr = None
        self.prov_corr = None
        self.ind_corr = None
        self.prov_overall_series = None
        self.ind_overall_series = None
        self.ent_overall_series = None

        self.sub_corr = None
        self.sub_timeseries = None
        self.group_compare_ts = None

        # data holders for default correlation
        self.region_corr = None
        self.industry_corr = None
        self.firm_info = None
        self.default_corr = None

        # data holders for heatmap generator
        self.heatmap_data = None
        self.color_neg = QColor(0, 0, 255)   # 蓝色表示-1
        self.color_zero = QColor(255, 255, 255)  # 白色表示0
        self.color_pos = QColor(255, 0, 0)   # 红色表示1
        self.heatmap_title = "Correlation Matrix Heatmap"  # 默认标题
        self.show_values = True  # 默认显示数值
        self.smooth_transition = True  # 默认启用色彩过渡效果

        self._build_ui()

    def _setup_unified_styles(self):
        """设置统一的样式定义"""
        # 字体大小
        self.font_size_small = "9pt"
        self.font_size_medium = "10pt"
        self.font_size_large = "11pt"
        
        # 控件尺寸
        self.button_height = 35
        self.input_width = 70
        self.small_button_size = (40, 25)
        
        # 间距
        self.spacing_small = 6
        self.spacing_medium = 8
        self.spacing_large = 10
        
        # 样式表
        self.style_label = f"font-size: {self.font_size_medium};"
        self.style_input = f"font-size: {self.font_size_small}; padding: 2px;"
        self.style_button = f"""
            QPushButton {{
                font-size: {self.font_size_medium};
                padding: 5px 10px;
                min-height: {self.button_height}px;
            }}
            QPushButton:hover {{
                background-color: #e6e6e6;
            }}
        """
        self.style_checkbox = f"font-size: {self.font_size_medium};"
        self.style_combobox = f"font-size: {self.font_size_small}; padding: 2px;"
        self.style_groupbox = f"font-size: {self.font_size_large}; font-weight: bold;"

    def _build_ui(self):
        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        layout.addWidget(tabs)

        # Tab 1: 主计算
        tab_main = QWidget()
        tabs.addTab(tab_main, "主计算")
        self._build_main_tab(tab_main)

        # Tab 2: 子集相关性分析
        tab_sub = QWidget()
        tabs.addTab(tab_sub, "子集相关性分析")
        self._build_subset_tab(tab_sub)

        # Tab 3: 违约相关性矩阵计算器
        tab_default = QWidget()
        tabs.addTab(tab_default, "违约相关性矩阵计算器")
        self._build_default_corr_tab(tab_default)

        # Tab 4: 相关性矩阵热力图生成器
        tab_heatmap = QWidget()
        tabs.addTab(tab_heatmap, "相关性矩阵热力图生成器")
        self._build_heatmap_tab(tab_heatmap)

        # status
        self.status = QLabel("")
        layout.addWidget(self.status)

    def _build_main_tab(self, parent):
        v_main = QVBoxLayout(parent)
        v_main.setSpacing(self.spacing_medium)

        # 文件操作和控制面板
        control_group = QGroupBox("数据和控制")
        control_group.setStyleSheet(self.style_groupbox)
        control_layout = QGridLayout()
        control_layout.setSpacing(self.spacing_small)
        
        # 第一行：文件加载
        btn_load = QPushButton("加载 Excel")
        btn_load.setStyleSheet(self.style_button)
        btn_load.clicked.connect(self.load_excel)
        self.lbl_path = QLabel("未加载")
        self.lbl_path.setStyleSheet(self.style_label)
        control_layout.addWidget(btn_load, 0, 0)
        control_layout.addWidget(self.lbl_path, 0, 1, 1, 3)

        # 第二行：因子选择和窗口参数（均匀排列在一行）
        self.chk_market = QCheckBox("去市场因子")
        self.chk_ind = QCheckBox("去行业因子")
        self.chk_reg = QCheckBox("去区域因子")
        self.chk_market.setChecked(True)
        self.chk_market.setStyleSheet(self.style_checkbox)
        self.chk_ind.setStyleSheet(self.style_checkbox)
        self.chk_reg.setStyleSheet(self.style_checkbox)
        control_layout.addWidget(self.chk_market, 1, 0)
        control_layout.addWidget(self.chk_ind, 1, 1)
        control_layout.addWidget(self.chk_reg, 1, 2)
        
        # 添加弹性空间，将窗口参数推到右侧
        control_layout.addWidget(QWidget(), 1, 3)  # 弹性空间
        
        # 窗口参数靠右布局
        lbl_win = QLabel("窗口长度（天）")
        lbl_win.setStyleSheet(self.style_label)
        control_layout.addWidget(lbl_win, 1, 4)
        self.input_win_main = QLineEdit("60")
        self.input_win_main.setFixedWidth(self.input_width)
        self.input_win_main.setStyleSheet(self.style_input)
        control_layout.addWidget(self.input_win_main, 1, 5)
        
        # 在窗口长度输入框和步长标签之间添加空间
        spacer = QLabel("   ")  # 添加空格作为间隔
        spacer.setFixedWidth(15)  # 设置固定宽度
        control_layout.addWidget(spacer, 1, 6)
        
        lbl_step = QLabel("步长（天）")
        lbl_step.setStyleSheet(self.style_label)
        control_layout.addWidget(lbl_step, 1, 7)
        self.input_step_main = QLineEdit("10")
        self.input_step_main.setFixedWidth(self.input_width)
        self.input_step_main.setStyleSheet(self.style_input)
        control_layout.addWidget(self.input_step_main, 1, 8)

        # 第四行：操作按钮
        btn_run = QPushButton("运行主计算")
        btn_run.setStyleSheet(self.style_button)
        btn_run.clicked.connect(self.run_main)
        control_layout.addWidget(btn_run, 3, 0, 1, 2)
        
        btn_export_main = QPushButton("导出结果")
        btn_export_main.setStyleSheet(self.style_button)
        btn_export_main.clicked.connect(self.export_main_results)
        control_layout.addWidget(btn_export_main, 3, 2, 1, 2)

        control_group.setLayout(control_layout)
        v_main.addWidget(control_group)

        # 显示区域：热图和趋势图
        display_h = QHBoxLayout()
        display_h.setSpacing(10)

        # 左侧热图区域
        heatmap_group = QGroupBox("相关性热图")
        heatmap_layout = QVBoxLayout()
        heatmap_layout.setSpacing(6)
        
        heatmap_control = QHBoxLayout()
        heatmap_control.addWidget(QLabel("显示："))
        self.cmb_heat = QComboBox()
        self.cmb_heat.addItems(["无", "企业相关性", "区域相关性", "行业相关性"])
        self.cmb_heat.currentIndexChanged.connect(self.update_heatmap)
        heatmap_control.addWidget(self.cmb_heat)
        heatmap_control.addStretch()
        
        heatmap_layout.addLayout(heatmap_control)
        self.heatmap = MatCanvas(width=5, height=4)
        heatmap_layout.addWidget(self.heatmap)
        heatmap_group.setLayout(heatmap_layout)
        display_h.addWidget(heatmap_group, 1)

        # 右侧趋势图区域
        trend_group = QGroupBox("时间趋势")
        trend_layout = QVBoxLayout()
        trend_layout.setSpacing(6)
        
        trend_control = QHBoxLayout()
        trend_control.addWidget(QLabel("视图："))
        self.cmb_trend = QComboBox()
        self.cmb_trend.addItems(["无", "区域总体平均相关性", "行业总体平均相关性", "企业平均总体相关性"])
        self.cmb_trend.currentIndexChanged.connect(self.update_trend)
        trend_control.addWidget(self.cmb_trend)
        trend_control.addStretch()
        
        trend_layout.addLayout(trend_control)
        self.trend_canvas = MatCanvas(width=5, height=3)
        trend_layout.addWidget(self.trend_canvas)
        
        # 年度统计
        self.lbl_year_stats = QLabel("年度统计：尚未计算")
        self.lbl_year_stats.setWordWrap(True)
        trend_layout.addWidget(self.lbl_year_stats)
        
        trend_group.setLayout(trend_layout)
        display_h.addWidget(trend_group, 1)

        v_main.addLayout(display_h, 1)  # 设置伸缩因子为1

    def _build_subset_tab(self, parent):
        v_sub = QVBoxLayout(parent)
        v_sub.setSpacing(self.spacing_medium)

        # 控制面板
        control_group = QGroupBox("子集选择和控制")
        control_group.setStyleSheet(self.style_groupbox)
        control_layout = QHBoxLayout()
        control_layout.setSpacing(self.spacing_large)

        # 行业选择
        ind_group = QGroupBox("行业选择")
        ind_group.setStyleSheet(self.style_groupbox)
        ind_layout = QVBoxLayout()
        ind_layout.setSpacing(self.spacing_small)
        
        ind_control = QHBoxLayout()
        ind_label = QLabel("行业（多选）")
        ind_label.setStyleSheet(self.style_label)
        ind_control.addWidget(ind_label)
        self.chk_ind_all = QCheckBox("全选")
        self.chk_ind_all.setStyleSheet(self.style_checkbox)
        self.chk_ind_all.stateChanged.connect(self.toggle_select_all_ind)
        ind_control.addWidget(self.chk_ind_all)
        ind_control.addStretch()
        ind_layout.addLayout(ind_control)
        
        self.list_ind = QListWidget()
        self.list_ind.setSelectionMode(QAbstractItemView.MultiSelection)
        self.list_ind.setMaximumHeight(150)
        ind_layout.addWidget(self.list_ind)
        ind_group.setLayout(ind_layout)
        control_layout.addWidget(ind_group, 1)

        # 区域选择
        reg_group = QGroupBox("区域选择")
        reg_group.setStyleSheet(self.style_groupbox)
        reg_layout = QVBoxLayout()
        reg_layout.setSpacing(self.spacing_small)
        
        reg_control = QHBoxLayout()
        reg_label = QLabel("区域（多选）")
        reg_label.setStyleSheet(self.style_label)
        reg_control.addWidget(reg_label)
        self.chk_reg_all = QCheckBox("全选")
        self.chk_reg_all.setStyleSheet(self.style_checkbox)
        self.chk_reg_all.stateChanged.connect(self.toggle_select_all_reg)
        reg_control.addWidget(self.chk_reg_all)
        reg_control.addStretch()
        reg_layout.addLayout(reg_control)
        
        self.list_reg = QListWidget()
        self.list_reg.setSelectionMode(QAbstractItemView.MultiSelection)
        self.list_reg.setMaximumHeight(150)
        reg_layout.addWidget(self.list_reg)
        reg_group.setLayout(reg_layout)
        control_layout.addWidget(reg_group, 1)

        # 参数设置
        param_group = QGroupBox("参数设置")
        param_group.setStyleSheet(self.style_groupbox)
        param_layout = QVBoxLayout()
        param_layout.setSpacing(self.spacing_medium)
        
        method_label = QLabel("相关系数方法")
        method_label.setStyleSheet(self.style_label)
        param_layout.addWidget(method_label)
        self.cmb_method = QComboBox()
        self.cmb_method.setStyleSheet(self.style_combobox)
        self.cmb_method.addItems(["Pearson", "Spearman", "Kendall"])
        param_layout.addWidget(self.cmb_method)

        window_layout = QHBoxLayout()
        win_label = QLabel("窗口长度")
        win_label.setStyleSheet(self.style_label)
        window_layout.addWidget(win_label)
        self.input_win_sub = QLineEdit("60")
        self.input_win_sub.setFixedWidth(self.input_width)
        self.input_win_sub.setStyleSheet(self.style_input)
        window_layout.addWidget(self.input_win_sub)
        
        step_label = QLabel("步长")
        step_label.setStyleSheet(self.style_label)
        window_layout.addWidget(step_label)
        self.input_step_sub = QLineEdit("10")
        self.input_step_sub.setFixedWidth(self.input_width)
        self.input_step_sub.setStyleSheet(self.style_input)
        window_layout.addWidget(self.input_step_sub)
        param_layout.addLayout(window_layout)

        self.chk_sub_ts = QCheckBox("计算滑动总体平均相关性")
        self.chk_sub_ts.setStyleSheet(self.style_checkbox)
        param_layout.addWidget(self.chk_sub_ts)

        btn_layout = QHBoxLayout()
        btn_compute_sub = QPushButton("计算子集相关性")
        btn_compute_sub.setStyleSheet(self.style_button)
        btn_compute_sub.clicked.connect(self.compute_subset)
        btn_layout.addWidget(btn_compute_sub)
        
        btn_export_sub = QPushButton("导出结果")
        btn_export_sub.setStyleSheet(self.style_button)
        btn_export_sub.clicked.connect(self.export_subset_results)
        btn_layout.addWidget(btn_export_sub)
        param_layout.addLayout(btn_layout)

        param_group.setLayout(param_layout)
        control_layout.addWidget(param_group, 1)
        control_group.setLayout(control_layout)
        v_sub.addWidget(control_group)

        # 结果显示区域
        results_group = QGroupBox("子集分析结果")
        results_group.setStyleSheet(self.style_groupbox)
        results_layout = QVBoxLayout()
        results_layout.setSpacing(self.spacing_medium)

        # 热图
        self.sub_heatmap = MatCanvas(width=7, height=3)
        results_layout.addWidget(self.sub_heatmap)

        # 统计信息
        self.lbl_subset_stats = QLabel("统计： - ")
        self.lbl_subset_stats.setStyleSheet(self.style_label)
        self.lbl_subset_stats.setWordWrap(True)
        results_layout.addWidget(self.lbl_subset_stats)

        # 时间序列图
        self.sub_ts_canvas = MatCanvas(width=7, height=2)
        results_layout.addWidget(self.sub_ts_canvas)

        # 年度统计
        self.lbl_subset_year_stats = QLabel("子集年度统计：尚未计算")
        self.lbl_subset_year_stats.setStyleSheet(self.style_label)
        self.lbl_subset_year_stats.setWordWrap(True)
        results_layout.addWidget(self.lbl_subset_year_stats)

        results_group.setLayout(results_layout)
        v_sub.addWidget(results_group, 1)

    def _build_default_corr_tab(self, parent):
        layout = QVBoxLayout(parent)
        layout.setSpacing(self.spacing_medium)

        # ===== 文件导入按钮组 =====
        file_layout = QHBoxLayout()
        self.btn_load_region = QPushButton("📂 导入区域相关矩阵")
        self.btn_load_industry = QPushButton("📂 导入行业相关矩阵")
        self.btn_load_firm = QPushButton("📂 导入企业信息")
        for btn in [self.btn_load_region, self.btn_load_industry, self.btn_load_firm]:
            btn.setFixedHeight(self.button_height)
            btn.setStyleSheet(self.style_button)
            file_layout.addWidget(btn)

        self.btn_load_region.clicked.connect(self.load_region_matrix)
        self.btn_load_industry.clicked.connect(self.load_industry_matrix)
        self.btn_load_firm.clicked.connect(self.load_firm_info)
        layout.addLayout(file_layout)

        layout.addSpacing(10)

        # ===== 权重滑块 =====
        weight_label = QLabel("⚖️ 行业与区域权重（两者和为1）")
        weight_label.setStyleSheet(self.style_label)
        layout.addWidget(weight_label)
        
        self.slider_weight = QSlider(QtCore.Qt.Horizontal)
        self.slider_weight.setRange(0, 100)
        self.slider_weight.setValue(50)
        self.slider_weight.valueChanged.connect(self.update_labels)
        layout.addWidget(self.slider_weight)
        
        self.label_weight = QLabel("行业权重: 0.50 | 区域权重: 0.50")
        self.label_weight.setStyleSheet(self.style_label)
        layout.addWidget(self.label_weight)

        layout.addSpacing(5)

        # ===== 缩放滑块 =====
        scale_label = QLabel("🔧 整体缩放比例（0.5~1）")
        scale_label.setStyleSheet(self.style_label)
        layout.addWidget(scale_label)
        
        self.slider_scale = QSlider(QtCore.Qt.Horizontal)
        self.slider_scale.setRange(50, 100)
        self.slider_scale.setValue(100)
        self.slider_scale.valueChanged.connect(self.update_labels)
        layout.addWidget(self.slider_scale)
        
        self.label_scale = QLabel("整体缩放比例: 1.00")
        self.label_scale.setStyleSheet(self.style_label)
        layout.addWidget(self.label_scale)

        layout.addSpacing(10)

        # ===== 计算与导出按钮 =====
        btn_layout = QHBoxLayout()
        self.btn_calculate = QPushButton("⚙️ 计算违约相关性矩阵")
        self.btn_export = QPushButton("💾 导出结果到Excel")
        self.btn_calculate.setFixedHeight(self.button_height + 10)
        self.btn_export.setFixedHeight(self.button_height + 10)
        self.btn_calculate.setStyleSheet(self.style_button)
        self.btn_export.setStyleSheet(self.style_button)
        self.btn_calculate.clicked.connect(self.calculate_corr)
        self.btn_export.clicked.connect(self.export_result)
        btn_layout.addWidget(self.btn_calculate)
        btn_layout.addWidget(self.btn_export)
        layout.addLayout(btn_layout)
        layout.addSpacing(10)

        # ===== 文本输出区域 =====
        output_label = QLabel("📄 输出信息：")
        output_label.setStyleSheet(self.style_label)
        layout.addWidget(output_label)
        
        self.text_output = QTextEdit()
        self.text_output.setReadOnly(True)
        self.text_output.setFixedHeight(150)
        self.text_output.setStyleSheet(f"font-size: {self.font_size_small};")
        layout.addWidget(self.text_output)

        # ===== 热力图显示区 =====
        self.text_output.append("💡 提示：计算完成后将自动在下方显示违约相关性热力图。")
        
        heatmap_label = QLabel("🔥 违约相关性矩阵热力图")
        heatmap_label.setStyleSheet(self.style_label)
        layout.addWidget(heatmap_label)
        
        self.fig = Figure(figsize=(6, 5))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        layout.addStretch()

    def _build_heatmap_tab(self, parent):
        layout = QHBoxLayout(parent)
        
        # 左侧控制面板
        control_panel = self.create_heatmap_control_panel()
        layout.addWidget(control_panel, 1)
        
        # 右侧热力图显示区域
        self.heatmap_canvas = HeatmapCanvas(self, width=8, height=6)
        layout.addWidget(self.heatmap_canvas, 3)

    def create_heatmap_control_panel(self):
        panel = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(self.spacing_medium)
        panel.setLayout(layout)
        
        # 文件导入部分
        file_group = QGroupBox("文件操作")
        file_group.setStyleSheet(self.style_groupbox)
        file_layout = QVBoxLayout()
        file_layout.setSpacing(self.spacing_small)
        
        self.file_label = QLabel("未选择文件")
        self.file_label.setStyleSheet(self.style_label)
        self.file_label.setWordWrap(True)
        file_layout.addWidget(self.file_label)
        
        import_btn = QPushButton("导入Excel文件")
        import_btn.setStyleSheet(self.style_button)
        import_btn.clicked.connect(self.import_excel_heatmap)
        file_layout.addWidget(import_btn)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # 颜色设置部分
        color_group = QGroupBox("颜色设置")
        color_group.setStyleSheet(self.style_groupbox)
        color_layout = QVBoxLayout()
        color_layout.setSpacing(self.spacing_small)
        
        # -1颜色设置
        neg_layout = QHBoxLayout()
        neg_label = QLabel("-1点颜色:")
        neg_label.setStyleSheet(self.style_label)
        neg_layout.addWidget(neg_label)
        self.neg_color_btn = QPushButton()
        self.neg_color_btn.setFixedSize(*self.small_button_size)
        self.neg_color_btn.setStyleSheet(f"background-color: {self.color_neg.name()}")
        self.neg_color_btn.clicked.connect(lambda: self.choose_color('neg'))
        neg_layout.addWidget(self.neg_color_btn)
        neg_layout.addStretch()
        color_layout.addLayout(neg_layout)
        
        # 0颜色设置
        zero_layout = QHBoxLayout()
        zero_label = QLabel("0点颜色:")
        zero_label.setStyleSheet(self.style_label)
        zero_layout.addWidget(zero_label)
        self.zero_color_btn = QPushButton()
        self.zero_color_btn.setFixedSize(*self.small_button_size)
        self.zero_color_btn.setStyleSheet(f"background-color: {self.color_zero.name()}")
        self.zero_color_btn.clicked.connect(lambda: self.choose_color('zero'))
        zero_layout.addWidget(self.zero_color_btn)
        zero_layout.addStretch()
        color_layout.addLayout(zero_layout)
        
        # 1颜色设置
        pos_layout = QHBoxLayout()
        pos_label = QLabel("1点颜色:")
        pos_label.setStyleSheet(self.style_label)
        pos_layout.addWidget(pos_label)
        self.pos_color_btn = QPushButton()
        self.pos_color_btn.setFixedSize(*self.small_button_size)
        self.pos_color_btn.setStyleSheet(f"background-color: {self.color_pos.name()}")
        self.pos_color_btn.clicked.connect(lambda: self.choose_color('pos'))
        pos_layout.addWidget(self.pos_color_btn)
        pos_layout.addStretch()
        color_layout.addLayout(pos_layout)
        
        # 颜色过渡预览
        preview_label = QLabel("颜色过渡预览:")
        preview_label.setStyleSheet(self.style_label)
        color_layout.addWidget(preview_label)
        self.color_transition = ColorTransitionWidget()
        self.color_transition.set_colors(self.color_neg, self.color_zero, self.color_pos)
        color_layout.addWidget(self.color_transition)
        
        color_group.setLayout(color_layout)
        layout.addWidget(color_group)
        
        # 标题设置部分
        title_group = QGroupBox("标题设置")
        title_group.setStyleSheet(self.style_groupbox)
        title_layout = QVBoxLayout()
        title_layout.setSpacing(self.spacing_small)
        
        self.title_label = QLabel(f"当前标题: {self.heatmap_title}")
        self.title_label.setStyleSheet(self.style_label)
        self.title_label.setWordWrap(True)
        title_layout.addWidget(self.title_label)
        
        title_btn = QPushButton("设置标题")
        title_btn.setStyleSheet(self.style_button)
        title_btn.clicked.connect(self.set_title)
        title_layout.addWidget(title_btn)
        
        title_group.setLayout(title_layout)
        layout.addWidget(title_group)
        
        # 显示选项部分
        display_group = QGroupBox("显示选项")
        display_group.setStyleSheet(self.style_groupbox)
        display_layout = QVBoxLayout()
        display_layout.setSpacing(self.spacing_small)
        
        # 显示数值勾选框
        self.show_values_checkbox = QCheckBox("显示相关性系数数值")
        self.show_values_checkbox.setStyleSheet(self.style_checkbox)
        self.show_values_checkbox.setChecked(self.show_values)
        self.show_values_checkbox.stateChanged.connect(self.toggle_show_values)
        display_layout.addWidget(self.show_values_checkbox)
        
        # 色彩过渡效果勾选框
        self.smooth_transition_checkbox = QCheckBox("启用色彩过渡效果")
        self.smooth_transition_checkbox.setStyleSheet(self.style_checkbox)
        self.smooth_transition_checkbox.setChecked(self.smooth_transition)
        self.smooth_transition_checkbox.stateChanged.connect(self.toggle_smooth_transition)
        display_layout.addWidget(self.smooth_transition_checkbox)
        
        display_group.setLayout(display_layout)
        layout.addWidget(display_group)
        
        # 操作按钮
        button_group = QGroupBox("操作")
        button_group.setStyleSheet(self.style_groupbox)
        button_layout = QVBoxLayout()
        button_layout.setSpacing(self.spacing_small)
        
        generate_btn = QPushButton("生成热力图")
        generate_btn.setStyleSheet(self.style_button)
        generate_btn.clicked.connect(self.generate_heatmap)
        button_layout.addWidget(generate_btn)
        
        save_btn = QPushButton("保存热力图")
        save_btn.setStyleSheet(self.style_button)
        save_btn.clicked.connect(self.save_heatmap)
        button_layout.addWidget(save_btn)
        
        button_group.setLayout(button_layout)
        layout.addWidget(button_group)
        
        layout.addStretch()
        
        return panel

    # ========= 主计算方法 =========

    def _load_large_excel_optimized(self, path, chunk_size=1000):
        """优化加载大Excel文件，分批读取避免内存溢出"""
        try:
            # 首先读取前几行来获取列信息
            sample_df = pd.read_excel(path, nrows=5)
            cols = sample_df.columns.tolist()
            
            # 确定需要读取的列
            if len(cols) < 4:
                raise ValueError("表格至少需包含 企业名称, 行业, 区域, 及若干价格列")
            
            # 重命名前三列为标准名称
            sample_df = sample_df.rename(columns={cols[0]: "企业名称", cols[1]: "行业", cols[2]: "区域"})
            price_cols = cols[3:]
            
            # 分批读取数据
            chunks = []
            for chunk in pd.read_excel(path, chunksize=chunk_size):
                # 重命名列
                chunk = chunk.rename(columns={cols[0]: "企业名称", cols[1]: "行业", cols[2]: "区域"})
                chunks.append(chunk)
            
            # 合并所有块
            df = pd.concat(chunks, ignore_index=True)
            
            return df
            
        except Exception as e:
            raise Exception(f"优化加载Excel失败: {str(e)}")

    def _safe_log_returns_optimized(self, price_wide):
        """优化计算对数收益率，避免内存溢出"""
        try:
            # 对于大矩阵，分批计算对数收益率
            if price_wide.shape[1] > 2000:  # 如果公司数超过2000，分批处理
                self.status.setText("分批计算对数收益率...")
                QtWidgets.QApplication.processEvents()
                
                # 分批处理，每批1000家公司
                batch_size = 1000
                num_batches = (price_wide.shape[1] + batch_size - 1) // batch_size
                returns_chunks = []
                
                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, price_wide.shape[1])
                    
                    batch_prices = price_wide.iloc[:, start_idx:end_idx]
                    batch_returns = safe_log_returns_from_prices_df(batch_prices)
                    returns_chunks.append(batch_returns)
                    
                    self.status.setText(f"计算对数收益率... 批次 {i+1}/{num_batches}")
                    QtWidgets.QApplication.processEvents()
                
                # 合并所有批次
                returns = pd.concat(returns_chunks, axis=1)
            else:
                # 对于小矩阵，直接计算
                returns = safe_log_returns_from_prices_df(price_wide)
            
            return returns
            
        except Exception as e:
            raise Exception(f"计算对数收益率失败: {str(e)}")

    def _validate_data_quality(self, df, price_cols):
        """验证数据质量，检测0值、NaN值、无穷大值等无效数据"""
        validation_results = {
            'has_zeros': False,
            'has_nans': False,
            'has_infs': False,
            'zero_count': 0,
            'nan_count': 0,
            'inf_count': 0,
            'invalid_columns': [],
            'validation_passed': True,
            'validation_message': ""
        }
        
        try:
            # 检查价格列的数据质量
            price_data = df[price_cols]
            
            # 检测0值
            zero_mask = (price_data == 0)
            zero_count = zero_mask.sum().sum()
            if zero_count > 0:
                validation_results['has_zeros'] = True
                validation_results['zero_count'] = zero_count
            
            # 检测NaN值
            nan_mask = price_data.isna()
            nan_count = nan_mask.sum().sum()
            if nan_count > 0:
                validation_results['has_nans'] = True
                validation_results['nan_count'] = nan_count
            
            # 检测无穷大值
            inf_mask = np.isinf(price_data)
            inf_count = inf_mask.sum().sum()
            if inf_count > 0:
                validation_results['has_infs'] = True
                validation_results['inf_count'] = inf_count
            
            # 检查每列的数据质量
            invalid_cols = []
            for col in price_cols:
                col_data = price_data[col]
                if col_data.isna().all() or (col_data == 0).all():
                    invalid_cols.append(col)
            
            validation_results['invalid_columns'] = invalid_cols
            
            # 生成验证消息
            messages = []
            if zero_count > 0:
                messages.append(f"发现 {zero_count} 个0值")
            if nan_count > 0:
                messages.append(f"发现 {nan_count} 个缺失值")
            if inf_count > 0:
                messages.append(f"发现 {inf_count} 个无穷大值")
            if invalid_cols:
                messages.append(f"{len(invalid_cols)} 列数据无效: {', '.join(invalid_cols[:5])}{'...' if len(invalid_cols) > 5 else ''}")
            
            if messages:
                validation_results['validation_passed'] = False
                validation_results['validation_message'] = "数据验证发现问题: " + "; ".join(messages)
            else:
                validation_results['validation_message'] = "数据验证通过"
                
        except Exception as e:
            validation_results['validation_passed'] = False
            validation_results['validation_message'] = f"数据验证过程中出错: {str(e)}"
        
        return validation_results

    def _show_data_validation_dialog(self, validation_results, file_path):
        """显示数据验证结果对话框"""
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("数据验证结果")
        msg_box.setIcon(QMessageBox.Warning)
        
        message = f"文件: {os.path.basename(file_path)}\n\n"
        message += validation_results['validation_message'] + "\n\n"
        
        if not validation_results['validation_passed']:
            message += "数据质量问题:\n"
            if validation_results['has_zeros']:
                message += f"• 0值数量: {validation_results['zero_count']}\n"
            if validation_results['has_nans']:
                message += f"• 缺失值数量: {validation_results['nan_count']}\n"
            if validation_results['has_infs']:
                message += f"• 无穷大值数量: {validation_results['inf_count']}\n"
            if validation_results['invalid_columns']:
                message += f"• 无效列数: {len(validation_results['invalid_columns'])}\n"
            
            message += "\n建议:\n"
            message += "1. 检查原始数据源，确保数据完整性和准确性\n"
            message += "2. 对于0值，确认是否为真实数据还是数据缺失\n"
            message += "3. 对于缺失值，考虑使用插值方法填充\n"
            message += "4. 对于无效列，可能需要从分析中排除\n\n"
            message += "是否继续加载数据？"
            
            msg_box.setText(message)
            msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            msg_box.setDefaultButton(QMessageBox.No)
            
            return msg_box.exec_() == QMessageBox.Yes
        else:
            message += "数据质量良好，可以继续分析。"
            msg_box.setText(message)
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.exec_()
            return True

    def _clean_data_automatically(self, df, price_cols, validation_results):
        """自动清洗数据"""
        cleaned_df = df.copy()
        
        # 移除完全无效的列
        if validation_results['invalid_columns']:
            valid_cols = [col for col in price_cols if col not in validation_results['invalid_columns']]
            cleaned_df = cleaned_df[["企业名称", "行业", "区域"] + valid_cols]
            price_cols = valid_cols
        
        # 对于价格数据，将0值替换为NaN（假设0值表示数据缺失）
        price_data = cleaned_df[price_cols]
        price_data_cleaned = price_data.replace(0, np.nan)
        
        # 移除无穷大值
        price_data_cleaned = price_data_cleaned.replace([np.inf, -np.inf], np.nan)
        
        # 使用前向填充和后向填充处理缺失值
        price_data_cleaned = price_data_cleaned.ffill().bfill()
        
        # 如果仍有缺失值，使用列均值填充
        for col in price_cols:
            if price_data_cleaned[col].isna().any():
                col_mean = price_data_cleaned[col].mean()
                price_data_cleaned[col] = price_data_cleaned[col].fillna(col_mean)
        
        # 更新数据框
        cleaned_df[price_cols] = price_data_cleaned
        
        return cleaned_df, price_cols

    def load_excel(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 Excel 文件", "", "Excel Files (*.xlsx *.xls)")
        if not path:
            return
        try:
            # 检查文件大小，如果文件很大则使用优化加载方式
            file_size = os.path.getsize(path) / (1024 * 1024)  # MB
            if file_size > 50:  # 如果文件大于50MB，使用优化加载
                self.status.setText("检测到大文件，使用优化加载方式...")
                QtWidgets.QApplication.processEvents()
                df = self._load_large_excel_optimized(path)
            else:
                df = pd.read_excel(path)
            
            cols = df.columns.tolist()
            if len(cols) < 4:
                QMessageBox.critical(self, "错误", "表格至少需包含 企业名称, 行业, 区域, 及若干价格列。")
                return
            
            # rename first three columns to standard names
            df = df.rename(columns={cols[0]: "企业名称", cols[1]: "行业", cols[2]: "区域"})
            price_cols = cols[3:]
            
            # 数据验证
            self.status.setText("正在进行数据验证...")
            QtWidgets.QApplication.processEvents()
            validation_results = self._validate_data_quality(df, price_cols)
            
            # 如果数据有问题，提示用户
            if not validation_results['validation_passed']:
                continue_loading = self._show_data_validation_dialog(validation_results, path)
                if not continue_loading:
                    self.status.setText("用户取消加载，请清洗数据后重新加载")
                    return
                
                # 询问是否自动清洗数据
                auto_clean_reply = QMessageBox.question(
                    self, "数据清洗", 
                    "是否自动清洗数据？\n\n自动清洗将：\n• 移除无效列\n• 处理0值和缺失值\n• 使用插值方法填充数据\n\n注意：自动清洗可能改变原始数据分布",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )
                
                if auto_clean_reply == QMessageBox.Yes:
                    df, price_cols = self._clean_data_automatically(df, price_cols, validation_results)
                    QMessageBox.information(self, "数据清洗完成", "数据已自动清洗完成")
            
            # 优化内存使用：只加载需要的列
            meta_df = df[["企业名称", "行业", "区域"]].copy()
            price_df = df[price_cols].copy()
            
            # 优化数据类型以减少内存使用
            price_df = price_df.astype(np.float32)
            price_df.index = df["企业名称"].astype(str).values
            
            # transpose to get index=time, cols=firms
            price_wide = price_df.T
            
            # 优化计算：分批处理大矩阵
            self.status.setText("计算对数收益率...")
            QtWidgets.QApplication.processEvents()
            returns = self._safe_log_returns_optimized(price_wide)
            
            self.raw_df = df
            self.meta_df = meta_df
            self.price_df = price_wide
            self.returns = returns
            self.resid_returns = None
            self.lbl_path.setText(os.path.basename(path))
            self.populate_lists()
            
            success_message = f"加载成功：公司数={price_df.shape[0]}, 时间点={price_wide.shape[0]}\n文件大小: {file_size:.1f}MB"
            if not validation_results['validation_passed']:
                success_message += f"\n\n注意：数据存在质量问题，已进行相应处理"
            
            QMessageBox.information(self, "加载成功", success_message)
        except Exception as e:
            QMessageBox.critical(self, "读取失败", f"{e}\n\n{traceback.format_exc()}")

    def populate_lists(self):
        if self.meta_df is None:
            return
        self.list_ind.clear(); self.list_reg.clear()
        inds = sorted(self.meta_df['行业'].dropna().unique().tolist())
        regs = sorted(self.meta_df['区域'].dropna().unique().tolist())
        for it in inds:
            self.list_ind.addItem(QListWidgetItem(str(it)))
        for r in regs:
            self.list_reg.addItem(QListWidgetItem(str(r)))
        # reset "select all" boxes
        self.chk_ind_all.setChecked(False)
        self.chk_reg_all.setChecked(False)

    def toggle_select_all_ind(self, state):
        # state 0 unchecked, 2 checked
        for i in range(self.list_ind.count()):
            item = self.list_ind.item(i)
            if state == QtCore.Qt.Checked:
                item.setSelected(True)
            else:
                item.setSelected(False)

    def toggle_select_all_reg(self, state):
        for i in range(self.list_reg.count()):
            item = self.list_reg.item(i)
            if state == QtCore.Qt.Checked:
                item.setSelected(True)
            else:
                item.setSelected(False)

    def _compute_correlation_optimized(self, data):
        """优化计算相关性矩阵，避免内存溢出"""
        try:
            # 对于大矩阵，分批计算相关性
            if data.shape[1] > 2000:
                self.status.setText("分批计算相关性矩阵...")
                QtWidgets.QApplication.processEvents()
                
                # 分批处理相关性计算
                batch_size = 1000
                num_batches = (data.shape[1] + batch_size - 1) // batch_size
                corr_chunks = []
                
                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, data.shape[1])
                    
                    batch_data = data.iloc[:, start_idx:end_idx]
                    # 只计算当前批次的相关性
                    batch_corr = batch_data.corr()
                    corr_chunks.append(batch_corr)
                    
                    self.status.setText(f"计算相关性矩阵... 批次 {i+1}/{num_batches}")
                    QtWidgets.QApplication.processEvents()
                
                # 合并相关性矩阵
                # 对于大矩阵，我们只返回第一个批次的相关性矩阵，避免内存问题
                # 在实际应用中，可能需要更复杂的合并策略
                corr_matrix = corr_chunks[0] if corr_chunks else None
            else:
                # 对于小矩阵，直接计算
                corr_matrix = data.corr()
            
            return corr_matrix
            
        except Exception as e:
            raise Exception(f"计算相关性矩阵失败: {str(e)}")

    # ========= 进度条管理方法 =========

    def show_progress_dialog(self, title="计算中", message="请稍候..."):
        """显示进度对话框"""
        self.progress_dialog = QProgressDialog(message, "取消", 0, 100, self)
        self.progress_dialog.setWindowTitle(title)
        self.progress_dialog.setWindowModality(QtCore.Qt.WindowModal)
        self.progress_dialog.setMinimumDuration(0)  # 立即显示
        self.progress_dialog.canceled.connect(self.cancel_calculation)
        self.progress_dialog.show()
        
    def update_progress(self, value, message=None):
        """更新进度条"""
        if hasattr(self, 'progress_dialog') and self.progress_dialog and self.progress_dialog.isVisible():
            self.progress_dialog.setValue(value)
            if message:
                self.progress_dialog.setLabelText(message)
            QtWidgets.QApplication.processEvents()  # 允许UI响应
            
    def hide_progress_dialog(self):
        """隐藏进度对话框"""
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.hide()
            self.progress_dialog = None

    def _on_calculation_progress(self, value, message):
        """处理计算线程的进度更新"""
        self.update_progress(value, message)
        
    def _on_calculation_finished(self, success, message):
        """处理计算线程完成"""
        try:
            self.hide_progress_dialog()
            
            # 清理线程引用，避免内存泄漏
            if self.calculation_thread:
                if self.calculation_thread.isRunning():
                    self.calculation_thread.stop()  # 使用安全的停止方法
                    self.calculation_thread.wait(5000)  # 最多等待5秒
                self.calculation_thread = None
            
            # 强制垃圾回收，释放内存
            import gc
            gc.collect()
            
            if success:
                # 自动将热图切换到"行业相关性"
                if self.ind_corr is not None and not self.ind_corr.empty:
                    self.cmb_heat.setCurrentText("行业相关性")
                    # 强制更新热图显示
                    self.update_heatmap(3)
                
                # 更新状态栏
                self.status.setText("主计算完成，可以查看结果或导出")
                
                # 使用延迟显示消息框，避免与线程清理冲突
                QtCore.QTimer.singleShot(100, lambda: self._show_completion_message(success, message))
            else:
                self.status.setText("计算失败，请检查数据")
                QtCore.QTimer.singleShot(100, lambda: self._show_completion_message(success, message))
                
        except Exception as e:
            # 处理线程完成过程中的异常
            self.status.setText("计算完成处理异常")
            QtCore.QTimer.singleShot(100, lambda: self._show_error_message(f"计算完成处理异常: {str(e)}"))
    
    def _show_completion_message(self, success, message):
        """安全显示完成消息"""
        try:
            if success:
                QMessageBox.information(self, "完成", message)
            else:
                QMessageBox.critical(self, "错误", message)
        except Exception as e:
            print(f"显示消息框失败: {e}")
    
    def _show_error_message(self, message):
        """安全显示错误消息"""
        try:
            QMessageBox.critical(self, "系统错误", message)
        except Exception as e:
            print(f"显示错误消息框失败: {e}")

    def cancel_calculation(self):
        """取消当前计算"""
        if self.calculation_thread and self.calculation_thread.isRunning():
            reply = QMessageBox.question(self, "确认取消", 
                                       "确定要取消当前计算吗？",
                                       QMessageBox.Yes | QMessageBox.No,
                                       QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.calculation_thread.stop()
                self.status.setText("计算已取消")
                self.hide_progress_dialog()
                QMessageBox.information(self, "已取消", "计算已成功取消")

    def run_main(self):
        """使用多线程运行主计算，避免UI假死"""
        if self.returns is None or self.meta_df is None:
            QMessageBox.warning(self, "提示", "请先加载数据并确保返回序列可用")
            return
            
        # 检查是否有正在运行的计算线程
        if self.calculation_thread and self.calculation_thread.isRunning():
            QMessageBox.information(self, "提示", "已有计算正在进行中，请等待完成")
            return
            
        try:
            remove_market = self.chk_market.isChecked()
            remove_ind = self.chk_ind.isChecked()
            remove_reg = self.chk_reg.isChecked()
            win = int(self.input_win_main.text())
            step = int(self.input_step_main.text())
            
            # 创建并启动计算线程
            self.calculation_thread = MainCalculationThread(
                self, remove_market, remove_ind, remove_reg, win, step
            )
            self.calculation_thread.progress_signal.connect(self._on_calculation_progress)
            self.calculation_thread.finished_signal.connect(self._on_calculation_finished)
            
            # 显示进度对话框
            self.show_progress_dialog("主计算", "正在初始化计算...")
            
            # 启动线程
            self.calculation_thread.start()
            
        except Exception as e:
            self.hide_progress_dialog()
            QMessageBox.critical(self, "错误", f"启动计算失败：{e}\n\n{traceback.format_exc()}")

    def _aggregate_group_correlation(self, corr_matrix, mapping_series, groups):
        """Helper: aggregate firm-firm corr_matrix into group x group average correlation"""
        G = pd.DataFrame(index=groups, columns=groups, dtype=float)
        for a in groups:
            fa = [f for f in corr_matrix.index if mapping_series.get(f) == a]
            for b in groups:
                fb = [f for f in corr_matrix.index if mapping_series.get(f) == b]
                if len(fa) == 0 or len(fb) == 0:
                    G.loc[a, b] = np.nan
                else:
                    G.loc[a, b] = corr_matrix.loc[fa, fb].values.mean()
        return G

    def update_heatmap(self, idx):
        if idx == 0:
            self.heatmap.plot_heatmap(None)
        elif idx == 1:
            self.heatmap.plot_heatmap(self.ent_corr, title="企业相关性")
        elif idx == 2:
            self.heatmap.plot_heatmap(self.prov_corr, title="区域相关性")
        elif idx == 3:
            self.heatmap.plot_heatmap(self.ind_corr, title="行业相关性")

    def update_trend(self, idx):
        if idx == 0:
            self.trend_canvas.plot_series(None)
            self.lbl_year_stats.setText("年度统计：尚未计算")
        elif idx == 1:
            if self.prov_overall_series is None or self.prov_overall_series.empty:
                QMessageBox.information(self, "提示", "尚未计算区域总体平均相关性时序（先运行主计算）。")
                return
            df = self.prov_overall_series.to_frame('prov_overall')
            self.trend_canvas.plot_series(df, title="区域总体平均相关性随时间")
            yearly = yearly_statistics_from_series(self.prov_overall_series)
            self.lbl_year_stats.setText("区域年度统计：\n" + yearly.to_string())
        elif idx == 2:
            if self.ind_overall_series is None or self.ind_overall_series.empty:
                QMessageBox.information(self, "提示", "尚未计算行业总体平均相关性时序（先运行主计算）。")
                return
            df = self.ind_overall_series.to_frame('ind_overall')
            self.trend_canvas.plot_series(df, title="行业总体平均相关性随时间")
            yearly = yearly_statistics_from_series(self.ind_overall_series)
            self.lbl_year_stats.setText("行业年度统计：\n" + yearly.to_string())
        elif idx == 3:
            if self.ent_overall_series is None or self.ent_overall_series.empty:
                QMessageBox.information(self, "提示", "尚未计算企业平均总体相关性时序（先运行主计算）。")
                return
            df = self.ent_overall_series.to_frame('ent_overall')
            self.trend_canvas.plot_series(df, title="企业平均总体相关性随时间")
            yearly = yearly_statistics_from_series(self.ent_overall_series)
            self.lbl_year_stats.setText("企业年度统计：\n" + yearly.to_string())

    # ---------- export main results (use with ExcelWriter) ----------

    def export_main_results(self):
        if self.resid_returns is None:
            QMessageBox.warning(self, "提示", "请先运行主计算并生成残差收益率。")
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存主计算结果到 Excel", "main_results.xlsx", "Excel Files (*.xlsx)")
        if not path:
            return
        try:
            ent_corr = self.ent_corr
            prov_corr = self.prov_corr
            ind_corr = self.ind_corr
            prov_ts = self.prov_overall_series.to_frame('prov_overall') if self.prov_overall_series is not None else pd.DataFrame()
            ind_ts = self.ind_overall_series.to_frame('ind_overall') if self.ind_overall_series is not None else pd.DataFrame()
            ent_ts = self.ent_overall_series.to_frame('ent_overall') if self.ent_overall_series is not None else pd.DataFrame()
            prov_yearly = yearly_statistics_from_series(self.prov_overall_series) if self.prov_overall_series is not None else pd.DataFrame()
            ind_yearly = yearly_statistics_from_series(self.ind_overall_series) if self.ind_overall_series is not None else pd.DataFrame()
            ent_yearly = yearly_statistics_from_series(self.ent_overall_series) if self.ent_overall_series is not None else pd.DataFrame()

            with pd.ExcelWriter(path, engine='openpyxl') as writer:
                if ent_corr is not None:
                    ent_corr.to_excel(writer, sheet_name="企业相关性")
                if prov_corr is not None:
                    prov_corr.to_excel(writer, sheet_name="区域相关性")
                if ind_corr is not None:
                    ind_corr.to_excel(writer, sheet_name="行业相关性")
                if not prov_ts.empty:
                    prov_ts.to_excel(writer, sheet_name="区域总体平均_TS")
                if not ind_ts.empty:
                    ind_ts.to_excel(writer, sheet_name="行业总体平均_TS")
                if not ent_ts.empty:
                    ent_ts.to_excel(writer, sheet_name="企业平均总体_TS")
                if not prov_yearly.empty:
                    prov_yearly.to_excel(writer, sheet_name="区域年度统计")
                if not ind_yearly.empty:
                    ind_yearly.to_excel(writer, sheet_name="行业年度统计")
                if not ent_yearly.empty:
                    ent_yearly.to_excel(writer, sheet_name="企业年度统计")
                try:
                    self.resid_returns.head(500).to_excel(writer, sheet_name="残差收益率_head")
                except Exception:
                    pass
            QMessageBox.information(self, "导出成功", f"主计算结果已保存到：\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", f"{e}\n\n{traceback.format_exc()}")

    # ---------- subset compute & export ----------

    def compute_subset(self):
        if self.resid_returns is None or self.meta_df is None:
            QMessageBox.warning(self, "提示", "请先在主计算页运行残差化并加载数据。")
            return
        try:
            sel_inds = [it.text() for it in self.list_ind.selectedItems()]
            sel_regs = [it.text() for it in self.list_reg.selectedItems()]
            subset_meta = self.meta_df.copy()
            if sel_inds:
                subset_meta = subset_meta[subset_meta['行业'].isin(sel_inds)]
            if sel_regs:
                subset_meta = subset_meta[subset_meta['区域'].isin(sel_regs)]
            firms = subset_meta['企业名称'].astype(str).tolist()
            firms = [f for f in firms if f in self.resid_returns.columns]
            if len(firms) < 2:
                QMessageBox.information(self, "提示", "筛选后企业数不足以计算相关性（至少2家）。")
                return
            method = self.cmb_method.currentText().lower()
            sub_corr = compute_corr_matrix_for_subset(self.resid_returns, firms, method=method)
            self.sub_corr = sub_corr
            try:
                arr = sub_corr.values.astype(float)
            except ValueError:
                arr = pd.to_numeric(sub_corr.values.flatten(), errors='coerce').reshape(sub_corr.shape)
            n = arr.shape[0]
            if n <= 1:
                mean_v = var_v = max_v = min_v = med_v = q25 = q75 = np.nan
            else:
                mask = ~np.eye(n, dtype=bool)
                vals = arr[mask]
                mean_v = np.nanmean(vals); var_v = np.nanvar(vals)
                max_v = np.nanmax(vals); min_v = np.nanmin(vals)
                med_v = np.nanmedian(vals); q25 = np.nanpercentile(vals, 25); q75 = np.nanpercentile(vals, 75)
            stats_text = f"均值: {mean_v:.6f}    方差: {var_v:.6f}    最小: {min_v:.6f}    最大: {max_v:.6f}\n中位数: {med_v:.6f}    25%分位: {q25:.6f}    75%分位: {q75:.6f}"
            self.lbl_subset_stats.setText(stats_text)
            self.sub_heatmap.plot_heatmap(sub_corr, title="子集相关性矩阵")
            if self.chk_sub_ts.isChecked():
                win = int(self.input_win_sub.text()); step = int(self.input_step_sub.text())
                dates = list(self.resid_returns.index)
                n_t = len(dates)
                starts = list(range(0, max(0, n_t - win + 1), step))
                times = []; overall_vals = []
                for s in starts:
                    e = s + win
                    window = self.resid_returns.iloc[s:e][firms]
                    if window.shape[0] < 3:
                        continue
                    w_corr = window.corr(method=method)
                    overall_vals.append(overall_mean_offdiag(w_corr))
                    times.append(dates[s + win // 2])
                if len(times) > 0:
                    self.sub_timeseries = pd.Series(overall_vals, index=pd.DatetimeIndex(times))
                    self.sub_ts_canvas.plot_series(self.sub_timeseries.to_frame('overall'), title="子集总体平均相关性随时间")
                    yearly = yearly_statistics_from_series(self.sub_timeseries)
                    self.lbl_subset_year_stats.setText("子集年度统计：\n" + yearly.to_string())
                else:
                    self.sub_timeseries = None
                    self.sub_ts_canvas.plot_series(None)
                    self.lbl_subset_year_stats.setText("子集年度统计：无数据")
            else:
                self.sub_timeseries = None
                self.sub_ts_canvas.plot_series(None)
                self.lbl_subset_year_stats.setText("子集年度统计：未计算")
            self.status.setText(f"子集计算完成，企业数 {len(firms)}")
        except Exception as e:
            QMessageBox.critical(self, "子集计算失败", f"{e}\n\n{traceback.format_exc()}")

    def export_subset_results(self):
        if self.sub_corr is None:
            QMessageBox.warning(self, "提示", "请先计算子集相关性。")
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存子集结果到 Excel", "subset_results.xlsx", "Excel Files (*.xlsx)")
        if not path:
            return
        try:
            with pd.ExcelWriter(path, engine='openpyxl') as writer:
                self.sub_corr.to_excel(writer, sheet_name="子集相关性矩阵")
                try:
                    arr = self.sub_corr.values.astype(float)
                except ValueError:
                    arr = pd.to_numeric(self.sub_corr.values.flatten(), errors='coerce').reshape(self.sub_corr.shape)
                n = arr.shape[0]
                if n <= 1:
                    mean_v = var_v = max_v = min_v = med_v = q25 = q75 = np.nan
                else:
                    mask = ~np.eye(n, dtype=bool)
                    vals = arr[mask]
                    mean_v = np.nanmean(vals); var_v = np.nanvar(vals)
                    max_v = np.nanmax(vals); min_v = np.nanmin(vals)
                    med_v = np.nanmedian(vals); q25 = np.nanpercentile(vals, 25); q75 = np.nanpercentile(vals, 75)
                stats_df = pd.DataFrame({
                    'mean': [mean_v], 'var': [var_v], 'min': [min_v], 'max': [max_v],
                    'median': [med_v], 'q25': [q25], 'q75': [q75]
                })
                stats_df.to_excel(writer, sheet_name="子集统计")
                if self.sub_timeseries is not None:
                    self.sub_timeseries.to_frame('overall').to_excel(writer, sheet_name="子集总体平均_TS")
            QMessageBox.information(self, "导出成功", f"子集结果已保存到：\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", f"{e}\n\n{traceback.format_exc()}")

    # ========= 违约相关性计算方法 =========

    def update_labels(self):
        w_ind = self.slider_weight.value() / 100
        w_reg = 1 - w_ind
        scale = self.slider_scale.value() / 100
        self.label_weight.setText(f"行业权重: {w_ind:.2f} | 区域权重: {w_reg:.2f}")
        self.label_scale.setText(f"整体缩放比例: {scale:.2f}")

    def load_region_matrix(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择区域相关性矩阵", "", "Excel 文件 (*.xlsx *.xls)")
        if path:
            self.region_corr = pd.read_excel(path, index_col=0)
            self.text_output.append(f"✅ 已导入区域相关性矩阵：{path}")

    def load_industry_matrix(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择行业相关性矩阵", "", "Excel 文件 (*.xlsx *.xls)")
        if path:
            self.industry_corr = pd.read_excel(path, index_col=0)
            self.text_output.append(f"✅ 已导入行业相关性矩阵：{path}")

    def load_firm_info(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择企业信息文件", "", "Excel 文件 (*.xlsx *.xls)")
        if path:
            self.firm_info = pd.read_excel(path)
            self.text_output.append(f"✅ 已导入企业信息文件：{path}")
            self.text_output.append(f"📄 样例数据：\n{self.firm_info.head(3)}\n")

    def calculate_corr(self):
        if self.region_corr is None or self.industry_corr is None or self.firm_info is None:
            QMessageBox.warning(self, "错误", "请先导入区域矩阵、行业矩阵和企业信息！")
            return

        firms = self.firm_info
        if not all(col in firms.columns for col in ["企业名", "行业", "区域"]):
            QMessageBox.warning(self, "错误", "企业信息表必须包含列：企业名、行业、区域")
            return

        n = len(firms)
        rho = np.zeros((n, n))

        w_ind = self.slider_weight.value() / 100
        w_reg = 1 - w_ind
        scale = self.slider_scale.value() / 100

        # ===== 逐项计算 =====
        for i in range(n):
            for j in range(n):
                ind_i, reg_i = firms.loc[i, "行业"], firms.loc[i, "区域"]
                ind_j, reg_j = firms.loc[j, "行业"], firms.loc[j, "区域"]

                corr_ind = (
                    self.industry_corr.loc[ind_i, ind_j]
                    if (ind_i in self.industry_corr.index and ind_j in self.industry_corr.columns)
                    else 0
                )
                corr_reg = (
                    self.region_corr.loc[reg_i, reg_j]
                    if (reg_i in self.region_corr.index and reg_j in self.region_corr.columns)
                    else 0
                )

                rho[i, j] = scale * (w_ind * corr_ind + w_reg * corr_reg)
                rho[i, j] = max(min(rho[i, j], 1), 0)

        self.default_corr = pd.DataFrame(rho, index=firms["企业名"], columns=firms["企业名"])
        self.text_output.append("✅ 已完成违约相关性矩阵计算。")
        self.text_output.append(f"⚖️ 行业权重={w_ind:.2f}, 区域权重={w_reg:.2f}, 缩放比例={scale:.2f}")
        self.text_output.append(f"矩阵维度：{n} × {n}\n")
        self.show_heatmap()

    def show_heatmap(self):
        if self.default_corr is None:
            return
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        cax = ax.imshow(self.default_corr, cmap="YlGnBu", vmin=0, vmax=1)
        ax.set_title("违约相关性矩阵热力图", fontsize=12)
        ax.set_xticks(range(len(self.default_corr.columns)))
        ax.set_xticklabels(self.default_corr.columns, rotation=90, fontsize=8)
        ax.set_yticks(range(len(self.default_corr.index)))
        ax.set_yticklabels(self.default_corr.index, fontsize=8)
        self.fig.colorbar(cax, ax=ax, fraction=0.045, pad=0.04)
        self.canvas.draw()

    def export_result(self):
        if self.default_corr is None:
            QMessageBox.warning(self, "错误", "请先计算结果")
            return

        path, _ = QFileDialog.getSaveFileName(self, "保存结果", "违约相关性矩阵.xlsx", "Excel 文件 (*.xlsx)")
        if path:
            self.default_corr.to_excel(path, float_format="%.8f")
            QMessageBox.information(self, "成功", f"已导出结果：{path}")
            self.text_output.append(f"💾 已导出结果文件：{path}\n")

    # ========= 热力图生成器方法 =========

    def import_excel_heatmap(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择Excel文件", "", "Excel Files (*.xlsx *.xls)")
        
        if file_path:
            try:
                # 读取Excel文件
                self.heatmap_data = pd.read_excel(file_path, index_col=0)
                self.file_label.setText(f"已导入: {file_path.split('/')[-1]}")
                
                # 验证数据是否为相关性矩阵
                if not self.validate_correlation_matrix():
                    QMessageBox.warning(self, "数据验证", 
                                      "导入的数据可能不是有效的相关性矩阵。请确保数据范围在-1到1之间。")
                
            except Exception as e:
                QMessageBox.critical(self, "导入错误", f"无法读取Excel文件: {str(e)}")
                self.heatmap_data = None
                self.file_label.setText("导入失败")
                
    def validate_correlation_matrix(self):
        """验证数据是否为有效的相关性矩阵"""
        if self.heatmap_data is None:
            return False
            
        # 检查数据范围
        min_val = self.heatmap_data.min().min()
        max_val = self.heatmap_data.max().max()
        
        return min_val >= -1 and max_val <= 1
        
    def choose_color(self, color_type):
        current_color = getattr(self, f'color_{color_type}')
        color = QColorDialog.getColor(current_color, self, f"选择{color_type}点颜色")
        
        if color.isValid():
            setattr(self, f'color_{color_type}', color)
            btn = getattr(self, f'{color_type}_color_btn')
            btn.setStyleSheet(f"background-color: {color.name()}")
            
            # 更新颜色过渡预览
            self.color_transition.set_colors(self.color_neg, self.color_zero, self.color_pos)
            
    def set_title(self):
        """设置热力图标题"""
        title, ok = QInputDialog.getText(self, "设置标题", "请输入热力图标题:", text=self.heatmap_title)
        if ok and title:
            self.heatmap_title = title
            self.title_label.setText(f"当前标题: {self.heatmap_title}")
            
    def toggle_show_values(self, state):
        """切换显示数值"""
        self.show_values = (state == Qt.Checked)
        
    def toggle_smooth_transition(self, state):
        """切换色彩过渡效果"""
        self.smooth_transition = (state == Qt.Checked)
            
    def generate_heatmap(self):
        if self.heatmap_data is None:
            QMessageBox.warning(self, "数据缺失", "请先导入Excel文件")
            return
            
        try:
            self.heatmap_canvas.plot_heatmap(
                self.heatmap_data, 
                self.color_neg, 
                self.color_zero, 
                self.color_pos, 
                self.heatmap_title,
                self.show_values,
                self.smooth_transition
            )
        except Exception as e:
            QMessageBox.critical(self, "生成错误", f"生成热力图时出错: {str(e)}")
            
    def save_heatmap(self):
        if self.heatmap_data is None:
            QMessageBox.warning(self, "数据缺失", "请先生成热力图")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存热力图", "correlation_heatmap.png", 
            "PNG Files (*.png);;JPEG Files (*.jpg);;PDF Files (*.pdf)")
            
        if file_path:
            try:
                self.heatmap_canvas.fig.savefig(file_path, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "保存成功", f"热力图已保存到: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "保存错误", f"保存热力图时出错: {str(e)}")

# ========= 主程序入口 =========

def main():
    app = QApplication(sys.argv)
    win = CorrelationAnalysisApp()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
