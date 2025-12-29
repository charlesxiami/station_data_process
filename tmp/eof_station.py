import os
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import shapefile
from shapely.geometry import shape
from scipy import linalg
from scipy.stats import f, chi2

# 配置参数（与原程序保持一致）
NC_FILE_PATH = 'zj_tem_19650101-20241231.nc'
OUTPUT_DIR = 'plots/eof'
VAR_NAME = 'tem'
VAR_UNITS = 'degC'
TITLE_PREFIX = 'Temperature'
PLOT_LON_RANGE = [117.5, 123.5]
PLOT_LAT_RANGE = [27.0, 31.5]
FIG_SIZE = (12, 6)
DPI = 200

# 复用原程序的NC数据加载函数
def load_nc_data(nc_path):
    """加载NetCDF数据并返回xarray Dataset（复用原程序函数）"""
    ds = xr.open_dataset(nc_path)
    try:
        ds = ds.convert_calendar('gregorian', use_cftime=False)
    except:
        time_units = ds['time'].attrs.get('units', 'days since 1965-01-01')
        unit_parts = time_units.split(' since ')
        if len(unit_parts) != 2:
            start_date = np.datetime64('1965-01-01')
            time_delta = np.timedelta64(1, 'D')
        else:
            unit = unit_parts[0].strip()
            start_date_str = unit_parts[1].split(' ')[0].strip()
            start_date = np.datetime64(start_date_str)
            unit_map = {
                'day': 'D', 'days': 'D',
                'hour': 'h', 'hours': 'h',
                'minute': 'm', 'minutes': 'm',
                'second': 's', 'seconds': 's'
            }
            time_delta = np.timedelta64(1, unit_map.get(unit, 'D'))
        time_vals = ds['time'].values.astype(np.float64)
        ds['time'] = start_date + (time_vals * time_delta).astype('timedelta64[D]')
    return ds

def preprocess_annual_average(ds, var_name):
    """
    功能1：数据预处理 - 对原始站点数据进行年平均处理
    参数：
        ds: xarray Dataset - 原始站点数据
        var_name: str - 待处理的变量名
    返回：
        ds_annual: xarray Dataset - 年平均后的站点数据
        annual_time: np.ndarray - 年平均对应的时间标签（年份）
    """
    # 提取变量数据，按时间维度重采样做年平均（YE=每年年末，兼容日期格式）
    var_daily = ds[var_name]
    # 按年平均，保留站点维度，生成年平均数据
    var_annual = var_daily.resample(time='YE').mean(dim='time', skipna=True)
    # 提取年平均对应的年份（用于后续时间序列标注）
    annual_years = var_annual['time'].dt.year.values
    # 构造年平均后的Dataset，保留经纬度信息
    ds_annual = xr.Dataset(
        {var_name: (('time', 'station'), var_annual.values)},
        coords={
            'time': annual_years,
            'longitude': ds['longitude'],
            'latitude': ds['latitude']
        }
    )
    print(f"年平均预处理完成：原始时间数={len(ds['time'])}，年平均时间数={len(ds_annual['time'])}")
    print(f"年平均数据维度：(年份数, 站点数) = {ds_annual[var_name].shape}")
    return ds_annual, annual_years

def remove_time_trend(ds_annual, var_name):
    """
    对年平均后的站点数据进行时间维度去趋势处理
    参数：
        ds_annual: xarray Dataset - 年平均后的站点数据
        var_name: str - 待去趋势的变量名
    返回：
        ds_annual_detrended: xarray Dataset - 时间去趋势后的年平均站点数据
    """
    # 提取年平均数据和年份（时间维度）
    var_annual = ds_annual[var_name].values  # 维度：(年份数, 站点数)
    n_years, n_station = var_annual.shape
    years = ds_annual['time'].values.astype(np.float64)  # 年份转为浮点型，用于线性拟合

    # 初始化去趋势后的数据数组
    var_annual_detrended = np.zeros_like(var_annual)

    # 对每个站点单独进行时间去趋势（避免站点间差异干扰）
    for station_idx in range(n_station):
        # 提取单个站点的年际时间序列
        station_series = var_annual[:, station_idx]
        
        # 跳过全为NaN的站点
        if np.all(np.isnan(station_series)):
            var_annual_detrended[:, station_idx] = station_series
            continue
        
        # 线性拟合：求解时间趋势（斜率+截距）
        valid_mask = ~np.isnan(station_series)
        x = years[valid_mask]
        y = station_series[valid_mask]
        
        # 最小二乘法拟合线性趋势
        slope, intercept = np.polyfit(x, y, 1)
        # 构建趋势线
        trend_line = slope * years + intercept
        # 去趋势：原始序列 - 趋势线
        station_series_detrended = station_series - trend_line
        
        # 赋值到去趋势数组中
        var_annual_detrended[:, station_idx] = station_series_detrended

    # 构造去趋势后的Dataset，保留原有坐标信息
    ds_annual_detrended = xr.Dataset(
        {var_name: (('time', 'station'), var_annual_detrended)},
        coords={
            'time': ds_annual['time'],
            'longitude': ds_annual['longitude'],
            'latitude': ds_annual['latitude']
        }
    )

    print(f"时间去趋势处理完成：年平均数据（{n_years}年×{n_station}站）已去除线性时间趋势")
    return ds_annual_detrended


def eof_analysis_station_data(ds, var_name, n_modes=3):
    """
    改进版EOF分析：基于年平均后的站点数据进行计算
    参数：
        ds: xarray Dataset - 年平均后的站点数据
        var_name: str - 要分析的变量名（如'tem'）
        n_modes: int - 要提取的EOF模态数（默认前3模）
    返回：
        eof_modes: np.ndarray - EOF空间模态（站点数 × 模态数）
        pc_timeseries: np.ndarray - 主成分时间序列（年份数 × 模态数）
        explained_variance: np.ndarray - 各模态解释的方差占比
        explained_variance_cumulative: np.ndarray - 累积方差占比
        eigenvalues: np.ndarray - 特征值（已排序）
    """
    # 1. 提取年平均后的数据（维度：年份 × 站点）
    var_data = ds[var_name].values
    n_time, n_station = var_data.shape
    print(f"\nEOF分析输入（年平均后）：年份数={n_time}，站点数={n_station}")

    # 2. 数据预处理：去除时间均值（每个站点的年均值长期均值），处理NaN
    station_time_mean = np.nanmean(var_data, axis=0)
    var_anomaly = var_data - station_time_mean
    # 填充NaN值：用对应站点的异常值均值填充
    for i in range(n_station):
        station_anomaly_mean = np.nanmean(var_anomaly[:, i])
        nan_mask = np.isnan(var_anomaly[:, i])
        var_anomaly[nan_mask, i] = station_anomaly_mean

    # 3. 计算协方差矩阵（站点 × 站点）
    cov_matrix = np.cov(var_anomaly.T, ddof=1)

    # 4. 特征值分解（求解EOF）
    eigenvalues, eigenvectors = linalg.eigh(cov_matrix)
    # 按特征值从大到小排序
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_idx]
    eigenvectors = eigenvectors[:, sorted_idx]

    # 5. 计算解释方差占比
    total_variance = np.sum(eigenvalues)
    explained_variance = (eigenvalues / total_variance) * 100
    explained_variance_cumulative = np.cumsum(explained_variance)

    # 6. 提取前n_modes个模态
    eof_modes = eigenvectors[:, :n_modes]
    pc_timeseries = np.dot(var_anomaly, eof_modes)

    # 7. 模态符号调整
    for i in range(n_modes):
        max_idx = np.argmax(np.abs(eof_modes[:, i]))
        sign = np.sign(eof_modes[max_idx, i])
        eof_modes[:, i] *= sign
        pc_timeseries[:, i] *= sign

    return eof_modes, pc_timeseries, explained_variance, explained_variance_cumulative, eigenvalues

def test_eof_significance(eigenvalues, n_time, n_station, significance_level=0.05):
    """
    功能2：EOF模态显著性检验（两种常用方法，输出详细结果）
    参数：
        eigenvalues: np.ndarray - EOF分析得到的特征值（已按从大到小排序）
        n_time: int - 时间样本数（年平均后的年份数）
        n_station: int - 站点数
        significance_level: float - 显著性水平（默认0.05，即95%置信度）
    返回：
        significance_results: dict - 各模态显著性检验结果
    """
    n_modes = len(eigenvalues)
    # 初始化结果字典
    significance_results = {
        'eigenvalues': eigenvalues,
        'significance_level': significance_level,
        'north_test': [],  # North检验（最常用，用于判断相邻模态是否可区分）
        'bartlett_test': []  # Bartlett检验（用于判断单个模态是否显著）
    }

    # ========== 修正后的 North 检验（核心修改） ==========
    #1. North检验（相邻模态可区分性）
    eigenvalue_errors = eigenvalues * np.sqrt(2 / n_time)  # 误差范围计算正确
    for i in range(n_modes):
        if i == n_modes - 1:
            north_result = {
                'mode': i+1,
                'eigenvalue': eigenvalues[i],
                'error': eigenvalue_errors[i],
                'lower_bound': eigenvalues[i] - eigenvalue_errors[i],
                'upper_bound': eigenvalues[i] + eigenvalue_errors[i],
                'is_distinguishable_from_next': None,
                'significance': None
            }
        else:
            # 提取当前模态和下一模态的误差边界（正确提取）
            current_lower = eigenvalues[i] - eigenvalue_errors[i]  # 当前模态下边界
            current_upper = eigenvalues[i] + eigenvalue_errors[i]  # 当前模态上边界
            next_lower = eigenvalues[i+1] - eigenvalue_errors[i+1]  # 下一模态下边界
            next_upper = eigenvalues[i+1] + eigenvalue_errors[i+1]  # 下一模态上边界

            # 修正核心判断条件：当前模态下边界 > 下一模态上边界 → 无重叠，可区分
            is_distinguishable = current_lower > next_upper
            north_result = {
                'mode': i+1,
                'eigenvalue': eigenvalues[i],
                'error': eigenvalue_errors[i],
                'lower_bound': current_lower,
                'upper_bound': current_upper,
                'next_mode_eigenvalue': eigenvalues[i+1],
                'next_mode_upper_bound': next_upper,  # 修正：存储下一模态上边界（更合理）
                'next_mode_lower_bound': next_lower,
                'is_distinguishable_from_next': is_distinguishable,
                'significance': 'significant' if is_distinguishable else 'not significant'
            }
        significance_results['north_test'].append(north_result)
    # =====================================================

    # 2. Bartlett检验（单个模态显著性，基于卡方分布）
    # 自由度：df = (n_station - i) * (n_station - i + 1) / 2 （第i个模态，从0开始）
    total_variance = np.sum(eigenvalues)
    for i in range(n_modes):
        # 计算检验统计量
        chi2_stat = -n_time * (n_station - i - 1) * np.log(1 - eigenvalues[i] / total_variance)
        # 计算自由度
        df = (n_station - i) * (n_station - i + 1) // 2 - 1
        # 计算p值
        p_value = 1 - chi2.cdf(chi2_stat, df)
        # 判断是否显著
        is_significant = p_value < significance_level
        bartlett_result = {
            'mode': i+1,
            'eigenvalue': eigenvalues[i],
            'chi2_statistic': chi2_stat,
            'degrees_of_freedom': df,
            'p_value': p_value,
            'is_significant': is_significant,
            'significance': 'significant' if is_significant else 'not significant'
        }
        significance_results['bartlett_test'].append(bartlett_result)

    # 打印显著性检验汇总结果
    # print(f"\n================ EOF模态显著性检验结果（显著性水平={significance_level}） ================")
    # print("--- North检验（相邻模态可区分性） ---")
    # for res in significance_results['north_test']:
    #     if res['is_distinguishable_from_next'] is not None:
    #         print(f"第{res['mode']}模 vs 第{res['mode']+1}模：特征值={res['eigenvalue']:.4f}，误差区间=[{res['lower_bound']:.4f}, {res['upper_bound']:.4f}]，下一模态误差区间=[{res['next_mode_lower_bound']:.4f}, {res['next_mode_upper_bound']:.4f}]，可区分性={res['is_distinguishable_from_next']}（{res['significance']}）")
    #     else:
    #         print(f"第{res['mode']}模：特征值={res['eigenvalue']:.4f}，误差区间=[{res['lower_bound']:.4f}, {res['upper_bound']:.4f}]（无后续模态，不进行可区分性检验）")
    
    # print("\n--- Bartlett检验（单个模态显著性） ---")
    # for res in significance_results['bartlett_test']:
    #     print(f"第{res['mode']}模：卡方统计量={res['chi2_statistic']:.4f}，自由度={res['degrees_of_freedom']}，p值={res['p_value']:.4f}，显著性={res['significance']}")
    # print("====================================================================================")

    # 保存显著性检验结果到文本文件
    sig_txt_path = os.path.join(OUTPUT_DIR, 'eof_significance_test_results.txt')
    with open(sig_txt_path, 'w', encoding='utf-8') as f:
        f.write(f"EOF模态显著性检验结果（显著性水平={significance_level}）\n")
        f.write("="*80 + "\n\n")

        f.write("--- North检验（相邻模态可区分性） ---\n")
        # 修正表头，增加下一模态上边界
        f.write("模态编号,特征值,误差,下界,上界,下一模态特征值,下一模态下界,下一模态上界,可区分性,显著性\n")
        for res in significance_results['north_test']:
            if res['is_distinguishable_from_next'] is not None:
                f.write(f"{res['mode']},{res['eigenvalue']:.4f},{res['error']:.4f},{res['lower_bound']:.4f},{res['upper_bound']:.4f},{res['next_mode_eigenvalue']:.4f},{res['next_mode_lower_bound']:.4f},{res['next_mode_upper_bound']:.4f},{res['is_distinguishable_from_next']},{res['significance']}\n")
            else:
                f.write(f"{res['mode']},{res['eigenvalue']:.4f},{res['error']:.4f},{res['lower_bound']:.4f},{res['upper_bound']:.4f},无,无,无,无,无\n")

        f.write("\n--- Bartlett检验（单个模态显著性） ---\n")
        f.write("模态编号,特征值,卡方统计量,自由度,p值,是否显著,显著性\n")
        for res in significance_results['bartlett_test']:
            f.write(f"{res['mode']},{res['eigenvalue']:.4f},{res['chi2_statistic']:.4f},{res['degrees_of_freedom']},{res['p_value']:.4f},{res['is_significant']},{res['significance']}\n")

    print(f"EOF显著性检验结果已保存至：{sig_txt_path}")
    return significance_results

def save_pc_timeseries_to_txt(annual_years, pc_timeseries, n_modes, output_dir, var_name):
    """
    将前n_modes个PC时间序列保存到TXT文件，格式：year, pc1, pc2, pc3,...
    参数：
        annual_years: np.ndarray - 年平均对应的年份数组
        pc_timeseries: np.ndarray - PC时间序列数据（年份数 × 模态数）
        n_modes: int - 要保存的前n个模态
        output_dir: str - 输出目录
        var_name: str - 变量名（用于文件名标识）
    """
    # 1. 校验数据维度一致性
    if len(annual_years) != pc_timeseries.shape[0]:
        raise ValueError(f"年份数（{len(annual_years)}）与PC时间序列行数（{pc_timeseries.shape[0]}）不一致！")
    if pc_timeseries.shape[1] < n_modes:
        raise ValueError(f"PC时间序列模态数（{pc_timeseries.shape[1]}）小于要保存的模态数（{n_modes}）！")

    # 2. 构造输出文件名
    pc_txt_path = os.path.join(output_dir, f'pc_timeseries_top{n_modes}_modes.txt')

    # 3. 构造表头（year, pc1, pc2, ..., pcn）
    header = 'year'
    for i in range(n_modes):
        header += f', pc{i+1}'

    # 4. 写入数据到TXT
    with open(pc_txt_path, 'w', encoding='utf-8') as f:
        # 写入表头
        f.write(header + '\n')
        # 逐行写入年份和对应PC值
        for year, pc_vals in zip(annual_years, pc_timeseries[:, :n_modes]):
            # 拼接该行数据：年份 + 各PC值（保留4位小数，便于后续分析）
            line = f'{int(year)}'  # 年份转为整数格式
            for pc_val in pc_vals:
                line += f', {pc_val:.4f}'
            f.write(line + '\n')

    print(f"前{n_modes}个PC时间序列已保存至：{pc_txt_path}")
    return pc_txt_path

def plot_eof_results(ds, eof_modes, pc_timeseries, explained_variance, annual_years, n_modes=3):
    """
    绘制EOF分析结果（适配年平均数据的时间标签）
    """
    lon = ds['longitude'].values
    lat = ds['latitude'].values
    shp_path = "/mnt/e/data/shapefile/zhejiang/zhejiang/zhejiang_city.shp"
    sf = shapefile.Reader(shp_path, encoding='gbk')
    shapes = [shape(rec.shape.__geo_interface__) for rec in sf.shapeRecords()]

    for mode in range(n_modes):
        mode_num = mode + 1
        exp_var = explained_variance[mode]

        # 绘制EOF空间分布图
        eof_spatial_path = os.path.join(OUTPUT_DIR, f'eof_mode_{mode_num}_spatial_annual.png')
        fig = plt.figure(figsize=FIG_SIZE, dpi=DPI)
        ax = plt.axes(projection=ccrs.PlateCarree())

        # 1. 提取当前模态的EOF数据
        current_eof = eof_modes[:, mode]
        # 2. 计算最大绝对值，确定对称上下界
        max_abs_eof = np.max(np.abs(current_eof))
        vmin = -max_abs_eof  # 下界：-最大绝对值
        vmax = max_abs_eof   # 上界：最大绝对值

        scatter = ax.scatter(
            lon, lat, c=eof_modes[:, mode],
            cmap='RdBu_r', s=60, alpha=0.9,
            transform=ccrs.PlateCarree(),
            edgecolors='black', linewidths=0.5,
            vmin=vmin, vmax=vmax  # 关键参数：指定对称上下界
        )

        # 叠加Shapefile边界
        for geom in shapes:
            ax.add_geometries(
                [geom],
                crs=ccrs.PlateCarree(),
                facecolor='none',
                edgecolor="k",
                linewidth=0.8,
                zorder=10
            )

        ax.set_xlim(PLOT_LON_RANGE)
        ax.set_ylim(PLOT_LAT_RANGE)

        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.05, aspect=30)
        cbar.set_label(f'EOF Mode {mode_num} Amplitude', fontsize=14, labelpad=10)
        cbar.ax.tick_params(labelsize=10)

        ax.set_title(
            f'{TITLE_PREFIX} - EOF Mode {mode_num} Spatial Distribution (Explained Variance: {exp_var:.2f}%)',
            fontsize=14, pad=20, weight='bold'
        )
        ax.set_xlabel('Longitude (°E)', fontsize=12, weight='medium')
        ax.set_ylabel('Latitude (°N)', fontsize=12, weight='medium')
        ax.grid(False)
        ax.tick_params(labelsize=12)

        plt.tight_layout()
        plt.savefig(eof_spatial_path, bbox_inches='tight')
        plt.close()
        print(f"EOF第{mode_num}模空间分布图（年平均）已保存：{eof_spatial_path}")

        # 绘制PC时间序列图（x轴为年份）
        pc_time_series_path = os.path.join(OUTPUT_DIR, f'eof_mode_{mode_num}_pc_timeseries_annual.png')
        fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)

        ax.plot(
            annual_years, pc_timeseries[:, mode],
            color='steelblue', linewidth=1.2, alpha=0.8, marker='o', markersize=5
        )
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
        ax.fill_between(
            annual_years, pc_timeseries[:, mode], 0,
            where=(pc_timeseries[:, mode] > 0),
            color='lightcoral', alpha=0.3, label='Positive Phase'
        )
        ax.fill_between(
            annual_years, pc_timeseries[:, mode], 0,
            where=(pc_timeseries[:, mode] < 0),
            color='lightblue', alpha=0.3, label='Negative Phase'
        )

        ax.set_title(
            f'{TITLE_PREFIX} - EOF Mode {mode_num} Principal Component Time Series (Annual Avg, Explained Variance: {exp_var:.2f}%)',
            fontsize=14, pad=20, weight='bold'
        )
        ax.set_xlabel('Year', fontsize=12, weight='medium')
        ax.set_ylabel(f'Principal Component Amplitude', fontsize=12, weight='medium')
        ax.legend(fontsize=10)
        ax.grid(False)
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(pc_time_series_path, bbox_inches='tight')
        plt.close()
        print(f"EOF第{mode_num}模PC时间序列图（年平均）已保存：{pc_time_series_path}")

    # 绘制解释方差占比图
    variance_plot_path = os.path.join(OUTPUT_DIR, 'eof_explained_variance_annual.png')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), dpi=DPI, sharex=True)

    modes_idx = np.arange(1, n_modes+1)
    ax1.bar(modes_idx, explained_variance[:n_modes], color='cornflowerblue', alpha=0.7, width=0.6)
    ax1.set_ylabel('Explained Variance (%)', fontsize=12, weight='medium')
    ax1.set_title(f'{TITLE_PREFIX} - EOF Explained Variance per Mode (Annual Avg)', fontsize=14, pad=15, weight='bold')
    ax1.grid(axis='y', alpha=0.3)
    for i, v in enumerate(explained_variance[:n_modes]):
        ax1.text(modes_idx[i], v + 0.5, f'{v:.2f}%', ha='center', va='bottom', fontsize=10)

    ax2.plot(modes_idx, np.cumsum(explained_variance[:n_modes]), color='darkred', linewidth=2, marker='o', markersize=6)
    ax2.set_xlabel('EOF Mode Number', fontsize=12, weight='medium')
    ax2.set_ylabel('Cumulative Explained Variance (%)', fontsize=12, weight='medium')
    ax2.set_title(f'{TITLE_PREFIX} - Cumulative Explained Variance (Annual Avg)', fontsize=14, pad=15, weight='bold')
    ax2.grid(axis='y', alpha=0.3)
    for i, v in enumerate(np.cumsum(explained_variance[:n_modes])):
        ax2.text(modes_idx[i], v + 0.5, f'{v:.2f}%', ha='center', va='bottom', fontsize=10)

    ax2.set_xticks(modes_idx)
    ax2.set_xticklabels([f'Mode {i}' for i in modes_idx])

    plt.tight_layout()
    plt.savefig(variance_plot_path, bbox_inches='tight')
    plt.close()
    print(f"EOF解释方差占比图（年平均）已保存：{variance_plot_path}")

# 主函数
def main_eof():
    """EOF分析主函数（包含年平均预处理和显著性检验）"""
    # 1. 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. 加载原始NC数据
    print("加载NetCDF数据...")
    ds_original = load_nc_data(NC_FILE_PATH)

    # 3. 功能1：年平均预处理
    print("开始进行年平均数据预处理...")
    ds_annual, annual_years = preprocess_annual_average(ds_original, VAR_NAME)

    print("开始对年平均数据进行时间去趋势处理...")
    ds_annual_detrended = remove_time_trend(ds_annual, VAR_NAME)
    # ======================================================

    # 4. 执行EOF分析（基于年平均数据）
    print("开始执行EOF分析...")
    n_modes = 3
    eof_modes, pc_timeseries, explained_variance, cum_var, eigenvalues = eof_analysis_station_data(ds_annual_detrended, VAR_NAME, n_modes)

    # 打印EOF基本信息
    print(f"\nEOF分析结果（前{n_modes}模，年平均数据）：")
    for i in range(n_modes):
        print(f"第{i+1}模解释方差占比：{explained_variance[i]:.2f}%，累积占比：{cum_var[i]:.2f}%")

    # 5. 功能2：EOF模态显著性检验
    n_time = len(ds_annual['time'])  # 年平均后的年份数
    n_station = len(ds_annual['station'])  # 站点数
    _ = test_eof_significance(eigenvalues, n_time, n_station, significance_level=0.05)

    # 6. 绘制EOF结果图像
    print("开始绘制EOF分析结果图像...")
    plot_eof_results(ds_annual, eof_modes, pc_timeseries, explained_variance, annual_years, n_modes)

     # ========== 新增：保存前n_modes个PC时间序列到TXT（核心调用） ==========
    save_pc_timeseries_to_txt(annual_years, pc_timeseries, n_modes, OUTPUT_DIR, VAR_NAME)
    # =====================================================================

    # 7. 裁剪图像
    os.system('mogrify -trim '+'./'+OUTPUT_DIR +'/*.png')
    print(f"\n所有EOF分析图像及结果已保存至目录：{OUTPUT_DIR}")

if __name__ == '__main__':
    main_eof()