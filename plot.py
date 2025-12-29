import os
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，适合服务器环境
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy import stats
from datetime import datetime
from netCDF4 import num2date
import shapefile
from shapely.geometry import shape
# from matplotlib.patches import Polygon
# from matplotlib.collections import PatchCollection

# ====================== 配置参数（根据实际情况修改）======================
NC_FILE_PATH = 'zj_prec_19650101-20241231.nc'  # 输入NC文件路径
OUTPUT_DIR = 'plots'  # 图像输出子目录
VAR_NAME = 'prec'  # 气象变量名（需与NC文件中一致）
VAR_UNITS = 'mm/day'  # 变量单位（用于图像标注）
TITLE_PREFIX = 'precipitation'  # 图像标题前缀
# 地图范围（根据经纬度极值调整）
# MAP_LON_RANGE = [118, 122]
# MAP_LAT_RANGE = [29, 32]

PLOT_LON_RANGE = [117.5, 123.5]
PLOT_LAT_RANGE = [27.0, 31.5]
# ====================== 初始化设置 =======================
# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
# 设置中文字体（避免中文乱码）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
# 设置图像样式
plt.style.use('seaborn-v0_8-whitegrid')
FIG_SIZE = (12, 6)
DPI = 300

plt.rcParams.update({
    'font.size': 12,         # 基础字体从10→12
    'axes.labelsize': 14,    # 坐标轴标签（新增，单独指定）
    'axes.titlesize': 16,    # 标题从14→16
    'xtick.labelsize': 12,   # X轴刻度从10→12
    'ytick.labelsize': 12,   # Y轴刻度从10→12
    'legend.fontsize': 12,   # 图例从10→12
    # 其他参数不变...
})

def load_nc_data(nc_path):
    """加载NetCDF数据并返回xarray Dataset"""
    ds = xr.open_dataset(nc_path)
   # 方案1：使用xarray原生的convert_calendar（推荐，兼容所有时间格式）
    try:
        # 自动转换时间维度为标准datetime64类型
        ds = ds.convert_calendar('gregorian', use_cftime=False)
    except:
        # 方案2：手动计算datetime（备用，避免溢出）
        time_units = ds['time'].attrs.get('units', 'days since 1965-01-01')
        # 解析单位和起始日期
        unit_parts = time_units.split(' since ')
        if len(unit_parts) != 2:
            start_date = np.datetime64('1965-01-01')
            time_delta = np.timedelta64(1, 'D')
        else:
            unit = unit_parts[0].strip()
            start_date_str = unit_parts[1].split(' ')[0].strip()
            start_date = np.datetime64(start_date_str)
            
            # 匹配时间单位（day/hour/minute/second）
            unit_map = {
                'day': 'D', 'days': 'D',
                'hour': 'h', 'hours': 'h',
                'minute': 'm', 'minutes': 'm',
                'second': 's', 'seconds': 's'
            }
            time_delta = np.timedelta64(1, unit_map.get(unit, 'D'))
        
        # 计算datetime（避免大数溢出）
        time_vals = ds['time'].values.astype(np.float64)
        ds['time'] = start_date + (time_vals * time_delta).astype('timedelta64[D]')
    
    return ds

def plot_spatial_distribution(ds, output_path):
    """1. 所有时间平均的站点空间分布图"""
    # 计算时间平均
    var_mean = ds[VAR_NAME].mean(dim='time')
    
    # 创建地图画布
    fig = plt.figure(figsize=FIG_SIZE, dpi=DPI)
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    shp_path = "/mnt/e/data/shapefile/zhejiang/zhejiang/zhejiang_city.shp"
    sf = shapefile.Reader(shp_path, encoding='gbk')
    records = sf.shapeRecords()
    shapes = [shape(rec.shape.__geo_interface__) for rec in records]

    # 设置地图范围和特征
    # ax.set_extent(MAP_LON_RANGE + MAP_LAT_RANGE, crs=ccrs.PlateCarree())
    # ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5)
    # ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.5)
    # ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
    
    # 绘制站点散点（颜色表示数值，大小固定）
    scatter = ax.scatter(
        ds['longitude'], ds['latitude'], c=var_mean, 
        cmap='viridis_r', s=50, alpha=0.8,
        transform=ccrs.PlateCarree(),
        edgecolors='black', linewidths=0.5
    )
    
    # ====（2）叠加 Shapefile 边界====
    for geom in shapes:
        ax.add_geometries(
            [geom],
            crs=ccrs.PlateCarree(),
            facecolor='none',
            edgecolor="k",          # 小黑边，便于识别
            linewidth=0.8,
            zorder=10
        )

    # 设置坐标轴范围（浙江省略大）
    ax.set_xlim(PLOT_LON_RANGE)
    ax.set_ylim(PLOT_LAT_RANGE)

    # 添加颜色条和标注（优化样式）
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.05, aspect=30)
    cbar.set_label(f'{VAR_NAME} ({VAR_UNITS})', fontsize=14, labelpad=10)
    cbar.ax.tick_params(labelsize=10)
    ax.set_title(f'{TITLE_PREFIX} - Spatial Distribution (Time Average)', fontsize=14, pad=20, weight='bold')
    ax.set_xlabel('Longitude (°E)', fontsize=12, weight='medium')
    ax.set_ylabel('Latitude (°N)', fontsize=12, weight='medium')
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)  # 弱化网格线
    ax.tick_params(labelsize=12)  # 调整刻度字体大小
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"空间分布图已保存：{output_path}")
    print("prec min:", np.nanmin(var_mean))
    print("prec max:", np.nanmax(var_mean))

def plot_daily_time_series(ds, output_path):
    """2. 所有站点平均的每日时间序列图"""
    # 计算站点平均
    daily_mean = ds[VAR_NAME].mean(dim='station', skipna=True)

    total_count = len(daily_mean)
    gt30_count = np.sum(daily_mean > 30)
    gt50_count = np.sum(daily_mean > 50)
    gt30_ratio = gt30_count / total_count * 100
    gt50_ratio = gt50_count / total_count * 100

    print(f"\n=== 每日数据统计（>30、>50）===")
    print(f"总有效数据量：{total_count}")
    print(f">30的数量：{gt30_count}，占比：{gt30_ratio:.2f}%")
    print(f">50的数量：{gt50_count}，占比：{gt50_ratio:.2f}%")
    print("================================")
    # ==================================================

    # 创建画布
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    
    # 绘制时间序列
    ax.plot(daily_mean['time'], daily_mean.values, 
            color='steelblue', linewidth=1.5, marker='.', markersize=4)
    
    # 新增：x轴范围 = 时间最小值-1天 ~ 最大值+1天
    time_min = daily_mean['time'].min().values
    time_max = daily_mean['time'].max().values
    ax.set_xlim(
        np.datetime64(time_min) - np.timedelta64(1, 'D'),
        np.datetime64(time_max) + np.timedelta64(1, 'D')
    )

    # 添加标注
    ax.set_title(f'{TITLE_PREFIX} - Daily Time Series (Station Average)', fontsize=14, pad=20)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(f'{VAR_NAME} ({VAR_UNITS})', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"每日时间序列图已保存：{output_path}")
    print("daily min:", np.nanmin(daily_mean))
    print("daily max:", np.nanmax(daily_mean))


def plot_annual_sum_trend(ds, output_path):
    """3. 所有站点平均的每年总和时间序列（含趋势线）"""
    # 计算站点平均→按年求和
    daily_mean = ds[VAR_NAME].mean(dim='station', skipna=True)
    annual_sum = daily_mean.resample(time='YE').sum(skipna=True)
    
    # 提取年份和对应数值
    years = annual_sum['time'].dt.year.values
    year_values = annual_sum.values
    
     # ========== 新增：保存年度总和到txt ==========
    txt_path = os.path.join(OUTPUT_DIR, '3_annual_sum_data.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('year,value\n')  # 表头
        for y, v in zip(years, year_values):
            f.write(f'{y},{v:.4f}\n')
    print(f"年度总和数据已保存：{txt_path}")
    # =============================================

    # 线性趋势拟合
    slope, intercept, r_value, p_value, std_err = stats.linregress(years, year_values)
    trend_line = slope * years + intercept
    
    # 创建画布
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    
    # 绘制年度总和和趋势线
    ax.bar(years, year_values, color='lightcoral', alpha=0.7, width=0.8, label='Annual Sum')
    ax.plot(years, trend_line, color='darkred', linewidth=2, 
            label=f'Trend (slope={slope:.2f}, p={p_value:.4f})')
    
    # 新增：x轴范围 = 年份最小值-1 ~ 最大值+1
    year_min = years.min()
    year_max = years.max()
    ax.set_xlim(year_min - 1, year_max + 1)

    # 添加标注
    ax.set_title(f'{TITLE_PREFIX} - Annual Sum (Station Average) + Trend', fontsize=14, pad=20)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel(f'{VAR_NAME} Sum (mm/year)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"年度总和趋势图已保存：{output_path}")
    print("year min:", np.nanmin(year_values))
    print("year max:", np.nanmax(year_values))

def plot_extreme_counts(ds, output_path_high, output_path_low):
    """4. 计算95%/5%阈值，绘制每年超阈值日数柱线图"""
    # 计算所有每日数据的95%和5%阈值（排除NaN）
    all_daily_data = ds[VAR_NAME].mean(dim='station').values
    all_daily_data = all_daily_data[~np.isnan(all_daily_data)]
    p95 = np.percentile(all_daily_data, 95)
    p5 = np.percentile(all_daily_data, 5)
    print(f"\n=== 极值阈值统计 ===")
    print(f"95% 阈值：{p95:.4f} ({VAR_UNITS})")
    print(f"5% 阈值：{p5:.4f} ({VAR_UNITS})")
    print("====================")
    
    # 计算每日站点均值，判断是否超阈值
    daily_mean = ds[VAR_NAME].mean(dim='station', skipna=True)
    daily_mean['is_above_p95'] = daily_mean > p95
    daily_mean['is_below_p5'] = daily_mean < p5

    # 按年统计超阈值日数
    annual_high_count = daily_mean['is_above_p95'].resample(time='YE').sum(skipna=True)
    annual_low_count = daily_mean['is_below_p5'].resample(time='YE').sum(skipna=True)
    years = annual_high_count['time'].dt.year.values
    
    # 绘制超95%阈值日数图
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    ax.bar(years, annual_high_count.values, color='orangered', alpha=0.7, width=0.8)

    # 新增：x轴范围 = 年份最小值-1 ~ 最大值+1
    year_min = years.min()
    year_max = years.max()
    ax.set_xlim(year_min - 1, year_max + 1)

    ax.set_title(f'{TITLE_PREFIX} - Annual Days Above 95% Threshold ({p95:.2f})', fontsize=14, pad=20)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Number of Days', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path_high, bbox_inches='tight')
    plt.close()
    print(f"超95%阈值日数图已保存：{output_path_high}")

    # ========== 新增：保存超95%阈值日数到txt ==========
    high_txt_path = os.path.join(OUTPUT_DIR, '4_annual_days_above_p95_data.txt')
    with open(high_txt_path, 'w', encoding='utf-8') as f:
        f.write('year,count\n')  # 表头
        for y, c in zip(years, annual_high_count.values):
            f.write(f'{y},{int(c)}\n')  # 日数为整数，转int
    print(f"超95%阈值日数数据已保存：{high_txt_path}")
    # ==================================================
    
    # # 绘制低于5%阈值日数图
    # fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    # ax.bar(years, annual_low_count.values, color='dodgerblue', alpha=0.7, width=0.8)
    # ax.set_title(f'{TITLE_PREFIX} - Annual Days Below 5% Threshold ({p5:.2f})', fontsize=14, pad=20)
    # ax.set_xlabel('Year', fontsize=12)
    # ax.set_ylabel('Number of Days', fontsize=12)
    # ax.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plt.savefig(output_path_low, bbox_inches='tight')
    # plt.close()
    # print(f"低于5%阈值日数图已保存：{output_path_low}")

def main():
    """主函数：执行所有绘图任务"""
    # 1. 加载数据
    print("加载NetCDF数据...")
    ds = load_nc_data(NC_FILE_PATH)
    
    # 2. 执行绘图
    # 2.1 空间分布图
    plot_spatial_distribution(
        ds, 
        os.path.join(OUTPUT_DIR, '1_spatial_distribution.png')
    )
    
    # 2.2 每日时间序列
    plot_daily_time_series(
        ds, 
        os.path.join(OUTPUT_DIR, '2_daily_time_series.png')
    )
    
    # 2.3 年度总和+趋势线
    plot_annual_sum_trend(
        ds, 
        os.path.join(OUTPUT_DIR, '3_annual_sum_trend.png')
    )
    
    # 2.4 极值阈值统计+年度超阈值日数
    plot_extreme_counts(
        ds,
        os.path.join(OUTPUT_DIR, '4_annual_days_above_p95.png'),
        os.path.join(OUTPUT_DIR, '4_annual_days_below_p5.png')
    )
    
    print("\n所有图像已保存至目录：", OUTPUT_DIR)

    os.system('mogrify -trim '+'./'+OUTPUT_DIR +'/*.png')

if __name__ == '__main__':
    main()