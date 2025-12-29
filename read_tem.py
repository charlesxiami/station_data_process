import os
import numpy as np
import pandas as pd
from netCDF4 import Dataset, date2num
from datetime import datetime, timedelta

def load_station_info(csv_path):
    """
    加载CSV站点信息，返回站点字典和站点列表
    :param csv_path: CSV文件路径
    :return: station_dict (区站号: (站名, 纬度, 经度, 高度)), station_codes (区站号列表)
    """
    # 读取CSV文件
    df = pd.read_csv(csv_path, encoding='utf-8')
    # 检查必要列是否存在
    required_cols = ['站名', '区站号', '纬度', '经度', '高度']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV文件必须包含列：{required_cols}")
    
    # 构建站点字典（区站号为键）
    station_dict = {}
    for _, row in df.iterrows():
        station_code = str(int(row['区站号']))  # 确保区站号为字符串
        station_dict[station_code] = (
            row['站名'],
            float(row['纬度'])/100,
            float(row['经度'])/100,
            float(row['高度'])
        )
    # 提取区站号列表（固定顺序）
    station_codes = list(station_dict.keys())
    return station_dict, station_codes

def generate_time_range(start_date_str, end_date_str):
    """
    生成指定时间范围的日期列表（每日）
    :param start_date_str: 开始日期，格式'YYYYMMDD'
    :param end_date_str: 结束日期，格式'YYYYMMDD'
    :return: date_list (datetime对象列表), date_str_list (日期字符串列表'YYYYMMDD')
    """
    start_date = datetime.strptime(start_date_str, '%Y%m%d')
    end_date = datetime.strptime(end_date_str, '%Y%m%d')
    
    date_list = []
    date_str_list = []
    current_date = start_date
    while current_date <= end_date:
        date_list.append(current_date)
        date_str_list.append(current_date.strftime('%Y%m%d'))
        current_date += timedelta(days=1)
    return date_list, date_str_list

def read_txt_data(txt_file_path, station_codes, default_value=np.nan):
    """
    读取单个TXT文件数据，返回对应站点的数据值
    :param txt_file_path: TXT文件路径
    :param station_codes: 目标区站号列表
    :param default_value: 缺省值
    :return: data_array (站点维度的数组)
    """
    # 初始化数据数组（默认缺省值）
    data_array = np.full(len(station_codes), default_value, dtype=np.float32)
    station_code_index = {code: idx for idx, code in enumerate(station_codes)}
    
    try:
        # 读取TXT文件，跳过前两行
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()[2:]  # 跳过前两行
        
        # 解析每一行数据
        for line in lines:
            line = line.strip()
            if not line:  # 跳过空行
                continue
            # 按逗号分割列（处理末尾逗号的情况）
            cols = [col.strip() for col in line.rstrip(',').split(',')]
            if len(cols) < 5:  # 至少需要经度、纬度、高度、数据值、站点号
                continue
            
            # 提取数据（容错处理）
            try:
                station_code = str(int(cols[4]))  # 区站号（第5列，索引4）
                data_value = float(cols[3])       # 数据值（第4列，索引3）
                if data_value > 200 or data_value < 0 or data_value == 999999:
                    data_value = default_value
            except (ValueError, IndexError):
                continue  # 解析失败则跳过该行
            
            # 如果站点在目标列表中，填充数据
            if station_code in station_code_index:
                idx = station_code_index[station_code]
                data_array[idx] = data_value
    except Exception as e:
        print(f"读取TXT文件失败 {txt_file_path}: {e}")
    
    return data_array

def build_data_array(root_dir, start_date_str, end_date_str, station_codes, keyword='TEM'):
    """
    构建时空数据数组（时间×站点）
    :param root_dir: TXT文件根目录
    :param start_date_str: 开始日期'YYYYMMDD'
    :param end_date_str: 结束日期'YYYYMMDD'
    :param station_codes: 区站号列表
    :param keyword: TXT文件名关键词
    :return: data_array (时间×站点), date_list (datetime列表)
    """
    # 生成时间范围
    date_list, date_str_list = generate_time_range(start_date_str, end_date_str)
    n_times = len(date_list)
    n_stations = len(station_codes)
    
    # 初始化数据数组（时间×站点）
    data_array = np.full((n_times, n_stations), np.nan, dtype=np.float32)
    
    # 遍历每个日期读取对应TXT文件
    for time_idx, date_str in enumerate(date_str_list):
        # 解析日期路径：YYYY/YYYYMM/YYYYMMDD/
        year = date_str[:4]
        year_month = date_str[:6]
        date_dir = os.path.join(root_dir, year, year_month, date_str)
        
        # 查找包含关键词的TXT文件
        txt_file = None
        if os.path.exists(date_dir):
            for file in os.listdir(date_dir):
                if keyword in file and file.endswith('.txt'):
                    txt_file = os.path.join(date_dir, file)
                    break
        
        # 读取TXT数据并填充
        if txt_file:
            station_data = read_txt_data(txt_file, station_codes)
            data_array[time_idx, :] = station_data
        else:
            print(f"警告：日期 {date_str} 未找到包含关键词 {keyword} 的TXT文件，填充缺省值")
    
    return data_array, date_list

def save_to_netcdf(output_path, data_array, date_list, station_dict, station_codes, 
                   var_name, var_longname, var_units):
    """
    将数据保存为NetCDF格式
    :param output_path: 输出NC文件路径
    :param data_array: 时间×站点的数据数组
    :param date_list: datetime列表
    :param station_dict: 站点字典
    :param station_codes: 区站号列表
    :param var_name: 变量名
    :param var_longname: 变量长名称
    """
    # 创建NetCDF文件
    nc = Dataset(output_path, 'w', format='NETCDF4')
    
    # 定义维度
    nc.createDimension('time', None)  # 无限时间维度
    nc.createDimension('station', len(station_codes))
    
    # 定义时间变量
    time_var = nc.createVariable('time', 'f8', ('time',))
    time_var.units = 'days since 1965-01-01 00:00:00'
    time_var.calendar = 'gregorian'
    time_var.long_name = 'Time'
    # 将datetime转换为数值
    time_vals = date2num(date_list, units=time_var.units, calendar=time_var.calendar)
    time_var[:] = time_vals
    
    # 定义站点相关变量
    # 区站号（VLEN字符串类型）
    station_code_var = nc.createVariable('station_code', 'U10', ('station',))  # U10=UTF-8字符串，长度10
    station_code_var.long_name = 'Station code'
    # 逐个索引赋值（兼容VLEN要求）
    for i, code in enumerate(station_codes):
        station_code_var[i] = code
    
    # 站名（VLEN字符串类型）
    station_name_var = nc.createVariable('station_name', 'U50', ('station',))  # U50=UTF-8字符串，长度50
    station_name_var.long_name = 'Station name'
    # 逐个索引赋值
    for i, code in enumerate(station_codes):
        station_name_var[i] = station_dict[code][0]
    
    # 纬度
    lat_var = nc.createVariable('latitude', 'f4', ('station',))
    lat_var.long_name = 'Station latitude'
    lat_var.units = 'degrees_north'
    lat_var.standard_name = 'latitude'  # 新增：CDO识别的标准名称
    lat_var.axis = 'Y'  # 新增：维度轴标识
    lat_var[:] = [station_dict[code][1] for code in station_codes]
    
    # 经度
    lon_var = nc.createVariable('longitude', 'f4', ('station',))
    lon_var.long_name = 'Station longitude'
    lon_var.units = 'degrees_east'
    lon_var.standard_name = 'longitude'  # 新增：CDO识别的标准名称
    lon_var.axis = 'X'  # 新增：维度轴标识
    lon_var[:] = [station_dict[code][2] for code in station_codes]
    
    # 高度
    # height_var = nc.createVariable('height', 'f4', ('station',))
    # height_var.long_name = 'Station height'
    # height_var.units = 'm'
    # height_var[:] = [station_dict[code][3] for code in station_codes]
    
    # 主要气象变量
    data_var = nc.createVariable(var_name, 'f4', ('time', 'station'), fill_value=np.nan)
    data_var.long_name = var_longname
    data_var.units = var_units  # 改为使用主函数传入的单位（原代码是固定'unknown'，需同步修改）
    data_var.coordinates = "longitude latitude"  # 告诉CDO该变量的坐标是经纬度
    data_var[:] = data_array
    
    # 添加全局属性
    nc.description = 'Meteorological data from TXT files'
    nc.history = f'Created {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
    nc.source = 'Processed from TXT files and CSV station info'
    
    # 关闭文件
    nc.close()
    print(f"NetCDF文件已保存至：{output_path}")

def main():
    """主函数：配置参数并执行完整流程"""
    # ====================== 配置参数（根据实际情况修改）======================
    CSV_PATH = '/mnt/e/data/observation/zhejiang/zj_station.csv'               # 站点信息CSV文件路径
    ROOT_DIR = '/mnt/e/data/observation/zhejiang/daily'                      # TXT文件根目录（包含YYYY/YYYYMM/YYYYMMDD结构）
    START_DATE = '19650101'                  # 开始日期（YYYYMMDD）
    END_DATE = '20241231'                    # 结束日期（YYYYMMDD）
    KEYWORD = 'TEM'                    # TXT文件名关键词
    BASE_NC_NAME = 'zj_tem'               # 基础NetCDF文件名
    OUTPUT_NC_PATH = f"{BASE_NC_NAME}_{START_DATE}-{END_DATE}.nc"    # 输出NetCDF文件路径
    VAR_NAME = 'tem'           # 气象变量名
    VAR_LONGNAME = 'Temperature'  # 变量长名称
    VAR_UNITS = 'degC'                    # 变量单位（根据实际数据修改）
    
    # ====================== 执行数据处理流程 ======================
    try:
        # 1. 加载站点信息
        print("加载站点信息...")
        station_dict, station_codes = load_station_info(CSV_PATH)
        print(f"成功加载 {len(station_codes)} 个站点信息")
        
        # 2. 构建时空数据数组
        print("读取TXT文件并构建数据数组...")
        data_array, date_list = build_data_array(
            root_dir=ROOT_DIR,
            start_date_str=START_DATE,
            end_date_str=END_DATE,
            station_codes=station_codes,
            keyword=KEYWORD
        )
        print(f"数据数组形状：时间×站点 = {data_array.shape}")
        
        # ====================== 新增：极值统计 ======================
        # 提取经纬度数组
        lats = np.array([station_dict[code][1] for code in station_codes])
        lons = np.array([station_dict[code][2] for code in station_codes])
        # 计算极值（排除NaN）
        lat_min, lat_max = np.nanmin(lats), np.nanmax(lats)
        lon_min, lon_max = np.nanmin(lons), np.nanmax(lons)
        data_min, data_max = np.nanmin(data_array), np.nanmax(data_array)
        # 打印极值
        print("\n=== 数据极值统计 ===")
        print(f"经度范围（已除以100）：{lon_min:.4f} ~ {lon_max:.4f}")
        print(f"纬度范围（已除以100）：{lat_min:.4f} ~ {lat_max:.4f}")
        print(f"{VAR_NAME}数据范围（异常值已替换为NaN）：{data_min:.4f} ~ {data_max:.4f}")
        print("====================\n")
        # ===========================================================

        # 3. 保存为NetCDF文件
        print("保存NetCDF文件...")
        save_to_netcdf(
            output_path=OUTPUT_NC_PATH,
            data_array=data_array,
            date_list=date_list,
            station_dict=station_dict,
            station_codes=station_codes,
            var_name=VAR_NAME,
            var_longname=VAR_LONGNAME,
            var_units=VAR_UNITS
        )
        
        print("数据处理完成！")
        
    except Exception as e:
        print(f"程序执行出错：{e}")

if __name__ == '__main__':
    main()