import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import re

def enhanced_negative_sample_generation():
    """
    基于多源数据和业务规则的增强型负样本生成 - 优化版
    """
    # 加载调令数据
    df = pd.read_csv('imports/11调令信息.csv')
    
    # 加载相关数据
    water_level_df = pd.read_csv('imports/闸下潮位.csv')
    flow_df = pd.read_csv('imports/实测流量.csv')
    rainfall_df = pd.read_csv('imports/实测降雨.csv')
    forecast_df = pd.read_csv('imports/降雨预报.csv')
    water_status_df = pd.read_csv('imports/水位工况.csv')
    
    # 转换时间格式
    df['SIGNTM'] = pd.to_datetime(df['SIGNTM'])
    df['日期'] = pd.to_datetime(df['日期'])
    
    # 从其他数据源也转换时间格式
    water_level_df['time'] = pd.to_datetime(water_level_df['time'])
    flow_df['监测日期'] = pd.to_datetime(flow_df['监测日期'])
    rainfall_df['监测日期'] = pd.to_datetime(rainfall_df['监测日期'])
    forecast_df['时间'] = pd.to_datetime(forecast_df['预计开始时间'])
    water_status_df['监测日期'] = pd.to_datetime(water_status_df['监测日期'])
    
    # 提取潮汐时间信息
    def extract_tide_time(text):
        if "预计涨潮时间" in text:
            try:
                tide_time_str = text.split("预计涨潮时间")[1].strip().split(" ")[0]
                return pd.to_datetime(tide_time_str)
            except:
                return None
        return None
    
    df['预计涨潮时间'] = df['调度信息'].apply(extract_tide_time)
    
    negative_samples = []
    
    # 策略1: 基于水位数据的无效调令 - 优化版
    def generate_water_based_negative_samples():
        samples = []
        for idx, row in df.iterrows():
            if pd.notna(row['SIGNTM']):
                # 获取调令发布时的水位数据
                sign_time = row['SIGNTM']
                water_data = water_level_df[
                    (water_level_df['time'] >= sign_time - timedelta(hours=1)) &
                    (water_level_df['time'] <= sign_time + timedelta(hours=1))
                ]
                
                if not water_data.empty:
                    current_water_level = water_data['water_level'].mean()
                    
                    # 优化: 更精确的水位阈值判断
                    if current_water_level < 0.8 and pd.notna(row['开闸孔数']) and row['开闸孔数'] > 0:
                        negative_sample = row.copy()
                        negative_sample['调度信息'] = f"无效调令: 当前水位({current_water_level:.2f}m)过低，不适合开闸"
                        negative_sample['有效调令'] = 0
                        negative_sample['开闸孔数'] = 0
                        negative_sample['开闸时间'] = None
                        negative_sample['关闸时间'] = None
                        negative_sample['开闸时长'] = 0
                        samples.append(negative_sample)
                    
                    # 如果水位过高但还继续开闸
                    if current_water_level > 3.5 and pd.notna(row['开闸孔数']) and row['开闸孔数'] > 5:
                        negative_sample = row.copy()
                        negative_sample['调度信息'] = f"无效调令: 当前水位({current_water_level:.2f}m)过高，不适合大幅开闸"
                        negative_sample['有效调令'] = 0
                        negative_sample['开闸孔数'] = max(0, row['开闸孔数'] - random.randint(3, 7))
                        samples.append(negative_sample)
                    
                    # 如果目标水位设置不合理
                    if "目标水位" in str(row['目标水位']) and not "无最低水位限制" in str(row['目标水位']):
                        try:
                            target_level = float(row['目标水位'])
                            # 更精确的目标水位判断
                            if target_level > current_water_level + 1.5 or target_level < current_water_level - 1.0:
                                negative_sample = row.copy()
                                negative_sample['目标水位'] = target_level
                                negative_sample['调度信息'] = f"无效调令: 目标水位({target_level:.2f}m)与当前水位({current_water_level:.2f}m)差异过大"
                                negative_sample['有效调令'] = 0
                                samples.append(negative_sample)
                        except:
                            pass
        
        return samples
    
    # 策略2: 基于流量和降雨数据的无效调令 - 优化版
    def generate_flow_rain_based_negative_samples():
        samples = []
        for idx, row in df.iterrows():
            if pd.notna(row['SIGNTM']):
                sign_time = row['SIGNTM']
                
                # 获取近期流量数据 (延长到12小时)
                flow_data = flow_df[
                    (flow_df['监测日期'] >= sign_time - timedelta(hours=12)) &
                    (flow_df['监测日期'] <= sign_time)
                ]
                
                # 获取近期降雨数据 (延长到24小时)
                rain_data = rainfall_df[
                    (rainfall_df['监测日期'] >= sign_time - timedelta(hours=24)) &
                    (rainfall_df['监测日期'] <= sign_time)
                ]
                
                # 如果流量已经很大但还增加开闸
                if not flow_data.empty:
                    avg_flow = flow_data['流量'].mean()
                    # 更精确的流量阈值
                    if avg_flow > 25.0 and pd.notna(row['开闸孔数']) and row['开闸孔数'] > 8:
                        negative_sample = row.copy()
                        negative_sample['调度信息'] = f"无效调令: 当前流量({avg_flow:.2f})已较大，不适合大幅增加开闸"
                        negative_sample['有效调令'] = 0
                        negative_sample['开闸孔数'] = max(0, row['开闸孔数'] - random.randint(3, 6))
                        samples.append(negative_sample)
                
                # 如果有强降雨预报但开闸不足
                if not rain_data.empty:
                    max_rain = rain_data['雨量'].max()
                    forecast_data = forecast_df[
                        (forecast_df['时间'] >= sign_time) &
                        (forecast_df['时间'] <= sign_time + timedelta(hours=24))
                    ]
                    
                    if not forecast_data.empty:
                        forecast_rain = forecast_data['降雨量'].mean()
                        
                        # 更精确的降雨阈值
                        if (max_rain > 8.0 or forecast_rain > 12.0) and pd.notna(row['开闸孔数']) and row['开闸孔数'] < 10:
                            negative_sample = row.copy()
                            negative_sample['调度信息'] = f"无效调令: 强降雨条件下({max_rain:.1f}mm)，开闸孔数不足"
                            negative_sample['有效调令'] = 0
                            negative_sample['开闸孔数'] = min(28, row['开闸孔数'] + random.randint(5, 8))
                            samples.append(negative_sample)
        
        return samples
    
    # 策略3: 基于潮汐时间和业务规则的无效调令 - 优化版
    def generate_tide_based_negative_samples():
        samples = []
        for idx, row in df.iterrows():
            if pd.notna(row['预计涨潮时间']):
                tide_time = row['预计涨潮时间']
                
                # 生成在涨潮时段发布的调令 (更精确的时间窗口)
                sign_time = tide_time + timedelta(hours=random.uniform(-0.5, 0.5))
                
                negative_sample = row.copy()
                negative_sample['SIGNTM'] = sign_time
                negative_sample['调度信息'] = f"无效调令: 在涨潮时段发布调令"
                negative_sample['开闸时间'] = None
                negative_sample['关闸时间'] = None
                negative_sample['开闸时长'] = None
                negative_sample['开闸孔数'] = 0
                negative_sample['有效调令'] = 0
                
                samples.append(negative_sample)
                
                # 生成开闸时间与涨潮时间冲突的调令
                if pd.notna(row['开闸时间']):
                    try:
                        # 使开闸时间接近涨潮时间（冲突）
                        conflict_open_time = tide_time + timedelta(hours=random.uniform(-0.3, 0.3))
                        
                        negative_sample = row.copy()
                        negative_sample['开闸时间'] = conflict_open_time.strftime("%H:%M")
                        negative_sample['调度信息'] = f"无效调令: 开闸时间与涨潮时间冲突"
                        negative_sample['有效调令'] = 0
                        
                        samples.append(negative_sample)
                    except:
                        pass
            else:
                # 对于没有预计涨潮时间的记录，也生成一些基于时间的负样本
                if random.random() < 0.2:  # 降低概率到20%
                    # 更合理的非退潮时段判断
                    sign_time = row['SIGNTM'] + timedelta(hours=random.uniform(3, 6))
                    
                    negative_sample = row.copy()
                    negative_sample['SIGNTM'] = sign_time
                    negative_sample['调度信息'] = f"无效调令: 在非退潮时段发布调令"
                    negative_sample['开闸时间'] = None
                    negative_sample['关闸时间'] = None
                    negative_sample['开闸时长'] = None
                    negative_sample['开闸孔数'] = 0
                    negative_sample['有效调令'] = 0
                    
                    samples.append(negative_sample)
        
        return samples
    
    # 策略4: 基于历史经验的不合理调令（一天开闸次数过多） - 优化版
    def generate_frequency_based_negative_samples():
        samples = []
        # 按日期分组，计算每天的开闸次数
        daily_ops = df.groupby(df['SIGNTM'].dt.date).size()
        
        for idx, row in df.iterrows():
            date = row['SIGNTM'].date()
            if date in daily_ops.index and daily_ops[date] >= 3:  # 提高阈值到3次
                # 如果这天已经有很多操作，再添加操作可能不合理
                if random.random() < 0.4:  # 降低概率到40%
                    negative_sample = row.copy()
                    negative_sample['调度信息'] = f"无效调令: 单日操作次数过多(已{daily_ops[date]}次)"
                    negative_sample['有效调令'] = 0
                    negative_sample['开闸孔数'] = 0
                    negative_sample['开闸时间'] = None
                    negative_sample['关闸时间'] = None
                    negative_sample['开闸时长'] = 0
                    samples.append(negative_sample)
        
        return samples
    
    # 策略5: 参数组合不合理的调令 - 优化版
    def generate_parameter_based_negative_samples():
        samples = []
        for idx, row in df.iterrows():
            if pd.notna(row['开闸孔数']) and pd.notna(row['开闸时长']):
                # 开闸孔数过多但时长过短
                if row['开闸孔数'] > 15 and row['开闸时长'] < 1.5:  # 调整阈值
                    negative_sample = row.copy()
                    negative_sample['开闸孔数'] = row['开闸孔数']
                    negative_sample['开闸时长'] = row['开闸时长'] * random.uniform(0.6, 0.8)
                    negative_sample['调度信息'] = f"无效调令: 开闸孔数多但时长过短"
                    negative_sample['有效调令'] = 0
                    samples.append(negative_sample)
                
                # 开闸孔数过少但时长过长
                if row['开闸孔数'] < 6 and row['开闸时长'] > 4.0:  # 调整阈值
                    negative_sample = row.copy()
                    negative_sample['开闸孔数'] = max(1, int(row['开闸孔数'] * random.uniform(0.6, 0.8)))
                    negative_sample['开闸时长'] = row['开闸时长']
                    negative_sample['调度信息'] = f"无效调令: 开闸孔数少但时长过长"
                    negative_sample['有效调令'] = 0
                    samples.append(negative_sample)
            
            # 目标水位不合理
            if "目标水位" in str(row['目标水位']) and not "无最低水位限制" in str(row['目标水位']):
                try:
                    target_level = float(row['目标水位'])
                    # 目标水位过高或过低
                    if target_level > 4.0 or target_level < 1.2:  # 调整阈值
                        negative_sample = row.copy()
                        negative_sample['目标水位'] = target_level
                        negative_sample['调度信息'] = f"无效调令: 目标水位({target_level:.2f}m)超出合理范围"
                        negative_sample['有效调令'] = 0
                        samples.append(negative_sample)
                except:
                    pass
        
        return samples
    
    # 策略6: 基于水位环境的不开闸样本 - 优化版
    def generate_no_operation_samples():
        samples = []
        # 获取所有正样本的时间点
        operation_times = df['SIGNTM'].tolist()
        
        # 从水位数据中随机选择一些时间点作为不开闸样本
        water_times = water_level_df['time'].unique()
        
        # 过滤出不在操作时间附近的时间点
        non_operation_times = []
        for wt in water_times:
            # 检查这个时间点是否接近任何操作时间
            is_near_operation = any(
                abs((pd.Timestamp(wt) - ot).total_seconds()) < 4 * 3600 for ot in operation_times  # 缩短到4小时
            )
            if not is_near_operation and random.random() < 0.15:  # 提高到15%的概率
                non_operation_times.append(wt)
        
        # 为每个选中的时间点创建不开闸样本
        for time_point in non_operation_times[:300]:  # 减少到最多300个不开闸样本
            # 获取此时的水位数据
            water_data = water_level_df[water_level_df['time'] == time_point]
            if not water_data.empty:
                water_level = water_data['water_level'].values[0]
                
                # 创建不开闸样本 - 使用与原始数据相同的结构
                sample = pd.Series({
                    'SIGNTM': time_point,
                    '日期': pd.Timestamp(time_point).normalize(),
                    '调度信息': f"无操作: 当前水位({water_level:.2f}m)适宜，无需开闸",
                    '开闸时间': None,
                    '关闸时间': None,
                    '开闸时长': 0,
                    '开闸孔数': 0,
                    '目标水位': "无操作",
                    '预计涨潮时间': None,
                    '有效调令': 0
                })
                
                samples.append(sample)
        
        return samples
    
    # 新增策略7: 基于季节和时间的不合理调令
    def generate_seasonal_negative_samples():
        samples = []
        for idx, row in df.iterrows():
            if pd.notna(row['SIGNTM']):
                month = row['SIGNTM'].month
                hour = row['SIGNTM'].hour
                
                # 冬季深夜开闸不合理
                if month in [12, 1, 2] and (hour < 6 or hour > 22):
                    if random.random() < 0.3:
                        negative_sample = row.copy()
                        negative_sample['调度信息'] = f"无效调令: 冬季深夜开闸不合理"
                        negative_sample['有效调令'] = 0
                        negative_sample['开闸孔数'] = 0
                        negative_sample['开闸时间'] = None
                        negative_sample['关闸时间'] = None
                        negative_sample['开闸时长'] = 0
                        samples.append(negative_sample)
                
                # 雨季过度开闸
                if month in [6, 7, 8, 9] and pd.notna(row['开闸孔数']) and row['开闸孔数'] > 20:
                    if random.random() < 0.4:
                        negative_sample = row.copy()
                        negative_sample['调度信息'] = f"无效调令: 雨季过度开闸可能引发洪水风险"
                        negative_sample['有效调令'] = 0
                        negative_sample['开闸孔数'] = max(0, row['开闸孔数'] - random.randint(5, 10))
                        samples.append(negative_sample)
        
        return samples
    
    # 生成所有类型的负样本
    print("生成基于水位的负样本...")
    negative_samples.extend(generate_water_based_negative_samples())
    
    print("生成基于流量和降雨的负样本...")
    negative_samples.extend(generate_flow_rain_based_negative_samples())
    
    print("生成基于潮汐时间的负样本...")
    negative_samples.extend(generate_tide_based_negative_samples())
    
    print("生成基于操作频率的负样本...")
    negative_samples.extend(generate_frequency_based_negative_samples())
    
    print("生成基于参数组合的负样本...")
    negative_samples.extend(generate_parameter_based_negative_samples())
    
    print("生成基于水位环境的不开闸样本...")
    negative_samples.extend(generate_no_operation_samples())
    
    print("生成基于季节时间的负样本...")
    negative_samples.extend(generate_seasonal_negative_samples())
    
    # 创建负样本DataFrame
    negative_df = pd.DataFrame(negative_samples)
    
    # 添加标签列
    df['有效调令'] = 1  # 正样本
    if not negative_df.empty:
        negative_df['有效调令'] = 0  # 负样本
    
    # 合并正负样本
    if not negative_df.empty:
        full_dataset = pd.concat([df, negative_df], ignore_index=True)
    else:
        full_dataset = df.copy()
    
    # 确保数据平衡
    positive_count = full_dataset['有效调令'].sum()
    negative_count = len(full_dataset) - positive_count
    
    print(f"正样本数量: {positive_count}")
    print(f"负样本数量: {negative_count}")
    print(f"正负样本比例: {positive_count/len(full_dataset):.3f}")
    
    # 如果负样本过多，进行下采样
    if negative_count > positive_count * 1.5:
        print("进行负样本下采样以平衡数据集...")
        negative_indices = full_dataset[full_dataset['有效调令'] == 0].index
        keep_negative_count = int(positive_count * 1.2)  # 保持1.2:1的比例
        remove_n = len(negative_indices) - keep_negative_count
        
        if remove_n > 0:
            drop_indices = np.random.choice(negative_indices, remove_n, replace=False)
            full_dataset = full_dataset.drop(drop_indices)
            
            positive_count = full_dataset['有效调令'].sum()
            negative_count = len(full_dataset) - positive_count
            print(f"下采样后正样本数量: {positive_count}")
            print(f"下采样后负样本数量: {negative_count}")
            print(f"下采样后正负样本比例: {positive_count/len(full_dataset):.3f}")
    
    # 保存结果
    full_dataset.to_csv('imports/调令数据_增强正负样本.csv', index=False, encoding='utf-8-sig')
    print("增强版正负样本数据集已保存")
    
    return full_dataset

# 在原有的train_data_analysis函数中替换负样本生成部分
def train_data_analysis():
    """
    调令数据分析 - 使用增强的负样本生成
    """
    # 使用增强的负样本生成
    full_dataset = enhanced_negative_sample_generation()
    
    # 这里可以添加进一步的分析代码
    print("数据集基本信息:")
    print(full_dataset.info())
    print("\n有效调令分布:")
    print(full_dataset['有效调令'].value_counts())
    
    return full_dataset

if __name__ == "__main__":
    train_data_analysis()