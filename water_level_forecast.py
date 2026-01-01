import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 设置字体，确保中文正常显示
plt.rcParams["font.family"] = ["WenQuanYi Zen Hei"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
USE_CHINESE = True

# 1. 数据加载与预处理（优化版）- 增加数据补齐功能
def load_and_preprocess(data_path, file_pattern):
    """加载并预处理多个潮位数据文件，确保时间序列连续完整"""
    files = glob.glob(f"{data_path}/**/{file_pattern}", recursive=True)
    if not files:
        raise FileNotFoundError(f"未找到匹配的文件: {file_pattern}")
    
    df_list = []
    for f in files:
        temp_df = pd.read_csv(f, header=None, names=['station_id', 'time', 'water_level'])
        df_list.append(temp_df)
    
    full_df = pd.concat(df_list, ignore_index=True)
    
    # 转换时间格式并排序
    full_df['time'] = pd.to_datetime(full_df['time'])
    full_df.sort_values(['station_id', 'time'], inplace=True)
    
    # 数据去重
    full_df = full_df.drop_duplicates(subset=['station_id', 'time'])
    
    # 新增：补齐缺失时间点
    resampled_dfs = []
    for station_id, group in full_df.groupby('station_id'):
        # 设置时间索引
        group = group.set_index('time')
        # 创建完整时间序列（每小时一个点）
        full_time_range = pd.date_range(
            start=group.index.min(),
            end=group.index.max(),
            freq='H'
        )
        # 重新采样并线性插值补齐
        group_resampled = group.reindex(full_time_range)
        group_resampled['water_level'] = group_resampled['water_level'].interpolate(method='linear', limit_direction='both')
        # 填充站点ID
        group_resampled['station_id'] = group_resampled['station_id'].fillna(station_id)
        resampled_dfs.append(group_resampled.reset_index().rename(columns={'index': 'time'}))
    
    full_df = pd.concat(resampled_dfs, ignore_index=True)
    
    # 添加时间特征
    full_df['hour'] = full_df['time'].dt.hour
    full_df['day'] = full_df['time'].dt.day
    full_df['month'] = full_df['time'].dt.month
    full_df['dayofweek'] = full_df['time'].dt.dayofweek
    
    return full_df

# 2. 特征工程函数（保持不变）
def create_features(df, station_id=3018, lag_hours=72):
    """为机器学习模型创建特征"""
    station_df = df[df['station_id'] == station_id].copy()
    station_df.set_index('time', inplace=True)
    
    # 创建滞后特征
    for i in range(1, lag_hours+1):
        station_df[f'lag_{i}'] = station_df['water_level'].shift(i)
    
    # 添加周期性特征
    station_df['hour_sin'] = np.sin(2 * np.pi * station_df['hour']/24)
    station_df['hour_cos'] = np.cos(2 * np.pi * station_df['hour']/24)
    
    # 添加月份特征
    station_df['month_sin'] = np.sin(2 * np.pi * (station_df['month']-1)/12)
    station_df['month_cos'] = np.cos(2 * np.pi * (station_df['month']-1)/12)
    
    # 删除包含NaN的行
    station_df.dropna(inplace=True)
    
    return station_df

# 3. 数据标准化（保持不变）
def scale_data(df, feature_columns, target_column):
    """标准化特征数据"""
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    
    X_scaled = scaler_X.fit_transform(df[feature_columns])
    y_scaled = scaler_y.fit_transform(df[[target_column]])
    
    return X_scaled, y_scaled, scaler_X, scaler_y

# 4. 创建LSTM输入数据（保持不变）
def create_lstm_dataset(data, n_steps):
    """将时间序列数据转换为LSTM输入格式"""
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:(i + n_steps), :])
        y.append(data[i + n_steps, -1])  # 预测最后一个特征（水位）
    return np.array(X), np.array(y)

# 5. 构建LSTM模型（保持不变）
def build_lstm_model(input_shape):
    """构建LSTM神经网络模型"""
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mse')
    return model

# 6. 预测函数（保持不变）
def predict_future(model, last_sequence, scaler_X, scaler_y, feature_columns, steps=24):
    """进行多步滚动预测"""
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(steps):
        # 预测下一步
        prediction = model.predict(current_sequence.reshape(1, current_sequence.shape[0], current_sequence.shape[1]))
        
        # 反归一化预测结果
        dummy_array = np.zeros((1, len(feature_columns)))
        dummy_array[0, -1] = prediction[0, 0]  # 水位值在最后一列
        water_level = scaler_y.inverse_transform(dummy_array[:, -1].reshape(-1, 1))[0, 0]
        predictions.append(water_level)
        
        # 更新序列：移除最旧的数据点，添加预测值
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1, -1] = prediction
        
        # 更新时间特征（如果需要预测更长时间）
        if steps > 24:
            last_time = pd.to_datetime(last_sequence.index[-1])
            new_time = last_time + pd.Timedelta(hours=1)
            # 更新新时间点的时间特征（此处简化处理）
    
    return predictions

# 7. 主函数（保持不变）
def main():
    # 加载数据
    data_path = "exports"
    file_pattern = "*闸下潮位.csv"
    station_id = 3018
    
    print("开始加载数据...")
    df = load_and_preprocess(data_path, file_pattern)
    print(f"成功加载 {len(df)} 条记录")
    
    # 特征工程
    print("创建特征...")
    feature_df = create_features(df, station_id, lag_hours=72)
    
    # 定义特征列和目标列
    feature_columns = [col for col in feature_df.columns if col not in ['station_id', 'water_level', 'day', 'month', 'hour']]
    target_column = 'water_level'
    
    # 数据标准化
    print("标准化数据...")
    X, y, scaler_X, scaler_y = scale_data(feature_df, feature_columns, target_column)
    
    # 创建LSTM数据集
    n_steps = 24  # 使用24小时数据预测未来
    X_lstm, y_lstm = create_lstm_dataset(np.hstack((X, y)), n_steps)
    
    # 划分训练集和测试集（最后10%作为测试）
    split_idx = int(len(X_lstm) * 0.9)
    X_train, X_test = X_lstm[:split_idx], X_lstm[split_idx:]
    y_train, y_test = y_lstm[:split_idx], y_lstm[split_idx:]
    
    print(f"训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}")
    
    # 构建LSTM模型
    print("构建LSTM模型...")
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    
    # 训练模型
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )
    
    # 评估模型
    print("评估模型...")
    y_pred_scaled = model.predict(X_test)
    
    # 反归一化
    dummy_array = np.zeros((len(y_pred_scaled), len(feature_columns)))
    dummy_array[:, -1] = y_pred_scaled.flatten()
    y_pred = scaler_y.inverse_transform(dummy_array[:, -1].reshape(-1, 1)).flatten()
    
    dummy_array[:, -1] = y_test.flatten()
    y_true = scaler_y.inverse_transform(dummy_array[:, -1].reshape(-1, 1)).flatten()
    
    # 计算误差
    mae = mean_absolute_error(y_true, y_pred)
    max_error = np.max(np.abs(y_true - y_pred))
    print(f"测试集MAE: {mae:.4f}米, 最大误差: {max_error:.4f}米")
    
    # 检查是否满足0.3米误差要求
    if max_error <= 0.3:
        print("✅ 模型满足0.3米误差要求")
    else:
        print("❌ 模型未满足0.3米误差要求，需要进一步优化")
    
    # 可视化结果
    plt.figure(figsize=(14, 6))
    plt.plot(feature_df.index[-len(y_true):], y_true, 'b-', label='实际水位')
    plt.plot(feature_df.index[-len(y_pred):], y_pred, 'r--', label='预测水位')
    plt.fill_between(feature_df.index[-len(y_pred):], 
                    y_pred - 0.3, 
                    y_pred + 0.3, 
                    color='gray', alpha=0.2, label='误差范围(0.3m)')
    plt.title(f'站点 {station_id} 水位预测结果 (MAE: {mae:.3f}m)')
    plt.xlabel('时间')
    plt.ylabel('水位 (m)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('models/water_level_forecast.png', dpi=300)
    plt.show()
    
    # 预测未来24小时
    print("预测未来24小时水位...")
    last_sequence = np.hstack((X, y))[-n_steps:]
    future_predictions = predict_future(model, last_sequence, scaler_X, scaler_y, feature_columns, steps=24)
    
    # 生成未来时间点
    last_time = feature_df.index[-1]
    future_times = [last_time + pd.Timedelta(hours=i+1) for i in range(24)]
    
    # 可视化预测结果
    plt.figure(figsize=(12, 6))
    plt.plot(feature_df.index[-72:], feature_df['water_level'][-72:], 'b-o', label='历史水位')
    plt.plot(future_times, future_predictions, 'r--o', label='预测水位')
    plt.axvline(last_time, color='gray', linestyle='--')
    plt.title(f'站点 {station_id} 未来24小时水位预测')
    plt.xlabel('时间')
    plt.ylabel('水位 (m)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('models/future_water_level_forecast.png', dpi=300)
    plt.show()
    
    # 保存预测结果
    forecast_df = pd.DataFrame({
        'time': future_times,
        'predicted_water_level': future_predictions
    })
    forecast_df.to_csv('models/water_level_forecast.csv', index=False)
    print("预测结果已保存到 water_level_forecast.csv")

if __name__ == "__main__":
    main()