import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, LSTM, Dense, Dropout, 
                                    ConvLSTM2D, Reshape, Multiply,
                                    Add, Activation, Flatten)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from pytides.tide import Tide
from datetime import datetime, timedelta
import joblib

# 天文潮汐生成器
class AstronomicalTideGenerator:
    def __init__(self, constituents, lat=30.0, lon=120.0):
        self.constituents = constituents
        self.lat = lat
        self.lon = lon
        
    def generate_tide(self, start_date, end_date, freq='H'):
        """生成指定时间范围内的天文潮汐"""
        # 创建潮汐模型
        tide = Tide(model=self.constituents, radians=False)
        
        # 生成时间序列
        times = pd.date_range(start=start_date, end=end_date, freq=freq)
        
        # 计算潮汐高度
        heights = []
        for t in times:
            # 计算每个时间点的天文潮汐
            h = tide.at(t)
            # 考虑地理位置影响
            lat_factor = np.cos(np.radians(self.lat))
            lon_factor = 0.5 * (1 + np.cos(np.radians(self.lon - 120)))
            h_adjusted = h * lat_factor * lon_factor
            heights.append(h_adjusted)
            
        return pd.Series(heights, index=times)

# 空间注意力模块
def spatial_attention_block(input_tensor):
    """空间注意力机制"""
    # 创建注意力权重
    attention = ConvLSTM2D(filters=1, kernel_size=(3, 3), 
                          padding='same', activation='sigmoid')(input_tensor)
    
    # 应用注意力权重
    weighted = Multiply()([input_tensor, attention])
    
    # 残差连接
    output = Add()([input_tensor, weighted])
    return Activation('relu')(output)

# 1. 数据加载与预处理（增强版）
def load_and_preprocess(data_path, file_pattern, tide_generator=None):
    """加载并预处理数据，集成天文潮汐特征"""
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
    
    # 添加时间特征
    full_df['hour'] = full_df['time'].dt.hour
    full_df['day'] = full_df['time'].dt.day
    full_df['month'] = full_df['time'].dt.month
    full_df['dayofweek'] = full_df['time'].dt.dayofweek
    full_df['dayofyear'] = full_df['time'].dt.dayofyear
    
    # 添加天文潮汐特征
    if tide_generator:
        min_time = full_df['time'].min()
        max_time = full_df['time'].max()
        tide_series = tide_generator.generate_tide(min_time, max_time)
        full_df = full_df.merge(tide_series.rename('astronomical_tide'), 
                               left_on='time', right_index=True, how='left')
        # 填充可能的缺失值
        full_df['astronomical_tide'] = full_df['astronomical_tide'].fillna(method='ffill')
    else:
        full_df['astronomical_tide'] = 0.0
    
    return full_df

# 2. 特征工程（增强版）
def create_features(df, station_id=3018, lag_hours=72, n_stations=2):
    """为模型创建特征，包含多站点信息"""
    # 获取目标站点数据
    target_df = df[df['station_id'] == station_id].copy()
    target_df.set_index('time', inplace=True)
    
    # 添加邻近站点数据作为特征
    other_stations = df[df['station_id'] != station_id]['station_id'].unique()[:n_stations-1]
    for i, sid in enumerate(other_stations):
        station_df = df[df['station_id'] == sid][['time', 'water_level']]
        station_df = station_df.rename(columns={'water_level': f'station_{sid}'})
        station_df.set_index('time', inplace=True)
        target_df = target_df.join(station_df, how='left')
    
    # 填充邻近站点缺失值
    for col in target_df.columns:
        if col.startswith('station_'):
            target_df[col] = target_df[col].fillna(method='ffill')
    
    # 创建滞后特征
    features_to_lag = ['water_level', 'astronomical_tide'] + \
                     [col for col in target_df.columns if col.startswith('station_')]
    
    for feature in features_to_lag:
        for i in range(1, lag_hours+1):
            target_df[f'{feature}_lag_{i}'] = target_df[feature].shift(i)
    
    # 添加周期性特征
    target_df['hour_sin'] = np.sin(2 * np.pi * target_df['hour']/24)
    target_df['hour_cos'] = np.cos(2 * np.pi * target_df['hour']/24)
    target_df['month_sin'] = np.sin(2 * np.pi * (target_df['month']-1)/12)
    target_df['month_cos'] = np.cos(2 * np.pi * (target_df['month']-1)/12)
    target_df['doy_sin'] = np.sin(2 * np.pi * target_df['dayofyear']/365)
    target_df['doy_cos'] = np.cos(2 * np.pi * target_df['dayofyear']/365)
    
    # 删除包含NaN的行
    target_df.dropna(inplace=True)
    
    return target_df

# 3. 数据标准化
def scale_data(df, feature_columns, target_column):
    """标准化特征数据"""
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    
    X_scaled = scaler_X.fit_transform(df[feature_columns])
    y_scaled = scaler_y.fit_transform(df[[target_column]])
    
    return X_scaled, y_scaled, scaler_X, scaler_y

# 4. 创建ConvLSTM输入数据
def create_convlstm_dataset(data, n_steps, n_features, spatial_dims=(1, 1)):
    """将数据转换为ConvLSTM输入格式"""
    X, y = [], []
    for i in range(len(data) - n_steps):
        # 重塑为ConvLSTM输入格式 [samples, timesteps, rows, cols, features]
        seq = data[i:(i + n_steps), :]
        seq_reshaped = seq.reshape((n_steps, spatial_dims[0], spatial_dims[1], n_features))
        X.append(seq_reshaped)
        y.append(data[i + n_steps, -1])  # 预测最后一个特征（水位）
    return np.array(X), np.array(y)

# 5. 构建时空注意力模型
def build_spatiotemporal_model(input_shape, spatial_dims=(1, 1)):
    """构建包含空间注意力机制的ConvLSTM模型"""
    inputs = Input(shape=input_shape)
    
    # 空间特征提取
    convlstm1 = ConvLSTM2D(filters=32, kernel_size=(3, 3), 
                          padding='same', return_sequences=True)(inputs)
    convlstm1 = Dropout(0.2)(convlstm1)
    
    # 空间注意力模块
    attention = spatial_attention_block(convlstm1)
    
    # 时间特征提取
    convlstm2 = ConvLSTM2D(filters=64, kernel_size=(3, 3), 
                          padding='same', return_sequences=False)(attention)
    convlstm2 = Dropout(0.2)(convlstm2)
    
    # 展平并添加全连接层
    flattened = Flatten()(convlstm2)
    dense1 = Dense(128, activation='relu')(flattened)
    dense2 = Dense(64, activation='relu')(dense1)
    output = Dense(1)(dense2)
    
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# 6. 预测函数（增强版）
def predict_future(model, last_sequence, scaler_X, scaler_y, feature_columns, 
                  spatial_dims, n_features, steps=24):
    """进行多步滚动预测"""
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(steps):
        # 重塑输入格式
        input_seq = current_sequence.reshape(1, current_sequence.shape[0], 
                                           spatial_dims[0], spatial_dims[1], n_features)
        
        # 预测下一步
        prediction = model.predict(input_seq)
        
        # 反归一化预测结果
        dummy_array = np.zeros((1, len(feature_columns)))
        dummy_array[0, -1] = prediction[0, 0]  # 水位值在最后一列
        water_level = scaler_y.inverse_transform(dummy_array[:, -1].reshape(-1, 1))[0, 0]
        predictions.append(water_level)
        
        # 更新序列：移除最旧的数据点，添加预测值
        current_sequence = np.roll(current_sequence, -1, axis=0)
        
        # 更新新数据点（只更新水位特征）
        current_sequence[-1, -1] = prediction
        
        # 更新时间特征（简化处理）
        # 在实际应用中，这里应该更新时间相关特征
        
    return predictions

# 7. 主函数（增强版）
def main():
    # 初始化天文潮汐生成器（使用主要潮汐成分）
    constituents = ['M2', 'S2', 'N2', 'K1', 'O1']
    tide_generator = AstronomicalTideGenerator(constituents, lat=30.0, lon=120.0)
    
    # 加载数据
    data_path = "exports"
    file_pattern = "*闸下潮位.csv"
    station_id = 3018
    
    print("开始加载数据...")
    df = load_and_preprocess(data_path, file_pattern, tide_generator)
    print(f"成功加载 {len(df)} 条记录")
    
    # 特征工程
    print("创建特征...")
    feature_df = create_features(df, station_id, lag_hours=72, n_stations=2)
    
    # 定义特征列和目标列
    feature_columns = [col for col in feature_df.columns 
                      if col not in ['station_id', 'water_level', 'day', 'month', 'hour', 'dayofyear']]
    target_column = 'water_level'
    
    # 数据标准化
    print("标准化数据...")
    X, y, scaler_X, scaler_y = scale_data(feature_df, feature_columns, target_column)
    
    # 设置ConvLSTM参数
    n_steps = 24  # 使用24小时数据预测未来
    spatial_dims = (1, 1)  # 空间维度（简化为1x1）
    n_features = len(feature_columns) // (spatial_dims[0] * spatial_dims[1])
    
    # 创建ConvLSTM数据集
    X_convlstm, y_convlstm = create_convlstm_dataset(
        np.hstack((X, y)), n_steps, n_features, spatial_dims
    )
    
    # 划分训练集和测试集
    split_idx = int(len(X_convlstm) * 0.9)
    X_train, X_test = X_convlstm[:split_idx], X_convlstm[split_idx:]
    y_train, y_test = y_convlstm[:split_idx], y_convlstm[split_idx:]
    
    print(f"训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}")
    
    # 构建时空注意力模型
    print("构建时空注意力模型...")
    input_shape = (n_steps, spatial_dims[0], spatial_dims[1], n_features)
    model = build_spatiotemporal_model(input_shape, spatial_dims)
    
    # 训练模型
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
    
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop, reduce_lr],
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
    plt.plot(y_true, 'b-', label='实际水位')
    plt.plot(y_pred, 'r--', label='预测水位')
    plt.fill_between(range(len(y_pred)), 
                    y_pred - 0.3, 
                    y_pred + 0.3, 
                    color='gray', alpha=0.2, label='误差范围(0.3m)')
    plt.title(f'站点 {station_id} 水位预测结果 (MAE: {mae:.3f}m, 最大误差: {max_error:.3f}m)')
    plt.xlabel('时间步')
    plt.ylabel('水位 (m)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('models/pywater_level_forecast_enhanced.png', dpi=300)
    plt.show()
    
    # 预测未来24小时
    print("预测未来24小时水位...")
    last_sequence = np.hstack((X, y))[-n_steps:]
    future_predictions = predict_future(
        model, last_sequence, scaler_X, scaler_y, feature_columns,
        spatial_dims, n_features, steps=24
    )
    
    # 生成未来时间点
    last_time = feature_df.index[-1]
    future_times = [last_time + timedelta(hours=i+1) for i in range(24)]
    
    # 可视化预测结果
    plt.figure(figsize=(12, 6))
    
    # 历史数据（最后72小时）
    history_hours = 72
    plt.plot(feature_df.index[-history_hours:], 
             feature_df['water_level'][-history_hours:], 
             'b-o', label='历史水位')
    
    # 预测数据
    plt.plot(future_times, future_predictions, 'r--o', label='预测水位')
    
    # 添加天文潮汐参考
    if 'astronomical_tide' in feature_df.columns:
        plt.plot(feature_df.index[-history_hours:], 
                 feature_df['astronomical_tide'][-history_hours:], 
                 'g:', alpha=0.7, label='天文潮汐')
    
    plt.axvline(last_time, color='gray', linestyle='--')
    plt.title(f'站点 {station_id} 未来24小时水位预测')
    plt.xlabel('时间')
    plt.ylabel('水位 (m)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('models/pyfuture_water_level_forecast_enhanced.png', dpi=300)
    plt.show()
    
    # 保存预测结果
    forecast_df = pd.DataFrame({
        'time': future_times,
        'predicted_water_level': future_predictions
    })
    forecast_df.to_csv('models/pywater_level_forecast_enhanced.csv', index=False)
    
    # 保存模型和标准化器
    model.save('models/pywater_level_forecast_model.h5')
    joblib.dump(scaler_X, 'scaler_X.pkl')
    joblib.dump(scaler_y, 'scaler_y.pkl')
    print("模型和预测结果已保存")

if __name__ == "__main__":
    main()