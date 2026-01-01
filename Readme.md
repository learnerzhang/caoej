已知已下数据格式，要求预测   开闸时间	开闸时长*开闸孔数	目标水位

已知数据格式如下：

1. 闸下水位csv文件，包含以下字段：
   - station_id：测站编号
   - time：时间
   - water_level：水位

2. 实测流量csv文件，包含以下字段：
   - 测点编码：测点编号
   - 监测日期：时间
   - 流量：流量值

3. 实测降雨csv文件，包含以下字段：
   - 测点编码：测点编号
   - 监测日期：时间
   - 雨量：雨量值

4. 降雨预报csv文件，包含以下字段：
   - 经度：经度
   - 维度：纬度
   - 时间：时间
   - 雨量：雨量值
   - 所属区域：所属区域
5. 水位工况csv文件，包含以下字段：
    测点编码	
    监测日期	
    水位


问题：
1）模型的预测->业务上的处理
   时长：7小时
   目标水位 1m

2）预测开闸时长*开孔数量 分配原则  
   LABEL_TO_GATES = {1: 4, 2: 8, 3: 10, 4: 12, 5: 18, 6: 28}
   28孔，全开
   落潮60%的时间  落7-8小时(6个小时窗口期)， 当前水位->目标水位差值，建立时间区间关联性


3）多模型如何组织
   模型x -> 多值

4) 不开闸样本，根据历史经验操作，一天开2次，考虑当时实际水位环境



闸下水位csv示例

	station_id	time	water_level
0	3018	2021-01-11 10:00:00	3.33
1	3018	2021-01-11 11:00:00	4.10
2	3018	2021-01-11 12:00:00	4.42
3	3018	2021-01-11 13:00:00	3.82
4	3018	2021-01-11 14:00:00	2.90
5	3018	2021-01-11 15:00:00	1.81
6	3018	2021-01-11 16:00:00	0.86
7	3018	2021-01-11 17:00:00	0.13
8	3018	2021-01-11 18:00:00	-0.33
9	3018	2021-01-11 19:00:00	-0.72


实测流量csv示例

	测点编码	监测日期	流量
0	8164	2021-01-11 10:00:00	0.00
1	8164	2021-01-11 11:00:00	11.93
2	8164	2021-01-11 12:00:00	3.58
3	8164	2021-01-11 13:00:00	0.00
4	8164	2021-01-11 14:00:00	0.00
5	8164	2021-01-11 15:00:00	16.77
6	8164	2021-01-11 16:00:00	5.99
7	8164	2021-01-11 17:00:00	46.72
8	8164	2021-01-11 18:00:00	7.19
9	8164	2021-01-11 19:00:00	9.58

实测降雨csv示例


测点编码	监测日期	雨量	经度	维度	所属区域
0	5427	2021-01-11 09:10:00	0.0	120.615600	29.765100	虞南山区
1	5425	2021-01-11 09:10:00	0.0	NaN	NaN	NaN
2	5422	2021-01-11 09:10:00	0.0	120.741999	29.861333	虞南山区
3	1532	2021-01-11 09:10:00	0.0	120.500000	29.894700	绍兴平原
4	3642	2021-01-11 09:10:00	0.0	120.698655	30.276754	NaN
5	3228	2021-01-11 09:10:00	0.0	120.598589	29.615754	嵊州
6	3840	2021-01-11 09:10:00	0.0	120.844937	29.917734	虞北平原
7	2310	2021-01-11 09:10:00	0.0	120.927700	30.059000	虞北平原
8	1601	2021-01-11 09:10:00	0.0	121.190897	29.523922	嵊州
9	7308	2021-01-11 09:10:00	0.0	120.744015	30.232381	NaN


降雨预报csv示例
0	1	2	3	4	5	6
0	120.25	30.15	2025-04-23 14:00:00.0000002025-04-23 15:00:00....	0.0	NaN	NaN	绍兴平原
1	120.35	30.05	2025-04-23 14:00:00.0000002025-04-23 15:00:00....	0.0	绍兴市	柯桥区	绍兴平原
2	120.35	30.15	2025-04-23 14:00:00.0000002025-04-23 15:00:00....	0.0	绍兴市	柯桥区	绍兴平原
3	120.45	29.95	2025-04-23 14:00:00.0000002025-04-23 15:00:00....	0.1	绍兴市	柯桥区	绍兴平原
4	120.45	30.05	2025-04-23 14:00:00.0000002025-04-23 15:00:00....	0.0	绍兴市	柯桥区	绍兴平原
5	120.45	30.15	2025-04-23 14:00:00.0000002025-04-23 15:00:00....	0.0	绍兴市	柯桥区	绍兴平原
6	120.55	29.35	2025-04-23 14:00:00.0000002025-04-23 15:00:00....	0.0	NaN	NaN	嵊州
7	120.55	29.45	2025-04-23 14:00:00.0000002025-04-23 15:00:00....	0.0	绍兴市	嵊州市	嵊州
8	120.55	29.55	2025-04-23 14:00:00.0000002025-04-23 15:00:00....	0.0	绍兴市	嵊州市	嵊州
9	120.55	29.65	2025-04-23 14:00:00.0000002025-04-23 15:00:00....	0.0	绍兴市	嵊州市	嵊州

水位工况csv示例
	测点编码	监测日期	水位
413	1529	2021-01-11 10:00:00	4.09
414	1529	2021-01-11 11:00:00	4.08
415	1529	2021-01-11 12:00:00	4.08
416	1529	2021-01-11 13:00:00	4.08
417	1529	2021-01-11 14:00:00	4.08
418	1529	2021-01-11 15:00:00	4.08
419	1529	2021-01-11 16:00:00	4.09
420	1529	2021-01-11 17:00:00	4.09
421	1529	2021-01-11 18:00:00	4.09
422	1529	2021-01-11 19:00:00	4.10


调令csv示例
SIGNTM	日期	调度信息	开闸时间	关闸时间	开闸时长	开闸孔数	目标水位
0	2025-07-30 16:00:11	2025-07-30	07月30日傍晚退潮时段开启28孔闸门，到潮前关闭全部闸门，预计涨潮时间07-31 01:40	NaN	NaN	NaN	28	无最低水位限制
1	2025-07-30 05:55:49	2025-07-30	07月30日上午退潮时段开启28孔闸门，到潮前关闭全部闸门，预计涨潮时间07-30 13:45	07:15	13:10	NaN	28	无最低水位限制
2	2025-07-29 18:44:19	2025-07-29	07月29日傍晚退潮时段开启16孔闸门，07-30 00:30关闭全部闸门，预计涨潮时间07...	19:20	00:30	NaN	16	无最低水位限制
3	2025-07-29 05:54:14	2025-07-29	07月29日上午退潮时段开启20孔闸门，07-29 12:30关闭全部闸门，预计涨潮时间07...	06:30	12:30	NaN	20	无最低水位限制
4	2025-07-28 16:01:16	2025-07-28	07月28日傍晚退潮时段开启20孔闸门，07-29 00:00关闭全部闸门，预计涨潮时间07...	17:40	00:00	NaN	20	无最低水位限制
5	2025-07-28 07:08:13	2025-07-28	07月28日上午退潮时段开启12孔闸门，开闸时间3小时，预计涨潮时间07-28 13:00	08:00	11:00	3.0	12	2.5
6	2025-07-27 14:00:41	2025-07-27	07月27日下午退潮时段开启20孔闸门，开闸时间5小时，预计涨潮时间07-27 23:55	17:00	22:00	5.0	20	2.00
7	2025-07-25 14:57:03	2025-07-25	07月25日下午退潮时段开启14孔闸门，开闸时间3小时，预计涨潮时间07-25 22:50	15:40	18:40	3.0	14	3.00
8	2025-07-19 07:55:02	2025-07-19	07月19日上午退潮时段开启6孔闸门，开闸时间3小时，预计涨潮时间07-19 16:00	08:30	11:30	3.0	6	3.40
9	2025-07-18 20:32:49	2025-07-18	07月18日晚上退潮时段开启14孔闸门，开闸时间3小时，预计涨潮时间07-19 03:30	21:20	00:20	3.0	14	3.0






# 开闸时间预测
Fitting 5 folds for each of 6 candidates, totalling 30 fits
最佳参数: {'classifier__C': 1, 'classifier__penalty': 'l2'}
最佳交叉验证准确率: 0.0745

测试集准确率: 0.1608

precision    recall  f1-score   support

         0.0       0.00      0.00      0.00         2
         1.0       0.00      0.00      0.00         2
         2.0       0.00      0.00      0.00         2
         3.0       0.00      0.00      0.00         1
         4.0       0.00      0.00      0.00         2
         5.0       0.00      0.00      0.00         3
         6.0       0.00      0.00      0.00         5
         7.0       0.25      0.38      0.30         8
         8.0       0.00      0.00      0.00        10
         9.0       0.35      0.54      0.42        13
        10.0       0.00      0.00      0.00        10
        11.0       0.00      0.00      0.00         8
        12.0       0.00      0.00      0.00         9
        13.0       0.00      0.00      0.00        11
        14.0       0.00      0.00      0.00        12
        15.0       0.00      0.00      0.00        15
        16.0       0.16      0.64      0.25        25
        17.0       0.00      0.00      0.00        20
        18.0       0.13      0.33      0.19        18
        19.0       0.00      0.00      0.00        10
        20.0       0.00      0.00      0.00         6
        21.0       0.00      0.00      0.00         3
        22.0       0.00      0.00      0.00         3
        23.0       0.00      0.00      0.00         1

    accuracy                           0.16       199
   macro avg       0.04      0.08      0.05       199
weighted avg       0.06      0.16      0.09       199





### 模型训练

python -m models.kz_time.train

python -m models.kz_dura.train

python -m models.kz_num.train

python -m models.kz_level.train

### 模型预测

python -m models.kz_time.evaluate

python -m models.kz_dura.evaluate

python -m models.kz_num.evaluate

python -m models.kz_level.evaluate



任务类型	模型名称

分类	
逻辑回归 (Logistic Regression)	处理二分类问题，计算效率高，结果有概率意义
支持向量机 (SVM - 用于分类)	善于找到复杂分类边界，尤其适合高维数据和小样本情况
决策树 (Decision Tree - 用于分类)	模型直观易解释，容易过拟合

回归	
线性回归 (Linear Regression)	建模变量间线性关系，简单、高效、可解释性强
岭回归 (Ridge Regression)	在线性回归基础上加入L2正则化，处理特征多重共线性，防止过拟合
Lasso回归 (Lasso Regression)	在线性回归基础上加入L1正则化，能进行特征选择，使模型更稀疏


分类与回归	支持向量机 (SVM)	通过不同核函数和损失函数，可灵活应用于分类和回归问题
决策树 (Decision Tree)	通过不同的分裂准则（如基尼系数、信息增益或均方误差）处理分类和回归任务
随机森林 (Random Forest)	集成多棵决策树，通过投票或平均进行预测，泛化能力强大，抗过拟合
梯度提升机 (XGBoost, LightGBM)	逐步修正前序模型的错误，预测精度高，在许多数据竞赛中表现出色




特征分布情况：
       hour_of_day  day_of_week       month  is_weekend    hour_sin    hour_cos  day_of_year  prev_gate_count  ...  flow_rain_ratio  is_rush_hour  water_status_mean  water_status_max  water_status_min  water_status_range  water_status_slope  water_status_missing
count   980.000000   980.000000  980.000000  980.000000  980.000000  980.000000   980.000000       980.000000  ...     9.800000e+02    980.000000         980.000000         980.00000        980.000000          980.000000        9.800000e+02            980.000000
mean     12.974490     2.924490    5.750000    0.254082   -0.102901   -0.473695   159.624490        10.806122  ...     3.347122e+05      0.317347           4.587168          26.06501          1.970969           24.094041        1.674770e-07              0.381633
std       4.407066     1.958823    3.013716    0.435566    0.713413    0.506801    91.809323         4.488000  ...     1.691327e+06      0.465681           3.611599          20.49194          1.667278           18.951099        3.067141e-06              0.486035
min       0.000000     0.000000    1.000000    0.000000   -1.000000   -1.000000     2.000000         0.000000  ...     0.000000e+00      0.000000           0.000000           0.00000         -1.870000            0.000000       -2.941310e-05              0.000000
25%       9.000000     1.000000    3.000000    0.000000   -0.707107   -0.866025    84.000000         8.000000  ...     0.000000e+00      0.000000           0.000000           0.00000          0.000000            0.000000        0.000000e+00              0.000000
50%      13.000000     3.000000    6.000000    0.000000   -0.258819   -0.707107   155.000000        10.000000  ...     3.821974e-02      0.000000           7.211200          41.92500          2.640000           38.440000        0.000000e+00              0.000000
75%      16.000000     5.000000    8.000000    1.000000    0.707107   -0.258819   221.000000        14.000000  ...     6.370912e-01      1.000000           7.490090          42.40250          3.617500           39.052500        7.430340e-07              1.000000
max      23.000000     6.000000   12.000000    1.000000    1.000000    1.000000   365.000000        28.000000  ...     2.514212e+07      1.000000           8.355417          42.81000          4.000000           44.230000        2.520487e-05              1.000000

[8 rows x 97 columns]

=== 分析目标变量: 开闸时间(小时) ===

特征与目标 '开闸时间(小时)' 的相关性：
hour_of_day                  0.339329 
tide_24h_tide_rise_rate      0.117882 
tide_24h_tide_fall_rate      0.105184 
future_water_missing         0.095228 
future_tide_tide_slope       0.091744 
tide_24h_tide_cycle_count    0.089262 
future_tide_tide_range       0.080782 
tide_12h_tide_fall_rate      0.064833 
flow_rain_ratio              0.063483 
future_tide_tide_max         0.059073
dtype: float64
随机森林模型性能 (开闸时间(小时)): R2=0.5474, MSE=10.7882

特征重要性排序 (开闸时间(小时)):
                    feature  importance
0               hour_of_day    0.391009
5                  hour_cos    0.160437
4                  hour_sin    0.073048
89          flow_rain_ratio    0.050522
9              prev_op_hour    0.042500
12  ops_week_total_duration    0.029763
38   future_tide_tide_slope    0.021621
6               day_of_year    0.019505
17      tide_24h_tide_slope    0.016856
1               day_of_week    0.015986

=== 分析目标变量: 开闸时长 ===

特征与目标 '开闸时长' 的相关性：
ops_week_total_duration    0.481209
water_status_missing       0.380520
flow_missing               0.379216
water_missing              0.377869
ops_week_count             0.375852
tide_24h_tide_min          0.375542
prev_gate_count            0.365668
tide_12h_tide_min          0.352008
ops_week_avg_gates         0.288154
prev_duration              0.274795
dtype: float64
随机森林模型性能 (开闸时长): R2=0.3937, MSE=0.4879

特征重要性排序 (开闸时长):
                    feature  importance
93         water_status_min    0.151503
12  ops_week_total_duration    0.148799
8             prev_duration    0.079649
11       ops_week_avg_gates    0.053617
6               day_of_year    0.051068
9              prev_op_hour    0.047107
7           prev_gate_count    0.043042
1               day_of_week    0.027432
5                  hour_cos    0.026518
0               hour_of_day    0.022259

=== 分析目标变量: 开闸孔数 ===

特征与目标 '开闸孔数' 的相关性：
prev_gate_count            0.458496
rain_actual_avg            0.386750
rain_actual_total          0.386750
ops_week_count             0.375011
ops_week_total_duration    0.374464
rain_actual_虞北平原_sum       0.370383
rain_actual_嵊州_sum         0.366215
rain_actual_绍兴平原_sum       0.361241
rain_actual_虞南山区_sum       0.360504
ops_week_avg_gates         0.355064
dtype: float64
随机森林模型性能 (开闸孔数): R2=0.3413, MSE=2.5584

特征重要性排序 (开闸孔数):
                       feature  importance
12     ops_week_total_duration    0.141965
82           rain_actual_total    0.069843
7              prev_gate_count    0.062412
84             rain_actual_avg    0.053470
93            water_status_min    0.051714
6                  day_of_year    0.050228
11          ops_week_avg_gates    0.041803
9                 prev_op_hour    0.040433
39  future_tide_tide_r_squared    0.025213
1                  day_of_week    0.023776

=== 分析目标变量: 目标水位 ===

特征与目标 '目标水位' 的相关性：
tide_24h_tide_cycle_count     0.384441
water_status_min              0.380125
tide_24h_tide_rise_rate       0.376914
tide_24h_tide_fall_rate       0.376012
future_tide_tide_r_squared    0.340304
water_status_mean             0.270662
water_status_max              0.259960
tide_12h_tide_cycle_count     0.254384
water_status_range            0.247654
tide_24h_tide_range           0.240236
dtype: float64
随机森林模型性能 (目标水位): R2=0.2738, MSE=0.5279

特征重要性排序 (目标水位):
                    feature  importance
93         water_status_min    0.199692
7           prev_gate_count    0.196222
12  ops_week_total_duration    0.074290
6               day_of_year    0.072297
11       ops_week_avg_gates    0.046322
9              prev_op_hour    0.043468
8             prev_duration    0.042399
1               day_of_week    0.020301
5                  hour_cos    0.019575
10           ops_week_count    0.019191

=== 分析目标变量: 时长孔数乘积 ===

特征与目标 '时长孔数乘积' 的相关性：   
prev_gate_count            0.479105    
ops_week_total_duration    0.473047    
ops_week_count             0.413503    
ops_week_avg_gates         0.350063    
rain_actual_avg            0.257980    
rain_actual_total          0.257980    
tide_24h_tide_min          0.256952    
rain_actual_嵊州_sum         0.247580  
tide_12h_tide_min          0.245707    
rain_actual_虞南山区_sum       0.244365
dtype: float64
随机森林模型性能 (时长孔数乘积): R2=0.4115, MSE=66.2193

特征重要性排序 (时长孔数乘积):
                    feature  importance
12  ops_week_total_duration    0.147010
7           prev_gate_count    0.130220
93         water_status_min    0.091072
11       ops_week_avg_gates    0.056557
6               day_of_year    0.056056
9              prev_op_hour    0.050736
8             prev_duration    0.048958
10           ops_week_count    0.027358
1               day_of_week    0.026619
89          flow_rain_ratio    0.025566

=== 分析目标变量: 时长孔数乘积(对数) ===

特征与目标 '时长孔数乘积(对数)' 的相关性：
ops_week_total_duration    0.471087
prev_gate_count            0.465736
ops_week_count             0.425601
ops_week_avg_gates         0.387984
rain_actual_avg            0.262709
rain_actual_total          0.262709
tide_24h_tide_min          0.261862
tide_12h_tide_min          0.251698
prev_duration              0.251621
rain_actual_嵊州_sum         0.251174
dtype: float64
随机森林模型性能 (时长孔数乘积(对数)): R2=0.3627, MSE=0.2361

特征重要性排序 (时长孔数乘积(对数)):
                       feature  importance
93            water_status_min    0.216666
12     ops_week_total_duration    0.088104
7              prev_gate_count    0.085818
6                  day_of_year    0.053693
11          ops_week_avg_gates    0.045369
8                prev_duration    0.044574
9                 prev_op_hour    0.040881
89             flow_rain_ratio    0.028558
39  future_tide_tide_r_squared    0.025660
12  ops_week_total_duration    0.147010
7           prev_gate_count    0.130220
93         water_status_min    0.091072
11       ops_week_avg_gates    0.056557
6               day_of_year    0.056056
9              prev_op_hour    0.050736
8             prev_duration    0.048958
10           ops_week_count    0.027358
1               day_of_week    0.026619
89          flow_rain_ratio    0.025566

=== 分析目标变量: 时长孔数乘积(对数) ===

特征与目标 '时长孔数乘积(对数)' 的相关性：
ops_week_total_duration    0.471087
prev_gate_count            0.465736
ops_week_count             0.425601
ops_week_avg_gates         0.387984
rain_actual_avg            0.262709
rain_actual_total          0.262709
tide_24h_tide_min          0.261862
tide_12h_tide_min          0.251698
prev_duration              0.251621
rain_actual_嵊州_sum         0.251174
dtype: float64
随机森林模型性能 (时长孔数乘积(对数)): R2=0.3627, MSE=0.2361

特征重要性排序 (时长孔数乘积(对数)):
                       feature  importance
93            water_status_min    0.216666
12     ops_week_total_duration    0.088104
7              prev_gate_count    0.085818
6                  day_of_year    0.053693
11          ops_week_avg_gates    0.045369
8                prev_duration    0.044574
9                 prev_op_hour    0.040881
89             flow_rain_ratio    0.028558
39  future_tide_tide_r_squared    0.025660
1                  day_of_week    0.025348

12  ops_week_total_duration    0.147010
7           prev_gate_count    0.130220
93         water_status_min    0.091072
11       ops_week_avg_gates    0.056557
6               day_of_year    0.056056
9              prev_op_hour    0.050736
8             prev_duration    0.048958
10           ops_week_count    0.027358
1               day_of_week    0.026619
89          flow_rain_ratio    0.025566

=== 分析目标变量: 时长孔数乘积(对数) ===

特征与目标 '时长孔数乘积(对数)' 的相关性：
ops_week_total_duration    0.471087
prev_gate_count            0.465736
ops_week_count             0.425601
ops_week_avg_gates         0.387984
rain_actual_avg            0.262709
rain_actual_total          0.262709
tide_24h_tide_min          0.261862
tide_12h_tide_min          0.251698
prev_duration              0.251621
rain_actual_嵊州_sum         0.251174
dtype: float64
随机森林模型性能 (时长孔数乘积(对数)): R2=0.3627, MSE=0.2361

特征重要性排序 (时长孔数乘积(对数)):
                       feature  importance
93            water_status_min    0.216666
12     ops_week_total_duration    0.088104
7              prev_gate_count    0.085818
6                  day_of_year    0.053693
11          ops_week_avg_gates    0.045369
8                prev_duration    0.044574
9                 prev_op_hour    0.040881
89             flow_rain_ratio    0.028558
39  future_tide_tide_r_squared    0.025660
9              prev_op_hour    0.050736
8             prev_duration    0.048958
10           ops_week_count    0.027358
1               day_of_week    0.026619
89          flow_rain_ratio    0.025566

=== 分析目标变量: 时长孔数乘积(对数) ===

特征与目标 '时长孔数乘积(对数)' 的相关性：
ops_week_total_duration    0.471087
prev_gate_count            0.465736
ops_week_count             0.425601
ops_week_avg_gates         0.387984
rain_actual_avg            0.262709
rain_actual_total          0.262709
tide_24h_tide_min          0.261862
tide_12h_tide_min          0.251698
prev_duration              0.251621
rain_actual_嵊州_sum         0.251174
dtype: float64
随机森林模型性能 (时长孔数乘积(对数)): R2=0.3627, MSE=0.2361

特征重要性排序 (时长孔数乘积(对数)):
                       feature  importance
93            water_status_min    0.216666
12     ops_week_total_duration    0.088104
7              prev_gate_count    0.085818
6                  day_of_year    0.053693
11          ops_week_avg_gates    0.045369
8                prev_duration    0.044574
9                 prev_op_hour    0.040881
89             flow_rain_ratio    0.028558
39  future_tide_tide_r_squared    0.025660

=== 分析目标变量: 时长孔数乘积(对数) ===

特征与目标 '时长孔数乘积(对数)' 的相关性：
ops_week_total_duration    0.471087
prev_gate_count            0.465736
ops_week_count             0.425601
ops_week_avg_gates         0.387984
rain_actual_avg            0.262709
rain_actual_total          0.262709
tide_24h_tide_min          0.261862
tide_12h_tide_min          0.251698
prev_duration              0.251621
rain_actual_嵊州_sum         0.251174
dtype: float64
随机森林模型性能 (时长孔数乘积(对数)): R2=0.3627, MSE=0.2361

特征重要性排序 (时长孔数乘积(对数)):
                       feature  importance
93            water_status_min    0.216666
12     ops_week_total_duration    0.088104
7              prev_gate_count    0.085818
6                  day_of_year    0.053693
11          ops_week_avg_gates    0.045369
8                prev_duration    0.044574
9                 prev_op_hour    0.040881
89             flow_rain_ratio    0.028558
39  future_tide_tide_r_squared    0.025660
rain_actual_total          0.262709
tide_24h_tide_min          0.261862
tide_12h_tide_min          0.251698
prev_duration              0.251621
rain_actual_嵊州_sum         0.251174
dtype: float64
随机森林模型性能 (时长孔数乘积(对数)): R2=0.3627, MSE=0.2361

特征重要性排序 (时长孔数乘积(对数)):
                       feature  importance
93            water_status_min    0.216666
12     ops_week_total_duration    0.088104
7              prev_gate_count    0.085818
6                  day_of_year    0.053693
11          ops_week_avg_gates    0.045369
8                prev_duration    0.044574
9                 prev_op_hour    0.040881
89             flow_rain_ratio    0.028558
39  future_tide_tide_r_squared    0.025660
dtype: float64
随机森林模型性能 (时长孔数乘积(对数)): R2=0.3627, MSE=0.2361

特征重要性排序 (时长孔数乘积(对数)):
                       feature  importance
93            water_status_min    0.216666
12     ops_week_total_duration    0.088104
7              prev_gate_count    0.085818
6                  day_of_year    0.053693
11          ops_week_avg_gates    0.045369
8                prev_duration    0.044574
9                 prev_op_hour    0.040881
89             flow_rain_ratio    0.028558
39  future_tide_tide_r_squared    0.025660
12     ops_week_total_duration    0.088104
7              prev_gate_count    0.085818
6                  day_of_year    0.053693
11          ops_week_avg_gates    0.045369
8                prev_duration    0.044574
9                 prev_op_hour    0.040881
89             flow_rain_ratio    0.028558
39  future_tide_tide_r_squared    0.025660
11          ops_week_avg_gates    0.045369
8                prev_duration    0.044574
9                 prev_op_hour    0.040881
89             flow_rain_ratio    0.028558
39  future_tide_tide_r_squared    0.025660
9                 prev_op_hour    0.040881
89             flow_rain_ratio    0.028558
39  future_tide_tide_r_squared    0.025660
89             flow_rain_ratio    0.028558
39  future_tide_tide_r_squared    0.025660
39  future_tide_tide_r_squared    0.025660
1                  day_of_week    0.025348

=== 目标变量之间的关系分析 ===
目标变量相关性矩阵:
            开闸时间(小时)      开闸时长      开闸孔数      目标水位    时长孔数乘积  时长孔数乘积(对数)
开闸时间(小时)    1.000000 -0.051780 -0.138659  0.121547 -0.097318   -0.098613
开闸时长       -0.051780  1.000000  0.493280 -0.697410  0.842913    0.806962
开闸孔数       -0.138659  0.493280  1.000000 -0.686679  0.852021    0.860638
目标水位        0.121547 -0.697410 -0.686679  1.000000 -0.801005   -0.755715
时长孔数乘积     -0.097318  0.842913  0.852021 -0.801005  1.000000    0.914692
时长孔数乘积(对数) -0.098613  0.806962  0.860638 -0.755715  0.914692    1.000000


预测开闸与否
POST 127.0.0.1:8001/predict_bin

{
    "metadata": {
        "model_version": "binary_1.0",
        "prediction_time": "2025-09-24 13:41:31",
        "threshold_used": 0.5
    },
    "prediction": {
        "confidence": 0.7303,
        "confidence_level": "medium",
        "explanation": "预测开闸 (概率: 0.730) - 中等置信度开闸",
        "features_used": 159,
        "prediction": 1,
        "probability": 0.7303,
        "recommendation": "Issue gate opening order"
    },
    "status": "success"
}



本防汛调度经验模型是基于多源监测数据与机器学习构建的智能决策系统。
1）融合水位、流量、降雨及潮汐预报等多维数据，通过特征工程提取97项关键指标，支撑精准分析；
2）采用多模型协同架构，分别预测开闸时间、时长、孔数和目标水位，并依据潮汐规律自动识别泄洪窗口，实现孔数分级智能分配；


