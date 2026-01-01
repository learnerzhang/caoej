import pprint
import requests
import json
from datetime import datetime, timedelta

def get_tide_predictions(station_id, start_date, end_date, api_key):
    """
    获取指定站点和日期范围的潮汐预测数据
    
    参数:
    station_id: 站点ID
    start_date: 开始日期 (YYYYMMDD)
    end_date: 结束日期 (YYYYMMDD)
    api_key: NOAA API 密钥
    
    返回:
    包含潮汐预测数据的字典
    """
    # NOAA CO-OPS API 端点
    url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
    
    # 请求参数
    params = {
        "begin_date": start_date,
        "end_date": end_date,
        "station": station_id,
        "product": "predictions",  # 产品类型：潮汐预测
        "datum": "MLLW",  # 基准面：平均低低潮面
        "units": "english",  # 单位：英制
        "time_zone": "lst_ldt",  # 时区：当地标准时间/当地夏令时
        "format": "json",  # 响应格式
        "interval": "hilo",  # 间隔：只返回高低潮
        "application": "Python_Tide_Example",  # 应用标识
        "api_key": api_key
    }
    
    try:
        # 发送请求
        response = requests.get(url, params=params)
        response.raise_for_status()  # 检查请求是否成功
        
        # 解析JSON响应
        data = response.json()
        
        # 检查是否有错误信息
        if "error" in data:
            print(f"API 错误: {data['error']['message']}")
            return None
            
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"请求错误: {e}")
        return None

def display_tide_data(tide_data):
    """展示潮汐数据"""
    # 如果没有潮汐数据或者预测数据，则返回
    if not tide_data or "predictions" not in tide_data:
        print("没有可用的潮汐数据")
        return
        
    # 打印站点名称和ID
    # 打印站点信息
    # 打印基准面
    print(f"站点: {tide_data['metadata']['name']} (ID: {tide_data['metadata']['id']})")
    print(f"基准面: {tide_data['metadata']['datum']}")
    print("\n潮汐预测:")
    print("-" * 60)
    # 打印表头
    print(f"{'时间':<20} {'类型':<10} {'高度(英尺)':<15}")
    print("-" * 60)
    
    # 遍历预测数据
    for prediction in tide_data['predictions']:
        # 格式化日期时间
        dt = datetime.strptime(prediction['t'], "%Y-%m-%d %H:%M")
        time_str = dt.strftime("%Y-%m-%d %H:%M")
        
        # 潮汐类型和高度
        tide_type = prediction['type']
        # 打印潮汐预测数据
        height = prediction['v']
        
        print(f"{time_str:<20} {tide_type:<10} {height:<15}")


def get_predictions(station_id, days, api_key):
    # 计算日期范围
    today = datetime.now()
    start_date = today.strftime("%Y%m%d")
    end_date = (today + timedelta(days=days)).strftime("%Y%m%d")
    
    print(f"获取 {station_id} 站点从 {start_date} 到 {end_date} 的潮汐数据...")
    
    # 获取潮汐数据
    tide_data = get_tide_predictions(station_id, start_date, end_date, api_key)
    pprint.pprint(tide_data)



def get_historical_tide_data(station_id, start_date, end_date, api_key):
    """
    获取指定站点和历史日期范围的潮汐数据
    
    参数:
    station_id: 站点ID
    start_date: 开始日期 (YYYYMMDD)
    end_date: 结束日期 (YYYYMMDD)
    api_key: NOAA API 密钥
    
    返回:
    包含历史潮汐数据的字典
    """
    url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
    
    # 对于历史数据，使用相同的API端点但指定历史日期范围
    params = {
        "begin_date": start_date,
        "end_date": end_date,
        "station": station_id,
        "product": "predictions",  # 历史预测数据
        "datum": "MLLW",
        "units": "metric",  # 这里改为公制单位（米）
        "time_zone": "gmt",  # 使用GMT时间避免时区问题
        "format": "json",
        "interval": "hilo",  # 高低潮数据
        "application": "Python_Historical_Tide_Example",
        "api_key": api_key
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if "error" in data:
            print(f"API 错误: {data['error']['message']}")
            return None
            
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"请求错误: {e}")
        return None

def display_historical_data(tide_data):
    """展示历史潮汐数据"""
    if not tide_data or "predictions" not in tide_data:
        print("没有可用的历史潮汐数据")
        return
        
    print(f"历史潮汐数据 - 站点: {tide_data['metadata']['name']}")
    print(f"日期范围: {tide_data['predictions'][0]['t'].split(' ')[0]} 至 {tide_data['predictions'][-1]['t'].split(' ')[0]}")
    print(f"基准面: {tide_data['metadata']['datum']}")
    print("\n历史潮汐记录:")
    print("-" * 65)
    print(f"{'日期时间':<22} {'类型':<10} {'高度(米)':<15} {'时区'}")
    print("-" * 65)
    
    for record in tide_data['predictions']:
        print(f"{record['t']:<22} {record['type']:<10} {record['v']:<15} GMT")

def get_histories(station_id, days, api_key):
    
    today = datetime.now()
    end_date = today.strftime("%Y%m%d")
    start_date = (today - timedelta(days=days)).strftime("%Y%m%d")
    
    print(f"获取 {station_id} 站点从 {start_date} 到 {end_date} 的潮汐数据...")
    
    # 获取历史数据
    historical_data = get_historical_tide_data(STATION_ID, start_date, end_date, api_key)
    pprint.pprint(historical_data)    


def get_CnStations():
    # 自然资源部公开数据示例
    import pandas as pd

    url = "https://global-tide.nmdis.org.cn/Api/Service.ashx"
    params = {"Server":"User","Command":"GetData","Data":{"code":"T072","date":"2025-08-02"}}
    resp = requests.get(url, params=params)
    print(resp.text)
    df = pd.read_json(resp.text)
    print(df.head())

if __name__ == "__main__":
    # 配置参数 - 请替换为你自己的信息
    API_KEY = "yzCzOJXFGzwxjfYMPLscwopuIbHkvHkG"  # 替换为你的NOAA API密钥
    STATION_ID = "8518750"    # 
    STATION_ID = "CH000003590"
    DAYS_TO_FETCH = 7         # 获取未来7天的数据
    """
    
    站点名称	站码 (Station ID)
    上海 (Shanghai)	GHCND:CH000003590
    青岛 (Qingdao)	GHCND:CH000004570
    厦门 (Xiamen)	GHCND:CH000005910
    香港 (Hong Kong)	GHCND:CH000004672
    杭州 (Hangzhou)	GHCND:CH000003580
    """
    # 计算日期范围
    # get_predictions(STATION_ID, DAYS_TO_FETCH, API_KEY)

    STATION_ID = "9410230"    # 
    STATION_ID = "CH000003590"
    # get_histories(STATION_ID, DAYS_TO_FETCH, API_KEY)
    # get_CnStations()
    




