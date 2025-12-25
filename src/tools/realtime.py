"""
实时信息工具模块
获取时间、日期、天气等实时信息
"""

import os
from datetime import datetime
from typing import Dict, Any
import urllib.request
import json


def get_current_datetime() -> Dict[str, Any]:
    """
    获取当前日期和时间
    
    Returns:
        包含日期时间信息的字典
    """
    now = datetime.now()
    
    # 中文星期
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    weekday = weekdays[now.weekday()]
    
    # 时间段
    hour = now.hour
    if 5 <= hour < 9:
        period = "早上"
    elif 9 <= hour < 12:
        period = "上午"
    elif 12 <= hour < 14:
        period = "中午"
    elif 14 <= hour < 18:
        period = "下午"
    elif 18 <= hour < 22:
        period = "晚上"
    else:
        period = "深夜"
    
    return {
        "success": True,
        "date": now.strftime("%Y年%m月%d日"),
        "time": now.strftime("%H:%M"),
        "weekday": weekday,
        "period": period,
        "formatted": f"{now.strftime('%Y年%m月%d日')} {weekday} {period} {now.strftime('%H:%M')}",
        "timestamp": now.isoformat()
    }


def get_weather(city: str = "深圳") -> Dict[str, Any]:
    """
    获取指定城市的天气信息
    使用多个API源，优先使用可用的
    
    Args:
        city: 城市名称
    
    Returns:
        天气信息字典
    """
    # 方案1：使用 open-meteo (免费、无需API Key、国际可用)
    result = _get_weather_open_meteo(city)
    if result["success"]:
        return result
    
    # 方案2：使用 wttr.in 作为备选
    result = _get_weather_wttr(city)
    if result["success"]:
        return result
    
    # 都失败了
    return {
        "success": False,
        "city": city,
        "error": "所有天气API均不可用",
        "formatted": f"抱歉，暂时无法获取{city}的天气信息，不过我还是在这里陪着你呀~"
    }


# 城市坐标映射（用于 open-meteo API）
CITY_COORDS = {
    "深圳": (22.55, 114.07),
    "北京": (39.90, 116.40),
    "上海": (31.23, 121.47),
    "广州": (23.13, 113.26),
    "杭州": (30.27, 120.15),
    "成都": (30.57, 104.07),
    "武汉": (30.58, 114.27),
    "西安": (34.27, 108.93),
    "南京": (32.06, 118.80),
    "重庆": (29.56, 106.55),
    "苏州": (31.30, 120.62),
    "天津": (39.12, 117.20),
    "长沙": (28.23, 112.94),
    "青岛": (36.07, 120.38),
    "东莞": (23.05, 113.75),
}


def _get_weather_open_meteo(city: str) -> Dict[str, Any]:
    """使用 Open-Meteo API 获取天气（免费、无需Key）"""
    try:
        # 获取城市坐标
        coords = CITY_COORDS.get(city)
        if not coords:
            # 默认深圳坐标
            coords = CITY_COORDS["深圳"]
            city = "深圳"
        
        lat, lon = coords
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,weather_code&daily=temperature_2m_max,temperature_2m_min&timezone=Asia/Shanghai&forecast_days=1"
        
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))
        
        current = data.get("current", {})
        daily = data.get("daily", {})
        
        temp = current.get("temperature_2m", "未知")
        humidity = current.get("relative_humidity_2m", "未知")
        weather_code = current.get("weather_code", 0)
        
        max_temp = daily.get("temperature_2m_max", ["未知"])[0]
        min_temp = daily.get("temperature_2m_min", ["未知"])[0]
        
        # 天气代码转描述
        weather_desc = _weather_code_to_desc(weather_code)
        
        return {
            "success": True,
            "city": city,
            "temperature": f"{temp}°C",
            "humidity": f"{humidity}%",
            "weather": weather_desc,
            "high": f"{max_temp}°C",
            "low": f"{min_temp}°C",
            "formatted": f"{city}现在{weather_desc}，{temp}°C，湿度{humidity}%，今天{min_temp}~{max_temp}°C"
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}


def _weather_code_to_desc(code: int) -> str:
    """WMO天气代码转中文描述"""
    weather_map = {
        0: "晴天",
        1: "晴朗", 2: "多云", 3: "阴天",
        45: "有雾", 48: "雾凇",
        51: "小毛毛雨", 53: "毛毛雨", 55: "大毛毛雨",
        61: "小雨", 63: "中雨", 65: "大雨",
        66: "冻雨", 67: "大冻雨",
        71: "小雪", 73: "中雪", 75: "大雪",
        77: "雪粒",
        80: "小阵雨", 81: "阵雨", 82: "大阵雨",
        85: "小阵雪", 86: "大阵雪",
        95: "雷阵雨",
        96: "雷阵雨伴小冰雹", 99: "雷阵雨伴冰雹",
    }
    return weather_map.get(code, "未知天气")


def _get_weather_wttr(city: str) -> Dict[str, Any]:
    """使用 wttr.in API 获取天气（备选）"""
    try:
        url = f"https://wttr.in/{city}?format=j1&lang=zh"
        req = urllib.request.Request(url, headers={"User-Agent": "curl/7.68.0"})
        
        with urllib.request.urlopen(req, timeout=3) as response:
            data = json.loads(response.read().decode("utf-8"))
        
        current = data.get("current_condition", [{}])[0]
        temp_c = current.get("temp_C", "未知")
        humidity = current.get("humidity", "未知")
        weather_desc = current.get("weatherDesc", [{}])[0].get("value", "未知")
        
        weather_data = data.get("weather", [{}])[0]
        max_temp = weather_data.get("maxtempC", "未知")
        min_temp = weather_data.get("mintempC", "未知")
        
        return {
            "success": True,
            "city": city,
            "temperature": f"{temp_c}°C",
            "humidity": f"{humidity}%",
            "weather": weather_desc,
            "high": f"{max_temp}°C",
            "low": f"{min_temp}°C",
            "formatted": f"{city}现在{weather_desc}，{temp_c}°C，湿度{humidity}%，今天{min_temp}~{max_temp}°C"
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    # 测试
    print("=== 测试时间 ===")
    print(get_current_datetime())
    
    print("\n=== 测试天气 ===")
    print(get_weather("深圳"))

