import os
import json
import time
import requests

"""
高德地图 - 逆地理编码网格扫描爬虫 (使用 基础LBS服务 150000次/月 额度)

原理揭秘：
我们不再用搜索栏搜“XX小区”，这是非常低效且受限于 5000 次配额的方法。
我们在北京五到六环内的地图上，每隔约 1 公里“扎一根针（生成经纬度点）”。
然后问 API：“这根针方圆 1.5 公里内，所有的住宅区（POIType: 120300）都报上名来！”
只要扫遍全网格，全北京市的小区就能一次性全部刮出来，并且合法合规，配额极其宽裕。
"""

API_KEY = "填写你的真实_API_KEY"

# 北京市区及郊区核心网格范围 (以天安门为中心向外扩散，大致覆盖六环内)
LNG_START = 115.80
LNG_END = 117.00
LAT_START = 39.50
LAT_END = 40.30

# 步长 0.015 度，约在地图上间隔 1.5 公里
STEP = 0.015  

OUTPUT_FILE = "grid_communities.jsonl"
STATE_FILE = "grid_state.json"

def generate_grid():
    grid = []
    lng = LNG_START
    while lng <= LNG_END:
        lat = LAT_START
        while lat <= LAT_END:
            grid.append((round(lng, 4), round(lat, 4)))
            lat += STEP
        lng += STEP
    return grid

def fetch_regeo_pois(lng, lat):
    # poitype=120300 是高德特定的 "住宅区" 类目
    # extensions=all 才会返回周边 POI
    url = f"https://restapi.amap.com/v3/geocode/regeo?key={API_KEY}&location={lng},{lat}&extensions=all&poitype=120300&radius=1500"
    
    try:
        req = requests.get(url, timeout=5)
        data = req.json()
        if str(data.get("status")) == "1" and "regeocode" in data:
            return data["regeocode"].get("pois", [])
        print(f"[{lng},{lat}] 请求失败状态码或报错: {data.get('info')}")
        return []
    except Exception as e:
        print(f"[{lng},{lat}] 网络请求错误: {e}")
        return None

if __name__ == "__main__":
    grid = generate_grid()
    print(f"初始化完成，总计生成 {len(grid)} 个扫描网格点。")
    print("这可能需要消耗大约 4000 ~ 5000 次基础 LBS 配额（完全在你 15 万次的额度保障内）。\n")
    
    scanned_points = set()
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            scanned_points = set(json.load(f))
            print(f"-> 恢复进度：发现已扫描 {len(scanned_points)} 个点位。")
            
    with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f:
        for idx, point in enumerate(grid):
            point_str = f"{point[0]},{point[1]}"
            if point_str in scanned_points:
                continue
                
            print(f"({idx+1}/{len(grid)}) 正在抛网扫描坐标点: {point[0]}, {point[1]} ...")
            pois = fetch_regeo_pois(point[0], point[1])
            
            if pois is not None:
                community_count = 0
                for p in pois:
                    out_f.write(json.dumps({
                        "id": p.get("id"),
                        "name": p.get("name"),
                        "location": p.get("location"),
                        "address": p.get("address"),
                        "district": p.get("adname", ""),  # 所属区县
                        "city": p.get("cityname", "")
                    }, ensure_ascii=False) + "\n")
                    community_count += 1
                    
                print(f"    └─ 成功捞起 {community_count} 个小区/住宅。")
                
                scanned_points.add(point_str)
                # 记录进度防断电
                with open(STATE_FILE, "w") as sf:
                    json.dump(list(scanned_points), sf)
            
            # API 规定有并发频率限制，停顿 0.1 秒不仅符合规范，也极其安全
            time.sleep(0.1)
