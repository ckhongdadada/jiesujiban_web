import os
import json
import time
import requests

"""
高德地图 - 逆地理道路拓扑词典爬虫 (使用 基础LBS服务 150000次/月 额度)

目标：
提取北京市六环内所有的道路名称，并将其绑定到精度极高的【区县 + 街道/乡镇】。

原理：
通过网格经纬度查询 `/v3/geocode/regeo?location=lng,lat&extensions=all`，
高德的 `addressComponent` 会直接返回当前点所在的区 (district) 和所属的极其精细的街镇 (township)。
高德的 `roads` 数组会一并返回当前坐标点周边 1-2 公里内的所有道路。
由于我们在网格内扫描，这就形成了一张绝对可靠的地名映射表：[任意道路] -> [所属区县]-[所属街镇]。
"""

API_KEY = os.getenv("AMAP_API_KEY", "2702f59ae73556e5b4029b4a8eaf1310").strip()
if not API_KEY:
    raise RuntimeError("请配置真实的环境变量 AMAP_API_KEY (或者直接替换此处代码) 才能运行本爬虫！")

# 剔除了房山深山、延庆密云保护区等彻底的无人区，只围猎北京冲积平原和六环核心带
LNG_START = 115.90
LNG_END = 116.85
LAT_START = 39.60
LAT_END = 40.25

# 步长 0.01 度，圆心间隔约 1 公里（相切但不重叠过度）
STEP = 0.01

OUTPUT_FILE = "road_township_dictionary.jsonl"
STATE_FILE = "road_grid_state.json"

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

def fetch_roads_and_township(lng, lat):
    # 把探测半径从过度密集的 500 米恢复到 1000 米，彻底吃满 1 公里的步长身位
    url = f"https://restapi.amap.com/v3/geocode/regeo?key={API_KEY}&location={lng},{lat}&extensions=all&radius=1000"
    
    try:
        req = requests.get(url, timeout=5)
        data = req.json()
        if str(data.get("status")) == "1" and "regeocode" in data:
            regeo = data["regeocode"]
            address_comp = regeo.get("addressComponent", {})
            
            district = address_comp.get("district", "")
            township = address_comp.get("township", "")
            
            # 极少数荒郊野外/高速路上没有街道划分，高德可能返回空数组 [] 而非字符串，需要容错处理
            if isinstance(district, list): district = ""
            if isinstance(township, list): township = ""
            
            roads = regeo.get("roads", [])
            return {
                "district": district,
                "township": township,
                "roads": [r.get("name") for r in roads if r.get("name")]
            }
        print(f"[{lng},{lat}] 请求失败或超限: {data.get('info')}")
        return None
    except Exception as e:
        print(f"[{lng},{lat}] 网络请求错误: {e}")
        return None

if __name__ == "__main__":
    grid = generate_grid()
    print(f"初始化完成，总计生成 {len(grid)} 个扫描网格点。")
    print("这可能需要消耗大约 4000 ~ 5000 次基础 LBS 配额（配额15万管够）。")
    print(f"结果将输出带有最高精度紧固带属性的词典到：{OUTPUT_FILE}\n")
    
    scanned_points = set()
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            scanned_points = set(json.load(f))
            print(f"-> 恢复进度：发现已扫描 {len(scanned_points)} 个点位。")
            
    # 使用集合在内存中对 (道路_区_街镇) 唯一键防重发，避免输出文件无脑膨胀
    seen_bindings = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as outf:
            for line in outf:
                if line.strip():
                    try:
                        record = json.loads(line)
                        seen_bindings.add(f"{record.get('road')}_{record.get('district')}_{record.get('township')}")
                    except ValueError:
                        pass
        print(f"-> 去重防重器：从已有词典中载入了 {len(seen_bindings)} 条道路防重记录！")

    with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f:
        for idx, point in enumerate(grid):
            point_str = f"{point[0]},{point[1]}"
            if point_str in scanned_points:
                continue
                
            print(f"({idx+1}/{len(grid)}) 正在探测坐标点 {point[0]}, {point[1]} 的路网与街镇归属...")
            result = fetch_roads_and_township(point[0], point[1])
            
            if result is not None:
                district = result["district"]
                township = result["township"]
                road_names = result["roads"]
                
                # 只有在这块地确实被明确划归了具体的行政“区”和“街镇”的情况下，这个紧固带映射才有高精度的价值
                if district and township:
                    bind_count = 0
                    for r_name in road_names:
                        # 复合唯一键防重：例如 长安街_东城区_东华门街道
                        unique_key = f"{r_name}_{district}_{township}"
                        
                        if unique_key not in seen_bindings:
                            seen_bindings.add(unique_key)
                            
                            dict_record = {
                                "road": r_name,
                                "district": district,      # 例如：朝阳区
                                "township": township,      # 例如：建外街道
                                # 最高精度绑定带！专供 NER 和大模型做街道级归属匹配使用
                                "high_precision_zone": f"{district}{township}"
                            }
                            out_f.write(json.dumps(dict_record, ensure_ascii=False) + "\n")
                            bind_count += 1
                            
                    print(f"    └─ 成功斩获 [{district}-{township}] 下辖的 {bind_count} 条未去重道路。")
                else:
                    print("    └─ 该点位处于行政边缘或无明确街镇划分，跳过强紧固记录。")
                
                scanned_points.add(point_str)
                with open(STATE_FILE, "w") as sf:
                    json.dump(list(scanned_points), sf)
            
            # API 规定有并发频率限制，停顿 0.1 秒极其安全且不会触发黑洞惩罚
            time.sleep(0.4)
