import os
import time
import json
import requests
from math import ceil

# ================= 核心配置 =================
API_KEY = os.environ.get("AMAP_API_KEY") or "2702f59ae73556e5b4029b4a8eaf1310"
if not API_KEY:
    raise RuntimeError("请配置真实的环境变量 AMAP_API_KEY！")

# 剔除深山、只围猎北京冲积平原和六环核心带
LNG_START = 115.90
LNG_END = 116.85
LAT_START = 39.60
LAT_END = 40.25

# 采用 1km 的步长，正好通过 1000 米半径的周边搜索圈完美接合
STEP = 0.01  
RADIUS = 1000

# 高德地图 POI 分类码：120300(住宅区), 120200(商务住宅), 120301(别墅), 120302(住宅小区), 120303(宿舍), 120304(社区中心)
# 通过指定分类码，任何名字多奇葩的小区都无法遁形！
POI_TYPES = "120200|120300|120301|120302|120303|120304"

OUTPUT_FILE = "grid_community_dictionary.jsonl"
STATE_FILE = "community_grid_state.json"

# ================= 状态管理 =================
def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f).get("last_processed_idx", -1)
        except Exception:
            return -1
    return -1

def save_state(last_idx):
    with open(STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump({"last_processed_idx": last_idx}, f)

# ================= 预热已存数据集 (UID全局去重) =================
unique_community_ids = set()
def load_existing_communities():
    count = 0
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        unique_community_ids.add(record["uid"])
                        count += 1
                    except json.JSONDecodeError:
                        continue
    return count

# ================= 核心请求逻辑 =================
def generate_grid_points():
    grid = []
    lng = LNG_START
    while lng <= LNG_END:
        lat = LAT_START
        while lat <= LAT_END:
            grid.append((lng, lat))
            lat = round(lat + STEP, 5)
        lng = round(lng + STEP, 5)
    return grid

def fetch_communities_around(lng, lat):
    """
    通过周边搜索 API (place/around)，强制要求返回上述 POI 类型的实体
    带分页处理，确保每一个圆圈内的小区都被抓干净
    """
    communities = []
    page = 1
    
    while True:
        url = (f"https://restapi.amap.com/v3/place/around?key={API_KEY}"
               f"&location={lng},{lat}&radius={RADIUS}&types={POI_TYPES}"
               f"&offset=25&page={page}&extensions=base")
        
        try:
            req = requests.get(url, timeout=5)
            # 限流保护，周边搜索配额消耗快
            time.sleep(0.3)
            
            data = req.json()
            if data.get("status") == "1":
                pois = data.get("pois", [])
                
                for poi in pois:
                    communities.append({
                        "uid": poi.get("id"),
                        "name": poi.get("name"),
                        "type": poi.get("type"),
                        "address": poi.get("address") or poi.get("adname", ""),
                        "district": poi.get("adname", ""),
                        "location": poi.get("location"),
                        "source": "amap_grid_around",
                        "scan_center": f"{lng},{lat}"
                    })
                
                # 若当前页无数据或未达到 25 条，说明该区域已穷尽
                if len(pois) < 25 or page >= 10:  # 强制限制最大10页防止死循环
                    break
                page += 1
            else:
                # API 错误或超过并发限制
                if data.get("info") == "USER_DAILY_QUOTA_OVER":
                    print("\n[!] 警告：高德地图配额已用尽！请保留状态文件，明天再跑。")
                    save_state(last_processed_idx)
                    exit(1)
                break
        except Exception as e:
            print(f" 请求异常: {e}")
            time.sleep(1)
            break
            
    return communities

# ================= 主体流程 =================
if __name__ == "__main__":
    print("-" * 60)
    print("🚀 终极杀手锏：网格周边分类爬虫 (高德去杂质物理扫描)")
    print("-" * 60)
    
    existing_count = load_existing_communities()
    print(f"[+] 当前已录入的独立小区数据数量: {existing_count} 条")
    
    grid_points = generate_grid_points()
    total_cells = len(grid_points)
    print(f"[+] 经纬度网格划分完毕，涵盖核心生活带，总网格数: {total_cells} 个")
    
    last_processed_idx = load_state()
    if last_processed_idx >= 0:
        print(f"[!] 检测到记忆断点，将从第 {last_processed_idx + 1} 个网格 ({grid_points[last_processed_idx]}) 继续...")
    else:
        print(f"[+] 新任务启动！")

    start_idx = last_processed_idx + 1
    
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f_out:
        for i in range(start_idx, total_cells):
            lng, lat = grid_points[i]
            
            # 使用 POI 分类码抓取方圆 1000 米内的所有小区
            found_communities = fetch_communities_around(lng, lat)
            new_added = 0
            
            for c in found_communities:
                # 使用唯一的高德 POI ID 去重，无视重复网格区域交叉带来的重复数据
                if c["uid"] not in unique_community_ids:
                    unique_community_ids.add(c["uid"])
                    f_out.write(json.dumps(c, ensure_ascii=False) + "\n")
                    new_added += 1
            
            print(f"\r正在强推第 {i+1}/{total_cells} 格 ({lng:.4f}, {lat:.4f}) | "
                  f"本圈发现: {len(found_communities)} | 新增净资产: {new_added} | 总宝库: {len(unique_community_ids)}", 
                  end="", flush=True)

            # 每扫描 50 个网格，持久化一次断点状态
            if (i + 1) % 50 == 0:
                save_state(i)
                f_out.flush()
                
    # 正常跑完保存终点
    save_state(total_cells)
    print(f"\n\n🎉 伟大的收割已完成！北京非山区核心带的所有奇葩名字小区已被您全维覆盖！总计入库 {len(unique_community_ids)} 条。")
