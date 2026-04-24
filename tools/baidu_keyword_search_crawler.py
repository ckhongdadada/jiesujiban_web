import os
import json
import time
import math
import datetime
import requests

"""
百度地图 API - 小区极速全量爬虫 (严格防超配额护城河版)

【核心防御机制】：
根据截图，你的地点检索配额仅为 0.01万次/天（即每天仅仅只有 100 次！），且并发上限为 3 QPS。
这套代码内置了极其严格的【每日 98 次安全熔断器】和【0.4秒并发限速器】。
哪怕你不小心忘记关掉了，它只要跑到 98 次就会死锁休眠，绝不会让你超额。
每天凌晨过后它会自动解锁，继续无缝接着旧进度向下爬。
"""

AK = "w4ybVBML4tf70w3GLecMbd7Q7PmrDteT"
URL = "https://api.map.baidu.com/place/v2/search"

OUTPUT_FILE = "baidu_community_records.jsonl"
STATE_FILE = "baidu_community_state.json"
DAILY_MAX_REQUESTS = 98  # 安全线，每天只查 98 次，给你留 2 次作为人工余量

DISTRICTS = [
    "东城区", "西城区", "朝阳区", "丰台区", "石景山区", "海淀区",
    "门头沟区", "房山区", "通州区", "顺义区", "昌平区", "大兴区",
    "怀柔区", "平谷区", "密云区", "延庆区"
]

KEYWORDS = [
    "小区", "苑", "家园", "园", "城", "公寓", "新村", "嘉园", "华庭", "国际",
    "名苑", "雅苑", "家苑", "里", "庄", "湾", "府", "郡", "阁", "公馆",
    "社区", "大厦", "广场", "家属院", "别墅", "一区", "二区", "三区", "胡同"
]

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"tasks": {}, "quota_date": "", "daily_used": 0}

def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def fetch_baidu_page(district, keyword, page_num):
    query_str = f"{district} {keyword}"
    params = {
        "query": query_str,
        "region": "北京",
        "city_limit": "true",
        "output": "json",
        "page_size": 20,
        "page_num": page_num,
        "ak": AK
    }
    try:
        req = requests.get(URL, params=params, timeout=10)
        data = req.json()
        if data.get("status") == 0:
            return data.get("results", []), data.get("total", 0)
        print(f"[{district}-{keyword}] 请求报错: {data.get('message')}")
        return None, 0
    except Exception as e:
        print(f"网络异常: {e}")
        return None, 0

if __name__ == "__main__":
    state = load_state()
    tasks_state = state.get("tasks", {})
    
    # 配额每日刷新逻辑
    today_str = datetime.date.today().isoformat()
    if state.get("quota_date") != today_str:
        state["quota_date"] = today_str
        state["daily_used"] = 0
        print(f"🔄 检测到新的一天 ({today_str})，已重置今日剩余配额。")
        
    # --- 新增：跨文件、跨词根的全局唯一去重器 ---
    seen_uids = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as outf:
            for line in outf:
                if line.strip():
                    try:
                        record = json.loads(line)
                        if "uid" in record:
                            seen_uids.add(record["uid"])
                    except ValueError:
                        pass
        print(f"-> 跨词根内存去重：从已有词典中载入了 {len(seen_uids)} 个确界小区的 UID，绝不重复索取！")
        
    print("🚀 百度地图高级切块爬虫启动！")
    print(f"🚨 你的地点检索并发上限: 3 QPS | 今日限额: 100 次 | 今日已用: {state['daily_used']} 次")
    
    if state["daily_used"] >= DAILY_MAX_REQUESTS:
        print("🛑 警告：今日安全额度早已耗尽，请明天再来运行吧！程序已保护性退出。")
        exit(0)
    
    quota_reached = False
    with open(OUTPUT_FILE, "a", encoding="utf-8") as outf:
        for district in DISTRICTS:
            if quota_reached: break
            for keyword in KEYWORDS:
                if quota_reached: break
                task_key = f"{district}_{keyword}"
                
                if task_key not in tasks_state:
                    tasks_state[task_key] = {"crawled_pages": [], "exhausted": False}
                    
                entry = tasks_state[task_key]
                if entry.get("exhausted"): 
                    continue
                
                page_num = 0
                while True:
                    if page_num in entry["crawled_pages"]:
                        page_num += 1
                        continue
                        
                    # 检查配额熔断机制 (核心防御)
                    if state["daily_used"] >= DAILY_MAX_REQUESTS:
                        print(f"\n🛑 【安全熔断触发】今日调用的配额已达 {DAILY_MAX_REQUESTS} 次！")
                        print("👉 强制关停爬虫以防账号被封禁超限扣费，请明天再执行本脚本，它会自动从最后一秒无缝衔接。")
                        quota_reached = True
                        break
                        
                    print(f"正在深挖: [{district}] 的「{keyword}」类小区 (第 {page_num+1} 页) ...")
                    pois, total_hits = fetch_baidu_page(district, keyword, page_num)
                    
                    # 只要发送了请求，立马计数扣除配额
                    state["daily_used"] += 1
                    state["tasks"] = tasks_state
                    save_state(state)
                    
                    if pois is None:
                        print("    └─ 接口抛错休息 3 秒，等会儿重试...")
                        time.sleep(3)
                        break
                        
                    if len(pois) == 0:
                        entry["exhausted"] = True
                        save_state(state)
                        print("    └─ 此碎块已被全线挖空！(零数据返回)")
                        break
                        
                    # 落地写文件
                    saved_count = 0
                    duplicate_count = 0
                    for p in pois:
                        if "uid" in p and p.get("name"):
                            uid = p.get("uid")
                            # 这里就是终极防重：只留完全没见过的小区
                            if uid in seen_uids:
                                duplicate_count += 1
                                continue
                            seen_uids.add(uid)
                            
                            loc = p.get("location", {})
                            outf.write(json.dumps({
                                "uid": p.get("uid"),
                                "name": p.get("name"),
                                "district": district,
                                "address": p.get("address", ""),
                                # 百度的经纬度天生是反过来的（先 lng，后 lat）
                                "location": f"{loc.get('lng', '')},{loc.get('lat', '')}",
                                "province": "北京市",
                                "source": "baidu_custom_spider"
                            }, ensure_ascii=False) + "\n")
                            saved_count += 1
                    
                    print(f"    └─ 成功写盘新增数据 {saved_count} 条，自动剔除了 {duplicate_count} 个重叠数据。")
                    print(f"    └─ 今日额度剩余 {DAILY_MAX_REQUESTS - state['daily_used']} 次")
                    
                    entry["crawled_pages"].append(page_num)
                    
                    # 分析是否已经抄底：如果拿到的数据不满足20条一页，或者翻到了总页数尽头
                    max_pages = math.ceil(total_hits / 20.0)
                    if page_num >= max_pages - 1 or len(pois) < 20: 
                        entry["exhausted"] = True
                        
                    save_state(state)
                    page_num += 1
                    
                    # 并发防御：配额图显示并发上限为 3 QPS，强制休息 0.4 秒确保极度安全
                    time.sleep(0.4)
                    
                    if entry["exhausted"]:
                        break
