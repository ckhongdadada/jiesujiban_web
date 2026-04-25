import requests

AK = "w4ybVBML4tf70w3GLecMbd7Q7PmrDteT"
URL = "https://api.map.baidu.com/place/v2/search"

params = {
    "query": "小区",
    "region": "北京",
    "output": "json",
    "ak": AK
}

print("========================================")
print("正在使用你的百度 AK 测试 API 联通性...")
print("========================================")

try:
    resp = requests.get(URL, params=params, timeout=5)
    data = resp.json()

    if data.get("status") == 0:
        print("✅ 百度 API 测试通行成功！这把 AK 钥匙是有效的。下面是数据示例：")
        pois = data.get("results", [])
        for poi in pois[:3]:
            print(f" - 小区名: {poi.get('name')} | 区域: {poi.get('area')} | 地址: {poi.get('address')}")
    else:
        print("❌ 百度 API 调用失败！请检查报错信息：")
        print(data)
except Exception as e:
    print(f"❌ 网络调用遇到错误: {e}")
