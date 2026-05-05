#!/usr/bin/env python3
"""
北京地名爬虫 —— 链家小区 + 安居客小区 + 北京地铁官网地铁站 + 8684公交站
输出格式兼容 place_alias_catalog.jsonl，可直接追加到现有词典。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Iterable
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.jsjb.location.rule_based import DISTRICT_NAMES

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)
COMMON_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

BLOCK_KEYWORDS = ["验证码", "verifycode", "antibot", "访问异常", "登录", "ke-passport", "安全验证"]

LIANJIA_DISTRICT_MAP = {
    "东城区": "dongcheng",
    "西城区": "xicheng",
    "朝阳区": "chaoyang",
    "海淀区": "haidian",
    "丰台区": "fengtai",
    "石景山区": "shijingshan",
    "通州区": "tongzhou",
    "顺义区": "shunyi",
    "昌平区": "changping",
    "大兴区": "daxing",
    "房山区": "fangshan",
    "门头沟区": "mentougou",
    "怀柔区": "huairou",
    "平谷区": "pinggu",
    "密云区": "miyun",
    "延庆区": "yanqing",
}

ANJUKE_DISTRICT_MAP = {
    "东城区": "dongcheng",
    "西城区": "xicheng",
    "朝阳区": "chaoyang",
    "海淀区": "haidian",
    "丰台区": "fengtai",
    "石景山区": "shijingshan",
    "通州区": "tongzhou",
    "顺义区": "shunyi",
    "昌平区": "changping",
    "大兴区": "daxing",
    "房山区": "fangshan",
    "门头沟区": "mentougou",
    "怀柔区": "huairou",
    "平谷区": "pinggu",
    "密云区": "miyun",
    "延庆区": "yanqing",
}


@dataclass
class PlaceRecord:
    alias: str
    district: str
    category: str
    confidence: float
    source: str


def normalize_name(name: str) -> str:
    return re.sub(r"\s+", "", (name or "").strip())


def looks_blocked(html: str) -> bool:
    lower = (html or "").lower()
    return any(kw.lower() in lower for kw in BLOCK_KEYWORDS)


def extract_district_from_text(text: str) -> str:
    for district in DISTRICT_NAMES:
        if district in text or district.replace("区", "") in text:
            return district
    return ""


def safe_get(session: requests.Session, url: str, timeout: int = 20) -> str | None:
    try:
        resp = session.get(url, headers=COMMON_HEADERS, timeout=timeout)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or resp.encoding or "utf-8"
        return resp.text
    except Exception as e:
        print(f"  [WARN] request failed: {url} -> {e}")
        return None


def deduplicate(records: Iterable[PlaceRecord]) -> list[PlaceRecord]:
    seen: dict[tuple[str, str, str], PlaceRecord] = {}
    for r in records:
        key = (r.alias, r.district, r.category)
        seen[key] = r
    return list(seen.values())


# ──────────────────────────────────────────────
# 1. 链家小区爬虫（可能被反爬拦截）
# ──────────────────────────────────────────────
def crawl_lianjia(session: requests.Session, max_pages: int = 5) -> list[PlaceRecord]:
    records: list[PlaceRecord] = []
    base = "https://bj.lianjia.com/xiaoqu"

    for district_cn, district_slug in LIANJIA_DISTRICT_MAP.items():
        print(f"[链家] 爬取 {district_cn} ...")
        for page in range(1, max_pages + 1):
            url = f"{base}/{district_slug}/" if page == 1 else f"{base}/{district_slug}/pg{page}/"
            html = safe_get(session, url)
            if not html:
                break
            if looks_blocked(html):
                print(f"  [链家] {district_cn} page {page} 被反爬拦截")
                break

            soup = BeautifulSoup(html, "lxml")
            items = soup.select("li.clear.xiaoquListItem, div.list-content li, ul.listContent li")
            if not items:
                items = soup.select("[class*='xiaoquList'] li, [class*='listContent'] li")
            if not items:
                print(f"  [链家] {district_cn} page {page} 无列表项，停止")
                break

            page_count = 0
            for item in items:
                name = ""
                name_el = item.select_one(".title a, [class*='Title'] a, a[title]")
                if name_el:
                    name = name_el.get_text(strip=True) or name_el.get("title", "")
                if not name:
                    continue

                district = district_cn
                pos_el = item.select_one(".positionInfo, [class*='positionInfo']")
                if pos_el:
                    pos_text = pos_el.get_text(" ", strip=True)
                    extracted = extract_district_from_text(pos_text)
                    if extracted:
                        district = extracted

                records.append(PlaceRecord(
                    alias=normalize_name(name),
                    district=district,
                    category="community",
                    confidence=0.85,
                    source="lianjia_community",
                ))
                page_count += 1

            if page_count == 0:
                print(f"  [链家] {district_cn} page {page} 解析到 0 条，停止")
                break
            time.sleep(random.uniform(1.0, 2.0))

    return records


# ──────────────────────────────────────────────
# 2. 安居客小区爬虫
# ──────────────────────────────────────────────
def crawl_anjuke(session: requests.Session, max_pages: int = 5) -> list[PlaceRecord]:
    records: list[PlaceRecord] = []
    base = "https://beijing.anjuke.com/community"

    for district_cn, district_slug in ANJUKE_DISTRICT_MAP.items():
        print(f"[安居客] 爬取 {district_cn} ...")
        for page in range(1, max_pages + 1):
            url = f"{base}/{district_slug}/" if page == 1 else f"{base}/{district_slug}/p{page}/"
            html = safe_get(session, url)
            if not html:
                break
            if looks_blocked(html):
                print(f"  [安居客] {district_cn} page {page} 被反爬拦截")
                break

            soup = BeautifulSoup(html, "lxml")
            items = soup.select("li.list-item, div.li-itemmod, [class*='community-list'] li")
            if not items:
                items = soup.select("[class*='itemmod'], [class*='ListItem']")
            if not items:
                print(f"  [安居客] {district_cn} page {page} 无列表项，停止")
                break

            page_count = 0
            for item in items:
                name = ""
                name_el = item.select_one(".item-title a, .li-info h3 a, a[title], [class*='title'] a")
                if name_el:
                    name = name_el.get_text(strip=True) or name_el.get("title", "")
                if not name:
                    continue

                district = district_cn
                props_el = item.select_one(".props, [class*='props'], [class*='address']")
                if props_el:
                    props_text = props_el.get_text(" ", strip=True)
                    extracted = extract_district_from_text(props_text)
                    if extracted:
                        district = extracted

                records.append(PlaceRecord(
                    alias=normalize_name(name),
                    district=district,
                    category="community",
                    confidence=0.80,
                    source="anjuke_community",
                ))
                page_count += 1

            if page_count == 0:
                print(f"  [安居客] {district_cn} page {page} 解析到 0 条，停止")
                break
            time.sleep(random.uniform(1.2, 2.5))

    return records


# ──────────────────────────────────────────────
# 2.5 高德 POI 搜索小区/公交站/地铁站（需 AMAP_API_KEY）
# ──────────────────────────────────────────────
AMAP_POI_TYPES = {
    "community": "120300",
    "bus_stop": "150700",
    "subway_station": "150500",
    "park": "110101",
    "mall": "060100",
    "government": "130100",
    "river_lake": "110200",
    "bridge": "190302",
    "school": "141200",
    "hospital": "090100",
    "market": "060400",
    "cultural_site": "140100",
}


def crawl_amap_poi(
    session: requests.Session,
    amap_key: str,
    categories: list[str] | None = None,
    max_pages: int = 10,
) -> list[PlaceRecord]:
    if not amap_key:
        print("[高德POI] 未设置 AMAP_API_KEY，跳过")
        return []

    if categories is None:
        categories = [
            "community", "bus_stop", "subway_station",
            "park", "mall", "government", "river_lake", "bridge",
            "school", "hospital", "market", "cultural_site",
        ]

    records: list[PlaceRecord] = []
    poi_url = "https://restapi.amap.com/v3/place/text"

    for district_cn in DISTRICT_NAMES:
        for cat in categories:
            type_code = AMAP_POI_TYPES.get(cat)
            if not type_code:
                continue

            print(f"[高德POI] {district_cn} / {cat} ...")
            for page in range(1, max_pages + 1):
                params = {
                    "key": amap_key,
                    "keywords": "",
                    "types": type_code,
                    "city": "北京",
                    "citylimit": "true",
                    "district": district_cn,
                    "offset": "25",
                    "page": str(page),
                    "output": "json",
                }
                try:
                    resp = session.get(poi_url, params=params, headers=COMMON_HEADERS, timeout=15)
                    resp.raise_for_status()
                    payload = resp.json()
                except Exception as e:
                    print(f"  [高德POI] 请求失败: {e}")
                    break

                pois = payload.get("pois") or []
                if not pois:
                    break

                for poi in pois:
                    name = poi.get("name", "").strip()
                    adname = poi.get("adname", "")
                    if not name:
                        continue

                    district = ""
                    if adname in DISTRICT_NAMES:
                        district = adname
                    else:
                        extracted = extract_district_from_text(adname)
                        if extracted:
                            district = extracted
                    if not district:
                        district = district_cn

                    records.append(PlaceRecord(
                        alias=normalize_name(name),
                        district=district,
                        category=cat,
                        confidence=0.92,
                        source=f"amap_poi_{cat}",
                    ))

                count = payload.get("count", "0")
                total = int(count) if count.isdigit() else 0
                if page * 25 >= total:
                    break
                time.sleep(random.uniform(0.12, 0.25))

    return records


# ──────────────────────────────────────────────
# 3. 地铁站爬虫（metroman.cn + bjsubway + 硬编码兜底）
# ──────────────────────────────────────────────
METROMAN_URL = "https://www.metroman.cn/cities/beijing/stations"

SUBWAY_LINE_URLS = [
    METROMAN_URL,
    "https://www.bjsubway.com/station/xltcx/",
    "https://www.bjsubway.com/station/xltcx_list.html",
]

SUBWAY_BLACKLIST = {
    "首页", "详情", "分钟", "票价", "复制", "路线", "拥挤", "方向",
    "信息公开", "联系我们", "输入站点名称", "关闭详情", "乘车方案",
    "出发时间", "返回", "取消", "关闭", "展开详情", "详情展开",
    "换乘", "运营", "出发", "完全程", "站内", "站外", "出入口",
    "无障碍", "卫生间", "自助", "客服", "安检", "查询", "更多",
    "线路", "站点", "首末车", "时刻表", "票价表", "线路图",
    "一卡通", "亿通行", "乘车码", "APP", "下载", "意见反馈",
    "版权", "京ICP", "技术支持", "浏览器", "推荐", "使用",
    "未找到相关数据",
}

SUBWAY_HARDCODED: list[str] = [
    "玉泉路", "五棵松", "万寿路", "公主坟", "军事博物馆", "木樨地", "南礼士路",
    "复兴门", "西单", "天安门西", "天安门东", "王府井", "东单", "建国门",
    "永安里", "国贸", "大望路", "四惠", "四惠东", "高碑店", "传媒大学",
    "双桥", "管庄", "八里桥", "通州北苑", "果园", "九棵树", "梨园",
    "临河里", "土桥", "花庄", "环球度假区",
    "西直门", "车公庄", "阜成门", "长椿街", "宣武门", "和平门", "前门",
    "崇文门", "北京站", "朝阳门", "东四十条", "东直门", "雍和宫", "安定门",
    "鼓楼大街", "积水潭",
    "工人体育场", "团结湖", "朝阳公园", "石佛营", "朝阳站", "姚家园",
    "东坝南", "东坝", "东坝北",
    "安河桥北", "北宫门", "西苑", "圆明园", "北京大学东门", "中关村",
    "海淀黄庄", "人民大学", "魏公村", "国家图书馆", "动物园", "新街口",
    "平安里", "西四", "灵境胡同", "菜市口", "陶然亭", "北京南站",
    "马家堡", "角门西", "公益西桥", "新宫", "西红门", "高米店北",
    "高米店南", "枣园", "清源路", "黄村西大街", "黄村火车站", "义和庄",
    "生物医药基地", "天宫院",
    "天通苑北", "天通苑", "天通苑南", "立水桥", "立水桥南", "北苑路北",
    "大屯路东", "惠新西街北口", "惠新西街南口", "和平西桥", "和平里北街",
    "北新桥", "张自忠路", "东四", "灯市口", "磁器口", "天坛东门",
    "蒲黄榆", "刘家窑", "宋家庄",
    "金安桥", "苹果园", "杨庄", "西黄村", "廖公庄", "田村", "海淀五路居",
    "慈寿寺", "花园桥", "白石桥南", "二里沟", "车公庄西", "北海北",
    "南锣鼓巷", "东大桥", "呼家楼", "金台路", "十里堡", "青年路",
    "褡裢坡", "黄渠", "常营", "草房", "物资学院路", "通州北关",
    "北运河西", "北运河东", "郝家府", "东夏园", "潞城", "潞阳",
    "北京西站", "湾子", "达官营", "广安门内", "虎坊桥", "珠市口",
    "桥湾", "广渠门内", "广渠门外", "双井", "九龙山", "大郊亭",
    "百子湾", "化工", "南楼梓庄", "欢乐谷景区", "垡头", "双合",
    "焦化厂", "黄厂", "郎辛庄", "黑庄户", "万盛西", "万盛东",
    "群芳", "高楼金",
    "朱辛庄", "育知路", "平西府", "回龙观东大街", "霍营", "育新",
    "西小口", "永泰庄", "林萃桥", "森林公园南门", "奥林匹克公园",
    "奥体中心", "北土城", "安华桥", "安德里北街", "什刹海",
    "中国美术馆", "金鱼胡同", "永定门外", "木樨园", "海户屯",
    "大红门", "大红门南", "和义", "东高地", "火箭万源", "五福堂",
    "德茂", "瀛海",
    "白堆子", "六里桥东", "六里桥", "七里庄", "丰台东大街", "丰台南路",
    "科怡路", "丰台科技园", "郭公庄",
    "巴沟", "苏州街", "知春里", "知春路", "西土城", "牡丹园",
    "健德门", "安贞门", "芍药居", "太阳宫", "三元桥", "亮马桥",
    "农业展览馆", "金台夕照", "劲松", "潘家园", "十里河", "分钟寺",
    "成寿寺", "石榴庄", "角门东", "草桥", "纪家庙", "首经贸",
    "丰台站", "泥洼", "西局", "莲花桥", "西钓鱼台", "车道沟",
    "长春桥", "火器营",
    "模式口", "北辛安", "新首钢",
    "四季青桥", "蓝靛厂", "苏州桥", "大钟寺", "蓟门桥", "北太平庄",
    "马甸桥", "安贞桥", "光熙门", "西坝河", "将台西", "高家园",
    "驼房营", "东坝西",
    "五道口", "上地", "清河站", "西二旗", "龙泽", "回龙观", "北苑",
    "望京西", "柳芳",
    "张郭庄", "园博园", "大瓦窑", "郭庄子", "大井", "东管头",
    "丽泽商务区", "菜户营", "西铁营", "景风门", "景泰", "方庄",
    "北工大西门", "平乐园", "枣营", "东风北桥", "将台", "望京南",
    "阜通", "望京", "东湖渠", "来广营", "善各庄",
    "俸伯", "顺义", "石门", "南法信", "后沙峪", "花梨坎", "国展",
    "孙河", "马泉营", "崔各庄", "望京东", "关庄", "安立路", "北沙滩",
    "六道口", "清华东路西口",
    "北安河", "温阳路", "稻香湖路", "屯佃", "永丰", "永丰南", "西北旺",
    "马连洼", "农大南路", "万泉河桥", "万寿寺", "甘家口", "玉渊潭东门",
    "红莲南路", "东管头南", "富丰桥", "看丹", "榆树庄", "洪泰庄", "宛平城",
    "未来科学城北", "未来科学城", "天通苑东", "清河营", "红军营",
    "左家庄", "潘家园西", "周家庄", "十八里店", "北神树", "次渠北",
    "次渠", "嘉会湖",
    "上地软件园", "东北旺", "龙泽西", "回龙观西大街", "文华路", "霍营东",
    "太平庄",
    "北太平庄", "太平桥", "牛街", "新发地",
    "肖村", "小红门", "旧宫", "亦庄桥", "亦庄文化园", "万源街",
    "荣京东街", "荣昌东街", "同济南路", "经海路", "次渠南", "亦庄火车站",
    "昌平西山口", "十三陵景区", "昌平", "昌平东关", "北邵洼", "南邵",
    "沙河高教园", "沙河", "巩华城", "生命科学园", "朱房北",
    "清河小营桥", "学知园", "学院桥",
    "花乡东桥", "白盆窑", "大葆台", "稻田", "长阳", "篱笆房",
    "广阳城", "良乡大学城北", "良乡大学城", "良乡大学城西", "良乡南关",
    "苏庄", "阎村东",
    "紫草坞", "阎村", "星城", "大石河东", "马各庄", "饶乐府",
    "房山城关", "燕山",
    "石厂", "小园", "栗园庄", "上岸", "桥户营", "四道桥",
    "香山", "国家植物园", "万安", "茶棚", "颐和园西门",
    "3号航站楼", "2号航站楼",
    "大兴新城", "大兴机场",
    "定海园", "定海园西", "经海一路", "亦创会展中心", "亦庄同仁",
    "鹿圈东", "泰河路", "九号村", "四海庄", "太和桥北", "瑞合庄",
    "融兴街", "屈庄",
]


def is_plausible_station(name: str) -> bool:
    if not name or len(name) < 2 or len(name) > 10:
        return False
    if name in SUBWAY_BLACKLIST:
        return False
    if re.search(r"[0-9]+分钟|方案|时间|拥挤|路段|运营|出发|换乘|完全程|时刻|票价", name):
        return False
    return bool(re.search(r"[\u4e00-\u9fa5]", name))


def _parse_metroman(html: str) -> set[str]:
    stations: set[str] = set()
    soup = BeautifulSoup(html, "lxml")
    for li in soup.select("li"):
        text = li.get_text(strip=True)
        parts = re.split(r"\s+", text, maxsplit=1)
        cn_name = parts[0].strip()
        if is_plausible_station(cn_name) and cn_name not in SUBWAY_BLACKLIST:
            stations.add(cn_name)
    for h2 in soup.select("h2"):
        line_text = h2.get_text(strip=True)
        if "号线" in line_text or "线" in line_text:
            next_ul = h2.find_next_sibling("ul")
            if next_ul:
                for li in next_ul.select("li"):
                    text = li.get_text(strip=True)
                    parts = re.split(r"\s+", text, maxsplit=1)
                    cn_name = parts[0].strip()
                    if is_plausible_station(cn_name):
                        stations.add(cn_name)
    return stations


def _parse_bjsubway(html: str) -> set[str]:
    stations: set[str] = set()
    soup = BeautifulSoup(html, "lxml")
    for a_tag in soup.select("a[href*='station'], a[href*='xltcx']"):
        text = a_tag.get_text(strip=True)
        cleaned = re.sub(r"[（(]地铁站[）)]", "", text).replace("地铁站", "").strip()
        if is_plausible_station(cleaned):
            stations.add(cleaned)
    for td in soup.select("td, th, span, div"):
        text = td.get_text(strip=True)
        if "站" in text and len(text) <= 10:
            cleaned = text.replace("站", "").strip()
            if is_plausible_station(cleaned):
                stations.add(cleaned)
    scripts = "\n".join(s.get_text(" ", strip=True) for s in soup.select("script"))
    for match in re.findall(r"[\u4e00-\u9fa5A-Za-z0-9]{2,10}站", scripts):
        cleaned = match.replace("站", "").strip()
        if is_plausible_station(cleaned):
            stations.add(cleaned)
    return stations


def crawl_subway(session: requests.Session) -> list[PlaceRecord]:
    all_stations: set[str] = set()
    runtime_dir = os.path.join(PROJECT_ROOT, "data", "runtime")

    subway_xlsx_path = os.path.join(runtime_dir, "subway_stations_official.xlsx")
    if not os.path.exists(subway_xlsx_path):
        print("[地铁] 下载北京市交通委官方轨道站点数据 ...")
        _download_xlsx(session, SUBWAY_XLSX_URL, subway_xlsx_path)

    if os.path.exists(subway_xlsx_path):
        print(f"[地铁] 解析官方 XLSX: {subway_xlsx_path}")
        recs = _parse_subway_xlsx(subway_xlsx_path)
        xlsx_names = {r.alias for r in recs}
        all_stations.update(xlsx_names)
        print(f"  官方数据: {len(recs)} 条")

    print(f"[地铁] 尝试 metroman.cn 补充 ...")
    html = safe_get(session, METROMAN_URL)
    if html and not looks_blocked(html):
        parsed = _parse_metroman(html)
        all_stations.update(parsed)
        print(f"  metroman.cn: 提取 {len(parsed)} 个站名")

    if len(all_stations) < 50:
        for url in SUBWAY_LINE_URLS[1:]:
            print(f"[地铁] 尝试 {url} ...")
            html = safe_get(session, url)
            if not html or looks_blocked(html):
                continue
            parsed = _parse_bjsubway(html)
            all_stations.update(parsed)
            print(f"  {url}: 提取 {len(parsed)} 个站名")

    if len(all_stations) < 50:
        print(f"[地铁] 在线爬取不足 50 站，启用硬编码兜底列表 ({len(SUBWAY_HARDCODED)} 站)")
        all_stations.update(SUBWAY_HARDCODED)

    print(f"[地铁] 共提取 {len(all_stations)} 个站名")

    records: list[PlaceRecord] = []
    for name in sorted(all_stations):
        records.append(PlaceRecord(
            alias=normalize_name(name),
            district="",
            category="subway_station",
            confidence=0.90,
            source="metroman_subway",
        ))
    return records


# ──────────────────────────────────────────────
# 4. 公交站爬虫（官方 XLSX + 8684 + 安居客公交频道）
# ──────────────────────────────────────────────
BUS_BLACKLIST_NAMES = {
    "更多", "下一页", "上一页", "首页", "尾页", "返回", "线路", "站点",
    "查询", "换乘", "地图", "下载", "APP", "意见反馈", "关于我们",
    "广告", "合作", "联系", "版权", "备案", "京ICP",
}

BUS_XLSX_URL = "https://service.jtw.beijing.gov.cn/cxsjxz/data/%E5%85%AC%E4%BA%A4%E7%AB%99%E7%82%B9%E4%BF%A1%E6%81%AF.xlsx"
SUBWAY_XLSX_URL = "https://service.jtw.beijing.gov.cn/cxsjxz/data/%E8%BD%A8%E9%81%93%E7%AB%99%E7%82%B9%E4%BF%A1%E6%81%AF.xlsx"


def _download_xlsx(session: requests.Session, url: str, save_path: str) -> bool:
    try:
        resp = session.get(url, headers=COMMON_HEADERS, timeout=60)
        resp.raise_for_status()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(resp.content)
        return True
    except Exception as e:
        print(f"  [下载] 失败: {e}")
        return False


def _parse_bus_xlsx(xlsx_path: str) -> list[PlaceRecord]:
    try:
        import openpyxl
    except ImportError:
        print("  [公交XLSX] 需要安装 openpyxl: pip install openpyxl")
        return []

    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    ws = wb.active

    station_names: set[str] = set()
    for row in ws.iter_rows(min_row=2, values_only=True):
        name = row[3] if len(row) > 3 else None
        if name and isinstance(name, str) and 2 <= len(name.strip()) <= 15:
            station_names.add(name.strip())
    wb.close()

    records: list[PlaceRecord] = []
    for name in sorted(station_names):
        records.append(PlaceRecord(
            alias=normalize_name(name),
            district="",
            category="bus_stop",
            confidence=0.90,
            source="beijing_gov_bus_xlsx",
        ))
    return records


def _parse_subway_xlsx(xlsx_path: str) -> list[PlaceRecord]:
    try:
        import openpyxl
    except ImportError:
        print("  [地铁XLSX] 需要安装 openpyxl: pip install openpyxl")
        return []

    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    ws = wb.active

    station_names: set[str] = set()
    for row in ws.iter_rows(min_row=2, values_only=True):
        name = row[2] if len(row) > 2 else None
        if name and isinstance(name, str) and 2 <= len(name.strip()) <= 15:
            station_names.add(name.strip())
    wb.close()

    records: list[PlaceRecord] = []
    for name in sorted(station_names):
        records.append(PlaceRecord(
            alias=normalize_name(name),
            district="",
            category="subway_station",
            confidence=0.95,
            source="beijing_gov_subway_xlsx",
        ))
    return records


def _crawl_8684(session: requests.Session) -> list[PlaceRecord]:
    records: list[PlaceRecord] = []
    base = "https://beijing.8684.cn"

    for district_cn in DISTRICT_NAMES:
        short = district_cn.replace("区", "")
        print(f"  [8684] {district_cn} ...")

        list_url = f"{base}/bus_stops/{short}"
        html = safe_get(session, list_url)
        if not html:
            continue

        soup = BeautifulSoup(html, "lxml")

        selectors = [
            "a[href*='/bus_stop/']",
            "a[href*='station']",
            ".bus-list a",
            ".stop-list a",
            "div.list a",
            "ul.list a",
            ".content a",
            ".wrapper a",
            "#bus_stop_list a",
        ]
        station_links: list = []
        for sel in selectors:
            found = soup.select(sel)
            if found:
                station_links = found
                break

        for link in station_links:
            name = link.get_text(strip=True)
            if not name or len(name) < 2 or len(name) > 15:
                continue
            if name in BUS_BLACKLIST_NAMES:
                continue
            if re.search(r"^\d+$", name):
                continue

            records.append(PlaceRecord(
                alias=normalize_name(name),
                district=district_cn,
                category="bus_stop",
                confidence=0.75,
                source="8684_bus_stop",
            ))

        time.sleep(random.uniform(0.8, 1.5))

    return records


def _crawl_anjuke_bus(session: requests.Session) -> list[PlaceRecord]:
    records: list[PlaceRecord] = []
    base = "https://beijing.anjuke.com/bus"

    for district_cn in DISTRICT_NAMES:
        short = district_cn.replace("区", "")
        print(f"  [安居客公交] {district_cn} ...")

        url = f"{base}/{short}"
        html = safe_get(session, url)
        if not html or looks_blocked(html):
            continue

        soup = BeautifulSoup(html, "lxml")
        for a_tag in soup.select("a[href*='station'], a[href*='bus'], .station-list a, .bus-list a"):
            name = a_tag.get_text(strip=True)
            if not name or len(name) < 2 or len(name) > 15:
                continue
            if name in BUS_BLACKLIST_NAMES:
                continue

            records.append(PlaceRecord(
                alias=normalize_name(name),
                district=district_cn,
                category="bus_stop",
                confidence=0.70,
                source="anjuke_bus_stop",
            ))

        time.sleep(random.uniform(1.0, 2.0))

    return records


def crawl_bus_stations(session: requests.Session, max_pages: int = 3) -> list[PlaceRecord]:
    records: list[PlaceRecord] = []
    runtime_dir = os.path.join(PROJECT_ROOT, "data", "runtime")

    bus_xlsx_path = os.path.join(runtime_dir, "bus_stations_official.xlsx")
    if not os.path.exists(bus_xlsx_path):
        print("[公交站] 下载北京市交通委官方公交站点数据 ...")
        _download_xlsx(session, BUS_XLSX_URL, bus_xlsx_path)

    if os.path.exists(bus_xlsx_path):
        print(f"[公交站] 解析官方 XLSX: {bus_xlsx_path}")
        recs = _parse_bus_xlsx(bus_xlsx_path)
        print(f"  官方数据: {len(recs)} 条")
        records.extend(recs)
    else:
        print("[公交站] 官方 XLSX 不可用，尝试 8684.cn ...")
        recs = _crawl_8684(session)
        print(f"  8684: {len(recs)} 条")
        records.extend(recs)

        print("[公交站] 尝试安居客公交频道 ...")
        recs = _crawl_anjuke_bus(session)
        print(f"  安居客公交: {len(recs)} 条")
        records.extend(recs)

    return records


# ──────────────────────────────────────────────
# 5. 用高德 API 补充地铁站/公交站的区归属
# ──────────────────────────────────────────────
def resolve_districts_via_amap(
    session: requests.Session,
    records: list[PlaceRecord],
    amap_key: str,
) -> list[PlaceRecord]:
    if not amap_key:
        return records

    geocode_url = "https://restapi.amap.com/v3/geocode/geo"
    resolved: list[PlaceRecord] = []
    missing = [r for r in records if not r.district]
    print(f"[高德] 需要解析区的记录: {len(missing)} 条")

    for i, record in enumerate(missing):
        params = {"key": amap_key, "address": f"北京市{record.alias}", "city": "北京"}
        try:
            resp = session.get(geocode_url, params=params, headers=COMMON_HEADERS, timeout=15)
            resp.raise_for_status()
            payload = resp.json()
            geocodes = payload.get("geocodes") or []
            if geocodes:
                district = geocodes[0].get("district", "")
                if district in DISTRICT_NAMES:
                    record = PlaceRecord(
                        alias=record.alias,
                        district=district,
                        category=record.category,
                        confidence=record.confidence,
                        source=record.source,
                    )
        except Exception:
            pass

        resolved.append(record)
        if (i + 1) % 50 == 0:
            print(f"  [高德] 已解析 {i+1}/{len(missing)}")
        time.sleep(random.uniform(0.15, 0.35))

    already_has = [r for r in records if r.district]
    return already_has + resolved


# ──────────────────────────────────────────────
# 输出
# ──────────────────────────────────────────────
def write_catalog_jsonl(records: list[PlaceRecord], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def write_district_json(records: list[PlaceRecord], path: str) -> None:
    grouped: dict[str, list[str]] = defaultdict(list)
    for r in records:
        if r.district and r.alias:
            grouped[r.district].append(r.alias)
    result = {d: sorted(set(names)) for d, names in grouped.items()}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def merge_into_existing_catalog(new_records: list[PlaceRecord], catalog_path: str) -> int:
    existing_keys: set[tuple[str, str, str]] = set()
    if os.path.exists(catalog_path):
        with open(catalog_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                existing_keys.add((d.get("alias", ""), d.get("district", ""), d.get("category", "")))

    added = 0
    with open(catalog_path, "a", encoding="utf-8") as f:
        for r in new_records:
            key = (r.alias, r.district, r.category)
            if key not in existing_keys:
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
                existing_keys.add(key)
                added += 1
    return added


def print_stats(records: list[PlaceRecord]) -> None:
    cat_counts: dict[str, int] = defaultdict(int)
    dist_counts: dict[str, int] = defaultdict(int)
    source_counts: dict[str, int] = defaultdict(int)
    for r in records:
        cat_counts[r.category] += 1
        if r.district:
            dist_counts[r.district] += 1
        source_counts[r.source] += 1

    print("\n" + "=" * 60)
    print("爬取统计")
    print("=" * 60)
    print(f"总记录数: {len(records)}")
    print("\n按类别:")
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {cnt}")
    print("\n按来源:")
    for src, cnt in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"  {src}: {cnt}")
    print("\n按区:")
    for dist, cnt in sorted(dist_counts.items(), key=lambda x: -x[1]):
        print(f"  {dist}: {cnt}")
    no_district = sum(1 for r in records if not r.district)
    if no_district:
        print(f"  (无区归属): {no_district}")


def main() -> None:
    parser = argparse.ArgumentParser(description="北京地名爬虫：小区 + 地铁站 + 公交站 + 公园/商场/政府/河流/桥梁等")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["subway", "bus", "amap"],
        choices=["lianjia", "anjuke", "subway", "bus", "amap"],
        help="要爬取的数据源（默认 subway+bus+amap；lianjia/anjuke 可能被反爬拦截）",
    )
    parser.add_argument(
        "--amap-categories",
        nargs="+",
        default=None,
        choices=list(AMAP_POI_TYPES.keys()),
        help="高德POI要爬取的类别（默认全部）",
    )
    parser.add_argument("--max-pages", type=int, default=5, help="每个区最大翻页数")
    parser.add_argument("--dry-run", action="store_true", help="只打印统计不写文件")
    parser.add_argument("--merge", action="store_true", help="直接合并到现有 place_alias_catalog.jsonl")
    args = parser.parse_args()

    amap_key = os.getenv("AMAP_API_KEY", "").strip()
    session = requests.Session()
    all_records: list[PlaceRecord] = []

    if "lianjia" in args.sources:
        recs = crawl_lianjia(session, args.max_pages)
        print(f"  链家: {len(recs)} 条")
        all_records.extend(recs)

    if "anjuke" in args.sources:
        recs = crawl_anjuke(session, args.max_pages)
        print(f"  安居客: {len(recs)} 条")
        all_records.extend(recs)

    if "amap" in args.sources:
        recs = crawl_amap_poi(session, amap_key, categories=args.amap_categories)
        print(f"  高德POI: {len(recs)} 条")
        all_records.extend(recs)

    if "subway" in args.sources:
        recs = crawl_subway(session)
        print(f"  地铁站: {len(recs)} 条")
        all_records.extend(recs)

    if "bus" in args.sources:
        recs = crawl_bus_stations(session, args.max_pages)
        print(f"  公交站: {len(recs)} 条")
        all_records.extend(recs)

    all_records = deduplicate(all_records)

    missing_district = sum(1 for r in all_records if not r.district)
    if missing_district > 0 and amap_key:
        print(f"\n[高德] 补充 {missing_district} 条记录的区归属...")
        all_records = resolve_districts_via_amap(session, all_records, amap_key)

    print_stats(all_records)

    if args.dry_run:
        return

    output_dir = os.path.join(PROJECT_ROOT, "data", "runtime")
    jsonl_path = os.path.join(output_dir, "crawled_places.jsonl")
    district_path = os.path.join(output_dir, "beijing_districts_crawled.json")

    write_catalog_jsonl(all_records, jsonl_path)
    write_district_json(all_records, district_path)
    print(f"\n结果已写入:")
    print(f"  {jsonl_path}")
    print(f"  {district_path}")

    if args.merge:
        catalog_path = os.path.join(output_dir, "place_alias_catalog.jsonl")
        added = merge_into_existing_catalog(all_records, catalog_path)
        print(f"\n已合并到 {catalog_path}，新增 {added} 条（去重后）")


if __name__ == "__main__":
    main()
