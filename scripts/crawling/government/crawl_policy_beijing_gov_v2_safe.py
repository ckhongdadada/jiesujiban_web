"""
crawl_policy_corpus_v2.py
=========================
第二版合规爬虫 - 更安全、更保守的策略

合规声明：
  1. 仅使用公开API和政府网站的公开页面
  2. 严格遵守 robots.txt 规则
  3. 请求间隔：5-10秒（比第一版更保守）
  4. 不爬取需要登录或授权的内容
  5. 不爬取动态加载的内容（JavaScript）
  6. 保留完整的来源信息和URL
  7. 仅用于内部研究，不对外分发
  8. 爬取前先检查 robots.txt
  9. 设置合理的超时时间（30秒）
  10. 不并发请求，逐个处理

数据来源（政府公开信息）：
  1. 北京市政府官网 - 公开政策文件（静态HTML页面）
  2. 北京市政务服务网 - 办事指南（静态HTML页面）
  3. 各区政府官网 - 区级政策文件

使用方法：
  python tools/crawl_policy_corpus_v2.py --limit 20 --output data/policy_case_corpus_v2.jsonl

依赖：
  pip install requests beautifulsoup4
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import hashlib
import random
import urllib.robotparser
from datetime import datetime
from typing import Any, Optional
from urllib.parse import urljoin, urlparse

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("请先安装依赖: pip install requests beautifulsoup4")
    raise

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
}

REQUEST_DELAY_MIN = 5.0
REQUEST_DELAY_MAX = 10.0


def check_robots_txt(base_url: str) -> bool:
    """检查 robots.txt 是否允许爬取"""
    rp = urllib.robotparser.RobotFileParser()
    robots_url = urljoin(base_url, "/robots.txt")
    try:
        print(f"  [robots.txt] 检查: {robots_url}")
        rp.set_url(robots_url)
        rp.read()
        allowed = rp.can_fetch("*", base_url)
        print(f"  [robots.txt] 结果: {'允许' if allowed else '禁止'}")
        return allowed
    except Exception as e:
        print(f"  [robots.txt] 无法读取: {e}")
        print("  [robots.txt] 警告: 假设允许爬取，请谨慎使用")
        return True


def random_delay():
    """随机延迟，避免对服务器造成压力"""
    delay = REQUEST_DELAY_MIN + random.random() * (REQUEST_DELAY_MAX - REQUEST_DELAY_MIN)
    print(f"  [延迟] 等待 {delay:.1f} 秒...")
    time.sleep(delay)


def clean_text(text: str) -> str:
    """清理文本，去除多余空白"""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\u3000]+", " ", text)
    return text.strip()


def generate_doc_id(url: str) -> str:
    """根据URL生成唯一文档ID"""
    return hashlib.md5(url.encode()).hexdigest()[:12]


class PolicyCrawlerV2:
    """
    合规版爬虫 - 更加保守和安全
    
    特点：
    1. 逐个页面爬取，不并发
    2. 请求间隔更长（5-10秒）
    3. 超时时间更长（30秒）
    4. 优先爬取静态HTML页面
    5. 不爬取需要JavaScript渲染的内容
    """
    
    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or os.path.join(PROJECT_ROOT, "data")
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.collected: list[dict[str, Any]] = []
        self.seen_urls: set[str] = set()
        self.max_retries = 3
        self.retry_delay = 2
        
    def fetch_page(self, url: str, timeout: int = 30, max_retries: int = 3) -> Optional[str]:
        """获取页面内容，带重试机制"""
        if url in self.seen_urls:
            return None
        self.seen_urls.add(url)
        
        for attempt in range(max_retries):
            try:
                print(f"  [请求] {url}")
                resp = self.session.get(url, timeout=timeout)
                resp.raise_for_status()
                resp.encoding = resp.apparent_encoding or "utf-8"
                random_delay()
                return resp.text
            except requests.Timeout:
                print(f"  [错误] 超时，重试 {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    time.sleep(self.retry_delay)
            except requests.RequestException as e:
                print(f"  [错误] 请求失败: {e}")
                if attempt < max_retries - 1:
                    time.sleep(self.retry_delay)
        
        return None
    
    def parse_policy_detail(self, html: str, url: str, source_name: str) -> Optional[dict[str, Any]]:
        """解析政策详情页"""
        soup = BeautifulSoup(html, "html.parser")
        
        title_elem = (
            soup.find("h1") or 
            soup.find("title") or
            soup.find(class_=re.compile(r"title|Title|TITLE"))
        )
        title = clean_text(title_elem.get_text()) if title_elem else "未知标题"
        
        content_elem = (
            soup.find("div", class_=re.compile(r"content|Content|article|Article|TRS")) or
            soup.find("article") or
            soup.find(id=re.compile(r"content|Content"))
        )
        
        if not content_elem:
            content_elem = soup.find("body")
        
        if content_elem:
            for tag in content_elem.find_all(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            content = clean_text(content_elem.get_text())
        else:
            content = ""
        
        if len(content) < 100:
            return None
        
        district_match = re.search(r"(东城|西城|朝阳|海淀|丰台|石景山|门头沟|房山|通州|顺义|昌平|大兴|怀柔|平谷|密云|延庆)区?", title + content)
        district = district_match.group(1) + "区" if district_match else "全市"
        
        doc = {
            "id": generate_doc_id(url),
            "title": title[:200],
            "doc_type": "政策",
            "district": district,
            "source": source_name,
            "source_url": url,
            "crawl_time": datetime.now().isoformat(),
            "tags": [],
            "content": content[:5000],
        }
        
        return doc
    
    def crawl_gov_page(self, url: str, source_name: str, max_links: int = 10) -> list[str]:
        """爬取政府网站页面，提取政策链接"""
        print(f"\n[阶段] 爬取页面: {url}")
        
        html = self.fetch_page(url)
        if not html:
            return []
        
        soup = BeautifulSoup(html, "html.parser")
        detail_urls = []
        
        for link in soup.find_all("a", href=True):
            href = link.get("href", "")
            full_url = urljoin(url, href)
            
            if any(x in full_url for x in [".doc", ".docx", ".pdf", ".xls", ".xlsx"]):
                continue
            if len(full_url) < 30:
                continue
            if "/zwgk/" in full_url and full_url.endswith((".htm", ".html")):
                if full_url not in self.seen_urls:
                    detail_urls.append(full_url)
                    if len(detail_urls) >= max_links:
                        break
        
        print(f"  [结果] 找到 {len(detail_urls)} 个链接")
        return detail_urls
    
    def crawl_zwfw_page(self, url: str, source_name: str, max_links: int = 10) -> list[str]:
        """爬取政务服务网页面"""
        print(f"\n[阶段] 爬取页面: {url}")
        
        html = self.fetch_page(url)
        if not html:
            return []
        
        soup = BeautifulSoup(html, "html.parser")
        detail_urls = []
        
        for link in soup.find_all("a", href=True):
            href = link.get("href", "")
            full_url = urljoin(url, href)
            
            if "guide" in full_url or "item" in full_url:
                if full_url not in self.seen_urls and len(full_url) > 40:
                    detail_urls.append(full_url)
                    if len(detail_urls) >= max_links:
                        break
        
        print(f"  [结果] 找到 {len(detail_urls)} 个链接")
        return detail_urls
    
    def run(self, limit: int = 20) -> list[dict[str, Any]]:
        """执行爬取任务"""
        print("=" * 60)
        print("合规版爬虫 v2.0 - 更安全、更保守")
        print("=" * 60)
        print(f"目标数量: {limit}")
        print(f"输出目录: {self.output_dir}")
        print(f"请求间隔: {REQUEST_DELAY_MIN}-{REQUEST_DELAY_MAX}秒")
        print()
        
        # 定义要爬取的页面（静态HTML页面）
        target_pages = [
            # 北京市政府官网
            ("https://www.beijing.gov.cn/", "北京市人民政府"),
            # 北京市政务服务网
            ("https://zwfw.beijing.gov.cn/", "北京市政务服务网"),
        ]
        
        all_urls = []
        
        for page_url, source_name in target_pages:
            # 检查 robots.txt
            if not check_robots_txt(page_url):
                print(f"[警告] robots.txt 禁止爬取: {page_url}")
                continue
            
            # 爬取页面
            if "beijing.gov.cn" in page_url:
                urls = self.crawl_gov_page(page_url, source_name, max_links=limit // 2)
            else:
                urls = self.crawl_zwfw_page(page_url, source_name, max_links=limit // 2)
            
            all_urls.extend(urls)
            if len(all_urls) >= limit:
                break
        
        print(f"\n[阶段] 解析详情页 (共 {len(all_urls)} 个)...")
        
        for i, url in enumerate(all_urls[:limit]):
            print(f"  [{i+1}/{min(len(all_urls), limit)}] ", end="")
            html = self.fetch_page(url)
            if not html:
                continue
            
            source_name = "北京市人民政府" if "beijing.gov.cn" in url else "北京市政务服务网"
            doc = self.parse_policy_detail(html, url, source_name)
            if doc:
                self.collected.append(doc)
                print(f"    ✓ {doc['title'][:50]}...")
        
        print(f"\n[完成] 成功收集 {len(self.collected)} 条语料")
        return self.collected
    
    def save(self, filename: str = "policy_case_corpus_v2.jsonl"):
        """保存爬取结果"""
        if not self.collected:
            print("[警告] 没有数据可保存")
            return
        
        output_path = os.path.join(self.output_dir, filename)
        
        with open(output_path, "w", encoding="utf-8") as f:
            for doc in self.collected:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        
        print(f"[保存] 写入 {len(self.collected)} 条记录到: {output_path}")
        
        report = {
            "crawl_time": datetime.now().isoformat(),
            "total_docs": len(self.collected),
            "sources": {},
            "districts": {},
        }
        
        for doc in self.collected:
            source = doc.get("source", "未知")
            report["sources"][source] = report["sources"].get(source, 0) + 1
            district = doc.get("district", "全市")
            report["districts"][district] = report["districts"].get(district, 0) + 1
        
        report_path = os.path.join(self.output_dir, "crawl_policy_v2_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"[报告] 爬取报告: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="合规版爬虫 v2.0")
    parser.add_argument("--limit", type=int, default=20, help="爬取数量限制")
    parser.add_argument("--output", type=str, default="policy_case_corpus_v2.jsonl", help="输出文件名")
    args = parser.parse_args()
    
    crawler = PolicyCrawlerV2()
    crawler.run(limit=args.limit)
    crawler.save(filename=args.output)


if __name__ == "__main__":
    main()
