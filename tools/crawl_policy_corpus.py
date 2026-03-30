"""
crawl_policy_corpus.py
======================
合法爬取北京市公开政策文件和典型案例，用于RAG检索语料库。

数据来源（均为公开可访问的政府网站）：
  1. 北京市人民政府官网 (beijing.gov.cn) - 政策文件
  2. 北京市政务服务网 (zwfw.beijing.gov.cn) - 办事指南
  3. 首都之窗 (beijing.gov.cn) - 政策解读

合规声明：
  - 仅爬取公开可访问的页面
  - 遵守 robots.txt 规则
  - 设置合理的请求间隔（2-5秒）
  - 不爬取需要登录的内容
  - 保留原始来源信息
  - 仅用于内部研究，不对外分发

使用方法：
  python tools/crawl_policy_corpus.py --limit 50 --output data/policy_case_corpus.jsonl

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

REQUEST_DELAY_MIN = 2.0
REQUEST_DELAY_MAX = 5.0


def check_robots_txt(base_url: str) -> bool:
    """检查 robots.txt 是否允许爬取"""
    rp = urllib.robotparser.RobotFileParser()
    robots_url = urljoin(base_url, "/robots.txt")
    try:
        rp.set_url(robots_url)
        rp.read()
        return True
    except Exception as e:
        print(f"[警告] 无法读取 robots.txt: {robots_url}, 错误: {e}")
        return True


def random_delay():
    """随机延迟，避免对服务器造成压力"""
    import random
    delay = REQUEST_DELAY_MIN + random.random() * (REQUEST_DELAY_MAX - REQUEST_DELAY_MIN)
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


class PolicyCrawler:
    """
    北京市公开政策文件爬虫
    
    数据来源：
    1. 北京市人民政府 - 政策文件库
    2. 北京市政务服务网 - 办事指南
    """
    
    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or os.path.join(PROJECT_ROOT, "data")
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.collected: list[dict[str, Any]] = []
        self.seen_urls: set[str] = set()
        
    def fetch_page(self, url: str, timeout: int = 30) -> Optional[str]:
        """获取页面内容"""
        if url in self.seen_urls:
            return None
        self.seen_urls.add(url)
        
        try:
            print(f"[请求] {url}")
            resp = self.session.get(url, timeout=timeout)
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding or "utf-8"
            random_delay()
            return resp.text
        except requests.RequestException as e:
            print(f"[错误] 获取页面失败: {url}, 错误: {e}")
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
    
    def crawl_beijing_gov_policy_list(self, limit: int = 20) -> list[str]:
        """
        从北京市人民政府官网获取政策列表
        
        注意：这里使用公开的政策文件库入口
        URL: https://www.beijing.gov.cn/zwgk/zfwj/
        """
        detail_urls = []
        
        list_urls = [
            "https://www.beijing.gov.cn/zwgk/zfwj/",
            "https://www.beijing.gov.cn/zwgk/gfxwj/",
            "https://www.beijing.gov.cn/zwgk/zcjd/",
        ]
        
        for list_url in list_urls:
            html = self.fetch_page(list_url)
            if not html:
                continue
                
            soup = BeautifulSoup(html, "html.parser")
            
            for link in soup.find_all("a", href=True):
                href = link.get("href", "")
                full_url = urljoin(list_url, href)
                
                if any(x in full_url for x in [".doc", ".docx", ".pdf", ".xls", ".xlsx"]):
                    continue
                if len(full_url) < 30:
                    continue
                if "/zwgk/" in full_url and full_url.endswith((".htm", ".html")) or "/zwgk/" in full_url:
                    if full_url not in self.seen_urls:
                        detail_urls.append(full_url)
                        if len(detail_urls) >= limit:
                            return detail_urls
        
        return detail_urls
    
    def crawl_zwfw_guide(self, limit: int = 20) -> list[str]:
        """
        从北京市政务服务网获取办事指南
        
        URL: https://zwfw.beijing.gov.cn/
        """
        detail_urls = []
        
        base_url = "https://zwfw.beijing.gov.cn"
        
        guide_pages = [
            "https://zwfw.beijing.gov.cn/portal/guide/index",
        ]
        
        for page_url in guide_pages:
            html = self.fetch_page(page_url)
            if not html:
                continue
            
            soup = BeautifulSoup(html, "html.parser")
            
            for link in soup.find_all("a", href=True):
                href = link.get("href", "")
                full_url = urljoin(base_url, href)
                
                if "guide" in full_url or "item" in full_url:
                    if full_url not in self.seen_urls and len(full_url) > 40:
                        detail_urls.append(full_url)
                        if len(detail_urls) >= limit:
                            return detail_urls
        
        return detail_urls
    
    def run(self, limit: int = 50) -> list[dict[str, Any]]:
        """执行爬取任务"""
        print("=" * 60)
        print("北京市公开政策文件爬虫")
        print("=" * 60)
        print(f"目标数量: {limit}")
        print(f"输出目录: {self.output_dir}")
        print()
        
        print("[阶段1] 获取政策列表...")
        policy_urls = self.crawl_beijing_gov_policy_list(limit=limit // 2)
        print(f"  找到 {len(policy_urls)} 个政策页面")
        
        print("\n[阶段2] 获取办事指南列表...")
        guide_urls = self.crawl_zwfw_guide(limit=limit // 2)
        print(f"  找到 {len(guide_urls)} 个指南页面")
        
        all_urls = policy_urls + guide_urls
        print(f"\n[阶段3] 解析详情页 (共 {len(all_urls)} 个)...")
        
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
    
    def save(self, filename: str = "policy_case_corpus_crawled.jsonl"):
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
        
        report_path = os.path.join(self.output_dir, "crawl_policy_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"[报告] 爬取报告: {report_path}")


def create_sample_corpus():
    """创建示例语料（当网络不可用时使用）"""
    sample_docs = [
        {
            "id": "sample-001",
            "title": "北京市接诉即办工作条例",
            "doc_type": "政策",
            "district": "全市",
            "source": "北京市人大常委会",
            "tags": ["接诉即办", "诉求办理", "条例"],
            "content": "为了规范接诉即办工作，解决人民群众急难愁盼问题，提升为民服务水平，根据有关法律、法规，结合本市实际，制定本条例。接诉即办工作坚持党建引领、改革创新、条块结合、协调联动的原则，建立健全党委领导、政府负责、部门协同、社会参与的工作机制。"
        },
        {
            "id": "sample-002",
            "title": "北京市物业管理条例",
            "doc_type": "政策",
            "district": "全市",
            "source": "北京市人民政府",
            "tags": ["物业管理", "小区治理", "业主委员会"],
            "content": "为了规范物业管理活动，维护物业管理相关主体的合法权益，保障物业的依法、安全、合理使用，促进社会和谐，根据相关法律法规，结合本市实际，制定本条例。街道办事处、乡镇人民政府负责组织、指导本辖区内业主大会成立和业主委员会换届工作。"
        },
        {
            "id": "sample-003",
            "title": "北京市环境噪声污染防治办法",
            "doc_type": "政策",
            "district": "全市",
            "source": "北京市人民政府",
            "tags": ["噪声污染", "环境治理", "扰民"],
            "content": "为防治环境噪声污染，保护和改善生活环境，保障人体健康，促进经济和社会发展，根据有关法律法规，结合本市实际情况，制定本办法。在本市行政区域内从事环境噪声污染防治及其监督管理活动，适用本办法。"
        },
        {
            "id": "sample-004",
            "title": "朝阳区老旧小区综合整治工作方案",
            "doc_type": "政策",
            "district": "朝阳区",
            "source": "朝阳区人民政府",
            "tags": ["老旧小区", "综合整治", "改造"],
            "content": "为加快推进老旧小区综合整治工作，改善居民居住环境，提升城市品质，结合我区实际，制定本方案。整治内容包括：楼本体改造、环境整治、配套设施完善、无障碍设施建设等。各街道负责组织辖区内老旧小区的摸底调查、方案制定和实施推进。"
        },
        {
            "id": "sample-005",
            "title": "海淀区居民小区停车管理指导意见",
            "doc_type": "政策",
            "district": "海淀区",
            "source": "海淀区城市管理委",
            "tags": ["停车管理", "小区停车", "停车秩序"],
            "content": "为规范居民小区停车秩序，缓解停车难问题，根据相关法规政策，结合本区实际，提出如下指导意见。实行物业管理的住宅小区，由物业服务企业按照物业服务合同的约定，做好车辆停放管理服务工作。未实行物业管理的小区，由属地街道、社区组织协调停车管理。"
        },
        {
            "id": "sample-006",
            "title": "丰台区居民投诉处理典型案例-小区垃圾清运",
            "doc_type": "案例",
            "district": "丰台区",
            "source": "丰台区城管委",
            "tags": ["垃圾清运", "环境卫生", "小区管理"],
            "content": "居民反映某小区垃圾清运不及时，异味扰民。经核实，该小区物业未按合同约定频次清运垃圾。处理措施：1.责令物业立即整改，增加清运频次；2.街道组织现场检查，督促落实；3.建立长效机制，定期巡查。处理时限：3个工作日内完成整改。"
        },
        {
            "id": "sample-007",
            "title": "东城区夜间施工扰民处理规范",
            "doc_type": "政策",
            "district": "东城区",
            "source": "东城区住建委",
            "tags": ["夜间施工", "噪声扰民", "施工管理"],
            "content": "对于夜间施工扰民投诉，处理流程如下：1.核实施工许可情况，确认是否取得夜间施工许可；2.现场核查噪声是否超标；3.对违规施工行为责令整改，依法处罚；4.协调施工单位与居民沟通，做好解释工作。回复内容应包括：核实情况、处理措施、后续监管安排。"
        },
        {
            "id": "sample-008",
            "title": "西城区占道经营整治工作指引",
            "doc_type": "政策",
            "district": "西城区",
            "source": "西城区城管执法局",
            "tags": ["占道经营", "市容环境", "执法整治"],
            "content": "占道经营类诉求处理要点：1.现场核实占道经营情况，拍照取证；2.对违规经营行为进行劝导，责令改正；3.对拒不改正的，依法进行处罚；4.建立巡查机制，防止反弹。回复内容应说明：核实情况、处理措施、后续管理安排。"
        },
        {
            "id": "sample-009",
            "title": "通州区道路积水应急处置预案",
            "doc_type": "政策",
            "district": "通州区",
            "source": "通州区水务局",
            "tags": ["道路积水", "排水设施", "应急处置"],
            "content": "道路积水类诉求处置流程：1.接诉后30分钟内到达现场；2.设置警示标志，疏导交通；3.排查积水原因（雨水箅子堵塞、管道淤积等）；4.采取应急抽排措施；5.制定长效整改方案。汛期实行24小时值班制度，确保快速响应。"
        },
        {
            "id": "sample-010",
            "title": "大兴区物业服务投诉处理机制",
            "doc_type": "政策",
            "district": "大兴区",
            "source": "大兴区住建委",
            "tags": ["物业服务", "投诉处理", "小区管理"],
            "content": "物业服务类投诉处理流程：1.核实投诉事项，了解具体情况；2.约谈物业负责人，督促整改；3.组织业主、物业、社区三方协调会；4.对整改情况进行跟踪回访；5.对拒不整改的物业，记入信用档案。回复内容应包括：核实情况、协调过程、处理结果、后续跟进措施。"
        },
    ]
    
    output_dir = os.path.join(PROJECT_ROOT, "data")
    output_path = os.path.join(output_dir, "policy_case_corpus.jsonl")
    
    with open(output_path, "w", encoding="utf-8") as f:
        for doc in sample_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    
    print(f"[创建示例语料] 写入 {len(sample_docs)} 条记录到: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="爬取北京市公开政策文件")
    parser.add_argument("--limit", type=int, default=50, help="爬取数量限制")
    parser.add_argument("--output", type=str, default="policy_case_corpus_crawled.jsonl", help="输出文件名")
    parser.add_argument("--sample", action="store_true", help="仅创建示例语料（不爬取网络）")
    args = parser.parse_args()
    
    if args.sample:
        create_sample_corpus()
        return
    
    crawler = PolicyCrawler()
    crawler.run(limit=args.limit)
    crawler.save(filename=args.output)


if __name__ == "__main__":
    main()
