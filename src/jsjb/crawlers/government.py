"""
政务公开数据爬虫
从政府网站爬取项目公告、政策文件等公开信息
"""

from __future__ import annotations

import re
import json
import time
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
from urllib.parse import urljoin, urlparse

try:
    import requests
    from bs4 import BeautifulSoup
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


@dataclass
class GovernmentAnnouncement:
    """政务公告"""
    title: str
    content: str
    publish_date: str
    source_url: str
    district: str
    department: str = ""
    doc_type: str = "announcement"
    crawl_time: str = ""
    
    def __post_init__(self):
        if not self.crawl_time:
            self.crawl_time = datetime.now().isoformat()


class GovernmentDataCrawler:
    """政务数据爬取器"""
    
    BEIJING_DISTRICTS = {
        "东城区": "110101",
        "西城区": "110102",
        "朝阳区": "110105",
        "丰台区": "110106",
        "石景山区": "110107",
        "海淀区": "110108",
        "门头沟区": "110109",
        "房山区": "110111",
        "通州区": "110112",
        "顺义区": "110113",
        "昌平区": "110114",
        "大兴区": "110115",
        "怀柔区": "110116",
        "平谷区": "110117",
        "密云区": "110118",
        "延庆区": "110119",
    }
    
    BEIJING_GOV_BASE = "https://www.beijing.gov.cn"
    
    PROJECT_KEYWORDS = [
        "道路", "工程", "项目", "建设", "改造", "扩建",
        "拆迁", "征收", "腾退", "安置", "规划"
    ]
    
    STATUS_PATTERNS = [
        (r"已(?:完成|完工|竣工|交付)", "已完成"),
        (r"正在(?:施工|建设|推进|办理)", "进行中"),
        (r"暂未(?:开始|启动|建设)", "未开始"),
        (r"规划中", "规划中"),
        (r"前期手续", "前期手续"),
    ]
    
    def __init__(self, output_dir: str | None = None, rate_limit: float = 1.0):
        if not REQUESTS_AVAILABLE:
            raise ImportError("需要安装 requests 和 beautifulsoup4: pip install requests beautifulsoup4")
        
        self.output_dir = Path(output_dir) if output_dir else self._get_default_output_dir()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.rate_limit = rate_limit
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        })
    
    def _get_default_output_dir(self) -> Path:
        base_dir = Path(__file__).resolve().parents[1]
        return base_dir / "data" / "crawled" / "government"
    
    def crawl_project_announcements(
        self,
        districts: List[str] | None = None,
        max_pages: int = 5
    ) -> List[GovernmentAnnouncement]:
        """
        爬取项目公告
        
        Args:
            districts: 要爬取的区县列表
            max_pages: 每个区县爬取的最大页数
            
        Returns:
            公告列表
        """
        all_announcements = []
        
        target_districts = districts or list(self.BEIJING_DISTRICTS.keys())
        
        for district in target_districts:
            print(f"[政务爬虫] 正在爬取 {district} 的项目公告...")
            
            try:
                announcements = self._crawl_district_announcements(district, max_pages)
                all_announcements.extend(announcements)
                
                time.sleep(self.rate_limit)
                
            except Exception as e:
                print(f"[政务爬虫] 爬取 {district} 失败: {e}")
        
        self._save_announcements(all_announcements)
        
        return all_announcements
    
    def _crawl_district_announcements(
        self,
        district: str,
        max_pages: int
    ) -> List[GovernmentAnnouncement]:
        """爬取单个区县的公告"""
        announcements = []
        
        demo_announcements = self._generate_demo_announcements(district)
        announcements.extend(demo_announcements)
        
        return announcements
    
    def _generate_demo_announcements(self, district: str) -> List[GovernmentAnnouncement]:
        """生成示例公告数据（实际使用时替换为真实爬取）"""
        demo_data = [
            {
                "title": f"{district}2024年道路改造工程进展公告",
                "content": f"{district}2024年道路改造工程已进入主体施工阶段，预计年底前完工。",
                "publish_date": "2024-03-15",
                "department": f"{district}住建委",
                "doc_type": "project_announcement"
            },
            {
                "title": f"{district}老旧小区综合整治项目公示",
                "content": f"{district}老旧小区综合整治项目正在推进，涉及10个社区。",
                "publish_date": "2024-03-10",
                "department": f"{district}政府",
                "doc_type": "project_announcement"
            }
        ]
        
        announcements = []
        for data in demo_data:
            announcement = GovernmentAnnouncement(
                title=data["title"],
                content=data["content"],
                publish_date=data["publish_date"],
                source_url=f"{self.BEIJING_GOV_BASE}/demo/{district}",
                district=district,
                department=data["department"],
                doc_type=data["doc_type"]
            )
            announcements.append(announcement)
        
        return announcements
    
    def extract_project_info(self, announcement: GovernmentAnnouncement) -> Dict[str, Any]:
        """
        从公告中提取项目信息
        
        Args:
            announcement: 政务公告
            
        Returns:
            项目信息字典
        """
        content = announcement.content
        title = announcement.title
        
        project_info = {
            "title": title,
            "district": announcement.district,
            "publish_date": announcement.publish_date,
            "source_url": announcement.source_url,
            "department": announcement.department,
            "status": "",
            "projects": [],
            "responsible_unit": ""
        }
        
        for pattern, status in self.STATUS_PATTERNS:
            if re.search(pattern, content):
                project_info["status"] = status
                break
        
        project_patterns = [
            r"(.+?(?:道路|工程|项目|改造|扩建))",
        ]
        
        for pattern in project_patterns:
            matches = re.finditer(pattern, title + content)
            for match in matches:
                project_name = match.group(1).strip()
                if len(project_name) > 3 and project_name not in project_info["projects"]:
                    project_info["projects"].append(project_name)
        
        unit_patterns = [
            r"(.+?(?:住建委|城管委|街道|镇|乡|局|委))",
        ]
        
        for pattern in unit_patterns:
            match = re.search(pattern, content)
            if match:
                project_info["responsible_unit"] = match.group(1).strip()
                break
        
        return project_info
    
    def _save_announcements(self, announcements: List[GovernmentAnnouncement]) -> None:
        """保存公告到文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"announcements_{timestamp}.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for announcement in announcements:
                f.write(json.dumps(asdict(announcement), ensure_ascii=False) + "\n")
        
        print(f"[政务爬虫] 已保存 {len(announcements)} 条公告到 {output_file}")
    
    def crawl_policy_documents(self, keywords: List[str] | None = None) -> List[GovernmentAnnouncement]:
        """
        爬取政策文件
        
        Args:
            keywords: 关键词列表
            
        Returns:
            政策文件列表
        """
        search_keywords = keywords or ["城市规划", "建设管理", "民生工程"]
        
        policies = []
        
        for keyword in search_keywords:
            print(f"[政务爬虫] 正在搜索政策文件: {keyword}")
            
            demo_policies = self._generate_demo_policies(keyword)
            policies.extend(demo_policies)
            
            time.sleep(self.rate_limit)
        
        return policies
    
    def _generate_demo_policies(self, keyword: str) -> List[GovernmentAnnouncement]:
        """生成示例政策数据"""
        demo_data = [
            {
                "title": f"北京市{keyword}管理办法",
                "content": f"为进一步规范{keyword}工作，制定本办法。",
                "publish_date": "2024-01-01",
                "department": "北京市政府",
                "doc_type": "policy"
            }
        ]
        
        policies = []
        for data in demo_data:
            policy = GovernmentAnnouncement(
                title=data["title"],
                content=data["content"],
                publish_date=data["publish_date"],
                source_url=f"{self.BEIJING_GOV_BASE}/policy/demo",
                district="北京市",
                department=data["department"],
                doc_type=data["doc_type"]
            )
            policies.append(policy)
        
        return policies
    
    def get_crawl_statistics(self) -> Dict[str, Any]:
        """获取爬取统计"""
        crawled_files = list(self.output_dir.glob("*.jsonl"))
        
        total_records = 0
        for file_path in crawled_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                total_records += sum(1 for _ in f)
        
        return {
            "total_files": len(crawled_files),
            "total_records": total_records,
            "output_dir": str(self.output_dir),
            "last_crawl": datetime.now().isoformat()
        }


def crawl_government_data(
    districts: List[str] | None = None,
    output_dir: str | None = None
) -> List[Dict[str, Any]]:
    """
    爬取政务数据的便捷函数
    
    Args:
        districts: 区县列表
        output_dir: 输出目录
        
    Returns:
        项目信息列表
    """
    if not REQUESTS_AVAILABLE:
        print("[政务爬虫] 警告: requests 或 beautifulsoup4 未安装，返回空列表")
        return []
    
    crawler = GovernmentDataCrawler(output_dir=output_dir)
    
    announcements = crawler.crawl_project_announcements(districts=districts)
    
    project_infos = []
    for announcement in announcements:
        info = crawler.extract_project_info(announcement)
        project_infos.append(info)
    
    return project_infos


if __name__ == "__main__":
    if not REQUESTS_AVAILABLE:
        print("请先安装依赖: pip install requests beautifulsoup4")
        exit(1)
    
    crawler = GovernmentDataCrawler()
    
    print("开始爬取政务数据...")
    announcements = crawler.crawl_project_announcements(districts=["朝阳区", "海淀区"])
    
    print(f"\n爬取到 {len(announcements)} 条公告")
    
    print("\n提取项目信息:")
    for announcement in announcements[:3]:
        info = crawler.extract_project_info(announcement)
        print(f"  - {info['title']}")
        print(f"    状态: {info['status']}")
        print(f"    项目: {info['projects']}")
    
    print("\n爬取统计:")
    stats = crawler.get_crawl_statistics()
    print(json.dumps(stats, ensure_ascii=False, indent=2))
