"""
新闻API连接器
从新闻媒体API获取项目相关信息
"""

from __future__ import annotations

import re
import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


@dataclass
class NewsArticle:
    """新闻文章"""
    title: str
    content: str
    publish_time: str
    source: str
    url: str = ""
    author: str = ""
    crawl_time: str = ""
    
    def __post_init__(self):
        if not self.crawl_time:
            self.crawl_time = datetime.now().isoformat()


@dataclass
class ProjectUpdate:
    """项目更新"""
    project: str
    old_status: str
    new_status: str
    update_time: str
    source: str
    confidence: float
    details: str = ""


class NewsAPIConnector:
    """新闻API连接器"""
    
    STATUS_UPDATE_PATTERNS = [
        (r"(.+?(?:道路|工程|项目)).{0,10}(?:已(?:完工|竣工|交付))", "已完成", "未开始"),
        (r"(.+?(?:道路|工程|项目)).{0,10}(?:正式通车)", "已通车", "建设中"),
        (r"(.+?(?:道路|工程|项目)).{0,10}(?:开工建设)", "建设中", "未开始"),
        (r"(.+?(?:道路|工程|项目)).{0,10}(?:进入收尾阶段)", "收尾阶段", "建设中"),
    ]
    
    PROJECT_KEYWORDS = [
        "道路", "工程", "项目", "建设", "改造", "扩建",
        "通车", "竣工", "开工", "交付"
    ]
    
    def __init__(self, api_keys: Dict[str, str] | None = None, output_dir: str | None = None):
        if not REQUESTS_AVAILABLE:
            raise ImportError("需要安装 requests: pip install requests")
        
        self.api_keys = api_keys or {}
        self.output_dir = Path(output_dir) if output_dir else self._get_default_output_dir()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def _get_default_output_dir(self) -> Path:
        base_dir = Path(__file__).resolve().parents[1]
        return base_dir / "data" / "crawled" / "news"
    
    def search_project_news(
        self,
        project_name: str,
        district: str = "",
        days: int = 30
    ) -> List[NewsArticle]:
        """
        搜索项目相关新闻
        
        Args:
            project_name: 项目名称
            district: 区县
            days: 搜索最近多少天的新闻
            
        Returns:
            新闻文章列表
        """
        all_news = []
        
        query = f"{district} {project_name}" if district else project_name
        
        demo_news = self._generate_demo_news(project_name, district)
        all_news.extend(demo_news)
        
        self._save_news(all_news, project_name)
        
        return all_news
    
    def _generate_demo_news(self, project_name: str, district: str) -> List[NewsArticle]:
        """生成示例新闻数据（实际使用时替换为真实API调用）"""
        demo_data = [
            {
                "title": f"{district}{project_name}建设进展顺利",
                "content": f"{district}{project_name}目前已完成主体工程，正在进行附属设施建设，预计年内通车。",
                "publish_time": (datetime.now() - timedelta(days=2)).isoformat(),
                "source": "北京日报"
            },
            {
                "title": f"{project_name}项目最新动态",
                "content": f"{project_name}施工单位正在加班加点推进工程进度，确保按期完工。",
                "publish_time": (datetime.now() - timedelta(days=5)).isoformat(),
                "source": "新京报"
            }
        ]
        
        news_list = []
        for data in demo_data:
            news = NewsArticle(
                title=data["title"],
                content=data["content"],
                publish_time=data["publish_time"],
                source=data["source"],
                url=f"https://news.example.com/{project_name}"
            )
            news_list.append(news)
        
        return news_list
    
    def extract_project_updates(self, news_list: List[NewsArticle]) -> List[ProjectUpdate]:
        """
        从新闻中提取项目更新
        
        Args:
            news_list: 新闻文章列表
            
        Returns:
            项目更新列表
        """
        updates = []
        
        for news in news_list:
            text = f"{news.title} {news.content}"
            
            for pattern, new_status, old_status in self.STATUS_UPDATE_PATTERNS:
                matches = re.finditer(pattern, text)
                for match in matches:
                    project_name = match.group(1).strip()
                    
                    if self._is_valid_project_name(project_name):
                        update = ProjectUpdate(
                            project=project_name,
                            old_status=old_status,
                            new_status=new_status,
                            update_time=news.publish_time,
                            source=news.source,
                            confidence=0.75,
                            details=news.title
                        )
                        updates.append(update)
        
        return updates
    
    def _is_valid_project_name(self, name: str) -> bool:
        """判断是否为有效的项目名称"""
        if not name or len(name) < 3:
            return False
        
        return any(keyword in name for keyword in self.PROJECT_KEYWORDS)
    
    def _save_news(self, news_list: List[NewsArticle], project_name: str) -> None:
        """保存新闻到文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = re.sub(r'[\\/:*?"<>|]', '_', project_name)
        output_file = self.output_dir / f"news_{safe_name}_{timestamp}.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for news in news_list:
                f.write(json.dumps(asdict(news), ensure_ascii=False) + "\n")
    
    def batch_search_projects(
        self,
        projects: List[Dict[str, str]],
        days: int = 30
    ) -> Dict[str, List[NewsArticle]]:
        """
        批量搜索多个项目的新闻
        
        Args:
            projects: 项目列表，每个项目包含 name 和 district
            days: 搜索天数
            
        Returns:
            项目名称到新闻列表的映射
        """
        results = {}
        
        for project in projects:
            name = project.get("name", "")
            district = project.get("district", "")
            
            if name:
                news = self.search_project_news(name, district, days)
                results[name] = news
                
                time.sleep(0.5)
        
        return results
    
    def get_trending_projects(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        获取热门项目（新闻提及次数多的项目）
        
        Args:
            days: 统计天数
            
        Returns:
            热门项目列表
        """
        all_files = list(self.output_dir.glob("news_*.jsonl"))
        
        project_counts: Dict[str, int] = {}
        
        cutoff_time = datetime.now() - timedelta(days=days)
        
        for file_path in all_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        news_data = json.loads(line)
                        publish_time = datetime.fromisoformat(news_data.get("publish_time", ""))
                        
                        if publish_time >= cutoff_time:
                            title = news_data.get("title", "")
                            
                            for keyword in self.PROJECT_KEYWORDS:
                                if keyword in title:
                                    match = re.search(rf"(.+?{keyword})", title)
                                    if match:
                                        project_name = match.group(1)
                                        project_counts[project_name] = project_counts.get(project_name, 0) + 1
                                    break
                    except Exception:
                        continue
        
        trending = sorted(project_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [{"project": name, "mention_count": count} for name, count in trending[:10]]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        all_files = list(self.output_dir.glob("news_*.jsonl"))
        
        total_news = 0
        for file_path in all_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                total_news += sum(1 for _ in f)
        
        return {
            "total_files": len(all_files),
            "total_news": total_news,
            "output_dir": str(self.output_dir)
        }


def search_project_news(
    project_name: str,
    district: str = "",
    days: int = 30
) -> List[Dict[str, Any]]:
    """
    搜索项目新闻的便捷函数
    
    Args:
        project_name: 项目名称
        district: 区县
        days: 搜索天数
        
    Returns:
        新闻列表
    """
    if not REQUESTS_AVAILABLE:
        print("[新闻API] 警告: requests 未安装，返回空列表")
        return []
    
    connector = NewsAPIConnector()
    news_list = connector.search_project_news(project_name, district, days)
    
    return [asdict(news) for news in news_list]


if __name__ == "__main__":
    if not REQUESTS_AVAILABLE:
        print("请先安装依赖: pip install requests")
        exit(1)
    
    connector = NewsAPIConnector()
    
    print("搜索项目新闻...")
    news_list = connector.search_project_news("鲁疃西路", "昌平区")
    
    print(f"\n找到 {len(news_list)} 条新闻:")
    for news in news_list:
        print(f"  - {news.title} ({news.source})")
    
    print("\n提取项目更新:")
    updates = connector.extract_project_updates(news_list)
    for update in updates:
        print(f"  - {update.project}: {update.old_status} -> {update.new_status}")
    
    print("\n统计信息:")
    stats = connector.get_statistics()
    print(json.dumps(stats, ensure_ascii=False, indent=2))
