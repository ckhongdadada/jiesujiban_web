from __future__ import annotations

import json
import os
import re
import time
import threading
import hashlib
from dataclasses import dataclass, asdict
from typing import Any
from collections import defaultdict


@dataclass
class LocationCandidate:
    matched_text: str
    district: str
    source: str
    confidence: float = 1.0
    category: str = ""
    alias_type: str = ""
    coordinates: tuple[float, float] | None = None
    context_score: float = 0.0
    geo_score: float = 0.0
    final_score: float = 0.0


@dataclass
class DisambiguationResult:
    matched_text: str
    district: str
    confidence: float
    source: str
    disambiguation_method: str
    candidates_count: int
    coordinates: tuple[float, float] | None = None


class ContextFeatureExtractor:
    """上下文特征提取器"""
    
    def __init__(self):
        self.district_keywords = {
            "东城区": ["故宫", "天安门", "王府井", "东单", "崇文门", "建国门", "朝阳门", "东直门"],
            "西城区": ["西单", "金融街", "什刹海", "后海", "南锣鼓巷", "鼓楼", "积水潭", "西直门"],
            "朝阳区": ["CBD", "国贸", "三里屯", "望京", "大望路", "呼家楼", "团结湖", "朝阳公园"],
            "海淀区": ["中关村", "五道口", "清华", "北大", "颐和园", "圆明园", "西苑", "上地"],
            "丰台区": ["丽泽", "丰台路口", "方庄", "大红门", "木樨园", "右安门", "卢沟桥"],
            "石景山区": ["八大处", "石景山游乐园", "首钢", "苹果园", "古城", "八角"],
            "通州区": ["通州北苑", "梨园", "九棵树", "临河里", "土桥", "张家湾"],
            "顺义区": ["首都机场", "天竺", "后沙峪", "马坡", "牛栏山", "南法信"],
            "昌平区": ["回龙观", "天通苑", "沙河", "昌平县城", "十三陵", "小汤山"],
            "大兴区": ["亦庄", "黄村", "旧宫", "瀛海", "青云店", "庞各庄"],
            "门头沟区": ["门头沟城区", "永定", "潭柘寺", "妙峰山"],
            "房山区": ["良乡", "长阳", "窦店", "燕山", "周口店", "琉璃河"],
            "怀柔区": ["怀柔城区", "雁栖湖", "红螺寺", "慕田峪"],
            "平谷区": ["平谷城区", "金海湖", "京东大峡谷"],
            "密云区": ["密云城区", "密云水库", "古北水镇"],
            "延庆区": ["延庆城区", "八达岭", "龙庆峡", "世园会"]
        }
        
        self.issue_type_districts = {
            "垃圾清运": ["朝阳区", "海淀区", "丰台区"],
            "噪声扰民": ["朝阳区", "海淀区", "西城区"],
            "停车秩序": ["朝阳区", "海淀区", "丰台区", "东城区"],
            "道路积水": ["朝阳区", "海淀区", "丰台区", "通州区"],
            "物业服务": ["朝阳区", "海淀区", "丰台区", "昌平区"]
        }
    
    def extract_features(self, text: str, location: str) -> dict[str, float]:
        """提取上下文特征"""
        features = {}
        
        for district, keywords in self.district_keywords.items():
            score = 0.0
            for keyword in keywords:
                if keyword in text:
                    score += 0.2
            features[f"district_keyword_{district}"] = min(score, 1.0)
        
        district_mentions = {}
        for district in self.district_keywords.keys():
            count = text.count(district)
            if count > 0:
                district_mentions[district] = count
        
        if district_mentions:
            total_mentions = sum(district_mentions.values())
            for district, count in district_mentions.items():
                features[f"district_mention_{district}"] = count / total_mentions
        
        for issue_type, districts in self.issue_type_districts.items():
            if issue_type in text:
                for district in districts:
                    features[f"issue_district_{district}"] = 0.3
        
        if "街道" in text or "社区" in text or "小区" in text:
            features["has_residential_area"] = 1.0
        
        if "路" in text or "街" in text:
            features["has_road"] = 1.0
        
        return features
    
    def calculate_context_score(self, text: str, location: str, candidate_district: str) -> float:
        """计算上下文匹配分数"""
        features = self.extract_features(text, location)
        
        score = 0.0
        
        keyword_key = f"district_keyword_{candidate_district}"
        if keyword_key in features:
            score += features[keyword_key] * 0.4
        
        mention_key = f"district_mention_{candidate_district}"
        if mention_key in features:
            score += features[mention_key] * 0.3
        
        issue_key = f"issue_district_{candidate_district}"
        if issue_key in features:
            score += features[issue_key] * 0.2
        
        if candidate_district in text:
            score += 0.1
        
        return min(score, 1.0)


class GeoCodingValidator:
    """地理编码验证器"""
    
    def __init__(self, amap_api_key: str | None = None):
        self.amap_api_key = amap_api_key or os.getenv("AMAP_API_KEY")
        self.cache = {}
        self.lock = threading.Lock()
        
        self.district_centers = {
            "东城区": (116.418, 39.928),
            "西城区": (116.366, 39.912),
            "朝阳区": (116.486, 39.921),
            "海淀区": (116.298, 39.959),
            "丰台区": (116.287, 39.858),
            "石景山区": (116.223, 39.906),
            "通州区": (116.656, 39.909),
            "顺义区": (116.654, 40.130),
            "昌平区": (116.231, 40.221),
            "大兴区": (116.341, 39.727),
            "门头沟区": (116.102, 39.940),
            "房山区": (116.143, 39.748),
            "怀柔区": (116.632, 40.316),
            "平谷区": (117.121, 40.141),
            "密云区": (116.843, 40.377),
            "延庆区": (115.975, 40.457)
        }
        
        self.district_bounds = {
            "东城区": ((116.38, 39.88), (116.46, 39.96)),
            "西城区": ((116.33, 39.87), (116.40, 39.95)),
            "朝阳区": ((116.40, 39.82), (116.58, 40.02)),
            "海淀区": ((116.20, 39.88), (116.38, 40.04)),
            "丰台区": ((116.18, 39.78), (116.40, 39.94)),
            "石景山区": ((116.15, 39.85), (116.30, 39.96)),
            "通州区": ((116.55, 39.80), (116.80, 40.02)),
            "顺义区": ((116.50, 40.00), (116.85, 40.28)),
            "昌平区": ((116.05, 40.08), (116.45, 40.38)),
            "大兴区": ((116.20, 39.58), (116.55, 39.85)),
            "门头沟区": ((115.90, 39.80), (116.25, 40.08)),
            "房山区": ((115.75, 39.55), (116.25, 39.95)),
            "怀柔区": ((116.40, 40.18), (116.90, 40.48)),
            "平谷区": ((116.95, 40.02), (117.32, 40.28)),
            "密云区": ((116.60, 40.22), (117.10, 40.55)),
            "延庆区": ((115.70, 40.32), (116.25, 40.60))
        }
    
    def geocode(self, address: str) -> tuple[float, float] | None:
        """获取地址的地理编码"""
        if address in self.cache:
            return self.cache[address]
        
        if not self.amap_api_key:
            return None
        
        try:
            import urllib.request
            import urllib.parse
            
            params = {
                "key": self.amap_api_key,
                "address": f"北京市{address}",
                "city": "北京"
            }
            url = f"https://restapi.amap.com/v3/geocode/geo?{urllib.parse.urlencode(params)}"
            
            with urllib.request.urlopen(url, timeout=5) as response:
                data = json.loads(response.read().decode("utf-8"))
            
            if data.get("status") != "1" or not data.get("geocodes"):
                return None
            
            geocode = data["geocodes"][0]
            location = geocode.get("location", "")
            
            if location:
                lng, lat = map(float, location.split(","))
                
                with self.lock:
                    self.cache[address] = (lng, lat)
                
                return (lng, lat)
            
            return None
        except Exception:
            return None
    
    def is_within_district(self, coords: tuple[float, float], district: str) -> bool:
        """检查坐标是否在指定区域内"""
        if district not in self.district_bounds:
            return False
        
        lng, lat = coords
        (min_lng, min_lat), (max_lng, max_lat) = self.district_bounds[district]
        
        return min_lng <= lng <= max_lng and min_lat <= lat <= max_lat
    
    def calculate_geo_score(self, location: str, candidate_district: str) -> float:
        """计算地理编码验证分数"""
        coords = self.geocode(location)
        
        if coords is None:
            return 0.5
        
        if self.is_within_district(coords, candidate_district):
            return 1.0
        
        if candidate_district in self.district_centers:
            center = self.district_centers[candidate_district]
            distance = self._calculate_distance(coords, center)
            
            if distance < 5:
                return 0.8
            elif distance < 10:
                return 0.6
            elif distance < 20:
                return 0.4
            else:
                return 0.2
        
        return 0.5
    
    def _calculate_distance(self, coord1: tuple[float, float], coord2: tuple[float, float]) -> float:
        """计算两点之间的距离（公里）"""
        import math
        
        lng1, lat1 = coord1
        lng2, lat2 = coord2
        
        dlng = math.radians(lng2 - lng1)
        dlat = math.radians(lat2 - lat1)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlng/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371
        
        return c * r


class MultiSourceFusion:
    """多源融合器"""
    
    def __init__(self):
        self.source_weights = {
            "direct_match": 1.0,
            "trie_match": 0.95,
            "lac_direct": 0.95,
            "lac_alias": 0.90,
            "alias_match": 0.85,
            "amap_api": 0.75,
            "amap_cache": 0.80,
            "context_disambiguation": 0.85,
            "geo_validation": 0.80
        }
    
    def fuse_results(
        self,
        candidates: list[LocationCandidate],
        context_score: float = 0.0,
        geo_score: float = 0.0
    ) -> DisambiguationResult | None:
        """融合多个候选结果"""
        if not candidates:
            return None
        
        district_scores = defaultdict(lambda: {"score": 0.0, "count": 0, "best_candidate": None})
        
        for candidate in candidates:
            district = candidate.district
            base_score = candidate.confidence * self.source_weights.get(candidate.source, 0.7)
            
            district_scores[district]["score"] += base_score
            district_scores[district]["count"] += 1
            
            if (district_scores[district]["best_candidate"] is None or 
                candidate.confidence > district_scores[district]["best_candidate"].confidence):
                district_scores[district]["best_candidate"] = candidate
        
        for district, info in district_scores.items():
            info["score"] /= info["count"]
            
            if context_score > 0:
                info["score"] += context_score * 0.2
            
            if geo_score > 0:
                info["score"] += geo_score * 0.1
        
        best_district = max(district_scores.items(), key=lambda x: x[1]["score"])
        district_name = best_district[0]
        district_info = best_district[1]
        
        best_candidate = district_info["best_candidate"]
        
        return DisambiguationResult(
            matched_text=best_candidate.matched_text,
            district=district_name,
            confidence=min(district_info["score"], 1.0),
            source=best_candidate.source,
            disambiguation_method="multi_source_fusion",
            candidates_count=len(candidates),
            coordinates=best_candidate.coordinates
        )


class LocationDisambiguator:
    """地名消歧器"""
    
    def __init__(
        self,
        amap_api_key: str | None = None,
        enable_context_disambiguation: bool = True,
        enable_geo_validation: bool = True
    ):
        self.context_extractor = ContextFeatureExtractor() if enable_context_disambiguation else None
        self.geo_validator = GeoCodingValidator(amap_api_key) if enable_geo_validation else None
        self.multi_source_fusion = MultiSourceFusion()
        
        self.enable_context_disambiguation = enable_context_disambiguation
        self.enable_geo_validation = enable_geo_validation
        
        self.stats = {
            "total_disambiguations": 0,
            "context_disambiguations": 0,
            "geo_disambiguations": 0,
            "multi_source_disambiguations": 0
        }
    
    def disambiguate(
        self,
        text: str,
        location: str,
        candidates: list[LocationCandidate]
    ) -> DisambiguationResult | None:
        """执行地名消歧"""
        self.stats["total_disambiguations"] += 1
        
        if not candidates:
            return None
        
        if len(candidates) == 1:
            candidate = candidates[0]
            return DisambiguationResult(
                matched_text=candidate.matched_text,
                district=candidate.district,
                confidence=candidate.confidence,
                source=candidate.source,
                disambiguation_method="single_candidate",
                candidates_count=1,
                coordinates=candidate.coordinates
            )
        
        district_groups = defaultdict(list)
        for candidate in candidates:
            district_groups[candidate.district].append(candidate)
        
        if len(district_groups) == 1:
            best_candidate = max(candidates, key=lambda c: c.confidence)
            return DisambiguationResult(
                matched_text=best_candidate.matched_text,
                district=best_candidate.district,
                confidence=best_candidate.confidence,
                source=best_candidate.source,
                disambiguation_method="single_district",
                candidates_count=len(candidates),
                coordinates=best_candidate.coordinates
            )
        
        context_score = 0.0
        if self.enable_context_disambiguation and self.context_extractor:
            context_scores = {}
            for district in district_groups.keys():
                context_scores[district] = self.context_extractor.calculate_context_score(
                    text, location, district
                )
            
            if context_scores:
                best_context_district = max(context_scores.items(), key=lambda x: x[1])
                context_score = best_context_district[1]
                self.stats["context_disambiguations"] += 1
        
        geo_score = 0.0
        if self.enable_geo_validation and self.geo_validator:
            geo_scores = {}
            for district in district_groups.keys():
                geo_scores[district] = self.geo_validator.calculate_geo_score(location, district)
            
            if geo_scores:
                best_geo_district = max(geo_scores.items(), key=lambda x: x[1])
                geo_score = best_geo_district[1]
                self.stats["geo_disambiguations"] += 1
        
        result = self.multi_source_fusion.fuse_results(
            candidates,
            context_score=context_score,
            geo_score=geo_score
        )
        
        if result:
            self.stats["multi_source_disambiguations"] += 1
        
        return result
    
    def get_stats(self) -> dict:
        return {
            **self.stats,
            "context_disambiguation_rate": (
                self.stats["context_disambiguations"] / self.stats["total_disambiguations"]
                if self.stats["total_disambiguations"] > 0 else 0
            ),
            "geo_validation_rate": (
                self.stats["geo_disambiguations"] / self.stats["total_disambiguations"]
                if self.stats["total_disambiguations"] > 0 else 0
            ),
            "multi_source_rate": (
                self.stats["multi_source_disambiguations"] / self.stats["total_disambiguations"]
                if self.stats["total_disambiguations"] > 0 else 0
            )
        }


def disambiguate_location(
    text: str,
    location: str,
    candidates: list[dict[str, Any]],
    amap_api_key: str | None = None
) -> dict[str, Any] | None:
    """地名消歧便捷函数"""
    disambiguator = LocationDisambiguator(amap_api_key=amap_api_key)
    
    candidate_objects = [
        LocationCandidate(
            matched_text=c.get("matched_text", ""),
            district=c.get("district", ""),
            source=c.get("source", ""),
            confidence=c.get("confidence", 1.0),
            category=c.get("category", ""),
            alias_type=c.get("alias_type", ""),
            coordinates=c.get("coordinates")
        )
        for c in candidates
    ]
    
    result = disambiguator.disambiguate(text, location, candidate_objects)
    
    if result:
        return asdict(result)
    
    return None
