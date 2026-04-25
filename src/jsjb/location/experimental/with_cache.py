from __future__ import annotations

import json
import os
import re
import time
import threading
import hashlib
from dataclasses import dataclass, asdict
from typing import Any
from collections import deque


@dataclass
class LocationHit:
    matched_text: str
    district: str
    source: str
    confidence: float = 1.0
    category: str = ""
    alias_type: str = ""


class TrieNode:
    """Trie树节点"""
    
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.value = None


class LocationTrie:
    """基于Trie树的地名匹配器"""
    
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, key: str, value: str):
        """插入键值对"""
        node = self.root
        for char in key:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
        node.value = value
    
    def build_from_dict(self, data: dict[str, str]):
        """从字典构建Trie树"""
        for key, value in data.items():
            self.insert(key, value)
    
    def find_all(self, text: str) -> list[tuple[str, str, int, int]]:
        """
        查找所有匹配项
        
        Returns:
            list of (matched_text, value, start_pos, end_pos)
        """
        results = []
        n = len(text)
        
        for i in range(n):
            node = self.root
            j = i
            last_match = None
            last_j = i
            
            while j < n and text[j] in node.children:
                node = node.children[text[j]]
                j += 1
                
                if node.is_end:
                    last_match = node.value
                    last_j = j
            
            if last_match:
                results.append((text[i:last_j], last_match, i, last_j))
        
        return results
    
    def find_longest(self, text: str) -> list[tuple[str, str, int, int]]:
        """查找最长匹配"""
        all_matches = self.find_all(text)
        
        if not all_matches:
            return []
        
        all_matches.sort(key=lambda x: (x[2], -(x[3] - x[2])))
        
        result = []
        last_end = -1
        
        for match in all_matches:
            if match[2] >= last_end:
                result.append(match)
                last_end = match[3]
        
        return result


class ResultCache:
    """结果缓存"""
    
    def __init__(self, max_size: int = 5000):
        self.cache = {}
        self.max_size = max_size
        self.lock = threading.Lock()
        self.stats = {'hits': 0, 'misses': 0}
    
    def _get_key(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()
    
    def get(self, text: str) -> dict[str, Any] | None:
        key = self._get_key(text)
        
        with self.lock:
            if key in self.cache:
                self.stats['hits'] += 1
                return self.cache[key]
            self.stats['misses'] += 1
            return None
    
    def set(self, text: str, result: dict[str, Any]):
        key = self._get_key(text)
        
        with self.lock:
            if len(self.cache) >= self.max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            self.cache[key] = result
    
    def get_stats(self) -> dict:
        with self.lock:
            total = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / total if total > 0 else 0
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'hit_rate': hit_rate
            }
    
    def clear(self):
        with self.lock:
            self.cache.clear()
            self.stats = {'hits': 0, 'misses': 0}


class AmapRateLimiter:
    """高德API限流器"""
    
    def __init__(self, max_calls_per_second: float = 10.0):
        self.max_calls_per_second = max_calls_per_second
        self.min_interval = 1.0 / max_calls_per_second
        self.last_call_time = 0.0
        self.lock = threading.Lock()
        self.call_count = 0
        self.total_wait_time = 0.0
    
    def acquire(self):
        """获取调用许可"""
        with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.last_call_time
            
            if time_since_last < self.min_interval:
                wait_time = self.min_interval - time_since_last
                time.sleep(wait_time)
                self.total_wait_time += wait_time
            
            self.last_call_time = time.time()
            self.call_count += 1
    
    def get_stats(self) -> dict:
        with self.lock:
            return {
                'call_count': self.call_count,
                'total_wait_time': self.total_wait_time,
                'max_calls_per_second': self.max_calls_per_second
            }


BEIJING_DISTRICTS = {
    "东城区", "西城区", "朝阳区", "丰台区", "石景山区", "海淀区",
    "门头沟区", "房山区", "通州区", "顺义区", "昌平区", "大兴区",
    "怀柔区", "平谷区", "密云区", "延庆区",
}

DISTRICT_ALIASES = {
    "东城": "东城区", "西城": "西城区", "朝阳": "朝阳区",
    "丰台": "丰台区", "石景山": "石景山区", "海淀": "海淀区",
    "门头沟": "门头沟区", "房山": "房山区", "通州": "通州区",
    "顺义": "顺义区", "昌平": "昌平区", "大兴": "大兴区",
    "怀柔": "怀柔区", "平谷": "平谷区", "密云": "密云区",
    "延庆": "延庆区", "北京": "北京市", "全市": "北京市",
}


class BeijingDistrictResolverOptimized:
    """优化的北京行政区识别器"""
    
    def __init__(
        self,
        alias_path: str | None = None,
        enable_lac: bool | None = None,
        amap_api_key: str | None = None,
        enable_cache: bool = True,
        cache_size: int = 5000,
        amap_rate_limit: float = 10.0
    ):
        self.amap_api_key = amap_api_key or os.getenv("AMAP_API_KEY")
        
        self.alias_paths = self._resolve_alias_paths(alias_path)
        self.alias_to_district = self._load_aliases(self.alias_paths)
        self.alias_catalog = self._load_catalog()
        
        self.enable_lac = enable_lac if enable_lac is not None else self._detect_lac()
        self.lac = None
        if self.enable_lac:
            try:
                from LAC import LAC
                self.lac = LAC(mode="lac")
            except ImportError:
                self.enable_lac = False
        
        self.enable_cache = enable_cache
        self.result_cache = ResultCache(max_size=cache_size) if enable_cache else None
        
        self.amap_cache = {}
        self.amap_rate_limiter = AmapRateLimiter(max_calls_per_second=amap_rate_limit) if self.amap_api_key else None
        
        self.trie = LocationTrie()
        self._build_trie()
        
        print(f"[地名识别] 初始化完成")
        print(f"  - 别名数量: {len(self.alias_to_district)}")
        print(f"  - LAC启用: {self.enable_lac}")
        print(f"  - 高德API: {'已配置' if self.amap_api_key else '未配置'}")
        print(f"  - 缓存启用: {enable_cache}")
    
    def _resolve_alias_paths(self, alias_path: str | None) -> list[str]:
        if alias_path and os.path.exists(alias_path):
            return [alias_path]
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidates = [
            os.path.join(base_dir, "data", "beijing_location_aliases.json"),
            os.path.join(base_dir, "enhancements", "beijing_location_aliases.json"),
        ]
        return [p for p in candidates if os.path.exists(p)]
    
    def _load_aliases(self, paths: list[str]) -> dict[str, str]:
        alias_dict = dict(DISTRICT_ALIASES)
        
        for path in paths:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        alias_dict.update(data)
            except Exception as e:
                print(f"[地名识别] 加载别名文件失败 {path}: {e}")
        
        return alias_dict
    
    def _load_catalog(self) -> dict[str, list[str]]:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        catalog_path = os.path.join(base_dir, "data", "beijing_location_catalog.json")
        
        if os.path.exists(catalog_path):
            try:
                with open(catalog_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        
        return {}
    
    def _detect_lac(self) -> bool:
        try:
            from LAC import LAC
            return True
        except ImportError:
            return False
    
    def _build_trie(self):
        """构建Trie树"""
        for alias, district in self.alias_to_district.items():
            self.trie.insert(alias, district)
        
        for district in BEIJING_DISTRICTS:
            self.trie.insert(district, district)
        
        print(f"[地名识别] Trie树构建完成")
    
    def _normalize(self, text: str) -> str:
        normalized = text.lower()
        normalized = re.sub(r"[^\u4e00-\u9fa5a-z0-9]", "", normalized)
        return normalized
    
    def _match_direct_districts(self, text: str) -> list[LocationHit]:
        hits = []
        for district in BEIJING_DISTRICTS:
            if district in text:
                hits.append(LocationHit(
                    matched_text=district,
                    district=district,
                    source="direct_match",
                    confidence=1.0
                ))
        return hits
    
    def _match_with_trie(self, text: str) -> list[LocationHit]:
        """使用Trie树匹配"""
        matches = self.trie.find_longest(text)
        
        hits = []
        for matched_text, district, start, end in matches:
            hits.append(LocationHit(
                matched_text=matched_text,
                district=district,
                source="trie_match",
                confidence=0.9
            ))
        
        return hits
    
    def _match_aliases(self, text: str) -> list[LocationHit]:
        hits = []
        for alias, district in self.alias_to_district.items():
            if alias in text:
                hits.append(LocationHit(
                    matched_text=alias,
                    district=district,
                    source="alias_match",
                    confidence=0.85,
                    alias_type="catalog" if alias in self.alias_catalog else "built_in"
                ))
        return hits
    
    def _extract_lac_loc_entities(self, text: str) -> list[str]:
        if not self.enable_lac or self.lac is None:
            return []
        
        try:
            result = self.lac.run(text)
            if not result or len(result) < 2:
                return []
            
            words, tags = result
            loc_entities = []
            current_loc = []
            
            for word, tag in zip(words, tags):
                if tag == "LOC":
                    current_loc.append(word)
                else:
                    if current_loc:
                        loc_entities.append("".join(current_loc))
                        current_loc = []
            
            if current_loc:
                loc_entities.append("".join(current_loc))
            
            return loc_entities
        except Exception:
            return []
    
    def _match_lac_entities(self, entities: list[str]) -> list[LocationHit]:
        hits = []
        for entity in entities:
            if entity in BEIJING_DISTRICTS:
                hits.append(LocationHit(
                    matched_text=entity,
                    district=entity,
                    source="lac_direct",
                    confidence=0.95
                ))
            elif entity in self.alias_to_district:
                hits.append(LocationHit(
                    matched_text=entity,
                    district=self.alias_to_district[entity],
                    source="lac_alias",
                    confidence=0.90
                ))
        return hits
    
    def _match_amap(self, text: str, lac_entities: list[str], existing_hits: list[LocationHit]) -> list[LocationHit]:
        if not self.amap_api_key:
            return []
        
        existing_districts = {hit.district for hit in existing_hits}
        if any(d in BEIJING_DISTRICTS for d in existing_districts):
            return []
        
        candidates = lac_entities[:3]
        if not candidates:
            loc_patterns = re.findall(r"[\u4e00-\u9fa5]{2,8}(?:路|街|道|小区|村|镇|乡|区)", text)
            candidates = loc_patterns[:3]
        
        hits = []
        for candidate in candidates:
            if candidate in self.amap_cache:
                cached_district = self.amap_cache[candidate]
                if cached_district:
                    hits.append(LocationHit(
                        matched_text=candidate,
                        district=cached_district,
                        source="amap_cache",
                        confidence=0.80
                    ))
                continue
            
            self.amap_rate_limiter.acquire()
            
            district = self._amap_geocode(candidate)
            self.amap_cache[candidate] = district
            
            if district:
                hits.append(LocationHit(
                    matched_text=candidate,
                    district=district,
                    source="amap_api",
                    confidence=0.75
                ))
        
        return hits
    
    def _amap_geocode(self, address: str) -> str | None:
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
            district = geocode.get("district", "")
            
            if district in BEIJING_DISTRICTS:
                return district
            
            return None
        except Exception:
            return None
    
    def _deduplicate_hits(self, hits: list[LocationHit]) -> list[LocationHit]:
        seen = {}
        for hit in hits:
            key = (hit.matched_text, hit.district)
            if key not in seen or hit.confidence > seen[key].confidence:
                seen[key] = hit
        return list(seen.values())
    
    def _pick_best_district(self, hits: list[LocationHit]) -> tuple[str, float, str]:
        if not hits:
            return "未识别", 0.0, "none"
        
        district_scores = {}
        for hit in hits:
            district = hit.district
            score = hit.confidence
            
            source_weights = {
                "direct_match": 1.0,
                "trie_match": 0.95,
                "lac_direct": 0.95,
                "lac_alias": 0.90,
                "alias_match": 0.85,
                "amap_api": 0.75,
                "amap_cache": 0.80,
            }
            score *= source_weights.get(hit.source, 0.7)
            
            if district not in district_scores:
                district_scores[district] = {"score": 0.0, "source": hit.source, "count": 0}
            
            district_scores[district]["score"] += score
            district_scores[district]["count"] += 1
        
        for district_info in district_scores.values():
            district_info["score"] /= district_info["count"]
        
        best_district = max(district_scores.items(), key=lambda x: x[1]["score"])
        
        return best_district[0], best_district[1]["score"], best_district[1]["source"]
    
    def resolve(self, text: str) -> dict[str, Any]:
        """识别地名"""
        if self.enable_cache and self.result_cache:
            cached = self.result_cache.get(text)
            if cached is not None:
                return cached
        
        normalized = self._normalize(text)
        hits: list[LocationHit] = []
        
        hits.extend(self._match_direct_districts(normalized))
        
        trie_hits = self._match_with_trie(normalized)
        hits.extend(trie_hits)
        
        if not trie_hits:
            hits.extend(self._match_aliases(normalized))
        
        lac_loc_entities = self._extract_lac_loc_entities(text)
        hits.extend(self._match_lac_entities(lac_loc_entities))
        
        if self.amap_api_key:
            hits.extend(self._match_amap(text, lac_loc_entities, hits))
        
        hits = self._deduplicate_hits(hits)
        district, confidence, source = self._pick_best_district(hits)
        
        result = {
            "district": district,
            "confidence": round(confidence, 3),
            "source": source,
            "places": [asdict(hit) for hit in hits[:10]],
            "text_length": len(text),
        }
        
        if self.enable_cache and self.result_cache:
            self.result_cache.set(text, result)
        
        return result
    
    def batch_resolve(self, texts: list[str]) -> list[dict[str, Any]]:
        """批量识别地名"""
        results = []
        for text in texts:
            result = self.resolve(text)
            results.append(result)
        return results
    
    def get_stats(self) -> dict[str, Any]:
        """获取统计信息"""
        stats = {
            "lac_enabled": self.enable_lac,
            "amap_enabled": self.amap_api_key is not None,
            "alias_count": len(self.alias_to_district),
        }
        
        if self.enable_cache and self.result_cache:
            stats["cache_stats"] = self.result_cache.get_stats()
        
        if self.amap_rate_limiter:
            stats["amap_stats"] = self.amap_rate_limiter.get_stats()
        
        return stats
    
    def clear_cache(self):
        """清空缓存"""
        if self.enable_cache and self.result_cache:
            self.result_cache.clear()
        self.amap_cache.clear()


def resolve_beijing_district(
    text: str,
    alias_path: str | None = None,
    enable_lac: bool | None = None,
    amap_api_key: str | None = None,
) -> dict[str, Any]:
    resolver = BeijingDistrictResolverOptimized(
        alias_path=alias_path,
        enable_lac=enable_lac,
        amap_api_key=amap_api_key
    )
    return resolver.resolve(text)


def batch_resolve_beijing_district(
    texts: list[str],
    alias_path: str | None = None,
    enable_lac: bool | None = None,
    amap_api_key: str | None = None,
) -> list[dict[str, Any]]:
    resolver = BeijingDistrictResolverOptimized(
        alias_path=alias_path,
        enable_lac=enable_lac,
        amap_api_key=amap_api_key
    )
    return resolver.batch_resolve(texts)
