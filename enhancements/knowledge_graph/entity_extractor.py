"""
实体提取器
从文本中提取项目、地点、单位、政策等实体
支持正则表达式和NER模型两种方式
"""

from __future__ import annotations

import re
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime


class EntityExtractor:
    """实体提取器"""
    
    PROJECT_KEYWORDS = {
        "道路": "道路建设",
        "学校": "教育设施",
        "医院": "医疗设施",
        "公园": "公共绿地",
        "停车场": "交通设施",
        "图书馆": "文化设施",
        "体育": "体育设施",
        "改造": "改造工程",
        "建设": "建设工程",
        "修缮": "维修工程",
        "供水": "供水设施",
        "供电": "供电设施",
        "供暖": "供暖设施",
        "排水": "排水设施",
        "绿化": "绿化工程",
        "照明": "照明设施",
        "电梯": "电梯工程",
        "消防": "消防设施",
    }
    
    ORG_KEYWORDS = {
        "城管": "城市管理",
        "环卫": "环境卫生",
        "交通": "交通管理",
        "教委": "教育管理",
        "卫健": "卫生健康",
        "住建": "住房建设",
        "规自": "规划自然资源",
        "街道": "街道办事处",
        "社区": "社区居委会",
        "公安": "公安管理",
        "民政": "民政管理",
        "人社": "人力资源",
        "财政": "财政管理",
        "环保": "环境保护",
        "水务": "水务管理",
        "园林": "园林绿化",
        "市场监管": "市场监管",
    }
    
    STATUS_KEYWORDS = {
        "规划": "规划中",
        "前期": "前期手续",
        "施工": "施工中",
        "建设": "建设中",
        "完工": "已完工",
        "竣工": "已完工",
        "验收": "验收中",
        "运营": "运营中",
        "使用": "使用中",
        "停工": "已停工",
        "延期": "已延期",
    }
    
    BEIJING_DISTRICTS = [
        "东城区", "西城区", "朝阳区", "丰台区", "石景山区", "海淀区",
        "门头沟区", "房山区", "通州区", "顺义区", "昌平区", "大兴区",
        "怀柔区", "平谷区", "密云区", "延庆区", "亦庄", "开发区"
    ]
    
    def __init__(self, use_ner: bool = False, ner_model_path: str = None):
        self.project_pattern = re.compile(r"([^，。！？\s]{2,20}(?:工程|项目|建设|改造|修缮|小区|公园|学校|医院))")
        self.org_pattern = re.compile(r"([^，。！？\s]{2,15}(?:局|委|办|处|中心|公司|单位|街道|社区))")
        self.date_pattern = re.compile(r"(\d{4}年\d{1,2}月\d{0,2}日?|\d{4}-\d{1,2}-\d{0,2}|\d{4}\.\d{1,2}\.\d{0,2})")
        self.phone_pattern = re.compile(r"(?:电话|联系方式|联系电话)[：:]\s*(\d{3,4}[-\s]?\d{7,8})")
        self.money_pattern = re.compile(r"(\d+(?:\.\d+)?[万亿]元)")
        
        self.use_ner = use_ner
        self.ner_model = None
        self.ner_tokenizer = None
        
        if use_ner and ner_model_path:
            self._load_ner_model(ner_model_path)
    
    def _load_ner_model(self, model_path: str):
        """加载NER模型"""
        try:
            from transformers import AutoTokenizer, AutoModelForTokenClassification
            import torch
            
            self.ner_tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.ner_model = AutoModelForTokenClassification.from_pretrained(model_path)
            self.ner_model.eval()
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.ner_model.to(self.device)
            
            print(f"[实体提取] NER模型已加载: {model_path}")
        except Exception as e:
            print(f"[实体提取] NER模型加载失败: {e}")
            self.use_ner = False
    
    def extract_from_document(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        从文档中提取实体
        
        Args:
            doc: 文档字典，包含title, content, district等字段
            
        Returns:
            提取的实体字典
        """
        text = f"{doc.get('title', '')} {doc.get('content', '')} {doc.get('snippet', '')}"
        
        entities = {
            "projects": self.extract_projects(text, doc),
            "locations": self.extract_locations(text, doc),
            "organizations": self.extract_organizations(text, doc),
            "policies": self.extract_policies(text, doc),
            "dates": self.extract_dates(text),
            "statuses": self.extract_statuses(text),
            "contacts": self.extract_contacts(text),
            "amounts": self.extract_amounts(text),
        }
        
        entities["relationships"] = self.extract_relationships(text, entities)
        
        return entities
    
    def extract_projects(self, text: str, doc: Dict) -> List[Dict]:
        """提取项目实体"""
        projects = []
        seen_names = set()
        
        matches = self.project_pattern.findall(text)
        
        for match in matches:
            if match in seen_names:
                continue
            seen_names.add(match)
            
            project_type = self._infer_project_type(match)
            status = self._infer_status(text)
            
            context = self._extract_context(text, match, window=50)
            
            project = {
                "name": match,
                "type": project_type,
                "status": status,
                "description": doc.get("snippet", "")[:200],
                "source_doc": doc.get("title", ""),
                "district": doc.get("district", ""),
                "context": context,
            }
            
            dates = self.extract_dates(text)
            if dates:
                project["start_date"] = dates[0] if len(dates) > 0 else ""
                project["end_date"] = dates[-1] if len(dates) > 1 else ""
            
            amounts = self.extract_amounts(text)
            if amounts:
                project["budget"] = amounts[0]
            
            projects.append(project)
        
        return projects
    
    def extract_locations(self, text: str, doc: Dict) -> List[Dict]:
        """提取地点实体"""
        locations = []
        seen_names = set()
        
        district = doc.get("district", "")
        
        for dist in self.BEIJING_DISTRICTS:
            if dist in text and dist not in seen_names:
                seen_names.add(dist)
                locations.append({
                    "name": dist,
                    "type": "区县",
                    "district": dist,
                    "street": "",
                    "coordinates": ""
                })
        
        street_pattern = re.compile(r"([^，。！？\s]{2,10}(?:街道|镇|乡))")
        streets = street_pattern.findall(text)
        for street in streets:
            if street not in seen_names:
                seen_names.add(street)
                locations.append({
                    "name": street,
                    "type": "街道",
                    "district": district,
                    "street": street,
                    "coordinates": ""
                })
        
        community_pattern = re.compile(r"([^，。！？\s]{2,15}(?:小区|社区|村|家园|花园|公寓|大院))")
        communities = community_pattern.findall(text)
        for community in communities:
            if community not in seen_names:
                seen_names.add(community)
                locations.append({
                    "name": community,
                    "type": "社区",
                    "district": district,
                    "street": "",
                    "coordinates": ""
                })
        
        road_pattern = re.compile(r"([^，。！？\s]{2,15}(?:路|街|巷|胡同|大道|大街))")
        roads = road_pattern.findall(text)
        for road in roads:
            if road not in seen_names and len(road) >= 3:
                seen_names.add(road)
                locations.append({
                    "name": road,
                    "type": "道路",
                    "district": district,
                    "street": "",
                    "coordinates": ""
                })
        
        return locations
    
    def extract_organizations(self, text: str, doc: Dict) -> List[Dict]:
        """提取单位实体"""
        organizations = []
        seen_names = set()
        
        matches = self.org_pattern.findall(text)
        
        for match in matches:
            if match in seen_names:
                continue
            seen_names.add(match)
            
            org_type = self._infer_org_type(match)
            level = self._infer_org_level(match)
            
            context = self._extract_context(text, match, window=30)
            
            org = {
                "name": match,
                "type": org_type,
                "level": level,
                "contact": "",
                "responsibilities": "",
                "district": doc.get("district", ""),
                "context": context,
            }
            
            organizations.append(org)
        
        if doc.get("responsible_unit") and doc["responsible_unit"] not in seen_names:
            organizations.append({
                "name": doc["responsible_unit"],
                "type": self._infer_org_type(doc["responsible_unit"]),
                "level": self._infer_org_level(doc["responsible_unit"]),
                "contact": "",
                "responsibilities": "",
                "district": doc.get("district", "")
            })
        
        return organizations
    
    def extract_policies(self, text: str, doc: Dict) -> List[Dict]:
        """提取政策实体"""
        policies = []
        seen_titles = set()
        
        policy_pattern = re.compile(r"《([^》]{4,50})》")
        matches = policy_pattern.findall(text)
        
        policy_keywords = ["办法", "规定", "条例", "通知", "意见", "方案", "标准", "规范", "指南", "政策"]
        
        for match in matches:
            if match in seen_titles:
                continue
            
            if any(kw in match for kw in policy_keywords):
                seen_titles.add(match)
                policy = {
                    "title": match,
                    "doc_number": "",
                    "publish_date": "",
                    "effective_date": "",
                    "content": "",
                    "source": doc.get("title", "")
                }
                policies.append(policy)
        
        doc_number_pattern = re.compile(r"([京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼][A-Z]?[A-Za-z0-9]+号)")
        doc_numbers = doc_number_pattern.findall(text)
        for doc_num in doc_numbers:
            if not any(p.get("doc_number") == doc_num for p in policies):
                policies.append({
                    "title": f"文件{doc_num}",
                    "doc_number": doc_num,
                    "publish_date": "",
                    "effective_date": "",
                    "content": "",
                    "source": doc.get("title", "")
                })
        
        return policies
    
    def extract_dates(self, text: str) -> List[str]:
        """提取日期"""
        matches = self.date_pattern.findall(text)
        dates = []
        for match in matches:
            normalized = match.replace("年", "-").replace("月", "-").replace("日", "").replace(".", "-")
            if normalized.endswith("-"):
                normalized = normalized[:-1]
            dates.append(normalized)
        return dates
    
    def extract_statuses(self, text: str) -> List[str]:
        """提取状态"""
        statuses = []
        for keyword, status in self.STATUS_KEYWORDS.items():
            if keyword in text:
                statuses.append(status)
        return list(set(statuses))
    
    def extract_contacts(self, text: str) -> List[Dict]:
        """提取联系方式"""
        contacts = []
        matches = self.phone_pattern.findall(text)
        for phone in matches:
            contacts.append({
                "type": "电话",
                "value": phone
            })
        return contacts
    
    def extract_amounts(self, text: str) -> List[str]:
        """提取金额"""
        return self.money_pattern.findall(text)
    
    def extract_relationships(self, text: str, entities: Dict) -> List[Dict]:
        """提取实体关系"""
        relationships = []
        
        for project in entities.get("projects", []):
            for location in entities.get("locations", []):
                if location["name"] in text and project["name"] in text:
                    relationships.append({
                        "from": project["name"],
                        "from_type": "Project",
                        "to": location["name"],
                        "to_type": "Location",
                        "rel_type": "LOCATED_IN"
                    })
        
        responsible_patterns = [
            (r"由([^，。！？]{2,15})负责", "RESPONSIBLE_FOR"),
            (r"([^，。！？]{2,15})负责该", "RESPONSIBLE_FOR"),
            (r"经([^，。！？]{2,15})核实", "VERIFIED_BY"),
            (r"([^，。！？]{2,15})办理", "HANDLED_BY"),
            (r"联系([^，。！？]{2,15})", "CONTACT"),
        ]
        
        for pattern, rel_type in responsible_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                for org in entities.get("organizations", []):
                    if org["name"] in match or match in org["name"]:
                        for project in entities.get("projects", []):
                            if project["name"] in text:
                                relationships.append({
                                    "from": org["name"],
                                    "from_type": "Organization",
                                    "to": project["name"],
                                    "to_type": "Project",
                                    "rel_type": rel_type
                                })
        
        for policy in entities.get("policies", []):
            for project in entities.get("projects", []):
                if policy["title"] in text and project["name"] in text:
                    relationships.append({
                        "from": project["name"],
                        "from_type": "Project",
                        "to": policy["title"],
                        "to_type": "Policy",
                        "rel_type": "BASED_ON"
                    })
        
        return self._deduplicate_relationships(relationships)
    
    def _extract_context(self, text: str, entity: str, window: int = 50) -> str:
        """提取实体上下文"""
        idx = text.find(entity)
        if idx == -1:
            return ""
        
        start = max(0, idx - window)
        end = min(len(text), idx + len(entity) + window)
        
        context = text[start:end]
        if start > 0:
            context = "..." + context
        if end < len(text):
            context = context + "..."
        
        return context
    
    def _deduplicate_relationships(self, relationships: List[Dict]) -> List[Dict]:
        """去重关系"""
        seen = set()
        unique = []
        for rel in relationships:
            key = (rel["from"], rel["to"], rel["rel_type"])
            if key not in seen:
                seen.add(key)
                unique.append(rel)
        return unique
    
    def _infer_project_type(self, project_name: str) -> str:
        """推断项目类型"""
        for keyword, ptype in self.PROJECT_KEYWORDS.items():
            if keyword in project_name:
                return ptype
        return "其他工程"
    
    def _infer_org_type(self, org_name: str) -> str:
        """推断单位类型"""
        for keyword, otype in self.ORG_KEYWORDS.items():
            if keyword in org_name:
                return otype
        return "其他单位"
    
    def _infer_org_level(self, org_name: str) -> str:
        """推断单位级别"""
        if "市" in org_name and "区" not in org_name and "街道" not in org_name:
            return "市级"
        elif any(d in org_name for d in self.BEIJING_DISTRICTS):
            return "区级"
        elif "街道" in org_name or "镇" in org_name:
            return "街道级"
        elif "社区" in org_name or "村" in org_name:
            return "社区级"
        return "未知"
    
    def _infer_status(self, text: str) -> str:
        """推断项目状态"""
        priority_order = ["运营", "完工", "竣工", "验收", "施工", "建设", "前期", "规划", "停工", "延期"]
        for keyword in priority_order:
            if keyword in text:
                return self.STATUS_KEYWORDS.get(keyword, "未知")
        return "未知"
    
    def extract_with_ner(self, text: str) -> List[Dict]:
        """使用NER模型提取实体"""
        if not self.use_ner or self.ner_model is None:
            return []
        
        try:
            import torch
            
            inputs = self.ner_tokenizer(
                text,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.ner_model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=-1)
            
            tokens = self.ner_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            labels = [self.ner_model.config.id2label[p.item()] for p in predictions[0]]
            
            entities = []
            current_entity = None
            
            for token, label in zip(tokens, labels):
                if label.startswith("B-"):
                    if current_entity:
                        entities.append(current_entity)
                    current_entity = {
                        "text": token.replace("##", ""),
                        "type": label[2:],
                        "start": 0,
                        "end": 0
                    }
                elif label.startswith("I-") and current_entity:
                    current_entity["text"] += token.replace("##", "")
                else:
                    if current_entity:
                        entities.append(current_entity)
                        current_entity = None
            
            if current_entity:
                entities.append(current_entity)
            
            return entities
            
        except Exception as e:
            print(f"[实体提取] NER提取失败: {e}")
            return []
