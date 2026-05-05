#!/usr/bin/env python3
"""
RAG 语料扩充脚本
1. 从 feedback.db 提取高质量案例入库
2. 从 fact_knowledge_base.json 提取审核通过事实
3. 补充北京市各区常见政策案例模板
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.jsjb.core.paths import get_feedback_db_path, get_fact_kb_path

CORPUS_PATH = os.path.join(PROJECT_ROOT, "data", "runtime", "policy_case_corpus.jsonl")

DISTRICTS = [
    "东城区", "西城区", "朝阳区", "丰台区", "石景山区", "海淀区",
    "门头沟区", "房山区", "通州区", "顺义区", "昌平区", "大兴区",
    "怀柔区", "平谷区", "密云区", "延庆区",
]

TEMPLATE_CASES = [
    {
        "title": "{district}老旧小区综合整治实施方案",
        "doc_type": "政策",
        "district": "{district}",
        "source": "{district}人民政府",
        "tags": ["老旧小区", "综合整治", "改造"],
        "content": "为加快推进老旧小区综合整治工作，改善居民居住环境，{district}制定本实施方案。整治范围包括：2000年底前建成的住宅小区，重点解决楼本体安全隐患、基础设施老化、公共服务缺失等问题。整治内容涵盖楼本体改造（外墙保温、屋面防水、门窗更换）、环境整治（绿化提升、道路整修、停车优化）、配套设施完善（加装电梯、无障碍设施、智能门禁）等。由区住建委牵头，各街道办事处负责具体实施，区房管局负责质量监督。居民可通过12345热线反映整治需求和意见。",
        "issue_type": "小区管理",
    },
    {
        "title": "{district}物业管理突出问题专项整治方案",
        "doc_type": "政策",
        "district": "{district}",
        "source": "{district}住建委",
        "tags": ["物业管理", "专项整治", "物业服务"],
        "content": "针对群众反映强烈的物业管理突出问题，{district}决定开展专项整治。重点整治：物业服务不到位（保洁绿化不达标、安保巡逻缺失）、公共收益不透明（未按规定公示收支、违规使用公共收益）、电梯维保不规范（超期未检、维保记录造假）、消防通道堵塞（占用消防通道、消防设施损坏）。业主可向属地街道物业科投诉，街道应在5个工作日内响应。对拒不整改的物业企业，将记入信用档案并限制承接新项目。",
        "issue_type": "物业服务",
    },
    {
        "title": "{district}噪声污染防治专项行动方案",
        "doc_type": "政策",
        "district": "{district}",
        "source": "{district}生态环境局",
        "tags": ["噪声污染", "扰民", "环境治理"],
        "content": "为有效防治噪声污染，保障居民生活安宁，{district}生态环境局联合公安分局、城管执法局开展噪声污染防治专项行动。重点整治：社会生活噪声（广场舞、商铺喇叭、装修施工）、建筑施工噪声（夜间违规施工、超时作业）、交通噪声（机动车鸣笛、轨道交通噪声）。居民发现噪声扰民可拨打12345或110投诉。对于夜间（22:00-次日6:00）施工噪声，由城管执法部门查处；对于社会生活噪声，由公安机关依法处理。",
        "issue_type": "噪声扰民",
    },
    {
        "title": "{district}停车秩序综合治理工作方案",
        "doc_type": "政策",
        "district": "{district}",
        "source": "{district}交通委",
        "tags": ["停车管理", "停车秩序", "违停"],
        "content": "为规范停车秩序，缓解停车难问题，{district}交通委制定本工作方案。主要措施：推进居住区停车位共享（鼓励商业停车场夜间向周边居民开放）、规范路侧停车管理（全面实施电子收费、取消免费时段）、加强违停执法（重点整治消防通道、人行道、公交站违停）、建设立体停车设施（在条件允许的区域建设机械式立体停车场）。居民小区内部停车管理由物业负责，道路公共停车由交通委管理，违停执法由交管部门负责。",
        "issue_type": "停车秩序",
    },
    {
        "title": "{district}垃圾分类工作推进方案",
        "doc_type": "政策",
        "district": "{district}",
        "source": "{district}城管委",
        "tags": ["垃圾分类", "垃圾清运", "环境卫生"],
        "content": "为深入推进垃圾分类工作，{district}城管委制定本推进方案。工作目标：居民小区垃圾分类参与率达到90%以上，厨余垃圾分出率持续提升。主要措施：完善分类投放设施（每个小区设置四类垃圾投放点，配备指导员）、优化收运体系（厨余垃圾日产日清，其他垃圾每日清运）、加强执法检查（对混投混运行为依法处罚）、推进源头减量（限制一次性用品、推广净菜上市）。居民如发现垃圾清运不及时、分类设施破损等问题，可向12345热线或属地街道反映。",
        "issue_type": "垃圾清运",
    },
    {
        "title": "{district}道路积水治理工作方案",
        "doc_type": "政策",
        "district": "{district}",
        "source": "{district}水务局",
        "tags": ["道路积水", "排水设施", "防汛"],
        "content": "为解决汛期道路积水问题，保障市民出行安全，{district}水务局制定本工作方案。治理范围：易积水路段、下穿式立交桥、低洼居民区。主要措施：排水管网清淤疏通（汛前完成全区排水管网排查和清淤）、泵站升级改造（对排水能力不足的泵站进行扩容）、建设海绵城市设施（增加透水铺装、雨水花园、调蓄池）、完善应急响应机制（暴雨预警时提前部署抽排设备）。居民发现道路积水可拨打12345或水务服务热线，水务部门应在1小时内到场处置。",
        "issue_type": "道路积水",
    },
    {
        "title": "{district}占道经营专项整治方案",
        "doc_type": "政策",
        "district": "{district}",
        "source": "{district}城管执法局",
        "tags": ["占道经营", "市容环境", "城管执法"],
        "content": "为维护市容环境秩序，保障道路通行安全，{district}城管执法局开展占道经营专项整治。整治重点：主次干道两侧占道经营、流动商贩聚集点、市场外溢经营、店外经营堆物堆料。执法方式：宣传教育先行、责令整改为主、行政处罚兜底。对屡教不改的，依法处以500元以上5000元以下罚款。同时合理设置便民疏导点，引导流动商贩入室入场规范经营。市民发现占道经营可拨打12345热线投诉，城管部门应在2小时内到场处置。",
        "issue_type": "占道经营",
    },
    {
        "title": "{district}消防通道专项整治行动方案",
        "doc_type": "政策",
        "district": "{district}",
        "source": "{district}消防救援支队",
        "tags": ["消防通道", "消防安全", "隐患整治"],
        "content": "为保障消防通道畅通，消除安全隐患，{district}消防救援支队联合住建委、城管执法局、交管部门开展专项整治。整治范围：居民小区消防通道、高层建筑消防登高面、商业街区疏散通道。重点问题：私家车占用消防通道、楼道堆物堆料、防火门被堵塞、消防设施损坏缺失。对占用消防通道的车辆，交管部门依法处罚并拖移；对楼道堆物，物业应限期清理，逾期由街道组织强制清理。居民发现消防隐患可拨打12345或119反映。",
        "issue_type": "消防通道",
    },
    {
        "title": "{district}施工扰民治理工作方案",
        "doc_type": "政策",
        "district": "{district}",
        "source": "{district}住建委",
        "tags": ["施工扰民", "建筑施工", "噪声治理"],
        "content": "为规范建筑施工现场管理，减少施工对周边居民的影响，{district}住建委制定本方案。管理要求：施工时间严格限制在7:00-20:00（中高考期间全天禁止产生噪声的施工）、施工现场必须设置围挡和防尘网、出土运输车辆必须覆盖并冲洗轮胎、夜间施工须提前公示并取得许可。投诉渠道：居民发现违规施工可向12345热线或属地住建委投诉，住建委应在1小时内到场核查。对违规施工单位，依法处以1万元以上10万元以下罚款，情节严重的责令停工整改。",
        "issue_type": "施工扰民",
    },
    {
        "title": "{district}绿化养护管理规范",
        "doc_type": "政策",
        "district": "{district}",
        "source": "{district}园林绿化局",
        "tags": ["园林绿化", "绿化养护", "环境提升"],
        "content": "为提升{district}绿化养护水平，改善城市生态环境，制定本管理规范。养护标准：行道树每年修剪不少于2次、草坪每月修剪1次、花坛每季度更换花卉、绿化带每周保洁。重点任务：补植补种缺株断垄、治理树木遮挡信号灯和路灯、防治病虫害、古树名木保护。居民如发现绿化损毁、树木遮挡、绿地被占用等问题，可向12345热线或区园林绿化局反映，园林绿化部门应在3个工作日内处理并反馈。",
        "issue_type": "园林绿化",
    },
    {
        "title": "{district}照明设施维护管理方案",
        "doc_type": "政策",
        "district": "{district}",
        "source": "{district}城管委",
        "tags": ["照明设施", "路灯", "夜间出行"],
        "content": "为保障市民夜间出行安全，{district}城管委制定照明设施维护管理方案。管理范围：城市道路路灯、小区路灯、景观照明、人行天桥照明。维护标准：主干道路灯亮灯率不低于98%、次干道不低于96%、小区路灯故障24小时内修复、路灯设施每月巡检1次。居民发现路灯不亮、灯杆倾斜、灯具损坏等问题，可拨打12345热线或城管服务热线反映，维护单位应在24小时内到场处理。",
        "issue_type": "照明设施",
    },
    {
        "title": "{district}排水设施管护工作方案",
        "doc_type": "政策",
        "district": "{district}",
        "source": "{district}水务局",
        "tags": ["排水设施", "排水管网", "防汛排涝"],
        "content": "为保障排水设施正常运行，提高防汛排涝能力，{district}水务局制定本方案。管护范围：雨水管网、污水管网、雨污合流管网、排水泵站、检查井、雨水口。管护标准：管网每年清淤不少于1次、泵站设备汛前全面检修、检查井盖缺失4小时内补齐、雨水口汛期每周清掏1次。居民发现排水设施堵塞、井盖缺失、污水外溢等问题，可拨打12345热线或水务服务热线，水务部门应在2小时内到场处置。",
        "issue_type": "排水设施",
    },
]

CITY_WIDE_CASES = [
    {
        "id": "policy-bj-001",
        "title": "北京市12345市民服务热线工作规范",
        "doc_type": "政策",
        "district": "全市",
        "source": "北京市政务服务局",
        "tags": ["12345", "市民热线", "接诉即办"],
        "content": "北京市12345市民服务热线是市委市政府设立的非紧急类政务服务便民热线，负责受理群众诉求、咨询、建议和投诉。工作流程：受理—派单—办理—回复—回访—评价。办理时限：普通诉求7个工作日、紧急诉求24小时、特别紧急诉求2小时。各承办单位须在规定时限内办结并回复诉求人，对超期未办结的将进行督办和通报。市民可通过拨打12345、登录北京通APP、访问首都之窗网站等方式提交诉求。",
        "issue_type": "政务服务",
    },
    {
        "id": "policy-bj-002",
        "title": "北京市物业管理条例实施细则",
        "doc_type": "政策",
        "district": "全市",
        "source": "北京市住建委",
        "tags": ["物业管理", "业主委员会", "物业服务标准"],
        "content": "根据《北京市物业管理条例》，制定本实施细则。物业服务标准：物业服务企业应当按照合同约定提供物业服务，包括共用部位维护、共用设施设备运行保养、环境卫生保洁、绿化养护、秩序维护等。业主权利：有权监督物业服务、提议召开业主大会、选举业主委员会、申请使用维修资金。物业费调整须经业主大会表决同意。对物业服务不满意的，业主可向属地街道物业科投诉，街道应协调处理。物业企业拒不整改的，可记入信用档案。",
        "issue_type": "物业服务",
    },
    {
        "id": "policy-bj-003",
        "title": "北京市大气污染防治条例",
        "doc_type": "政策",
        "district": "全市",
        "source": "北京市生态环境局",
        "tags": ["空气质量", "大气污染", "扬尘治理"],
        "content": "为防治大气污染，改善空气质量，保障公众健康，制定本条例。重点管控：施工扬尘（施工现场须设置围挡、洒水降尘、物料覆盖）、道路扬尘（增加道路清扫保洁频次、推广机械清扫）、餐饮油烟（餐饮单位须安装油烟净化设施并定期清洗）、机动车排放（实施排放检验与维护制度）。市民发现大气污染问题可拨打12345或12369环保举报热线反映，生态环境部门应及时查处并反馈。",
        "issue_type": "空气质量",
    },
    {
        "id": "policy-bj-004",
        "title": "北京市供热采暖管理办法",
        "doc_type": "政策",
        "district": "全市",
        "source": "北京市城管委",
        "tags": ["供暖", "供热", "冬季取暖"],
        "content": "本市采暖期为当年11月15日至次年3月15日。供热标准：采暖期内用户卧室、起居室温度不低于18℃。供热单位应保证供热设施正常运行，不得擅自推迟供热或提前停热。用户室内温度不达标的，可向供热单位报修，供热单位应在24小时内测温并处理。温度不达标的，按不达标天数退还采暖费。居民发现供暖问题可拨打12345热线或供热服务电话96069投诉。供热单位接到投诉后应在1小时内响应，24小时内处理完毕。",
        "issue_type": "供暖问题",
    },
    {
        "id": "policy-bj-005",
        "title": "北京市房屋建筑使用安全管理办法",
        "doc_type": "政策",
        "district": "全市",
        "source": "北京市住建委",
        "tags": ["房屋安全", "外墙脱落", "建筑安全"],
        "content": "房屋建筑所有权人是房屋使用安全责任人，应当定期检查房屋安全状况。发现外墙脱落、墙体开裂等安全隐患的，应立即设置警示标志并报告属地街道。街道接到报告后应组织房屋安全鉴定，鉴定为危险房屋的须设置围挡并督促产权人治理。物业服务企业应定期巡查共用部位，发现安全隐患及时报告业主委员会和属地街道。居民发现房屋安全隐患可拨打12345热线反映，住建部门应督促相关责任单位限期整改。",
        "issue_type": "房屋安全",
    },
    {
        "id": "policy-bj-006",
        "title": "北京市无障碍环境建设条例",
        "doc_type": "政策",
        "district": "全市",
        "source": "北京市人大常委会",
        "tags": ["无障碍", "残疾人", "适老化改造"],
        "content": "为保障残疾人、老年人等社会成员平等参与社会生活，制定本条例。建设要求：新建、改建、扩建公共建筑和居住建筑须配套建设无障碍设施、城市道路须设置盲道和缘石坡道、公共交通须配备无障碍车辆和设施、居住区须设置无障碍出入口和电梯。既有建筑应逐步推进无障碍改造，重点改造政府办公场所、医院、学校、商场等公共场所。市民发现无障碍设施被占用或损坏，可向12345热线反映，城管部门应依法查处。",
        "issue_type": "无障碍设施",
    },
    {
        "id": "policy-bj-007",
        "title": "北京市生活垃圾分类管理条例",
        "doc_type": "政策",
        "district": "全市",
        "source": "北京市人大常委会",
        "tags": ["垃圾分类", "厨余垃圾", "可回收物"],
        "content": "本市生活垃圾分为四类：厨余垃圾、可回收物、有害垃圾、其他垃圾。产生生活垃圾的单位和个人是垃圾分类的责任主体，应当依法在指定的分类投放点分类投放。违反规定的，由城管执法部门责令改正，拒不改正的，对个人处20元以上200元以下罚款，对单位处500元以上5000元以下罚款。物业服务企业是居住区垃圾分类管理责任人，应设置分类投放收集容器、指导居民分类投放、将生活垃圾交由有资质的单位收运。",
        "issue_type": "垃圾清运",
    },
    {
        "id": "policy-bj-008",
        "title": "北京市电动自行车消防安全管理规定",
        "doc_type": "政策",
        "district": "全市",
        "source": "北京市消防救援总队",
        "tags": ["电动自行车", "消防安全", "充电安全"],
        "content": "为加强电动自行车消防安全管理，预防和减少火灾事故，制定本规定。禁止行为：在建筑内的共用走道、楼梯间、安全出口处等公共区域停放电动自行车或为其充电、将电动自行车蓄电池带入电梯或室内充电、违反用电安全要求私拉电线充电。物业服务企业应设置集中充电设施并加强日常巡查。对违规停放充电的，由消防救援机构责令改正，拒不改正的，对经营性单位处2000元以上1万元以下罚款，对个人处500元以上1000元以下罚款。",
        "issue_type": "消防安全",
    },
]


def load_existing_ids(path: str) -> set[str]:
    ids = set()
    if not os.path.exists(path):
        return ids
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            ids.add(d.get("id", ""))
    return ids


def expand_from_feedback_db(corpus_path: str, existing_ids: set[str]) -> int:
    db_path = str(get_feedback_db_path())
    if not os.path.exists(db_path):
        print("[语料扩充] feedback.db 不存在，跳过")
        return 0

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        "SELECT id, tag, title, body, reply, unit, district, is_helpful "
        "FROM user_feedback WHERE is_helpful = 1 AND reply IS NOT NULL "
        "AND length(reply) > 50 ORDER BY id DESC LIMIT 500"
    )
    rows = cursor.fetchall()
    conn.close()

    added = 0
    with open(corpus_path, "a", encoding="utf-8") as f:
        for row in rows:
            doc_id = f"feedback-{row['id']}"
            if doc_id in existing_ids:
                continue
            title = (row["title"] or "").strip()
            body = (row["body"] or "").strip()
            reply = (row["reply"] or "").strip()
            if not reply or len(reply) < 50:
                continue

            content = f"诉求：{title}。{body}\n回复：{reply}" if body else f"诉求：{title}\n回复：{reply}"
            entry = {
                "id": doc_id,
                "title": title[:80] if title else "用户反馈案例",
                "doc_type": "案例",
                "district": row["district"] or "全市",
                "source": "用户反馈（审核通过）",
                "tags": [row["tag"]] if row["tag"] else [],
                "content": content[:2000],
                "issue_type": row["tag"] or "",
                "unit": row["unit"] or "",
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            existing_ids.add(doc_id)
            added += 1

    print(f"[语料扩充] 从 feedback.db 导入 {added} 条高质量案例")
    return added


def expand_from_fact_kb(corpus_path: str, existing_ids: set[str]) -> int:
    fact_path = str(get_fact_kb_path())
    if not os.path.exists(fact_path):
        print("[语料扩充] fact_knowledge_base.json 不存在，跳过")
        return 0

    with open(fact_path, "r", encoding="utf-8") as f:
        try:
            facts = json.load(f)
        except json.JSONDecodeError:
            print("[语料扩充] fact_knowledge_base.json 格式错误，跳过")
            return 0

    if not isinstance(facts, list):
        facts = [facts]

    added = 0
    with open(corpus_path, "a", encoding="utf-8") as f:
        for i, fact in enumerate(facts):
            doc_id = f"fact-{i:04d}"
            if doc_id in existing_ids:
                continue

            content = fact.get("content", "") or fact.get("fact_content", "")
            if not content:
                continue

            entry = {
                "id": doc_id,
                "title": fact.get("title", f"事实记录-{i+1}"),
                "doc_type": "事实记录",
                "district": fact.get("district", "全市"),
                "source": fact.get("source", "知识图谱审核"),
                "tags": fact.get("tags", []),
                "content": content[:2000],
                "issue_type": fact.get("issue_type", fact.get("fact_type", "")),
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            existing_ids.add(doc_id)
            added += 1

    print(f"[语料扩充] 从 fact_knowledge_base.json 导入 {added} 条事实记录")
    return added


def expand_from_templates(corpus_path: str, existing_ids: set[str]) -> int:
    added = 0
    with open(corpus_path, "a", encoding="utf-8") as f:
        for d_idx, district in enumerate(DISTRICTS):
            for t_idx, template in enumerate(TEMPLATE_CASES):
                doc_id = f"template-{district[:2]}-{d_idx:02d}-{t_idx:02d}"
                if doc_id in existing_ids:
                    continue

                entry = {
                    "id": doc_id,
                    "title": template["title"].format(district=district),
                    "doc_type": template["doc_type"],
                    "district": district,
                    "source": template["source"].format(district=district),
                    "tags": template["tags"],
                    "content": template["content"].format(district=district),
                    "issue_type": template["issue_type"],
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                existing_ids.add(doc_id)
                added += 1

    print(f"[语料扩充] 从模板生成 {added} 条区级政策文档")
    return added


def expand_city_wide(corpus_path: str, existing_ids: set[str]) -> int:
    added = 0
    with open(corpus_path, "a", encoding="utf-8") as f:
        for case in CITY_WIDE_CASES:
            doc_id = case["id"]
            if doc_id in existing_ids:
                continue
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
            existing_ids.add(doc_id)
            added += 1

    print(f"[语料扩充] 补充 {added} 条全市级政策文档")
    return added


def main() -> None:
    existing_ids = load_existing_ids(CORPUS_PATH)
    print(f"[语料扩充] 现有语料 {len(existing_ids)} 条")

    total = 0
    total += expand_city_wide(CORPUS_PATH, existing_ids)
    total += expand_from_templates(CORPUS_PATH, existing_ids)
    total += expand_from_feedback_db(CORPUS_PATH, existing_ids)
    total += expand_from_fact_kb(CORPUS_PATH, existing_ids)

    final_count = len(load_existing_ids(CORPUS_PATH))
    print(f"\n[语料扩充] 共新增 {total} 条，总计 {final_count} 条")


if __name__ == "__main__":
    main()
