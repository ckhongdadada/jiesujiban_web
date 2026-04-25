"""
知识图谱和主动学习使用示例
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from enhancements.knowledge_graph.graph_manager import KnowledgeGraphManager
from enhancements.knowledge_graph.entity_extractor import EntityExtractor
from enhancements.knowledge_graph.graph_query import GraphQueryEngine
from enhancements.active_learning.sample_collector import SampleCollector
from enhancements.active_learning.annotation_manager import AnnotationManager
from enhancements.active_learning.incremental_trainer import IncrementalTrainer


def example_knowledge_graph():
    """知识图谱使用示例"""
    print("=" * 60)
    print("知识图谱使用示例")
    print("=" * 60)
    
    # 1. 初始化图谱管理器（使用内存模式）
    graph = KnowledgeGraphManager()
    
    # 2. 创建实体
    print("\n[步骤1] 创建实体...")
    
    # 创建项目
    project_id = graph.create_project(
        name="朝阳路改造工程",
        type="道路建设",
        status="已完工",
        start_date="2023-06",
        end_date="2024-03",
        description="朝阳路全线改造，包括路面、排水、绿化"
    )
    print(f"  创建项目: 朝阳路改造工程 (ID: {project_id})")
    
    # 创建地点
    location_id = graph.create_location(
        name="朝阳区",
        type="区县",
        district="朝阳区"
    )
    print(f"  创建地点: 朝阳区 (ID: {location_id})")
    
    # 创建单位
    org_id = graph.create_organization(
        name="朝阳区城管委",
        type="城市管理",
        level="区级"
    )
    print(f"  创建单位: 朝阳区城管委 (ID: {org_id})")
    
    parent_org_id = graph.create_organization(
        name="北京市城管委",
        type="城市管理",
        level="市级"
    )
    print(f"  创建单位: 北京市城管委 (ID: {parent_org_id})")
    
    # 创建政策
    policy_id = graph.create_policy(
        title="北京市市政工程管理办法",
        doc_number="京政发[2023]10号",
        publish_date="2023-01-01"
    )
    print(f"  创建政策: 北京市市政工程管理办法 (ID: {policy_id})")
    
    # 3. 创建关系
    print("\n[步骤2] 创建关系...")
    
    graph.add_located_in(project_id, location_id)
    print("  添加关系: 朝阳路改造工程 --位于--> 朝阳区")
    
    graph.add_responsible_for(org_id, project_id)
    print("  添加关系: 朝阳区城管委 --负责--> 朝阳路改造工程")
    
    graph.add_reports_to(org_id, parent_org_id)
    print("  添加关系: 朝阳区城管委 --上级单位--> 北京市城管委")
    
    graph.add_based_on(project_id, policy_id)
    print("  添加关系: 朝阳路改造工程 --依据政策--> 北京市市政工程管理办法")
    
    graph.add_status_change(project_id, "施工中", "已完工", "2024-03-15")
    print("  添加关系: 朝阳路改造工程 --状态变更--> 已完工")
    
    # 4. 查询图谱
    print("\n[步骤3] 查询图谱...")
    
    query_engine = GraphQueryEngine(graph)
    
    # 查询项目完整信息
    print("\n查询: 朝阳路改造工程的完整信息")
    info = query_engine.query_project_info("朝阳路改造工程")
    summary = query_engine._format_project_summary(info)
    print(summary)
    
    # 查询责任链
    print("\n查询: 朝阳路改造工程的责任链")
    chain = query_engine.query_responsibility_chain("朝阳路改造工程")
    print("责任链:")
    for i, org in enumerate(chain, 1):
        print(f"  {i}. {org['properties']['name']} ({org['properties']['level']})")
    
    # 多跳查询
    print("\n查询: 朝阳路改造工程的负责单位的上级单位")
    result = query_engine.multi_hop_query("朝阳路改造工程", ["RESPONSIBLE_FOR", "REPORTS_TO"])
    if result:
        print(f"  结果: {result[0]['properties']['name']}")
    
    # 5. 保存图谱
    print("\n[步骤4] 保存图谱...")
    graph.save("data/runtime/knowledge_graph.json")
    print("  已保存到: data/runtime/knowledge_graph.json")
    
    graph.close()
    print("\n知识图谱示例完成!")


def example_active_learning():
    """主动学习使用示例"""
    print("\n" + "=" * 60)
    print("主动学习使用示例")
    print("=" * 60)
    
    # 1. 初始化组件
    print("\n[步骤1] 初始化组件...")
    
    collector = SampleCollector("data/runtime/active_learning.db")
    annotation_manager = AnnotationManager(collector)
    trainer = IncrementalTrainer(
        sample_collector=collector,
        base_data_path="data/runtime/training_data.jsonl",
        model_dir="final_model_fgm"
    )
    
    print("  样本收集器: 已初始化")
    print("  标注管理器: 已初始化")
    print("  增量训练器: 已初始化")
    
    # 2. 添加待标注样本
    print("\n[步骤2] 添加待标注样本...")
    
    samples = [
        {
            "sample_id": "sample_001",
            "tag": "环境卫生",
            "title": "小区垃圾清运不及时",
            "body": "望京南湖东园小区垃圾桶经常满溢，清运不及时",
            "district": "朝阳区",
            "predicted_unit": "环卫中心",
            "confidence": 0.45,
            "prediction_probs": {"环卫中心": 0.45, "城管委": 0.30, "街道办": 0.15}
        },
        {
            "sample_id": "sample_002",
            "tag": "交通出行",
            "title": "路灯损坏",
            "body": "朝阳路与望京街交叉口路灯损坏多日未修",
            "district": "朝阳区",
            "predicted_unit": "市政管理委",
            "confidence": 0.35,
            "prediction_probs": {"市政管理委": 0.35, "城管委": 0.32, "电力公司": 0.20},
            "user_feedback": "不满意"
        }
    ]
    
    for sample in samples:
        collector.add_sample(**sample)
        print(f"  添加样本: {sample['sample_id']} (置信度: {sample['confidence']})")
    
    # 3. 获取待标注样本
    print("\n[步骤3] 获取待标注样本...")
    
    pending = annotation_manager.get_next_batch(batch_size=5, strategy="priority")
    print(f"  获取到 {len(pending)} 条待标注样本")
    
    for sample in pending:
        formatted = annotation_manager.format_for_annotation(sample)
        print(f"\n  样本ID: {formatted['id']}")
        print(f"  优先级: {formatted['priority']}")
        print(f"  内容: {formatted['content'][:50]}...")
        print(f"  预测: {formatted['predicted']} (置信度: {formatted['confidence']})")
    
    # 4. 标注样本
    print("\n[步骤4] 标注样本...")
    
    annotations = [
        {"sample_id": "sample_001", "correct_unit": "朝阳区环卫中心", "notes": "确认正确"},
        {"sample_id": "sample_002", "correct_unit": "朝阳区城管委", "notes": "路灯归城管委管理"}
    ]
    
    result = annotation_manager.submit_batch_annotations(annotations, annotator="admin")
    print(f"  {result['message']}")
    
    # 5. 查看进度
    print("\n[步骤5] 查看标注进度...")
    
    progress = annotation_manager.get_annotation_progress()
    print(f"  待标注: {progress['pending']} 条")
    print(f"  已标注: {progress['annotated']} 条")
    print(f"  可用于训练: {progress['ready_for_training']} 条")
    print(f"  进度: {progress['progress_percent']}%")
    
    # 6. 获取建议
    print("\n[步骤6] 获取下一步建议...")
    
    suggestion = annotation_manager.suggest_next_action()
    print(f"  建议操作: {suggestion['action']}")
    print(f"  原因: {suggestion['reason']}")
    
    # 7. 查看统计
    print("\n[步骤7] 查看统计信息...")
    
    stats = collector.get_statistics()
    print(f"  待标注样本: {stats['pending_samples']} 条")
    print(f"  已标注样本: {stats['annotated_samples']} 条")
    print(f"  平均置信度: {stats['avg_confidence']}")
    print(f"  负面反馈: {stats['negative_feedback_samples']} 条")
    
    print("\n主动学习示例完成!")


def example_integration():
    """集成使用示例"""
    print("\n" + "=" * 60)
    print("集成使用示例")
    print("=" * 60)
    
    print("\n在Flask应用中集成:")
    print("""
from flask import Flask
from enhancements.active_learning.sample_collector import SampleCollector
from enhancements.active_learning.annotation_manager import AnnotationManager
from enhancements.active_learning.incremental_trainer import IncrementalTrainer
from enhancements.active_learning.api_integration import (
    create_active_learning_blueprint,
    add_sample_collection_middleware
)

app = Flask(__name__)

# 初始化组件
collector = SampleCollector()
annotation_manager = AnnotationManager(collector)
trainer = IncrementalTrainer(collector, "data.jsonl", "model_dir")

# 注册蓝图
bp = create_active_learning_blueprint(collector, annotation_manager, trainer)
app.register_blueprint(bp)

# 添加中间件（自动收集低置信度样本）
add_sample_collection_middleware(app, collector)

# 现在可以访问以下API:
# GET  /api/active-learning/samples/pending
# POST /api/active-learning/samples/annotate
# POST /api/active-learning/samples/annotate-batch
# GET  /api/active-learning/progress
# GET  /api/active-learning/statistics
# POST /api/active-learning/train
    """)


if __name__ == "__main__":
    # 运行示例
    example_knowledge_graph()
    example_active_learning()
    example_integration()
    
    print("\n" + "=" * 60)
    print("所有示例完成!")
    print("=" * 60)
