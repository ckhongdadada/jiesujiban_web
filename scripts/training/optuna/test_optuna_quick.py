"""
Optuna快速测试脚本
用于验证环境和配置
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_imports():
    """测试必要的导入"""
    print("=" * 60)
    print("测试1: 检查依赖包")
    print("=" * 60)
    
    try:
        import optuna
        print(f"✓ optuna: {optuna.__version__}")
    except ImportError:
        print("✗ optuna 未安装")
        print("  请运行: pip install optuna")
        return False
    
    try:
        import torch
        print(f"✓ torch: {torch.__version__}")
        print(f"  CUDA可用: {torch.cuda.is_available()}")
    except ImportError:
        print("✗ torch 未安装")
        return False
    
    try:
        import transformers
        print(f"✓ transformers: {transformers.__version__}")
    except ImportError:
        print("✗ transformers 未安装")
        return False
    
    try:
        import pandas
        print(f"✓ pandas: {pandas.__version__}")
    except ImportError:
        print("✗ pandas 未安装")
        return False
    
    print("\n所有依赖包检查通过!\n")
    return True


def test_data_path():
    """测试数据路径"""
    print("=" * 60)
    print("测试2: 检查数据文件")
    print("=" * 60)
    
    from training.train_unit_classifier_fgm import get_default_classifier_data_path
    
    data_path = get_default_classifier_data_path()
    print(f"数据路径: {data_path}")
    
    if Path(data_path).exists():
        print(f"✓ 数据文件存在")
        file_size = Path(data_path).stat().st_size / (1024 * 1024)
        print(f"  文件大小: {file_size:.2f} MB")
        return True
    else:
        print(f"✗ 数据文件不存在")
        print(f"  请确保数据文件在以下位置之一:")
        from training.train_unit_classifier_fgm import DEFAULT_CLASSIFIER_DATA_CANDIDATES
        for path in DEFAULT_CLASSIFIER_DATA_CANDIDATES:
            print(f"    - {path}")
        return False


def test_model_path():
    """测试模型路径"""
    print("\n" + "=" * 60)
    print("测试3: 检查基础模型")
    print("=" * 60)
    
    from src.jsjb.core.config import load_runtime_config
    
    config = load_runtime_config()
    base_model = config.classifier_base_model
    
    print(f"基础模型路径: {base_model}")
    
    if Path(base_model).exists():
        print(f"✓ 基础模型存在")
        return True
    else:
        print(f"✗ 基础模型不存在")
        print(f"  请确保RoBERTa模型已下载到: {base_model}")
        return False


def test_output_dir():
    """测试输出目录"""
    print("\n" + "=" * 60)
    print("测试4: 检查输出目录")
    print("=" * 60)
    
    from src.jsjb.core.paths import get_training_reports_dir
    
    output_dir = get_training_reports_dir() / "optuna"
    print(f"输出目录: {output_dir}")
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ 输出目录可写")
        return True
    except Exception as e:
        print(f"✗ 输出目录创建失败: {e}")
        return False


def test_simple_trial():
    """测试简单的trial"""
    print("\n" + "=" * 60)
    print("测试5: 运行简单trial")
    print("=" * 60)
    
    try:
        import optuna
        
        def objective(trial):
            x = trial.suggest_float("x", -10, 10)
            return (x - 2) ** 2
        
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=3, show_progress_bar=False)
        
        print(f"✓ Optuna运行正常")
        print(f"  最优值: {study.best_value:.4f}")
        print(f"  最优参数: {study.best_params}")
        return True
        
    except Exception as e:
        print(f"✗ Optuna运行失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("Optuna环境检查")
    print("=" * 60 + "\n")
    
    results = []
    
    results.append(("依赖包", test_imports()))
    results.append(("数据文件", test_data_path()))
    results.append(("基础模型", test_model_path()))
    results.append(("输出目录", test_output_dir()))
    results.append(("Optuna功能", test_simple_trial()))
    
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{name:12s}: {status}")
    
    all_passed = all(r[1] for r in results)
    
    print("=" * 60)
    
    if all_passed:
        print("\n✓ 所有测试通过! 可以运行Optuna搜索")
        print("\n运行命令:")
        print("  python run_optuna_search.py --n-trials 20")
        return 0
    else:
        print("\n✗ 部分测试失败，请先解决上述问题")
        return 1


if __name__ == "__main__":
    sys.exit(main())
