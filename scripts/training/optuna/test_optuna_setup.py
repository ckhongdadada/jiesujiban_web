import sys
sys.path.insert(0, ".")

print("测试导入...")

try:
    from tools.optuna_search_classifier import parse_args, objective, build_training_args
    print("✓ 导入成功")
except Exception as e:
    print(f"✗ 导入失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

try:
    from training.train_unit_classifier_fgm import BertClassifier, DataProcessor, TextDataset, Config, build_arg_parser
    print("✓ 训练模块导入成功")
except Exception as e:
    print(f"✗ 训练模块导入失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

try:
    args = parse_args()
    args.n_trials = 2
    args.epochs = 1
    args.sample_size = 500
    args.study_name = "test_run"
    print(f"✓ 参数解析成功: {args.study_name}")
except Exception as e:
    print(f"✗ 参数解析失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n所有测试通过，准备运行Optuna搜索...")
