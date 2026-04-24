import sys
import os

sys.path.insert(0, ".")

log_file = open("optuna_test_log.txt", "w", encoding="utf-8")

def log(msg):
    print(msg)
    log_file.write(msg + "\n")
    log_file.flush()

log("测试导入...")

try:
    from tools.optuna_search_classifier import parse_args, objective, build_training_args
    log("✓ 导入成功")
except Exception as e:
    log(f"✗ 导入失败: {e}")
    import traceback
    log(traceback.format_exc())
    log_file.close()
    exit(1)

try:
    from training.train_unit_classifier_fgm import BertClassifier, DataProcessor, TextDataset, Config, build_arg_parser
    log("✓ 训练模块导入成功")
except Exception as e:
    log(f"✗ 训练模块导入失败: {e}")
    import traceback
    log(traceback.format_exc())
    log_file.close()
    exit(1)

try:
    args = parse_args()
    args.n_trials = 2
    args.epochs = 1
    args.sample_size = 500
    args.study_name = "test_run"
    log(f"✓ 参数解析成功: {args.study_name}")
except Exception as e:
    log(f"✗ 参数解析失败: {e}")
    import traceback
    log(traceback.format_exc())
    log_file.close()
    exit(1)

log("\n所有测试通过!")

log_file.close()
