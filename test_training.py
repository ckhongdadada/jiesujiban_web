import sys
sys.path.insert(0, ".")

print("测试训练流程...")

from training.train_unit_classifier_fgm import BertClassifier, DataProcessor, TextDataset, Config, build_arg_parser
import torch
from torch.utils.data import DataLoader

parser = build_arg_parser()
args = parser.parse_args([
    "--epochs", "1",
    "--batch-size", "4",
    "--sample-size", "100",
    "--save-dir", "data/models/test_run",
])

print(f"数据路径: {args.data_path}")
print(f"样本大小: {args.sample_size}")
print(f"批大小: {args.batch_size}")

config = Config(args)
config.set_seed()

print("加载数据...")
processor = DataProcessor(config)
train_df, val_df, train_tfidf, val_tfidf = processor.load_and_process()

print(f"训练集大小: {len(train_df)}")
print(f"验证集大小: {len(val_df)}")

print("创建分类器...")
classifier = BertClassifier(config)

print("创建数据集...")
train_dataset = TextDataset(
    train_df,
    classifier.tokenizer,
    config.max_len,
    tfidf_data=train_tfidf,
    augment=True,
)
val_dataset = TextDataset(
    val_df,
    classifier.tokenizer,
    config.max_len,
    tfidf_data=val_tfidf,
    augment=False,
)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

print("开始训练...")
classifier.train(train_loader, val_loader)

print("评估模型...")
metrics = classifier.evaluate(val_loader)

print(f"评估结果: {metrics}")

print("测试完成!")
