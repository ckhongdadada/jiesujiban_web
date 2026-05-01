# 接诉即办项目 - 冗余文件清理清单

**生成时间**：2026-04-25  
**预计可释放空间**：约 10 GB+

---

## 一、磁盘占用分析

| 目录 | 大小 | 说明 |
|-----|------|------|
| `checkpoints/classifier/` | ~7.43 GB | 分类模型检查点 |
| `checkpoints/generator/` | ~3.90 GB | 生成模型检查点 |
| `data/models/classifier/` | ~1.96 GB | 分类模型副本 |
| `legacy/` | ~0 GB | 遗留代码（体积很小） |
| `training/` | ~0 GB | 训练脚本（体积很小） |

---

## 二、必须保留的文件

以下文件是当前系统运行所必需的，**请勿删除**：

### 2.1 生成模型依赖
| 路径 | 说明 |
|-----|------|
| `checkpoints/generator/base_models/` | 生成链路依赖的基础 Qwen 模型 |
| `checkpoints/generator/lora/qwen_reply_model/` | 当前生成适配器 |

### 2.2 分类模型主线
| 路径 | 说明 |
|-----|------|
| `checkpoints/classifier/final_model_hybrid_v3_32cls/` | 当前主线分类模型 |

### 2.3 其他保留项
| 路径 | 说明 |
|-----|------|
| `legacy/` | 体积很小，保留可降低回溯风险 |
| `data/reports/training/*_latest.json` | 最新训练报告 |
| `logs/complaint_system.log` | 最近使用的日志 |
| `competition_materials/` | 交付材料 |

---

## 三、建议删除的文件（按优先级排序）

### 🔴 优先级 1：Optuna Trial 产物（高收益、低风险）

**预计释放空间**：约 2-3 GB

| 路径 | 大小 | 说明 |
|-----|------|------|
| `data/reports/training/optuna/trials/` | 大 | Optuna trial 模型副本 |
| `data/reports/training/optuna_trials/` | 大 | Optuna trial 产物 |

**删除命令**：
```powershell
Remove-Item -Path "c:\Users\28414\PycharmProjects\接诉即办项目\data\reports\training\optuna\trials" -Recurse -Force
Remove-Item -Path "c:\Users\28414\PycharmProjects\接诉即办项目\data\reports\training\optuna_trials" -Recurse -Force
```

---

### 🔴 优先级 2：空目录/空模型壳

**预计释放空间**：几乎为 0，但可清理目录结构

| 路径 | 说明 |
|-----|------|
| `data/models/classifier/bert_cnn_attn_test/` | 空目录 |
| `data/models/classifier/bert_cnn_attn_v1/` | 空目录 |
| `checkpoints/classifier/final_model_fgm_master_smoke/` | 几乎为空 |

**删除命令**：
```powershell
Remove-Item -Path "c:\Users\28414\PycharmProjects\接诉即办项目\data\models\classifier\bert_cnn_attn_test" -Recurse -Force
Remove-Item -Path "c:\Users\28414\PycharmProjects\接诉即办项目\data\models\classifier\bert_cnn_attn_v1" -Recurse -Force
Remove-Item -Path "c:\Users\28414\PycharmProjects\接诉即办项目\checkpoints\classifier\final_model_fgm_master_smoke" -Recurse -Force
```

---

### 🔴 优先级 3：旧模型的 training_state.pt（高收益、中风险）

**预计释放空间**：约 2-3 GB

这些是断点续训状态文件，如果不需要断点续训，可以删除：

| 路径 | 大小 | 说明 |
|-----|------|------|
| `checkpoints/classifier/final_model_distillation/training_state.pt` | 大 | 旧蒸馏模型断点 |
| `checkpoints/classifier/final_model_hybrid/training_state.pt` | 大 | 旧混合模型断点 |
| `checkpoints/classifier/final_model_hybrid_v3_32cls/training_state.pt` | 大 | 当前模型断点（谨慎） |
| `checkpoints/classifier/final_model_fgm/training_state.pt` | 大 | FGM模型断点 |
| `data/models/classifier/hybrid_ema_soft_label_bank/training_state.pt` | 大 | EMA软标签库断点 |

**删除命令**：
```powershell
Remove-Item -Path "c:\Users\28414\PycharmProjects\接诉即办项目\checkpoints\classifier\final_model_distillation\training_state.pt" -Force
Remove-Item -Path "c:\Users\28414\PycharmProjects\接诉即办项目\checkpoints\classifier\final_model_hybrid\training_state.pt" -Force
Remove-Item -Path "c:\Users\28414\PycharmProjects\接诉即办项目\checkpoints\classifier\final_model_fgm\training_state.pt" -Force
Remove-Item -Path "c:\Users\28414\PycharmProjects\接诉即办项目\data\models\classifier\hybrid_ema_soft_label_bank\training_state.pt" -Force
```

---

### 🟡 优先级 4：旧分类模型目录（高收益、中风险）

**预计释放空间**：约 5.5 GB

如果只保留一个分类主模型，可以删除以下目录：

| 路径 | 大小 | 说明 |
|-----|------|------|
| `checkpoints/classifier/final_model_distillation/` | ~1.96 GB | 旧蒸馏模型 |
| `checkpoints/classifier/final_model_hybrid/` | ~1.96 GB | 旧混合模型 |
| `checkpoints/classifier/final_model_fgm/` | ~1.56 GB | FGM模型 |
| `data/models/classifier/hybrid_ema_soft_label_bank/` | ~1.96 GB | EMA软标签库副本 |

**删除命令**：
```powershell
Remove-Item -Path "c:\Users\28414\PycharmProjects\接诉即办项目\checkpoints\classifier\final_model_distillation" -Recurse -Force
Remove-Item -Path "c:\Users\28414\PycharmProjects\接诉即办项目\checkpoints\classifier\final_model_hybrid" -Recurse -Force
Remove-Item -Path "c:\Users\28414\PycharmProjects\接诉即办项目\checkpoints\classifier\final_model_fgm" -Recurse -Force
Remove-Item -Path "c:\Users\28414\PycharmProjects\接诉即办项目\data\models\classifier\hybrid_ema_soft_label_bank" -Recurse -Force
```

---

### 🟡 优先级 5：Python 缓存文件（低收益、低风险）

**预计释放空间**：约 10-50 MB

| 路径 | 说明 |
|-----|------|
| `src/__pycache__/` | 源码缓存 |
| `src/jsjb/__pycache__/` | jsjb模块缓存 |
| `src/jsjb/core/__pycache__/` | 核心模块缓存 |
| `src/jsjb/location/__pycache__/` | 位置模块缓存 |
| `src/jsjb/unit_classifier/__pycache__/` | 分类器模块缓存 |
| `tests/__pycache__/` | 测试缓存 |
| `tests/core/__pycache__/` | 核心测试缓存 |
| `tests/unit_classifier/__pycache__/` | 分类器测试缓存 |

**删除命令**：
```powershell
Get-ChildItem -Path "c:\Users\28414\PycharmProjects\接诉即办项目" -Recurse -Directory -Filter "__pycache__" | Where-Object { $_.FullName -notlike "*\.venv\*" } | Remove-Item -Recurse -Force
```

---

### 🟡 优先级 6：重复的爬虫脚本

| 路径 | 说明 |
|-----|------|
| `scripts/crawling/government/crawl_policy_beijing_gov_v1.py` | 旧版爬虫（保留v2） |

**删除命令**：
```powershell
Remove-Item -Path "c:\Users\28414\PycharmProjects\接诉即办项目\scripts\crawling\government\crawl_policy_beijing_gov_v1.py" -Force
```

---

### 🟡 优先级 7：重复的 Optuna 脚本

`scripts/training/optuna/` 目录下有多个类似的脚本，建议只保留核心脚本：

| 路径 | 建议 |
|-----|------|
| `optuna_search_10_trials.py` | ❌ 删除 |
| `optuna_search_100_trials.py` | ❌ 删除 |
| `optuna_search_default.py` | ❌ 删除 |
| `optuna_search_direct_env.py` | ❌ 删除 |
| `optuna_search_v3_custom.py` | ❌ 删除 |
| `run_optuna_50_trials.py` | ❌ 删除 |
| `run_optuna_search.py` | ❌ 删除 |
| `run_optuna_wrapper.py` | ❌ 删除 |
| `test_optuna_log.py` | ❌ 删除 |
| `test_optuna_quick.py` | ❌ 删除 |
| `test_optuna_setup.py` | ❌ 删除 |
| `optuna_search_classifier.py` | ✅ 保留 |
| `check_optuna_results.py` | ✅ 保留 |
| `check_optuna_status.py` | ✅ 保留 |
| `export_optuna_results.py` | ✅ 保留 |
| `monitor_optuna.py` | ✅ 保留 |

**删除命令**：
```powershell
$optunaDir = "c:\Users\28414\PycharmProjects\接诉即办项目\scripts\training\optuna"
Remove-Item -Path "$optunaDir\optuna_search_10_trials.py" -Force
Remove-Item -Path "$optunaDir\optuna_search_100_trials.py" -Force
Remove-Item -Path "$optunaDir\optuna_search_default.py" -Force
Remove-Item -Path "$optunaDir\optuna_search_direct_env.py" -Force
Remove-Item -Path "$optunaDir\optuna_search_v3_custom.py" -Force
Remove-Item -Path "$optunaDir\run_optuna_50_trials.py" -Force
Remove-Item -Path "$optunaDir\run_optuna_search.py" -Force
Remove-Item -Path "$optunaDir\run_optuna_wrapper.py" -Force
Remove-Item -Path "$optunaDir\test_optuna_log.py" -Force
Remove-Item -Path "$optunaDir\test_optuna_quick.py" -Force
Remove-Item -Path "$optunaDir\test_optuna_setup.py" -Force
```

---

### 🟢 优先级 8：遗留测试文件（可选）

`tests/legacy/` 目录下的旧测试文件：

| 路径 | 说明 |
|-----|------|
| `tests/legacy/test_backup_model.py` | 备份模型测试 |
| `tests/legacy/test_prediction.py` | 旧版预测测试 |
| `tests/legacy/test_rag_bge_basic.py` | 旧版RAG测试 |
| `tests/legacy/test_rag_bge_extended.py` | 旧版RAG扩展测试 |
| `tests/legacy/test_rag_bge_quick.py` | 旧版快速测试 |
| `tests/legacy/test_training.py` | 旧版训练测试 |

**删除命令**：
```powershell
Remove-Item -Path "c:\Users\28414\PycharmProjects\接诉即办项目\tests\legacy" -Recurse -Force
```

---

### 🟢 优先级 9：调试脚本（可选）

`scripts/debug/` 目录下的调试脚本，生产环境可删除：

| 路径 | 说明 |
|-----|------|
| `scripts/debug/_debug_test.py` | 调试测试 |
| `scripts/debug/_simple_test.py` | 简单测试 |
| `scripts/debug/check_*.py` | 各种检查脚本 |

---

## 四、清理顺序建议

按照风险从低到高的顺序执行：

1. **第一步**：删除 Optuna trial 模型副本（优先级 1）
2. **第二步**：删除空目录（优先级 2）
3. **第三步**：删除旧模型的 training_state.pt（优先级 3）
4. **第四步**：删除 Python 缓存文件（优先级 5）
5. **第五步**：删除重复脚本（优先级 6、7）
6. **第六步**：删除旧分类模型目录（优先级 4）- **需确认不再使用**
7. **第七步**：删除遗留测试和调试脚本（优先级 8、9）- **可选**

---

## 五、预计释放空间汇总

| 优先级 | 清理内容 | 预计释放空间 |
|-------|---------|-------------|
| 1 | Optuna trial 产物 | ~2-3 GB |
| 2 | 空目录 | ~0 GB |
| 3 | training_state.pt | ~2-3 GB |
| 4 | 旧分类模型目录 | ~5.5 GB |
| 5 | Python 缓存 | ~10-50 MB |
| 6-7 | 重复脚本 | ~1 MB |
| 8-9 | 遗留测试/调试 | ~1 MB |
| **合计** | | **~10-12 GB** |

---

## 六、一键清理脚本

将以下内容保存为 `cleanup.ps1` 并执行：

```powershell
# 接诉即办项目清理脚本
# 请在执行前仔细阅读并确认

$projectRoot = "c:\Users\28414\PycharmProjects\接诉即办项目"

Write-Host "开始清理冗余文件..." -ForegroundColor Green

# 1. 删除 Optuna trial 产物
Write-Host "清理 Optuna trial 产物..." -ForegroundColor Yellow
Remove-Item -Path "$projectRoot\data\reports\training\optuna\trials" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path "$projectRoot\data\reports\training\optuna_trials" -Recurse -Force -ErrorAction SilentlyContinue

# 2. 删除空目录
Write-Host "清理空目录..." -ForegroundColor Yellow
Remove-Item -Path "$projectRoot\data\models\classifier\bert_cnn_attn_test" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path "$projectRoot\data\models\classifier\bert_cnn_attn_v1" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path "$projectRoot\checkpoints\classifier\final_model_fgm_master_smoke" -Recurse -Force -ErrorAction SilentlyContinue

# 3. 删除 Python 缓存
Write-Host "清理 Python 缓存..." -ForegroundColor Yellow
Get-ChildItem -Path $projectRoot -Recurse -Directory -Filter "__pycache__" | 
    Where-Object { $_.FullName -notlike "*\.venv\*" } | 
    Remove-Item -Recurse -Force

# 4. 删除重复脚本
Write-Host "清理重复脚本..." -ForegroundColor Yellow
$optunaDir = "$projectRoot\scripts\training\optuna"
@(
    "optuna_search_10_trials.py",
    "optuna_search_100_trials.py", 
    "optuna_search_default.py",
    "optuna_search_direct_env.py",
    "optuna_search_v3_custom.py",
    "run_optuna_50_trials.py",
    "run_optuna_search.py",
    "run_optuna_wrapper.py",
    "test_optuna_log.py",
    "test_optuna_quick.py",
    "test_optuna_setup.py"
) | ForEach-Object { Remove-Item -Path "$optunaDir\$_" -Force -ErrorAction SilentlyContinue }

Remove-Item -Path "$projectRoot\scripts\crawling\government\crawl_policy_beijing_gov_v1.py" -Force -ErrorAction SilentlyContinue

Write-Host "清理完成！" -ForegroundColor Green
Write-Host "如需删除旧模型目录，请手动执行优先级4的删除命令" -ForegroundColor Cyan
```

---

## 七、注意事项

1. **执行前备份**：建议在执行删除前备份重要数据
2. **分步执行**：建议按优先级分步执行，每步确认后再进行下一步
3. **保留主线模型**：确保 `final_model_hybrid_v3_32cls` 不被删除
4. **检查依赖**：删除模型目录前，确认没有其他脚本依赖该模型
5. **文档归档**：建议保留 `legacy/` 目录，仅删除代码文件

---

**文档生成时间**：2026-04-25