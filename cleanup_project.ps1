# cleanup_project.ps1
# 项目文件自动整理脚本

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  接诉即办项目文件整理脚本" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 询问用户是否继续
$confirm = Read-Host "此脚本将移动和删除文件，是否继续？(y/n)"
if ($confirm -ne 'y' -and $confirm -ne 'Y') {
    Write-Host "操作已取消" -ForegroundColor Yellow
    exit
}

Write-Host "`n开始项目整理..." -ForegroundColor Green

# 1. 创建目录
Write-Host "`n[1/5] 创建必要目录..." -ForegroundColor Yellow
$directories = @(
    "archive/old_scripts",
    "archive/test_logs",
    "archive/deprecated_docs",
    "docs",
    "tools/testing"
)

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "  ✓ 创建 $dir" -ForegroundColor Green
    } else {
        Write-Host "  ○ $dir 已存在" -ForegroundColor Gray
    }
}

# 2. 移动文档到 docs/
Write-Host "`n[2/5] 移动文档文件到 docs/..." -ForegroundColor Yellow
$docs = @(
    "PROJECT_REVIEW.md",
    "TRAINING_README.md",
    "TRAINING_ARTIFACTS_README.md",
    "ENHANCEMENT_README.md",
    "RAG_ENGINEERING_README.md",
    "CRAWLER_README.md",
    "STABILITY_README.md",
    "技术架构与原理详细解析报告.md",
    "技术架构精简版.md",
    "项目技术概念详解.md"
)

$moved_docs = 0
foreach ($doc in $docs) {
    if (Test-Path $doc) {
        Move-Item -Path $doc -Destination "docs/" -Force
        Write-Host "  ✓ 移动 $doc" -ForegroundColor Green
        $moved_docs++
    } else {
        Write-Host "  ○ $doc 不存在，跳过" -ForegroundColor Gray
    }
}
Write-Host "  共移动 $moved_docs 个文档文件" -ForegroundColor Cyan

# 3. 归档过时文档
Write-Host "`n[3/5] 归档过时文档..." -ForegroundColor Yellow
$deprecated = @(
    "PROJECT_OPTIMIZATION.md",
    "PROJECT_TREE.md"
)

$archived_docs = 0
foreach ($doc in $deprecated) {
    if (Test-Path $doc) {
        Move-Item -Path $doc -Destination "archive/deprecated_docs/" -Force
        Write-Host "  ✓ 归档 $doc" -ForegroundColor Green
        $archived_docs++
    } else {
        Write-Host "  ○ $doc 不存在，跳过" -ForegroundColor Gray
    }
}
Write-Host "  共归档 $archived_docs 个过时文档" -ForegroundColor Cyan

# 4. 移动测试文件
Write-Host "`n[4/5] 移动测试文件..." -ForegroundColor Yellow
if (Test-Path "测试生成回复.py") {
    Move-Item -Path "测试生成回复.py" -Destination "tools/testing/" -Force
    Write-Host "  ✓ 移动 测试生成回复.py 到 tools/testing/" -ForegroundColor Green
} else {
    Write-Host "  ○ 测试生成回复.py 不存在，跳过" -ForegroundColor Gray
}

if (Test-Path "test_logs") {
    Move-Item -Path "test_logs" -Destination "archive/" -Force
    Write-Host "  ✓ 移动 test_logs/ 到 archive/" -ForegroundColor Green
} else {
    Write-Host "  ○ test_logs/ 不存在，跳过" -ForegroundColor Gray
}

# 5. 删除空目录和备份文件
Write-Host "`n[5/5] 清理空目录和备份文件..." -ForegroundColor Yellow
$deleted = 0

if (Test-Path "test") {
    $isEmpty = (Get-ChildItem -Path "test" -Force | Measure-Object).Count -eq 0
    if ($isEmpty) {
        Remove-Item -Path "test" -Force -Recurse
        Write-Host "  ✓ 删除空目录 test/" -ForegroundColor Green
        $deleted++
    } else {
        Write-Host "  ⚠ test/ 不为空，跳过删除" -ForegroundColor Yellow
    }
} else {
    Write-Host "  ○ test/ 不存在，跳过" -ForegroundColor Gray
}

if (Test-Path "legacy/enhancements/rag_retriever.py.backup") {
    Remove-Item -Path "legacy/enhancements/rag_retriever.py.backup" -Force
    Write-Host "  ✓ 删除备份文件 rag_retriever.py.backup" -ForegroundColor Green
    $deleted++
} else {
    Write-Host "  ○ rag_retriever.py.backup 不存在，跳过" -ForegroundColor Gray
}

Write-Host "  共删除 $deleted 个文件/目录" -ForegroundColor Cyan

# 6. 可选：清理 __pycache__
Write-Host "`n[可选] 是否清理所有 __pycache__ 目录？(y/n)" -ForegroundColor Yellow
$clean_cache = Read-Host
if ($clean_cache -eq 'y' -or $clean_cache -eq 'Y') {
    $caches = Get-ChildItem -Path . -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue
    $cache_count = ($caches | Measure-Object).Count
    if ($cache_count -gt 0) {
        $caches | Remove-Item -Recurse -Force
        Write-Host "  ✓ 删除 $cache_count 个 __pycache__ 目录" -ForegroundColor Green
    } else {
        Write-Host "  ○ 未找到 __pycache__ 目录" -ForegroundColor Gray
    }
}

# 完成
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  ✅ 项目整理完成！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan

Write-Host "`n📋 整理总结：" -ForegroundColor Cyan
Write-Host "  • 移动文档：$moved_docs 个" -ForegroundColor White
Write-Host "  • 归档文档：$archived_docs 个" -ForegroundColor White
Write-Host "  • 删除文件：$deleted 个" -ForegroundColor White

Write-Host "`n⚠️ 请检查以下内容：" -ForegroundColor Yellow
Write-Host "  1. 文档是否正确移动到 docs/ 目录"
Write-Host "  2. 测试文件是否正确移动到 tools/testing/"
Write-Host "  3. 运行 python app.py 测试应用是否正常"
Write-Host "  4. 如果使用Git，请提交变更"

Write-Host "`n💡 建议的Git命令：" -ForegroundColor Cyan
Write-Host "  git status" -ForegroundColor Gray
Write-Host "  git add ." -ForegroundColor Gray
Write-Host "  git commit -m '项目结构整理：移动文档到docs/，归档过时文件'" -ForegroundColor Gray

Write-Host ""
