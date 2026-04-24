"""
测试运行脚本
运行所有测试并生成报告
"""

import unittest
import sys
import json
from pathlib import Path
from datetime import datetime


def run_all_tests(verbose: bool = True):
    """运行所有测试"""
    project_root = Path(__file__).resolve().parents[1]
    tests_dir = project_root / "tests"
    
    loader = unittest.TestLoader()
    suite = loader.discover(str(tests_dir), pattern="test_*.py")
    
    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result = runner.run(suite)
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_tests": result.testsRun,
        "passed": result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped),
        "failed": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "success_rate": f"{((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.2f}%" if result.testsRun > 0 else "0%",
        "failures": [
            {
                "test": str(test),
                "traceback": traceback
            }
            for test, traceback in result.failures
        ],
        "errors": [
            {
                "test": str(test),
                "traceback": traceback
            }
            for test, traceback in result.errors
        ]
    }
    
    report_path = project_root / "test_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print("测试报告摘要")
    print(f"{'='*60}")
    print(f"总测试数: {report['total_tests']}")
    print(f"通过: {report['passed']}")
    print(f"失败: {report['failed']}")
    print(f"错误: {report['errors']}")
    print(f"跳过: {report['skipped']}")
    print(f"成功率: {report['success_rate']}")
    print(f"\n详细报告已保存到: {report_path}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
