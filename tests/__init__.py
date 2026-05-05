"""
测试框架
提供单元测试、集成测试和端到端测试的基础设施
"""

import unittest
import json
import tempfile
import os
from pathlib import Path
from typing import Any, Dict, List


def _on_rmtree_error(func, path, exc_info):
    import stat
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass


class BaseTestCase(unittest.TestCase):
    """测试基类"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        cls.test_dir = tempfile.mkdtemp()
        cls.test_data_dir = Path(cls.test_dir) / "test_data"
        cls.test_data_dir.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def tearDownClass(cls):
        from src.jsjb.feedback.repository import FeedbackDatabase
        FeedbackDatabase.reset_singleton()
        import shutil
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir, onerror=_on_rmtree_error)
    
    def create_test_file(self, filename: str, content: str) -> Path:
        """创建测试文件"""
        file_path = self.test_data_dir / filename
        file_path.write_text(content, encoding='utf-8')
        return file_path
    
    def create_test_json(self, filename: str, data: Dict[str, Any]) -> Path:
        """创建测试JSON文件"""
        file_path = self.test_data_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return file_path
    
    def assertDictContainsSubset(self, subset: Dict, dictionary: Dict):
        """断言字典包含子集"""
        for key, value in subset.items():
            self.assertIn(key, dictionary)
            self.assertEqual(dictionary[key], value)


class TestResult:
    """测试结果"""
    
    def __init__(self):
        self.total = 0
        self.passed = 0
        self.failed = 0
        self.errors = 0
        self.skipped = 0
        self.details: List[Dict[str, Any]] = []
    
    def add_result(self, test_name: str, status: str, message: str = "", duration: float = 0):
        """添加测试结果"""
        self.total += 1
        if status == "passed":
            self.passed += 1
        elif status == "failed":
            self.failed += 1
        elif status == "error":
            self.errors += 1
        elif status == "skipped":
            self.skipped += 1
        
        self.details.append({
            "test_name": test_name,
            "status": status,
            "message": message,
            "duration": duration
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "skipped": self.skipped,
            "success_rate": f"{(self.passed / self.total * 100):.2f}%" if self.total > 0 else "0%",
            "details": self.details
        }


def run_tests(test_pattern: str = "test_*.py", verbose: bool = True) -> TestResult:
    """
    运行测试
    
    Args:
        test_pattern: 测试文件模式
        verbose: 是否显示详细输出
        
    Returns:
        测试结果
    """
    loader = unittest.TestLoader()
    suite = loader.discover('tests', pattern=test_pattern)
    
    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result = runner.run(suite)
    
    test_result = TestResult()
    
    for test, traceback in result.failures:
        test_result.add_result(str(test), "failed", traceback)
    
    for test, traceback in result.errors:
        test_result.add_result(str(test), "error", traceback)
    
    for test in result.skipped:
        test_result.add_result(str(test[0]), "skipped")
    
    for test in result.success:
        test_result.add_result(str(test), "passed")
    
    return test_result


if __name__ == "__main__":
    result = run_tests()
    print("\n测试结果:")
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
