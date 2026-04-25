"""
输入验证和错误处理系统
实现输入清理、验证、统一错误处理
"""

import re
import html
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ValidationError(Exception):
    """验证错误异常"""
    def __init__(self, message: str, field: str = "", error_code: str = "VALIDATION_ERROR"):
        self.message = message
        self.field = field
        self.error_code = error_code
        super().__init__(self.message)


class SecurityLevel(Enum):
    """安全级别枚举"""
    LOW = "low"      # 低风险内容
    MEDIUM = "medium" # 中等风险内容  
    HIGH = "high"    # 高风险内容


@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    cleaned_value: str
    errors: List[str]
    warnings: List[str]


class InputValidator:
    """输入验证器"""
    
    # 恶意模式检测
    MALICIOUS_PATTERNS = [
        (r'<script[^>]*>.*?</script>', '脚本注入'),
        (r'javascript:', 'JavaScript注入'),
        (r'on\w+\s*=', '事件处理器注入'),
        (r'<iframe[^>]*>', 'iframe注入'),
        (r'\b(?:select|insert|update|delete|drop|create|alter)\b', 'SQL注入'),
        (r'\b(?:exec|xp_cmdshell|sp_)\b', '命令注入'),
        (r'\b(?:union|select)\b', 'SQL联合查询'),
        (r'\b(?:eval|exec|system)\b', '代码执行'),
    ]
    
    # 敏感词检测
    SENSITIVE_WORDS = [
        '习近平', '共产党', '政府', '领导人', '政治', '敏感词'
    ]
    
    def __init__(self, max_length: int = 5000, security_level: SecurityLevel = SecurityLevel.MEDIUM):
        self.max_length = max_length
        self.security_level = security_level
    
    def validate_text(self, text: str, field_name: str = "text", 
                     max_length: Optional[int] = None) -> ValidationResult:
        """验证文本输入"""
        max_len = max_length or self.max_length
        errors = []
        warnings = []
        
        # 基础验证
        if not text or text.strip() == "":
            errors.append(f"{field_name}不能为空")
            return ValidationResult(False, "", errors, warnings)
        
        if len(text) > max_len:
            errors.append(f"{field_name}长度不能超过{max_len}个字符")
        
        # HTML转义清理
        cleaned_text = html.escape(text)
        
        # 恶意模式检测
        malicious_detected = self._detect_malicious_patterns(text)
        if malicious_detected:
            if self.security_level == SecurityLevel.HIGH:
                errors.append(f"检测到潜在安全威胁: {malicious_detected}")
            else:
                warnings.append(f"检测到可疑内容: {malicious_detected}")
        
        # 敏感词检测（根据安全级别）
        if self.security_level == SecurityLevel.HIGH:
            sensitive_words = self._detect_sensitive_words(text)
            if sensitive_words:
                warnings.append(f"检测到敏感词汇: {', '.join(sensitive_words)}")
        
        # 清理多余空格和换行
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, cleaned_text, errors, warnings)
    
    def validate_request_data(self, data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """验证完整的请求数据"""
        cleaned_data = {}
        all_errors = []
        
        # 验证标签
        tag = data.get('tag', '')
        tag_result = self.validate_text(tag, '标签', 50)
        if tag_result.is_valid:
            cleaned_data['tag'] = tag_result.cleaned_value
        else:
            all_errors.extend(tag_result.errors)
        
        # 验证标题
        title = data.get('title', '')
        title_result = self.validate_text(title, '标题', 200)
        if title_result.is_valid:
            cleaned_data['title'] = title_result.cleaned_value
        else:
            all_errors.extend(title_result.errors)
        
        # 验证正文
        body = data.get('body', '')
        body_result = self.validate_text(body, '正文', 3000)
        if body_result.is_valid:
            cleaned_data['body'] = body_result.cleaned_value
        else:
            all_errors.extend(body_result.errors)
        
        # 验证强制单位（可选）
        forced_unit = data.get('_force_unit', '')
        if forced_unit:
            unit_result = self.validate_text(forced_unit, '强制单位', 100)
            if unit_result.is_valid:
                cleaned_data['_force_unit'] = unit_result.cleaned_value
            else:
                all_errors.extend(unit_result.errors)
        
        return cleaned_data, all_errors
    
    def _detect_malicious_patterns(self, text: str) -> Optional[str]:
        """检测恶意模式"""
        for pattern, description in self.MALICIOUS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return description
        return None
    
    def _detect_sensitive_words(self, text: str) -> List[str]:
        """检测敏感词"""
        detected = []
        for word in self.SENSITIVE_WORDS:
            if word.lower() in text.lower():
                detected.append(word)
        return detected


class ErrorHandler:
    """统一错误处理器"""
    
    def __init__(self, logger):
        self.logger = logger
    
    def handle_validation_error(self, errors: List[str]) -> Dict[str, Any]:
        """处理验证错误"""
        self.logger.warning("输入验证失败", extra={'errors': errors})
        return {
            "error": "输入验证失败",
            "details": errors,
            "error_code": "VALIDATION_ERROR"
        }
    
    def handle_model_error(self, error: Exception, model_type: str) -> Dict[str, Any]:
        """处理模型错误"""
        error_msg = f"{model_type}模型处理失败"
        self.logger.error(error_msg, extra={
            'model_type': model_type,
            'error': str(error),
            'error_type': type(error).__name__
        })
        
        # 生产环境中隐藏详细错误信息
        return {
            "error": "系统处理失败，请稍后重试",
            "error_code": "MODEL_ERROR"
        }
    
    def handle_rate_limit(self, client_ip: str) -> Dict[str, Any]:
        """处理频率限制"""
        self.logger.warning("频率限制触发", extra={'client_ip': client_ip})
        return {
            "error": "请求过于频繁，请稍候重试",
            "error_code": "RATE_LIMIT_EXCEEDED"
        }
    
    def handle_unexpected_error(self, error: Exception) -> Dict[str, Any]:
        """处理未知错误"""
        self.logger.error("未知错误发生", extra={
            'error': str(error),
            'error_type': type(error).__name__
        })
        
        return {
            "error": "系统内部错误，请联系管理员",
            "error_code": "INTERNAL_ERROR"
        }


# 全局验证器和错误处理器
validator = InputValidator()
error_handler = None  # 将在应用初始化时设置


def get_validator() -> InputValidator:
    """获取全局验证器"""
    return validator


def set_error_handler(handler: ErrorHandler) -> None:
    """设置全局错误处理器"""
    global error_handler
    error_handler = handler


def validate_and_clean_input(data: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """验证和清理输入数据"""
    cleaned_data, errors = validator.validate_request_data(data)
    
    if errors:
        return {}, error_handler.handle_validation_error(errors) if error_handler else {
            "error": "输入验证失败",
            "details": errors
        }
    
    return cleaned_data, None


# 错误处理装饰器
def safe_execute(error_message: str = "处理失败"):
    """安全执行装饰器，捕获异常并返回统一错误响应"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except ValidationError as e:
                if error_handler:
                    return error_handler.handle_validation_error([e.message])
                return {"error": e.message, "error_code": e.error_code}
            except Exception as e:
                if error_handler:
                    return error_handler.handle_unexpected_error(e)
                return {"error": error_message, "error_code": "UNKNOWN_ERROR"}
        return wrapper
    return decorator


if __name__ == "__main__":
    # 测试验证功能
    validator = InputValidator()
    
    # 测试正常输入
    normal_text = "小区垃圾没人清理"
    result = validator.validate_text(normal_text, "测试输入")
    print(f"正常输入验证: {result}")
    
    # 测试恶意输入
    malicious_text = "<script>alert('xss')</script>小区垃圾"
    result = validator.validate_text(malicious_text, "恶意输入")
    print(f"恶意输入验证: {result}")
    
    # 测试超长输入
    long_text = "a" * 6000
    result = validator.validate_text(long_text, "超长输入")
    print(f"超长输入验证: {result}")
    
    print("输入验证测试完成")