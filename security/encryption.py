"""
数据加密模块
实现敏感数据加密、解密、哈希等功能
"""

from __future__ import annotations

import os
import base64
import hashlib
import secrets
import json
from typing import Any, Dict, Optional
from pathlib import Path


try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False


class EncryptionManager:
    """加密管理器"""
    
    def __init__(self, key: str | bytes | None = None):
        """
        初始化加密管理器
        
        Args:
            key: 加密密钥，如果不提供则从环境变量读取或生成新密钥
        """
        self._key = self._get_or_create_key(key)
        
        if CRYPTO_AVAILABLE:
            self._fernet = Fernet(self._key)
        else:
            self._fernet = None
    
    def _get_or_create_key(self, key: str | bytes | None) -> bytes:
        """获取或创建密钥"""
        if key:
            if isinstance(key, str):
                return base64.urlsafe_b64encode(key.encode().ljust(32)[:32])
            return key
        
        env_key = os.environ.get("ENCRYPTION_KEY")
        if env_key:
            return base64.urlsafe_b64encode(env_key.encode().ljust(32)[:32])
        
        if CRYPTO_AVAILABLE:
            return Fernet.generate_key()
        else:
            return base64.urlsafe_b64encode(secrets.token_bytes(32))
    
    def encrypt(self, data: str | bytes) -> str:
        """
        加密数据
        
        Args:
            data: 待加密数据
            
        Returns:
            加密后的Base64字符串
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        if CRYPTO_AVAILABLE and self._fernet:
            encrypted = self._fernet.encrypt(data)
            return base64.urlsafe_b64encode(encrypted).decode('utf-8')
        else:
            encrypted = self._simple_encrypt(data)
            return base64.urlsafe_b64encode(encrypted).decode('utf-8')
    
    def decrypt(self, encrypted_data: str) -> str:
        """
        解密数据
        
        Args:
            encrypted_data: 加密的Base64字符串
            
        Returns:
            解密后的字符串
        """
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode('utf-8'))
        
        if CRYPTO_AVAILABLE and self._fernet:
            decrypted = self._fernet.decrypt(encrypted_bytes)
            return decrypted.decode('utf-8')
        else:
            decrypted = self._simple_decrypt(encrypted_bytes)
            return decrypted.decode('utf-8')
    
    def _simple_encrypt(self, data: bytes) -> bytes:
        """简单加密（无cryptography库时使用）"""
        key_bytes = self._key
        result = bytearray()
        
        for i, byte in enumerate(data):
            result.append(byte ^ key_bytes[i % len(key_bytes)])
        
        return bytes(result)
    
    def _simple_decrypt(self, data: bytes) -> bytes:
        """简单解密（无cryptography库时使用）"""
        return self._simple_encrypt(data)
    
    def encrypt_dict(self, data: Dict[str, Any]) -> str:
        """
        加密字典
        
        Args:
            data: 待加密的字典
            
        Returns:
            加密后的字符串
        """
        json_str = json.dumps(data, ensure_ascii=False)
        return self.encrypt(json_str)
    
    def decrypt_dict(self, encrypted_data: str) -> Dict[str, Any]:
        """
        解密字典
        
        Args:
            encrypted_data: 加密的字符串
            
        Returns:
            解密后的字典
        """
        json_str = self.decrypt(encrypted_data)
        return json.loads(json_str)


class HashManager:
    """哈希管理器"""
    
    @staticmethod
    def hash_password(password: str, salt: str | None = None) -> Dict[str, str]:
        """
        哈希密码
        
        Args:
            password: 密码
            salt: 盐值，如果不提供则生成新的
            
        Returns:
            包含哈希值和盐值的字典
        """
        if salt is None:
            salt = secrets.token_hex(16)
        
        hash_value = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        )
        
        return {
            "hash": base64.urlsafe_b64encode(hash_value).decode('utf-8'),
            "salt": salt
        }
    
    @staticmethod
    def verify_password(password: str, hash_value: str, salt: str) -> bool:
        """
        验证密码
        
        Args:
            password: 密码
            hash_value: 哈希值
            salt: 盐值
            
        Returns:
            是否匹配
        """
        result = HashManager.hash_password(password, salt)
        return result["hash"] == hash_value
    
    @staticmethod
    def hash_data(data: str, algorithm: str = "sha256") -> str:
        """
        哈希数据
        
        Args:
            data: 待哈希数据
            algorithm: 哈希算法 (md5/sha1/sha256/sha512)
            
        Returns:
            哈希值
        """
        hash_func = getattr(hashlib, algorithm, hashlib.sha256)
        return hash_func(data.encode('utf-8')).hexdigest()
    
    @staticmethod
    def hash_file(file_path: str | Path, algorithm: str = "sha256") -> str:
        """
        哈希文件
        
        Args:
            file_path: 文件路径
            algorithm: 哈希算法
            
        Returns:
            哈希值
        """
        hash_func = getattr(hashlib, algorithm, hashlib.sha256)()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()


class SecureDataStorage:
    """安全数据存储"""
    
    SENSITIVE_FIELDS = [
        "password", "token", "api_key", "secret",
        "phone", "email", "id_card", "bank_account"
    ]
    
    def __init__(self, storage_path: str | None = None):
        """
        初始化安全存储
        
        Args:
            storage_path: 存储路径
        """
        self.storage_path = Path(storage_path) if storage_path else self._get_default_path()
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.encryption = EncryptionManager()
    
    def _get_default_path(self) -> Path:
        """获取默认存储路径"""
        base_dir = Path(__file__).resolve().parents[1]
        data_dir = base_dir / "data" / "secure"
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir / "secure_data.json"
    
    def _is_sensitive(self, key: str) -> bool:
        """判断是否为敏感字段"""
        key_lower = key.lower()
        return any(field in key_lower for field in self.SENSITIVE_FIELDS)
    
    def _encrypt_sensitive_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """加密敏感字段"""
        result = {}
        
        for key, value in data.items():
            if self._is_sensitive(key) and isinstance(value, str):
                result[key] = {
                    "_encrypted": True,
                    "value": self.encryption.encrypt(value)
                }
            elif isinstance(value, dict):
                result[key] = self._encrypt_sensitive_fields(value)
            else:
                result[key] = value
        
        return result
    
    def _decrypt_sensitive_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """解密敏感字段"""
        result = {}
        
        for key, value in data.items():
            if isinstance(value, dict):
                if value.get("_encrypted"):
                    result[key] = self.encryption.decrypt(value["value"])
                else:
                    result[key] = self._decrypt_sensitive_fields(value)
            else:
                result[key] = value
        
        return result
    
    def save(self, data: Dict[str, Any], filename: str | None = None) -> bool:
        """
        保存数据（自动加密敏感字段）
        
        Args:
            data: 待保存数据
            filename: 文件名
            
        Returns:
            是否成功
        """
        target_path = self.storage_path if filename is None else self.storage_path.parent / filename
        
        encrypted_data = self._encrypt_sensitive_fields(data)
        
        try:
            with open(target_path, 'w', encoding='utf-8') as f:
                json.dump(encrypted_data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"保存失败: {e}")
            return False
    
    def load(self, filename: str | None = None) -> Dict[str, Any]:
        """
        加载数据（自动解密敏感字段）
        
        Args:
            filename: 文件名
            
        Returns:
            解密后的数据
        """
        target_path = self.storage_path if filename is None else self.storage_path.parent / filename
        
        if not target_path.exists():
            return {}
        
        try:
            with open(target_path, 'r', encoding='utf-8') as f:
                encrypted_data = json.load(f)
            
            return self._decrypt_sensitive_fields(encrypted_data)
        except Exception as e:
            print(f"加载失败: {e}")
            return {}
    
    def delete(self, filename: str | None = None) -> bool:
        """删除数据文件"""
        target_path = self.storage_path if filename is None else self.storage_path.parent / filename
        
        try:
            if target_path.exists():
                target_path.unlink()
            return True
        except Exception as e:
            print(f"删除失败: {e}")
            return False


class DataMasker:
    """数据脱敏"""
    
    @staticmethod
    def mask_phone(phone: str) -> str:
        """脱敏手机号"""
        if len(phone) != 11:
            return phone
        return f"{phone[:3]}****{phone[7:]}"
    
    @staticmethod
    def mask_email(email: str) -> str:
        """脱敏邮箱"""
        if '@' not in email:
            return email
        parts = email.split('@')
        username = parts[0]
        domain = parts[1]
        
        if len(username) <= 2:
            masked_username = '*' * len(username)
        else:
            masked_username = username[0] + '*' * (len(username) - 2) + username[-1]
        
        return f"{masked_username}@{domain}"
    
    @staticmethod
    def mask_id_card(id_card: str) -> str:
        """脱敏身份证号"""
        if len(id_card) < 8:
            return id_card
        return f"{id_card[:4]}********{id_card[-4:]}"
    
    @staticmethod
    def mask_bank_account(account: str) -> str:
        """脱敏银行账号"""
        if len(account) < 8:
            return account
        return f"{account[:4]}****{account[-4:]}"
    
    @staticmethod
    def mask_name(name: str) -> str:
        """脱敏姓名"""
        if len(name) <= 1:
            return name
        return name[0] + '*' * (len(name) - 1)
    
    @staticmethod
    def mask_address(address: str) -> str:
        """脱敏地址"""
        if len(address) <= 6:
            return address[:2] + '****'
        return address[:6] + '****' + address[-2:]
    
    @staticmethod
    def auto_mask(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        自动脱敏
        
        Args:
            data: 待脱敏数据
            
        Returns:
            脱敏后的数据
        """
        result = {}
        
        for key, value in data.items():
            key_lower = key.lower()
            
            if not isinstance(value, str):
                result[key] = value
                continue
            
            if 'phone' in key_lower or '手机' in key:
                result[key] = DataMasker.mask_phone(value)
            elif 'email' in key_lower or '邮箱' in key:
                result[key] = DataMasker.mask_email(value)
            elif 'id_card' in key_lower or '身份证' in key:
                result[key] = DataMasker.mask_id_card(value)
            elif 'bank' in key_lower or '银行卡' in key:
                result[key] = DataMasker.mask_bank_account(value)
            elif 'name' in key_lower or '姓名' in key:
                result[key] = DataMasker.mask_name(value)
            elif 'address' in key_lower or '地址' in key:
                result[key] = DataMasker.mask_address(value)
            else:
                result[key] = value
        
        return result


_encryption_manager: Optional[EncryptionManager] = None
_secure_storage: Optional[SecureDataStorage] = None


def get_encryption_manager() -> EncryptionManager:
    """获取加密管理器单例"""
    global _encryption_manager
    if _encryption_manager is None:
        _encryption_manager = EncryptionManager()
    return _encryption_manager


def get_secure_storage() -> SecureDataStorage:
    """获取安全存储单例"""
    global _secure_storage
    if _secure_storage is None:
        _secure_storage = SecureDataStorage()
    return _secure_storage


if __name__ == "__main__":
    print("数据加密模块测试")
    print("=" * 50)
    
    encryption = EncryptionManager()
    
    print("\n1. 字符串加密解密")
    original = "这是一段敏感数据"
    encrypted = encryption.encrypt(original)
    decrypted = encryption.decrypt(encrypted)
    print(f"原文: {original}")
    print(f"加密: {encrypted[:50]}...")
    print(f"解密: {decrypted}")
    
    print("\n2. 字典加密解密")
    data = {"user": "张三", "phone": "13800138000", "secret": "密码123"}
    encrypted_dict = encryption.encrypt_dict(data)
    decrypted_dict = encryption.decrypt_dict(encrypted_dict)
    print(f"原文: {data}")
    print(f"解密: {decrypted_dict}")
    
    print("\n3. 密码哈希")
    password = "mypassword123"
    hash_result = HashManager.hash_password(password)
    is_valid = HashManager.verify_password(password, hash_result["hash"], hash_result["salt"])
    print(f"密码: {password}")
    print(f"哈希: {hash_result['hash'][:30]}...")
    print(f"验证: {is_valid}")
    
    print("\n4. 数据脱敏")
    sensitive_data = {
        "name": "张三",
        "phone": "13800138000",
        "email": "zhangsan@example.com",
        "id_card": "110101199001011234",
        "address": "北京市朝阳区望京街道阜通东大街6号院"
    }
    masked_data = DataMasker.auto_mask(sensitive_data)
    print("原文:")
    print(json.dumps(sensitive_data, ensure_ascii=False, indent=2))
    print("\n脱敏后:")
    print(json.dumps(masked_data, ensure_ascii=False, indent=2))
    
    print("\n5. 安全存储")
    storage = SecureDataStorage()
    storage.save({"api_key": "sk_test123456", "user": "admin"})
    loaded = storage.load()
    print(f"加载的数据: {loaded}")
