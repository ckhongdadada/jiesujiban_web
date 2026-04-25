"""
安全模块单元测试
"""

import unittest
from tests import BaseTestCase
from security.auth import AuthManager
from security.encryption import (
    EncryptionManager, HashManager, SecureDataStorage, DataMasker
)


class TestAuthManager(BaseTestCase):
    """认证管理器测试"""
    
    def setUp(self):
        """每个测试前的初始化"""
        self.db_path = str(self.test_data_dir / "test_auth.db")
        self.auth = AuthManager(self.db_path)
    
    def test_create_user(self):
        """测试创建用户"""
        result = self.auth.create_user("testuser", "password123", "user")
        
        self.assertTrue(result["success"])
        self.assertEqual(result["username"], "testuser")
        self.assertIsNotNone(result["api_key"])
    
    def test_create_duplicate_user(self):
        """测试创建重复用户"""
        self.auth.create_user("user1", "pass1")
        result = self.auth.create_user("user1", "pass2")
        
        self.assertFalse(result["success"])
    
    def test_authenticate_success(self):
        """测试认证成功"""
        self.auth.create_user("authuser", "correctpass")
        result = self.auth.authenticate("authuser", "correctpass")
        
        self.assertTrue(result["success"])
        self.assertIsNotNone(result["access_token"])
    
    def test_authenticate_wrong_password(self):
        """测试认证失败"""
        self.auth.create_user("authuser2", "correctpass")
        result = self.auth.authenticate("authuser2", "wrongpass")
        
        self.assertFalse(result["success"])
    
    def test_verify_token(self):
        """测试令牌验证"""
        self.auth.create_user("tokenuser", "pass")
        auth_result = self.auth.authenticate("tokenuser", "pass")
        
        token = auth_result["access_token"]
        verify_result = self.auth.verify_token(token)
        
        self.assertTrue(verify_result["success"])
        self.assertEqual(verify_result["username"], "tokenuser")
    
    def test_create_api_key(self):
        """测试创建API密钥"""
        user_result = self.auth.create_user("apiuser", "pass")
        user_id = user_result["user_id"]
        
        key_result = self.auth.create_api_key(user_id, ["read", "write"])
        
        self.assertTrue(key_result["success"])
        self.assertIsNotNone(key_result["api_key"])
    
    def test_verify_api_key(self):
        """测试API密钥验证"""
        user_result = self.auth.create_user("apikeyuser", "pass")
        user_id = user_result["user_id"]
        
        key_result = self.auth.create_api_key(user_id)
        api_key = key_result["api_key"]
        
        verify_result = self.auth.verify_api_key(api_key)
        
        self.assertTrue(verify_result["success"])
    
    def test_check_permission(self):
        """测试权限检查"""
        self.assertTrue(self.auth.check_permission("admin", "admin"))
        self.assertTrue(self.auth.check_permission("admin", "read"))
        self.assertTrue(self.auth.check_permission("user", "read"))
        self.assertFalse(self.auth.check_permission("user", "admin"))
        self.assertFalse(self.auth.check_permission("viewer", "write"))


class TestEncryptionManager(BaseTestCase):
    """加密管理器测试"""
    
    def setUp(self):
        """每个测试前的初始化"""
        self.encryption = EncryptionManager()
    
    def test_encrypt_decrypt_string(self):
        """测试字符串加密解密"""
        original = "这是一段测试文本"
        
        encrypted = self.encryption.encrypt(original)
        decrypted = self.encryption.decrypt(encrypted)
        
        self.assertNotEqual(original, encrypted)
        self.assertEqual(original, decrypted)
    
    def test_encrypt_decrypt_dict(self):
        """测试字典加密解密"""
        original = {"key": "value", "number": 123}
        
        encrypted = self.encryption.encrypt_dict(original)
        decrypted = self.encryption.decrypt_dict(encrypted)
        
        self.assertEqual(original, decrypted)
    
    def test_different_encryptions(self):
        """测试相同内容不同加密结果"""
        text = "same text"
        
        encrypted1 = self.encryption.encrypt(text)
        encrypted2 = self.encryption.encrypt(text)
        
        if hasattr(self.encryption, '_fernet') and self.encryption._fernet:
            self.assertNotEqual(encrypted1, encrypted2)


class TestHashManager(BaseTestCase):
    """哈希管理器测试"""
    
    def test_hash_password(self):
        """测试密码哈希"""
        password = "mypassword"
        
        result = HashManager.hash_password(password)
        
        self.assertIn("hash", result)
        self.assertIn("salt", result)
    
    def test_verify_password(self):
        """测试密码验证"""
        password = "testpassword"
        
        hash_result = HashManager.hash_password(password)
        
        self.assertTrue(
            HashManager.verify_password(
                password,
                hash_result["hash"],
                hash_result["salt"]
            )
        )
        
        self.assertFalse(
            HashManager.verify_password(
                "wrongpassword",
                hash_result["hash"],
                hash_result["salt"]
            )
        )
    
    def test_hash_data(self):
        """测试数据哈希"""
        data = "test data"
        
        hash1 = HashManager.hash_data(data)
        hash2 = HashManager.hash_data(data)
        
        self.assertEqual(hash1, hash2)
        self.assertEqual(len(hash1), 64)  # SHA-256


class TestSecureDataStorage(BaseTestCase):
    """安全存储测试"""
    
    def setUp(self):
        """每个测试前的初始化"""
        self.storage_path = str(self.test_data_dir / "secure")
        self.storage = SecureDataStorage(self.storage_path)
    
    def test_save_and_load(self):
        """测试保存和加载"""
        data = {
            "username": "admin",
            "password": "secret123",
            "api_key": "sk_test_key"
        }
        
        self.storage.save(data)
        loaded = self.storage.load()
        
        self.assertEqual(loaded["username"], "admin")
        self.assertEqual(loaded["password"], "secret123")
        self.assertEqual(loaded["api_key"], "sk_test_key")
    
    def test_auto_encrypt_sensitive_fields(self):
        """测试自动加密敏感字段"""
        data = {
            "name": "张三",
            "password": "mypassword",
            "token": "mytoken"
        }
        
        self.storage.save(data)
        
        import json
        with open(self.storage.storage_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        self.assertNotIn("_encrypted", raw_data.get("name", {}))
        self.assertTrue(raw_data.get("password", {}).get("_encrypted", False))
        self.assertTrue(raw_data.get("token", {}).get("_encrypted", False))


class TestDataMasker(BaseTestCase):
    """数据脱敏测试"""
    
    def test_mask_phone(self):
        """测试手机号脱敏"""
        phone = "13800138000"
        masked = DataMasker.mask_phone(phone)
        
        self.assertEqual(masked, "138****8000")
    
    def test_mask_email(self):
        """测试邮箱脱敏"""
        email = "zhangsan@example.com"
        masked = DataMasker.mask_email(email)
        
        self.assertIn("*", masked)
        self.assertIn("@", masked)
    
    def test_mask_id_card(self):
        """测试身份证脱敏"""
        id_card = "110101199001011234"
        masked = DataMasker.mask_id_card(id_card)
        
        self.assertEqual(masked[:4], "1101")
        self.assertEqual(masked[-4:], "1234")
        self.assertIn("*", masked)
    
    def test_auto_mask(self):
        """测试自动脱敏"""
        data = {
            "name": "张三",
            "phone": "13800138000",
            "email": "test@example.com",
            "id_card": "110101199001011234"
        }
        
        masked = DataMasker.auto_mask(data)
        
        self.assertNotEqual(masked["phone"], data["phone"])
        self.assertNotEqual(masked["email"], data["email"])
        self.assertNotEqual(masked["id_card"], data["id_card"])


if __name__ == "__main__":
    unittest.main()
