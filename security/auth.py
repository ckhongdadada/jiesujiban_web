"""
认证授权模块
实现用户认证、权限控制、API密钥管理
"""

from __future__ import annotations

import os
import json
import hashlib
import secrets
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from functools import wraps
import sqlite3


try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False


@dataclass
class User:
    """用户信息"""
    user_id: str
    username: str
    password_hash: str
    role: str
    api_key: str
    created_at: str
    last_login: str
    is_active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "username": self.username,
            "role": self.role,
            "created_at": self.created_at,
            "last_login": self.last_login,
            "is_active": self.is_active
        }


@dataclass
class APIKey:
    """API密钥"""
    key_id: str
    user_id: str
    api_key: str
    permissions: List[str]
    rate_limit: int
    created_at: str
    expires_at: str
    is_active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "key_id": self.key_id,
            "user_id": self.user_id,
            "permissions": self.permissions,
            "rate_limit": self.rate_limit,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "is_active": self.is_active
        }


class AuthManager:
    """认证管理器"""
    
    SECRET_KEY = os.environ.get("SECRET_KEY", secrets.token_hex(32))
    TOKEN_EXPIRE_HOURS = 24
    ALGORITHM = "HS256"
    
    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or self._get_default_db_path()
        self._init_database()
    
    def _get_default_db_path(self) -> str:
        """获取默认数据库路径"""
        base_dir = Path(__file__).resolve().parents[1]
        data_dir = base_dir / "data" / "runtime"
        data_dir.mkdir(parents=True, exist_ok=True)
        return str(data_dir / "auth.db")
    
    def _init_database(self) -> None:
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                api_key TEXT UNIQUE,
                created_at TEXT NOT NULL,
                last_login TEXT,
                is_active INTEGER DEFAULT 1
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_keys (
                key_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                api_key TEXT UNIQUE NOT NULL,
                permissions TEXT NOT NULL,
                rate_limit INTEGER DEFAULT 100,
                created_at TEXT NOT NULL,
                expires_at TEXT,
                is_active INTEGER DEFAULT 1,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS access_logs (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                api_key TEXT,
                endpoint TEXT,
                method TEXT,
                ip_address TEXT,
                timestamp TEXT NOT NULL,
                success INTEGER DEFAULT 1
            )
        ''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_username ON users(username)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_api_key ON api_keys(api_key)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON access_logs(timestamp)')
        
        conn.commit()
        conn.close()
    
    def _hash_password(self, password: str) -> str:
        """哈希密码"""
        salt = os.environ.get("PASSWORD_SALT", "default_salt")
        return hashlib.sha256(f"{password}{salt}".encode()).hexdigest()
    
    def _generate_user_id(self) -> str:
        """生成用户ID"""
        return f"user_{secrets.token_hex(8)}"
    
    def _generate_api_key(self) -> str:
        """生成API密钥"""
        return f"sk_{secrets.token_hex(16)}"
    
    def create_user(
        self,
        username: str,
        password: str,
        role: str = "user"
    ) -> Dict[str, Any]:
        """
        创建用户
        
        Args:
            username: 用户名
            password: 密码
            role: 角色 (admin/user/viewer)
            
        Returns:
            创建结果
        """
        password_hash = self._hash_password(password)
        user_id = self._generate_user_id()
        api_key = self._generate_api_key()
        created_at = datetime.now().isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO users (user_id, username, password_hash, role, api_key, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (user_id, username, password_hash, role, api_key, created_at))
            
            conn.commit()
            
            return {
                "success": True,
                "user_id": user_id,
                "username": username,
                "api_key": api_key,
                "message": "用户创建成功"
            }
        except sqlite3.IntegrityError:
            return {
                "success": False,
                "error": "用户名已存在"
            }
        finally:
            conn.close()
    
    def authenticate(self, username: str, password: str) -> Dict[str, Any]:
        """
        用户认证
        
        Args:
            username: 用户名
            password: 密码
            
        Returns:
            认证结果，包含访问令牌
        """
        password_hash = self._hash_password(password)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT user_id, username, role, api_key, is_active
            FROM users
            WHERE username = ? AND password_hash = ?
        ''', (username, password_hash))
        
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return {
                "success": False,
                "error": "用户名或密码错误"
            }
        
        user_id, username, role, api_key, is_active = row
        
        if not is_active:
            conn.close()
            return {
                "success": False,
                "error": "用户已被禁用"
            }
        
        cursor.execute('''
            UPDATE users SET last_login = ? WHERE user_id = ?
        ''', (datetime.now().isoformat(), user_id))
        
        conn.commit()
        conn.close()
        
        token = self._generate_token(user_id, username, role)
        
        return {
            "success": True,
            "access_token": token,
            "token_type": "Bearer",
            "expires_in": self.TOKEN_EXPIRE_HOURS * 3600,
            "user": {
                "user_id": user_id,
                "username": username,
                "role": role
            }
        }
    
    def _generate_token(self, user_id: str, username: str, role: str) -> str:
        """生成JWT令牌"""
        if not JWT_AVAILABLE:
            return f"{user_id}:{username}:{role}:{int(time.time())}"
        
        payload = {
            "user_id": user_id,
            "username": username,
            "role": role,
            "exp": datetime.utcnow() + timedelta(hours=self.TOKEN_EXPIRE_HOURS),
            "iat": datetime.utcnow()
        }
        
        return jwt.encode(payload, self.SECRET_KEY, algorithm=self.ALGORITHM)
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """
        验证令牌
        
        Args:
            token: 访问令牌
            
        Returns:
            验证结果
        """
        if not JWT_AVAILABLE:
            parts = token.split(":")
            if len(parts) >= 3:
                return {
                    "success": True,
                    "user_id": parts[0],
                    "username": parts[1],
                    "role": parts[2]
                }
            return {
                "success": False,
                "error": "无效的令牌"
            }
        
        try:
            payload = jwt.decode(token, self.SECRET_KEY, algorithms=[self.ALGORITHM])
            return {
                "success": True,
                "user_id": payload["user_id"],
                "username": payload["username"],
                "role": payload["role"]
            }
        except jwt.ExpiredSignatureError:
            return {
                "success": False,
                "error": "令牌已过期"
            }
        except jwt.InvalidTokenError:
            return {
                "success": False,
                "error": "无效的令牌"
            }
    
    def verify_api_key(self, api_key: str) -> Dict[str, Any]:
        """
        验证API密钥
        
        Args:
            api_key: API密钥
            
        Returns:
            验证结果
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT k.key_id, k.user_id, k.permissions, k.rate_limit, k.expires_at, k.is_active,
                   u.username, u.role
            FROM api_keys k
            JOIN users u ON k.user_id = u.user_id
            WHERE k.api_key = ?
        ''', (api_key,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return {
                "success": False,
                "error": "无效的API密钥"
            }
        
        key_id, user_id, permissions, rate_limit, expires_at, is_active, username, role = row
        
        if not is_active:
            return {
                "success": False,
                "error": "API密钥已被禁用"
            }
        
        if expires_at:
            expires_dt = datetime.fromisoformat(expires_at)
            if datetime.now() > expires_dt:
                return {
                    "success": False,
                    "error": "API密钥已过期"
                }
        
        return {
            "success": True,
            "key_id": key_id,
            "user_id": user_id,
            "username": username,
            "role": role,
            "permissions": json.loads(permissions),
            "rate_limit": rate_limit
        }
    
    def create_api_key(
        self,
        user_id: str,
        permissions: List[str] | None = None,
        rate_limit: int = 100,
        expires_days: int | None = None
    ) -> Dict[str, Any]:
        """
        创建API密钥
        
        Args:
            user_id: 用户ID
            permissions: 权限列表
            rate_limit: 速率限制 (请求/分钟)
            expires_days: 过期天数
            
        Returns:
            创建结果
        """
        key_id = f"key_{secrets.token_hex(8)}"
        api_key = self._generate_api_key()
        permissions = permissions or ["read"]
        created_at = datetime.now().isoformat()
        
        expires_at = None
        if expires_days:
            expires_at = (datetime.now() + timedelta(days=expires_days)).isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO api_keys (key_id, user_id, api_key, permissions, rate_limit, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (key_id, user_id, api_key, json.dumps(permissions), rate_limit, created_at, expires_at))
            
            conn.commit()
            
            return {
                "success": True,
                "key_id": key_id,
                "api_key": api_key,
                "permissions": permissions,
                "rate_limit": rate_limit,
                "expires_at": expires_at,
                "message": "API密钥创建成功"
            }
        except sqlite3.IntegrityError as e:
            return {
                "success": False,
                "error": str(e)
            }
        finally:
            conn.close()
    
    def log_access(
        self,
        user_id: str | None,
        api_key: str | None,
        endpoint: str,
        method: str,
        ip_address: str,
        success: bool = True
    ) -> None:
        """记录访问日志"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO access_logs (user_id, api_key, endpoint, method, ip_address, timestamp, success)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, api_key, endpoint, method, ip_address, datetime.now().isoformat(), 1 if success else 0))
        
        conn.commit()
        conn.close()
    
    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """获取用户信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT user_id, username, role, created_at, last_login, is_active
            FROM users
            WHERE user_id = ?
        ''', (user_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        return {
            "user_id": row[0],
            "username": row[1],
            "role": row[2],
            "created_at": row[3],
            "last_login": row[4],
            "is_active": bool(row[5])
        }
    
    def list_users(self) -> List[Dict[str, Any]]:
        """列出所有用户"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT user_id, username, role, created_at, last_login, is_active
            FROM users
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                "user_id": row[0],
                "username": row[1],
                "role": row[2],
                "created_at": row[3],
                "last_login": row[4],
                "is_active": bool(row[5])
            }
            for row in rows
        ]
    
    def deactivate_user(self, user_id: str) -> bool:
        """禁用用户"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE users SET is_active = 0 WHERE user_id = ?
        ''', (user_id,))
        
        affected = cursor.rowcount
        conn.commit()
        conn.close()
        
        return affected > 0
    
    def check_permission(self, role: str, permission: str) -> bool:
        """检查权限"""
        role_permissions = {
            "admin": ["read", "write", "delete", "admin"],
            "user": ["read", "write"],
            "viewer": ["read"]
        }
        
        return permission in role_permissions.get(role, [])


_auth_manager: Optional[AuthManager] = None


def get_auth_manager() -> AuthManager:
    """获取认证管理器单例"""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthManager()
    return _auth_manager


def require_auth(permissions: List[str] | None = None):
    """
    认证装饰器
    
    Args:
        permissions: 所需权限列表
    """
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args, **kwargs):
            from flask import request, jsonify
            
            auth_header = request.headers.get("Authorization", "")
            
            if not auth_header:
                return jsonify({
                    "success": False,
                    "error": "缺少认证信息"
                }), 401
            
            if auth_header.startswith("Bearer "):
                token = auth_header[7:]
                result = get_auth_manager().verify_token(token)
            else:
                result = get_auth_manager().verify_api_key(auth_header)
            
            if not result.get("success"):
                return jsonify({
                    "success": False,
                    "error": result.get("error", "认证失败")
                }), 401
            
            if permissions:
                role = result.get("role", "viewer")
                auth_manager = get_auth_manager()
                
                for permission in permissions:
                    if not auth_manager.check_permission(role, permission):
                        return jsonify({
                            "success": False,
                            "error": "权限不足"
                        }), 403
            
            request.current_user = result
            
            return f(*args, **kwargs)
        
        return decorated_function
    
    return decorator


if __name__ == "__main__":
    print("认证授权模块测试")
    print("=" * 50)
    
    auth = AuthManager()
    
    print("\n1. 创建用户")
    result = auth.create_user("admin", "admin123", "admin")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    print("\n2. 用户认证")
    result = auth.authenticate("admin", "admin123")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    if result.get("success"):
        token = result["access_token"]
        
        print("\n3. 验证令牌")
        result = auth.verify_token(token)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    
    print("\n4. 列出用户")
    users = auth.list_users()
    print(json.dumps(users, ensure_ascii=False, indent=2))
