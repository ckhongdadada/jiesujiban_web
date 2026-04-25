from __future__ import annotations

import os
import json
import threading
import time
import hashlib
from pathlib import Path
from typing import Any, Callable
from dataclasses import dataclass, field
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent


@dataclass
class ConfigItem:
    key: str
    value: Any
    default: Any
    type: str
    description: str
    mutable: bool = True
    validators: list[Callable] = field(default_factory=list)


class ConfigManager:
    def __init__(
        self,
        config_file: str = "config.json",
        watch_changes: bool = True
    ):
        self.config_file = Path(config_file)
        self.watch_changes = watch_changes
        
        self._config: dict[str, ConfigItem] = {}
        self._callbacks: dict[str, list[Callable]] = {}
        self._lock = threading.RLock()
        self._file_hash: str | None = None
        self._observer: Observer | None = None
        
        self._load_config()
        
        if watch_changes:
            self._start_watcher()
    
    def _load_config(self):
        with self._lock:
            if self.config_file.exists():
                try:
                    with open(self.config_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        self._file_hash = hashlib.md5(content.encode()).hexdigest()
                        data = json.loads(content)
                        
                        for key, value in data.items():
                            if key in self._config:
                                self._update_config_value(key, value)
                            else:
                                self._config[key] = ConfigItem(
                                    key=key,
                                    value=value,
                                    default=value,
                                    type=type(value).__name__,
                                    description="",
                                    mutable=True
                                )
                except Exception as e:
                    print(f"[配置管理器] 加载配置失败: {e}")
    
    def _start_watcher(self):
        if not self.config_file.exists():
            return
        
        class ConfigFileHandler(FileSystemEventHandler):
            def __init__(self, manager):
                self.manager = manager
            
            def on_modified(self, event):
                if isinstance(event, FileModifiedEvent):
                    if Path(event.src_path).resolve() == self.manager.config_file.resolve():
                        self.manager._check_and_reload()
        
        self._observer = Observer()
        self._observer.schedule(
            ConfigFileHandler(self),
            str(self.config_file.parent),
            recursive=False
        )
        self._observer.start()
    
    def _check_and_reload(self):
        if not self.config_file.exists():
            return
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                content = f.read()
                new_hash = hashlib.md5(content.encode()).hexdigest()
                
                if new_hash != self._file_hash:
                    self._file_hash = new_hash
                    data = json.loads(content)
                    
                    changes = []
                    with self._lock:
                        for key, value in data.items():
                            if key in self._config:
                                old_value = self._config[key].value
                                if old_value != value:
                                    self._update_config_value(key, value)
                                    changes.append((key, old_value, value))
                            else:
                                self._config[key] = ConfigItem(
                                    key=key,
                                    value=value,
                                    default=value,
                                    type=type(value).__name__,
                                    description="",
                                    mutable=True
                                )
                                changes.append((key, None, value))
                    
                    for key, old_value, new_value in changes:
                        self._notify_callbacks(key, old_value, new_value)
                    
                    print(f"[配置管理器] 配置已热更新，变更项: {len(changes)}")
        except Exception as e:
            print(f"[配置管理器] 热更新失败: {e}")
    
    def _update_config_value(self, key: str, value: Any):
        if key in self._config:
            item = self._config[key]
            if not item.mutable:
                raise ValueError(f"配置项 '{key}' 不可修改")
            
            for validator in item.validators:
                if not validator(value):
                    raise ValueError(f"配置项 '{key}' 的值 '{value}' 验证失败")
            
            item.value = value
    
    def _notify_callbacks(self, key: str, old_value: Any, new_value: Any):
        if key in self._callbacks:
            for callback in self._callbacks[key]:
                try:
                    callback(key, old_value, new_value)
                except Exception as e:
                    print(f"[配置管理器] 回调执行失败: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            if key in self._config:
                return self._config[key].value
            return default
    
    def set(self, key: str, value: Any, persist: bool = False) -> bool:
        with self._lock:
            try:
                old_value = self._config[key].value if key in self._config else None
                self._update_config_value(key, value)
                
                if persist:
                    self._persist_config()
                
                self._notify_callbacks(key, old_value, value)
                return True
            except Exception as e:
                print(f"[配置管理器] 设置配置失败: {e}")
                return False
    
    def _persist_config(self):
        with self._lock:
            data = {key: item.value for key, item in self._config.items()}
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self._file_hash = hashlib.md5(json.dumps(data).encode()).hexdigest()
    
    def register_callback(
        self,
        key: str,
        callback: Callable[[str, Any, Any], None]
    ):
        with self._lock:
            if key not in self._callbacks:
                self._callbacks[key] = []
            self._callbacks[key].append(callback)
    
    def unregister_callback(
        self,
        key: str,
        callback: Callable
    ):
        with self._lock:
            if key in self._callbacks and callback in self._callbacks[key]:
                self._callbacks[key].remove(callback)
    
    def register_config(
        self,
        key: str,
        default: Any,
        type: str = None,
        description: str = "",
        mutable: bool = True,
        validators: list[Callable] = None
    ):
        with self._lock:
            if key not in self._config:
                self._config[key] = ConfigItem(
                    key=key,
                    value=default,
                    default=default,
                    type=type or type(default).__name__,
                    description=description,
                    mutable=mutable,
                    validators=validators or []
                )
    
    def get_all(self) -> dict[str, Any]:
        with self._lock:
            return {key: item.value for key, item in self._config.items()}
    
    def get_metadata(self) -> dict[str, dict]:
        with self._lock:
            return {
                key: {
                    'value': item.value,
                    'default': item.default,
                    'type': item.type,
                    'description': item.description,
                    'mutable': item.mutable
                }
                for key, item in self._config.items()
            }
    
    def stop(self):
        if self._observer:
            self._observer.stop()
            self._observer.join()


class DynamicRateLimiter:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        
        self.config_manager.register_config(
            key='rate_limit_seconds',
            default=3.0,
            type='float',
            description='请求频率限制（秒）',
            mutable=True,
            validators=[lambda x: x >= 0]
        )
        
        self.config_manager.register_config(
            key='rate_limit_max_requests',
            default=100,
            type='int',
            description='最大并发请求数',
            mutable=True,
            validators=[lambda x: x > 0]
        )
        
        self._request_times: dict[str, list[float]] = {}
        self._lock = threading.Lock()
    
    def check_rate_limit(self, client_ip: str) -> tuple[bool, float]:
        rate_limit_seconds = self.config_manager.get('rate_limit_seconds', 3.0)
        max_requests = self.config_manager.get('rate_limit_max_requests', 100)
        
        with self._lock:
            now = time.time()
            
            if client_ip not in self._request_times:
                self._request_times[client_ip] = []
            
            self._request_times[client_ip] = [
                t for t in self._request_times[client_ip]
                if now - t < rate_limit_seconds * 10
            ]
            
            recent_requests = [
                t for t in self._request_times[client_ip]
                if now - t < rate_limit_seconds
            ]
            
            if len(recent_requests) >= max_requests:
                oldest = min(recent_requests)
                wait_time = rate_limit_seconds - (now - oldest)
                return False, wait_time
            
            self._request_times[client_ip].append(now)
            return True, 0
    
    def get_current_limit(self) -> dict:
        return {
            'rate_limit_seconds': self.config_manager.get('rate_limit_seconds'),
            'max_requests': self.config_manager.get('rate_limit_max_requests')
        }


class FeatureFlags:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        
        self._register_default_flags()
    
    def _register_default_flags(self):
        flags = {
            'enable_disambiguation': True,
            'enable_cache': True,
            'enable_batch_processing': True,
            'enable_metrics': True,
            'enable_detailed_logging': False,
            'enable_geo_validation': False,
            'enable_context_disambiguation': True,
        }
        
        for flag, default in flags.items():
            self.config_manager.register_config(
                key=f'feature_{flag}',
                default=default,
                type='bool',
                description=f'功能开关: {flag}',
                mutable=True
            )
    
    def is_enabled(self, flag: str) -> bool:
        return self.config_manager.get(f'feature_{flag}', False)
    
    def enable(self, flag: str):
        self.config_manager.set(f'feature_{flag}', True)
    
    def disable(self, flag: str):
        self.config_manager.set(f'feature_{flag}', False)
    
    def toggle(self, flag: str) -> bool:
        current = self.is_enabled(flag)
        self.config_manager.set(f'feature_{flag}', not current)
        return not current
    
    def get_all_flags(self) -> dict[str, bool]:
        flags = {}
        for key, item in self.config_manager._config.items():
            if key.startswith('feature_'):
                flag_name = key[8:]
                flags[flag_name] = item.value
        return flags


config_manager: ConfigManager | None = None
rate_limiter: DynamicRateLimiter | None = None
feature_flags: FeatureFlags | None = None


def init_config_manager(
    config_file: str = "config.json",
    watch_changes: bool = True
) -> ConfigManager:
    global config_manager, rate_limiter, feature_flags
    
    config_manager = ConfigManager(config_file, watch_changes)
    rate_limiter = DynamicRateLimiter(config_manager)
    feature_flags = FeatureFlags(config_manager)
    
    return config_manager


def get_config_manager() -> ConfigManager | None:
    return config_manager


def get_rate_limiter() -> DynamicRateLimiter | None:
    return rate_limiter


def get_feature_flags() -> FeatureFlags | None:
    return feature_flags
