"""模型配置管理器

支持从环境变量读取不同服务商的配置信息。
"""

import os
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """模型配置类"""
    api_key: str
    base_url: str
    model_name: str
    service_name: str


class ConfigManager:
    """配置管理器
    
    从环境变量读取不同服务商的配置信息。
    配置格式：{SERVICE}_API_KEY, {SERVICE}_BASE_URL, {SERVICE}_MODEL_NAME
    """
    
    def __init__(self):
        self._configs: Dict[str, ModelConfig] = {}
        self._load_configs()
    
    def _load_configs(self):
        """从环境变量加载所有可用的服务配置"""
        # 获取所有以 _API_KEY 结尾的环境变量
        api_key_vars = [key for key in os.environ.keys() if key.endswith('_API_KEY')]
        
        for api_key_var in api_key_vars:
            service_name = api_key_var.replace('_API_KEY', '')
            
            api_key = os.getenv(api_key_var)
            base_url = os.getenv(f"{service_name}_BASE_URL")
            model_name = os.getenv(f"{service_name}_MODEL_NAME")
            
            if api_key and base_url and model_name:
                config = ModelConfig(
                    api_key=api_key,
                    base_url=base_url,
                    model_name=model_name,
                    service_name=service_name
                )
                self._configs[service_name] = config
    
    def get_config(self, service_name: str) -> Optional[ModelConfig]:
        """获取指定服务的配置
        
        Args:
            service_name: 服务名称（如 'deepseek', 'openai' 等）
            
        Returns:
            ModelConfig 对象，如果服务不存在则返回 None
        """
        return self._configs.get(service_name)
    
    def list_services(self) -> list[str]:
        """列出所有可用的服务
        
        Returns:
            可用服务名称列表
        """
        return list(self._configs.keys())
    
    def has_service(self, service_name: str) -> bool:
        """检查服务是否可用
        
        Args:
            service_name: 服务名称
            
        Returns:
            服务是否可用
        """
        return service_name in self._configs