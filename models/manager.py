"""统一的模型管理器

整合配置管理和客户端功能，提供简化的模型调用接口。
"""

import logging
from typing import Dict, List, Optional, Any, Iterator
from .config import ConfigManager, ModelConfig
from .client import ModelClient


class ModelManager:
    """统一的模型管理器
    
    提供简化的大模型调用接口，自动管理配置和客户端。
    """
    
    def __init__(self, 
                 default_service: Optional[str] = None,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 model_name: Optional[str] = None):
        """初始化模型管理器
        
        Args:
            default_service: 默认使用的服务名称
            api_key: 直接指定的API密钥（优先级高于环境变量）
            base_url: 直接指定的API基础URL
            model_name: 直接指定的模型名称
        """
        self.logger = logging.getLogger('AgentsChat.ModelManager')
        self.config_manager = ConfigManager()
        self._clients: Dict[str, ModelClient] = {}
        self.default_service = default_service
        self._direct_config = None
        
        # 如果提供了直接配置参数，创建直接配置
        if api_key and base_url and model_name:
            self._direct_config = ModelConfig(
                api_key=api_key,
                base_url=base_url,
                model_name=model_name,
                service_name='DIRECT'
            )
            self.default_service = 'DIRECT'
            self.logger.info("使用直接传入的API配置")
        else:
            # 如果指定了默认服务，验证其可用性
            if default_service and not self.config_manager.has_service(default_service):
                available_services = self.config_manager.list_services()
                raise ValueError(f"默认服务 '{default_service}' 不可用。可用服务: {available_services}")
            
            # 如果没有指定默认服务，使用第一个可用的服务
            if not default_service:
                available_services = self.config_manager.list_services()
                if available_services:
                    self.default_service = available_services[0]
                    self.logger.info(f"使用默认服务: {self.default_service}")
    
    def get_client(self, service_name: Optional[str] = None) -> ModelClient:
        """获取指定服务的客户端
        
        Args:
            service_name: 服务名称，如果为 None 则使用默认服务
            
        Returns:
            ModelClient 实例
        """
        service = service_name or self.default_service
        
        if not service:
            raise ValueError("没有可用的服务，请检查环境变量配置")
        
        if service not in self._clients:
            # 如果是直接配置的服务
            if service == 'DIRECT' and self._direct_config:
                config = self._direct_config
            else:
                config = self.config_manager.get_config(service)
                if not config:
                    available_services = self.config_manager.list_services()
                    raise ValueError(f"服务 '{service}' 不可用。可用服务: {available_services}")
            
            self._clients[service] = ModelClient(config)
            self.logger.info(f"创建客户端: {service}")
        
        return self._clients[service]
    
    def chat(self, 
             message: str, 
             service_name: Optional[str] = None,
             **kwargs) -> str:
        """简单聊天接口
        
        Args:
            message: 用户消息
            service_name: 服务名称，如果为 None 则使用默认服务
            **kwargs: 其他参数
            
        Returns:
            模型回复内容
        """
        client = self.get_client(service_name)
        return client.simple_chat(message, **kwargs)
    
    def chat_completion(self, 
                       messages: List[Dict[str, str]], 
                       service_name: Optional[str] = None,
                       **kwargs) -> Dict[str, Any]:
        """聊天补全接口
        
        Args:
            messages: 消息列表
            service_name: 服务名称，如果为 None 则使用默认服务
            **kwargs: 其他参数
            
        Returns:
            API 响应结果
        """
        client = self.get_client(service_name)
        beta_header = None
        if self.default_service == "CLAUDE":
            beta_header = {
                "anthropic-beta": "context-1m-2025-08-07"
            }
        
        return client.chat_completion(messages, service_name=self.default_service, extra_headers=beta_header, **kwargs)
    
    def stream_chat(self, 
                    messages: List[Dict[str, str]], 
                    service_name: Optional[str] = None,
                    **kwargs) -> Iterator[Dict[str, Any]]:
        """流式聊天接口
        
        Args:
            messages: 消息列表
            service_name: 服务名称，如果为 None 则使用默认服务
            **kwargs: 其他参数
            
        Yields:
            流式响应数据
        """
        client = self.get_client(service_name)
        return client.chat_completion(messages, stream=True, **kwargs)
    
    def list_services(self) -> List[str]:
        """列出所有可用的服务
        
        Returns:
            可用服务名称列表
        """
        return self.config_manager.list_services()
    
    def get_service_info(self, service_name: Optional[str] = None) -> Dict[str, Any]:
        """获取服务信息
        
        Args:
            service_name: 服务名称，如果为 None 则使用默认服务
            
        Returns:
            服务信息
        """
        client = self.get_client(service_name)
        return client.get_model_info()
    
    def set_default_service(self, service_name: str):
        """设置默认服务
        
        Args:
            service_name: 服务名称
        """
        if not self.config_manager.has_service(service_name):
            available_services = self.config_manager.list_services()
            raise ValueError(f"服务 '{service_name}' 不可用。可用服务: {available_services}")
        
        self.default_service = service_name
        self.logger.info(f"设置默认服务: {service_name}")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        # 关闭所有客户端
        for client in self._clients.values():
            client.session.close()
        self._clients.clear()