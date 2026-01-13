import json
import logging
from typing import Dict, List, Optional, Any, Iterator
import requests
from .config import ModelConfig
import anthropic # 确保已安装: pip install anthropic

class ModelClient:
    """通用大模型调用客户端"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = logging.getLogger(f'AgentsChat.ModelClient.{config.service_name}')
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.api_key}",
        })
        self.claude_client = None
        if "CLAUDE" in self.config.service_name.upper():
            self.claude_client = anthropic.Anthropic(
                api_key=config.api_key,
                base_url=config.base_url if config.base_url else None
            )

    def chat_completion(self, 
                       messages: List[Dict[str, str]], 
                       temperature: float = 0.7,
                       max_tokens: Optional[int] = None,
                       stream: bool = False,
                       extra_headers: Optional[Dict[str, str]] = None,
                       service_name: Optional[str] = None,
                       **kwargs) -> Dict[str, Any]:

        current_service = service_name if service_name else self.config.service_name
        if "CLAUDE" in current_service.upper() and self.claude_client:
            return self._chat_completion_claude_sdk(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                **kwargs
            )
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
            **kwargs
        }
        
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        
        self.logger.debug(f"发送请求到 {url}")
        
        # (原有 requests 逻辑...)
        try:
            if stream:
                return self._stream_request(url, payload, headers=extra_headers)
            else:
                response = self.session.post(url, json=payload, headers=extra_headers, timeout=1200)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            self.logger.error(f"请求失败: {e}")
            raise

    def _chat_completion_claude_sdk(self, messages, temperature, max_tokens, stream, **kwargs) -> Dict[str, Any]:
        """使用 Claude SDK 处理请求，并转换回 OpenAI 格式"""
        system_prompt = None
        claude_messages = []
        for msg in messages:
            if msg['role'] == 'system':
                system_prompt = msg['content']
            else:
                if msg['role'] == 'tool':
                    tool_response = {
                        "type": "tool_result",
                        "tool_use_id": msg['tool_call_id'],
                        "content": msg['content']
                    }
                    msg = {
                        "role": "user",
                        "content": [tool_response] # 加 []
                    }
                elif 'tool_calls' in msg and msg['tool_calls']:
                    tc = msg['tool_calls'][0] 
                    
                    tool_use_block = {
                        "type": "tool_use",
                        "id": tc['id'],
                        "name": tc['function']['name'],
                        # 修正：必须 json.loads 将字符串转为字典
                        "input": json.loads(tc['function']['arguments']) 
                    }
                    content_list = []
                    if msg.get('content'):
                        content_list.append({"type": "text", "text": msg['content']})
                    content_list.append(tool_use_block)
                    
                    msg = {
                        "role": "assistant",
                        "content": content_list
                    }
                    
                claude_messages.append(msg)
                
        params = {
            "model": self.config.model_name,
            "messages": claude_messages,
            "max_tokens": max_tokens if max_tokens else 1024,
            "temperature": temperature,
            "betas": ["context-1m-2025-08-07"],
            **kwargs
        }
        if system_prompt:
            params["system"] = system_prompt

        try:
            response = self.claude_client.beta.messages.create(**params)

            content_text = ""
            if response.content:
                content_text = "".join([block.text for block in response.content if block.type == 'text'])
                
            return {
                "id": response.id,
                "object": "chat.completion",
                "created": 0,
                "model": response.model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content_text
                    },
                    "finish_reason": response.stop_reason
                }],
                "usage": {
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                }
            }
            
        except anthropic.APIError as e:
            self.logger.error(f"Claude SDK Error: {e}")
            raise
    def _stream_request(self, url: str, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Iterator[Dict[str, Any]]:
        with self.session.post(url, json=payload, headers=headers, stream=True, timeout=1200) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]
                        if data_str.strip() == '[DONE]': break
                        try:
                            yield json.loads(data_str)
                        except json.JSONDecodeError: continue

    def simple_chat(self, message: str, **kwargs) -> str:
        messages = [{"role": "user", "content": message}]
        response = self.chat_completion(messages, **kwargs)
        if 'choices' in response and len(response['choices']) > 0:
            msg = response['choices'][0]['message']
            return msg.get('content', '') or ''
        else:
            raise ValueError("无效的 API 响应格式")
            
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "service_name": self.config.service_name.lower(),
            "model_name": self.config.model_name,
            "base_url": self.config.base_url
        }
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.close()