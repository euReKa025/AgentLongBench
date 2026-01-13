# AgentsChat Models 工具包

通用大模型调用工具包，提供统一的 API 接口支持多种大模型服务商。

## 📋 目录

- [功能特性](#功能特性)
- [快速开始](#快速开始)
- [环境配置](#环境配置)
- [API 接口](#api-接口)
- [使用示例](#使用示例)
- [支持的服务商](#支持的服务商)
- [模块说明](#模块说明)

## 🚀 功能特性

- **统一接口**：基于 OpenAI 兼容 API，支持多种大模型服务商
- **自动配置**：从环境变量自动加载服务配置
- **流式支持**：支持流式和非流式聊天模式
- **多服务商**：支持 DEEPSEEK、OpenAI、Claude、Qwen 等主流服务
- **简单易用**：提供简化的聊天接口和完整的 API 接口
- **资源管理**：支持上下文管理器，自动清理资源
- **错误处理**：统一的错误处理和重试机制

## 🏃 快速开始

### 基本使用

```python
from src.models import ModelManager

# 创建模型管理器（自动使用第一个可用服务）
manager = ModelManager()

# 简单聊天
reply = manager.chat("你好，请介绍一下你自己")
print(reply)

# 指定服务商
reply = manager.chat("Hello", service_name="DEEPSEEK")
print(reply)
```

### 流式聊天

```python
# 流式聊天
messages = [{"role": "user", "content": "请介绍人工智能"}]

for chunk in manager.stream_chat(messages):
    if chunk and 'choices' in chunk:
        delta = chunk['choices'][0].get('delta', {})
        content = delta.get('content', '')
        if content:
            print(content, end='', flush=True)
```

### 上下文管理器

```python
# 使用上下文管理器自动清理资源
with ModelManager(default_service="DEEPSEEK") as manager:
    reply = manager.chat("你好")
    print(reply)
```

## ⚙️ 环境配置

在项目根目录创建 `.env` 文件，配置所需的服务商信息：

```bash
# DEEPSEEK 配置
DEEPSEEK_API_KEY=sk-your-deepseek-api-key
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL_NAME=deepseek-chat

# OpenAI 配置
OPENAI_API_KEY=sk-your-openai-api-key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL_NAME=gpt-3.5-turbo

# 其他服务商...
```

### 配置格式规则

- `{SERVICE}_API_KEY`：API 密钥
- `{SERVICE}_BASE_URL`：API 基础 URL
- `{SERVICE}_MODEL_NAME`：模型名称

其中 `{SERVICE}` 为服务商名称（如 DEEPSEEK、OPENAI 等）。

## 📚 API 接口

### ModelManager 类

主要的统一管理器类，推荐使用。

#### 初始化

```python
manager = ModelManager(default_service="DEEPSEEK")
```

#### 主要方法

| 方法 | 说明 | 参数 | 返回值 |
|------|------|------|--------|
| `chat(message, service_name=None, **kwargs)` | 简单聊天接口 | message: 用户消息<br>service_name: 服务名称 | str: 模型回复 |
| `chat_completion(messages, service_name=None, **kwargs)` | 完整聊天补全接口 | messages: 消息列表<br>service_name: 服务名称 | Dict: API 响应 |
| `stream_chat(messages, service_name=None, **kwargs)` | 流式聊天接口 | messages: 消息列表<br>service_name: 服务名称 | Iterator: 流式响应 |
| `list_services()` | 列出可用服务 | 无 | List[str]: 服务列表 |
| `get_service_info(service_name=None)` | 获取服务信息 | service_name: 服务名称 | Dict: 服务信息 |
| `set_default_service(service_name)` | 设置默认服务 | service_name: 服务名称 | 无 |

### ModelClient 类

底层客户端类，提供更细粒度的控制。

#### 主要方法

| 方法 | 说明 | 参数 | 返回值 |
|------|------|------|--------|
| `chat_completion(messages, temperature=0.7, max_tokens=None, stream=False, **kwargs)` | 聊天补全 | messages: 消息列表<br>temperature: 温度参数<br>max_tokens: 最大 token 数<br>stream: 是否流式 | Dict/Iterator |
| `simple_chat(message, **kwargs)` | 简单聊天 | message: 用户消息 | str: 模型回复 |
| `get_model_info()` | 获取模型信息 | 无 | Dict: 模型信息 |

### ModelConfig 类

配置数据类，包含服务配置信息。

#### 属性

- `api_key`: API 密钥
- `base_url`: API 基础 URL
- `model_name`: 模型名称
- `service_name`: 服务名称

### ConfigManager 类

配置管理器，负责从环境变量加载配置。

#### 主要方法

| 方法 | 说明 | 参数 | 返回值 |
|------|------|------|--------|
| `get_config(service_name)` | 获取服务配置 | service_name: 服务名称 | ModelConfig/None |
| `list_services()` | 列出可用服务 | 无 | List[str] |
| `has_service(service_name)` | 检查服务可用性 | service_name: 服务名称 | bool |

## 💡 使用示例

### 1. 基础聊天

```python
from src.models import ModelManager

# 初始化管理器
manager = ModelManager()

# 简单对话
user_input = "请解释什么是机器学习"
response = manager.chat(user_input)
print(f"用户: {user_input}")
print(f"AI: {response}")
```

### 2. 多轮对话

```python
# 多轮对话
messages = [
    {"role": "user", "content": "你好，我想学习 Python"},
    {"role": "assistant", "content": "你好！我很乐意帮助你学习 Python。你想从哪里开始？"},
    {"role": "user", "content": "请推荐一些入门资源"}
]

response = manager.chat_completion(messages)
print(response['choices'][0]['message']['content'])
```

### 3. 流式输出

```python
# 实时流式输出
messages = [{"role": "user", "content": "请写一首关于春天的诗"}]

print("AI 正在创作...")
for chunk in manager.stream_chat(messages):
    if chunk and 'choices' in chunk:
        delta = chunk['choices'][0].get('delta', {})
        content = delta.get('content', '')
        if content:
            print(content, end='', flush=True)
print("\n创作完成！")
```

### 4. 服务管理

```python
# 查看可用服务
services = manager.list_services()
print(f"可用服务: {services}")

# 切换服务
manager.set_default_service("OPENAI")
response = manager.chat("Hello, how are you?")

# 临时使用其他服务
response = manager.chat("你好", service_name="DEEPSEEK")
```

### 5. 直接使用客户端

```python
from src.models import ModelClient, ConfigManager

# 获取配置
config_manager = ConfigManager()
config = config_manager.get_config("DEEPSEEK")

# 创建客户端
with ModelClient(config) as client:
    response = client.simple_chat("你好")
    print(response)
    
    # 获取模型信息
    info = client.get_model_info()
    print(f"服务: {info['service_name']}")
    print(f"模型: {info['model_name']}")
```

## 🌐 支持的服务商

目前支持所有兼容 OpenAI API 格式的服务商，包括但不限于：

- **DEEPSEEK**：DeepSeek 大模型服务
- **OpenAI**：GPT-3.5、GPT-4 等模型
- **Claude**：Anthropic Claude 系列
- **Qwen**：阿里云通义千问
- **其他**：任何兼容 OpenAI API 的服务

### 添加新服务商

只需在 `.env` 文件中添加对应的配置即可：

```bash
# 新服务商配置
NEW_SERVICE_API_KEY=your-api-key
NEW_SERVICE_BASE_URL=https://api.newservice.com/v1
NEW_SERVICE_MODEL_NAME=model-name
```

## 📁 模块说明

### 文件结构

```
src/models/
├── __init__.py          # 包初始化，导出主要类
├── manager.py           # ModelManager - 统一管理器（推荐使用）
├── client.py            # ModelClient - 底层客户端
├── config.py            # ConfigManager, ModelConfig - 配置管理
└── README.md            # 本文档
```

### 模块职责

- **manager.py**：提供高级统一接口，自动管理配置和客户端
- **client.py**：实现底层 HTTP 客户端，处理 API 调用
- **config.py**：管理环境变量配置，支持多服务商
- **__init__.py**：包入口，导出主要类供外部使用

### 推荐使用方式

1. **日常使用**：直接使用 `ModelManager`，它提供了最简单的接口
2. **高级控制**：需要更细粒度控制时使用 `ModelClient`
3. **配置管理**：需要动态管理配置时使用 `ConfigManager`

## 🔧 开发和测试

### 运行测试

```bash
# 运行所有测试
python tests/test_models.py

# 测试包含真实 API 调用，需要配置 .env 文件
```

### 测试覆盖

- 配置管理测试
- 客户端功能测试
- 管理器接口测试
- 真实 API 集成测试
- 流式响应测试

---

**注意**：使用前请确保已正确配置环境变量，并且拥有相应服务商的有效 API 密钥。