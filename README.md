# 🌸 小虹 - 情感陪伴机器人

一个基于 MemGPT 架构的情感陪伴机器人，具备长期记忆能力，能够记住你的名字、生日、喜好，陪伴你度过每一天。

## ✨ 特性

- 🧠 **长期记忆** - 基于 MemGPT 架构，跨会话记住用户信息
- 💝 **情感理解** - 识别和理解用户的情感状态
- 🎯 **个性化陪伴** - 记住用户画像，提供温暖的个性化回复
- 🕐 **实时信息** - 支持查询当前时间和天气
- 💾 **本地存储** - SQLite + FTS5 全文搜索，数据安全可控

## 🏗️ 技术架构

```
┌─────────────────────────────────────────────┐
│                 Gradio Web UI               │
├─────────────────────────────────────────────┤
│              EmotionalAgent                 │
│  ┌─────────────┐  ┌───────────────────┐    │
│  │   Memory    │  │  Emotion Analyzer │    │
│  │   Manager   │  │   (混合分析引擎)    │    │
│  └─────────────┘  └───────────────────┘    │
├─────────────────────────────────────────────┤
│              LLM Client (千问/DeepSeek)      │
├─────────────────────────────────────────────┤
│           SQLite + FTS5 (持久化存储)         │
└─────────────────────────────────────────────┘
```

## 🚀 快速开始

### 环境要求

- Python 3.10+
- 千问 API Key 或 DeepSeek API Key

### 本地运行

1. **克隆项目**
```bash
git clone https://github.com/your-username/robot-hong.git
cd robot-hong
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **配置环境变量**
```bash
cp .env.example .env
# 编辑 .env 文件，填入你的 API Key
```

4. **启动服务**
```bash
./run.sh
# 或者
python src/main.py
```

5. **访问界面**

打开浏览器访问 http://localhost:7860

### Docker 部署

```bash
# 构建镜像
docker build -t robot-hong .

# 运行容器
docker run -d \
  --name robot-hong \
  -p 7860:7860 \
  -e DASHSCOPE_API_KEY=your_api_key_here \
  -v $(pwd)/data:/app/data \
  robot-hong
```

## ⚙️ 配置说明

### 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `LLM_PROVIDER` | LLM 提供商 (qwen/deepseek) | qwen |
| `DASHSCOPE_API_KEY` | 千问 API Key | - |
| `DEEPSEEK_API_KEY` | DeepSeek API Key | - |
| `DATABASE_PATH` | 数据库路径 | ./data/emotional_bot.db |
| `GRADIO_SERVER_PORT` | 服务端口 | 7860 |
| `LOG_LEVEL` | 日志级别 | INFO |
| `ENV` | 运行环境 (development/production) | development |

### .env.example

```bash
# LLM 配置（二选一）
LLM_PROVIDER=qwen
DASHSCOPE_API_KEY=your_dashscope_api_key_here
# DEEPSEEK_API_KEY=your_deepseek_api_key_here

# 服务配置
GRADIO_SERVER_PORT=7860
DATABASE_PATH=./data/emotional_bot.db
LOG_LEVEL=INFO
ENV=production
```

## 📁 项目结构

```
robot-hong/
├── src/
│   ├── main.py              # 主入口
│   ├── config.py            # 配置管理
│   ├── agent/               # 核心 Agent 模块
│   │   ├── emotional_agent.py
│   │   ├── memory.py
│   │   └── context.py
│   ├── emotion/             # 情感分析模块
│   │   └── analyzer.py
│   ├── llm/                 # LLM 接口模块
│   │   ├── client.py
│   │   ├── prompts.py
│   │   └── functions.py
│   ├── storage/             # 存储模块
│   │   ├── database.py
│   │   └── repository.py
│   ├── tools/               # 工具模块
│   │   └── realtime.py
│   └── utils/               # 工具函数
│       └── logger.py
├── data/                    # 数据目录
├── docs/                    # 文档
├── tests/                   # 测试
├── Dockerfile               # Docker 配置
├── requirements.txt         # Python 依赖
├── run.sh                   # 启动脚本
└── README.md
```

## 🎯 核心功能

### 用户画像记忆

小虹会自动记住你告诉她的信息：

- 👤 名字、年龄、生日
- 📍 住址/城市
- 💼 职业
- 🎨 兴趣爱好
- 📅 重要生活事件

### 情感分析

采用混合分析策略：
- 规则层：关键词匹配，快速低成本
- LLM层：复杂情况深度分析

### 实时信息

- 🕐 查询当前时间
- 🌤️ 查询天气信息

## 🔒 安全与隐私

- API 密钥通过环境变量管理，不提交到代码仓库
- 所有数据本地存储，不上传云端
- SQLite 数据库可随时备份或删除

## 📝 开发日志

- ✅ 基于 MemGPT 架构的长期记忆
- ✅ 混合情感分析引擎
- ✅ Function Calling 工具调用
- ✅ 用户画像持久化
- ✅ 实时时间/天气查询
- ✅ Gradio Web 界面

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 License

MIT License

