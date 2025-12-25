#!/bin/bash
# 情感机器人启动脚本

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}      小虹 - 情感陪伴机器人             ${NC}"
echo -e "${GREEN}========================================${NC}"

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 检查.env文件
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}警告: .env 文件不存在${NC}"
    echo -e "正在从 .env.example 创建..."
    cp .env.example .env
    echo -e "${RED}请编辑 .env 文件配置你的 API Key${NC}"
    exit 1
fi

# 检查API Key是否配置
if grep -q "your_.*_api_key_here" .env; then
    echo -e "${RED}错误: 请在 .env 文件中配置真实的 API Key${NC}"
    exit 1
fi

# 激活conda环境（如果存在）
if command -v conda &> /dev/null; then
    echo -e "${YELLOW}检测到conda，尝试激活 ai-study-env 环境...${NC}"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate ai-study-env 2>/dev/null || echo -e "${YELLOW}ai-study-env 环境不存在，使用当前环境${NC}"
fi

# 检查Python
if ! command -v python &> /dev/null; then
    echo -e "${RED}错误: 未找到Python${NC}"
    exit 1
fi

echo -e "${GREEN}Python版本:${NC} $(python --version)"

# 检查依赖
echo -e "${YELLOW}检查依赖...${NC}"
python -c "import gradio, openai, dotenv" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}安装依赖...${NC}"
    pip install -r requirements.txt
fi

# 启动应用
echo -e "${GREEN}启动情感机器人...${NC}"
echo -e "${YELLOW}访问地址将显示在下方${NC}"
echo ""

python src/main.py

