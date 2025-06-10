#!/bin/bash
# RunPod Setup Script for GR00T Inference Service
# This script installs everything in the persistent network volume

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Starting GR00T RunPod Setup${NC}"

# Check if persistent volume is mounted
PERSISTENT_DIR="/persistent"
if [ ! -d "$PERSISTENT_DIR" ]; then
    echo -e "${RED}❌ Persistent volume not found at $PERSISTENT_DIR${NC}"
    echo -e "${YELLOW}Please make sure you have mounted a network volume named 'persistent' to /persistent${NC}"
    exit 1
fi

echo -e "${BLUE}📁 Using persistent directory: $PERSISTENT_DIR${NC}"

# Install Miniconda if not already installed
MINICONDA_DIR="$PERSISTENT_DIR/miniconda3"
if [ ! -d "$MINICONDA_DIR" ]; then
    echo -e "${BLUE}🐍 Installing Miniconda...${NC}"
    cd "$PERSISTENT_DIR"
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p "$MINICONDA_DIR"
    rm miniconda.sh
    echo -e "${GREEN}✅ Miniconda installed${NC}"
else
    echo -e "${GREEN}✅ Miniconda already exists${NC}"
fi

# Initialize conda and add to PATH
export PATH="$MINICONDA_DIR/bin:$PATH"
source "$MINICONDA_DIR/etc/profile.d/conda.sh"

# Set up environment variables
export PYTHONPATH="$PERSISTENT_DIR/Isaac-GR00T:$PYTHONPATH"
export HF_HOME="$PERSISTENT_DIR/huggingface_cache"
export PIP_CACHE_DIR="$PERSISTENT_DIR/pip_cache"
export CONDA_PKGS_DIRS="$PERSISTENT_DIR/conda_pkgs"

# Create necessary directories
echo -e "${BLUE}📂 Creating directories...${NC}"
mkdir -p "$PERSISTENT_DIR/huggingface_cache"
mkdir -p "$PERSISTENT_DIR/pip_cache"
mkdir -p "$PERSISTENT_DIR/conda_pkgs"
mkdir -p "$PERSISTENT_DIR/models"

# Install system dependencies if not already installed
echo -e "${BLUE}📦 Installing system dependencies...${NC}"
apt-get update -qq
apt-get install -y -qq git git-lfs ffmpeg libsm6 libxext6 libgl1-mesa-glx

# Check if Isaac-GR00T is already installed
if [ ! -d "$PERSISTENT_DIR/Isaac-GR00T" ]; then
    echo -e "${BLUE}📥 Cloning Isaac-GR00T repository...${NC}"
    cd "$PERSISTENT_DIR"
    git clone https://github.com/iowathe3rd/Isaac-GR00T
    cd Isaac-GR00T
else
    echo -e "${GREEN}✅ Isaac-GR00T already exists, updating...${NC}"
    cd "$PERSISTENT_DIR/Isaac-GR00T"
    git pull
fi

# Install Python dependencies
echo -e "${BLUE}🐍 Installing Python dependencies...${NC}"
cd "$PERSISTENT_DIR/Isaac-GR00T"

# Check if virtual environment exists
VENV_DIR="$PERSISTENT_DIR/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${BLUE}🔧 Creating conda environment...${NC}"
    # Use conda to create environment with Python 3.10
    conda create -p "$VENV_DIR" python=3.10 -y
    echo -e "${GREEN}✅ Conda environment created${NC}"
fi

# Activate conda environment
source activate "$VENV_DIR"

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install package in editable mode
echo -e "${BLUE}📋 Installing GR00T package...${NC}"
pip install -e .

# Install flash attention
echo -e "${BLUE}⚡ Installing Flash Attention...${NC}"
pip install flash-attn==2.7.1.post4 --no-build-isolation

# Install additional dependencies
echo -e "${BLUE}📚 Installing additional dependencies...${NC}"
pip install accelerate>=0.26.0

# Create activation script
echo -e "${BLUE}📝 Creating activation script...${NC}"
cat > "$PERSISTENT_DIR/activate.sh" << 'EOF'
#!/bin/bash
# Activation script for GR00T environment

export PATH="/persistent/miniconda3/bin:$PATH"
source /persistent/miniconda3/etc/profile.d/conda.sh
export PYTHONPATH="/persistent/Isaac-GR00T:$PYTHONPATH"
export HF_HOME="/persistent/huggingface_cache"
export PIP_CACHE_DIR="/persistent/pip_cache"
source activate /persistent/venv

echo "🤖 GR00T environment activated!"
echo "📁 Project dir: /persistent/Isaac-GR00T"
echo "🏠 HF cache: $HF_HOME"
echo "🐍 Python: $(which python)"

cd /persistent/Isaac-GR00T
EOF

chmod +x "$PERSISTENT_DIR/activate.sh"

# Create start script for inference service
echo -e "${BLUE}🎯 Creating inference service start script...${NC}"
cat > "$PERSISTENT_DIR/start_inference.sh" << 'EOF'
#!/bin/bash
# Start GR00T Inference Service

# Activate environment
source /persistent/activate.sh

# Default environment variables (can be overridden)
export MODEL_PATH=${MODEL_PATH:-"nvidia/GR00T-N1-2B"}
export EMBODIMENT_TAG=${EMBODIMENT_TAG:-""}
export NUM_ARMS=${NUM_ARMS:-"1"}
export NUM_CAMS=${NUM_CAMS:-"2"}
export DENOISING_STEPS=${DENOISING_STEPS:-"4"}
export HOST=${HOST:-"0.0.0.0"}
export PORT=${PORT:-"5555"}

echo "🚀 Starting GR00T Inference Service"
echo "📦 Model: $MODEL_PATH"
echo "🤖 Embodiment: $EMBODIMENT_TAG"
echo "🔧 Config: $NUM_ARMS arms, $NUM_CAMS cameras"
echo "🌐 Server: $HOST:$PORT"

cd /persistent/Isaac-GR00T

# Start the inference service
python scripts/inference_service.py \
    --model_path "$MODEL_PATH" \
    --embodiment_tag "$EMBODIMENT_TAG" \
    --num_arms "$NUM_ARMS" \
    --num_cams "$NUM_CAMS" \
    --denoising_steps "$DENOISING_STEPS" \
    --host "$HOST" \
    --port "$PORT"
EOF

chmod +x "$PERSISTENT_DIR/start_inference.sh"

# Create quick setup script for new pods
echo -e "${BLUE}⚡ Creating quick setup script...${NC}"
cat > "$PERSISTENT_DIR/quick_setup.sh" << 'EOF'
#!/bin/bash
# Quick setup for new RunPod instances

echo "⚡ Quick GR00T Setup for new pod"

# Set environment variables
export PATH="/persistent/miniconda3/bin:$PATH"
source /persistent/miniconda3/etc/profile.d/conda.sh
export PYTHONPATH="/persistent/Isaac-GR00T:$PYTHONPATH"
export HF_HOME="/persistent/huggingface_cache"
export PIP_CACHE_DIR="/persistent/pip_cache"

# Activate conda environment
source activate /persistent/venv

# Update system dependencies (usually pre-installed in RunPod)
apt-get update -qq && apt-get install -y -qq git git-lfs ffmpeg libsm6 libxext6 libgl1-mesa-glx

echo "✅ Quick setup complete!"
echo "🚀 Run: /persistent/start_inference.sh"
EOF

chmod +x "$PERSISTENT_DIR/quick_setup.sh"

# Create Docker start command
echo -e "${BLUE}🐳 Creating Docker start command...${NC}"
cat > "$PERSISTENT_DIR/docker_start_command.txt" << 'EOF'
# Docker start command for RunPod template:
bash -c "cd /persistent && ./quick_setup.sh && ./start_inference.sh"

# Or with environment variables:
bash -c "export MODEL_PATH=your-model-path && cd /persistent && ./quick_setup.sh && ./start_inference.sh"
EOF

# Test installation
echo -e "${BLUE}🧪 Testing installation...${NC}"
export PATH="$MINICONDA_DIR/bin:$PATH"
source activate "$VENV_DIR"
python -c "
try:
    from gr00t.model.policy import Gr00tPolicy
    from gr00t.experiment.data_config import ConfigGenerator
    print('✅ GR00T imports successful')
except Exception as e:
    print(f'❌ Import failed: {e}')
    exit(1)
"

echo -e "${GREEN}🎉 Setup complete!${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}📋 Usage Instructions:${NC}"
echo -e "${BLUE}1.${NC} For new pods, run: ${GREEN}/persistent/quick_setup.sh${NC}"
echo -e "${BLUE}2.${NC} Start inference: ${GREEN}/persistent/start_inference.sh${NC}"
echo -e "${BLUE}3.${NC} Or activate env: ${GREEN}source /persistent/activate.sh${NC}"
echo ""
echo -e "${GREEN}🔧 Environment Variables (set before starting):${NC}"
echo -e "${BLUE}MODEL_PATH${NC}=your-model-repo (default: nvidia/GR00T-N1-2B)"
echo -e "${BLUE}EMBODIMENT_TAG${NC}=your-tag (auto-detect if empty)"
echo -e "${BLUE}NUM_ARMS${NC}=1"
echo -e "${BLUE}NUM_CAMS${NC}=2"
echo -e "${BLUE}PORT${NC}=5555"
echo ""
echo -e "${GREEN}📦 Files saved in:${NC} $PERSISTENT_DIR"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
