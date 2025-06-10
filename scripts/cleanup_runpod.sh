#!/bin/bash
# Cleanup script to remove all RunPod setup results

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${YELLOW}🧹 GR00T RunPod Cleanup Script${NC}"
echo -e "${RED}⚠️  This will delete ALL GR00T setup files and environments!${NC}"

# Ask for confirmation
read -p "Are you sure you want to continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${BLUE}Cleanup cancelled.${NC}"
    exit 0
fi

PERSISTENT_DIR="/persistent"

if [ -d "$PERSISTENT_DIR" ]; then
    echo -e "${BLUE}📁 Cleaning persistent directory: $PERSISTENT_DIR${NC}"
    
    # Remove conda installation
    if [ -d "$PERSISTENT_DIR/miniconda3" ]; then
        echo -e "${BLUE}🐍 Removing Miniconda...${NC}"
        rm -rf "$PERSISTENT_DIR/miniconda3"
        echo -e "${GREEN}✅ Miniconda removed${NC}"
    fi
    
    # Remove conda environment
    if [ -d "$PERSISTENT_DIR/venv" ]; then
        echo -e "${BLUE}🔧 Removing conda environment...${NC}"
        rm -rf "$PERSISTENT_DIR/venv"
        echo -e "${GREEN}✅ Conda environment removed${NC}"
    fi
    
    # Remove Isaac-GR00T repo
    if [ -d "$PERSISTENT_DIR/Isaac-GR00T" ]; then
        echo -e "${BLUE}📥 Removing Isaac-GR00T repository...${NC}"
        rm -rf "$PERSISTENT_DIR/Isaac-GR00T"
        echo -e "${GREEN}✅ Isaac-GR00T repository removed${NC}"
    fi
    
    # Remove cache directories
    echo -e "${BLUE}🗑️ Removing cache directories...${NC}"
    rm -rf "$PERSISTENT_DIR/huggingface_cache"
    rm -rf "$PERSISTENT_DIR/pip_cache"
    rm -rf "$PERSISTENT_DIR/conda_pkgs"
    rm -rf "$PERSISTENT_DIR/models"
    echo -e "${GREEN}✅ Cache directories removed${NC}"
    
    # Remove setup scripts
    echo -e "${BLUE}📝 Removing setup scripts...${NC}"
    rm -f "$PERSISTENT_DIR/activate.sh"
    rm -f "$PERSISTENT_DIR/start_inference.sh"
    rm -f "$PERSISTENT_DIR/quick_setup.sh"
    rm -f "$PERSISTENT_DIR/docker_start_command.txt"
    echo -e "${GREEN}✅ Setup scripts removed${NC}"
    
    echo -e "${GREEN}🎉 Cleanup complete!${NC}"
    echo -e "${BLUE}📁 Persistent directory is now empty and ready for fresh setup.${NC}"
else
    echo -e "${YELLOW}⚠️  Persistent directory not found at $PERSISTENT_DIR${NC}"
    echo -e "${BLUE}Nothing to clean up.${NC}"
fi

# Optional: Clean system-wide conda if installed
if command -v conda &> /dev/null; then
    echo ""
    echo -e "${YELLOW}🔍 Found system conda installation${NC}"
    read -p "Do you want to remove system conda as well? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}🐍 Removing system conda...${NC}"
        
        # Remove conda environments
        conda env list | grep -v "^#" | awk '{print $1}' | grep -v "base" | xargs -I {} conda env remove -n {} -y 2>/dev/null || true
        
        # Remove conda installation directories
        rm -rf ~/miniconda3 2>/dev/null || true
        rm -rf ~/anaconda3 2>/dev/null || true
        rm -rf ~/.conda 2>/dev/null || true
        rm -rf ~/.condarc 2>/dev/null || true
        
        # Clean pip cache
        pip cache purge 2>/dev/null || true
        
        echo -e "${GREEN}✅ System conda removed${NC}"
    fi
fi

echo -e "${GREEN}🏁 All cleanup operations completed!${NC}"
