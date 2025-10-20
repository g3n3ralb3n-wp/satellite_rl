#!/bin/bash
# Setup script for Satellite RL Project (Linux/macOS)
# This script automates the virtual environment creation and dependency installation

set -e  # Exit on error

echo "========================================="
echo "Satellite RL Project Setup"
echo "========================================="
echo ""

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
echo -e "${BLUE}Checking Python version...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 is not installed${NC}"
    echo "Please install Python 3.9 or higher"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo -e "${GREEN}Found Python ${PYTHON_VERSION}${NC}"
echo ""

# Create virtual environment
echo -e "${BLUE}Creating virtual environment (venv_linux)...${NC}"
if [ -d "venv_linux" ]; then
    echo -e "${RED}Warning: venv_linux already exists${NC}"
    read -p "Do you want to remove it and create a fresh one? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf venv_linux
        python3 -m venv venv_linux
        echo -e "${GREEN}Created fresh virtual environment${NC}"
    else
        echo -e "${BLUE}Using existing virtual environment${NC}"
    fi
else
    python3 -m venv venv_linux
    echo -e "${GREEN}Virtual environment created${NC}"
fi
echo ""

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source venv_linux/bin/activate
echo -e "${GREEN}Virtual environment activated${NC}"
echo ""

# Upgrade pip
echo -e "${BLUE}Upgrading pip...${NC}"
pip install --upgrade pip --quiet
echo -e "${GREEN}pip upgraded${NC}"
echo ""

# Install dependencies
echo -e "${BLUE}Installing dependencies from requirements.txt...${NC}"
echo "This may take a few minutes..."
pip install -r requirements.txt --quiet
echo -e "${GREEN}Dependencies installed${NC}"
echo ""

# Install Jupyter kernel
echo -e "${BLUE}Installing Jupyter kernel...${NC}"
python -m ipykernel install --user --name=satellite_rl
echo -e "${GREEN}Jupyter kernel 'satellite_rl' installed${NC}"
echo ""

# Verify installation
echo -e "${BLUE}Verifying installation...${NC}"
python -c "import numpy, matplotlib, gymnasium, jupyter" 2>/dev/null && \
    echo -e "${GREEN}Core packages verified successfully${NC}" || \
    echo -e "${RED}Warning: Some packages may not be installed correctly${NC}"
echo ""

# Display next steps
echo "========================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Activate the virtual environment:"
echo -e "   ${BLUE}source venv_linux/bin/activate${NC}"
echo ""
echo "2. Launch Jupyter Lab:"
echo -e "   ${BLUE}jupyter lab${NC}"
echo ""
echo "3. Open the notebook:"
echo -e "   ${BLUE}notebook/satellite_sensor_tasking.ipynb${NC}"
echo ""
echo "4. Select kernel: satellite_rl"
echo ""
echo "5. Run all cells!"
echo ""
echo "========================================="
echo "To run tests:"
echo -e "   ${BLUE}pytest tests/ -v${NC}"
echo "========================================="
