#!/bin/bash
# GitHub Repository Setup Script for Satellite Sensor Tasking RL

set -e  # Exit on error

echo "🚀 Satellite Sensor Tasking RL - GitHub Repository Setup"
echo "========================================================="
echo ""

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get current directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo -e "${BLUE}📁 Current directory: $PROJECT_DIR${NC}"
echo ""

# Step 1: Check if git is installed
echo -e "${BLUE}Step 1: Checking if git is installed...${NC}"
if ! command -v git &> /dev/null; then
    echo -e "${YELLOW}⚠️  Git is not installed. Please install git first.${NC}"
    echo "   Visit: https://git-scm.com/downloads"
    exit 1
fi
echo -e "${GREEN}✓ Git is installed${NC}"
echo ""

# Step 2: Check if already a git repository
if [ -d ".git" ]; then
    echo -e "${YELLOW}⚠️  This directory is already a git repository.${NC}"
    read -p "Do you want to reinitialize? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf .git
        echo -e "${GREEN}✓ Removed existing .git directory${NC}"
    else
        echo "Keeping existing repository. Exiting."
        exit 0
    fi
fi

# Step 3: Initialize git repository
echo -e "${BLUE}Step 2: Initializing git repository...${NC}"
git init
echo -e "${GREEN}✓ Git repository initialized${NC}"
echo ""

# Step 4: Set default branch to main
echo -e "${BLUE}Step 3: Setting default branch to 'main'...${NC}"
git branch -M main
echo -e "${GREEN}✓ Default branch set to 'main'${NC}"
echo ""

# Step 5: Add all files
echo -e "${BLUE}Step 4: Adding files to git...${NC}"
git add .
echo -e "${GREEN}✓ Files added to staging${NC}"
echo ""

# Step 6: Create initial commit
echo -e "${BLUE}Step 5: Creating initial commit...${NC}"
git commit -m "Initial commit: Satellite Sensor Tasking RL Project

- Gymnasium-compatible grid environment for satellite sensor tasking
- Q-learning agent with epsilon-greedy exploration
- Transfer learning utilities (Q-table resizing)
- 2D visualizations (learning curves, heatmaps, policy arrows)
- 3D satellite gimbal visualization
- Comprehensive unit tests (42+ test cases)
- Educational Jupyter notebook with step-by-step examples
- Full documentation and setup instructions

Features:
- All files <500 lines (clean, modular code)
- Google-style docstrings and type hints
- Virtual environment setup
- <10 minute quick start for students
- MIT License (open source)"

echo -e "${GREEN}✓ Initial commit created${NC}"
echo ""

# Step 7: Instructions for GitHub
echo -e "${BLUE}=========================================================${NC}"
echo -e "${GREEN}✓ Local repository setup complete!${NC}"
echo ""
echo -e "${YELLOW}Next steps to push to GitHub:${NC}"
echo ""
echo "1. Create a new repository on GitHub:"
echo -e "   ${BLUE}https://github.com/new${NC}"
echo ""
echo "2. Repository settings (recommended):"
echo "   - Name: satellite-sensor-tasking-rl"
echo "   - Description: Educational RL project for satellite sensor tasking with Q-learning, transfer learning, and 3D visualization"
echo "   - Visibility: Public (for portfolio/sharing)"
echo "   - DO NOT initialize with README, .gitignore, or license (already have them)"
echo ""
echo "3. After creating the repository, run these commands:"
echo ""
echo -e "   ${BLUE}git remote add origin https://github.com/YOUR_USERNAME/satellite-sensor-tasking-rl.git${NC}"
echo -e "   ${BLUE}git push -u origin main${NC}"
echo ""
echo "4. (Optional) Add repository topics on GitHub for discoverability:"
echo "   - reinforcement-learning"
echo "   - q-learning"
echo "   - satellite"
echo "   - machine-learning"
echo "   - python"
echo "   - jupyter-notebook"
echo "   - gymnasium"
echo "   - transfer-learning"
echo ""
echo -e "${GREEN}Happy coding! 🎉${NC}"
echo ""

# Step 8: Show git status
echo -e "${BLUE}Current git status:${NC}"
git status
echo ""

# Optional: Show file count
echo -e "${BLUE}Repository contents:${NC}"
echo "  Source files: $(find src -name '*.py' | wc -l) Python modules"
echo "  Test files: $(find tests -name 'test_*.py' | wc -l) test suites"
echo "  Notebook: 1 educational Jupyter notebook"
echo "  Documentation: README.md, LICENSE"
echo ""
