# GitHub Repository Setup Guide

## Quick Start

### Option 1: Using the Setup Script (Recommended)

```bash
cd /mnt/c/Users/benja/OneDrive\ -\ Johns\ Hopkins/Documents/RLproject/satellite_rl
./setup_git_repo.sh
```

The script will:
- Initialize git repository
- Add all files
- Create initial commit
- Provide instructions for GitHub setup

### Option 2: Manual Setup

```bash
cd /mnt/c/Users/benja/OneDrive\ -\ Johns\ Hopkins/Documents/RLproject/satellite_rl

# Initialize git
git init

# Set default branch
git branch -M main

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Satellite Sensor Tasking RL Project"

# Now follow "Push to GitHub" steps below
```

## Push to GitHub

### 1. Create Repository on GitHub

Go to: https://github.com/new

**Settings:**
- **Name**: `satellite-sensor-tasking-rl`
- **Description**: `Educational RL project for satellite sensor tasking with Q-learning, transfer learning, and 3D visualization`
- **Visibility**: Public (recommended for portfolio)
- **DO NOT** check:
  - ❌ Add a README file (already have one)
  - ❌ Add .gitignore (already have one)
  - ❌ Choose a license (already have MIT license)

### 2. Connect Local Repository to GitHub

Replace `YOUR_USERNAME` with your GitHub username:

```bash
git remote add origin https://github.com/YOUR_USERNAME/satellite-sensor-tasking-rl.git
git push -u origin main
```

### 3. Add Repository Topics (Optional but Recommended)

On GitHub repository page, click "⚙️ Settings" → "Topics" and add:
- `reinforcement-learning`
- `q-learning`
- `satellite`
- `machine-learning`
- `python`
- `jupyter-notebook`
- `gymnasium`
- `transfer-learning`
- `education`
- `3d-visualization`

This helps others discover your project!

## Verify Repository Contents

After pushing, your GitHub repository should show:

```
satellite-sensor-tasking-rl/
├── .gitignore              ← Excludes venv, cache, etc.
├── LICENSE                 ← MIT License (most permissive)
├── README.md               ← Project documentation
├── requirements.txt        ← Python dependencies
├── .env.example           ← Configuration template
├── notebook/
│   └── satellite_sensor_tasking.ipynb
├── src/
│   ├── environment/
│   ├── agents/
│   ├── visualization/
│   └── utils/
└── tests/
    ├── test_environment.py
    ├── test_agents.py
    └── test_transfer_learning.py
```

## Next Steps

### 1. Update README with Your GitHub URL

In `README.md`, add a clone command at the top:

```markdown
## Installation

git clone https://github.com/YOUR_USERNAME/satellite-sensor-tasking-rl.git
cd satellite-sensor-tasking-rl
```

### 2. Add a Badge (Optional)

Add this to the top of your README:

```markdown
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
```

### 3. Enable GitHub Pages for Documentation (Optional)

If you want to host documentation:
1. Go to repository Settings → Pages
2. Source: Deploy from branch `main`
3. Folder: `/docs` (create a docs folder with sphinx or mkdocs)

### 4. Add GitHub Actions for Testing (Advanced)

Create `.github/workflows/tests.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run tests
        run: pytest tests/ -v
```

## Sharing Your Project

### For Your Portfolio
- Link to GitHub repository in resume/CV
- Showcase Jupyter notebook (GitHub renders it automatically)
- Highlight in LinkedIn projects section

### For Students/Educators
Share clone instructions:

```bash
git clone https://github.com/YOUR_USERNAME/satellite-sensor-tasking-rl.git
cd satellite-sensor-tasking-rl
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m ipykernel install --user --name=satellite_rl
jupyter lab
```

### For Collaborators
- Enable Issues for bug reports/feature requests
- Add CONTRIBUTING.md with guidelines
- Consider adding CODE_OF_CONDUCT.md

## Troubleshooting

### "Permission denied (publickey)"
Use HTTPS instead of SSH, or set up SSH keys:
```bash
git remote set-url origin https://github.com/YOUR_USERNAME/satellite-sensor-tasking-rl.git
```

### "Repository not found"
Verify the repository exists and URL is correct:
```bash
git remote -v
```

### "Failed to push some refs"
Pull first if repository has changes:
```bash
git pull origin main --allow-unrelated-histories
git push -u origin main
```

## Additional Resources

- [GitHub Quickstart](https://docs.github.com/en/get-started/quickstart)
- [Git Basics](https://git-scm.com/book/en/v2/Getting-Started-Git-Basics)
- [GitHub Pages Documentation](https://docs.github.com/en/pages)
- [GitHub Actions](https://docs.github.com/en/actions)

---

**Ready to share your work with the world!** 🚀
