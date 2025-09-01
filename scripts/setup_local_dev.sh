#!/bin/bash
# Setup script for local development environment on Mac

set -e

echo "Setting up local development environment for GPU Deep Learning..."

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "This script is designed for macOS. Proceeding anyway..."
fi

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "uv not found. Please install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "uv found: $(uv --version)"

# Initialize project if pyproject.toml doesn't exist
if [ ! -f "pyproject.toml" ]; then
    echo "Initializing uv project..."
    uv init --no-readme
fi

# Add dependencies using uv
echo "Adding dependencies with uv..."
echo "Note: Triton won't install on Apple Silicon, but that's expected"
uv add torch matplotlib numpy jupyter pandas modal

# Try to add triton (will fail on Apple Silicon, but that's ok)
if uv add triton 2>/dev/null; then
    echo "Triton added successfully"
else
    echo "Triton skipped (not available for Apple Silicon - will run remotely)"
fi

# Add development dependencies
echo "Adding development dependencies..."
uv add --dev black flake8 pytest ipython

# Verify installations
echo "Verifying installations..."

uv run python -c "
import torch
import matplotlib
import numpy as np
print('Core packages imported successfully')
print(f'PyTorch version: {torch.__version__}')
print(f'NumPy version: {np.__version__}')

try:
    import triton
    print(f'Triton version: {triton.__version__}')
except ImportError:
    print('Triton not available locally (expected on Apple Silicon)')
"

# Check if git is configured
if ! git config user.name &> /dev/null; then
    echo "Git user name not configured. Run:"
    echo "   git config --global user.name 'Your Name'"
    echo "   git config --global user.email 'your.email@example.com'"
fi

# Create .gitignore if it doesn't exist
if [ ! -f ".gitignore" ]; then
    echo "Creating .gitignore..."
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Jupyter Notebook
.ipynb_checkpoints

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log

# Model files and data
*.pth
*.pt
*.onnx
data/
checkpoints/
EOF
fi

# Create scripts directory and make them executable
chmod +x scripts/*.py
chmod +x scripts/*.sh

echo ""
echo "Setup complete! Next steps:"
echo ""
echo "1. Generate sync tools:"
echo "   uv run python scripts/kaggle_sync.py --git-sync"
echo "   uv run python scripts/lightning_sync.py --git-sync"
echo ""
echo "2. Push to GitHub:"
echo "   git add ."
echo "   git commit -m 'Setup GPU workflow'"
echo "   git push"
echo ""
echo "3. Run on GPU:"
echo "   - Kaggle: Copy generated kaggle_git_sync.py content to new notebook"
echo "   - Lightning: Upload lightning_quick_setup.sh to GPU studio"
echo ""