# Simple Kaggle Workflow

## How to run code on Kaggle GPU

### One-time setup:
1. Run: `uv run python scripts/kaggle_sync.py --git-sync`
2. This creates `kaggle_git_sync.py` with this content:

```python
# Kaggle GPU Sync Notebook
# This notebook automatically pulls latest code from GitHub

# Install dependencies
!pip install triton -q

# Clone/pull latest code
import os
if os.path.exists('/kaggle/working/gpu-dl-playground'):
    print('Repository exists, pulling latest changes...')
    os.chdir('/kaggle/working/gpu-dl-playground')
    !git pull
else:
    print('Cloning repository...')
    os.chdir('/kaggle/working')
    !git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
    os.chdir('/kaggle/working/gpu-dl-playground')

# Check GPU
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'   GPU: {torch.cuda.get_device_name(0)}')

# Run script (change this line as needed)
print("Running code...")
exec(open('triton_kernels.py').read())
```

### Daily workflow:
1. **Code locally** and push to GitHub: `git push`
2. **Go to Kaggle** (kaggle.com/code) 
3. **New Notebook** → **Enable GPU** (T4 x2 in settings)
4. **Paste the content** from `kaggle_git_sync.py` into the notebook
5. **Run all cells** - the latest GitHub code runs automatically!

### That's it!
- No manual file uploads
- Always runs latest code from GitHub  
- 30-second cycle from local → GPU
- Keep the Kaggle notebook tab open for quick re-runs
