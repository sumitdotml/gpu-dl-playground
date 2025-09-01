#!/usr/bin/env python3
"""
Seamless Kaggle workflow - push code directly to Kaggle dataset
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def setup_kaggle_dataset():
    """One-time setup of Kaggle dataset for code sync"""

    # Check if kaggle CLI is available
    try:
        subprocess.run(["kaggle", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Kaggle CLI not found. Install with:")
        print("   pip install kaggle")
        print("   Then setup API key: https://www.kaggle.com/settings/account")
        return False

    # Create dataset metadata
    dataset_metadata = {
        "title": "GPU Deep Learning Code Sync",
        "id": "your-username/gpu-dl-code",  # User needs to update this
        "licenses": [{"name": "Apache 2.0"}],
        "resources": [],
    }

    with open("dataset-metadata.json", "w") as f:
        json.dump(dataset_metadata, f, indent=2)

    print("📦 Created dataset-metadata.json")
    print("🔧 Setup steps:")
    print(
        "   1. Edit dataset-metadata.json - change 'your-username' to your Kaggle username"
    )
    print("   2. Run: kaggle datasets create -p .")
    print("   3. Then use: python scripts/kaggle_sync.py your_script.py")

    return True


def push_to_kaggle(script_name, dataset_id=None):
    """Push code to existing Kaggle dataset"""

    if not os.path.exists(script_name):
        print(f"❌ Script {script_name} not found")
        return False

    # Try to get dataset ID from metadata
    if not dataset_id and os.path.exists("dataset-metadata.json"):
        with open("dataset-metadata.json", "r") as f:
            metadata = json.load(f)
            dataset_id = metadata.get("id")

    if not dataset_id:
        print("❌ No dataset ID found. Run setup first or provide --dataset-id")
        return False

    print(f"🚀 Pushing {script_name} to Kaggle dataset {dataset_id}")

    try:
        # Update dataset with new version
        result = subprocess.run(
            [
                "kaggle",
                "datasets",
                "version",
                "-p",
                ".",
                "-m",
                f"Updated {script_name}",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print("✅ Code pushed to Kaggle successfully!")

            # Generate notebook that imports from dataset
            notebook_code = f"""# Install if needed
!pip install triton -q

# Mount dataset
import os
os.chdir('/kaggle/input/gpu-dl-code')

# Run your script
exec(open('{script_name}').read())
"""

            print(f"\n📝 Quick Kaggle notebook code to run {script_name}:")
            print("=" * 50)
            print(notebook_code)
            print("=" * 50)
            print(f"💡 Or direct link: https://www.kaggle.com/datasets/{dataset_id}")

        else:
            print("❌ Failed to push to Kaggle")
            print(result.stderr)
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

    return True


def create_git_sync_notebook():
    """Create a notebook that syncs from GitHub"""

    # Get git remote URL and ensure it's HTTPS for Kaggle compatibility
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"], capture_output=True, text=True
        )
        git_url = result.stdout.strip()
        
        # Convert SSH to HTTPS for Kaggle (no SSH keys available there)
        if git_url.startswith("git@github.com:"):
            git_url = git_url.replace("git@github.com:", "https://github.com/")
        elif git_url.startswith("git@github-"):
            # Handle custom SSH hosts like git@github-sumitdotml:
            git_url = git_url.split(":")[1]  # Extract user/repo part
            git_url = f"https://github.com/{git_url}"
            
        # Remove .git suffix if present
        if git_url.endswith(".git"):
            git_url = git_url[:-4]
            
    except:
        git_url = "https://github.com/YOUR_USERNAME/YOUR_REPO"

    notebook_code = f"""# Kaggle GPU Sync Notebook
# This notebook automatically pulls your latest code from GitHub

import os
import subprocess
import sys

# =============================================================================
# CONFIGURATION: Set which script to run
# =============================================================================
SCRIPT_TO_RUN = 'triton_kernels.py'  # Change this to run different files
# Examples:
# SCRIPT_TO_RUN = 'test_workflow.py'
# SCRIPT_TO_RUN = 'scripts/my_experiment.py'
# SCRIPT_TO_RUN = 'experiments/model_training.py'
# =============================================================================

def list_python_files():
    \"\"\"List all available Python files in the repository\"\"\"
    print("Available Python files:")
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file).replace('./', '')
                print(f"  {{filepath}}")

def run_script(script_path):
    \"\"\"Run a Python script and handle errors\"\"\"
    if not os.path.exists(script_path):
        print(f"Error: Script '{{script_path}}' not found")
        print("\\nTip: Set SCRIPT_TO_RUN = '--list' to see all available files")
        list_python_files()
        return False

    print(f"Running {{script_path}}...")
    try:
        exec(open(script_path).read())
        print(f"Successfully completed {{script_path}}")
        return True
    except Exception as e:
        print(f"Error running {{script_path}}: {{e}}")
        return False

def download_files():
    \"\"\"Download files directly from GitHub using requests (fallback for git issues)\"\"\"
    import requests
    import tempfile
    
    base_url = "{git_url}"
    files_to_download = [
        'triton_kernels.py',
        'test_workflow.py'
    ]
    
    print("Git not available. Downloading files directly from GitHub...")
    
    # Create working directory
    work_dir = '/kaggle/working/gpu-dl-playground'
    os.makedirs(work_dir, exist_ok=True)
    os.chdir(work_dir)
    
    success_count = 0
    for filename in files_to_download:
        try:
            raw_url = f"{{base_url}}/raw/main/{{filename}}"
            print(f"Downloading {{filename}}...")
            
            response = requests.get(raw_url, timeout=30)
            response.raise_for_status()
            
            with open(filename, 'w') as f:
                f.write(response.text)
            
            print(f"Successfully downloaded {{filename}}")
            success_count += 1
            
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {{filename}}: {{e}}")
        except Exception as e:
            print(f"Error saving {{filename}}: {{e}}")
    
    if success_count > 0:
        print(f"\\nSuccessfully downloaded {{success_count}}/{{len(files_to_download)}} files")
        return True
    else:
        print("\\nFailed to download any files")
        return False

def get_code():
    \"\"\"Get code using git or direct download fallback\"\"\"
    repo_name = "gpu-dl-playground"
    
    # Try git first
    if os.path.exists(f'/kaggle/working/{{repo_name}}'):
        print("Repository exists, trying to pull latest changes...")
        os.chdir(f'/kaggle/working/{{repo_name}}')
        try:
            result = subprocess.run(['git', 'pull'], capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                print("Successfully updated repository")
                return True
            else:
                print("Git pull failed, continuing with existing files...")
                return True
        except:
            print("Git pull failed, continuing with existing files...")
            return True
    else:
        print("Trying to clone repository...")
        os.chdir('/kaggle/working')
        
        try:
            result = subprocess.run(['git', 'clone', '{git_url}.git'], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                print("Successfully cloned repository")
                os.chdir(f'/kaggle/working/{{repo_name}}')
                return True
        except:
            pass
        
        print("Git clone failed. Using direct download fallback...")
        return download_files()

# Install dependencies
print("Installing dependencies...")
!pip install triton requests -q

# Get code (git or direct download)
if not get_code():
    print("\\nFailed to get code. Cannot proceed.")
    sys.exit(1)

# Check GPU
import torch
print(f'\\nCUDA available: {{torch.cuda.is_available()}}')
if torch.cuda.is_available():
    print(f'   GPU: {{torch.cuda.get_device_name(0)}}')

print(f"\\nCurrent directory: {{os.getcwd()}}")
print(f"Repository contents:")
!ls -la

# Handle special commands
if SCRIPT_TO_RUN == '--list':
    list_python_files()
else:
    run_script(SCRIPT_TO_RUN)
"""

    with open("kaggle_git_sync.py", "w") as f:
        f.write(notebook_code)

    print("📝 Created kaggle_git_sync.py")
    print(f"🔗 Git URL detected: {git_url}")
    print("\n🎯 Streamlined workflow:")
    print("   1. Code locally, commit & push to GitHub")
    print("   2. Run kaggle_git_sync.py in Kaggle (with GPU enabled)")
    print("   3. Your latest code runs automatically!")

    return True


def main():
    parser = argparse.ArgumentParser(description="Seamless Kaggle code sync")
    parser.add_argument("script", nargs="?", help="Python script to sync")
    parser.add_argument("--setup", action="store_true", help="Setup Kaggle dataset")
    parser.add_argument(
        "--git-sync", action="store_true", help="Create git sync notebook"
    )
    parser.add_argument(
        "--dataset-id", help="Kaggle dataset ID (username/dataset-name)"
    )

    args = parser.parse_args()

    if args.setup:
        setup_kaggle_dataset()
    elif args.git_sync:
        create_git_sync_notebook()
    elif args.script:
        push_to_kaggle(args.script, args.dataset_id)
    else:
        print("Usage examples:")
        print("  python scripts/kaggle_sync.py --setup           # One-time setup")
        print("  python scripts/kaggle_sync.py --git-sync        # Git-based workflow")
        print(
            "  python scripts/kaggle_sync.py script.py         # Push specific script"
        )


if __name__ == "__main__":
    main()
