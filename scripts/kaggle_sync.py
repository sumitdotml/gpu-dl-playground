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

    # Get git remote URL
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"], capture_output=True, text=True
        )
        git_url = result.stdout.strip()
        if git_url.startswith("git@"):
            # Convert SSH to HTTPS
            git_url = git_url.replace("git@github.com:", "https://github.com/")
        if git_url.endswith(".git"):
            git_url = git_url[:-4]
    except:
        git_url = "https://github.com/YOUR_USERNAME/YOUR_REPO"

    notebook_code = f"""# Kaggle GPU Sync Notebook
# This notebook automatically pulls your latest code from GitHub

# Install dependencies
!pip install triton -q

# Clone/pull latest code
import os
if os.path.exists('/kaggle/working/gpu-dl-playground'):
    print("📁 Repository exists, pulling latest changes...")
    os.chdir('/kaggle/working/gpu-dl-playground')
    !git pull
else:
    print("📥 Cloning repository...")
    os.chdir('/kaggle/working')
    !git clone {git_url}.git
    os.chdir('/kaggle/working/gpu-dl-playground')

# Check GPU
import torch
print(f'🖥️  CUDA available: {{torch.cuda.is_available()}}')
if torch.cuda.is_available():
    print(f'   GPU: {{torch.cuda.get_device_name(0)}}')

# Run your script (change this line as needed)
print("🚀 Running your code...")
exec(open('triton_kernels.py').read())
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
