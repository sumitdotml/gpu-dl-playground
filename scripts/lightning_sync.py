#!/usr/bin/env python3
"""
Seamless Lightning AI workflow - git-based sync
"""

import argparse
import os
import subprocess
import sys


def create_lightning_git_runner():
    """Create a Lightning runner that syncs from git"""

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

    # Create Lightning app that pulls from git
    runner_content = f'''#!/usr/bin/env python3
"""
Lightning AI Git Sync Runner
Run this in Lightning AI Studio with GPU enabled
"""

import subprocess
import os
import sys

def main():
    print("⚡ Lightning AI GPU Runner")
    print("=" * 40)

    # Install dependencies
    print("📦 Installing dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip",
                          "install", "triton", "torch", "matplotlib", "numpy"])

    # Clone or pull latest code
    repo_dir = "gpu-dl-playground"
    if os.path.exists(repo_dir):
        print("📁 Repository exists, pulling latest changes...")
        os.chdir(repo_dir)
        subprocess.run(["git", "pull"])
    else:
        print("📥 Cloning repository...")
        subprocess.run(["git", "clone", "{git_url}.git", repo_dir])
        os.chdir(repo_dir)

    # Check GPU
    print("🖥️  Checking GPU...")
    try:
        subprocess.run(["nvidia-smi"], check=True)
    except:
        print("⚠️  No NVIDIA GPU detected")

    # Import torch and check CUDA
    import torch
    print(f"🔥 CUDA available: {{torch.cuda.is_available()}}")
    if torch.cuda.is_available():
        print(f"   GPU: {{torch.cuda.get_device_name(0)}}")
        print(
            f"   Memory: {{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}} GB")

    # Run the main script
    script_to_run = sys.argv[1] if len(sys.argv) > 1 else "triton_kernels.py"
    print(f"🚀 Running {{script_to_run}}...")

    try:
        exec(open(script_to_run).read())
        print("✅ Execution completed successfully!")
    except Exception as e:
        print(f"❌ Execution failed: {{e}}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''

    with open("lightning_git_runner.py", "w") as f:
        f.write(runner_content)

    os.chmod("lightning_git_runner.py", 0o755)

    # Create simple setup script for Lightning Studio
    setup_content = """#!/bin/bash
# Lightning AI Studio Quick Setup
# Just run this script in Lightning Studio with GPU enabled

echo "⚡ Lightning AI Quick Setup"

# Install git if not available
which git || apt-get update && apt-get install -y git

# Run the git sync runner
python3 lightning_git_runner.py "$@"
"""

    with open("lightning_quick_setup.sh", "w") as f:
        f.write(setup_content)

    os.chmod("lightning_quick_setup.sh", 0o755)

    print("⚡ Created Lightning AI sync files:")
    print("   - lightning_git_runner.py")
    print("   - lightning_quick_setup.sh")
    print(f"🔗 Git URL: {git_url}")
    print("\n🎯 Streamlined Lightning workflow:")
    print("   1. Code locally, commit & push to GitHub")
    print("   2. Go to lightning.ai/studios, create GPU studio")
    print("   3. Upload lightning_quick_setup.sh")
    print("   4. Run: ./lightning_quick_setup.sh your_script.py")
    print("   5. Your latest code runs automatically!")

    return True


def create_lightning_web_runner():
    """Create a web-based runner for Lightning AI"""

    web_runner = '''import requests
import tempfile
import os

def run_from_github(github_url, script_name):
    """Download and run script directly from GitHub"""

    # Convert GitHub URL to raw URL
    if "github.com" in github_url:
        raw_url = github_url.replace("github.com", "raw.githubusercontent.com")
        if "/blob/" in raw_url:
            raw_url = raw_url.replace("/blob/", "/")
        script_url = f"{raw_url}/{script_name}"
    else:
        print("❌ Please provide a valid GitHub repository URL")
        return False

    print(f"📥 Downloading {script_name} from {script_url}")
    
    try:
        response = requests.get(script_url)
        response.raise_for_status()

        # Write to temporary file and execute
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(response.text)
            temp_script = f.name
        
        print(f"🚀 Running {script_name}...")
        exec(open(temp_script).read())
        
        # Clean up
        os.unlink(temp_script)
        print("✅ Execution completed!")
        
    except Exception as e:
        print(f"❌ Failed to download/run script: {e}")
        return False
    
    return True

# Example usage:
# run_from_github("https://github.com/your-username/gpu-dl-playground", "triton_kernels.py")
'''

    with open("lightning_web_runner.py", "w") as f:
        f.write(web_runner)

    print("🌐 Created lightning_web_runner.py")
    print("   Use this to run scripts directly from GitHub URLs")

    return True


def main():
    parser = argparse.ArgumentParser(description="Seamless Lightning AI code sync")
    parser.add_argument(
        "--git-sync", action="store_true", help="Create git sync runner"
    )
    parser.add_argument("--web-runner", action="store_true", help="Create web runner")

    args = parser.parse_args()

    if args.git_sync:
        create_lightning_git_runner()
    elif args.web_runner:
        create_lightning_web_runner()
    else:
        print("Lightning AI sync options:")
        print("  --git-sync    Create git-based sync runner")
        print("  --web-runner  Create web-based runner")
        print("\nRecommended: --git-sync for most seamless workflow")


if __name__ == "__main__":
    main()
