"""
Test script to verify the workflow is working
This script will run a simple Triton kernel and output results
"""

import os
import sys

import torch
import triton
import triton.language as tl


@triton.jit
def multiply_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Simple element-wise multiplication kernel"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x * y
    tl.store(output_ptr + offsets, output, mask=mask)


def multiply(x: torch.Tensor, y: torch.Tensor):
    """Host function to launch the multiplication kernel"""
    output = torch.empty_like(x)
    n_elements = output.numel()

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    multiply_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


def main():
    print("🧪 Testing Triton GPU Workflow")
    print("=" * 40)

    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        print("⚠️  CUDA not available, using CPU (this will fail for Triton kernels)")
        device = "cpu"

    # Create test tensors
    size = 1024
    if device == "cuda":
        x = torch.randn(size, device="cuda")
        y = torch.randn(size, device="cuda")

        print(f"📊 Created tensors of size {size} on GPU")

        # Run our Triton kernel
        try:
            result = multiply(x, y)
            expected = x * y

            # Verify correctness
            if torch.allclose(result, expected, rtol=1e-5):
                print("✅ Triton kernel executed successfully and results match!")
                print(f"   Mean result: {result.mean().item():.6f}")
                print(
                    f"   Max difference: {
                        (result - expected).abs().max().item():.2e}"
                )
            else:
                print("❌ Results don't match expected values")
                print(
                    f"   Max difference: {
                        (result - expected).abs().max().item():.2e}"
                )
        except Exception as e:
            print(f"❌ Triton kernel failed: {str(e)}")
    else:
        # CPU fallback
        x = torch.randn(size)
        y = torch.randn(size)
        result = x * y
        print(f"📊 CPU fallback: multiplied tensors of size {size}")
        print(f"   Mean result: {result.mean().item():.6f}")

    # Environment info
    print("\n🔧 Environment Info:")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   Triton version: {triton.__version__}")
    print(f"   Python version: {sys.version.split()[0]}")
    print(f"   Platform: {sys.platform}")

    # Check GPU memory if available
    if torch.cuda.is_available():
        print(
            f"   GPU memory: {torch.cuda.get_device_properties(
                0).total_memory / 1e9:.1f} GB"
        )
        print(
            f"   GPU memory used: {
                torch.cuda.memory_allocated() / 1e6:.1f} MB"
        )

    print("\n🎉 Workflow test completed!")


if __name__ == "__main__":
    main()
