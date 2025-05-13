## Phase 0: Prerequisites (Week 0)

Before diving into GPU programming, the following would be helpful:

1. **Programming Knowledge:**

   - Proficient in C++ (for CUDA) and Python (for Triton)
   - Understanding of basic data structures and algorithms
   - Familiarity with linear algebra concepts (matrices, vectors, transformations)

2. **Hardware Understanding:**

   - Basic computer architecture concepts
   - Understanding the difference between CPU and GPU execution models
   - Access to an NVIDIA GPU (preferably with Tensor Cores - Volta architecture or newer). _See Mac Setup Guide if using Apple Silicon._

3. **Development Environment:**
   - Linux-based system (preferred) or Windows with WSL for native CUDA execution. For users on Apple Silicon (M-series) Macs, see **[Mac Setup Guide for CUDA/Triton Development](MAC_SETUP_GUIDE.md)** for important notes on working with NVIDIA GPU technologies via remote/cloud resources.
   - Code editor/IDE of choice (VSCode recommended with CUDA extensions)
   - Git for version control
   - Familiarity with using a terminal/command line.

### Essential Math for GPU Programming

- Matrix/tensor layout fundamentals (row-major vs column-major)
- Arithmetic intensity calculations (FLOPs/byte ratio)
- Basic parallel algorithm complexity (work vs span)
- Memory access pattern analysis (stride patterns, spatial locality)
- Basic understanding of floating-point arithmetic (precision, potential for NaNs/Infs, non-associativity)

Resources:

- "C++ Primer" (5th Edition) for C++ refresher
- "Numerical Linear Algebra" by Trefethen and Bau for math foundations
- "Computer Systems: A Programmer's Perspective" for general computing background

## Phase 1: CUDA Fundamentals (Weeks 1-3)

_Note: Phase 1 covers many foundational and advanced topics. The goal in these initial weeks is to gain exposure and a solid foundational understanding. Deep mastery of every advanced technique will come with continued practice and experience beyond this phase._

### Week 1 - Core Concepts & First Kernels:

1. Install CUDA Toolkit + NSight Systems/Compute (on a CUDA-enabled machine/VM).
2. Learn GPU architecture basics:
   - Thread hierarchy (threads < blocks < grid)
   - Memory hierarchy (global, shared, registers) - _Why is shared faster? What are their scopes & lifetimes?_
   - SIMT (Single Instruction, Multiple Threads) execution model & concept of warps - _Key for understanding warp behavior and divergence._
   - Basic awareness of my target GPU architecture (SM count, memory bandwidth, L1/L2 cache sizes).
3. Write first kernels:
   - Vector addition (CPU vs GPU baseline)
   - Simple element-wise operations (e.g., scalar multiplication, ReLU activation) - _Start with these for basic kernel launch and data handling._
   - Matrix multiplication (naive version – three nested loops, focus on understanding global memory access patterns and their inefficiency).
4. _Suggestion:_ Start a learning log/GitHub repo to track progress, notes, and code examples.

Resources:

- NVIDIA CUDA C++ Programming Guide (Ch 1-3)
- CUDA by Example (Book) - First 4 chapters
- NVIDIA Developer Blogs (CUDA section) for deep dives.
- NVIDIA's "CUDA C++ Best Practices Guide" - essential reading for optimization principles.

### Week 2 - Core Memory Optimization & Profiling Introduction:

1. Understanding Global Memory: Latency, Bandwidth, and Access Patterns (Strided, Uncoalesced). _Analyze the naive matrix multiplication kernel's memory access._
2. Memory Coalescing: _What is it, why is it critical? Implement and verify coalesced global memory access for a simple kernel (e.g., matrix transpose or a modified vector operation)._
3. Shared Memory: _Purpose and benefits. Understand its explicit management by the programmer._
4. Shared Memory Tiling for Matrix Multiplication: Implement matrix multiplication using shared memory to load tiles of input matrices. _This is a cornerstone CUDA optimization pattern._
5. Bank Conflicts in Shared Memory: _What are they and how to avoid them for optimal shared memory throughput?_
6. Basic Profiling: Introduce Nsight Compute. Profile my naive and shared-memory matrix multiplication kernels. Look at memory throughput, instruction execution, and basic bottleneck identification.
7. Understanding memory bandwidth vs. computational throughput (e.g., theoretical peak FLOPs and bandwidth of my GPU). _Compare my kernel's performance against these._
8. **Memory footprint analysis** - Understanding and measuring how much global/shared memory my kernels use.
   - Tools: `cudaMemGetInfo()`, Nsight Memory Profiler basics.

Resources:

- Revisit relevant sections of the CUDA C++ Programming Guide and Best Practices Guide.
- Nsight Compute Documentation (getting started).

### Week 3 - Advanced CUDA Patterns, Concurrency & Deeper Profiling:

1. Warp-Level Operations: (e.g., shuffles like `__shfl_sync`, broadcasts `__broadcast_sync`, and warp-wide reductions/queries like `__ballot_sync`, `__activemask()`). _Understand their use cases for inter-thread communication within a warp. Why are these efficient?_
2. Atomic Operations: Common use cases (e.g., building histograms, lock-free updates for shared counters, parallel reductions where conflicts are possible). _What are their performance implications and when to use them?_
3. Streams and Concurrency: Using `cudaStream_t` to overlap computation with data movement (`cudaMemcpyAsync`) and kernel execution. _Implement a simple example of overlapping a kernel with data copies._
4. Deeper Profiling with Nsight Systems/Compute:
   - Analyzing the timeline view in Nsight Systems for concurrent operations.
   - Using Nsight Compute to dive deeper into kernel performance: occupancy, instruction stalls, memory latency effects. Further explore the Roofline Model concept.
5. **Introduction to Tensor Core Programming (Conceptual):**
   - Overview of NVIDIA Tensor Cores: their purpose (accelerating matrix multiply-accumulate operations) and architecture.
   - Understanding mixed-precision techniques (FP16, BF16, TF32) and their use with Tensor Cores.
   - _Awareness point:_ WMMA API (`nvcuda::wmma`) for direct CUDA C++ control and the CUTLASS library for high-performance templated primitives. (Deep dive into these is for further study).
6. **Advanced CUDA Concepts (Awareness & Further Study):**
   - Register blocking/tiling - _Conceptual understanding._
   - Prefetching techniques - _Conceptual understanding._
   - Persistent thread blocks - _Concept and use cases for reducing launch overhead._

## Phase 2: Triton for DL Research (Weeks 4-5)

### Week 4 - Triton Basics & Benchmarking Framework:

1. Install Triton + PyTorch integration (on my CUDA-enabled machine/VM).
2. Learn Triton DSL:
   - `@triton.jit` decorator, kernel launch syntax.
   - `tl.load`/`tl.store` (masked and unmasked), `tl.make_block_ptr`.
   - Program IDs (`tl.program_id`), axes, and ranges (`tl.arange`).
   - `tl.constexpr` for compile-time constants.
   - Understanding Triton's block-level programming model (writing programs that operate on _tiles_ of data) vs. CUDA's thread-level programming.
3. Reimplement CUDA examples in Triton:
   - Vector add.
   - Simple element-wise operations (e.g., scalar multiplication, ReLU activation).
   - Matrix multiplication (compare performance and code complexity with CUDA version and PyTorch `torch.matmul`).
4. **Create a standardized benchmarking framework:**
   - Build a simple Python utility (e.g., using `time` module, `torch.cuda.Event` for timing, `triton.testing.do_bench`) to compare:
     - Runtime performance (wall clock time).
     - Memory usage (if measurable simply).
     - Computational throughput (GFLOPs/s).
     - Bandwidth utilization (GB/s).
   - Compare implementations across: PyTorch native, my CUDA kernels, my Triton kernels.
   - Add qualitative comparison dimensions:
     - Lines of Code (LOC) complexity.
     - Estimated Development time.
     - Portability across GPU architectures (conceptual for now, Triton aims for better portability).
     - Maintainability & Readability.
     - Ease of Debugging.

Resources:

- Official Triton Tutorials (Parts 1-3): `https://github.com/openai/triton/tree/main/python/tutorials` (_Prioritize these_)
- Triton Documentation (`https://triton-lang.org/main/index.html`)
- "Triton: An Intermediate Language" paper (_Review for foundational concepts, but docs/tutorials are more current._)
- **OpenAI Triton Kernels repository**: `https://github.com/openai/triton/tree/main/python/triton/` for real-world examples.

### Week 5 - DL-specific Optimizations in Triton:

1. **Gradual progression toward attention mechanisms:**
   - Start with implementing a simple softmax kernel in Triton (row-wise).
   - Progress to matrix multiplication with an element-wise operation (e.g., scaling) and a masking operation.
   - Then implement a basic scaled dot-product attention (without complex masking or fusion initially).
2. Fused attention implementation - _Further develop the basic attention by fusing softmax and matmuls. Compare with non-fused versions before tackling FlashAttention replicas._
3. Kernel fusion techniques in Triton: (e.g., fusing element-wise operations with matmuls, combining multiple element-wise ops, implementing custom activation functions directly within a kernel). _Why? Reduces memory bandwidth bottlenecks by keeping data in faster on-chip memory (registers/shared memory if Triton compiler stages it there)._
4. Understanding Triton & Autograd: How Triton kernels integrate with PyTorch's automatic differentiation (e.g., via `torch.compile` or by writing a custom `torch.autograd.Function`).
5. Memory-bound vs compute-bound analysis for Triton kernels using profilers (Nsight Compute on generated PTX, `torch.profiler`).
6. **Introduction to Sparse computation techniques in Triton:**
   - Implementing simple sparse matrix operations (e.g., SpMV with CSR format if feasible as an intro).
   - Understanding challenges with irregular memory access patterns.
7. **Triton Autotuning Workflow:**
   - Identify tunable parameters (`BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_K`, `num_warps`, `num_stages`).
   - Define search space constraints using `triton.Config`.
   - Implement automated tuning with `@triton.autotune` decorator.
   - Analyze performance/accuracy tradeoffs from different configurations.

## Phase 3: Integration & Projects (Weeks 6-8)

### Week 6 - Mixed Workflow & Advanced Studies:

1. Profile a baseline PyTorch model (e.g., a simple Transformer from scratch or a small Hugging Face model) using `torch.profiler`.
2. Identify computational bottlenecks (e.g., attention, GEMM, LayerNorm).
3. Develop decision criteria: When to choose Triton (rapid prototyping, Pythonic interface, good performance for many DL ops, easier kernel fusion) vs. CUDA (maximum fine-grained control, squeezing every bit of performance, highly specialized or novel low-level operations not easily expressed in Triton).
4. Prototype optimization in Triton for an identified bottleneck.
5. _Optional:_ Attempt to optimize the same critical path further with CUDA if Triton doesn't meet hypothetical performance goals.
6. **Case studies of real-world optimizations:**
   - Study and understand the high-level ideas behind:
     - FlashAttention (original, V2, V3) - Tiling, recomputation, online softmax, minimizing HBM access.
     - DeepSpeed ZeRO - Optimizations for distributed training (conceptual understanding of memory sharding).
     - Megatron-LM - Model parallelism techniques (conceptual).
   - Analyzing `torch.compile` outputs with the Triton backend to see how PyTorch applies optimizations.

### Week 7-8 - Capstone Project:

Implement a custom Transformer variant or another DL model component that presents optimization opportunities:

1. Use Triton for experimental components or parts where Pythonic expression is beneficial (e.g., novel attention mechanism, custom activation function, fused layer norm).
2. Use CUDA for highly optimized components _only if necessary for performance goals and Triton proves insufficient or less suitable for the specific low-level control needed_ (e.g., a very specific reduction pattern, complex atomic interactions).
3. Benchmark against PyTorch baseline (native and `torch.compile`).
4. Profile and optimize memory transfers and kernel performance using Nsight tools and `torch.profiler`.
5. **Document and share my optimization journey:**
   - Create detailed write-ups of my optimization process (e.g., blog post, GitHub project README).
   - Include performance measurements before/after each significant optimization, clearly stating the hardware and conditions.
   - Share insights about debugging challenges and bottlenecks encountered.
   - Open-source my implementations when possible to contribute and get feedback.

## Key Resources

- NVIDIA CUDA Zone (`https://developer.nvidia.com/cuda-zone`)
- Triton Documentation (`https://triton-lang.org`)
- Triton GitHub Tutorials (`https://github.com/openai/triton/tree/main/python/tutorials`)
- NVIDIA Developer Blogs (`https://developer.nvidia.com/blog/`)
- _Awareness of CUDA Ecosystem Libraries:_ cuBLAS, cuDNN, cuSPARSE, CUTLASS, NCCL (for multi-GPU, future study).
- Courses:
  - "Intro to GPU Programming" (Udacity)
  - "Heterogeneous Parallel Programming" (Coursera - UIUC)
  - Check for publicly available university course materials (e.g., Stanford CS149/CS249, CMU 15-418/618).
- Books:
  - "Programming Massively Parallel Processors" (4th Ed) by Hwu, Kirk, and El Hajj
  - "CUDA Handbook" by Nicholas Wilt
  - "Professional CUDA C Programming" by Cheng, Grossman, and McKercher
- **Community Resources:**
  - NVIDIA Developer Forums (`https://forums.developer.nvidia.com/`)
  - PyTorch Forums (`https://discuss.pytorch.org/`)
  - Triton GitHub Issues & Discussions. (Check for official Discord/community links on their site).
  - HuggingFace Discord (has active GPU optimization community).
  - Reddit communities: r/CUDA, r/MachineLearning.
  - _Awareness:_ Relevant academic conferences (NeurIPS, ICML, ICLR, PPoPP, SC, ASPLOS) for cutting-edge research papers on GPU optimization for ML.

## Debugging and Profiling Workflow

- **General:**
  - Always profile before/after optimizations to quantify impact.
  - Start with small tensor sizes for functional debugging; ensure correctness before performance tuning.
  - Use `torch.cuda.synchronize()` (in Python) or `cudaDeviceSynchronize()` (in CUDA C++) for accurate GPU timing.
- **CUDA:**
  - _Functional Debugging:_ `printf` from kernels (use sparingly, can affect execution), `cuda-gdb` for interactive debugging.
  - _Memory Debugging:_ `compute-sanitizer` (preferred for memory errors, race conditions) or `cuda-memcheck`.
  - _Performance Profiling:_ **Nsight Compute** (for deep kernel-level analysis: stalls, occupancy, memory traffic), **Nsight Systems** (application-level, CPU-GPU interactions, API calls, concurrency).
- **Triton:**
  - _Functional Debugging:_ `TRITON_INTERPRET=1` env var (runs on CPU via NumPy, allows `pdb`), `print()` within kernels (requires `TRITON_DEBUG=1` env var, recompilation needed), `assert` statements.
  - _Launch Debugging:_ `triton.runtime.jit.PRINT_LAUNCH_CONFIG=True` to inspect kernel launch parameters.
  - _Performance Profiling:_ `torch.profiler` (when called from PyTorch), `triton.testing.do_bench`, Nsight Compute (can be used on generated PTX code), Triton Autotuner (`@triton.autotune`).
- **Common Debugging Examples:**
  - Debugging race conditions in atomics or shared memory.
  - Fixing shared memory bank conflicts.
  - Resolving warp divergence issues and their performance impact.
  - Troubleshooting incorrect kernel launches (wrong grid/block dimensions).
  - Understanding and debugging precision issues (FP32 vs FP16/BF16 differences, NaN/Inf propagation, catastrophic cancellation).

## Tips for Success (Incorporating DeepSeek's Advice)

1. **Profile Systematically:** Profile before optimizing, profile after. Understand _what_ I'm improving and _why_.
2. **Start Small & Iterate:** Debug kernel logic with small, manageable tensor sizes. Incrementally add complexity and features.
3. **Accurate Timing:** Use proper synchronization for benchmarking GPU code.
4. **Study Real-world Code & Papers:**
   - FlashAttention (CUDA implementations & paper) - _Advanced example_.
   - OpenAI Triton Kernels repository - _Excellent practical examples_.
   - Kernels generated by `torch.compile` (explore the generated code if possible).
   - Other open-source high-performance libraries (e.g., CUTLASS, xFormers) and relevant research papers.
5. **Maintain a Learning Log/Portfolio:** Document progress, challenges, solutions, key learnings, and link to my implemented kernels (e.g., on GitHub).
6. **Seek & Offer Peer Code Reviews:** If possible, have my CUDA/Triton code reviewed. Join online communities; sharing and explaining my work solidifies understanding.
7. **Regular Incremental Progress:** Set small, achievable weekly goals rather than trying to optimize everything at once.
8. **Hardware Awareness & Continuous Learning:** Always keep my target hardware architecture in mind. GPU architectures evolve (Hopper, Blackwell, etc.), bringing new features, performance characteristics, and compiler behaviors; continuous learning is key.
9. **Don't be afraid to experiment:** Breaking things and debugging them is a core part of the learning process. Understand _why_ things broke.

## Weekly Time Commitment

- 6-8 hours/week (e.g., 2h theory/reading, 4h coding/experimenting, 2h reviewing/debugging)
- Focused 30-60 minute daily practice/review often beats infrequent marathon sessions for long-term retention of fundamentals.

This path balances theory with hands-on implementation. This will develop the ability to prototype new architectures in Triton while understanding how to push performance limits with CUDA - exactly the skills needed for DL research and optimization roles.

## Mandatory Validation Steps

1. Unit tests with edge cases (empty tensors, odd dimensions, specific values like 0, 1, large/small numbers, NaNs, Infs).
2. Numerical stability checks (monitoring for NaN/Inf propagation, potential catastrophic cancellation, differences between FP32 and lower precisions).
3. Testing against reference implementations: Use `torch.allclose` (or equivalent) with appropriate absolute and relative tolerances for floating-point results. Bitwise equivalence is more for integer logic or ensuring specific data movement/shuffle patterns are identical.
4. Gradient checking for autograd integration (e.g., using `torch.autograd.gradcheck`) if implementing custom backward passes for CUDA/Triton kernels in PyTorch.
