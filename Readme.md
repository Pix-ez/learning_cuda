# 🚀 CUDA Deep Learning Kernels: From Basics to High Performance

Welcome to my learning repository on **CUDA programming for Deep Learning acceleration**.  
This is a hands-on exploration starting from first principles of GPU computing to building high-performance CUDA kernels for matrix multiplication — a foundational operation in deep learning workloads.

---

## 🧠 Purpose

This repository documents my journey to mastering CUDA kernel development for deep learning, progressing step by step:

1. 🔤 **Learning Basic CUDA Concepts**  
   Threads, blocks, warps, streaming multiprocessors (SMs), memory hierarchy (global, shared, registers), and occupancy.

2. ✳️ **Basic Matrix Multiplication Kernel**  
   Implemented naive `A x B = C` kernel using global memory only.

3. 📦 **Tiled Matrix Multiplication Kernel**  
   Leveraged shared memory tiling to optimize memory access and reduce global memory reads.

4. 🔮 **(Coming Next)** Advanced CUDA Kernels  
   - Thread coarsening  
   - Tensor core usage via WMMA API  
   - cuBLASLt integration  
   - Mixed-precision (FP16/BF16) compute  
   - Batched GEMM with streams

---

## 📈 Benchmarks

All benchmarks run on:
- **GPU:** NVIDIA GeForce RTX 2060 SUPER (SM 7.5 or similar)
- **Driver Version:** 566.36
- **Architecture:** Turing
- **Precision:** FP32,FP16,INT8 
- **Matrix size:** M = 512, K = 256, N = 126  
- **Iterations:** 20 runs (after 3 warmup rus) 
- **Timing:** Wall-clock using `clock_gettime(CLOCK_MONOTONIC)`

| Kernel Version        | Avg. Execution Time (ms) |
|-----------------------|--------------------------|
| CPU (Reference)       | 57115.670 (57 sec💀)     |
| Naive CUDA Kernel     | 120.713 😃               |
| Tiled Shared Kernel   | 78.532 🤩                |
| CublasLt FP32         | 79.177 🤩                |



---



