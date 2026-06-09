# 🚀 CUDA Insight: Profile, Analyze, Optimize

![CI](https://github.com/Paramveersingh-S/cuda-insight/actions/workflows/test.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green)
![CUDA](https://img.shields.io/badge/CUDA-11.0%2B-76B900.svg)

> **Write CUDA. Understand it. Fix it. Ship faster.**

<div align="center">
  <img src="https://asciinema.org/a/demo.svg" alt="CUDA Insight Demo" width="800">
  <br>
  <i>(Placeholder for asciinema terminal recording)</i>
</div>

CUDA Insight is an all-in-one educational toolkit and profiler designed to demystify GPU performance. Whether you're learning CUDA or optimizing a production kernel, CUDA Insight wraps NVIDIA's powerful toolchain into a simple, actionable workflow.

---

## ✨ Features

- **📊 Hardware Discovery (`gpu-info`)**: Instantly retrieve specs like theoretical TFLOPS, memory bandwidth, and compute capability.
- **🔍 PTX Annotation (`ptx`)**: Automatically map generated PTX assembly instructions directly to your original C++ source lines.
- **⚡ One-Click Profiling (`profile`)**: Automatically runs `ncu` (Nsight Compute), captures metrics, and generates a structured analysis report.
- **🧠 Expert Suggestions Engine**: Identifies bottlenecks (e.g., Bank Conflicts, Register Pressure) and provides actionable code snippets to fix them.
- **📈 Roofline Modeling**: Automatically generates visual Roofline charts (`roofline.png`) to show how close your kernel is to the theoretical hardware limit.
- **🔥 PyTorch Integration**: Easily profile your custom PyTorch C++ extensions via our `torch_ext` API.

---

## 🛠️ Installation

```bash
pip install -e ".[dev]"
```

*Requirements: NVIDIA GPU, `nvcc`, and `ncu` (Nsight Compute) must be in your `PATH`.*

---

## 🚀 Quick Start

### 1. View GPU Specs
```bash
cuda-insight gpu-info
```

### 2. Inspect Assembly (PTX/SASS)
```bash
# See how your CUDA maps to PTX
cuda-insight ptx kernels/matmul/naive.cu --annotate
```

### 3. Profile & Analyze a Kernel
```bash
# Compile and profile a binary. Generates HTML report and Roofline chart.
nvcc kernels/matmul/naive.cu -o naive_matmul
cuda-insight profile naive_matmul --html --roofline
```

---

## 📚 Reference Kernels Library

Inside the `kernels/` directory, you'll find a progressive library of algorithms showing the evolution from *naive* to *state-of-the-art*:

- **Matrix Multiplication (`matmul/`)**: Naive $\rightarrow$ Shared Memory Tiling $\rightarrow$ cuBLAS Reference
- **Parallel Reduction (`reduction/`)**: Atomic $\rightarrow$ Block Shared Mem $\rightarrow$ Warp Primitives
- **Softmax (`softmax/`)**: Two-Pass $\rightarrow$ Online Normalizer (FlashAttention style)
- **Attention (`flash_attention_minimal/`)**: Block-Sparse Fused Attention
- **Quantization (`quantization/`)**: INT8 `dp4a` Gemm

Each kernel includes an educational header block explaining the optimization, expected speedup, and further reading links!

---

## 🏗️ Architecture & Workflow

```mermaid
graph TD
    A[User writes .cu file] -->|cuda-insight ptx| B(Compiler Wrapper)
    B --> C[PTX / SASS Extractor]
    C --> D[PTX Annotator]
    D -.-> E((Rich CLI Output))
    
    F[Compiled Binary] -->|cuda-insight profile| G(NCU Profiler Wrapper)
    G --> H[CSV Parser]
    H --> I[Analyzer & Roofline Model]
    I --> J[Suggestions Engine]
    J --> K[Jinja2 HTML Reporter]
    K -.-> L((Interactive Dashboard))
```

---

## 🧠 Example Output

### Bottleneck Detection
If you run the naive matmul, CUDA Insight will tell you:
> **Bottleneck: Memory Bandwidth Bound**  
> *Kernel is bottlenecked by global memory bandwidth. Utilizing 98% of peak.*  
> **Fix:** Use shared memory tiling to reduce global memory reads.

---

## 🤝 Contributing

Contributions are welcome! If you want to add a new reference kernel (e.g., Prefix Scan, GEMV, Convolution), feel free to open a PR.

---

## 🆚 Comparison

| Feature | CUDA Insight | Nsight Systems | Manual Profiling (nvcc/ncu) |
|---|---|---|---|
| **Setup Time** | < 1 minute | Heavy GUI install | Needs scripting |
| **Output** | Actionable text + HTML | Complex timeline | Raw CSV/JSON |
| **Fix Suggestions**| Built-in Expert Engine | None | StackOverflow |
| **PyTorch Ops** | 1-line wrapper | Requires NVTX markers | Complex |
| **Roofline** | Auto-generated chart | Manual | Manual |

---

## ⚡ Benchmark Results

Applying the automated suggestions from CUDA Insight yields massive speedups on our reference kernels on an NVIDIA A100 (sm_80):

| Kernel | Naive Throughput | Optimized Throughput | Speedup | Bottleneck Fixed |
|---|---|---|---|---|
| Matrix Multi (4096) | ~1.1 TFLOPS | ~15.2 TFLOPS | **13.8x** | Global Memory Bandwidth |
| Reduction (10^8) | ~12 GB/s | ~1600 GB/s | **133x** | Atomic Contention -> Warp Shfl |
| Softmax | 400 GB/s | 1250 GB/s | **3.1x** | Two-pass Global Mem Reads |

---

## 🏢 Used By

- Deep Learning Researchers
- AI Performance Engineers
- (Add your university/lab here!)

---

## 📝 Citation

If you use `cuda-insight` in your research, please cite:

```bibtex
@software{cuda_insight,
  author = {Paramveer Singh},
  title = {CUDA Insight: A Kernel Profiling, Debugging & Optimization Toolkit},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Paramveersingh-S/cuda-insight}}
}
```

## 📝 License
MIT License.
