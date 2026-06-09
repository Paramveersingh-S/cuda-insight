# 🚀 CUDA Deep Learning Tooling Project — Master Prompt

> **Purpose:** This is a comprehensive guide for building a real, publishable, research-grade project that solves genuine pain points for deep learning researchers and CUDA developers. Follow every section in order. Do not skip phases.

---

## 🧠 Project Identity

**Project Name (working title):** `cuda-insight` — *A CUDA Kernel Profiling, Debugging & Optimization Toolkit for Deep Learning Researchers*

**Tagline:** "Write CUDA. Understand it. Fix it. Ship faster."

**Core Problem Being Solved:**
Deep learning researchers who write custom CUDA kernels — for attention mechanisms, custom activations, fused ops, quantization kernels, etc. — face a brutal workflow:

1. They write a kernel and it's slow, but they don't know *why* — occupancy? memory bandwidth? warp divergence?
2. cuBLAS/cuDNN are black boxes. Replacing them with custom kernels requires painful trial-and-error.
3. PTX and SASS (assembly) output from `nvcc` is hard to interpret without expert knowledge.
4. Memory access pattern bugs (bank conflicts, uncoalesced accesses) silently kill performance.
5. There is no easy-to-use CLI tool that gives a researcher a clear, human-readable report: "here's what's wrong with your kernel and here's how to fix it."

**What this project delivers:**
A CLI + Python library that takes a CUDA `.cu` file (or a PyTorch custom op), compiles it, runs it on the GPU, profiles it using NVIDIA's profiling APIs (NVTX, CUPTI, Nsight), and returns a structured, human-readable report with:
- Achieved vs theoretical occupancy
- Memory bandwidth utilization
- Warp divergence hotspots
- Bank conflict detection
- Register pressure analysis
- Side-by-side PTX annotation
- Actionable fix suggestions (rule-based + optionally LLM-assisted)

---

## 📦 Repository Structure to Build

```
cuda-insight/
├── README.md                        # Rich README with badges, examples, benchmarks
├── CONTRIBUTING.md
├── LICENSE                          # Apache 2.0
├── pyproject.toml                   # Modern Python packaging
├── setup.cfg
├── Makefile                         # build, test, lint, docs targets
│
├── cuda_insight/                    # Core Python package
│   ├── __init__.py
│   ├── cli.py                       # Click-based CLI entrypoint
│   ├── compiler.py                  # nvcc wrapper: compile .cu → .so / PTX / SASS
│   ├── profiler.py                  # CUPTI / Nsight Systems integration
│   ├── analyzer.py                  # Parse profiler output → structured metrics
│   ├── reporter.py                  # Render human-readable + JSON reports
│   ├── suggestions.py               # Rule-based fix engine
│   ├── ptx_annotator.py             # Map PTX lines back to source lines
│   └── utils/
│       ├── gpu_info.py              # GPU capability query (sm version, warps, SMs)
│       ├── ncu_parser.py            # Parse Nsight Compute CSV/NV-report output
│       └── perf_model.py            # Roofline model calculator
│
├── kernels/                         # Reference & benchmark kernels
│   ├── matmul/
│   │   ├── naive.cu
│   │   ├── tiled.cu
│   │   ├── cublas_ref.cu
│   │   └── README.md
│   ├── softmax/
│   │   ├── naive.cu
│   │   ├── online_softmax.cu
│   │   └── README.md
│   ├── reduction/
│   │   ├── atomic.cu
│   │   ├── warp_reduce.cu
│   │   └── README.md
│   ├── flash_attention_minimal/
│   │   ├── flash_attn_v1.cu
│   │   └── README.md
│   └── quantization/
│       ├── int8_gemm.cu
│       └── README.md
│
├── benchmarks/
│   ├── run_all.py
│   └── results/                     # Pre-run JSON results for README charts
│
├── tests/
│   ├── test_compiler.py
│   ├── test_analyzer.py
│   ├── test_reporter.py
│   └── fixtures/                    # Small .cu files used in tests
│
├── docs/
│   ├── index.md
│   ├── getting_started.md
│   ├── metrics_explained.md         # Deep explanation of every metric
│   ├── roofline_model.md
│   └── contributing_kernels.md
│
└── examples/
    ├── profile_matmul.py
    ├── profile_custom_attention.py
    └── pytorch_custom_op_example.py
```

---

## 🔧 Tech Stack & Dependencies

| Layer | Technology | Why |
|---|---|---|
| GPU profiling | NVIDIA Nsight Compute CLI (`ncu`) | Industry standard, outputs CSV/JSON |
| GPU profiling (runtime) | CUPTI (via `pycupti` or ctypes) | Fine-grained kernel-level metrics |
| Compilation | `nvcc` subprocess wrapper | Compile .cu files to PTX/SASS/shared lib |
| PTX/SASS parsing | Regex + `cuobjdump` | Map instructions back to source |
| CLI | `click` + `rich` | Beautiful terminal output |
| Report rendering | `rich` tables + `jinja2` HTML | Both terminal and browser reports |
| Roofline model | Pure numpy + matplotlib | Visualize compute vs memory bound |
| Python binding | `ctypes` / `cffi` | Load compiled kernels for benchmarking |
| Testing | `pytest` + `pytest-mock` | Standard |
| Packaging | `pyproject.toml` (PEP 517) | Modern, pip-installable |
| Optional LLM layer | Anthropic API | "Explain this bottleneck in plain English" |
| CI | GitHub Actions | Auto-test on CPU (mocked), GPU runner optional |

---

## 🗺️ Build Phases — Follow Strictly In Order

### ─── PHASE 0: Project Skeleton & Tooling ───

**Goal:** Repo exists, installs cleanly, CI passes, README skeleton is live.

**Tasks:**
- [ ] Initialize git repo with `.gitignore` (CUDA build artifacts, `*.o`, `*.so`, `__pycache__`)
- [ ] Write `pyproject.toml` with metadata, dependencies, optional extras (`[llm]`, `[viz]`)
- [ ] Write `Makefile` with targets: `build`, `test`, `lint`, `docs`, `clean`
- [ ] Write skeleton `__init__.py` that exposes top-level API: `profile()`, `analyze()`, `report()`
- [ ] Set up `pytest` with a passing dummy test
- [ ] Write `README.md` skeleton: badges (PyPI, license, Python version), one-paragraph description, "coming soon" sections
- [ ] Create GitHub Actions workflow: install Python, run `pytest`

**Deliverable check:** `pip install -e .` works. `pytest` passes. README renders on GitHub.

---

### ─── PHASE 1: GPU Info & Environment Detection ───

**Goal:** Tool knows exactly what GPU it's running on and surfaces that information.

**Tasks:**
- [ ] Write `gpu_info.py`: query `nvidia-smi` and `nvcc --version` via subprocess
  - Capture: GPU name, compute capability (sm_XY), VRAM, # SMs, warp size, max threads/block, L2 cache size, memory bandwidth (theoretical)
  - Output as a typed dataclass `GPUInfo`
- [ ] Handle gracefully: no GPU present, driver too old, `nvcc` not in PATH
- [ ] Add CLI command: `cuda-insight gpu-info` → pretty `rich` table
- [ ] Write unit tests with mocked subprocess output

**Key data to capture per GPU:**
```python
@dataclass
class GPUInfo:
    name: str                    # "NVIDIA A100-SXM4-80GB"
    compute_capability: str      # "8.0"
    sm_count: int                # 108
    cuda_cores_per_sm: int       # 64
    total_cuda_cores: int        # 6912
    vram_gb: float               # 80.0
    memory_bandwidth_gbps: float # 2000.0
    l2_cache_mb: float           # 40.0
    peak_tflops_fp32: float      # 19.5
    peak_tflops_fp16: float      # 77.97
    warp_size: int               # 32
    max_threads_per_block: int   # 1024
    max_shared_mem_per_block_kb: int  # 164
    driver_version: str
    cuda_version: str
```

---

### ─── PHASE 2: CUDA Compiler Wrapper ───

**Goal:** Compile any `.cu` file programmatically, capture PTX and SASS, handle errors cleanly.

**Tasks:**
- [ ] Write `compiler.py`:
  - `compile(source_path, sm_arch, output_dir, extra_flags=[])` → `CompileResult`
  - Runs `nvcc` with flags: `-O3 -lineinfo --ptx --generate-code arch=compute_XY,code=sm_XY`
  - Also dump SASS via `cuobjdump --dump-sass`
  - Capture stdout/stderr; on error, parse `nvcc` error format and return structured `CompileError`
  - Extract kernel names from PTX (`__global__` function names)
- [ ] Write `ptx_annotator.py`:
  - Parse PTX output line by line
  - Map PTX instructions back to source file + line number (using `.loc` directives in PTX)
  - Output: `List[PTXLine]` with fields: `ptx_instruction`, `source_file`, `source_line`, `source_snippet`
- [ ] CLI command: `cuda-insight compile mykernel.cu --arch sm_80 --show-ptx`
- [ ] Tests: compile a minimal `__global__ void add(...)` kernel, assert PTX is captured

**PTX parsing note:** PTX files contain `.loc` directives like:
```
.loc 1 42 8   // file index 1, line 42, column 8
```
Use these to build the source↔PTX map.

---

### ─── PHASE 3: Reference Kernels Library ───

**Goal:** Ship a library of well-documented reference kernels that researchers can learn from AND that serve as benchmarks for the profiler.

**Build these kernels (each with a naive and optimized version):**

#### 3a. Matrix Multiplication
- `naive.cu` — global memory only, no shared memory
- `tiled.cu` — shared memory tiling (classic optimization)
- Notes in each file explaining the optimization and expected speedup

#### 3b. Parallel Reduction (Sum)
- `atomic.cu` — atomic adds (slow, shows contention)
- `warp_reduce.cu` — warp-level `__shfl_down_sync` reduction (fast)
- `block_reduce.cu` — full block reduction with warp primitives

#### 3c. Softmax
- `naive.cu` — two-pass (max then exp+sum) with poor memory access
- `online_softmax.cu` — single-pass numerically stable (Flash Attention style)

#### 3d. Minimal Flash Attention
- `flash_attn_v1.cu` — block-sparse attention with shared memory tiling
- Heavily commented explaining the tiling strategy

#### 3e. INT8 GEMM (Quantization)
- `int8_gemm.cu` — demonstrate `dp4a` instruction usage

**Each kernel file must have a header comment block:**
```c
/*
 * KERNEL: Tiled Matrix Multiplication
 * OPTIMIZATION: Shared memory tiling (TILE_SIZE x TILE_SIZE)
 * KEY INSIGHT: Reduces global memory reads from O(N^3) to O(N^3 / TILE_SIZE)
 * EXPECTED SPEEDUP: 10-20x over naive on A100 for large N
 * KNOWN BOTTLENECK: Still memory-bandwidth bound for large tiles
 * FURTHER READING: https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/
 */
```

---

### ─── PHASE 4: Profiler Integration (Core Feature) ───

**Goal:** Run a kernel and capture real hardware performance counters.

**Strategy:** Use `ncu` (Nsight Compute CLI) as the primary backend because:
- It ships with CUDA toolkit
- Outputs structured CSV/JSON
- Captures hundreds of hardware counters without writing C++ profiling code

**Tasks:**
- [ ] Write `profiler.py`:
  - `profile_kernel(binary_path, kernel_name, args, gpu_info)` → `ProfileResult`
  - Runs: `ncu --csv --metrics <metric_list> --kernel-name <name> <binary>`
  - Parse CSV output into structured `ProfileResult`
- [ ] Write `ncu_parser.py`:
  - Parse `ncu` CSV format (it has a specific multi-header format)
  - Map metric names to human-readable descriptions
- [ ] Define the metric set to collect:

```python
NCU_METRICS = [
    # Occupancy
    "sm__warps_active.avg.pct_of_peak_sustained_active",   # achieved occupancy %
    "launch__occupancy_theoretical",                         # theoretical max occupancy
    "launch__registers_per_thread",                          # register usage
    "launch__shared_mem_per_block_static",                   # static shared mem
    
    # Memory
    "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum",         # global load bytes
    "l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum",         # global store bytes
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld", # shared mem bank conflicts
    "dram__bytes_read.sum",                                  # DRAM reads
    "dram__bytes_write.sum",                                 # DRAM writes
    
    # Compute
    "sm__inst_executed.sum",                                 # total instructions
    "sm__warps_eligible.sum",                               # warp scheduling
    "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",  # FP32 adds
    "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",  # FP32 muls
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",  # FP32 fmas
    
    # Timing
    "gpu__time_duration.sum",                               # kernel wall time
]
```

---

### ─── PHASE 5: Analyzer & Roofline Model ───

**Goal:** Turn raw profiler numbers into actionable insight.

**Tasks:**
- [ ] Write `analyzer.py`:
  - `analyze(profile_result, gpu_info)` → `AnalysisReport`
  - Compute derived metrics:
    - **Arithmetic Intensity** = FLOPs / bytes transferred
    - **Bandwidth Utilization** = achieved GB/s / theoretical peak GB/s
    - **Compute Utilization** = achieved TFLOPS / theoretical peak TFLOPS
    - **Occupancy Gap** = theoretical - achieved occupancy
    - **Memory Bound vs Compute Bound** classification using roofline

- [ ] Write `perf_model.py` — Roofline Model:
  - Inputs: arithmetic intensity, peak FLOPS, peak bandwidth
  - Output: ridge point, performance ceiling, which roof the kernel is under
  - Generate matplotlib roofline chart with kernel's operating point plotted

- [ ] Classify bottlenecks with clear rules:

```python
class Bottleneck(Enum):
    MEMORY_BANDWIDTH = "memory_bandwidth"
    SHARED_MEMORY_BANK_CONFLICTS = "shared_memory_bank_conflicts"
    LOW_OCCUPANCY_REGISTER_PRESSURE = "low_occupancy_register_pressure"
    LOW_OCCUPANCY_SHARED_MEM = "low_occupancy_shared_mem"
    WARP_DIVERGENCE = "warp_divergence"
    INSTRUCTION_THROUGHPUT = "instruction_throughput"
    LAUNCH_CONFIG = "launch_config"
    MEMORY_COALESCING = "memory_coalescing"
```

---

### ─── PHASE 6: Suggestions Engine ───

**Goal:** For each bottleneck, produce specific, actionable, copy-paste-ready suggestions.

**Tasks:**
- [ ] Write `suggestions.py` with a rule table:

| Bottleneck | Condition | Suggestion |
|---|---|---|
| Bank conflicts | `bank_conflicts > 0` | "Use padding: declare `__shared__ float smem[TILE][TILE+1]` to offset bank indices" |
| Low occupancy (registers) | `registers_per_thread > 64` | "Add `__launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)` to cap register usage" |
| Low occupancy (shared mem) | `shared_mem_per_block > 48KB` | "Split kernel into two passes or reduce tile size to increase occupancy" |
| Memory bandwidth | `bandwidth_util < 0.6` | "Check access stride — use `float4` loads for coalesced 128-byte transactions" |
| Uncoalesced access | detected from stride analysis | "Ensure thread N accesses address base + N*sizeof(T) for coalesced reads" |
| Compute bound | `compute_util < 0.7` | "Use `__fma_rn` intrinsics; consider Tensor Core ops via `wmma` API if on sm_70+" |

- [ ] Each suggestion must include:
  - Plain English explanation of *why* this is a problem
  - Code snippet showing the fix
  - Expected impact (e.g., "typically 2-4x speedup")
  - Link to NVIDIA docs / paper

---

### ─── PHASE 7: Reporter — Beautiful Output ───

**Goal:** Make the output something researchers will screenshot and share.

**Tasks:**
- [ ] Write `reporter.py` with multiple output modes:

**Terminal output (using `rich`):**
```
╔══════════════════════════════════════════════╗
║   cuda-insight Report: tiled_matmul          ║
║   GPU: NVIDIA A100 (sm_80)  |  CUDA 12.1    ║
╚══════════════════════════════════════════════╝

📊 Performance Summary
┌──────────────────────────────┬─────────────────┬──────────┐
│ Metric                       │ Value           │ Status   │
├──────────────────────────────┼─────────────────┼──────────┤
│ Kernel Duration              │ 1.24 ms         │          │
│ Achieved Occupancy           │ 62.5%           │ ⚠️  Low  │
│ Theoretical Occupancy        │ 100%            │          │
│ Memory Bandwidth Util        │ 78.3%           │ ✅ Good  │
│ Arithmetic Intensity         │ 14.2 FLOP/byte  │          │
│ Performance                  │ 8.4 TFLOPS      │          │
│ Peak FP32 Performance        │ 19.5 TFLOPS     │ 43% peak │
│ Bank Conflicts               │ 0               │ ✅ None  │
└──────────────────────────────┴─────────────────┴──────────┘

🔍 Bottleneck Analysis
  → Kernel is MEMORY BANDWIDTH BOUND (Arithmetic Intensity < Ridge Point)

⚠️  Issues Found (2)
  1. LOW OCCUPANCY — Register Pressure
     Registers/thread: 96 (limit for 100% occ: 64)
     Fix: Add __launch_bounds__(256, 2) to your kernel signature
     Expected gain: +20-35% occupancy → +10-15% throughput

  2. UNCOALESCED GLOBAL LOADS (estimated)
     Thread stride appears non-unit for matrix B access
     Fix: Transpose B before kernel call, or use tiled loading into shared mem
     Expected gain: +1.5-2x memory throughput

📈 Roofline Model: [saved to report/roofline.png]
```

- [ ] JSON output mode: `--format json` → machine-readable for CI integration
- [ ] HTML output mode: `--format html` → self-contained report with embedded chart
- [ ] Add `--compare file1.cu file2.cu` mode: side-by-side diff of two kernels' metrics

---

### ─── PHASE 8: CLI Polish ───

**Goal:** A CLI that researchers will actually use and recommend.

**Commands to implement:**

```bash
# Profile a CUDA file
cuda-insight profile matmul.cu --kernel tiled_matmul --block 256 --grid 1024

# Profile a PyTorch custom op
cuda-insight profile-torch my_op.py --op my_custom_attention --input-shape 8,1024,64

# Show GPU info
cuda-insight gpu-info

# Compile and show PTX with source annotations
cuda-insight ptx matmul.cu --annotate --kernel tiled_matmul

# Run roofline analysis only
cuda-insight roofline matmul.cu --flops 1e12 --bytes 5e9

# Compare two kernel implementations
cuda-insight compare naive_matmul.cu tiled_matmul.cu --kernel-names naive,tiled

# Run built-in benchmark suite
cuda-insight benchmark --suite matmul --sizes 256,512,1024,2048
```

**CLI quality requirements:**
- `--help` on every command is self-documenting and includes examples
- Progress bars for long operations (using `rich.progress`)
- `--verbose` flag shows subprocess commands being run
- `--no-color` for CI/pipe output
- Exit codes: 0 = success, 1 = profiling error, 2 = compile error, 3 = GPU not found

---

### ─── PHASE 9: PyTorch Integration ───

**Goal:** Let researchers profile their PyTorch custom CUDA extensions without writing a standalone `.cu` harness.

**Tasks:**
- [ ] Write `pytorch_integration.py`:
  - Detect if PyTorch is installed
  - Accept a `torch.nn.Module` or a `torch.autograd.Function`
  - Use PyTorch's `torch.utils.cpp_extension` to locate the compiled `.so`
  - Use NVTX markers around the forward pass so `ncu` can isolate just that kernel
  - Auto-generate input tensors of specified shapes for benchmarking
- [ ] Example showing profiling of a custom FlashAttention implementation

---

### ─── PHASE 10: Testing & CI ───

**Goal:** Robust test suite that runs even without a GPU (using mocks).

**Test strategy:**
- Unit tests: mock all subprocess calls (`nvcc`, `ncu`, `nvidia-smi`) using `pytest-mock`
- Integration tests: marked `@pytest.mark.gpu` — skip if no GPU present
- Fixture kernels: small `.cu` files in `tests/fixtures/` that are known-good

**Coverage targets:**
- `compiler.py`: 90%+ (all error paths tested with mocked nvcc errors)
- `analyzer.py`: 95%+ (all bottleneck rules tested with synthetic metric data)
- `reporter.py`: 80%+ (test both terminal and JSON output)
- `suggestions.py`: 100% (every rule must have a test)

**GitHub Actions:**
```yaml
# .github/workflows/test.yml
- Run on: push, pull_request
- Python versions: 3.9, 3.10, 3.11, 3.12
- Steps: install, lint (ruff), type-check (mypy), test (pytest --cov)
- GPU tests: optional self-hosted runner with NVIDIA GPU
```

---

### ─── PHASE 11: Documentation ───

**Goal:** Docs good enough that a researcher can be productive in 10 minutes.

**docs/getting_started.md** must cover:
1. Prerequisites (CUDA toolkit, `ncu` in PATH, Python 3.9+)
2. `pip install cuda-insight`
3. First profile in 3 commands
4. Reading the report (explain every metric)
5. Common fixes (top 5 bottlenecks with fixes)

**docs/metrics_explained.md** — deep dive on every metric:
- What it measures at the hardware level
- What a good value looks like
- What a bad value implies
- How to fix it

**docs/roofline_model.md** — explain the roofline model from scratch, targeted at ML researchers who know PyTorch but not microarchitecture.

---

### ─── PHASE 12: Publication & Community ───

**Goal:** Get real users and citations.

**README must include:**
- [ ] Animated terminal GIF showing the tool in action (use `asciinema`)
- [ ] Comparison table: cuda-insight vs Nsight Systems vs manual profiling
- [ ] Benchmark results: speedup achieved on reference kernels after applying suggestions
- [ ] "Used by" section (start with your own research / university)
- [ ] Citation block (BibTeX) for researchers to cite the tool

**Publish to:**
- [ ] PyPI: `pip install cuda-insight`
- [ ] Post on r/MachineLearning and r/CUDA
- [ ] Post on Hacker News (Show HN)
- [ ] Submit to Papers With Code (as a tool paper, not a research paper)
- [ ] Tweet thread: "I built a tool that tells you WHY your CUDA kernel is slow" with example output screenshot

**Future research directions to mention in README:**
- Auto-tuning launch configurations
- LLM-assisted kernel rewriting
- Integration with Triton compiler
- Support for HIP/ROCm (AMD GPUs)

---

## 🧱 Non-Negotiable Code Quality Rules

Follow these in every file you write:

1. **Type hints everywhere** — every function parameter and return type annotated
2. **Dataclasses for structured data** — no raw dicts for results/configs
3. **Graceful degradation** — if `ncu` is not installed, fall back to basic `nvprof` or tell user clearly
4. **No hardcoded paths** — everything configurable via env vars or CLI flags
5. **Subprocess calls** — always use `subprocess.run(..., capture_output=True, text=True, check=False)` and check returncode explicitly
6. **Logging** — use Python `logging` module, not `print()` in library code
7. **Error messages are human-readable** — include what went wrong, what was expected, and what to do
8. **Every public function has a docstring** with Args, Returns, Raises, Example

---

## ⚠️ Known Hard Parts — Solve These Carefully

### 1. `ncu` requires root / special permissions
- `ncu` often requires `sudo` or setting `kernel.perf_event_paranoid=0`
- Solution: detect this, print clear instructions, provide a `--sudo` flag

### 2. PTX varies by SM architecture  
- PTX generated for `sm_80` ≠ PTX for `sm_86`
- Always detect GPU's compute capability first and compile for that exact arch

### 3. Kernel naming in `ncu`
- CUDA kernel names get mangled in C++ (e.g., `_Z17tiled_matmul_kernelPfS_S_i`)
- Solution: use `c++filt` to demangle, or use regex matching on the original name

### 4. `ncu` CSV format is complex
- Has two header rows, metric names span columns, units are in a sub-row
- Write a robust parser and test it against saved `ncu` outputs in fixtures

### 5. Shared library loading for benchmarking
- Compiled `.cu` → `.so` needs proper `extern "C"` wrappers and correct `ctypes` signatures
- Test with a simple add kernel before complex ones

---

## 🎯 Success Criteria

The project is ready to share publicly when:

- [ ] `pip install cuda-insight` works from PyPI
- [ ] `cuda-insight profile kernels/matmul/naive.cu` produces a full report
- [ ] `cuda-insight profile kernels/matmul/tiled.cu` shows better metrics and fewer warnings
- [ ] The suggestions for `naive.cu` are actionable and accurate
- [ ] All 5 reference kernel pairs (naive vs optimized) are included
- [ ] `pytest` passes with 80%+ coverage
- [ ] README has animated demo GIF and benchmark numbers
- [ ] At least one researcher outside your team has tested it and given feedback
- [ ] Posted to at least 2 communities (Reddit, HN, Twitter)

---

## 📝 Session Protocol

When working on this project, each session should:

1. **Start** by stating which Phase we are in and what the specific goal is
2. **Write code** file by file, never skipping error handling
3. **Write tests** immediately after each module — never defer tests to later
4. **Commit message format**: `feat(phase-N): description` or `fix(module): description`
5. **End** each session by listing: what was completed, what's next, and any open questions/blockers

If you hit a blocker (e.g., `ncu` not available in the environment), say so clearly and build a mock/stub that can be replaced later — don't skip the feature.

---

## 🔑 First Session Starter

**Begin with Phase 0.** Your first output should be:
1. The complete `pyproject.toml`
2. The complete directory skeleton (`mkdir -p` commands)
3. The skeleton `__init__.py` with the public API stubs
4. A passing `test_skeleton.py` that imports the package
5. The `Makefile`
6. The `.gitignore`

After that, move immediately to Phase 1 (GPU Info).

Let's build something researchers will actually use.