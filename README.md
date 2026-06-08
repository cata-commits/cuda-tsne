# CUDA t-SNE — Project Plan (INF351)

## What are we doing

Implementing **t-SNE** (t-distributed Stochastic Neighbor Embedding) — a dimensionality reduction algorithm that takes high-dimensional data (like images with 784 features) and maps it to 2D for visualization, while preserving the neighborhood structure of the data.

We are implementing it 4 ways and comparing performance:

| Version | Description |
|---|---|
| CPU Baseline | Sequential C++ — nested loops, no parallelism |
| CUDA v1 — Naive | One thread per pair (i, j), reads from global memory |
| CUDA v2 — Shared Memory | Tiled computation, data loaded into shared memory to reduce global reads |
| CUDA v3 — Reduction + Coalescing | Parallel reduction for normalization step + coalesced memory access |

---

## Why these versions

Each version targets the main bottleneck of the algorithm: **pairwise distance computation** between all N×N points. This is O(N²) and has to run every iteration, so it's the most expensive part.

- **v1** shows the baseline GPU gain just from parallelism
- **v2** reduces memory bandwidth waste using shared memory (same idea as tiled matmul)
- **v3** optimizes the reduction and memory layout for warp-level efficiency

---

## How we work

Everything runs on **Google Colab** (free GPU). We write CUDA kernels inline using `%%writefile` + `!nvcc`. Each version lives in its own notebook.

Notebooks are uploaded to the repo. Colab links go in the README.

---

## Repo structure

```
cuda-tsne/
├── README.md                          ← Colab links + how to run
├── notebooks/
│   ├── 01_tsne_cpu_baseline.ipynb
│   ├── 02_tsne_cuda_naive.ipynb
│   ├── 03_tsne_cuda_shared_memory.ipynb
│   └── 04_tsne_cuda_final.ipynb       ← benchmarks + plots + conclusions
├── src/                               ← .cu kernel files extracted from notebooks
│   ├── tsne_cpu.cu
│   ├── tsne_cuda_v1.cu
│   └── tsne_cuda_v2.cu
└── data/
    └── (dataset — likely MNIST subset)
```

---

## Implementation order

```
01 CPU baseline → 02 CUDA naive → 03 shared memory → 04 benchmarks + final plots
```

Each notebook builds on the previous one. The last notebook runs all 4 versions and generates all the timing plots for the presentation.

---

## Dataset

We'll use a subset of **MNIST** (e.g. 3000–5000 samples, 784 dimensions → 2D). Small enough to run reasonably on Colab, big enough to show meaningful speedups.

---

## Deliverables (per project rules)

- Code that compiles and runs, with a README explaining how
- All results must be reproducible from the code
- Presentation slides covering: problem description, solution, results, timings, conclusions
