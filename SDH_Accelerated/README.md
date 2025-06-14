# Optimized Spatial Distance Histogram (SDH) - CUDA Project

This project presents an **optimized CUDA implementation** of the Spatial Distance Histogram (SDH) algorithm for 3D datasets. It builds upon a baseline implementation by leveraging key GPU features to significantly enhance performance and scalability.

---

## 🚀 Project Overview

The goal is to compute a histogram that categorizes distances between all pairs of 3D points into fixed-width buckets. This is useful in computational physics, astronomy, and other domains where pairwise spatial relationships matter.

The key objective of this project was to **accelerate** this histogram calculation using GPU programming via CUDA, and to **optimize** it further by reducing bottlenecks related to memory and thread contention.

---

## ⚡ Key Optimizations

### 1. Output Privatization with Shared Memory
- Each CUDA block uses a **shared histogram array** for local accumulation of results.
- Threads update shared memory first, then merge results into the global histogram using fewer `atomicAdd()` operations—reducing contention and speeding up execution.

### 2. Efficient Work Distribution & Memory Access
- Threads process balanced chunks of the point set to ensure **equal workload distribution**.
- Designed for **coalesced memory access**, improving memory throughput.

### 3. CUDA Event Timing
- **High-resolution timing** using `cudaEventRecord()` is used to accurately measure kernel performance and avoid artifacts from CPU-side timing methods.

---

## 📊 Experimental Setup

### Dataset Size Variation
- `100,000` and `500,000` points
- Bucket width = `100`
- Block size = `256`
```bash
/apps/GPU_course/runScript.sh proj2-akmal.cu 100000 100 256
/apps/GPU_course/runScript.sh proj2-akmal.cu 500000 100 256
```

### Block Size Variation
- Dataset: `100,000` points, Bucket width = `100`
- Tested with block sizes: `64`, `128`, `256`, `512`
```bash
/apps/GPU_course/runScript.sh proj2-akmal.cu 100000 100 64
/apps/GPU_course/runScript.sh proj2-akmal.cu 100000 100 128
/apps/GPU_course/runScript.sh proj2-akmal.cu 100000 100 256
/apps/GPU_course/runScript.sh proj2-akmal.cu 100000 100 512
```

---

## ⚖️ Performance Results

| Dataset Size | CPU Time (ms) | GPU Time (ms) | Speedup |
|--------------|----------------|----------------|----------|
| 100,000      | 98,300         | 1,019          | 96.5×    |
| 500,000      | 2,459,081      | 24,569         | ~100×    |

Optimizations led to:
- **Major reductions in global memory access time**
- **Near-linear scalability for larger datasets**
- **Block size 256** offering best performance balance

---

## 📁 Files Included

- `sdh_accelerated.cu` — Optimized CUDA implementation
- `Report.pdf` — In-depth explanation of optimization strategies and experimental findings

---

## License & Academic Integrity

© 2025 **Akmal Kurbanov**. All rights reserved.

This repository is shared **solely for learning and personal experimentation.**

* **Do not clone, copy, or submit any part of this project as coursework** or graded assignments. Doing so violates academic integrity policies and the author’s explicit prohibition against plagiarism and cheating.
* You may study the code, draw inspiration, and create your own original implementation—provided you follow your institution’s honor code.