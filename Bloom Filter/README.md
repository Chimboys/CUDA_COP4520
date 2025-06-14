# GPU-Accelerated Bloom Filter with SipHash - CUDA Project

This project presents a high-performance CUDA-based implementation of a **Bloom Filter**, optimized for large-scale string sets and high-speed insertions and queries. The implementation is built around the cryptographically strong **SipHash-2-4** function and leverages GPU parallelism to deliver up to **350× speedups** compared to a serial CPU version.

---

## 🌱 Project Overview

A Bloom Filter is a space-efficient probabilistic data structure that supports fast membership queries. The serial CPU baseline computes `k` hashes per string, inserts or queries bits in a byte array, and performs all operations linearly.

This CUDA-accelerated version maps **one thread per string**, computes all `k` SipHash hashes in **registers**, and uses `atomicOr` operations to safely write to shared memory, enabling massive speedups while preserving the same false-positive rate.

---

## ⚡ CPU vs. GPU Approaches

### CPU Baseline
- Runs serially on each input string.
- Uses `siphash_cpu()` `k` times per string.
- Updates byte-array filter with poor memory locality.
- High DRAM latency and limited parallelism.

### CUDA Optimization
Each thread on the GPU:
1. Computes `k` SipHash values inline (entirely in registers).
2. Converts each hash to a bit index (32-bit word + bitmask).
3. Uses `atomicOr()` to update the Bloom filter safely.
4. Exits early on queries once a zero bit is detected.

Benefits:
- High SM occupancy with millions of concurrent threads.
- Fully coalesced memory access patterns.
- Drastically reduced contention and latency.

---

## 🔧 Key Optimizations

1. **Register-Only SipHash**: All hashing logic kept in registers, avoiding spills.
2. **One-Thread-Per-String**: Uniform, balanced work distribution.
3. **Fine-Grained Atomics**: Bitwise updates via `atomicOr()` on 32-bit words.
4. **Memory Coalescing**: String table and filter array accesses are coalesced.
5. **Query Early-Exit**: Membership tests stop at the first unset bit.
6. **Configurable Block Size**: Allows tuning occupancy vs. resource usage.

---

## ⏱️ Performance Timing with CUDA Events

- Uses `cudaEventRecord()` and `cudaEventElapsedTime()` around GPU kernel launches.
- Ensures reported times reflect *only kernel computation*, excluding memory transfers or host-device syncs.

---

## 🧪 Experimental Setup

### Parameters
- **Dataset size (n)**: 1 × 10⁴, 1 × 10⁶, 1 × 10⁸
- **False-positive rate (p)**: 0.01, 0.00001
- **Block sizes**: 64, 128, 256, 512 threads

### Example Run Commands
```bash
/apps/GPU_course/runScript.sh proj3_akmal.cu 1000000 0.00001 256
/apps/GPU_course/runScript.sh proj3_akmal.cu 100000000 0.001 256
```

---

## 📈 Performance Highlights

| Configuration                  | Speedup over CPU |
|-------------------------------|------------------|
| n = 1e4, p = 0.001             | 190×             |
| n = 1e6, p = 0.01              | 307–326×         |
| n = 1e6, p = 1e-5             | 327–350×         |
| n = 1e8, p = 0.001             | 316×             |

**Best block size:** 256 threads per block delivered the best performance across all datasets.

---

## 📁 Files Included

- `bloom.cu` — Fully optimized GPU Bloom Filter source code
- `Report.pdf` — Detailed explanation of optimizations, experiment results, and methodology

---

## License & Academic Integrity

© 2025 **Akmal Kurbanov**. All rights reserved.

This repository is shared **solely for learning and personal experimentation.**

* **Do not clone, copy, or submit any part of this project as coursework** or graded assignments. Doing so violates academic integrity policies and the author’s explicit prohibition against plagiarism and cheating.
* You may study the code, draw inspiration, and create your own original implementation—provided you follow your institution’s honor code.