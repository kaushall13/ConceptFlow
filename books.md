# AI Infrastructure Mastery — Reading List

Sequenced foundation-first, domain-specific second, cutting-edge third. Each tier feeds the next. The reader has already finished an LLM inference systems book, so foundational ML theory is assumed — the list starts at computer architecture and works up to compiler backends and serving systems.

---

## Tier 1 — Computer Architecture Foundations

These are not optional prerequisites. Without them, GPU memory hierarchies, SIMD datapaths, and hardware-software co-design are just vocabulary, not understanding.

---

**1. Computer Organization and Design: RISC-V Edition (2nd ed.)**
*David A. Patterson & John L. Hennessy — Morgan Kaufmann, 2020*

The canonical undergraduate computer architecture text. Covers the full hardware-software interface: instruction sets, pipelining, memory hierarchies (caches, DRAM, virtual memory), I/O, and basic parallelism. The RISC-V edition strips away x86 noise and teaches concepts cleanly.

**Why:** Every GPU optimization decision — bandwidth saturation, cache blocking, memory coalescing — is incomprehensible without a working mental model of memory hierarchies and pipeline stages. This is the substrate.

**Difficulty:** Undergraduate. Accessible but dense.
**Gain:** The hardware model that every other book in this list assumes.

---

**2. Computer Architecture: A Quantitative Approach (6th ed.)**
*John L. Hennessy & David A. Patterson — Morgan Kaufmann, 2017*

The graduate-level sequel. Goes deep on ILP, out-of-order execution, vector/SIMD/GPU architectures (Chapter 4), memory hierarchy design, multiprocessors, and domain-specific architectures. The data-level parallelism chapter is a direct bridge to GPU programming.

**Why:** This is where you get the analytical vocabulary — throughput vs. latency trade-offs, Amdahl's Law applied to accelerators, roofline models, DRAM bandwidth ceilings. The domain-specific architectures chapter is directly relevant to TPUs and AI accelerators.

**Difficulty:** Graduate. Requires book 1 or equivalent.
**Gain:** The ability to reason quantitatively about hardware bottlenecks rather than just observe them empirically.

---

## Tier 2 — GPU Architecture & CUDA Programming

The hardware substrate for all modern AI inference. Go deep here before touching any inference optimization library.

---

**3. Programming Massively Parallel Processors: A Hands-on Approach (4th/5th ed.)**
*Wen-mei W. Hwu, David B. Kirk & Izzat El Hajj — Elsevier, 2022/2024*

The definitive CUDA textbook, written by the people who built CUDA at NVIDIA and Illinois. Covers the GPU execution model (warps, thread blocks, SMs), memory hierarchy (global, shared, constant, texture), parallel patterns (reduction, scan, histogram, stencil), and performance optimization. Updated editions include tensor cores and cuDNN.

**Why:** vLLM's PagedAttention, FlashAttention's tiling strategy, quantized kernel design — all require intimate knowledge of how GPU memory works. This book builds that knowledge systematically, not by osmosis.

**Difficulty:** Intermediate. Requires C/C++ familiarity.
**Gain:** The ability to read CUDA kernel code, understand what a memory access pattern costs, and reason about occupancy, bank conflicts, and warp divergence.

---

**4. Professional CUDA C Programming**
*John Cheng, Max Grossman & Ty McKercher — Wrox/Wiley, 2014*

Complements book 3 by going deeper on the execution model, stream concurrency, multi-GPU programming, and profiling/performance tuning methodology. More oriented toward production-grade optimization than pedagogy.

**Why:** This is where CUDA moves from understanding to craft. The profiling and performance tuning chapters are directly applicable to kernel optimization work in inference systems.

**Difficulty:** Intermediate-to-advanced. Read after book 3.
**Gain:** Multi-GPU coordination patterns, CUDA streams and events, and a systematic profiling workflow.

---

## Tier 3 — Hardware-Software Co-design for Neural Networks

The bridge between chip architecture and ML system design. This is the intellectual core of the AI infrastructure discipline.

---

**5. Efficient Processing of Deep Neural Networks**
*Vivienne Sze, Yu-Hsin Chen, Tien-Ju Yang & Joel S. Emer — Morgan & Claypool, 2020*

Written by the MIT group behind the Eyeriss accelerator and an NVIDIA research scientist. Covers DNN computational requirements, hardware dataflow architectures (weight stationary, output stationary, row stationary), quantization at the hardware level, sparsity exploitation, and the design space for custom accelerators.

**Why:** This is the book that explains *why* certain quantization schemes work, why structured pruning matters for hardware, and how hardware accelerators (TPUs, Eyeriss) are designed around the math of convolutions and attention. Every AI hardware decision traces back to concepts here.

**Difficulty:** Advanced. Requires both ML fundamentals and architecture background.
**Gain:** A rigorous framework for hardware-software co-design. The ability to look at a new chip spec and understand its trade-offs.

---

**6. Machine Learning Systems: Principles and Practices of Engineering Artificially Intelligent Systems**
*Vijay Janapa Reddi (Harvard) — MIT Press / mlsysbook.ai (free PDF)*

The emerging standard textbook for the ML systems field, developed through Harvard's CS249r course. Covers data engineering, hardware-aware training, inference acceleration, model optimization (quantization, pruning, distillation), deployment, and the broader systems context. Open-source and continuously updated.

**Why:** The most comprehensive single-volume treatment of the ML systems stack that exists as a proper textbook. Connects hardware to software to deployment in a structured, teachable way.

**Difficulty:** Intermediate-to-advanced.
**Gain:** A unified mental model of the entire ML systems stack, from chip to serving endpoint.

---

## Tier 4 — Model Compression (Quantization, Pruning, Distillation)

No standalone book covers this field end-to-end at book length — the canonical treatments live in book 5 above and in survey papers. This book brackets the topic from the embedded constraint angle, which makes every optimization decision unavoidable.

---

**7. TinyML: Machine Learning with TensorFlow Lite on Arduino and Ultra-Low-Power Microcontrollers**
*Pete Warden & Daniel Situnayake — O'Reilly, 2020*

Written by the creator of TensorFlow Lite (Warden). Covers the complete compression and deployment pipeline: quantization-aware training, post-training quantization, pruning, and running inference on microcontrollers with 256KB of RAM.

**Why:** The embedded constraint is extreme, which makes every optimization decision explicit and unavoidable. The concepts transfer directly to understanding INT8/INT4 inference on GPUs — the mechanisms are the same, the stakes are just rendered more starkly.

**Difficulty:** Accessible. The most readable compression-focused book available.
**Gain:** Practical fluency with quantization and model optimization pipelines. A concrete understanding of what compression actually costs and gains.

---

## Tier 5 — AI Systems Performance Engineering & Serving

The operational layer: where compressed, optimized models actually get deployed.

---

**8. AI Systems Performance Engineering: Optimizing Model Training and Inference Workloads with GPUs, CUDA, and PyTorch**
*Chris Fregly — O'Reilly, December 2025*

1,058 pages covering GPU performance optimization end-to-end: CUDA kernel optimization, storage I/O, PyTorch compiler stack (TorchDynamo, AOT Autograd), OpenAI Triton, XLA backends, multi-node distributed training and inference.

**Why:** The most comprehensive single-volume treatment of AI performance engineering at the system level. Covers the PyTorch/Triton/XLA compiler stack in depth — the material that's hardest to find organized coherently elsewhere. Directly actionable for inference infrastructure work.

**Difficulty:** Advanced. Assumes CUDA familiarity and production ML experience.
**Gain:** End-to-end optimization methodology from CUDA kernels through compiler backends to multi-node serving.

---

**9. Hands-On LLM Serving and Optimization**
*Chi Wang & Peiheng Hu — O'Reilly, 2025*

Covers LLM serving infrastructure: vLLM and NVIDIA Triton, request batching, model compression for serving, long-context serving, and multi-model design patterns.

**Why:** The closest thing to a production manual for LLM serving infrastructure. Complements the inference systems book already read by going deeper on the serving stack rather than the core inference algorithms.

**Difficulty:** Intermediate-to-advanced.
**Gain:** Practical serving systems architecture: batching strategies, hardware provisioning patterns, and optimization levers in production deployments.

---

**10. Designing Machine Learning Systems: An Iterative Process for Production-Ready Applications**
*Chip Huyen — O'Reilly, 2022*

Covers the full production ML lifecycle: data pipelines, training infrastructure, model deployment, monitoring, and system architecture. Strong on systems trade-offs (latency vs. throughput, batch vs. online, retraining cadences) that govern serving infrastructure decisions.

**Why:** The inference systems book covers the engine; this book covers the vehicle it's installed in. Understanding deployment patterns, serving architectures, and monitoring systems is necessary for infrastructure that works in practice, not just benchmarks.

**Difficulty:** Intermediate. Accessible to any practicing engineer.
**Gain:** The systems engineering mindset for production ML.

---

## Tier 6 — Distributed Systems Foundations

Distributed inference (tensor parallelism, pipeline parallelism, disaggregated prefill/decode) requires solid distributed systems fundamentals.

---

**11. Designing Data-Intensive Applications**
*Martin Kleppmann — O'Reilly, 2017*

The standard text on distributed systems engineering for practitioners. Covers replication, partitioning, consistency models, consensus, stream processing, and fault tolerance.

**Why:** Distributed LLM inference is distributed systems. Tensor parallelism introduces synchronization overhead. Pipeline parallelism introduces pipeline bubbles and scheduling complexity. Disaggregated serving introduces network consistency challenges. Kleppmann's framework applies directly.

**Difficulty:** Intermediate. Systems experience helps.
**Gain:** The conceptual vocabulary for distributed systems — replication lag, linearizability, consensus, write-ahead logs — that comes up constantly in distributed inference engineering.

---

## Reading Order Summary

| # | Book | Topic |
|---|------|-------|
| 1 | Patterson & Hennessy — *Computer Organization and Design* (RISC-V) | Hardware foundations |
| 2 | Hennessy & Patterson — *Computer Architecture: A Quantitative Approach* (6th) | Quantitative architecture, GPU chapter |
| 3 | Hwu, Kirk & El Hajj — *Programming Massively Parallel Processors* (4th/5th) | CUDA & GPU programming |
| 4 | Cheng et al. — *Professional CUDA C Programming* | CUDA optimization craft |
| 5 | Sze, Chen, Yang & Emer — *Efficient Processing of Deep Neural Networks* | Hardware-software co-design, compression |
| 6 | Reddi — *Machine Learning Systems* (mlsysbook.ai, free) | Full ML systems stack |
| 7 | Warden & Situnayake — *TinyML* | Quantization & compression pipeline |
| 8 | Fregly — *AI Systems Performance Engineering* | GPU perf, Triton, XLA, serving |
| 9 | Wang & Hu — *Hands-On LLM Serving and Optimization* | LLM serving infrastructure |
| 10 | Huyen — *Designing Machine Learning Systems* | Production ML system architecture |
| 11 | Kleppmann — *Designing Data-Intensive Applications* | Distributed systems foundations |

---

## Note on Compiler Backends (MLIR, XLA, Triton)

No book-length pedagogical treatment of MLIR, XLA, or Triton currently exists. The authoritative material lives in:
- The Triton paper (Tillet et al., 2019)
- The MLIR documentation and tutorials
- The compiler chapters of Fregly's *AI Systems Performance Engineering* (book 8)
- The ML compiler chapter in Reddi's *Machine Learning Systems* (book 6)

These are the best entry points available in book form until a dedicated text emerges.
