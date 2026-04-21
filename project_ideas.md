# Next Project Ideas — Compression & Inference Focus

These ideas are grounded in the infrastructure side of AI — compression, quantization, inference optimization, and serving systems. Ordered roughly from "build and learn fast" to "research-grade".

---

## 1. Quantization Explorer — Interactive W4A16 / W8A8 Benchmark Harness

**What:** A CLI + small web UI that takes any HuggingFace model, applies different quantization schemes (GPTQ, AWQ, bitsandbytes INT8, FP8), and produces a structured comparison report: perplexity delta, throughput (tokens/s), memory footprint, and latency breakdown.

**Why it's interesting:** Quantization is easy to read about and hard to reason about concretely. The comparison surface (accuracy vs. speed vs. VRAM) has no single right answer — it's task- and hardware-dependent. Building this forces you to understand the calibration dataset problem, the weight vs. activation distinction, and why outlier channels break naive INT8.

**Core challenge:** Getting apples-to-apples latency measurements requires careful CUDA synchronization, warm-up runs, and accounting for KV cache differently across schemes.

**Tech:** `transformers`, `auto-gptq`, `autoawq`, `bitsandbytes`, `lm-eval-harness` for perplexity.

---

## 2. KV Cache Compression Library

**What:** A drop-in module that wraps any HuggingFace attention layer and applies one of three KV cache reduction strategies: (a) sliding window eviction, (b) H2O (Heavy-Hitter Oracle) token scoring, (c) quantized KV cache (INT8 keys/values). Expose a unified API so strategies are swappable.

**Why it's interesting:** KV cache is the dominant memory consumer during inference. Most engineers know this is "the problem" but few have touched the actual attention score accumulation that makes some token slots evictable. This project lives at the intersection of algorithm and memory layout.

**Core challenge:** Making eviction decisions without breaking positional encoding (especially RoPE). The naive sliding window breaks long-range dependencies in ways that take careful eval to detect.

**Tech:** PyTorch hooks into attention layers, `transformers` model internals.

---

## 3. Speculative Decoding Testbed

**What:** Implement speculative decoding from scratch — draft model proposes K tokens, target model verifies in a single forward pass, accepted tokens committed. Build a harness that measures acceptance rate as a function of: draft model size, temperature delta, sequence length, and prompt domain.

**Why it's interesting:** Speculative decoding is one of the few latency wins that doesn't trade accuracy — but its benefit is highly domain-dependent. A math reasoning prompt might have 20% acceptance rate; a code completion prompt might have 70%. Understanding why is the entire project.

**Core challenge:** The tree-based batched verification (SpecInfer-style) vs. the simple linear proposal — implementing both to see where the complexity pays off.

**Tech:** Pure PyTorch, any two models from the same family (Llama 3.2 1B + 8B, for example).

---

## 4. Sparse Attention Profiler

**What:** A profiling tool that runs an LLM on a dataset, records the actual attention sparsity pattern (which heads attend to which positions, across layers and sequence positions), and produces a report showing: which layers could be safely replaced with sliding window attention, which heads are consistently attending to early tokens (induction heads), and what the theoretical FLOP savings would be.

**Why it's interesting:** Most sparse attention papers report aggregate sparsity. Real models have highly heterogeneous attention patterns — some heads are always sparse, some are dense, and the mixture changes with context. This project teaches you to read a model's actual behavior rather than trusting theoretical guarantees.

**Core challenge:** Efficiently capturing attention weights without running out of memory (use hooks, don't materialize the full NxN matrix for every layer simultaneously).

**Tech:** HuggingFace `transformers` with attention weight output, matplotlib for pattern visualization.

---

## 5. Continuous Batching Inference Server (from scratch)

**What:** A minimal inference server that implements continuous batching — requests join a running batch mid-generation rather than waiting for the current batch to finish. No vLLM dependency. Build the iteration-level scheduler, the paged KV cache block manager, and the token stream API.

**Why it's interesting:** This is the core of every production inference system (vLLM, TGI, TensorRT-LLM). Understanding why continuous batching requires paged attention (you can't know how long a sequence will be when it starts) is one of those ideas that clicks once and reframes everything else.

**Core challenge:** The block allocator — allocating and freeing fixed-size KV blocks correctly under preemption, without fragmentation, while keeping GPU memory utilization high.

**Tech:** PyTorch, asyncio for request handling. Start with a single GPU.

---

## 6. Pruning + Distillation Pipeline for a Specific Task

**What:** Take a 7B model, prune it to ~3B using structured pruning (remove attention heads and FFN dimensions that contribute least to a target task), then distill the full 7B as a teacher to recover accuracy on that task. Measure the resulting model vs. a purpose-trained 3B.

**Why it's interesting:** Structured pruning is messy and underexplored compared to quantization — most engineers default to quantization because pruning requires retraining and task-specific calibration. Understanding why magnitude-based pruning fails and why gradient-based methods are necessary is the lesson.

**Core challenge:** Identifying which heads to prune without evaluating all 2^N subsets. Approximation methods (Taylor expansion, gradient magnitude) are the interesting engineering here.

**Tech:** `transformers`, `torch.nn.utils.prune`, a small task-specific dataset (SQuAD, GSM8K subset, etc.).

---

## 7. GGUF / GGML Format Deep Dive — Custom Loader

**What:** Write a GGUF file parser and minimal model loader from scratch in Python (or Rust if you want the performance). Load a Llama-compatible GGUF model, inspect its quantization metadata, and run a single forward pass using dequantized weights. No llama.cpp dependency.

**Why it's interesting:** GGUF is the de facto format for quantized LLMs on consumer hardware. Understanding the file format — how Q4_K_M differs from Q4_0 at the bit level, how superblock quantization works, why group size matters — grounds every "what quantization should I use" conversation in actual structure rather than benchmarks.

**Core challenge:** Correctly dequantizing Q4_K_M (k-quants with block-wise scales and minimums) without reference to llama.cpp internals.

**Tech:** Python `struct` for binary parsing, numpy for dequantization math. The GGUF spec is publicly documented.

---

## Common Thread

All seven projects share a pattern: take something you've read about as a concept (quantization, speculative decoding, KV cache) and force it to become something you've debugged, measured, and explained through its failure modes. That's the gap between "I understand inference engineering" and "I understand what actually happens in production."
