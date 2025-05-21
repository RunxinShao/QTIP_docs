# vLLM Overview

`vLLM` is a high-performance, open-source inference engine for large language models (LLMs). It is designed to maximize throughput, minimize latency, and reduce memory usage for serving models like LLaMA, GPT-J, Falcon, and others in production and research environments.

vLLM combines system-level optimizations with deep model-aware scheduling to enable **fast, parallel, and memory-efficient inference**.

---

## Key Features of vLLM

### 1. PagedAttention

vLLM introduces **PagedAttention**, a novel memory layout that:
- Enables efficient dynamic batching
- Minimizes memory fragmentation
- Reduces the cost of KV cache management
- Supports partial KV eviction and reuse

This makes vLLM particularly suitable for serving many simultaneous requests with varying sequence lengths.

---

### 2. Kernel Fusion and GPU Efficiency

vLLM implements:
- Fused multi-head attention
- Optimized matmul and softmax kernels
- Efficient quantized inference support

These optimizations allow it to fully utilize modern GPUs with high compute-to-memory ratios (e.g., A100, H100).

---

### 3. Continuous Batching Scheduler

Unlike frameworks that batch requests synchronously, vLLM supports:
- **Asynchronous token streaming**
- **Request prioritization**
- **Fine-grained batching**

This allows it to achieve **sub-50ms latency** even with many concurrent users.

---

## Model Support and Quantization

vLLM supports models trained in Hugging Face or PyTorch format, including:
- LLaMA / LLaMA 2 / LLaMA 3
- Falcon
- OPT
- GPT-J / GPT-NeoX
- MPT

It also supports:
- **GPTQ** quantized models
- **AWQ** quantized models
- **LoRA** adapters

However, **advanced quantization methods like QTIP are not natively supported**, which is where this project contributes.

---

## vLLM Structure Overview

| Component                     | Description                                    |
|------------------------------|------------------------------------------------|
| `model_executor/`            | Loads models, processes forward pass           |
| `model_executor/layers/`     | Defines custom Linear, Attention, Norm layers  |
| `model_executor/quantization`| Handles quantization wrappers (e.g., GPTQ)     |
| `engine/`                    | Handles request scheduling and batching        |
| `_custom_ops/`               | CUDA kernels and fused ops                     |
| `quantization_config.json`   | Defines external quantization configuration    |

---

## Where QTIP Fits In

To integrate QTIP into vLLM, we must:
- Define a new quantization method class (`QTIPLinearMethod`)
- Replace `torch.nn.Linear` with a custom `BitshiftLinear` or `QuantizedLinear` class
- Hook into the loading pipeline via `load_quant_config` and `QuantizationConfig`
- Optionally implement a CUDA kernel path or use Python fallback logic

This enables vLLM to serve QTIP-compressed models with minimal accuracy loss and maximum runtime efficiency.

---

## Summary

vLLM is an inference-first engine built for LLMs at scale. Its modular architecture, performance-focused kernel design, and flexible quantization interface make it an ideal target for integrating cutting-edge compression techniques like QTIP.