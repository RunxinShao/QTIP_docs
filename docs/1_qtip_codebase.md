# QTIP codebase
This document provides a structured overview of the QTIP implementation and its decoding, performance, and integration details.

---

## 1. Core Components and Methods

### 1.1 BitshiftCodebook  
*Location: `bitshift.py`*

This module handles all aspects of quantization, trellis management, and decoding logic.

**Key Methods:**
- `recons_lut()`  
  Reconstructs the lookup table based on the current decoding mode.
  
- `recons(encoded)`  
  Reconstructs floating-point values from encoded quantized indices.
  
- `update(cost, thing)`  
  Updates the trellis state and cumulative cost during Viterbi search.
  
- `viterbi(X, overlap=None)`  
  Performs Viterbi decoding to find the optimal quantization path.
  
- `quantize_seq(X, overlap=None)`  
  Quantizes a sequence of input weights using trellis-structured optimization.
  
- `quantize(X)`  
  Main method to quantize an input block.
  
- `pack_trellis(trellis)`  
  Compresses the trellis into a bit-packed format.
  
- `unpack_trellis(packed, T)`  
  Decompresses the packed trellis back to state sequences.

---

### 1.2 BitshiftLinear  
*Location: `bitshift.py`*

This class defines a linear layer that supports fast matrix multiplication using compressed weights.

**Key Methods:**
- `get_hatW(unpacked_trellis, m, n)`  
  Reconstructs the full weight matrix from trellis codes.

- `get_hatW_kernel(trellis, m, n)`  
  CUDA kernel version of weight reconstruction.

- `cache_hatW(...)`  
  Caches reconstructed weight matrices to avoid redundant decoding during inference.

- `forward(...)`  
  Executes the forward pass using trellis-decoded weights. Supports distributed and low-level kernel options.

---

### 1.3 QuantizedLinear  
*Location: `bitshift.py`*

Defines a wrapper around `BitshiftLinear` with support for gradient checkpointing.

**Key Methods:**
- `no_ckpt_forward(input)`  
  Forward pass without checkpointing (faster inference).

- `forward(input)`  
  Main forward pass. Supports gradient checkpointing for memory efficiency.

---

## 2. Decoding Modes

QTIP supports multiple decoding modes, each balancing speed, accuracy, and memory differently.

### 2.1 Available Modes

- `'lut'`  
  Basic lookup table decoding; simple and fast.
  
- `'1mad'`  
  Uses a single multiply-add operation to approximate a Gaussian.
  
- `'2mad'`  
  Uses two MADs for better approximation.
  
- `'3inst'`  
  Uses three instructions (LCG + bitwise) for accurate decoding.
  
- `'quantlut'`  
  Decodes using a compact quantized lookup table.
  
- `'quantlut_sym'`  
  Symmetric version of quantized lookup for best accuracy.

---

## 3. Performance Optimization

### 3.1 CUDA Implementation  
QTIP includes CUDA kernels for:
- Bitshift trellis decoding
- Matrix multiplication using `BitshiftLinearKernelAG`
- Custom ops via PyTorch extensions

### 3.2 Memory Optimization Techniques

**Trellis Packing**
- `pack_trellis()`: Packs trellis into a compact bit format.
- `unpack_trellis()`: Reconstructs full trellis from packed form.

**Weight Caching**
- `cache_hatW(...)`: Caches decoded weights to avoid recomputation during repeated inference.

---

## 4. Best Practices

### Parameter Selection

| Parameter | Recommendation |
|-----------|----------------|
| `L` (trellis window) | Use 16 (default) |
| `K` (bits per weight) | Use 2–4 for good compression |
| `V` (group size) | Use 2 for most cases |
| `tlut_bits` | Should match `L` in `lut` mode |

### Mode Selection

| Use Case | Recommended Mode |
|----------|------------------|
| Highest accuracy | `quantlut_sym` |
| Fastest decoding | `1mad` or `2mad` |
| Simplicity | `lut` |

### Block Size Selection

- `td_x`, `td_y` should be powers of 2
- Common sizes: 16, 32, 64
- Larger blocks improve compression but increase computation

---

## 5. Important Notes


### Performance Considerations

- Different decode modes impact latency and precision
- Use CUDA kernels for best speed
- Block sizes affect cache efficiency

### Accuracy Considerations

- Quantization affects model accuracy
- Post-quantization fine-tuning may be necessary
- Mixed-precision training and inference can help recover accuracy

---

## Citation


@inproceedings{tseng2024qtip,
    title={{QTIP}: Quantization with Trellises and Incoherence Processing},
    author={Albert Tseng and Qingyao Sun and David Hou and Christopher De Sa},
    booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
    year={2024}
}