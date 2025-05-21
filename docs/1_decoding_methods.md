# QTIP Decoding Methods

After quantization using the bitshift trellis, QTIP decodes compressed representations into approximate floating-point weights at inference time. Unlike traditional decoding methods which rely on explicit codebooks or tables, QTIP supports multiple decoding strategies, including **lookup-free**, **hardware-efficient**, and **hybrid** techniques.

This section explains the core logic of each decoding method implemented in QTIP.

---

## 1. Lookup Table (LUT)

**Method**: Store a full codebook of all possible output values, and use the quantized code (e.g., 4-bit index) to retrieve the value.

**Example**: For a 4-bit weight encoding:

```python
lut = {
    0b0000: -2.0,
    0b0001: -1.5,
    ...
    0b1111: 2.0
}
decoded = lut[code]
```

**Characteristics**:
- Fastest decoding
- High memory usage (1 table per block/channel)
- Precomputed, no computation required at inference

LUT-based decoding is conceptually straightforward, but expensive when the codebook size is large or memory is constrained.

## 2. 1MAD: One Multiply-Add (Algorithm 1)

**Goal**: Approximate a Gaussian-distributed weight using only simple arithmetic operations — no table required.

### Steps:
1. Use a linear congruential generator (LCG) to pseudorandomize the compressed index.
2. Interpret the 32-bit LCG result as four 8-bit integers.
3. Sum the 4 components — the result resembles a Gaussian (by central limit theorem).
4. Apply a scale and shift to normalize.

**Pseudocode**:
```python
x = (a * index + b) % 2^32
x = (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + ((x >> 24) & 255)
x = (x - 510) / 147.8  # Normalize to N(0,1)
```

**Parameters**:
- a = 34038481
- b = 76625530

**Features**:
- Only 2–3 instructions per weight
- Fully LUT-free
- Produces approximately Gaussian weights

## 3. 3INST: Three-Instruction Decode (Algorithm 2)

**Goal**: Improve decoding resolution and Gaussianity using a small number of hardware-friendly operations.

### Steps:
1. Run an LCG to generate a 32-bit pseudorandom number X.
2. XOR bits of X with a carefully constructed float16 "template" (m) to inject entropy into mantissa, exponent, and sign.
3. Reinterpret the result as two float16s, sum them.

**Pseudocode (simplified)**:
```python
X = (a * index + b) % 2^32
m = 0.922 (in float16)
# Pack and XOR with both halves of X
x1 = reinterpret((X & 0xFFFF) ^ mask(m))
x2 = reinterpret((X >> 16) ^ mask(m))
decoded = x1 + x2
```

**Parameters**:
- a = 89226354
- b = 64248484
- m = 0.922 (float16)

**Notes**:
- Results approximate the sum of two mirrored exponential variables → close to Gaussian
- Achieves higher entropy than 1MAD
- Can be implemented in 3 ALU instructions on modern GPUs

## 4. HYB: Hybrid Decode (Algorithm 3)

**Goal**: Combine the flexibility of a small LUT with the randomness of a hash function for high-quality decoding.

### Core Idea:
- Use a hashed version of the index to look up a 2D vector from a small shared LUT.
- Then apply a sign flip to increase representational range.

### Steps:
1. Compute a fast hash:
   ```python
   x = x * x + x
   ```
2. Use bits 14 - Q + 1 to 14 as index into a 2D LUT of size 2^Q × 2.
3. Flip the sign of the second float in the 2D vector if bit 15 is 1.

**Pseudocode**:
```python
x = (index * index + index) % 2^32
i = (x >> (15 - Q)) & (2^Q - 1)
v = LUT[i]  # returns [v0, v1]
if x & (1 << 15):
    v[1] = -v[1]
decoded = v
```

**Notes**:
- Amortized ~2 instructions per weight
- LUT can be fine-tuned offline via K-means
- Fits in L1/shared memory on GPUs
- Supports quantizing V=2 weight groups jointly

## 5. Configuration in QTIP

Each decoding method can be selected with a simple config field in `quantize_config.json`:

```json
{
  "decode_mode": "lut"  // or "1mad", "3inst", "hyb"
}
```

Changing this mode automatically triggers the corresponding decoding logic during model inference.

## Comparison of Decoding Methods

| Method | Memory Usage | Computation | Quality | Best For |
|--------|-------------|-------------|---------|----------|
| LUT | High | Minimal | Exact | Small models, memory-rich environments |
| 1MAD | None | Low (2-3 ops) | Good | Memory-constrained, compute-rich environments |
| 3INST | None | Medium (3 ops) | Better | Best quality/performance trade-off |
| HYB | Low | Low-Medium | Best | Production deployments requiring high accuracy |

## Implementation Considerations

When implementing QTIP decoding, consider these factors:

1. **Hardware Target**: Different hardware may favor different decoding strategies:
   - GPUs: HYB or 3INST leverage parallel compute well
   - CPUs: 1MAD can be very efficient with SIMD
   - Mobile: LUT for small models, 1MAD for larger ones

2. **Memory Budget**: If memory is severely constrained, prefer compute-based methods (1MAD, 3INST)

3. **Accuracy Requirements**: For highest quality, use HYB with an optimized LUT

4. **Model Size**: The impact of decoding method grows with model size