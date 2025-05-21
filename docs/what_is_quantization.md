# Understanding Quantization in Deep Learning

## What is Quantization?

Quantization is the process of approximating continuous or high-precision values with discrete, lower-precision ones. In deep learning, it commonly refers to converting 32-bit floating-point weights and activations into lower-bit formats like int8, int4, or even binary.

> **Example:** Instead of representing a weight as 3.14159265 (float32), we can represent it as 3 (int8), accepting a small error in exchange for speed and memory savings.

This transformation drastically reduces model size and computational cost—enabling efficient inference on:
- Mobile devices
- Embedded systems
- Large-scale cloud services

## Why Quantization?

Modern large language models (LLMs) contain billions of parameters. Deploying these models at scale presents several challenges:

| Challenge | Description |
|-----------|-------------|
| **Memory Bandwidth** | Loading float32 weights for each layer consumes massive bandwidth |
| **Inference Latency** | Matrix multiplication with float32 is slower than with int8 |
| **Hardware Limitations** | Many devices (smartphones, IoT boards) cannot efficiently process float32 |

Quantization addresses these issues by reducing data precision while preserving as much model accuracy as possible, enabling real-time inference with less power and hardware cost.

## Quantization Techniques

### From Uniform Scalar Quantization to Scale-ZeroPoint

#### Uniform Scalar Quantizer (USQ)

The most basic quantizer maps floating-point values to fixed intervals (buckets):

1. Define float value range (e.g., −10 to +10)
2. Choose number of levels (e.g., 256 levels for 8 bits)
3. Calculate bucket width = (max − min) / (levels - 1) ≈ 0.078
4. Round each value to the nearest bucket center

#### Scale and Zero-Point Framework

To generalize the process and make it hardware-friendly, we use two parameters:

- **Scale (α)**: the step size between quantization levels
- **Zero-point (z)**: the integer value that corresponds to 0 in float

**Formulas:**
- Quantization: `q = round(x / scale) + zero_point`
- Dequantization: `x = scale * (q - zero_point)`

**Example:**  
For values in range [0, 6] using 8-bit unsigned integers (0-255):
- Scale = (6 - 0) / 255 = 0.0235
- Zero-point = 0 (since 0 maps to 0)
- Float 3.0 → Quantized = round(3.0 / 0.0235) = 128
- Back to float: 0.0235 * (128 - 0) ≈ 3.008

This simple method is known as affine quantization.

## Types of Quantization

### 1. Value Mapping

- **Symmetric:** Zero-point is 0; float 0 always maps to int 0
- **Asymmetric:** Zero-point can be non-zero, better for unbalanced ranges

### 2. Granularity

- **Per-tensor:** One scale/zero-point for whole weight tensor
- **Per-channel:** Different scale/zero-point for each output channel

### 3. Timing

- **Post-Training Quantization (PTQ):** Apply quantization after model training
- **Quantization-Aware Training (QAT):** Simulate quantization during training

## Methods of Quantization

### 1. Uniform Quantization
- Uses fixed step sizes (as explained above)
- Fast and easy to implement

### 2. Non-Uniform Quantization
- Step size varies (e.g., logarithmic)
- Can better preserve small values but needs lookup tables

### 3. Weight-Only Quantization
- Only compresses model weights, not activations
- Useful when activation computation is not a bottleneck

### 4. Activation Quantization
- Applies to intermediate layer outputs
- Reduces memory bandwidth

### 5. Mixed-Precision Quantization
- Combines multiple bit-widths depending on sensitivity of layers
- Often used in practice (e.g., 8-bit activations, 4-bit weights)

## Applications of Quantization


### 1. On-Device AI
Quantization enables efficient inference on phones, edge devices, and embedded systems.

### 2. Cloud Inference at Scale
Cloud models save compute and power costs by running quantized versions.

### 3. Faster Model Loading
Compressed weights reduce I/O load and load times.

### 4. Custom Hardware Acceleration
Quantized models align better with AI chips like TPUs, NPUs, and Tensor Cores.

### 5. Advanced Research
Methods like Trellis-Coded Quantization (TCQ) push boundaries of low-bit compression.

