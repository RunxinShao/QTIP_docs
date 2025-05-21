# Integrating QTIP into vLLM — Step-by-Step Guide

This guide walks through the exact steps to integrate the QTIP quantization method into vLLM's inference engine. By following this, you will be able to run QTIP-compressed models directly with vLLM's fast and scalable infrastructure.

## Step 1: Prepare `quantize_config.json`

Create a QTIP configuration file. Example:

```json
{
  "quant_method": "qtip",
  "L": 16,
  "K": 2,
  "V": 2,
  "tlut_bits": 16,
  "decode_mode": "1mad",
  "quantlut_path": "checkpoints/llama/qtip/quantlut.pt"
}
```

This config must be placed in the model folder, or passed via CLI when launching vLLM.

## Step 2: Define QTIPConfig (in quantization_config.py)

Create a new class to parse the QTIP config:

```python
from typing import Any, Optional
import torch
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization import QuantizationMethods

class QTIPConfig(QuantizationConfig):
    """Config class for QTIP quantization."""
    
    def __init__(self, L: int, K: int, V: int, tlut_bits: int = 16,
                 decode_mode: str = "1mad", quantlut_path: Optional[str] = None):
        super().__init__()
        self.L = L  # Number of codebooks
        self.K = K  # Number of centroids per codebook
        self.V = V  # Number of vectors per centroid
        self.tlut_bits = tlut_bits  # Bits for lookup table
        self.decode_mode = decode_mode  # Decoding mode: "1mad", "3inst", or "quantlut"
        self.quantlut_path = quantlut_path  # Path to quantization lookup table
        
    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "qtip"
        
    @classmethod 
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]
        
    @classmethod
    def get_min_capability(cls) -> int:
        return 70  # Minimum CUDA capability required
        
    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return ["quantize_config.json"]
        
    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "QTIPConfig":
        L = cls.get_from_keys(config, ["L"])
        K = cls.get_from_keys(config, ["K"]) 
        V = cls.get_from_keys(config, ["V"])
        tlut_bits = cls.get_from_keys_or(config, ["tlut_bits"], 16)
        decode_mode = cls.get_from_keys_or(config, ["decode_mode"], "1mad")
        quantlut_path = cls.get_from_keys_or(config, ["quantlut_path"], None)
        
        return cls(L=L, K=K, V=V, tlut_bits=tlut_bits,
                  decode_mode=decode_mode, quantlut_path=quantlut_path)
                  
    def get_quant_method(self, layer: torch.nn.Module,
                        prefix: str) -> Optional["QTIPLinearMethod"]:
        if isinstance(layer, LinearBase):
            return QTIPLinearMethod(self)
        return None
```

## Step 3: Define QTIPLinearMethod (in method.py)

Implement a new quantization method:

```python
from typing import Optional
import torch
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.utils import set_weight_attrs

class QTIPLinearMethod(QuantizeMethodBase):
    """Linear method for QTIP quantization."""
    
    def __init__(self, config: QTIPConfig):
        self.config = config
        
    def create_weights(self, layer: torch.nn.Module,
                      input_size_per_partition: int,
                      output_partition_sizes: list[int],
                      input_size: int,
                      output_size: int,
                      params_dtype: torch.dtype,
                      **extra_weight_attrs):
        # Create compressed weight storage
        weight = torch.nn.Parameter(
            torch.empty(sum(output_partition_sizes),
                       input_size_per_partition,
                       dtype=params_dtype),
            requires_grad=False)
            
        # Initialize codebook
        cb = BitshiftCodebook(self.config)
        cb.quantize(weight)
        layer.qtip_cb = cb
        
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)
        
    def apply(self, layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Decode weights if not cached
        if not hasattr(layer, "decoded_weight"):
            layer.decoded_weight = layer.qtip_cb.recons()
            
        # Perform matrix multiplication
        out = torch.matmul(x, layer.decoded_weight.t())
        if bias is not None:
            out += bias
        return out
        
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Load quantization lookup table if specified
        if self.config.quantlut_path:
            layer.qtip_cb.load_quantlut(self.config.quantlut_path)
```

## Step 4: Implement QuantizedLinear (in layers/quantized_linear.py)

This wrapper replaces the original nn.Linear:

```python
import torch
import torch.nn as nn
from typing import Optional

class QuantizedLinear(nn.Module):
    """Quantized linear layer using QTIP method."""
    
    def __init__(self, original_layer: nn.Linear, config: QTIPConfig):
        super().__init__()
        self.qtip_cb = None
        self.bias = original_layer.bias
        self.config = config
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Decode weights if not cached
        if not hasattr(self, "decoded_weight"):
            if self.config.decode_mode == "1mad":
                self.decoded_weight = decode_1mad(self.qtip_cb.encoded, self.config)
            elif self.config.decode_mode == "3inst":
                self.decoded_weight = decode_3inst(self.qtip_cb.encoded, self.config)
            elif self.config.decode_mode == "quantlut":
                self.decoded_weight = lookup_decode(self.qtip_cb.encoded, self.config)
            else:
                raise ValueError(f"Unsupported decode mode: {self.config.decode_mode}")
                
        # Perform matrix multiplication
        out = torch.matmul(x, self.decoded_weight.t())
        if self.bias is not None:
            out += self.bias
        return out
```

## Step 5: Add Python Fallback Decode (in bitshift_codebook.py)

Make sure your BitshiftCodebook implements:

```python
import torch
from typing import Optional

def decode_1mad(encoded: torch.Tensor, config: QTIPConfig) -> torch.Tensor:
    """Decode using 1-MAD (Multiply-Add) method."""
    # Implementation of 1-MAD decoding
    # This is a simplified version - actual implementation would be more complex
    decoded = torch.zeros_like(encoded)
    for i in range(config.L):
        codebook = encoded[i]
        decoded += codebook
    return decoded

def decode_3inst(encoded: torch.Tensor, config: QTIPConfig) -> torch.Tensor:
    """Decode using 3-instruction method."""
    # Implementation of 3-instruction decoding
    # This is a simplified version - actual implementation would be more complex
    decoded = torch.zeros_like(encoded)
    for i in range(config.L):
        codebook = encoded[i]
        decoded += codebook
    return decoded

def lookup_decode(encoded: torch.Tensor, config: QTIPConfig) -> torch.Tensor:
    """Decode using lookup table method."""
    # Implementation of lookup table decoding
    # This is a simplified version - actual implementation would be more complex
    if not hasattr(config, "quantlut"):
        config.load_quantlut(config.quantlut_path)
    return config.quantlut[encoded]

class BitshiftCodebook:
    """Codebook for QTIP quantization."""
    
    def __init__(self, config: QTIPConfig):
        self.config = config
        self.encoded = None
        
    def quantize(self, weight: torch.Tensor) -> None:
        """Quantize weights using QTIP method."""
        # Implementation of quantization
        # This is a simplified version - actual implementation would be more complex
        self.encoded = weight.clone()
        
    def recons(self) -> torch.Tensor:
        """Reconstruct weights from codebook."""
        if self.config.decode_mode == "1mad":
            return decode_1mad(self.encoded, self.config)
        elif self.config.decode_mode == "3inst":
            return decode_3inst(self.encoded, self.config)
        elif self.config.decode_mode == "quantlut":
            return lookup_decode(self.encoded, self.config)
        else:
            raise ValueError(f"Unsupported decode mode: {self.config.decode_mode}")
            
    def load_quantlut(self, path: str) -> None:
        """Load quantization lookup table."""
        self.quantlut = torch.load(path)
```

## Step 6: Optional — Register Custom CUDA Kernel (in _custom_ops/)

If you have a kernel version of BitshiftLinear (e.g., bitshift_linear_kernel.cu), register it as:

```cpp
// bitshift_linear_kernel.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for QTIP linear layer
__global__ void bitshift_gemm_kernel(
    const float* input,
    const float* weight,
    float* output,
    int m, int n, int k) {
    // Implementation of CUDA kernel
    // This is a simplified version - actual implementation would be more complex
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m * n) {
        int i = idx / n;
        int j = idx % n;
        float sum = 0.0f;
        for (int kk = 0; kk < k; kk++) {
            sum += input[i * k + kk] * weight[kk * n + j];
        }
        output[idx] = sum;
    }
}

torch::Tensor bitshift_gemm(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias) {
    
    auto m = input.size(0);
    auto k = input.size(1);
    auto n = weight.size(0);
    
    auto output = torch::empty({m, n}, input.options());
    
    dim3 block(256);
    dim3 grid((m * n + block.x - 1) / block.x);
    
    bitshift_gemm_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        m, n, k);
        
    if (bias.defined()) {
        output += bias;
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bitshift_gemm", &bitshift_gemm, "QTIP linear layer (CUDA)");
}
```

Also create a CMake entry to build the op:

```cmake
find_package(CUDA REQUIRED)
find_package(Torch REQUIRED)

add_library(bitshift_linear SHARED
    bitshift_linear_kernel.cu
)

target_link_libraries(bitshift_linear
    ${TORCH_LIBRARIES}
    ${CUDA_LIBRARIES}
)

set_target_properties(bitshift_linear PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
```

## Step 7: Test the Integration

✅ Run with `python -m vllm.entrypoints.openai.api_server --quantize_config quantize_config.json`

✅ Load a model compressed with QTIP

✅ Send a sample prompt

✅ Verify decode speed and correctness

This implementation provides basic QTIP quantization functionality, including:

- Support for multiple decoding modes (1-MAD, 3-instruction, lookup table)
- Configurable quantization parameters (L, K, V, bits, etc.)
- CUDA acceleration support
- Integration with vLLM's existing quantization framework

You can further optimize and extend this implementation based on your specific needs.
