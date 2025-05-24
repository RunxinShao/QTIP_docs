# Integrating QTIP into vLLM — Step-by-Step Guide

This guide walks through the exact steps to integrate the QTIP (Quantization with Trellis Index Packing) quantization method into vLLM's inference engine. By following this, you will be able to run QTIP-compressed models directly with vLLM's fast and scalable infrastructure.

current code branch here: https://github.com/RunxinShao/vllm/tree/add-qtip-inference

## Step 1: Prepare `quantize_config.json`

Create a QTIP configuration file. Example:

```json
{
  "quant_method": "qtip",
  "td_x": 16,
  "td_y": 16,
  "L": 16,
  "K": 2,
  "V": 1,
  "tlut_bits": 16,
  "decode_mode": "1mad",
  "scale": 32.0
}
```

This config must be placed in the model folder, or passed via CLI when launching vLLM.

## Step 2: Define QTIPConfig and QTIPLinearMethod (in vllm/model_executor/layers/quantization/qtip.py)

Create a new class to parse the QTIP config:

```python
# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import torch
from torch import nn

from vllm._custom_ops import bitshift_codebook, bitshift_gemm
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.parameter import PackedvLLMParameter


class QTIPConfig(QuantizationConfig):
    """
    QTIP (Quantization with Trellis Index Packing) static quantization 
    configuration class
    """

    def __init__(self,
                 td_x: int,
                 td_y: int,
                 L: int,
                 K: int,
                 V: int,
                 tlut_bits: int,
                 decode_mode: str,
                 scale: float = 32.0):
        self.td_x = td_x
        self.td_y = td_y
        self.L = L
        self.K = K
        self.V = V
        self.tlut_bits = tlut_bits
        self.decode_mode = decode_mode
        self.scale = scale
        # Number of elements in each block
        self.pack_factor = td_x * td_y

    def __repr__(self) -> str:
        return (
            f"QTIPConfig(td_x={self.td_x}, td_y={self.td_y},"
            f" L={self.L}, K={self.K}, V={self.V},"
            f" tlut_bits={self.tlut_bits}, decode_mode='{self.decode_mode}',"
            f" scale={self.scale})")

    @classmethod
    def get_name(cls) -> str:
        return "qtip"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 60

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: dict) -> "QTIPConfig":
        td_x = cls.get_from_keys(config, ["td_x"])
        td_y = cls.get_from_keys(config, ["td_y"])
        L = cls.get_from_keys(config, ["L"])
        K = cls.get_from_keys(config, ["K"])
        V = cls.get_from_keys(config, ["V"])
        tlut_bits = cls.get_from_keys(config, ["tlut_bits"])
        decode_mode = cls.get_from_keys(config, ["decode_mode"])
        scale = cls.get_from_keys_or(config, ["scale"], default=32.0)
        return cls(td_x, td_y, L, K, V, tlut_bits, decode_mode, scale)

    def get_quant_method(self, layer: nn.Module,
                         prefix: str) -> "QTIPLinearMethod":
        return QTIPLinearMethod(self)


class QTIPLinearMethod(LinearMethodBase):
    """
    QTIP linear layer quantization method
    """

    def __init__(self, quant_config: QTIPConfig):
        self.cfg = quant_config
        # Build lookup table (codebook)
        self.cb = bitshift_codebook(L=self.cfg.L,
                                    K=self.cfg.K,
                                    V=self.cfg.V,
                                    tlut_bits=self.cfg.tlut_bits,
                                    decode_mode=self.cfg.decode_mode)
        self.scale = self.cfg.scale

    def create_weights(self, layer: nn.Module, input_size_per_partition: int,
                       output_partition_sizes: list[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):
        output_size_per_partition = sum(output_partition_sizes)
        pack_factor = self.cfg.pack_factor
        assert input_size_per_partition % pack_factor == 0, \
            "input size must be multiple of pack_factor"
        rows = input_size_per_partition // pack_factor

        qweight = PackedvLLMParameter(
            data=torch.empty(rows,
                             output_size_per_partition,
                             dtype=torch.int32),
            input_dim=0,
            output_dim=1,
            packed_dim=0,
            packed_factor=pack_factor,
            weight_loader=lambda p, w: p.data.copy_(w))
        layer.register_parameter("qweight", qweight)

        # Register SU and SV
        layer.register_buffer(
            "SU", torch.ones(input_size_per_partition, dtype=params_dtype))
        layer.register_buffer(
            "SV", torch.ones(output_size_per_partition, dtype=torch.float32))

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        """
        Unpack loaded quantized indices and restore to int32 index matrix
        """
        packed = layer.qweight.data
        unpacked = self.cb.unpack_trellis(packed, self.cfg.pack_factor)
        layer.qweight.data = unpacked.to(torch.int32)

    def apply(self,
              layer: nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        output = bitshift_gemm(input=x,
                               trellis=layer.qweight,
                               codebook=self.cb,
                               td_x=self.cfg.td_x,
                               td_y=self.cfg.td_y,
                               scale=self.scale,
                               SU=layer.SU,
                               SV=layer.SV)
        if bias is not None:
            output = output + bias
        return output


```

## Step 4: Define the Bitshift Codebook (in _custom_ops.py)

Implement the `bitshift_codebook` class that handles decoding quantized weights:

```python

import numpy as np
import torch.nn as nn

class bitshift_codebook(nn.Module):
    def __init__(self,
                 L: int,
                 K: int,
                 V: int,
                 tlut_bits: int,
                 decode_mode: str,
                 tlut: torch.Tensor = None):
        super().__init__()
        self.L = L
        self.K = K
        self.V = V
        self.tlut_bits = tlut_bits
        self.decode_mode = decode_mode

        levels = 1 << L

        if decode_mode == 'lut':
            # Use externally provided tlut or random initialization
            if tlut is None:
                assert tlut_bits == L
                tbl = torch.randn(levels, V, dtype=torch.float16)
            else:
                tbl = tlut
            # Store as [V, levels]
            self.register_buffer('lut', tbl.T.contiguous())

        elif decode_mode == '1mad':
            assert V == 1
            vals = decode_1mad(torch.arange(levels, device='cpu'))
            self.register_buffer('lut', vals.unsqueeze(0).to(torch.float16))

        elif decode_mode == '2mad':
            assert V == 1
            vals = decode_2mad(torch.arange(levels, device='cpu'))
            self.register_buffer('lut', vals.unsqueeze(0).to(torch.float16))

        elif decode_mode == '3inst':
            assert V == 1
            vals = decode_3inst(torch.arange(levels, device='cpu'))
            self.register_buffer('lut', vals.unsqueeze(0).to(torch.float16))

        elif decode_mode == 'quantlut':
            assert tlut is not None
            tbl = quantlut(tlut, L, tlut_bits)
            self.register_buffer('lut', tbl)

        elif decode_mode == 'quantlut_sym':
            assert tlut is not None
            tbl = quantlut_sym(tlut, L, tlut_bits)
            self.register_buffer('lut', tbl)

        else:
            raise ValueError(f"Unsupported decode_mode: {decode_mode!r}")

    def recons(self, encoded: torch.Tensor) -> torch.Tensor:
        """
        encoded: LongTensor([num_blocks, td_x*td_y])
        returns: FloatTensor([V, num_blocks, td_x*td_y])
        """
        # encoded.long() ensures proper dtype, then gather from lut
        # self.lut shape = [V, levels]
        return self.lut[:, encoded.long()]
        
    def unpack_trellis(self, packed: torch.Tensor, pack_factor: int) -> torch.Tensor:
        """
        Unpack trellis indices from packed format
        
        Args:
            packed: Packed tensor of shape [rows, cols]
            pack_factor: Number of elements per block
            
        Returns:
            Unpacked tensor of shape [rows, cols*pack_factor]
        """
        rows, cols = packed.shape
        unpacked = torch.zeros((rows * pack_factor, cols), 
                              device=packed.device, 
                              dtype=torch.int32)
        
        # Unpack each element based on the bit pattern
        for i in range(pack_factor):
            mask = (1 << self.L) - 1
            shift = i * self.L
            unpacked[i::pack_factor, :] = (packed >> shift) & mask
            
        return unpacked
```


## Step 5: Implement Decoding Functions for QTIP (in _custom_ops.py)

Add these decoding functions to support different modes:

```python
def decode_1mad(x: torch.Tensor) -> torch.Tensor:
    """
    Decode using 1-MAD (Multiply-Add) method.
    This decoding uses a single multiply-add operation.
    """
    x = x.to(torch.int64) & ((1 << 32) - 1)
    x = x * 34038481 + 76625530
    x = x & ((1 << 32) - 1)
    y = (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + ((x >> 24) & 0xFF)
    y = y - 510
    return (y.to(torch.float32) / 147.800537109375)

def decode_2mad(x: torch.Tensor) -> torch.Tensor:
    """
    Decode using 2-MAD method.
    Uses two multiply-add operations for potentially better accuracy.
    """
    x = x.to(torch.int64) & ((1 << 32) - 1)
    x = x * 264435761 + 1013904223
    x = x & ((1 << 32) - 1)
    x = ((x * 1664525) >> 32) + x
    x = x & ((1 << 32) - 1)
    y = (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + ((x >> 24) & 0xFF)
    y = y - 510
    return (y.to(torch.float32) / 147.800537109375)

def decode_3inst(x: torch.Tensor) -> torch.Tensor:
    """
    Decode using 3-instruction method.
    This uses floating point bit manipulation for high precision.
    """
    def bfe16_to_fp16(z: torch.Tensor) -> torch.Tensor:
        arr = z.to(torch.int32)
        mask = arr >= (1 << 15)
        arr[mask] -= (1 << 16)
        
        tmp = arr.to(torch.int16).cpu().numpy().view(np.float16)
        return torch.from_numpy(tmp).to(z.device)

    a, b = 89226354, 64248484
    fpmask = 996162400
    z = x.to(torch.int64) & ((1 << 32) - 1)
    z = z * a + b
    mask = ((1 << 15) | ((1 << 12) - 1)) << 16
    res = (mask & z) ^ fpmask
    top = bfe16_to_fp16(res >> 16)
    bot = bfe16_to_fp16(res & 0xFFFF)
    return (top + bot).float()

def quantlut(tlut: torch.Tensor, L: int, nbits: int) -> torch.Tensor:
    """
    Create a quantized lookup table from a provided table.
    """
    lut = torch.arange(1 << L, device=tlut.device)
    lut = (lut + 1) * lut
    lut = (lut >> (16 - nbits)) & ((1 << nbits) - 1)
    return tlut[lut].T.contiguous()

def quantlut_sym(tlut: torch.Tensor, L: int, nbits: int) -> torch.Tensor:
    """
    Create a symmetric quantized lookup table.
    """
    lut = torch.arange(1 << L, device=tlut.device)
    lut = (lut + 1) * lut
    sign = 1 - ((lut >> 15) & 1) * 2
    lut = (lut >> (16 - nbits - 1)) & ((1 << nbits) - 1)
    out = tlut[lut]
    out[:, 0] = out[:, 0] * sign
    return out.T.contiguous()
```

## Step 6: Implement the Bitshift GEMM Function (in _custom_ops.py)
This function performs the matrix multiplication with the quantized weights:

```python
def bitshift_gemm(
    input: torch.Tensor,
    trellis: torch.Tensor,
    codebook: bitshift_codebook,
    td_x: int,
    td_y: int,
    scale: float,
    SU: torch.Tensor,
    SV: torch.Tensor
) -> torch.Tensor:
    """
    Python fallback for QTIP Bitshift GEMM.

    Steps:
      1. Decode each block via codebook.recons → [V, num_blocks, block_size]
      2. (V must be 1 in fallback) take recons[0] → [num_blocks, block_size]
      3. Reshape & transpose to reconstruct full weight matrix hatW [m, n]
      4. Compute input @ hatW^T and divide by scale → output [B, n]

    Args:
        input:    [B, m]  activation tensor
        trellis:  [num_blocks, td_x*td_y]  block-wise index matrix
        codebook: bitshift_codebook instance (with .recons method)
        td_x:     block row size
        td_y:     block col size
        scale:    dequantization scale factor
        SU:       scale factor for input
        SV:       scale factor for output

    Returns:
        output: [B, n]  result of dequantized GEMM
    """
    B, m = input.shape
    input = input.to(torch.float32) * SU  # ← SU corrects input

    # decode
    recons = codebook.recons(trellis)
    assert recons.shape[0] == 1
    recons = recons[0]

    row_blocks = m // td_x
    col_blocks = recons.shape[0] // row_blocks
    n = col_blocks * td_y

    hatW = (
        recons
        .view(row_blocks, col_blocks, td_x, td_y)
        .transpose(1, 2)
        .reshape(m, n)
    )

    out = input.matmul(hatW.T)  # [B, n]
    return (out * SV * scale).to(input.dtype)  # ← SV corrects output
```

## Step 7: Register QTIP in the Quantization Methods Registry(in vllm/model_executor/layers/quantization/__init__.py)

Update the `__init__.py` file in the quantization directory to include QTIP:

```python
# SPDX-License-Identifier: Apache-2.0

from typing import Literal, get_args

from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)

QuantizationMethods = Literal[
    "aqlm",
    "awq",
    "deepspeedfp",
    "tpu_int8",
    "fp8",
    "ptpc_fp8",
    "fbgemm_fp8",
    "modelopt",
    "nvfp4",
    "marlin",
    "bitblas",
    "gguf",
    "gptq_marlin_24",
    "gptq_marlin",
    "gptq_bitblas",
    "awq_marlin",
    "gptq",
    "compressed-tensors",
    "bitsandbytes",
    "qqq",
    "hqq",
    "experts_int8",
    "neuron_quant",
    "ipex",
    "quark",
    "moe_wna16",
    "qtip",
    "torchao",
]
QUANTIZATION_METHODS: list[str] = list(get_args(QuantizationMethods))

# The customized quantization methods which will be added to this dict.
_CUSTOMIZED_METHOD_TO_QUANT_CONFIG = {}


def register_quantization_config(quantization: str):
    """Register a customized vllm quantization config.

    When a quantization method is not supported by vllm, you can register a customized
    quantization config to support it.

    Args:
        quantization (str): The quantization method name.

    Examples:
        >>> from vllm.model_executor.layers.quantization import register_quantization_config
        >>> from vllm.model_executor.layers.quantization import get_quantization_config
        >>> from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
        >>>
        >>> @register_quantization_config("my_quant")
        ... class MyQuantConfig(QuantizationConfig):
        ...     pass
        >>>
        >>> get_quantization_config("my_quant")
        <class 'MyQuantConfig'>
    """  # noqa: E501

    def _wrapper(quant_config_cls):
        if quantization in QUANTIZATION_METHODS:
            raise ValueError(
                f"The quantization method `{quantization}` is already exists.")
        if not issubclass(quant_config_cls, QuantizationConfig):
            raise ValueError("The quantization config must be a subclass of "
                             "`QuantizationConfig`.")
        _CUSTOMIZED_METHOD_TO_QUANT_CONFIG[quantization] = quant_config_cls
        QUANTIZATION_METHODS.append(quantization)
        return quant_config_cls

    return _wrapper


def get_quantization_config(quantization: str) -> type[QuantizationConfig]:
    if quantization not in QUANTIZATION_METHODS:
        raise ValueError(f"Invalid quantization method: {quantization}")

    # lazy import to avoid triggering `torch.compile` too early
    from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig

    from .aqlm import AQLMConfig
    from .awq import AWQConfig
    from .awq_marlin import AWQMarlinConfig
    from .bitblas import BitBLASConfig
    from .bitsandbytes import BitsAndBytesConfig
    from .compressed_tensors.compressed_tensors import (  # noqa: E501
        CompressedTensorsConfig)
    from .deepspeedfp import DeepSpeedFPConfig
    from .experts_int8 import ExpertsInt8Config
    from .fbgemm_fp8 import FBGEMMFp8Config
    from .fp8 import Fp8Config
    from .gguf import GGUFConfig
    from .gptq import GPTQConfig
    from .gptq_bitblas import GPTQBitBLASConfig
    from .gptq_marlin import GPTQMarlinConfig
    from .gptq_marlin_24 import GPTQMarlin24Config
    from .hqq_marlin import HQQMarlinConfig
    from .ipex_quant import IPEXConfig
    from .marlin import MarlinConfig
    from .modelopt import ModelOptFp8Config, ModelOptNvFp4Config
    from .moe_wna16 import MoeWNA16Config
    from .neuron_quant import NeuronQuantConfig
    from .ptpc_fp8 import PTPCFp8Config
    from .qqq import QQQConfig
    from .qtip import QTIPConfig
    from .torchao import TorchAOConfig
    from .tpu_int8 import Int8TpuConfig
    
    method_to_config: dict[str, type[QuantizationConfig]] = {
        "aqlm": AQLMConfig,
        "awq": AWQConfig,
        "deepspeedfp": DeepSpeedFPConfig,
        "tpu_int8": Int8TpuConfig,
        "fp8": Fp8Config,
        "fbgemm_fp8": FBGEMMFp8Config,
        "modelopt": ModelOptFp8Config,
        "nvfp4": ModelOptNvFp4Config,
        "marlin": MarlinConfig,
        "bitblas": BitBLASConfig,
        "gguf": GGUFConfig,
        "gptq_marlin_24": GPTQMarlin24Config,
        "gptq_marlin": GPTQMarlinConfig,
        "gptq_bitblas": GPTQBitBLASConfig,
        "awq_marlin": AWQMarlinConfig,
        "gptq": GPTQConfig,
        "compressed-tensors": CompressedTensorsConfig,
        "bitsandbytes": BitsAndBytesConfig,
        "ptpc_fp8": PTPCFp8Config,
        "qqq": QQQConfig,
        "hqq": HQQMarlinConfig,
        "experts_int8": ExpertsInt8Config,
        "neuron_quant": NeuronQuantConfig,
        "ipex": IPEXConfig,
        "quark": QuarkConfig,
        "moe_wna16": MoeWNA16Config,
        "qtip": QTIPConfig,
        "torchao": TorchAOConfig,
    }
    # Update the `method_to_config` with customized quantization methods.
    method_to_config.update(_CUSTOMIZED_METHOD_TO_QUANT_CONFIG)

    return method_to_config[quantization]


__all__ = [
    "QuantizationConfig",
    "QuantizationMethods",
    "get_quantization_config",
    "QUANTIZATION_METHODS",
]
```

## Step 8: Test the Integration (in tests/kernels/quantization/test_qtip.py)

Create a test file to validate your implementation:

```python
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm import _custom_ops as ops

HIDDEN_SIZES = [4096]
OUT_SIZES = [4096]
L_VALUES = [16]
K_VALUES = [3]
V_VALUES = [1]


@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("out_size", OUT_SIZES)
@pytest.mark.parametrize("L", L_VALUES)
@pytest.mark.parametrize("K", K_VALUES)
@pytest.mark.parametrize("V", V_VALUES)
def test_bitshift_codebook(hidden_size, out_size, L, K, V):
    td_x, td_y = 8, 8
    block_size = td_x * td_y
    row_blocks = hidden_size // td_x
    col_blocks = out_size // td_y
    num_blocks = row_blocks * col_blocks

    codebook = ops.bitshift_codebook(
        L=L,
        K=K,
        V=V,
        tlut_bits=L,
        decode_mode="lut"
    )

    encoded = torch.randint(
        0, 2**L,
        (num_blocks, block_size),
        device='cpu', 
        dtype=torch.long
    )

    decoded = codebook.recons(encoded)

    assert decoded.shape == (V, num_blocks, block_size)


@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("out_size", OUT_SIZES)
@pytest.mark.parametrize("L", L_VALUES)
@pytest.mark.parametrize("K", K_VALUES)
@pytest.mark.parametrize("V", V_VALUES)
def test_bitshift_gemm(hidden_size, out_size, L, K, V):
    td_x = 8
    td_y = 8
    block_size = td_x * td_y

    m = hidden_size
    n = out_size
    row_blocks = m // td_x
    col_blocks = n // td_y
    num_blocks = row_blocks * col_blocks
    
    codebook = ops.bitshift_codebook(
        L=L,
        K=K,
        V=V,
        tlut_bits=L,
        decode_mode="lut"
    )

    x = torch.rand((1, m), device='cpu', dtype=torch.float16)
    
    trellis = torch.randint(
        0, 2 ** L,
        (num_blocks, block_size),
        device='cpu',
        dtype=torch.long
    )

    SU = torch.ones(m, device='cpu', dtype=torch.float16)
    SV = torch.ones(n, device='cpu', dtype=torch.float16)

    output = ops.bitshift_gemm(
        input=x,
        trellis=trellis,
        codebook=codebook,
        td_x=td_x,
        td_y=td_y,
        scale=32.0,
        SU=SU,
        SV=SV
    )

    assert output.shape == (1, n)
```
## Step 8: Test the Config (in tests\quantization\test_qtip_config.py)
```python
# SPDX-License-Identifier: Apache-2.0
"""Tests for QTIP quantization configuration.

Run `pytest tests/quantization/test_qtip_config.py`.
"""

import pytest
import torch

from vllm.model_executor.layers.quantization.qtip import (QTIPConfig,
                                                          QTIPLinearMethod)


def test_qtip_config_creation():
    """Test QTIP configuration creation and validation."""
    
    config = QTIPConfig(td_x=8,
                        td_y=8,
                        L=16,
                        K=2,
                        V=2,
                        tlut_bits=16,
                        decode_mode="1mad",
                        scale=32.0)

    
    assert config.td_x == 8
    assert config.td_y == 8
    assert config.L == 16
    assert config.K == 2
    assert config.V == 2
    assert config.tlut_bits == 16
    assert config.decode_mode == "1mad"
    assert config.scale == 32.0
    assert config.pack_factor == 64  # td_x * td_y

    
    config_dict = {
        "td_x": 8,
        "td_y": 8,
        "L": 16,
        "K": 2,
        "V": 2,
        "tlut_bits": 16,
        "decode_mode": "1mad",
        "scale": 32.0
    }
    config_from_dict = QTIPConfig.from_config(config_dict)
    assert config_from_dict.td_x == config.td_x
    assert config_from_dict.td_y == config.td_y
    assert config_from_dict.L == config.L
    assert config_from_dict.K == config.K
    assert config_from_dict.V == config.V
    assert config_from_dict.tlut_bits == config.tlut_bits
    assert config_from_dict.decode_mode == config.decode_mode
    assert config_from_dict.scale == config.scale


def test_qtip_config_methods():
    """Test QTIP configuration methods."""
    config = QTIPConfig(td_x=8,
                        td_y=8,
                        L=16,
                        K=2,
                        V=2,
                        tlut_bits=16,
                        decode_mode="1mad",
                        scale=32.0)

   
    assert config.get_name() == "qtip"

   
    assert torch.half in config.get_supported_act_dtypes()

    
    assert config.get_min_capability() == 60

    
    assert "quantize_config.json" in config.get_config_filenames()




# needs cuda, currently using cpu
# @pytest.mark.parametrize(
#     "model",
#     [
#         "meta-llama/Llama-2-7b-hf",  
#     ])
# def test_qtip_inference(vllm_runner, model, monkeypatch):
#     """Test inference with QTIP quantization."""
    
#     monkeypatch.setenv("VLLM_USE_V1", "0")

    
#     qtip_config = {
#         "quant_method": "qtip",
#         "td_x": 8,
#         "td_y": 8,
#         "L": 16,
#         "K": 2,
#         "V": 2,
#         "tlut_bits": 16,
#         "decode_mode": "1mad",
#         "scale": 32.0
#     }

   
#     with vllm_runner(model_name=model,
#                      quantization="qtip",
#                      enforce_eager=True,
#                      ) as llm:
        
#         model = llm.model.llm_engine.model_executor.driver_worker.model_runner.model
#         layer = model.model.layers[0]

#         assert isinstance(layer.self_attn.qkv_proj.quant_method,
#                           QTIPLinearMethod)

#         output = llm.generate_greedy("Hello my name is", max_tokens=20)
#         assert output

```

## Step 9: Testing command (in the workspace of docker)
```python
pip install -r requirements/dev.txt
pytest tests/kernels/quantization/test_qtip.py
pytest tests/quantization/test_qtip_config.py
```