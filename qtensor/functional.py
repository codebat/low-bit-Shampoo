# code is based the implementation of "bitsandbytes"
# see https://github.com/TimDettmers/bitsandbytes/tree/main/bitsandbytes

import math
import ctypes as ct
import torch
from torch import Tensor
from typing import Optional, Tuple
from qtensor.cextension import get_lib


try:
    lib = get_lib()
except Exception:
    lib = None

if lib is not None:
    lib_quan = {
        8 : (
            lib.cquantize_blockwise_8bit_fp32,
            lib.cquantize_blockwise_8bit_bf16,
        ),
        4 : (
            lib.cquantize_blockwise_4bit_fp32,
            lib.cquantize_blockwise_4bit_bf16,
        ),
    }

    lib_dequan = {
        8 : (
            lib.cdequantize_blockwise_8bit_fp32,
            lib.cdequantize_blockwise_8bit_bf16,
        ),
        4 : (
            lib.cdequantize_blockwise_4bit_fp32,
            lib.cdequantize_blockwise_4bit_bf16,
        ),
    }

    lib_quan_diagreal = {
        8 : (
            lib.cquantize_blockwise_diagreal_8bit_fp32,
            lib.cquantize_blockwise_diagreal_8bit_bf16,
        ),
        4 : (
            lib.cquantize_blockwise_diagreal_4bit_fp32,
            lib.cquantize_blockwise_diagreal_4bit_bf16,
        ),
    }

    lib_dequan_diagreal = {
        8 : (
            lib.cdequantize_blockwise_diagreal_8bit_fp32,
            lib.cdequantize_blockwise_diagreal_8bit_bf16,
        ),
        4 : (
            lib.cdequantize_blockwise_diagreal_4bit_fp32,
            lib.cdequantize_blockwise_diagreal_4bit_bf16,
        ),
    }
else:
    lib_quan = {}
    lib_dequan = {}
    lib_quan_diagreal = {}
    lib_dequan_diagreal = {}


def get_ptr(A: Tensor) -> ct.c_void_p:
    """
    Get the ctypes pointer from a PyTorch Tensor.

    Parameters
    ----------
    A : torch.tensor
        The PyTorch tensor.

    Returns
    -------
    ctypes.c_void_p
    """
    if A is None:
        return None
    else:
        return ct.c_void_p(A.data.data_ptr())


def create_dynamic_map(signed=True, total_bits=8, power=1):
    data = []
    max_exponent_bits = total_bits - 1
    for i in range(max_exponent_bits):
        fraction_items = int((2 ** i + 1 if signed else 2 ** (i + 1) + 1))
        boundaries = torch.linspace(0.1, 1, fraction_items)
        means = (boundaries[:-1] + boundaries[1:]) / 2.0
        data += ((10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()
        if signed:
            data += (-(10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()

    data.append(0)
    data.append(1.0)
    data.sort()

    data = torch.Tensor(data)
    return data.sign() * data.abs().pow(power)


def create_linear_map(signed=True, total_bits=8, power=2):
    if signed:
        data = torch.linspace(-1, 1, (2 ** total_bits))
        data[2 ** (total_bits-1) - 1] = 0
    else:
        data = torch.linspace(0, 1, (2 ** total_bits))

    return data.sign() * data.abs().pow(power)


def quantize_blockwise(
    A: Tensor,
    code: Tensor,
    order: int,
    absmax: Optional[Tensor] = None,
    out: Optional[Tensor] = None,
    blocksize: int = 256,
    bits: int = 8,
) -> Tuple[Tensor, Tensor]:
#    assert bits in [8, 4]

    if A.is_cuda:
        if lib is None:
            raise RuntimeError("CUDA quantization library is unavailable; rebuild the extension or install a CUDA-enabled PyTorch build.")
        if absmax is None:
            blocks = order // blocksize
            blocks += 1 if order % blocksize > 0 else 0
            blocks *= order
            absmax = torch.empty((blocks,), device=A.device, dtype=torch.float32)

        if out is None:
            n = order * order
            m = 8 // bits
            out_numel = n // m
            out_numel += 1 if n % m > 0 else 0
            out = torch.empty(out_numel, device=A.device, dtype=torch.uint8)

        assert blocksize in [2048, 1024, 512, 256, 128, 64]
        assert code.shape == torch.Size([2 ** bits])
        cblocksize = ct.c_int32(blocksize)
        corder = ct.c_int32(order)

        if A.dtype == torch.float32:
            quan_func = lib_quan[bits][0]
        elif A.dtype == torch.bfloat16:
            quan_func = lib_quan[bits][1]
        else:
            raise ValueError(f"data type of A is not supported: {A.dtype}")

        quan_func(get_ptr(A), get_ptr(code), corder, get_ptr(absmax), get_ptr(out), cblocksize)
    else:
        out, absmax = _quantize_blockwise_cpu(
            A=A,
            code=code,
            order=order,
            absmax=absmax,
            out=out,
            blocksize=blocksize,
            bits=bits,
        )

    return out, absmax


def dequantize_blockwise(
    A: Tensor,
    code: Tensor,
    order: int,
    absmax: Tensor,
    outdtype = torch.float32,
    out: Optional[Tensor] = None,
    blocksize: int = 256,
    bits: int = 8,
) -> Tensor:
#    assert bits in [8, 4]

    if A.is_cuda:
        if lib is None:
            raise RuntimeError("CUDA quantization library is unavailable; rebuild the extension or install a CUDA-enabled PyTorch build.")
        if out is None:
            out = torch.empty((order, order), device=A.device, dtype=outdtype)

        assert blocksize in [2048, 1024, 512, 256, 128, 64]
        assert code.shape == torch.Size([2 ** bits])
        cblocksize = ct.c_int32(blocksize)
        corder = ct.c_int32(order)

        if out.dtype == torch.float32:
            dequan_func = lib_dequan[bits][0]
        elif out.dtype == torch.bfloat16:
            dequan_func = lib_dequan[bits][1]
        else:
            raise ValueError(f"data type of out is not supported: {out.dtype}")

        dequan_func(get_ptr(A), get_ptr(code), corder, get_ptr(absmax), get_ptr(out), cblocksize)
    else:
        out = _dequantize_blockwise_cpu(
            A=A,
            code=code,
            order=order,
            absmax=absmax,
            out=out,
            blocksize=blocksize,
            bits=bits,
            outdtype=outdtype,
        )

    return out


def quantize_blockwise_diagreal(
    A: Tensor,
    code: Tensor,
    order: int,
    absmax: Optional[Tensor] = None,
    diag: Optional[Tensor] = None,
    out: Optional[Tensor] = None,
    blocksize: int = 256,
    bits: int = 8,
) -> Tuple[Tensor, Tensor, Tensor]:
#    assert bits in [8, 4]

    if A.is_cuda:
        if lib is None:
            raise RuntimeError("CUDA quantization library is unavailable; rebuild the extension or install a CUDA-enabled PyTorch build.")
        if absmax is None:
            blocks = order // blocksize
            blocks += 1 if order % blocksize > 0 else 0
            blocks *= order
            absmax = torch.empty((blocks,), device=A.device, dtype=torch.float32)

        if diag is None:
            diag = torch.empty((order,), device=A.device, dtype=torch.float32)

        if out is None:
            n = order * order
            m = 8 // bits
            out_numel = n // m
            out_numel += 1 if n % m > 0 else 0
            out = torch.empty(out_numel, device=A.device, dtype=torch.uint8)

        assert blocksize in [2048, 1024, 512, 256, 128, 64]
        assert code.shape == torch.Size([2 ** bits])
        cblocksize = ct.c_int32(blocksize)
        corder = ct.c_int32(order)

        if A.dtype == torch.float32:
            quan_func = lib_quan_diagreal[bits][0]
        elif A.dtype == torch.bfloat16:
            quan_func = lib_quan_diagreal[bits][1]
        else:
            raise ValueError(f"data type of A is not supported: {A.dtype}")

        quan_func(get_ptr(A), get_ptr(code), corder, get_ptr(absmax), get_ptr(diag), get_ptr(out), cblocksize)
    else:
        out, absmax, diag = _quantize_blockwise_cpu(
            A=A,
            code=code,
            order=order,
            absmax=absmax,
            out=out,
            blocksize=blocksize,
            bits=bits,
            store_diag=True,
            diag=diag,
        )

    return out, absmax, diag


def dequantize_blockwise_diagreal(
    A: Tensor,
    code: Tensor,
    order: int,
    absmax: Tensor,
    diag: Tensor,
    outdtype = torch.float32,
    out: Optional[Tensor] = None,
    blocksize: int = 256,
    bits: int = 8,
) -> Tensor:
#    assert bits in [8, 4]

    if A.is_cuda:
        if lib is None:
            raise RuntimeError("CUDA quantization library is unavailable; rebuild the extension or install a CUDA-enabled PyTorch build.")
        if out is None:
            out = torch.empty((order, order), device=A.device, dtype=outdtype)

        assert blocksize in [2048, 1024, 512, 256, 128, 64]
        assert code.shape == torch.Size([2 ** bits])
        cblocksize = ct.c_int32(blocksize)
        corder = ct.c_int32(order)

        if out.dtype == torch.float32:
            dequan_func = lib_dequan_diagreal[bits][0]
        elif out.dtype == torch.bfloat16:
            dequan_func = lib_dequan_diagreal[bits][1]
        else:
            raise ValueError(f"data type of out is not supported: {out.dtype}")

        dequan_func(get_ptr(A), get_ptr(code), corder, get_ptr(absmax), get_ptr(diag), get_ptr(out), cblocksize)
    else:
        out = _dequantize_blockwise_cpu(
            A=A,
            code=code,
            order=order,
            absmax=absmax,
            out=out,
            blocksize=blocksize,
            bits=bits,
            outdtype=outdtype,
            diag=diag,
        )

    return out


def _quantize_blockwise_cpu(
    A: Tensor,
    code: Tensor,
    order: int,
    absmax: Optional[Tensor],
    out: Optional[Tensor],
    blocksize: int,
    bits: int,
    store_diag: bool = False,
    diag: Optional[Tensor] = None,
):
    if bits not in (4, 8):
        raise ValueError(f"Unsupported bit width on CPU quantization: {bits}")
    if blocksize <= 0:
        raise ValueError("blocksize must be positive.")

    device = A.device
    dtype = torch.float32
    values_per_byte = 8 // bits
    num_blocks_row = math.ceil(order / blocksize)
    total_blocks = order * num_blocks_row
    total_elems = order * order

    absmax = _prepare_buffer(absmax, total_blocks, device, torch.float32)
    if store_diag:
        diag = _prepare_buffer(diag, order, device, torch.float32)

    expected_out_len = (total_elems + values_per_byte - 1) // values_per_byte
    out = _prepare_buffer(out, expected_out_len, device, torch.uint8)
    out.zero_()

    code_float = code.to(device=device, dtype=dtype)
    if code_float.ndim != 1 or code_float.numel() != 2 ** bits:
        raise ValueError(f"code tensor must be 1-D with length 2**bits (got shape {code_float.shape})")

    if torch.any(code_float[1:] < code_float[:-1]):
        code_sorted, sort_idx = torch.sort(code_float)
    else:
        code_sorted = code_float
        sort_idx = None

    code_min = code_sorted[0].item()
    code_max = code_sorted[-1].item()

    flat = A.to(dtype).contiguous().view(-1).clone()
    if flat.numel() != total_elems:
        raise ValueError(f"Expected tensor with {total_elems} elements, found {flat.numel()}.")

    if store_diag:
        diag_vals = torch.diagonal(A, 0).to(torch.float32)
        diag.copy_(diag_vals)
        row_indices = torch.arange(order, device=device)
        flat[row_indices * order + row_indices] = 0.0

    indices = torch.empty(total_elems, dtype=torch.int64, device=device)

    for row in range(order):
        row_offset = row * order
        for block_idx in range(num_blocks_row):
            col_start = block_idx * blocksize
            if col_start >= order:
                absmax[row * num_blocks_row + block_idx] = 0.0
                continue
            col_end = min(col_start + blocksize, order)
            start = row_offset + col_start
            end = row_offset + col_end
            block = flat[start:end]
            if block.numel() == 0:
                absmax[row * num_blocks_row + block_idx] = 0.0
                continue

            amax_value = float(block.abs().max())
            absmax[row * num_blocks_row + block_idx] = amax_value

            if amax_value == 0.0:
                indices[start:end] = 0
                continue

            scaled = (block / amax_value).clamp(min=code_min, max=code_max)
            idx = _nearest_code_indices(scaled, code_sorted, sort_idx)
            indices[start:end] = idx.to(torch.int64)

    packed = _pack_indices(indices, bits)
    if packed.numel() != out.numel():
        out.resize_(packed.shape)
    out.copy_(packed)

    if store_diag:
        return out, absmax, diag
    return out, absmax


def _dequantize_blockwise_cpu(
    A: Tensor,
    code: Tensor,
    order: int,
    absmax: Tensor,
    out: Optional[Tensor],
    blocksize: int,
    bits: int,
    outdtype: torch.dtype,
    diag: Optional[Tensor] = None,
):
    if bits not in (4, 8):
        raise ValueError(f"Unsupported bit width on CPU dequantization: {bits}")
    if blocksize <= 0:
        raise ValueError("blocksize must be positive.")

    device = A.device
    values_per_byte = 8 // bits
    num_blocks_row = math.ceil(order / blocksize)
    total_blocks = order * num_blocks_row
    total_elems = order * order

    if absmax.device != device:
        raise ValueError("absmax tensor must live on the same device as quantized data.")
    if absmax.numel() != total_blocks:
        raise ValueError(f"absmax has {absmax.numel()} entries, expected {total_blocks}.")

    code_float = code.to(device=device, dtype=torch.float32)
    indices = _unpack_indices(A, total_elems, bits, device)
    values = torch.empty(total_elems, dtype=torch.float32, device=device)

    for row in range(order):
        row_offset = row * order
        for block_idx in range(num_blocks_row):
            col_start = block_idx * blocksize
            if col_start >= order:
                continue
            col_end = min(col_start + blocksize, order)
            start = row_offset + col_start
            end = row_offset + col_end
            block_indices = indices[start:end]
            amax = absmax[row * num_blocks_row + block_idx]
            if block_indices.numel() == 0:
                continue
            amax_value = float(amax)
            if amax_value == 0.0:
                values[start:end] = 0.0
                continue

            block_vals = code_float[block_indices.to(torch.long)] * amax_value
            values[start:end] = block_vals

    matrix = values.view(order, order)
    result = matrix.to(outdtype)

    if diag is not None:
        if diag.device != device:
            raise ValueError("diag tensor must live on the same device as quantized data.")
        if diag.numel() != order:
            raise ValueError(f"diag tensor must have length {order}.")
        result.diagonal().copy_(diag.to(outdtype))

    if out is None:
        return result

    if out.shape != (order, order):
        out.resize_(order, order)
    if out.dtype != outdtype:
        raise ValueError("Provided output tensor has incorrect dtype.")
    out.copy_(result)
    return out


def _prepare_buffer(tensor: Optional[Tensor], length: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    if tensor is None:
        return torch.empty(length, device=device, dtype=dtype)
    if tensor.device != device:
        raise ValueError("Buffer device mismatch.")
    if tensor.dtype != dtype:
        raise ValueError("Buffer dtype mismatch.")
    if tensor.numel() != length:
        tensor.resize_(length)
    return tensor


def _nearest_code_indices(values: Tensor, code_sorted: Tensor, sort_idx: Optional[Tensor]) -> Tensor:
    insert = torch.searchsorted(code_sorted, values, right=True)
    insert = torch.clamp(insert, max=code_sorted.numel())
    lower = torch.clamp(insert - 1, min=0)
    upper = torch.clamp(insert, max=code_sorted.numel() - 1)

    lower_vals = code_sorted[lower]
    upper_vals = code_sorted[upper]
    choose_upper = (values - lower_vals).abs() > (upper_vals - values).abs()
    chosen = torch.where(choose_upper, upper, lower)
    if sort_idx is not None:
        chosen = sort_idx[chosen]
    return chosen.to(torch.int64)


def _pack_indices(indices: Tensor, bits: int) -> Tensor:
    values_per_byte = 8 // bits
    if bits == 8:
        return indices.to(torch.uint8)

    mask = (1 << bits) - 1
    pad = (-indices.numel()) % values_per_byte
    if pad:
        indices = torch.cat(
            [
                indices,
                torch.zeros(pad, dtype=indices.dtype, device=indices.device),
            ],
            dim=0,
        )
    shifts = torch.arange(values_per_byte - 1, -1, -1, device=indices.device, dtype=torch.int64) * bits
    packed = ((indices.view(-1, values_per_byte).to(torch.int64) & mask) << shifts).sum(dim=1)
    return packed.to(torch.uint8)


def _unpack_indices(packed: Tensor, numel: int, bits: int, device: torch.device) -> Tensor:
    values_per_byte = 8 // bits
    if bits == 8:
        return packed.to(torch.int64)[:numel]

    mask = (1 << bits) - 1
    shifts = torch.arange(values_per_byte - 1, -1, -1, device=device, dtype=torch.int64) * bits
    expanded = ((packed.to(torch.int64).unsqueeze(1) >> shifts) & mask).reshape(-1)
    return expanded[:numel]


@torch.no_grad()
def compute_power(Vt, S, p, iter_count=4, ridge_epsilon=1e-6):
    for j in range(iter_count):
        Vt = 1.5 * Vt - 0.5 * Vt @ Vt.T @ Vt
    rho = ridge_epsilon * S.max()

    return Vt.T @ (1 / (S + rho).pow(1 / p)).diag() @ Vt
