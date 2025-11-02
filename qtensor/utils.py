import torch
import qtensor.functional as F
from pathlib import Path
import re


class QTensor:
    def __init__(self, var, bits=32, name2qmap={}, code='', blocksize=2048, min_lowbit_size=4096):
        self.bits = bits

        if self.bits in [32, 16] or var.numel() < min_lowbit_size:
            if self.bits == 16:
                self.var = var.bfloat16()
                self.bits = 16
            else:
                self.var = var.float()
                self.bits = 32
        elif self.bits in [8, 4]:
            name2qmap[code] = name2qmap[code].to(var.device)
            self.name2qmap = name2qmap
            self.code = code
            self.blocksize = blocksize

            self.var_order = var.shape[0]
            self.var_dtype = var.dtype
            self.var, self.absmax = F.quantize_blockwise(var, code=self.name2qmap[self.code], order=self.var_order, blocksize=self.blocksize, bits=self.bits)
        else:
            raise ValueError(f'num of bits is not supported: {self.bits}')
    
    def quantize(self, var):
        if self.bits < 16:
            # Allow dynamic change of order if the target matrix size changes
            if var.shape[0] != self.var_order:
                self.var_order = var.shape[0]
                self.var_dtype = var.dtype
                self.var = None  # trigger reallocation in functional
                self.absmax = None
            out, absmax = F.quantize_blockwise(
                var.contiguous(),
                code=self.name2qmap[self.code],
                order=self.var_order,
                absmax=self.absmax,
                out=self.var,
                blocksize=self.blocksize,
                bits=self.bits,
            )
            self.var, self.absmax = out, absmax
        else:
            self.var = var.to(self.var.dtype)

    def dequantize(self):
        if self.bits < 16:
            return F.dequantize_blockwise(self.var, code=self.name2qmap[self.code], order=self.var_order, absmax=self.absmax, outdtype=self.var_dtype, blocksize=self.blocksize, bits=self.bits)
        else:
            return self.var

    def set_device(self, device):
        if self.bits < 16:
            self.name2qmap[self.code] = self.name2qmap[self.code].to(device)
            self.absmax = self.absmax.to(device)
        self.var = self.var.to(device)


class QTensorDiagReal:
    def __init__(self, var, bits=32, name2qmap={}, code='', blocksize=2048, min_lowbit_size=4096):
        self.bits = bits

        if self.bits in [32, 16] or var.numel() < min_lowbit_size:
            if self.bits == 16:
                self.var = var.bfloat16()
                self.bits = 16
            else:
                self.var = var.float()
                self.bits = 32
        elif self.bits in [8, 4]:
            name2qmap[code] = name2qmap[code].to(var.device)
            self.name2qmap = name2qmap
            self.code = code
            self.blocksize = blocksize

            self.var_order = var.shape[0]
            self.var_dtype = var.dtype
            self.var, self.absmax, self.diag = F.quantize_blockwise_diagreal(var, code=self.name2qmap[self.code], order=self.var_order, blocksize=self.blocksize, bits=self.bits)
        else:
            raise ValueError(f'num of bits is not supported: {self.bits}')
    
    def quantize(self, var):
        if self.bits < 16:
            # Allow dynamic change of order if the target matrix size changes
            if var.shape[0] != self.var_order:
                self.var_order = var.shape[0]
                self.var_dtype = var.dtype
                self.var = None  # trigger reallocation in functional
                self.absmax = None
                self.diag = None
            out, absmax, diag = F.quantize_blockwise_diagreal(
                var.contiguous(),
                code=self.name2qmap[self.code],
                order=self.var_order,
                absmax=self.absmax,
                diag=self.diag,
                out=self.var,
                blocksize=self.blocksize,
                bits=self.bits,
            )
            self.var, self.absmax, self.diag = out, absmax, diag
        else:
            self.var = var.to(self.var.dtype)

    def dequantize(self):
        if self.bits < 16:
            return F.dequantize_blockwise_diagreal(self.var, code=self.name2qmap[self.code], order=self.var_order, absmax=self.absmax, diag=self.diag, outdtype=self.var_dtype, blocksize=self.blocksize, bits=self.bits)
        else:
            return self.var

    def set_device(self, device):
        if self.bits < 16:
            self.name2qmap[self.code] = self.name2qmap[self.code].to(device)
            self.absmax = self.absmax.to(device)
            self.diag = self.diag.to(device)
        self.var = self.var.to(device)


class QTensorSVDFast:

    def __init__(self, var, bits=32, name2qmap={}, code='', blocksize=2048, min_lowbit_size=4096, rect_t1=1, rect_t2=4, log_prefix=None):
        self.bits = bits
        self.rect_t1 = rect_t1
        self.rect_t2 = rect_t2
        self.log_prefix = log_prefix
        self._log_written = False

        Vt = torch.eye(var.shape[0], device=var.device)
        n = var.shape[0]
        keep = max(1, n // 2)
        base_vals = var[0][0] * Vt.diag()
        self.keep_indices = torch.arange(n, device=var.device, dtype=torch.int64)[:keep]
        self.Svalue = base_vals[self.keep_indices].clone()
        self.max_eigenvalue = base_vals.max().clone()

        if self.bits in [32, 16] or var.numel() < min_lowbit_size:
            self.var = Vt
            self.bits = 32
        elif self.bits in [8, 4]:
            name2qmap[code] = name2qmap[code].to(var.device)
            self.name2qmap = name2qmap
            self.code = code
            self.blocksize = blocksize

            self.var_order = var.shape[0]
            self.var_dtype = Vt.dtype
            self.var, self.absmax = F.quantize_blockwise(Vt, code=self.name2qmap[self.code], order=self.var_order, blocksize=self.blocksize, bits=self.bits)
        else:
            raise ValueError(f'num of bits is not supported: {self.bits}')
    
    def quantize(self, var, Vt=None):
        V, _ = torch.linalg.qr(var.float() @ Vt.T.float())
        full_eigs = (V.T @ var.float() @ V).diag()

        keep = max(1, full_eigs.numel() // 2)
        sorted_vals, sorted_idx = torch.sort(full_eigs)
        keep_idx = sorted_idx[:keep]
        self.keep_indices = keep_idx.to(dtype=torch.int64)
        self.Svalue = sorted_vals[:keep].to(var.dtype)
        discarded_vals = sorted_vals[keep:].to(var.dtype)
        self.max_eigenvalue = full_eigs.max().to(var.dtype)

        if self.bits < 16:
            F.quantize_blockwise(V.T.contiguous(), code=self.name2qmap[self.code], order=self.var_order, absmax=self.absmax, out=self.var, blocksize=self.blocksize, bits=self.bits)
        else:
            self.var = V.T.contiguous()

        if self.log_prefix and not self._log_written:
            log_dir = Path("logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            kept = self.Svalue.detach().cpu()
            discarded = discarded_vals.detach().cpu()
            total_sum = full_eigs.sum().item()
            kept_sum = kept.sum().item()
            discarded_sum = discarded.sum().item()
            safe_prefix = re.sub(r"[^A-Za-z0-9_.-]", "_", self.log_prefix)
            path = log_dir / f"eigenvalues_{safe_prefix}.csv"
            with open(path, "w", encoding="utf-8") as f:
                f.write("rank,value\n")
                for idx, val in enumerate(kept, start=1):
                    f.write(f"{idx},{float(val)}\n")
                for offset, val in enumerate(discarded, start=len(kept) + 1):
                    f.write(f"{offset},{float(val)}\n")
            self._log_written = True


    def dequantize(self):
        if self.bits < 16:
            Vt = F.dequantize_blockwise(self.var, code=self.name2qmap[self.code], order=self.var_order, absmax=self.absmax, outdtype=self.var_dtype, blocksize=self.blocksize, bits=self.bits)
            for j in range(self.rect_t1):
                Vt = 1.5 * Vt - 0.5 * Vt @ Vt.T @ Vt
        else:
            Vt = self.var

        keep_idx = self.keep_indices.to(device=Vt.device)
        Svals = self.Svalue.to(device=Vt.device, dtype=Vt.dtype)
        V_subset = Vt[keep_idx]
        return V_subset.T @ Svals.diag() @ V_subset, Vt

    def computepower(self, exp, ridge_epsilon=1e-6):
        if self.bits < 16:
            Vt = F.dequantize_blockwise(self.var, code=self.name2qmap[self.code], order=self.var_order, absmax=self.absmax, outdtype=self.var_dtype, blocksize=self.blocksize, bits=self.bits)
            return F.compute_power(
                Vt,
                self.Svalue,
                exp,
                iter_count=self.rect_t2,
                ridge_epsilon=ridge_epsilon,
                indices=self.keep_indices,
                max_eigen=self.max_eigenvalue,
            )
        else:
            Vt = self.var
            return F.compute_power(
                Vt,
                self.Svalue,
                exp,
                iter_count=0,
                ridge_epsilon=ridge_epsilon,
                indices=self.keep_indices,
                max_eigen=self.max_eigenvalue,
            )

    def set_device(self, device):
        if self.bits < 16:
            self.name2qmap[self.code] = self.name2qmap[self.code].to(device)
            self.absmax = self.absmax.to(device)
        self.Svalue = self.Svalue.to(device)
        self.keep_indices = self.keep_indices.to(device)
        self.max_eigenvalue = self.max_eigenvalue.to(device)
        self.var = self.var.to(device)
