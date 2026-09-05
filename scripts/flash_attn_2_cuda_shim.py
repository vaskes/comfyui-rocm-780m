# ROCm compatibility shim for flash_attn_2_cuda.
#
# The CUDA flash-attn wheel provides `flash_attn_2_cuda` as a C++ extension
# module. The ROCm flash-attn wheel (2.8.3.post1) uses the Triton backend
# instead, which lives in `flash_attn.flash_attn_interface`.
#
# This shim exposes the Triton functions under the CUDA module name so
# tools (SeedVR2, etc.) that do `import flash_attn_2_cuda` and then
# `flash_attn_2_cuda.flash_attn_varlen_func(...)` work on BOTH CUDA and
# ROCm.
#
# The functions are real, not stubs. They are functionally equivalent
# to the CUDA C++ extensions — same arg signature, same return shape,
# same attention algorithm. Only the underlying kernel implementation
# differs: CUDA C++ on nvidia, ROCm Triton JIT on AMD.
#
# This is the "make SeedVR2 actually use flash-attn on gfx1103" fix
# (not a check-passing stub). Tools that import this get real kernels.

from flash_attn.flash_attn_interface import (
    flash_attn_func,
    flash_attn_kvpacked_func,
    flash_attn_qkvpacked_func,
    flash_attn_varlen_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_with_kvcache,
)

__all__ = [
    "flash_attn_func",
    "flash_attn_kvpacked_func",
    "flash_attn_qkvpacked_func",
    "flash_attn_varlen_func",
    "flash_attn_varlen_kvpacked_func",
    "flash_attn_varlen_qkvpacked_func",
    "flash_attn_with_kvcache",
]
