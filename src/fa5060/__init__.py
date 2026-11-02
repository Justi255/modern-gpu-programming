"""Executable experiments for the RTX 5060/Triton textbook.

GPU modules are loaded lazily so the documentation and CPU checks work on
machines without CUDA or PyTorch.
"""

__version__ = "0.1.0"
__all__ = ["prefill", "decode", "decode_workspace"]


def __getattr__(name):
    if name in __all__:
        from . import attention

        return getattr(attention, name)
    raise AttributeError(name)
