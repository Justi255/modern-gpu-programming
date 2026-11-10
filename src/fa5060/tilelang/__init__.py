"""Standalone TileLang teaching kernels, imported only when requested.

Importing this namespace needs neither a GPU compiler nor PyTorch.  The
fundamentals and attention modules validate inputs before compiling kernels.
"""

from importlib import import_module

__all__ = ["vector_add", "row_softmax", "transpose", "matmul", "prefill", "decode", "decode_workspace"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    module = "fundamentals" if name in {"vector_add", "row_softmax", "transpose", "matmul"} else "attention"
    value = getattr(import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value
