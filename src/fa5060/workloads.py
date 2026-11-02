"""Frozen attention workloads and deterministic native-layout inputs.

The full matrix is intentionally independent of discovered provider support.
An unsupported case stays in the report instead of disappearing from it.
"""

from dataclasses import dataclass
from itertools import product
from typing import Any


@dataclass(frozen=True, slots=True)
class AttentionCase:
    phase: str
    batch: int
    kv_len: int
    head_dim: int
    heads_kv: int
    dtype: str
    layout: str
    heads_q: int = 32
    page_size: int = 16

    def __post_init__(self):
        if self.phase not in ("prefill", "decode"):
            raise ValueError("phase must be prefill or decode")
        if self.layout not in ("dense", "paged") or self.dtype not in ("fp16", "bf16"):
            raise ValueError("unsupported layout or dtype")
        if min(self.batch, self.kv_len, self.head_dim, self.heads_kv, self.heads_q) <= 0:
            raise ValueError("benchmark dimensions must be positive")
        if self.heads_q % self.heads_kv or self.head_dim not in (32, 64, 128):
            raise ValueError("heads_q must divide into KV groups; D must be 32, 64 or 128")
        if self.page_size not in (16, 32, 64):
            raise ValueError("page size must be 16, 32 or 64")

    @property
    def causal(self) -> bool:
        return self.phase == "prefill"

    @property
    def query_len(self) -> int:
        return self.kv_len if self.phase == "prefill" else 1

    @property
    def case_id(self) -> str:
        return (f"{self.phase}-b{self.batch}-l{self.kv_len}-hq{self.heads_q}"
                f"-hkv{self.heads_kv}-d{self.head_dim}-{self.dtype}-{self.layout}"
                + (f"{self.page_size}" if self.layout == "paged" else ""))

    def to_dict(self) -> dict[str, Any]:
        q_shape = ([self.batch, self.query_len, self.heads_q, self.head_dim]
                   if self.phase == "prefill" else [self.batch, self.heads_q, self.head_dim])
        pages = (self.kv_len + self.page_size - 1) // self.page_size
        kv_shape = ([self.batch, self.kv_len, self.heads_kv, self.head_dim]
                    if self.layout == "dense" else
                    [self.batch * pages, self.page_size, self.heads_kv, self.head_dim])
        return {
            "id": self.case_id, "phase": self.phase, "batch": self.batch,
            "kv_len": self.kv_len, "query_len": self.query_len,
            "head_dim": self.head_dim, "heads_q": self.heads_q,
            "heads_kv": self.heads_kv, "dtype": self.dtype, "layout": self.layout,
            "page_size": self.page_size, "causal": self.causal,
            "shape": {
                "q": q_shape, "k": kv_shape, "v": list(kv_shape),
                "kv_lens": [self.kv_len] * self.batch,
                "block_table": [self.batch, pages] if self.layout == "paged" else None,
                "heads_q": self.heads_q, "heads_kv": self.heads_kv,
                "head_dim": self.head_dim, "batch": self.batch,
                "query_len": self.query_len, "kv_len": self.kv_len,
            },
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AttentionCase":
        fields = ("phase", "batch", "kv_len", "head_dim", "heads_kv", "dtype", "layout")
        case = cls(**{name: value[name] for name in fields},
                   heads_q=value.get("heads_q", 32), page_size=value.get("page_size", 16))
        for name, expected in (("causal", case.causal), ("query_len", case.query_len),
                               ("id", case.case_id), ("shape", case.to_dict()["shape"])):
            if name in value and value[name] != expected:
                raise ValueError(f"inconsistent serialized {name}")
        return case


def full_suite() -> tuple[AttentionCase, ...]:
    """Return all 144 causal-prefill and 168 decode cases in frozen order."""
    sizes = {
        "prefill": ((1, 512), (1, 2048), (1, 8192), (4, 512), (4, 2048), (4, 8192)),
        "decode": ((1, 512), (1, 8192), (1, 32768), (4, 2048), (4, 8192),
                   (16, 512), (16, 2048)),
    }
    return tuple(
        AttentionCase(phase, batch, length, dim, heads, dtype, layout)
        for phase, pairs in sizes.items() for batch, length in pairs
        for dim, heads, dtype, layout in product(
            (64, 128), (32, 8, 1), ("fp16", "bf16"), ("dense", "paged"))
    )


def smoke_suite() -> tuple[AttentionCase, ...]:
    """Eight short cases cover both phases, dtypes, layouts, D and GQA."""
    return tuple(
        AttentionCase(phase, 1, 512, 64 if dtype == "fp16" else 128,
                      (32 if dtype == "fp16" else 8) if layout == "dense" else 1,
                      dtype, layout)
        for phase, dtype, layout in product(("prefill", "decode"),
                                            ("fp16", "bf16"), ("dense", "paged"))
    )


@dataclass(frozen=True, slots=True)
class AttentionInputs:
    q: Any
    k: Any
    v: Any
    kv_lens: Any
    block_table: Any = None


def make_inputs(case: AttentionCase, *, device: str = "cuda", seed: int = 20261102) -> AttentionInputs:
    """Allocate directly in the provider-visible layout, with no dense paging shim."""
    import torch

    generator = torch.Generator(device=device).manual_seed(seed)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[case.dtype]
    shape = case.to_dict()["shape"]
    q = torch.randn(shape["q"], generator=generator, dtype=dtype, device=device)
    if case.layout == "paged":
        # FMHAv2's native separate-page API addresses both caches relative
        # to K's allocation. Each cache remains an ordinary contiguous 4D view.
        pool = torch.randn((2, *shape["k"]), generator=generator, dtype=dtype, device=device)
        k, v = pool[0], pool[1]
    else:
        k = torch.randn(shape["k"], generator=generator, dtype=dtype, device=device)
        v = torch.randn(shape["v"], generator=generator, dtype=dtype, device=device)
    lengths = torch.full((case.batch,), case.kv_len, dtype=torch.int32, device=device)
    table = None
    if case.layout == "paged":
        pages = shape["block_table"][1]
        table = torch.randperm(case.batch * pages, generator=generator, device=device)
        table = table.to(torch.int32).reshape(case.batch, pages)
    return AttentionInputs(q, k, v, lengths, table)
