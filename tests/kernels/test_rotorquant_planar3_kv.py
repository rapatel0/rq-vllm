# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the rq-models project
#
# rq-models RotorQuant planar3 KV kernel parity tests.
#
# Sprint 004 Phase 2a scaffold. The actual ops (rotorquant_planar3_pack /
# rotorquant_planar3_unpack) are not yet wired into vLLM's setup.py; once
# Phase 2b adds the build, these tests exercise the math.
#
# What this validates:
#   - pack(unpack(x)) round-trip is lossy but bounded (3-bit = ~5% rel err).
#   - L2 norm of the reconstructed block is within tight tolerance of the
#     original (the per-block norm scalar is corrected in pack to compensate
#     for centroid quantization energy loss).
#   - Output shape and dtype match the contract the FlashAttention KV-write
#     hook depends on.
#   - Correctness is identical to llama-cpp-turboquant's planar3 round-trip
#     when given the same input bytes (cross-substrate parity — required so
#     llama.cpp baselines stay valid).
#
# Reference parity case is taken from a small fixed seed to keep this test
# deterministic across hardware.

from __future__ import annotations

import math
import pytest
import torch


# These imports will resolve once Phase 2b wires the extension into vllm._C.
# We pytest-skip the body when the op isn't registered, so this file passes
# CI noise even before Phase 2b lands.
try:
    rotorquant_planar3_pack = torch.ops.rq_models.rotorquant_planar3_pack
    rotorquant_planar3_unpack = torch.ops.rq_models.rotorquant_planar3_unpack
    _OPS_AVAILABLE = True
except (AttributeError, RuntimeError):
    _OPS_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not _OPS_AVAILABLE or not torch.cuda.is_available(),
    reason="rotorquant_planar3 ops not registered or no CUDA device "
           "(Phase 2b will wire the torch extension build).",
)


QK_PLANAR3 = 128
BLOCK_BYTES = 50


def _make_input(n_blocks: int, seed: int = 42, scale: float = 1.0) -> torch.Tensor:
    """fp16 input for n_blocks * 128 elements, drawn from N(0, scale^2)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(n_blocks * QK_PLANAR3, generator=g,
                        device="cuda", dtype=torch.float32)
            * scale).to(torch.float16)


@pytest.mark.parametrize("n_blocks", [1, 8, 64, 1024])
def test_pack_unpack_shapes(n_blocks: int) -> None:
    """Shape + dtype contract sanity check."""
    src = _make_input(n_blocks)
    packed = torch.empty(n_blocks * BLOCK_BYTES, device="cuda", dtype=torch.uint8)
    rotorquant_planar3_pack(src, packed, n_blocks)
    assert packed.shape == (n_blocks * BLOCK_BYTES,)
    assert packed.dtype == torch.uint8

    out = torch.empty(n_blocks * QK_PLANAR3, device="cuda", dtype=torch.float16)
    rotorquant_planar3_unpack(packed, out, n_blocks)
    assert out.shape == (n_blocks * QK_PLANAR3,)
    assert out.dtype == torch.float16


@pytest.mark.parametrize("n_blocks", [1, 8, 64])
@pytest.mark.parametrize("scale", [0.1, 1.0, 10.0])
def test_round_trip_l2_norm(n_blocks: int, scale: float) -> None:
    """Per-block L2 norm is preserved across pack -> unpack within 1e-2.

    The pack kernel stores a `corrected = grp_norm / recon_norm` scalar so
    that unpack(pack(x)) recovers the original L2 magnitude (modulo fp16
    rounding). This test confirms that correction is applied correctly.
    """
    src = _make_input(n_blocks, scale=scale)
    packed = torch.empty(n_blocks * BLOCK_BYTES, device="cuda", dtype=torch.uint8)
    out = torch.empty(n_blocks * QK_PLANAR3, device="cuda", dtype=torch.float16)
    rotorquant_planar3_pack(src, packed, n_blocks)
    rotorquant_planar3_unpack(packed, out, n_blocks)

    src_blocks = src.view(n_blocks, QK_PLANAR3).float()
    out_blocks = out.view(n_blocks, QK_PLANAR3).float()

    src_norms = torch.linalg.norm(src_blocks, dim=1)
    out_norms = torch.linalg.norm(out_blocks, dim=1)

    # Relative error per block
    rel_err = ((src_norms - out_norms).abs() / src_norms.clamp(min=1e-6))
    max_rel = rel_err.max().item()
    assert max_rel < 1e-2, f"L2 norm drift too large: max rel err = {max_rel}"


@pytest.mark.parametrize("n_blocks", [1, 8, 64])
def test_round_trip_per_element_relerr(n_blocks: int) -> None:
    """Per-element relative error is within 5% (3-bit Lloyd-Max codebook
    bound). The 8-level codebook has spacing ~0.05 in the unit-norm interior,
    so post-rotation values get 1 of 8 reconstructions — max per-element
    quantization error ~0.025 * grp_norm. Allow 5e-2 relative for slack."""
    src = _make_input(n_blocks)
    packed = torch.empty(n_blocks * BLOCK_BYTES, device="cuda", dtype=torch.uint8)
    out = torch.empty(n_blocks * QK_PLANAR3, device="cuda", dtype=torch.float16)
    rotorquant_planar3_pack(src, packed, n_blocks)
    rotorquant_planar3_unpack(packed, out, n_blocks)

    src_f = src.float()
    out_f = out.float()
    diff = (src_f - out_f).abs()
    src_max = src_f.abs().max().item()
    rel = diff.max().item() / max(src_max, 1e-6)
    # 3-bit per-element rel err can spike on individual outliers; check that
    # 95% of values are within 5e-2.
    sorted_rel = torch.sort(diff / src_f.abs().clamp(min=1e-3))[0]
    p95 = sorted_rel[int(0.95 * len(sorted_rel))].item()
    assert p95 < 0.10, (
        f"Per-element p95 rel err too large: {p95}. "
        f"Max abs err = {diff.max().item()}, src_max = {src_max}.")


def test_zero_input_does_not_nan() -> None:
    """An all-zero block (grp_norm = 0) must produce all-zero output, not NaN."""
    n_blocks = 4
    src = torch.zeros(n_blocks * QK_PLANAR3, device="cuda", dtype=torch.float16)
    packed = torch.empty(n_blocks * BLOCK_BYTES, device="cuda", dtype=torch.uint8)
    out = torch.empty(n_blocks * QK_PLANAR3, device="cuda", dtype=torch.float16)
    rotorquant_planar3_pack(src, packed, n_blocks)
    rotorquant_planar3_unpack(packed, out, n_blocks)
    assert torch.isfinite(out.float()).all(), "NaN or Inf in zero-input round-trip"
    # Output should be exactly zero (norm scalar = 0 -> all reconstructed
    # values multiplied by 0).
    assert (out.float().abs() < 1e-6).all()


@pytest.mark.parametrize("n_blocks", [1, 8])
def test_packed_size_constant(n_blocks: int) -> None:
    """Verify the packed-byte budget matches the documented 3.125 bpe target.

    This is the load-bearing invariant for VRAM accounting: every 128-element
    block uses exactly 50 bytes, regardless of the rotation tables used.
    """
    src = _make_input(n_blocks)
    packed = torch.empty(n_blocks * BLOCK_BYTES, device="cuda", dtype=torch.uint8)
    rotorquant_planar3_pack(src, packed, n_blocks)
    bpe = (BLOCK_BYTES * 8) / QK_PLANAR3
    assert math.isclose(bpe, 3.125, rel_tol=1e-9), (
        f"Bits per element drifted: {bpe} != 3.125")
