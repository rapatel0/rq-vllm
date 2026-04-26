# rq-vllm

This is the rq-models fork of [vLLM](https://github.com/vllm-project/vllm),
maintained by Ravi Patel for adding **RotorQuant KV cache compression** as
a vLLM-native quantization plug-in.

The fork is the working substrate for [rq-models](https://github.com/rapatel0/rq-models)
Sprint 004 onward.

## Purpose

rq-models's RotorQuant KV cache compression scheme (iso3, planar3,
planar4, iso4 — see the rq-models repo) currently runs on a llama.cpp
fork. This vLLM fork is the second substrate target, motivated by:

- **Continuous batching** — vLLM's PagedAttention + scheduler is
  materially stronger than llama.cpp's parallel mode for multi-tenant
  serving.
- **Speculative decoding** — vLLM has a maintained plug-in architecture
  (Eagle-3, DFlash via Speculators, n-gram lookup). Required for future
  rq-models speculation work.
- **Multi-GPU** — vLLM tensor + pipeline parallelism is years ahead of
  llama.cpp's split-mode.

## Pinning

This fork is pinned to upstream vLLM **v0.19.1** (released
2026-04-18). The `feature/rotorquant` branch is the working branch for
all RotorQuant additions.

Rebasing onto a newer upstream tag is its own per-sprint task; not
automated.

## Status

- Sprint 004 (current): bring-up + planar3 kernel port end-to-end.
- See [rq-models/docs/sprints/SPRINT-004.md](https://github.com/rapatel0/rq-models/blob/main/docs/sprints/SPRINT-004.md)
  for the active sprint plan.

## Repository layout

This fork follows upstream vLLM's structure. RotorQuant additions live
under:

- `vllm/model_executor/layers/quantization/rotorquant.py` — config + quantize-method classes (added in Sprint 004 Phase 1).
- `vllm/model_executor/layers/quantization/kernels/rotorquant/` — CUDA kernels (added in Sprint 004 Phase 2 and onward).
- `tests/quantization/test_rotorquant.py` — parity tests against the llama.cpp baseline.

`README.md` (upstream's) is left unmodified to minimize merge conflicts
on rebase. This file (`ROTORQUANT.md`) is the fork-specific entry point.

## License

vLLM is Apache 2.0; this fork inherits.
