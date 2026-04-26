// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the rq-models project
//
// Torch extension bindings for rq-models RotorQuant planar3 KV kernels.
// Exposes two ops to Python:
//   torch.ops.rq_models.rotorquant_planar3_pack(src_f16, dst_uint8, n_blocks)
//   torch.ops.rq_models.rotorquant_planar3_unpack(src_uint8, dst_f16, n_blocks)
//
// These are LOW-LEVEL dispatch ops that operate on flat tensors. The
// vLLM-side glue in vllm/v1/attention/ops/rotorquant_kv.py is responsible
// for unpacking the [num_tokens, num_kv_heads, head_size] -> flat
// [n_blocks * 128] mapping and re-shaping the output back. Keeping this
// op signature minimal lets us iterate on the per-token block-table
// integration without touching the kernel.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace rq_models {
namespace rotorquant {

// Forward declarations of the launchers from planar3_kv.cu.
struct BlockPlanar3;  // opaque to the binding TU

extern "C" {
void rq_planar3_pack_f16(
    const __half* d_src,
    BlockPlanar3* d_dst,
    int64_t n_blocks,
    cudaStream_t stream);

void rq_planar3_unpack_f16(
    const BlockPlanar3* d_src,
    __half* d_dst,
    int64_t n_blocks,
    cudaStream_t stream);
}

constexpr int64_t QK_PLANAR3   = 128;
constexpr int64_t BLOCK_BYTES  = 50;

// Pack: flat fp16 (length n_blocks * 128) -> packed uint8 (length n_blocks * 50).
// Args:
//   src    fp16 CUDA tensor, contiguous, length n_blocks * 128
//   dst    uint8 CUDA tensor, contiguous, length n_blocks * 50
//   n_blocks   number of 128-element blocks to encode
void rotorquant_planar3_pack(
    const torch::Tensor& src,
    torch::Tensor& dst,
    int64_t n_blocks)
{
    TORCH_CHECK(src.is_cuda(), "src must be a CUDA tensor");
    TORCH_CHECK(dst.is_cuda(), "dst must be a CUDA tensor");
    TORCH_CHECK(src.dtype() == torch::kFloat16, "src must be float16");
    TORCH_CHECK(dst.dtype() == torch::kUInt8,   "dst must be uint8");
    TORCH_CHECK(src.is_contiguous(), "src must be contiguous");
    TORCH_CHECK(dst.is_contiguous(), "dst must be contiguous");
    TORCH_CHECK(src.numel() == n_blocks * QK_PLANAR3,
                "src.numel() ", src.numel(),
                " != n_blocks * 128 = ", n_blocks * QK_PLANAR3);
    TORCH_CHECK(dst.numel() == n_blocks * BLOCK_BYTES,
                "dst.numel() ", dst.numel(),
                " != n_blocks * 50 = ", n_blocks * BLOCK_BYTES);

    auto stream = at::cuda::getCurrentCUDAStream(src.get_device()).stream();
    rq_planar3_pack_f16(
        reinterpret_cast<const __half*>(src.data_ptr<at::Half>()),
        reinterpret_cast<BlockPlanar3*>(dst.data_ptr<uint8_t>()),
        n_blocks,
        stream);
}

// Unpack: packed uint8 (length n_blocks * 50) -> fp16 (length n_blocks * 128).
void rotorquant_planar3_unpack(
    const torch::Tensor& src,
    torch::Tensor& dst,
    int64_t n_blocks)
{
    TORCH_CHECK(src.is_cuda(), "src must be a CUDA tensor");
    TORCH_CHECK(dst.is_cuda(), "dst must be a CUDA tensor");
    TORCH_CHECK(src.dtype() == torch::kUInt8,   "src must be uint8");
    TORCH_CHECK(dst.dtype() == torch::kFloat16, "dst must be float16");
    TORCH_CHECK(src.is_contiguous(), "src must be contiguous");
    TORCH_CHECK(dst.is_contiguous(), "dst must be contiguous");
    TORCH_CHECK(src.numel() == n_blocks * BLOCK_BYTES,
                "src.numel() ", src.numel(),
                " != n_blocks * 50 = ", n_blocks * BLOCK_BYTES);
    TORCH_CHECK(dst.numel() == n_blocks * QK_PLANAR3,
                "dst.numel() ", dst.numel(),
                " != n_blocks * 128 = ", n_blocks * QK_PLANAR3);

    auto stream = at::cuda::getCurrentCUDAStream(src.get_device()).stream();
    rq_planar3_unpack_f16(
        reinterpret_cast<const BlockPlanar3*>(src.data_ptr<uint8_t>()),
        reinterpret_cast<__half*>(dst.data_ptr<at::Half>()),
        n_blocks,
        stream);
}

}  // namespace rotorquant
}  // namespace rq_models

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rotorquant_planar3_pack",
          &rq_models::rotorquant::rotorquant_planar3_pack,
          "rq-models RotorQuant planar3 KV pack (fp16 -> 3-bit packed)");
    m.def("rotorquant_planar3_unpack",
          &rq_models::rotorquant::rotorquant_planar3_unpack,
          "rq-models RotorQuant planar3 KV unpack (3-bit packed -> fp16)");
}
