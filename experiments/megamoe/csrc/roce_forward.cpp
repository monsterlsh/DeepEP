// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026

#include "roce_forward.hpp"
#include "timer.hpp"
#include "kernels/mega_moe_hierarchical/kernel_launch.hpp"

#include <cstdint>
#include <limits>

#include <acl/acl.h>
#include "shmem.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"

#include "check.hpp"
#include "kernels/mega_moe_hierarchical/mc2_common/mc2_moe_context.hpp"

namespace deepep::megamoe::roce {

Plan::Plan(torch::Tensor example, int64_t intermediate, int64_t topk, int64_t experts, int64_t world,
           int64_t max_tokens, int64_t shared_experts, uint64_t ffts, uint64_t symmetric_address,
           int64_t symmetric_bytes, const std::string& activation, const std::string& mma_type,
           int64_t topk_weights_type)
    : ffts_(ffts)
{
    using namespace HierarchicalMegaMoeImpl;
    CheckTensor(example, 2, "plan input");
    const auto mma_mode = MmaMode(mma_type);
    TORCH_CHECK(example.scalar_type() == torch::kBFloat16 || example.scalar_type() == c10::ScalarType::Float8_e4m3fn,
                "MTE+RoCE plan input must be BF16 or match the quantized activation type");
    const int64_t h = example.size(1);
    platform_ = GetPlatformInfo();
    tiling_ = MakeTiling(example.size(0), h, intermediate, topk, experts, world, max_tokens, shared_experts,
                         platform_, activation, mma_mode, topk_weights_type);
    const auto required = LayoutArena(world, experts, max_tokens, topk, h, HIERARCHICAL_RANKS_PER_HOST, 0, 1,
                                      topk_weights_type == 1, tiling_.arena);
    TORCH_CHECK(symmetric_address != 0 && symmetric_address % HIERARCHICAL_FLAG_BYTES == 0 && symmetric_bytes >= required,
                "MTE+RoCE requires a 512-byte aligned symmetric address and ", required, " bytes, got ", symmetric_bytes);
    int32_t device = -1;
    ASCEND_DEEPEP_CHECK_ACL(aclrtGetDevice(&device));
    TORCH_CHECK(example.device().index() == device, "MTE+RoCE plan input must be on the current NPU");
    TORCH_CHECK(aclshmem_n_pes() == world, "MTE+RoCE plan must match the initialized SHMEM world size");
    static_assert(sizeof(HierarchicalMegaMoeTilingData) == 368, "MegaMoE tiling ABI changed");
    workspace_ = torch::empty({WorkspaceBytes(tiling_, platform_)}, example.options().dtype(torch::kUInt8));
    context_ = torch::empty({static_cast<int64_t>(sizeof(HierarchicalMc2Aclnn::Mc2MoeContext))}, workspace_.options());

    // peer 地址和 context 只在准备阶段解析，不在每次 forward 中重新分配、同步拷贝。
    HierarchicalMc2Aclnn::Mc2MoeContext context{};
    const int rank = aclshmem_my_pe();
    TORCH_CHECK(rank >= 0 && rank < world && world <= HierarchicalMc2Aclnn::HCCL_MAX_RANK_SIZE, "Invalid SHMEM rank/world: ", rank,
                "/", world);
    context.epRankId = static_cast<uint32_t>(rank);
    context.rankSizePerServer = static_cast<uint32_t>(world);
    for (int64_t peer = 0; peer < world; ++peer) {
        if (HierarchicalMegaMoeImpl::HierarchicalHost(static_cast<uint32_t>(peer)) !=
                HierarchicalMegaMoeImpl::HierarchicalHost(static_cast<uint32_t>(rank))) {
            // Cross-host operations use the local symmetric address plus PE, never
            // an address from the MTE peer table. Keep these entries deliberately null.
            continue;
        }
        void* address = aclshmem_ptr(reinterpret_cast<void*>(symmetric_address), static_cast<int32_t>(peer));
        TORCH_CHECK(address != nullptr, "rank=", rank, " prepare: cannot resolve peer=", peer);
        context.epHcclBuffer_[peer] = reinterpret_cast<uint64_t>(address);
    }
    ASCEND_DEEPEP_CHECK_ACL(
        aclrtMemcpy(context_.data_ptr(), sizeof(context), &context, sizeof(context), ACL_MEMCPY_HOST_TO_DEVICE));
}

std::tuple<torch::Tensor, torch::Tensor> Plan::forward(
    torch::Tensor x, c10::optional<torch::Tensor> x_scales, torch::Tensor topk_ids, torch::Tensor topk_weights,
    torch::Tensor weight1, torch::Tensor weight2, torch::Tensor weight_scales1, torch::Tensor weight_scales2,
    c10::optional<torch::Tensor> timer, c10::optional<torch::Tensor> shared_weight1,
    c10::optional<torch::Tensor> shared_weight2, c10::optional<torch::Tensor> shared_weight_scales1,
    c10::optional<torch::Tensor> shared_weight_scales2, c10::optional<torch::Tensor> out,
    uint64_t sequence, float activation_clamp)
{
    using namespace HierarchicalMegaMoeImpl;
    CheckTensor(x, 2, "x");
    CheckTensor(topk_ids, 2, "topk_ids");
    CheckTensor(topk_weights, 2, "topk_weights");
    CheckTensor(weight1, 3, "weight1");
    CheckTensor(weight2, 3, "weight2");
    CheckTensor(weight_scales1, 4, "weight_scales1");
    CheckTensor(weight_scales2, 4, "weight_scales2");
    const bool prequantized = x_scales.has_value();
    const auto mma_mode = tiling_.groupedMatmulMode;
    TORCH_CHECK(x.scalar_type() == (prequantized ? c10::ScalarType::Float8_e4m3fn : torch::kBFloat16),
                "x must be BF16 or the plan's quantized activation dtype with E8M0 scales");
    const int64_t bs = tiling_.bs, h = tiling_.h, topk = tiling_.topK;
    const int64_t experts = tiling_.moeExpertPerRank, hidden = tiling_.hiddenDim;
    const int64_t intermediate = IntermediateWidth(tiling_);
    TORCH_CHECK(x.device() == workspace_.device() && x.sizes() == torch::IntArrayRef({bs, h}),
                "Input shape and device must match the MTE+RoCE plan");
    if (prequantized) {
        CheckTensor(*x_scales, 2, "x_scales");
        TORCH_CHECK(x_scales->scalar_type() == c10::ScalarType::Float8_e8m0fnu &&
                        x_scales->sizes() == torch::IntArrayRef({bs, h / 32}),
                    "x_scales must be E8M0 [BS, H/32]");
    }
    TORCH_CHECK(topk_ids.scalar_type() == torch::kInt32, "topk_ids must be int32");
    TORCH_CHECK(topk_weights.scalar_type() == torch::kFloat32, "topk_weights must be float32");
    TORCH_CHECK(topk_ids.sizes() == torch::IntArrayRef({bs, topk}) && topk_weights.sizes() == topk_ids.sizes(),
                "top-k tensors must match the plan's [BS, K]");
    const bool fp8_weights = mma_mode == GROUPED_MATMUL_MODE_GENERAL;
    const auto weight_dtype = fp8_weights ? c10::ScalarType::Float8_e4m3fn : torch::kUInt8;
    const int64_t weight_elements_per_byte = fp8_weights ? 1 : 2;
    TORCH_CHECK(weight1.scalar_type() == weight_dtype && weight2.scalar_type() == weight_dtype,
                "Routed weight dtype must match the MTE+RoCE plan");
    TORCH_CHECK(weight_scales1.scalar_type() == c10::ScalarType::Float8_e8m0fnu &&
                    weight_scales2.scalar_type() == c10::ScalarType::Float8_e8m0fnu,
                "weight scales must be FP8 E8M0 tensors");
    for (const auto* tensor : {&topk_ids, &topk_weights, &weight1, &weight2, &weight_scales1, &weight_scales2}) {
        TORCH_CHECK(tensor->device() == x.device(), "MegaMoE tensors must be on the same NPU");
        TORCH_CHECK(!out || !out->is_alias_of(*tensor), "out must not alias routed inputs");
    }
    for (const auto* tensor : {&x_scales, &out}) {
        TORCH_CHECK(!tensor->has_value() || tensor->value().device() == x.device(),
                    "optional MegaMoE tensors must be on the same NPU");
    }
    TORCH_CHECK(!out || (!out->is_alias_of(x) && (!x_scales || !out->is_alias_of(*x_scales))),
                "out must not alias x or x_scales");
    TORCH_CHECK(weight1.sizes() == torch::IntArrayRef({experts, hidden, h / weight_elements_per_byte}) &&
                    weight2.sizes() == torch::IntArrayRef({experts, h, intermediate / weight_elements_per_byte}),
                "GMM1 and GMM2 weight shapes must match the MTE+RoCE plan's activation and packing");
    TORCH_CHECK(weight_scales1.sizes() == torch::IntArrayRef({experts, hidden, CeilDiv(h, int64_t{64}), 2}),
                "weight_scales1 shape must be [local_experts, hiddenDim, ceil(h/64), 2]");
    TORCH_CHECK(weight_scales2.sizes() == torch::IntArrayRef({experts, h, CeilDiv(intermediate, int64_t{64}), 2}),
                "weight_scales2 shape must be [local_experts, h, ceil(intermediate_hidden/64), 2]");

    const bool has_shared = shared_weight1.has_value();
    TORCH_CHECK(shared_weight2.has_value() == has_shared && shared_weight_scales1.has_value() == has_shared &&
                    shared_weight_scales2.has_value() == has_shared,
                "all four shared weight and scale tensors must be supplied together");
    int64_t shared_experts = 0;
    if (has_shared) {
        CheckTensor(*shared_weight1, 3, "shared_weight1");
        CheckTensor(*shared_weight2, 3, "shared_weight2");
        CheckTensor(*shared_weight_scales1, 4, "shared_weight_scales1");
        CheckTensor(*shared_weight_scales2, 4, "shared_weight_scales2");
        shared_experts = shared_weight1->size(0);
        TORCH_CHECK(shared_experts > 0 && shared_experts <= std::numeric_limits<uint32_t>::max(),
                    "shared expert count must be positive and fit uint32");
        for (const auto* tensor :
             {&*shared_weight1, &*shared_weight2, &*shared_weight_scales1, &*shared_weight_scales2}) {
            TORCH_CHECK(tensor->device() == x.device(), "shared tensors must be on the same device as x");
            TORCH_CHECK(!out || !out->is_alias_of(*tensor), "out must not alias shared inputs");
        }
        const bool fp8 = shared_weight1->scalar_type() == c10::ScalarType::Float8_e4m3fn;
        TORCH_CHECK((fp8 || shared_weight1->scalar_type() == torch::kUInt8) &&
                        shared_weight1->scalar_type() == shared_weight2->scalar_type(),
                    "invalid shared projection types");
        const int64_t packing = fp8 ? 1 : 2;
        TORCH_CHECK(shared_weight_scales1->scalar_type() == c10::ScalarType::Float8_e8m0fnu &&
                        shared_weight_scales2->scalar_type() == c10::ScalarType::Float8_e8m0fnu,
                    "shared weight scales must be FP8 E8M0");
        TORCH_CHECK(shared_weight1->sizes() == torch::IntArrayRef({shared_experts, hidden, h / packing}) &&
                        shared_weight2->sizes() == torch::IntArrayRef({shared_experts, h, intermediate / packing}),
                    "shared weights must match routed H and intermediate widths");
        TORCH_CHECK(shared_weight_scales1->sizes() ==
                            torch::IntArrayRef({shared_experts, hidden, CeilDiv(h, int64_t{64}), 2}) &&
                        shared_weight_scales2->sizes() ==
                            torch::IntArrayRef({shared_experts, h, CeilDiv(intermediate, int64_t{64}), 2}),
                    "shared scales must match [S, N, ceil(K/64), 2] for each shared weight");
    }

    TORCH_CHECK(shared_experts == tiling_.sharedExpertNum, "Shared experts differ from the prepared plan");
    torch::Tensor y = out.has_value() ? *out : torch::empty({bs, h}, x.options().dtype(torch::kBFloat16));
    CheckTensor(y, 2, "out");
    TORCH_CHECK(y.scalar_type() == torch::kBFloat16 && y.sizes() == torch::IntArrayRef({bs, h}),
                "out must be BF16 [BS, H]");
    torch::Tensor expert_tokens = torch::empty({experts}, x.options().dtype(torch::kInt32));

    TORCH_CHECK(sequence > 0 && sequence < static_cast<uint32_t>(std::numeric_limits<int32_t>::max()),
                "RoCE invocation sequence exhausted; recreate SymmBuffer collectively");
    TORCH_CHECK(activation_clamp >= 0 && activation_clamp <= std::numeric_limits<float>::max(),
                "invalid activation clamp");
    tiling_.sharedFp8 = has_shared && shared_weight1->scalar_type() == c10::ScalarType::Float8_e4m3fn;
    SetWaveSize(tiling_);
    tiling_.clampLimit = activation_clamp;
    tiling_.hierarchicalSequence = static_cast<uint32_t>(sequence);
    const auto stream = c10_npu::getCurrentNPUStream();
    TORCH_CHECK(stream.device_index() == x.device().index(), "MTE+RoCE current NPU must match the prepared plan");
    if (timer) {
        deepep::megamoe::validate_timer(*timer, "mte_roce");
        TORCH_CHECK(timer->device() == x.device(), "MTE+RoCE timer must be on the input device");
        for (const auto* tensor : {&x, &topk_ids, &topk_weights, &weight1, &weight2, &weight_scales1,
                                   &weight_scales2, &y, &workspace_, &context_}) {
            TORCH_CHECK(!timer->is_alias_of(*tensor), "MTE+RoCE timer must have separate storage");
        }
        for (const auto* tensor : {&x_scales, &shared_weight1, &shared_weight2, &shared_weight_scales1,
                                   &shared_weight_scales2}) {
            TORCH_CHECK(!*tensor || !timer->is_alias_of(**tensor), "MTE+RoCE timer must have separate storage");
        }
        ASCEND_DEEPEP_CHECK_ACL(
            aclrtMemsetAsync(timer->data_ptr(), timer->nbytes(), 0, timer->nbytes(), stream.stream()));
        c10_npu::NPUCachingAllocator::recordStream(timer->storage().data_ptr(), stream);
    }
    auto ptr = [](const torch::Tensor& tensor) { return static_cast<uint8_t*>(tensor.data_ptr()); };
    auto opt = [&](const c10::optional<torch::Tensor>& tensor) { return tensor ? ptr(*tensor) : nullptr; };
    mega_moe_hierarchical_kernel_do(
        platform_.block_dim, platform_.ub, stream.stream(), mma_mode == GROUPED_MATMUL_MODE_A8W4,
        ptr(context_), ptr(x), opt(x_scales),
        ptr(topk_ids), ptr(topk_weights), ptr(weight1), ptr(weight2), ptr(weight_scales1), ptr(weight_scales2),
        opt(shared_weight1), opt(shared_weight2), opt(shared_weight_scales1), opt(shared_weight_scales2),
        ptr(y), ptr(expert_tokens), ptr(workspace_), opt(timer), reinterpret_cast<const uint8_t*>(&tiling_), ffts_);
    // The direct kernel launch bypasses the PyTorch dispatcher; record that
    // every argument remains live until the current NPU stream reaches it.
    for (const auto* tensor : {&x, &topk_ids, &topk_weights, &weight1, &weight2, &weight_scales1, &weight_scales2,
                               &workspace_, &context_, &y, &expert_tokens}) {
        c10_npu::NPUCachingAllocator::recordStream(tensor->storage().data_ptr(), stream);
    }
    if (has_shared) {
        for (const auto* tensor :
             {&*shared_weight1, &*shared_weight2, &*shared_weight_scales1, &*shared_weight_scales2}) {
            c10_npu::NPUCachingAllocator::recordStream(tensor->storage().data_ptr(), stream);
        }
    }
    if (x_scales) {
        c10_npu::NPUCachingAllocator::recordStream(x_scales->storage().data_ptr(), stream);
    }
    return {y, expert_tokens};
}

int64_t buffer_size(int64_t world, int64_t experts, int64_t tokens, int64_t topk, int64_t hidden,
                    int64_t topk_weights_type, const std::string& mma_type)
{
    using namespace HierarchicalMegaMoeImpl;
    CheckArenaGeometry(world, experts, tokens, topk, hidden, topk_weights_type);
    MmaMode(mma_type);
    HierarchicalMegaMoeArenaOffsets layout{};
    return LayoutArena(world, experts, tokens, topk, hidden, HIERARCHICAL_RANKS_PER_HOST, 0, 1,
                       topk_weights_type == 1, layout);
}

int64_t plan_bytes(int64_t world, int64_t experts, int64_t tokens, int64_t capacity, int64_t topk, int64_t hidden,
                   int64_t intermediate, const std::string& activation, const std::string& mma_type,
                   int64_t topk_weights_type, int64_t shared_experts)
{
    const auto platform = GetPlatformInfo();
    const auto tiling = MakeTiling(tokens, hidden, intermediate, topk, experts, world, capacity, shared_experts,
                                   platform, activation, MmaMode(mma_type), topk_weights_type);
    return WorkspaceBytes(tiling, platform);
}

void bind(pybind11::module_& module)
{
    namespace py = pybind11;
    module.attr("mte_roce_enabled") = true;
    module.attr("mte_roce_mx_enabled") = true;
#ifdef DEEPEP_MEGAMOE_TIMER
    module.attr("mte_roce_timer_enabled") = true;
#else
    module.attr("mte_roce_timer_enabled") = false;
#endif
    module.def("roce_buffer_size", &buffer_size, py::arg("world"), py::arg("experts"), py::arg("tokens"),
               py::arg("topk"), py::arg("hidden"), py::arg("topk_weights_type") = 0,
               py::arg("mma_type") = "fp8xfp4");
    module.def("roce_plan_bytes", &plan_bytes, py::arg("world"), py::arg("experts"), py::arg("tokens"),
               py::arg("capacity"), py::arg("topk"), py::arg("hidden"), py::arg("intermediate"),
               py::arg("activation"), py::arg("mma_type"), py::arg("topk_weights_type"), py::arg("shared_experts"));
    py::class_<Plan>(module, "RocePlan")
        .def(py::init<torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, uint64_t, uint64_t, int64_t,
                      const std::string&, const std::string&, int64_t>(),
             py::arg("example"), py::arg("intermediate"), py::arg("topk"), py::arg("experts"), py::arg("world"),
             py::arg("max_tokens"), py::arg("shared_experts"), py::arg("ffts"), py::arg("symmetric_address"),
             py::arg("symmetric_bytes"), py::arg("activation") = "swiglu", py::arg("mma_type") = "fp8xfp4",
             py::arg("topk_weights_type") = 0)
        .def("forward", &Plan::forward)
        .def_property_readonly("workspace_bytes", &Plan::workspace_bytes);
}

}  // namespace deepep::megamoe::roce
