// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026

#pragma once

#include <cstdint>
#include <string>
#include <tuple>
#include <torch/extension.h>
#include "roce_tiling.hpp"

namespace deepep::megamoe::roce {

// 一个 BS 对应一个准备好的计划；Python owner 保证同流串行复用及销毁顺序。
class Plan {
public:
    Plan(torch::Tensor example, int64_t intermediate, int64_t topk, int64_t experts, int64_t world, int64_t max_tokens,
         int64_t shared_experts, uint64_t ffts, uint64_t symmetric_address, int64_t symmetric_bytes,
         const std::string& activation, const std::string& mma_type, int64_t topk_weights_type);

    std::tuple<torch::Tensor, torch::Tensor> forward(
        torch::Tensor x, c10::optional<torch::Tensor> x_scales, torch::Tensor topk_ids, torch::Tensor topk_weights,
        torch::Tensor weight1, torch::Tensor weight2, torch::Tensor weight_scales1, torch::Tensor weight_scales2,
        c10::optional<torch::Tensor> timer, c10::optional<torch::Tensor> shared_weight1,
        c10::optional<torch::Tensor> shared_weight2, c10::optional<torch::Tensor> shared_weight_scales1,
        c10::optional<torch::Tensor> shared_weight_scales2, c10::optional<torch::Tensor> out,
        uint64_t sequence, float activation_clamp);

    int64_t workspace_bytes() const
    {
        return workspace_.numel();
    }

private:
    PlatformInfo platform_;
    HierarchicalMegaMoeTilingData tiling_;
    torch::Tensor workspace_;
    torch::Tensor context_;
    uint64_t ffts_;
};

int64_t buffer_size(int64_t world, int64_t experts, int64_t tokens, int64_t topk, int64_t hidden,
                    int64_t topk_weights_type, const std::string& mma_type);

int64_t plan_bytes(int64_t world, int64_t experts, int64_t tokens, int64_t capacity, int64_t topk, int64_t hidden,
                   int64_t intermediate, const std::string& activation, const std::string& mma_type,
                   int64_t topk_weights_type, int64_t shared_experts);

}  // namespace deepep::megamoe::roce
