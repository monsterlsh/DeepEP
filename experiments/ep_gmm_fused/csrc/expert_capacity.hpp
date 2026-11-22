// Copyright (c) 2026, Lu Lu

#pragma once

#include <cstdint>

namespace deepep::ep_gmm_fused {

inline constexpr uint32_t kForwardSmallLocalExperts = 64;
inline constexpr uint32_t kInverseSmallLocalExperts = 32;
inline constexpr uint32_t kMaxLocalExperts = 128;

template <class T>
struct TypeTag {
    using type = T;
};

template <class Config, class Launch>
void LaunchWithExpertCapacity(TypeTag<Config> config, uint32_t localExperts, const Launch& launch)
{
    if (localExperts <= Config::MoeConstraints::LOCAL_EXPERT_NUM_LIMIT) {
        launch(config);
    } else {
        launch(TypeTag<typename Config::template WithExpertCapacity<kMaxLocalExperts>>{});
    }
}

}  // namespace deepep::ep_gmm_fused
