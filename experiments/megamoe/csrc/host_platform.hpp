// Copyright (c) 2026, Lu Lu

#pragma once

#include <cstdint>

namespace at {
class Tensor;
}

namespace deepep::megamoe {

struct PlatformInfo {
    uint32_t aic;
    uint32_t aiv;
    uint32_t ub;
    uint32_t block_dim;
    uint32_t system_workspace;
};

// Raw launches reserve this space in addition to the kernel and AscendC workspaces.
constexpr int64_t kReservedWorkspaceBytes = 50LL * 1024LL * 1024LL;

PlatformInfo GetPlatformInfo();
void CheckTensor(const at::Tensor& tensor, int64_t dims, const char* name);

}  // namespace deepep::megamoe
