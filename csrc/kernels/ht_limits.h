// Copyright (c) 2026, Lu Lu

#pragma once
#include <cstdint>

namespace ascend_deepep {
// Public shape limits of the Elastic HT path. Kernel tiling and workspace
// layout are already parameterized; these bounds only gate the host entries.
// Keep in sync with deep_ep/backends/mesh_clos/_elastic_layout.py.
inline constexpr uint32_t kHtMaxTokensPerRank = 65536U;
inline constexpr uint32_t kHtMaxTopk = 32U;
}  // namespace ascend_deepep
