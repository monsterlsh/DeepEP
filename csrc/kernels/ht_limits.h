// Copyright (c) 2026, Lu Lu

#pragma once

namespace ascend_deepep {
// Public shape limits of the Elastic HT path. Kernel tiling and workspace
// layout are already parameterized; these bounds only gate the host entries.
// Keep in sync with deep_ep/backends/mesh_clos/_elastic_layout.py.
inline constexpr int kHtMaxTokensPerRank = 65536;
inline constexpr int kHtMaxTopk = 32;
}  // namespace ascend_deepep
