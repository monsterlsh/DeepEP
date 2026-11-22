// Copyright (c) 2026, Lu Lu

#pragma once

namespace ascend_deepep {
// Shape limits shared by Elastic HT host validation and device routing.
// Keep in sync with deep_ep/backends/mesh_clos/_elastic_layout.py.
inline constexpr int kHtMaxTokensPerRank = 65536;
inline constexpr int kHtMaxTopk = 32;
}  // namespace ascend_deepep
