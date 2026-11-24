// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once

#include "kernels/mega_moe_common/implementation_namespace.hpp"
#include "kernel_operator.h"

namespace DEEPEP_MEGAMOE_IMPL_NAMESPACE::GmmKernel {
template <typename T>
__aicore__ inline constexpr T ScalarMin(T a, T b)
{
    return a < b ? a : b;
}
}  // namespace DEEPEP_MEGAMOE_IMPL_NAMESPACE::GmmKernel
