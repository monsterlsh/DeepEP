// Copyright (c) 2026, Lu Lu

#pragma once

#include <string>

namespace at {
class Tensor;
}

namespace deepep::megamoe {

at::Tensor allocate_timer(at::Tensor example, const std::string& backend = "mte");
void validate_timer(const at::Tensor& timer, const std::string& backend);

}  // namespace deepep::megamoe
