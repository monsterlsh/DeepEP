// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026

#include "forward.hpp"
#include "shmem.h"
#include "kernels/mega_moe_common/common/ascend_timer_v2.hpp"
#include "kernels/mega_moe_common/common/mega_moe_constants.hpp"
#include "kernels/mega_moe_urma/mega_moe_urma_tiling.hpp"

namespace py = pybind11;

#ifdef DEEPEP_MEGAMOE_MTE_ROCE
namespace deepep::megamoe::roce {
void bind(pybind11::module_& module);
}
#endif

PYBIND11_MODULE(_C, module)
{
    using deepep::megamoe::Plan;
    module.attr("api_protocol") = 13;
    module.attr("small_batch_max_tokens") = MegaMoeImpl::W4A8_SMALL_BATCH_MAX_TOKENS;
    module.attr("urma_ep8_window_tokens") = MegaMoeUrma::kWindowTokens;
    module.attr("urma_w8a8_enabled") = true;
    module.attr("mte_a4w4_enabled") = true;
    module.attr("urma_w8a8_swiglu_enabled") = true;
    module.attr("urma_windowed_swiglu_enabled") = true;
#ifdef DEEPEP_MEGAMOE_MTE_ROCE
    deepep::megamoe::roce::bind(module);
#else
    module.attr("mte_roce_enabled") = false;
#endif
    module.def("buffer_size", &deepep::megamoe::buffer_size, py::arg("world"), py::arg("experts"), py::arg("tokens"),
               py::arg("topk"), py::arg("hidden"), py::arg("backend") = "mte", py::arg("topk_weights_type") = 0,
               py::arg("mma_type") = "", py::arg("windowed") = false);
    module.def("mte_plan_bytes", &deepep::megamoe::mte_plan_bytes,
               py::arg("world"), py::arg("experts"), py::arg("tokens"), py::arg("capacity"), py::arg("topk"),
               py::arg("hidden"), py::arg("intermediate"), py::arg("activation"), py::arg("mma_type"),
               py::arg("shared_experts") = 0, py::arg("topk_weights_type") = -1);
    module.def("urma_window_bytes", &deepep::megamoe::urma_window_bytes, py::arg("world"), py::arg("local_experts"),
               py::arg("tokens"), py::arg("topk"), py::arg("hidden"), py::arg("intermediate"),
               py::arg("mma_type") = "fp8xfp4");
    module.def("allocate_timer", &deepep::megamoe::allocate_timer, py::arg("example"), py::arg("backend") = "mte");
    module.def("validate_timer", &deepep::megamoe::validate_timer, py::arg("timer"), py::arg("backend"));
    py::class_<Plan>(module, "Plan")
        .def(py::init<torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, bool, int64_t, int64_t, uint64_t,
                      uint64_t, int64_t, const std::string&, int64_t, const std::string&, const std::string&, bool, int64_t>(),
             py::arg("example"), py::arg("intermediate"), py::arg("topk"), py::arg("experts"), py::arg("world"),
             py::arg("max_tokens"), py::arg("is_a8w4"), py::arg("shared_experts"), py::arg("ranks_per_node"),
             py::arg("ffts"), py::arg("symmetric_address"), py::arg("symmetric_bytes"), py::arg("backend") = "mte",
             py::arg("topk_weights_type") = -1, py::arg("activation") = "swiglu", py::arg("mma_type") = "",
             py::arg("windowed") = false, py::arg("timeout_seconds") = 120)
        .def("forward", &Plan::forward, py::arg("x"), py::arg("x_scales"), py::arg("topk_ids"), py::arg("topk_weights"),
             py::arg("weight1"), py::arg("weight2"), py::arg("weight_scales1"), py::arg("weight_scales2"),
             py::arg("timer"), py::arg("shared_weight1"), py::arg("shared_weight2"), py::arg("shared_weight_scales1"),
             py::arg("shared_weight_scales2"), py::arg("out"), py::arg("generation") = 0, py::arg("activation_clamp") = std::numeric_limits<float>::max())
        .def_property_readonly("workspace_bytes", &Plan::workspace_bytes);
    module.def("ffts_config", [] { return shmemx_get_ffts_config(); });
    module.attr("timer_numel") = AscendTimer::TOTAL_BUFFER_SIZE;
#ifdef DEEPEP_MEGAMOE_TIMER
    module.attr("timer_enabled") = true;
#else
    module.attr("timer_enabled") = false;
#endif
}
