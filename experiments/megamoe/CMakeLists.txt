# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

# MegaMoE is an optional Torch-bound extension. The core _C keeps its original ABI.
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/Headers.cmake")
ascend_deepep_find_mega_moe_headers()
find_program(BISHENG bisheng PATHS "${ASCEND_HOME_PATH}/tools/bisheng_compiler/bin" NO_DEFAULT_PATH REQUIRED)
execute_process(COMMAND "${Python_EXECUTABLE}" -c
    "import torch,torch_npu; from pathlib import Path; print(torch.utils.cmake_prefix_path); print(Path(torch_npu.__file__).parent)"
    OUTPUT_VARIABLE framework_paths OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
string(REPLACE "\n" ";" framework_paths "${framework_paths}")
list(GET framework_paths 0 torch_cmake)
list(GET framework_paths 1 torch_npu_root)
list(APPEND CMAKE_PREFIX_PATH "${torch_cmake}")
find_package(Torch REQUIRED)
find_library(TORCH_NPU_LIBRARY torch_npu PATHS "${torch_npu_root}/lib" NO_DEFAULT_PATH REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib" NO_DEFAULT_PATH REQUIRED)

set(beta_include
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc"
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/mega_moe"
    "${CATLASS_ROOT}/include" "${CATLASS_ROOT}/examples/common"
    "${SHMEM_ROOT}/include" "${SHMEM_ROOT}/src" "${SHMEM_ROOT}/src/device" "${SHMEM_ROOT}/src/device_simt"
    "${ASCEND_HOME_PATH}/include" "${ASCEND_HOME_PATH}/pkg_inc"
    "${ASCEND_HOME_PATH}/include/experiment/runtime"
    "${ASCEND_HOME_PATH}/pkg_inc/runtime"
    "${ASCEND_HOME_PATH}/include/op_common"
    "${ASCEND_HOME_PATH}/pkg_inc/op_common"
    "${ASCEND_HOME_PATH}/compiler/ascendc/act"
    "${ASCEND_HOME_PATH}/aarch64-linux/asc/atcos"
    "${ASCEND_DEEPEP_MEGA_MOE_COMMON_INCLUDE_DIR}"
    "${ASCEND_DEEPEP_MEGA_MOE_COMMON_INCLUDE_DIR}/tensor_api"
    "${ASCEND_DEEPEP_MEGA_MOE_COMMON_INCLUDE_DIR}/tensor_api/include")
set(kernel_flags
    -O2 -std=c++17 -xasc --npu-arch=dav-3510 --cce-auto-infer-kernel-type=false
    --cce-auto-sync=off --cce-fatobj-link -shared -fPIC
    -DCATLASS_ARCH=3510 -DASCEND_DEEPEP_DIRECT_TENSOR_INPUT=1
    -DASCEND_DEEPEP_DISABLE_MEGA_MOE_EXCEPTION_DUMP=1 -DENABLE_TENSOR_API
    -DNOT_DYNAMIC_COMPILE -DHIGH_PERFORMANCE=1 -DDETERMINISTIC_MODE=0 -DL2_CACHE_HINT)
if(DEEPEP_MEGAMOE_TIMER)
    list(APPEND kernel_flags -DDEEPEP_MEGAMOE_TIMER=1)
endif()
foreach(path IN LISTS beta_include)
    list(APPEND kernel_flags "-I${path}")
endforeach()
find_file(api_header ascendc_api_version.h
    PATHS "${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/lib"
          "${ASCEND_HOME_PATH}/aarch64-linux/tikcpp/tikcfw/lib" NO_DEFAULT_PATH)
if(api_header)
    file(STRINGS "${api_header}" api_version REGEX "^#define ASCENDC_API_VERSION [0-9]+")
    string(REGEX REPLACE ".*ASCENDC_API_VERSION ([0-9]+).*" "\\1" api_version "${api_version}")
    list(APPEND kernel_flags "-DASCENDC_API_VERSION=${api_version}")
endif()
# Bisheng compiles separate host and device passes from each entry point.
# Track device-only headers even when the host preprocessor skips them.
function(megamoe_header_dependencies result)
    file(GLOB_RECURSE headers CONFIGURE_DEPENDS ${ARGN})
    list(FILTER headers INCLUDE REGEX "\\.(h|hh|hpp|hxx|inc|inl|ipp|tpp)$")
    # A removed header must invalidate the kernel even after it leaves the glob.
    set(manifest "${CMAKE_CURRENT_BINARY_DIR}/${result}.txt")
    file(GENERATE OUTPUT "${manifest}" CONTENT "${headers}\n")
    set(${result} ${headers} "${manifest}" PARENT_SCOPE)
endfunction()
megamoe_header_dependencies(shared_headers "csrc/kernels/mega_moe_common/*"
    "${CATLASS_ROOT}/include/*" "${CATLASS_ROOT}/examples/common/*")
megamoe_header_dependencies(wave_headers "csrc/kernels/mega_moe/*")
megamoe_header_dependencies(urma_headers "csrc/kernels/mega_moe_urma/*")
megamoe_header_dependencies(hierarchical_headers "csrc/kernels/mega_moe_hierarchical/*"
    "${PROJECT_SOURCE_DIR}/csrc/common/megamoe_combine_config.hpp")
function(megamoe_device_kernel target source extra_flags)
    set(soname "lib${target}.so")
    set(library "${CMAKE_CURRENT_BINARY_DIR}/${soname}")
    add_custom_command(OUTPUT "${library}"
        COMMAND "${BISHENG}" ${extra_flags} ${kernel_flags} "-Wl,-soname,${soname}" ${source}
            "-L${ASCEND_HOME_PATH}/lib64" "-L${SHMEM_ROOT}/lib"
            -lruntime -lascendcl -lshmem -lgraph_base -ltiling_api -lplatform -lregister -ldl -lstdc++ -lm
            -o "${library}"
        DEPENDS ${source} ${ARGN} ${shared_headers}
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" VERBATIM)
    add_library(${target} SHARED IMPORTED)
    set_target_properties(${target} PROPERTIES IMPORTED_LOCATION "${library}")
    install(FILES "${library}" DESTINATION ../deep_ep_experimental/megamoe)
    set(${target}_file "${library}" PARENT_SCOPE)
endfunction()

set(hns_flags -fvisibility=hidden -DACLSHMEMI_RDMA_K_BACKEND_HNS_1825=1
    -DACLSHMEMI_HNS_1825_PI_ON_CHIP=${DEEPEP_SHMEM_HNS_1825_PI_ON_CHIP})
set(urma_flags "-I${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/mega_moe_urma"
    -DDEEPEP_MEGAMOE_URMA=1 ${hns_flags})
set(hierarchical_flags "-I${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/mega_moe_hierarchical"
    "-I${PROJECT_SOURCE_DIR}/csrc" -DDEEPEP_MEGAMOE_MTE_ROCE=1 -DDEEPEP_MEGAMOE_ROCE=1 ${hns_flags})

megamoe_device_kernel(megamoe_kernel csrc/kernels/mega_moe/mega_moe.cpp
    "" csrc/kernel_launch.hpp ${wave_headers})
megamoe_device_kernel(megamoe_urma_kernel csrc/kernels/mega_moe_urma/mega_moe_urma.cpp
    "${urma_flags}" ${urma_headers})
megamoe_device_kernel(megamoe_mte_roce_kernel csrc/kernels/mega_moe_hierarchical/mega_moe_hierarchical.cpp
    "${hierarchical_flags}" ${hierarchical_headers})
add_custom_target(megamoe_device DEPENDS "${megamoe_kernel_file}" "${megamoe_urma_kernel_file}"
    "${megamoe_mte_roce_kernel_file}")
pybind11_add_module(megamoe_ext csrc/bindings.cpp csrc/forward.cpp csrc/kernel_launch.cpp csrc/roce_forward.cpp)
set_source_files_properties(csrc/roce_forward.cpp PROPERTIES COMPILE_DEFINITIONS "DEEPEP_MEGAMOE_ROCE=1")
set_target_properties(megamoe_ext PROPERTIES OUTPUT_NAME _C
    INSTALL_RPATH "$ORIGIN" INSTALL_RPATH_USE_LINK_PATH FALSE)
target_include_directories(megamoe_ext PRIVATE ${beta_include} ${TORCH_INCLUDE_DIRS} "${torch_npu_root}/include"
    "${PROJECT_SOURCE_DIR}/csrc")
separate_arguments(torch_flags NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(megamoe_ext PRIVATE ${torch_flags} -O2 -fstack-protector-strong)
target_compile_definitions(megamoe_ext PRIVATE CATLASS_ARCH=3510 DEEPEP_MEGAMOE_MTE_ROCE=1)
if(DEEPEP_MEGAMOE_TIMER)
    target_compile_definitions(megamoe_ext PRIVATE DEEPEP_MEGAMOE_TIMER=1)
endif()
add_dependencies(megamoe_ext megamoe_device)
target_link_libraries(megamoe_ext PRIVATE ${TORCH_LIBRARIES} "${TORCH_PYTHON_LIBRARY}"
    "${TORCH_NPU_LIBRARY}" megamoe_kernel megamoe_urma_kernel megamoe_mte_roce_kernel shmem ascendcl)
target_link_directories(megamoe_ext PRIVATE "${ASCEND_HOME_PATH}/lib64")
target_link_libraries(megamoe_ext PRIVATE platform tiling_api)
target_link_options(megamoe_ext PRIVATE -Wl,-z,relro,-z,now,-z,noexecstack)
install(TARGETS megamoe_ext LIBRARY DESTINATION ../deep_ep_experimental/megamoe)
