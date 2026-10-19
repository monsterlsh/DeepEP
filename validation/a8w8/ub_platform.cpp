// Query the same CANN platform API used by MegaMoE, without modifying its kernel.
#include <acl/acl.h>
#include <cstdint>
#include <iostream>
#include "tiling/platform/platform_ascendc.h"

int main()
{
    if (aclInit(nullptr) != ACL_SUCCESS) return 2;
    if (aclrtSetDevice(0) != ACL_SUCCESS) return 3;
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (platform == nullptr) return 4;
    uint64_t ub = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    const auto aic = platform->GetCoreNumAic();
    const auto aiv = platform->GetCoreNumAiv();
    std::cout << "UBPROBE_JSON={\"ub_bytes\":" << ub
              << ",\"aic\":" << aic << ",\"aiv\":" << aiv
              << ",\"block_dim\":" << platform->CalcTschBlockDim(aiv, aic, aiv) << "}\n";
    aclrtResetDevice(0);
    aclFinalize();
    return ub > 0 && aic > 0 && aiv > 0 ? 0 : 5;
}
