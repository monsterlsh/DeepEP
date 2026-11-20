#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

#include "kernels/mega_moe_common/common/mega_moe_constants.hpp"

namespace deepep::megamoe {

// The directory and the distinct pages must fit together in the AIV lookup buffer.
inline std::vector<uint32_t> PackGeluLookupTable(const std::vector<uint32_t>& factors)
{
    using namespace MegaMoeImpl;
    if (factors.size() != MTE_GELU_LUT_KEY_COUNT) {
        return {};
    }
    using Page = std::array<uint32_t, MTE_GELU_LUT_PAGE_SIZE>;
    std::map<Page, uint32_t> offsets;
    constexpr std::size_t pageCount = MTE_GELU_LUT_KEY_COUNT / MTE_GELU_LUT_PAGE_SIZE;
    std::vector<uint32_t> packed(pageCount);
    packed.reserve(MTE_GELU_LUT_WORD_CAPACITY);
    for (std::size_t i = 0; i < pageCount; ++i) {
        Page page;
        std::copy_n(factors.begin() + i * page.size(), page.size(), page.begin());
        auto [it, fresh] = offsets.emplace(page, static_cast<uint32_t>(packed.size()));
        if (fresh) {
            if (packed.size() + page.size() > MTE_GELU_LUT_WORD_CAPACITY) {
                return {};
            }
            packed.insert(packed.end(), page.begin(), page.end());
        }
        packed[i] = it->second;
    }
    return packed;
}

}  // namespace deepep::megamoe
