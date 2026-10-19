// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once

#include <cstdint>

namespace MegaMoeImpl::Fp4Expanded {

// Exact BF16 midpoints between E2M1 values. Ties alternate lower/upper
// according to the low bit of the E2M1 code (round to nearest, ties to even).
constexpr uint16_t kQuarter = 0x3e80;
constexpr uint16_t kThreeQuarters = 0x3f40;
constexpr uint16_t kOneAndQuarter = 0x3fa0;
constexpr uint16_t kOneAndThreeQuarters = 0x3fe0;
constexpr uint16_t kTwoAndHalf = 0x4020;
constexpr uint16_t kThreeAndHalf = 0x4060;
constexpr uint16_t kFive = 0x40a0;

// The output is the exact E4M3 encoding of the rounded E2M1 value, not an
// E4M3 quantization. E8M0 scales are unchanged, including near their endpoints.
constexpr uint8_t EncodeScaledBf16(uint16_t bits)
{
    const uint16_t magnitude = bits & 0x7fffU;
    const uint8_t sign = static_cast<uint8_t>((bits & 0x8000U) >> 8U);
    uint8_t value = 0;
    if (magnitude > kQuarter) value = 0x30;
    if (magnitude >= kThreeQuarters) value = 0x38;
    if (magnitude > kOneAndQuarter) value = 0x3c;
    if (magnitude >= kOneAndThreeQuarters) value = 0x40;
    if (magnitude > kTwoAndHalf) value = 0x44;
    if (magnitude >= kThreeAndHalf) value = 0x48;
    if (magnitude > kFive) value = 0x4c;
    if (magnitude > 0x7f80U) value = 0x7f;
    return sign | value;
}

constexpr uint8_t ExpandE2M1(uint8_t code)
{
    const uint8_t magnitude = code & 7U;
    const uint8_t value =
        magnitude >= 2U ? static_cast<uint8_t>((magnitude << 2U) + 0x30U) : (magnitude == 1U ? 0x30U : 0U);
    return static_cast<uint8_t>(((code & 8U) << 4U) | value);
}

}  // namespace MegaMoeImpl::Fp4Expanded
