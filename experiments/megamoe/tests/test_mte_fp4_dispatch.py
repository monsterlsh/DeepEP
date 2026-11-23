"""Exercise MTE dispatch with the logical-nibble indexing used by AscendC FP4 tensors."""

from pathlib import Path
import shutil
import subprocess

import pytest


def test_packed_fp4_record_fields_and_workspace_rows_keep_byte_offsets(tmp_path):
    compiler = next((path for name in ("c++", "g++", "clang++") if (path := shutil.which(name))), None)
    if compiler is None:
        pytest.skip("The dispatch probe requires a host C++17 compiler")
    path = Path(__file__).resolve().parents[1] / "csrc/kernels/mega_moe/stage/mega_moe_token_dispatch.hpp"
    source = path.read_text(encoding="utf-8")
    functions = source[
        source.rindex("template <typename ActivationType>", 0, source.index("GetDispatchCopyBuffer")) :
        source.index("// 环形槽持续跨 peer")
    ]
    fixture = tmp_path / "dispatch.cpp"
    fixture.write_text(
        r"""
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <type_traits>
#define __aicore__
#define __gm__
using TEventID = int;
enum class TPosition { VECCALC };
namespace AscendC {
enum class HardEvent { MTE3_MTE2, MTE2_MTE3, MTE3_S, MTE2_S, S_MTE3 };
}
template <AscendC::HardEvent> void SetFlag(TEventID) {}
template <AscendC::HardEvent> void WaitFlag(TEventID) {}
struct Fp4 { uint8_t bits; };
template <typename T> constexpr int elementsPerByte = std::is_same_v<T, Fp4> ? 2 : 1;
std::array<uint8_t, 8192> ub{};
template <typename T> struct LocalTensor {
    uint8_t* address;
    explicit LocalTensor(uint8_t* p) : address(p) {}
    LocalTensor(TPosition, uint32_t offset, uint32_t) : address(ub.data() + offset) {}
    LocalTensor operator[](uint64_t offset) const { return LocalTensor(address + offset * sizeof(T) / elementsPerByte<T>); }
    template <typename U> LocalTensor<U> ReinterpretCast() const { return LocalTensor<U>(address); }
    T GetValue(uint32_t i) const { T value; std::memcpy(&value, (*this)[i].address, sizeof(T)); return value; }
    void SetValue(uint32_t i, T value) { std::memcpy((*this)[i].address, &value, sizeof(T)); }
};
template <typename T> struct GlobalTensor {
    uint8_t* address;
    void SetGlobalBuffer(T* p) { address = reinterpret_cast<uint8_t*>(p); }
    GlobalTensor operator[](uint64_t offset) const { return {address + offset * sizeof(T) / elementsPerByte<T>}; }
};
template <typename T> void DataCopy(LocalTensor<T> dst, GlobalTensor<T> src, uint32_t elements) {
    std::memcpy(dst.address, src.address, elements * sizeof(T) / elementsPerByte<T>);
}
template <typename T> void DataCopy(GlobalTensor<T> dst, LocalTensor<T> src, uint32_t elements) {
    std::memcpy(dst.address, src.address, elements * sizeof(T) / elementsPerByte<T>);
}
struct DataCopyExtParams { uint32_t count, bytes, srcStride, dstStride, padding; };
template <typename T> void DataCopyPad(GlobalTensor<T> dst, LocalTensor<T> src, DataCopyExtParams copy) {
    assert(copy.count == 1);
    std::memcpy(dst.address, src.address, copy.bytes);
}
constexpr uint32_t INT32_PER_256B = 64, RANK_ID = 0, TOKEN_ID = 1, TOPK_INDEX = 2, WEIGHT_INDEX = 3;
uint64_t MapWaveRow(uint32_t rowOffset, uint64_t row) { return rowOffset + row; }
struct TokenDispatchConfig {
    uint32_t quantTokenAlignBytes, quantScaleAlignBytes, quantTokenScaleAlignBytes, waveMapping;
};
template <typename> struct TokenDispatchScratch {
    uint32_t copyTmpBaseAddr, revTokenElemCnt, revScaleElemCnt;
    LocalTensor<int32_t> metaInfoTensor;
};
struct Tiling { uint32_t topK; };
struct Params {
    const Tiling* tilingData;
    struct { uint8_t* dispatchRevDataPtr; uint8_t* dispatchRevScalePtr; uint8_t* metaInfoPtr; } workspaceInfo;
};
"""
        + functions
        + r"""
int main() {
    constexpr uint32_t tokenBytes = 2048, scaleBytes = 128, recordBytes = 2304;
    std::array<uint8_t, 3 * recordBytes> peer{};
    std::array<uint8_t, 4 * tokenBytes> tokens{};
    std::array<uint8_t, 4 * scaleBytes> scales{};
    std::array<uint8_t, 4 * 256> meta{};
    Tiling tiling{25};
    TokenDispatchConfig config{tokenBytes, scaleBytes, recordBytes, 1};
    Params params{&tiling, {tokens.data(), scales.data(), meta.data()}};
    TokenDispatchScratch<Fp4> scratch{0, tokenBytes, scaleBytes, LocalTensor<int32_t>(ub.data() + 6144)};
    auto record = peer.data() + 2 * recordBytes;
    std::memset(record, 0x5a, tokenBytes);
    std::memset(record + tokenBytes, 0x77, scaleBytes);
    int32_t weight = 0x3f000000;
    std::memcpy(record + tokenBytes + scaleBytes + 4 * 19, &weight, 4);
    GlobalTensor<uint8_t> peerBytes{peer.data()};
    FetchDispatchTokenAndMetaInfo<false, true>(config, params, scratch, 1, 2 * 25 + 19, 17, peerBytes);
    StoreDispatchTokenAndMetaInfo<Fp4, uint8_t, true>(config, params, scratch, 1, 2);
    assert(std::memcmp(tokens.data() + 3 * tokenBytes, record, tokenBytes) == 0);
    assert(std::memcmp(scales.data() + 3 * scaleBytes, record + tokenBytes, scaleBytes) == 0);
    assert(tokens[3 * tokenBytes - 1] == 0 && scales[3 * scaleBytes - 1] == 0);
    std::array<int32_t, 4> metadata{};
    std::memcpy(metadata.data(), meta.data() + 3 * 256, sizeof(metadata));
    assert((metadata == std::array<int32_t, 4>{17, 2, 19, weight}));
}
""",
        encoding="utf-8",
    )
    binary = tmp_path / "dispatch.exe"
    subprocess.run([compiler, "-std=c++17", str(fixture), "-o", str(binary)], check=True)
    subprocess.run([binary], check=True)
