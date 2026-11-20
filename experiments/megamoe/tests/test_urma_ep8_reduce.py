"""Execute the device reducers with synchronous host tensor operations."""

from pathlib import Path
import shutil
import subprocess
import sys

import pytest


def test_window_reduce_preserves_original_slot_order(tmp_path):
    compiler = next((path for name in ("c++", "g++", "clang++") if (path := shutil.which(name))), None)
    if compiler is None:
        pytest.skip("A C++17 compiler is required for the host reducer test")
    csrc = Path(__file__).resolve().parents[1] / "csrc"
    kernel = csrc / "kernels/mega_moe_urma/stage/mega_moe_reduce.hpp"
    first_definition = "__aicore__ inline void BuildReduceDependencies"
    reducer = "namespace MegaMoeUrma {\n" + first_definition + kernel.read_text(encoding="utf-8").split(
        first_definition, 1
    )[1]
    source = tmp_path / "reduce.cpp"
    source.write_text(
        r'''#include "kernels/mega_moe_urma/common/mega_moe_route_plan.hpp"
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <vector>

#define __gm__
#define __aicore__

namespace AscendC {
alignas(64) std::byte ub[MegaMoeUrma::kRequiredUbBytes];
uint32_t multiplyCalls = 0U;
enum class TPosition { VECCALC };
enum class CacheMode { CACHE_MODE_DISABLE };
enum class RoundMode { CAST_NONE, CAST_RINT };
enum class HardEvent { V_MTE2, MTE2_V, V_MTE3, MTE3_V, S_MTE2, MTE2_S };
using TEventID = uint32_t;
constexpr TEventID EVENT_ID0 = 0;
constexpr int PIPE_ALL = 0, PIPE_V = 1;
template <HardEvent> void SetFlag(TEventID) {}
template <HardEvent> void WaitFlag(TEventID) {}
template <int> void PipeBarrier() {}
template <typename T> T ReadGmByPassDCache(T* p) { return *p; }
uint64_t GetSystemCycle() { static uint64_t tick = 0; return ++tick; }

struct bfloat16_t {
    uint16_t bits = 0U;
    bfloat16_t() = default;
    explicit bfloat16_t(float value) {
        uint32_t word;
        std::memcpy(&word, &value, sizeof(word));
        bits = uint16_t((word + 0x7fffU + ((word >> 16U) & 1U)) >> 16U);
    }
    explicit operator float() const {
        const uint32_t word = uint32_t(bits) << 16U;
        float value;
        std::memcpy(&value, &word, sizeof(value));
        return value;
    }
};
static_assert(sizeof(bfloat16_t) == 2U);

template <typename T> struct Tensor {
    T* ptr = nullptr;
    Tensor() = default;
    explicit Tensor(T* address) : ptr(address) {}
    Tensor(TPosition, uint32_t offset, uint32_t size) : ptr(reinterpret_cast<T*>(ub + offset)) {
        assert(offset + size * sizeof(T) <= sizeof(ub));
    }
    void SetGlobalBuffer(T* address) { ptr = address; }
    void SetL2CacheHint(CacheMode) {}
    Tensor operator[](uint64_t offset) const { return Tensor(ptr + offset); }
    template <typename D> Tensor<D> ReinterpretCast() const { return Tensor<D>(reinterpret_cast<D*>(ptr)); }
    T GetValue(uint32_t index) const { return ptr[index]; }
    void SetValue(uint32_t index, T value) { ptr[index] = value; }
};
template <typename T> using LocalTensor = Tensor<T>;
template <typename T> using GlobalTensor = Tensor<T>;
template <typename T> void DataCopy(Tensor<T> dst, Tensor<T> src, uint32_t n) {
    std::memmove(dst.ptr, src.ptr, n * sizeof(T));
}
struct CopyParams { uint32_t blocks, bytes, srcStride, dstStride, reserved; };
template <typename T> struct PadParams { bool pad; uint8_t left, right; T value; };
template <typename T> void DataCopyPad(Tensor<T> dst, Tensor<T> src, CopyParams p, PadParams<T>) {
    assert(p.blocks == 1U && p.srcStride == 0U && p.dstStride == 0U);
    std::memmove(dst.ptr, src.ptr, p.bytes);
}
template <typename D, typename S> void Cast(Tensor<D> dst, Tensor<S> src, RoundMode, uint32_t n) {
    for (uint32_t i = 0U; i < n; ++i) dst.ptr[i] = static_cast<D>(src.ptr[i]);
}
void Muls(Tensor<float> dst, Tensor<float> src, float w, uint32_t n) {
    ++multiplyCalls;
    for (uint32_t i = 0U; i < n; ++i) dst.ptr[i] = src.ptr[i] * w;
}
void Add(Tensor<float> dst, Tensor<float> a, Tensor<float> b, uint32_t n) {
    for (uint32_t i = 0U; i < n; ++i) dst.ptr[i] = a.ptr[i] + b.ptr[i];
}
}  // namespace AscendC

namespace MegaMoeUrma {
using AscendC::GetSystemCycle;
struct TestTimer {
    uint64_t TraceBegin() const { return 0U; }
    void TraceEnd(TraceEvent, uint64_t, TraceMeta) const {}
};
void aclshmemi_kernel_abort(const char*) { std::abort(); }
void WaitPackInverseReady(const Arguments&) {}
const DeviceRouteInfo* RouteInfo(const Arguments& args) {
    return reinterpret_cast<const DeviceRouteInfo*>(args.routeInfo);
}
}  // namespace MegaMoeUrma
'''
        + reducer
        + r'''
template <bool KeepFp32>
void Run(uint32_t tokens, uint32_t topk, uint32_t hidden, bool weighted, uint32_t world = 8U) {
    using namespace MegaMoeUrma;
    using AscendC::bfloat16_t;
    using Output = std::conditional_t<KeepFp32, float, bfloat16_t>;
    Arguments args{};
    args.config.world = world;
    args.config.bs = tokens;
    args.config.rank = world - 1U;
    args.config.localExperts = 25U;
    args.config.cvRows = 128U;
    args.topkIdBytes = sizeof(int32_t);
    args.config.topk = topk;
    args.config.hidden = hidden;
    args.config.topkWeightsType = weighted ? 0U : 1U;
    args.config.maxPeerChunks = (tokens * topk + 127U) / 128U;
    args.config.epoch = 3U;
    const uint32_t routes = tokens * topk;
    const size_t elements = size_t(routes) * hidden;
    args.config.combineReadyOffset = elements * sizeof(bfloat16_t);
    const uint32_t flagCount = world * args.config.maxPeerChunks;
    std::vector<uint64_t> symmetric(elements / 4U + flagCount * kFlagBytes / sizeof(uint64_t));
    auto* returned = reinterpret_cast<bfloat16_t*>(symmetric.data());
    symmetric[elements / 4U] = args.config.epoch;
    for (uint32_t cv = 0; cv < args.config.maxPeerChunks; ++cv) {
        const uint32_t idx = (world - 1U) * args.config.maxPeerChunks + cv;
        symmetric[elements / 4U + idx * kFlagBytes / sizeof(uint64_t)] = args.config.epoch;
    }
    alignas(64) std::array<int32_t, 4U * kFlagBytes / sizeof(int32_t)> flags{};
    // Force per-route readiness probes, with visible payload in both dependencies.
    flags[kLocalFlagIncomingDone * kFlagBytes / sizeof(int32_t)] = world - 1U;
    // One row is routed to PE0. The remainder use the last PE, which owns
    // the highest readiness index and includes the rank's direct self return.
    DeviceRouteInfo info{};
    for (uint32_t id = 1U; id <= world * args.config.localExperts; ++id) info.outgoing[id] = 1U;
    info.outgoing[world * args.config.localExperts] = routes;
    std::vector<int32_t> ids(routes, world * args.config.localExperts - 1U);
    ids[routes - 1U] = 0;
    args.routeInfo = reinterpret_cast<uint64_t>(&info);
    args.topkIds = reinterpret_cast<uint64_t>(ids.data());
    std::vector<int64_t> inverse(routes);
    std::vector<float> weights(routes);
    std::vector<Output> output(size_t(tokens) * hidden, static_cast<Output>(1234.5F));
    for (uint32_t route = 0; route < routes; ++route) {
        const uint32_t token = route / topk, slot = route % topk;
        inverse[route] = routes - 1U - route;
        // Cancellation makes a different accumulation order visible at FP32 precision.
        const float sign = token % 2U ? -1.0F : 1.0F;
        const float value = (slot % 3U == 0U ? 67108864.0F : slot % 3U == 1U ? -67108864.0F : 1.0F);
        weights[route] = slot % 3U == 2U ? 0.5F : 1.0F;
        for (uint32_t col = 0U; col < hidden; ++col) {
            const float factor = (col / 2048U + 1U) * (col % 2U ? 0.25F : 1.0F);
            returned[size_t(inverse[route]) * hidden + col] = bfloat16_t(value * sign * factor);
        }
    }
    args.symmetric = reinterpret_cast<uint64_t>(symmetric.data());
    args.localFlags = reinterpret_cast<uint64_t>(flags.data());
    args.reduceIndex = reinterpret_cast<uint64_t>(inverse.data());
    // Preweighted returns need neither a weights address nor a vector multiply.
    args.weights = weighted ? reinterpret_cast<uint64_t>(weights.data()) : 0U;
    args.output = args.sharedRoutedOutput = reinterpret_cast<uint64_t>(output.data());
    const uint32_t workers = KeepFp32 ? kLocalWorkers : ReduceWorkers(args.config.world);
    TestTimer timer;
    NoReduceObserver observer;
    const bool legacy = !KeepFp32 && UseLegacyReduce(args.config);
    AscendC::multiplyCalls = 0U;
    for (uint32_t worker = 0U; worker < workers; ++worker) {
        std::memset(AscendC::ub, 0xff, sizeof(AscendC::ub));
        if (legacy) Reduce(args, worker, timer);
        else ReduceGeneric<KeepFp32>(args, worker, workers, timer, observer);
    }
    const uint32_t slices = legacy ? 1U : (hidden + GenericReduce::ReduceUbContract::hidden - 1U) /
                                           GenericReduce::ReduceUbContract::hidden;
    assert(AscendC::multiplyCalls == (weighted ? routes * slices : 0U));
    for (uint32_t token = 0U; token < tokens; ++token) {
        for (uint32_t col = 0U; col < hidden; ++col) {
            float expected = 0.0F;
            for (uint32_t slot = 0U; slot < topk; ++slot) {
                const uint32_t route = token * topk + slot;
                const float value = static_cast<float>(returned[size_t(inverse[route]) * hidden + col]);
                expected += value * (weighted ? weights[route] : 1.0F);
            }
            const auto result = output[size_t(token) * hidden + col];
            assert(static_cast<float>(result) == static_cast<float>(static_cast<Output>(expected)));
        }
    }
}

int main() {
    for (uint32_t width = 1U; width <= MegaMoeUrma::kWindowMaxTopk; ++width) {
        Run<false>(17U, width, 2304U, false);
        Run<true>(17U, width, 256U, true);
    }
    for (uint32_t world : {8U, 16U, 32U}) {
        Run<false>(MegaMoeUrma::kWindowTokens, MegaMoeUrma::kWindowMaxTopk, 256U, true, world);
        Run<false>(112U, 25U, 4096U, false, world);
        Run<false>(255U, 32U, 256U, false, world);
    }
    Run<false>(17U, 8U, 3584U, false);
    Run<false>(17U, 8U, 3584U, true);
}
''',
        encoding="utf-8",
    )
    executable = tmp_path / ("reduce.exe" if sys.platform == "win32" else "reduce")
    result = subprocess.run(
        [
            compiler, "-std=c++17", "-O2", "-ffp-contract=off", "-fno-strict-aliasing",
            "-I", str(csrc), str(source), "-o", str(executable),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(executable)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
