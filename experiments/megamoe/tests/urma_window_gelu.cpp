// Exercise the routed device loop with a simulated mailbox and DMA endpoints.
#include <array>
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include "kernels/mega_moe_urma/common/mega_moe_compute_contract.hpp"

using namespace MegaMoeUrma;
#define __aicore__
#define __gm__
constexpr unsigned EVENT_ID0 = 0;
struct Capture {
    uint32_t rows, total = 0, batch = 0, mailbox = 0, column = 0, calls = 0, signals = 0;
    uint64_t rowBegin, weightIndex = 0;
    bool lookup, early;
};
static Capture capture;

namespace AscendC {
enum class HardEvent { MTE2_V };
template <HardEvent> void SetFlag(unsigned) {}
template <HardEvent> void WaitFlag(unsigned) {}
template <unsigned, unsigned> void SetCtrlSpr(unsigned) {}
template <class T> struct GlobalTensor {
    void SetGlobalBuffer(T*) {}
    uint64_t operator[](uint64_t index) const { return index; }
};
void DataCopy(int, uint64_t index, uint32_t count) {
    capture.weightIndex = index;
    capture.batch = count / 8;
}
void AtomicAdd(uint64_t, int delta) {
    assert(capture.signals == 0 && capture.total == capture.rows && delta == 1);
    ++capture.signals;
}
}
enum class Phase { Activation };
struct Timer {
    void Begin(Phase) {}
    int TraceBegin() { return 0; }
    void TraceEnd(TraceEvent, int, int) {}
    void End(Phase) {}
};
int FragmentTraceMeta(const Fragment&, uint32_t) { return 0; }
int ComputeTileTraceMeta(int, unsigned, unsigned, unsigned, unsigned) { return 0; }
uint64_t ComputeFlag(uint64_t base, unsigned group) { return base + group; }

struct Epilogue {
    using Offsets = std::array<int64_t, 6>;
    void UpdateGlobalAddr(Offsets offset) {
        assert(offset[0] == int64_t(capture.rowBegin * 1536));
        assert(offset[1] == int64_t(capture.rowBegin * 48));
    }
    int GetTopkWeightTensor() { return 0; }
    void SelectGeluMailboxRows(unsigned first) { capture.mailbox = first; }
    void operator()(std::array<int64_t, 4> shape, Offsets out, unsigned) {
        assert(shape[1] == 256 && capture.mailbox == capture.total && capture.signals == 0);
        assert(shape[0] > 0 && shape[0] <= (capture.lookup ? 128 : 256));
        const auto value = int64_t(capture.total) * 1536 + capture.column;
        assert(out[0] == value && out[1] == value / 32);
        if (capture.early) {
            assert(capture.batch == shape[0] && capture.weightIndex == (capture.rowBegin + capture.total) * 8);
        }
        capture.total += shape[0];
        ++capture.calls;
    }
};
struct TestActivation {
    static unsigned Columns(const Arguments&) { return 256; }
#include "routed_activation.inc"
};
int main() {
    TestActivation activation;
    Epilogue op;
    Timer timer;
    Arguments args{};
    args.config.intermediate = 1536;
    for (unsigned rows : {1, 31, 127, 128, 129, 255, 256}) {
        Fragment f{};
        f.rowBegin = 12345;
        f.rows = rows;
        for (unsigned tile = 0; tile < 6; ++tile) {
            for (unsigned mode : {0, 1, 2}) {
                capture = {};
                capture.rows = rows;
                capture.column = tile * 256;
                capture.rowBegin = f.rowBegin;
                capture.lookup = mode == 2;
                capture.early = mode > 0;
                if (mode == 2) activation.RunImpl<true, true>(args, f, 3, tile, timer, op);
                if (mode == 1) activation.RunImpl<true>(args, f, 3, tile, timer, op);
                if (mode == 0) activation.RunImpl<false>(args, f, 3, tile, timer, op);
                assert(capture.signals == 1 && capture.calls == 1U + (mode == 2 && rows > 128));
            }
        }
    }
}
