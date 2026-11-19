"""Exercise EP8 GELU route metadata using the production host/device planner."""

from pathlib import Path
import shutil
import subprocess
import sys

import pytest


def test_ep8_window_route_metadata_at_topk32(tmp_path):
    compiler = next((path for name in ("c++", "g++", "clang++") if (path := shutil.which(name))), None)
    if compiler is None:
        pytest.skip("A host C++17 compiler is required for the route planner check")
    csrc = Path(__file__).resolve().parents[1] / "csrc"
    source = tmp_path / "ep8_routing.cpp"
    executable = tmp_path / ("ep8_routing.exe" if sys.platform == "win32" else "ep8_routing")
    source.write_text(
        r"""
#include <cassert>
#include <initializer_list>
#include "kernels/mega_moe_urma/common/mega_moe_route_plan.hpp"

using namespace MegaMoeUrma;

struct PackCheck {
    const Config& config;
    uint32_t destination;
    uint64_t packed = 0;

    void PackChunk(uint32_t cv, uint32_t peer, const Chunk& chunk) const
    {
        if (peer != destination) {
            assert(chunk.rows == 0U);
            return;
        }
        const uint32_t row = cv * config.cvRows;
        assert(chunk.cv == cv && chunk.peer == destination && chunk.localRow == row);
        assert(chunk.remoteRow == config.rank * config.sourceRowCapacity + row);
        assert(chunk.rows == RouteMin(config.cvRows, config.bs * config.topk - row));
    }

    void PackRows(uint32_t, uint64_t rows)
    {
        packed += rows;
    }
};

void check(uint32_t tokens, uint32_t topk, uint32_t cvRows, bool allSources)
{
    Config config{};
    config.world = 8U;
    config.rank = 3U;
    config.localExperts = 100U;
    config.hidden = 4096U;
    config.intermediate = 1536U;
    config.activationMode = static_cast<uint32_t>(MegaMoeImpl::MegaMoeActMode::GELU);
    config.topkWeightsType = 1U;
    config.bs = tokens;
    config.topk = topk;
    config.cvRows = cvRows;
    config.sourceRowCapacity = kEp8WindowTokens * topk;
    config.maxPeerChunks = (config.sourceRowCapacity + cvRows - 1U) / cvRows;

    DeviceRouteInfo routes{};
    const uint32_t firstExpert = config.localExperts - topk - 1U;
    const uint32_t lastExpert = firstExpert + topk;
    const uint32_t destination = allSources ? config.rank : config.world - 1U;
    const uint32_t localRows = tokens * topk;
    for (uint32_t p = 0U; p < config.world; ++p) {
        for (uint32_t e = 0U; e < config.localExperts; ++e) {
            const uint32_t slots = e >= firstExpert && e < lastExpert ? tokens : 0U;
            const uint32_t globalExpert = p * config.localExperts + e;
            routes.outgoing[globalExpert + 1U] = routes.outgoing[globalExpert] + (p == destination ? slots : 0U);
            routes.incoming[p][e + 1U] = routes.incoming[p][e] + (allSources || p == 1U ? slots : 0U);
        }
    }
    for (uint32_t core = 0U; core < kAic; ++core) {
        const auto owned = RouteWork(config.localExperts, core, kAic);
        SummarizeRouteExperts(config, routes, owned.first, owned.last);
    }
    FinishRouteSummaryPrefix(config, routes, false);

    const uint32_t sources = allSources ? config.world : 1U;
    const uint32_t receiveCvs = (localRows + cvRows - 1U) / cvRows;
    const uint32_t groupsPerExpert = (sources * tokens + 255U) / 256U;
    assert(config.maxPeerChunks <= kMaxPeerChunks && receiveCvs <= config.maxPeerChunks);
    assert(localRows <= kMaxPeerReorderSpans);
    assert(config.receiveRows == sources * localRows);
    assert(config.fragmentCount == groupsPerExpert * topk);
    assert(config.gmm1Tiles == config.fragmentCount * (config.intermediate / 256U));
    assert(config.gmm2Tiles == config.fragmentCount * (config.hidden / 256U));
    assert(config.selfRows == (allSources ? localRows : 0U));
    assert(config.sendChunkCount == config.packRoundCount && config.packRoundCount == receiveCvs);
    assert(config.recvChunkCount == sources * receiveCvs);
    for (uint32_t p = 0U; p < config.world; ++p) {
        assert(routes.recvCvBase[p + 1U] - routes.recvCvBase[p] == (allSources || p == 1U ? receiveCvs : 0U));
    }
    for (uint32_t cv = 0U; cv < config.recvChunkCount; ++cv) {
        const uint32_t peer = PlanPeerForCv(cv, config, routes, false);
        assert(peer == (allSources ? cv / receiveCvs : 1U));
        const uint32_t peerRow = (cv - routes.recvCvBase[peer]) * cvRows;
        assert(PlanExpertForPeerRow(peer, peerRow, config, routes) == firstExpert + peerRow / tokens);
    }

    PackCheck pack{config, destination};
    EmitRoutePack(config, routes, pack, {0U, config.packRoundCount});
    assert(pack.packed == localRows);
}

int main()
{
    assert(kEp8WindowTokens == 256U && kEp8MaxTopk == 32U);
    for (uint32_t cv : {128U, 256U}) {
        for (uint32_t tokens : {1U, 17U, 255U, kEp8WindowTokens}) {
            for (uint32_t topk = 1U; topk <= kEp8MaxTopk; ++topk) {
                check(tokens, topk, cv, true);
                check(tokens, topk, cv, false);
            }
        }
    }
}
""",
        encoding="utf-8",
    )
    subprocess.run(
        [compiler, "-std=c++17", "-I", str(csrc), str(source), "-o", str(executable)],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    subprocess.run([str(executable)], check=True, capture_output=True, text=True, timeout=30)


@pytest.mark.parametrize("topk", [1, 8, 9, 25, 32])
def test_window_route_preflight_scans_tail_and_preserves_slot_order(topk):
    torch = pytest.importorskip("torch")
    from deep_ep_experimental.megamoe.validation import check_window_routes

    ids = torch.arange(topk - 1, -1, -1, dtype=torch.int32).repeat(257, 1)
    weights = torch.linspace(-1, 1, topk).repeat(257, 1)
    original = ids.clone()
    check_window_routes(ids, weights, 32)
    assert torch.equal(ids, original)

    for invalid in (-1, 32):
        ids[-1, -1] = invalid
        with pytest.raises(ValueError, match="expert IDs"):
            check_window_routes(ids, weights, 32)
    if topk > 1:
        ids[-1, -1] = ids[-1, 0]
        with pytest.raises(ValueError, match="duplicate"):
            check_window_routes(ids, weights, 32)
    for invalid in (float("nan"), float("inf"), -float("inf")):
        weights[-1, -1] = invalid
        with pytest.raises(ValueError, match="finite"):
            check_window_routes(original, weights, 32)
