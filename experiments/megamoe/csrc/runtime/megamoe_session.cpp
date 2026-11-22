// Copyright (c) 2026, Lu Lu
// Modified by SimpleBrightMan 2026

#include "runtime/megamoe_session.hpp"

#include <cstdlib>
#include <stdexcept>

#include "host/shmem_host_def.h"
#include "host_device/shmem_common_types.h"

namespace deepep::megamoe {
namespace {

SessionOptions options_for(const std::string& backend, int size, bool windowed)
{
    SessionOptions options;
    options.megamoe = true;
    if (backend == "mte") {
        if (windowed) throw std::invalid_argument("The UBX transport requires the URMA kernel");
        options.maximum_bytes = 1ULL << 30;
        return options;
    }
    if (backend == "mte_roce") {
        if (windowed || size != 32) throw std::invalid_argument("MegaMoE MTE+RoCE requires the non-URMA EP32 runtime");
        options.maximum_bytes = 1ULL << 30;
        options.required_peer_transports = ACLSHMEM_TRANSPORT_MTE;
        options.megamoe_hierarchical_roce = true;
        return options;
    }
    if (backend != "urma") throw std::invalid_argument("MegaMoE backend must be mte, mte_roce or urma");

    options.maximum_bytes = 32ULL << 30;
    if (windowed) {
        if (size != 8 && size != 16 && size != 32)
            throw std::invalid_argument("UBX windowed transport requires EP8, EP16 or EP32");
        options.required_peer_transports = ACLSHMEM_TRANSPORT_MTE;
        options.window_ranks_per_host = size == 32 ? 16 : size;
        return options;
    }
    if (size == 32) throw std::invalid_argument("UBX EP32 requires windowed transport");
    options.udma = true;
    options.required_peer_transports = ACLSHMEM_TRANSPORT_MTE | ACLSHMEM_TRANSPORT_UDMA;
    options.dedicated_peer_qp = true;

    // This is an internal A/B switch for the resource smoke test.  Production
    // MegaMoE keeps one QP per peer; setting it to elastic selects the existing
    // shared-jetty UDMA allocation without changing the Python API.
    const char* resource_mode = std::getenv("DEEPEP_MEGAMOE_RESOURCE_MODE");
    if (resource_mode != nullptr && resource_mode[0] != '\0') {
        const std::string mode(resource_mode);
        if (mode == "elastic") {
            options.dedicated_peer_qp = false;
        } else if (mode != "dedicated") {
            throw std::invalid_argument("DEEPEP_MEGAMOE_RESOURCE_MODE must be dedicated or elastic");
        }
    }
    return options;
}

}  // namespace

std::unique_ptr<Session> create_runtime(int rank, int size, uint64_t bytes, const std::string& endpoint,
                                        unsigned timeout, const std::string& backend, bool windowed)
{
    return std::make_unique<Session>(rank, size, bytes, endpoint, timeout, options_for(backend, size, windowed));
}

}  // namespace deepep::megamoe
