#include <cassert>
#include <type_traits>

#define __aicore__
#include "kernels/mega_moe_common/common/mega_moe_peer_rows.hpp"
#include "kernels/mega_moe_hierarchical/mc2_common/mc2_moe_context.hpp"
#include "kernels/mega_moe_common/mc2_common/mc2_moe_context.hpp"

#if defined(DEEPEP_MEGAMOE_HIERARCHICAL_IMPL)
namespace Kernel = HierarchicalMegaMoeImpl;
#else
namespace Kernel = MegaMoeImpl;
#endif

static_assert(std::is_same_v<HierarchicalMc2Aclnn::Mc2MoeContext, Mc2Aclnn::Mc2MoeContext>);
static_assert(std::is_same_v<HierarchicalMc2Aclnn::MoeCommContext, Mc2Aclnn::MoeCommContext>);

int main()
{
    assert(Kernel::GetDispatchPeer(0, 0, 32, 16) == 16);
    assert(Kernel::GetDispatchPeer(1, 0, 32, 16) == 0);
    assert(Kernel::GetDispatchPeer(2, 0, 32, 16) == 17);
}
