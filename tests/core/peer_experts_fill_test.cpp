// Peer fills and scratch lending preserve blob bytes with or without driver VMM support.
#include "basalt/core/peer_experts.hpp"
#include "basalt/core/vmm.hpp"
#include "basalt/kernels/cpu/expert_layout.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

namespace {
namespace core = basalt::core;
namespace cpu = basalt::kernels::cpu;

struct Source : core::ExpertSource {
    std::vector<std::vector<uint8_t>> blobs;
    explicit Source(size_t bytes) : blobs(2, std::vector<uint8_t>(bytes)) {
        for (size_t e = 0; e < blobs.size(); ++e)
            for (size_t i = 0; i < bytes; ++i) blobs[e][i] = (uint8_t) (i * 7 + e * 31);
    }
    const uint8_t* blob(int64_t, int64_t expert) override { return blobs[(size_t) expert].data(); }
};

bool same(core::PeerExperts& peer, const Source& src) {
    int prev = 0;
    cudaGetDevice(&prev);
    cudaSetDevice(peer.device());
    bool ok = true;
    for (int64_t e = 0; e < 2; ++e) {
        std::vector<uint8_t> actual(src.blobs[(size_t) e].size());
        ok = ok && peer.has(0, e) &&
             cudaMemcpy(actual.data(), peer.slot_ptr(0, e), actual.size(), cudaMemcpyDeviceToHost) == cudaSuccess &&
             actual == src.blobs[(size_t) e];
    }
    cudaSetDevice(prev);
    return ok;
}
}  // namespace

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2) {
        std::puts("two CUDA devices required: skipped");
        return 77;
    }
    std::string err;
    auto fail = [&](const char* step) {
        std::fprintf(stderr, "%s: %s\n", step, err.c_str());
        return 1;
    };
    cpu::NativeFmt fmt;
    if (!cpu::native_fmt(12, 7, 256, 256, fmt, err)) return fail("format");
    std::ostringstream layout;
    layout << "# experts v6: test\n0 12 7 0 " << fmt.bytes << '\n';
    if (!cpu::expert_layout_load_text(layout.str(), 1, 2, err, 256, 256)) return fail("layout");
    Source src((size_t) fmt.bytes);
    cudaSetDevice(1);
    const bool segmented = core::vmm::supported();
    cudaSetDevice(0);
    core::PeerExperts peer;
    for (int repeat = 0; repeat < 2; ++repeat) {
        if (!peer.prepare(1, src, 1, 2, 256, 256, err) || !peer.fill({{0, 0}, {0, 1}}, err))
            return fail("fill");
        int current = -1;
        cudaGetDevice(&current);
        if (current != 0 || peer.cache().segmented() != segmented || !same(peer, src))
            return fail("filled bytes and device");
        std::vector<std::pair<int32_t, int32_t>> lent;
        if (!peer.lend(1, lent, err) || lent.size() != 1 || peer.has(0, 1)) return fail("lend");
        cudaSetDevice(1);
        const cudaError_t cleared = cudaMemset(const_cast<uint8_t*>(peer.slot_address(1)), 0, src.blobs[1].size());
        const cudaError_t synced = cudaDeviceSynchronize();
        cudaSetDevice(0);
        if (cleared != cudaSuccess || synced != cudaSuccess || !peer.give_back(lent, err) || !peer.give_back_wait(lent, err) ||
            !lent.empty() || !same(peer, src)) return fail("give back");
        const bool slept = peer.sleep(err);
        if (segmented) {
            if (!slept || !peer.wake(err) || !peer.wake_wait(err) || !same(peer, src)) return fail("sleep and wake");
        } else if (slept || peer.asleep() || !same(peer, src)) {
            return fail("sleep refused without losing bytes");
        }
    }
    std::printf("peer fill: bytes, lending, reopen and sleep policy passed (VMM=%d)\n", segmented);
    return 0;
}
