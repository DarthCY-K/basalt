// Portable peer transport: exact input/result bytes across repeated layers and more groups than the event ring.
#include "../../../../src/models/qwen4exp/prefill/peer.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
using basalt::prefill::PeerPrefill;
constexpr size_t kRow = 2560 * sizeof(float);
constexpr int kRows = 17, kLayers = 5, kGroups = 2 * PeerPrefill::kGrpEv + 5;

void check(cudaError_t e) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "peer transfer: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}
cudaEvent_t event() {
    cudaEvent_t e = nullptr;
    check(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    return e;
}
cudaStream_t stream() {
    cudaStream_t s = nullptr;
    check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    return s;
}
float* device_bytes(size_t bytes) {
    float* p = nullptr;
    check(cudaMalloc((void**) &p, bytes));
    return p;
}
float* host_bytes(size_t bytes) {
    float* p = nullptr;
    check(cudaHostAlloc((void**) &p, bytes, cudaHostAllocPortable));
    return p;
}

bool run_pair(int primary, int peer) {
    const size_t input_bytes = kRows * kRow;
    const size_t layer_bytes = kGroups * input_bytes;
    std::vector<unsigned char> inputs(kLayers * input_bytes), expected(kLayers * layer_bytes, 0);
    for (size_t i = 0; i < inputs.size(); ++i) inputs[i] = (unsigned char) (i * 17 + i / input_bytes * 29);
    check(cudaSetDevice(primary));
    float* input = device_bytes(input_bytes);
    float* output = device_bytes(expected.size());
    check(cudaMemset(output, 0, expected.size()));
    cudaStream_t ready = stream();
    bool same = true;
    {
        PeerPrefill p;
        p.prim = primary;
        p.dev = peer;
        p.host_staged = true;
        p.x_out = stream();
        p.res_in = stream();
        p.ev_in = event();
        p.ev_hx = event();
        p.ev_res = event();
        p.hx = host_bytes(input_bytes);
        for (int b = 0; b < PeerPrefill::kRes; ++b) {
            p.ev_h2d[b] = event();
            p.hres[b] = host_bytes(input_bytes);
        }
        check(cudaSetDevice(peer));
        p.s = stream();
        p.s_out = stream();
        p.ev_hx_free = event();
        for (auto& e : p.ev_grp) e = event();
        for (auto& e : p.ev_dm) e = event();
        for (auto& e : p.ev_d2h) e = event();
        p.mixed = device_bytes(input_bytes);
        p.Dm_b[0] = device_bytes(input_bytes);
        p.Dm_b[1] = device_bytes(input_bytes);
        float* snapshots = device_bytes(inputs.size());
        p.owned = {p.mixed, p.Dm_b[0], p.Dm_b[1], snapshots};
        for (int layer = 0; layer < kLayers; ++layer) {
            check(cudaSetDevice(primary));
            if (layer) check(cudaStreamWaitEvent(ready, p.ev_res, 0));
            check(cudaMemcpyAsync(input, inputs.data() + layer * input_bytes, input_bytes, cudaMemcpyHostToDevice, ready));
            p.stage_input(input, ready, kRows);
            check(cudaMemcpyAsync((unsigned char*) snapshots + layer * input_bytes, p.mixed,
                                  input_bytes, cudaMemcpyDeviceToDevice, p.s));
            for (int group = 0; group < kGroups; ++group) {
                const int b = group & 1;
                const int rows = 1 + (group * 7 + layer * 3) % kRows;
                const int pattern = (layer * kGroups + group) % 251 + 1;
                const size_t at = layer * layer_bytes + group * input_bytes;
                std::memset(expected.data() + at, pattern, rows * kRow);
                if (p.dm_live[b]) check(cudaStreamWaitEvent(p.s, p.ev_dm[b], 0));
                check(cudaMemsetAsync(p.Dm_b[b], pattern, rows * kRow, p.s));
                p.stage_rows((float*) ((unsigned char*) output + at), b, rows, (size_t) group);
            }
            check(cudaSetDevice(primary));
            check(cudaEventRecord(p.ev_res, p.res_in));
        }
        check(cudaStreamSynchronize(p.res_in));
        check(cudaStreamSynchronize(ready));
        std::vector<unsigned char> got(expected.size()), got_inputs(inputs.size());
        check(cudaMemcpy(got.data(), output, got.size(), cudaMemcpyDeviceToHost));
        check(cudaSetDevice(peer));
        check(cudaStreamSynchronize(p.s));
        check(cudaMemcpy(got_inputs.data(), snapshots, got_inputs.size(), cudaMemcpyDeviceToHost));
        same = got == expected && got_inputs == inputs;
    }
    check(cudaSetDevice(primary));
    check(cudaStreamDestroy(ready));
    check(cudaFree(input));
    check(cudaFree(output));
    std::printf("peer transfer %d -> %d: %d layers x %d groups, input and result bytes %s\n",
                primary, peer, kLayers, kGroups, same ? "identical" : "DIFFER");
    return same;
}
}  // namespace

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) return 77;
    std::vector<int> devices;
    for (int d = 0; d < count; ++d) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, d) == cudaSuccess && prop.major == 12) devices.push_back(d);
    }
    if (devices.size() < 2) {
        std::printf("peer transfer: two Blackwell devices required, skipped\n");
        return 77;
    }
    const bool forward = run_pair(devices[0], devices[1]);
    const bool reverse = run_pair(devices[1], devices[0]);
    return forward && reverse ? 0 : 1;
}
