// Device tables and counters shared by the main and peer windows.
#include "basalt/models/qwen4exp/kernels/dec_moe.hpp"
#include "basalt/log.hpp"
#include <cuda_runtime.h>

namespace basalt::kernels {
__global__ void dm_table_put_kernel(const __grid_constant__ DmTablePut a) {
    for (int i = (int) threadIdx.x; i < a.n; i += (int) blockDim.x) a.tab[a.idx[i]] = a.val[i];
}

__global__ void dm_zero_words_kernel(unsigned int* p, int n) {
    if ((int) threadIdx.x < n) p[threadIdx.x] = 0u;
}

bool dec_moe_table_put(const DmTablePut& a, void* stream) {
    if (a.n <= 0) return true;
    void* params[] = {(void*) &a};
    const cudaError_t e = cudaLaunchKernel((const void*) dm_table_put_kernel, dim3(1), dim3(128), params, 0,
                                           (cudaStream_t) stream);
    if (e != cudaSuccess) {
        basalt::log::error("basalt", "dec_moe_table_put: %s", cudaGetErrorString(e));
        return false;
    }
    return true;
}

bool dec_moe_table_load(void* stream) {
    DmTablePut a;
    unsigned int* p = nullptr;
    int n = 0;
    void* pa[] = {(void*) &a};
    void* pz[] = {(void*) &p, (void*) &n};
    const cudaError_t e = cudaLaunchKernel((const void*) dm_table_put_kernel, dim3(1), dim3(128), pa, 0,
                                           (cudaStream_t) stream);
    const cudaError_t f = e != cudaSuccess ? e : cudaLaunchKernel((const void*) dm_zero_words_kernel, dim3(1), dim3(256),
                                                                    pz, 0, (cudaStream_t) stream);
    if (f != cudaSuccess) {
        basalt::log::error("basalt", "dec_moe_table_load: %s", cudaGetErrorString(f));
        return false;
    }
    return true;
}

bool dec_moe_zero_words(unsigned int* p, int n, void* stream) {
    if (n <= 0) return true;
    if (n > 256) return false;
    void* params[] = {(void*) &p, (void*) &n};
    const cudaError_t e = cudaLaunchKernel((const void*) dm_zero_words_kernel, dim3(1), dim3(256), params, 0,
                                           (cudaStream_t) stream);
    if (e != cudaSuccess) {
        basalt::log::error("basalt", "dec_moe_zero_words: %s", cudaGetErrorString(e));
        return false;
    }
    return true;
}


}  // namespace basalt::kernels
