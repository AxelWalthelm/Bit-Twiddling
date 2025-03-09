//
// Copyright 2025 Axel Walthelm
//
#include "cuda_device_time.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include "cuda_and_host.h"

namespace cuda_device_time
{
    __device__ __constant__
    int clockRate = 0; // kHz

    bool DoStaticInit()
    {
        // Get clock_rate at program start from host code because:
        // - CUDA_OK(cudaDeviceGetAttribute(&clock_rate, cudaDevAttrClockRate, 0)); // error: calling a __host__ function("cudaDeviceGetAttribute") [...] is not allowed
        // - CUDA_OK(cudaGetDeviceProperty(&prop, 0)); // error: identifier "cudaGetDeviceProperty" is undefined

        int clock_rate; // kHz
        CUDA_OK(cudaDeviceGetAttribute(&clock_rate, cudaDevAttrClockRate, 0));
        CUDA_OK(cudaMemcpyToSymbol(clockRate, &clock_rate, sizeof(clockRate)));
        return true;
    }

    static bool isStaticInit = DoStaticInit();
}
