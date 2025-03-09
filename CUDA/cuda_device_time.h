#pragma once
//
// Copyright 2025 Axel Walthelm
//
// Helper to be included in *.cu files that do timing in CUDA device code

#ifdef __CUDACC__
#include <stdint.h>

// Implemented as a namespace instead of a class because of cuda error
// "memory qualifier on data member is not allowed"
// on static class member clockRate.
// The class would be instantiated on device, but needs static initialization from host code,
// so conceptually the class would also be instantiated on the host.
namespace cuda_device_time
{
    __device__ __constant__
    extern int clockRate; // kHz

    __device__ inline
    long long int GetCudaTime()
    {
        return clock64();
    }

    __device__ inline
    uint64_t GetElapsedTimeUs(long long int start, long long int stop)
    {
        uint64_t diff = stop - start;
        return diff * 1000 / clockRate; // microseconds
    }

    __device__ inline
    uint64_t GetElapsedTimeUs(long long int start)
    {
        return GetElapsedTimeUs(start, GetCudaTime());
    }

    struct Timer
    {
        long long int start;

        __device__
        Timer() : start(GetCudaTime()) {}

        __device__
        uint64_t ElapsedTimeUs()
        {
            return GetElapsedTimeUs(start, GetCudaTime());
        }
    };
}

#endif // #ifdef __CUDACC__
