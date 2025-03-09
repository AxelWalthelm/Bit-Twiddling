//
// Copyright 2025 Axel Walthelm
//
// Some definitions and mappings that allow code to be compiled as CUDA device code as well as host code.
#if defined(min) || defined(max) || defined(abs)
#error "min/max/abs must not be macros"
#endif

#ifndef CUDA_AND_HOST_H_INCLUDED
#define CUDA_AND_HOST_H_INCLUDED

#include <stdio.h>

#ifndef __CUDACC__
#include <cmath>
#include <algorithm>
using std::min;
using std::max;
using std::abs;
#endif


#define STRINGIFY(...) STRINGIFY2(__VA_ARGS__)
#define STRINGIFY2(...) #__VA_ARGS__


#ifdef __GNUC__
#define strong_inline __attribute__((always_inline)) inline
#else
#define strong_inline inline
#endif

// C++ functions that are intended to be used by normal C++ compiler as well as
// CUDA compiler host code, and CUDA compiler device code are marked by CUDA_ALL.
// Functions that are CUDA device code only should be surrounded by "#ifdef __CUDACC__" and marked as "__device__".
#ifdef __CUDACC__
#define CUDA_ALL __host__ __device__
#else
#define CUDA_ALL
#endif

#ifdef __CUDACC__
namespace
{

#ifdef __CUDA_ARCH__
#define CUDA_exit(error) assert(0)
#define CUDA_error() cudaSuccess
#define CUDA_error_string(error) "on device"
#else
#define CUDA_exit(error) exit(error)
#define CUDA_error() cudaGetLastError()
const char* CUDA_error_string(cudaError_t error) { const char* s = cudaGetErrorString(error); return s && *s ? s : "???"; }
#endif

CUDA_ALL
void CUDA_errfnc(cudaError_t last_error, cudaError_t expr_error, const char* expression, const char* function_name, const char* file_name, int line_number, int do_sync = 0)
{
    const bool is_old = last_error != cudaSuccess;
    if (!is_old)
        last_error = CUDA_error();
    if (last_error != cudaSuccess && last_error != expr_error)
    {
        printf("%s:%d: CUDA error %d: \"%s\" %s '%s' in function '%s'.\n",
            file_name, line_number, last_error, CUDA_error_string(last_error), is_old ? "before" : "after", expression, function_name);
    }

    if (expr_error != cudaSuccess)
    {
        printf("%s:%d: CUDA error %d: \"%s\" %s '%s' in function '%s'.\n",
            file_name, line_number, expr_error, CUDA_error_string(expr_error), do_sync != 2 ? "in" : "in cudaDeviceSynchronize() after", expression, function_name);
    }

    if (expr_error != cudaSuccess || last_error != cudaSuccess)
    {
        CUDA_exit(expr_error != cudaSuccess ? expr_error : last_error);
    }

#ifndef __CUDA_ARCH__
    if (do_sync == 1)
    {
        CUDA_errfnc(cudaSuccess, cudaDeviceSynchronize(), expression, function_name, file_name, line_number, 2);
    }
#endif
}

#define CUDA_OK(...) do { cudaError_t e=CUDA_error(); CUDA_errfnc(e, e==cudaSuccess?(__VA_ARGS__):cudaSuccess, #__VA_ARGS__, __func__, __FILE__, __LINE__); } while(0)

#ifndef __CUDA_ARCH__
#define CUDA_DO(...) do { cudaError_t e=CUDA_error(); if(e==cudaSuccess){__VA_ARGS__;} CUDA_errfnc(e, cudaSuccess, #__VA_ARGS__, __func__, __FILE__, __LINE__); } while(0)
#define CUDA_DO_SYNC(...) do { cudaError_t e=CUDA_error(); if(e==cudaSuccess){__VA_ARGS__;} CUDA_errfnc(e, cudaSuccess, #__VA_ARGS__, __func__, __FILE__, __LINE__, 1); } while(0)
#else
#define CUDA_DO(...) __VA_ARGS__
#define CUDA_DO_SYNC(...) __VA_ARGS__; cudaDeviceSynchronize()
#endif

} // namespace
#endif // #ifdef __CUDACC__


CUDA_ALL static
void CUDA_throwfnc(const char* expression, const char* function_name, const char* file_name, int line_number)
{
    printf("%s:%d: CUDA_THROW(%s) in function '%s'.\n", file_name, line_number, expression, function_name);
}

#ifndef __CUDA_ARCH__
#include <stdexcept>
#define CUDA_THROW(...) do { CUDA_throwfnc(#__VA_ARGS__, __func__, __FILE__, __LINE__); throw std::runtime_error(#__VA_ARGS__); } while(0)
#else
#define CUDA_THROW(...) do { CUDA_throwfnc(#__VA_ARGS__, __func__, __FILE__, __LINE__); assert(!#__VA_ARGS__); } while(0)
#endif
#define THROW_NOT_IMPLEMENTED CUDA_THROW(not implemented yet)


struct MemoryPool
{
    // TODO/TBD: use enum cudaMemoryType instead?
    enum class Type { None, Malloc, Cuda, CudaHost }; // note: new[] and delete[] must use correct types => not supported

    CUDA_ALL
    static void* malloc(size_t bytes, MemoryPool::Type pool = MemoryPool::Type::Malloc)
    {
        switch (pool)
        {
            case Type::Malloc:   return ::malloc(bytes);
#if defined(__CUDACC__) && !defined(__CUDA_ARCH__)
            case Type::Cuda:     { void* m; CUDA_OK(cudaMalloc(&m, bytes)); return m; }
            case Type::CudaHost: { void* m; CUDA_OK(cudaMallocHost(&m, bytes)); return m; }
#endif
            default: CUDA_THROW(unsupported memory pool); return nullptr;
        }
    }

    CUDA_ALL
    static void free(void* memory, MemoryPool::Type pool)
    {
        // TODO/TBD: use cudaPointerGetAttributes() to determine memory pool?
        switch (pool)
        {
            case Type::None:     return;
            case Type::Malloc:   ::free(memory); return;
#if defined(__CUDACC__) && !defined(__CUDA_ARCH__)
            case Type::Cuda:     CUDA_OK(cudaFree(memory)); return;
            case Type::CudaHost: CUDA_OK(cudaFreeHost(memory)); return;
#endif
            default: CUDA_THROW(unsupported memory pool);
        }
    }
};

#endif // CUDA_AND_HOST_H_INCLUDED
