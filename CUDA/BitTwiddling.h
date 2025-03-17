#pragma once

#include <stdio.h>
#include <inttypes.h>
#include <memory>

#include "cuda_and_host.h"
#include "cuda_math.h"
#include "HighResolutionTimer.h"

#include "../k_out_of_n_bits.hpp"
#include "../semi_exhaustive_search_for_8bit_rev.hpp"

void semi_exhaustive_search_for_8bit_rev_cuda();
void semi_exhaustive_search_for_8bit_rev_cpu(unsigned int gridDim_x, unsigned int blockDim_x, uint64_t start_index, int steps, int n_rep, int n_sel, int n_shi);

namespace{

strong_inline CUDA_ALL
void semi_exhaustive_search_for_8bit_rev_impl(unsigned int blockIdx_x, unsigned int blockDim_x, unsigned int threadIdx_x, uint64_t start_index, int steps, int n_rep, int n_sel, int n_shi)
{
    const uint64_t block_index = blockIdx_x *(uint64_t) blockDim_x + threadIdx_x;
    const uint64_t index = start_index + block_index * steps;

    generate_replicator replicate(n_rep);
    k_out_of_n_bits<uint32_t> select(n_sel, 32);
    k_out_of_n_bits<uint32_t> shift(n_shi, 32);

    if (generators::set_index_of(steps, index, shift, select, replicate))
    {
        //uint64_t istart = generators::get_index(shift, select, replicate);
        //printf("%08lx-\n", istart);
        do
        {
            if (is_8bit_reverse(*replicate, *select, *shift))
            {
                printf("solution at index=%" PRIu64 "!!!\n", generators::get_index(shift, select, replicate));
                printf("counters:");
                replicate.print("replicate", " ", " ");
                select.print("select", " ", " ");
                shift.print("shift", " ", "\n");
            }
        }
        while (generators::next_of(steps, shift, select, replicate));
        //uint64_t istop = generators::get_index(shift, select, replicate);
        //printf("%" PRIu64 "-%" PRIu64 " \r", istart, istop);
    }
    //if (generators::get_index(shift, select, replicate) >= generators::get_count(shift, select, replicate)) printf("\n");
}

#ifdef __CUDACC__

__global__
void semi_exhaustive_search_for_8bit_rev_kernel(uint64_t start_index, int steps, int n_rep, int n_sel, int n_shi)
{
    IF_CUDA_TIMER(cuda_device_time::Timer timer);

    assert(blockDim.y == 1);
    assert(blockDim.z == 1);
    semi_exhaustive_search_for_8bit_rev_impl(blockIdx.x, blockDim.x, threadIdx.x, start_index, steps, n_rep, n_sel, n_shi);

    IF_CUDA_TIMER(__syncthreads(); if (threadIdx.x == 0) printf("block %lld in %lld us\n", blockIdx.x *(uint64_t) blockDim.x, timer.ElapsedTimeUs()));
}

#endif // #ifdef __CUDACC__

} // anonymous namespace
