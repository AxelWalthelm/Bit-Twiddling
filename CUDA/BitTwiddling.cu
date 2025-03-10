#include "BitTwiddling.h"
#include <stdio.h>
#include <inttypes.h>
#include <memory>
#include <cuda_runtime.h>
#include <cuda.h>

#include "cuda_and_host.h"
#include "cuda_math.h"

#include "../k_out_of_n_bits.hpp"
#include "../semi_exhaustive_search_for_8bit_rev.hpp"

#if defined(NDEBUG) && !defined(__OPTIMIZE__)
#pragma message("warning: CUDA host code not optimized")
#endif

#define PROFILE 0
#if PROFILE
#define IF_PROFILE(x) x
#else
#define IF_PROFILE(x)
#endif

#define CUDA_TIMER 0
#if CUDA_TIMER
#include "cuda_device_time.h"
#define IF_CUDA_TIMER(x) x
#else
#define IF_CUDA_TIMER(x)
#endif

namespace{


__global__
void semi_exhaustive_search_for_8bit_rev_kernel(uint64_t start, int steps, int n_rep, int n_sel, int n_shi)
{
    IF_CUDA_TIMER(cuda_device_time::Timer timer);

    assert(blockDim.y == 1);
    assert(blockDim.z == 1);
    const uint64_t block_index = blockIdx.x *(uint64_t) blockDim.x + threadIdx.x;
    const uint64_t index = start + block_index * steps;

    k_out_of_n_bits<uint32_t> replicate(n_rep, 32);
    k_out_of_n_bits<uint32_t> select(n_sel, 32);
    k_out_of_n_bits<uint32_t> shift(n_shi, 32);

    if (generators::set_index(index, shift, select, replicate))
    {
        for (int step = 0; step < steps; step++)
        {
            if (is_8bit_reverse(*replicate, *select, *shift))
            {
                printf("solution!!!\n");
                printf("counters:");
                replicate.print("replicate", " ", " ");
                select.print("select", " ", " ");
                shift.print("shift", " ", "\n");
            }

            if (!generators::next(shift, select, replicate))
                break;
        }
    }

    IF_CUDA_TIMER(__syncthreads(); if (threadIdx.x == 0) printf("block %lld in %lld us\n", block_index, timer.ElapsedTimeUs()));
}

} // anonymous namespace


void semi_exhaustive_search_for_8bit_rev_cuda()
{
    int n_rep = 4;
    int n_sel = 8;
    int n_shi = 5;

    k_out_of_n_bits<uint32_t> replicate(n_rep, 32);
    k_out_of_n_bits<uint32_t> select(n_sel, 32);
    k_out_of_n_bits<uint32_t> shift(n_shi, 32);
    const uint64_t N = generators::get_count(shift, select, replicate);
	printf("Total number of combinations: %" PRIu64 "\n", N);

    const int steps = 2000; // number of tests done in each thread; empiric value: threads run ca. 50 ms for 1000 steps; avoid timeout
    const int threads = 32; // threads per block; must be a multiple of 32; more than 32 is no good and even slightly bad
    const uint64_t blocks_total = div_ceil(N, threads * (uint64_t)steps);
    const uint64_t blocks_x_max = 0x1000; //0x7fffffff; // CUDA 5 or higher: Maximum x-dimension of a grid of thread blocks [thread blocks] is 2^31-1
    const uint64_t iteration_blocks = std::min(blocks_total, blocks_x_max);
    const uint64_t iterations = div_ceil(blocks_total, iteration_blocks);
	printf("Total number of iterations: %" PRIu64 "\n", iterations);
    for (uint64_t iteration = 0; iteration < iterations; iteration++)
    {
        uint64_t start = iteration * iteration_blocks;
        printf("%" PRIu64 "/%" PRIu64 ": semi_exhaustive_search_for_8bit_rev_kernel<<<0x%08x, %d>>>(0x%lx, 0x%x, %d, %d, %d)\n",
            iteration, iterations, (uint32_t)iteration_blocks, threads, start, steps, n_rep, n_sel, n_shi);

        CUDA_DO_SYNC(semi_exhaustive_search_for_8bit_rev_kernel<<<(uint32_t)iteration_blocks, threads>>>(start, steps, n_rep, n_sel, n_shi));
    }

    printf("Done.");
}
