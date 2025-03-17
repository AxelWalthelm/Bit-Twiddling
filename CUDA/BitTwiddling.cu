

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

#include "BitTwiddling.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include "../parallel_for_items.hpp"

#if defined(NDEBUG) && !defined(__OPTIMIZE__)
#pragma message("warning: CUDA host code not optimized")
#endif


void semi_exhaustive_search_for_8bit_rev_cuda()
{
    int n_rep = 1; // 3
    int n_sel = 8; // 8
    int n_shi = 5; // 4-6

    // iteration per iterations
    // blocks per iteration (iteration=grid)
    // threads per block
    // steps per thread

    generate_replicator replicate(n_rep);
    k_out_of_n_bits<uint32_t> select(n_sel, 32);
    k_out_of_n_bits<uint32_t> shift(n_shi, 32);
    const uint64_t N = generators::get_count(shift, select, replicate);
	printf("Total number of combinations: %" PRIu64 "\n", N);

    const int steps = 2000; // number of tests done in each thread; empiric value: threads run ca. 50 ms for 1000 steps; avoid timeout
    const int threads = 32; // threads per block; must be a multiple of 32; more than 32 is not better and even slightly bad
    const uint64_t blocks_total = div_ceil(N, threads * steps);
    const uint64_t blocks_x_max = 0x1000; //0x7fffffff; // CUDA 5 or higher: Maximum x-dimension of a grid of thread blocks [thread blocks] is 2^31-1
    const uint64_t iteration_blocks = std::min(blocks_total, blocks_x_max);
    const uint64_t iterations = div_ceil(blocks_total, iteration_blocks);
    printf("Total number of iterations: %" PRIu64 "\n", iterations);
    printf("Combinations per iteration: %" PRIu64 " >= %" PRIu64 "\n", iteration_blocks * threads * steps, div_ceil(N, iterations));
    parallel_for_range(21642 /*div_ceil(uint64_t(35) * select.count * shift.count, iteration_blocks * threads * steps)*/, iterations,
        [=](uint64_t iteration)
        {
            auto start_time = GetHighResolutionTime();
            uint64_t start_index = iteration * iteration_blocks * threads * steps;
            semi_exhaustive_search_for_8bit_rev_cpu((uint32_t)iteration_blocks, threads, start_index, steps, n_rep, n_sel, n_shi);
            printf("%" PRIu64 "/%" PRIu64 " CPU in %.3fs\n", iteration, iterations, GetHighResolutionTimeElapsedNs(start_time) * 1e-9);
        },
        [=](uint64_t iteration)
        {
            auto start_time = GetHighResolutionTime();
            uint64_t start_index = iteration * iteration_blocks * threads * steps;
            CUDA_DO_SYNC(semi_exhaustive_search_for_8bit_rev_kernel<<<(uint32_t)iteration_blocks, threads>>>(start_index, steps, n_rep, n_sel, n_shi));
            printf("%" PRIu64 "/%" PRIu64 " GPU in %.3fs\n", iteration, iterations, GetHighResolutionTimeElapsedNs(start_time) * 1e-9);

            static int next_long_print_in = 0;
            if (--next_long_print_in <= 0)
            {
                next_long_print_in = 10;

                generate_replicator replicate(n_rep);
                k_out_of_n_bits<uint32_t> select(n_sel, 32);
                k_out_of_n_bits<uint32_t> shift(n_shi, 32);

                generators::set_index(start_index, shift, select, replicate);
                printf("counters:");
                replicate.print("replicate", " ", " ");
                select.print("select", " ", " ");
                shift.print("shift", " ", "\n");

                uint64_t new_index = generators::get_index(shift, select, replicate);
                if (new_index > start_index)
                {
                    uint64_t new_item = new_index / (iteration_blocks * threads * steps);
                    return new_item > 2 ? new_item - 2 : 0; // request to skip items
                }
            }

            return iteration;
        });

    printf("Done.");
}
