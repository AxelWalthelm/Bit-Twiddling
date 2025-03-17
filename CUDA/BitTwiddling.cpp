
#define PROFILE 0
#if PROFILE
#define IF_PROFILE(x) x
#else
#define IF_PROFILE(x)
#endif

#include "BitTwiddling.h"


// emulate CUDA kernel execution on host
void semi_exhaustive_search_for_8bit_rev_cpu(unsigned int gridDim_x, unsigned int blockDim_x, uint64_t start_index, int steps, int n_rep, int n_sel, int n_shi)
{
    for (unsigned int blockIdx_x = 0; blockIdx_x < gridDim_x; blockIdx_x++)
    {
        IF_PROFILE(auto start_time = GetHighResolutionTime());

        for (unsigned int threadIdx_x = 0; threadIdx_x < blockDim_x; threadIdx_x++)
        {
            semi_exhaustive_search_for_8bit_rev_impl(blockIdx_x, blockDim_x, threadIdx_x, start_index, steps, n_rep, n_sel, n_shi);
        }

        IF_PROFILE(auto stop_time = GetHighResolutionTime());
        IF_PROFILE(printf("cpu block %ld in %.3f us\n", blockIdx_x *(uint64_t) blockDim_x, GetHighResolutionTimeElapsedNs(start_time, stop_time) * 0.001));
    }
}
