#include "semi_exhaustive_search_for_8bit_rev.h"
#include "semi_exhaustive_search_for_8bit_rev.hpp"
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <assert.h>
#include "HighResolutionTimer.h"
#include "k_out_of_n_bits.hpp"
#include "k_out_of_n_bits_test.h"


void semi_exhaustive_search_for_8bit_rev()
{
	HighResolutionTime_t last_print_time = 0;

#if 0
	// check correctness of quick test table
	for (int i = 0; i < int(sizeof(rev8_quick_tests) / sizeof(*rev8_quick_tests)); i++)
	{
		assert(bits_rev8(rev8_quick_tests[i][0]) == rev8_quick_tests[i][1]);
	}
#endif

	// a solution is expected to have this number of bits in its parameters
	// replicate: 4 (4-6...) // less than 4 does not give new options, because select can as well ignore any of the replications; more than 4 means that some relications overlap and some bits will be corrupted due to carry in multiplication (highly doubful, but it could lead to a solution)
	// select: 8 (8-10...) // less than 8 would mean not all input bits are used in the end result; more than 8 means some input bits are used muliple times (doubful, but it could lead to a solution)
	// shift: 4-8... // 4 means the input bits are shifted in 4 groups, 5 is more likely as this is used in 7 bit reversal, 8 means every input bit is shifted differently (doubtful, but may lead to a solution), more than 8 will probably lead to overlap and corrupted bits due to carry in multiplication  (highly doubful, but it could lead to a solution).

	generate_replicator       replicate(2);
	k_out_of_n_bits<uint32_t> select(8, 32);
	k_out_of_n_bits<uint32_t> shift(5, 32);

	printf("Total number of combinations: %" PRIu64 "\n", generators::get_count(shift, select, replicate));

	do
	{
		// use counters
		HighResolutionTime_t now = GetHighResolutionTime();
		if (last_print_time == 0 || GetHighResolutionTimeElapsedNs(last_print_time, now) >= 500000000)
		{
			last_print_time = now;
			printf("counters:");
			replicate.print("replicate", " ", " ");
			select.print("select", " ", " ");
			shift.print("shift", " ", "\n");
		}

		if (is_8bit_reverse(*replicate, *select, shift[0]))
		{
			printf("solution!!!\n");
			getchar();
		}
	} while (generators::next(shift, select, replicate));
}
