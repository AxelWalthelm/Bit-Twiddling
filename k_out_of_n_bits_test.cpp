
#include "k_out_of_n_bits.hpp"
#include "k_out_of_n_bits_test.h"
#include <stdio.h>
#include <stdint.h>
#include "HighResolutionTimer.h"

// Use assert() for test.
#ifdef NDEBUG
#undef NDEBUG
#include <assert.h>
#define NDEBUG
#else
#include <assert.h>
#endif

void k_out_of_n_bits_test()
{
	const int pars[][2] = { {0, 0}, {0, 1}, {1, 1}, {1, 2}, {2, 3}, {3, 5}, {4, 32}, {3, 64}
#ifdef NDEBUG
		, {8, 32} // too slow in debug mode
#endif
	};
	for (auto& par: pars)
	{
		int k = par[0];
		int n = par[1];
		k_out_of_n_bits<> seq(k, n);
		k_out_of_n_bits<> rnd(k, n);
		printf("k=%d n=%d\n", k, n);

		bool do_print = seq.count < 100;

		do
		{
			if (do_print) seq.print("seq");

			rnd.set_index(seq.index);
			if (do_print) rnd.print("rnd");

			assert(seq.k == rnd.k);
			assert(seq.n == rnd.n);
			assert(seq.index == rnd.index);
			for (int i = 0; i < seq.k; i++)
				assert(seq.counters[i] == rnd.counters[i]);
			for (int i = 0; i < seq.bit_array_items; i++)
				assert(seq.bit_array[i] == rnd.bit_array[i]);

		} while (seq.next());

		printf("\n");
	}
}
