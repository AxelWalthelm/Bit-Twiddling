
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
	const int pars[][2] = { {0, 0}, {0, 1}, {1, 1}, {1, 2}, {2, 3}, {3, 5}, {4, 32}, {3, 64}, {8, 32}
	};
	for (auto& par: pars)
	{
		int k = par[0];
		int n = par[1];
		k_out_of_n_bits<> seq(k, n);
		k_out_of_n_bits<> rnd(k, n);
		printf("k=%d n=%d\n", k, n);

		do
		{
			bool do_print = seq.index < 100;

			if (do_print) seq.print("seq");

			rnd.set_index(seq.index);
			if (do_print) rnd.print("rnd");

			if (seq.index > 1000000)
			{
				break;
			}

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

	for (int pi = 0; pi < sizeof(pars) / sizeof(*pars) - 1; pi++)
	{
		int k1 = pars[pi][0];
		int n1 = pars[pi][1];
		int k2 = pars[pi + 1][0];
		int n2 = pars[pi + 1][1];
		k_out_of_n_bits<> seq1(k1, n1);
		k_out_of_n_bits<> rnd1(k1, n1);
		k_out_of_n_bits<> seq2(k2, n2);
		k_out_of_n_bits<> rnd2(k2, n2);
		printf("k1=%d n1=%d  k2=%d n2=%d\n", k1, n1, k2, n2);

		printf("Total number of combinations: %lld\n", generators::get_count(seq1, seq2));
		assert(generators::get_count(seq1, seq2) == generators::get_count(rnd1, rnd2));

		do
		{
			auto index = generators::get_index(seq1, seq2);

			bool do_print = index < 100;

			if (do_print)
			{
				seq1.print("seq1", "", " ");
				seq1.print("seq2", "");
			}

			generators::set_index(index, rnd1, rnd2);
			if (do_print)
			{
				rnd1.print("rnd1", "", " ");
				rnd1.print("rnd2", "");
			}

			if (index > 1000000)
			{
				break;
			}


			assert(index == generators::get_index(rnd1, rnd2));
			for (int i = 0; i < seq1.k; i++)
				assert(seq1.counters[i] == rnd1.counters[i]);
			for (int i = 0; i < seq2.k; i++)
				assert(seq2.counters[i] == rnd2.counters[i]);

			for (int i = 0; i < seq1.k; i++)
				assert(seq1.counters[i] == rnd1.counters[i]);
			for (int i = 0; i < seq2.k; i++)
				assert(seq2.counters[i] == rnd2.counters[i]);

			for (int i = 0; i < seq1.bit_array_items; i++)
				assert(seq1.bit_array[i] == rnd1.bit_array[i]);
			for (int i = 0; i < seq2.bit_array_items; i++)
				assert(seq2.bit_array[i] == rnd2.bit_array[i]);

		} while (generators::next(seq1, seq2));

		printf("\n");
	}
}
