#include "semi_exhaustive_search_for_8bit_rev.h"
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include "HighResolutionTimer.h"
#include "k_out_of_n_bits.hpp"
#include "k_out_of_n_bits_test.h"
#include "BitTwiddling.hpp"

namespace {

	inline uint8_t bits_mul_and_mul(uint8_t x, uint32_t param[3])
	{
		//if (param[1] & 2) return bits_rev8(x);
		return uint8_t((((uint32_t(x) * param[0]) & param[1]) * param[2]) >> 24);
	}

	static constexpr uint8_t rev8_quick_tests[][2] =
	{
		{0xff, 0xff},
		{0x0f, 0xf0}, {0xcc, 0x33}, {0xaa, 0x55},
		{0xf0, 0x0f}, {0x33, 0xcc}, {0x55, 0xaa},
		{0x01, 0x80}, {0x02, 0x40}, {0x04, 0x20}, {0x08, 0x10},
		{0x80, 0x01}, {0x40, 0x02}, {0x20, 0x04}, {0x10, 0x08}
	};

	bool is_8bit_reverse(uint32_t params[3])
	{
		constexpr uint8_t test_mask = 0xff;

		for (int i = 0; i < sizeof(rev8_quick_tests) / sizeof(*rev8_quick_tests); i++)
		{
			uint8_t value = bits_mul_and_mul(rev8_quick_tests[i][0], params) & test_mask;
			uint8_t expected = rev8_quick_tests[i][1] & test_mask;
			if (value != expected)
				return false; // quick test failed
		}

		printf("full test: 0x%08x 0x%08x 0x%08x\n", params[0], params[1], params[2]);
		for (int i = 0; i < 256; i++)
		{
			uint8_t value = bits_mul_and_mul(i, params) & test_mask;
			uint8_t expected = bits_rev8(i) & test_mask;
			printf("    i=0x%02x value=0x%02x expected=0x%02x\n", i, value, expected);
			if (value != expected)
				return false;
		}

		return true;
	}

	inline
		void set_bit(uint32_t* array, unsigned int bit, bool value)
	{
		//bit = (bit >> 3) | ((bit & 7) << (3 * 32 - 1 - 3));
		if (value)
			array[bit >> 5] |= uint32_t(1) << (bit & 31);
		else
			array[bit >> 5] &= ~(uint32_t(1) << (bit & 31));
	}

} // anonymous namespace


void semi_exhaustive_search_for_8bit_rev()
{
	HighResolutionTime_t last_print_time = 0;

	// check correctness of quick test table
	for (int i = 0; i < sizeof(rev8_quick_tests) / sizeof(*rev8_quick_tests); i++)
	{
		assert(bits_rev8(rev8_quick_tests[i][0]) == rev8_quick_tests[i][1]);
	}


	constexpr int nr_bits = 3 * 32;
	constexpr int nr_bits_set_min = 17;
	constexpr int nr_bits_set_max = 24;
	int counters[nr_bits_set_max] = {};
	for (int nr_bits_set = nr_bits_set_min; nr_bits_set <= nr_bits_set_max; nr_bits_set++)
	{
		//printf("nr_bits_set: %d\n", nr_bits_set);

		uint32_t params[3] = {};

		for (int i = 0; i < nr_bits_set; i++)
		{
			counters[i] = i;
			set_bit(params, i, true);
		}

		while (counters[nr_bits_set - 1] < nr_bits)
		{
			// use counters
			HighResolutionTime_t now = GetHighResolutionTime();
			if (last_print_time == 0 || GetHighResolutionTimeElapsedNs(last_print_time, now) >= 500000000)
			{
				last_print_time = now;
				printf("counters: %2d/%2d-%2d ", nr_bits_set, nr_bits_set_min, nr_bits_set_max);
				for (int i = 0; i < nr_bits_set; i++)
				{
					printf(" %2d", counters[i]);
				}
				printf("  0x%08x 0x%08x 0x%08x\n", params[0], params[1], params[2]);
			}

			if (is_8bit_reverse(params))
			{
				printf("solution!!!\n");
				getchar();
			}

			// find counter to increment
			int level = nr_bits_set - 1;
			if (counters[level] + 1 >= nr_bits)
			{
				--level;
				while (level >= 0 && counters[level] + 1 >= counters[level + 1])
					--level;

				if (level < 0)
					break;
			}

			// increment counter
			set_bit(params, counters[level], false);
			++counters[level];
			set_bit(params, counters[level], true);
			// reset all overflown counters
			for (int l = level + 1; l < nr_bits_set; l++)
			{
				set_bit(params, counters[l], false);
				counters[l] = counters[l-1] + 1;
				set_bit(params, counters[l], true);
			}
		}
	}
}
