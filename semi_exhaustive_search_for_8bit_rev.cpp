#include "semi_exhaustive_search_for_8bit_rev.h"
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include "HighResolutionTimer.h"
#include "k_out_of_n_bits.hpp"
#include "k_out_of_n_bits_test.h"
#include "BitTwiddling.hpp"

namespace {

	inline uint8_t bits_mul_and_mul(uint8_t x, uint32_t replicate, uint32_t select, uint32_t shift)
	{
		return uint8_t((((uint32_t(x) * replicate) & select) * shift) >> 24);
	}

	static constexpr uint8_t rev8_quick_tests[][2] =
	{
		{0xff, 0xff},
		{0x0f, 0xf0}, {0xcc, 0x33}, {0xaa, 0x55},
		{0xf0, 0x0f}, {0x33, 0xcc}, {0x55, 0xaa},
		{0x01, 0x80}, {0x02, 0x40}, {0x04, 0x20}, {0x08, 0x10},
		{0x80, 0x01}, {0x40, 0x02}, {0x20, 0x04}, {0x10, 0x08}
	};

	bool is_8bit_reverse(uint32_t replicate, uint32_t select, uint32_t shift)
	{
		constexpr uint8_t test_mask = 0xff;

		for (int i = 0; i < sizeof(rev8_quick_tests) / sizeof(*rev8_quick_tests); i++)
		{
			uint8_t value = bits_mul_and_mul(rev8_quick_tests[i][0], replicate, select, shift) & test_mask;
			uint8_t expected = rev8_quick_tests[i][1] & test_mask;
			if (value != expected)
				return false; // quick test failed
		}

		printf("full test: 0x%08x 0x%08x 0x%08x\n", replicate, select, shift);
		for (int i = 0; i < 256; i++)
		{
			uint8_t value = bits_mul_and_mul(i, replicate, select, shift) & test_mask;
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
		if (value)
			array[bit >> 5] |= uint32_t(1) << (bit & 31);
		else
			array[bit >> 5] &= ~(uint32_t(1) << (bit & 31));
	}

	inline
	void set_bit(uint32_t& replicate, uint32_t& select, uint32_t& shift, unsigned int bit, bool value)
	{
		if (bit >= 2 * 32)
			set_bit(&shift, bit - 2 * 32, value);
		else if (bit > 32)
			set_bit(&select, bit - 32, value);
		else
			set_bit(&replicate, bit, value);
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

	// a solution is expected to have this number of bits in its parameters
	// replicate: 4 (4-6...) // less than 4 does not give new options, because select can as well ignore any of the replications; more than 4 means that some relications overlap and some bits will be corrupted due to carry in multiplication (highly doubful, but it could lead to a solution)
	// select: 8 (8-10...) // less than 8 would mean not all input bits are used in the end result; more than 8 means some input bits are used muliple times (doubful, but it could lead to a solution)
	// shift: 4-8... // 4 means the input bits are shifted in 4 groups, 5 is more likely as this is used in 7 bit reversal, 8 means every input bit is shifted differently (doubtful, but may lead to a solution), more than 8 will probably lead to overlap and corrupted bits due to carry in multiplication  (highly doubful, but it could lead to a solution).

	constexpr int nr_bits = 3 * 32;
	constexpr int nr_bits_set_min = 17;
	constexpr int nr_bits_set_max = 24;
	int counters[nr_bits_set_max] = {};
	for (int nr_bits_set = nr_bits_set_min; nr_bits_set <= nr_bits_set_max; nr_bits_set++)
	{
		//printf("nr_bits_set: %d\n", nr_bits_set);

		uint32_t replicate = 0;
		uint32_t select = 0;
		uint32_t shift = 0;

		for (int i = 0; i < nr_bits_set; i++)
		{
			counters[i] = i;
			set_bit(replicate, select, shift, i, true);
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
				printf("  0x%08x 0x%08x 0x%08x\n", replicate, select, shift);
			}

			if (is_8bit_reverse(replicate, select, shift))
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
			set_bit(replicate, select, shift, counters[level], false);
			++counters[level];
			set_bit(replicate, select, shift, counters[level], true);
			// reset all overflown counters
			for (int l = level + 1; l < nr_bits_set; l++)
			{
				set_bit(replicate, select, shift, counters[l], false);
				counters[l] = counters[l-1] + 1;
				set_bit(replicate, select, shift, counters[l], true);
			}
		}
	}
}
