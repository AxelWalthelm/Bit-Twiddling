#pragma once
#include <assert.h>

template<typename INT = uint32_t>
class k_out_of_n_bits
{
	constexpr inline static size_t log2_of_pow2(size_t pow2) noexcept
	{
		return pow2 <= 1 ? 0 : 1 + log2_of_pow2(pow2 / 2);
	}

	constexpr static int INT_bit_count = sizeof(INT) * 8; // e.g. 32
	constexpr static size_t INT_addr_mask = INT_bit_count - 1; // e.g. 31
	constexpr static size_t INT_addr_shift = log2_of_pow2(INT_bit_count); // e.g. 5

	static_assert(INT_bit_count == (1 << INT_addr_shift), "INT must have power of 2 bits"); // e.g. INT can not be a 24 bit integer


	template<typename INT1, typename INT2>
	static inline bool multiplication_overflows(INT1 n1, INT2 n2)
	{
		return n2 != 0 && (n1 * n2) / n2 != n1;
	}

	template<typename INT1, typename INT2>
	static inline bool division_truncates(INT1 n1, INT2 n2)
	{
		return n2 != 0 && (n1 / n2) * n2 != n1;
	}

	static unsigned binomial_coefficient(unsigned n, unsigned k)
	{
		assert(k <= n);

		unsigned p = n - k;
		if ( k < p )
		{
			p = k;
			k = n - p;
		}

		unsigned r = p == 0 ? 1 : k + 1;
		for (unsigned i = 2; i <= p; i++)
		{
			assert(!multiplication_overflows(r, (k + i))); // with "unsigned long r" we would cover a larger range, but be slower (e.g. much slower on CUDA)
			assert(!division_truncates(r * (k + i), i)); // because r is always a combinatorial number
			r = (r * (k + i)) / i;
		}

		return r;
	}


	// Generate the [index]th lexicographically ordered set of [k] elements in [n] into counters,
	// where 0 <= index < binomial_coefficient(n, k) and
	// counters has k items with 0 <= counters[i] < n for 0 <= i < k.
	static void set_combination(int* counters, int n, int k, int index)
	{
		if (k <= 1)
		{
			if (k == 1)
				counters[0] = index;
			return;
		}

		int sum = 0; // sum of lexicographical steps done so far
		for (int ci = 0; ci < k - 1; ci++) // counter index
		{
			counters[ci] = ci == 0 ? -1 : counters[ci - 1];
			int steps; // lexicographical steps until counter ci overflows again
			do
			{
				counters[ci]++;
				steps = binomial_coefficient(n - 1 - counters[ci], k - 1 - ci);
				sum += steps;
			} while (sum <= index);
			sum -= steps;
		}
		counters[k-1] = counters[k-2] + 1 + index - sum;
	}

	static void invert_bit(INT* array, unsigned int bit) { array[bit >> INT_addr_shift] ^= INT(1) << (bit & INT_addr_mask); }

	// optimized version of "invert_bit(array, bit); invert_bit(array, bit + 1);"
	static void invert_2bits(INT* array, unsigned int bit)
	{
		unsigned int item = bit >> INT_addr_shift;
		unsigned int bit_in_item = bit & INT_addr_mask;
		array[item] ^= INT(3) << bit_in_item;
		if (bit_in_item == INT_bit_count - 1)
			array[item + 1] ^= INT(1);
	}

	INT get_bits(int offset = 0) const
	{
		INT bits = 0;
		for (int i = 0; i < k; i++)
			if (offset <= counters[i] && counters[i] < offset + INT_bit_count)
				bits |= INT(1) << (counters[i] - offset);

		return bits;
	}

public:

	const int k;
	const int n;
	const int count;
	int index;
	int* const counters;
	const int bit_array_items;
	INT* const bit_array;

	k_out_of_n_bits(int k, int n) :
		k(k),
		n(n),
		count(binomial_coefficient(n, k)),
		index(0),
		counters(new int[k]),
		bit_array_items((n + 31) / 32),
		bit_array(new INT[bit_array_items])
	{
		assert(n >= 0);
		assert(k >= 0);
		assert(k <= n);

		for (int i = 0; i < k; i++)
			counters[i] = i;

		for (int i = 0; i < bit_array_items; i++)
			bit_array[i] = get_bits(i * INT_bit_count);
	}


	~k_out_of_n_bits()
	{
		delete [] counters;
		delete [] bit_array;
	}


	void print(const char* name)
	{
		printf("k_out_of_n_bits %s(%d, %d) %3d/%d: ", name, k, n, index, count);
		for (int i = 0; i < k; i++)
			printf(" %2d", counters[i]);
		for (int i = 0; i < bit_array_items; i++)
			printf(" 0x%08x", bit_array[i]);
		printf("\n");
	}

	bool set_index(int index)
	{
		if (index < 0 || index >= count || k <= 0)
			return false;

		this->index = index;

		set_combination(counters, n, k, index);
		assert(counters[k - 1] < n);

		for (int i = 0; i < bit_array_items; i++)
			bit_array[i] = get_bits(i * INT_bit_count);

		return true;
	}

	bool next()
	{
		// find counter to increment
		int ci = k - 1;
		if (ci < 0)
			return false;
		if (counters[ci] >= n - 1)
			do
			{
				if (--ci < 0)
					return false;
			} while (counters[ci] + 1 >= counters[ci + 1]);

			// increment counter
			invert_2bits(bit_array, counters[ci]++);

			// reset all overflown counters
			for (int i = ci + 1; i < k; i++)
			{
				invert_bit(bit_array, counters[i]);
				counters[i] = counters[i-1] + 1;
				invert_bit(bit_array, counters[i]);
			}

			index++;

			return true;
	}
};
