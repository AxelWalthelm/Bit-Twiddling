#pragma once
#include <assert.h>
#include <stdint.h>
#include <limits.h>
#include <limits>

namespace
{
// The following numeric checks are intended to be used in debug assert
// to detect numeric boundary problems during debug sessions.
// These checks are slow and may not catch all overflow/underflow problems.
// Future hint: better use standard checks like stdckdint.h from C23

template<typename NUM1, typename NUM2>
static inline bool addition_overflows(NUM1 n1, NUM2 n2)
{
	// n2 > 0: n1 + n2 > MAX
	// n2 < 0: n1 + n2 < MIN
	return n2 >= 0
		? n1 > std::numeric_limits<decltype(n1 + n2)>::max() - n2
		: n1 < std::numeric_limits<decltype(n1 + n2)>::lowest() - n2;
}

template<typename NUM1, typename NUM2>
static inline bool multiplication_overflows(NUM1 n1, NUM2 n2)
{
	// n1 > 0, n2 > 0: n1 * n2 > MAX
	// n1 > 0, n2 < 0: n1 * n2 < MIN
	// n1 < 0, n2 > 0: n1 * n2 < MIN
	// n1 < 0, n2 < 0: n1 * n2 > MAX
	return (n1 < 0) == (n2 < 0)
		? n1 > std::numeric_limits<decltype(n1 + n2)>::max() / n2
		: n1 < std::numeric_limits<decltype(n1 + n2)>::lowest() / n2;
}

template<typename INT1, typename INT2>
static inline bool division_truncates(INT1 n1, INT2 n2)
{
	return n2 != 0 && (n1 / n2) * n2 != n1;
}

template<typename NUM1, typename NUM2>
static inline bool numeric_cast_fails(NUM2 n2)
{
	NUM1 n1 = static_cast<NUM1>(n2);
	return n1 != n2 || (n1 < 0) != (n2 < 0);
}

} // anonymous namespace

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


	void reset()
	{
		index = 0;

		for (int i = 0; i < k; i++)
			counters[i] = i;

		for (int i = 0; i < bit_array_items; i++)
			bit_array[i] = get_bits(i * INT_bit_count);
	}


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

		reset();
	}


	// For best performance copies shall be avoided. Note: default copy constructor does not work.
	k_out_of_n_bits(k_out_of_n_bits const& other) = delete;


	~k_out_of_n_bits()
	{
		delete [] counters;
		delete [] bit_array;
	}


	void print(const char* name, const char* begin = "k_out_of_n_bits ", const char* end = "\n")
	{
		printf("%s%s(%d, %d) %3d/%d: ", begin, name, k, n, index, count);
		for (int i = 0; i < k; i++)
			printf(" %2d", counters[i]);
		for (int i = 0; i < bit_array_items; i++)
			printf(" 0x%08x", bit_array[i]);
		printf(end);
	}

	INT const& get(int item = 0) const { assert(0 <= item && item < bit_array_items); return bit_array[item]; }

	INT const& operator*() const { return get(); }

	INT const& operator[](int i) const { return get(i); }

	int get_count() const { return count; }

	int get_index() const { return index; }

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

	bool set_index(uint64_t index)
	{
		assert(!numeric_cast_fails<int>(index));
		return index <= INT_MAX && set_index(int(index));
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

namespace generators
{
	// A recursive variadic function to increment multiple generators like k_out_of_n_bits

	template <typename TGenerator>
	inline bool next(TGenerator& generator) 
	{
		return generator.next();
	}

	template<typename TGenerator, typename ... TGenerators>
	inline bool next(TGenerator& generator, TGenerators& ... generators)
	{
		if (generator.next())
			return true;

		generator.reset();

		return next(generators ...);
	}


	// A recursive variadic function to get the possible number of combinations of multiple generators like k_out_of_n_bits

	template <typename TGenerator>
	inline uint64_t get_count(TGenerator const& generator) 
	{
		return generator.get_count();
	}

	template<typename TGenerator, typename ... TGenerators>
	inline uint64_t get_count(TGenerator const& generator, TGenerators const& ... generators)
	{
		uint64_t count = generator.get_count();
		uint64_t counts = get_count(generators ...);
		assert(!multiplication_overflows(count, counts));
		return count * counts;
	}


	// A recursive variadic function to get index of multiple generators like k_out_of_n_bits

	template <typename TGenerator>
	inline uint64_t get_index(TGenerator const& generator) 
	{
		return generator.get_index();
	}

	template<typename TGenerator, typename ... TGenerators>
	inline uint64_t get_index(TGenerator const& generator, TGenerators const& ... generators)
	{
		uint64_t count1 = generator.get_count();
		uint64_t index1 = generator.get_index();
		uint64_t index2 = get_index(generators ...);
		assert(!multiplication_overflows(index2, count1));
		assert(!addition_overflows(index2 * count1, index1));
		uint64_t index = index2 * count1 + index1;
		return index;
	}


	// A recursive variadic function to set index of multiple generators like k_out_of_n_bits

	template <typename TGenerator>
	inline bool set_index(uint64_t index, TGenerator& generator) 
	{
		return index < generator.get_count() && generator.set_index(index);
	}

	template<typename TGenerator, typename ... TGenerators>
	inline bool set_index(uint64_t index, TGenerator& generator, TGenerators& ... generators)
	{
		auto count = generator.get_count();
		auto div = index / count;
		auto rem = index % count;
		return set_index(div, generators ...) && generator.set_index(rem);
	}
}
