#pragma once
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <limits.h>
#include <limits>

#ifndef CUDA_ALL
#define CUDA_ALL
#endif

namespace
{
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
// ignored by CUDA nvcc?
#pragma GCC diagnostic ignored "-Wtype-limits"
//#pragma GCC diagnostic ignored "-Wtautological-compare"
//#pragma GCC diagnostic ignored "-Wfloat-equal"
//#pragma GCC diagnostic ignored "-Wsign-conversion"
//#pragma GCC diagnostic ignored "-Wfloat-conversion"
//#pragma GCC diagnostic ignored "-Wuseless-cast"
#endif

// The following numeric checks are intended to be used in debug assert
// to detect numeric boundary problems during debug sessions.
// These checks are slow and may not catch all overflow/underflow problems.
// Future hint: better use standard checks like stdckdint.h from C23

template<typename NUM1, typename NUM2>
CUDA_ALL
static inline bool addition_overflows(NUM1 n1, NUM2 n2)
{
	// n2 > 0: n1 + n2 > MAX
	// n2 < 0: n1 + n2 < MIN
	return n2 >= 0
		? n1 > std::numeric_limits<decltype(n1 + n2)>::max() - n2
		: n1 < std::numeric_limits<decltype(n1 + n2)>::lowest() - n2;
}

template<typename NUM1, typename NUM2>
CUDA_ALL
static inline bool multiplication_overflows(NUM1 n1, NUM2 n2)
{
	// n1 > 0, n2 > 0: n1 * n2 > MAX
	// n1 > 0, n2 < 0: n1 * n2 < MIN
	// n1 < 0, n2 > 0: n1 * n2 < MIN
	// n1 < 0, n2 < 0: n1 * n2 > MAX
	return (n1 <= 0) == (n2 <= 0)
		? n1 > std::numeric_limits<decltype(n1 + n2)>::max() / n2
		: n1 < std::numeric_limits<decltype(n1 + n2)>::lowest() / n2;
}

template<typename INT1, typename INT2>
CUDA_ALL
static inline bool division_truncates(INT1 n1, INT2 n2)
{
	return n2 != 0 && (n1 / n2) * n2 != n1;
}

template<typename NUM1, typename NUM2>
CUDA_ALL
static inline bool numeric_cast_fails(NUM2 n2)
{
	NUM1 n1 = static_cast<NUM1>(n2);
	return n1 != n2 || (n1 <= 0) != (n2 <= 0);
}

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
} // anonymous namespace

namespace
{
	CUDA_ALL
	constexpr inline static size_t log2_of_pow2(size_t pow2) noexcept
	{
		return pow2 <= 1 ? 0 : 1 + log2_of_pow2(pow2 / 2);
	}
}

template<typename INT = uint32_t>
class k_out_of_n_bits
{
	constexpr static int INT_bit_count = sizeof(INT) * 8; // e.g. 32
	constexpr static size_t INT_addr_mask = INT_bit_count - 1; // e.g. 31
	constexpr static size_t INT_addr_shift = log2_of_pow2(INT_bit_count); // e.g. 5

	static_assert(INT_bit_count == (1 << INT_addr_shift), "INT must have power of 2 bits"); // e.g. INT can not be a 24 bit integer


	CUDA_ALL
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
	CUDA_ALL
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

	CUDA_ALL
	static void invert_bit(INT* array, unsigned int bit) { array[bit >> INT_addr_shift] ^= INT(1) << (bit & INT_addr_mask); }

	// optimized version of "invert_bit(array, bit); invert_bit(array, bit + 1);"
	CUDA_ALL
	static void invert_2bits(INT* array, unsigned int bit)
	{
		unsigned int item = bit >> INT_addr_shift;
		unsigned int bit_in_item = bit & INT_addr_mask;
		array[item] ^= INT(3) << bit_in_item;
		if (bit_in_item == INT_bit_count - 1)
			array[item + 1] ^= INT(1);
	}

	CUDA_ALL
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


	CUDA_ALL
	void reset()
	{
		index = 0;

		for (int i = 0; i < k; i++)
			counters[i] = i;

		for (int i = 0; i < bit_array_items; i++)
			bit_array[i] = get_bits(i * INT_bit_count);
	}


	CUDA_ALL
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
	CUDA_ALL
	k_out_of_n_bits(k_out_of_n_bits const& other) = delete;


	CUDA_ALL
	~k_out_of_n_bits()
	{
		delete [] counters;
		delete [] bit_array;
	}


	CUDA_ALL
	void print(const char* name, const char* begin = "k_out_of_n_bits ", const char* end = "\n")
	{
		printf("%s%s(%d, %d) %3d/%d: ", begin, name, k, n, index, count);
		for (int i = 0; i < k; i++)
			printf(" %2d", counters[i]);
		for (int i = 0; i < bit_array_items; i++)
			printf(" 0x%08x", bit_array[i]);
		printf("%s", end);
	}

	CUDA_ALL
	INT const& get(int item = 0) const { assert(0 <= item && item < bit_array_items); return bit_array[item]; }

	CUDA_ALL
	INT const& operator*() const { return get(); }

	CUDA_ALL
	INT const& operator[](int i) const { return get(i); }

	CUDA_ALL
	int get_count() const { return count; }

	CUDA_ALL
	int get_index() const { return index; }

	CUDA_ALL
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

	CUDA_ALL
	bool set_index(uint64_t index)
	{
		assert(!numeric_cast_fails<int>(index));
		return index <= INT_MAX && set_index(int(index));
	}

	CUDA_ALL
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

// Replicator mask generator
/*
    //      00000001000000010000000100000001
    // k=0         *       *       *       *   1^4 = 1
    // k=1  *     ***     ***     ***     **   3^4 = 81
    // k=2  **   *****   *****   *****   ***   5^4 = 625
    // k=3  *** ******* ******* ******* ****   7^4 = 2401
    // k=4  ********************************   9^4 = 6561

	Bit 0 of replicator mask should always be 1, because:
	assuming there is a solution where Bit 0 to n are 0 and Bit n+1 is 1,
	then we can construct a solution with Bit 0 set to 1 by
	- shift replicator mask right n times
	- shift selector mask right n times
	- shift shift mask left n times
	Note that this argumentation does not conflict with the quite similar reasoning,
	that the result should always be in the highest bits, because:
	assuming there is a solution where the result is n bits to the right,
	then we can construct a solution with the result at the highest bits by
	- shift shift mask left n times

	It would make sense to limit the distances between set bits.
	Unfortunately this makes the generator more complicated,
	because in the cases where sum of the distances is less than 32,
	a 5th bit set to 1 has to be considered.

	It seems easier to stay with a fixed distance with some variation
	and use a 5th bit at default position 32 (out of the mask)
	which may variate into the mask with negative variation offset.
*/
class generate_replicator
{
	constexpr static int INT_bit_count = sizeof(uint32_t) * 8; // e.g. 32
	constexpr static size_t INT_addr_mask = INT_bit_count - 1; // e.g. 31
	constexpr static size_t INT_addr_shift = log2_of_pow2((size_t)INT_bit_count); // e.g. 5

	CUDA_ALL
	uint32_t get_bits() const
	{
		uint32_t bits = 0;
		for (int i = 0; i < 4; i++)
		{
			int c = counters[i] + 1; // c in [1, k]
			// c   1   2   3   4   5   6   7   8   9
			// di  0   1  -1   2  -2   3  -3   4  -4
			const int di = (c & 1) ? -(c >> 1) : (c >> 1);
			const int bi = (8 * i + di + 32) % 32;
			assert(0 <= bi && bi <= 31);
			bits |= uint32_t(1) << bi;
		}

		return bits;
	}

public:

	const int k;
	const int n;
	const int count;
	int index;
	int counters[4];
	uint32_t bits;

	CUDA_ALL
	void reset()
	{
		index = 0;

		for (int i = 0; i < 4; i++)
			counters[i] = 0;

		bits = get_bits();
	}


	CUDA_ALL
	generate_replicator(int k) :
		k(k),
		n(2 * k + 1),
		count(n * n * n * n),
		index(0)
	{
		assert(k >= 0);
		assert(k < 32);

		reset();
	}


	// For best performance copies shall be avoided.
	CUDA_ALL
	generate_replicator(generate_replicator const& other) = delete;


	CUDA_ALL
	void print(const char* name, const char* begin = "generate_replicator ", const char* end = "\n")
	{
		printf("%s%s(%d) %3d/%d: ", begin, name, k, index, count);
		for (int i = 0; i < 4; i++)
			printf(" %2d", counters[i]);
		printf(" 0x%08x", bits);
		printf("%s", end);
	}

	CUDA_ALL
	uint32_t get() const { assert(get_bits() == bits); return bits; }

	CUDA_ALL
	uint32_t operator*() const { return get(); }

	CUDA_ALL
	uint32_t operator[](int i) const { assert(i == 0); return i == 0 ? get() : 0; }

	CUDA_ALL
	int get_count() const { return count; }

	CUDA_ALL
	int get_index() const { return index; }

	CUDA_ALL
	bool set_index(int index)
	{
		if (index < 0 || index >= count)
			return false;

		this->index = index;

		for (int i = 4 - 1; i >= 0; i--)
		{
			counters[i] = index % n;
			index = index / n;
		}
		assert(index == 0);

		bits = get_bits();

		return true;
	}

	CUDA_ALL
	bool set_index(uint64_t index)
	{
		assert(!numeric_cast_fails<int>(index));
		return index <= INT_MAX && set_index(int(index));
	}

	CUDA_ALL
	bool next()
	{
		// find counter to increment
		int ci = 4 - 1;
		if (ci < 0)
			return false;
		if (counters[ci] >= n - 1)
			do
			{
				if (--ci < 0)
					return false;
			} while (counters[ci] >= n - 1);

		// increment counter
		counters[ci]++;

		// reset all overflown counters
		for (int i = ci + 1; i < 4; i++)
		{
			counters[i] = 0;
		}

		index++;

		bits = get_bits();

		#ifndef NDEBUG
		// self-test
		{
			printf("generate_replicator self-test\n");
			generate_replicator other(k);
			other.set_index(index);
			for (int i = 0; i < 4; i++)
				assert(counters[i] == other.counters[i]);
		}
		#endif

		return true;
	}
};


namespace generators
{
	// A recursive variadic function to increment multiple generators like k_out_of_n_bits

	template <typename TGenerator>
	CUDA_ALL
	inline bool next(TGenerator& generator) 
	{
		return generator.next();
	}

	template<typename TGenerator, typename ... TGenerators>
	CUDA_ALL
	inline bool next(TGenerator& generator, TGenerators& ... generators)
	{
		if (generator.next())
			return true;

		generator.reset();

		return next(generators ...);
	}


	// A recursive variadic function to get the possible number of combinations of multiple generators like k_out_of_n_bits

	template <typename TGenerator>
	CUDA_ALL
	inline uint64_t get_count(TGenerator const& generator) 
	{
		return generator.get_count();
	}

	template<typename TGenerator, typename ... TGenerators>
	CUDA_ALL
	inline uint64_t get_count(TGenerator const& generator, TGenerators const& ... generators)
	{
		uint64_t count = generator.get_count();
		uint64_t counts = get_count(generators ...);
		assert(!multiplication_overflows(count, counts));
		return count * counts;
	}


	// A recursive variadic function to get index of multiple generators like k_out_of_n_bits

	template <typename TGenerator>
	CUDA_ALL
	inline uint64_t get_index(TGenerator const& generator) 
	{
		return generator.get_index();
	}

	template<typename TGenerator, typename ... TGenerators>
	CUDA_ALL
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
	CUDA_ALL
	inline bool set_index(uint64_t index, TGenerator& generator) 
	{
		return index < (uint64_t)generator.get_count() && generator.set_index(index);
	}

	template<typename TGenerator, typename ... TGenerators>
	CUDA_ALL
	inline bool set_index(uint64_t index, TGenerator& generator, TGenerators& ... generators)
	{
		auto count = generator.get_count();
		auto div = index / count;
		auto rem = index % count;
		return set_index(div, generators ...) && generator.set_index(rem);
	}
}
