#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include "HighResolutionTimer.h"

namespace {

inline int bits_popcount(uint32_t x)
{
	x = ((x >> 0x01) & 0x55555555u) + (x & 0x55555555u);
	x = ((x >> 0x02) & 0x33333333u) + (x & 0x33333333u);
	x = ((x >> 0x04) & 0x0f0f0f0fu) + (x & 0x0f0f0f0fu);
	x = ((x >> 0x08) & 0x00ff00ffu) + (x & 0x00ff00ffu);
	x = ((x >> 0x10) & 0x0000ffffu) + (x & 0x0000ffffu);
	return x;
}

int bits_popcount_ref(uint32_t x)
{
	int count = 0;
	do
	{
		if (x & 1)
			count++;
	}
	while (x >>= 1);

	return count;
}

inline int bits_ffs(uint32_t x)
{
	if (x == 0)
		return 0;

	int n = 2;
	if ((x & 0x0000FFFFu) == 0) n += 0x10, x >>= 0x10;
	if ((x & 0x000000FFu) == 0) n += 0x08, x >>= 0x08;
	if ((x & 0x0000000Fu) == 0) n += 0x04, x >>= 0x04;
	if ((x & 0x00000003u) == 0) n += 0x02, x >>= 0x02;
	return n - (x & 1);
}

inline int bits_ffs_ref(uint32_t x)
{
	if (x == 0)
		return 0;

	int count = 1;
	while (!(x & 1))
	{
		count++;
		x >>= 1;
	}

	return count;
}

// abcdefgh -> hgfedcb, where a, b, c, d, e, f, g, h in {0, 1}
/*
                                    abcdefgh
            abcdefghabcdefghabcdefghabcdefgh       (!)      FEDCBA9876543210FEDCBA9876543210
1            ...e...                                  <<1   00000000000000000000000000000010
2                              ..f...b                <<19  00000000000010000000000000000000
1                        .g...c.                      <<13  00000000000000000010000000000000
-                                          h...-..    <<31  10000000000000000000000000000000
+                  -...d..                            <<7   00000000000000000000000010000000
            ------------------------------------            10000000000010000010000010000010 = 0x80082082u 
            00001000000100100010010001000001 = 0x08122441u
                e      d  g   c  f   b     h
            ------------------------------------
            hgfedcb
*/
inline uint8_t bits_rev7(uint8_t x)
{
	return uint8_t(((uint32_t(x) * 0x01010101u) & 0x08122441u) * 0x80082082u >> 25); // 4 ops (2 mult)
}


/*
full: [22, 28, 10, 0, 22, 4, 10]

    hgfedcbahgfedcbahgfedcbahgfedcba   zeros l/r: 0/0
                          b......         <<22 00000000010000000000000000000000
                                .c.....   <<28 00010000000000000000000000000000
              ..d....                     <<10 00000000000000000000010000000000
    ...e...                               << 0 00000000000000000000000000000001
                          ....f..         <<22 00000000010000000000000000000000
        .....g.                           << 4 00000000000000000000000000010000
              ......h                     <<10 00000000000000000000010000000000
    --------------------------------
    bcdefgh

f(x) = (x * 0x1010101u & 0x10488224u) * 0x10400411u
f(x) = (x * 0b1000000010000000100000001u & 0b10000010010001000001000100100u) * 0b10000010000000000010000010001u

full: [30, 12, 18, 0, 6, 12, 18]

    hgfedcbahgfedcbahgfedcbahgfedcba   zeros l/r: 0/0
                                  b......   <<30 01000000000000000000000000000000
                .c.....                     <<12 00000000000000000001000000000000
                      ..d....               <<18 00000000000001000000000000000000
    ...e...                                 << 0 00000000000000000000000000000001
          ....f..                           << 6 00000000000000000000000001000000
                .....g.                     <<12 00000000000000000001000000000000
                      ......h               <<18 00000000000001000000000000000000
    --------------------------------
    bcdefgh

f(x) = (x * 0x1010101u & 0x10244882u) * 0x40041041u
f(x) = (x * 0b1000000010000000100000001u & 0b10000001001000100100010000010u) * 0b1000000000001000001000001000001u
*/

inline uint8_t bits_rev8_new(uint8_t x)
{
	//f(x) = (x * 0x1010101u & 0x10488224u) * 0x10400411u
	return uint8_t((((uint32_t(x) * 0x01010101u) & 0x10488224u) * 0x10400411u >> 25) | (x << 7)); // 6 ops (2 mult, 32 bit only)
}



/*
            abcdefghabcdefghabcdefghabcdefgh  ???            FEDCBA9876543210FEDCBA9876543210
             ...e...a                                  <<1   00000000000000000000000000000010
                               ..f...b.                <<19  00000000000010000000000000000000
                         .g...c..                      <<13  00000000000000000010000000000000
                                           h...d...    <<31  10000000000000000000000000000000
                                                             10000000000010000010000000000010
            ------------------------------------             8   0   0   8   2   0   0   2    
            00001000100000100010010001000001  = 0x08822441
                e   a     g   c  f   b     h
            ------------------------------------
            hgfedcba
*/
inline uint8_t bits_rev8(uint8_t x)
{
	//return uint8_t(((uint32_t(x) * 0x00000802u & 0x00022110u) | (uint32_t(x) * 0x00008020u & 0x00088440u)) * 0x00010101u >> 16); // 7 ops (3 mult), Sean Eron Anderson https://graphics.stanford.edu/~seander/bithacks.html
	//return uint8_t(((uint32_t(x) * 0x80200802u) & 0x84422110u) * 0x01010101u >> 24) | ((x >> 1) & 8u); // 7 ops (2 mult), derived from https://graphics.stanford.edu/~seander/bithacks.html#ReverseByteWith64Bits
	//return uint8_t(((uint32_t(x) * 0x01010101u) & 0x08822441u) * 0x80082002u >> 24) | ((x >> 1) & 8u); // 7 ops (2 mult)
	return (bits_rev7(x) << 1) | (x >> 7); // 7 ops (2 mult)
}


// https://graphics.stanford.edu/~seander/bithacks.html#ReverseByteWith32Bits
inline uint8_t bits_rev8_anderson_32bit(uint8_t x)
{
	return uint8_t(((uint32_t(x) * 0x00000802u & 0x00022110u) | (uint32_t(x) * 0x00008020u & 0x00088440u)) * 0x00010101u >> 16); // 7 ops (3 mult)
}

// https://graphics.stanford.edu/~seander/bithacks.html#ReverseByteWith64Bits
inline uint8_t bits_rev8_anderson_64bit(uint8_t x)
{
	return uint8_t(((uint64_t(x) * 0x80200802u) & 0x0884422110u) * 0x0101010101u >> 32); // 4 ops (2 mult)
}


inline uint16_t bits_rev16(uint16_t x)
{
	return (bits_rev8_new(uint8_t(x)) << 8) | bits_rev8_new(uint8_t(x >> 8));  // 17 ops (4 mult)
}


// https://graphics.stanford.edu/~seander/bithacks.html#ReverseParallel
inline uint32_t bits_rev32(uint32_t x)
{
	x = ((x >> 0x01) & 0x55555555u) | ((x & 0x55555555u) << 0x01);
	x = ((x >> 0x02) & 0x33333333u) | ((x & 0x33333333u) << 0x02);
	x = ((x >> 0x04) & 0x0F0F0F0Fu) | ((x & 0x0F0F0F0Fu) << 0x04);
	x = ((x >> 0x08) & 0x00FF00FFu) | ((x & 0x00FF00FFu) << 0x08);
	return (x >> 0x10) | (x << 0x10); // 23 ops
}

// https://graphics.stanford.edu/~seander/bithacks.html#BitReverseTable
static const unsigned char BitReverseTable256[256] = 
{
#   define R2(n)     n,     n + 2*64,     n + 1*64,     n + 3*64
#   define R4(n) R2(n), R2(n + 2*16), R2(n + 1*16), R2(n + 3*16)
#   define R6(n) R4(n), R4(n + 2*4 ), R4(n + 1*4 ), R4(n + 3*4 )
	R6(0), R6(2), R6(1), R6(3)
};

inline uint32_t bits_rev32_mem(uint32_t x)
{
	unsigned int v = x; // reverse 32-bit value, 8 bits at time
	unsigned int c; // c will get v reversed

#if 0
	// Option 1:
	c = (BitReverseTable256[v & 0xff] << 24) |
		(BitReverseTable256[(v >> 8) & 0xff] << 16) |
		(BitReverseTable256[(v >> 16) & 0xff] << 8) |
		(BitReverseTable256[(v >> 24) & 0xff]);
#else
	// Option 2:
	unsigned char * p = (unsigned char *)&v;
	unsigned char * q = (unsigned char *)&c;
	q[3] = BitReverseTable256[p[0]];
	q[2] = BitReverseTable256[p[1]];
	q[1] = BitReverseTable256[p[2]];
	q[0] = BitReverseTable256[p[3]];
#endif
	return c;
}

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
		uint8_t expected = bits_rev8_new(i) & test_mask;
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

void semi_exhaustive_search_for_8bit_rev()
{
	HighResolutionTime_t last_print_time = 0;

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

} // namespace


void main()
{
	// test k_out_of_n_bits
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
	printf("<return>\n");
	getchar();
	return;

	semi_exhaustive_search_for_8bit_rev();
	printf("<return>\n");
	getchar();
	return;

	printf("\nbits_rev\n");
	{
		for (int i = 0; i < 256; i++)
		{
			int r8 = bits_rev8(i);
			int r8n = bits_rev8_new(i);
			int r16 = bits_rev16(i) >> 8;
			int r32 = bits_rev32(i) >> 24;
			int i2 = bits_rev8(r8);
			bool ok = r8 == r8n && r8 == r16 && r16 == r32 && i2 == i;
			if (!ok)
			{
				printf(" 0x%02x -> 0x%02x 0x%02x 0x%02x 0x%02x -> 0x%02x -- ok=%d\n", i, r8, r8n, r16, r32, i2, ok);
				goto end;
			}

			int r7 = bits_rev7(i);
			int r7i = bits_rev7(r7);

			ok = ok && r7 == (r8 >> 1);
			ok = ok && r7i == i %128;
			if (!ok)
			{
				printf(" 0x%02x -> 0x%02x -> 0x%02x -- ok=%d\n", i, r7, r7i, ok);
				goto end;
			}
		}
	}

	printf("\nbits_rev performance\n");
	{
		const int times = 0x10000000;

		HighResolutionTime_t start, stop;
		start = GetHighResolutionTime();
		int dummy_sum = 0; // something to prevent compiler to optimze test calls away
		for (int i = 0; i < times; i++)
		{
			dummy_sum += bits_rev7(uint8_t(i));
		}
		stop = GetHighResolutionTime();
		printf("bits_rev7  %.6lf us  [Walthelm  2025 ] sum=%d\n", GetHighResolutionTimeElapsedNs(start, stop) * (0.001 / times), dummy_sum);

		start = GetHighResolutionTime();
		dummy_sum = 0; // something to prevent compiler to optimze test calls away
		for (int i = 0; i < times; i++)
		{
			dummy_sum += bits_rev8(uint8_t(i));
		}
		stop = GetHighResolutionTime();
		printf("bits_rev8  %.6lf us  [Walthelm  2025 ] sum=%d\n", GetHighResolutionTimeElapsedNs(start, stop) * (0.001 / times), dummy_sum);

		start = GetHighResolutionTime();
		dummy_sum = 0; // something to prevent compiler to optimze test calls away
		for (int i = 0; i < times; i++)
		{
			dummy_sum += bits_rev8_new(uint8_t(i));
		}
		stop = GetHighResolutionTime();
		printf("bits_rev8n %.6lf us  [Walthelm  2025 ] sum=%d\n", GetHighResolutionTimeElapsedNs(start, stop) * (0.001 / times), dummy_sum);

		start = GetHighResolutionTime();
		dummy_sum = 0; // something to prevent compiler to optimze test calls away
		for (int i = 0; i < times; i++)
		{
			dummy_sum += bits_rev8_anderson_32bit(uint8_t(i));
		}
		stop = GetHighResolutionTime();
		printf("bits_rev8  %.6lf us  [Anderson 32 bit] sum=%d\n", GetHighResolutionTimeElapsedNs(start, stop) * (0.001 / times), dummy_sum);

		start = GetHighResolutionTime();
		dummy_sum = 0; // something to prevent compiler to optimze test calls away
		for (int i = 0; i < times; i++)
		{
			dummy_sum += bits_rev8_anderson_64bit(uint8_t(i));
		}
		stop = GetHighResolutionTime();
		printf("bits_rev8  %.6lf us  [Anderson 64 bit] sum=%d\n", GetHighResolutionTimeElapsedNs(start, stop) * (0.001 / times), dummy_sum);

		start = GetHighResolutionTime();
		dummy_sum = 0; // something to prevent compiler to optimze test calls away
		for (int i = 0; i < times; i++)
		{
			dummy_sum += bits_rev16(uint16_t(i));
		}
		stop = GetHighResolutionTime();
		printf("bits_rev16 %.6lf us  [Walthelm  2025 ] sum=%d\n", GetHighResolutionTimeElapsedNs(start, stop) * (0.001 / times), dummy_sum);
		
		start = GetHighResolutionTime();
		dummy_sum = 0; // something to prevent compiler to optimze test calls away
		for (int i = 0; i < times; i++)
		{
			dummy_sum += bits_rev32(uint32_t(i));
		}
		stop = GetHighResolutionTime();
		printf("bits_rev32 %.6lf us  [Dr. Dobb's 1983] sum=%d\n", GetHighResolutionTimeElapsedNs(start, stop) * (0.001 / times), dummy_sum);

		start = GetHighResolutionTime();
		dummy_sum = 0; // something to prevent compiler to optimze test calls away
		for (int i = 0; i < times; i++)
		{
			dummy_sum += bits_rev32_mem(uint32_t(i));
		}
		stop = GetHighResolutionTime();
		printf("bits_rev_m %.6lf us  [lookup table   ] sum=%d\n", GetHighResolutionTimeElapsedNs(start, stop) * (0.001 / times), dummy_sum);

		
		// Intel(R) Celeron(R) CPU  J1900  @ 1.99GHz
		// Microsoft Visual Studio Community 2015

		/* Release x86
			bits_rev performance
			bits_rev7  0.004695 us  [Walthelm  2025 ] sum=-134217728
			bits_rev8  0.006412 us  [Walthelm  2025 ] sum=-134217728
			bits_rev8n 0.005220 us  [Walthelm  2025 ] sum=-134217728
			bits_rev8  0.005515 us  [Anderson 32 bit] sum=-134217728
			bits_rev8  0.030848 us  [Anderson 64 bit] sum=-134217728
			bits_rev16 0.010534 us  [Walthelm  2025 ] sum=-134217728
			bits_rev32 0.017210 us  [Dr. Dobb's 1983] sum=-2147483648
			bits_rev_m 0.019883 us  [lookup table   ] sum=-2147483648
		*/

		/* Release x64
			bits_rev performance
			bits_rev7  0.003496 us  [Walthelm  2025 ] sum=-134217728
			bits_rev8  0.004917 us  [Walthelm  2025 ] sum=-134217728
			bits_rev8n 0.004269 us  [Walthelm  2025 ] sum=-134217728
			bits_rev8  0.005169 us  [Anderson 32 bit] sum=-134217728
			bits_rev8  0.004447 us  [Anderson 64 bit] sum=-134217728
			bits_rev16 0.008890 us  [Walthelm  2025 ] sum=-134217728
			bits_rev32 0.016171 us  [Dr. Dobb's 1983] sum=-2147483648
			bits_rev_m 0.021386 us  [lookup table   ] sum=-2147483648
		*/
	}

	printf("\nbits_ffs\n");
	{
		bool is_first = true;
		for (uint32_t i = 0; is_first || i != 0; i++, is_first = false)
		{
			int c1 = bits_ffs(i);
			int c2 = bits_ffs_ref(i);
			if (c1 != c2)
			{
				printf(" 0x%02x -> %d %d\n", i, c1, c2);
				goto end;
			}
		}
	}

	printf("\nbits_popcount\n");
	{
		bool is_first = true;
		for (uint32_t i = 0; is_first || i != 0; i++, is_first = false)
		{
			int c1 = bits_popcount(i);
			int c2 = bits_popcount_ref(i);
			if (c1 != c2)
			{
				printf(" 0x%02x -> %d %d\n", i, c1, c2);
				goto end;
			}
		}
	}

end:
	printf("<return>\n");
	getchar();
}
