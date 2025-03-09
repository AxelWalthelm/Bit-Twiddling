#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include "HighResolutionTimer.h"
#include "k_out_of_n_bits.hpp"
#include "k_out_of_n_bits_test.h"
#include "semi_exhaustive_search_for_8bit_rev.h"
#include "BitTwiddling.hpp"

namespace {

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
inline uint8_t bits_rev8_old(uint8_t x)
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
	return (bits_rev8(uint8_t(x)) << 8) | bits_rev8(uint8_t(x >> 8));  // 17 ops (4 mult)
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

} // namespace


void main()
{
#if 0
	// test k_out_of_n_bits
	k_out_of_n_bits_test();
	printf("<return>\n");
	getchar();
	return;
#endif

	semi_exhaustive_search_for_8bit_rev();
	printf("<return>\n");
	getchar();
	return;

	printf("\nbits_rev\n");
	{
		for (int i = 0; i < 256; i++)
		{
			int r8 = bits_rev8_old(i);
			int r8n = bits_rev8(i);
			int r16 = bits_rev16(i) >> 8;
			int r32 = bits_rev32(i) >> 24;
			int i2 = bits_rev8_old(r8);
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
			dummy_sum += bits_rev8_old(uint8_t(i));
		}
		stop = GetHighResolutionTime();
		printf("bits_rev8  %.6lf us  [Walthelm  2025 ] sum=%d\n", GetHighResolutionTimeElapsedNs(start, stop) * (0.001 / times), dummy_sum);

		start = GetHighResolutionTime();
		dummy_sum = 0; // something to prevent compiler to optimze test calls away
		for (int i = 0; i < times; i++)
		{
			dummy_sum += bits_rev8(uint8_t(i));
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
