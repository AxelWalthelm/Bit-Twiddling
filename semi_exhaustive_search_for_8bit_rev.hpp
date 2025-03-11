#pragma once

#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <assert.h>
#include "BitTwiddling.hpp"

#ifndef CUDA_ALL
#define CUDA_ALL
#endif

namespace {

	CUDA_ALL
	inline uint8_t bits_mul_and_mul(uint8_t x, uint32_t replicate, uint32_t select, uint32_t shift)
	{
		return uint8_t((((uint32_t(x) * replicate) & select) * shift) >> 24);
	}

	CUDA_ALL
	bool is_8bit_reverse(uint32_t replicate, uint32_t select, uint32_t shift)
	{
		static constexpr uint8_t rev8_quick_tests[][2] =
		{
			{0xff, 0xff},
			{0x0f, 0xf0}, {0xcc, 0x33}, {0xaa, 0x55},
			{0xf0, 0x0f}, {0x33, 0xcc}, {0x55, 0xaa},
			{0x01, 0x80}, {0x02, 0x40}, {0x04, 0x20}, {0x08, 0x10},
			{0x80, 0x01}, {0x40, 0x02}, {0x20, 0x04}, {0x10, 0x08}
		};

		constexpr uint8_t test_mask = 0xff;

		for (int i = 0; i < int(sizeof(rev8_quick_tests) / sizeof(*rev8_quick_tests)); i++)
		{
			uint8_t value = bits_mul_and_mul(rev8_quick_tests[i][0], replicate, select, shift) & test_mask;
			uint8_t expected = rev8_quick_tests[i][1] & test_mask;
			if (value != expected)
				return false; // quick test failed
		}

		#ifndef __CUDA_ARCH__
		printf("full test: 0x%08x 0x%08x 0x%08x\n", replicate, select, shift);
		#endif
		for (int i = 0; i < 256; i++)
		{
			uint8_t value = bits_mul_and_mul(i, replicate, select, shift) & test_mask;
			uint8_t expected = bits_rev8(i) & test_mask;
			#ifndef __CUDA_ARCH__
			printf("    i=0x%02x value=0x%02x expected=0x%02x\n", i, value, expected);
			#endif
			if (value != expected)
				return false;
		}

		return true;
	}

} // anonymous namespace
