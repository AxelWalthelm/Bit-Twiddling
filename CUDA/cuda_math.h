#pragma once
//
// Copyright 2025 Axel Walthelm
//
// Some math functionality that may be implemented differently for cuda device and host code,
// e.g. to use different hardware for better performance.

#include <stdint.h>

namespace
{
#ifdef __CUDACC__
namespace device
{
	// Use fast PTX assembler command rsqrt to approximate sqrt(x) by value*rsqrt(x).
	// Resulting calculation error is much smaller than 0.5, but it may err in both directions.
	// Instead of testing both directions we can consistently over- or underestimate
	// by adding or subtracting 0.5.

	__device__ inline // using __noinline__ is slow
	uint32_t uint32_sqrt_asm1_over(uint32_t value)
	{
		float value_float = value == 0 ? 1.0f : float(value); // Avoid undefined behavior for value zero.
		float rsqrt_float;
		asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(rsqrt_float) : "f"(value_float));
		uint32_t s = uint32_t(value_float * rsqrt_float + 0.5f); // Always overestimate, never undefined.
		// Result s is never too small.
		// Result s is too big if:
		//     s * s > value  // the left side can overflow numerically if s = 0x10000, in which case the result is 0xffff
		return s == 0x10000 || s * s > value ? s - 1 : s;
	}

	// Avoid check for value == 0 by using more PTX - about 20% faster than uint32_sqrt_asm1_over
	__device__ inline // using __noinline__ is slow
	uint32_t uint32_sqrt_asm2_over(uint32_t value)
	{
		float value_float = float(value);
		float rsqrt_float; // = rsqrt(value_float); // CUDA intrinsic __frsqrt_rn is slower than PTX rsqrt
		asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(rsqrt_float) : "f"(value_float)); // PTX ISA 8.7: Input +0.0 => Result +Inf
		float s_float = value_float * rsqrt_float + 0.5f; // Always overestimate, NaN for value = 0, because IEEE 754: 0 * Inf = NaN
		uint32_t s; // = isnan(s_float) ? 0 : uint32_t(s_float); // Cast of NaN to integer is undefined behavior for C++ and CUDA intrinsic __float2uint_rd.
		asm("cvt.rzi.u32.f32 %0, %1;" : "=r"(s) : "f"(s_float)); // PTX ISA 8.7: "In float-to-integer conversion, NaN inputs are converted to 0."
		// Result s is never too small.
		// Result s is too big if:
		//     s * s > value  // the left side can overflow numerically if s = 0x10000, in which case the result is 0xffff
		return s == 0x10000 || s * s > value ? s - 1 : s;
	}

	__device__ inline
	uint32_t uint32_sqrt_asm1_under(uint32_t value)
	{
		float value_float = value == 0 ? 1.0f : float(value); // Avoid undefined behavior for value zero.
		float rsqrt_float;
		asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(rsqrt_float) : "f"(value_float));
		uint32_t s = uint32_t(value_float * rsqrt_float - 0.5f); // Always underestimate, never undefined.
		// Result s is never too big.
		// Result s is too small if:
		//     (s + 1) * (s + 1) <= value  // but the left side can overflow numerically
		// <=> s * s + 2 * s + 1 <= value
		// <=> value - s * s >= 2 * s + 1  // the left side does not underflow numerically
		// <=> value - s * s > 2 * s
		return value - s * s > 2 * s ? s + 1 : s;
	}

	// Avoid check for value == 0 by using more PTX - about 20% faster than uint32_sqrt_asm1_under
	__device__ inline
	uint32_t uint32_sqrt_asm2_under(uint32_t value)
	{
		float value_float = float(value);
		float rsqrt_float; // = rsqrt(value_float); // CUDA intrinsic __frsqrt_rn is slower than PTX rsqrt
		asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(rsqrt_float) : "f"(value_float)); // PTX ISA 8.7: Input +0.0 => Result +Inf
		float s_float = value_float * rsqrt_float - 0.5f; // Always underestimate, NaN for value = 0, because IEEE 754: 0 * Inf = NaN
		uint32_t s; // = isnan(s_float) ? 0 : uint32_t(s_float); // Cast of NaN to int is undefined behavior for C++ and CUDA intrinsic __float2uint_rd.
		asm("cvt.rzi.u32.f32 %0, %1;" : "=r"(s) : "f"(s_float)); // PTX ISA 8.7: "In float-to-integer conversion, NaN inputs are converted to 0."
		// Result s is never too big.
		// Result s is too small if:
		//     (s + 1) * (s + 1) <= value  // but the left side can overflow numerically
		// <=> s * s + 2 * s + 1 <= value
		// <=> value - s * s >= 2 * s + 1  // the left side does not underflow numerically
		// <=> value - s * s > 2 * s
		return value - s * s > 2 * s ? s + 1 : s;
	}

} // namespace device
#endif // #ifdef __CUDACC__

namespace host
{
	inline
	uint32_t uint32_sqrt_float_and_check(uint32_t value)
	{
		float sf = sqrtf(float(value));
		if (value <= 0xffffff) // Float mantissa has 23 bits and implicitly a leading 1 bit.
			return uint32_t(sf); // No rounding errors: just take exact result.

		uint32_t s = uint32_t(sf - 0.5f); // Always underestimate (calculation error is less than 0.01).
		// Result s is never too big.
		// Result s is too small if:
		//     (s + 1) * (s + 1) <= value  // but the left side can overflow numerically
		// <=> s * s + 2 * s + 1 <= value
		// <=> value - s * s >= 2 * s + 1  // the left side does not underflow numerically
		// <=> value - s * s > 2 * s
		if (value - s * s > 2 * s)
			s++;

		return s;
	}

	inline
	uint32_t uint32_sqrt_float_or_double(uint32_t value)
	{
		return value <= 0xffffff // Float mantissa has 23 bits and implicitly a leading 1 bit.
			? uint32_t(sqrtf(float(value)))
			: uint32_t(sqrt(double(value)));
	}
} // namespace host


#ifdef __CUDA_ARCH__
__device__ inline
uint32_t uint32_sqrt(uint32_t value) { return device::uint32_sqrt_asm2_under(value); }
#else // #ifdef __CUDA_ARCH__
inline
uint32_t uint32_sqrt(uint32_t value) { return host::uint32_sqrt_float_and_check(value); }
#endif // #ifdef __CUDA_ARCH__

#ifdef __CUDACC__
// CUDA extended API contains template function ceil_div,
// but "#include <cuda/cmath>" is not always available and documentation states that
// it is better for 64 bit or unsigned, but not for int.
// For unsigned types "min(value, 1 + ((value - 1) / divisor)" is used.
__device__ __host__ inline
int div_ceil(int a, int b) { return (a + b - 1) / b; }
#else // #ifdef __CUDACC__
inline
int div_ceil(int a, int b) { return (a + b - 1) / b; }
#endif // #ifdef __CUDACC__


// Population count of bits in number.
#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
__device__ inline int bits_popcount(uint32_t x) { return __popc(x); }
#elif defined(__GNUC__)
inline int bits_popcount(uint32_t x) { return __builtin_popcount(x); }
#else
inline int bits_popcount(uint32_t x)
{
	x = ((x >> 0x01) & 0x55555555u) + (x & 0x55555555u);
	x = ((x >> 0x02) & 0x33333333u) + (x & 0x33333333u);
	x = ((x >> 0x04) & 0x0f0f0f0fu) + (x & 0x0f0f0f0fu);
	x = ((x >> 0x08) & 0x00ff00ffu) + (x & 0x00ff00ffu);
	x = ((x >> 0x10) & 0x0000ffffu) + (x & 0x0000ffffu);
	return x;
}
#endif


#if 0 // TODO?
// Count trailing zero bits in number.
// For input value 0, all bits are counted as trailing bits.
#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
__device__ inline int bits_ctz(uint32_t x) { assert(x != 0); return x == 0 ? 32 : __ffs(x) - 1; }
#elif defined(__GNUC__)
inline int bits_ctz(uint32_t x) { return __builtin_ctz(x); }
#else
inline int bits_ctz(uint32_t x)
{
	int n = 1;
	if ((x & 0x0000FFFFu) == 0) n += 0x10, x >>= 0x10;
	if ((x & 0x000000FFu) == 0) n += 0x08, x >>= 0x08;
	if ((x & 0x0000000Fu) == 0) n += 0x04, x >>= 0x04;
	if ((x & 0x00000003u) == 0) n += 0x02, x >>= 0x02;
	return n - (x & 1);
}
#endif
#endif


// Find first set bit in number, counting from least significant bit as bit 1.
#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
__device__ inline int bits_ffs(uint32_t x) { return __ffs(x); }
#elif defined(__GNUC__)
inline int bits_ffs(uint32_t x) { return __builtin_ffs(x); }
#else
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
#endif

#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
__device__ inline uint8_t bits_rev(uint8_t x) { return __brev(uint32_t(x)) >> 24; }
__device__ inline uint16_t bits_rev(uint16_t x) { return __brev(uint32_t(x)) >> 16; }
__device__ inline uint32_t bits_rev(uint32_t x) { return __brev(x); }
#else
// Note: there is currently no __builtin_bitreverse in GCC yet.
inline uint8_t bits_rev(uint8_t x)
{
#if defined(UINT64_MAX) && (defined(__LP64__) || defined(_WIN64))
	return uint8_t(((uint64_t(x) * 0x0000000080200802u) & 0x0000000884422110u) * 0x0000000101010101u >> 32); // 4 ops (64 bit)
#else
	return uint8_t(((uint32_t(x) * 0x00000802u & 0x00022110u) | (uint32_t(x) * 0x00008020u & 0x00088440u)) * 0x00010101u >> 16); // 7 ops (32 bit)
#endif
}
inline uint16_t bits_rev(uint16_t x)
{
	return (bits_rev(uint8_t(x)) << 8) | bits_rev(uint8_t(x >> 8)); // 17 ops
}
inline uint32_t bits_rev(uint32_t x)
{
	x = ((x >> 0x01) & 0x55555555u) | ((x & 0x55555555u) << 0x01);
	x = ((x >> 0x02) & 0x33333333u) | ((x & 0x33333333u) << 0x02);
	x = ((x >> 0x04) & 0x0F0F0F0Fu) | ((x & 0x0F0F0F0Fu) << 0x04);
	x = ((x >> 0x08) & 0x00FF00FFu) | ((x & 0x00FF00FFu) << 0x08);
	return (x >> 0x10) | (x << 0x10); // 23 ops (32 bit)
}
#endif

} // namespace
