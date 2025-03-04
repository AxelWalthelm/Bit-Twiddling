# Bit-Twiddling

Experimenting with C/C++ implementations for bit reversal, find first set bit, population count, etc.

Many recepies for "bit twiddling" were collected by
[Sean Eron Anderson](https://graphics.stanford.edu/~seander/bithacks.html). Appearantly this did became the main online reference for what to do if no compiler intrinsic is available for your current platform.

## Bit Reversal

GCC seems to have no
[intrinsic](https://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html)
for this yet, but may have
[__builtin_bitreverse](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=50481)
for 8, 16, 32, and 64 bit in the future.

Microsoft Visual Studio seems to have no [intrinsic](https://learn.microsoft.com/en-us/cpp/intrinsics/alphabetical-listing-of-intrinsic-functions) for this at all.

CUDA has [intrinsics](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__INT.html) __brev and __brevll for 32 and 64 bit.

[Bit Twiddling Hacks](https://graphics.stanford.edu/~seander/bithacks.html#BitReverseObvious) show a number of options to work around a missing intrinsic. Especially interesting is the idea to use the multiplication hardware to reverse 8 bits with [4 operations, two 64-bit multiply](https://graphics.stanford.edu/~seander/bithacks.html#ReverseByteWith64Bits) or with [7 operations, three 32 bit multiply](https://graphics.stanford.edu/~seander/bithacks.html#ReverseByteWith32Bits).

I did look into the 32 bit case and found: a 32 bit multiplication unit can reverse 7 bits this way, but not 8. But there are several options to do the 7 bit reverse, including one that lead me to a solution with **6 operations, two 32 bit multiply** only, which is faster on my Intel Celeron and will &mdash; in all likelihood &mdash; also perform better on many other hardware platforms:

```
inline uint8_t bits_rev8_new(uint8_t x)
{
	return uint8_t((((uint32_t(x) * 0x01010101u) & 0x10488224u) * 0x10400411u >> 25) | (x << 7));
}
```

Here is an attempt to visualize how the left part of the formula does a 7 bit reversal:

```
    hgfedcbahgfedcbahgfedcbahgfedcba   zeros l/r: 0/0
                          b......         <<22 00000000010000000000000000000000
                                .c.....   <<28 00010000000000000000000000000000
              ..d....                     <<10 00000000000000000000010000000000
    ...e...                               << 0 00000000000000000000000000000001
                          ....f..         <<22 00000000010000000000000000000000
        .....g.                           << 4 00000000000000000000000000010000
              ......h                     <<10 00000000000000000000010000000000
    ---------------------------------------------------------------------------
    00010000010010001000001000100100           00010000010000000000010000010001
    --------------------------------
    bcdefgh

f(x) = (x * 0x1010101u & 0x10488224u) * 0x10400411u
f(x) = (x * 0b1000000010000000100000001u & 0b10000010010001000001000100100u) * 0b10000010000000000010000010001u
```

The result is then shifted to the right by 25 bits into the least significant position.

The right part is just fixing the missing 'a' bit with a shift and a bitwise or.

The cast to uint8_t then removes some extra bits. But this is not counted as an extra operation in the "Bit Twiddling Hacks", so neither do I.
Maybe someone can even find a use for the resulting [palindrome](https://en.wikipedia.org/wiki/Palindrome) bit pattern of length 15?
