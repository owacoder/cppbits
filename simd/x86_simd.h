/*
 * x86_simd.h
 *
 * Copyright © 2018 Oliver Adams
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the “Software”), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef X86_SIMD_H
#define X86_SIMD_H

#include "../environment.h"

#if defined CPPBITS_GCC && (defined CPPBITS_X86 || defined CPPBITS_X86_64) && defined CPPBITS_SSE
#include <x86intrin.h>
#include "generic_simd.h"

template<unsigned int element_bits, typename EffectiveType>
class x86_sse_vector
{
    static constexpr unsigned int vector_digits = 128;
    static constexpr unsigned int alignment = 16;

    static_assert(element_bits, "Number of bits per element must be greater than zero");

public:
    typedef typename impl::unsigned_int<element_bits>::type underlying_element_type;

private:
    static constexpr bool elements_are_floats = std::is_floating_point<EffectiveType>::value;
    static constexpr bool elements_are_signed = std::is_signed<EffectiveType>::value && !elements_are_floats;
    typedef x86_sse_vector<element_bits, EffectiveType> type;
    typedef underlying_element_type value_type;
    static constexpr underlying_element_type ones = -1;
    static constexpr unsigned int elements = vector_digits / element_bits;
    static constexpr underlying_element_type element_mask = ones >> (std::numeric_limits<underlying_element_type>::digits - element_bits);

    union alignas(alignment) packed_union
    {
        packed_union() : vector{} {}
        packed_union(__m128i v) : vector(v) {}
        packed_union(__m128 v) : vector((__m128i) v) {}
        packed_union(__m128d v) : vector((__m128i) v) {}

        underlying_element_type store[elements];
        __m128i vector;
    };

    static constexpr bool effective_type_is_exact_size = (sizeof(EffectiveType) * CHAR_BIT == element_bits);
    static_assert(!elements_are_floats || (element_bits == 32 || element_bits == 64), "Floating-point element size must be 32 or 64 bits");
    static_assert(element_bits <= 64, "Element size is too large for specified type");
    static_assert(elements_are_floats || (sizeof(EffectiveType) * CHAR_BIT) >= element_bits, "Element size is too large for specified effective type `EffectiveType`");

    constexpr explicit x86_sse_vector(__m128i value) : data_(value) {}
    constexpr explicit x86_sse_vector(__m128 value) : data_(value) {}
    constexpr explicit x86_sse_vector(__m128d value) : data_(value) {}

    constexpr static underlying_element_type make_t_from_effective(EffectiveType v)
    {
        return elements_are_floats? element_bits == 32? underlying_element_type(float_cast_to_ieee_754(v)): underlying_element_type(double_cast_to_ieee_754(v)): underlying_element_type(v);
    }
    static EffectiveType make_effective_from_t(underlying_element_type v)
    {
        if (elements_are_floats)
            return element_bits == 32? EffectiveType(float_cast_from_ieee_754(v)):
                                       EffectiveType(double_cast_from_ieee_754(v));
        else if (elements_are_signed)
        {
            const underlying_element_type val = v & (element_mask >> 1);
            const underlying_element_type negative = v >> (element_bits - 1);
            return negative? val == 0? scalar_min(): -EffectiveType((~val + 1) & (element_mask >> 1)): EffectiveType(val);
        }
        return EffectiveType(v);
    }

public:
    enum broadcast_type
    {
        broadcast_none, /* Initialize entire vector with provided value */
        broadcast_scalar, /* Single scalar value in element position 0 */
        broadcast_all /* Broadcast value to all positions */
    };

    enum math_type
    {
        /* Integral math modes (When used on a floating-point vector, math_accurate is used instead) */
        math_saturate, /* Saturating arithmetic */
        math_keephigh, /* Keep high part of result */
        math_keeplow, /* Rollover arithmetic, keep low part of result */

        /* Floating-point math modes (When used on an integral vector, math_keeplow is used instead) */
        math_accurate = math_keeplow, /* As accurate a result as possible */
        math_approximate /* An approximate result is okay, if it is available and speeds things up */
    };

    enum shift_type
    {
        shift_natural, /* Either logical or arithmetic, depending on the effective element type */
        shift_logical, /* Logical shift shifts in zeros */
        shift_arithmetic /* Arithmetic shift copies the sign bit in from the left, zeros from the right */
    };

    enum compare_type
    {
        compare_less, /* Compare `a < b` */
        compare_lessequal, /* Compare `a <= b` */
        compare_greater, /* Compare `a > b` */
        compare_greaterequal, /* Compare `a >= b` */
        compare_equal, /* Compare `a == b` */
        compare_nequal /* Compare `a != b` */
    };

    /*
     * Default constructor zeros the vector
     */
    constexpr x86_sse_vector() : data_{} {}

    /*
     * Returns a new representation of the vector, casted to a different type.
     * The data is not modified.
     */
    template<typename NewType>
    constexpr native_simd_vector<underlying_element_type, element_bits, NewType> cast() const {return native_simd_vector<underlying_element_type, element_bits, NewType>::make_vector(data_.vector);}

    /*
     * Returns a new representation of the vector, casted to a different type with different element size.
     * The data is not modified.
     */
    template<unsigned int element_size, typename NewType>
    native_simd_vector<underlying_element_type, element_size, NewType> cast() const {return native_simd_vector<underlying_element_type, element_size, NewType>::make_vector(data_.vector);}

    /*
     * Returns a new representation of the vector, casted to a different type with different element size, with a different underlying type.
     * The data is not modified.
     */
    template<typename UnderlyingType, unsigned int element_size, typename NewType>
    native_simd_vector<UnderlyingType, element_size, NewType> cast() const {return native_simd_vector<UnderlyingType, element_size, NewType>::make_vector(data_.vector);}

    /*
     * Returns a representation of a vector with specified value assigned to the entire vector
     */

    constexpr static type make_vector(__m128i value) {return type(value);}

    /*
     * Returns a representation of a vector with specified value assigned to element 0
     * TODO: support 8, 16, 64 bit element sizes
     */
    static type make_scalar(EffectiveType value)
    {
        underlying_element_type temp = make_t_from_effective(value);
        switch (element_bits)
        {
            default: /* 8 */
            case 16:
            case 32:
            {
                alignas(alignment) union
                {
                    uint32_t i;
                    float f;
                } cvt;
                cvt.i = temp;
                return type(_mm_load_ss(&cvt.f));
            }
            case 64:
#ifdef CPPBITS_SSE2
            {
                alignas(alignment) union
                {
                    uint64_t i;
                    double f;
                } cvt;
                cvt.i = temp;
                return type(_mm_load_sd(&cvt.f));
            }
#else
            {
                alignas(alignment) union
                {
                    uint64_t i;
                    __m64 v;
                } cvt;
                cvt.i = temp;
                return type(_mm_loadl_pi(_mm_setzero_ps(), &cvt.v));
            }
#endif
        };
    }

    /*
     * Returns a representation of a vector with specified value assigned to every element in the vector
     */
    constexpr static type make_broadcast(EffectiveType value) {return type(expand_value(value));}

    /*
     * Sets this vector to a representation of a vector with specified value assigned to the entire vector
     */
    type &vector(__m128i value) {return *this = make_vector(value);}

    /*
     * Sets this vector to a representation of a vector with specified value assigned to element 0
     * All elements other than element 0 are zeroed.
     */
    type &scalar(EffectiveType value) {return *this = make_scalar(value);}

    /*
     * Sets this vector to a representation of a vector with specified value assigned to every element
     */
    type &broadcast(EffectiveType value) {return *this = make_broadcast(value);}

    /*
     * Returns the internal representation of the entire vector
     */
    constexpr __m128i vector() const {return data_.vector;}

    /*
     * Returns the value of element 0
     */
    EffectiveType scalar() const {return get<0>();}

    /*
     * Logical NOT's the entire vector and returns it
     */
    constexpr type operator~() const
    {
        return type(_mm_xor_ps((__m128) data_.vector, (__m128) vector_mask()));
    }

    /*
     * Logical OR's the entire vector with `vec` and returns it
     */
    constexpr type operator|(x86_sse_vector vec) const
    {
        return type(_mm_or_ps((__m128) data_.vector, (__m128) vec.vector()));
    }

    /*
     * Logical AND's the entire vector with `vec` and returns it
     */
    constexpr type operator&(x86_sse_vector vec) const
    {
        return type(_mm_and_ps((__m128) data_.vector, (__m128) vec.vector()));
    }

    /*
     * Logical AND's the entire vector with negated `vec` and returns it
     */
    constexpr type and_not(x86_sse_vector vec) const
    {
        return type(_mm_andnot_ps((__m128) data_.vector, (__m128) vec.vector()));
    }

    /*
     * Logical XOR's the entire vector with `vec` and returns it
     */
    constexpr type operator^(x86_sse_vector vec) const
    {
        return type(_mm_xor_ps((__m128) data_.vector, (__m128) vec.vector()));
    }

    /*
     * Adds all elements of `this` and returns the resulting sum
     */
    constexpr EffectiveType horizontal_sum() const
    {
        CPPBITS_ERROR("Horizontal sum not implemented for x86 SIMD vectors yet");
    }

    /*
     * Adds elements of `vec` to `this` (using rollover addition) and returns the result
     */
    type operator+(x86_sse_vector vec) const {return add(vec, math_keeplow);}

    /*
     * Adds elements of `vec` to `this` (using specified math method) and returns the result
     */
    type add(x86_sse_vector vec, math_type math) const
    {
        if (elements_are_floats)
        {
            if (element_bits == 32)
                return type(_mm_add_ps((__m128) data_.vector, (__m128) vec.vector()));
            else
            {
#ifdef CPPBITS_SSE2
                return type(_mm_add_pd((__m128d) data_.vector, (__m128d) vec.vector()));
#else
                return make_init_values(vec, [](EffectiveType a, EffectiveType b){return a + b;});
#endif
            }
        }
        else
        {
            switch (math) {
                default: /* Rollover arithmetic, math_keeplow */
#ifdef CPPBITS_SSE2
                    switch (element_bits) {
                        default: return type(_mm_add_epi8(data_.vector, vec.vector()));
                        case 16: return type(_mm_add_epi16(data_.vector, vec.vector()));
                        case 32: return type(_mm_add_epi32(data_.vector, vec.vector()));
                        case 64: return type(_mm_add_epi64(data_.vector, vec.vector()));
                    }
#else
                    return make_init_values(vec, [](EffectiveType a, EffectiveType b){return a + b;});
#endif
                case math_saturate:
                    if (elements_are_signed)
                    {
#ifdef CPPBITS_SSE2
                        switch (element_bits) {
                            case  8: return type(_mm_adds_epi8(data_.vector, vec.vector()));
                            case 16: return type(_mm_adds_epi16(data_.vector, vec.vector()));

                        }
#else
                        return make_init_raw_values(vec, [](underlying_element_type a, underlying_element_type b)
                        {
                            underlying_element_type temp = a + b;
                            if (temp < a)
                                temp = (element_mask >> 1) + (a >> (element_bits - 1));
                        });
#endif
                    }
                    else
                    {
#ifdef CPPBITS_SSE2
                        switch (element_bits) {
                            case  8: return type(_mm_adds_epu8(data_.vector, vec.vector()));
                            case 16: return type(_mm_adds_epu16(data_.vector, vec.vector()));
                            default:
                                return make_init_raw_values(vec, [](underlying_element_type a, underlying_element_type b)
                                {
                                    underlying_element_type temp = a + b;
                                    return temp < a? element_mask: temp;
                                });
                        }
#else
                        return make_init_raw_values(vec, [](underlying_element_type a, underlying_element_type b)
                        {
                            underlying_element_type temp = a + b;
                            return temp < a? element_mask: temp;
                        });
#endif
                    }
                case math_keephigh: /* TODO: not yet verified as there is no builtin to keep the high part of an addition */
                {
                    CPPBITS_ERROR("Addition using math_keephigh not implemented for x86 SIMD vectors yet");
                }
            }
        }
    }

    /*
     * Subtracts elements of `vec` to `this` (using rollover subtraction) and returns the result
     */
    type operator-(x86_sse_vector vec) const {return sub(vec, math_keeplow);}

    /*
     * Subtracts elements of `vec` from `this` (using specified math method) and returns the result
     */
    type sub(x86_sse_vector vec, math_type math) const
    {
        if (elements_are_floats)
        {
            if (element_bits == 32)
                return type(_mm_sub_ps((__m128) data_.vector, (__m128) vec.vector()));
            else
            {
#ifdef CPPBITS_SSE2
                return type(_mm_sub_pd((__m128d) data_.vector, (__m128d) vec.vector()));
#else
                return make_init_values(vec, [](EffectiveType a, EffectiveType b){return a - b;});
#endif
            }
        }
        else
        {
            switch (math) {
                default: /* Rollover arithmetic, math_keeplow */
#ifdef CPPBITS_SSE2
                    switch (element_bits) {
                        default: return type(_mm_sub_epi8(data_.vector, vec.vector()));
                        case 16: return type(_mm_sub_epi16(data_.vector, vec.vector()));
                        case 32: return type(_mm_sub_epi32(data_.vector, vec.vector()));
                        case 64: return type(_mm_sub_epi64(data_.vector, vec.vector()));
                    }
#else
                    return make_init_values(vec, [](EffectiveType a, EffectiveType b){return a - b;});
#endif
                case math_saturate:
                    if (elements_are_signed)
                    {
                        CPPBITS_ERROR("Signed saturation not implemented for x86 SIMD vectors yet");
                    }
                    else
                    {
                        CPPBITS_ERROR("Unsigned saturation not implemented for x86 SIMD vectors yet");
                    }
                case math_keephigh: /* TODO: not yet verified as there is no builtin to keep the high part of an addition */
                {
                    CPPBITS_ERROR("Subtraction using math_keephigh not implemented for x86 SIMD vectors yet");
                }
            }
        }
    }

    /*
     * Multiplies elements of `vec` by `this` (using rollover multiplication) and returns the result
     */
    type operator*(x86_sse_vector vec) const {return mul(vec, math_keeplow);}

    /*
     * Multiplies elements of `vec` by `this` (using specified math method) and returns the result
     * TODO: implement integral calculations
     */
    type mul(x86_sse_vector vec, math_type math) const
    {
        if (elements_are_floats)
        {
            if (element_bits == 32)
                return type(_mm_mul_ps((__m128) data_.vector, (__m128) vec.vector()));
            else
            {
#ifdef CPPBITS_SSE2
                return type(_mm_mul_pd((__m128d) data_.vector, (__m128d) vec.vector()));
#else
                return make_init_values(vec, [](EffectiveType a, EffectiveType b){return a * b;});
#endif
            }
        }
        else
        {
            switch (math) {
                default: /* Rollover arithmetic, math_keeplow */
                    switch (element_bits) {
                        /* TODO: 8-bit multiplication is slower than naive, but much smaller code size when naive is unrolled. Is there a way to speed this up? */
#ifdef CPPBITS_SSE2
                        default:
                        {
                            __m128i lo = _mm_unpacklo_epi8(data_.vector, _mm_setzero_si128());
                            __m128i other_lo = _mm_unpacklo_epi8(vec.vector(), _mm_setzero_si128());
                            __m128i hi = _mm_unpackhi_epi8(data_.vector, _mm_setzero_si128());
                            __m128i other_hi = _mm_unpackhi_epi8(vec.vector(), _mm_setzero_si128());
                            __m128i mask = expand_mask(element_mask, element_bits * 2);

                            return type(_mm_packus_epi16(_mm_and_si128(_mm_mullo_epi16(lo, other_lo), mask),
                                                         _mm_and_si128(_mm_mullo_epi16(hi, other_hi), mask)));
                        }
#else
                        default: return make_init_values(vec, [](EffectiveType a, EffectiveType b){return a * b;});
#endif
#ifdef CPPBITS_SSE2
                        case 16: return type(_mm_mullo_epi16(data_.vector, vec.vector()));
#else
                        case 16: return make_init_values(vec, [](EffectiveType a, EffectiveType b){return a * b;});
#endif
#ifdef CPPBITS_SSE4_1
                        case 32: return type(_mm_mullo_epi32(data_.vector, vec.vector()));
#elif defined CPPBITS_SSE2
                        case 32:
                        {
                            __m128i temp1, temp2;

                            temp1 = _mm_mul_epu32(data_.vector, vec.vector());
                            // 0x31 = 0b00110001, map element 0 to elements 1 and 3 (unused), element 1 to element 0, and element 3 to element 2
                            temp2 = _mm_mul_epu32(_mm_shuffle_epi32(data_.vector, 0x31),
                                                  _mm_shuffle_epi32(vec.vector(), 0x31));
                            // Now temp1 contains elements 0 and 2 in the correct positions (with garbage in elements 1 and 3)
                            // and temp2 contains element 1 in position 0 and element 3 in position 2 (with garbage in elements 1 and 3)
                            return type(_mm_movelh_ps((__m128) _mm_unpacklo_epi32(temp1, temp2),
                                                      (__m128) _mm_unpackhi_epi32(temp1, temp2)));
                        }
#else
                        case 32: return make_init_values(vec, [](EffectiveType a, EffectiveType b){return a * b;});
#endif
                        case 64: return make_init_values(vec, [](EffectiveType a, EffectiveType b){return a * b;});
                    }
                case math_saturate:
                    if (elements_are_signed)
                    {
                        CPPBITS_ERROR("Signed saturation not implemented for x86 SIMD vectors yet");
                    }
                    else
                    {
                        CPPBITS_ERROR("Unsigned saturation not implemented for x86 SIMD vectors yet");
                    }
                case math_keephigh: /* TODO: not yet verified as there is no builtin to keep the high part of an addition */
                {
                    CPPBITS_ERROR("Multiplication using math_keephigh not implemented for x86 SIMD vectors yet");
                }
            }
        }
    }

    /*
     * Multiplies elements of `vec` by `this` (using specified math method), adds `add`, and returns the result
     */
    constexpr type mul_add(x86_sse_vector vec, x86_sse_vector add, math_type math) const
    {
        return mul(vec, math).add(add, math);
    }

    /*
     * Multiplies elements of `vec` by `this` (using specified math method), subtracts `sub`, and returns the result
     */
    constexpr type mul_sub(x86_sse_vector vec, x86_sse_vector sub, math_type math) const
    {
        return mul(vec, math).sub(sub, math);
    }

    /*
     * Divides elements of `this` by `vec` (using rollover division) and returns the result
     */
    type operator/(x86_sse_vector vec) const {return div(vec, math_keeplow);}

    /*
     * Divides elements of `this` by `vec` (using specified math method) and returns the result
     * TODO: implement integral calculations
     */
    type div(x86_sse_vector vec, math_type math) const
    {
        if (elements_are_floats)
        {
            if (element_bits == 32)
            {
                switch (math) {
                    default: return type(_mm_div_ps((__m128) data_.vector, (__m128) vec.vector()));
                    case math_approximate: return *this * vec.reciprocal(math);
                }
            }
            else
            {
#ifdef CPPBITS_SSE2
                return type(_mm_div_pd((__m128d) data_.vector, (__m128d) vec.vector()));
#else
                return make_init_values(vec, [](EffectiveType a, EffectiveType b){return a / b;});
#endif
            }
        }
        else if (vec.has_zero_element())
            CPPBITS_ERROR("Division by zero");
        else
        {
            switch (math) {
                default: /* Rollover arithmetic, math_keeplow */
                    return make_init_values(vec, [](EffectiveType a, EffectiveType b){return a / b;});
                case math_saturate:
                    if (elements_are_signed)
                    {
                        CPPBITS_ERROR("Signed saturation not implemented for x86 SIMD vectors yet");
                    }
                    else
                    {
                        CPPBITS_ERROR("Unsigned saturation not implemented for x86 SIMD vectors yet");
                    }
                case math_keephigh:
                {
                    CPPBITS_ERROR("Division using math_keephigh not implemented for x86 SIMD vectors yet");
                }
            }
        }
    }

    /*
     * Computes the average of each element of `this` and `vec` as `((element of this) + (element of vec) + 1)/2` for integral values,
     * or `((element of this) + (element of vec))/2` for floating-point values
     */
    type avg(x86_sse_vector) const
    {
        CPPBITS_ERROR("Element average not implemented for x86 SIMD vectors yet");
    }

    /*
     * Shifts each element of `this` to the left by `amount` and returns the resulting vector
     * If the invariant `0 <= amount <= element_bits` does not hold, undefined behavior results
     */
    constexpr type operator<<(unsigned int amount) const {return shl(amount);}

    /*
     * Shifts each element of `this` to the left by `amount` and returns the resulting vector
     * If the invariant `0 <= amount <= element_bits` does not hold, undefined behavior results
     * TODO: support floating-point values
     */
    constexpr type shl(unsigned int amount) const
    {
        switch (element_bits)
        {
            default: /* 8 */ return make_init_raw_value([amount](underlying_element_type a) {return a << amount;});
#ifdef CPPBITS_SSE2
            case 16: return type(_mm_slli_epi16(data_.vector, amount));
#else
            case 16: return make_init_raw_value([amount](underlying_element_type a) {return a << amount;});
#endif
#ifdef CPPBITS_SSE2
            case 32: return type(_mm_slli_epi32(data_.vector, amount));
#else
            case 32: return make_init_raw_value([amount](underlying_element_type a) {return a << amount;});
#endif
#ifdef CPPBITS_SSE2
            case 64: return type(_mm_slli_epi64(data_.vector, amount));
#else
            case 64: return make_init_raw_value([amount](underlying_element_type a) {return a << amount;});
#endif
        }
    }

    /*
     * Shifts each element of `this` to the right by `amount` (using shift_natural behavior) and returns the resulting vector
     * If the invariant `0 <= amount <= element_bits` does not hold, undefined behavior results
     */
    constexpr type operator>>(unsigned int amount) const {return shr(amount, shift_natural);}

    /*
     * Shifts each element of `this` to the right by `amount` (using specified shift type) and returns the resulting vector
     * If the invariant `0 <= amount <= element_bits` does not hold, undefined behavior results
     * TODO: support floating-point values
     */
    type shr(unsigned int amount, shift_type shift) const
    {
        switch (shift) {
            default: /* shift_natural */
                return elements_are_signed? shr(amount, shift_arithmetic): shr(amount, shift_logical);
            case shift_logical:
                switch (element_bits) {
                    default: /* 8 */ return make_init_raw_value([amount](underlying_element_type a) {return a >> amount;});
#ifdef CPPBITS_SSE2
                    case 16: return type(_mm_srli_epi16(data_.vector, amount));
#else
                    case 16: return make_init_raw_value([amount](underlying_element_type a) {return a >> amount;});
#endif
#ifdef CPPBITS_SSE2
                    case 32: return type(_mm_srli_epi32(data_.vector, amount));
#else
                    case 32: return make_init_raw_value([amount](underlying_element_type a) {return a >> amount;});
#endif
#ifdef CPPBITS_SSE2
                    case 64: return type(_mm_srli_epi64(data_.vector, amount));
#else
                    case 64: return make_init_raw_value([amount](underlying_element_type a) {return a >> amount;});
#endif
                }
            case shift_arithmetic:
                switch (element_bits) {
                    default: /* 8 */ return make_init_raw_value([amount](EffectiveType a) {return a >> amount;});
#ifdef CPPBITS_SSE2
                    case 16: return type(_mm_srai_epi16(data_.vector, amount));
#else
                    case 16: return make_init_raw_value([amount](EffectiveType a) {return a >> amount;});
#endif
#ifdef CPPBITS_SSE2
                    case 32: return type(_mm_srai_epi32(data_.vector, amount));
#else
                    case 32: return make_init_raw_value([amount](EffectiveType a) {return a >> amount;});
#endif
                    case 64: return make_init_raw_value([amount](EffectiveType a) {return a >> amount;});
                }
        }
    }

    /*
     * Extracts MSB from each element and places them in the low bits of the result
     * Each bit position in the result corresponds to the element position in the source vector
     * (i.e. Element 0 MSB -> Bit 0, Element 1 MSB -> Bit 1, etc.)
     */
    unsigned int movmsk() const
    {
        switch (element_bits) {
            default: /* 8 */
            {
#ifdef CPPBITS_SSE2
                return _mm_movemask_epi8(data_.vector);
#else
                packed_union pack;
                unsigned result = 0;

                pack.vector = data_;
                for (unsigned i = 0; i < elements; ++i)
                    result |= (pack.store[i] >> (element_bits - 1)) << i;
                return result;
#endif
            }
            case 16:
            {
#ifndef CPPBITS_SSE2
                const unsigned int map[16] = {/* no bits set */ 0,
                                              /* bit 0 is set, bit 2 of result */ 4,
                                              /* bit 1 is set, bit 0 of result */ 1,
                                              /* bits 0,1 set, bits 0,2 of result */ 5,
                                              /* bit 2 set, bit 3 of result */ 8,
                                              /* bits 0,2 set, bits 2,3 of result */ 12,
                                              /* bits 1,2 set, bits 0,3 of result */ 9,
                                              /* bits 0,1,2 set, bits 0,2,3 of result */ 13,
                                              /* bit 3 set, bit 1 of result */ 2,
                                              /* bits 0,3 set, bits 1,2 of result */ 6,
                                              /* bits 1,3 set, bits 0,1 of result */ 3,
                                              /* bits 0,1,3 set, bits 0,1,2 of result */ 7,
                                              /* bits 2,3 set, bits 1,3 of result */ 10,
                                              /* bits 0,2,3 set, bits 1,2,3 of result */ 14,
                                              /* bits 1,2,3 set, bits 0,1,3 of result */ 11,
                                              /* bits 0,1,2,3 set, bits 0,1,2,3 of result */ 15
                                             };
                int mask = _mm_movemask_epi8(data_.vector) & 0xaa; // 0xAA = 0b10101010
                mask |= (mask >> 5);
                // Bit 0 of mask contains bit 2 of result
                // Bit 1 of mask contains bit 0 of result
                // Bit 2 of mask contains bit 3 of result
                // Bit 3 of mask contains bit 1 of result
                // That's why the lookup table is used
                return map[mask & 0xf];
#else
                unsigned result = 0;

                for (unsigned i = 0; i < elements; ++i)
                    result |= (data_.store[i] >> (element_bits - 1)) << i;
                return result;
#endif
            }
            case 32: return _mm_movemask_ps((__m128) data_.vector);
            case 64:
            {
#ifdef CPPBITS_SSE2
                return _mm_movemask_pd((__m128d) data_.vector);
#else
                int n = _mm_movemask_ps((__m128) data_.vector) & 0xa; // Mask with 0b1010
                return ((n >> 1) | (n >> 2)) & 0x3; // Move bits to the lowest two bits
#endif
            }
        }
    }

    /*
     * Negates elements of `this` and returns the resulting vector
     * Note that this function returns the value unmodified if EffectiveType is not a signed type
     * TODO: are floating-point values supported properly?
     * TODO: implement integral calculations
     */
    type negate() const
    {
        CPPBITS_ERROR("Negation not implemented for x86 SIMD vectors yet");
    }

    /*
     * Computes the absolute value of elements of `this` and returns the resulting vector
     * Note that this function returns the value unmodified if EffectiveType is not a signed type
     * TODO: are floating-point values supported properly?
     * TODO: implement integral calculations
     */
    type abs() const
    {
        CPPBITS_ERROR("Absolute-value not implemented for x86 SIMD vectors yet");
    }

    /*
     * Sets each element of `this` to zero if the element is zero, or all ones otherwise, and returns the resulting vector
     * TODO: need to implement
     */
    type fill_if_nonzero() const
    {
#ifdef CPPBITS_SSE2
        return ~type(_mm_cmpeq_epi32(data_.vector, _mm_setzero_si128()));
#else
        return make_init_raw_value([](underlying_element_type a) {return (a != 0) * element_mask;});
#endif
    }

    /*
     * Returns true if vector has at least one element equal to zero (sign of zero irrelevant in the case of floating-point), false otherwise
     */
    constexpr bool has_zero_element() const
    {
        return count_zero_elements() != 0;
    }

    /*
     * Returns number of zero elements in vector (sign of zero irrelevant in the case of floating-point)
     */
    unsigned int count_zero_elements() const
    {
        if (elements_are_floats)
        {
            if (element_bits == 32)
            {
                __m128 cmp = _mm_cmpeq_ps((__m128) data_.vector, _mm_setzero_ps());
                return __builtin_popcount(_mm_movemask_ps((__m128) cmp));
            }
            else
            {
#ifdef CPPBITS_SSE2
                __m128d cmp = _mm_cmpeq_pd((__m128d) data_.vector, _mm_setzero_pd());
                return __builtin_popcount(_mm_movemask_pd((__m128d) cmp));
#else
                unsigned result = 0;

                for (unsigned i = 0; i < elements; ++i)
                    result += make_effective_from_t(data_.store[i]) == 0;
                return result;
#endif
            }
        }
        else
        {
#ifdef CPPBITS_SSE2
            switch (element_bits) {
                default: /* 8 */ return __builtin_popcount(type(_mm_cmpeq_epi8(data_.vector, _mm_setzero_si128())).movmsk());
                case 16: return __builtin_popcount(type(_mm_cmpeq_epi16(data_.vector, _mm_setzero_si128())).movmsk());
                case 32: return __builtin_popcount(type(_mm_cmpeq_epi32(data_.vector, _mm_setzero_si128())).movmsk());
                case 64:
                {
                    __m128i temp = _mm_cmpeq_epi32(data_.vector, _mm_setzero_si128());
                    return __builtin_popcount(_mm_movemask_pd((__m128d) _mm_and_si128(temp, _mm_slli_epi64(temp, 32))));
                }
            }
#else
            unsigned result = 0;

            for (unsigned i = 0; i < elements; ++i)
                result += make_effective_from_t(data_.store[i]) == 0;
            return result;
#endif
        }
    }

    /*
     * Returns true if vector has at least one element equal to `v`, false otherwise
     */
    constexpr bool has_equal_element(EffectiveType v) const
    {
        return count_equal_elements(v) != 0;
    }

    /*
     * Returns number of elements equal to v in vector
     */
    unsigned int count_equal_elements(EffectiveType v) const
    {
        if (elements_are_floats)
        {
            if (element_bits == 32)
            {
                __m128 cmp = _mm_cmpeq_ps((__m128) data_.vector, (__m128) expand_mask(make_t_from_effective(v) & element_mask, element_bits));
                return __builtin_popcount(_mm_movemask_ps((__m128) cmp));
            }
            else
            {
#ifdef CPPBITS_SSE2
                __m128d cmp = _mm_cmpeq_pd((__m128d) data_.vector, (__m128d) expand_mask(make_t_from_effective(v) & element_mask, element_bits));
                return __builtin_popcount(_mm_movemask_pd((__m128d) cmp));
#else
                packed_union pack;
                unsigned result = 0;

                pack.vector = data_;
                for (unsigned i = 0; i < elements; ++i)
                    result += make_effective_from_t(pack.store[i]) == v;
                return result;
#endif
            }
        }
        else
        {
#ifdef CPPBITS_SSE2
            switch (element_bits) {
                default: /* 8 */ return __builtin_popcount(type(_mm_cmpeq_epi8(data_.vector, expand_mask(make_t_from_effective(v) & element_mask, element_bits))).movmsk());
                case 16: return __builtin_popcount(type(_mm_cmpeq_epi16(data_.vector, expand_mask(make_t_from_effective(v) & element_mask, element_bits))).movmsk());
                case 32: return __builtin_popcount(type(_mm_cmpeq_epi32(data_.vector, expand_mask(make_t_from_effective(v) & element_mask, element_bits))).movmsk());
                case 64:
                {
                    __m128i temp = _mm_cmpeq_epi32(data_.vector, expand_mask(make_t_from_effective(v) & element_mask, element_bits));
                    return __builtin_popcount(_mm_movemask_pd((__m128d) _mm_and_si128(temp, _mm_slli_epi64(temp, 32))));
                }
            }
#else
            packed_union pack;
            unsigned result = 0;

            pack.vector = data_;
            for (unsigned i = 0; i < elements; ++i)
                result += make_effective_from_t(pack.store[i]) == v;
            return result;
#endif
        }
    }

    /*
     * Compares `this` to `vec`. If the comparison result is true, the corresponding element is set to all 1's
     * Otherwise, if the comparison result is false, the corresponding element is set to all 0's
     */
    type cmp(x86_sse_vector vec, compare_type compare) const
    {
        if (elements_are_floats)
        {
            if (element_bits == 32)
            {
                switch (compare) {
                    default: return type(_mm_cmpeq_ps((__m128) data_.vector, (__m128) vec.vector()));
                    case compare_nequal: return type(_mm_cmpneq_ps((__m128) data_.vector, (__m128) vec.vector()));
                    case compare_less: return type(_mm_cmplt_ps((__m128) data_.vector, (__m128) vec.vector()));
                    case compare_lessequal: return type(_mm_cmple_ps((__m128) data_.vector, (__m128) vec.vector()));
                    case compare_greater: return type(_mm_cmple_ps((__m128) vec.vector(), (__m128) data_.vector));
                    case compare_greaterequal: return type(_mm_cmplt_ps((__m128) vec.vector(), (__m128) data_.vector));
                }
            }
            else // element_bits == 64
            {
#ifdef CPPBITS_SSE2
                switch (compare) {
                    default: return type(_mm_cmpeq_pd((__m128d) data_.vector, (__m128d) vec.vector()));
                    case compare_nequal: return type(_mm_cmpneq_pd((__m128d) data_.vector, (__m128d) vec.vector()));
                    case compare_less: return type(_mm_cmplt_pd((__m128d) data_.vector, (__m128d) vec.vector()));
                    case compare_lessequal: return type(_mm_cmple_pd((__m128d) data_.vector, (__m128d) vec.vector()));
                    case compare_greater: return type(_mm_cmple_pd((__m128d) vec.vector(), (__m128d) data_.vector));
                    case compare_greaterequal: return type(_mm_cmplt_pd((__m128d) vec.vector(), (__m128d) data_.vector));
                }
#else
                switch (compare) {
                    default: return make_init_cmp_values(vec, [](EffectiveType a, EffectiveType b) {return a == b;});
                    case compare_nequal: return make_init_cmp_values(vec, [](EffectiveType a, EffectiveType b) {return a != b;});
                    case compare_less: return make_init_cmp_values(vec, [](EffectiveType a, EffectiveType b) {return a < b;});
                    case compare_lessequal: return make_init_cmp_values(vec, [](EffectiveType a, EffectiveType b) {return a <= b;});
                    case compare_greater: return make_init_cmp_values(vec, [](EffectiveType a, EffectiveType b) {return a > b;});
                    case compare_greaterequal: return make_init_cmp_values(vec, [](EffectiveType a, EffectiveType b) {return a >= b;});
                }
#endif
            }
        }
        else // elements are integers
        {
#ifdef CPPBITS_SSE2
            if (elements_are_signed)
            {
                switch (element_bits) {
                    default: /* 8 */
                        switch (compare) {
                            default: return type(_mm_cmpeq_epi8(data_.vector, vec.vector()));
                            case compare_nequal: return ~type(_mm_cmpeq_epi8(data_.vector, vec.vector()));
                            case compare_less: return type(_mm_cmpgt_epi8(vec.vector(), data_.vector));
                            case compare_lessequal: return type(_mm_or_si128(_mm_cmpgt_epi8(vec.vector(), data_.vector), _mm_cmpeq_epi8(data_.vector, vec.vector())));
                            case compare_greater: return type(_mm_cmpgt_epi8(data_.vector, vec.vector()));
                            case compare_greaterequal: return type(_mm_or_si128(_mm_cmpgt_epi8(data_.vector, vec.vector()), _mm_cmpeq_epi8(data_.vector, vec.vector())));
                        }
                    case 16:
                        switch (compare) {
                            default: return type(_mm_cmpeq_epi16(data_.vector, vec.vector()));
                            case compare_nequal: return ~type(_mm_cmpeq_epi16(data_.vector, vec.vector()));
                            case compare_less: return type(_mm_cmpgt_epi16(vec.vector(), data_.vector));
                            case compare_lessequal: return type(_mm_or_si128(_mm_cmpgt_epi16(vec.vector(), data_.vector), _mm_cmpeq_epi16(data_.vector, vec.vector())));
                            case compare_greater: return type(_mm_cmpgt_epi16(data_.vector, vec.vector()));
                            case compare_greaterequal: return type(_mm_or_si128(_mm_cmpgt_epi16(data_.vector, vec.vector()), _mm_cmpeq_epi16(data_.vector, vec.vector())));
                        }
                    case 32:
                        switch (compare) {
                            default: return type(_mm_cmpeq_epi32(data_.vector, vec.vector()));
                            case compare_nequal: return ~type(_mm_cmpeq_epi32(data_.vector, vec.vector()));
                            case compare_less: return type(_mm_cmpgt_epi32(vec.vector(), data_.vector));
                            case compare_lessequal: return type(_mm_or_si128(_mm_cmpgt_epi32(vec.vector(), data_.vector), _mm_cmpeq_epi32(data_.vector, vec.vector())));
                            case compare_greater: return type(_mm_cmpgt_epi32(data_.vector, vec.vector()));
                            case compare_greaterequal: return type(_mm_or_si128(_mm_cmpgt_epi32(data_.vector, vec.vector()), _mm_cmpeq_epi32(data_.vector, vec.vector())));
                        }
                    case 64:
#ifdef CPPBITS_SSE4_1
                        switch (compare) {
                            default: return type(_mm_cmpeq_epi64(data_.vector, vec.vector()));
                            case compare_nequal: return ~type(_mm_cmpeq_epi64(data_.vector, vec.vector()));
                            case compare_less: return type(_mm_cmpgt_epi64(vec.vector(), data_.vector));
                            case compare_lessequal: return type(_mm_or_si128(_mm_cmpgt_epi64(vec.vector(), data_.vector), _mm_cmpeq_epi64(data_.vector, vec.vector())));
                            case compare_greater: return type(_mm_cmpgt_epi64(data_.vector, vec.vector()));
                            case compare_greaterequal: return type(_mm_or_si128(_mm_cmpgt_epi64(data_.vector, vec.vector()), _mm_cmpeq_epi64(data_.vector, vec.vector())));
                        }
#else
                        switch (compare) {
                            default: return make_init_cmp_values(vec, [](EffectiveType a, EffectiveType b) {return a == b;});
                            case compare_nequal: return make_init_cmp_values(vec, [](EffectiveType a, EffectiveType b) {return a != b;});
                            case compare_less: return make_init_cmp_values(vec, [](EffectiveType a, EffectiveType b) {return a < b;});
                            case compare_lessequal: return make_init_cmp_values(vec, [](EffectiveType a, EffectiveType b) {return a <= b;});
                            case compare_greater: return make_init_cmp_values(vec, [](EffectiveType a, EffectiveType b) {return a > b;});
                            case compare_greaterequal: return make_init_cmp_values(vec, [](EffectiveType a, EffectiveType b) {return a >= b;});
                        }
#endif
                }
            }
            else // elements are unsigned
            {
                switch (element_bits) {
                    default: /* 8 */
                        switch (compare) {
                            default: return type(_mm_cmpeq_epi8(data_.vector, vec.vector()));
                            case compare_nequal: return ~type(_mm_cmpeq_epi8(data_.vector, vec.vector()));
                            case compare_less:
                            {
                                const __m128i mask = expand_mask(element_mask ^ (element_mask >> 1), element_bits);
                                return type(_mm_cmpgt_epi8(_mm_xor_si128(vec.vector(), mask), _mm_xor_si128(data_.vector, mask)));
                            }
                            case compare_lessequal:
                            {
                                const __m128i mask = expand_mask(element_mask ^ (element_mask >> 1), element_bits);
                                return type(_mm_or_si128(_mm_cmpgt_epi8(_mm_xor_si128(vec.vector(), mask), _mm_xor_si128(data_.vector, mask)), _mm_cmpeq_epi8(data_.vector, vec.vector())));
                            }
                            case compare_greater:
                            {
                                const __m128i mask = expand_mask(element_mask ^ (element_mask >> 1), element_bits);
                                return type(_mm_cmpgt_epi8(_mm_xor_si128(data_.vector, mask), _mm_xor_si128(vec.vector(), mask)));
                            }
                            case compare_greaterequal:
                            {
                                const __m128i mask = expand_mask(element_mask ^ (element_mask >> 1), element_bits);
                                return type(_mm_or_si128(_mm_cmpgt_epi8(_mm_xor_si128(data_.vector, mask), _mm_xor_si128(vec.vector(), mask)), _mm_cmpeq_epi8(data_.vector, vec.vector())));
                            }
                        }
                    case 16:
                        switch (compare) {
                            default: return type(_mm_cmpeq_epi16(data_.vector, vec.vector()));
                            case compare_nequal: return ~type(_mm_cmpeq_epi16(data_.vector, vec.vector()));
                            case compare_less:
                            {
                                const __m128i mask = expand_mask(element_mask ^ (element_mask >> 1), element_bits);
                                return type(_mm_cmpgt_epi16(_mm_xor_si128(vec.vector(), mask), _mm_xor_si128(data_.vector, mask)));
                            }
                            case compare_lessequal:
                            {
                                const __m128i mask = expand_mask(element_mask ^ (element_mask >> 1), element_bits);
                                return type(_mm_or_si128(_mm_cmpgt_epi16(_mm_xor_si128(vec.vector(), mask), _mm_xor_si128(data_.vector, mask)), _mm_cmpeq_epi16(data_.vector, vec.vector())));
                            }
                            case compare_greater:
                            {
                                const __m128i mask = expand_mask(element_mask ^ (element_mask >> 1), element_bits);
                                return type(_mm_cmpgt_epi16(_mm_xor_si128(data_.vector, mask), _mm_xor_si128(vec.vector(), mask)));
                            }
                            case compare_greaterequal:
                            {
                                const __m128i mask = expand_mask(element_mask ^ (element_mask >> 1), element_bits);
                                return type(_mm_or_si128(_mm_cmpgt_epi16(_mm_xor_si128(data_.vector, mask), _mm_xor_si128(vec.vector(), mask)), _mm_cmpeq_epi16(data_.vector, vec.vector())));
                            }
                        }
                    case 32:
                        switch (compare) {
                            default: return type(_mm_cmpeq_epi32(data_.vector, vec.vector()));
                            case compare_nequal: return ~type(_mm_cmpeq_epi32(data_.vector, vec.vector()));
                            case compare_less:
                            {
                                const __m128i mask = expand_mask(element_mask ^ (element_mask >> 1), element_bits);
                                return type(_mm_cmpgt_epi32(_mm_xor_si128(vec.vector(), mask), _mm_xor_si128(data_.vector, mask)));
                            }
                            case compare_lessequal:
                            {
                                const __m128i mask = expand_mask(element_mask ^ (element_mask >> 1), element_bits);
                                return type(_mm_or_si128(_mm_cmpgt_epi32(_mm_xor_si128(vec.vector(), mask), _mm_xor_si128(data_.vector, mask)), _mm_cmpeq_epi32(data_.vector, vec.vector())));
                            }
                            case compare_greater:
                            {
                                const __m128i mask = expand_mask(element_mask ^ (element_mask >> 1), element_bits);
                                return type(_mm_cmpgt_epi32(_mm_xor_si128(data_.vector, mask), _mm_xor_si128(vec.vector(), mask)));
                            }
                            case compare_greaterequal:
                            {
                                const __m128i mask = expand_mask(element_mask ^ (element_mask >> 1), element_bits);
                                return type(_mm_or_si128(_mm_cmpgt_epi32(_mm_xor_si128(data_.vector, mask), _mm_xor_si128(vec.vector(), mask)), _mm_cmpeq_epi32(data_.vector, vec.vector())));
                            }
                        }
                    case 64:
#ifdef CPPBITS_SSE4_1
                        switch (compare) {
                            default: return type(_mm_cmpeq_epi64(data_.vector, vec.vector()));
                            case compare_nequal: return ~type(_mm_cmpeq_epi64(data_.vector, vec.vector()));
                            case compare_less:
                            {
                                const __m128i mask = expand_mask(element_mask ^ (element_mask >> 1), element_bits);
                                return type(_mm_cmpgt_epi64(_mm_xor_si128(vec.vector(), mask), _mm_xor_si128(data_.vector, mask)));
                            }
                            case compare_lessequal:
                            {
                                const __m128i mask = expand_mask(element_mask ^ (element_mask >> 1), element_bits);
                                return type(_mm_or_si128(_mm_cmpgt_epi64(_mm_xor_si128(vec.vector(), mask), _mm_xor_si128(data_.vector, mask)), _mm_cmpeq_epi64(data_.vector, vec.vector())));
                            }
                            case compare_greater:
                            {
                                const __m128i mask = expand_mask(element_mask ^ (element_mask >> 1), element_bits);
                                return type(_mm_cmpgt_epi64(_mm_xor_si128(data_.vector, mask), _mm_xor_si128(vec.vector(), mask)));
                            }
                            case compare_greaterequal:
                            {
                                const __m128i mask = expand_mask(element_mask ^ (element_mask >> 1), element_bits);
                                return type(_mm_or_si128(_mm_cmpgt_epi64(_mm_xor_si128(data_.vector, mask), _mm_xor_si128(vec.vector(), mask)), _mm_cmpeq_epi64(data_.vector, vec.vector())));
                            }
                        }
#else
                        switch (compare) {
                            default: return make_init_cmp_values(vec, [](EffectiveType a, EffectiveType b) {return a == b;});
                            case compare_nequal: return make_init_cmp_values(vec, [](EffectiveType a, EffectiveType b) {return a != b;});
                            case compare_less: return make_init_cmp_values(vec, [](EffectiveType a, EffectiveType b) {return a < b;});
                            case compare_lessequal: return make_init_cmp_values(vec, [](EffectiveType a, EffectiveType b) {return a <= b;});
                            case compare_greater: return make_init_cmp_values(vec, [](EffectiveType a, EffectiveType b) {return a > b;});
                            case compare_greaterequal: return make_init_cmp_values(vec, [](EffectiveType a, EffectiveType b) {return a >= b;});
                        }
#endif
                }
            }
#else
            switch (compare) {
                default: return make_init_cmp_values(vec, [](EffectiveType a, EffectiveType b) {return a == b;});
                case compare_nequal: return make_init_cmp_values(vec, [](EffectiveType a, EffectiveType b) {return a != b;});
                case compare_less: return make_init_cmp_values(vec, [](EffectiveType a, EffectiveType b) {return a < b;});
                case compare_lessequal: return make_init_cmp_values(vec, [](EffectiveType a, EffectiveType b) {return a <= b;});
                case compare_greater: return make_init_cmp_values(vec, [](EffectiveType a, EffectiveType b) {return a > b;});
                case compare_greaterequal: return make_init_cmp_values(vec, [](EffectiveType a, EffectiveType b) {return a >= b;});
            }
#endif
        }
    }

    type operator==(x86_sse_vector vec) const {return cmp(vec, compare_equal);}
    type operator!=(x86_sse_vector vec) const {return cmp(vec, compare_nequal);}
    type operator<(x86_sse_vector vec) const {return cmp(vec, compare_less);}
    type operator<=(x86_sse_vector vec) const {return cmp(vec, compare_lessequal);}
    type operator>(x86_sse_vector vec) const {return cmp(vec, compare_greater);}
    type operator>=(x86_sse_vector vec) const {return cmp(vec, compare_greaterequal);}

    /*
     * Sets each element in output to reciprocal of respective element in `this`
     * TODO: doesn't work properly for integral values
     */
    type reciprocal(math_type math) const
    {
        if (elements_are_floats)
        {
            if (element_bits == 32)
            {
                switch (math) {
                    default: return make_broadcast(1.0) / *this;
                    case math_approximate: return type(_mm_rcp_ps((__m128) data_.vector));
                }
            }
            else
                return make_broadcast(1.0) / *this;
        }
        else
        {
            CPPBITS_ERROR("Reciprocal requested for integral vector");
            return *this;
        }
    }

    /*
     * Sets each element in output to square-root of respective element in `this`
     * TODO: doesn't work properly for integral values
     */
    type sqrt(math_type) const
    {
        if (elements_are_floats)
        {
            if (element_bits == 32)
                return type(_mm_sqrt_ps((__m128) data_.vector));
            else
#ifdef CPPBITS_SSE2
                return type(_mm_sqrt_pd((__m128d) data_.vector));
#else
                return make_init_value([](EffectiveType a) {return std::sqrt(a);});
#endif
        }
        else
        {
            CPPBITS_ERROR("Square-root requested for integral vector");
            return *this;
        }
    }

    /*
     * Sets each element in output to reciprocal of square-root of respective element in `this`
     * TODO: doesn't work properly for integral values
     */
    type rsqrt(math_type math) const
    {
        if (elements_are_floats)
        {
            if (element_bits == 32)
            {
                switch (math) {
                    default: return sqrt(math).reciprocal(math);
                    case math_approximate: return type(_mm_rsqrt_ps((__m128) data_.vector));
                }
            }
            else
                return sqrt(math).reciprocal(math);
        }
        else
        {
            CPPBITS_ERROR("Reciprocal of square-root requested for integral vector");
            return *this;
        }
    }

    /*
     * Sets each element in output to maximum of respective elements of `this` and `vec`
     */
    type max(x86_sse_vector vec) const
    {
        if (elements_are_floats)
        {
            if (element_bits == 32)
                return type(_mm_max_ps((__m128) data_.vector, (__m128) vec.vector()));
            else
            {
#ifdef CPPBITS_SSE2
                return type(_mm_max_pd((__m128d) data_.vector, (__m128d) vec.vector()));
#else
                return make_init_values(vec, [](EffectiveType a, EffectiveType b) {using namespace std; return max(a, b);});
#endif
            }
        }
        else
        {
            CPPBITS_ERROR("Integral max not implemented for x86 SIMD vectors yet");
        }
    }

    /*
     * Sets each element in output to minimum of respective elements of `this` and `vec`
     */
    type min(x86_sse_vector vec) const
    {
        if (elements_are_floats)
        {
            if (element_bits == 32)
                return type(_mm_min_ps((__m128) data_.vector, (__m128) vec.vector()));
            else
            {
#ifdef CPPBITS_SSE2
                return type(_mm_min_pd((__m128d) data_.vector, (__m128d) vec.vector()));
#else
                return make_init_values(vec, [](EffectiveType a, EffectiveType b) {using namespace std; return min(a, b);});
#endif
            }
        }
        else
        {
            CPPBITS_ERROR("Integral min not implemented for x86 SIMD vectors yet");
        }
    }

    /*
     * Sets element `idx` to `value`
     */
    template<unsigned int idx>
    type &set(EffectiveType value)
    {
        data_.store[idx] = make_t_from_effective(value) & element_mask;
        return *this;
    }

    /*
     * Sets element `idx` to `value`
     */
    type &set(unsigned int idx, EffectiveType value)
    {
        data_.store[idx] = make_t_from_effective(value) & element_mask;
        return *this;
    }

    /*
     * Sets all bits of element `idx` to 1
     */
    template<unsigned int idx>
    type &set()
    {
        data_.store[idx] = element_mask;
        return *this;
    }

    /*
     * Sets all bits of element `idx` to 1
     */
    type &set(unsigned int idx)
    {
        data_.store[idx] = element_mask;
        return *this;
    }

    /*
     * Sets all bits of element `idx` to 1 if `v` is true, 0 otherwise
     */
    template<unsigned int idx>
    type &set_bits(bool v)
    {
        data_.store[idx] = v * element_mask;
        return *this;
    }

    /*
     * Sets all bits of element `idx` to 1 if `v` is true, 0 otherwise
     */
    type &set_bits(unsigned int idx, bool v)
    {
        data_.store[idx] = v * element_mask;
        return *this;
    }

    /*
     * Sets all bits of element `idx` to 0
     */
    template<unsigned int idx>
    type &reset()
    {
        data_.store[idx] = 0;
        return *this;
    }

    /*
     * Sets all bits of element `idx` to 0
     */
    type &reset(unsigned int idx)
    {
        data_.store[idx] = 0;
        return *this;
    }

    /*
     * Flips all bits of element `idx`
     */
    template<unsigned int idx>
    type &flip()
    {
        data_.store[idx] ^= data_.store[idx];
        return *this;
    }

    /*
     * Flips all bits of element `idx`
     */
    type &flip(unsigned int idx)
    {
        data_.store[idx] ^= data_.store[idx];
        return *this;
    }

    /*
     * Gets value of element `idx`
     */
    template<unsigned int idx>
    EffectiveType get() const
    {
        packed_union pack;
        pack.vector = data_;
        return make_effective_from_t(pack.store[idx]);
    }

    /*
     * Gets value of element `idx`
     */
    EffectiveType get(unsigned int idx) const
    {
        return make_effective_from_t(data_.store[idx]);
    }

    /*
     * Returns minumum value of vector
     */
    static constexpr __m128i min() {return {};}

    /*
     * Returns maximum value of vector
     */
    static constexpr __m128i max() {return vector_mask();}

    /*
     * Returns unsigned mask that can contain all element values
     */
    static constexpr underlying_element_type scalar_mask() {return element_mask;}

    /*
     * Returns the minimum value an element can contain
     */
    static constexpr EffectiveType scalar_min() {return elements_are_signed? -make_effective_from_t(element_mask >> 1) - 1: make_effective_from_t(0);}

    /*
     * Returns the maximum value an element can contain
     */
    static constexpr EffectiveType scalar_max() {return elements_are_signed? make_effective_from_t(element_mask >> 1): make_effective_from_t(element_mask);}

    /*
     * Returns maximum value of vector
     */
    static constexpr __m128i vector_mask()
    {
        return _mm_set1_epi32(-1);
    }

    /*
     * Returns whether elements in this vector are viewed as signed values
     */
    static constexpr bool is_signed() {return elements_are_signed || elements_are_floats;}

    /*
     * Returns the maximum number of elements this vector can contain
     */
    static constexpr unsigned int max_elements() {return elements;}

    /*
     * Returns the size in bits of each element
     */
    static constexpr unsigned int element_size() {return element_bits;}

    /* Dumps vector to memory location (non-portable, so don't use for saving values permanently unless they're read with the same computing configuration) */
    void dump_packed(__m128i *mem)
    {
        *mem = data_;
    }
    void dump_packed(__m128 *mem)
    {
        *mem = (__m128) data_;
    }
    void dump_packed(__m128d *mem)
    {
        *mem = (__m128d) data_;
    }

    /* Reads vector from memory location (non-portable, so don't use for saving values permanently unless they're read with the same computing configuration) */
    type &load_packed(const __m128i *mem)
    {
        data_ = *mem;
        return *this;
    }
    type &load_packed(const __m128 *mem)
    {
        data_ = (__m128i) *mem;
        return *this;
    }
    type &load_packed(const __m128d *mem)
    {
        data_ = (__m128i) *mem;
        return *this;
    }

    /* Dumps vector to array (with an identical number of elements to this vector) of EffectiveType values
     * Undefined behavior results if array is not large enough to contain this vector
     * Element 0 is placed in the first position
     */
    void dump_unpacked(EffectiveType *array)
    {
        if (effective_type_is_exact_size)
            _mm_storeu_ps(impl::type_punning_cast<float *>(array), (__m128) data_.vector);
        else
        {
            for (unsigned i = 0; i < max_elements(); ++i)
                *array++ = make_effective_from_t(data_.store[i]);
        }
    }

    /* Loads vector from array (with an identical number of elements to this vector) of EffectiveType values
     * Undefined behavior results if array is not large enough to contain this vector
     * Element 0 is read from the first position
     */
    type &load_unpacked(const EffectiveType *array)
    {
        if (effective_type_is_exact_size)
            data_.vector = (__m128i) _mm_loadu_ps(impl::type_punning_cast<const float *>(array));
        else
        {
            for (unsigned i = 0; i < max_elements(); ++i)
                data_.store[i] = make_t_from_effective(*array++) & element_mask;
        }
        return *this;
    }

    /* Dumps vector to aligned array (with an identical number of elements to this vector) of EffectiveType values
     * Undefined behavior results if array is not large enough to contain this vector
     * Element 0 is placed in the first position
     */
    void dump_unpacked_aligned(EffectiveType *array)
    {
        if (effective_type_is_exact_size)
            _mm_store_ps(impl::type_punning_cast<float *>(array), (__m128) data_.vector);
        else
        {
            for (unsigned i = 0; i < max_elements(); ++i)
                *array++ = make_effective_from_t(data_.store[i]);
        }
    }

    /* Loads vector from aligned array (with an identical number of elements to this vector) of EffectiveType values
     * Undefined behavior results if array is not large enough to contain this vector
     * Element 0 is read from the first position
     */
    type &load_unpacked_aligned(const EffectiveType *array)
    {
        if (effective_type_is_exact_size)
            data_.vector = (__m128i) _mm_load_ps(impl::type_punning_cast<const float *>(array));
        else
        {
            for (unsigned i = 0; i < max_elements(); ++i)
                data_.store[i] = make_t_from_effective(*array++) & element_mask;
        }
        return *this;
    }

private:
    // Lambda should be a functor taking an EffectiveType and returning an EffectiveType value
    template<typename Lambda>
    type make_init_value(Lambda callable) const
    {
        packed_union res;

        for (unsigned i = 0; i < max_elements(); ++i)
            res.store[i] = make_t_from_effective(callable(make_effective_from_t(data_.store[i]))) & element_mask;

        return type(res.vector);
    }

    // Lambda should be a functor taking 2 EffectiveType values and returning a boolean value
    template<typename Lambda>
    type make_init_cmp_values(x86_sse_vector vec, Lambda callable) const
    {
        packed_union res;

        for (unsigned i = 0; i < max_elements(); ++i)
            res.store[i] = callable(data_.store[i], vec.data_.store[i]) * element_mask;

        return type(res.vector);
    }

    // Lambda should be a functor taking an underlying_element_type and returning an underlying_element_type value
    template<typename Lambda>
    type make_init_raw_value(Lambda callable) const
    {
        packed_union res;

        for (unsigned i = 0; i < max_elements(); ++i)
            res.store[i] = callable(data_.store[i]);

        return type(res.vector);
    }

    // Lambda should be a functor taking 2 underlying_element_type values and returning an underlying_element_type value
    template<typename Lambda>
    type make_init_raw_values(x86_sse_vector vec, Lambda callable) const
    {
        packed_union res;

        for (unsigned i = 0; i < max_elements(); ++i)
            res.store[i] = callable(data_.store[i], vec.data_.store[i]);

        return type(res.vector);
    }

    // Lambda should be a functor taking 2 EffectiveType values and returning an EffectiveType value
    template<typename Lambda>
    type make_init_values(x86_sse_vector vec, Lambda callable) const
    {
        packed_union res;

        for (unsigned i = 0; i < max_elements(); ++i)
            res.store[i] = make_t_from_effective(callable(make_effective_from_t(data_.store[i]),
                                                          make_effective_from_t(vec.data_.store[i]))) & element_mask;

        return type(res.vector);
    }

    static constexpr uint64_t expand_mask_helper(uint64_t mask, unsigned int mask_size, unsigned int level)
    {
        return level == 0? mask:
                           expand_mask_helper(mask | ((mask & (uint64_t(-1) >> (64 - mask_size))) <<
                                               ((level-1) * mask_size))
                                       , mask_size
                                       , level-1);
    }
    static __m128i expand_mask(underlying_element_type mask, unsigned int mask_size)
    {
#ifdef CPPBITS_SSE2
        return _mm_set_epi64x(expand_mask_helper(mask, mask_size, (elements / 2 / (mask_size / element_bits))),
                              expand_mask_helper(mask, mask_size, (elements / 2 / (mask_size / element_bits))));
#else
        alignas(alignment) union {
            uint64_t store[2];
            __m128i vec;
        } pack;

        pack.store[0] = pack.store[1] = expand_mask_helper(mask, mask_size, (elements / 2 / (mask_size / element_bits)));
        return pack.vec;
#endif
    }
    static __m128i expand_value(EffectiveType value)
    {
        switch (element_bits) {
            default: /* 8 */ return _mm_set1_epi8(value);
            case 16: return _mm_set1_epi16(value);
            case 32:
                if (elements_are_floats)
                    return (__m128i) _mm_set1_ps(value);
                else
                    return _mm_set1_epi32(value);
            case 64:
#ifdef CPPBITS_SSE2
                if (elements_are_floats)
                    return  (__m128i) _mm_set1_pd(value);
                else
                    return _mm_set1_epi64x(value);
#else
                return expand_mask(make_t_from_effective(value) & element_mask, element_bits);
#endif
        }
    }

    packed_union data_;
};

template<typename T, typename EffectiveType>
struct native_simd_vector<T, 8, EffectiveType> : public x86_sse_vector<8, EffectiveType>
{
    native_simd_vector() {}
    native_simd_vector(x86_sse_vector<8, EffectiveType> v)
        : x86_sse_vector<8, EffectiveType>(v)
    {}
};

template<typename T, typename EffectiveType>
struct native_simd_vector<T, 16, EffectiveType> : public x86_sse_vector<16, EffectiveType>
{
    native_simd_vector() {}
    native_simd_vector(x86_sse_vector<16, EffectiveType> v)
        : x86_sse_vector<16, EffectiveType>(v)
    {}
};

template<typename T, typename EffectiveType>
struct native_simd_vector<T, 32, EffectiveType> : public x86_sse_vector<32, EffectiveType>
{
    native_simd_vector() {}
    native_simd_vector(x86_sse_vector<32, EffectiveType> v)
        : x86_sse_vector<32, EffectiveType>(v)
    {}
};

template<typename T, typename EffectiveType>
struct native_simd_vector<T, 64, EffectiveType> : public x86_sse_vector<64, EffectiveType>
{
    native_simd_vector() {}
    native_simd_vector(x86_sse_vector<64, EffectiveType> v)
        : x86_sse_vector<64, EffectiveType>(v)
    {}
};

template<unsigned int bits, typename EffectiveType>
std::ostream &operator<<(std::ostream &os, x86_sse_vector<bits, EffectiveType> vec)
{
    os.put('[');
    for (unsigned i = 0; i < vec.max_elements(); ++i)
    {
        if (i != 0)
            os.put(' ');
        os << vec.get(i);
    }
    return os.put(']');
}

template<typename EffectiveType>
std::ostream &operator<<(std::ostream &os, x86_sse_vector<8, EffectiveType> vec)
{
    os.put('[');
    for (unsigned i = 0; i < vec.max_elements(); ++i)
    {
        if (i != 0)
            os.put(' ');
        os << (signed) vec.get(i);
    }
    return os.put(']');
}
#endif // defined CPPBITS_GCC && (defined CPPBITS_X86 || defined CPPBITS_X86_64) && defined CPPBITS_SSE

#endif // X86_SIMD_H
