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

template<typename T, typename EffectiveType>
class simd_vector<T, 32, EffectiveType>
{
    static constexpr unsigned int element_bits = 32;
    static constexpr unsigned int vector_digits = 128;
    static constexpr unsigned int alignment = 16;

    static_assert(std::is_unsigned<T>::value, "T must be an unsigned integral type");
    static_assert(element_bits, "Number of bits per element must be greater than zero");

    static constexpr bool elements_are_signed = std::is_signed<EffectiveType>::value;
    static constexpr bool elements_are_floats = std::is_floating_point<EffectiveType>::value;
    typedef simd_vector<T, element_bits, EffectiveType> type;
    typedef uint64_t value_type;
    static constexpr uint64_t ones = -1;
    static constexpr unsigned int elements = vector_digits / element_bits;
    static constexpr T element_mask = ones >> (64 - element_bits);

    static_assert(!elements_are_floats || (element_bits == 32 || element_bits == 64), "Floating-point element size must be 32 or 64 bits");
    static_assert(elements, "Element size is too large for specified type");
    static_assert(elements_are_floats || std::numeric_limits<EffectiveType>::digits + elements_are_signed >= element_bits, "Element size is too large for specified effective type `EffectiveType`");

    constexpr explicit simd_vector(__m128i value) : data_(value) {}
    constexpr explicit simd_vector(__m128 value) : data_((__m128i) value) {}

    constexpr static T make_t_from_effective(EffectiveType v)
    {
        return elements_are_floats? element_bits == 32? T(float_cast_to_ieee_754(v)): T(double_cast_to_ieee_754(v)): T(v);
    }
    constexpr static EffectiveType make_effective_from_t(T v)
    {
        return elements_are_floats? element_bits == 32? EffectiveType(float_cast_from_ieee_754(v)): EffectiveType(double_cast_from_ieee_754(v)): EffectiveType(v);
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
        math_saturate, /* Saturating arithmetic */
        math_keephigh, /* Keep high part of result */
        math_keeplow, /* Rollover arithmetic, keep low part of result */
    };

    enum shift_type
    {
        shift_natural, /* Either logical or arithmetic, depending on the effective element type */
        shift_logical, /* Logical shift shifts in zeros */
        shift_arithmetic, /* Arithmetic shift copies the sign bit in from the left, zeros from the right */
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
    constexpr simd_vector() : data_{} {}

    /*
     * Returns a new representation of the vector, casted to a different type.
     * The data is not modified.
     */
    template<typename NewType>
    constexpr simd_vector<T, element_bits, NewType> cast() const {return simd_vector<T, element_bits, NewType>::make_vector(data_);}

    /*
     * Returns a new representation of the vector, casted to a different type with different element size.
     * The data is not modified.
     */
    template<unsigned int element_size, typename NewType>
    simd_vector<T, element_size, NewType> cast() const {return simd_vector<T, element_size, NewType>::make_vector(data_);}

    /*
     * Returns a new representation of the vector, casted to a different type with different element size, with a different underlying type.
     * The data is not modified.
     */
    template<typename UnderlyingType, unsigned int element_size, typename NewType>
    simd_vector<UnderlyingType, element_size, NewType> cast() const {return simd_vector<UnderlyingType, element_size, NewType>::make_vector(data_);}

    /*
     * Returns a representation of a vector with specified value assigned to the entire vector
     */

    constexpr static type make_vector(__m128i value) {return type(value);}

    /*
     * Returns a representation of a vector with specified value assigned to element 0
     */
    static type make_scalar(EffectiveType value)
    {
        alignas(alignment) uint32_t temp = make_t_from_effective(value);
        return type(_mm_load_ss((float *) &temp));
    }

    /*
     * Returns a representation of a vector with specified value assigned to every element in the vector
     */
    constexpr static type make_broadcast(EffectiveType value) {return type(expand_mask(make_t_from_effective(value) & element_mask, element_bits));}

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
    constexpr __m128i vector() const {return data_;}

    /*
     * Returns the value of element 0
     */
    EffectiveType scalar() const {return get<0>();}

    /*
     * Logical NOT's the entire vector and returns it
     */
    type operator~() const
    {
        return type(_mm_xor_ps((__m128) data_, (__m128) expand_mask(element_mask, element_bits)));
    }

    /*
     * Logical OR's the entire vector with `vec` and returns it
     */
    constexpr type operator|(const simd_vector &vec) const
    {
        return type(_mm_or_ps((__m128) data_, (__m128) vec.vector()));
    }

    /*
     * Logical AND's the entire vector with `vec` and returns it
     */
    constexpr type operator&(const simd_vector &vec) const
    {
        return type(_mm_and_ps((__m128) data_, (__m128) vec.vector()));
    }

    /*
     * Logical ANDT's the entire vector with negated `vec` and returns it
     */
    constexpr type and_not(const simd_vector &vec) const
    {
        return type(_mm_andnot_ps((__m128) data_, (__m128) vec.vector()));
    }

    /*
     * Logical XOR's the entire vector with `vec` and returns it
     */
    constexpr type operator^(const simd_vector &vec) const
    {
        return type(_mm_xor_ps((__m128) data_, (__m128) vec.vector()));
    }

    /*
     * Adds all elements of `this` and returns the resulting sum
     */
    constexpr EffectiveType horizontal_sum() const
    {
        CPPBITS_ERROR("Horizontal sum not implemented for x86 32-bit SIMD vectors yet");
    }

    /*
     * Adds elements of `vec` to `this` (using rollover addition) and returns the result
     */
    type operator+(const simd_vector &vec) const {return add(vec, math_keeplow);}

    /*
     * Adds elements of `vec` to `this` (using specified math method) and returns the result
     */
    type add(const simd_vector &vec, math_type math) const
    {
        if (elements_are_floats)
            return type(_mm_add_ps((__m128) data_, (__m128) vec.vector()));
        else
        {
            switch (math) {
                default: /* Rollover arithmetic, math_keeplow */
#ifdef CPPBITS_SSE2
                    return type(_mm_add_epi32(data_, vec.vector()));
#else
                {
                    constexpr uint64_t add_mask = expand_small_mask(element_mask >> 1, element_bits, 64 / element_bits);
                    alignas(alignment) uint64_t store[elements];

                    _mm_store_ps((float *) store, (__m128) data_);
                    _mm_store_ps((float *) (store + 2), (__m128) vec.vector());

                    store[0] = ((store[0] & add_mask) + (store[2] & add_mask)) ^ ((store[0] ^ store[2]) & ~add_mask);
                    store[1] = ((store[1] & add_mask) + (store[3] & add_mask)) ^ ((store[1] ^ store[3]) & ~add_mask);

                    return type(_mm_load_ps((float *) store));
                }
#endif
                case math_saturate:
                    if (elements_are_signed)
                    {
                        CPPBITS_ERROR("Signed saturation not implemented for x86 32-bit SIMD vectors yet");
                    }
                    else
                    {
                        CPPBITS_ERROR("Unsigned saturation not implemented for x86 32-bit SIMD vectors yet");
                    }
                case math_keephigh: /* TODO: not yet verified as there is no builtin to keep the high part of an addition */
                {
                    CPPBITS_ERROR("Addition using math_keephigh not implemented for x86 32-bit SIMD vectors yet");
                }
            }
        }
    }

    /*
     * Subtracts elements of `vec` to `this` (using rollover subtraction) and returns the result
     */
    type operator-(const simd_vector &vec) const {return sub(vec, math_keeplow);}

    /*
     * Subtracts elements of `vec` from `this` (using specified math method) and returns the result
     */
    type sub(const simd_vector &vec, math_type math) const
    {
        if (elements_are_floats)
            return type(_mm_sub_ps((__m128) data_, (__m128) vec.vector()));
        else
        {
            switch (math) {
                default: /* Rollover arithmetic, math_keeplow */
#ifdef CPPBITS_SSE2
                    return type(_mm_sub_epi32(data_, vec.vector()));
#else
                {
                    constexpr uint64_t sub_mask = expand_small_mask(element_mask >> 1, element_bits, 64 / element_bits);
                    alignas(alignment) uint64_t store[elements];

                    _mm_store_ps((float *) store, (__m128) data_);
                    _mm_store_ps((float *) (store + 2), (__m128) vec.vector());

                    store[0] = ((store[0] | ~sub_mask) - (store[2] & sub_mask)) ^ ((store[0] ^ ~store[2]) & ~sub_mask);
                    store[1] = ((store[1] | ~sub_mask) - (store[3] & sub_mask)) ^ ((store[1] ^ ~store[3]) & ~sub_mask);

                    return type(_mm_load_ps((float *) store));
                }
#endif
                case math_saturate:
                    if (elements_are_signed)
                    {
                        CPPBITS_ERROR("Signed saturation not implemented for x86 32-bit SIMD vectors yet");
                    }
                    else
                    {
                        CPPBITS_ERROR("Unsigned saturation not implemented for x86 32-bit SIMD vectors yet");
                    }
                case math_keephigh: /* TODO: not yet verified as there is no builtin to keep the high part of an addition */
                {
                    CPPBITS_ERROR("Subtraction using math_keephigh not implemented for x86 32-bit SIMD vectors yet");
                }
            }
        }
    }

    /*
     * Multiplies elements of `vec` by `this` (using rollover multiplication) and returns the result
     */
    type operator*(const simd_vector &vec) const {return mul(vec, math_keeplow);}

    /*
     * Multiplies elements of `vec` by `this` (using specified math method) and returns the result
     */
    type mul(const simd_vector &vec, math_type) const
    {
        if (elements_are_floats)
            return _mm_mul_ps((__m128) data_, (__m128) vec.vector());
        else
            CPPBITS_ERROR("Integral multiplication not implemented for x86 32-bit SIMD vectors yet");
    }

    /*
     * Multiplies elements of `vec` by `this` (using specified math method) and returns the result
     */
    constexpr type mul_add(const simd_vector &vec, const simd_vector &add, math_type math) const
    {
        return mul(vec, math).add(add, math);
    }

    /*
     * Divides elements of `this` by `vec` (using rollover division) and returns the result
     */
    type operator/(const simd_vector &vec) const {return div(vec, math_keeplow);}

    /*
     * Divides elements of `this` by `vec` (using specified math method) and returns the result
     */
    type div(const simd_vector &vec, math_type) const
    {
        if (elements_are_floats)
            return _mm_div_ps((__m128) data_, (__m128) vec.vector());
        else if (vec.has_zero_element())
            CPPBITS_ERROR("Division by zero");
        else
            CPPBITS_ERROR("Integral division not implemented for x86 32-bit SIMD vectors yet");
    }

    /*
     * Computes the average of each element of `this` and `vec` as `((element of this) + (element of vec) + 1)/2` for integral values,
     * or `((element of this) + (element of vec))/2` for floating-point values
     * TODO: if element_bits == number of bits in T, undefined behavior results
     */
    type avg(const simd_vector &) const
    {
        CPPBITS_ERROR("Element average not implemented for x86 32-bit SIMD vectors yet");
    }

    /*
     * Shifts each element of `this` to the left by `amount` and returns the resulting vector
     * If the invariant `0 <= amount <= element_bits` does not hold, undefined behavior results
     * TODO: support floating-point values
     */
    constexpr type operator<<(unsigned int amount) const {return shl(amount, shift_natural);}

    /*
     * Shifts each element of `this` to the left by `amount` (using specified shift type) and returns the resulting vector
     * If the invariant `0 <= amount <= element_bits` does not hold, undefined behavior results
     * TODO: support floating-point values
     */
    constexpr type shl(unsigned int, shift_type) const
    {
        CPPBITS_ERROR("Left shift not implemented for x86 32-bit SIMD vectors yet");
    }

    /*
     * Shifts each element of `this` to the right by `amount` (using shift_natural behavior) and returns the resulting vector
     * If the invariant `0 <= amount <= element_bits` does not hold, undefined behavior results
     * TODO: support floating-point values
     */
    constexpr type operator>>(unsigned int amount) const {return shr(amount, shift_natural);}

    /*
     * Shifts each element of `this` to the right by `amount` (using specified shift type) and returns the resulting vector
     * If the invariant `0 <= amount <= element_bits` does not hold, undefined behavior results
     * TODO: support floating-point values
     */
    type shr(unsigned int, shift_type) const
    {
        CPPBITS_ERROR("Right shift not implemented for x86 32-bit SIMD vectors yet");
    }

    /*
     * Extracts MSB from each element and places them in the low bits of the result
     * Each bit position in the result corresponds to the element position in the source vector
     * (i.e. Element 0 MSB -> Bit 0, Element 1 MSB -> Bit 1, etc.)
     */
    constexpr unsigned int movmsk() const
    {
        return _mm_movemask_ps((__m128) data_);
    }

    /*
     * Negates elements of `this` and returns the resulting vector
     * Note that this function returns the value unmodified if EffectiveType is not a signed type
     * TODO: are floating-point values supported properly?
     */
    type negate() const
    {
        CPPBITS_ERROR("Negation not implemented for x86 32-bit SIMD vectors yet");
    }

    /*
     * Computes the absolute value of elements of `this` and returns the resulting vector
     * Note that this function returns the value unmodified if EffectiveType is not a signed type
     * TODO: are floating-point values supported properly?
     */
    type abs() const
    {
        CPPBITS_ERROR("Absolute-value not implemented for x86 32-bit SIMD vectors yet");
    }

    /*
     * Sets each element of `this` to zero if the element is zero, or all ones otherwise, and returns the resulting vector
     */
    type fill_if_nonzero() const
    {
#ifdef CPPBITS_SSE2
        return ~type(_mm_cmpeq_epi32(data_, _mm_setzero_si128()));
#else
        alignas(alignment) uint32_t store[elements];

        _mm_store_ps((float *) store, (__m128) data_);
        for (unsigned i = 0; i < elements; ++i)
            store[i] = (store[i] != 0) * element_mask;
        return type(_mm_load_ps((float *) store));
#endif
    }

    /*
     * Returns true if vector has at least one element equal to zero (sign of zero irrelevant in the case of floating-point), false otherwise
     * See <http://graphics.stanford.edu/~seander/bithacks.html#HasLessInWord>
     * This is a rewrite of hasless(u, 1)
     */
    constexpr bool has_zero_element() const
    {
        return count_zero_elements() != 0;
    }

    /*
     * Returns number of zero elements in vector (+0 in the case of floating-point)
     * See <http://graphics.stanford.edu/~seander/bithacks.html#HasLessInWord>
     * This is a rewrite of countless(u, 1)
     * TODO: doesn't work properly with small element sizes
     */
    unsigned int count_zero_elements() const
    {
        const unsigned int count[16] = {0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4};

        if (elements_are_floats)
        {
            __m128 cmp = _mm_cmpeq_ps((__m128) data_, _mm_setzero_ps());
            return count[_mm_movemask_ps((__m128) cmp)];
        }
        else
        {
#ifdef CPPBITS_SSE2
            __m128i cmp = _mm_cmpeq_epi32(data_, _mm_setzero_si128());
            return count[_mm_movemask_ps((__m128) cmp)];
#else
            alignas(alignment) uint32_t store[elements];
            unsigned result = 0;

            _mm_store_ps((float *) store, (__m128) data_);
            for (unsigned i = 0; i < elements; ++i)
                result += store[i] != 0;
            return result;
#endif
        }
    }

    /*
     * Returns true if vector has at least one element equal to `v`, false otherwise
     * See <http://graphics.stanford.edu/~seander/bithacks.html#HasLessInWord>
     * This is a rewrite of hasless(u, 1)
     */
    constexpr bool has_equal_element(EffectiveType v) const
    {
        return count_equal_elements(v) != 0;
    }

    /*
     * Returns number of zero elements in vector
     * See <http://graphics.stanford.edu/~seander/bithacks.html#HasLessInWord>
     * This is a rewrite of countless(u, 1)
     */
    constexpr unsigned int count_equal_elements(EffectiveType v) const
    {
        const unsigned int count[16] = {0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4};

        if (elements_are_floats)
        {
            __m128 cmp = _mm_cmpeq_ps((__m128) data_, expand_mask(make_t_from_effective(v), element_bits));
            return count[_mm_movemask_ps((__m128) cmp)];
        }
        else
        {
#ifdef CPPBITS_SSE2
            __m128i cmp = _mm_cmpeq_epi32(data_, expand_mask(make_t_from_effective(v), element_bits));
            return count[_mm_movemask_ps((__m128) cmp)];
#else
            alignas(alignment) uint32_t store[elements];
            unsigned result = 0;

            _mm_store_ps((float *) store, (__m128) data_);
            for (unsigned i = 0; i < elements; ++i)
                result += make_effective_from_t(store[i]) == v;
            return result;
#endif
        }
    }

    /*
     * Compares `this` to `vec`. If the comparison result is true, the corresponding element is set to all 1's
     * Otherwise, if the comparison result is false, the corresponding element is set to all 0's
     * TODO: verify comparisons for floating-point values
     */
    type cmp(const simd_vector &vec, compare_type compare) const
    {
        if (elements_are_floats)
        {
            switch (compare) {
                default: return type(_mm_cmpeq_ps((__m128) data_, (__m128) vec.vector()));
                case compare_nequal: return type(_mm_cmpneq_ps((__m128) data_, (__m128) vec.vector()));
                case compare_less: return type(_mm_cmplt_ps((__m128) data_, (__m128) vec.vector()));
                case compare_lessequal: return type(_mm_cmple_ps((__m128) data_, (__m128) vec.vector()));
                case compare_greater: return type(_mm_cmple_ps((__m128) vec.vector(), (__m128) data_));
                case compare_greaterequal: return type(_mm_cmplt_ps((__m128) vec.vector(), (__m128) data_));
            }
        }
        else
        {
#ifdef CPPBITS_SSE2
            if (elements_are_signed)
            {
                switch (compare) {
                    default: return type(_mm_cmpeq_epi32(data_, vec.vector()));
                    case compare_nequal: return ~type(_mm_cmpeq_epi32(data_, vec.vector()));
                    case compare_less: return type(_mm_cmpgt_epi32(vec.vector(), data_));
                    case compare_lessequal: return type(_mm_or_si128(_mm_cmpgt_epi32(vec.vector(), data_), _mm_cmpeq_epi32(data_, vec.vector())));
                    case compare_greater: return type(_mm_cmpgt_epi32(data_, vec.vector()));
                    case compare_greaterequal: return type(_mm_or_si128(_mm_cmpgt_epi32(data_, vec.vector()), _mm_cmpeq_epi32(data_, vec.vector())));
                }
            }
            else
            {
                switch (compare) {
                    default: return type(_mm_cmpeq_epi32(data_, vec.vector()));
                    case compare_nequal: return ~type(_mm_cmpeq_epi32(data_, vec.vector()));
                    case compare_less:
                    {
                        const __m128i mask = expand_mask(element_mask ^ (element_mask >> 1), element_bits);
                        return type(_mm_cmpgt_epi32(_mm_xor_si128(vec.vector(), mask), _mm_xor_si128(data_, mask)));
                    }
                    case compare_lessequal:
                    {
                        const __m128i mask = expand_mask(element_mask ^ (element_mask >> 1), element_bits);
                        return type(_mm_or_si128(_mm_cmpgt_epi32(_mm_xor_si128(vec.vector(), mask), _mm_xor_si128(data_, mask)), _mm_cmpeq_epi32(data_, vec.vector())));
                    }
                    case compare_greater:
                    {
                        const __m128i mask = expand_mask(element_mask ^ (element_mask >> 1), element_bits);
                        return type(_mm_cmpgt_epi32(_mm_xor_si128(data_, mask), _mm_xor_si128(vec.vector(), mask)));
                    }
                    case compare_greaterequal:
                    {
                        const __m128i mask = expand_mask(element_mask ^ (element_mask >> 1), element_bits);
                        return type(_mm_or_si128(_mm_cmpgt_epi32(_mm_xor_si128(data_, mask), _mm_xor_si128(vec.vector(), mask)), _mm_cmpeq_epi32(data_, vec.vector())));
                    }
                }
            }
#else
            type result;
            for (unsigned i = 0; i < elements; ++i)
            {
                switch (compare) {
                    default: result.init_bits(i, get(i) == vec.get(i)); break;
                    case compare_nequal: result.init_bits(i, get(i) != vec.get(i)); break;
                    case compare_less: result.init_bits(i, get(i) < vec.get(i)); break;
                    case compare_lessequal: result.init_bits(i, get(i) <= vec.get(i)); break;
                    case compare_greater: result.init_bits(i, get(i) > vec.get(i)); break;
                    case compare_greaterequal: result.init_bits(i, get(i) >= vec.get(i)); break;
                }
            }
            return result;
#endif
        }
        CPPBITS_ERROR("Integral compare not implemented for x86 32-bit SIMD vectors yet");
    }

    type operator==(const simd_vector &vec) const {return cmp(vec, compare_equal);}
    type operator!=(const simd_vector &vec) const {return cmp(vec, compare_nequal);}
    type operator<(const simd_vector &vec) const {return cmp(vec, compare_less);}
    type operator<=(const simd_vector &vec) const {return cmp(vec, compare_lessequal);}
    type operator>(const simd_vector &vec) const {return cmp(vec, compare_greater);}
    type operator>=(const simd_vector &vec) const {return cmp(vec, compare_greaterequal);}

    /*
     * Sets each element in output to reciprocal of respective element in `this`
     * TODO: doesn't work properly for integral values
     */
    type reciprocal() const
    {
        if (elements_are_floats)
            return type(_mm_rcp_ps((__m128) data_));
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
    type sqrt() const
    {
        if (elements_are_floats)
            return type(_mm_sqrt_ps((__m128) data_));
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
    type rsqrt() const
    {
        if (elements_are_floats)
            return type(_mm_rsqrt_ps((__m128) data_));
        else
        {
            CPPBITS_ERROR("Reciprocal of square-root requested for integral vector");
            return *this;
        }
    }

    /*
     * Sets each element in output to maximum of respective elements of `this` and `vec`
     */
    type max(const simd_vector &vec) const
    {
        if (elements_are_floats)
            return type(_mm_max_ps((__m128) data_, (__m128) vec.vector()));
        else
        {
            CPPBITS_ERROR("Integral max not implemented for x86 32-bit SIMD vectors yet");
        }
    }

    /*
     * Sets each element in output to maximum of respective elements of `this` and `vec`
     */
    type min(const simd_vector &vec) const
    {
        if (elements_are_floats)
            return type(_mm_min_ps((__m128) data_, (__m128) vec.vector()));
        else
        {
            CPPBITS_ERROR("Integral min not implemented for x86 32-bit SIMD vectors yet");
        }
    }

    /*
     * Sets element `idx` to `value`
     * TODO: improve performance?
     */
    template<unsigned int idx>
    type &set(EffectiveType value)
    {
        alignas(alignment) uint32_t store[elements];
        _mm_store_ps((float *) store, (__m128) data_);
        store[idx] = make_t_from_effective(value);
        data_ = (__m128i) _mm_load_ps((float *) store);
        return *this;
    }

    /*
     * Sets element `idx` to `value`
     * TODO: improve performance?
     */
    type &set(unsigned int idx, EffectiveType value)
    {
        alignas(alignment) uint32_t store[elements];
        _mm_store_ps((float *) store, (__m128) data_);
        store[idx] = make_t_from_effective(value);
        data_ = (__m128i) _mm_load_ps((float *) store);
        return *this;
    }

    /*
     * Sets all bits of element `idx` to 1
     */
    template<unsigned int idx>
    type &set()
    {
        alignas(alignment) uint32_t store[elements];
        _mm_store_ps((float *) store, (__m128) data_);
        store[idx] = element_mask;
        data_ = (__m128i) _mm_load_ps((float *) store);
        return *this;
    }

    /*
     * Sets all bits of element `idx` to 1
     */
    type &set(unsigned int idx)
    {
        alignas(alignment) uint32_t store[elements];
        _mm_store_ps((float *) store, (__m128) data_);
        store[idx] = element_mask;
        data_ = (__m128i) _mm_load_ps((float *) store);
        return *this;
    }

    /*
     * Sets all bits of element `idx` to 1 if `v` is true, 0 otherwise
     */
    type &set_bits(unsigned int idx, bool v)
    {
        alignas(alignment) uint32_t store[elements];
        _mm_store_ps((float *) store, (__m128) data_);
        store[idx] = v * element_mask;
        data_ = (__m128i) _mm_load_ps((float *) store);
        return *this;
    }

    /*
     * Sets all bits of element `idx` to 1 if `v` is true, 0 otherwise
     */
    template<unsigned int idx>
    type &set_bits(bool v)
    {
        alignas(alignment) uint32_t store[elements];
        _mm_store_ps((float *) store, (__m128) data_);
        store[idx] = v * element_mask;
        data_ = (__m128i) _mm_load_ps((float *) store);
        return *this;
    }

    /*
     * Sets all bits of element `idx` to 0
     */
    template<unsigned int idx>
    type &reset()
    {
        alignas(alignment) uint32_t store[elements];
        _mm_store_ps((float *) store, (__m128) data_);
        store[idx] = 0;
        data_ = (__m128i) _mm_load_ps((float *) store);
        return *this;
    }

    /*
     * Sets all bits of element `idx` to 0
     */
    type &reset(unsigned int idx)
    {
        alignas(alignment) uint32_t store[elements];
        _mm_store_ps((float *) store, (__m128) data_);
        store[idx] = 0;
        data_ = (__m128i) _mm_load_ps((float *) store);
        return *this;
    }

    /*
     * Flips all bits of element `idx`
     */
    template<unsigned int idx>
    type &flip()
    {
        alignas(alignment) uint32_t store[elements];
        _mm_store_ps((float *) store, (__m128) data_);
        store[idx] ^= store[idx];
        data_ = (__m128i) _mm_load_ps((float *) store);
        return *this;
    }

    /*
     * Flips all bits of element `idx`
     */
    type &flip(unsigned int idx)
    {
        alignas(alignment) uint32_t store[elements];
        _mm_store_ps((float *) store, (__m128) data_);
        store[idx] ^= store[idx];
        data_ = (__m128i) _mm_load_ps((float *) store);
        return *this;
    }

    /*
     * Gets value of element `idx`
     */
    template<unsigned int idx>
    EffectiveType get() const
    {
        alignas(alignment) uint32_t space[elements];
        _mm_store_ps((float *) space, (__m128) data_);

        if (elements_are_signed && !elements_are_floats)
        {
            const T val = space[idx] & (element_mask >> 1);
            const T negative = (space[idx] >> (element_bits - 1)) & 1;
            return negative? val == 0? scalar_min(): -make_effective_from_t((~val + 1) & (element_mask >> 1)): make_effective_from_t(val);
        }
        return make_effective_from_t(space[idx]);
    }

    /*
     * Gets value of element `idx`
     */
    EffectiveType get(unsigned int idx) const
    {
        alignas(alignment) uint32_t space[elements];
        _mm_store_ps((float *) space, (__m128) data_);

        if (elements_are_signed && !elements_are_floats)
        {
            const T val = space[idx] & (element_mask >> 1);
            const T negative = (space[idx] >> (element_bits - 1)) & 1;
            return negative? val == 0? scalar_min(): -make_effective_from_t((~val + 1) & (element_mask >> 1)): make_effective_from_t(val);
        }
        return make_effective_from_t(space[idx]);
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
    static constexpr T scalar_mask() {return element_mask;}

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
    static __m128i vector_mask()
    {
        __m128i v;
        return _mm_cmpeq_epi32(v, v);
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

private:
    /*
     * Initializes element `idx` to `value`
     * Expects that element contains 0 prior to function call
     */
    template<unsigned int idx>
    type &init(EffectiveType value)
    {
        return set<idx>(value);
    }

    /*
     * Sets element `idx` to `value`
     * Expects that element contains 0 prior to function call
     */
    type &init(unsigned int idx, EffectiveType value)
    {
        return set(idx, value);
    }

    /*
     * Sets all bits of element `idx` to 1 if `v` is true, 0 otherwise
     * Expects that element contains 0 prior to function call
     */
    type &init_bits(unsigned int idx, bool v)
    {
        return set_bits(idx, v);
    }

    /*
     * Sets all bits of element `idx` to 1 if `v` is true, 0 otherwise
     * Expects that element contains 0 prior to function call
     */
    template<unsigned int idx>
    type &init_bits(bool v)
    {
        return set_bits<idx>(v);
    }

    static constexpr uint64_t expand_mask_helper(uint64_t mask, unsigned int mask_size, unsigned int level)
    {
        return level == 0? mask:
                           expand_mask_helper(mask | ((mask & (ones >> (64 - mask_size))) <<
                                               ((level-1) * element_bits))
                                       , mask_size
                                       , level-1);
    }
    static __m128i expand_mask(T mask, unsigned int mask_size)
    {
#ifdef CPPBITS_SSE2
        return _mm_set_epi64x(expand_mask_helper(mask, mask_size, elements / 2),
                              expand_mask_helper(mask, mask_size, elements / 2));
#else
        alignas(alignment) uint64_t store[elements/2];
        store[0] = store[1] = expand_mask_helper(mask, mask_size, elements / 2);
        return (__m128i) _mm_load_ps((float *) store);
#endif
    }

    static constexpr uint64_t expand_small_mask(uint64_t mask, unsigned int mask_size, unsigned int level)
    {
        return level == 0? mask:
                           expand_small_mask(mask | ((mask & (ones >> (64 - mask_size))) <<
                                               ((level-1) * mask_size))
                                       , mask_size
                                       , level-1);
    }

    __m128i data_;
};
#endif // defined CPPBITS_GCC && (defined CPPBITS_X86 || defined CPPBITS_X86_64) && defined CPPBITS_SSE

#endif // X86_SIMD_H
