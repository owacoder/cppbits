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

#if CPPBITS_GCC && defined CPPBITS_X86
#include <x86intrin.h>

template<typename T, typename EffectiveType>
class simd_vector<T, 8, EffectiveType>
{
    static constexpr unsigned int element_bits = 8;
    static constexpr unsigned int vector_digits = 128;

    static_assert(std::is_unsigned<T>::value, "T must be an unsigned integral type");

    static constexpr bool elements_are_signed = std::is_signed<EffectiveType>::value;
    typedef simd_vector<T, element_bits, EffectiveType> type;
    typedef T value_type;
    static constexpr uint64_t ones = -1;
    static constexpr unsigned int elements = vector_digits / element_bits;
    static constexpr uint64_t mask = ones;
    static constexpr uint64_t element_mask = ones >> ((vector_digits / 2) - element_bits);

    static_assert(elements, "Element size is too large for specified type");

    constexpr explicit simd_vector(__m128i value) : data_(value) {}

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

    constexpr simd_vector() : data_{} {}

    template<typename NewType>
    constexpr simd_vector<T, element_bits, NewType> cast() const {return simd_vector<T, element_bits, NewType>::make_vector(data_);}

    template<unsigned int element_size, typename NewType>
    simd_vector<T, element_size, NewType> cast() const {return simd_vector<T, element_size, NewType>::make_vector(data_);}

    constexpr static type make_vector(__m128i value) {return type(value);}
    constexpr static type make_scalar(EffectiveType value) {return type(make_t_from_effective(value) & element_mask);}
    constexpr static type make_broadcast(EffectiveType value) {return type(expand_mask(make_t_from_effective(value) & element_mask, element_bits));}

    type &vector(__m128i value) {return *this = make_vector(value);}
    type &scalar(EffectiveType value) {return *this = make_scalar(value);}
    type &broadcast(EffectiveType value) {return *this = make_broadcast(value);}

    constexpr __m128i vector() const {return data_;}
    EffectiveType scalar() const {return get(0);}

    /*
     * Logical NOT's the entire vector and returns it
     */
    constexpr type operator~() const
    {
        return type(_mm_xor_si128(data_, expand_mask(element_mask, element_bits)));
    }

    /*
     * Logical OR's the entire vector with `vec` and returns it
     */
    constexpr type operator|(const simd_vector &vec) const
    {
        return type(_mm_or_si128(data_, vec.vector()));
    }

    /*
     * Logical AND's the entire vector with `vec` and returns it
     */
    constexpr type operator&(const simd_vector &vec) const
    {
        return type(_mm_and_si128(data_, vec.vector()));
    }

    /*
     * Logical AND NOT's the entire vector with `vec` and returns it
     */
    constexpr type andnot(const simd_vector &vec) const
    {
        return type(_mm_andnot_si128(data_, vec.vector()));
    }

    /*
     * Logical XOR's the entire vector with `vec` and returns it
     */
    constexpr type operator^(const simd_vector &vec) const
    {
        return type(_mm_xor_si128(data_, vec.vector()));
    }

    /*
     * Adds elements of `vec` to `this` (using rollover addition) and stores the result
     */
    constexpr type operator+(const simd_vector &vec) const
    {
        return type(_mm_add_epi8(data_, vec.vector()));
    }

    /*
     * Adds elements of `vec` to `this` (using specified math method) and stores the result
     */
    type &add(const simd_vector &vec, math_type math)
    {
        switch (math) {
            default: /* Rollover arithmetic, math_keeplow */
                data_ = _mm_add_epi8(data_, vec.vector());
                break;
            case math_saturate:
                if (elements_are_signed)
                    data_ = _mm_adds_epi8(data_, vec.vector());
                else
                    data_ = _mm_adds_epu8(data_, vec.vector());
                break;
            case math_keephigh: /* TODO: not yet verified as there is no builtin to keep the high part of an addition */
                __m128i temp = _mm_add_epi8(data_, vec.vector());
                data_ = _mm_cmpgt_epi8(data_, temp);
                if (!elements_are_signed)
                    data_ = _mm_abs_epi8(data_);
                break;
        }
        return *this;
    }

    /*
     * Subtracts elements of `vec` to `this` (using rollover subtraction) and stores the result
     */
    constexpr type operator-(const simd_vector &vec) const
    {
        return type(_mm_sub_epi8(data_, vec.vector()));
    }

    type &sub(const simd_vector &vec, math_type math)
    {
        switch (math) {
            default: /* Rollover arithmetic, math_keeplow */
                data_ = _mm_sub_epi8(data_, vec.vector());
                break;
            case math_saturate:
                if (elements_are_signed)
                    data_ = _mm_subs_epi8(data_, vec.vector());
                else
                    data_ = _mm_subs_epu8(data_, vec.vector());
                break;
            case math_keephigh: /* TODO: not yet implemented as there is no builtin to keep the high part of an addition */
                break;
        }
        return *this;
    }

    /*
     * Negates elements of `this` and returns the resulting vector
     * Note that this function returns the value unmodified if EffectiveType is not a signed type
     * TODO: not yet implemented
     */
    constexpr type negate() const
    {
        return elements_are_signed? type(_mm_add_epi8(_mm_xor_si128(data_, expand_mask(element_mask, element_bits)), expand_mask(1, element_bits))): *this;
    }

    /*
     * Computes the absolute value of elements of `this` and returns the resulting vector
     * Note that this function returns the value unmodified if EffectiveType is not a signed type
     */
    constexpr type abs() const
    {
        return elements_are_signed? type(_mm_abs_epi8(data_)): *this;
    }

    /*
     * Sets each element of `this` to zero if the element is zero, or all ones otherwise, and returns the resulting vector
     */
    constexpr type fill_if_nonzero() const
    {
        return ~type(_mm_cmpeq_epi8(data_, expand_mask(0, element_bits)));
    }

    /*
     * TODO: not yet implemented
     */
    template<unsigned int lsb_pos, unsigned int length>
    type &set()
    {
        data_ |= bitfield_member<T, lsb_pos, length>::bitfield_mask() & mask;
        return *this;
    }

    /*
     * TODO: not yet implemented
     */
    template<unsigned int idx>
    type &set(EffectiveType value)
    {
        data_ |= bitfield_member<T, idx * element_bits, element_bits>(make_t_from_effective(value)).bitfield_value() & mask;
        return *this;
    }

    /*
     * TODO: not yet implemented
     */
    type &set(unsigned int idx, EffectiveType value)
    {
        const unsigned int shift = idx * element_bits;
        data_ = (data_ & ~(element_mask << shift)) | ((T(value) & element_mask) << shift);
        return *this;
    }

    /*
     * TODO: not yet implemented
     */
    type &set(unsigned int idx)
    {
        const unsigned int shift = idx * element_bits;
        data_ |= (element_mask << shift);
        return *this;
    }

    /*
     * TODO: not yet implemented
     */
    template<unsigned int idx>
    type &reset()
    {
        data_ &= ~bitfield_member<T, idx * element_bits, element_bits>::bitfield_mask();
        return *this;
    }

    /*
     * TODO: not yet implemented
     */
    type &reset(unsigned int idx)
    {
        const unsigned int shift = idx * element_bits;
        data_ &= ~(element_mask << shift);
        return *this;
    }

    /*
     * TODO: not yet implemented
     */
    template<unsigned int idx>
    type &flip()
    {
        data_ ^= bitfield_member<T, idx * element_bits, element_bits>::bitfield_mask();
        return *this;
    }

    /*
     * TODO: not yet implemented
     */
    type &flip(unsigned int idx)
    {
        const unsigned int shift = idx * element_bits;
        data_ ^= (element_mask << shift);
        return *this;
    }

    template<unsigned int idx>
    EffectiveType get() const
    {
        alignas(16) uint8_t space[16];
        _mm_store_si128((__m128i *) space, data_);

        if (elements_are_signed)
        {
            const T val = space[idx] & (element_mask >> 1);
            const T negative = (space[idx] >> (element_bits - 1)) & 1;
            return negative? val == 0? scalar_min(): -EffectiveType((~val + 1) & (element_mask >> 1)): EffectiveType(val);
        }
        return EffectiveType(space[idx]);
    }

    EffectiveType get(unsigned int idx) const
    {
        alignas(16) uint8_t space[16];
        _mm_store_si128((__m128i *) space, data_);

        if (elements_are_signed)
        {
            const T val = space[idx] & (element_mask >> 1);
            const T negative = (space[idx] >> (element_bits - 1)) & 1;
            return negative? val == 0? scalar_min(): -EffectiveType((~val + 1) & (element_mask >> 1)): EffectiveType(val);
        }
        return EffectiveType(space[idx]);
    }

    static constexpr T min() {return 0;}
    static constexpr T max() {return mask;}

    static constexpr T scalar_mask() {return element_mask;}
    static constexpr EffectiveType scalar_min() {return elements_are_signed? -EffectiveType(element_mask >> 1) - 1: EffectiveType(0);}
    static constexpr EffectiveType scalar_max() {return elements_are_signed? EffectiveType(element_mask >> 1): EffectiveType(element_mask);}
    static __m128i vector_mask()
    {
        __m128i v;
        return _mm_cmpeq_epi8(v, v);
    }

    static constexpr bool is_signed() {return elements_are_signed;}
    static constexpr unsigned int max_elements() {return elements;}
    static constexpr unsigned int element_size() {return element_bits;}

private:
    static constexpr uint64_t expand_mask_helper(uint64_t mask, unsigned int mask_size, unsigned int level)
    {
        return level == 0? mask:
                           expand_mask_helper(mask | ((mask & (ones >> ((vector_digits / 2) - mask_size))) <<
                                               ((level-1) * element_bits))
                                       , mask_size
                                       , level-1);
    }
    static constexpr __m128i expand_mask(T mask, unsigned int mask_size)
    {
        return _mm_set_epi64x(expand_mask_helper(mask, mask_size, elements / 2),
                              expand_mask_helper(mask, mask_size, elements / 2));
    }

    __m128i data_;
};
#endif // defined __GNUC__ && (defined __x86_64 || defined __x86_64__)

#endif // X86_SIMD_H
