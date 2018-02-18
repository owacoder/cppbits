/*
 * generic_simd.h
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

#ifndef GENERIC_SIMD_H
#define GENERIC_SIMD_H

#include "../fp_convert.h"
#include "../bitfield.h"
#include "../environment.h"

template<typename T, unsigned int element_bits, typename EffectiveType>
class simd_vector
{
    static_assert(std::is_unsigned<T>::value, "T must be an unsigned integral type");
    static_assert(element_bits, "Number of bits per element must be greater than zero");

    static constexpr unsigned int vector_digits = std::numeric_limits<T>::digits;
    static constexpr bool elements_are_signed = std::is_signed<EffectiveType>::value;
    static constexpr bool elements_are_floats = std::is_floating_point<EffectiveType>::value;
    typedef simd_vector<T, element_bits, EffectiveType> type;
    typedef T value_type;
    static constexpr T ones = -1;
    static constexpr unsigned int elements = vector_digits / element_bits;
    static constexpr T mask = ones >> (vector_digits % element_bits);
    static constexpr T element_mask = ones >> (vector_digits - element_bits);

    static_assert(!elements_are_floats || (element_bits == 32 || element_bits == 64), "Floating-point element size must be 32 or 64 bits");
    static_assert(elements, "Element size is too large for specified type");

    constexpr explicit simd_vector(T value) : data_(value) {}

#if defined CPPBITS_X86 || defined CPPBITS_X86_64
    constexpr static T make_t_from_effective(EffectiveType v)
    {
        return elements_are_floats? element_bits == 32? T(float_cast_to_ieee_754(v)): T(double_cast_to_ieee_754(v)): T(v);
    }
    constexpr static EffectiveType make_effective_from_t(T v)
    {
        return elements_are_floats? element_bits == 32? EffectiveType(float_cast_from_ieee_754(v)): EffectiveType(double_cast_from_ieee_754(v)): EffectiveType(v);
    }
#else
    constexpr static T make_t_from_effective(EffectiveType v)
    {
        return elements_are_floats? element_bits == 32? T(float_to_ieee_754(v)): T(double_to_ieee_754(v)): T(v);
    }
    constexpr static EffectiveType make_effective_from_t(T v)
    {
        return elements_are_floats? element_bits == 32? EffectiveType(float_from_ieee_754(v)): EffectiveType(double_from_ieee_754(v)): EffectiveType(v);
    }
#endif

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
     * Returns a representation of a vector with specified value assigned to the entire vector
     */
    constexpr static type make_vector(T value) {return type(value);}

    /*
     * Returns a representation of a vector with specified value assigned to element 0
     */
    constexpr static type make_scalar(EffectiveType value) {return type(make_t_from_effective(value) & element_mask);}

    /*
     * Returns a representation of a vector with specified value assigned to every element in the vector
     */
    constexpr static type make_broadcast(EffectiveType value) {return type(expand_mask(make_t_from_effective(value) & element_mask, element_bits, elements));}

    /*
     * Sets this vector to a representation of a vector with specified value assigned to the entire vector
     */
    type &vector(T value) {return *this = make_vector(value);}

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
    constexpr T vector() const {return data_;}

    /*
     * Returns the value of element 0
     */
    EffectiveType scalar() const {return get<0>();}

    /*
     * Logical NOT's the entire vector and returns it
     */
    constexpr type operator~() const
    {
        return type(~data_ & mask);
    }

    /*
     * Logical OR's the entire vector with `vec` and returns it
     */
    constexpr type operator|(const simd_vector &vec) const
    {
        return type(data_ | vec.vector());
    }

    /*
     * Logical AND's the entire vector with `vec` and returns it
     */
    constexpr type operator&(const simd_vector &vec) const
    {
        return type(data_ & vec.vector());
    }

    /*
     * Logical AND's the entire vector with negated `vec` and returns it
     */
    constexpr type and_not(const simd_vector &vec) const
    {
        return type(data_ & ~vec.vector());
    }

    /*
     * Logical XOR's the entire vector with `vec` and returns it
     */
    constexpr type operator^(const simd_vector &vec) const
    {
        return type(data_ ^ vec.vector());
    }

    /*
     * Adds all elements of `this` and returns the resulting sum
     */
    constexpr EffectiveType horizontal_sum() const
    {
        return horizontal_sum_helper(elements);
    }

    /*
     * Adds elements of `vec` to `this` (using rollover addition) and returns the result
     */
    type operator+(const simd_vector &vec) const
    {
        if (elements_are_floats)
        {
            type result;
            for (unsigned i = 0; i < max_elements(); ++i)
                result.set(i, get(i) + vec.get(i));
            return result;
        }
        else
        {
            constexpr T add_mask = expand_mask(element_mask >> 1, element_bits, elements);
            return type(((data_ & add_mask) + (vec.vector() & add_mask)) ^ ((data_ ^ vec.vector()) & ~add_mask));
        }
    }

    /*
     * Adds elements of `vec` to `this` (using specified math method) and returns the result
     */
    type add(const simd_vector &vec, math_type math) const
    {
        if (elements_are_floats)
        {
            type result;
            for (unsigned i = 0; i < max_elements(); ++i)
                result.set(i, get(i) + vec.get(i));
            return result;
        }
        else
        {
            constexpr T add_mask = expand_mask(element_mask >> 1, element_bits, elements);
            switch (math) {
                default: /* Rollover arithmetic, math_keeplow */
                    return type(((data_ & add_mask) + (vec.vector() & add_mask)) ^ ((data_ ^ vec.vector()) & ~add_mask));
                case math_saturate:
                    if (elements_are_signed)
                    {
                        const T temp = ((data_ & add_mask) + (vec.vector() & add_mask)) ^ ((data_ ^ vec.vector()) & ~add_mask);
                        const T overflow = ((data_ ^ temp) & (vec.vector() ^ temp) & ~add_mask) >> (element_bits - 1);
                        return type((temp & ((expand_mask(1, element_bits, elements) - overflow) * element_mask)) |
                                    (overflow * (element_mask ^ (element_mask >> 1)) - ((temp >> (element_bits - 1)) & overflow)));
                    }
                    else
                    {
                        const T temp = ((data_ & add_mask) + (vec.vector() & add_mask)) ^ ((data_ ^ vec.vector()) & ~add_mask);
                        return type(temp | ((((data_ & vec.vector()) | (~temp & (data_ | vec.vector()))) & ~add_mask) >> (element_bits - 1)) * element_mask);
                    }
                    break;
                case math_keephigh:
                {
                    constexpr T overflow_mask = expand_mask(1, element_bits, elements);
                    if (elements_are_signed)
                        return type(((((data_ & vec.vector()) | (((data_ & add_mask) + (vec.vector() & add_mask)) & (data_ ^ vec.vector()))) >> (element_bits - 1)) & overflow_mask) * element_mask);
                    else
                        return type((((data_ & vec.vector()) | (((data_ & add_mask) + (vec.vector() & add_mask)) & (data_ ^ vec.vector()))) >> (element_bits - 1)) & overflow_mask);
                    break;
                }
            }
        }
        return *this;
    }

    /*
     * Subtracts elements of `vec` to `this` (using rollover subtraction) and returns the result
     */
    type operator-(const simd_vector &vec) const
    {
        if (elements_are_floats)
        {
            type result;
            for (unsigned i = 0; i < max_elements(); ++i)
                result.set(i, get(i) - vec.get(i));
            return result;
        }
        else
        {
            constexpr T sub_mask = expand_mask(element_mask >> 1, element_bits, elements);
            return type(((data_ | ~sub_mask) - (vec.vector() & sub_mask)) ^ ((data_ ^ ~vec.vector()) & ~sub_mask));
        }
    }

    /*
     * Subtracts elements of `vec` from `this` (using specified math method) and returns the result
     */
    type sub(const simd_vector &vec, math_type math) const
    {
        if (elements_are_floats)
        {
            type result;
            for (unsigned i = 0; i < max_elements(); ++i)
                result.set(i, get(i) - vec.get(i));
            return result;
        }
        else
        {
            constexpr T sub_mask = expand_mask(element_mask >> 1, element_bits, elements);
            switch (math) {
                default: /* Rollover arithmetic, math_keeplow */
                    return type(((data_ | ~sub_mask) - (vec.vector() & sub_mask)) ^ ((data_ ^ ~vec.vector()) & ~sub_mask));
                    break;
                case math_saturate:
                    if (elements_are_signed)
                    {
                        const T temp = ((data_ | ~sub_mask) - (vec.vector() & sub_mask)) ^ ((data_ ^ ~vec.vector()) & ~sub_mask);
                        const T overflow = ((data_ ^ temp) & ~(vec.vector() ^ temp) & ~sub_mask) >> (element_bits - 1);
                        return type((temp & ((expand_mask(1, element_bits, elements) - overflow) * element_mask)) |
                                    (overflow * (element_mask ^ (element_mask >> 1)) - ((temp >> (element_bits - 1)) & overflow)));
                    }
                    else
                    {
                        const T temp = ((data_ | ~sub_mask) - (vec.vector() & sub_mask)) ^ ((data_ ^ ~vec.vector()) & ~sub_mask);
                        return type(temp & ((~((vec.vector() & temp) | (~data_ & (vec.vector() | temp))) & ~sub_mask) >> (element_bits - 1)) * element_mask);
                    }
                    break;
                case math_keephigh: /* TODO: not accurate right now */
                    constexpr T overflow_mask = expand_mask(1, element_bits, elements);
                    if (elements_are_signed)
                        return type((((((data_ | ~sub_mask) - (vec.vector() & sub_mask)) & (data_ ^ ~vec.vector())) >> (element_bits - 1)) & overflow_mask) * element_mask);
                    else
                        return type(((((data_ | ~sub_mask) - (vec.vector() & sub_mask)) & (data_ ^ ~vec.vector())) >> (element_bits - 1)) & overflow_mask);
                    break;
            }
        }
        return *this;
    }

    /*
     * Computes the average of each element of `this` and `vec` as `((element of this) + (element of vec) + 1)/2`
     * TODO: if element_bits == number of bits in T, undefined behavior results
     */
    type avg(const simd_vector &vec) const
    {
        constexpr T ones_mask = expand_mask(1, element_bits * 2, (elements+1) / 2);
        constexpr T avg_mask = expand_mask(element_mask, element_bits * 2, (elements+1) / 2);
        return type((((ones_mask + ((data_ >> element_bits) & avg_mask) + ((vec.vector() >> element_bits) & avg_mask)) << (element_bits - 1)) & (avg_mask << element_bits)) |
                    (((ones_mask + (data_ & avg_mask) + (vec.vector() & avg_mask)) >> 1) & avg_mask));
    }

    /*
     * Shifts each element of `this` to the left by `amount` and returns the resulting vector
     * If the invariant `0 <= amount <= element_bits` does not hold, undefined behavior results
     */
    constexpr type operator<<(unsigned int amount) const
    {
        return type((data_ << amount) & ~(expand_mask(1, element_bits, elements) * (element_mask >> (element_bits - amount))));
    }

    /*
     * Shifts each element of `this` to the left by `amount` (using specified shift type) and returns the resulting vector
     * If the invariant `0 <= amount <= element_bits` does not hold, undefined behavior results
     */
    constexpr type shl(unsigned int amount, shift_type) const
    {
        return type((data_ << amount) & ~(expand_mask(1, element_bits, elements) * (element_mask >> (element_bits - amount))));
    }

    /*
     * Shifts each element of `this` to the right by `amount` (using shift_natural behavior) and returns the resulting vector
     * If the invariant `0 <= amount <= element_bits` does not hold, undefined behavior results
     */
    constexpr type operator>>(unsigned int amount) const
    {
        return shr(amount, shift_natural);
    }

    /*
     * Shifts each element of `this` to the right by `amount` (using specified shift type) and returns the resulting vector
     * If the invariant `0 <= amount <= element_bits` does not hold, undefined behavior results
     */
    type shr(unsigned int amount, shift_type shift) const
    {
        switch (shift) {
            default: // shift_natural
                if (elements_are_signed)
                {
                    constexpr T shift_ones = expand_mask(1, element_bits, elements);
                    const T shift_mask = element_mask >> amount;
                    const T temp = (data_ >> amount) & (shift_mask * shift_ones);
                    return type(temp | ((~shift_mask & element_mask) * ((data_ >> (element_bits - 1)) & shift_ones)));
                }
                else
                    return type((data_ >> amount) & (expand_mask(1, element_bits, elements) * (element_mask >> amount)));
                break;
            case shift_arithmetic:
            {
                constexpr T shift_ones = expand_mask(1, element_bits, elements);
                const T shift_mask = element_mask >> amount;
                const T temp = (data_ >> amount) & (shift_mask * shift_ones);
                return type(temp | ((~shift_mask & element_mask) * ((data_ >> (element_bits - 1)) & shift_ones)));
            }
            case shift_logical:
                return type((data_ >> amount) & (expand_mask(1, element_bits, elements) * (element_mask >> amount)));
        }
    }

    /*
     * Negates elements of `this` and returns the resulting vector
     * Note that this function returns the value unmodified if EffectiveType is not a signed type
     * TODO: support floating-point values
     */
    type negate() const
    {
        if (elements_are_signed)
        {
            constexpr T neg_mask = expand_mask(element_mask >> 1, element_bits, elements);
            constexpr T add_mask = expand_mask(1, element_bits, elements);
            return {(~data_ & neg_mask) + add_mask};
        }
        return *this;
    }

    /*
     * Computes the absolute value of elements of `this` and returns the resulting vector
     * Note that this function returns the value unmodified if EffectiveType is not a signed type
     * TODO: support floating-point values
     */
    type abs() const
    {
        if (elements_are_signed)
        {
            constexpr T abs_mask = expand_mask(element_mask >> 1, element_bits, elements);
            const T neg = (data_ & abs_mask) >> (element_bits - 1);
            return {((data_ ^ (neg * element_mask)) & abs_mask) + neg};
        }
        return *this;
    }

    /*
     * Sets each element of `this` to zero if the element is zero, or all ones otherwise, and returns the resulting vector
     */
    type fill_if_nonzero() const
    {
        return type(fill_elements_if_nonzero(data_, element_bits, element_bits));
    }

    /*
     * Sets all bits of the vector between `lsb_pos` and `lsb_pos + length` to 1
     */
    template<unsigned int lsb_pos, unsigned int length>
    type &set()
    {
        data_ |= bitfield_member<T, lsb_pos, length>::bitfield_mask() & mask;
        return *this;
    }

    /*
     * Sets all bits of element `idx` to 1
     */
    template<unsigned int idx>
    type &set(EffectiveType value)
    {
        data_ |= bitfield_member<T, idx * element_bits, element_bits>(make_t_from_effective(value)).bitfield_value() & mask;
        return *this;
    }

    /*
     * Sets element `idx` to `value`
     */
    type &set(unsigned int idx, EffectiveType value)
    {
        const unsigned int shift = idx * element_bits;
        data_ = (data_ & ~(element_mask << shift)) | ((make_t_from_effective(value) & element_mask) << shift);
        return *this;
    }

    /*
     * Sets all bits of element `idx` to 1
     */
    type &set(unsigned int idx)
    {
        const unsigned int shift = idx * element_bits;
        data_ |= (element_mask << shift);
        return *this;
    }

    /*
     * Sets all bits of element `idx` to 0
     */
    template<unsigned int idx>
    type &reset()
    {
        data_ &= ~bitfield_member<T, idx * element_bits, element_bits>::bitfield_mask();
        return *this;
    }

    /*
     * Sets all bits of element `idx` to 0
     */
    type &reset(unsigned int idx)
    {
        const unsigned int shift = idx * element_bits;
        data_ &= ~(element_mask << shift);
        return *this;
    }

    /*
     * Flips all bits of element `idx`
     */
    template<unsigned int idx>
    type &flip()
    {
        data_ ^= bitfield_member<T, idx * element_bits, element_bits>::bitfield_mask();
        return *this;
    }

    /*
     * Flips all bits of element `idx`
     */
    type &flip(unsigned int idx)
    {
        const unsigned int shift = idx * element_bits;
        data_ ^= (element_mask << shift);
        return *this;
    }

    /*
     * Gets value of element `idx`
     */
    template<unsigned int idx>
    EffectiveType get() const
    {
        const unsigned int shift = idx * element_bits;
        if (elements_are_signed && !elements_are_floats)
        {
            const T val = (data_ >> shift) & (element_mask >> 1);
            const T negative = (data_ >> (shift + element_bits - 1)) & 1;
            return negative? val == 0? scalar_min(): -make_effective_from_t((~val + 1) & (element_mask >> 1)): make_effective_from_t(val);
        }
        return make_effective_from_t((data_ >> shift) & element_mask);
    }

    /*
     * Gets value of element `idx`
     */
    EffectiveType get(unsigned int idx) const
    {
        const unsigned int shift = idx * element_bits;
        if (elements_are_signed && !elements_are_floats)
        {
            const T val = (data_ >> shift) & (element_mask >> 1);
            const T negative = (data_ >> (shift + element_bits - 1)) & 1;
            return negative? val == 0? scalar_min(): -make_effective_from_t((~val + 1) & (element_mask >> 1)): make_effective_from_t(val);
        }
        return make_effective_from_t((data_ >> shift) & element_mask);
    }

    /*
     * Returns minumum value of vector
     */
    static constexpr T min() {return 0;}

    /*
     * Returns maximum value of vector
     */
    static constexpr T max() {return mask;}

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
    static constexpr T vector_mask() {return mask;}

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
    static constexpr T expand_mask(T mask, unsigned int mask_size, unsigned int level)
    {
        return level == 0? mask:
                           expand_mask(mask | ((mask & (ones >> (vector_digits - mask_size))) <<
                                               ((level-1) * mask_size))
                                       , mask_size
                                       , level-1);
    }
    static constexpr T fill_elements_if_nonzero(T vector, unsigned int mask_size, unsigned int number_of_bits_to_merge)
    {
        return number_of_bits_to_merge == 0? (vector & expand_mask(1, mask_size, elements)) * element_mask:
                           fill_elements_if_nonzero(((vector >> number_of_bits_to_merge/2) & expand_mask(element_mask >> number_of_bits_to_merge/2, mask_size, elements)) |
                                                    (vector & expand_mask(element_mask >> number_of_bits_to_merge/2, mask_size, elements))
                                                    , mask_size
                                                    , number_of_bits_to_merge / 2);
    }
    constexpr EffectiveType horizontal_sum_helper(unsigned int level) const
    {
        return level == 0? EffectiveType(0): EffectiveType(get(level-1) + horizontal_sum_helper(level-1));
    }

    T data_;
};

#endif // GENERIC_SIMD_H
