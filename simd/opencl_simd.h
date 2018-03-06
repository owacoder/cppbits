/*
 * opencl_simd.h
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

#ifndef OPENCL_SIMD_H
#define OPENCL_SIMD_H

#include "../environment.h"
#include "CL/cl.hpp"
#include <limits.h>

namespace cppbits {
template<unsigned int desired_elements, unsigned int element_bits, typename EffectiveType>
class opencl_simd_vector {
    static_assert(std::is_unsigned<T>::value, "T must be an unsigned integral type");
    static_assert(element_bits, "Number of bits per element must be greater than zero");

    static constexpr unsigned int vector_digits = desired_elements * element_bits;
    static constexpr bool elements_are_floats = std::is_floating_point<EffectiveType>::value;
    static constexpr bool elements_are_signed = std::is_signed<EffectiveType>::value && !elements_are_floats;
    typedef opencl_simd_vector<desired_elements, element_bits, EffectiveType> type;

public:
    typedef typename impl::unsigned_int<element_bits>::type underlying_element_type;
    typedef cl::Buffer underlying_vector_type;

private:
    static constexpr underlying_element_type ones = -1;
    static constexpr unsigned int elements = vector_digits / element_bits;
    static constexpr underlying_element_type element_mask = ones >> (std::numeric_limits<underlying_element_type>::digits - element_bits);

    static constexpr bool effective_type_is_exact_size = sizeof(EffectiveType) * CHAR_BIT == element_bits;
    static_assert(!elements_are_floats || (element_bits == 32 || element_bits == 64), "Floating-point element size must be 32 or 64 bits");
    static_assert(element_bits <= 64, "Element size is too large");
    static_assert(elements_are_floats || (sizeof(EffectiveType) * CHAR_BIT) >= element_bits, "Element size is too large for specified effective type `EffectiveType`");

    constexpr explicit opencl_simd_vector(T value) : data_(value) {}

    constexpr static T make_t_from_effective(EffectiveType v)
    {
        return elements_are_floats? element_bits == 32? T(float_to_ieee_754(v)): T(double_to_ieee_754(v)): T(v);
    }
    static EffectiveType make_effective_from_t(T v)
    {
        if (elements_are_floats)
            return element_bits == 32? EffectiveType(float_from_ieee_754(v)):
                                       EffectiveType(double_from_ieee_754(v));
        else if (elements_are_signed)
        {
            const T val = v & (element_mask >> 1);
            const T negative = v >> (element_bits - 1);
            return negative? val == 0? scalar_min(): -EffectiveType((~val + 1) & (element_mask >> 1)): EffectiveType(val);
        }
        return EffectiveType(v);
    }

public:
    /*
     * Default constructor zeros the vector
     */
    constexpr opencl_simd_vector() : data_{}, ctx_(CL_DEVICE_TYPE_ALL), buf_(ctx_, CL_MEM_USE_HOST_PTR, vector_digits / 8, data_) {}

    /*
     * Copy constructor copies the data and context, initializes new buffer
     */
    constexpr opencl_simd_vector(const opencl_simd_vector &other)
        : data_{}
        , ctx_(other.ctx_)
        , buf_(ctx_, CL_MEM_USE_HOST_PTR, vector_digits / 8, data_)
    {}

    /*
     * Returns a new representation of the vector, casted to a different type.
     * The data is not modified.
     */
    template<typename NewType>
    constexpr native_simd_vector<T, element_bits, NewType> cast() const {return native_simd_vector<T, element_bits, NewType>::make_vector(data_);}

    /*
     * Returns a new representation of the vector, casted to a different type with different element size.
     * The data is not modified.
     */
    template<unsigned int element_size, typename NewType>
    native_simd_vector<T, element_size, NewType> cast() const {return native_simd_vector<T, element_size, NewType>::make_vector(data_);}

    /*
     * Returns a new representation of the vector, casted to a different type with different element size, with a different underlying type.
     * The data is not modified.
     */
    template<typename UnderlyingType, unsigned int element_size, typename NewType>
    native_simd_vector<UnderlyingType, element_size, NewType> cast() const {return native_simd_vector<UnderlyingType, element_size, NewType>::make_vector(data_);}

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
    constexpr static type make_broadcast(EffectiveType value) {return type((make_t_from_effective(value) & element_mask) * expand_mask(1, element_bits, elements));}

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
    constexpr type operator|(generic_simd_vector vec) const
    {
        return type(data_ | vec.vector());
    }

    /*
     * Logical AND's the entire vector with `vec` and returns it
     */
    constexpr type operator&(generic_simd_vector vec) const
    {
        return type(data_ & vec.vector());
    }

    /*
     * Logical AND's the entire vector with negated `vec` and returns it
     */
    constexpr type and_not(generic_simd_vector vec) const
    {
        return type(data_ & ~vec.vector());
    }

    /*
     * Logical XOR's the entire vector with `vec` and returns it
     */
    constexpr type operator^(generic_simd_vector vec) const
    {
        return type(data_ ^ vec.vector());
    }

    /*
     * Adds all elements of `this` and returns the resulting sum
     */
    constexpr EffectiveType horizontal_sum() const
    {
        return horizontal_sum_helper(data_, elements);
    }

    /*
     * Adds elements of `vec` to `this` (using rollover addition) and returns the result
     */
    type operator+(generic_simd_vector vec) const {return add(vec, cppbits::math_keeplow);}

    /*
     * Adds elements of `vec` to `this` (using specified math method) and returns the result
     */
    type add(generic_simd_vector vec, cppbits::math_type math) const
    {
        if (elements_are_floats)
            return make_init_values(vec, [](EffectiveType a, EffectiveType b) {return a + b;});
        else
        {
            constexpr T add_mask = expand_mask(element_mask >> 1, element_bits, elements);
            constexpr T inverted_mask = ~add_mask & mask;
            switch (math) {
                default: /* Rollover arithmetic, math_keeplow */
                    /// Cost: 6 - Should compile down to three ANDs, one addition, and two XORs
                    return type(((data_ & add_mask) + (vec.vector() & add_mask)) ^ ((data_ ^ vec.vector()) & inverted_mask));
                case cppbits::math_saturate:
                    if (elements_are_signed)
                    {
                        /// Cost: 19 - Should compile down to seven ANDs, one addition, four XORs, two shifts, two subtractions, two multiplies, and one OR
                        const T temp = ((data_ & add_mask) + (vec.vector() & add_mask)) ^ ((data_ ^ vec.vector()) & inverted_mask);
                        const T overflow = ((data_ ^ temp) & (vec.vector() ^ temp) & inverted_mask) >> (element_bits - 1);
                        return type((temp & ((expand_mask(1, element_bits, elements) - overflow) * element_mask)) |
                                    (overflow * (element_mask ^ (element_mask >> 1)) - ((temp >> (element_bits - 1)) & overflow)));
                    }
                    else
                    {
                        /// Cost: 14 - Should compile down to six ANDs, one addition, three ORs, one shift, one multiply, and two XORs
                        const T temp = ((data_ & add_mask) + (vec.vector() & add_mask)) ^ ((data_ ^ vec.vector()) & inverted_mask);
                        return type(temp | ((((data_ & vec.vector()) | (~temp & (data_ | vec.vector()))) & inverted_mask) >> (element_bits - 1)) * element_mask);
                    }
                case cppbits::math_keephigh:
                {
                    constexpr T overflow_mask = expand_mask(1, element_bits, elements);
                    if (elements_are_signed)
                        /// Cost: 10 - Should compile down to five ANDs, one addition, one OR, one shift, one multiply, and one XOR
                        return type(((((data_ & vec.vector()) | (((data_ & add_mask) + (vec.vector() & add_mask)) & (data_ ^ vec.vector()))) >> (element_bits - 1)) & overflow_mask) * element_mask);
                    else
                        /// Cost: 9 - Should compile down to five ANDs, one addition, one OR, one shift, and one XOR
                        return type((((data_ & vec.vector()) | (((data_ & add_mask) + (vec.vector() & add_mask)) & (data_ ^ vec.vector()))) >> (element_bits - 1)) & overflow_mask);
                }
            }
        }
    }

    /*
     * Subtracts elements of `vec` to `this` (using rollover subtraction) and returns the result
     */
    type operator-(generic_simd_vector vec) const {return sub(vec, cppbits::math_keeplow);}

    /*
     * Subtracts elements of `vec` from `this` (using specified math method) and returns the result
     */
    type sub(generic_simd_vector vec, cppbits::math_type math) const
    {
        if (elements_are_floats)
            return make_init_values(vec, [](EffectiveType a, EffectiveType b) {return a - b;});
        else
        {
            constexpr T sub_mask = expand_mask(element_mask >> 1, element_bits, elements);
            constexpr T inverted_mask = ~sub_mask & mask;
            switch (math) {
                default: /* Rollover arithmetic, math_keeplow */
                    /// Cost: 7 - Should compile down to two ANDs, one subtraction, one OR, one NOT, and two XORs
                    return type(((data_ | inverted_mask) - (vec.vector() & sub_mask)) ^ ((data_ ^ ~vec.vector()) & inverted_mask));
                case cppbits::math_saturate:
                    if (elements_are_signed)
                    {
                        const T temp = ((data_ | inverted_mask) - (vec.vector() & sub_mask)) ^ ((data_ ^ ~vec.vector()) & inverted_mask);
                        const T overflow = ((data_ ^ temp) & ~(vec.vector() ^ temp) & inverted_mask) >> (element_bits - 1);
                        return type((temp & ((expand_mask(1, element_bits, elements) - overflow) * element_mask)) |
                                    (overflow * (element_mask ^ (element_mask >> 1)) - ((temp >> (element_bits - 1)) & overflow)));
                    }
                    else
                    {
                        const T temp = ((data_ | inverted_mask) - (vec.vector() & sub_mask)) ^ ((data_ ^ ~vec.vector()) & inverted_mask);
                        return type(temp & ((~((vec.vector() & temp) | (~data_ & (vec.vector() | temp))) & inverted_mask) >> (element_bits - 1)) * element_mask);
                    }
                case cppbits::math_keephigh: /* TODO: not accurate right now */
                    CPPBITS_ERROR("Subtraction with math_keephigh not implemented yet");
                    constexpr T overflow_mask = expand_mask(1, element_bits, elements);
                    if (elements_are_signed)
                        return type((((((data_ | inverted_mask) - (vec.vector() & sub_mask)) & (data_ ^ ~vec.vector())) >> (element_bits - 1)) & overflow_mask) * element_mask);
                    else
                        return type(((((data_ | inverted_mask) - (vec.vector() & sub_mask)) & (data_ ^ ~vec.vector())) >> (element_bits - 1)) & overflow_mask);
            }
        }
    }

    /*
     * Multiplies elements of `vec` by `this` (using rollover multiplication) and returns the result
     */
    type operator*(generic_simd_vector vec) const {return mul(vec, cppbits::math_keeplow);}

    /*
     * Multiplies elements of `vec` by `this` (using specified math method) and returns the result
     */
    type mul(generic_simd_vector vec, cppbits::math_type math) const
    {
        if (elements_are_floats)
            return make_init_values(vec, [](EffectiveType a, EffectiveType b) {return a * b;});
        else
        {
            switch (math)
            {
                default: /* Rollover arithmetic, math_keeplow */
                    return make_init_values(vec, [](EffectiveType a, EffectiveType b)
                    {
                        return a * b;
                    });
                case cppbits::math_saturate:
                    return make_init_values(vec, [](EffectiveType a, EffectiveType b)
                    {
                        /* TODO: may overflow, and EffectiveType may not be able to hold the double-size result */
                        const EffectiveType temp = a * b;

                        if (temp > scalar_max())
                            return scalar_max();
                        else if (elements_are_signed && temp < scalar_min())
                            return scalar_min();
                        else
                            return temp;
                    });
                case cppbits::math_keephigh:
                    CPPBITS_ERROR("Multiplication with math_keephigh not implemented yet");
            }
        }
    }

    /*
     * Multiplies elements of `vec` by `this` (using specified math method), adds `add`, and returns the result
     */
    constexpr type mul_add(generic_simd_vector vec, generic_simd_vector add, cppbits::math_type math) const
    {
        return mul(vec, math).add(add, math);
    }

    /*
     * Multiplies elements of `vec` by `this` (using specified math method), subtracts `sub`, and returns the result
     */
    constexpr type mul_sub(generic_simd_vector vec, generic_simd_vector sub, cppbits::math_type math) const
    {
        return mul(vec, math).sub(sub, math);
    }

    /*
     * Divides elements of `this` by `vec` (using rollover division) and returns the result
     */
    type operator/(generic_simd_vector vec) const {return div(vec, cppbits::math_keeplow);}

    /*
     * Divides elements of `this` by `vec` (using specified math method) and returns the result
     */
    type div(generic_simd_vector vec, cppbits::math_type math) const
    {
        if (elements_are_floats)
            return make_init_values(vec, [](EffectiveType a, EffectiveType b) {return a / b;});
        else if (vec.has_zero_element())
            CPPBITS_ERROR("Division by zero");
        else
        {
            switch (math)
            {
                default: /* Rollover arithmetic, math_keeplow, or math_saturate */
                    return make_init_values(vec, [](EffectiveType a, EffectiveType b)
                    {
                        return a / b;
                    });
                case cppbits::math_keephigh:
                    CPPBITS_ERROR("Division with math_keephigh not implemented yet");
            }
        }
    }

    /*
     * Computes the average of each element of `this` and `vec` as `((element of this) + (element of vec) + 1)/2` for integral values,
     * or `((element of this) + (element of vec))/2` for floating-point values
     * TODO: if element_bits == number of bits in T, undefined behavior results
     */
    type avg(generic_simd_vector vec) const
    {
        if (elements_are_floats)
            return make_init_values(vec, [](EffectiveType a, EffectiveType b) {return (a + b) * 0.5;});
        else
        {
            constexpr T ones_mask = expand_mask(1, element_bits * 2, (elements+1) / 2);
            constexpr T avg_mask = expand_mask(element_mask, element_bits * 2, (elements+1) / 2);
            return type((((ones_mask + ((data_ >> element_bits) & avg_mask) + ((vec.vector() >> element_bits) & avg_mask)) << (element_bits - 1)) & (avg_mask << element_bits)) |
                        (((ones_mask + (data_ & avg_mask) + (vec.vector() & avg_mask)) >> 1) & avg_mask));
        }
    }

    /*
     * Shifts each element of `this` to the left by `amount` and returns the resulting vector
     * If the invariant `0 <= amount <= element_bits` does not hold, undefined behavior results
     * TODO: support floating-point values
     */
    constexpr type operator<<(unsigned int amount) const {return shl(amount);}

    /*
     * Shifts each element of `this` to the left by `amount` and returns the resulting vector
     * If the invariant `0 <= amount <= element_bits` does not hold, undefined behavior results
     * TODO: if element_bits == number of bits in T, undefined behavior results
     * TODO: support floating-point values
     */
    constexpr type shl(unsigned int amount) const
    {
        return type((data_ << amount) & ((~expand_mask(1, element_bits, elements) & mask) * (element_mask >> (element_bits - amount))));
    }

    /*
     * Shifts each element of `this` to the right by `amount` (using shift_natural behavior) and returns the resulting vector
     * If the invariant `0 <= amount <= element_bits` does not hold, undefined behavior results
     * TODO: support floating-point values
     */
    constexpr type operator>>(unsigned int amount) const {return shr(amount, cppbits::shift_natural);}

    /*
     * Shifts each element of `this` to the right by `amount` (using specified shift type) and returns the resulting vector
     * If the invariant `0 <= amount <= element_bits` does not hold, undefined behavior results
     * TODO: if element_bits == number of bits in T, undefined behavior results
     * TODO: support floating-point values
     */
    type shr(unsigned int amount, cppbits::shift_type shift) const
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
            case cppbits::shift_arithmetic:
            {
                constexpr T shift_ones = expand_mask(1, element_bits, elements);
                const T shift_mask = element_mask >> amount;
                const T temp = (data_ >> amount) & (shift_mask * shift_ones);
                return type(temp | ((~shift_mask & element_mask) * ((data_ >> (element_bits - 1)) & shift_ones)));
            }
            case cppbits::shift_logical:
                return type((data_ >> amount) & (expand_mask(1, element_bits, elements) * (element_mask >> amount)));
        }
    }

    /*
     * Extracts MSB from each element and places them in the low bits of the result
     * Each bit position in the result corresponds to the element position in the source vector
     * (i.e. Element 0 MSB -> Bit 0, Element 1 MSB -> Bit 1, etc.)
     */
    T movmsk() const
    {
        if (element_bits == 1)
            return data_;
        else
        {
            T result = 0;
            for (unsigned i = 0; i < max_elements(); ++i)
                result |= ((data_ >> (i * element_bits + element_bits - 1)) & 1) << i;
            return result;
        }
    }

    /*
     * Negates elements of `this` and returns the resulting vector
     * Note that this function returns the value unmodified if EffectiveType is not a signed type
     * TODO: are floating-point values supported properly?
     */
    type negate() const
    {
        if (elements_are_floats)
        {
            constexpr T neg_mask = expand_mask(element_mask ^ (element_mask >> 1), element_bits, elements);
            return {data_ ^ neg_mask};
        }
        else if (elements_are_signed)
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
     * TODO: are floating-point values supported properly?
     */
    type abs() const
    {
        if (elements_are_floats)
        {
            constexpr T neg_mask = expand_mask(element_mask >> 1, element_bits, elements);
            return {data_ & neg_mask};
        }
        else if (elements_are_signed)
        {
            constexpr T abs_mask = expand_mask(element_mask >> 1, element_bits, elements);
            const T neg = (data_ & abs_mask) >> (element_bits - 1);
            return {((data_ ^ (neg * element_mask)) & abs_mask) + neg};
        }
        return *this;
    }

    /*
     * Computes the hypotenuse length (`sqrt(x^2 + y^2)`) and returns the resulting vector
     */
    constexpr type hypot(generic_simd_vector vec, cppbits::math_type math) const
    {
        return mul_add(*this, vec.mul(vec, math), math).sqrt(math);
    }

    /*
     * Sets each element of `this` to zero if the element is zero (sign of zero irrelevant in the case of floating-point), or all ones otherwise, and returns the resulting vector
     */
    type fill_if_nonzero() const
    {
        return type(fill_elements_if_nonzero(data_, element_bits, element_bits));
    }

    /*
     * Returns true if vector has at least one element equal to zero (sign of zero irrelevant in the case of floating-point), false otherwise
     * See <http://graphics.stanford.edu/~seander/bithacks.html#HasLessInWord>
     * This is a rewrite of hasless(u, 1)
     */
    bool has_zero_element() const
    {
        if (elements_are_floats)
        {
            bool has_zero = false;
            for (unsigned i = 0; i < max_elements(); ++i)
                has_zero |= get(i) == 0.0;
            return has_zero;
        }
        else
        {
            return element_bits == 1? data_ != mask: ((data_ - expand_mask(1, element_bits, elements)) &
                    (~data_ & expand_mask(element_mask ^ (element_mask >> 1), element_bits, elements))) != 0;
        }
    }

    /*
     * Returns number of zero elements in vector (sign of zero irrelevant in the case of floating-point)
     * See <http://graphics.stanford.edu/~seander/bithacks.html#HasLessInWord>
     * This is a rewrite of countless(u, 1)
     */
    unsigned int count_zero_elements() const
    {
        if (elements_are_floats || element_mask <= elements)
        {
            unsigned zeros = 0;
            for (unsigned i = 0; i < max_elements(); ++i)
                zeros += get(i) == 0;
            return zeros;
        }
        else
        {
            constexpr T test_mask = expand_mask(element_mask >> 1, element_bits, elements);
            constexpr T inverted_mask = ~test_mask & mask;
            return (((inverted_mask - (data_ & test_mask)) & ~data_ & inverted_mask) >> (element_bits - 1)) % element_mask;
        }
    }

    /*
     * Returns true if vector has at least one element equal to `v`, false otherwise
     * See <http://graphics.stanford.edu/~seander/bithacks.html#HasLessInWord>
     * This is a rewrite of hasless(u, 1)
     */
    bool has_equal_element(EffectiveType v) const
    {
        if (elements_are_floats)
        {
            bool has_equal = false;
            for (unsigned i = 0; i < max_elements(); ++i)
                has_equal |= get(i) == v;
            return has_equal;
        }
        else
        {
            return (*this ^ make_broadcast(v)).has_zero_element();
        }
    }

    /*
     * Returns number of elements in vector equal to `v`
     * See <http://graphics.stanford.edu/~seander/bithacks.html#HasLessInWord>
     * This is a rewrite of countless(u, 1)
     */
    unsigned int count_equal_elements(EffectiveType v) const
    {
        if (elements_are_floats || element_mask <= elements)
        {
            unsigned equals = 0;
            for (unsigned i = 0; i < max_elements(); ++i)
                equals += get(i) == v;
            return equals;
        }
        else
        {
            return (*this ^ make_broadcast(v)).count_zero_elements();
        }
    }

    /*
     * Compares `this` to `vec`. If the comparison result is true, the corresponding element is set to all 1's
     * Otherwise, if the comparison result is false, the corresponding element is set to all 0's
     * TODO: verify comparisons for floating-point values
     */
    type cmp(generic_simd_vector vec, cppbits::compare_type compare) const
    {
        switch (compare) {
            default: return make_init_cmp_values(vec, [](EffectiveType a, EffectiveType b) {return a == b;});
            case cppbits::compare_nequal: return make_init_cmp_values(vec, [](EffectiveType a, EffectiveType b) {return a != b;});
            case cppbits::compare_less: return make_init_cmp_values(vec, [](EffectiveType a, EffectiveType b) {return a < b;});
            case cppbits::compare_lessequal: return make_init_cmp_values(vec, [](EffectiveType a, EffectiveType b) {return a <= b;});
            case cppbits::compare_greater: return make_init_cmp_values(vec, [](EffectiveType a, EffectiveType b) {return a > b;});
            case cppbits::compare_greaterequal: return make_init_cmp_values(vec, [](EffectiveType a, EffectiveType b) {return a >= b;});
        }
    }

    type operator==(generic_simd_vector vec) const {return cmp(vec, cppbits::compare_equal);}
    type operator!=(generic_simd_vector vec) const {return cmp(vec, cppbits::compare_nequal);}
    type operator<(generic_simd_vector vec) const {return cmp(vec, cppbits::compare_less);}
    type operator<=(generic_simd_vector vec) const {return cmp(vec, cppbits::compare_lessequal);}
    type operator>(generic_simd_vector vec) const {return cmp(vec, cppbits::compare_greater);}
    type operator>=(generic_simd_vector vec) const {return cmp(vec, cppbits::compare_greaterequal);}

    /*
     * Sets each element in output to reciprocal of respective element in `this`
     * TODO: doesn't work properly for integral values
     */
    type reciprocal(cppbits::math_type) const
    {
        if (elements_are_floats)
            return make_init_value([](EffectiveType a) {return 1.0 / a;});
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
    type sqrt(cppbits::math_type) const
    {
        if (elements_are_floats)
            return make_init_value([](EffectiveType a) {return std::sqrt(a);});
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
    type rsqrt(cppbits::math_type) const
    {
        if (elements_are_floats)
            return make_init_value([](EffectiveType a) {return 1.0 / std::sqrt(a);});
        else
        {
            CPPBITS_ERROR("Reciprocal of square-root requested for integral vector");
            return *this;
        }
    }

    /*
     * Sets each element in output to maximum of respective elements of `this` and `vec`
     * TODO: verify comparisons for floating-point values
     */
    type max(generic_simd_vector vec) const
    {
        return make_init_values(vec, [](EffectiveType a, EffectiveType b) {using namespace std; return max(a, b);});
    }

    /*
     * Sets each element in output to minimum of respective elements of `this` and `vec`
     * TODO: verify comparisons for floating-point values
     */
    type min(generic_simd_vector vec) const
    {
        return make_init_values(vec, [](EffectiveType a, EffectiveType b) {using namespace std; return min(a, b);});
    }

    /*
     * Sets element `idx` to `value`
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
    template<unsigned int idx>
    type &set()
    {
        const unsigned int shift = idx * element_bits;
        data_ |= (element_mask << shift);
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
     * Sets all bits of element `idx` to 1 if `v` is true, 0 otherwise
     */
    template<unsigned int idx>
    type &set_bits(bool v)
    {
        const unsigned int shift = idx * element_bits;
        data_ = (data_ & ~(element_mask << shift)) | ((v * element_mask) << shift);
        return *this;
    }

    /*
     * Sets all bits of element `idx` to 1 if `v` is true, 0 otherwise
     */
    type &set_bits(unsigned int idx, bool v)
    {
        const unsigned int shift = idx * element_bits;
        data_ = (data_ & ~(element_mask << shift)) | ((v * element_mask) << shift);
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
    constexpr EffectiveType get() const
    {
        return make_effective_from_t((data_ >> (idx * element_bits)) & element_mask);
    }

    /*
     * Gets value of element `idx`
     */
    constexpr EffectiveType get(unsigned int idx) const
    {
        return make_effective_from_t((data_ >> (idx * element_bits)) & element_mask);
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

    /* Dumps vector to memory location (non-portable, so don't use for saving values permanently unless they're read with the same computing configuration) */
    void dump_packed(underlying_vector_type *mem)
    {
        *mem = data_;
    }

    /* Reads vector from memory location (non-portable, so don't use for saving values permanently unless they're read with the same computing configuration) */
    type &load_packed(const underlying_vector_type *mem)
    {
        data_ = *mem & mask;
        return *this;
    }

    /* Dumps vector to array (with an identical number of elements to this vector) of EffectiveType values
     * Undefined behavior results if array is not large enough to contain this vector
     * Element 0 is placed in the first position
     */
    void dump_unpacked(EffectiveType *array)
    {
        for (unsigned i = 0; i < max_elements(); ++i)
            *array++ = get(i);
    }

    /* Loads vector from array (with an identical number of elements to this vector) of EffectiveType values
     * Undefined behavior results if array is not large enough to contain this vector
     * Element 0 is read from the first position
     */
    type &load_unpacked(const EffectiveType *array)
    {
        for (unsigned i = 0; i < max_elements(); ++i)
            set(i, *array++);
        return *this;
    }

    /* Dumps vector to aligned array (with an identical number of elements to this vector) of EffectiveType values
     * Undefined behavior results if array is not large enough to contain this vector
     * Element 0 is placed in the first position
     */
    void dump_unpacked_aligned(EffectiveType *array)
    {
        dump_unpacked(array);
    }

    /* Loads vector from aligned array (with an identical number of elements to this vector) of EffectiveType values
     * Undefined behavior results if array is not large enough to contain this vector
     * Element 0 is read from the first position
     */
    type &load_unpacked_aligned(const EffectiveType *array)
    {
        return load_unpacked(array);
    }

    /* Determines if aligned accesses are possible with specified pointer (does nothing with generic_simd_vector) */
    static constexpr bool ptr_is_aligned(EffectiveType *)
    {
        return true;
    }

private:
    // Lambda should be a functor taking an EffectiveType and returning an EffectiveType value
    template<typename Lambda>
    type make_init_value(Lambda callable) const
    {
        T result = 0;
        constexpr unsigned int shift = element_bits % vector_digits;

        for (unsigned i = 0; i < max_elements(); ++i)
            result |= (make_t_from_effective(callable(get(i))) & element_mask) << (i * shift);

        return type(result);
    }

    // Lambda should be a functor taking 2 EffectiveType values and returning an EffectiveType value
    template<typename Lambda>
    type make_init_values(generic_simd_vector vec, Lambda callable) const
    {
        T result = 0;
        constexpr unsigned int shift = element_bits % vector_digits;

        for (unsigned i = 0; i < max_elements(); ++i)
            result |= (make_t_from_effective(callable(get(i), vec.get(i))) & element_mask) << (i * shift);

        return type(result);
    }

    // Lambda should be a functor taking 2 EffectiveType values and returning a boolean value
    template<typename Lambda>
    type make_init_cmp_values(generic_simd_vector vec, Lambda callable) const
    {
        T result = 0;
        constexpr unsigned int shift = element_bits % vector_digits;

        for (unsigned i = 0; i < max_elements(); ++i)
            result |= (callable(get(i), vec.get(i)) * element_mask) << (i * shift);

        return type(result);
    }

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
    constexpr EffectiveType horizontal_sum_helper(T vector, unsigned int level) const
    {
        return level == 0? EffectiveType(0): EffectiveType(type(vector).get(level-1) + horizontal_sum_helper(vector, level-1));
    }

    underlying_element_type data_[desired_elements];
    underlying_vector_type buf_;
    cl::Context ctx_;
};

template<typename T, unsigned int element_bits, typename EffectiveType>
struct native_simd_vector : public generic_simd_vector<T, element_bits, EffectiveType>
{
    native_simd_vector() {}
    native_simd_vector(generic_simd_vector<T, element_bits, EffectiveType> v)
        : generic_simd_vector<T, element_bits, EffectiveType>(v)
    {}
};
}

#endif // OPENCL_SIMD_H
