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

#ifndef CPPBITS_H
#define CPPBITS_H

#include "simd/generic_simd.h"
#include "simd/x86_simd.h"

template<unsigned int desired_elements, typename T, unsigned int element_bits, typename EffectiveType>
class simd_vector
{
    typedef native_simd_vector<T, element_bits, EffectiveType> native_vector;
    typedef simd_vector<desired_elements, T, element_bits, EffectiveType> type;
    static constexpr unsigned int native_vector_count = desired_elements? (desired_elements + native_vector::max_elements() - 1) / native_vector::max_elements(): 1;

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
     * Computes the average of each element of `this` and `vec` as `((element of this) + (element of vec) + 1)/2` for integral values,
     * or `((element of this) + (element of vec))/2` for floating-point values
     */
    type avg(type vec) const
    {
        type result;
        for (unsigned i = 0; i < native_vector_count; ++i)
            result.array_[i] = array_[i].avg(vec);
        return result;
    }

    /*
     * Shifts each element of `this` to the left by `amount` and returns the resulting vector
     * If the invariant `0 <= amount <= element_bits` does not hold, undefined behavior results
     */
    constexpr type operator<<(unsigned int amount) const {return shl(amount);}

    /*
     * Shifts each element of `this` to the left by `amount` and returns the resulting vector
     * If the invariant `0 <= amount <= element_bits` does not hold, undefined behavior results
     */
    constexpr type shl(unsigned int amount) const
    {
        type result;
        for (unsigned i = 0; i < native_vector_count; ++i)
            result.array_[i] = array_[i].shl(amount);
        return result;
    }

    /*
     * Shifts each element of `this` to the right by `amount` (using shift_natural behavior) and returns the resulting vector
     * If the invariant `0 <= amount <= element_bits` does not hold, undefined behavior results
     */
    constexpr type operator>>(unsigned int amount) const {return shr(amount, shift_natural);}

    /*
     * Shifts each element of `this` to the right by `amount` (using specified shift type) and returns the resulting vector
     * If the invariant `0 <= amount <= element_bits` does not hold, undefined behavior results
     */
    type shr(unsigned int amount, shift_type shift) const
    {
        type result;
        for (unsigned i = 0; i < native_vector_count; ++i)
            result.array_[i] = array_[i].shr(amount, shift);
        return result;
    }

    /*
     * Negates elements of `this` and returns the resulting vector
     * Note that this function returns the value unmodified if EffectiveType is not a signed type
     */
    type negate() const
    {
        type result;
        for (unsigned i = 0; i < native_vector_count; ++i)
            result.array_[i] = array_[i].negate();
        return result;
    }

    /*
     * Computes the absolute value of elements of `this` and returns the resulting vector
     * Note that this function returns the value unmodified if EffectiveType is not a signed type
     */
    type abs() const
    {
        type result;
        for (unsigned i = 0; i < native_vector_count; ++i)
            result.array_[i] = array_[i].abs();
        return result;
    }

    /*
     * Sets each element of `this` to zero if the element is zero, or all ones otherwise, and returns the resulting vector
     */
    type fill_if_nonzero() const
    {
        type result;
        for (unsigned i = 0; i < native_vector_count; ++i)
            result.array_[i] = array_[i].fill_if_nonzero();
        return result;
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
        unsigned int result = 0;
        for (unsigned i = 0; i < native_vector_count; ++i)
            result += array_[i].count_zero_elements();
        return result;
    }

    /*
     * Returns true if vector has at least one element equal to `v` (sign of zero irrelevant in the case of floating-point), false otherwise
     */
    constexpr bool has_equal_element(EffectiveType v) const
    {
        return count_equal_elements(v) != 0;
    }

    /*
     * Returns number of elements equal to v in vector (sign of zero irrelevant in the case of floating-point)
     */
    unsigned int count_equal_elements(EffectiveType v) const
    {
        unsigned int result = 0;
        for (unsigned i = 0; i < native_vector_count; ++i)
            result += array_[i].count_equal_elements(v);
        return result;
    }

    /*
     * Compares `this` to `vec`. If the comparison result is true, the corresponding element is set to all 1's
     * Otherwise, if the comparison result is false, the corresponding element is set to all 0's
     */
    type cmp(type vec, compare_type compare) const
    {
        type result;
        for (unsigned i = 0; i < native_vector_count; ++i)
            result.array_[i] = array_[i].cmp(vec.array_[i], compare);
        return result;
    }

    type operator==(type vec) const {return cmp(vec, compare_equal);}
    type operator!=(type vec) const {return cmp(vec, compare_nequal);}
    type operator<(type vec) const {return cmp(vec, compare_less);}
    type operator<=(type vec) const {return cmp(vec, compare_lessequal);}
    type operator>(type vec) const {return cmp(vec, compare_greater);}
    type operator>=(type vec) const {return cmp(vec, compare_greaterequal);}

    /*
     * Sets each element in output to reciprocal of respective element in `this`
     */
    type reciprocal(math_type math) const
    {
        type result;
        for (unsigned i = 0; i < native_vector_count; ++i)
            result.array_[i] = array_[i].reciprocal(math);
        return result;
    }

    /*
     * Sets each element in output to square-root of respective element in `this`
     */
    type sqrt(math_type math) const
    {
        type result;
        for (unsigned i = 0; i < native_vector_count; ++i)
            result.array_[i] = array_[i].sqrt(math);
        return result;
    }

    /*
     * Sets each element in output to reciprocal of square-root of respective element in `this`
     */
    type rsqrt(math_type math) const
    {
        type result;
        for (unsigned i = 0; i < native_vector_count; ++i)
            result.array_[i] = array_[i].rsqrt(math);
        return result;
    }

    /*
     * Sets each element in output to maximum of respective elements of `this` and `vec`
     */
    type max(type vec) const
    {
        type result;
        for (unsigned i = 0; i < native_vector_count; ++i)
            result.array_[i] = array_[i].max(vec.array_[i]);
        return result;
    }

    /*
     * Sets each element in output to minimum of respective elements of `this` and `vec`
     */
    type min(type vec) const
    {
        type result;
        for (unsigned i = 0; i < native_vector_count; ++i)
            result.array_[i] = array_[i].min(vec.array_[i]);
        return result;
    }

    /*
     * Sets element `idx` to `value`
     */
    template<unsigned int idx>
    type &set(EffectiveType value)
    {
        array_[idx / native_vector::max_elements()].set<idx % native_vector::max_elements()>(value);
        return *this;
    }

    /*
     * Sets element `idx` to `value`
     */
    type &set(unsigned int idx, EffectiveType value)
    {
        array_[idx / native_vector::max_elements()].set(idx % native_vector::max_elements(), value);
        return *this;
    }

    /*
     * Sets all bits of element `idx` to 1
     */
    template<unsigned int idx>
    type &set()
    {
        array_[idx / native_vector::max_elements()].set<idx % native_vector::max_elements()>();
        return *this;
    }

    /*
     * Sets all bits of element `idx` to 1
     */
    type &set(unsigned int idx)
    {
        array_[idx / native_vector::max_elements()].set(idx % native_vector::max_elements());
        return *this;
    }

    /*
     * Sets all bits of element `idx` to 1 if `v` is true, 0 otherwise
     */
    template<unsigned int idx>
    type &set_bits(bool v)
    {
        array_[idx / native_vector::max_elements()].set_bits<idx % native_vector::max_elements()>(v);
        return *this;
    }

    /*
     * Sets all bits of element `idx` to 1 if `v` is true, 0 otherwise
     */
    type &set_bits(unsigned int idx, bool v)
    {
        array_[idx / native_vector::max_elements()].set_bits(idx % native_vector::max_elements(), v);
        return *this;
    }

    /*
     * Sets all bits of element `idx` to 0
     */
    template<unsigned int idx>
    type &reset()
    {
        array_[idx / native_vector::max_elements()].reset<idx % native_vector::max_elements()>();
        return *this;
    }

    /*
     * Sets all bits of element `idx` to 0
     */
    type &reset(unsigned int idx)
    {
        array_[idx / native_vector::max_elements()].reset(idx % native_vector::max_elements());
        return *this;
    }

    /*
     * Flips all bits of element `idx`
     */
    template<unsigned int idx>
    type &flip()
    {
        array_[idx / native_vector::max_elements()].flip<idx % native_vector::max_elements()>();
        return *this;
    }

    /*
     * Flips all bits of element `idx`
     */
    type &flip(unsigned int idx)
    {
        array_[idx / native_vector::max_elements()].flip(idx % native_vector::max_elements());
        return *this;
    }

    /*
     * Gets value of element `idx`
     */
    template<unsigned int idx>
    constexpr EffectiveType get() const
    {
        return array_[idx / native_vector::max_elements()].get<idx % native_vector::max_elements()>();
    }

    /*
     * Gets value of element `idx`
     */
    constexpr EffectiveType get(unsigned int idx) const
    {
        return array_[idx / native_vector::max_elements()].get(idx % native_vector::max_elements());
    }

    /*
     * Returns minumum value of vector
     */
    static constexpr type min() {return {};}

    /*
     * Returns maximum value of vector
     */
    static constexpr type max() {return ~min();}

    /*
     * Returns unsigned mask that can contain all element values
     */
    static constexpr typename native_vector::underlying_element_type scalar_mask() {return native_vector::scalar_mask();}

    /*
     * Returns the minimum value an element can contain
     */
    static constexpr EffectiveType scalar_min() {return native_vector::scalar_min();}

    /*
     * Returns the maximum value an element can contain
     */
    static constexpr EffectiveType scalar_max() {return native_vector::scalar_max();}

    /*
     * Returns maximum value of vector
     */
    static constexpr type vector_mask() {return max();}

    /*
     * Returns whether elements in this vector are viewed as signed values
     */
    static constexpr bool is_signed() {return native_vector::is_signed();}

    /*
     * Returns the maximum number of elements this vector can contain
     */
    static constexpr unsigned int max_elements() {return native_vector_count * native_vector::max_elements();}

    /*
     * Returns the size in bits of each element
     */
    static constexpr unsigned int element_size() {return native_vector::element_size();}

    /* Reads vector from memory location (non-portable, so don't use for saving values permanently unless they're read with the same computing configuration) */
    type &load_unpacked(const EffectiveType *array)
    {
        for (unsigned i = 0; i < native_vector_count; ++i)
            array_[i].load_unpacked(array + i * native_vector::max_elements());
        return *this;
    }

    /* Reads vector from memory location (non-portable, so don't use for saving values permanently unless they're read with the same computing configuration) */
    type &load_unpacked_aligned(const EffectiveType *array)
    {
        for (unsigned i = 0; i < native_vector_count; ++i)
            array_[i].load_unpacked_aligned(array + i * native_vector::max_elements());
        return *this;
    }

    /* Dumps vector to memory location (non-portable, so don't use for saving values permanently unless they're read with the same computing configuration) */
    void dump_unpacked(EffectiveType *array)
    {
        for (unsigned i = 0; i < native_vector_count; ++i)
            array_[i].dump_unpacked(array + i * native_vector::max_elements());
    }

    /* Dumps vector to aligned memory location (non-portable, so don't use for saving values permanently unless they're read with the same computing configuration) */
    void dump_unpacked_aligned(EffectiveType *array)
    {
        for (unsigned i = 0; i < native_vector_count; ++i, array += native_vector::max_elements())
            array_[i].dump_unpacked_aligned(array + i * native_vector::max_elements());
    }

private:
    native_vector array_[native_vector_count];
};

template<typename T, unsigned int bits, typename EffectiveType>
std::ostream &operator<<(std::ostream &os, native_simd_vector<T, bits, EffectiveType> vec)
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

#endif // CPPBITS_H
