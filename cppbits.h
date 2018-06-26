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
#include "simd/opencl_simd.h"

namespace cppbits {
template<unsigned int desired_elements, template<typename, unsigned int, typename> class native_vector_type, typename T, unsigned int element_bits, typename EffectiveType>
class simd_vector
{
    typedef native_vector_type<T, element_bits, EffectiveType> native_vector;
    typedef simd_vector<desired_elements, native_vector_type, T, element_bits, EffectiveType> type;
    static constexpr unsigned int native_vector_count = desired_elements? (desired_elements + native_vector::max_elements() - 1) / native_vector::max_elements(): 1;

public:
    /*
     * Returns a new representation of the vector, casted to a different type.
     * The data is not modified.
     */
    template<typename NewType>
    simd_vector<desired_elements, native_vector_type, T, element_bits, NewType> cast() const
    {
        simd_vector<desired_elements, native_vector_type, T, element_bits, NewType> result;
        for (unsigned i = 0; i < native_vector_count; ++i)
            result.array_[i] = array_[i].template cast<NewType>();
        return result;
    }

    /*
     * Returns a new representation of the vector, casted to a different type with different element size.
     * The data is not modified.
     * TODO: needs work. Changing element sizes across the entire vector is a tricky endeavor.
     */
    template<unsigned int element_size, typename NewType>
    simd_vector<desired_elements, native_vector_type, T, element_size, NewType> cast() const
    {
        simd_vector<desired_elements, native_vector_type, T, element_size, NewType> result;
        for (unsigned i = 0; i < std::min(native_vector_count, result.native_vector_count); ++i)
            result.array_[i] = array_[i].template cast<element_size, NewType>();
        return result;
    }

    /*
     * Returns a representation of a vector with specified value assigned to element 0
     */
    static type make_scalar(EffectiveType value)
    {
        type result;
        result.array_[0].scalar(value);
        return result;
    }

    /*
     * Returns a representation of a vector with specified value assigned to every element in the vector
     */
    static type make_broadcast(EffectiveType value)
    {
        type result;
        for (unsigned i = 0; i < native_vector_count; ++i)
            result.array_[i].broadcast(value);
        return result;
    }

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
     * Returns the value of element 0
     */
    EffectiveType scalar() const {return get<0>();}

    /*
     * Logical NOT's the entire vector and returns it
     */
    type operator~() const
    {
        type result;
        for (unsigned i = 0; i < native_vector_count; ++i)
            result.array_[i] = ~array_[i];
        return result;
    }

    /*
     * Logical OR's the entire vector with `vec` and returns it
     */
    type operator|(type vec) const
    {
        type result;
        for (unsigned i = 0; i < native_vector_count; ++i)
            result.array_[i] = array_[i] | vec;
        return result;
    }

    /*
     * Logical AND's the entire vector with `vec` and returns it
     */
    type operator&(type vec) const
    {
        type result;
        for (unsigned i = 0; i < native_vector_count; ++i)
            result.array_[i] = array_[i] & vec;
        return result;
    }

    /*
     * Logical AND's the entire vector with negated `vec` and returns it
     */
    type and_not(type vec) const
    {
        type result;
        for (unsigned i = 0; i < native_vector_count; ++i)
            result.array_[i] = array_[i].and_not(vec);
        return result;
    }

    /*
     * Logical XOR's the entire vector with `vec` and returns it
     */
    type operator^(type vec) const
    {
        type result;
        for (unsigned i = 0; i < native_vector_count; ++i)
            result.array_[i] = array_[i] ^ vec;
        return result;
    }

    /*
     * Adds all elements of `this` and returns the resulting sum
     */
    EffectiveType horizontal_sum() const
    {
        EffectiveType result = 0;
        for (unsigned i = 0; i < native_vector_count; ++i)
            result += array_[i].horizontal_sum();
        return result;
    }

    /*
     * Adds elements of `vec` to `this` (using rollover addition) and returns the result
     */
    type operator+(type vec) const {return add(vec, cppbits::math_keeplow);}

    /*
     * Adds elements of `vec` to `this` (using specified math method) and returns the result
     */
    type add(type vec, cppbits::math_type math) const
    {
        type result;
        for (unsigned i = 0; i < native_vector_count; ++i)
            result.array_[i] = array_[i].add(vec.array_[i], math);
        return result;
    }

    /*
     * Subtracts elements of `vec` to `this` (using rollover subtraction) and returns the result
     */
    type operator-(type vec) const {return sub(vec, cppbits::math_keeplow);}

    /*
     * Subtracts elements of `vec` from `this` (using specified math method) and returns the result
     */
    type sub(type vec, cppbits::math_type math) const
    {
        type result;
        for (unsigned i = 0; i < native_vector_count; ++i)
            result.array_[i] = array_[i].sub(vec.array_[i], math);
        return result;
    }

    /*
     * Multiplies elements of `vec` by `this` (using rollover multiplication) and returns the result
     */
    type operator*(type vec) const {return mul(vec, cppbits::math_keeplow);}

    /*
     * Multiplies elements of `vec` by `this` (using specified math method) and returns the result
     */
    type mul(type vec, cppbits::math_type math) const
    {
        type result;
        for (unsigned i = 0; i < native_vector_count; ++i)
            result.array_[i] = array_[i].mul(vec.array_[i], math);
        return result;
    }

    /*
     * Multiplies elements of `vec` by `this` (using specified math method), adds `add`, and returns the result
     */
    type mul_add(type vec, type add, cppbits::math_type math) const
    {
        type result;
        for (unsigned i = 0; i < native_vector_count; ++i)
            result.array_[i] = array_[i].mul_add(vec.array_[i], add.array_[i], math);
        return result;
    }

    /*
     * Multiplies elements of `vec` by `this` (using specified math method), subtracts `sub`, and returns the result
     */
    type mul_sub(type vec, type sub, cppbits::math_type math) const
    {
        type result;
        for (unsigned i = 0; i < native_vector_count; ++i)
            result.array_[i] = array_[i].mul_sub(vec.array_[i], sub.array_[i], math);
        return result;
    }

    /*
     * Divides elements of `this` by `vec` (using rollover division) and returns the result
     */
    type operator/(type vec) const {return div(vec, cppbits::math_keeplow);}

    /*
     * Divides elements of `this` by `vec` (using specified math method) and returns the result
     */
    type div(type vec, cppbits::math_type math) const
    {
        type result;
        for (unsigned i = 0; i < native_vector_count; ++i)
            result.array_[i] = array_[i].div(vec.array_[i], math);
        return result;
    }

    /*
     * Computes the average of each element of `this` and `vec` as `((element of this) + (element of vec) + 1)/2` for integral values,
     * or `((element of this) + (element of vec))/2` for floating-point values
     */
    type avg(type vec) const
    {
        type result;
        for (unsigned i = 0; i < native_vector_count; ++i)
            result.array_[i] = array_[i].avg(vec.array_[i]);
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
    type shl(unsigned int amount) const
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
    constexpr type operator>>(unsigned int amount) const {return shr(amount, cppbits::shift_natural);}

    /*
     * Shifts each element of `this` to the right by `amount` (using specified shift type) and returns the resulting vector
     * If the invariant `0 <= amount <= element_bits` does not hold, undefined behavior results
     */
    type shr(unsigned int amount, cppbits::shift_type shift) const
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
     * Computes the hypotenuse length (`sqrt(x^2 + y^2)`) and returns the resulting vector
     */
    type hypot(type vec, cppbits::math_type math) const
    {
        type result;
        for (unsigned i = 0; i < native_vector_count; ++i)
            result.array_[i] = array_[i].mul_add(array_[i], vec.array_[i].mul(vec.array_[i], math), math).sqrt(math);
        return result;
    }

    /*
     * Sets each element of `this` to zero if the element is zero (sign of zero irrelevant in the case of floating-point), or all ones otherwise, and returns the resulting vector
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
    type cmp(type vec, cppbits::compare_type compare) const
    {
        type result;
        for (unsigned i = 0; i < native_vector_count; ++i)
            result.array_[i] = array_[i].cmp(vec.array_[i], compare);
        return result;
    }

    type operator==(type vec) const {return cmp(vec, cppbits::compare_equal);}
    type operator!=(type vec) const {return cmp(vec, cppbits::compare_nequal);}
    type operator<(type vec) const {return cmp(vec, cppbits::compare_less);}
    type operator<=(type vec) const {return cmp(vec, cppbits::compare_lessequal);}
    type operator>(type vec) const {return cmp(vec, cppbits::compare_greater);}
    type operator>=(type vec) const {return cmp(vec, cppbits::compare_greaterequal);}

    /*
     * Sets each element in output to reciprocal of respective element in `this`
     */
    type reciprocal(cppbits::math_type math) const
    {
        type result;
        for (unsigned i = 0; i < native_vector_count; ++i)
            result.array_[i] = array_[i].reciprocal(math);
        return result;
    }

    /*
     * Sets each element in output to square-root of respective element in `this`
     */
    type sqrt(cppbits::math_type math) const
    {
        type result;
        for (unsigned i = 0; i < native_vector_count; ++i)
            result.array_[i] = array_[i].sqrt(math);
        return result;
    }

    /*
     * Sets each element in output to reciprocal of square-root of respective element in `this`
     */
    type rsqrt(cppbits::math_type math) const
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
        array_[idx / native_vector::max_elements()].template set<idx % native_vector::max_elements()>();
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
        array_[idx / native_vector::max_elements()].template set_bits<idx % native_vector::max_elements()>(v);
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
        array_[idx / native_vector::max_elements()].template reset<idx % native_vector::max_elements()>();
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
        array_[idx / native_vector::max_elements()].template flip<idx % native_vector::max_elements()>();
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
        return array_[idx / native_vector::max_elements()].template get<idx % native_vector::max_elements()>();
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
     * Returns the desired number of elements specified
     */
    static constexpr unsigned int specified_elements() {return desired_elements;}

    /*
     * Returns the size in bits of each element
     */
    static constexpr unsigned int element_size() {return native_vector::element_size();}

    /* Dumps vector to memory location (non-portable, so don't use for saving values permanently unless they're read with the same computing configuration) */
    void dump_packed(typename native_vector::underlying_vector_type *mem)
    {
        for (unsigned i = 0; i < native_vector_count; ++i)
            array_[i].dump_packed(mem + i);
    }

    /* Reads vector from memory location (non-portable, so don't use for saving values permanently unless they're read with the same computing configuration) */
    type &load_packed(const typename native_vector::underlying_vector_type *mem)
    {
        for (unsigned i = 0; i < native_vector_count; ++i)
            array_[i].load_packed(mem + i);
        return *this;
    }

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

    /* Determines if aligned accesses are possible with specified pointer (depends on underlying vector's alignment requirements) */
    static constexpr bool ptr_is_aligned(EffectiveType *ptr)
    {
        return native_vector::ptr_is_aligned(ptr);
    }

private:
    native_vector array_[native_vector_count];
};

template<unsigned int desired_elements, typename T, unsigned int element_bits, typename EffectiveType>
class default_simd_vector : public simd_vector<desired_elements, native_simd_vector, T, element_bits, EffectiveType>
{
    typedef simd_vector<desired_elements, native_simd_vector, T, element_bits, EffectiveType> type;

public:
    default_simd_vector() : type() {}
    default_simd_vector(type v) : type(v) {}

    default_simd_vector &operator=(type v) {type::operator=(v); return *this;}
};

template<typename T, template<typename, unsigned int, typename> class underlying_vector = native_simd_vector>
class vector2 : public simd_vector<2, underlying_vector, uint64_t, sizeof(T) * CHAR_BIT, T>
{
    typedef simd_vector<2, underlying_vector, uint64_t, sizeof(T) * CHAR_BIT, T> type;

public:
    vector2() : type() {}
    vector2(type v) : type(v) {}

    vector2 &operator=(type v) {type::operator=(v); return *this;}
};

typedef vector2<int8_t> vector2x8;
typedef vector2<int16_t> vector2x16;
typedef vector2<int32_t> vector2x32;
typedef vector2<int64_t> vector2x64;
typedef vector2<uint8_t> vector2xu8;
typedef vector2<uint16_t> vector2xu16;
typedef vector2<uint32_t> vector2xu32;
typedef vector2<uint64_t> vector2xu64;
typedef vector2<float> vector2f;
typedef vector2<double> vector2d;

template<typename T, template<typename, unsigned int, typename> class underlying_vector = native_simd_vector>
class vector3 : public simd_vector<3, underlying_vector, uint64_t, sizeof(T) * CHAR_BIT, T>
{
    typedef simd_vector<3, underlying_vector, uint64_t, sizeof(T) * CHAR_BIT, T> type;

public:
    vector3() : type() {}
    vector3(type v) : type(v) {}

    vector3 &operator=(type v) {type::operator=(v); return *this;}
};

typedef vector3<int8_t> vector3x8;
typedef vector3<int16_t> vector3x16;
typedef vector3<int32_t> vector3x32;
typedef vector3<int64_t> vector3x64;
typedef vector3<uint8_t> vector3xu8;
typedef vector3<uint16_t> vector3xu16;
typedef vector3<uint32_t> vector3xu32;
typedef vector3<uint64_t> vector3xu64;
typedef vector3<float> vector3f;
typedef vector3<double> vector3d;

template<typename T, template<typename, unsigned int, typename> class underlying_vector = native_simd_vector>
class vector4 : public simd_vector<4, underlying_vector, uint64_t, sizeof(T) * CHAR_BIT, T>
{
    typedef simd_vector<4, underlying_vector, uint64_t, sizeof(T) * CHAR_BIT, T> type;

public:
    vector4() : type() {}
    vector4(type v) : type(v) {}

    vector4 &operator=(type v) {type::operator=(v); return *this;}
};

typedef vector4<int8_t> vector4x8;
typedef vector4<int16_t> vector4x16;
typedef vector4<int32_t> vector4x32;
typedef vector4<int64_t> vector4x64;
typedef vector4<uint8_t> vector4xu8;
typedef vector4<uint16_t> vector4xu16;
typedef vector4<uint32_t> vector4xu32;
typedef vector4<uint64_t> vector4xu64;
typedef vector4<float> vector4f;
typedef vector4<double> vector4d;

template<typename T, template<typename, unsigned int, typename> class underlying_vector = native_simd_vector>
class vector8 : public simd_vector<8, underlying_vector, uint64_t, sizeof(T) * CHAR_BIT, T>
{
    typedef simd_vector<8, underlying_vector, uint64_t, sizeof(T) * CHAR_BIT, T> type;

public:
    vector8() : type() {}
    vector8(type v) : type(v) {}

    vector8 &operator=(type v) {type::operator=(v); return *this;}
};

typedef vector8<int8_t> vector8x8;
typedef vector8<int16_t> vector8x16;
typedef vector8<int32_t> vector8x32;
typedef vector8<int64_t> vector8x64;
typedef vector8<uint8_t> vector8xu8;
typedef vector8<uint16_t> vector8xu16;
typedef vector8<uint32_t> vector8xu32;
typedef vector8<uint64_t> vector8xu64;
typedef vector8<float> vector8f;
typedef vector8<double> vector8d;

template<typename T, template<typename, unsigned int, typename> class underlying_vector = native_simd_vector>
class vector16 : public simd_vector<16, underlying_vector, uint64_t, sizeof(T) * CHAR_BIT, T>
{
    typedef simd_vector<16, underlying_vector, uint64_t, sizeof(T) * CHAR_BIT, T> type;

public:
    vector16() : type() {}
    vector16(type v) : type(v) {}

    vector16 &operator=(type v) {type::operator=(v); return *this;}
};

typedef vector16<int8_t> vector16x8;
typedef vector16<int16_t> vector16x16;
typedef vector16<int32_t> vector16x32;
typedef vector16<int64_t> vector16x64;
typedef vector16<uint8_t> vector16xu8;
typedef vector16<uint16_t> vector16xu16;
typedef vector16<uint32_t> vector16xu32;
typedef vector16<uint64_t> vector16xu64;
typedef vector16<float> vector16f;
typedef vector16<double> vector16d;

template<typename T, template<typename, unsigned int, typename> class underlying_vector = native_simd_vector>
class vector32 : public simd_vector<32, underlying_vector, uint64_t, sizeof(T) * CHAR_BIT, T>
{
    typedef simd_vector<32, underlying_vector, uint64_t, sizeof(T) * CHAR_BIT, T> type;

public:
    vector32() : type() {}
    vector32(type v) : type(v) {}

    vector32 &operator=(type v) {type::operator=(v); return *this;}
};

typedef vector32<int8_t> vector32x8;
typedef vector32<int16_t> vector32x16;
typedef vector32<int32_t> vector32x32;
typedef vector32<int64_t> vector32x64;
typedef vector32<uint8_t> vector32xu8;
typedef vector32<uint16_t> vector32xu16;
typedef vector32<uint32_t> vector32xu32;
typedef vector32<uint64_t> vector32xu64;
typedef vector32<float> vector32f;
typedef vector32<double> vector32d;

template<typename T, template<typename, unsigned int, typename> class underlying_vector = native_simd_vector>
class vector64 : public simd_vector<64, underlying_vector, uint64_t, sizeof(T) * CHAR_BIT, T>
{
    typedef simd_vector<64, underlying_vector, uint64_t, sizeof(T) * CHAR_BIT, T> type;

public:
    vector64() : type() {}
    vector64(type v) : type(v) {}

    vector64 &operator=(type v) {type::operator=(v); return *this;}
};

typedef vector64<int8_t> vector64x8;
typedef vector64<int16_t> vector64x16;
typedef vector64<int32_t> vector64x32;
typedef vector64<int64_t> vector64x64;
typedef vector64<uint8_t> vector64xu8;
typedef vector64<uint16_t> vector64xu16;
typedef vector64<uint32_t> vector64xu32;
typedef vector64<uint64_t> vector64xu64;
typedef vector64<float> vector64f;
typedef vector64<double> vector64d;

template<typename T, template<typename, unsigned int, typename> class underlying_vector = native_simd_vector>
class vector128 : public simd_vector<128, underlying_vector, uint64_t, sizeof(T) * CHAR_BIT, T>
{
    typedef simd_vector<128, underlying_vector, uint64_t, sizeof(T) * CHAR_BIT, T> type;

public:
    vector128() : type() {}
    vector128(type v) : type(v) {}

    vector128 &operator=(type v) {type::operator=(v); return *this;}
};

typedef vector128<int8_t> vector128x8;
typedef vector128<int16_t> vector128x16;
typedef vector128<int32_t> vector128x32;
typedef vector128<int64_t> vector128x64;
typedef vector128<uint8_t> vector128xu8;
typedef vector128<uint16_t> vector128xu16;
typedef vector128<uint32_t> vector128xu32;
typedef vector128<uint64_t> vector128xu64;
typedef vector128<float> vector128f;
typedef vector128<double> vector128d;

template<typename T, template<typename, unsigned int, typename> class underlying_vector = native_simd_vector>
class vector256 : public simd_vector<256, underlying_vector, uint64_t, sizeof(T) * CHAR_BIT, T>
{
    typedef simd_vector<256, underlying_vector, uint64_t, sizeof(T) * CHAR_BIT, T> type;

public:
    vector256() : type() {}
    vector256(type v) : type(v) {}

    vector256 &operator=(type v) {type::operator=(v); return *this;}
};

typedef vector256<int8_t> vector256x8;
typedef vector256<int16_t> vector256x16;
typedef vector256<int32_t> vector256x32;
typedef vector256<int64_t> vector256x64;
typedef vector256<uint8_t> vector256xu8;
typedef vector256<uint16_t> vector256xu16;
typedef vector256<uint32_t> vector256xu32;
typedef vector256<uint64_t> vector256xu64;
typedef vector256<float> vector256f;
typedef vector256<double> vector256d;
}

template<typename T, unsigned int bits, typename EffectiveType>
std::ostream &operator<<(std::ostream &os, cppbits::native_simd_vector<T, bits, EffectiveType> vec)
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
