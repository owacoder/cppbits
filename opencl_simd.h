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

#include <array>
#include <vector>
#include <limits.h>

template<typename ElementType, bool is_floating_point, bool is_signed>
class opencl_compile
{
    cl::Program prog_;
    cl::Context ctx_;
    cl::CommandQueue queue_;
    std::vector<cl::Kernel> kernels_;

    static constexpr const char *type = impl::type_name<ElementType, is_signed>::value;
    static constexpr const char *double_type = impl::double_type_name<ElementType, is_signed>::value;

public:
    opencl_compile() {init();}

    cl::Context &context() {return ctx_;}
    cl::CommandQueue &queue() {return queue_;}
    cl::Kernel &kernel(size_t number) {return kernels_[number];}

private:
    void init()
    {
        cl::Platform platform;
        std::vector<cl::Device> devices;

        platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

        ctx_ = cl::Context(devices);
        std::string source;

        if (is_floating_point)
        {
            // TODO: compares should result in all bits set to 1

            source = "__kernel void add(__global const T *A, __global const T *B, __global T *C)\n"
                    "{size_t i = get_global_id(0); C[i] = A[i] + B[i];}\n\n"
                    ""
                    "__kernel void add_hi_signed(__global const T *A, __global const T *B, __global T *C)\n"
                    "{size_t i = get_global_id(0); C[i] = A[i] + B[i];}\n\n"
                    ""
                    "__kernel void add_hi_unsigned(__global const T *A, __global const T *B, __global T *C)\n"
                    "{size_t i = get_global_id(0); C[i] = A[i] + B[i];}\n\n"
                    ""
                    "__kernel void add_saturate_signed(__global const T *A, __global const T *B, __global T *C)\n"
                    "{size_t i = get_global_id(0); C[i] = A[i] + B[i];}\n\n"
                    ""
                    "__kernel void add_saturate_unsigned(__global const T *A, __global const T *B, __global T *C)\n"
                    "{size_t i = get_global_id(0); C[i] = A[i] + B[i];}\n\n"
                    ""
                    "__kernel void sub(__global const T *A, __global const T *B, __global T *C)\n"
                    "{size_t i = get_global_id(0); C[i] = A[i] - B[i];}\n\n"
                    ""
                    "__kernel void sub_hi_signed(__global const T *A, __global const T *B, __global T *C)\n"
                    "{size_t i = get_global_id(0); C[i] = A[i] - B[i];}\n\n"
                    ""
                    "__kernel void sub_hi_unsigned(__global const T *A, __global const T *B, __global T *C)\n"
                    "{size_t i = get_global_id(0); C[i] = A[i] - B[i];}\n\n"
                    ""
                    "__kernel void sub_saturate_signed(__global const T *A, __global const T *B, __global T *C)\n"
                    "{size_t i = get_global_id(0); C[i] = A[i] - B[i];}\n\n"
                    ""
                    "__kernel void sub_saturate_unsigned(__global const T *A, __global const T *B, __global T *C)\n"
                    "{size_t i = get_global_id(0); C[i] = A[i] - B[i];}\n\n"
                    ""
                    "__kernel void mul(__global const T *A, __global const T *B, __global T *C)\n"
                    "{size_t i = get_global_id(0); C[i] = A[i] * B[i];}\n\n"
                    ""
                    "__kernel void div(__global const T *A, __global const T *B, __global T *C)\n"
                    "{size_t i = get_global_id(0); C[i] = A[i] / B[i];}\n\n"
                    ""
                    "__kernel void shl(__global const T *A, unsigned int B, __global T *C)\n"
                    "{size_t i = get_global_id(0); C[i] = A[i] * pow(2.0, B);}\n\n"
                    ""
                    "__kernel void shr(__global const T *A, unsigned int B, __global T *C)\n"
                    "{size_t i = get_global_id(0); C[i] = A[i] / pow(2.0, B);}\n\n"
                    ""
                    "__kernel void negate(__global const T *A, __global T *B)\n"
                    "{size_t i = get_global_id(0); B[i] = -A[i];}\n\n"
                    ""
                    "__kernel void absolute(__global const T *A, __global T *B)\n"
                    "{size_t i = get_global_id(0); B[i] = abs(A[i]);}\n\n"
                    ""
                    "__kernel void avg(__global const T *A, __global const T *B, __global T *C)\n"
                    "{size_t i = get_global_id(0); C[i] = (A[i] + B[i]) / 2.0;}\n\n"
                    ""
                    "__kernel void eq(__global const T *A, __global const T *B, __global T *C)\n"
                    "{size_t i = get_global_id(0); C[i] = A[i] == B[i]? -1: 0;}\n\n"
                    ""
                    "__kernel void ne(__global const T *A, __global const T *B, __global T *C)\n"
                    "{size_t i = get_global_id(0); C[i] = A[i] != B[i]? -1: 0;}\n\n"
                    ""
                    "__kernel void lt(__global const T *A, __global const T *B, __global T *C)\n"
                    "{size_t i = get_global_id(0); C[i] = A[i] < B[i]? -1: 0;}\n\n"
                    ""
                    "__kernel void gt(__global const T *A, __global const T *B, __global T *C)\n"
                    "{size_t i = get_global_id(0); C[i] = A[i] > B[i]? -1: 0;}\n\n"
                    ""
                    "__kernel void le(__global const T *A, __global const T *B, __global T *C)\n"
                    "{size_t i = get_global_id(0); C[i] = A[i] <= B[i]? -1: 0;}\n\n"
                    ""
                    "__kernel void ge(__global const T *A, __global const T *B, __global T *C)\n"
                    "{size_t i = get_global_id(0); C[i] = A[i] >= B[i]? -1: 0;}\n\n";
        }
        else
        {
            // TODO: signed saturation
            // TODO: signed hi part

            source = "__kernel void add(__global const T *A, __global const T *B, __global T *C)\n"
                    "{size_t i = get_global_id(0); C[i] = A[i] + B[i];}\n\n"
                    ""
                    "__kernel void add_hi_signed(__global const T *A, __global const T *B, __global T *C)\n"
                    "{size_t i = get_global_id(0); C[i] = A[i] + B[i];}\n\n"
                    ""
                    "__kernel void add_hi_unsigned(__global const T *A, __global const T *B, __global T *C)\n"
                    "{size_t i = get_global_id(0); T temp = A[i] + B[i]; C[i] = temp < A[i];}\n\n"
                    ""
                    "__kernel void add_saturate_signed(__global const T *A, __global const T *B, __global T *C)\n"
                    "{size_t i = get_global_id(0); C[i] = A[i] + B[i];}\n\n"
                    ""
                    "__kernel void add_saturate_unsigned(__global const T *A, __global const T *B, __global T *C)\n"
                    "{size_t i = get_global_id(0); T temp = A[i] + B[i]; C[i] = temp < A[i]? -1: temp;}\n\n"
                    ""
                    "__kernel void sub(__global const T *A, __global const T *B, __global T *C)\n"
                    "{size_t i = get_global_id(0); C[i] = A[i] - B[i];}\n\n"
                    ""
                    "__kernel void sub_hi_signed(__global const T *A, __global const T *B, __global T *C)\n"
                    "{size_t i = get_global_id(0); C[i] = A[i] - B[i];}\n\n"
                    ""
                    "__kernel void sub_hi_unsigned(__global const T *A, __global const T *B, __global T *C)\n"
                    "{size_t i = get_global_id(0); T temp = A[i] - B[i]; C[i] = temp > A[i];}\n\n"
                    ""
                    "__kernel void sub_saturate_signed(__global const T *A, __global const T *B, __global T *C)\n"
                    "{size_t i = get_global_id(0); C[i] = A[i] - B[i];}\n\n"
                    ""
                    "__kernel void sub_saturate_unsigned(__global const T *A, __global const T *B, __global T *C)\n"
                    "{size_t i = get_global_id(0); T temp = A[i] - B[i]; C[i] = temp > A[i]? 0: temp;}\n\n"
                    ""
                    "__kernel void mul(__global const T *A, __global const T *B, __global T *C)\n"
                    "{size_t i = get_global_id(0); C[i] = A[i] * B[i];}\n\n"
                    ""
                    "__kernel void div(__global const T *A, __global const T *B, __global T *C)\n"
                    "{size_t i = get_global_id(0); C[i] = A[i] / B[i];}\n\n"
                    ""
                    "__kernel void shl(__global const T *A, unsigned int B, __global T *C)\n"
                    "{size_t i = get_global_id(0); C[i] = A[i] << B;}\n\n"
                    ""
                    "__kernel void shr(__global const T *A, unsigned int B, __global T *C)\n"
                    "{size_t i = get_global_id(0); C[i] = A[i] >> B;}\n\n"
                    ""
                    "__kernel void negate(__global const T *A, __global T *B)\n"
                    "{size_t i = get_global_id(0); B[i] = -A[i];}\n\n"
                    ""
                    "__kernel void absolute(__global const T *A, __global T *B)\n"
                    "{size_t i = get_global_id(0); B[i] = abs(A[i]);}\n\n"
                    ""
                    "__kernel void avg(__global const T *A, __global const T *B, __global T *C)\n"
                    "{size_t i = get_global_id(0); C[i] = (A[i] + B[i] + 1) / 2;}\n\n"
                    ""
                    "__kernel void eq(__global const T *A, __global const T *B, __global T *C)\n"
                    "{size_t i = get_global_id(0); C[i] = A[i] == B[i]? -1: 0;}\n\n"
                    ""
                    "__kernel void ne(__global const T *A, __global const T *B, __global T *C)\n"
                    "{size_t i = get_global_id(0); C[i] = A[i] != B[i]? -1: 0;}\n\n"
                    ""
                    "__kernel void lt(__global const T *A, __global const T *B, __global T *C)\n"
                    "{size_t i = get_global_id(0); C[i] = A[i] < B[i]? -1: 0;}\n\n"
                    ""
                    "__kernel void gt(__global const T *A, __global const T *B, __global T *C)\n"
                    "{size_t i = get_global_id(0); C[i] = A[i] > B[i]? -1: 0;}\n\n"
                    ""
                    "__kernel void le(__global const T *A, __global const T *B, __global T *C)\n"
                    "{size_t i = get_global_id(0); C[i] = A[i] <= B[i]? -1: 0;}\n\n"
                    ""
                    "__kernel void ge(__global const T *A, __global const T *B, __global T *C)\n"
                    "{size_t i = get_global_id(0); C[i] = A[i] >= B[i]? -1: 0;}\n\n";
        }

        for (size_t i = 0; i = source.find('T', i), i != source.npos;)
            source.replace(i, 1, type);

        for (size_t i = 0; i = source.find('X', i), i != source.npos;)
            source.replace(i, 1, double_type);

        prog_ = cl::Program(ctx_, source, true);
        prog_.createKernels(&kernels_);

        queue_ = cl::CommandQueue(ctx_);
    }
};

namespace cppbits {
template<unsigned int desired_elements, unsigned int element_bits, typename EffectiveType>
class opencl_simd_vector {
    static_assert(element_bits, "Number of bits per element must be greater than zero");

    static constexpr unsigned int vector_digits = desired_elements * element_bits;
    static constexpr bool elements_are_floats = std::is_floating_point<EffectiveType>::value;
    static constexpr bool elements_are_signed = std::is_signed<EffectiveType>::value && !elements_are_floats;
    typedef opencl_simd_vector<desired_elements, element_bits, EffectiveType> type;

public:
    typedef typename impl::unsigned_int<element_bits>::type underlying_element_type;
    typedef cl::Buffer underlying_vector_type;

private:
    static opencl_compile<EffectiveType, elements_are_floats, elements_are_signed> &interface()
    {
        static opencl_compile<EffectiveType, elements_are_floats, elements_are_signed> compile;
        return compile;
    }

    enum kernel_entry
    {
        kernel_abs,
        kernel_add,
        kernel_add_hi_signed,
        kernel_add_hi_unsigned,
        kernel_add_sat_signed,
        kernel_add_sat_unsigned,
        kernel_avg,
        kernel_div,
        kernel_eq,
        kernel_ge,
        kernel_gt,
        kernel_le,
        kernel_lt,
        kernel_mul,
        kernel_ne,
        kernel_neg,
        kernel_shl,
        kernel_shr,
        kernel_sub,
        kernel_sub_hi_signed,
        kernel_sub_hi_unsigned,
        kernel_sub_sat_signed,
        kernel_sub_sat_unsigned
    };

    static constexpr underlying_element_type ones = -1;
    static constexpr unsigned int elements = vector_digits / element_bits;
    static constexpr underlying_element_type element_mask = ones >> (std::numeric_limits<underlying_element_type>::digits - element_bits);

    static constexpr bool effective_type_is_exact_size = sizeof(EffectiveType) * CHAR_BIT == element_bits;
    static_assert(effective_type_is_exact_size, "EffectiveType must be the same size as each element for OpenCL SIMD wrappers to work properly");
    static_assert(!elements_are_floats || (element_bits == 32 || element_bits == 64), "Floating-point element size must be 32 or 64 bits");
    static_assert(element_bits <= 64, "Element size is too large");
    static_assert(elements_are_floats || (sizeof(EffectiveType) * CHAR_BIT) >= element_bits, "Element size is too large for specified effective type `EffectiveType`");

    constexpr static underlying_element_type make_t_from_effective(EffectiveType v)
    {
        return elements_are_floats? element_bits == 32? underlying_element_type(float_to_ieee_754(v)): underlying_element_type(double_to_ieee_754(v)): underlying_element_type(v);
    }
    static EffectiveType make_effective_from_t(underlying_element_type v)
    {
        if (elements_are_floats)
            return element_bits == 32? EffectiveType(float_from_ieee_754(v)):
                                       EffectiveType(double_from_ieee_754(v));
        else if (elements_are_signed)
        {
            const underlying_element_type val = v & (element_mask >> 1);
            const underlying_element_type negative = v >> (element_bits - 1);
            return negative? val == 0? scalar_min(): -EffectiveType((~val + 1) & (element_mask >> 1)): EffectiveType(val);
        }
        return EffectiveType(v);
    }

    static constexpr unsigned int buf_size = (((vector_digits / 8) >> 5) + 1) << 5;

    /*
     *  Special constructor for output of operations
     */
    constexpr opencl_simd_vector(bool)
        : data_{}
        , push_first_(false)
        , pull_first_(true)
        , buf_(interface().context(), CL_MEM_READ_WRITE, buf_size)
    {}

public:
    /*
     * Default constructor zeros the vector
     */
    constexpr opencl_simd_vector()
        : data_{}
        , push_first_(true)
        , pull_first_(false)
        , buf_(interface().context(), CL_MEM_READ_WRITE, buf_size)
    {}

    /*
     * Copy constructor copies the data and context, initializes new buffer
     */
    opencl_simd_vector(const opencl_simd_vector &other)
        : data_(other.data_)
        , push_first_(other.push_first_)
        , pull_first_(other.pull_first_)
        , buf_(other.buf_)
    {}

    /*
     * Move constructor copies the data and context, initializes new buffer
     */
    opencl_simd_vector(opencl_simd_vector &&other)
        : data_(std::move(other.data_))
        , push_first_(other.push_first_)
        , pull_first_(other.pull_first_)
        , buf_(std::move(other.buf_))
    {}

    /*
     * Copy assignment copies data and buffer
     */
    type &operator=(const opencl_simd_vector &other)
    {
        using namespace std;
        if (!other.pull_first_)
            copy(other.data_.begin(), other.data_.end(), data_.begin());
        push_first_ = other.push_first_;
        pull_first_ = other.pull_first_;
        buf_ = other.buf_;
        return *this;
    }

    /*
     * Move assignment moves data and buffer
     */
    type &operator=(opencl_simd_vector &&other)
    {
        using namespace std;
        if (!other.pull_first_)
            move(other.data_.begin(), other.data_.end(), data_.begin());
        push_first_ = other.push_first_;
        pull_first_ = other.pull_first_;
        buf_ = std::move(other.buf_);
        return *this;
    }

    /*
     * Returns a representation of a vector with specified value assigned to element 0
     */
    static type make_scalar(EffectiveType value)
    {
        type result;
        result.data_[0] = make_t_from_effective(value);
        return result;
    }

    /*
     * Returns a representation of a vector with specified value assigned to every element in the vector
     */
    static type make_broadcast(EffectiveType value)
    {
        type result;
        result.data_.fill(make_t_from_effective(value));
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
    constexpr type operator~() const
    {
        return make_init_raw_value([](underlying_element_type a) {return ~a;});
    }

    /*
     * Logical OR's the entire vector with `vec` and returns it
     */
    constexpr type operator|(opencl_simd_vector vec) const
    {
        return make_init_raw_values(vec, [](underlying_element_type a, underlying_element_type b) {return a | b;});
    }

    /*
     * Logical AND's the entire vector with `vec` and returns it
     */
    constexpr type operator&(opencl_simd_vector vec) const
    {
        return type(data_ & vec.vector());
    }

    /*
     * Logical AND's the entire vector with negated `vec` and returns it
     */
    constexpr type and_not(opencl_simd_vector vec) const
    {
        return type(data_ & ~vec.vector());
    }

    /*
     * Logical XOR's the entire vector with `vec` and returns it
     */
    constexpr type operator^(opencl_simd_vector vec) const
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
    type operator+(opencl_simd_vector vec) const {return add(vec, cppbits::math_keeplow);}

    /*
     * Adds elements of `vec` to `this` (using specified math method) and returns the result
     */
    type add(opencl_simd_vector vec, cppbits::math_type math) const
    {
        type result(true);

        push();
        vec.push();

        cl::Kernel *ref;

        if (elements_are_floats)
            ref = &interface().kernel(kernel_add);
        else
        {
            switch (math)
            {
                default: /* Rollover arithmetic */ ref = &interface().kernel(kernel_add); break;
                case cppbits::math_saturate: ref = &interface().kernel(elements_are_signed? kernel_add_sat_signed: kernel_add_sat_unsigned); break;
                case cppbits::math_keephigh: ref = &interface().kernel(elements_are_signed? kernel_add_hi_signed: kernel_add_hi_unsigned); break;
            }
        }

        ref->setArg(0, buf_);
        ref->setArg(1, vec.buf_);
        ref->setArg(2, result.buf_);

        interface().queue().enqueueNDRangeKernel(*ref, cl::NullRange, cl::NDRange(buf_size), cl::NDRange(32));

        return result;
    }

    /*
     * Subtracts elements of `vec` to `this` (using rollover subtraction) and returns the result
     */
    type operator-(opencl_simd_vector vec) const {return sub(vec, cppbits::math_keeplow);}

    /*
     * Subtracts elements of `vec` from `this` (using specified math method) and returns the result
     */
    type sub(opencl_simd_vector vec, cppbits::math_type math) const
    {
        type result(true);

        push();
        vec.push();

        cl::Kernel *ref;

        if (elements_are_floats)
            ref = &interface().kernel(kernel_sub);
        else
        {
            switch (math)
            {
                default: /* Rollover arithmetic */ ref = &interface().kernel(kernel_sub); break;
                case cppbits::math_saturate: ref = &interface().kernel(elements_are_signed? kernel_sub_sat_signed: kernel_sub_sat_unsigned); break;
                case cppbits::math_keephigh: ref = &interface().kernel(elements_are_signed? kernel_sub_hi_signed: kernel_sub_hi_unsigned); break;
            }
        }

        ref->setArg(0, buf_);
        ref->setArg(1, vec.buf_);
        ref->setArg(2, result.buf_);

        interface().queue().enqueueNDRangeKernel(*ref, cl::NullRange, cl::NDRange(buf_size), cl::NDRange(32));

        return result;
    }

    /*
     * Multiplies elements of `vec` by `this` (using rollover multiplication) and returns the result
     */
    type operator*(opencl_simd_vector vec) const {return mul(vec, cppbits::math_keeplow);}

    /*
     * Multiplies elements of `vec` by `this` (using specified math method) and returns the result
     */
    type mul(opencl_simd_vector vec, cppbits::math_type math) const
    {
        type result(true);

        push();
        vec.push();

        cl::Kernel *ref;

        if (elements_are_floats)
            ref = &interface().kernel(kernel_mul);
        else
        {
            switch (math)
            {
                default: /* Rollover arithmetic */ ref = &interface().kernel(kernel_mul); break;
                case cppbits::math_saturate: CPPBITS_ERROR("Multiplication with math_saturate not implemented yet");
                case cppbits::math_keephigh: CPPBITS_ERROR("Multiplication with math_keephigh not implemented yet");
            }
        }

        ref->setArg(0, buf_);
        ref->setArg(1, vec.buf_);
        ref->setArg(2, result.buf_);

        interface().queue().enqueueNDRangeKernel(*ref, cl::NullRange, cl::NDRange(buf_size), cl::NDRange(32));

        return result;
    }

    /*
     * Multiplies elements of `vec` by `this` (using specified math method), adds `add`, and returns the result
     */
    constexpr type mul_add(opencl_simd_vector vec, opencl_simd_vector add, cppbits::math_type math) const
    {
        return mul(vec, math).add(add, math);
    }

    /*
     * Multiplies elements of `vec` by `this` (using specified math method), subtracts `sub`, and returns the result
     */
    constexpr type mul_sub(opencl_simd_vector vec, opencl_simd_vector sub, cppbits::math_type math) const
    {
        return mul(vec, math).sub(sub, math);
    }

    /*
     * Divides elements of `this` by `vec` (using rollover division) and returns the result
     */
    type operator/(opencl_simd_vector vec) const {return div(vec, cppbits::math_keeplow);}

    /*
     * Divides elements of `this` by `vec` (using specified math method) and returns the result
     */
    type div(opencl_simd_vector vec, cppbits::math_type math) const
    {
        type result(true);

        push();
        vec.push();

        cl::Kernel *ref;

        if (elements_are_floats)
            ref = &interface().kernel(kernel_div);
        else if (vec.has_zero_element())
            CPPBITS_ERROR("Division by zero");
        else
        {
            switch (math)
            {
                default: /* Rollover arithmetic */ ref = &interface().kernel(kernel_div); break;
                case cppbits::math_saturate: CPPBITS_ERROR("division with math_saturate not implemented yet");
                case cppbits::math_keephigh: CPPBITS_ERROR("division with math_keephigh not implemented yet");
            }
        }

        ref->setArg(0, buf_);
        ref->setArg(1, vec.buf_);
        ref->setArg(2, result.buf_);

        interface().queue().enqueueNDRangeKernel(*ref, cl::NullRange, cl::NDRange(buf_size), cl::NDRange(32));

        return result;
    }

    /*
     * Computes the average of each element of `this` and `vec` as `((element of this) + (element of vec) + 1)/2` for integral values,
     * or `((element of this) + (element of vec))/2` for floating-point values
     * TODO: if element_bits == number of bits in T, undefined behavior results
     */
    type avg(opencl_simd_vector vec) const
    {
        type result(true);

        push();
        vec.push();

        cl::Kernel *ref = &interface().kernel(kernel_avg);

        ref->setArg(0, buf_);
        ref->setArg(1, vec.buf_);
        ref->setArg(2, result.buf_);

        interface().queue().enqueueNDRangeKernel(*ref, cl::NullRange, cl::NDRange(buf_size), cl::NDRange(32));

        return result;
    }

    /*
     * Shifts each element of `this` to the left by `amount` and returns the resulting vector
     * If the invariant `0 <= amount <= element_bits` does not hold, undefined behavior results
     * TODO: support floating-point values
     */
    type operator<<(unsigned int amount) const {return shl(amount);}

    /*
     * Shifts each element of `this` to the left by `amount` and returns the resulting vector
     * If the invariant `0 <= amount <= element_bits` does not hold, undefined behavior results
     * TODO: if element_bits == number of bits in T, undefined behavior results
     * TODO: support floating-point values
     */
    type shl(unsigned int amount) const
    {
        type result(true);

        push();

        cl::Kernel *ref = &interface().kernel(kernel_shl);

        ref->setArg(0, buf_);
        ref->setArg(1, amount);
        ref->setArg(2, result.buf_);

        interface().queue().enqueueNDRangeKernel(*ref, cl::NullRange, cl::NDRange(buf_size), cl::NDRange(32));

        return result;
    }

    /*
     * Shifts each element of `this` to the right by `amount` (using shift_natural behavior) and returns the resulting vector
     * If the invariant `0 <= amount <= element_bits` does not hold, undefined behavior results
     * TODO: support floating-point values
     */
#if 0
    constexpr type operator>>(unsigned int amount) const {return shr(amount, cppbits::shift_natural);}
#endif

    /*
     * Shifts each element of `this` to the right by `amount` (using specified shift type) and returns the resulting vector
     * If the invariant `0 <= amount <= element_bits` does not hold, undefined behavior results
     * TODO: if element_bits == number of bits in T, undefined behavior results
     * TODO: support floating-point values
     */
#if 0
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
#endif

    /*
     * Extracts MSB from each element and places them in the low bits of the result
     * Each bit position in the result corresponds to the element position in the source vector
     * (i.e. Element 0 MSB -> Bit 0, Element 1 MSB -> Bit 1, etc.)
     */
#if 0
    unsigned int movmsk() const
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
#endif

    /*
     * Negates elements of `this` and returns the resulting vector
     * Note that this function returns the value unmodified if EffectiveType is not a signed type
     * TODO: are floating-point values supported properly?
     */
    type negate() const
    {
        type result(true);

        push();

        cl::Kernel *ref = &interface().kernel(kernel_neg);

        ref->setArg(0, buf_);
        ref->setArg(1, result.buf_);

        interface().queue().enqueueNDRangeKernel(*ref, cl::NullRange, cl::NDRange(buf_size), cl::NDRange(32));

        return result;
    }

    /*
     * Computes the absolute value of elements of `this` and returns the resulting vector
     * Note that this function returns the value unmodified if EffectiveType is not a signed type
     */
    type abs() const
    {
        if (!elements_are_floats && !elements_are_signed)
            return *this;

        type result(true);

        push();

        cl::Kernel *ref = &interface().kernel(kernel_abs);

        ref->setArg(0, buf_);
        ref->setArg(1, result.buf_);

        interface().queue().enqueueNDRangeKernel(*ref, cl::NullRange, cl::NDRange(buf_size), cl::NDRange(32));

        return result;
    }

    /*
     * Computes the hypotenuse length (`sqrt(x^2 + y^2)`) and returns the resulting vector
     */
    constexpr type hypot(opencl_simd_vector vec, cppbits::math_type math) const
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
    bool has_zero_element()
    {
        bool has_zero = false;
        for (unsigned i = 0; i < max_elements(); ++i)
            has_zero |= get(i) == EffectiveType(0);
        return has_zero;
    }

    /*
     * Returns number of zero elements in vector (sign of zero irrelevant in the case of floating-point)
     * See <http://graphics.stanford.edu/~seander/bithacks.html#HasLessInWord>
     * This is a rewrite of countless(u, 1)
     */
    unsigned int count_zero_elements() const
    {
        unsigned zeros = 0;
        for (unsigned i = 0; i < max_elements(); ++i)
            zeros += get(i) == EffectiveType(0);
        return zeros;
    }

    /*
     * Returns true if vector has at least one element equal to `v`, false otherwise
     * See <http://graphics.stanford.edu/~seander/bithacks.html#HasLessInWord>
     * This is a rewrite of hasless(u, 1)
     */
    bool has_equal_element(EffectiveType v) const
    {
        bool has_equal = false;
        for (unsigned i = 0; i < max_elements(); ++i)
            has_equal |= get(i) == v;
        return has_equal;
    }

    /*
     * Returns number of elements in vector equal to `v`
     * See <http://graphics.stanford.edu/~seander/bithacks.html#HasLessInWord>
     * This is a rewrite of countless(u, 1)
     */
    unsigned int count_equal_elements(EffectiveType v) const
    {
        unsigned equals = 0;
        for (unsigned i = 0; i < max_elements(); ++i)
            equals += get(i) == v;
        return equals;
    }

    /*
     * Compares `this` to `vec`. If the comparison result is true, the corresponding element is set to all 1's
     * Otherwise, if the comparison result is false, the corresponding element is set to all 0's
     * TODO: verify comparisons for floating-point values
     */
    type cmp(opencl_simd_vector vec, cppbits::compare_type compare) const
    {
        type result(true);

        push();
        vec.push();

        cl::Kernel *ref;

        switch (compare) {
            default: ref = &interface().kernel(kernel_eq); break;
            case cppbits::compare_nequal: ref = &interface().kernel(kernel_ne); break;
            case cppbits::compare_less: ref = &interface().kernel(kernel_lt); break;
            case cppbits::compare_lessequal: ref = &interface().kernel(kernel_le); break;
            case cppbits::compare_greater: ref = &interface().kernel(kernel_gt); break;
            case cppbits::compare_greaterequal: ref = &interface().kernel(kernel_ge); break;
        }

        ref->setArg(0, buf_);
        ref->setArg(1, vec.buf_);
        ref->setArg(2, result.buf_);

        interface().queue().enqueueNDRangeKernel(*ref, cl::NullRange, cl::NDRange(buf_size), cl::NDRange(32));

        return result;
    }

    type operator==(opencl_simd_vector vec) const {return cmp(vec, cppbits::compare_equal);}
    type operator!=(opencl_simd_vector vec) const {return cmp(vec, cppbits::compare_nequal);}
    type operator<(opencl_simd_vector vec) const {return cmp(vec, cppbits::compare_less);}
    type operator<=(opencl_simd_vector vec) const {return cmp(vec, cppbits::compare_lessequal);}
    type operator>(opencl_simd_vector vec) const {return cmp(vec, cppbits::compare_greater);}
    type operator>=(opencl_simd_vector vec) const {return cmp(vec, cppbits::compare_greaterequal);}

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
#if 0
    type max(opencl_simd_vector vec) const
    {
        return make_init_values(vec, [](EffectiveType a, EffectiveType b) {using namespace std; return max(a, b);});
    }
#endif

    /*
     * Sets each element in output to minimum of respective elements of `this` and `vec`
     * TODO: verify comparisons for floating-point values
     */
#if 0
    type min(opencl_simd_vector vec) const
    {
        return make_init_values(vec, [](EffectiveType a, EffectiveType b) {using namespace std; return min(a, b);});
    }
#endif

    /*
     * Sets element `idx` to `value`
     */
#if 0
    template<unsigned int idx>
    type &set(EffectiveType value)
    {
        data_ |= bitfield_member<T, idx * element_bits, element_bits>(make_t_from_effective(value)).bitfield_value() & mask;
        return *this;
    }
#endif

    /*
     * Sets element `idx` to `value`
     */
    type &set(unsigned int idx, EffectiveType value)
    {
        pull();
        push_first_ = true;
        data_[idx] = make_t_from_effective(value);
        return *this;
    }

    /*
     * Sets all bits of element `idx` to 1
     */
    template<unsigned int idx>
    type &set()
    {
        pull();
        push_first_ = true;
        data_[idx] = ones;
        return *this;
    }

    /*
     * Sets all bits of element `idx` to 1
     */
    type &set(unsigned int idx)
    {
        pull();
        push_first_ = true;
        data_[idx] = ones;
        return *this;
    }

    /*
     * Sets all bits of element `idx` to 1 if `v` is true, 0 otherwise
     */
    template<unsigned int idx>
    type &set_bits(bool v)
    {
        pull();
        push_first_ = true;
        data_[idx] = v? ones: 0;
        return *this;
    }

    /*
     * Sets all bits of element `idx` to 1 if `v` is true, 0 otherwise
     */
    type &set_bits(unsigned int idx, bool v)
    {
        pull();
        push_first_ = true;
        data_[idx] = v? ones: 0;
        return *this;
    }

    /*
     * Sets all bits of element `idx` to 0
     */
#if 0
    template<unsigned int idx>
    type &reset()
    {
        pull();
        push_first_ = true;
        data_ &= ~bitfield_member<T, idx * element_bits, element_bits>::bitfield_mask();
        return *this;
    }
#endif

    /*
     * Sets all bits of element `idx` to 0
     */
    type &reset(unsigned int idx)
    {
        const unsigned int shift = idx * element_bits;
        pull();
        push_first_ = true;
        data_ &= ~(element_mask << shift);
        return *this;
    }

    /*
     * Flips all bits of element `idx`
     */
#if 0
    template<unsigned int idx>
    type &flip()
    {
        pull();
        push_first_ = true;
        data_ ^= bitfield_member<T, idx * element_bits, element_bits>::bitfield_mask();
        return *this;
    }
#endif

    /*
     * Flips all bits of element `idx`
     */
    type &flip(unsigned int idx)
    {
        const unsigned int shift = idx * element_bits;
        pull();
        push_first_ = true;
        data_ ^= (element_mask << shift);
        return *this;
    }

    /*
     * Gets value of element `idx`
     */
    template<unsigned int idx>
    EffectiveType get()
    {
        pull();
        return make_effective_from_t(data_[idx]);
    }

    /*
     * Gets value of element `idx`
     */
    EffectiveType get(unsigned int idx)
    {
        pull();
        return make_effective_from_t(data_[idx]);
    }

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
        for (unsigned i = 0; i < max_elements(); ++i)
            data_[i] = *mem++ & element_mask;
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

    /* Determines if aligned accesses are possible with specified pointer (does nothing with opencl_simd_vector) */
    static constexpr bool ptr_is_aligned(EffectiveType *)
    {
        return true;
    }

private:
    // Pushes from data to buffer if necessary
    void push() const
    {
        if (push_first_)
        {
            interface().queue().enqueueWriteBuffer(buf_, CL_TRUE, 0, vector_digits / 8, data_.data());
            push_first_ = false;
        }
    }

    // Pushes from buffer to data if necessary
    void pull()
    {
        if (pull_first_)
        {
            interface().queue().enqueueReadBuffer(buf_, CL_TRUE, 0, vector_digits / 8, data_.data());
            pull_first_ = false;
        }
    }

    // Lambda should be a functor taking an EffectiveType and returning an EffectiveType value
    template<typename Lambda>
    type make_init_value(Lambda callable) const
    {
        type result;
        constexpr unsigned int shift = element_bits % vector_digits;

        for (unsigned i = 0; i < max_elements(); ++i)
            result |= (make_t_from_effective(callable(get(i))) & element_mask) << (i * shift);

        return type(result);
    }

    // Lambda should be a functor taking an underlying_element_type and returning an underlying_element_type value
    template<typename Lambda>
    type make_init_raw_value(Lambda callable) const
    {
        type res;

        for (unsigned i = 0; i < max_elements(); ++i)
            res.data_[i] = callable(data_[i]);

        return res;
    }

    // Lambda should be a functor taking 2 EffectiveType values and returning an EffectiveType value
    template<typename Lambda>
    type make_init_values(opencl_simd_vector vec, Lambda callable) const
    {
        type result;
        constexpr unsigned int shift = element_bits % vector_digits;

        for (unsigned i = 0; i < max_elements(); ++i)
            result |= (make_t_from_effective(callable(get(i), vec.get(i))) & element_mask) << (i * shift);

        return type(result);
    }

    // Lambda should be a functor taking 2 underlying_element_type values and returning an underlying_element_type value
    template<typename Lambda>
    type make_init_raw_values(opencl_simd_vector vec, Lambda callable) const
    {
        type res;

        for (unsigned i = 0; i < max_elements(); ++i)
            res.data_[i] = callable(data_[i], vec.data_[i]);

        return res;
    }

    // Lambda should be a functor taking 2 EffectiveType values and returning a boolean value
    template<typename Lambda>
    type make_init_cmp_values(opencl_simd_vector vec, Lambda callable) const
    {
        type result;
        constexpr unsigned int shift = element_bits % vector_digits;

        for (unsigned i = 0; i < max_elements(); ++i)
            result |= (callable(get(i), vec.get(i)) * element_mask) << (i * shift);

        return type(result);
    }

    std::array<underlying_element_type, desired_elements> data_;
    mutable bool push_first_, pull_first_;
    underlying_vector_type buf_;
};

#if 0
template<typename T, unsigned int element_bits, typename EffectiveType>
struct native_simd_vector : public opencl_simd_vector<T, element_bits, EffectiveType>
{
    native_simd_vector() {}
    native_simd_vector(opencl_simd_vector<T, element_bits, EffectiveType> v)
        : opencl_simd_vector<T, element_bits, EffectiveType>(v)
    {}
};
#endif
}

template<unsigned int desired_elements, unsigned int bits, typename EffectiveType>
std::ostream &operator<<(std::ostream &os, cppbits::opencl_simd_vector<desired_elements, bits, EffectiveType> vec)
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

#endif // OPENCL_SIMD_H
