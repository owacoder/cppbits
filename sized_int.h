/*
 * sized_int.h
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

#ifndef SIZED_INT_H
#define SIZED_INT_H

#include <type_traits>

template<typename T, unsigned int length>
class sized_int
{
protected:
    static_assert(std::is_unsigned<T>::value, "T must be an unsigned integral type");

    typedef sized_int<T, length> type;
    typedef T value_type;
    static constexpr T ones = -1;
    static constexpr T mask = ones >> (std::numeric_limits<T>::digits - length);

public:
    sized_int() : data_(0) {}
    sized_int(T value) : data_(value & mask) {}

    type &operator=(const type &value) {data_ = value.data_; return *this;}
    template<typename V, unsigned int len>
    type &operator=(const sized_int<V, len> &value) {data_ = static_cast<T>(value.value()) & mask; return *this;}
    type &operator=(T value) {data_ = value & mask; return *this;}

    template<typename V>
    type &operator+=(const sized_int<V, length> &value) {data_ = (data_ + static_cast<T>(value.value())) & mask; return *this;}
    template<typename V>
    type &operator-=(const sized_int<V, length> &value) {data_ = (data_ - static_cast<T>(value.value())) & mask; return *this;}
    template<typename V>
    type &operator*=(const sized_int<V, length> &value) {data_ = (data_ * static_cast<T>(value.value())) & mask; return *this;}
    template<typename V>
    type &operator/=(const sized_int<V, length> &value) {data_ = (data_ / static_cast<T>(value.value())) & mask; return *this;}
    template<typename V>
    type &operator%=(const sized_int<V, length> &value) {data_ = (data_ % static_cast<T>(value.value())) & mask; return *this;}
    template<typename V>
    type &operator|=(const sized_int<V, length> &value) {data_ |= static_cast<T>(value.value()); return *this;}
    template<typename V>
    type &operator&=(const sized_int<V, length> &value) {data_ &= static_cast<T>(value.value()); return *this;}
    template<typename V>
    type &operator^=(const sized_int<V, length> &value) {data_ ^= static_cast<T>(value.value()); return *this;}
    type &operator<<=(unsigned int shift) {data_ = (data_ << shift) & mask; return *this;}
    type &operator>>=(unsigned int shift) {data_ = (data_ >> shift) & mask; return *this;}

    type &operator+=(T value) {data_ = (data_ + (value & mask)) & mask; return *this;}
    type &operator-=(T value) {data_ = (data_ - (value & mask)) & mask; return *this;}
    type &operator*=(T value) {data_ = (data_ * (value & mask)) & mask; return *this;}
    type &operator/=(T value) {data_ = (data_ / (value & mask)) & mask; return *this;}
    type &operator%=(T value) {data_ = (data_ % (value & mask)) & mask; return *this;}
    type &operator|=(T value) {data_ = (data_ | value) & mask; return *this;}
    type &operator&=(T value) {data_ = (data_ & value) & mask; return *this;}
    type &operator^=(T value) {data_ = (data_ ^ value) & mask; return *this;}

    template<typename V>
    type operator+(const sized_int<V, length> &value) {return type(*this) += value;}
    template<typename V>
    type operator-(const sized_int<V, length> &value) {return type(*this) -= value;}
    template<typename V>
    type operator*(const sized_int<V, length> &value) {return type(*this) *= value;}
    template<typename V>
    type operator/(const sized_int<V, length> &value) {return type(*this) /= value;}
    template<typename V>
    type operator%(const sized_int<V, length> &value) {return type(*this) %= value;}
    template<typename V>
    type operator|(const sized_int<V, length> &value) {return type(*this) |= value;}
    template<typename V>
    type operator&(const sized_int<V, length> &value) {return type(*this) &= value;}
    template<typename V>
    type operator^(const sized_int<V, length> &value) {return type(*this) ^= value;}

    type operator<<(unsigned int shift) {return type(*this) <<= shift;}
    type operator>>(unsigned int shift) {return type(*this) >>= shift;}

    type &operator++() {++data_; return *this;}
    type operator++(int) {type t(*this); ++data_; return t;}
    type &operator--() {--data_; return *this;}
    type operator--(int) {type t(*this); --data_; return t;}

    static constexpr T min() {return 0;}
    static constexpr T max() {return mask;}
    explicit operator T() const {return data_;}
    T value() const {return data_;}

protected:
    T data_;
};

#endif // SIZED_INT_H
