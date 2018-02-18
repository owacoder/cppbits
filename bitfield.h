/*
 * bitfield.h
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

#ifndef BITFIELD_H
#define BITFIELD_H

#include "sized_int.h"

template<typename T, unsigned int lsb_pos, unsigned int length>
class bitfield_member : public sized_int<T, length>
{
    typedef bitfield_member<T, lsb_pos, length> type;
    typedef sized_int<T, length> inttype;
    typedef T value_type;
    static constexpr T bitmask = inttype::mask << lsb_pos;

public:
    bitfield_member() : inttype() {}
    explicit bitfield_member(T value) : inttype(value) {}
    template<typename V>
    bitfield_member(const sized_int<V, length> &value) : inttype(static_cast<T>(value.value())) {}

    T bitfield_value() const {return inttype::value() << lsb_pos;}
    static constexpr T bitfield_mask() {return bitmask;}
    static constexpr unsigned int start() {return lsb_pos;}
    static constexpr unsigned int size() {return length;}

    bool any() const {return inttype::value() != 0;}
    bool none() const {return inttype::value() == 0;}
    bool all() const {return inttype::value() == bitfield_mask();}
};

template<typename T>
class bitfield
{
    typedef bitfield<T> type;
    typedef T value_type;
    static constexpr T mask = -1;

public:
    bitfield() : data_(0) {}
    explicit bitfield(T value) : data_(value) {}

    template<typename V, unsigned int lsb_pos, unsigned int length>
    type &operator=(const bitfield_member<V, lsb_pos, length> &bitfield)
    {
        data_ = static_cast<T>(bitfield.bitfield_value());
        return *this;
    }

    template<typename U>
    type &operator=(U value) {data_ = static_cast<T>(value); return *this;}

    template<typename V, unsigned int lsb_pos, unsigned int length>
    type &operator|=(const bitfield_member<V, lsb_pos, length> &bitfield)
    {
        data_ |= static_cast<T>(bitfield.bitfield_value());
        return *this;
    }

    template<typename V, unsigned int lsb_pos, unsigned int length>
    type &operator&=(const bitfield_member<V, lsb_pos, length> &bitfield)
    {
        data_ &= static_cast<T>(bitfield.bitfield_value());
        return *this;
    }

    template<typename V, unsigned int lsb_pos, unsigned int length>
    type &operator^=(const bitfield_member<V, lsb_pos, length> &bitfield)
    {
        data_ ^= static_cast<T>(bitfield.bitfield_value());
        return *this;
    }

    template<typename V, unsigned int lsb_pos, unsigned int length>
    type &set(const bitfield_member<V, lsb_pos, length> &bitfield)
    {
        data_ = (data_ & ~static_cast<T>(bitfield.bitfield_mask())) | static_cast<T>(bitfield.bitfield_value());
        return *this;
    }

    template<unsigned int lsb_pos, unsigned int length>
    type &set()
    {
        data_ |= bitfield_member<T, lsb_pos, length>::bitfield_mask();
        return *this;
    }

    template<unsigned int lsb_pos, unsigned int length, T value>
    type &set()
    {
        data_ |= bitfield_member<T, lsb_pos, length>(value).bitfield_value();
        return *this;
    }

    template<unsigned int lsb_pos, unsigned int length>
    type &reset()
    {
        data_ &= ~bitfield_member<T, lsb_pos, length>::bitfield_mask();
        return *this;
    }

    template<unsigned int lsb_pos, unsigned int length>
    type &flip()
    {
        data_ ^= bitfield_member<T, lsb_pos, length>::bitfield_mask();
        return *this;
    }

    template<unsigned int lsb_pos, unsigned int length>
    bitfield_member<T, lsb_pos, length> get() const
    {
        return bitfield_member<T, lsb_pos, length>(data_ >> lsb_pos);
    }

    template<unsigned int lsb_pos, unsigned int length>
    explicit operator bitfield_member<T, lsb_pos, length>() const {return get<lsb_pos, length>();}
    explicit operator T() const {return data_;}
    T value() const {return data_;}

    bool any() const {return data_ != 0;}
    bool none() const {return data_ == 0;}
    bool all() const {return data_ == mask;}

    static constexpr T min() {return 0;}
    static constexpr T max() {return mask;}

private:
    T data_;
};

#endif // BITFIELD_H
