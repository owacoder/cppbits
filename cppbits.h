/*
 * cppbits.h
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

template<typename T, unsigned int element_bits, typename EffectiveType = T>
class simd_int_vector
{
    static_assert(std::is_unsigned<T>::value, "T must be an unsigned integral type");
    static_assert(element_bits, "Number of bits per element must be greater than zero");

    static constexpr bool elements_are_signed = std::is_signed<EffectiveType>::value;
    typedef simd_int_vector<T, element_bits, EffectiveType> type;
    typedef T value_type;
    static constexpr T ones = -1;
    static constexpr unsigned int elements = std::numeric_limits<T>::digits / element_bits;
    static constexpr T mask = ones >> (std::numeric_limits<T>::digits % element_bits);
    static constexpr T element_mask = ones >> (std::numeric_limits<T>::digits - element_bits);

    static_assert(elements, "Element size is too large for specified type");

    constexpr explicit simd_int_vector(T value) : data_(value) {}

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

    constexpr simd_int_vector() : data_{} {}

    template<typename NewType>
    constexpr simd_int_vector<T, element_bits, NewType> cast() const {return simd_int_vector<T, element_bits, NewType>::make_vector(data_);}

    template<unsigned int element_size, typename NewType>
    simd_int_vector<T, element_size, NewType> cast() const {return simd_int_vector<T, element_size, NewType>::make_vector(data_);}

    constexpr static type make_vector(T value) {return type(value);}
    constexpr static type make_scalar(EffectiveType value) {return type(T(value) & element_mask);}
    constexpr static type make_broadcast(EffectiveType value) {return type(expand_mask(T(value) & element_mask, element_bits, elements));}

    type &vector(T value) {return *this = make_vector(value);}
    type &scalar(EffectiveType value) {return *this = make_scalar(value);}
    type &broadcast(EffectiveType value) {return *this = make_broadcast(value);}

    constexpr T vector() const {return data_;}
    EffectiveType scalar() const {return get(0);}

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
    constexpr type operator|(const simd_int_vector &vec) const
    {
        return type(data_ | vec.vector());
    }

    /*
     * Logical AND's the entire vector with `vec` and returns it
     */
    constexpr type operator&(const simd_int_vector &vec) const
    {
        return type(data_ & vec.vector());
    }

    /*
     * Logical XOR's the entire vector with `vec` and returns it
     */
    constexpr type operator^(const simd_int_vector &vec) const
    {
        return type(data_ ^ vec.vector());
    }

    /*
     * Adds elements of `vec` to `this` (using rollover addition) and stores the result
     */
    type operator+(const simd_int_vector &vec) const
    {
        constexpr T add_mask = expand_mask(element_mask >> 1, element_bits, elements);
        return type(((data_ & add_mask) + (vec.vector() & add_mask)) ^ ((data_ ^ vec.vector()) & ~add_mask));
    }

    /*
     * Adds elements of `vec` to `this` (using specified math method) and stores the result
     */
    type &add(const simd_int_vector &vec, math_type math)
    {
        constexpr T add_mask = expand_mask(element_mask >> 1, element_bits, elements);
        switch (math) {
            default: /* Rollover arithmetic, math_keeplow */
                data_ = ((data_ & add_mask) + (vec.vector() & add_mask)) ^ ((data_ ^ vec.vector()) & ~add_mask);
                break;
            case math_saturate:
                if (elements_are_signed)
                {
                    const T temp = ((data_ & add_mask) + (vec.vector() & add_mask)) ^ ((data_ ^ vec.vector()) & ~add_mask);
                    const T overflow = ((data_ ^ temp) & (vec.vector() ^ temp) & ~add_mask) >> (element_bits - 1);
                    data_ = (temp & ((expand_mask(1, element_bits, elements) - overflow) * element_mask)) |
                            (overflow * (element_mask ^ (element_mask >> 1)) - ((temp >> (element_bits - 1)) & overflow));
                }
                else
                {
                    const T temp = ((data_ & add_mask) + (vec.vector() & add_mask)) ^ ((data_ ^ vec.vector()) & ~add_mask);
                    data_ = temp | ((((data_ & vec.vector()) | (~temp & (data_ | vec.vector()))) & ~add_mask) >> (element_bits - 1)) * element_mask;
                }
                break;
            case math_keephigh:
            {
                constexpr T overflow_mask = expand_mask(1, element_bits, elements);
                if (elements_are_signed)
                    data_ = ((((data_ & vec.vector()) | (((data_ & add_mask) + (vec.vector() & add_mask)) & (data_ ^ vec.vector()))) >> (element_bits - 1)) & overflow_mask) * element_mask;
                else
                    data_ = (((data_ & vec.vector()) | (((data_ & add_mask) + (vec.vector() & add_mask)) & (data_ ^ vec.vector()))) >> (element_bits - 1)) & overflow_mask;
                break;
            }
        }
        return *this;
    }

    /*
     * Subtracts elements of `vec` to `this` (using rollover subtraction) and stores the result
     */
    type operator-(const simd_int_vector &vec) const
    {
        constexpr T sub_mask = expand_mask(element_mask >> 1, element_bits, elements);
        return type(((data_ | ~sub_mask) - (vec.vector() & sub_mask)) ^ ((data_ ^ ~vec.vector()) & ~sub_mask));
    }

    type &sub(const simd_int_vector &vec, math_type math)
    {
        constexpr T sub_mask = expand_mask(element_mask >> 1, element_bits, elements);
        switch (math) {
            default: /* Rollover arithmetic, math_keeplow */
                data_ = ((data_ | ~sub_mask) - (vec.vector() & sub_mask)) ^ ((data_ ^ ~vec.vector()) & ~sub_mask);
                break;
            case math_saturate:
                if (elements_are_signed)
                {
                    const T temp = ((data_ | ~sub_mask) - (vec.vector() & sub_mask)) ^ ((data_ ^ ~vec.vector()) & ~sub_mask);
                    const T overflow = ((data_ ^ temp) & ~(vec.vector() ^ temp) & ~sub_mask) >> (element_bits - 1);
                    data_ = (temp & ((expand_mask(1, element_bits, elements) - overflow) * element_mask)) |
                            (overflow * (element_mask ^ (element_mask >> 1)) - ((temp >> (element_bits - 1)) & overflow));
                }
                else
                {
                    const T temp = ((data_ | ~sub_mask) - (vec.vector() & sub_mask)) ^ ((data_ ^ ~vec.vector()) & ~sub_mask);
                    data_ = temp & ((~((vec.vector() & temp) | (~data_ & (vec.vector() | temp))) & ~sub_mask) >> (element_bits - 1)) * element_mask;
                }
                break;
            case math_keephigh: /* TODO: not accurate right now */
                constexpr T overflow_mask = expand_mask(1, element_bits, elements);
                if (elements_are_signed)
                    data_ = (((((data_ | ~sub_mask) - (vec.vector() & sub_mask)) & (data_ ^ ~vec.vector())) >> (element_bits - 1)) & overflow_mask) * element_mask;
                else
                    data_ = ((((data_ | ~sub_mask) - (vec.vector() & sub_mask)) & (data_ ^ ~vec.vector())) >> (element_bits - 1)) & overflow_mask;
                break;
        }
        return *this;
    }

    /*
     * Negates elements of `this` and returns the resulting vector
     * Note that this function returns the value unmodified if EffectiveType is not a signed type
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

    template<unsigned int lsb_pos, unsigned int length>
    type &set()
    {
        data_ |= bitfield_member<T, lsb_pos, length>::bitfield_mask() & mask;
        return *this;
    }

    template<unsigned int idx>
    type &set(EffectiveType value)
    {
        data_ |= bitfield_member<T, idx * element_bits, element_bits>(T(value)).bitfield_value() & mask;
        return *this;
    }

    type &set(unsigned int idx, EffectiveType value)
    {
        const unsigned int shift = idx * element_bits;
        data_ = (data_ & ~(element_mask << shift)) | ((T(value) & element_mask) << shift);
        return *this;
    }

    type &set(unsigned int idx)
    {
        const unsigned int shift = idx * element_bits;
        data_ |= (element_mask << shift);
        return *this;
    }

    template<unsigned int idx>
    type &reset()
    {
        data_ &= ~bitfield_member<T, idx * element_bits, element_bits>::bitfield_mask();
        return *this;
    }

    type &reset(unsigned int idx)
    {
        const unsigned int shift = idx * element_bits;
        data_ &= ~(element_mask << shift);
        return *this;
    }

    template<unsigned int idx>
    type &flip()
    {
        data_ ^= bitfield_member<T, idx * element_bits, element_bits>::bitfield_mask();
        return *this;
    }

    type &flip(unsigned int idx)
    {
        const unsigned int shift = idx * element_bits;
        data_ ^= (element_mask << shift);
        return *this;
    }

    template<unsigned int idx>
    EffectiveType get() const
    {
        const unsigned int shift = idx * element_bits;
        if (elements_are_signed)
        {
            const T val = (data_ >> shift) & (element_mask >> 1);
            const T negative = (data_ >> (shift + element_bits - 1)) & 1;
            return negative? val == 0? scalar_min(): -EffectiveType((~val + 1) & (element_mask >> 1)): EffectiveType(val);
        }
        return EffectiveType((data_ >> shift) & element_mask);
    }

    EffectiveType get(unsigned int idx) const
    {
        const unsigned int shift = idx * element_bits;
        if (elements_are_signed)
        {
            const T val = (data_ >> shift) & (element_mask >> 1);
            const T negative = (data_ >> (shift + element_bits - 1)) & 1;
            return negative? val == 0? scalar_min(): -EffectiveType((~val + 1) & (element_mask >> 1)): EffectiveType(val);
        }
        return EffectiveType((data_ >> shift) & element_mask);
    }

    static constexpr T min() {return 0;}
    static constexpr T max() {return mask;}

    static constexpr T scalar_mask() {return element_mask;}
    static constexpr EffectiveType scalar_min() {return elements_are_signed? -EffectiveType(element_mask >> 1) - 1: EffectiveType(0);}
    static constexpr EffectiveType scalar_max() {return elements_are_signed? EffectiveType(element_mask >> 1): EffectiveType(element_mask);}
    static constexpr T vector_mask() {return mask;}

    static constexpr bool is_signed() {return elements_are_signed;}
    static constexpr unsigned int max_elements() {return elements;}
    static constexpr unsigned int element_size() {return element_bits;}

private:
    static constexpr T expand_mask(T mask, unsigned int mask_size, unsigned int level)
    {
        return level == 0? mask:
                           expand_mask(mask | ((mask & (ones >> (std::numeric_limits<T>::digits - mask_size))) <<
                                               ((level-1) * element_bits))
                                       , mask_size
                                       , level-1);
    }

    T data_;
};
