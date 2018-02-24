#define CPPBITS_ERROR_PRINT_AND_TERMINATE

#include "simd/generic_simd.h"
#include "simd/x86_simd.h"
#include <iostream>

#include <typeinfo>

template<typename T, unsigned int size, typename Effective>
void test_addition_low()
{
    typedef simd_vector<T, size, Effective> type;
    Effective src1[type::max_elements()];
    Effective src2[type::max_elements()];
    Effective dst[type::max_elements()];

    type vec1, vec2, vdst;

    for (unsigned i = 0; i < vec1.max_elements(); ++i)
    {
        src1[i] = rand();
        src2[i] = rand();
        dst[i] = src1[i] + src2[i];

        vec1.set(i, src1[i]);
        vec2.set(i, src2[i]);
        vdst.set(i, dst[i]);
    }

    vec1 = vec1.add(vec2, vec1.math_keeplow);

    for (unsigned i = 0; i < vec1.max_elements(); ++i)
    {
        if (vec1.get(i) != vdst.get(i))
        {
            std::cerr << "test_addition_low " << size << " failed: expected " << vdst.get(i) << " but got " << vec1.get(i) << std::endl;
            return;
        }
    }

    std::cout << "test_addition_low " << size << " passed!\n";
}

template<typename T, unsigned int size, typename Effective>
void test_subtraction_low()
{
    typedef simd_vector<T, size, Effective> type;
    Effective src1[type::max_elements()];
    Effective src2[type::max_elements()];
    Effective dst[type::max_elements()];

    type vec1, vec2, vdst;

    for (unsigned i = 0; i < vec1.max_elements(); ++i)
    {
        src1[i] = rand();
        src2[i] = rand();
        dst[i] = src1[i] - src2[i];

        vec1.set(i, src1[i]);
        vec2.set(i, src2[i]);
        vdst.set(i, dst[i]);
    }

    vec1 = vec1.sub(vec2, vec1.math_keeplow);

    for (unsigned i = 0; i < vec1.max_elements(); ++i)
    {
        if (vec1.get(i) != vdst.get(i))
        {
            std::cerr << "test_subtraction_low " << size << " failed: expected " << vdst.get(i) << " but got " << vec1.get(i) << std::endl;
            return;
        }
    }

    std::cout << "test_subtraction_low " << size << " passed!\n";
}

template<typename T, unsigned int size, typename Effective>
void test_multiplication_low()
{
    typedef simd_vector<T, size, Effective> type;
    Effective src1[type::max_elements()];
    Effective src2[type::max_elements()];
    Effective dst[type::max_elements()];

    type vec1, vec2, vdst;

    for (unsigned i = 0; i < vec1.max_elements(); ++i)
    {
        src1[i] = rand() & vec1.scalar_mask();
        src2[i] = rand() & vec1.scalar_mask();
        dst[i] = src1[i] * src2[i];

        vec1.set(i, src1[i]);
        vec2.set(i, src2[i]);
        vdst.set(i, dst[i]);
    }

    vec1 = vec1.mul(vec2, vec1.math_keeplow);

    for (unsigned i = 0; i < vec1.max_elements(); ++i)
    {
        if (vec1.get(i) != vdst.get(i))
        {
            std::cerr << "test_multiplication_low " << size << " failed: expected " << vdst.get(i) << " but got " << vec1.get(i) << std::endl;
            return;
        }
    }

    std::cout << "test_multiplication_low " << size << " passed!\n";
}

template<typename T, unsigned int size, typename Effective>
void test_division_low()
{
    typedef simd_vector<T, size, Effective> type;
    Effective src1[type::max_elements()];
    Effective src2[type::max_elements()];
    Effective dst[type::max_elements()];

    type vec1, vec2, vdst;

    for (unsigned i = 0; i < vec1.max_elements(); ++i)
    {
        src1[i] = rand() & vec1.scalar_mask();
        do src2[i] = rand() & vec1.scalar_mask(); while (src2[i] == 0);
        dst[i] = src1[i] / src2[i];

        vec1.set(i, src1[i]);
        vec2.set(i, src2[i]);
        vdst.set(i, dst[i]);
    }

    vec1 = vec1.div(vec2, vec1.math_keeplow);

    for (unsigned i = 0; i < vec1.max_elements(); ++i)
    {
        if (vec1.get(i) != vdst.get(i))
        {
            std::cerr << "test_division_low " << size << " failed: expected " << vdst.get(i) << " but got " << vec1.get(i) << std::endl;
            return;
        }
    }

    std::cout << "test_division_low " << size << " passed!\n";
}

#include <ctime>

int main(int, char **)
{
    srand(std::time(NULL));

    test_addition_low<uint8_t, 3, uint8_t>();

    test_addition_low<uint8_t, 8, uint8_t>();
    test_addition_low<uint8_t, 8, int8_t>();
    test_addition_low<uint16_t, 8, uint8_t>();
    test_addition_low<uint16_t, 8, int8_t>();
    test_addition_low<uint32_t, 8, uint8_t>();
    test_addition_low<uint32_t, 8, int8_t>();
    test_addition_low<uint64_t, 8, uint8_t>();
    test_addition_low<uint64_t, 8, int8_t>();

    test_addition_low<uint16_t, 16, uint16_t>();
    test_addition_low<uint16_t, 16, int16_t>();
    test_addition_low<uint32_t, 16, uint16_t>();
    test_addition_low<uint32_t, 16, int16_t>();
    test_addition_low<uint64_t, 16, uint16_t>();
    test_addition_low<uint64_t, 16, int16_t>();

    test_addition_low<uint32_t, 32, uint32_t>();
    test_addition_low<uint32_t, 32, int32_t>();
    test_addition_low<uint64_t, 32, uint32_t>();
    test_addition_low<uint64_t, 32, int32_t>();
    test_addition_low<uint32_t, 32, float>();
    test_addition_low<uint64_t, 32, float>();

    test_addition_low<uint64_t, 64, uint64_t>();
    test_addition_low<uint64_t, 64, int64_t>();
    test_addition_low<uint64_t, 64, double>();

    test_subtraction_low<uint8_t, 8, uint8_t>();
    test_subtraction_low<uint8_t, 8, int8_t>();
    test_subtraction_low<uint16_t, 8, uint8_t>();
    test_subtraction_low<uint16_t, 8, int8_t>();
    test_subtraction_low<uint32_t, 8, uint8_t>();
    test_subtraction_low<uint32_t, 8, int8_t>();
    test_subtraction_low<uint64_t, 8, uint8_t>();
    test_subtraction_low<uint64_t, 8, int8_t>();

    test_subtraction_low<uint16_t, 16, uint16_t>();
    test_subtraction_low<uint16_t, 16, int16_t>();
    test_subtraction_low<uint32_t, 16, uint16_t>();
    test_subtraction_low<uint32_t, 16, int16_t>();
    test_subtraction_low<uint64_t, 16, uint16_t>();
    test_subtraction_low<uint64_t, 16, int16_t>();

    test_subtraction_low<uint32_t, 32, uint32_t>();
    test_subtraction_low<uint32_t, 32, int32_t>();
    test_subtraction_low<uint64_t, 32, uint32_t>();
    test_subtraction_low<uint64_t, 32, int32_t>();
    test_subtraction_low<uint32_t, 32, float>();
    test_subtraction_low<uint64_t, 32, float>();

    test_subtraction_low<uint64_t, 64, uint64_t>();
    test_subtraction_low<uint64_t, 64, int64_t>();
    test_subtraction_low<uint64_t, 64, double>();

    test_multiplication_low<uint8_t, 3, uint8_t>();

    test_multiplication_low<uint8_t, 8, uint8_t>();
    test_multiplication_low<uint8_t, 8, int8_t>();
    test_multiplication_low<uint16_t, 8, uint8_t>();
    test_multiplication_low<uint16_t, 8, int8_t>();
    test_multiplication_low<uint32_t, 8, uint8_t>();
    test_multiplication_low<uint32_t, 8, int8_t>();
    test_multiplication_low<uint64_t, 8, uint8_t>();
    test_multiplication_low<uint64_t, 8, int8_t>();

    test_multiplication_low<uint16_t, 16, uint16_t>();
    test_multiplication_low<uint16_t, 16, int16_t>();
    test_multiplication_low<uint32_t, 16, uint16_t>();
    test_multiplication_low<uint32_t, 16, int16_t>();
    test_multiplication_low<uint64_t, 16, uint16_t>();
    test_multiplication_low<uint64_t, 16, int16_t>();

    test_multiplication_low<uint32_t, 32, uint32_t>();
    test_multiplication_low<uint32_t, 32, int32_t>();
    test_multiplication_low<uint64_t, 32, uint32_t>();
    test_multiplication_low<uint64_t, 32, int32_t>();
    test_multiplication_low<uint32_t, 32, float>();
    test_multiplication_low<uint64_t, 32, float>();

    test_multiplication_low<uint64_t, 64, uint64_t>();
    test_multiplication_low<uint64_t, 64, int64_t>();
    test_multiplication_low<uint64_t, 64, double>();

#if 0
    test_division_low<uint8_t, 3, uint8_t>();

    test_division_low<uint8_t, 8, uint8_t>();
    test_division_low<uint8_t, 8, int8_t>();
    test_division_low<uint16_t, 8, uint8_t>();
    test_division_low<uint16_t, 8, int8_t>();
    test_division_low<uint32_t, 8, uint8_t>();
    test_division_low<uint32_t, 8, int8_t>();
    test_division_low<uint64_t, 8, uint8_t>();
    test_division_low<uint64_t, 8, int8_t>();

    test_division_low<uint16_t, 16, uint16_t>();
    test_division_low<uint16_t, 16, int16_t>();
    test_division_low<uint32_t, 16, uint16_t>();
    test_division_low<uint32_t, 16, int16_t>();
    test_division_low<uint64_t, 16, uint16_t>();
    test_division_low<uint64_t, 16, int16_t>();

    test_division_low<uint32_t, 32, uint32_t>();
    test_division_low<uint32_t, 32, int32_t>();
    test_division_low<uint64_t, 32, uint32_t>();
    test_division_low<uint64_t, 32, int32_t>();
    test_division_low<uint32_t, 32, float>();
    test_division_low<uint64_t, 32, float>();

    test_division_low<uint64_t, 64, uint64_t>();
    test_division_low<uint64_t, 64, int64_t>();
    test_division_low<uint64_t, 64, double>();
#endif
}
