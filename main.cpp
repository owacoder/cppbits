#define CPPBITS_ERROR_PRINT_AND_TERMINATE

#include "cppbits.h"
#include <iostream>

#include <typeinfo>

template<typename T, unsigned int size, typename Effective>
void test_addition_low()
{
    typedef cppbits::default_simd_vector<0, T, size, Effective> type;
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

    vec1 = vec1.add(vec2, cppbits::math_keeplow);

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
    typedef cppbits::default_simd_vector<0, T, size, Effective> type;
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

    vec1 = vec1.sub(vec2, cppbits::math_keeplow);

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
    typedef cppbits::default_simd_vector<0, T, size, Effective> type;
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

    vec1 = vec1.mul(vec2, cppbits::math_keeplow);

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
    typedef cppbits::default_simd_vector<0, T, size, Effective> type;
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

    vec1 = vec1.div(vec2, cppbits::math_keeplow);

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
    try
    {
        cppbits::opencl_simd_vector<50, 32, float> vec;

        for (size_t i = 0; i < vec.max_elements(); ++i)
            vec.set(i, i*7);

        std::cout << vec << std::endl;

        vec = vec.add(vec, cppbits::math_keeplow);
        vec = vec.add(vec, cppbits::math_keeplow);
        vec = vec.add(vec, cppbits::math_keeplow);
        //vec = vec.avg(vec.make_broadcast(37));
        //vec = vec.negate().abs();

        std::cout << vec << std::endl;
        //return 0;

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

        typedef float underlying_type;
        typedef cppbits::opencl_simd_vector<100000, 32, underlying_type> vector;
        constexpr unsigned long long iters = 1000;
        constexpr unsigned long long size = vector::max_elements() * 2;
        underlying_type *data = new underlying_type[size];
        underlying_type *data2 = new underlying_type[size];
        underlying_type *dest = new underlying_type[size];
        underlying_type *dest2 = new underlying_type[size];

        for (unsigned i = 0; i < size; ++i)
            data[i] = rand(), data2[i] = rand();

        clock_t first = clock();
        for (unsigned long long j = 0; j < size; ++j)
        {
            dest[j] = data[j];
            for (unsigned long long i = 0; i < iters; ++i)
                dest[j] = sqrt(((data2[j] + dest[j] + 1) / 2));
        }
        double first_result = ((double) clock() - first) / CLOCKS_PER_SEC;
        std::cout << "\nFirst computation took: " << first_result << "\n";

        vector a, b;

        std::cout << "aligned? " << vector::ptr_is_aligned(data+1) << "\n";

        clock_t second = clock();
        for (unsigned long long j = 0; j < size; j += vector::max_elements())
        {
            a.load_unpacked(data + j);
            b.load_unpacked(data2 + j);
            for (unsigned long long i = 0; i < iters; ++i)
                a = b.avg(a).sqrt(cppbits::math_accurate);
            a.dump_unpacked(dest2 + j);
        }
        double second_result = ((double) clock() - second) / CLOCKS_PER_SEC;
        std::cout << "Second computation took: " << second_result << "\n";
        std::cout << "SIMD was " << (first_result / second_result) << " times faster\n\n";

        for (unsigned i = 0; i < vector::max_elements(); ++i)
            if (dest[i] != dest2[i])
            {
                std::cerr << "Invalid result " << dest2[i] << " at element " << i << ": expected " << dest[i] << "\n";
                std::cerr << "Difference: " << (dest2[i] - dest[i]) << "\n";
                return 1;
            }

        return 0;
    } catch (simd_error e) {
        std::cerr << e << std::endl;
    }
}
