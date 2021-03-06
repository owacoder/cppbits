#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#ifdef CPPBITS_OPTIONS
# error Define CPPBITS_ERROR_EXCEPTIONS if you want errors to be thrown as exceptions
# error Define CPPBITS_ERROR_PRINT_AND_TERMINATE if you want errors to be written to STDERR and the program terminated
# error Define CPPBITS_ERROR_TERMINATE if you want errors to terminate immediately
#endif

#ifdef __GNUC__
# define CPPBITS_GCC
#endif

#ifdef __clang__
# define CPPBITS_CLANG
#endif

#if defined i386 || defined __i386 || defined __i386__ || defined __IA32__ || defined _M_I86 || defined _M_IX86 || defined __X86__ || defined _X86_ || defined __THW_INTEL__ || defined __I86__ || defined __INTEL__ || defined __386
# define CPPBITS_X86
#
# ifdef __SSE__
#  define CPPBITS_SSE __SSE__
# endif
#
# ifdef __SSE2__
#  define CPPBITS_SSE2 __SSE2__
# endif
#
# ifdef __SSE3__
#  define CPPBITS_SSE3 __SSE3__
# endif
#
# ifdef __SSSE3__
#  define CPPBITS_SSSE3 __SSSE3__
# endif
#
# ifdef __SSE4_1__
#  define CPPBITS_SSE4_1 __SSE4_1__
# endif
#
# ifdef __SSE4_2__
#  define CPPBITS_SSE4_2 __SSE4_2__
# endif
#elif defined __ia64 || defined __ia64__ || defined _IA64 || defined __IA64__ || defined _M_IA64 || defined __itanium__ || defined __x86_64 || defined __x86_64__
# define CPPBITS_X86_64
#
# ifdef __SSE__
#  define CPPBITS_SSE __SSE__
# endif
#
# ifdef __SSE2__
#  define CPPBITS_SSE2 __SSE2__
# endif
#
# ifdef __SSE3__
#  define CPPBITS_SSE3 __SSE3__
# endif
#
# ifdef __SSSE3__
#  define CPPBITS_SSSE3 __SSSE3__
# endif
#
# ifdef __SSE4_1__
#  define CPPBITS_SSE4_1 __SSE4_1__
# endif
#
# ifdef __SSE4_2__
#  define CPPBITS_SSE4_2 __SSE4_2__
# endif
#endif

#if !defined CPPBITS_X86 && !defined CPPBITS_X86_64
# define CPPBITS_ONLY_GENERIC_SIMD
#endif

#include <iostream>

struct simd_error
{
    simd_error(std::string what) : what_(what) {}

    const char *what() const {return what_.c_str();}
    friend std::ostream &operator<<(std::ostream &os, simd_error e)
    {
        return os << "cppbits: " << e.what();
    }

private:
    std::string what_;
};

namespace impl
{
    template<typename T, bool is_signed>
    struct type_name {};
    template<bool is_signed>
    struct type_name<char, is_signed> {static constexpr const char *value = is_signed? "signed char": "unsigned char";};
    template<bool is_signed>
    struct type_name<signed char, is_signed> {static constexpr const char *value = is_signed? "signed char": "unsigned char";};
    template<bool is_signed>
    struct type_name<unsigned char, is_signed> {static constexpr const char *value = is_signed? "signed char": "unsigned char";};
    template<bool is_signed>
    struct type_name<signed short, is_signed> {static constexpr const char *value = is_signed? "signed short": "unsigned short";};
    template<bool is_signed>
    struct type_name<unsigned short, is_signed> {static constexpr const char *value = is_signed? "signed short": "unsigned short";};
    template<bool is_signed>
    struct type_name<signed int, is_signed> {static constexpr const char *value = is_signed? "signed int": "unsigned int";};
    template<bool is_signed>
    struct type_name<unsigned int, is_signed> {static constexpr const char *value = is_signed? "signed int": "unsigned int";};
    template<bool is_signed>
    struct type_name<signed long, is_signed> {static constexpr const char *value = is_signed? "signed long": "unsigned long";};
    template<bool is_signed>
    struct type_name<unsigned long, is_signed> {static constexpr const char *value = is_signed? "signed long": "unsigned long";};
    template<bool is_signed>
    struct type_name<signed long long, is_signed> {static constexpr const char *value = is_signed? "signed long long": "unsigned long long";};
    template<bool is_signed>
    struct type_name<unsigned long long, is_signed> {static constexpr const char *value = is_signed? "signed long long": "unsigned long long";};
    template<bool is_signed>
    struct type_name<float, is_signed> {static constexpr const char *value = "float";};
    template<bool is_signed>
    struct type_name<double, is_signed> {static constexpr const char *value = "double";};

    template<typename T, bool is_signed>
    struct double_type_name {};
    template<bool is_signed>
    struct double_type_name<char, is_signed> {static constexpr const char *value = is_signed? "signed short": "unsigned short";};
    template<bool is_signed>
    struct double_type_name<signed char, is_signed> {static constexpr const char *value = is_signed? "signed short": "unsigned short";};
    template<bool is_signed>
    struct double_type_name<unsigned char, is_signed> {static constexpr const char *value = is_signed? "signed short": "unsigned short";};
    template<bool is_signed>
    struct double_type_name<signed short, is_signed> {static constexpr const char *value = is_signed? "signed int": "unsigned int";};
    template<bool is_signed>
    struct double_type_name<unsigned short, is_signed> {static constexpr const char *value = is_signed? "signed int": "unsigned int";};
    template<bool is_signed>
    struct double_type_name<signed int, is_signed> {static constexpr const char *value = is_signed? "signed long": "unsigned long";};
    template<bool is_signed>
    struct double_type_name<unsigned int, is_signed> {static constexpr const char *value = is_signed? "signed long": "unsigned long";};
    template<bool is_signed>
    struct double_type_name<signed long, is_signed> {static constexpr const char *value = is_signed? "signed long long": "unsigned long long";};
    template<bool is_signed>
    struct double_type_name<unsigned long, is_signed> {static constexpr const char *value = is_signed? "signed long long": "unsigned long long";};
    template<bool is_signed>
    struct double_type_name<signed long long, is_signed> {static constexpr const char *value = is_signed? "signed long long": "unsigned long long";};
    template<bool is_signed>
    struct double_type_name<unsigned long long, is_signed> {static constexpr const char *value = is_signed? "signed long long": "unsigned long long";};
    template<bool is_signed>
    struct double_type_name<float, is_signed> {static constexpr const char *value = "float";};
    template<bool is_signed>
    struct double_type_name<double, is_signed> {static constexpr const char *value = "double";};

    template<unsigned int size>
    struct unsigned_int {};
    template<>
    struct unsigned_int<8> {typedef uint8_t type;};
    template<>
    struct unsigned_int<16> {typedef uint16_t type;};
    template<>
    struct unsigned_int<32> {typedef uint32_t type;};
    template<>
    struct unsigned_int<64> {typedef uint64_t type;};

    template<typename To, typename From, unsigned int alignment = (alignof(From) > alignof(To)? alignof(From): alignof(To))>
    To type_punning_cast(From from)
    {
        union alignas(alignment) packed_union
        {
            From from;
            To to;
        } u;

        u.from = from;
        return u.to;
    }
}

namespace cppbits
{
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

    enum shuffle_type
    {
        shuffle_0_same = 0x0, /* Keep element 0 */
        shuffle_1_to_0 = 0x1, /* Shuffle element 1 to element 0 */
        shuffle_2_to_0 = 0x2, /* Shuffle element 2 to element 0 */
        shuffle_3_to_0 = 0x3, /* Shuffle element 3 to element 0 */
        shuffle_0_to_1 = 0x0 << 2, /* Shuffle element 0 to element 1 */
        shuffle_1_same = 0x1 << 2, /* Keep element 1 */
        shuffle_2_to_1 = 0x2 << 2, /* Shuffle element 2 to element 1 */
        shuffle_3_to_1 = 0x3 << 2, /* Shuffle element 3 to element 1 */
        shuffle_0_to_2 = 0x0 << 4, /* Shuffle element 0 to element 2 */
        shuffle_1_to_2 = 0x1 << 4, /* Shuffle element 1 to element 2 */
        shuffle_2_same = 0x2 << 4, /* Keep element 2 */
        shuffle_3_to_2 = 0x3 << 4, /* Shuffle element 3 to element 2 */
        shuffle_0_to_3 = 0x0 << 6, /* Shuffle element 0 to element 3 */
        shuffle_1_to_3 = 0x1 << 6, /* Shuffle element 1 to element 3 */
        shuffle_2_to_3 = 0x2 << 6, /* Shuffle element 2 to element 3 */
        shuffle_3_same = 0x3 << 6, /* Shuffle element 3 to element 3 */

        shuffle_same = shuffle_0_same | shuffle_1_same | shuffle_2_same | shuffle_3_same, /* Keep all elements identical */
        shuffle_broadcast_0 = shuffle_0_same | shuffle_0_to_1 | shuffle_0_to_2 | shuffle_0_to_3, /* Broadcast element 0 to all elements */
        shuffle_broadcast_1 = shuffle_1_to_0 | shuffle_1_same | shuffle_1_to_2 | shuffle_1_to_3, /* Broadcast element 1 to all elements */
        shuffle_broadcast_2 = shuffle_2_to_0 | shuffle_2_to_1 | shuffle_2_same | shuffle_2_to_3, /* Broadcast element 2 to all elements */
        shuffle_broadcast_3 = shuffle_3_to_0 | shuffle_3_to_1 | shuffle_3_to_2 | shuffle_3_same /* Broadcast element 3 to all elements */
    };
}

#if !defined CPPBITS_ERROR_EXCEPTIONS && !defined CPPBITS_ERROR_PRINT_AND_TERMINATE && !defined CPPBITS_ERROR_TERMINATE
#define CPPBITS_ERROR_EXCEPTIONS
#endif

#if defined CPPBITS_ERROR_EXCEPTIONS
#define CPPBITS_ERROR(x) throw simd_error(x)
#elif defined CPPBITS_ERROR_PRINT_AND_TERMINATE
#include <cstdlib>
#define CPPBITS_ERROR(x) do {std::cerr << "cppbits: " << x << std::endl; std::terminate();} while (0)
#elif defined CPPBITS_ERROR_TERMINATE
#include <cstdlib>
#define CPPBITS_ERROR(x) std::terminate()
#endif

#endif // ENVIRONMENT_H
