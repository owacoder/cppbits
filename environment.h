#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

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
#endif

#if defined __ia64 || defined __ia64__ || defined _IA64 || defined __IA64__ || defined _M_IA64 || defined __itanium__ || defined __x86_64 || defined __x86_64__
# define CPPBITS_X86_64
#endif

#include <iostream>

struct simd_error
{
    simd_error(const char *what) : what_(what) {}

    const char *what() const {return what_;}
    friend std::ostream &operator<<(std::ostream &os, simd_error e)
    {
        return os << "cppbits: " << e.what();
    }

private:
    const char *what_;
};

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
