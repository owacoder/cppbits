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
#endif

#if defined __ia64 || defined __ia64__ || defined _IA64 || defined __IA64__ || defined _M_IA64 || defined __itanium__ || defined __x86_64 || defined __x86_64__
# define CPPBITS_X86_64
#endif

#endif // ENVIRONMENT_H
