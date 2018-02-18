#include "simd/generic_simd.h"
#include <iostream>

int main(int, char **)
{
#if 0
    try
    {
        std::vector<int> item;
        std::map<int, int> map;

        map = container_cast<decltype(map)>(item);
    }
    catch (std::exception &e)
    {
        std::cerr << e.what() << std::endl;
    }
#endif

    typedef simd_vector<unsigned, 8, signed> mvector;
    typedef simd_vector<unsigned, 8, unsigned> muvector;
    typedef simd_vector<unsigned, 16, unsigned> mu16vector;
    typedef simd_vector<unsigned, 32, float> mufloatvector;

#if 0
    std::cout << "element mask: " << mvector::scalar_mask() << "\n";
    std::cout << "element min: " << mvector::scalar_min() << "\n";
    std::cout << "element max: " << mvector::scalar_max() << "\n";
#endif

    mvector vec, vec2;

    vec.broadcast(vec.scalar_max());
    vec2.broadcast(vec.scalar_max());

    vec.sub(vec2, vec.math_keeplow);

#if 1
    vec.broadcast(-128);
    vec2.broadcast(-1);

    for (int i = 0; i < mvector::max_elements(); ++i)
    {
        std::cout << "vec1(" << i << "): " << vec.get(i) << "\n";
        std::cout << "vec2(" << i << "): " << vec2.get(i) << "\n";
    }

    //std::cout << std::hex << vec.vector() << std::endl;

    std::cout << "After addition:\n" << std::dec;

    vec.add(vec2, vec.math_keeplow);
    mu16vector uvec;// = vec.cast<16, unsigned>();
    mufloatvector fvec = vec.cast<32, float>();
    //vec = vec.fill_if_nonzero();

    for (unsigned i = 0; i < mvector::max_elements(); ++i)
    {
        std::cout << "vec1(" << i << "): " << vec.get(i) << "\n";
        std::cout << "vec2(" << i << "): " << vec2.get(i) << "\n";
        std::cout << "uvec(" << i << "): " << uvec.get(i) << "\n";
    }

    for (unsigned i = 0; i < mufloatvector::max_elements(); ++i)
        std::cout << "float(" << i << "): " << fvec.get(i) << "\n";

    //std::cout << std::hex << vec.vector() << std::endl;
#endif
}
