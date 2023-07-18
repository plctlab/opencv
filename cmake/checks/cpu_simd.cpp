#include <stdio.h>

#if __cplusplus >= 201703L && __has_include(<experimental/simd>)
#  include <experimental/simd>
#  define CV_SIMD 1
#endif

#if defined CV_SIMD
int test()
{
    namespace stdx = std::experimental;
    const float src[] = { 0.0f, 0.0f, 0.0f, 0.0f };
    stdx::native_simd<float> val(src, stdx::vector_aligned);
    return (int)stdx::reduce(val);
}
#else
#error "C++17 is not enabled or <experimental/simd> is not supported"
#endif

int main()
{
    printf("%d\n", test());
    return 0;
}