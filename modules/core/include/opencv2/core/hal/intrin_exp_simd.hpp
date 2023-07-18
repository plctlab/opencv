// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// The original implementation has been contributed by Yaossg.


#ifndef OPENCV_HAL_INTRIN_SIMD_HPP
#define OPENCV_HAL_INTRIN_SIMD_HPP

namespace cv
{

CV_CPU_OPTIMIZATION_HAL_NAMESPACE_BEGIN

#define CV_SIMD_SCALABLE 1
#define CV_SIMD_SCALABLE_64F 1

namespace stdx = std::experimental;

using v_uint8 =  stdx::native_simd<uint8_t>;
using v_int8 =   stdx::native_simd<int8_t>;
using v_uint16 = stdx::native_simd<uint16_t>;
using v_int16 =  stdx::native_simd<int16_t>;
using v_uint32 = stdx::native_simd<uint32_t>;
using v_int32 =  stdx::native_simd<int32_t>;
using v_uint64 = stdx::native_simd<uint64_t>;
using v_int64 =  stdx::native_simd<int64_t>;

using v_float32 = stdx::native_simd<float>;
#if CV_SIMD_SCALABLE_64F
using v_float64 = stdx::native_simd<double>;
#endif

using uchar = uint8_t;
using schar = int8_t;
using ushort = uint16_t;
using uint = uint32_t;
using uint64 = uint64_t;
using int64 = int64_t;

template <class T>
struct VTraits;

template <class TYPE>
struct VTraits<stdx::native_simd<TYPE>> {
    static constexpr inline int vlanes() { return stdx::native_simd<TYPE>::size(); }
    using lane_type = TYPE;
    enum { nlanes = vlanes(), max_nlanes = vlanes() };
};

#define OPENCV_HAL_IMPL_INIT(_Tpvec, _Tp, suffix) \
inline v_##_Tpvec v_setzero_##suffix() \
{ \
    return 0; \
} \
inline v_##_Tpvec v_setall_##suffix(_Tp v) \
{ \
    return v; \
}

OPENCV_HAL_IMPL_INIT(uint8, uchar, u8)
OPENCV_HAL_IMPL_INIT(int8, schar, s8)
OPENCV_HAL_IMPL_INIT(uint16, ushort, u16)
OPENCV_HAL_IMPL_INIT(int16, short, s16)
OPENCV_HAL_IMPL_INIT(uint32, uint, u32)
OPENCV_HAL_IMPL_INIT(int32, int, s32)
OPENCV_HAL_IMPL_INIT(uint64, uint64, u64)
OPENCV_HAL_IMPL_INIT(int64, int64, s64)

OPENCV_HAL_IMPL_INIT(float32, float, f32)
#if CV_SIMD_SCALABLE_64F
OPENCV_HAL_IMPL_INIT(float64, double, f64)
#endif

template<typename T>
inline auto v_load(const T* ptr) {
    return stdx::native_simd<T>(ptr, stdx::element_aligned);
}

template<typename T>
inline auto v_load_aligned(const T* ptr) {
    return stdx::native_simd<T>(ptr, stdx::vector_aligned);
}

template<typename T>
inline auto v_load_low(const T* ptr) {
    stdx::native_simd<T> c;
    for (size_t i = 0; i < c.size() / 2; ++i) {
        c[i] = ptr[i];
    }
    return c;
}

template<typename T>
inline auto v_load_halves(const T* loptr, const T* hiptr) {
    stdx::native_simd<T> c;
    for (size_t i = 0; i < c.size() / 2; ++i) {
        c[i] = loptr[i];
        c[i + c.size() / 2] = hiptr[i];
    }
    return c;
}

template<typename T>
inline auto v_load_expand(const T* ptr) {
    return stdx::native_simd<typename V_TypeTraits<T>::w_type>
            ([ptr](size_t i){ return ptr[i]; });
}

// FP16 support
inline auto v_load_expand(const float16_t* ptr) {
    return stdx::native_simd<float>
            ([ptr](size_t i){ return ptr[i]; });
}

template<typename T>
inline auto v_load_expand_q(const T* ptr) {
    return stdx::native_simd<typename V_TypeTraits<T>::q_type>
            ([ptr](size_t i){ return ptr[i]; });
}

template<typename T>
inline auto v_lut(const T* tab, const int* idx) {
    return stdx::native_simd<T>
            ([tab, idx](size_t i){ return tab[idx[i]]; });
}

template<typename T>
inline auto v_lut_pairs(const T* tab, const int* idx) {
    return stdx::native_simd<T>
            ([tab, idx](size_t i){ return tab[idx[i / 2] + i % 2]; });
}

template<typename T>
inline auto v_lut_quads(const T* tab, const int* idx) {
    return stdx::native_simd<T>
            ([tab, idx](size_t i){ return tab[idx[i / 4] + i % 4]; });
}

inline void v_cleanup() {}

template<typename T>
inline void v_store(T* dst, stdx::native_simd<T> v) {
    v.template copy_to(dst, stdx::element_aligned);
}

template<typename T>
inline void v_store_aligned(T* dst, stdx::native_simd<T> v) {
    v.template copy_to(dst, stdx::vector_aligned);
}

CV_CPU_OPTIMIZATION_HAL_NAMESPACE_END
}

#endif //OPENCV_HAL_INTRIN_SIMD_HPP
