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
inline void v_store(T* dst, const stdx::native_simd<T>& v,
                    hal::StoreMode /*mode*/=hal::STORE_UNALIGNED) {
    v.template copy_to(dst, stdx::element_aligned);
}

template<typename T>
inline void v_store_aligned(T* dst, const stdx::native_simd<T>& v) {
    v.template copy_to(dst, stdx::vector_aligned);
}

template<typename T>
inline auto v_get0(const stdx::native_simd<T>& v) {
    return v[0];
}

template<typename T>
inline auto v_load(std::initializer_list<T> v) {
    return v_load(v.begin());
}

template<typename T>
inline void v_load_deinterleave(const T* ptr,
                                stdx::native_simd<T>& a,
                                stdx::native_simd<T>& b) {
    a = stdx::native_simd<T>([ptr](size_t i){ return ptr[2 * i]; });
    b = stdx::native_simd<T>([ptr](size_t i){ return ptr[2 * i + 1]; });
}

template<typename T>
inline void v_load_deinterleave(const T* ptr,
                                stdx::native_simd<T>& a,
                                stdx::native_simd<T>& b,
                                stdx::native_simd<T>& c) {
    a = stdx::native_simd<T>([ptr](size_t i){ return ptr[3 * i]; });
    b = stdx::native_simd<T>([ptr](size_t i){ return ptr[3 * i + 1]; });
    c = stdx::native_simd<T>([ptr](size_t i){ return ptr[3 * i + 2]; });
}

template<typename T>
inline void v_load_deinterleave(const T* ptr,
                                stdx::native_simd<T>& a,
                                stdx::native_simd<T>& b,
                                stdx::native_simd<T>& c,
                                stdx::native_simd<T>& d) {
    a = stdx::native_simd<T>([ptr](size_t i){ return ptr[4 * i]; });
    b = stdx::native_simd<T>([ptr](size_t i){ return ptr[4 * i + 1]; });
    c = stdx::native_simd<T>([ptr](size_t i){ return ptr[4 * i + 2]; });
    d = stdx::native_simd<T>([ptr](size_t i){ return ptr[4 * i + 3]; });
}

template<typename T>
inline void v_store_interleave(T* ptr,
                               const stdx::native_simd<T>& a,
                               const stdx::native_simd<T>& b,
                               hal::StoreMode /*mode*/=hal::STORE_UNALIGNED) {
    size_t i, i2;
    for(i = i2 = 0; i < a.size(); i++, i2 += 2) {
        ptr[i2] = a[i];
        ptr[i2 + 1] = b[i];
    }
}

template<typename T>
inline void v_store_interleave(T* ptr,
                               const stdx::native_simd<T>& a,
                               const stdx::native_simd<T>& b,
                               const stdx::native_simd<T>& c,
                               hal::StoreMode /*mode*/=hal::STORE_UNALIGNED) {
    size_t i, i2;
    for(i = i2 = 0; i < a.size(); i++, i2 += 3) {
        ptr[i2] = a[i];
        ptr[i2 + 1] = b[i];
        ptr[i2 + 2] = c[i];
    }
}

template<typename T>
inline void v_store_interleave(T* ptr,
                               const stdx::native_simd<T>& a,
                               const stdx::native_simd<T>& b,
                               const stdx::native_simd<T>& c,
                               const stdx::native_simd<T>& d,
                               hal::StoreMode /*mode*/=hal::STORE_UNALIGNED) {
    size_t i, i2;
    for(i = i2 = 0; i < a.size(); i++, i2 += 4) {
        ptr[i2] = a[i];
        ptr[i2 + 1] = b[i];
        ptr[i2 + 2] = c[i];
        ptr[i2 + 3] = d[i];
    }
}

template<typename T>
inline auto v_interleave_pairs(const stdx::native_simd<T>& v) {
    constexpr size_t mapping[] = {0, 2, 1, 3};
    return stdx::native_simd<T>([&](size_t i) { return v[(i & ~3) + mapping[i % 4]]; });
}

template<typename T>
inline auto v_interleave_quads(const stdx::native_simd<T>& v) {
    constexpr size_t mapping[] = {0, 4, 1, 5, 2, 6, 3, 7};
    return stdx::native_simd<T>([&](size_t i) { return v[(i & ~7) + mapping[i % 8]]; });
}

template<typename T>
inline auto v_zip(const stdx::native_simd<T>& a0,
                  const stdx::native_simd<T>& a1,
                  stdx::native_simd<T>& b0,
                  stdx::native_simd<T>& b1) {
    b0 = stdx::native_simd<T>([&](size_t i) { return (i & 1) ? a1[i >> 1] : a0[i >> 1]; });
    b1 = stdx::native_simd<T>([&](size_t i) { return (i & 1) ? a1[(i >> 1) + a1.size() / 2] : a0[(i >> 1) + a0.size() / 2]; });
}

template<typename T>
inline auto v_combine_low(const stdx::native_simd<T>& a,
                          const stdx::native_simd<T>& b) {
    return stdx::native_simd<T>([&](size_t i) { return i < a.size() / 2 ? a[i] : b[i - a.size() / 2]; });
}

template<typename T>
inline auto v_combine_high(const stdx::native_simd<T>& a,
                           const stdx::native_simd<T>& b) {
    return stdx::native_simd<T>([&](size_t i) { return i < a.size() / 2 ? a[a.size() / 2 + i] : b[i]; });
}

template<typename T>
inline auto v_recombine(const stdx::native_simd<T>& a0,
                       const stdx::native_simd<T>& a1,
                       stdx::native_simd<T>& b0,
                       stdx::native_simd<T>& b1) {
    b0 = v_combine_low(a0, a1);
    b1 = v_combine_high(a0, a1);
}

// NOTE: stdx::split_by is not available yet in current implementation of <experimental/simd>
// Use ordinary loop instead.
template<typename T>
inline void v_store_low(T* dst, const stdx::native_simd<T>& v) {
    for(size_t i = 0; i < v.size() / 2; ++i)
        dst[i] = v[i];
}

template<typename T>
inline void v_store_high(T* dst, const stdx::native_simd<T>& v) {
    for(size_t i = 0; i < v.size() / 2; ++i)
        dst[i] = v[i + v.size() / 2];
}

template<typename T>
inline void v_expand(const stdx::native_simd<T>& a,
                     stdx::native_simd<typename V_TypeTraits<T>::w_type>& b0,
                     stdx::native_simd<typename V_TypeTraits<T>::w_type>& b1) {
    typedef typename V_TypeTraits<T>::w_type result_type;
    b0 = stdx::native_simd<result_type>([&](size_t i) { return a[i]; });
    b1 = stdx::native_simd<result_type>([&](size_t i) { return a[i + a.size() / 2]; });
}

template<typename T>
inline auto v_expand_low(const stdx::native_simd<T>& a) {
    return stdx::native_simd<typename V_TypeTraits<T>::w_type>([&](size_t i) { return a[i]; });
}


template<typename T>
inline auto v_expand_high(const stdx::native_simd<T>& a) {
    return stdx::native_simd<typename V_TypeTraits<T>::w_type>([&](size_t i) { return a[i + a.size() / 2]; });
}


#define OPENCV_HAL_IMPL_C_PACK(_Tp, _Tpn, pack_suffix, cast) \
inline auto v_##pack_suffix(const stdx::native_simd<_Tp>& a, const stdx::native_simd<_Tp>& b) { \
    return stdx::native_simd<_Tpn>([a, b](size_t i) { return cast<_Tpn>(i < a.size() ? a[i] : b[i - a.size()]); });   \
}
OPENCV_HAL_IMPL_C_PACK(ushort, uchar, pack, saturate_cast)
OPENCV_HAL_IMPL_C_PACK(short, schar, pack, saturate_cast)
OPENCV_HAL_IMPL_C_PACK(unsigned, ushort, pack, saturate_cast)
OPENCV_HAL_IMPL_C_PACK(int, short, pack, saturate_cast)
OPENCV_HAL_IMPL_C_PACK(uint64, unsigned, pack, static_cast)
OPENCV_HAL_IMPL_C_PACK(int64, int, pack, static_cast)
OPENCV_HAL_IMPL_C_PACK(short, uchar, pack_u, saturate_cast)
OPENCV_HAL_IMPL_C_PACK(int, ushort, pack_u, saturate_cast)


#define OPENCV_HAL_IMPL_C_REINTERPRET(_Tp, suffix) \
template<typename _Tp0> inline stdx::native_simd<_Tp> \
    v_reinterpret_as_##suffix(const stdx::native_simd<_Tp0>& a) \
{ return reinterpret_cast<const stdx::native_simd<_Tp>&>(a); }

OPENCV_HAL_IMPL_C_REINTERPRET(uchar, u8)
OPENCV_HAL_IMPL_C_REINTERPRET(schar, s8)
OPENCV_HAL_IMPL_C_REINTERPRET(ushort, u16)
OPENCV_HAL_IMPL_C_REINTERPRET(short, s16)
OPENCV_HAL_IMPL_C_REINTERPRET(unsigned, u32)
OPENCV_HAL_IMPL_C_REINTERPRET(int, s32)
OPENCV_HAL_IMPL_C_REINTERPRET(float, f32)
OPENCV_HAL_IMPL_C_REINTERPRET(double, f64)
OPENCV_HAL_IMPL_C_REINTERPRET(uint64, u64)
OPENCV_HAL_IMPL_C_REINTERPRET(int64, s64)

#define OPENCV_HAL_IMPL_C_RSHR_PACK(_Tp, _Tpn, pack_suffix, cast) \
template<int shift>                                               \
inline auto v_rshr_##pack_suffix(const stdx::native_simd<_Tp>& a, const stdx::native_simd<_Tp>& b) { \
    return stdx::native_simd<_Tpn>([a, b](size_t i) {             \
        return cast<_Tpn>(((i < a.size() ? a[i] : b[i - a.size()]) + ((_Tp)1 << (shift - 1))) >> shift);     \
    }); \
}

OPENCV_HAL_IMPL_C_RSHR_PACK(ushort, uchar, pack, saturate_cast)
OPENCV_HAL_IMPL_C_RSHR_PACK(short, schar, pack, saturate_cast)
OPENCV_HAL_IMPL_C_RSHR_PACK(unsigned, ushort, pack, saturate_cast)
OPENCV_HAL_IMPL_C_RSHR_PACK(int, short, pack, saturate_cast)
OPENCV_HAL_IMPL_C_RSHR_PACK(uint64, unsigned, pack, static_cast)
OPENCV_HAL_IMPL_C_RSHR_PACK(int64, int, pack, static_cast)
OPENCV_HAL_IMPL_C_RSHR_PACK(short, uchar, pack_u, saturate_cast)
OPENCV_HAL_IMPL_C_RSHR_PACK(int, ushort, pack_u, saturate_cast)

#define OPENCV_HAL_IMPL_C_PACK_STORE(_Tp, _Tpn, pack_suffix, cast) \
inline void v_##pack_suffix##_store(_Tpn* ptr, const stdx::native_simd<_Tp>& a) { \
    for (size_t i = 0; i < a.size(); ++i) \
        ptr[i] = cast<_Tpn>(a[i]); \
}

OPENCV_HAL_IMPL_C_PACK_STORE(ushort, uchar, pack, saturate_cast)
OPENCV_HAL_IMPL_C_PACK_STORE(short, schar, pack, saturate_cast)
OPENCV_HAL_IMPL_C_PACK_STORE(unsigned, ushort, pack, saturate_cast)
OPENCV_HAL_IMPL_C_PACK_STORE(int, short, pack, saturate_cast)
OPENCV_HAL_IMPL_C_PACK_STORE(uint64, unsigned, pack, static_cast)
OPENCV_HAL_IMPL_C_PACK_STORE(int64, int, pack, static_cast)
OPENCV_HAL_IMPL_C_PACK_STORE(short, uchar, pack_u, saturate_cast)
OPENCV_HAL_IMPL_C_PACK_STORE(int, ushort, pack_u, saturate_cast)

inline void v_pack_store(float16_t* ptr, const stdx::native_simd<float>& v) {
    for (size_t i = 0; i < v.size(); ++i) {
        ptr[i] = float16_t(v[i]);
    }
}

#define OPENCV_HAL_IMPL_C_RSHR_PACK_STORE(_Tp, _Tpn, pack_suffix, cast) \
template<int shift>                                                     \
inline void v_rshr_##pack_suffix##_store(_Tpn* ptr, const stdx::native_simd<_Tp>& a) { \
    for (size_t i = 0; i < a.size(); i++ ) \
        ptr[i] = cast<_Tpn>((a[i] + ((_Tp)1 << (shift - 1))) >> shift); \
}

OPENCV_HAL_IMPL_C_RSHR_PACK_STORE(ushort, uchar, pack, saturate_cast)
OPENCV_HAL_IMPL_C_RSHR_PACK_STORE(short, schar, pack, saturate_cast)
OPENCV_HAL_IMPL_C_RSHR_PACK_STORE(unsigned, ushort, pack, saturate_cast)
OPENCV_HAL_IMPL_C_RSHR_PACK_STORE(int, short, pack, saturate_cast)
OPENCV_HAL_IMPL_C_RSHR_PACK_STORE(uint64, unsigned, pack, static_cast)
OPENCV_HAL_IMPL_C_RSHR_PACK_STORE(int64, int, pack, static_cast)
OPENCV_HAL_IMPL_C_RSHR_PACK_STORE(short, uchar, pack_u, saturate_cast)
OPENCV_HAL_IMPL_C_RSHR_PACK_STORE(int, ushort, pack_u, saturate_cast)

template<typename T>
void _pack_b(stdx::native_simd<uchar>& s, size_t& j, const stdx::native_simd<T>& a) {
    for (size_t i = 0; i < a.size(); ++i) {
        s[j++] = a[i] == 0 ? 0 : 0xFF;
    }
}

template<typename... T>
auto _pack_b_varargs(const T... args) {
    stdx::native_simd<uchar> s;
    size_t j = 0;
    (_pack_b(s, j, args), ...);
    return s;
}

inline auto v_pack_b(const stdx::native_simd<ushort>& a, const stdx::native_simd<ushort>& b) {
    return _pack_b_varargs(a, b);
}

inline auto v_pack_b(const stdx::native_simd<unsigned>& a, const stdx::native_simd<unsigned>& b,
                     const stdx::native_simd<unsigned>& c, const stdx::native_simd<unsigned>& d) {
    return _pack_b_varargs(a, b, c, d);
}

inline auto v_pack_b(const stdx::native_simd<uint64>& a, const stdx::native_simd<uint64>& b,
                     const stdx::native_simd<uint64>& c, const stdx::native_simd<uint64>& d,
                     const stdx::native_simd<uint64>& e, const stdx::native_simd<uint64>& f,
                     const stdx::native_simd<uint64>& g, const stdx::native_simd<uint64>& h) {
    return _pack_b_varargs(a, b, c, d, e, f, g, h);
}

template<typename T>
inline auto v_pack_triplets(const stdx::native_simd<T>& a) {
    stdx::native_simd<T> c;
    for (size_t i = 0; i < a.size()/4; i++) {
        c[3*i  ] = a[4*i  ];
        c[3*i+1] = a[4*i+1];
        c[3*i+2] = a[4*i+2];
    }
    return c;
}

template<int i, typename T>
inline stdx::native_simd<T> v_broadcast_element(const stdx::native_simd<T>& a) {
    return a[i];
}

inline auto v_round(const stdx::native_simd<float>& a) {
    return stdx::native_simd<int>([&](size_t i) { return cvRound(a[i]); });
}

inline auto v_round(const stdx::native_simd<double>& a, const stdx::native_simd<double>& b) {
    return stdx::native_simd<int>([&](size_t i) { return cvRound(i < a.size() ? a[i] : b[i - a.size()]); });
}

inline auto v_floor(const stdx::native_simd<float>& a) {
    return stdx::native_simd<int>([&](size_t i) { return cvFloor(a[i]); });
}

inline auto v_ceil(const stdx::native_simd<float>& a) {
    return stdx::native_simd<int>([&](size_t i) { return cvCeil(a[i]); });
}

inline auto v_trunc(const stdx::native_simd<float>& a) {
    return stdx::native_simd<int>([&](size_t i) { return int(a[i]); });
}

inline auto v_round(const stdx::native_simd<double>& a) {
    return stdx::native_simd<int>([&](size_t i) { return cvRound(i < a.size() ? a[i] : 0); });
}

inline auto v_floor(const stdx::native_simd<double>& a) {
    return stdx::native_simd<int>([&](size_t i) { return cvFloor(i < a.size() ? a[i] : 0); });
}

inline auto v_ceil(const stdx::native_simd<double>& a) {
    return stdx::native_simd<int>([&](size_t i) { return cvCeil(i < a.size() ? a[i] : 0); });
}

inline auto v_trunc(const stdx::native_simd<double>& a) {
    return stdx::native_simd<int>([&](size_t i) { return int(i < a.size() ? a[i] : 0); });
}

inline auto v_cvt_f32(const stdx::native_simd<int>& a) {
    return stdx::static_simd_cast<stdx::native_simd<float>>(a);
}

inline auto v_cvt_f32(const stdx::native_simd<double>& a) {
    return stdx::native_simd<float>([&](size_t i) { return float(i < a.size() ? a[i] : 0); });
}

inline auto v_cvt_f32(const stdx::native_simd<double>& a, const stdx::native_simd<double>& b) {
    return stdx::native_simd<float>([&](size_t i) { return float(i < a.size() ? a[i] : b[i - a.size()]); });
}

inline auto v_cvt_f64(const stdx::native_simd<int>& a) {
    return stdx::native_simd<double>([&](size_t i) { return double(a[i]); });
}

inline auto v_cvt_f64_high(const stdx::native_simd<int>& a) {
    return stdx::native_simd<double>([&](size_t i) { return double(a[i + a.size() / 2]); });
}

inline auto v_cvt_f64(const stdx::native_simd<float>& a) {
    return stdx::native_simd<double>([&](size_t i) { return double(a[i]); });
}

inline auto v_cvt_f64_high(const stdx::native_simd<float>& a) {
    return stdx::native_simd<double>([&](size_t i) { return double(a[i + a.size() / 2]); });
}

inline auto v_cvt_f64(const stdx::native_simd<int64>& a) {
    return stdx::static_simd_cast<stdx::native_simd<double>>(a);
}

template<typename T>
inline int v_signmask(const stdx::native_simd<T>& a) {
    int mask = 0;
    for(size_t i = 0; i < a.size(); ++i)
        mask |= (V_TypeTraits<T>::reinterpret_int(a[i]) < 0) << i;
    return mask;
}

template <typename T>
inline int v_scan_forward(const stdx::native_simd<int64>& a) {
    for (size_t i = 0; i < a.size(); ++i)
        if(V_TypeTraits<T>::reinterpret_int(a[i]) < 0)
            return i;
    return 0;
}

template<typename T>
inline bool v_check_all(const stdx::native_simd<T>& a) {
    return stdx::all_of(reinterpret_cast<const stdx::native_simd<typename V_TypeTraits<T>::int_type>&>(a) < 0);
}

template<typename T>
inline bool v_check_any(const stdx::native_simd<T>& a) {
    return stdx::any_of(reinterpret_cast<const stdx::native_simd<typename V_TypeTraits<T>::int_type>&>(a) < 0);
}

template<typename T>
inline auto v_add_wrap(const stdx::native_simd<T>& a, const stdx::native_simd<T>& b) {
    return a + b;
}

template<typename T>
inline auto v_add(const stdx::native_simd<T>& a, const stdx::native_simd<T>& b) {
    if constexpr (std::is_same_v<decltype(a[0] + b[0]), T>) {
        return a + b;
    } else {
        return stdx::native_simd<T>([&](size_t i) { return saturate_cast<T>(a[i] + b[i]); });
    }
}

template<typename T>
inline auto v_add(const stdx::native_simd<T>& a, const stdx::native_simd<T>& b, const stdx::native_simd<T>& c) {
    if constexpr (std::is_same_v<decltype(a[0] + b[0] + c[0]), T>) {
        return a + b + c;
    } else {
        return stdx::native_simd<T>([&](size_t i) { return saturate_cast<T>(a[i] + b[i] + c[i]); });
    }
}

template<typename T>
inline auto v_sub_wrap(const stdx::native_simd<T>& a, const stdx::native_simd<T>& b) {
    return a - b;
}

template<typename T>
inline auto v_sub(const stdx::native_simd<T>& a, const stdx::native_simd<T>& b) {
    if constexpr (std::is_same_v<decltype(a[0] - b[0]), T>) {
        return a - b;
    } else {
        return stdx::native_simd<T>([&](size_t i) { return saturate_cast<T>(a[i] - b[i]); });
    }
}

template<typename T>
inline auto v_mul_wrap(const stdx::native_simd<T>& a, const stdx::native_simd<T>& b) {
    return a * b;
}

template<typename T>
inline auto v_mul_hi(const stdx::native_simd<T>& a, const stdx::native_simd<T>& b) {
    typedef typename V_TypeTraits<T>::w_type w_type;
    return stdx::native_simd<T>([&](size_t i) { return (T)((w_type)a[i] * b[i] >> sizeof(T) * 8); });
}

template<typename T>
inline auto v_mul(const stdx::native_simd<T>& a, const stdx::native_simd<T>& b) {
    if constexpr (std::is_same_v<decltype(a[0] * b[0]), T>) {
        return a * b;
    } else {
        return stdx::native_simd<T>([&](size_t i) { return saturate_cast<T>(a[i] * b[i]); });
    }
}

template<typename T>
inline auto v_mul(const stdx::native_simd<T>& a, const stdx::native_simd<T>& b, const stdx::native_simd<T>& c) {
    if constexpr (std::is_same_v<decltype(a[0] * b[0] * c[0]), T>) {
        return a * b * c;
    } else {
        return stdx::native_simd<T>([&](size_t i) { return saturate_cast<T>(a[i] * b[i] * c[i]); });
    }
}

template<typename T>
inline void v_mul_expand(const stdx::native_simd<T>& a, const stdx::native_simd<T>& b,
                         stdx::native_simd<typename V_TypeTraits<T>::w_type>& b0,
                         stdx::native_simd<typename V_TypeTraits<T>::w_type>& b1) {
    typedef typename V_TypeTraits<T>::w_type result_type;
    b0 = stdx::native_simd<result_type>([&](size_t i) { return (result_type)a[i] * b[i]; });
    b1 = stdx::native_simd<result_type>([&](size_t i) { return (result_type)a[i + a.size() / 2] * b[i + b.size() / 2]; });
}

template<typename T>
inline auto v_dotprod(const stdx::native_simd<T>& a, const stdx::native_simd<T>& b) {
    typedef typename V_TypeTraits<T>::w_type result_type;
    return stdx::native_simd<result_type>([&](size_t i) {
        return (result_type)a[2 * i] * b[2 * i] + (result_type)a[2 * i + 1] * b[2 * i + 1];
    });
}

template<typename T>
inline auto v_dotprod(const stdx::native_simd<T>& a, const stdx::native_simd<T>& b,
                      const stdx::native_simd<typename V_TypeTraits<T>::w_type>& c) {
    typedef typename V_TypeTraits<T>::w_type result_type;
    return stdx::native_simd<result_type>([&](size_t i) {
        return (result_type)a[2 * i] * b[2 * i] + (result_type)a[2 * i + 1] * b[2 * i + 1] + c[i];
    });
}

template<typename T>
inline auto v_dotprod_fast(const stdx::native_simd<T>& a, const stdx::native_simd<T>& b) {
    return v_dotprod(a, b);
}

template<typename T>
inline auto v_dotprod_fast(const stdx::native_simd<T>& a, const stdx::native_simd<T>& b,
                           const stdx::native_simd<typename V_TypeTraits<T>::w_type>& c) {
    return v_dotprod(a, b, c);
}


template<typename T>
inline auto v_dotprod_expand(const stdx::native_simd<T>& a, const stdx::native_simd<T>& b) {
    typedef typename V_TypeTraits<T>::q_type result_type;
    return stdx::native_simd<result_type>([&](size_t i) {
        return (result_type)a[4 * i    ] * b[4 * i    ] + (result_type)a[4 * i + 1] * b[4 * i + 1]
             + (result_type)a[4 * i + 2] * b[4 * i + 2] + (result_type)a[4 * i + 3] * b[4 * i + 3];
    });
}

template<typename T>
inline auto v_dotprod_expand(const stdx::native_simd<T>& a, const stdx::native_simd<T>& b,
                      const stdx::native_simd<typename V_TypeTraits<T>::q_type>& c) {
    typedef typename V_TypeTraits<T>::q_type result_type;
    return stdx::native_simd<result_type>([&](size_t i) {
        return (result_type)a[4 * i    ] * b[4 * i    ] + (result_type)a[4 * i + 1] * b[4 * i + 1]
             + (result_type)a[4 * i + 2] * b[4 * i + 2] + (result_type)a[4 * i + 3] * b[4 * i + 3] + c[i];
    });
}

template<typename T>
inline auto v_dotprod_expand_fast(const stdx::native_simd<T>& a, const stdx::native_simd<T>& b) {
    return v_dotprod_expand(a, b);
}

template<typename T>
inline auto v_dotprod_expand_fast(const stdx::native_simd<T>& a, const stdx::native_simd<T>& b,
                           const stdx::native_simd<typename V_TypeTraits<T>::q_type>& c) {
    return v_dotprod_expand(a, b, c);
}

template<typename T>
inline auto v_div(const stdx::native_simd<T>& a, const stdx::native_simd<T>& b) {
    return a / b;
}

template<typename T>
inline auto v_not(const stdx::native_simd<T>& a) {
    return ~a;
}

template<typename T>
inline auto v_and(const stdx::native_simd<T>& a, const stdx::native_simd<T>& b) {
    return a & b;
}

template<typename T>
inline auto v_or(const stdx::native_simd<T>& a, const stdx::native_simd<T>& b) {
    return a | b;
}

template<typename T>
inline auto v_xor(const stdx::native_simd<T>& a, const stdx::native_simd<T>& b) {
    return a ^ b;
}

template<typename T>
inline auto v_reverse(const stdx::native_simd<T>& a) {
    return stdx::native_simd<T>([&](size_t i) { return a[a.size() - i - 1]; });
}

template<int s, typename T>
inline auto v_extract(const stdx::native_simd<T>& a, const stdx::native_simd<T>& b) {
    const int shift = a.size() - s;
    return stdx::native_simd<T>([&](size_t i) { return i < shift ? a[i + s] : b[i - shift]; });
}

template<int s, typename T>
inline auto v_rotate_left(const stdx::native_simd<T>& a) {
    return stdx::native_simd<T>([&](int i) {
        const int n = a.size();
        int sIndex = i - s;
        return 0 <= sIndex && sIndex < n ? a[sIndex] : T(0);
    });
}

template<int s, typename T>
inline auto v_rotate_right(const stdx::native_simd<T>& a) {
    return stdx::native_simd<T>([&](int i) {
        const int n = a.size();
        int sIndex = i + s;
        return 0 <= sIndex && sIndex < n ? a[sIndex] : T(0);
    });
}


template<int s, typename T>
inline auto v_rotate_left(const stdx::native_simd<T>& a, const stdx::native_simd<T>& b) {
    return stdx::native_simd<T>([&](int i) {
        const int n = a.size();
        int aIndex = i - s;
        int bIndex = i - s + n;
        if (0 <= aIndex && aIndex < n) return a[aIndex];
        else if (0 <= bIndex && bIndex < n) return b[bIndex];
        else return T(0);
    });
}

template<int s, typename T>
inline auto v_rotate_right(const stdx::native_simd<T>& a, const stdx::native_simd<T>& b) {
    return stdx::native_simd<T>([&](int i) {
        const int n = a.size();
        int aIndex = i + s;
        int bIndex = i + s - n;
        if (0 <= aIndex && aIndex < n) return a[aIndex];
        else if (0 <= bIndex && bIndex < n) return b[bIndex];
        else return T(0);
    });
}

template<typename T>
inline auto v_sqrt(const stdx::native_simd<T>& a) {
    return stdx::sqrt(a);
}

template<typename T>
inline auto v_invsqrt(const stdx::native_simd<T>& a) {
    return 1 / stdx::sqrt(a);
}

template<int n, typename T>
inline auto v_shl(const stdx::native_simd<T>& a) {
    return a << n;
}

template<int n, typename T>
inline auto v_shr(const stdx::native_simd<T>& a) {
    return a >> n;
}

template<typename T>
inline auto v_absdiff(const stdx::native_simd<T>& a, const stdx::native_simd<T>& b) {
    if constexpr (std::is_floating_point_v<T>) {
        return stdx::abs(a - b);
    } else {
        typedef typename V_TypeTraits<T>::abs_type rtype;
        constexpr rtype mask = (rtype)(std::numeric_limits<T>::is_signed ? (1 << (sizeof(rtype)*8 - 1)) : 0);
        typedef stdx::native_simd<rtype> result_type;
        auto u = reinterpret_cast<const result_type&>(a) ^ mask;
        auto v = reinterpret_cast<const result_type&>(b) ^ mask;
        return result_type([&](size_t i) { return u[i] > v[i] ? u[i] - v[i] : v[i] - u[i]; });
    }
}

template<typename T>
inline auto v_absdiffs(const stdx::native_simd<T>& a, const stdx::native_simd<T>& b) {
    if constexpr (std::is_unsigned_v<T>) { // according to the doc this branch is unreachable, but it is reached anyway
        return stdx::native_simd<T>([&](size_t i) { return saturate_cast<T>(a[i] > b[i] ? a[i] - b[i] : b[i] - a[i]); });
    } else {
        if constexpr (std::is_same_v<decltype(std::abs(a[0] - b[0])), T>) {
            return stdx::abs(a - b);
        } else {
            return stdx::native_simd<T>([&](size_t i) { return saturate_cast<T>(std::abs(a[i] - b[i])); });
        }
    }
}

template<typename T>
inline auto v_reduce_sad(const stdx::native_simd<T>& a, const stdx::native_simd<T>& b) {
    return stdx::reduce(v_absdiffs(a, b));
}


template<typename T>
inline auto _mask(const stdx::native_simd_mask<T>& mask) {
    typedef typename V_TypeTraits<T>::int_type itype;
    return stdx::native_simd<T>([&](size_t i) { return V_TypeTraits<T>::reinterpret_from_int((itype)-(int)mask[i]); });
}

template<typename T>
inline auto v_eq(const stdx::native_simd<T>& a, const stdx::native_simd<T>& b) {
    return _mask(a == b);
}

template<typename T>
inline auto v_ne(const stdx::native_simd<T>& a, const stdx::native_simd<T>& b) {
    return _mask(a != b);
}

template<typename T>
inline auto v_lt(const stdx::native_simd<T>& a, const stdx::native_simd<T>& b) {
    return _mask(a < b);
}

template<typename T>
inline auto v_le(const stdx::native_simd<T>& a, const stdx::native_simd<T>& b) {
    return _mask(a <= b);
}

template<typename T>
inline auto v_gt(const stdx::native_simd<T>& a, const stdx::native_simd<T>& b) {
    return _mask(a > b);
}

template<typename T>
inline auto v_ge(const stdx::native_simd<T>& a, const stdx::native_simd<T>& b) {
    return _mask(a >= b);
}

template<typename T>
inline auto v_select(const stdx::native_simd<T>& mask, const stdx::native_simd<T>& a, const stdx::native_simd<T>& b) {
    return stdx::native_simd<T>([&](size_t i) { return V_TypeTraits<T>::reinterpret_int(mask[i]) ? a[i] : b[i]; });
}

template<typename T>
inline auto v_fma(const stdx::native_simd<T>& a, const stdx::native_simd<T>& b, const stdx::native_simd<T>& c) {
    return stdx::fma(a, b, c);
}

template<typename T>
inline auto v_muladd(const stdx::native_simd<T>& a, const stdx::native_simd<T>& b, const stdx::native_simd<T>& c) {
    return v_fma(a, b, c);
}

template<typename T>
inline auto v_min(const stdx::native_simd<T>& a, const stdx::native_simd<T>& b) {
    return stdx::min(a, b);
}

template<typename T>
inline auto v_max(const stdx::native_simd<T>& a, const stdx::native_simd<T>& b) {
    return stdx::max(a, b);
}

template<typename T>
inline void v_minmax(const stdx::native_simd<T>& a, const stdx::native_simd<T>& b,
                     stdx::native_simd<T>& min, stdx::native_simd<T>& max) {
    std::tie(min, max) = stdx::minmax(a, b);
}

template<typename T>
inline auto v_abs(const stdx::native_simd<T>& a) {
    if constexpr (std::is_floating_point_v<T>) {
        return stdx::abs(a);
    } else {
        typedef typename V_TypeTraits<T>::abs_type result_type;
        return stdx::native_simd<result_type>([&](size_t i) { return result_type(std::abs(a[i])); });
    }
}

template<int s, typename T>
inline auto v_extract_n(const stdx::native_simd<T>& a) {
    return a[s];
}

template<typename T>
inline auto v_extract_highest(const stdx::native_simd<T>& a) {
    return a[a.size() - 1];
}

template<typename T>
inline stdx::native_simd<T> v_broadcast_highest(const stdx::native_simd<T>& a) {
    return a[a.size() - 1];
}

template<typename T>
inline auto v_reduce_min(const stdx::native_simd<T>& a) {
    return stdx::hmin(a);
}

template<typename T>
inline auto v_reduce_max(const stdx::native_simd<T>& a) {
    return stdx::hmax(a);
}

template<typename T>
inline auto v_reduce_sum(const stdx::native_simd<T>& a) {
    typedef typename V_TypeTraits<T>::sum_type result_type;
    if constexpr (std::is_same_v<result_type, T>) {
        return stdx::reduce(a);
    } else {
        result_type result = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            result += a[i];
        }
        return result;
    }
}

template<typename T>
inline auto v_popcount(const stdx::native_simd<T>& a) {
    typedef typename V_TypeTraits<T>::abs_type result_type;
    return stdx::native_simd<result_type>([&](size_t i) { return result_type(__builtin_popcountll(uint64(a[i]))); });
}

template<typename T>
inline auto v_reduce_sum4(const stdx::native_simd<T>& a, const stdx::native_simd<T>& b,
                          const stdx::native_simd<T>& c, const stdx::native_simd<T>& d) {
    return v_load({stdx::reduce(a), stdx::reduce(b), stdx::reduce(c), stdx::reduce(d)});
}


template<typename T>
inline auto v_magnitude(const stdx::native_simd<T>& a, const stdx::native_simd<T>& b) {
    return stdx::hypot(a, b);
}

template<typename T>
inline auto v_sqr_magnitude(const stdx::native_simd<T>& a, const stdx::native_simd<T>& b) {
    return a * a + b * b;
}

inline stdx::native_simd<float> v_matmul(const stdx::native_simd<float>& v,
                                         const stdx::native_simd<float>& a, const stdx::native_simd<float>& b,
                                         const stdx::native_simd<float>& c, const stdx::native_simd<float>& d) {
    stdx::native_simd<float> res;
    for (size_t i = 0; i < v.size() / 4; i++) {
        res[0 + i*4] = v[0 + i*4] * a[0 + i*4] + v[1 + i*4] * b[0 + i*4] + v[2 + i*4] * c[0 + i*4] + v[3 + i*4] * d[0 + i*4];
        res[1 + i*4] = v[0 + i*4] * a[1 + i*4] + v[1 + i*4] * b[1 + i*4] + v[2 + i*4] * c[1 + i*4] + v[3 + i*4] * d[1 + i*4];
        res[2 + i*4] = v[0 + i*4] * a[2 + i*4] + v[1 + i*4] * b[2 + i*4] + v[2 + i*4] * c[2 + i*4] + v[3 + i*4] * d[2 + i*4];
        res[3 + i*4] = v[0 + i*4] * a[3 + i*4] + v[1 + i*4] * b[3 + i*4] + v[2 + i*4] * c[3 + i*4] + v[3 + i*4] * d[3 + i*4];
    }
    return res;
}

inline stdx::native_simd<float> v_matmuladd(const stdx::native_simd<float>& v,
                                            const stdx::native_simd<float>& a, const stdx::native_simd<float>& b,
                                            const stdx::native_simd<float>& c, const stdx::native_simd<float>& d) {
    stdx::native_simd<float> res;
    for (size_t i = 0; i < v.size() / 4; i++) {
        res[0 + i * 4] = v[0 + i * 4] * a[0 + i * 4] + v[1 + i * 4] * b[0 + i * 4] + v[2 + i * 4] * c[0 + i * 4] + d[0 + i * 4];
        res[1 + i * 4] = v[0 + i * 4] * a[1 + i * 4] + v[1 + i * 4] * b[1 + i * 4] + v[2 + i * 4] * c[1 + i * 4] + d[1 + i * 4];
        res[2 + i * 4] = v[0 + i * 4] * a[2 + i * 4] + v[1 + i * 4] * b[2 + i * 4] + v[2 + i * 4] * c[2 + i * 4] + d[2 + i * 4];
        res[3 + i * 4] = v[0 + i * 4] * a[3 + i * 4] + v[1 + i * 4] * b[3 + i * 4] + v[2 + i * 4] * c[3 + i * 4] + d[3 + i * 4];
    }
    return res;
}

template<typename T>
inline void v_transpose4x4(stdx::native_simd<T>& a0, const stdx::native_simd<T>& a1,
                           const stdx::native_simd<T>& a2, const stdx::native_simd<T>& a3,
                           stdx::native_simd<T>& b0, stdx::native_simd<T>& b1,
                           stdx::native_simd<T>& b2, stdx::native_simd<T>& b3) {
    for (size_t i = 0; i < a0.size() / 4; i++) {
        b0[0 + i*4] = a0[0 + i*4]; b0[1 + i*4] = a1[0 + i*4];
        b0[2 + i*4] = a2[0 + i*4]; b0[3 + i*4] = a3[0 + i*4];
        b1[0 + i*4] = a0[1 + i*4]; b1[1 + i*4] = a1[1 + i*4];
        b1[2 + i*4] = a2[1 + i*4]; b1[3 + i*4] = a3[1 + i*4];
        b2[0 + i*4] = a0[2 + i*4]; b2[1 + i*4] = a1[2 + i*4];
        b2[2 + i*4] = a2[2 + i*4]; b2[3 + i*4] = a3[2 + i*4];
        b3[0 + i*4] = a0[3 + i*4]; b3[1 + i*4] = a1[3 + i*4];
        b3[2 + i*4] = a2[3 + i*4]; b3[3 + i*4] = a3[3 + i*4];
    }
}

CV_CPU_OPTIMIZATION_HAL_NAMESPACE_END
}

#endif //OPENCV_HAL_INTRIN_SIMD_HPP
