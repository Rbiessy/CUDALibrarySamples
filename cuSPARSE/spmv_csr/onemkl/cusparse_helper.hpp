/***************************************************************************
*  Copyright (C) Codeplay Software Limited
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*
*  For your convenience, a copy of the License has been included in this
*  repository.
*
*  Unless required by applicable law or agreed to in writing, software
*  distributed under the License is distributed on an "AS IS" BASIS,
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  See the License for the specific language governing permissions and
*  limitations under the License.
*
**************************************************************************/

/**
 * @file cusparse_*.cpp : contain the implementation of all the routines
 * for CUDA backend
 */
#ifndef _CUSPARSE_HELPER_HPP_
#define _CUSPARSE_HELPER_HPP_
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include <cusparse.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <complex>

namespace oneapi {
namespace mkl {
namespace sparse {
namespace cusparse {

// The static assert to make sure that all index types used in
// src/oneMKL/backend/cusparse/sparse.hpp interface are int64_t
template <typename... Next>
struct is_int64 : std::false_type {};

template <typename First>
struct is_int64<First> : std::is_same<std::int64_t, First> {};

template <typename First, typename... Next>
struct is_int64<First, Next...>
        : std::integral_constant<bool, std::is_same<std::int64_t, First>::value &&
                                           is_int64<Next...>::value> {};

template <typename... T>
struct Overflow {
    static void inline check(T...) {}
};

template <typename Index, typename... T>
struct Overflow<Index, T...> {
    static void inline check(Index index, T... next) {
        if (std::abs(index) >= (1LL << 31)) {
            throw std::runtime_error(
                "Cusparse index overflow. cusparse does not support 64 bit integer as "
                "data size. Thus, the data size should not be greater that maximum "
                "supported size by 32 bit integer.");
        }
        Overflow<T...>::check(next...);
    }
};

template <typename Index, typename... Next>
void overflow_check(Index index, Next... indices) {
    static_assert(is_int64<Index, Next...>::value, "oneMKL index type must be 64 bit integer.");
    Overflow<Index, Next...>::check(index, indices...);
}

class cusparse_error : virtual public std::runtime_error {
protected:
    inline const char *cusparse_error_map(cusparseStatus_t error) {
        switch (error) {
            case CUSPARSE_STATUS_SUCCESS: return "CUSPARSE_STATUS_SUCCESS";

            case CUSPARSE_STATUS_NOT_INITIALIZED: return "CUSPARSE_STATUS_NOT_INITIALIZED";

            case CUSPARSE_STATUS_ALLOC_FAILED: return "CUSPARSE_STATUS_ALLOC_FAILED";

            case CUSPARSE_STATUS_INVALID_VALUE: return "CUSPARSE_STATUS_INVALID_VALUE";

            case CUSPARSE_STATUS_ARCH_MISMATCH: return "CUSPARSE_STATUS_ARCH_MISMATCH";

            case CUSPARSE_STATUS_EXECUTION_FAILED: return "CUSPARSE_STATUS_EXECUTION_FAILED";

            case CUSPARSE_STATUS_INTERNAL_ERROR: return "CUSPARSE_STATUS_INTERNAL_ERROR";

            case CUSPARSE_STATUS_NOT_SUPPORTED: return "CUSPARSE_STATUS_NOT_SUPPORTED";

            case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED: return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";

            case CUSPARSE_STATUS_INSUFFICIENT_RESOURCES: return "CUSPARSE_STATUS_INSUFFICIENT_RESOURCES";

            default: return "<unknown>";
        }
    }

    int error_number; ///< Error number
public:
    /** Constructor (C++ STL string, cusparseStatus_t).
   *  @param msg The error message
   *  @param err_num error number
   */
    explicit cusparse_error(std::string message, cusparseStatus_t result)
            : std::runtime_error((message + std::string(cusparse_error_map(result)))) {
        error_number = static_cast<int>(result);
    }

    /** Destructor.
   *  Virtual to allow for subclassing.
   */
    virtual ~cusparse_error() throw() {}

    /** Returns error number.
   *  @return #error_number
   */
    virtual int getErrorNumber() const throw() {
        return error_number;
    }
};

class cuda_error : virtual public std::runtime_error {
protected:
    inline const char *cuda_error_map(CUresult result) {
        switch (result) {
            case CUDA_SUCCESS: return "CUDA_SUCCESS";
            case CUDA_ERROR_NOT_PERMITTED: return "CUDA_ERROR_NOT_PERMITTED";
            case CUDA_ERROR_INVALID_CONTEXT: return "CUDA_ERROR_INVALID_CONTEXT";
            case CUDA_ERROR_INVALID_DEVICE: return "CUDA_ERROR_INVALID_DEVICE";
            case CUDA_ERROR_INVALID_VALUE: return "CUDA_ERROR_INVALID_VALUE";
            case CUDA_ERROR_OUT_OF_MEMORY: return "CUDA_ERROR_OUT_OF_MEMORY";
            case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES: return "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES";
            default: return "<unknown>";
        }
    }
    int error_number; ///< error number
public:
    /** Constructor (C++ STL string, CUresult).
   *  @param msg The error message
   *  @param err_num Error number
   */
    explicit cuda_error(std::string message, CUresult result)
            : std::runtime_error((message + std::string(cuda_error_map(result)))) {
        error_number = static_cast<int>(result);
    }

    /** Destructor.
   *  Virtual to allow for subclassing.
   */
    virtual ~cuda_error() throw() {}

    /** Returns error number.
   *  @return #error_number
   */
    virtual int getErrorNumber() const throw() {
        return error_number;
    }
};

#define CUDA_ERROR_FUNC(name, err, ...)                                 \
    err = name(__VA_ARGS__);                                            \
    if (err != CUDA_SUCCESS) {                                          \
        throw cuda_error(std::string(#name) + std::string(" : "), err); \
    }

#define CUSPARSE_ERROR_FUNC(name, err, ...)                                 \
    err = name(__VA_ARGS__);                                              \
    if (err != CUSPARSE_STATUS_SUCCESS) {                                   \
        throw oneapi::mkl::sparse::cusparse::cusparse_error(std::string(#name) + std::string(" : "), err); \
    }

#define CUSPARSE_ERROR_FUNC_SYNC(name, err, handle, ...)                    \
    err = name(handle, __VA_ARGS__);                                      \
    if (err != CUSPARSE_STATUS_SUCCESS) {                                   \
        throw oneapi::mkl::sparse::cusparse::cusparse_error(std::string(#name) + std::string(" : "), err); \
    }                                                                     \
    cudaStream_t currentStreamId;                                         \
    CUSPARSE_ERROR_FUNC(cusparseGetStream, err, handle, &currentStreamId);    \
    cuStreamSynchronize(currentStreamId);

#define CUSPARSE_ERROR_FUNC_T_SYNC(name, func, err, handle, ...)           \
    err = func(handle, __VA_ARGS__);                                     \
    if (err != CUSPARSE_STATUS_SUCCESS) {                                  \
        throw oneapi::mkl::sparse::cusparse::cusparse_error(std::string(name) + std::string(" : "), err); \
    }                                                                    \
    cudaStream_t currentStreamId;                                        \
    CUSPARSE_ERROR_FUNC(cusparseGetStream, err, handle, &currentStreamId);   \
    cuStreamSynchronize(currentStreamId);

/*
inline cusparseOperation_t get_cusparse_operation(oneapi::mkl::transpose trn) {
    switch (trn) {
        case oneapi::mkl::transpose::nontrans: return CUSPARSE_OP_N;
        case oneapi::mkl::transpose::trans: return CUSPARSE_OP_T;
        case oneapi::mkl::transpose::conjtrans: return CUSPARSE_OP_C;
        default: throw "Wrong transpose Operation.";
    }
}

inline cusparseFillMode_t get_cusparse_fill_mode(oneapi::mkl::uplo ul) {
    switch (ul) {
        case oneapi::mkl::uplo::upper: return CUSPARSE_FILL_MODE_UPPER;
        case oneapi::mkl::uplo::lower: return CUSPARSE_FILL_MODE_LOWER;
        default: throw "Wrong fill mode.";
    }
}

inline cusparseDiagType_t get_cusparse_diag_type(oneapi::mkl::diag un) {
    switch (un) {
        case oneapi::mkl::diag::unit: return CUSPARSE_DIAG_UNIT;
        case oneapi::mkl::diag::nonunit: return CUSPARSE_DIAG_NON_UNIT;
        default: throw "Wrong diag type.";
    }
}

inline cusparseSideMode_t get_cusparse_side_mode(oneapi::mkl::side lr) {
    switch (lr) {
        case oneapi::mkl::side::left: return CUSPARSE_SIDE_LEFT;
        case oneapi::mkl::side::right: return CUSPARSE_SIDE_RIGHT;
        default: throw "Wrong side mode.";
    }
}
*/

/*converting std::complex<T> to cu<T>Complex*/
/*converting sycl::half to __half*/
template <typename T>
struct CudaEquivalentType {
    using Type = T;
};
template <>
struct CudaEquivalentType<sycl::half> {
    using Type = __half;
};
template <>
struct CudaEquivalentType<std::complex<float>> {
    using Type = cuComplex;
};
template <>
struct CudaEquivalentType<std::complex<double>> {
    using Type = cuDoubleComplex;
};

} // namespace cusparse
} // namespace sparse
} // namespace mkl
} // namespace oneapi
#endif // _CUSPARSE_HELPER_HPP_
