/***************************************************************************
*  Copyright (C) Codeplay Software Limited
*  Copyright (C) 2022 Heidelberg University, Engineering Mathematics and Computing Lab (EMCL) and Computing Centre (URZ)
*
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

#ifndef _MKL_SPARSE_CUSPARSE_TASK_HPP_
#define _MKL_SPARSE_CUSPARSE_TASK_HPP_
#include <cusparse.h>
#include <cuda.h>
#include <complex>
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#ifndef __HIPSYCL__
#include "cusparse_scope_handle.hpp"
#if __has_include(<sycl/detail/pi.hpp>)
#include <sycl/detail/pi.hpp>
#else
#include <CL/sycl/detail/pi.hpp>
#endif
#else
#include "cusparse_scope_handle_hipsycl.hpp"
namespace sycl {
using interop_handler = sycl::interop_handle;
}
#endif
namespace oneapi {
namespace mkl {
namespace sparse {
namespace cusparse {

#ifdef __HIPSYCL__
template <typename H, typename F>
static inline void host_task_internal(H &cgh, sycl::queue queue, F f) {
    cgh.hipSYCL_enqueue_custom_operation([f, queue](sycl::interop_handle ih) {
        auto sc = CusparseScopedContextHandler(queue, ih);
        f(sc);
    });
}
#else
template <typename H, typename F>
static inline void host_task_internal(H &cgh, sycl::queue queue, F f) {
    cgh.host_task([f, queue](sycl::interop_handle ih) {
        auto sc = CusparseScopedContextHandler(queue, ih);
        f(sc);
        sc.wait_stream(queue);
    });
}
#endif
template <typename H, typename F>
static inline void onemkl_cusparse_host_task(H &cgh, sycl::queue queue, F f) {
    (void)host_task_internal(cgh, queue, f);
}

} // namespace cusparse
} // namespace sparse
} // namespace mkl
} // namespace oneapi
#endif // _MKL_SPARSE_CUSPARSE_TASK_HPP_
