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
#include "cusparse_scope_handle.hpp"
#if __has_include(<sycl/detail/common.hpp>)
#include <sycl/detail/common.hpp>
#else
#include <CL/sycl/detail/common.hpp>
#endif

namespace oneapi {
namespace mkl {
namespace sparse {
namespace cusparse {

/**
 * Inserts a new element in the map if its key is unique. This new element
 * is constructed in place using args as the arguments for the construction
 * of a value_type (which is an object of a pair type). The insertion only
 * takes place if no other element in the container has a key equivalent to
 * the one being emplaced (keys in a map container are unique).
 */
thread_local cusparse_handle<pi_context> CusparseScopedContextHandler::handle_helper =
    cusparse_handle<pi_context>{};

CusparseScopedContextHandler::CusparseScopedContextHandler(sycl::queue queue)
        : needToRecover_(false) {
    placedContext_ = new sycl::context(queue.get_context());
    auto device = queue.get_device();
    auto desired = sycl::get_native<sycl::backend::ext_oneapi_cuda>(*placedContext_);
    CUresult err;
    CUDA_ERROR_FUNC(cuCtxGetCurrent, err, &original_);
    if (original_ != desired) {
        // Sets the desired context as the active one for the thread
        CUDA_ERROR_FUNC(cuCtxSetCurrent, err, desired);
        // No context is installed and the suggested context is primary
        // This is the most common case. We can activate the context in the
        // thread and leave it there until all the PI context referring to the
        // same underlying CUDA primary context are destroyed. This emulates
        // the behaviour of the CUDA runtime api, and avoids costly context
        // switches. No action is required on this side of the if.
        needToRecover_ = !(original_ == nullptr);
    }
}

CusparseScopedContextHandler::~CusparseScopedContextHandler() noexcept(false) {
    if (needToRecover_) {
        CUresult err;
        CUDA_ERROR_FUNC(cuCtxSetCurrent, err, original_);
    }
    delete placedContext_;
}

void ContextCallback(void *userData) {
    auto *ptr = static_cast<std::atomic<cusparseHandle_t> *>(userData);
    if (!ptr) {
        return;
    }
    auto handle = ptr->exchange(nullptr);
    if (handle != nullptr) {
        cusparseStatus_t err1;
        CUSPARSE_ERROR_FUNC(cusparseDestroy, err1, handle);
        handle = nullptr;
    }
    else {
        // if the handle is nullptr it means the handle was already destroyed by
        // the cusparse_handle destructor and we're free to delete the atomic
        // object.
        delete ptr;
    }
}

cusparseHandle_t CusparseScopedContextHandler::get_handle(const sycl::queue &queue) {
    auto piPlacedContext_ = reinterpret_cast<pi_context>(
        sycl::get_native<sycl::backend::ext_oneapi_cuda>(*placedContext_));
    CUstream streamId = get_stream(queue);
    cusparseStatus_t err;
    auto it = handle_helper.cusparse_handle_mapper_.find(piPlacedContext_);
    if (it != handle_helper.cusparse_handle_mapper_.end()) {
        if (it->second == nullptr) {
            handle_helper.cusparse_handle_mapper_.erase(it);
        }
        else {
            auto handle = it->second->load();
            if (handle != nullptr) {
                cudaStream_t currentStreamId;
                CUSPARSE_ERROR_FUNC(cusparseGetStream, err, handle, &currentStreamId);
                if (currentStreamId != streamId) {
                    CUSPARSE_ERROR_FUNC(cusparseSetStream, err, handle, streamId);
                }
                return handle;
            }
            else {
                handle_helper.cusparse_handle_mapper_.erase(it);
            }
        }
    }

    cusparseHandle_t handle;

    CUSPARSE_ERROR_FUNC(cusparseCreate, err, &handle);
    CUSPARSE_ERROR_FUNC(cusparseSetStream, err, handle, streamId);

    auto insert_iter = handle_helper.cusparse_handle_mapper_.insert(
        std::make_pair(piPlacedContext_, new std::atomic<cusparseHandle_t>(handle)));

    sycl::detail::pi::contextSetExtendedDeleter(*placedContext_, ContextCallback,
                                                insert_iter.first->second);

    return handle;
}

CUstream CusparseScopedContextHandler::get_stream(const sycl::queue &queue) {
    return sycl::get_native<sycl::backend::ext_oneapi_cuda>(queue);
}
sycl::context CusparseScopedContextHandler::get_context(const sycl::queue &queue) {
    return queue.get_context();
}

} // namespace cusparse
} // namespace sparse
} // namespace mkl
} // namespace oneapi
