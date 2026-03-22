// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/fwd.h>

#include "lib/common/container/small_vector.hpp"
#include "lib/common/synchronized.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/hsa/aql_packet.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_info_session.hpp"
#include "lib/rocprofiler-sdk/hsa/rocprofiler_packet.hpp"
#include "lib/rocprofiler-sdk/kernel_dispatch/profiling_time.hpp"

#include <hsa/amd_hsa_kernel_code.h>
#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa/hsa_ven_amd_aqlprofile.h>
#include <hsa/hsa_ven_amd_loader.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace rocprofiler
{
namespace context
{
struct context;
struct correlation_id;
}  // namespace context
namespace hsa
{
struct async_signal_slot;

enum class queue_state
{
    normal       = 0,
    to_destroy   = 1,
    done_destroy = 2
};

// Interceptor for a single specific queue
class Queue
{
public:
    using context_t            = context::context;
    using context_array_t      = common::container::small_vector<const context_t*>;
    using callback_t           = void (*)(hsa_status_t status, hsa_queue_t* source, void* data);
    using queue_info_session_t = queue_info_session;

    struct pkt_and_serialize_t
    {
        std::unique_ptr<AQLPacket> pkt{nullptr};
        bool                       request_serialize{false};
    };

    // Function prototype used to notify consumers that a kernel has been enqueued.
    // Pair first: An AQL packet can be returned that will be injected into the queue.
    // Pair second: Boolean flag indicating the dispatch needs to be serialized.
    using queue_cb_t =
        std::function<pkt_and_serialize_t(const Queue&,
                                          const rocprofiler_packet&,
                                          rocprofiler_kernel_id_t,
                                          rocprofiler_dispatch_id_t,
                                          rocprofiler_user_data_t*,
                                          const queue_info_session_t::external_corr_id_map_t&,
                                          const context::correlation_id*)>;
    // Signals the completion of the kernel packet.
    using completed_cb_t = std::function<void(const Queue&,
                                              const rocprofiler_packet&,
                                              std::shared_ptr<Queue::queue_info_session_t>&,
                                              inst_pkt_t&,
                                              kernel_dispatch::profiling_time)>;
    using callback_map_t = std::unordered_map<ClientID, std::pair<queue_cb_t, completed_cb_t>>;

    // Used when creating a Queue from a previously created intercept queue.
    // When the constructor with this parameter type is called, the provided function will be called
    // with the intended Queue WriteInterceptor function (hsa_amd_queue_intercept_handler).
    using set_write_interceptor_t = std::function<void(hsa_amd_queue_intercept_handler, void*)>;

    Queue(const AgentCache& agent, CoreApiTable table);
    Queue(const AgentCache&  agent,
          uint32_t           size,
          hsa_queue_type32_t type,
          callback_t         callback,
          void*              data,
          uint32_t           private_segment_size,
          uint32_t           group_segment_size,
          CoreApiTable       core_api,
          AmdExtTable        ext_api,
          hsa_queue_t**      queue);

    // Used when creating a Queue from a previously created intercept queue.
    Queue(const AgentCache&       agent,
          CoreApiTable            core_api,
          AmdExtTable             ext_api,
          hsa_queue_t*            queue,
          set_write_interceptor_t set_write_interceptor);
    virtual ~Queue();

    const hsa_queue_t*        intercept_queue() const { return _intercept_queue; };
    virtual const AgentCache& get_agent() const { return _agent; }

    void create_signal(uint32_t attribute, hsa_signal_t* signal, bool direct_path = false) const;
    void signal_async_handler(const hsa_signal_t& signal, void* data) const;
    async_signal_slot* acquire_async_signal_slot() const;
    void               arm_async_signal_slot(async_signal_slot*                   slot,
                                             std::shared_ptr<queue_info_session_t> session) const;
    void retire_signal(hsa_signal_t signal) const;
    void drain_retired_signals() const;
    void destroy_async_signal_slots() const;
    void complete_async_signal_slot(async_signal_slot* slot) const;

    template <typename FuncT>
    void signal_callback(FuncT&& func) const;

    template <typename FuncT>
    void lock_queue(FuncT&& func);

    virtual rocprofiler_queue_id_t get_id() const;

    // Fast check to see if we have any callbacks we need to notify
    int get_notifiers() const { return _notifiers; }

    // Tracks the number of in flight kernel executions we
    // are waiting on. We cannot destroy Queue until all kernels
    // have comleted.
    void    interceptor_started()
    {
        _active_async_packets.fetch_add(1, std::memory_order_acq_rel);
    }
    void interceptor_complete()
    {
        _active_async_packets.fetch_sub(1, std::memory_order_acq_rel);
    }
    int64_t active_interceptors() const
    {
        return _active_async_packets.load(std::memory_order_acquire);
    }
    void async_handler_started()
    {
        _active_async_handlers.fetch_add(1, std::memory_order_acq_rel);
    }
    void async_handler_complete()
    {
        _active_async_handlers.fetch_sub(1, std::memory_order_acq_rel);
    }
    int64_t active_async_handlers() const
    {
        return _active_async_handlers.load(std::memory_order_acquire);
    }
    void async_started() { _core_api.hsa_signal_add_relaxed_fn(_active_kernels, 1); }
    void async_complete() { _core_api.hsa_signal_subtract_relaxed_fn(_active_kernels, 1); }
    bool has_active_kernel_signal() const { return _active_kernels.handle != 0u; }
    int64_t active_async_packets() const
    {
        return _core_api.hsa_signal_load_scacquire_fn(_active_kernels);
    }
    void sync() const;

    void register_callback(ClientID id, queue_cb_t enqueue_cb, completed_cb_t complete_cb);
    void remove_callback(ClientID id);

    const CoreApiTable&             core_api() const { return _core_api; }
    const AmdExtTable&              ext_api() const { return _ext_api; }
    mutable std::mutex              cv_mutex;
    mutable std::condition_variable cv_ready_signal;
    hsa_signal_t                    block_signal;
    hsa_signal_t                    ready_signal;
    queue_state                     get_state() const;
    void                            set_state(queue_state state) const;

private:
    struct queue_signal_trace_stats_t
    {
        std::atomic<uint64_t> create_signal_calls          = {0};
        std::atomic<uint64_t> create_signal_total_ns       = {0};
        std::atomic<uint64_t> create_signal_max_ns         = {0};
        std::atomic<uint64_t> direct_create_signal_calls   = {0};
        std::atomic<uint64_t> async_register_calls         = {0};
        std::atomic<uint64_t> async_register_total_ns      = {0};
        std::atomic<uint64_t> async_register_max_ns        = {0};
        std::atomic<uint64_t> slot_prepare_calls           = {0};
        std::atomic<uint64_t> slot_prepare_total_ns        = {0};
        std::atomic<uint64_t> slot_prepare_max_ns          = {0};
        std::atomic<uint64_t> slot_builder_loops           = {0};
        std::atomic<uint64_t> slot_builder_create_attempts = {0};
        std::atomic<uint64_t> slot_builder_created         = {0};
        std::atomic<uint64_t> slot_acquire_attempts        = {0};
        std::atomic<uint64_t> slot_acquire_hits            = {0};
        std::atomic<uint64_t> slot_acquire_misses          = {0};
        std::atomic<uint64_t> slot_arm_calls               = {0};
        std::atomic<uint64_t> slot_complete_calls          = {0};
        std::atomic<uint64_t> retired_signal_calls         = {0};
        std::atomic<uint64_t> retired_signal_drained       = {0};
        std::atomic<uint64_t> pool_slots_last              = {0};
        std::atomic<uint64_t> pool_slots_max               = {0};
        std::atomic<uint64_t> pool_ready_last              = {0};
        std::atomic<uint64_t> pool_ready_max               = {0};
        std::atomic<uint64_t> pool_in_use_last             = {0};
        std::atomic<uint64_t> pool_in_use_max              = {0};
        std::atomic<uint64_t> pool_tombstones_last         = {0};
        std::atomic<uint64_t> pool_tombstones_max          = {0};
    };

    void ensure_async_signal_slot_builder_started() const;
    void notify_async_signal_slot_builder() const;
    void stop_async_signal_slot_builder() const;
    void build_async_signal_slots() const;
    void fill_async_signal_slots(uint64_t target_slots) const;
    void prepare_async_signal_slot(async_signal_slot* slot) const;
    void emit_queue_signal_trace(const char* reason) const;
    void maybe_emit_queue_signal_trace(const char* reason, uint64_t count) const;
    void update_queue_signal_pool_state(uint64_t total_slots,
                                        uint64_t ready_slots,
                                        uint64_t in_use_slots,
                                        uint64_t tombstone_slots) const;

    std::atomic<int>                     _notifiers             = {0};
    std::atomic<int64_t>                 _active_async_handlers = {0};
    std::atomic<int64_t>                 _active_async_packets  = {0};
    CoreApiTable                         _core_api              = {};
    AmdExtTable                          _ext_api               = {};
    const AgentCache&                    _agent;
    common::Synchronized<callback_map_t> _callbacks            = {};
    hsa_queue_t*                         _intercept_queue      = nullptr;
    mutable std::atomic<queue_state>     _state                = queue_state::normal;
    mutable std::mutex                   _retired_signals_mutex;
    mutable std::vector<hsa_signal_t>    _retired_signals      = {};
    mutable std::mutex                   _async_signal_slots_mutex;
    mutable std::vector<std::unique_ptr<async_signal_slot>> _async_signal_slots = {};
    mutable std::deque<async_signal_slot*> _ready_async_signal_slots = {};
    mutable std::mutex                   _async_signal_slot_builder_mutex;
    mutable std::condition_variable      _async_signal_slot_builder_cv = {};
    mutable std::thread                  _async_signal_slot_builder = {};
    mutable std::atomic<bool>            _async_signal_slot_builder_shutdown = {false};
    mutable std::atomic<bool>            _async_signal_slot_builder_started = {false};
    mutable std::atomic<bool>            _async_signal_slot_builder_requested = {false};
    mutable queue_signal_trace_stats_t   _queue_signal_trace = {};
    std::mutex                           _lock_queue;
    hsa_signal_t                         _active_kernels = {.handle = 0};
};

inline rocprofiler_queue_id_t
Queue::get_id() const
{
    return {.handle = intercept_queue()->id};
};

template <typename FuncT>
inline void
Queue::signal_callback(FuncT&& func) const
{
    _callbacks.rlock([&func](const auto& data) { func(data); });
}

template <typename FuncT>
void
Queue::lock_queue(FuncT&& func)
{
    std::unique_lock<std::mutex> lock(_lock_queue);
    func();
}

struct async_signal_handler_data
{
    std::shared_ptr<Queue::queue_info_session_t> session = {};
    Queue*                                       owner   = nullptr;
    async_signal_slot*                           slot    = nullptr;
    std::mutex                                   mutex   = {};
    std::atomic<bool>                            handled = false;
};

struct async_signal_slot
{
    Queue*                                     owner        = nullptr;
    hsa_signal_t                               signal       = {};
    std::shared_ptr<async_signal_handler_data> handler_data = {};
    std::mutex                                 mutex        = {};
    std::atomic<bool>                          in_use       = false;
    std::atomic<bool>                          ready        = false;
};
}  // namespace hsa
}  // namespace rocprofiler
