// MIT License
//
/* Copyright (c) 2022-2025 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/common/scope_destructor.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/code_object/code_object.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/hsa/details/fmt.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/kernel_dispatch/profiling_time.hpp"
#include "lib/rocprofiler-sdk/kernel_dispatch/tracing.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/hsa_adapter.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/service.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"
#include "lib/rocprofiler-sdk/tracing/tracing.hpp"

#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/external_correlation.h>
#include <rocprofiler-sdk/fwd.h>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <memory>
#include <thread>

// static assert for rocprofiler_packet ABI compatibility
static_assert(sizeof(hsa_ext_amd_aql_pm4_packet_t) == sizeof(hsa_kernel_dispatch_packet_t),
              "unexpected ABI incompatibility");
static_assert(sizeof(hsa_ext_amd_aql_pm4_packet_t) == sizeof(hsa_barrier_and_packet_t),
              "unexpected ABI incompatibility");
static_assert(sizeof(hsa_ext_amd_aql_pm4_packet_t) == sizeof(hsa_barrier_or_packet_t),
              "unexpected ABI incompatibility");
static_assert(offsetof(hsa_ext_amd_aql_pm4_packet_t, completion_signal) ==
                  offsetof(hsa_kernel_dispatch_packet_t, completion_signal),
              "unexpected ABI incompatibility");
static_assert(offsetof(hsa_ext_amd_aql_pm4_packet_t, completion_signal) ==
                  offsetof(hsa_barrier_and_packet_t, completion_signal),
              "unexpected ABI incompatibility");
static_assert(offsetof(hsa_ext_amd_aql_pm4_packet_t, completion_signal) ==
                  offsetof(hsa_barrier_or_packet_t, completion_signal),
              "unexpected ABI incompatibility");

namespace rocprofiler
{
namespace hsa
{
namespace
{
int
prefetched_async_signal_slot_count()
{
    static const auto _v =
        std::max(0, common::get_env("ROCPROFILER_PREFETCH_ASYNC_SIGNAL_SLOTS", 64));
    return _v;
}

bool
use_prefetched_async_signal_slots()
{
    return prefetched_async_signal_slot_count() > 0;
}

bool
queue_signal_trace_enabled()
{
    static const auto _v = common::get_env("ROCPROFILER_QUEUE_SIGNAL_TRACE", false);
    return _v;
}

uint64_t
queue_signal_trace_period()
{
    static const auto _v = static_cast<uint64_t>(
        std::max(1, common::get_env("ROCPROFILER_QUEUE_SIGNAL_TRACE_PERIOD", 32768)));
    return _v;
}

template <typename Tp>
void
update_atomic_max(std::atomic<Tp>& dst, Tp value)
{
    auto current = dst.load(std::memory_order_relaxed);
    while(current < value &&
          !dst.compare_exchange_weak(current, value, std::memory_order_relaxed))
    {}
}

template <typename DomainT, typename... Args>
inline bool
context_filter(const context::context* ctx, DomainT domain, Args... args)
{
    if constexpr(std::is_same<DomainT, rocprofiler_buffer_tracing_kind_t>::value)
    {
        return (ctx->buffered_tracer && ctx->buffered_tracer->domains(domain, args...));
    }
    else if constexpr(std::is_same<DomainT, rocprofiler_callback_tracing_kind_t>::value)
    {
        return (ctx->callback_tracer && ctx->callback_tracer->domains(domain, args...));
    }
    else
    {
        static_assert(common::mpl::assert_false<DomainT>::value, "unsupported domain type");
        return false;
    }
}

bool
context_filter(const context::context* ctx)
{
    return (context_filter(ctx, ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH) ||
            context_filter(ctx, ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH));
}

auto&
get_async_signal_handler_data_map()
{
    static auto _v = common::Synchronized<
        std::unordered_map<void*, std::shared_ptr<async_signal_handler_data>>>{};
    return _v;
}

void
register_async_signal_handler_data(const std::shared_ptr<async_signal_handler_data>& data)
{
    get_async_signal_handler_data_map().wlock([&](auto& map) { map.emplace(data.get(), data); });
}

std::shared_ptr<async_signal_handler_data>
get_async_signal_handler_data(void* data)
{
    return get_async_signal_handler_data_map().rlock([&](const auto& map) {
        auto itr = map.find(data);
        if(itr == map.end()) return std::shared_ptr<async_signal_handler_data>{};
        return itr->second;
    });
}

void
drain_async_signal_handler_data(const Queue& queue)
{
    get_async_signal_handler_data_map().wlock([&](auto& map) {
        for(auto itr = map.begin(); itr != map.end();)
        {
            auto& entry = itr->second;
            if(entry && entry->owner == &queue &&
               entry->handled.load(std::memory_order_acquire))
            {
                itr = map.erase(itr);
            }
            else
            {
                ++itr;
            }
        }
    });
}

void
ProcessDispatchCompletion(std::shared_ptr<Queue::queue_info_session_t>& shared_ptr_info,
                          bool                                          retire_signals)
{
    if(!shared_ptr_info) return;

    auto& queue_info_session = *shared_ptr_info;
    auto dispatch_time = kernel_dispatch::get_dispatch_time(queue_info_session);
    kernel_dispatch::dispatch_complete(queue_info_session, dispatch_time);

    // Calls our internal callbacks to callers who need to be notified post
    // kernel execution.
    queue_info_session.queue.signal_callback([&](const auto& map) {
        for(const auto& [client_id, cb_pair] : map)
        {
            cb_pair.second(queue_info_session.queue,
                           queue_info_session.kernel_pkt,
                           shared_ptr_info,
                           queue_info_session.inst_pkt,
                           dispatch_time);
        }
    });

    if(queue_info_session.is_serialized)
    {
        CHECK_NOTNULL(hsa::get_queue_controller())
            ->serializer(&queue_info_session.queue)
            .wlock([&](auto& serializer) {
                serializer.kernel_completion_signal(queue_info_session.queue);
            });
    }

    // Delete signals and packets, signal we have completed.
    if(queue_info_session.interrupt_signal.handle != 0u)
    {
#if !defined(NDEBUG)
        if(retire_signals)
        {
            CHECK_NOTNULL(hsa::get_queue_controller())->_debug_signals.wlock([&](auto& signals) {
                signals.erase(queue_info_session.interrupt_signal.handle);
            });
        }
#endif
        hsa::get_core_table()->hsa_signal_store_screlease_fn(queue_info_session.interrupt_signal,
                                                             -1);
        if(retire_signals)
        {
            queue_info_session.queue.retire_signal(queue_info_session.interrupt_signal);
        }
    }
    if(retire_signals &&
       queue_info_session.kernel_pkt.ext_amd_aql_pm4.completion_signal.handle != 0u)
    {
        queue_info_session.queue.retire_signal(
            queue_info_session.kernel_pkt.ext_amd_aql_pm4.completion_signal);
    }

    // we need to decrement this reference count at the end of the functions
    auto* _corr_id = queue_info_session.correlation_id;
    if(_corr_id)
    {
        ROCP_FATAL_IF(_corr_id->get_ref_count() == 0)
            << "reference counter for correlation id " << _corr_id->internal << " from thread "
            << _corr_id->thread_idx << " has no reference count";
        _corr_id->sub_kern_count();
        _corr_id->sub_ref_count();
    }

    queue_info_session.queue.async_complete();
    shared_ptr_info.reset();
}

bool
AsyncSignalHandler(hsa_signal_value_t /*signal_v*/, void* data)
{
    if(!data) return true;

    auto handler_data = get_async_signal_handler_data(data);
    if(!handler_data) return false;

    std::shared_ptr<Queue::queue_info_session_t> session = {};
    {
        std::lock_guard<std::mutex> lk{handler_data->mutex};
        session = handler_data->session;
        if(!session) return true;
        if(handler_data->handled.exchange(true, std::memory_order_acq_rel)) return false;
        handler_data->session.reset();
    }

    // if we have fully finalized, discard the retained session and return
    if(registration::get_fini_status() > 0)
    {
        session.reset();
        return false;
    }

    if(session) ProcessDispatchCompletion(session, true);

    if(handler_data->slot && handler_data->owner)
    {
        handler_data->owner->complete_async_signal_slot(handler_data->slot);
    }

    return false;
}

template <typename Integral = uint64_t>
constexpr Integral
bit_mask(int first, int last)
{
    assert(last >= first && "Error: hsa_support::bit_mask -> invalid argument");
    size_t num_bits = last - first + 1;
    return ((num_bits >= sizeof(Integral) * 8) ? ~Integral{0}
                                               /* num_bits exceed the size of Integral */
                                               : ((Integral{1} << num_bits) - 1))
           << first;
}

/* Extract bits [last:first] from t.  */
template <typename Integral>
constexpr Integral
bit_extract(Integral x, int first, int last)
{
    return (x >> first) & bit_mask<Integral>(0, last - first);
}

/**
 * @brief This function is a queue write interceptor. It intercepts the
 * packet write function. Creates an instance of packet class with the raw
 * pointer. invoke the populate function of the packet class which returns a
 * pointer to the packet. This packet is written into the queue by this
 * interceptor by invoking the writer function.
 */
void
WriteInterceptor(const void* packets,
                 uint64_t    pkt_count,
                 uint64_t,
                 void*                                 data,
                 hsa_amd_queue_intercept_packet_writer writer)
{
    if(registration::get_fini_status() > 0)
    {
        writer(packets, pkt_count);
        return;
    }

    using callback_record_t = Queue::queue_info_session_t::callback_record_t;

    // unique sequence id for the dispatch
    static auto sequence_counter = std::atomic<rocprofiler_dispatch_id_t>{0};

    auto&& CreateBarrierPacket = [](hsa_signal_t*                    dependency_signal,
                                    hsa_signal_t*                    completion_signal,
                                    std::vector<rocprofiler_packet>& _packets) {
        hsa_barrier_and_packet_t barrier{};
        barrier.header = HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE;
        barrier.header |= 1 << HSA_PACKET_HEADER_BARRIER;
        if(dependency_signal != nullptr) barrier.dep_signal[0] = *dependency_signal;
        if(completion_signal != nullptr) barrier.completion_signal = *completion_signal;
        _packets.emplace_back(barrier);
    };

    ROCP_FATAL_IF(data == nullptr) << "WriteInterceptor was not passed a pointer to the queue";

    auto& queue = *static_cast<Queue*>(data);
    queue.interceptor_started();
    auto _interceptor_dtor =
        common::scope_destructor{[&queue]() { queue.interceptor_complete(); }};

    if(queue.get_state() != queue_state::normal || !queue.has_active_kernel_signal())
    {
        writer(packets, pkt_count);
        return;
    }

    // Graph replay can keep ROCr async-handler bookkeeping alive briefly after a dispatch
    // completes. Retire completion signals in the callback and only reclaim them once the queue
    // is fully idle.
    if(queue.active_interceptors() == 1 && queue.active_async_handlers() == 0 &&
       queue.active_async_packets() == 0)
    {
        drain_async_signal_handler_data(queue);
        queue.drain_retired_signals();
    }

    // We have no packets or no one who needs to be notified, do nothing.
    if(pkt_count == 0 ||
       (queue.get_notifiers() == 0 && context::get_active_contexts(context_filter).empty()))
    {
        writer(packets, pkt_count);
        return;
    }

    auto tracing_data_v = tracing::tracing_data{};
    tracing::populate_contexts(ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH,
                               ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
                               tracing_data_v);
    // these are for the services (dispatch counter collection, pc sampling, ATT) which use
    // the queue/queue_controller callback mechanism
    const auto queue_callback_context_filter = [](const context::context* ctx) {
        return (ctx->counter_collection || ctx->pc_sampler || ctx->dispatch_thread_trace);
    };

    for(const auto* itr : context::get_active_contexts(queue_callback_context_filter))
        tracing_data_v.external_correlation_ids.emplace(itr, tracing::empty_user_data);

    const auto* packets_arr         = static_cast<const rocprofiler_packet*>(packets);
    auto        transformed_packets = std::vector<rocprofiler_packet>{};

    // Searching accross all the packets given during this write
    for(size_t i = 0; i < pkt_count; ++i)
    {
        const auto& original_packet = packets_arr[i].kernel_dispatch;
        auto        packet_type     = bit_extract(original_packet.header,
                                       HSA_PACKET_HEADER_TYPE,
                                       HSA_PACKET_HEADER_TYPE + HSA_PACKET_HEADER_WIDTH_TYPE - 1);
        if(packet_type != HSA_PACKET_TYPE_KERNEL_DISPATCH)
        {
            transformed_packets.emplace_back(packets_arr[i]);
            continue;
        }

        const auto               current_tid  = common::get_tid_no_cache();
        auto*                    corr_id      = (context::thread_has_correlation_id(current_tid))
                                                    ? context::get_latest_correlation_id()
                                                    : nullptr;
        context::correlation_id* _corr_id_pop = nullptr;

        // Graph replay and other queue writes can be intercepted on the HSA async event thread,
        // outside any active host API scope. In that case we still want the dispatch trace, but
        // synthesizing a fresh correlation ID per dispatch here adds allocator pressure on the
        // async thread and is not required for kernel trace collection.
        if(corr_id)
        {
            // increase the reference count to denote that this correlation id is being used in a
            // kernel
            corr_id->add_ref_count();
            corr_id->add_kern_count();
        }

        auto thr_id           = (corr_id) ? corr_id->thread_idx : current_tid;
        auto user_data        = rocprofiler_user_data_t{.value = 0};
        auto internal_corr_id = (corr_id) ? corr_id->internal : 0;
        auto ancestor_corr_id = (corr_id) ? corr_id->ancestor : 0;

        // if we constructed a correlation id, this decrements the reference count after the
        // underlying function returns
        auto _corr_id_dtor = common::scope_destructor{[_corr_id_pop]() {
            if(_corr_id_pop)
            {
                context::pop_latest_correlation_id(_corr_id_pop);
                _corr_id_pop->sub_ref_count();
            }
        }};

        tracing::populate_external_correlation_ids(
            tracing_data_v.external_correlation_ids,
            thr_id,
            ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_KERNEL_DISPATCH,
            ROCPROFILER_KERNEL_DISPATCH_ENQUEUE,
            internal_corr_id);

        queue.async_started();

        const auto     original_completion_signal = original_packet.completion_signal;
        const bool     existing_completion_signal = (original_completion_signal.handle != 0);
        const uint64_t kernel_id = code_object::get_kernel_id(original_packet.kernel_object);

        // Copy kernel pkt, copy is to allow for signal to be modified
        rocprofiler_packet kernel_pkt = packets_arr[i];

        // computes the "size" based on the offset of reserved_padding field
        constexpr auto kernel_dispatch_info_rt_size =
            common::compute_runtime_sizeof<rocprofiler_kernel_dispatch_info_t>();

        static_assert(kernel_dispatch_info_rt_size < sizeof(rocprofiler_kernel_dispatch_info_t),
                      "failed to compute size field based on offset of reserved_padding field");

        auto dispatch_id     = ++sequence_counter;
        auto callback_record = callback_record_t{
            sizeof(callback_record_t),
            rocprofiler_timestamp_t{0},
            rocprofiler_timestamp_t{0},
            rocprofiler_kernel_dispatch_info_t{
                .size                 = kernel_dispatch_info_rt_size,
                .agent_id             = queue.get_agent().get_rocp_agent()->id,
                .queue_id             = queue.get_id(),
                .kernel_id            = kernel_id,
                .dispatch_id          = dispatch_id,
                .private_segment_size = kernel_pkt.kernel_dispatch.private_segment_size,
                .group_segment_size   = kernel_pkt.kernel_dispatch.group_segment_size,
                .workgroup_size   = rocprofiler_dim3_t{kernel_pkt.kernel_dispatch.workgroup_size_x,
                                                     kernel_pkt.kernel_dispatch.workgroup_size_y,
                                                     kernel_pkt.kernel_dispatch.workgroup_size_z},
                .grid_size        = rocprofiler_dim3_t{kernel_pkt.kernel_dispatch.grid_size_x,
                                                kernel_pkt.kernel_dispatch.grid_size_y,
                                                kernel_pkt.kernel_dispatch.grid_size_z},
                .reserved_padding = {0}}};

        {
            auto tracer_data = callback_record;
            tracing::execute_phase_enter_callbacks(tracing_data_v.callback_contexts,
                                                   thr_id,
                                                   internal_corr_id,
                                                   tracing_data_v.external_correlation_ids,
                                                   ancestor_corr_id,
                                                   ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH,
                                                   ROCPROFILER_KERNEL_DISPATCH_ENQUEUE,
                                                   tracer_data);
        }

        // map all the external correlation ids (after enqueue enter phase) for all the contexts
        // captured by the info session
        tracing::update_external_correlation_ids(
            tracing_data_v.external_correlation_ids,
            thr_id,
            ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_KERNEL_DISPATCH);

        // Stores the instrumentation pkt (i.e. AQL packets for counter collection)
        // along with an ID of the client we got the packet from (this will be returned via
        // completed_cb_t)
        auto inst_pkt = inst_pkt_t{};

        // True if any service (ATT,SPM,CC) requests this dispatch to be serialized
        bool bRequest_Serialize = false;

        // Signal callbacks that a kernel_pkt is being enqueued
        queue.signal_callback([&](const auto& map) {
            for(const auto& [client_id, cb_pair] : map)
            {
                auto [packet, bSerial] = cb_pair.first(queue,
                                                       kernel_pkt,
                                                       kernel_id,
                                                       dispatch_id,
                                                       &user_data,
                                                       tracing_data_v.external_correlation_ids,
                                                       corr_id);
                bRequest_Serialize |= bSerial;
                if(packet) inst_pkt.push_back(std::make_pair(std::move(packet), client_id));
            }
        });

        bool inserted_before = false;
        if(bRequest_Serialize)
        {
            inserted_before = true;
            CHECK_NOTNULL(hsa::get_queue_controller())
                ->serializer(&queue)
                .rlock([&](const auto& serializer) {
                    for(auto& s_pkt : serializer.kernel_dispatch(queue))
                        transformed_packets.emplace_back(s_pkt.ext_amd_aql_pm4);
                });
        }
        for(const auto& pkt_injection : inst_pkt)
        {
            for(const auto& pkt : pkt_injection.first->before_krn_pkt)
            {
                inserted_before = true;
                transformed_packets.emplace_back(pkt);
            }
        }

        const bool injected_end_pkt =
            std::any_of(inst_pkt.begin(), inst_pkt.end(), [](const auto& pkt_injection) {
                return !pkt_injection.first->after_krn_pkt.empty();
            });

        async_signal_slot* prefetched_signal_slot = nullptr;

        // When enabled, consume a background-built one-shot completion slot in the common path.
        // If the ready pool is empty, fall back to the stable per-dispatch path below.
        if(!injected_end_pkt && use_prefetched_async_signal_slots())
        {
            prefetched_signal_slot = queue.acquire_async_signal_slot();
            if(prefetched_signal_slot)
            {
                kernel_pkt.kernel_dispatch.completion_signal = prefetched_signal_slot->signal;
            }
            else
            {
                queue.create_signal(0, &kernel_pkt.kernel_dispatch.completion_signal, true);
            }
        }
        else
        {
            // create our own signal that we can get a callback on. if there is an original
            // completion signal we will create a barrier packet, assign the original completion
            // signal that that barrier packet, and add it right after the kernel packet
            queue.create_signal(0, &kernel_pkt.kernel_dispatch.completion_signal, true);
        }

#if ROCPROFILER_SDK_HSA_PC_SAMPLING > 0
        if(pc_sampling::is_pc_sample_service_configured(queue.get_agent().get_rocp_agent()->id))
        {
            transformed_packets.emplace_back(pc_sampling::hsa::generate_marker_packet_for_kernel(
                corr_id, tracing_data_v.external_correlation_ids, dispatch_id));
        }
#endif

        // emplace the kernel packet
        transformed_packets.emplace_back(kernel_pkt);
        // If a profiling packet was inserted, wait for completion before executing the dispatch
        if(inserted_before)
            transformed_packets.back().kernel_dispatch.header |= 1 << HSA_PACKET_HEADER_BARRIER;

        // if the original completion signal exists, trigger it via a barrier packet
        if(existing_completion_signal)
        {
            auto barrier   = hsa_barrier_and_packet_t{};
            barrier.header = HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE;
            barrier.header |= (1 << HSA_PACKET_HEADER_BARRIER);
            barrier.completion_signal = original_completion_signal;
            transformed_packets.emplace_back(barrier);
        }

        for(const auto& pkt_injection : inst_pkt)
        {
            for(const auto& pkt : pkt_injection.first->after_krn_pkt)
            {
                transformed_packets.emplace_back(pkt);
            }
        }

        auto completion_signal = hsa_signal_t{.handle = 0};
        auto interrupt_signal  = hsa_signal_t{.handle = 0};
        if(injected_end_pkt)
        {
            // Adding a barrier packet with the original packet's completion signal.
            queue.create_signal(0, &interrupt_signal, true);
            completion_signal                                            = interrupt_signal;
            transformed_packets.back().ext_amd_aql_pm4.completion_signal = interrupt_signal;
            CreateBarrierPacket(&interrupt_signal, &interrupt_signal, transformed_packets);
        }
        else
        {
            completion_signal = kernel_pkt.kernel_dispatch.completion_signal;
        }

        ROCP_FATAL_IF(packet_type != HSA_PACKET_TYPE_KERNEL_DISPATCH)
            << "get_kernel_id below might need to be updated";

        // Enqueue the signal into the handler. Will call completed_cb when
        // signal completes.

        {
            Queue::queue_info_session_t info_session{.queue            = queue,
                                                     .inst_pkt         = std::move(inst_pkt),
                                                     .interrupt_signal = interrupt_signal,
                                                     .tid              = thr_id,
                                                     .enqueue_ts       = common::timestamp_ns(),
                                                     .user_data        = user_data,
                                                     .correlation_id   = corr_id,
                                                     .kernel_pkt       = kernel_pkt,
                                                     .callback_record  = callback_record,
                                                     .tracing_data     = tracing_data_v,
                                                     .is_serialized    = bRequest_Serialize};

            auto shared = std::make_shared<Queue::queue_info_session_t>(std::move(info_session));
            if(prefetched_signal_slot)
            {
                queue.arm_async_signal_slot(prefetched_signal_slot, shared);
            }
            else
            {
                auto async_handler_data = std::make_shared<async_signal_handler_data>();
                async_handler_data->session = shared;
                async_handler_data->owner   = &queue;
                register_async_signal_handler_data(async_handler_data);

                queue.signal_async_handler(completion_signal, async_handler_data.get());
            }

            auto tracer_data = callback_record;
            tracing::execute_phase_exit_callbacks(tracing_data_v.callback_contexts,
                                                  tracing_data_v.external_correlation_ids,
                                                  ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH,
                                                  ROCPROFILER_KERNEL_DISPATCH_ENQUEUE,
                                                  tracer_data);
        }
    }

    // Command is only executed if GLOG_v=2 or higher, otherwise it is a no-op
    ROCP_TRACE << fmt::format(
        "QueueID {}: {}", queue.get_id().handle, fmt::join(transformed_packets, fmt::format(" ")));

    writer(transformed_packets.data(), transformed_packets.size());
}
}  // namespace

Queue::Queue(const AgentCache& agent, CoreApiTable table)
: _core_api(table)
, _agent(agent)
{
    _core_api.hsa_signal_create_fn(0, 0, nullptr, &_active_kernels);
}

Queue::Queue(const AgentCache&  agent,
             uint32_t           size,
             hsa_queue_type32_t type,
             void (*callback)(hsa_status_t status, hsa_queue_t* source, void* data),
             void*         data,
             uint32_t      private_segment_size,
             uint32_t      group_segment_size,
             CoreApiTable  core_api,
             AmdExtTable   ext_api,
             hsa_queue_t** queue)
: _core_api(core_api)
, _ext_api(ext_api)
, _agent(agent)
{
    ROCP_HSA_TABLE_CALL(FATAL,
                        _ext_api.hsa_amd_queue_intercept_create_fn(_agent.get_hsa_agent(),
                                                                   size,
                                                                   type,
                                                                   callback,
                                                                   data,
                                                                   private_segment_size,
                                                                   group_segment_size,
                                                                   &_intercept_queue))
        << "Could not create intercept queue";

    ROCP_HSA_TABLE_CALL(FATAL,
                        _ext_api.hsa_amd_profiling_set_profiler_enabled_fn(_intercept_queue, true))
        << "Could not setup intercept profiler";

    if(!context::get_registered_contexts([](const context::context* ctx) {
            return (ctx->counter_collection || ctx->device_counter_collection ||
                    ctx->dispatch_thread_trace || ctx->device_thread_trace);
        }).empty())
    {
        CHECK(_agent.cpu_pool().handle != 0);
        CHECK(_agent.get_hsa_agent().handle != 0);

        // Set state of the queue to allow profiling
        aql::set_profiler_active_on_queue(
            _agent.cpu_pool(), _agent.get_hsa_agent(), [&](hsa::rocprofiler_packet pkt) {
                hsa_signal_t completion;
                create_signal(0, &completion);
                pkt.ext_amd_aql_pm4.completion_signal = completion;
                counters::submitPacket(_intercept_queue, &pkt);
                constexpr auto timeout_hint =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::seconds{1});
                hsa_signal_value_t val;
                for(int i = 0; i < 3; i++)
                {
                    val = core_api.hsa_signal_wait_scacquire_fn(completion,
                                                                HSA_SIGNAL_CONDITION_EQ,
                                                                0,
                                                                timeout_hint.count(),
                                                                HSA_WAIT_STATE_ACTIVE);
                    if(val == 0)
                    {
                        core_api.hsa_signal_destroy_fn(completion);
                        return;
                    }
                }
                ROCP_FATAL << "Could not set agent to be profiled - Signal Value: " << val;
            });
    }

    create_signal(0, &ready_signal);
    create_signal(0, &block_signal);
    create_signal(0, &_active_kernels);
    _core_api.hsa_signal_store_screlease_fn(ready_signal, 0);
    _core_api.hsa_signal_store_screlease_fn(_active_kernels, 0);

    ROCP_HSA_TABLE_CALL(
        FATAL,
        _ext_api.hsa_amd_queue_intercept_register_fn(_intercept_queue, WriteInterceptor, this))
        << "Could not register interceptor";

    if(use_prefetched_async_signal_slots())
    {
        fill_async_signal_slots(prefetched_async_signal_slot_count());
        ensure_async_signal_slot_builder_started();
    }
    *queue = _intercept_queue;
}

Queue::Queue(
    const AgentCache&       agent,
    CoreApiTable            core_api,
    AmdExtTable             ext_api,
    hsa_queue_t*            queue,
    set_write_interceptor_t set_write_interceptor)  // NOLINT(performance-unnecessary-value-param)
: _core_api(core_api)
, _ext_api(ext_api)
, _agent(agent)
, _intercept_queue(queue)
{
    if(!context::get_registered_contexts([](const context::context* ctx) {
            return (ctx->counter_collection || ctx->device_counter_collection ||
                    ctx->dispatch_thread_trace || ctx->device_thread_trace);
        }).empty())
    {
        CHECK(_agent.cpu_pool().handle != 0);
        CHECK(_agent.get_hsa_agent().handle != 0);

        // Set state of the queue to allow profiling
        aql::set_profiler_active_on_queue(
            _agent.cpu_pool(), _agent.get_hsa_agent(), [&](hsa::rocprofiler_packet pkt) {
                hsa_signal_t completion;
                create_signal(0, &completion);
                pkt.ext_amd_aql_pm4.completion_signal = completion;
                counters::submitPacket(_intercept_queue, &pkt);
                constexpr auto timeout_hint =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::seconds{1});
                if(core_api.hsa_signal_wait_relaxed_fn(completion,
                                                       HSA_SIGNAL_CONDITION_EQ,
                                                       0,
                                                       timeout_hint.count(),
                                                       HSA_WAIT_STATE_ACTIVE) != 0)
                {
                    ROCP_FATAL << "Could not set agent to be profiled";
                }
                core_api.hsa_signal_destroy_fn(completion);
            });
    }

    create_signal(0, &ready_signal);
    create_signal(0, &block_signal);
    create_signal(0, &_active_kernels);
    _core_api.hsa_signal_store_screlease_fn(ready_signal, 0);
    _core_api.hsa_signal_store_screlease_fn(_active_kernels, 0);

    set_write_interceptor(WriteInterceptor, this);

    if(use_prefetched_async_signal_slots())
    {
        fill_async_signal_slots(prefetched_async_signal_slot_count());
        ensure_async_signal_slot_builder_started();
    }
}

Queue::~Queue()
{
    sync();
    emit_queue_signal_trace("destructor-sync");
    stop_async_signal_slot_builder();
    destroy_async_signal_slots();
    emit_queue_signal_trace("destructor-destroy");
    _core_api.hsa_signal_destroy_fn(_active_kernels);
}

void
Queue::update_queue_signal_pool_state(uint64_t total_slots,
                                      uint64_t ready_slots,
                                      uint64_t in_use_slots,
                                      uint64_t tombstone_slots) const
{
    _queue_signal_trace.pool_slots_last.store(total_slots, std::memory_order_relaxed);
    _queue_signal_trace.pool_ready_last.store(ready_slots, std::memory_order_relaxed);
    _queue_signal_trace.pool_in_use_last.store(in_use_slots, std::memory_order_relaxed);
    _queue_signal_trace.pool_tombstones_last.store(tombstone_slots, std::memory_order_relaxed);
    update_atomic_max(_queue_signal_trace.pool_slots_max, total_slots);
    update_atomic_max(_queue_signal_trace.pool_ready_max, ready_slots);
    update_atomic_max(_queue_signal_trace.pool_in_use_max, in_use_slots);
    update_atomic_max(_queue_signal_trace.pool_tombstones_max, tombstone_slots);
}

void
Queue::maybe_emit_queue_signal_trace(const char* reason, uint64_t count) const
{
    if(!queue_signal_trace_enabled()) return;
    if(count == 0 || (count % queue_signal_trace_period()) != 0) return;
    emit_queue_signal_trace(reason);
}

void
Queue::emit_queue_signal_trace(const char* reason) const
{
    if(!queue_signal_trace_enabled()) return;

    const auto avg_us = [](uint64_t total_ns, uint64_t calls) {
        return (calls > 0) ? static_cast<double>(total_ns) / (1000.0 * calls) : 0.0;
    };

    auto total_slots     = uint64_t{0};
    auto ready_slots     = uint64_t{0};
    auto in_use_slots    = uint64_t{0};
    auto tombstone_slots = uint64_t{0};
    {
        std::lock_guard<std::mutex> lk{_async_signal_slots_mutex};
        for(const auto& slot : _async_signal_slots)
        {
            if(!slot) continue;
            ++total_slots;

            const auto ready = slot->ready.load(std::memory_order_acquire);
            const auto in_use = slot->in_use.load(std::memory_order_acquire);
            if(ready) ++ready_slots;
            if(in_use) ++in_use_slots;
            if(slot->signal.handle == 0u && !ready && !in_use) ++tombstone_slots;
        }
    }
    update_queue_signal_pool_state(total_slots, ready_slots, in_use_slots, tombstone_slots);

    auto retired_pending = uint64_t{0};
    {
        std::lock_guard<std::mutex> lk{_retired_signals_mutex};
        retired_pending = _retired_signals.size();
    }

    const auto create_calls = _queue_signal_trace.create_signal_calls.load(std::memory_order_relaxed);
    const auto create_total_ns =
        _queue_signal_trace.create_signal_total_ns.load(std::memory_order_relaxed);
    const auto register_calls =
        _queue_signal_trace.async_register_calls.load(std::memory_order_relaxed);
    const auto register_total_ns =
        _queue_signal_trace.async_register_total_ns.load(std::memory_order_relaxed);
    const auto prepare_calls =
        _queue_signal_trace.slot_prepare_calls.load(std::memory_order_relaxed);
    const auto prepare_total_ns =
        _queue_signal_trace.slot_prepare_total_ns.load(std::memory_order_relaxed);

    fmt::print(
        stderr,
        "ROCP queue-signal queue={} reason={} create_calls={} create_avg_us={:.3f} "
        "create_max_us={:.3f} direct_create_calls={} register_calls={} register_avg_us={:.3f} "
        "register_max_us={:.3f} prepare_calls={} prepare_avg_us={:.3f} prepare_max_us={:.3f} "
        "builder_loops={} builder_create_attempts={} builder_created={} acquire_attempts={} "
        "acquire_hits={} acquire_misses={} arm_calls={} complete_calls={} retired_calls={} "
        "retired_drained={} retired_pending={} pool_slots={} pool_slots_max={} ready={} "
        "ready_max={} in_use={} in_use_max={} tombstones={} tombstones_max={} "
        "active_handlers={} active_packets={} active_interceptors={}\n",
        get_id().handle,
        reason,
        create_calls,
        avg_us(create_total_ns, create_calls),
        static_cast<double>(
            _queue_signal_trace.create_signal_max_ns.load(std::memory_order_relaxed)) /
            1000.0,
        _queue_signal_trace.direct_create_signal_calls.load(std::memory_order_relaxed),
        register_calls,
        avg_us(register_total_ns, register_calls),
        static_cast<double>(
            _queue_signal_trace.async_register_max_ns.load(std::memory_order_relaxed)) /
            1000.0,
        prepare_calls,
        avg_us(prepare_total_ns, prepare_calls),
        static_cast<double>(
            _queue_signal_trace.slot_prepare_max_ns.load(std::memory_order_relaxed)) /
            1000.0,
        _queue_signal_trace.slot_builder_loops.load(std::memory_order_relaxed),
        _queue_signal_trace.slot_builder_create_attempts.load(std::memory_order_relaxed),
        _queue_signal_trace.slot_builder_created.load(std::memory_order_relaxed),
        _queue_signal_trace.slot_acquire_attempts.load(std::memory_order_relaxed),
        _queue_signal_trace.slot_acquire_hits.load(std::memory_order_relaxed),
        _queue_signal_trace.slot_acquire_misses.load(std::memory_order_relaxed),
        _queue_signal_trace.slot_arm_calls.load(std::memory_order_relaxed),
        _queue_signal_trace.slot_complete_calls.load(std::memory_order_relaxed),
        _queue_signal_trace.retired_signal_calls.load(std::memory_order_relaxed),
        _queue_signal_trace.retired_signal_drained.load(std::memory_order_relaxed),
        retired_pending,
        total_slots,
        _queue_signal_trace.pool_slots_max.load(std::memory_order_relaxed),
        ready_slots,
        _queue_signal_trace.pool_ready_max.load(std::memory_order_relaxed),
        in_use_slots,
        _queue_signal_trace.pool_in_use_max.load(std::memory_order_relaxed),
        tombstone_slots,
        _queue_signal_trace.pool_tombstones_max.load(std::memory_order_relaxed),
        active_async_handlers(),
        active_async_packets(),
        active_interceptors());
}

void
Queue::signal_async_handler(const hsa_signal_t& signal, void* data) const
{
#if !defined(NDEBUG)
    CHECK_NOTNULL(hsa::get_queue_controller())->_debug_signals.wlock([&](auto& signals) {
        signals[signal.handle] = signal;
    });
#endif
    const auto start_ns = common::timestamp_ns();
    hsa_status_t status = _ext_api.hsa_amd_signal_async_handler_fn(
        signal, HSA_SIGNAL_CONDITION_LT, 1, AsyncSignalHandler, data);
    const auto elapsed_ns = static_cast<uint64_t>(common::timestamp_ns() - start_ns);
    const auto calls =
        _queue_signal_trace.async_register_calls.fetch_add(1, std::memory_order_relaxed) + 1;
    _queue_signal_trace.async_register_total_ns.fetch_add(elapsed_ns, std::memory_order_relaxed);
    update_atomic_max(_queue_signal_trace.async_register_max_ns, elapsed_ns);
    maybe_emit_queue_signal_trace("async-register", calls);
    ROCP_FATAL_IF(status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK)
        << "Error: hsa_amd_signal_async_handler failed with error code " << status
        << " :: " << hsa::get_hsa_status_string(status);
}

void
Queue::arm_async_signal_slot(async_signal_slot*                    slot,
                             std::shared_ptr<queue_info_session_t> session) const
{
    ROCP_FATAL_IF(slot == nullptr) << "attempting to arm a null async signal slot";

    std::shared_ptr<async_signal_handler_data> handler_data = {};
    {
        std::lock_guard<std::mutex> lk{slot->mutex};
        handler_data = slot->handler_data;
    }

    ROCP_FATAL_IF(!handler_data)
        << "attempting to arm async signal slot without a registered handler";

    {
        std::lock_guard<std::mutex> lk{handler_data->mutex};
        handler_data->session = std::move(session);
    }

    const auto calls =
        _queue_signal_trace.slot_arm_calls.fetch_add(1, std::memory_order_relaxed) + 1;

    notify_async_signal_slot_builder();
    maybe_emit_queue_signal_trace("slot-arm", calls);
}

async_signal_slot*
Queue::acquire_async_signal_slot() const
{
    if(!use_prefetched_async_signal_slots()) return nullptr;

    ensure_async_signal_slot_builder_started();

    std::lock_guard<std::mutex> lk{_async_signal_slots_mutex};
    const auto attempts =
        _queue_signal_trace.slot_acquire_attempts.fetch_add(1, std::memory_order_relaxed) + 1;
    auto total_slots  = static_cast<uint64_t>(_async_signal_slots.size());
    auto ready_slots  = static_cast<uint64_t>(_ready_async_signal_slots.size());
    auto in_use_slots = uint64_t{0};
    auto tombstone_slots = uint64_t{0};

    while(!_ready_async_signal_slots.empty())
    {
        auto* slot = _ready_async_signal_slots.front();
        _ready_async_signal_slots.pop_front();
        if(!slot) continue;

        bool expected = false;
        if(slot->ready.load(std::memory_order_acquire) &&
           slot->in_use.compare_exchange_strong(expected, true, std::memory_order_acq_rel))
        {
            slot->ready.store(false, std::memory_order_release);
            ready_slots = static_cast<uint64_t>(_ready_async_signal_slots.size());
            update_queue_signal_pool_state(total_slots, ready_slots, in_use_slots, tombstone_slots);
            const auto hits =
                _queue_signal_trace.slot_acquire_hits.fetch_add(1, std::memory_order_relaxed) + 1;
            notify_async_signal_slot_builder();
            maybe_emit_queue_signal_trace("slot-acquire-hit", hits);
            return slot;
        }
    }
    update_queue_signal_pool_state(total_slots, ready_slots, in_use_slots, tombstone_slots);
    const auto misses =
        _queue_signal_trace.slot_acquire_misses.fetch_add(1, std::memory_order_relaxed) + 1;
    notify_async_signal_slot_builder();
    maybe_emit_queue_signal_trace("slot-acquire-attempt", attempts);
    maybe_emit_queue_signal_trace("slot-acquire-miss", misses);
    return nullptr;
}

void
Queue::ensure_async_signal_slot_builder_started() const
{
    if(_async_signal_slot_builder_started.load(std::memory_order_acquire)) return;

    std::lock_guard<std::mutex> lk{_async_signal_slot_builder_mutex};
    if(_async_signal_slot_builder_started.load(std::memory_order_relaxed)) return;

    _async_signal_slot_builder_shutdown.store(false, std::memory_order_release);
    _async_signal_slot_builder_requested.store(true, std::memory_order_release);
    _async_signal_slot_builder = std::thread{[this]() { build_async_signal_slots(); }};
    _async_signal_slot_builder_started.store(true, std::memory_order_release);
    _async_signal_slot_builder_cv.notify_one();
}

void
Queue::notify_async_signal_slot_builder() const
{
    if(!_async_signal_slot_builder_started.load(std::memory_order_acquire)) return;
    _async_signal_slot_builder_requested.store(true, std::memory_order_release);
    _async_signal_slot_builder_cv.notify_one();
}

void
Queue::stop_async_signal_slot_builder() const
{
    if(!_async_signal_slot_builder_started.load(std::memory_order_acquire)) return;

    _async_signal_slot_builder_shutdown.store(true, std::memory_order_release);
    _async_signal_slot_builder_cv.notify_all();
    if(_async_signal_slot_builder.joinable()) _async_signal_slot_builder.join();
    _async_signal_slot_builder_started.store(false, std::memory_order_release);
}

void
Queue::fill_async_signal_slots(uint64_t target_slots) const
{
    while(!_async_signal_slot_builder_shutdown.load(std::memory_order_acquire))
    {
        auto ready_slots     = uint64_t{0};
        auto in_use_slots    = uint64_t{0};
        auto tombstone_slots = uint64_t{0};
        auto total_slots     = uint64_t{0};
        {
            std::lock_guard<std::mutex> lk{_async_signal_slots_mutex};
            ready_slots = static_cast<uint64_t>(_ready_async_signal_slots.size());
            total_slots = static_cast<uint64_t>(_async_signal_slots.size());
        }
        update_queue_signal_pool_state(total_slots, ready_slots, in_use_slots, tombstone_slots);

        if(static_cast<uint64_t>(ready_slots) >= target_slots) break;

        auto create_count = static_cast<int>(target_slots - ready_slots);
        _queue_signal_trace.slot_builder_create_attempts.fetch_add(create_count,
                                                                   std::memory_order_relaxed);

        for(int i = 0; i < create_count; ++i)
        {
            if(_async_signal_slot_builder_shutdown.load(std::memory_order_acquire)) break;

            auto slot      = std::make_unique<async_signal_slot>();
            auto* slot_ptr = slot.get();
            slot_ptr->owner = const_cast<Queue*>(this);
            prepare_async_signal_slot(slot_ptr);
            {
                std::lock_guard<std::mutex> lk{_async_signal_slots_mutex};
                _async_signal_slots.emplace_back(std::move(slot));
                _ready_async_signal_slots.emplace_back(slot_ptr);
            }
            _queue_signal_trace.slot_builder_created.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

void
Queue::prepare_async_signal_slot(async_signal_slot* slot) const
{
    ROCP_FATAL_IF(slot == nullptr) << "attempting to prepare a null async signal slot";
    const auto start_ns = common::timestamp_ns();

    auto signal = hsa_signal_t{.handle = 0};
    create_signal(0, &signal);
    auto handler_data     = std::make_shared<async_signal_handler_data>();
    handler_data->owner   = const_cast<Queue*>(this);
    handler_data->slot    = slot;
    register_async_signal_handler_data(handler_data);
    signal_async_handler(signal, handler_data.get());

    {
        std::lock_guard<std::mutex> lk{slot->mutex};
        slot->owner        = const_cast<Queue*>(this);
        slot->signal       = signal;
        slot->handler_data = std::move(handler_data);
    }
    slot->ready.store(true, std::memory_order_release);
    slot->in_use.store(false, std::memory_order_release);
    const auto elapsed_ns = static_cast<uint64_t>(common::timestamp_ns() - start_ns);
    const auto calls =
        _queue_signal_trace.slot_prepare_calls.fetch_add(1, std::memory_order_relaxed) + 1;
    _queue_signal_trace.slot_prepare_total_ns.fetch_add(elapsed_ns, std::memory_order_relaxed);
    update_atomic_max(_queue_signal_trace.slot_prepare_max_ns, elapsed_ns);
    maybe_emit_queue_signal_trace("slot-prepare", calls);
}

void
Queue::build_async_signal_slots() const
{
    while(true)
    {
        _queue_signal_trace.slot_builder_loops.fetch_add(1, std::memory_order_relaxed);
        {
            std::unique_lock<std::mutex> lk{_async_signal_slot_builder_mutex};
            _async_signal_slot_builder_cv.wait(lk, [this]() {
                return _async_signal_slot_builder_shutdown.load(std::memory_order_acquire) ||
                       _async_signal_slot_builder_requested.load(std::memory_order_acquire);
            });
        }

        if(_async_signal_slot_builder_shutdown.load(std::memory_order_acquire)) break;
        _async_signal_slot_builder_requested.store(false, std::memory_order_release);

        fill_async_signal_slots(prefetched_async_signal_slot_count());

        const auto loops = _queue_signal_trace.slot_builder_loops.load(std::memory_order_relaxed);
        maybe_emit_queue_signal_trace("slot-builder", loops);
    }

}

void
Queue::complete_async_signal_slot(async_signal_slot* slot) const
{
    if(!slot) return;

    {
        std::lock_guard<std::mutex> lk{slot->mutex};
        // The completion signal is retired by ProcessDispatchCompletion. The slot object remains
        // as a tombstone until queue teardown so signal identity is never reused mid-run.
        slot->signal       = {};
        slot->handler_data = {};
    }

    slot->ready.store(false, std::memory_order_release);
    slot->in_use.store(false, std::memory_order_release);
    const auto calls =
        _queue_signal_trace.slot_complete_calls.fetch_add(1, std::memory_order_relaxed) + 1;
    notify_async_signal_slot_builder();
    maybe_emit_queue_signal_trace("slot-complete", calls);
}

void
Queue::create_signal(uint32_t attribute, hsa_signal_t* signal, bool direct_path) const
{
    const auto start_ns = common::timestamp_ns();
    hsa_status_t status = _ext_api.hsa_amd_signal_create_fn(1, 0, nullptr, attribute, signal);
    const auto elapsed_ns = static_cast<uint64_t>(common::timestamp_ns() - start_ns);
    const auto calls =
        _queue_signal_trace.create_signal_calls.fetch_add(1, std::memory_order_relaxed) + 1;
    _queue_signal_trace.create_signal_total_ns.fetch_add(elapsed_ns, std::memory_order_relaxed);
    update_atomic_max(_queue_signal_trace.create_signal_max_ns, elapsed_ns);
    if(direct_path)
        _queue_signal_trace.direct_create_signal_calls.fetch_add(1, std::memory_order_relaxed);
    maybe_emit_queue_signal_trace(direct_path ? "signal-create-direct" : "signal-create", calls);
    ROCP_FATAL_IF(status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK)
        << "Error: hsa_amd_signal_create failed with error code " << status
        << " :: " << hsa::get_hsa_status_string(status);
}

void
Queue::retire_signal(hsa_signal_t signal) const
{
    if(signal.handle == 0u) return;
    std::lock_guard<std::mutex> lk{_retired_signals_mutex};
    _retired_signals.emplace_back(signal);
    const auto calls =
        _queue_signal_trace.retired_signal_calls.fetch_add(1, std::memory_order_relaxed) + 1;
    maybe_emit_queue_signal_trace("retire-signal", calls);
}

void
Queue::drain_retired_signals() const
{
    auto pending = std::vector<hsa_signal_t>{};
    {
        std::lock_guard<std::mutex> lk{_retired_signals_mutex};
        pending.swap(_retired_signals);
    }

    const auto drained =
        _queue_signal_trace.retired_signal_drained.fetch_add(pending.size(),
                                                             std::memory_order_relaxed) +
        pending.size();
    for(const auto& signal : pending)
    {
        _core_api.hsa_signal_destroy_fn(signal);
    }

    if(!pending.empty()) maybe_emit_queue_signal_trace("retired-drain", drained);
}

void
Queue::destroy_async_signal_slots() const
{
    auto pending = std::vector<std::unique_ptr<async_signal_slot>>{};
    {
        std::lock_guard<std::mutex> lk{_async_signal_slots_mutex};
        _ready_async_signal_slots.clear();
        pending.swap(_async_signal_slots);
    }

    for(auto& slot : pending)
    {
        if(!slot) continue;

        {
            std::lock_guard<std::mutex> lk{slot->mutex};
            slot->owner        = nullptr;
            slot->handler_data = {};
            slot->ready.store(false, std::memory_order_release);
        }
        slot->in_use.store(false, std::memory_order_release);

        if(slot->signal.handle != 0u)
        {
#if !defined(NDEBUG)
            CHECK_NOTNULL(hsa::get_queue_controller())->_debug_signals.wlock([&](auto& signals) {
                signals.erase(slot->signal.handle);
            });
#endif
            _core_api.hsa_signal_destroy_fn(slot->signal);
        }
    }
}

void
Queue::sync() const
{
    if(_active_kernels.handle == 0u) return;

    constexpr auto wait_timeout =
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::milliseconds{1})
            .count();
    auto spin_count = uint64_t{0};

    while(true)
    {
        const auto interceptors = active_interceptors();
        const auto handlers     = active_async_handlers();
        const auto kernels      = _core_api.hsa_signal_load_scacquire_fn(_active_kernels);

        if(interceptors == 0 && handlers == 0 && kernels == 0) break;

        ++spin_count;
        if((spin_count % 1000) == 0)
        {
            std::fprintf(stderr,
                         "[rocprofiler-sdk][queue-sync] queue=%llu spins=%llu interceptors=%lld "
                         "handlers=%lld kernels=%lld\n",
                         static_cast<unsigned long long>(get_id().handle),
                         static_cast<unsigned long long>(spin_count),
                         static_cast<long long>(interceptors),
                         static_cast<long long>(handlers),
                         static_cast<long long>(kernels));
            std::fflush(stderr);
        }

        if(kernels != 0)
        {
            _core_api.hsa_signal_wait_relaxed_fn(_active_kernels,
                                                 HSA_SIGNAL_CONDITION_EQ,
                                                 0,
                                                 wait_timeout,
                                                 HSA_WAIT_STATE_ACTIVE);
        }
        else if(handlers != 0)
        {
            std::this_thread::yield();
        }
        else
        {
            std::this_thread::yield();
        }
    }

    drain_async_signal_handler_data(*this);
    drain_retired_signals();
    emit_queue_signal_trace("sync");
}

void
Queue::register_callback(ClientID id, queue_cb_t enqueue_cb, completed_cb_t complete_cb)
{
    _callbacks.wlock([&](auto& map) {
        ROCP_FATAL_IF(rocprofiler::common::get_val(map, id)) << "ID already exists!";
        _notifiers++;
        map[id] = std::make_pair(enqueue_cb, complete_cb);
    });
}

void
Queue::remove_callback(ClientID id)
{
    _callbacks.wlock([&](auto& map) {
        if(map.erase(id) == 1) _notifiers--;
    });
}

queue_state
Queue::get_state() const
{
    return _state.load(std::memory_order_acquire);
}

void
Queue::set_state(queue_state state) const
{
    _state.store(state, std::memory_order_release);
}
}  // namespace hsa
}  // namespace rocprofiler
