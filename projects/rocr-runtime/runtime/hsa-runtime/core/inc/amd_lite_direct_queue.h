////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef HSA_RUNTIME_CORE_INC_AMD_LITE_DIRECT_QUEUE_H_
#define HSA_RUNTIME_CORE_INC_AMD_LITE_DIRECT_QUEUE_H_

#include <array>
#include <cstddef>
#include <cstdint>

#include "inc/hsa.h"

namespace rocr {
namespace AMD {
namespace lite {

constexpr uint32_t kMqdSize = 0x1000;
constexpr uint32_t kMqdDwordCount = kMqdSize / sizeof(uint32_t);
constexpr uint32_t kDirectComputeRingSize = 0x8000;  // 32KB=8192dw (was 0x1000=1024dw). Ring region is [0x2000,0x10000) so 0x8000 is the max power-of-2 that fits below EOP. The 1024-dw ring wrapped at the first boundary-crossing dispatch (966+131) and the MES-backed CP halted at the wrap; 8192dw avoids the wrap for torch dispatches. Proper ring-wrap-at-boundary handling is a follow-up.
constexpr uint32_t kDirectComputeEopSize = 0x1000;
constexpr uint32_t kDirectComputeDoorbellBase = 0x20;
constexpr uint32_t kDirectComputeDoorbellStride = 2;

struct DirectQueueMemory {
  uint64_t size = 0;
  uint64_t gpu_addr = 0;
  void* cpu = nullptr;
  uint64_t platform_handle = 0;
  uint64_t platform_bus_addr = 0;
  uint64_t platform_mmap_offset = 0;
};

struct DirectQueueLayout {
  uint64_t base_offset = 0;
  uint64_t base_gpu = 0;
  void* cpu_base = nullptr;
  uint64_t cpu_size = 0;
  uint64_t mqd_offset = 0;
  uint64_t ring_offset = 0;
  uint64_t eop_offset = 0;
  uint64_t rptr_offset = 0;
  uint64_t wptr_offset = 0;
  uint64_t mqd_gpu = 0;
  uint64_t ring_gpu = 0;
  uint64_t eop_gpu = 0;
  uint64_t rptr_gpu = 0;
  uint64_t wptr_gpu = 0;
};

using DirectQueueMqd = std::array<uint32_t, kMqdDwordCount>;

struct DirectQueueState {
  uint32_t queue_id = 0;
  uint32_t queue_index = 0;
  uint32_t doorbell_index = 0;
  uint32_t ring_size_bytes = 0;
  uint32_t ring_align_mask = 0;
  uint32_t ring_nop = 0;
  uint64_t ring_gpu = 0;
  uint64_t wptr = 0;
  bool mes_backed = false;
  uint64_t framebuffer_base = 0;
  DirectQueueLayout layout{};
  DirectQueueMemory memory;
  volatile uint32_t* ring_cpu = nullptr;
  volatile uint64_t* wptr_cpu = nullptr;
  volatile uint64_t* rptr_cpu = nullptr;
  volatile uint64_t* doorbell_cpu = nullptr;
};

struct DirectQueueOptions {
  bool force_reclaim = false;
  bool use_mes_queue = false;
  bool use_firmware_dequeue = true;
  // Doorbell-dead amdgpu_lite transport (Linux): poll the in-memory wptr and use
  // the routed MEC ring doorbell (0x6) instead of the unassigned 0x20 slot.
  // Default false so macOS/Windows direct queues are unchanged; set by
  // LinuxDirectQueueOptions.
  bool poll_wptr = false;
  bool mec_doorbell = false;
  bool skip_destroy = false;
  bool trace = false;
  bool trace_verbose = false;
  uint32_t dequeue_settle_us = 100000;
  uint32_t activate_sleep_us = 10000;
  const char* trace_prefix = "ROCR lite direct queue";
};

class DirectQueuePlatform {
 public:
  virtual ~DirectQueuePlatform() = default;

  virtual hsa_status_t EnsureDoorbellAperture() const = 0;
  virtual hsa_status_t ReadMmio32(uint32_t base, uint32_t reg,
                                  uint32_t* value) const = 0;
  virtual hsa_status_t WriteMmio32(uint32_t base, uint32_t reg,
                                   uint32_t value) const = 0;
  virtual hsa_status_t ZeroGpuMemory(uint64_t offset, uint64_t size) const = 0;
  virtual hsa_status_t WriteGpuMemory32(uint64_t offset, uint32_t value) const = 0;
  virtual hsa_status_t FlushHdp() const { return HSA_STATUS_SUCCESS; }
  virtual void* GpuMemoryCpuPointer(uint64_t offset) const = 0;
  virtual bool PreferAllocatedQueueMemory() const { return false; }
  virtual hsa_status_t AllocateQueueMemory(uint64_t size,
                                           DirectQueueMemory* memory) const {
    (void)size;
    (void)memory;
    return HSA_STATUS_ERROR;
  }
  virtual hsa_status_t FreeQueueMemory(DirectQueueMemory* memory) const {
    if (memory != nullptr) *memory = {};
    return HSA_STATUS_SUCCESS;
  }
  virtual volatile uint64_t* DoorbellCpuPointer(uint32_t doorbell_index) const = 0;
  virtual void SleepUs(uint32_t usec) const = 0;
};

uint32_t DirectQueuePipe(uint32_t queue_index);
uint32_t DirectQueueHqd(uint32_t queue_index);
uint32_t DirectQueueDoorbell(uint32_t queue_index, bool mec_doorbell);

DirectQueueLayout BuildDirectQueueLayout(uint64_t framebuffer_base,
                                         uint32_t queue_index);
DirectQueueMqd BuildPm4DirectQueueMqd(const DirectQueueLayout& layout,
                                      uint32_t doorbell_index);

hsa_status_t CreateDirectQueue(const DirectQueuePlatform& platform,
                               DirectQueueState* queue,
                               uint32_t queue_index,
                               uint64_t framebuffer_base,
                               const DirectQueueOptions& options);
hsa_status_t DestroyDirectQueue(const DirectQueuePlatform& platform,
                                DirectQueueState& queue,
                                const DirectQueueOptions& options);
hsa_status_t SubmitDirectQueue(const DirectQueuePlatform& platform,
                               DirectQueueState& queue,
                               const uint32_t* pm4,
                               size_t dword_count,
                               const DirectQueueOptions& options);
hsa_status_t ReadDirectQueueRptr(const DirectQueuePlatform& platform,
                                 const DirectQueueState& queue,
                                 uint32_t* rptr);
// Patch the MQD scratch persistent-state (compute_dispatch_scratch_base +
// tmpring) and re-activate the queue so the MEC/MES re-initializes FLAT_SCRATCH
// from per-queue state. scratch_base_256 = backing VA >> 8.
hsa_status_t SetDirectQueueScratch(const DirectQueuePlatform& platform,
                                   DirectQueueState& queue,
                                   uint64_t scratch_base_256,
                                   uint32_t tmpring_size,
                                   const DirectQueueOptions& options);

// Release the MES engine (clear reset/halt, set PIPE0/1_ACTIVE) when the
// firmware autoloaded the MES ucode but left the engine halted (macOS/Windows
// have no kernel-driver MES start). mes_entry is the ucode entry PC (the
// 64-bit ucode_start_addr from the uni_mes firmware header); the engine pipes
// are programmed with mes_entry >> 2. Returns HSA_STATUS_SUCCESS when both MES
// pipes latch ACTIVE. Must run before EnsureMesScheduler / the first
// use_mes_queue CreateDirectQueue. No-op-safe to skip on the direct path.
hsa_status_t StartMesEngine(const DirectQueuePlatform& platform,
                            uint64_t mes_entry,
                            const DirectQueueOptions& options);

}  // namespace lite
}  // namespace AMD
}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_INC_AMD_LITE_DIRECT_QUEUE_H_
