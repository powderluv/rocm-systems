////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef HSA_RUNTIME_CORE_INC_AMD_LITE_LINUX_DRIVER_H_
#define HSA_RUNTIME_CORE_INC_AMD_LITE_LINUX_DRIVER_H_

#if !defined(__linux__)
#error "amd_lite_linux_driver.h is Linux-only"
#endif

#include <cstddef>
#include <cstdint>
#include <array>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/inc/amd_lite_direct_queue.h"
#include "core/inc/amd_lite_linux_transport.h"
#include "core/inc/driver.h"
#include "core/inc/memory_region.h"

namespace rocr {
namespace core {
class Agent;
class Queue;
}  // namespace core

namespace AMD {

class LinuxAmdgpuLiteDriver final : public core::Driver {
 public:
  using DirectComputeQueue = lite::DirectQueueState;

  explicit LinuxAmdgpuLiteDriver(std::string devnode_name);

  static hsa_status_t DiscoverDriver(std::unique_ptr<core::Driver>& driver);

  hsa_status_t Init() override;
  hsa_status_t ShutDown() override;
  hsa_status_t QueryKernelModeDriver(core::DriverQuery query) override;
  hsa_status_t Open() override;
  hsa_status_t Close() override;

  hsa_status_t GetSystemProperties(HsaSystemProperties& sys_props) const override;
  hsa_status_t GetNodeProperties(HsaNodeProperties& node_props,
                                 uint32_t node_id) const override;
  hsa_status_t GetEdgeProperties(std::vector<HsaIoLinkProperties>& io_link_props,
                                 uint32_t node_id) const override;
  hsa_status_t GetMemoryProperties(uint32_t node_id,
                                   std::vector<HsaMemoryProperties>& mem_props) const override;
  hsa_status_t GetCacheProperties(uint32_t node_id, uint32_t processor_id,
                                  std::vector<HsaCacheProperties>& cache_props) const override;

  hsa_status_t AllocateMemory(const core::MemoryRegion& mem_region,
                              core::MemoryRegion::AllocateFlags alloc_flags,
                              size_t size, uint32_t node_id,
                              core::DriverMemoryHandle* handle) override;
  hsa_status_t FreeMemory(const core::DriverMemoryHandle& handle) override;
  // lite-internal pointer-based free for queue-owned VRAM (marker/scratch);
  // the ABI FreeMemory recovers the pointer from handle.handle and delegates here.
  hsa_status_t FreeMemoryPtr(void* mem);

  // Flush the HDP so CPU writes to VRAM via the BAR reach DRAM and
  // become visible to the GPU MC (H2D coherency; macOS #27 analog).
  hsa_status_t FlushHdp() const { return transport_.FlushHdp(); }

  hsa_status_t CreateQueue(uint32_t node_id, HSA_QUEUE_TYPE type, uint32_t queue_pct,
                           HSA::hsa_amd_queue_priority_internal_t priority,
                           uint32_t sdma_engine_id, void* queue_addr,
                           uint64_t queue_size_bytes,
                           uint64_t queue_metadata_size_bytes, HsaEvent* event,
                           HsaQueueResource& queue_resource) const override;
  hsa_status_t DestroyQueue(HSA_QUEUEID queue_id) const override;
  hsa_status_t UpdateQueue(HSA_QUEUEID queue_id, uint32_t queue_pct,
                           HSA::hsa_amd_queue_priority_internal_t priority,
                           void* queue_addr, uint64_t queue_size_bytes,
                           HsaEvent* event) const override;
  hsa_status_t SetQueueCUMask(HSA_QUEUEID queue_id, uint32_t cu_mask_count,
                              uint32_t* queue_cu_mask) const override;
  hsa_status_t AllocQueueGWS(HSA_QUEUEID queue_id, uint32_t num_gws,
                             uint32_t* first_gws) const override;

  hsa_status_t ExportMemoryHandle(const core::Agent& agent,
                                  const core::DriverMemoryHandle& handle,
                                  core::ShareType type,
                                  void* export_handle) override;
  hsa_status_t ImportMemoryHandle(const core::Agent& agent,
                                  core::DriverMemoryHandle* handle,
                                  core::ShareType type, void* import_handle,
                                  void* mem) override;
  hsa_status_t Map(const core::DriverMemoryHandle& handle, void* mem,
                   size_t offset, size_t size, hsa_access_permission_t perms,
                   uint32_t node_id) override;
  hsa_status_t Unmap(const core::DriverMemoryHandle& handle, void* mem,
                     size_t offset, size_t size, uint32_t node_id) override;
  hsa_status_t CreateShareableHandle(core::DriverMemoryHandle* handle,
                                     const core::Agent& agent,
                                     uint64_t* offset) override;
  hsa_status_t DestroyMemoryHandle(core::DriverMemoryHandle* handle) override;

  hsa_status_t SPMAcquire(uint32_t preferred_node_id) const override;
  hsa_status_t SPMRelease(uint32_t preferred_node_id) const override;
  hsa_status_t SPMSetDestBuffer(uint32_t preferred_node_id, uint32_t size_bytes,
                                uint32_t* timeout, uint32_t* size_copied,
                                void* dest_mem_addr,
                                bool* is_spm_data_loss) const override;
  hsa_status_t SetTrapHandler(uint32_t node_id, const void* base,
                              uint64_t base_size, const void* buffer_base,
                              uint64_t buffer_base_size) const override;
  hsa_status_t GetDeviceHandle(uint32_t node_id, void** device_handle) const override;
  hsa_status_t GetClockCounters(uint32_t node_id,
                                HsaClockCounters* clock_counter) const override;
  hsa_status_t GetTileConfig(uint32_t node_id, HsaGpuTileConfig* config) const override;
  hsa_status_t IsModelEnabled(bool* enable) const override;
  hsa_status_t GetWallclockFrequency(uint32_t node_id,
                                     uint64_t* frequency) const override;
  hsa_status_t AllocateScratchMemory(uint32_t node_id, uint64_t size,
                                     void** mem) const override;
  hsa_status_t AvailableMemory(uint32_t node_id,
                               uint64_t* available_size) const override;
  hsa_status_t RegisterMemory(void* ptr, uint64_t size,
                              HsaMemFlags mem_flags) const override;
  hsa_status_t DeregisterMemory(void* ptr) const override;
  hsa_status_t GetDeviceFd(uint32_t node_id, int* fd) const override;
  hsa_status_t CheckAcceleratorReadiness(core::Agent& agent,
                                         bool* ready) const override;
  hsa_status_t MakeMemoryResident(const void* mem, size_t size,
                                  uint64_t* alternate_va,
                                  const HsaMemFlags* mem_flags,
                                  uint32_t num_nodes,
                                  const uint32_t* nodes) const override;
  hsa_status_t MakeMemoryUnresident(const void* mem) const override;
  hsa_status_t GetQueueSaveAreaInfo(HSA_QUEUEID queue_id, void** address,
                                    size_t* size) const override;

  hsa_status_t AllocateVram(size_t size, size_t align, void** cpu_addr,
                            uint64_t* gpu_addr);
  hsa_status_t HostToGpuAddress(const void* ptr, uint64_t* gpu_addr) const;
  void RegisterVramShadow(const void* cpu_addr, size_t size, const void* src);
  hsa_status_t VramShadowAddress(const void* cpu_addr, size_t size,
                                 const void** shadow_addr) const;
  hsa_status_t CreateDirectComputeQueue(DirectComputeQueue* queue);
  hsa_status_t DestroyDirectComputeQueue(DirectComputeQueue& queue);
  hsa_status_t SubmitDirectCompute(DirectComputeQueue& queue,
                                   const uint32_t* pm4,
                                   size_t dword_count) const;
  hsa_status_t ReadDirectComputeRptr(const DirectComputeQueue& queue,
                                     uint32_t* rptr) const;
  // Program the MQD scratch persistent-state + per-VMID SH_MEM aperture so a
  // spilling kernel's FLAT_SCRATCH is initialized. scratch_base_256 = VA >> 8.
  hsa_status_t SetDirectComputeScratch(DirectComputeQueue& queue,
                                       uint64_t scratch_base_256,
                                       uint32_t tmpring_size) const;

 private:
  struct VramAllocation {
    lite::LinuxLiteBuffer buffer;
    std::vector<uint8_t> shadow;
  };

  lite::LinuxAmdgpuLiteTransport transport_;
  mutable std::mutex gpu_lock_;
  std::array<bool, 8> direct_queue_in_use_ = {};
  std::unordered_map<void*, VramAllocation> vram_allocations_;
};

}  // namespace AMD
}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_INC_AMD_LITE_LINUX_DRIVER_H_
