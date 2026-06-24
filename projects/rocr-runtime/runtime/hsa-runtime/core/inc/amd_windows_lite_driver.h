////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef HSA_RUNTIME_CORE_INC_AMD_WINDOWS_LITE_DRIVER_H_
#define HSA_RUNTIME_CORE_INC_AMD_WINDOWS_LITE_DRIVER_H_

#if !defined(_WIN32)
#error "amd_windows_lite_driver.h should only be used in the Windows build"
#endif

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/inc/amd_lite_direct_queue.h"
#include "core/inc/driver.h"
#include "core/inc/memory_region.h"

namespace rocr {
namespace core {
class Queue;
class Agent;
}  // namespace core

namespace AMD {

// Opaque holder for the wddm_lite WddmLite instance + cached IP-discovery and
// compute-bring-up state. Defined in the .cpp so this header stays free of
// wddm_lite.h (and thus <windows.h>). When wddm_lite_state_ is non-null the
// driver drives the GPU through the proven wddm_lite recipe (bootload + MEC
// enable + direct-queue transport) instead of the raw amdgpu_mcdm escapes.
struct WddmLiteState;

/// @brief ROCR driver backend for Windows userspace → the amdgpu_mcdm WDDM
/// kernel-mode driver (MCDM compute KMD for the gfx1201 / RDNA4 eGPU).
///
/// This is the Stage-1 scaffold, the third tri-OS lite:: backend alongside
/// LinuxAmdgpuLiteDriver (amdgpu_lite DRM) and MacOsDriver (DriverKit DEXT).
/// It fulfills the core::Driver pure-virtual interface so the rest of ROCR
/// links on Windows, and registers in the topology driver-discovery array
/// alongside KfdDriver. DiscoverDriver() currently reports "no device"
/// unconditionally — the real D3DKMT/escape plumbing lands in follow-up
/// commits.
///
/// Transport: every kernel interaction is a D3DKMTEscape() round-trip the
/// amdgpu_mcdm KMD services. The class mirrors MacOsDriver (one class, no
/// separate transport object), swapping the macgpu_* DEXT calls for WDDM
/// escapes:
///   - Open()        → D3DKMTOpenAdapterFromDeviceName + D3DKMTCreateDevice,
///                     then an AMDGPU_ESCAPE_GET_INFO to cache device_info_.
///   - ReadMmio32 /  → cached MMIO-BAR (BAR0) pointer at (base+reg)*4, with a
///     WriteMmio32     READ_REG32 / WRITE_REG32 escape fallback.
///   - AllocateVram  → MAP_VRAM window (bump-allocated); GPU addr =
///                     framebuffer_base_ + offset (no IOMMU dependency).
///   - DoorbellCpuPointer → MAP_BAR'd doorbell aperture.
///
/// Two verified KMD gaps gate later milestones (see the design notes):
///   1. The doorbell BAR index is not reported by GET_INFO and only BAR0 is
///      kernel-mapped — the mappable doorbell aperture must be confirmed
///      against a real GET_INFO BAR dump before the direct-HQD queue path
///      (do NOT hardcode the macOS index 2).
///   2. ALLOC_DMA returns a raw physical address with no IOMMU/DMA-adapter
///      mapping while GET_IOMMU_INFO claims remapping is active — so
///      PreferAllocatedQueueMemory() stays false (queue memory lives in the
///      VRAM-BAR window) until the KMD exposes a GPU-reachable DMA/GPUVM
///      address. This is the Windows analog of the macOS coherent-data work.
class WindowsLiteDriver final : public core::Driver,
                                private lite::DirectQueuePlatform {
 public:
  using DirectComputeQueue = lite::DirectQueueState;

  explicit WindowsLiteDriver(std::string devnode_name);

  /// @brief Probe for a usable amdgpu_mcdm adapter. Returns
  /// HSA_STATUS_SUCCESS with a live driver on success. Returns
  /// HSA_STATUS_ERROR (driver left nullptr) when no compatible WDDM adapter
  /// is present.
  static hsa_status_t DiscoverDriver(std::unique_ptr<core::Driver>& driver);

  // core::Driver overrides.

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
                              void** mem, size_t size, uint32_t node_id) override;
  hsa_status_t FreeMemory(void* mem, size_t size) override;

  hsa_status_t CreateQueue(uint32_t node_id, HSA_QUEUE_TYPE type, uint32_t queue_pct,
                           HSA::hsa_amd_queue_priority_internal_t priority,
                           uint32_t sdma_engine_id, void* queue_addr,
                           uint64_t queue_size_bytes,
                           uint64_t queue_metadata_size_bytes, HsaEvent* event,
                           HsaQueueResource& queue_resource) const override;
  hsa_status_t DestroyQueue(HSA_QUEUEID queue_id) const override;
  hsa_status_t UpdateQueue(HSA_QUEUEID queue_id, uint32_t queue_pct,
                           HSA::hsa_amd_queue_priority_internal_t priority, void* queue_addr,
                           uint64_t queue_size_bytes, HsaEvent* event) const override;
  hsa_status_t SetQueueCUMask(HSA_QUEUEID queue_id, uint32_t cu_mask_count,
                              uint32_t* queue_cu_mask) const override;
  hsa_status_t AllocQueueGWS(HSA_QUEUEID queue_id, uint32_t num_gws,
                             uint32_t* first_gws) const override;

  hsa_status_t ExportDMABuf(void* mem, size_t size, int* dmabuf_fd,
                            size_t* offset) override;
  hsa_status_t ImportDMABuf(int dmabuf_fd, const core::Agent& agent,
                            core::ShareableHandle* handle, void* mem) override;
  hsa_status_t DestroyImportedShareableHandle(core::ShareableHandle* handle) override;
  hsa_status_t Map(core::ShareableHandle handle, void* mem, size_t offset,
                   size_t size, hsa_access_permission_t perms) override;
  hsa_status_t Unmap(core::ShareableHandle handle, void* mem, size_t offset,
                     size_t size) override;
  hsa_status_t CreateShareableHandle(void* va, void* mem, size_t size,
                                     const core::Agent& agent,
                                     core::ShareableHandle* handle, uint64_t* offset,
                                     int* drm_fd, uint64_t* drm_fd_offset) override;
  hsa_status_t DestroyShareableHandle(core::ShareableHandle* handle) override;

  hsa_status_t SPMAcquire(uint32_t preferred_node_id) const override;
  hsa_status_t SPMRelease(uint32_t preferred_node_id) const override;
  hsa_status_t SPMSetDestBuffer(uint32_t preferred_node_id, uint32_t size_bytes,
                                uint32_t* timeout, uint32_t* size_copied,
                                void* dest_mem_addr, bool* is_spm_data_loss) const override;

  hsa_status_t SetTrapHandler(uint32_t node_id, const void* base, uint64_t base_size,
                              const void* buffer_base, uint64_t buffer_base_size) const override;
  hsa_status_t GetDeviceHandle(uint32_t node_id, void** device_handle) const override;
  hsa_status_t GetClockCounters(uint32_t node_id, HsaClockCounters* clock_counter) const override;
  hsa_status_t GetTileConfig(uint32_t node_id, HsaGpuTileConfig* config) const override;
  hsa_status_t IsModelEnabled(bool* enable) const override;
  hsa_status_t GetWallclockFrequency(uint32_t node_id, uint64_t* frequency) const override;
  hsa_status_t AllocateScratchMemory(uint32_t node_id, uint64_t size, void** mem) const override;
  hsa_status_t AvailableMemory(uint32_t node_id, uint64_t* available_size) const override;
  hsa_status_t RegisterMemory(void* ptr, uint64_t size, HsaMemFlags mem_flags) const override;
  hsa_status_t DeregisterMemory(void* ptr) const override;
  hsa_status_t MakeMemoryResident(const void* mem, size_t size, uint64_t* alternate_va,
                                  const HsaMemMapFlags* mem_flags,
                                  uint32_t num_nodes, const uint32_t* nodes) const override;
  hsa_status_t MakeMemoryUnresident(const void* mem) const override;

  hsa_status_t GetQueueSaveAreaInfo(HSA_QUEUEID queue_id, void** address,
                                    size_t* size) const override;

  // VRAM-BAR-window allocation (the first-pass, IOMMU-free queue/data path).
  hsa_status_t AllocateVram(size_t size, size_t align, void** cpu_addr,
                            uint64_t* gpu_addr);
  // Translate a CPU pointer inside a registered allocation to the GPU address
  // the CP fetches from. DMA (IOVA) tier first, then the VRAM-BAR window.
  hsa_status_t HostToGpuAddress(const void* ptr, uint64_t* gpu_addr) const;

  hsa_status_t CreateDirectComputeQueue(DirectComputeQueue* queue);
  hsa_status_t DestroyDirectComputeQueue(DirectComputeQueue& queue);
  hsa_status_t SubmitDirectCompute(DirectComputeQueue& queue,
                                   const uint32_t* pm4, size_t dword_count) const;
  hsa_status_t ReadDirectComputeRptr(const DirectComputeQueue& queue,
                                     uint32_t* rptr) const;
  hsa_status_t SetQueueScratch(DirectComputeQueue& queue,
                               uint64_t scratch_base_256,
                               uint32_t tmpring_size) const;

  /// @brief Bring-up validation (Windows only; NOT part of the HSA API).
  /// Stages fill_kernel_raw.co from the firmware dir, dispatches it through a
  /// direct compute queue created by THIS driver (CreateDirectComputeQueue ->
  /// SubmitDirectCompute -> the shared lite:: path over wddm_lite), and verifies
  /// out[0..N] == 0xDEADBEEF. Logs progress to stdout. Returns
  /// HSA_STATUS_SUCCESS iff the RELEASE_MEM fence signaled, the GPUVM fault
  /// status is 0, and every output dword verified.
  hsa_status_t DispatchKernelSelfTest();

 private:
  // Cached AMDGPU_ESCAPE_GET_INFO results, populated on Open().
  struct DeviceInfo {
    uint32_t vendor_id = 0;
    uint32_t device_id = 0;
    uint32_t revision_id = 0;
    uint64_t vram_size = 0;
    uint64_t visible_vram_size = 0;
  };

  struct VramAllocation {
    uint64_t offset = 0;
    uint64_t size = 0;
    uint64_t gpu_addr = 0;
  };

  // A KMD ALLOC_DMA-backed allocation. Wired up at the coherent-queue
  // milestone, once the KMD returns a GPU-reachable bus address.
  struct DmaAllocation {
    uint64_t alloc_handle = 0;  // FREE_DMA handle
    uint64_t bus_addr = 0;      // GPU-addressable DMA address
    uint64_t size = 0;
  };

  hsa_status_t EnsureBarMappingsLocked();

  // lite::DirectQueuePlatform overrides.
  hsa_status_t EnsureDoorbellAperture() const override;
  hsa_status_t ReadMmio32(uint32_t base, uint32_t reg,
                          uint32_t* value) const override;
  hsa_status_t WriteMmio32(uint32_t base, uint32_t reg,
                           uint32_t value) const override;
  hsa_status_t ZeroGpuMemory(uint64_t offset, uint64_t size) const override;
  hsa_status_t WriteGpuMemory32(uint64_t offset, uint32_t value) const override;
  hsa_status_t FlushHdp() const override;
  void* GpuMemoryCpuPointer(uint64_t offset) const override;
  bool PreferAllocatedQueueMemory() const override;
  hsa_status_t AllocateQueueMemory(uint64_t size,
                                   lite::DirectQueueMemory* memory) const override;
  hsa_status_t FreeQueueMemory(lite::DirectQueueMemory* memory) const override;
  volatile uint64_t* DoorbellCpuPointer(uint32_t doorbell_index) const override;
  void SleepUs(uint32_t usec) const override;

  // D3DKMT adapter/device handles (D3DKMT_HANDLE is a UINT32). 0 == not open.
  // Stored as uint32_t so this header stays free of <windows.h> / <d3dkmthk.h>.
  uint32_t adapter_ = 0;
  uint32_t device_ = 0;
  DeviceInfo info_{};
  mutable std::mutex gpu_lock_;
  // MAP_BAR'd apertures (cached once in EnsureBarMappingsLocked).
  void* mmio_bar_cpu_ = nullptr;
  uint64_t mmio_bar_size_ = 0;
  void* vram_bar_cpu_ = nullptr;
  uint64_t vram_bar_size_ = 0;
  // Opaque MappingHandle from the MAP_VRAM escape, replayed on UNMAP at Close().
  void* vram_map_handle_ = nullptr;
  void* doorbell_bar_cpu_ = nullptr;
  uint64_t doorbell_bar_size_ = 0;
  // Opaque MappingHandle from the doorbell MAP_BAR escape (for UNMAP at Close()).
  void* doorbell_map_handle_ = nullptr;
  uint64_t framebuffer_base_ = 0;
  uint64_t next_vram_offset_ = 0;
  uint32_t next_direct_queue_index_ = 0;
  std::unordered_map<void*, VramAllocation> vram_allocations_;
  std::unordered_map<void*, DmaAllocation> dma_allocations_;

  // wddm_lite-backed bring-up + transport. Allocated in Init() (which runs the
  // proven recipeBootload + MEC enable + doorbell aperture) and used to back
  // the DirectQueuePlatform overrides in allocated-queue-memory mode. Null
  // until Init() succeeds; the overrides fall back to the amdgpu_mcdm escape
  // path (above) when null so the scaffold still links/runs without bring-up.
  std::unique_ptr<WddmLiteState> wddm_lite_state_;
  // Brings the GPU to BOOTLOAD_COMPLETE + MEC enabled via wddm_lite. Idempotent.
  hsa_status_t EnsureGpuBringUpLocked();
};

}  // namespace AMD
}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_INC_AMD_WINDOWS_LITE_DRIVER_H_
