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

#ifdef _WIN32

#include "core/inc/amd_windows_lite_driver.h"

// Minimize windows.h's macro surface (drops winsock/GDI/etc. sub-headers) and
// keep min/max as std:: — defensively defined here even though the build sets
// them target-wide, so a standalone compile of this TU is clean too.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
// clang-format off
#include <windows.h>
// d3dkmthk.h must follow windows.h: it supplies D3DKMT_HANDLE / the D3DKMT_*
// argument structs and the D3DKMTEnumAdapters3 / OpenAdapterFromLuid /
// CreateDevice / DestroyDevice / CloseAdapter / Escape entry points. gdi32 is
// already on the link line (the topology layer uses it); if the linker reports
// the D3DKMT* symbols unresolved, fall back to the GetProcAddress path below.
#include <d3dkmthk.h>
// clang-format on

#include <atomic>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <utility>

#include "core/inc/amd_lite_direct_queue.h"
#include "core/inc/amd_memory_region.h"
#include "core/inc/memory_region.h"

// wddm_lite: the proven gfx1201 userspace driver (recipe bootload + MEC enable
// + bump allocator + escape transport). Backs the GPU bring-up and the
// DirectQueuePlatform overrides. wddm_lite.h pulls in <windows.h>, which is
// already included above; it is intentionally NOT in the public header.
#include "wddm_lite.h"

namespace rocr {
namespace AMD {

// Opaque state declared in the header: the WddmLite instance + cached
// IP-discovery + bring-up context. Populated by EnsureGpuBringUpLocked().
struct WddmLiteState {
  WddmLite gpu;
  IpDiscoveryResult ipd{};
  WddmComputeContext ctx{};
  bool brought_up = false;
  // Cache of per-index doorbell BAR2 mappings (mapBar(2, idx*4, 8)).
  struct DoorbellMap {
    uint32_t index;
    volatile uint64_t* cpu;
  };
  std::vector<DoorbellMap> doorbells;
};

namespace {

// Firmware directory the wddm_lite recipe reads gc/psp/smu .bin from, as seen
// by the guest. Default mirrors wddm_lite_test (Z:\winfw); override with the
// ROCR_WINDOWS_FW_DIR env var.
const char* WindowsFirmwareDir() {
  const char* dir = std::getenv("ROCR_WINDOWS_FW_DIR");
  return (dir != nullptr && dir[0] != '\0') ? dir : "Z:\\winfw";
}

// gfx1201 register namespace bases (dword index; the (base+reg)*4 byte-offset
// convention matches the Linux transport and the macOS DEXT backend).
constexpr uint32_t GC_B0 = 0x1260;
constexpr uint32_t NBIO_B2 = 0xD20;

constexpr uint32_t regRCC_DEV0_EPF0_RCC_DOORBELL_APER_EN = 0x00c0;
constexpr uint32_t regGDC_S2A0_S2A_DOORBELL_ENTRY_0_CTRL = 0x01cb;
constexpr uint32_t regGDC_S2A0_S2A_DOORBELL_ENTRY_3_CTRL = 0x01ce;
constexpr uint32_t regCP_MEC_DOORBELL_RANGE_LOWER = 0x1dfc;
constexpr uint32_t regCP_MEC_DOORBELL_RANGE_UPPER = 0x1dfd;

// Keep the general-purpose VRAM bump allocator away from the low queue scratch
// window used by the direct-compute bring-up path.
constexpr uint64_t kVramAllocBaseOffset = 64ull * 1024 * 1024;

uint64_t AlignUpU64(uint64_t value, uint64_t align) {
  return (value + align - 1) & ~(align - 1);
}

bool EnvEnabled(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

bool TraceDirectQueue() { return std::getenv("ROCR_WINDOWS_TRACE_DIRECT_QUEUE") != nullptr; }
bool TraceDirectQueueVerbose() {
  return EnvEnabled("ROCR_WINDOWS_TRACE_DIRECT_QUEUE_VERBOSE");
}

lite::DirectQueueOptions WindowsDirectQueueOptions() {
  lite::DirectQueueOptions options;
  options.use_firmware_dequeue = true;
  options.trace = TraceDirectQueue();
  options.trace_verbose = TraceDirectQueueVerbose();
  // First pass: direct HQD, no MES. Matches the pre-MES macOS/Linux bring-up.
  options.use_mes_queue = std::getenv("ROCR_WINDOWS_USE_MES_QUEUE") != nullptr;
  // Without a KMD to reset queues on process exit, an HQD activated by a
  // prior process stays active across runs (and the bring-up deliberately
  // skips re-bootload for repeatability). Reclaim a stale active HQD on a
  // fresh queue create, mirroring macOS (AMD_GPU_MACOS_FORCE_DIRECT_COMPUTE)
  // and Linux (ROCR_AMDGPU_LITE_FORCE_DIRECT_COMPUTE).
  options.force_reclaim =
      std::getenv("ROCR_WINDOWS_FORCE_DIRECT_COMPUTE") != nullptr;
  options.trace_prefix = "ROCR windows direct queue";
  return options;
}

// Tracks VirtualAlloc'd host memory so FreeMemory can route an untyped void*
// back to VirtualFree.
struct HostAllocRegistry {
  std::mutex m;
  std::unordered_map<void*, size_t> allocations;
};
HostAllocRegistry& GetHostAllocRegistry() {
  // Keep alive until process exit: HIP/ROCclr can free from a foreign module's
  // static destructor, which would otherwise race this registry's teardown.
  static HostAllocRegistry* reg = new HostAllocRegistry();
  return *reg;
}

// ---- amdgpu_mcdm escape ABI -------------------------------------------------
//
// The userspace copy of kernel_driver/amdgpu_mcdm.h. Every field's order and
// size must byte-match the KMD; the layout mirrors the Python ctypes structs in
// userspace_driver/.../windows/driver_interface.py. Windows types are used
// throughout; the NTSTATUS Status field is a LONG (32-bit signed) — the KMD
// fills it on return (0 == success, <0 == driver error).
enum AmdgpuEscapeCode : ULONG {
  AMDGPU_ESCAPE_GET_INFO = 0x0001,
  AMDGPU_ESCAPE_READ_REG32 = 0x0010,
  AMDGPU_ESCAPE_WRITE_REG32 = 0x0011,
  AMDGPU_ESCAPE_MAP_BAR = 0x0020,
  AMDGPU_ESCAPE_UNMAP_BAR = 0x0021,
  AMDGPU_ESCAPE_ALLOC_DMA = 0x0030,
  AMDGPU_ESCAPE_FREE_DMA = 0x0031,
  AMDGPU_ESCAPE_MAP_VRAM = 0x0040,
  AMDGPU_ESCAPE_GET_IOMMU_INFO = 0x0060,
};

// Natural alignment (NO #pragma pack): the KMD's amdgpu_mcdm.h structs and the
// Python ctypes structs both use the compiler's default alignment, so these
// must match it. LARGE_INTEGER forces 8-byte alignment of AmdgpuBarInfo and the
// VRAM Offset — packing to 1 would place Bars[]/Offset at the wrong byte offsets
// and send garbage to the KMD.
struct AmdgpuEscapeHeader {
  ULONG Command;  // AmdgpuEscapeCode value (4 bytes)
  LONG Status;    // NTSTATUS filled by driver (0 == success)
  ULONG Size;     // Total size including header
};

struct AmdgpuBarInfo {
  LARGE_INTEGER PhysicalAddress;
  ULONGLONG Length;
  BOOLEAN IsMemory;
  BOOLEAN Is64Bit;
  BOOLEAN IsPrefetchable;
  UCHAR Reserved;
};

struct AmdgpuGetInfoData {
  AmdgpuEscapeHeader Header;
  USHORT VendorId;
  USHORT DeviceId;
  USHORT SubsystemVendorId;
  USHORT SubsystemId;
  UCHAR RevisionId;
  UCHAR Reserved[3];
  ULONG NumBars;
  AmdgpuBarInfo Bars[6];
  ULONGLONG VramSizeBytes;
  ULONGLONG VisibleVramSizeBytes;
};

struct AmdgpuReg32Data {
  AmdgpuEscapeHeader Header;
  ULONG BarIndex;  // which BAR (0 for MMIO)
  ULONG Offset;    // byte offset within BAR
  ULONG Value;     // value read/written
};

struct AmdgpuMapBarData {
  AmdgpuEscapeHeader Header;
  ULONG BarIndex;       // input: which BAR to map
  ULONGLONG Offset;     // input: offset within BAR
  ULONGLONG Length;     // input: length (0 == entire BAR)
  PVOID MappedAddress;  // output: usermode VA
  PVOID MappingHandle;  // output: opaque handle for UNMAP
};

struct AmdgpuMapVramData {
  AmdgpuEscapeHeader Header;
  ULONGLONG Offset;     // input: offset within VRAM (via BAR2)
  ULONGLONG Length;     // input: length to map
  PVOID MappedAddress;  // output: usermode VA
  PVOID MappingHandle;  // output: opaque handle for unmap
};

// D3DKMT entry points. Resolved at first use: prefer the import-library symbols
// (gdi32 is already linked); the GetProcAddress fallback covers a toolchain
// whose gdi32 import lib omits the D3DKMT* thunks.
typedef NTSTATUS(APIENTRY* PfnD3DKMTEnumAdapters3)(D3DKMT_ENUMADAPTERS3*);
typedef NTSTATUS(APIENTRY* PfnD3DKMTOpenAdapterFromLuid)(D3DKMT_OPENADAPTERFROMLUID*);
typedef NTSTATUS(APIENTRY* PfnD3DKMTCreateDevice)(D3DKMT_CREATEDEVICE*);
typedef NTSTATUS(APIENTRY* PfnD3DKMTDestroyDevice)(const D3DKMT_DESTROYDEVICE*);
typedef NTSTATUS(APIENTRY* PfnD3DKMTCloseAdapter)(const D3DKMT_CLOSEADAPTER*);
typedef NTSTATUS(APIENTRY* PfnD3DKMTEscape)(const D3DKMT_ESCAPE*);

struct D3dkmtApi {
  PfnD3DKMTEnumAdapters3 EnumAdapters3 = nullptr;
  PfnD3DKMTOpenAdapterFromLuid OpenAdapterFromLuid = nullptr;
  PfnD3DKMTCreateDevice CreateDevice = nullptr;
  PfnD3DKMTDestroyDevice DestroyDevice = nullptr;
  PfnD3DKMTCloseAdapter CloseAdapter = nullptr;
  PfnD3DKMTEscape Escape = nullptr;
  bool valid = false;
};

const D3dkmtApi& GetD3dkmtApi() {
  static D3dkmtApi api = [] {
    D3dkmtApi a;
    HMODULE gdi = ::GetModuleHandleW(L"gdi32.dll");
    if (gdi == nullptr) gdi = ::LoadLibraryW(L"gdi32.dll");
    if (gdi == nullptr) return a;
    a.EnumAdapters3 =
        reinterpret_cast<PfnD3DKMTEnumAdapters3>(::GetProcAddress(gdi, "D3DKMTEnumAdapters3"));
    a.OpenAdapterFromLuid = reinterpret_cast<PfnD3DKMTOpenAdapterFromLuid>(
        ::GetProcAddress(gdi, "D3DKMTOpenAdapterFromLuid"));
    a.CreateDevice =
        reinterpret_cast<PfnD3DKMTCreateDevice>(::GetProcAddress(gdi, "D3DKMTCreateDevice"));
    a.DestroyDevice =
        reinterpret_cast<PfnD3DKMTDestroyDevice>(::GetProcAddress(gdi, "D3DKMTDestroyDevice"));
    a.CloseAdapter =
        reinterpret_cast<PfnD3DKMTCloseAdapter>(::GetProcAddress(gdi, "D3DKMTCloseAdapter"));
    a.Escape = reinterpret_cast<PfnD3DKMTEscape>(::GetProcAddress(gdi, "D3DKMTEscape"));
    // EnumAdapters3 is the only Win10-1903+ entry; the rest predate it. All
    // six must resolve for the backend to function.
    a.valid = a.EnumAdapters3 != nullptr && a.OpenAdapterFromLuid != nullptr &&
              a.CreateDevice != nullptr && a.DestroyDevice != nullptr &&
              a.CloseAdapter != nullptr && a.Escape != nullptr;
    return a;
  }();
  return api;
}

// Issue a driver-private escape on (adapter, device). Returns false on a
// transport failure (D3DKMTEscape NTSTATUS < 0); the caller separately checks
// the per-command Header.Status for driver-level errors. The escape buffer must
// begin with an AmdgpuEscapeHeader.
bool IssueEscape(D3DKMT_HANDLE adapter, D3DKMT_HANDLE device, void* buffer, size_t size) {
  const D3dkmtApi& api = GetD3dkmtApi();
  if (!api.valid || adapter == 0) return false;
  D3DKMT_ESCAPE esc{};
  esc.hAdapter = adapter;
  esc.hDevice = device;
  esc.Type = D3DKMT_ESCAPE_DRIVERPRIVATE;
  esc.Flags.Value = 0;
  esc.pPrivateDriverData = buffer;
  esc.PrivateDriverDataSize = static_cast<UINT>(size);
  esc.hContext = 0;
  const NTSTATUS st = api.Escape(&esc);
  return st >= 0;
}

}  // namespace

WindowsLiteDriver::WindowsLiteDriver(std::string devnode_name)
    : core::Driver(core::DriverType::WINDOWS_WDDM_LITE, std::move(devnode_name)) {}

hsa_status_t WindowsLiteDriver::DiscoverDriver(std::unique_ptr<core::Driver>& driver) {
  // Construct a tentative instance, attempt to open the amdgpu_mcdm adapter,
  // and keep it if the handshake succeeds. devnode_name_ is informational on
  // Windows (D3DKMT matches the adapter by device name / vendor id).
  auto tmp = std::make_unique<WindowsLiteDriver>(std::string("amdgpu_mcdm"));
  hsa_status_t s = tmp->Open();
  if (s != HSA_STATUS_SUCCESS) {
    // HSA_STATUS_ERROR for "no compatible WDDM adapter" is the normal path at
    // discovery time — the topology layer treats it as "no device."
    return s;
  }
  s = tmp->QueryKernelModeDriver(core::DriverQuery::GET_DRIVER_VERSION);
  if (s != HSA_STATUS_SUCCESS) {
    tmp->Close();
    return s;
  }
  driver = std::move(tmp);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t WindowsLiteDriver::Init() {
  // Bring the GPU to BOOTLOAD_COMPLETE + MEC enabled via the proven wddm_lite
  // recipe so the lite:: direct-queue path can dispatch. Best-effort: if the
  // firmware directory is absent (e.g. discovery-only / no-HW CI), bring-up is
  // skipped and the driver still links/serves topology queries. A real
  // CreateDirectComputeQueue then re-attempts bring-up under the lock.
  std::lock_guard<std::mutex> g(gpu_lock_);
  hsa_status_t status = EnsureGpuBringUpLocked();
  if (status != HSA_STATUS_SUCCESS && TraceDirectQueue()) {
    std::fprintf(stderr,
                 "WindowsLiteDriver::Init: GPU bring-up deferred (status=%u); "
                 "queue creation will retry\n",
                 status);
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t WindowsLiteDriver::ShutDown() { return Close(); }

hsa_status_t WindowsLiteDriver::EnsureGpuBringUpLocked() {
  if (wddm_lite_state_ != nullptr && wddm_lite_state_->brought_up) {
    return HSA_STATUS_SUCCESS;
  }
  if (wddm_lite_state_ == nullptr) {
    wddm_lite_state_ = std::make_unique<WddmLiteState>();
  }
  WddmLiteState& s = *wddm_lite_state_;

  // 1. Open the AMD adapter through wddm_lite (its own D3DKMT device, so the
  //    escape protocol matches the recipe exactly).
  if (!s.gpu.isOpen() && !s.gpu.open()) {
    return HSA_STATUS_ERROR;
  }

  // 2. GET_INFO + IP discovery + GMC (needed for vram_mc_base / framebuffer).
  AMDGPU_ESCAPE_GET_INFO_DATA info{};
  if (!s.gpu.getInfo(&info) || info.VendorId != 0x1002) {
    return HSA_STATUS_ERROR;
  }
  if (!ipDiscovery(s.gpu, info, s.ipd) || !s.ipd.valid) {
    return HSA_STATUS_ERROR;
  }
  GmcState gmc{};
  if (!gmcInit(s.gpu, s.ipd, gmc) || gmc.vramSize == 0) {
    return HSA_STATUS_ERROR;
  }

  // 3. recipeBootload (-> BOOTLOAD_COMPLETE) + NBIO doorbell aperture +
  //    cqInitGfxForCompute (MEC enable). Seeds the VRAM bump allocator.
  if (!wddmGfxBringUp(s.gpu, s.ipd, WindowsFirmwareDir(), gmc.vramStart,
                      s.ctx)) {
    return HSA_STATUS_ERROR;
  }

  // framebuffer_base_ = GMC VRAM MC base (== lite:: framebuffer_base, and the
  // base wddmAllocVram offsets are added to). Mirrors the Linux transport.
  framebuffer_base_ = s.ctx.vramMcBase;
  s.brought_up = true;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t WindowsLiteDriver::QueryKernelModeDriver(core::DriverQuery query) {
  if (query != core::DriverQuery::GET_DRIVER_VERSION) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  if (adapter_ == 0 && !lite_discovered_) return HSA_STATUS_ERROR;
  // AMDGPU_ESCAPE_GET_INFO carries no version field yet — seed a plausible
  // value from the successful Open() handshake, mirroring the macOS backend.
  // Real versioning lands when the KMD grows a GET_DRIVER_VERSION escape.
  version_.KernelInterfaceMajorVersion = 1;
  version_.KernelInterfaceMinorVersion = 0;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t WindowsLiteDriver::Open() {
  if (adapter_ != 0) return HSA_STATUS_SUCCESS;

  const D3dkmtApi& api = GetD3dkmtApi();
  if (!api.valid) {
    // D3DKMTEnumAdapters3 needs Win10 1903+. Without it (or any of the other
    // entry points) there is no amdgpu_mcdm transport — report "no device".
    return HSA_STATUS_ERROR;
  }

  // Two-call D3DKMTEnumAdapters3: first to learn the count, then to fill the
  // array. Filter=0 (no filter — enumerate every adapter).
  D3DKMT_ENUMADAPTERS3 enum_args{};
  enum_args.Filter.Value = 0;
  enum_args.NumAdapters = 0;
  enum_args.pAdapters = nullptr;
  if (api.EnumAdapters3(&enum_args) < 0) return HSA_STATUS_ERROR;
  if (enum_args.NumAdapters == 0) return HSA_STATUS_ERROR;

  std::vector<D3DKMT_ADAPTERINFO> adapters(enum_args.NumAdapters);
  enum_args.pAdapters = adapters.data();
  if (api.EnumAdapters3(&enum_args) < 0) return HSA_STATUS_ERROR;
  const UINT num_adapters = std::min<UINT>(enum_args.NumAdapters,
                                           static_cast<UINT>(adapters.size()));

  for (UINT i = 0; i < num_adapters; ++i) {
    D3DKMT_OPENADAPTERFROMLUID open_args{};
    open_args.AdapterLuid = adapters[i].AdapterLuid;
    if (api.OpenAdapterFromLuid(&open_args) < 0) continue;
    const D3DKMT_HANDLE adapter = open_args.hAdapter;

    D3DKMT_CREATEDEVICE create_args{};
    create_args.hAdapter = adapter;
    if (api.CreateDevice(&create_args) < 0) {
      D3DKMT_CLOSEADAPTER close_args{};
      close_args.hAdapter = adapter;
      api.CloseAdapter(&close_args);
      continue;
    }
    const D3DKMT_HANDLE device = create_args.hDevice;

    // GET_INFO handshake: a transport-OK escape with Header.Status==0 and the
    // AMD vendor id identifies our amdgpu_mcdm adapter.
    AmdgpuGetInfoData info{};
    info.Header.Command = AMDGPU_ESCAPE_GET_INFO;
    info.Header.Size = sizeof(info);
    const bool transport_ok = IssueEscape(adapter, device, &info, sizeof(info));
    if (transport_ok && info.Header.Status == 0 && info.VendorId == 0x1002) {
      adapter_ = adapter;
      device_ = device;
      info_.vendor_id = info.VendorId;
      info_.device_id = info.DeviceId;
      info_.revision_id = info.RevisionId;
      info_.vram_size = info.VramSizeBytes;
      info_.visible_vram_size = info.VisibleVramSizeBytes;
      return HSA_STATUS_SUCCESS;
    }

    // Not our adapter: tear down this device/adapter and try the next.
    D3DKMT_DESTROYDEVICE destroy_args{};
    destroy_args.hDevice = device;
    api.DestroyDevice(&destroy_args);
    D3DKMT_CLOSEADAPTER close_args{};
    close_args.hAdapter = adapter;
    api.CloseAdapter(&close_args);
  }

  // No amdgpu_mcdm escape adapter found. Fall back to the proven wddm_lite
  // adapter (the lite:: dispatch path uses it anyway) so the driver is still
  // discoverable. Probe a transient WddmLite for the AMD vendor id, then close
  // it -- EnsureGpuBringUpLocked opens its own instance for dispatch.
  {
    WddmLite probe;
    if (probe.open()) {
      AMDGPU_ESCAPE_GET_INFO_DATA li{};
      if (probe.getInfo(&li) && li.VendorId == 0x1002) {
        info_.vendor_id = li.VendorId;
        lite_discovered_ = true;
        probe.close();
        return HSA_STATUS_SUCCESS;
      }
      probe.close();
    }
  }
  return HSA_STATUS_ERROR;
}

hsa_status_t WindowsLiteDriver::Close() {
  const D3dkmtApi& api = GetD3dkmtApi();
  if (adapter_ != 0 && api.valid) {
    // UNMAP each cached aperture before destroying the device. UNMAP_BAR
    // carries the opaque MappingHandle from the MAP_VRAM / MAP_BAR escape.
    if (vram_map_handle_ != nullptr) {
      AmdgpuMapBarData unmap{};
      unmap.Header.Command = AMDGPU_ESCAPE_UNMAP_BAR;
      unmap.Header.Size = sizeof(unmap);
      unmap.MappingHandle = vram_map_handle_;
      IssueEscape(adapter_, device_, &unmap, sizeof(unmap));
    }
    if (doorbell_map_handle_ != nullptr) {
      AmdgpuMapBarData unmap{};
      unmap.Header.Command = AMDGPU_ESCAPE_UNMAP_BAR;
      unmap.Header.Size = sizeof(unmap);
      unmap.MappingHandle = doorbell_map_handle_;
      IssueEscape(adapter_, device_, &unmap, sizeof(unmap));
    }
    if (device_ != 0) {
      D3DKMT_DESTROYDEVICE destroy_args{};
      destroy_args.hDevice = device_;
      api.DestroyDevice(&destroy_args);
    }
    D3DKMT_CLOSEADAPTER close_args{};
    close_args.hAdapter = adapter_;
    api.CloseAdapter(&close_args);
  }
  adapter_ = 0;
  device_ = 0;
  mmio_bar_cpu_ = nullptr;
  mmio_bar_size_ = 0;
  vram_bar_cpu_ = nullptr;
  vram_bar_size_ = 0;
  vram_map_handle_ = nullptr;
  doorbell_bar_cpu_ = nullptr;
  doorbell_bar_size_ = 0;
  doorbell_map_handle_ = nullptr;
  framebuffer_base_ = 0;
  next_vram_offset_ = 0;
  // Release the wddm_lite instance (its destructor closes the D3DKMT device).
  if (wddm_lite_state_ != nullptr) {
    wddm_lite_state_->gpu.close();
    wddm_lite_state_.reset();
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t WindowsLiteDriver::GetSystemProperties(HsaSystemProperties& sys_props) const {
  sys_props.NumNodes = (adapter_ != 0 || lite_discovered_) ? 1 : 0;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t WindowsLiteDriver::GetNodeProperties(HsaNodeProperties& node_props,
                                                  uint32_t node_id) const {
  if (node_id != 0) return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  std::memset(&node_props, 0, sizeof(node_props));

  SYSTEM_INFO si{};
  GetSystemInfo(&si);
  node_props.NumCPUCores =
      static_cast<HSAuint32>(si.dwNumberOfProcessors > 0 ? si.dwNumberOfProcessors : 1);
  // Non-zero NumFComputeCores is the signal to DiscoverGpu() that this node is
  // a GPU. Navi48 / RDNA4 shape, matching the macOS backend.
  node_props.NumFComputeCores = 64 * 2;
  node_props.NumMemoryBanks = 1;
  node_props.NumCaches = 0;
  node_props.NumIOLinks = 0;
  node_props.VendorId = info_.vendor_id;
  node_props.DeviceId = info_.device_id;
  node_props.LocalMemSize = info_.vram_size;
  node_props.EngineId.ui32.Major = 12;
  node_props.EngineId.ui32.Minor = 0;
  node_props.EngineId.ui32.Stepping = 1;
  node_props.WaveFrontSize = 32;
  node_props.NumSIMDPerCU = 2;
  node_props.NumCUPerArray = 8;
  node_props.NumArrays = 2;
  node_props.NumShaderBanks = 4;
  node_props.MaxWavesPerSIMD = 16;
  node_props.LDSSizeInKB = 64;
  node_props.MaxEngineClockMhzFCompute = 2500;
  node_props.NumSdmaEngines = 1;
  node_props.NumSdmaQueuesPerEngine = 1;
  node_props.NumCpQueues = 1;
  node_props.NumXcc = 1;
  node_props.Capability.ui32.QueueSizePowerOfTwo = 1;
  node_props.Capability.ui32.QueueSize32bit = 1;
  node_props.Capability.ui32.ASICRevision = info_.revision_id & 0xF;
  node_props.Capability.ui32.SVMAPISupported = 1;
  std::snprintf(reinterpret_cast<char*>(node_props.AMDName), HSA_PUBLIC_NAME_SIZE, "gfx1201");
  const char name[] = "AMD Radeon RX 9000 (Windows WDDM)";
  for (size_t i = 0; i < sizeof(name) && i < HSA_PUBLIC_NAME_SIZE; ++i) {
    node_props.MarketingName[i] = static_cast<HSAuint16>(name[i]);
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t WindowsLiteDriver::GetEdgeProperties(std::vector<HsaIoLinkProperties>& io_link_props,
                                                  uint32_t) const {
  io_link_props.clear();
  return HSA_STATUS_SUCCESS;
}

hsa_status_t WindowsLiteDriver::GetMemoryProperties(
    uint32_t node_id, std::vector<HsaMemoryProperties>& mem_props) const {
  if (node_id != 0) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  if (mem_props.size() < 1) mem_props.resize(1);

  MEMORYSTATUSEX ms{};
  ms.dwLength = sizeof(ms);
  uint64_t mem_bytes = 0;
  if (GlobalMemoryStatusEx(&ms)) mem_bytes = ms.ullTotalPhys;

  std::memset(&mem_props[0], 0, sizeof(HsaMemoryProperties));
  mem_props[0].HeapType = HSA_HEAPTYPE_SYSTEM;
  mem_props[0].SizeInBytes = mem_bytes;
  mem_props[0].VirtualBaseAddress = 0;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t WindowsLiteDriver::GetCacheProperties(
    uint32_t, uint32_t, std::vector<HsaCacheProperties>& cache_props) const {
  cache_props.clear();
  return HSA_STATUS_SUCCESS;
}

hsa_status_t WindowsLiteDriver::AllocateMemory(const core::MemoryRegion& mem_region,
                                               core::MemoryRegion::AllocateFlags /*alloc_flags*/,
                                               void** mem, size_t size,
                                               uint32_t /*node_id*/) {
  if (!mem) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  *mem = nullptr;

  const auto& amd_region = static_cast<const AMD::MemoryRegion&>(mem_region);
  if (amd_region.IsLocalMemory()) {
    // First pass: device tensors come from the VRAM-BAR window (no IOMMU
    // dependency). The coherent ALLOC_DMA path waits on the KMD gap.
    uint64_t gpu_addr = 0;
    return AllocateVram(size, 4096, mem, &gpu_addr);
  }

  // Host memory: page-aligned, zero-filled, RW — the Win32 analog of the
  // Linux/macOS mmap(MAP_ANON) path.
  SYSTEM_INFO si{};
  GetSystemInfo(&si);
  const size_t page = si.dwPageSize > 0 ? si.dwPageSize : 4096;
  const size_t rounded = (size + page - 1) & ~(page - 1);
  void* ptr = ::VirtualAlloc(nullptr, rounded, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
  if (ptr == nullptr) return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  {
    auto& reg = GetHostAllocRegistry();
    std::lock_guard<std::mutex> g(reg.m);
    reg.allocations[ptr] = rounded;
  }
  *mem = ptr;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t WindowsLiteDriver::FreeMemory(void* mem, size_t /*size*/) {
  if (!mem) return HSA_STATUS_SUCCESS;
  {
    std::lock_guard<std::mutex> g(gpu_lock_);
    auto dit = dma_allocations_.find(mem);
    if (dit != dma_allocations_.end()) {
      // FREE_DMA escape lands with the coherent-queue milestone.
      dma_allocations_.erase(dit);
      return HSA_STATUS_SUCCESS;
    }
    auto vit = vram_allocations_.find(mem);
    if (vit != vram_allocations_.end()) {
      // Bump-allocated VRAM-BAR window is intentionally not reused yet.
      vram_allocations_.erase(vit);
      return HSA_STATUS_SUCCESS;
    }
  }
  auto& reg = GetHostAllocRegistry();
  {
    std::lock_guard<std::mutex> g(reg.m);
    auto it = reg.allocations.find(mem);
    if (it == reg.allocations.end()) return HSA_STATUS_ERROR_INVALID_ALLOCATION;
    reg.allocations.erase(it);
  }
  // MEM_RELEASE requires size 0 and the base address returned by VirtualAlloc.
  if (!::VirtualFree(mem, 0, MEM_RELEASE)) return HSA_STATUS_ERROR;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t WindowsLiteDriver::EnsureBarMappingsLocked() {
  if (adapter_ == 0) return HSA_STATUS_ERROR;

  if (vram_bar_cpu_ == nullptr) {
    // Map the visible-VRAM window via BAR2. Cap the request at 256MB so the
    // mapping fits within the host BAR/PTE budget even when the KMD reports a
    // larger visible aperture.
    constexpr uint64_t kVramMapWindowMax = 256ull * 1024 * 1024;
    uint64_t length = info_.visible_vram_size;
    if (length == 0 || length > kVramMapWindowMax) length = kVramMapWindowMax;
    AmdgpuMapVramData map{};
    map.Header.Command = AMDGPU_ESCAPE_MAP_VRAM;
    map.Header.Size = sizeof(map);
    map.Offset = 0;
    map.Length = length;
    if (!IssueEscape(adapter_, device_, &map, sizeof(map)) || map.Header.Status != 0) {
      return HSA_STATUS_ERROR;
    }
    vram_bar_cpu_ = map.MappedAddress;
    vram_bar_size_ = map.Length;
    vram_map_handle_ = map.MappingHandle;
  }

  if (doorbell_bar_cpu_ == nullptr) {
    // Doorbell aperture is BAR2 (ring_init.py:750 map_bar(2, ...)). Length 0
    // requests the entire BAR.
    AmdgpuMapBarData map{};
    map.Header.Command = AMDGPU_ESCAPE_MAP_BAR;
    map.Header.Size = sizeof(map);
    map.BarIndex = 2;
    map.Offset = 0;
    map.Length = 0;
    if (!IssueEscape(adapter_, device_, &map, sizeof(map)) || map.Header.Status != 0) {
      return HSA_STATUS_ERROR;
    }
    doorbell_bar_cpu_ = map.MappedAddress;
    doorbell_bar_size_ = map.Length;
    doorbell_map_handle_ = map.MappingHandle;
  }

  if (framebuffer_base_ == 0) {
    // MMHUB base 0x1A000, regMMMC_VM_FB_LOCATION_BASE 0x0554 (gmc_init.py:32);
    // FB base bits are the low 24 bits, shifted up by 24 (same as the macOS
    // backend).
    uint32_t fb = 0;
    hsa_status_t status = ReadMmio32(0x1A000, 0x0554, &fb);
    if (status != HSA_STATUS_SUCCESS) return status;
    framebuffer_base_ = static_cast<uint64_t>(fb & 0xFFFFFF) << 24;
  }

  if (next_vram_offset_ == 0) {
    next_vram_offset_ = kVramAllocBaseOffset;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t WindowsLiteDriver::AllocateVram(size_t size, size_t align, void** cpu_addr,
                                             uint64_t* gpu_addr) {
  if (cpu_addr == nullptr || gpu_addr == nullptr || size == 0) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  *cpu_addr = nullptr;
  *gpu_addr = 0;
  align = std::max<size_t>(align, 4096);
  if ((align & (align - 1)) != 0) return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  std::lock_guard<std::mutex> g(gpu_lock_);
  // When bring-up went through wddm_lite (discovered via the probe -> adapter_==0,
  // so the amdgpu_mcdm BAR window in EnsureBarMappingsLocked is unavailable),
  // allocate VRAM through the proven wddm_lite bump allocator -- the SAME cursor the
  // lite:: queue uses, so no collision. Record it in vram_allocations_ keyed by the
  // returned CPU pointer so HostToGpuAddress/IsRegisteredVramPointer can translate it
  // (each wddm_lite alloc is its own MAP_VRAM mapping, not a slice of one window).
  if (wddm_lite_state_ != nullptr && wddm_lite_state_->brought_up) {
    const uint64_t rounded_lite = AlignUpU64(size, align);
    void* cpu = nullptr;
    uint64_t gpu = 0, handle = 0;
    if (!wddmAllocVram(wddm_lite_state_->gpu, rounded_lite, &cpu, &gpu, &handle)) {
      return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    }
    VramAllocation alloc;
    alloc.offset = 0;
    alloc.size = rounded_lite;
    alloc.gpu_addr = gpu;
    vram_allocations_[cpu] = alloc;
    *cpu_addr = cpu;
    *gpu_addr = gpu;
    return HSA_STATUS_SUCCESS;
  }

  hsa_status_t status = EnsureBarMappingsLocked();
  if (status != HSA_STATUS_SUCCESS) return status;

  const uint64_t rounded = AlignUpU64(size, align);
  const uint64_t offset = AlignUpU64(next_vram_offset_, align);
  if (offset + rounded > vram_bar_size_) return HSA_STATUS_ERROR_OUT_OF_RESOURCES;

  auto* ptr = static_cast<char*>(vram_bar_cpu_) + offset;
  next_vram_offset_ = offset + rounded;
  VramAllocation alloc;
  alloc.offset = offset;
  alloc.size = rounded;
  alloc.gpu_addr = framebuffer_base_ + offset;
  vram_allocations_[ptr] = alloc;
  *cpu_addr = ptr;
  *gpu_addr = alloc.gpu_addr;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t WindowsLiteDriver::HostToGpuAddress(const void* ptr, uint64_t* gpu_addr) const {
  if (ptr == nullptr || gpu_addr == nullptr) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  *gpu_addr = 0;
  std::lock_guard<std::mutex> g(gpu_lock_);
  // DMA (IOVA) tier first (wired at the coherent-queue milestone).
  const auto dp = reinterpret_cast<uintptr_t>(ptr);
  for (const auto& kv : dma_allocations_) {
    const auto base = reinterpret_cast<uintptr_t>(kv.first);
    if (dp >= base && dp < base + kv.second.size) {
      *gpu_addr = kv.second.bus_addr + (dp - base);
      return HSA_STATUS_SUCCESS;
    }
  }
  // wddm_lite VRAM allocations (each its own MAP_VRAM mapping).
  for (const auto& kv : vram_allocations_) {
    const auto vbase = reinterpret_cast<uintptr_t>(kv.first);
    if (dp >= vbase && dp < vbase + kv.second.size) {
      *gpu_addr = kv.second.gpu_addr + (dp - vbase);
      return HSA_STATUS_SUCCESS;
    }
  }
  if (vram_bar_cpu_ == nullptr || framebuffer_base_ == 0) {
    return HSA_STATUS_ERROR_INVALID_ALLOCATION;
  }
  const auto base = reinterpret_cast<uintptr_t>(vram_bar_cpu_);
  if (dp < base || dp >= base + vram_bar_size_) return HSA_STATUS_ERROR_INVALID_ALLOCATION;
  *gpu_addr = framebuffer_base_ + (dp - base);
  return HSA_STATUS_SUCCESS;
}

bool WindowsLiteDriver::IsRegisteredVramPointer(const void* ptr) const {
  if (ptr == nullptr) return false;
  std::lock_guard<std::mutex> g(gpu_lock_);
  const auto p = reinterpret_cast<uintptr_t>(ptr);
  for (const auto& kv : dma_allocations_) {
    const auto base = reinterpret_cast<uintptr_t>(kv.first);
    if (p >= base && p < base + kv.second.size) return true;
  }
  for (const auto& kv : vram_allocations_) {
    const auto base = reinterpret_cast<uintptr_t>(kv.first);
    if (p >= base && p < base + kv.second.size) return true;
  }
  if (vram_bar_cpu_ != nullptr) {
    const auto base = reinterpret_cast<uintptr_t>(vram_bar_cpu_);
    if (p >= base && p < base + vram_bar_size_) return true;
  }
  return false;
}

// ---- lite::DirectQueuePlatform overrides ------------------------------------

hsa_status_t WindowsLiteDriver::ReadMmio32(uint32_t base, uint32_t reg,
                                           uint32_t* value) const {
  if (value == nullptr) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  // wddm_lite path: byte_offset = (base+reg)*4 on BAR0, identical to the Linux
  // transport's ReadMmio32 and gpu_init.cpp's gcReg/mmhubRead.
  if (wddm_lite_state_ != nullptr) {
    const uint32_t off = (base + reg) * 4u;
    return wddm_lite_state_->gpu.readReg32(off, value, /*barIndex=*/0)
               ? HSA_STATUS_SUCCESS
               : HSA_STATUS_ERROR;
  }
  if (adapter_ == 0) return HSA_STATUS_ERROR;
  // MMIO is BAR0; the (base+reg)*4 byte-offset convention matches the
  // macOS/Linux backends and the Python read_reg32 usage.
  AmdgpuReg32Data cmd{};
  cmd.Header.Command = AMDGPU_ESCAPE_READ_REG32;
  cmd.Header.Size = sizeof(cmd);
  cmd.BarIndex = 0;
  cmd.Offset = static_cast<ULONG>((static_cast<uint64_t>(base) + reg) * 4);
  if (!IssueEscape(adapter_, device_, &cmd, sizeof(cmd)) || cmd.Header.Status != 0) {
    return HSA_STATUS_ERROR;
  }
  *value = cmd.Value;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t WindowsLiteDriver::WriteMmio32(uint32_t base, uint32_t reg,
                                            uint32_t value) const {
  if (wddm_lite_state_ != nullptr) {
    const uint32_t off = (base + reg) * 4u;
    return wddm_lite_state_->gpu.writeReg32(off, value, /*barIndex=*/0)
               ? HSA_STATUS_SUCCESS
               : HSA_STATUS_ERROR;
  }
  if (adapter_ == 0) return HSA_STATUS_ERROR;
  AmdgpuReg32Data cmd{};
  cmd.Header.Command = AMDGPU_ESCAPE_WRITE_REG32;
  cmd.Header.Size = sizeof(cmd);
  cmd.BarIndex = 0;
  cmd.Offset = static_cast<ULONG>((static_cast<uint64_t>(base) + reg) * 4);
  cmd.Value = value;
  if (!IssueEscape(adapter_, device_, &cmd, sizeof(cmd)) || cmd.Header.Status != 0) {
    return HSA_STATUS_ERROR;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t WindowsLiteDriver::EnsureDoorbellAperture() const {
  // wddm_lite path: reuse the recipe's NBIO doorbell-aperture + framebuffer
  // enable (resolves the NBIF base via IP discovery, matching the recipe).
  if (wddm_lite_state_ != nullptr) {
    return wddmEnsureDoorbellAperture(wddm_lite_state_->gpu,
                                      wddm_lite_state_->ipd)
               ? HSA_STATUS_SUCCESS
               : HSA_STATUS_ERROR;
  }
  // The register sequence is architectural (identical to the macOS/Linux
  // backends); it becomes reachable once WriteMmio32 is wired.
  hsa_status_t status = WriteMmio32(NBIO_B2, regRCC_DEV0_EPF0_RCC_DOORBELL_APER_EN, 1);
  if (status != HSA_STATUS_SUCCESS) return status;
  status = WriteMmio32(NBIO_B2, regGDC_S2A0_S2A_DOORBELL_ENTRY_0_CTRL,
                       (1u << 0) | (3u << 1) | (3u << 28));
  if (status != HSA_STATUS_SUCCESS) return status;
  status = WriteMmio32(NBIO_B2, regGDC_S2A0_S2A_DOORBELL_ENTRY_3_CTRL,
                       (1u << 0) | (6u << 1) | (3u << 28));
  if (status != HSA_STATUS_SUCCESS) return status;
  status = WriteMmio32(GC_B0, regCP_MEC_DOORBELL_RANGE_LOWER, 0);
  if (status != HSA_STATUS_SUCCESS) return status;
  return WriteMmio32(GC_B0, regCP_MEC_DOORBELL_RANGE_UPPER, (0x8Au * 2u) << 2);
}

hsa_status_t WindowsLiteDriver::WriteGpuMemory32(uint64_t offset, uint32_t value) const {
  if (vram_bar_cpu_ == nullptr || offset + sizeof(uint32_t) > vram_bar_size_) {
    return HSA_STATUS_ERROR_INVALID_ALLOCATION;
  }
  auto* ptr = reinterpret_cast<volatile uint32_t*>(static_cast<char*>(vram_bar_cpu_) + offset);
  *ptr = value;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t WindowsLiteDriver::ZeroGpuMemory(uint64_t offset, uint64_t size) const {
  for (uint64_t i = 0; i < size; i += sizeof(uint32_t)) {
    hsa_status_t status = WriteGpuMemory32(offset + i, 0);
    if (status != HSA_STATUS_SUCCESS) return status;
  }
  return HSA_STATUS_SUCCESS;
}

void* WindowsLiteDriver::GpuMemoryCpuPointer(uint64_t offset) const {
  if (vram_bar_cpu_ == nullptr || offset >= vram_bar_size_) return nullptr;
  return static_cast<char*>(vram_bar_cpu_) + offset;
}

bool WindowsLiteDriver::PreferAllocatedQueueMemory() const {
  // wddm_lite path: each queue gets one VRAM buffer (gpu_addr = vramMcBase +
  // offset) via the recipe bump allocator, so lite:: uses
  // BuildQueueLayoutFromMemory (cpu_base) directly -- no VRAM-window offset
  // translation. Matches the proven recipe's rcpAllocVram queue layout.
  if (wddm_lite_state_ != nullptr) return true;
  // Escape fallback (no bring-up): queue ring/MQD/rptr/wptr live in the
  // VRAM-BAR window, requiring GpuMemoryCpuPointer offsets.
  return false;
}

hsa_status_t WindowsLiteDriver::AllocateQueueMemory(uint64_t size,
                                                    lite::DirectQueueMemory* memory) const {
  if (memory == nullptr) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  *memory = {};
  if (wddm_lite_state_ != nullptr) {
    // One FB-MC VRAM buffer covering the whole queue stride (ring/MQD/EOP/
    // rptr/wptr). gpu_addr is the FB-MC address the CP fetches from.
    void* cpu = nullptr;
    uint64_t gpu_addr = 0;
    uint64_t handle = 0;
    if (!wddmAllocVram(wddm_lite_state_->gpu, size, &cpu, &gpu_addr, &handle)) {
      return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    }
    memory->size = size;
    memory->gpu_addr = gpu_addr;
    memory->cpu = cpu;
    memory->platform_handle = handle;
    return HSA_STATUS_SUCCESS;
  }
  // Escape fallback: lite:: drops to the VRAM-BAR window.
  return HSA_STATUS_ERROR;
}

hsa_status_t WindowsLiteDriver::FreeQueueMemory(lite::DirectQueueMemory* memory) const {
  // The recipe bump allocator does not reclaim (matches wddm_lite + the macOS
  // backend); just clear the record.
  if (memory != nullptr) *memory = {};
  return HSA_STATUS_SUCCESS;
}

hsa_status_t WindowsLiteDriver::FlushHdp() const {
  // HDP flush: write 0 to HDP_MEM_COHERENCY_FLUSH (0x00F7) on the NBIF base
  // (base_idx 2), identical to the Linux transport's FlushHdp and gpu_init.cpp
  // cqHdpFlush.
  if (wddm_lite_state_ != nullptr) {
    if (!wddm_lite_state_->ctx.hasNbif) return HSA_STATUS_SUCCESS;
    return WriteMmio32(wddm_lite_state_->ctx.nbifBase2, 0x00F7, 0);
  }
  return HSA_STATUS_SUCCESS;
}

volatile uint64_t* WindowsLiteDriver::DoorbellCpuPointer(uint32_t doorbell_index) const {
  if (wddm_lite_state_ != nullptr) {
    // BAR2 at doorbell_index * 4 (8 bytes), mapped+cached per index. Matches
    // gpu_init.cpp cqInitComputeQueue's mapBar(2, idx*4, 8) and the lite::
    // Linux transport's doorbell_index * sizeof(uint32_t) offset.
    auto& s = *wddm_lite_state_;
    for (const auto& m : s.doorbells) {
      if (m.index == doorbell_index) return m.cpu;
    }
    void* addr = nullptr;
    void* handle = nullptr;
    if (!s.gpu.mapBar(2, static_cast<uint64_t>(doorbell_index) * 4, 8, &addr,
                      &handle) ||
        addr == nullptr) {
      return nullptr;
    }
    auto* cpu = reinterpret_cast<volatile uint64_t*>(addr);
    s.doorbells.push_back({doorbell_index, cpu});
    return cpu;
  }
  const uint64_t byte_offset = static_cast<uint64_t>(doorbell_index) * sizeof(uint32_t);
  if (doorbell_bar_cpu_ == nullptr || byte_offset + sizeof(uint64_t) > doorbell_bar_size_) {
    return nullptr;
  }
  return reinterpret_cast<volatile uint64_t*>(static_cast<char*>(doorbell_bar_cpu_) + byte_offset);
}

void WindowsLiteDriver::SleepUs(uint32_t usec) const {
  // Sleep()'s ~1-15ms granularity is far too coarse for the lite:: dequeue /
  // activate settle timings; use a QueryPerformanceCounter spin/yield for
  // sub-millisecond accuracy.
  LARGE_INTEGER freq{};
  if (!QueryPerformanceFrequency(&freq) || freq.QuadPart == 0) {
    if (usec >= 1000) ::Sleep(usec / 1000);
    return;
  }
  LARGE_INTEGER start{};
  QueryPerformanceCounter(&start);
  const long long target_ticks =
      start.QuadPart + (static_cast<long long>(usec) * freq.QuadPart) / 1000000ll;
  for (;;) {
    LARGE_INTEGER now{};
    QueryPerformanceCounter(&now);
    if (now.QuadPart >= target_ticks) break;
    ::SwitchToThread();
  }
}

// ---- direct-compute queue wrappers ------------------------------------------

hsa_status_t WindowsLiteDriver::CreateDirectComputeQueue(DirectComputeQueue* queue) {
  if (queue == nullptr) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  std::lock_guard<std::mutex> g(gpu_lock_);
  // Prefer the wddm_lite recipe path (bootload + MEC enable + FB-MC queue
  // memory). Bring-up may have been deferred at Init() time (no firmware),
  // so attempt it here; fall back to the escape VRAM-window path otherwise.
  hsa_status_t status = EnsureGpuBringUpLocked();
  if (status != HSA_STATUS_SUCCESS) {
    status = EnsureBarMappingsLocked();
    if (status != HSA_STATUS_SUCCESS) return status;
  }
  if (next_direct_queue_index_ >= 8) return HSA_STATUS_ERROR_OUT_OF_RESOURCES;

  const uint32_t queue_index = next_direct_queue_index_++;
  status = lite::CreateDirectQueue(*this, queue, queue_index, framebuffer_base_,
                                   WindowsDirectQueueOptions());
  if (status != HSA_STATUS_SUCCESS) {
    --next_direct_queue_index_;
    *queue = {};
  }
  return status;
}

hsa_status_t WindowsLiteDriver::DestroyDirectComputeQueue(DirectComputeQueue& queue) {
  std::lock_guard<std::mutex> g(gpu_lock_);
  return lite::DestroyDirectQueue(*this, queue, WindowsDirectQueueOptions());
}

hsa_status_t WindowsLiteDriver::SubmitDirectCompute(DirectComputeQueue& queue,
                                                    const uint32_t* pm4,
                                                    size_t dword_count) const {
  std::lock_guard<std::mutex> g(gpu_lock_);
  return lite::SubmitDirectQueue(*this, queue, pm4, dword_count, WindowsDirectQueueOptions());
}

hsa_status_t WindowsLiteDriver::ReadDirectComputeRptr(const DirectComputeQueue& queue,
                                                      uint32_t* rptr) const {
  std::lock_guard<std::mutex> g(gpu_lock_);
  return lite::ReadDirectQueueRptr(*this, queue, rptr);
}

hsa_status_t WindowsLiteDriver::SetQueueScratch(DirectComputeQueue& queue,
                                                uint64_t scratch_base_256,
                                                uint32_t tmpring_size) const {
  std::lock_guard<std::mutex> g(gpu_lock_);
  return lite::SetDirectQueueScratch(*this, queue, scratch_base_256, tmpring_size,
                                     WindowsDirectQueueOptions());
}

// Bring-up validation (Windows only; not part of the HSA API): dispatch a real
// compiled kernel through THIS WindowsLiteDriver object. Mirrors the standalone
// meskern probe but routes creation + submit through the driver's own
// CreateDirectComputeQueue / SubmitDirectCompute (the shared lite:: path over
// this driver's WddmLite instance), and stages the kernel GPUVM against that
// same instance.
hsa_status_t WindowsLiteDriver::DispatchKernelSelfTest() {
  // 1. Bring up the GPU + create a direct compute queue through this driver.
  DirectComputeQueue queue{};
  hsa_status_t status = CreateDirectComputeQueue(&queue);
  if (status != HSA_STATUS_SUCCESS) {
    std::fprintf(stderr,
                 "DispatchKernelSelfTest: CreateDirectComputeQueue failed (%u)\n",
                 status);
    return status;
  }
  std::printf("DispatchKernelSelfTest: queue qid=%u doorbell=0x%X mes_backed=%d\n",
              queue.queue_id, queue.doorbell_index, queue.mes_backed ? 1 : 0);
  if (wddm_lite_state_ == nullptr) {
    DestroyDirectComputeQueue(queue);
    return HSA_STATUS_ERROR;
  }
  WddmLiteState& s = *wddm_lite_state_;

  // 2. Stage the kernel GPUVM (code + kernarg + output + 4-level page table +
  //    GCVM_CONTEXT0). skipMecReassert=true: the direct HQD is already active
  //    from CreateDirectComputeQueue, so do NOT pulse-reset the MEC pipes.
  WddmKernelStage stage;
  if (!wddmStageKernelGpuvm(s.gpu, s.ipd, s.ctx, WindowsFirmwareDir(), stage,
                            /*skipMecReassert=*/true)) {
    std::fprintf(stderr, "DispatchKernelSelfTest: wddmStageKernelGpuvm failed\n");
    DestroyDirectComputeQueue(queue);
    return HSA_STATUS_ERROR;
  }

  // 3. Fence buffer (FB-MC addressable + CPU-mapped), separate from the queue.
  void* fence_cpu = nullptr;
  uint64_t fence_gpu = 0, fence_handle = 0;
  if (!wddmAllocVram(s.gpu, 4096, &fence_cpu, &fence_gpu, &fence_handle)) {
    std::fprintf(stderr, "DispatchKernelSelfTest: fence alloc failed\n");
    DestroyDirectComputeQueue(queue);
    return HSA_STATUS_ERROR;
  }
  volatile uint64_t* fence = static_cast<volatile uint64_t*>(fence_cpu);
  *fence = 0;
  std::atomic_thread_fence(std::memory_order_seq_cst);
  FlushHdp();

  // 4. Build the dispatch PM4 (RELEASE_MEM writes 1 to fence_gpu) and submit it
  //    through this driver's queue.
  std::vector<uint32_t> pm4;
  if (!wddmBuildKernelDispatchPm4(stage, fence_gpu, pm4)) {
    std::fprintf(stderr, "DispatchKernelSelfTest: wddmBuildKernelDispatchPm4 failed\n");
    DestroyDirectComputeQueue(queue);
    return HSA_STATUS_ERROR;
  }
  std::printf("DispatchKernelSelfTest: dispatch PM4 = %zu dwords\n", pm4.size());
  status = SubmitDirectCompute(queue, pm4.data(), pm4.size());
  if (status != HSA_STATUS_SUCCESS) {
    std::fprintf(stderr, "DispatchKernelSelfTest: SubmitDirectCompute failed (%u)\n",
                 status);
    DestroyDirectComputeQueue(queue);
    return status;
  }

  // 5. Poll the RELEASE_MEM fence (5 s).
  bool signaled = false;
  for (int i = 0; i < 5000; ++i) {
    if (*fence == 1) { signaled = true; break; }
    ::Sleep(1);
  }
  FlushHdp();

  // 6. GPUVM fault status + output verification.
  uint32_t fault_status = 0;
  ReadMmio32(stage.gcBase0, 0x15D0, &fault_status);
  const volatile uint32_t* out =
      static_cast<const volatile uint32_t*>(stage.outCpu);
  uint32_t nbad = 0, first_bad = 0, first_bad_val = 0;
  for (uint32_t i = 0; i < stage.fillN; ++i) {
    uint32_t v = out[i];
    if (v != stage.fillVal) {
      if (nbad == 0) { first_bad = i; first_bad_val = v; }
      ++nbad;
    }
  }
  std::printf("DispatchKernelSelfTest: FAULT_STATUS=0x%08X FENCE=%llu (exp 1) "
              "out[0]=0x%08X exp=0x%08X bad=%u/%u\n",
              fault_status, static_cast<unsigned long long>(*fence), out[0],
              stage.fillVal, nbad, stage.fillN);
  if (nbad)
    std::printf("DispatchKernelSelfTest: first mismatch at %u: 0x%08X\n",
                first_bad, first_bad_val);

  DestroyDirectComputeQueue(queue);

  const bool pass = signaled && fault_status == 0 && nbad == 0;
  std::printf("WINDOWSLITEDRIVER KERN %s\n",
              pass ? "PASS (fence signaled, no fault, output verified)"
                   : (!signaled ? "FAIL (fence timeout)"
                                : (fault_status ? "FAIL (GPUVM fault)"
                                                : "FAIL (output mismatch)")));
  return pass ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR;
}

// ---- queue / sharing / SPM / misc: not supported by the lite:: backend ------

hsa_status_t WindowsLiteDriver::CreateQueue(uint32_t, HSA_QUEUE_TYPE, uint32_t,
                                            HSA::hsa_amd_queue_priority_internal_t,
                                            uint32_t, void*, uint64_t, uint64_t, HsaEvent*,
                                            HsaQueueResource&) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t WindowsLiteDriver::DestroyQueue(HSA_QUEUEID) const { return HSA_STATUS_ERROR; }

hsa_status_t WindowsLiteDriver::UpdateQueue(HSA_QUEUEID, uint32_t,
                                            HSA::hsa_amd_queue_priority_internal_t,
                                            void*, uint64_t, HsaEvent*) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t WindowsLiteDriver::SetQueueCUMask(HSA_QUEUEID, uint32_t, uint32_t*) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t WindowsLiteDriver::AllocQueueGWS(HSA_QUEUEID, uint32_t, uint32_t*) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t WindowsLiteDriver::ExportDMABuf(void*, size_t, int*, size_t*) {
  // No dma-buf fd passing on Windows; a shared NT HANDLE path is a future
  // milestone.
  return HSA_STATUS_ERROR;
}

hsa_status_t WindowsLiteDriver::ImportDMABuf(int, const core::Agent&,
                                             core::ShareableHandle*, void*) {
  return HSA_STATUS_ERROR;
}

hsa_status_t WindowsLiteDriver::DestroyImportedShareableHandle(core::ShareableHandle*) {
  return HSA_STATUS_ERROR;
}

hsa_status_t WindowsLiteDriver::Map(core::ShareableHandle, void*, size_t, size_t,
                                    hsa_access_permission_t) {
  return HSA_STATUS_ERROR;
}

hsa_status_t WindowsLiteDriver::Unmap(core::ShareableHandle, void*, size_t, size_t) {
  return HSA_STATUS_ERROR;
}

hsa_status_t WindowsLiteDriver::CreateShareableHandle(void*, void*, size_t,
                                                      const core::Agent&,
                                                      core::ShareableHandle*, uint64_t*,
                                                      int*, uint64_t*) {
  return HSA_STATUS_ERROR;
}

hsa_status_t WindowsLiteDriver::DestroyShareableHandle(core::ShareableHandle*) {
  return HSA_STATUS_ERROR;
}

hsa_status_t WindowsLiteDriver::SPMAcquire(uint32_t) const { return HSA_STATUS_ERROR; }
hsa_status_t WindowsLiteDriver::SPMRelease(uint32_t) const { return HSA_STATUS_ERROR; }
hsa_status_t WindowsLiteDriver::SPMSetDestBuffer(uint32_t, uint32_t, uint32_t*, uint32_t*,
                                                 void*, bool*) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t WindowsLiteDriver::SetTrapHandler(uint32_t, const void*, uint64_t,
                                               const void*, uint64_t) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t WindowsLiteDriver::GetDeviceHandle(uint32_t, void**) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t WindowsLiteDriver::GetClockCounters(uint32_t, HsaClockCounters*) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t WindowsLiteDriver::GetTileConfig(uint32_t, HsaGpuTileConfig*) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t WindowsLiteDriver::IsModelEnabled(bool* enable) const {
  if (enable) *enable = false;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t WindowsLiteDriver::GetWallclockFrequency(uint32_t, uint64_t* frequency) const {
  if (frequency == nullptr) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  *frequency = 1000000000ull;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t WindowsLiteDriver::AllocateScratchMemory(uint32_t, uint64_t, void**) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t WindowsLiteDriver::AvailableMemory(uint32_t, uint64_t* available_size) const {
  if (available_size == nullptr) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  *available_size = info_.vram_size;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t WindowsLiteDriver::RegisterMemory(void*, uint64_t, HsaMemFlags) const {
  return HSA_STATUS_SUCCESS;
}

hsa_status_t WindowsLiteDriver::DeregisterMemory(void*) const { return HSA_STATUS_SUCCESS; }

hsa_status_t WindowsLiteDriver::MakeMemoryResident(const void* mem, size_t, uint64_t* alternate_va,
                                                   const HsaMemMapFlags*,
                                                   uint32_t, const uint32_t*) const {
  if (alternate_va != nullptr) {
    uint64_t gpu_addr = 0;
    *alternate_va = HostToGpuAddress(mem, &gpu_addr) == HSA_STATUS_SUCCESS ? gpu_addr : 0;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t WindowsLiteDriver::MakeMemoryUnresident(const void*) const {
  return HSA_STATUS_SUCCESS;
}

hsa_status_t WindowsLiteDriver::GetQueueSaveAreaInfo(HSA_QUEUEID, void**, size_t*) const {
  return HSA_STATUS_ERROR;
}

}  // namespace AMD
}  // namespace rocr

#endif  // _WIN32
