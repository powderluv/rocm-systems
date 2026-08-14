////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
////////////////////////////////////////////////////////////////////////////////

#include "core/inc/amd_lite_direct_queue.h"

#include <algorithm>
#include <atomic>
#include <array>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <mutex>
#include <unordered_map>

namespace rocr {
namespace AMD {
namespace lite {
namespace {

constexpr uint32_t kGcBase0 = 0x1260;
constexpr uint32_t kGcBase1 = 0xA000;

constexpr uint32_t regGRBM_GFX_CNTL = 0x0900;
constexpr uint32_t regSH_MEM_BASES = 0x09E3;   // base_idx 1 (kGcBase1)
constexpr uint32_t regSH_MEM_CONFIG = 0x09E4;  // base_idx 1 (kGcBase1)
constexpr uint32_t kShMemBasesGfx12 = 0x00010002u;   // (shared_base<<16)|private_base(2)
constexpr uint32_t kShMemConfigGfx12 = 0x0000C00Cu;  // ALIGN_UNALIGNED(3<<2)|INIT_PREFETCH(3<<14)
constexpr uint32_t regGCVM_L2_PROTECTION_FAULT_STATUS = 0x15D0;     // base_idx 0 (kGcBase0)
constexpr uint32_t regGCVM_L2_PROTECTION_FAULT_ADDR_LO32 = 0x15D2;  // base_idx 0
constexpr uint32_t regGCVM_L2_PROTECTION_FAULT_ADDR_HI32 = 0x15D3;  // base_idx 0
constexpr uint32_t regCP_MQD_BASE_ADDR = 0x1fa9;
constexpr uint32_t regCP_MQD_BASE_ADDR_HI = 0x1faa;
constexpr uint32_t regCP_HQD_ACTIVE = 0x1fab;
constexpr uint32_t regCP_HQD_VMID = 0x1fac;
constexpr uint32_t regCP_HQD_PERSISTENT_STATE = 0x1fad;
constexpr uint32_t regCP_HQD_PQ_BASE = 0x1fb1;
constexpr uint32_t regCP_HQD_PQ_BASE_HI = 0x1fb2;
constexpr uint32_t regCP_HQD_PQ_RPTR = 0x1fb3;
constexpr uint32_t regCP_HQD_PQ_RPTR_REPORT_ADDR = 0x1fb4;
constexpr uint32_t regCP_HQD_PQ_RPTR_REPORT_ADDR_HI = 0x1fb5;
constexpr uint32_t regCP_HQD_PQ_WPTR_POLL_ADDR = 0x1fb6;
constexpr uint32_t regCP_HQD_PQ_WPTR_POLL_ADDR_HI = 0x1fb7;
constexpr uint32_t regCP_HQD_PQ_DOORBELL_CONTROL = 0x1fb8;
constexpr uint32_t regCP_HQD_PQ_CONTROL = 0x1fba;
constexpr uint32_t regCP_HQD_GFX_CONTROL = 0x1e9f;
constexpr uint32_t regCP_HQD_DEQUEUE_REQUEST = 0x1fc1;
// Pair RESET_WAVES dequeue with an SPI compute-queue reset (matches tinygrad's
// gfx12 _dequeue_hqds). GC BASE_IDX=0.
constexpr uint32_t regSPI_COMPUTE_QUEUE_RESET = 0x1f73;
constexpr uint32_t regCP_MQD_CONTROL = 0x1fcb;
constexpr uint32_t regCP_HQD_EOP_BASE_ADDR = 0x1fce;
constexpr uint32_t regCP_HQD_EOP_BASE_ADDR_HI = 0x1fcf;
constexpr uint32_t regCP_HQD_EOP_CONTROL = 0x1fd0;
constexpr uint32_t regCP_HQD_PQ_WPTR_LO = 0x1fdf;
constexpr uint32_t regCP_HQD_PQ_WPTR_HI = 0x1fe0;
constexpr uint32_t regCP_HQD_DEQUEUE_STATUS = 0x1fe8;
constexpr uint32_t regCP_UNMAPPED_DOORBELL = 0x0880;
constexpr uint32_t regSCRATCH_REG0 = 0x2040;
// mmGRBM_STATUS: absolute BAR0 DWORD 0x0050 (base_idx 0 -> use base=0, NOT
// gc-base-relative). Global shader/CP-busy status; bit31 GUI_ACTIVE, bit29
// CP_BUSY, bit22 SPI_BUSY. Verified in-repo: wddm_lite_test.cpp readReg32(
// 0x0050 * 4). Read as ReadMmio32(/*base=*/0, regGRBM_STATUS_ABS, &v).
constexpr uint32_t regGRBM_STATUS_ABS = 0x0050;
constexpr uint32_t regCP_MES_CNTL = 0x2807;
// MES engine-start registers (base_idx 1, mirrors ring_init.py / wddmStartMes).
constexpr uint32_t regCP_MES_PRGRM_CNTR_START = 0x2800;
constexpr uint32_t regCP_MES_PRGRM_CNTR_START_HI = 0x289d;
constexpr uint32_t regCP_MES_HEADER_DUMP = 0x280d;
constexpr uint32_t regCP_MES_INSTR_PNTR = 0x2813;
constexpr uint32_t regCP_MES_DOORBELL_CONTROL1 = 0x283c;
constexpr uint32_t regCP_MES_DOORBELL_CONTROL2 = 0x283d;
constexpr uint32_t regCP_MES_DOORBELL_CONTROL3 = 0x283e;
constexpr uint32_t regCP_MES_DOORBELL_CONTROL4 = 0x283f;
constexpr uint32_t regCP_MES_DOORBELL_CONTROL5 = 0x2840;
constexpr uint32_t regCP_MES_GP3_LO = 0x2849;
constexpr uint32_t regCP_PQ_WPTR_POLL_CNTL = 0x1e23;
constexpr uint32_t regCP_PQ_STATUS = 0x1e58;
constexpr uint32_t regRLC_CP_SCHEDULERS = 0x098a;

constexpr uint32_t kCpHqdPersistentStateDefault = 0x0be05501;
constexpr uint32_t kCpMqdControlDefault = 0x00000100;
constexpr uint32_t kCpHqdPqControlDefault = 0x00308509;
constexpr uint32_t kCpHqdPqControlPm4 =
    (kCpHqdPqControlDefault & ~0x00003f00u) | (9u << 8) | (1u << 28) |
    (1u << 30) | (1u << 31);
constexpr uint32_t kCpHqdPqControlMes = 0xd8300909;
constexpr uint32_t kCpHqdDequeueDrainPipe = 0x1;
constexpr uint32_t kCpHqdDequeueResetWaves = 0x2;
constexpr uint32_t kCpMesCntlInvalidateIcache = 1u << 4;
constexpr uint32_t kCpMesCntlPipe0Reset = 1u << 16;
constexpr uint32_t kCpMesCntlPipe1Reset = 1u << 17;
constexpr uint32_t kCpMesCntlPipe0Active = 1u << 26;
constexpr uint32_t kCpMesCntlPipe1Active = 1u << 27;
constexpr uint32_t kCpMesCntlHalt = 1u << 30;
// RLC_CP_SCHEDULERS KIQ routing byte: (me=3<<5)|(pipe=1<<3)|(hqd=0)|enable(0x80).
constexpr uint32_t kRlcCpSchedulersEnable = 0x80u;
constexpr uint32_t kCpUnmappedDoorbellEnable = 1u << 0;
constexpr uint32_t kCpUnmappedDoorbellProcLsbMask = 0x00001f00u;
constexpr uint32_t kCpUnmappedDoorbellProcLsbShift = 8;
constexpr uint32_t kCpMesDoorbellOffsetMask = 0x0ffffffcu;
constexpr uint32_t kCpMesDoorbellEnable = 1u << 30;
constexpr uint32_t kCpMesDoorbellHit = 1u << 31;
constexpr uint32_t kCpHqdGfxControlDbUpdatedMsgEn = 1u << 15;

constexpr uint64_t kDirectComputeBaseOffset = 0x1900000;
constexpr uint64_t kDirectComputeStride = 0x40000;
constexpr uint64_t kDirectComputeMqdRelativeOffset = 0x00000;
constexpr uint64_t kDirectComputeRingRelativeOffset = 0x02000;
constexpr uint64_t kDirectComputeEopRelativeOffset = 0x10000;
constexpr uint64_t kDirectComputeRptrRelativeOffset = 0x20000;
constexpr uint64_t kDirectComputeWptrRelativeOffset = 0x21000;
constexpr uint32_t kMesRingSize = 0x8000;
constexpr uint64_t kMesSchedulerBaseOffset = 0x1800000;
constexpr uint64_t kMesKiqBaseOffset = 0x1840000;
[[maybe_unused]] constexpr uint64_t kLegacyKiqBaseOffset = 0x1880000;
constexpr uint64_t kMesSchedulerContextRelativeOffset = 0x22000;
constexpr uint64_t kMesQueryFenceRelativeOffset = 0x23000;
constexpr uint64_t kMesApiFenceRelativeOffset = 0x24000;
constexpr uint64_t kMesCleanerFenceRelativeOffset = 0x24020;

constexpr uint32_t kMesApiFrameDwords = 64;
constexpr uint32_t kMesApiTypeScheduler = 1;
constexpr uint32_t kMesOpcodeSetHwResources = 0;
constexpr uint32_t kMesOpcodeAddQueue = 2;
constexpr uint32_t kMesOpcodeRemoveQueue = 3;
constexpr uint32_t kMesOpcodeQuerySchedulerStatus = 11;
constexpr uint32_t kMesOpcodeSetHwResources1 = 19;
constexpr uint32_t kMesQueueTypeCompute = 1;
constexpr uint32_t kMesQueueTypeScheduler = 3;
constexpr uint32_t kMesSchedulerDoorbell = 0x00b << 1;
constexpr uint32_t kMesKiqDoorbell = 0x00c << 1;
constexpr uint32_t kMesKiqMe = 3;
constexpr uint32_t kMesKiqPipe = 1;
constexpr uint32_t kMesKiqHqd = 0;
[[maybe_unused]] constexpr uint32_t kLegacyKiqDoorbell = 0x000 << 1;
[[maybe_unused]] constexpr uint32_t kLegacyKiqMe = 1;
[[maybe_unused]] constexpr uint32_t kLegacyKiqPipe = 0;
[[maybe_unused]] constexpr uint32_t kLegacyKiqHqd = 0;
constexpr uint32_t kMesApiStatusSetHwResourcesDw = 50;
constexpr uint32_t kMesApiStatusSetHwResources1Dw = 2;
constexpr uint32_t kMesApiStatusAddQueueDw = 38;
constexpr uint32_t kMesApiStatusRemoveQueueDw = 6;
constexpr uint32_t kMesSetHwResourcesFlags = 0x00080447;
constexpr uint32_t kMesAddQueueMapLegacyKq = 1u << 13;
constexpr uint32_t kMesRemoveQueueUnmapLegacy = 1u << 3;
constexpr uint32_t kMesAggregatedDoorbellBase = 0x80;
constexpr uint32_t kPacketType3 = 3;
[[maybe_unused]] constexpr uint32_t kPacket3Nop = 0x10;
constexpr uint32_t kPacket3WriteData = 0x37;
constexpr uint32_t kPacket3MapQueues = 0xa2;

const char* TracePrefix(const DirectQueueOptions& options) {
  return options.trace_prefix ? options.trace_prefix : "ROCR lite direct queue";
}

// MES doorbell-dead workaround toggle: drive the MES KIQ / mapped compute HQD
// by MMIO-poking CP_HQD_PQ_WPTR instead of relying on the doorbell. Proven on
// Windows (ROCR_WINDOWS_MES_MMIO_WPTR); the generic ROCR_MES_MMIO_WPTR lets
// other transports (macOS) opt in to the same poke. Either flag being set (to a
// non-empty, non-"0" value) enables it. Default (neither set) = unchanged, so
// the proven Linux/macOS direct paths are byte-identical.
bool EnvFlagSet(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

bool MesMmioWptrPokeEnabled() {
  return EnvFlagSet("ROCR_WINDOWS_MES_MMIO_WPTR") ||
         EnvFlagSet("ROCR_MES_MMIO_WPTR");
}

// Windows WDDM/passthrough only: host-activate the MES scheduler ring's HQD
// (me=3,pipe=0) with the WPTR_POLL workaround, exactly like the KIQ. Without it
// the software MAP_SCHEDULER alone never brings pipe-0 online (rptr stays 0 and
// the scheduler SET_HW_RESOURCES times out -> MES map status=4096). Off by
// default so macOS/Linux (live doorbell) are unaffected.
bool MesActivateSchedulerHqdEnabled() {
  return EnvFlagSet("ROCR_WINDOWS_MES_ACTIVATE_SCHED_HQD");
}

// Windows-only: program the gfx12 SH_MEM private/scratch aperture (needed for
// architected flat scratch on a register-spilling kernel). Default ON wherever
// the MES mmio-wptr poke is on; override via ROCR_WINDOWS_SHMEM_APERTURE.
bool ShMemApertureEnabled() {
  const char* v = std::getenv("ROCR_WINDOWS_SHMEM_APERTURE");
  if (v == nullptr) v = std::getenv("ROCR_SHMEM_APERTURE");
  if (v != nullptr && v[0] != '\0') return v[0] != '0';
  return MesMmioWptrPokeEnabled();
}

// Windows MES-backed scratch (#57): skip the SetDirectQueueScratch REMOVE_QUEUE
// + ADD_QUEUE(map_legacy) remap. HW showed the remap freezes the already-working
// compute HQD (rptr stalls at the pre-scratch wptr, CP_PQ_WPTR_POLL_CNTL reset,
// no GPUVM fault) so the first register-spilling dispatch is never consumed.
// Scratch is already carried PER-DISPATCH by the AQL translator's SET_SH_REG of
// COMPUTE_DISPATCH_SCRATCH_BASE_LO/HI + COMPUTE_TMPRING_SIZE (amd_windows_aql_
// queue.cpp:763-767, same registers/values proven on the direct-HQD path in
// gpu_init.cpp), so the remap is not what programs scratch. The MQD scratch-
// field patch + SH_MEM re-assert + fault-status clear still run (harmless, and
// keep the MQD correct for any later state reload). Off by default so the
// existing remap path is byte-identical unless opted in.
bool SkipScratchRemapEnabled() {
  return EnvFlagSet("ROCR_WINDOWS_SKIP_SCRATCH_REMAP");
}

uint32_t MesHeader(uint32_t opcode) {
  return kMesApiTypeScheduler | (opcode << 4) | (kMesApiFrameDwords << 12);
}

uint32_t Packet3(uint32_t opcode, uint32_t count) {
  return (kPacketType3 << 30) | ((opcode & 0xffu) << 8) |
         ((count & 0x3fffu) << 16);
}

void PutU64(uint32_t* frame, uint32_t dword_offset, uint64_t value) {
  frame[dword_offset] = static_cast<uint32_t>(value);
  frame[dword_offset + 1] = static_cast<uint32_t>(value >> 32);
}

DirectQueueLayout BuildQueueLayoutAt(uint64_t framebuffer_base,
                                     uint64_t base_offset) {
  DirectQueueLayout layout;
  layout.base_offset = base_offset;
  layout.base_gpu = framebuffer_base + layout.base_offset;
  layout.mqd_offset = layout.base_offset + kDirectComputeMqdRelativeOffset;
  layout.ring_offset = layout.base_offset + kDirectComputeRingRelativeOffset;
  layout.eop_offset = layout.base_offset + kDirectComputeEopRelativeOffset;
  layout.rptr_offset = layout.base_offset + kDirectComputeRptrRelativeOffset;
  layout.wptr_offset = layout.base_offset + kDirectComputeWptrRelativeOffset;
  layout.mqd_gpu = layout.base_gpu + kDirectComputeMqdRelativeOffset;
  layout.ring_gpu = layout.base_gpu + kDirectComputeRingRelativeOffset;
  layout.eop_gpu = layout.base_gpu + kDirectComputeEopRelativeOffset;
  layout.rptr_gpu = layout.base_gpu + kDirectComputeRptrRelativeOffset;
  layout.wptr_gpu = layout.base_gpu + kDirectComputeWptrRelativeOffset;
  return layout;
}

DirectQueueLayout BuildQueueLayoutFromMemory(const DirectQueueMemory& memory) {
  DirectQueueLayout layout;
  layout.base_offset = 0;
  layout.base_gpu = memory.gpu_addr;
  layout.cpu_base = memory.cpu;
  layout.cpu_size = memory.size;
  layout.mqd_offset = kDirectComputeMqdRelativeOffset;
  layout.ring_offset = kDirectComputeRingRelativeOffset;
  layout.eop_offset = kDirectComputeEopRelativeOffset;
  layout.rptr_offset = kDirectComputeRptrRelativeOffset;
  layout.wptr_offset = kDirectComputeWptrRelativeOffset;
  layout.mqd_gpu = layout.base_gpu + kDirectComputeMqdRelativeOffset;
  layout.ring_gpu = layout.base_gpu + kDirectComputeRingRelativeOffset;
  layout.eop_gpu = layout.base_gpu + kDirectComputeEopRelativeOffset;
  layout.rptr_gpu = layout.base_gpu + kDirectComputeRptrRelativeOffset;
  layout.wptr_gpu = layout.base_gpu + kDirectComputeWptrRelativeOffset;
  return layout;
}

uint64_t LayoutRelativeOffset(const DirectQueueLayout& layout,
                              uint64_t offset) {
  return offset - layout.base_offset;
}

void* LayoutCpuPointer(const DirectQueuePlatform& platform,
                       const DirectQueueLayout& layout,
                       uint64_t offset,
                       uint64_t size = 1) {
  if (layout.cpu_base != nullptr) {
    if (offset < layout.base_offset) return nullptr;
    const uint64_t relative = LayoutRelativeOffset(layout, offset);
    if (relative > layout.cpu_size || size > layout.cpu_size - relative) {
      return nullptr;
    }
    return static_cast<char*>(layout.cpu_base) + relative;
  }
  return platform.GpuMemoryCpuPointer(offset);
}

hsa_status_t ZeroLayoutMemory(const DirectQueuePlatform& platform,
                              const DirectQueueLayout& layout,
                              uint64_t offset,
                              uint64_t size) {
  if (layout.cpu_base != nullptr) {
    void* cpu = LayoutCpuPointer(platform, layout, offset, size);
    if (cpu == nullptr) return HSA_STATUS_ERROR_INVALID_ALLOCATION;
    std::memset(cpu, 0, static_cast<size_t>(size));
    std::atomic_thread_fence(std::memory_order_release);
    return HSA_STATUS_SUCCESS;
  }
  return platform.ZeroGpuMemory(offset, size);
}

hsa_status_t WriteLayoutMemory32(const DirectQueuePlatform& platform,
                                 const DirectQueueLayout& layout,
                                 uint64_t offset,
                                 uint32_t value) {
  if (layout.cpu_base != nullptr) {
    void* cpu = LayoutCpuPointer(platform, layout, offset, sizeof(uint32_t));
    if (cpu == nullptr) return HSA_STATUS_ERROR_INVALID_ALLOCATION;
    *reinterpret_cast<volatile uint32_t*>(cpu) = value;
    return HSA_STATUS_SUCCESS;
  }
  return platform.WriteGpuMemory32(offset, value);
}

hsa_status_t PrepareQueueLayout(const DirectQueuePlatform& platform,
                                uint64_t framebuffer_base,
                                uint64_t base_offset,
                                DirectQueueLayout* layout,
                                DirectQueueMemory* memory) {
  if (layout == nullptr || memory == nullptr) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  *memory = {};
  const char* force_vram = std::getenv("ROCR_AMDGPU_LITE_FORCE_VRAM_QUEUE_MEMORY");
  if (platform.PreferAllocatedQueueMemory() &&
      (force_vram == nullptr || force_vram[0] == '\0' ||
       force_vram[0] == '0')) {
    hsa_status_t status =
        platform.AllocateQueueMemory(kDirectComputeStride, memory);
    if (status != HSA_STATUS_SUCCESS) return status;
    *layout = BuildQueueLayoutFromMemory(*memory);
    return HSA_STATUS_SUCCESS;
  }
  *layout = BuildQueueLayoutAt(framebuffer_base, base_offset);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t SelectHqd(const DirectQueuePlatform& platform, uint32_t me,
                       uint32_t pipe, uint32_t queue) {
  return platform.WriteMmio32(kGcBase1, regGRBM_GFX_CNTL,
                              ((pipe & 0x3u) << 0) | ((me & 0x3u) << 2) |
                                  ((0u & 0xFu) << 4) | ((queue & 0x7u) << 8));
}

hsa_status_t DeselectHqd(const DirectQueuePlatform& platform) {
  return platform.WriteMmio32(kGcBase1, regGRBM_GFX_CNTL, 0);
}

// Program the gfx12 per-VMID SH_MEM private (scratch) aperture for all 16
// VMIDs, mirroring the proven gpu_init.cpp cqInitGfxForCompute / ring_init.py:
// grbm-select each VMID (vmid in GRBM_GFX_CNTL[7:4]) then write SH_MEM_CONFIG +
// SH_MEM_BASES (base_idx 1). Absent this, a spilling kernel FLAT_SCRATCH -> VA0
// -> GCVM permission fault -> CP hang. Restores grbm to vmid 0 on exit.
hsa_status_t ProgramShMemAllVmids(const DirectQueuePlatform& platform) {
  for (uint32_t vmid = 0; vmid < 16; ++vmid) {
    hsa_status_t status =
        platform.WriteMmio32(kGcBase1, regGRBM_GFX_CNTL, (vmid & 0xFu) << 4);
    if (status != HSA_STATUS_SUCCESS) return status;
    status = platform.WriteMmio32(kGcBase1, regSH_MEM_CONFIG, kShMemConfigGfx12);
    if (status != HSA_STATUS_SUCCESS) return status;
    status = platform.WriteMmio32(kGcBase1, regSH_MEM_BASES, kShMemBasesGfx12);
    if (status != HSA_STATUS_SUCCESS) return status;
  }
  return platform.WriteMmio32(kGcBase1, regGRBM_GFX_CNTL, 0);
}

// Read the global GCVM L2 protection fault status + address (write-to-clear).
// Diagnoses the scratch CP stall: fault_va=0 => aperture still wrong; fault_va
// == scratch VA => backing not GPUVM-mapped; status=0 => not a VM fault.
void TraceScratchFault(const DirectQueuePlatform& platform, const char* tag,
                       const DirectQueueOptions& options) {
  uint32_t fs = 0, fa_lo = 0, fa_hi = 0;
  platform.ReadMmio32(kGcBase0, regGCVM_L2_PROTECTION_FAULT_STATUS, &fs);
  platform.ReadMmio32(kGcBase0, regGCVM_L2_PROTECTION_FAULT_ADDR_LO32, &fa_lo);
  platform.ReadMmio32(kGcBase0, regGCVM_L2_PROTECTION_FAULT_ADDR_HI32, &fa_hi);
  const uint64_t va = ((static_cast<uint64_t>(fa_hi) << 32) | fa_lo) << 12;
  std::fprintf(stderr,
               "%s scratch-fault-probe[%s] fault_status=0x%08x [walker=%u "
               "perm=0x%x] fault_va=0x%llx\n",
               TracePrefix(options), tag, fs,
               static_cast<unsigned>((fs >> 1) & 0x7u),
               static_cast<unsigned>((fs >> 4) & 0xFu),
               static_cast<unsigned long long>(va));
}

// MAX-INFO stall probe for the MES-backed spilling-dispatch hang (#57). Called
// with the compute HQD already GRBM-selected (me=1,pipe,hqd). Answers: is the CP
// stuck FETCHING (no wave) or is a spilling WAVE hung (no GPUVM fault)? Reads
// only verified offsets/bases:
//   (a) ring CONTENTS at the frozen rptr  -- host memory, NO MMIO
//   (b) GRBM_STATUS (abs DWORD 0x0050, base 0) -- shader/CP busy vs idle
//   (c) report-page rptr (host) vs CP_HQD_PQ_RPTR register (MMIO, selected HQD)
[[maybe_unused]] void TraceStallMaxInfo(const DirectQueuePlatform& platform,
                       const DirectQueueState& queue, const char* tag,
                       const DirectQueueOptions& options) {
  // (c) register rptr under the already-selected compute HQD.
  uint32_t hw_rptr = 0;
  platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_RPTR, &hw_rptr);
  const uint64_t report_rptr = queue.rptr_cpu != nullptr ? *queue.rptr_cpu : 0;

  // (b) GRBM_STATUS: absolute mmGRBM_STATUS DWORD 0x0050 (base 0), the only
  // in-repo-verified shader-busy register (wddm_lite_test.cpp). Raw word is
  // printed; the bit decode is advisory only.
  uint32_t grbm_status = 0;
  platform.ReadMmio32(/*base=*/0, regGRBM_STATUS_ABS, &grbm_status);
  const unsigned gui_active = (grbm_status >> 31) & 0x1u;  // any GUI work
  const unsigned cp_busy = (grbm_status >> 29) & 0x1u;     // CP busy
  const unsigned spi_busy = (grbm_status >> 22) & 0x1u;    // SPI busy

  // (a) ring dwords at the frozen read pointer. rptr is a DWORD index into the
  // ring; use the register value (register wins if the report page is stale).
  // NO MMIO -- pure host reads of the mapped ring.
  const uint64_t ring_dw =
      queue.ring_size_bytes != 0 ? queue.ring_size_bytes / sizeof(uint32_t) : 0;
  uint32_t r[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  if (queue.ring_cpu != nullptr && ring_dw != 0) {
    const uint64_t b = hw_rptr % ring_dw;
    for (uint64_t i = 0; i < 8; ++i) r[i] = queue.ring_cpu[(b + i) % ring_dw];
  }

  std::fprintf(stderr,
               "%s stall-maxinfo[%s] hw_rptr=0x%x report_rptr=%llu "
               "rptr_match=%d grbm_status=0x%08x [gui=%u cp=%u spi=%u] "
               "ring[rptr..+8]=0x%08x,0x%08x,0x%08x,0x%08x,0x%08x,0x%08x,"
               "0x%08x,0x%08x\n",
               TracePrefix(options), tag, hw_rptr,
               static_cast<unsigned long long>(report_rptr),
               (report_rptr == hw_rptr) ? 1 : 0, grbm_status, gui_active,
               cp_busy, spi_busy, r[0], r[1], r[2], r[3], r[4], r[5], r[6],
               r[7]);
}

// SETTLED stall+fault probe for the MES-backed spilling GEMM dispatch (#57).
// Called with the compute HQD already GRBM-selected (me=1,pipe,hqd). The old
// two-probe sequence read the fault regs after a fixed 3ms SleepUs, which on
// the LAST (scratch) submit fires while the CP is still mid-consume (rptr=966
// of the 73-dword block) -- BEFORE the DISPATCH_DIRECT launches the spilling
// wave. Here we first POLL CP_HQD_PQ_RPTR until the CP reaches the target
// wptr (drained) or the read pointer stops advancing for kSettleReads reads
// (stalled), capped at ~500ms, THEN read -- at that settled point -- the GCVM
// fault status+VA, GRBM_STATUS, and ring[rptr..+8]. So the scratch dispatch's
// own probe captures the 1019 hang (CP parked at DISPATCH_DIRECT / the post-
// dispatch fence). Reuses only in-repo-verified regs/bases; the HQD must be
// selected on entry (caller SelectHqd's it; DeselectHqd stays with the caller).
void TraceSettledStall(const DirectQueuePlatform& platform,
                       const DirectQueueState& queue, uint64_t target_wptr,
                       const char* tag, const DirectQueueOptions& options) {
  // Poll CP_HQD_PQ_RPTR until drained (rptr == target) or settled (rptr
  // unchanged for kSettleReads consecutive reads), whichever comes first.
  // ~500ms cap: kMaxReads iterations * kPollUs between reads.
  constexpr uint32_t kPollUs = 2000;    // 2ms per poll step
  constexpr uint32_t kMaxReads = 250;   // 250 * 2ms = ~500ms cap
  constexpr uint32_t kSettleReads = 8;  // frozen for 8 reads (~16ms) => stalled
  const uint64_t ring_dw =
      queue.ring_size_bytes != 0 ? queue.ring_size_bytes / sizeof(uint32_t) : 0;
  const uint32_t target_rptr =
      ring_dw != 0 ? static_cast<uint32_t>(target_wptr % ring_dw) : 0;
  uint32_t hw_rptr = 0;
  platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_RPTR, &hw_rptr);
  uint32_t last_rptr = hw_rptr;
  uint32_t unchanged = 0;
  uint32_t reads = 0;
  bool drained = false;
  bool settled = false;
  for (reads = 0; reads < kMaxReads; ++reads) {
    if (ring_dw != 0 && hw_rptr == target_rptr) {
      drained = true;
      break;
    }
    if (hw_rptr == last_rptr) {
      if (++unchanged >= kSettleReads) {
        settled = true;
        break;
      }
    } else {
      unchanged = 0;
      last_rptr = hw_rptr;
    }
    platform.SleepUs(kPollUs);
    platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_RPTR, &hw_rptr);
  }

  // Settled point reached: read the GCVM fault regs (merged from the old
  // TraceScratchFault) -- status=0 => not a VM fault (fence hang), va ==
  // scratch VA => backing not GPUVM-mapped.
  uint32_t fs = 0, fa_lo = 0, fa_hi = 0;
  platform.ReadMmio32(kGcBase0, regGCVM_L2_PROTECTION_FAULT_STATUS, &fs);
  platform.ReadMmio32(kGcBase0, regGCVM_L2_PROTECTION_FAULT_ADDR_LO32, &fa_lo);
  platform.ReadMmio32(kGcBase0, regGCVM_L2_PROTECTION_FAULT_ADDR_HI32, &fa_hi);
  const uint64_t fault_va = ((static_cast<uint64_t>(fa_hi) << 32) | fa_lo) << 12;

  // GRBM_STATUS: absolute mmGRBM_STATUS DWORD 0x0050 (base 0). grbm busy +
  // rptr short of target => spilling wave launched and hung (the 1019 case).
  uint32_t grbm_status = 0;
  platform.ReadMmio32(/*base=*/0, regGRBM_STATUS_ABS, &grbm_status);
  const unsigned gui_active = (grbm_status >> 31) & 0x1u;
  const unsigned cp_busy = (grbm_status >> 29) & 0x1u;
  const unsigned spi_busy = (grbm_status >> 22) & 0x1u;

  // ring dwords at the frozen read pointer (host reads, NO MMIO); the packet
  // the CP is parked on -- expect the DISPATCH_DIRECT / fence tail.
  uint32_t r[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  if (queue.ring_cpu != nullptr && ring_dw != 0) {
    const uint64_t b = hw_rptr % ring_dw;
    for (uint64_t i = 0; i < 8; ++i) r[i] = queue.ring_cpu[(b + i) % ring_dw];
  }

  const uint64_t report_rptr = queue.rptr_cpu != nullptr ? *queue.rptr_cpu : 0;
  std::fprintf(stderr,
               "%s settled-stall[%s] state=%s reads=%u hw_rptr=0x%x "
               "target_rptr=0x%x report_rptr=%llu grbm_status=0x%08x "
               "[gui=%u cp=%u spi=%u] fault_status=0x%08x [walker=%u "
               "perm=0x%x] fault_va=0x%llx "
               "ring[rptr..+8]=0x%08x,0x%08x,0x%08x,0x%08x,0x%08x,0x%08x,"
               "0x%08x,0x%08x\n",
               TracePrefix(options), tag,
               drained ? "drained" : (settled ? "stalled" : "timeout"),
               reads, hw_rptr, target_rptr,
               static_cast<unsigned long long>(report_rptr), grbm_status,
               gui_active, cp_busy, spi_busy, fs,
               static_cast<unsigned>((fs >> 1) & 0x7u),
               static_cast<unsigned>((fs >> 4) & 0xFu),
               static_cast<unsigned long long>(fault_va), r[0], r[1], r[2],
               r[3], r[4], r[5], r[6], r[7]);
}

hsa_status_t WaitForDirectHqdIdle(const DirectQueuePlatform& platform,
                                  uint32_t pipe, uint32_t queue,
                                  const char* phase,
                                  const DirectQueueOptions& options) {
  constexpr uint32_t kStepUs = 1000;
  const uint32_t timeout_us = options.dequeue_settle_us != 0 ? options.dequeue_settle_us : 100000;
  const uint32_t max_samples = std::max<uint32_t>(1, timeout_us / kStepUs);
  uint32_t active = 0;
  uint32_t pq_control = 0;
  uint32_t doorbell_control = 0;
  uint32_t dequeue_status = 0;
  for (uint32_t i = 0; i < max_samples; ++i) {
    hsa_status_t status = platform.ReadMmio32(kGcBase0, regCP_HQD_ACTIVE, &active);
    if (status != HSA_STATUS_SUCCESS) return status;
    status = platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_CONTROL, &pq_control);
    if (status != HSA_STATUS_SUCCESS) return status;
    status = platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_DOORBELL_CONTROL, &doorbell_control);
    if (status != HSA_STATUS_SUCCESS) return status;
    status = platform.ReadMmio32(kGcBase0, regCP_HQD_DEQUEUE_STATUS, &dequeue_status);
    if (status != HSA_STATUS_SUCCESS) return status;
    if (active == 0 && (doorbell_control & 0x40000000u) == 0) {
      if (options.trace && i > 0) {
        std::fprintf(stderr,
                     "%s hqd idle phase=%s pipe=%u hqd=%u samples=%u "
                     "active=0x%x pq_control=0x%x doorbell_control=0x%x "
                     "dequeue_status=0x%x\n",
                     TracePrefix(options), phase ? phase : "unknown", pipe, queue, i + 1,
                     active, pq_control, doorbell_control, dequeue_status);
      }
      return HSA_STATUS_SUCCESS;
    }
    platform.SleepUs(kStepUs);
  }
  if (options.trace) {
    std::fprintf(stderr,
                 "%s hqd idle timeout phase=%s pipe=%u hqd=%u active=0x%x "
                 "pq_control=0x%x doorbell_control=0x%x dequeue_status=0x%x "
                 "timeout_us=%u\n",
                 TracePrefix(options), phase ? phase : "unknown", pipe, queue, active,
                 pq_control, doorbell_control, dequeue_status, timeout_us);
  }
  return HSA_STATUS_ERROR;
}

hsa_status_t ReclaimActiveHqd(const DirectQueuePlatform& platform,
                              const DirectQueueState& queue,
                              const DirectQueueOptions& options,
                              const char* phase) {
  const uint32_t pipe = DirectQueuePipe(queue.queue_index);
  const uint32_t hqd_queue = DirectQueueHqd(queue.queue_index);
  uint32_t active = 0;
  hsa_status_t status = platform.ReadMmio32(kGcBase0, regCP_HQD_ACTIVE, &active);
  if (status != HSA_STATUS_SUCCESS) return status;
  if (active == 0) return HSA_STATUS_SUCCESS;

  if (options.use_firmware_dequeue) {
    if (options.trace) {
      std::fprintf(stderr,
                   "%s reclaim active HQD with dequeue index=%u pipe=%u "
                   "hqd=%u active=0x%x\n",
                   TracePrefix(options), queue.queue_index, pipe, hqd_queue, active);
    }
    platform.WriteMmio32(kGcBase0, regCP_HQD_DEQUEUE_REQUEST,
                         kCpHqdDequeueResetWaves);
    platform.WriteMmio32(kGcBase0, regSPI_COMPUTE_QUEUE_RESET, 1);
    uint32_t now_active = active;
    for (uint32_t i = 0; i < 1000; ++i) {
      platform.ReadMmio32(kGcBase0, regCP_HQD_ACTIVE, &now_active);
      if (now_active == 0) break;
      platform.SleepUs(1000);
    }
    platform.WriteMmio32(kGcBase0, regCP_HQD_DEQUEUE_REQUEST, 0);
    status = WaitForDirectHqdIdle(platform, pipe, hqd_queue, phase, options);
    if (status != HSA_STATUS_SUCCESS) return status;
    if (options.trace) {
      std::fprintf(stderr,
                   "%s dequeue reclaim complete index=%u pipe=%u hqd=%u "
                   "active=0x%x\n",
                   TracePrefix(options), queue.queue_index, pipe, hqd_queue, now_active);
    }
  } else {
    if (options.trace) {
      std::fprintf(stderr,
                   "%s reclaim active HQD without dequeue index=%u pipe=%u "
                   "hqd=%u active=0x%x\n",
                   TracePrefix(options), queue.queue_index, pipe, hqd_queue, active);
    }
    platform.WriteMmio32(kGcBase0, regCP_HQD_ACTIVE, 0);
    uint32_t now_active = active;
    for (uint32_t i = 0; i < 1000; ++i) {
      platform.ReadMmio32(kGcBase0, regCP_HQD_ACTIVE, &now_active);
      if (now_active == 0) break;
      platform.SleepUs(1000);
    }
    if (options.trace) {
      std::fprintf(stderr,
                   "%s direct-disable reclaim complete index=%u pipe=%u hqd=%u "
                   "active=0x%x\n",
                   TracePrefix(options), queue.queue_index, pipe, hqd_queue, now_active);
    }
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t ResetSelectedHqdForProgramming(const DirectQueuePlatform& platform,
                                            uint32_t pipe,
                                            uint32_t hqd_queue,
                                            const DirectQueueOptions& options,
                                            const char* label) {
  constexpr uint32_t kStepUs = 1;
  const uint32_t timeout_us =
      options.dequeue_settle_us != 0 ? options.dequeue_settle_us : 100000;
  uint32_t active = 0;
  hsa_status_t status = platform.ReadMmio32(kGcBase0, regCP_HQD_ACTIVE, &active);
  if (status != HSA_STATUS_SUCCESS) return status;

  if ((active & 1u) != 0) {
    if (options.trace) {
      std::fprintf(stderr,
                   "%s %s reset active HQD before program pipe=%u hqd=%u "
                   "active=0x%x\n",
                   TracePrefix(options), label ? label : "HQD", pipe,
                   hqd_queue, active);
    }
    status = platform.WriteMmio32(kGcBase0, regCP_HQD_DEQUEUE_REQUEST,
                                  kCpHqdDequeueDrainPipe);
    if (status != HSA_STATUS_SUCCESS) return status;

    for (uint32_t elapsed = 0; elapsed < timeout_us; elapsed += kStepUs) {
      status = platform.ReadMmio32(kGcBase0, regCP_HQD_ACTIVE, &active);
      if (status != HSA_STATUS_SUCCESS) return status;
      if ((active & 1u) == 0) break;
      platform.SleepUs(kStepUs);
    }
    platform.WriteMmio32(kGcBase0, regCP_HQD_DEQUEUE_REQUEST, 0);
    if ((active & 1u) != 0) {
      if (options.trace) {
        std::fprintf(stderr,
                     "%s %s reset dequeue timeout; clearing selected HQD "
                     "directly pipe=%u hqd=%u active=0x%x timeout_us=%u\n",
                     TracePrefix(options), label ? label : "HQD", pipe,
                     hqd_queue, active, timeout_us);
      }
    }
  } else {
    status = platform.WriteMmio32(kGcBase0, regCP_HQD_DEQUEUE_REQUEST, 0);
    if (status != HSA_STATUS_SUCCESS) return status;
  }

  uint32_t doorbell_ctl = 0;
  status = platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_DOORBELL_CONTROL,
                               &doorbell_ctl);
  if (status != HSA_STATUS_SUCCESS) return status;
  doorbell_ctl &= ~kCpMesDoorbellEnable;
  doorbell_ctl |= kCpMesDoorbellHit;
  status = platform.WriteMmio32(kGcBase0, regCP_HQD_PQ_DOORBELL_CONTROL,
                                doorbell_ctl);
  if (status != HSA_STATUS_SUCCESS) return status;
  status = platform.WriteMmio32(kGcBase0, regCP_HQD_PQ_DOORBELL_CONTROL, 0);
  if (status != HSA_STATUS_SUCCESS) return status;

  if (platform.WriteMmio32(kGcBase0, regCP_HQD_ACTIVE, 0) !=
          HSA_STATUS_SUCCESS ||
      platform.WriteMmio32(kGcBase0, regCP_PQ_WPTR_POLL_CNTL, 0) !=
          HSA_STATUS_SUCCESS ||
      platform.WriteMmio32(kGcBase0, regCP_HQD_PQ_WPTR_LO, 0) !=
          HSA_STATUS_SUCCESS ||
      platform.WriteMmio32(kGcBase0, regCP_HQD_PQ_WPTR_HI, 0) !=
          HSA_STATUS_SUCCESS ||
      platform.WriteMmio32(kGcBase0, regCP_HQD_PQ_RPTR, 0) !=
          HSA_STATUS_SUCCESS) {
    return HSA_STATUS_ERROR;
  }

  if (options.trace_verbose) {
    uint32_t rptr = 0;
    uint32_t wptr = 0;
    uint32_t wptr_hi = 0;
    uint32_t post_doorbell_ctl = 0;
    platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_RPTR, &rptr);
    platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_WPTR_LO, &wptr);
    platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_WPTR_HI, &wptr_hi);
    platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_DOORBELL_CONTROL,
                        &post_doorbell_ctl);
    std::fprintf(stderr,
                 "%s %s reset complete pipe=%u hqd=%u rptr=0x%x "
                 "wptr=0x%08x:%08x doorbell_ctl=0x%08x\n",
                 TracePrefix(options), label ? label : "HQD", pipe, hqd_queue,
                 rptr, wptr_hi, wptr, post_doorbell_ctl);
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t ZeroQueueMemory(const DirectQueuePlatform& platform,
                             const DirectQueueLayout& layout,
                             uint32_t ring_size = kDirectComputeRingSize) {
  hsa_status_t status =
      ZeroLayoutMemory(platform, layout, layout.mqd_offset, kMqdSize);
  if (status == HSA_STATUS_SUCCESS) {
    status = ZeroLayoutMemory(platform, layout, layout.ring_offset, ring_size);
  }
  if (status == HSA_STATUS_SUCCESS) {
    status =
        ZeroLayoutMemory(platform, layout, layout.eop_offset, kDirectComputeEopSize);
  }
  if (status == HSA_STATUS_SUCCESS) {
    status = ZeroLayoutMemory(platform, layout, layout.rptr_offset, 0x20);
  }
  if (status == HSA_STATUS_SUCCESS) {
    status = ZeroLayoutMemory(platform, layout, layout.wptr_offset, 0x20);
  }
  if (status != HSA_STATUS_SUCCESS) return status;
  std::atomic_thread_fence(std::memory_order_release);
  return platform.FlushHdp();
}

hsa_status_t WriteMqdImage(const DirectQueuePlatform& platform,
                           const DirectQueueLayout& layout,
                           const DirectQueueMqd& mqd) {
  for (size_t i = 0; i < mqd.size(); ++i) {
    hsa_status_t status =
        WriteLayoutMemory32(platform, layout, layout.mqd_offset + i * 4, mqd[i]);
    if (status != HSA_STATUS_SUCCESS) return status;
  }
  std::atomic_thread_fence(std::memory_order_seq_cst);
  return platform.FlushHdp();
}

[[maybe_unused]] hsa_status_t ProgramHqdRegisters(
    const DirectQueuePlatform& platform,
    uint32_t me,
    uint32_t pipe,
    uint32_t hqd_queue,
    const DirectQueueMqd& mqd,
    const DirectQueueOptions& options,
    const char* label) {
  hsa_status_t status = SelectHqd(platform, me, pipe, hqd_queue);
  if (status != HSA_STATUS_SUCCESS) return status;

  auto write = [&](uint32_t base, uint32_t reg, uint32_t value) {
    status = platform.WriteMmio32(base, reg, value);
    return status == HSA_STATUS_SUCCESS;
  };
  auto read = [&](uint32_t base, uint32_t reg, uint32_t* value) {
    status = platform.ReadMmio32(base, reg, value);
    return status == HSA_STATUS_SUCCESS;
  };

  uint32_t vmid = 0;
  uint32_t doorbell_ctl = 0;
  if (!write(kGcBase0, regCP_HQD_ACTIVE, 0) ||
      !write(kGcBase0, regCP_PQ_WPTR_POLL_CNTL, 0) ||
      !write(kGcBase0, regCP_HQD_PQ_RPTR, 0) ||
      !write(kGcBase0, regCP_HQD_PQ_WPTR_LO, 0) ||
      !write(kGcBase0, regCP_HQD_PQ_WPTR_HI, 0) ||
      !read(kGcBase0, regCP_HQD_VMID, &vmid) ||
      !write(kGcBase0, regCP_HQD_VMID, vmid & ~0xFu) ||
      !read(kGcBase0, regCP_HQD_PQ_DOORBELL_CONTROL, &doorbell_ctl) ||
      !write(kGcBase0, regCP_HQD_PQ_DOORBELL_CONTROL,
             doorbell_ctl & ~0x40000000u) ||
      !write(kGcBase0, regCP_MQD_BASE_ADDR, mqd[0x80]) ||
      !write(kGcBase0, regCP_MQD_BASE_ADDR_HI, mqd[0x81]) ||
      !write(kGcBase0, regCP_MQD_CONTROL, mqd[0xA2]) ||
      !write(kGcBase0, regCP_HQD_EOP_BASE_ADDR, mqd[0xA5]) ||
      !write(kGcBase0, regCP_HQD_EOP_BASE_ADDR_HI, mqd[0xA6]) ||
      !write(kGcBase0, regCP_HQD_EOP_CONTROL, mqd[0xA7]) ||
      !write(kGcBase0, regCP_HQD_PQ_BASE, mqd[0x88]) ||
      !write(kGcBase0, regCP_HQD_PQ_BASE_HI, mqd[0x89]) ||
      !write(kGcBase0, regCP_HQD_PQ_RPTR_REPORT_ADDR, mqd[0x8B]) ||
      !write(kGcBase0, regCP_HQD_PQ_RPTR_REPORT_ADDR_HI, mqd[0x8C]) ||
      !write(kGcBase0, regCP_HQD_PQ_CONTROL, mqd[0x91]) ||
      !write(kGcBase0, regCP_HQD_PQ_WPTR_POLL_ADDR, mqd[0x8D]) ||
      !write(kGcBase0, regCP_HQD_PQ_WPTR_POLL_ADDR_HI, mqd[0x8E]) ||
      !write(kGcBase0, regCP_HQD_PQ_DOORBELL_CONTROL, mqd[0x8F]) ||
      !write(kGcBase0, regCP_HQD_PERSISTENT_STATE, mqd[0x84]) ||
      !write(kGcBase0, regCP_HQD_GFX_CONTROL,
             kCpHqdGfxControlDbUpdatedMsgEn) ||
      !write(kGcBase0, regCP_HQD_ACTIVE, 1)) {
    DeselectHqd(platform);
    return status;
  }
  uint32_t pq_status = 0;
  if (!read(kGcBase0, regCP_PQ_STATUS, &pq_status) ||
      !write(kGcBase0, regCP_PQ_STATUS, pq_status | (1u << 1))) {
    DeselectHqd(platform);
    return status;
  }

  platform.SleepUs(options.activate_sleep_us);
  uint32_t active = 0;
  status = platform.ReadMmio32(kGcBase0, regCP_HQD_ACTIVE, &active);
  DeselectHqd(platform);
  if (status != HSA_STATUS_SUCCESS) return status;
  if (active == 0) {
    if (options.trace) {
      std::fprintf(stderr,
                   "%s %s HQD activation failed me=%u pipe=%u hqd=%u\n",
                   TracePrefix(options), label ? label : "MES", me, pipe,
                   hqd_queue);
    }
    return HSA_STATUS_ERROR;
  }
  if (options.trace) {
    std::fprintf(stderr, "%s %s HQD active me=%u pipe=%u hqd=%u\n",
                 TracePrefix(options), label ? label : "MES", me, pipe,
                 hqd_queue);
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t ProgramMesQueueRegisters(const DirectQueuePlatform& platform,
                                      uint32_t pipe,
                                      const DirectQueueMqd& mqd,
                                      const DirectQueueOptions& options,
                                      const char* label,
                                      bool set_active) {
  hsa_status_t status = SelectHqd(platform, kMesKiqMe, pipe, kMesKiqHqd);
  if (status != HSA_STATUS_SUCCESS) return status;

  auto write = [&](uint32_t base, uint32_t reg, uint32_t value) {
    status = platform.WriteMmio32(base, reg, value);
    return status == HSA_STATUS_SUCCESS;
  };
  auto read = [&](uint32_t base, uint32_t reg, uint32_t* value) {
    status = platform.ReadMmio32(base, reg, value);
    return status == HSA_STATUS_SUCCESS;
  };

  status = ResetSelectedHqdForProgramming(platform, pipe, kMesKiqHqd, options,
                                          label ? label : "MES KIQ");
  if (status != HSA_STATUS_SUCCESS) {
    DeselectHqd(platform);
    return status;
  }

  uint32_t vmid = 0;
  uint32_t doorbell_ctl = 0;
  // Windows doorbell-dead workaround (ROCR_WINDOWS_MES_WPTR_POLL):
  // On the Windows WDDM/passthrough setup, host doorbell writes do NOT
  // reach the GPU (proven by dbprobe), so the MES KIQ -- which relies
  // solely on the doorbell -- never services SET_HW_RESOURCES. The WORKING
  // direct compute queue advances via WPTR_POLL: CreateDirectQueue never
  // touches CP_PQ_WPTR_POLL_CNTL, so it keeps the firmware/RLC default
  // (dbprobe read back 0x1 = enabled). When the env flag is set we mirror
  // that here for the KIQ HQD: write the enabled value instead of 0 so the
  // MES pipe's CP polls the KIQ ring's in-memory wptr
  // (CP_HQD_PQ_WPTR_POLL_ADDR/_HI = mqd[0x8D/0x8E] = layout.wptr_gpu, set
  // below) without a doorbell event. Default (flag unset) = 0 = unchanged,
  // so macOS/Linux behavior is identical.
  uint32_t mes_wptr_poll_cntl = 0;
  {
    const char* poll_env = std::getenv("ROCR_WINDOWS_MES_WPTR_POLL");
    if (poll_env != nullptr && poll_env[0] != '\0' &&
        std::strcmp(poll_env, "0") != 0) {
      // 0x1 = the enabled value the working direct queue runs with (the
      // firmware default CreateDirectQueue leaves untouched; dbprobe saw 0x1).
      mes_wptr_poll_cntl = 0x1u;
      if (options.trace) {
        std::fprintf(stderr,
                     "%s %s WPTR_POLL workaround enabled: "
                     "CP_PQ_WPTR_POLL_CNTL=0x%x\n",
                     TracePrefix(options), label ? label : "MES KIQ",
                     mes_wptr_poll_cntl);
      }
    }
  }
  if (!write(kGcBase0, regCP_HQD_ACTIVE, 0) ||
      !write(kGcBase0, regCP_PQ_WPTR_POLL_CNTL, mes_wptr_poll_cntl) ||
      !write(kGcBase0, regCP_HQD_PQ_RPTR, 0) ||
      !write(kGcBase0, regCP_HQD_PQ_WPTR_LO, 0) ||
      !write(kGcBase0, regCP_HQD_PQ_WPTR_HI, 0) ||
      !read(kGcBase0, regCP_HQD_VMID, &vmid) ||
      !write(kGcBase0, regCP_HQD_VMID, vmid & ~0xFu) ||
      !read(kGcBase0, regCP_HQD_PQ_DOORBELL_CONTROL, &doorbell_ctl) ||
      !write(kGcBase0, regCP_HQD_PQ_DOORBELL_CONTROL,
             doorbell_ctl & ~0x40000000u) ||
      !write(kGcBase0, regCP_MQD_BASE_ADDR, mqd[0x80]) ||
      !write(kGcBase0, regCP_MQD_BASE_ADDR_HI, mqd[0x81]) ||
      !write(kGcBase0, regCP_MQD_CONTROL, mqd[0xA2]) ||
      // gfx12 requires a valid EOP window for the KIQ/MES HQD to latch ACTIVE.
      // The compute HQD path (ProgramHqdRegisters) writes these; the MES path
      // previously omitted them (+ MQD_CONTROL=0), so the KIQ never activated.
      !write(kGcBase0, regCP_HQD_EOP_BASE_ADDR, mqd[0xA5]) ||
      !write(kGcBase0, regCP_HQD_EOP_BASE_ADDR_HI, mqd[0xA6]) ||
      !write(kGcBase0, regCP_HQD_EOP_CONTROL, mqd[0xA7]) ||
      !write(kGcBase0, regCP_HQD_PQ_BASE, mqd[0x88]) ||
      !write(kGcBase0, regCP_HQD_PQ_BASE_HI, mqd[0x89]) ||
      !write(kGcBase0, regCP_HQD_PQ_RPTR_REPORT_ADDR, mqd[0x8B]) ||
      !write(kGcBase0, regCP_HQD_PQ_RPTR_REPORT_ADDR_HI, mqd[0x8C]) ||
      !write(kGcBase0, regCP_HQD_PQ_CONTROL, mqd[0x91]) ||
      !write(kGcBase0, regCP_HQD_PQ_WPTR_POLL_ADDR, mqd[0x8D]) ||
      !write(kGcBase0, regCP_HQD_PQ_WPTR_POLL_ADDR_HI, mqd[0x8E]) ||
      !write(kGcBase0, regCP_HQD_PQ_DOORBELL_CONTROL, mqd[0x8F]) ||
      !write(kGcBase0, regCP_HQD_PERSISTENT_STATE, mqd[0x84]) ||
      !write(kGcBase0, regCP_HQD_GFX_CONTROL, kCpHqdGfxControlDbUpdatedMsgEn) ||
      (set_active && !write(kGcBase0, regCP_HQD_ACTIVE, 1))) {
    DeselectHqd(platform);
    return status;
  }
  uint32_t pq_status = 0;
  if (!read(kGcBase0, regCP_PQ_STATUS, &pq_status) ||
      !write(kGcBase0, regCP_PQ_STATUS, pq_status | (1u << 1))) {
    DeselectHqd(platform);
    return status;
  }

  platform.SleepUs(options.activate_sleep_us);
  uint32_t active = 0;
  status = platform.ReadMmio32(kGcBase0, regCP_HQD_ACTIVE, &active);
  DeselectHqd(platform);
  if (status != HSA_STATUS_SUCCESS) return status;
  if (active == 0 && set_active) {
    if (options.trace) {
      std::fprintf(stderr,
                   "%s %s activation failed me=%u pipe=%u hqd=%u\n",
                   TracePrefix(options), label ? label : "MES KIQ", kMesKiqMe,
                   pipe, kMesKiqHqd);
    }
    return HSA_STATUS_ERROR;
  }
  if (options.trace) {
    std::fprintf(stderr, "%s %s active me=%u pipe=%u hqd=%u\n",
                 TracePrefix(options), label ? label : "MES KIQ", kMesKiqMe,
                 pipe, kMesKiqHqd);
  }
  if (options.trace_verbose) {
    hsa_status_t select_status =
        SelectHqd(platform, kMesKiqMe, pipe, kMesKiqHqd);
    if (select_status == HSA_STATUS_SUCCESS) {
      uint32_t pq_control = 0;
      uint32_t rptr = 0;
      uint32_t wptr = 0;
      uint32_t wptr_hi = 0;
      uint32_t doorbell_control = 0;
      platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_CONTROL, &pq_control);
      platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_RPTR, &rptr);
      platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_WPTR_LO, &wptr);
      platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_WPTR_HI, &wptr_hi);
      platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_DOORBELL_CONTROL,
                          &doorbell_control);
      DeselectHqd(platform);
      std::fprintf(stderr,
                   "%s %s post-program pipe=%u hqd=%u pq_control=0x%08x "
                   "rptr=0x%x wptr=0x%08x:%08x doorbell_ctl=0x%08x\n",
                   TracePrefix(options), label ? label : "MES KIQ", pipe,
                   kMesKiqHqd, pq_control, rptr, wptr_hi, wptr,
                   doorbell_control);
    }
  }
  return HSA_STATUS_SUCCESS;
}

DirectQueueState MakeRingState(const DirectQueuePlatform& platform,
                               const DirectQueueLayout& layout,
                               uint32_t doorbell_index,
                               uint32_t ring_size = kDirectComputeRingSize,
                               uint32_t align_mask = 0,
                               uint32_t nop_packet = 0) {
  DirectQueueState ring{};
  ring.doorbell_index = doorbell_index;
  ring.ring_size_bytes = ring_size;
  ring.ring_align_mask = align_mask;
  ring.ring_nop = nop_packet;
  ring.ring_gpu = layout.ring_gpu;
  ring.ring_cpu = static_cast<volatile uint32_t*>(
      LayoutCpuPointer(platform, layout, layout.ring_offset,
                       ring_size));
  ring.rptr_cpu = static_cast<volatile uint64_t*>(
      LayoutCpuPointer(platform, layout, layout.rptr_offset,
                       sizeof(uint64_t)));
  ring.wptr_cpu = static_cast<volatile uint64_t*>(
      LayoutCpuPointer(platform, layout, layout.wptr_offset,
                       sizeof(uint64_t)));
  ring.doorbell_cpu = platform.DoorbellCpuPointer(doorbell_index);
  return ring;
}

hsa_status_t SubmitRingPm4(const DirectQueuePlatform& platform,
                           DirectQueueState& ring, const uint32_t* pm4,
                           size_t dword_count) {
  if (ring.ring_cpu == nullptr || ring.wptr_cpu == nullptr ||
      ring.doorbell_cpu == nullptr || pm4 == nullptr || dword_count == 0) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  const uint64_t ring_dw = ring.ring_size_bytes / sizeof(uint32_t);
  // A PM4 packet must not straddle the ring-end boundary: the CP reads each
  // packet as a contiguous run, so if this packet would wrap past the end,
  // NOP-fill the tail and place the packet at offset 0. Without this the first
  // ring wrap (~ring_size/packet_size submits) writes a split packet and the CP
  // faults (HSA 0x1000) — which capped sustained dispatch at ~13 on the 4 KiB
  // ring (the macOS multi-op limit).
  uint64_t start = ring.wptr % ring_dw;
  if (start + dword_count > ring_dw) {
    const uint64_t tail = ring_dw - start;
    for (uint64_t i = 0; i < tail; ++i) ring.ring_cpu[start + i] = ring.ring_nop;
    ring.wptr += tail;  // advance past the NOP tail so wptr % ring_dw == 0
    start = 0;
  }
  for (size_t i = 0; i < dword_count; ++i) {
    ring.ring_cpu[(start + i) % ring_dw] = pm4[i];
  }
  size_t pad_count = 0;
  if (ring.ring_align_mask != 0) {
    const uint64_t unpadded_wptr = ring.wptr + dword_count;
    pad_count = (ring.ring_align_mask + 1 -
                 (unpadded_wptr & ring.ring_align_mask)) &
                ring.ring_align_mask;
    for (size_t i = 0; i < pad_count; ++i) {
      ring.ring_cpu[(start + dword_count + i) % ring_dw] = ring.ring_nop;
    }
  }
  std::atomic_thread_fence(std::memory_order_release);
  const uint64_t new_wptr = ring.wptr + dword_count + pad_count;
  *ring.wptr_cpu = new_wptr;
  std::atomic_thread_fence(std::memory_order_release);
  hsa_status_t status = platform.FlushHdp();
  if (status != HSA_STATUS_SUCCESS) return status;
  *ring.doorbell_cpu = new_wptr;
  ring.wptr = new_wptr;
  return HSA_STATUS_SUCCESS;
}

struct MesSchedulerState {
  bool initialized = false;
  DirectQueueState ring;
  DirectQueueState kiq;
  DirectQueueMemory scheduler_memory;
  DirectQueueMemory kiq_memory;
  uint64_t sch_ctx_gpu = 0;
  uint64_t query_status_fence_gpu = 0;
  uint64_t api_fence_gpu = 0;
  volatile uint64_t* api_fence_cpu = nullptr;
  uint64_t cleaner_fence_gpu = 0;
  uint64_t kiq_sch_ctx_gpu = 0;
  uint64_t kiq_query_status_fence_gpu = 0;
  uint64_t kiq_api_fence_gpu = 0;
  volatile uint64_t* kiq_api_fence_cpu = nullptr;
  uint64_t kiq_cleaner_fence_gpu = 0;
  uint64_t next_fence_value = 0;
  std::array<uint32_t, 5> aggregated_doorbells{};
};

std::mutex& MesSchedulerMutex() {
  static std::mutex* m = new std::mutex();
  return *m;
}

std::unordered_map<const DirectQueuePlatform*, MesSchedulerState>& MesSchedulers() {
  static auto* states =
      new std::unordered_map<const DirectQueuePlatform*, MesSchedulerState>();
  return *states;
}

void ResetMesSchedulerState(const DirectQueuePlatform& platform,
                            MesSchedulerState& state) {
  platform.FreeQueueMemory(&state.scheduler_memory);
  platform.FreeQueueMemory(&state.kiq_memory);
  state = {};
}

uint32_t MesPipeForRing(const DirectQueueState& ring) {
  return ring.doorbell_index == kMesKiqDoorbell ? kMesKiqPipe : 0;
}

hsa_status_t SyncMesRingPointersFromHardware(const DirectQueuePlatform& platform,
                                             DirectQueueState& ring,
                                             const DirectQueueOptions& options,
                                             const char* label) {
  if (ring.wptr_cpu == nullptr || ring.rptr_cpu == nullptr) {
    return HSA_STATUS_ERROR;
  }

  const uint32_t pipe = MesPipeForRing(ring);
  hsa_status_t status = SelectHqd(platform, kMesKiqMe, pipe, kMesKiqHqd);
  if (status != HSA_STATUS_SUCCESS) return status;

  uint32_t hw_wptr = 0;
  uint32_t hw_wptr_hi = 0;
  uint32_t hw_rptr = 0;
  status = platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_WPTR_LO, &hw_wptr);
  if (status == HSA_STATUS_SUCCESS) {
    status = platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_WPTR_HI, &hw_wptr_hi);
  }
  if (status == HSA_STATUS_SUCCESS) {
    status = platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_RPTR, &hw_rptr);
  }
  if (status != HSA_STATUS_SUCCESS) {
    DeselectHqd(platform);
    return status;
  }

  const uint64_t initial_wptr =
      (static_cast<uint64_t>(hw_wptr_hi) << 32) | hw_wptr;
  if (initial_wptr != ring.wptr ||
      (ring.rptr_cpu != nullptr && *ring.rptr_cpu != initial_wptr) ||
      (ring.wptr_cpu != nullptr && *ring.wptr_cpu != initial_wptr)) {
    ring.wptr = initial_wptr;
    *ring.wptr_cpu = initial_wptr;
    *ring.rptr_cpu = initial_wptr;
    std::atomic_thread_fence(std::memory_order_release);
    status = platform.FlushHdp();
    if (status == HSA_STATUS_SUCCESS) {
      status = platform.WriteMmio32(kGcBase0, regCP_HQD_PQ_RPTR,
                                    static_cast<uint32_t>(initial_wptr));
    }
    if (status != HSA_STATUS_SUCCESS) {
      DeselectHqd(platform);
      return status;
    }
  }

  uint32_t post_rptr = 0;
  platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_RPTR, &post_rptr);
  DeselectHqd(platform);

  if (options.trace_verbose) {
    std::fprintf(stderr,
                 "%s MES %s pointer sync pipe=%u doorbell=0x%x "
                 "hw_wptr=0x%08x:%08x hw_rptr=0x%x ring_wptr=%llu "
                 "post_rptr=0x%x\n",
                 TracePrefix(options), label ? label : "ring", pipe,
                 ring.doorbell_index, hw_wptr_hi, hw_wptr, hw_rptr,
                 static_cast<unsigned long long>(ring.wptr), post_rptr);
  }
  return HSA_STATUS_SUCCESS;
}

void DumpMesRingState(const DirectQueuePlatform& platform,
                      const DirectQueueState& ring,
                      const DirectQueueOptions& options,
                      const char* phase) {
  if (!options.trace_verbose) return;

  const uint32_t pipe = MesPipeForRing(ring);
  hsa_status_t status = SelectHqd(platform, kMesKiqMe, pipe, kMesKiqHqd);
  if (status != HSA_STATUS_SUCCESS) {
    std::fprintf(stderr, "%s MES %s select failed pipe=%u status=%u\n",
                 TracePrefix(options), phase ? phase : "diagnostic", pipe,
                 status);
    return;
  }

  uint32_t mes_cntl = 0;
  uint32_t schedulers = 0;
  uint32_t version = 0;
  uint32_t active = 0;
  uint32_t vmid = 0;
  uint32_t mqd_base = 0;
  uint32_t mqd_base_hi = 0;
  uint32_t pq_base = 0;
  uint32_t pq_base_hi = 0;
  uint32_t pq_control = 0;
  uint32_t rptr = 0;
  uint32_t wptr = 0;
  uint32_t wptr_hi = 0;
  uint32_t rptr_report = 0;
  uint32_t rptr_report_hi = 0;
  uint32_t wptr_poll = 0;
  uint32_t wptr_poll_hi = 0;
  uint32_t wptr_poll_cntl = 0;
  uint32_t doorbell_control = 0;
  uint32_t persistent = 0;
  uint32_t dequeue_status = 0;
  platform.ReadMmio32(kGcBase1, regCP_MES_CNTL, &mes_cntl);
  platform.ReadMmio32(kGcBase1, regRLC_CP_SCHEDULERS, &schedulers);
  platform.ReadMmio32(kGcBase1, regCP_MES_GP3_LO, &version);
  platform.ReadMmio32(kGcBase0, regCP_HQD_ACTIVE, &active);
  platform.ReadMmio32(kGcBase0, regCP_HQD_VMID, &vmid);
  platform.ReadMmio32(kGcBase0, regCP_MQD_BASE_ADDR, &mqd_base);
  platform.ReadMmio32(kGcBase0, regCP_MQD_BASE_ADDR_HI, &mqd_base_hi);
  platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_BASE, &pq_base);
  platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_BASE_HI, &pq_base_hi);
  platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_CONTROL, &pq_control);
  platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_RPTR, &rptr);
  platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_WPTR_LO, &wptr);
  platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_WPTR_HI, &wptr_hi);
  platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_RPTR_REPORT_ADDR, &rptr_report);
  platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_RPTR_REPORT_ADDR_HI,
                      &rptr_report_hi);
  platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_WPTR_POLL_ADDR, &wptr_poll);
  platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_WPTR_POLL_ADDR_HI,
                      &wptr_poll_hi);
  // WPTR_POLL workaround signal: is the KIQ pipe's CP polling enabled?
  platform.ReadMmio32(kGcBase0, regCP_PQ_WPTR_POLL_CNTL, &wptr_poll_cntl);
  platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_DOORBELL_CONTROL,
                      &doorbell_control);
  platform.ReadMmio32(kGcBase0, regCP_HQD_PERSISTENT_STATE, &persistent);
  platform.ReadMmio32(kGcBase0, regCP_HQD_DEQUEUE_STATUS, &dequeue_status);
  DeselectHqd(platform);

  const uint64_t cpu_wptr = ring.wptr_cpu != nullptr ? *ring.wptr_cpu : 0;
  const uint64_t cpu_rptr = ring.rptr_cpu != nullptr ? *ring.rptr_cpu : 0;
  std::fprintf(stderr,
               "%s MES %s pipe=%u doorbell=0x%x ring_gpu=0x%llx "
               "wptr=%llu cpu_wptr=%llu cpu_rptr=%llu mes_cntl=0x%08x "
               "rlc_sched=0x%08x version=0x%08x active=0x%x vmid=0x%08x\n",
               TracePrefix(options), phase ? phase : "diagnostic", pipe,
               ring.doorbell_index,
               static_cast<unsigned long long>(ring.ring_gpu),
               static_cast<unsigned long long>(ring.wptr),
               static_cast<unsigned long long>(cpu_wptr),
               static_cast<unsigned long long>(cpu_rptr), mes_cntl,
               schedulers, version, active, vmid);
  std::fprintf(stderr,
               "%s MES %s regs mqd=0x%08x:%08x pq=0x%08x:%08x "
               "pq_ctl=0x%08x rptr=0x%x wptr=0x%08x:%08x "
               "rptr_report=0x%08x:%08x wptr_poll=0x%08x:%08x "
               "wptr_poll_cntl=0x%08x "
               "doorbell_ctl=0x%08x persistent=0x%08x dequeue=0x%08x\n",
               TracePrefix(options), phase ? phase : "diagnostic",
               mqd_base_hi, mqd_base, pq_base_hi, pq_base, pq_control, rptr,
               wptr_hi, wptr, rptr_report_hi, rptr_report, wptr_poll_hi,
               wptr_poll, wptr_poll_cntl, doorbell_control, persistent,
               dequeue_status);

  if (ring.ring_cpu != nullptr) {
    std::fprintf(stderr, "%s MES %s ring[0..15]=", TracePrefix(options),
                 phase ? phase : "diagnostic");
    for (uint32_t i = 0; i < 16; ++i) {
      std::fprintf(stderr, "%s0x%08x", i == 0 ? "" : ",", ring.ring_cpu[i]);
    }
    std::fprintf(stderr, "\n");
  }
}

hsa_status_t SubmitMesApiFrameOnRing(
    const DirectQueuePlatform& platform,
    DirectQueueState& ring,
    volatile uint64_t* api_fence_cpu,
    uint64_t api_fence_gpu,
    uint64_t& next_fence_value,
    std::array<uint32_t, kMesApiFrameDwords>& frame,
    uint32_t api_status_dw,
    const DirectQueueOptions& options,
    const char* opcode_name) {
  if (ring.ring_cpu == nullptr || ring.wptr_cpu == nullptr ||
      ring.doorbell_cpu == nullptr || api_fence_cpu == nullptr) {
    return HSA_STATUS_ERROR;
  }
  const uint64_t fence_value = ++next_fence_value;
  api_fence_cpu[0] = 0;
  api_fence_cpu[1] = 0;
  PutU64(frame.data(), api_status_dw, api_fence_gpu);
  PutU64(frame.data(), api_status_dw + 2, fence_value);

  std::array<uint32_t, kMesApiFrameDwords> query{};
  query[0] = MesHeader(kMesOpcodeQuerySchedulerStatus);
  PutU64(query.data(), 2, api_fence_gpu + sizeof(uint64_t));
  PutU64(query.data(), 4, fence_value);

  const uint64_t ring_dw = ring.ring_size_bytes / sizeof(uint32_t);
  const uint64_t start = ring.wptr % ring_dw;
  for (uint32_t i = 0; i < kMesApiFrameDwords; ++i) {
    ring.ring_cpu[(start + i) % ring_dw] = frame[i];
  }
  for (uint32_t i = 0; i < kMesApiFrameDwords; ++i) {
    ring.ring_cpu[(start + kMesApiFrameDwords + i) % ring_dw] = query[i];
  }
  std::atomic_thread_fence(std::memory_order_release);

  const uint64_t new_wptr = ring.wptr + kMesApiFrameDwords * 2;
  *ring.wptr_cpu = new_wptr;
  std::atomic_thread_fence(std::memory_order_release);
  hsa_status_t status = platform.FlushHdp();
  if (status != HSA_STATUS_SUCCESS) return status;
  if (options.trace_verbose) {
    std::fprintf(stderr,
                 "%s MES API %s submit start=%llu old_wptr=%llu "
                 "new_wptr=%llu api_fence=0x%llx status_dw=%u "
                 "frame0=0x%08x frame1=0x%08x frame22=0x%08x "
                 "frame24=0x%08x frame50=0x%08x query0=0x%08x\n",
                 TracePrefix(options), opcode_name ? opcode_name : "unknown",
                 static_cast<unsigned long long>(start),
                 static_cast<unsigned long long>(ring.wptr),
                 static_cast<unsigned long long>(new_wptr),
                 static_cast<unsigned long long>(api_fence_gpu),
                 api_status_dw, frame[0], frame[1], frame[22], frame[24],
                 frame[50], query[0]);
    DumpMesRingState(platform, ring, options, "pre-doorbell");
  }
  // Windows doorbell-dead workaround (ROCR_WINDOWS_MES_MMIO_WPTR):
  // Host doorbell writes do NOT reach the GPU on this WDDM/passthrough setup
  // (proven by dbprobe: doorbell-only FAILS), and WPTR_POLL on the KIQ does
  // NOT make the MES service the ring either (KIQ RPTR stayed 0). The only
  // mechanism proven to advance an HQD on this setup is the direct MMIO wptr
  // poke that SubmitDirectQueue uses for the MEC (me=1): after writing the
  // ring + the in-mem wptr, it does SelectHqd(me,pipe,hqd) and writes
  // CP_HQD_PQ_WPTR_LO/HI, and the MEC's CP fetches the ring. Mirror that
  // exactly for the KIQ HQD here (me=3/pipe=1/queue=0, the same select the
  // MES path already uses in ProgramMesQueueRegisters/DumpMesRingState) so
  // the MES pipe's CP picks up the wptr and fetches the KIQ ring without a
  // doorbell event. Default (flag unset) = unchanged, so macOS/Linux behave
  // identically. Independent of ROCR_WINDOWS_MES_WPTR_POLL (both can be set).
  // Accepts ROCR_WINDOWS_MES_MMIO_WPTR (Windows) or generic ROCR_MES_MMIO_WPTR.
  const bool mmio_wptr_poke = MesMmioWptrPokeEnabled();
  if (mmio_wptr_poke) {
    const uint32_t pipe = MesPipeForRing(ring);
    status = SelectHqd(platform, kMesKiqMe, pipe, kMesKiqHqd);
    if (status == HSA_STATUS_SUCCESS) {
      // Same registers SubmitDirectQueue's MEC poke writes.
      status = platform.WriteMmio32(kGcBase0, regCP_HQD_PQ_WPTR_LO,
                                    static_cast<uint32_t>(new_wptr));
      if (status == HSA_STATUS_SUCCESS) {
        status = platform.WriteMmio32(kGcBase0, regCP_HQD_PQ_WPTR_HI,
                                      static_cast<uint32_t>(new_wptr >> 32));
      }
    }
    if (options.trace) {
      // Diagnostics: did the poke latch (CP_HQD_PQ_WPTR) and does the KIQ
      // RPTR advance now? RPTR advancing + api_fence reaching expected = win.
      uint32_t poke_active = 0;
      uint32_t poke_rptr = 0;
      uint32_t poke_wptr = 0;
      uint32_t poke_wptr_hi = 0;
      platform.ReadMmio32(kGcBase0, regCP_HQD_ACTIVE, &poke_active);
      platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_RPTR, &poke_rptr);
      platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_WPTR_LO, &poke_wptr);
      platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_WPTR_HI, &poke_wptr_hi);
      std::fprintf(stderr,
                   "%s MES API %s mmio-wptr poke me=%u pipe=%u hqd=%u "
                   "new_wptr=%llu active=0x%x rptr=0x%x wptr=0x%08x:%08x "
                   "status=%u\n",
                   TracePrefix(options), opcode_name ? opcode_name : "unknown",
                   kMesKiqMe, pipe, kMesKiqHqd,
                   static_cast<unsigned long long>(new_wptr), poke_active,
                   poke_rptr, poke_wptr_hi, poke_wptr, status);
    }
    DeselectHqd(platform);
    if (status != HSA_STATUS_SUCCESS) return status;
  }
  *ring.doorbell_cpu = new_wptr;
  ring.wptr = new_wptr;
  if (options.trace_verbose) {
    platform.SleepUs(100);
    DumpMesRingState(platform, ring, options, "post-doorbell");
    // WPTR_POLL workaround win-signal: pair the KIQ RPTR (printed by
    // DumpMesRingState above) with the api_fence here. If the KIQ RPTR
    // advanced and api_fence reaches the expected value WITHOUT the doorbell
    // delivering, the MES serviced the frame purely via the in-memory
    // wptr-poll => the Windows doorbell-dead workaround works.
    std::fprintf(stderr,
                 "%s MES API %s post-doorbell api_fence[0]=0x%llx "
                 "api_fence[1]=0x%llx expected=%llu wptr_dword=%llu\n",
                 TracePrefix(options), opcode_name ? opcode_name : "unknown",
                 static_cast<unsigned long long>(api_fence_cpu[0]),
                 static_cast<unsigned long long>(api_fence_cpu[1]),
                 static_cast<unsigned long long>(fence_value),
                 static_cast<unsigned long long>(
                     ring.wptr_cpu != nullptr ? *ring.wptr_cpu : 0));
  }

  constexpr uint32_t kStepUs = 1000;
  constexpr uint32_t kTimeoutUs = 5000000;
  uint64_t observed_api = 0;
  uint64_t observed_query = 0;
  for (uint32_t elapsed = 0; elapsed < kTimeoutUs; elapsed += kStepUs) {
    std::atomic_thread_fence(std::memory_order_acquire);
    observed_api = api_fence_cpu[0];
    observed_query = api_fence_cpu[1];
    if (observed_api == fence_value && observed_query == fence_value) {
      if (options.trace) {
        std::fprintf(stderr,
                     "%s MES API %s complete fence=%llu query=%llu "
                     "wptr=%llu\n",
                     TracePrefix(options), opcode_name ? opcode_name : "unknown",
                     static_cast<unsigned long long>(fence_value),
                     static_cast<unsigned long long>(observed_query),
                     static_cast<unsigned long long>(new_wptr));
      }
      return HSA_STATUS_SUCCESS;
    }
    if ((observed_api != 0 && observed_api != fence_value) ||
        (observed_query != 0 && observed_query != fence_value)) {
      break;
    }
    platform.SleepUs(kStepUs);
  }

  if (options.trace) {
    uint32_t rptr =
        ring.rptr_cpu != nullptr ? static_cast<uint32_t>(*ring.rptr_cpu) : 0;
    std::fprintf(stderr,
                 "%s MES API %s timeout/error observed_api=0x%llx "
                 "observed_query=0x%llx expected=%llu wptr=%llu rptr=%u\n",
                 TracePrefix(options), opcode_name ? opcode_name : "unknown",
                 static_cast<unsigned long long>(observed_api),
                 static_cast<unsigned long long>(observed_query),
                 static_cast<unsigned long long>(fence_value),
                 static_cast<unsigned long long>(ring.wptr), rptr);
    DumpMesRingState(platform, ring, options, "timeout");
  }
  return HSA_STATUS_ERROR;
}

hsa_status_t SubmitMesApiFrame(const DirectQueuePlatform& platform,
                               MesSchedulerState& state,
                               std::array<uint32_t, kMesApiFrameDwords>& frame,
                               uint32_t api_status_dw,
                               const DirectQueueOptions& options,
                               const char* opcode_name) {
  return SubmitMesApiFrameOnRing(platform, state.ring, state.api_fence_cpu,
                                 state.api_fence_gpu, state.next_fence_value,
                                 frame, api_status_dw, options, opcode_name);
}

std::array<uint32_t, kMesApiFrameDwords> BuildMesSetHwResourcesFrame(
    const MesSchedulerState& state,
    uint64_t sch_ctx_gpu,
    uint64_t query_status_fence_gpu,
    bool include_scheduler_resources,
    uint32_t oversubscription_timer) {
  std::array<uint32_t, kMesApiFrameDwords> frame{};
  frame[0] = MesHeader(kMesOpcodeSetHwResources);
  if (include_scheduler_resources) {
    frame[1] = 0x0000ff00u;
    frame[2] = 0x0000ffffu;
    frame[3] = 0;
    frame[4] = 0;
    for (uint32_t i = 0; i < 8; ++i) frame[5 + i] = 0xffu;
    frame[13] = 0xffu;
    frame[14] = 0xffu;
    frame[15] = 0xfcu;
    frame[16] = 0xfcu;
    for (uint32_t i = 0; i < state.aggregated_doorbells.size(); ++i) {
      frame[17 + i] = state.aggregated_doorbells[i];
    }
  }
  PutU64(frame.data(), 22, sch_ctx_gpu);
  PutU64(frame.data(), 24, query_status_fence_gpu);
  constexpr uint32_t kGcBases[] = {
      kGcBase0, kGcBase1, 0x0001c000u, 0x02402c00u, 0x02000112u};
  constexpr uint32_t kMmhubBases[] = {
      0x0001a000u, 0x02408800u, 0x0f0000ffu, 0x0003000eu,
      0x00016000u};
  constexpr uint32_t kOsssysBases[] = {
      0x000010a0u, 0x0240a000u, 0x03000046u, 0x00000106u,
      0x02411800u};
  for (uint32_t i = 0; i < 5; ++i) {
    frame[26 + i] = kGcBases[i];
    frame[34 + i] = kMmhubBases[i];
    frame[42 + i] = kOsssysBases[i];
  }
  frame[54] = kMesSetHwResourcesFlags;
  frame[55] = oversubscription_timer;
  return frame;
}

uint32_t MesOversubscriptionTimer(uint32_t version) {
  (void)version;
  return 50;
}

uint32_t ReadMesPipeVersion(const DirectQueuePlatform& platform,
                            uint32_t pipe) {
  uint32_t version = 0;
  if (SelectHqd(platform, kMesKiqMe, pipe, kMesKiqHqd) ==
      HSA_STATUS_SUCCESS) {
    platform.ReadMmio32(kGcBase1, regCP_MES_GP3_LO, &version);
    DeselectHqd(platform);
  }
  return version;
}

std::array<uint32_t, kMesApiFrameDwords> BuildMesSetHwResources1Frame(
    uint64_t cleaner_shader_fence_gpu) {
  std::array<uint32_t, kMesApiFrameDwords> frame{};
  frame[0] = MesHeader(kMesOpcodeSetHwResources1);
  frame[13] = 0x0a;
  PutU64(frame.data(), 16, cleaner_shader_fence_gpu);
  return frame;
}

std::array<uint32_t, kMesApiFrameDwords> BuildMesMapLegacySchedulerFrame(
    const DirectQueueLayout& scheduler_layout) {
  std::array<uint32_t, kMesApiFrameDwords> frame{};
  frame[0] = MesHeader(kMesOpcodeAddQueue);
  frame[18] = kMesSchedulerDoorbell;
  PutU64(frame.data(), 20, scheduler_layout.mqd_gpu);
  PutU64(frame.data(), 22, scheduler_layout.wptr_gpu);
  frame[28] = kMesQueueTypeScheduler;
  frame[37] = kMesAddQueueMapLegacyKq;
  frame[50] = 0;
  frame[51] = 0;
  return frame;
}

void InitMesAggregatedDoorbells(const DirectQueuePlatform& platform,
                                const MesSchedulerState& state) {
  constexpr uint32_t regs[] = {
      regCP_MES_DOORBELL_CONTROL1, regCP_MES_DOORBELL_CONTROL2,
      regCP_MES_DOORBELL_CONTROL3, regCP_MES_DOORBELL_CONTROL4,
      regCP_MES_DOORBELL_CONTROL5};
  for (uint32_t i = 0; i < state.aggregated_doorbells.size(); ++i) {
    uint32_t data = 0;
    if (platform.ReadMmio32(kGcBase1, regs[i], &data) != HSA_STATUS_SUCCESS) {
      continue;
    }
    data &= ~(kCpMesDoorbellOffsetMask | kCpMesDoorbellEnable |
              kCpMesDoorbellHit);
    data |= (state.aggregated_doorbells[i] << 2) | kCpMesDoorbellEnable;
    platform.WriteMmio32(kGcBase1, regs[i], data);
  }
  platform.WriteMmio32(kGcBase0, regCP_HQD_GFX_CONTROL,
                       kCpHqdGfxControlDbUpdatedMsgEn);
}

void EnableUnmappedDoorbellHandling(const DirectQueuePlatform& platform) {
  uint32_t data = 0;
  if (platform.ReadMmio32(kGcBase1, regCP_UNMAPPED_DOORBELL, &data) !=
      HSA_STATUS_SUCCESS) {
    return;
  }
  data &= ~kCpUnmappedDoorbellProcLsbMask;
  data |= 0xdu << kCpUnmappedDoorbellProcLsbShift;
  data |= kCpUnmappedDoorbellEnable;
  platform.WriteMmio32(kGcBase1, regCP_UNMAPPED_DOORBELL, data);
}

uint32_t QueueSizeField(uint32_t ring_size_bytes) {
  uint32_t dwords = ring_size_bytes / sizeof(uint32_t);
  uint32_t log2_dwords = 0;
  while (dwords > 1) {
    dwords >>= 1;
    ++log2_dwords;
  }
  return log2_dwords == 0 ? 0 : log2_dwords - 1;
}

uint32_t MesPqControl(uint32_t ring_size_bytes) {
  return (kCpHqdPqControlMes & ~0x3fu) |
         (QueueSizeField(ring_size_bytes) & 0x3fu);
}

DirectQueueMqd BuildMesKernelQueueMqd(const DirectQueueLayout& layout,
                                      uint32_t doorbell_index,
                                      uint32_t ring_size) {
  DirectQueueMqd mqd{};
  mqd[0] = 0xC0310800;
  mqd[0x0B] = 1;
  constexpr uint32_t kStaticThreadMgmtDwords[] = {0x17u, 0x18u, 0x1Au, 0x1Bu};
  for (uint32_t dw : kStaticThreadMgmtDwords) mqd[dw] = 0xFFFFFFFFu;
  mqd[0x20] = 7;

  const uint64_t eop_base_shifted = layout.eop_gpu >> 8;
  mqd[0xA5] = static_cast<uint32_t>(eop_base_shifted);
  mqd[0xA6] = static_cast<uint32_t>(eop_base_shifted >> 32);
  mqd[0xA7] = 8;  // Linux MES uses a 2 KiB EOP window.

  mqd[0x80] = static_cast<uint32_t>(layout.mqd_gpu) & 0xFFFFFFFCu;
  mqd[0x81] = static_cast<uint32_t>(layout.mqd_gpu >> 32);
  mqd[0x82] = 1;
  mqd[0x84] = (kCpHqdPersistentStateDefault & ~(0x3FFu << 8)) | (0x55u << 8);
  mqd[0x87] = 0x111;

  const uint64_t pq_base_shifted = layout.ring_gpu >> 8;
  mqd[0x88] = static_cast<uint32_t>(pq_base_shifted);
  mqd[0x89] = static_cast<uint32_t>(pq_base_shifted >> 32);
  mqd[0x8B] = static_cast<uint32_t>(layout.rptr_gpu) & 0xFFFFFFFCu;
  mqd[0x8C] = static_cast<uint32_t>(layout.rptr_gpu >> 32) & 0xFFFFu;
  mqd[0x8D] = static_cast<uint32_t>(layout.wptr_gpu) & 0xFFFFFFF8u;
  mqd[0x8E] = static_cast<uint32_t>(layout.wptr_gpu >> 32) & 0xFFFFu;
  mqd[0x8F] = ((doorbell_index & 0x03FFFFFFu) << 2) | (1u << 30);

  mqd[0x91] = MesPqControl(ring_size);
  mqd[0x95] = 0x00300000;
  mqd[0xA2] = kCpMqdControlDefault;
  mqd[0xB8] = 1u << 15;
  return mqd;
}

[[maybe_unused]] hsa_status_t SubmitKiqScratchTest(
    const DirectQueuePlatform& platform,
    MesSchedulerState& state,
    const DirectQueueOptions& options) {
  constexpr uint32_t kScratchValue = 0xdeadbeef;
  const uint32_t scratch_offset = kGcBase1 + regSCRATCH_REG0;
  platform.WriteMmio32(kGcBase1, regSCRATCH_REG0, 0xcafedead);
  const uint32_t pm4[] = {
      Packet3(kPacket3WriteData, 3),
      1u << 16,
      scratch_offset,
      0,
      kScratchValue,
  };
  hsa_status_t status =
      SubmitRingPm4(platform, state.kiq, pm4, sizeof(pm4) / sizeof(pm4[0]));
  if (status != HSA_STATUS_SUCCESS) return status;
  for (uint32_t i = 0; i < 100000; ++i) {
    uint32_t value = 0;
    platform.ReadMmio32(kGcBase1, regSCRATCH_REG0, &value);
    if (value == kScratchValue) {
      if (options.trace) {
        std::fprintf(stderr, "%s KIQ scratch test complete samples=%u\n",
                     TracePrefix(options), i + 1);
      }
      return HSA_STATUS_SUCCESS;
    }
    platform.SleepUs(1);
  }
  if (options.trace) {
    uint32_t rptr = state.kiq.rptr_cpu != nullptr
                        ? static_cast<uint32_t>(*state.kiq.rptr_cpu)
                        : 0;
    std::fprintf(stderr,
                 "%s KIQ scratch test timeout wptr=%llu rptr=%u\n",
                 TracePrefix(options),
                 static_cast<unsigned long long>(state.kiq.wptr), rptr);
  }
  return HSA_STATUS_ERROR;
}

[[maybe_unused]] hsa_status_t MapSchedulerWithKiq(
    const DirectQueuePlatform& platform,
    MesSchedulerState& state,
    const DirectQueueLayout& scheduler_layout,
    const DirectQueueOptions& options) {
  constexpr uint32_t kScratchValue = 0xdeadbeef;
  const uint32_t scratch_offset = kGcBase1 + regSCRATCH_REG0;
  platform.WriteMmio32(kGcBase1, regSCRATCH_REG0, 0xcafedead);
  const uint32_t map_control =
      (0u << 4) |   // queue select
      (0u << 8) |   // VMID
      (0u << 13) |  // queue
      (0u << 16) |  // pipe
      (2u << 18) |  // MES engine queue selector
      (0u << 21) |  // normal queue
      (0u << 24) |  // all on one pipe
      (5u << 26) |  // MES engine
      (1u << 29);
  const uint32_t pm4[] = {
      Packet3(kPacket3MapQueues, 5),
      map_control,
      kMesSchedulerDoorbell << 2,
      static_cast<uint32_t>(scheduler_layout.mqd_gpu),
      static_cast<uint32_t>(scheduler_layout.mqd_gpu >> 32),
      static_cast<uint32_t>(scheduler_layout.wptr_gpu),
      static_cast<uint32_t>(scheduler_layout.wptr_gpu >> 32),
      Packet3(kPacket3WriteData, 3),
      1u << 16,
      scratch_offset,
      0,
      kScratchValue,
  };
  hsa_status_t status =
      SubmitRingPm4(platform, state.kiq, pm4, sizeof(pm4) / sizeof(pm4[0]));
  if (status != HSA_STATUS_SUCCESS) return status;
  bool scratch_done = false;
  for (uint32_t i = 0; i < 100000; ++i) {
    uint32_t value = 0;
    platform.ReadMmio32(kGcBase1, regSCRATCH_REG0, &value);
    if (value == kScratchValue) {
      scratch_done = true;
      break;
    }
    platform.SleepUs(1);
  }
  if (options.trace) {
    const uint32_t kiq_rptr = state.kiq.rptr_cpu != nullptr
                                  ? static_cast<uint32_t>(*state.kiq.rptr_cpu)
                                  : 0;
    std::fprintf(stderr,
                 "%s MES scheduler MAP_QUEUES submitted by KIQ kiq_wptr=%llu "
                 "kiq_rptr=%u scratch=%s mqd=0x%llx wptr=0x%llx\n",
                 TracePrefix(options),
                 static_cast<unsigned long long>(state.kiq.wptr), kiq_rptr,
                 scratch_done ? "done" : "timeout",
                 static_cast<unsigned long long>(scheduler_layout.mqd_gpu),
                 static_cast<unsigned long long>(scheduler_layout.wptr_gpu));
  }
  return scratch_done ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR;
}

// --- Per-process MES scheduler teardown (#66) -------------------------------
// Windows has no KMD to reset queues on process exit and the driver's C++
// teardown does not run on a hard exit, so a process that exits leaving its MES
// scheduler-ring HQD (me=3/pipe=0) ACTIVE wedges the next process: the fresh
// EnsureMesScheduler's dequeue-drain of that HQD times out on the now-dead ring
// and the scheduler SET_HW_RESOURCES is never serviced (MES map status=4096 ->
// hsa_queue_create fails from the 3rd process on). Deactivating THIS process's
// scheduler HQD while the MES is still healthy (its drain completes at once)
// leaves active=0 so the next process's reset is clean. Default ON on Windows
// (opt out with ROCR_WINDOWS_MES_TEARDOWN_AT_EXIT=0); macOS/Linux stay opt-in so
// their exit path is byte-identical unless the env is explicitly set.
bool MesTeardownAtExitEnabled() {
  const char* v = std::getenv("ROCR_MACOS_MES_TEARDOWN_AT_EXIT");
  if (v == nullptr) v = std::getenv("ROCR_WINDOWS_MES_TEARDOWN_AT_EXIT");
#if defined(_WIN32) || defined(__APPLE__)
  // Default ON (Windows + macOS): on the MES-backed path a prior process leaves
  // the MES scheduler-ring HQD active, wedging the next process's scheduler
  // SET_HW_RESOURCES (status=4096; Windows returns an error, macOS retries the
  // create -> hang). HW-validated on both: macOS MES isolate 9/9 (was 0/9) +
  // 5/5 sequential torch procs. An explicit env value wins (set =0 to opt out).
  // No-op unless an MES scheduler was actually initialized (e.g. macOS default
  // direct path never registers this).
  if (v != nullptr && v[0] != '\0') return v[0] != '0';
  return true;
#else
  // Linux: opt-in only.
  return v != nullptr && v[0] != '\0' && v[0] != '0';
#endif
}

void DeactivateMesSchedulerHqd(const DirectQueuePlatform& platform,
                               MesSchedulerState& state) {
  const uint32_t pipe = MesPipeForRing(state.ring);
  if (SelectHqd(platform, kMesKiqMe, pipe, 0) != HSA_STATUS_SUCCESS) return;
  DirectQueueOptions options;  // trace off, default dequeue timeout
  // Reuse the exact reset the next process would run on this HQD, but now, while
  // the MES is healthy: the drain completes immediately (active -> 0) instead of
  // timing out on a dead ring, leaving a clean HQD for the next bring-up.
  ResetSelectedHqdForProgramming(platform, pipe, 0, options, "TEARDOWN");
  DeselectHqd(platform);
}

void TeardownAllMesSchedulersAtExit() {
  std::lock_guard<std::mutex> lock(MesSchedulerMutex());
  for (auto& entry : MesSchedulers()) {
    MesSchedulerState& state = entry.second;
    if (!state.initialized) continue;
    DeactivateMesSchedulerHqd(*entry.first, state);
    state.initialized = false;
  }
}

hsa_status_t EnsureMesScheduler(const DirectQueuePlatform& platform,
                                uint64_t framebuffer_base,
                                const DirectQueueOptions& options) {
  std::lock_guard<std::mutex> lock(MesSchedulerMutex());
  MesSchedulerState& state = MesSchedulers()[&platform];
  if (state.initialized) return HSA_STATUS_SUCCESS;

  uint32_t mes_cntl = 0;
  hsa_status_t status = platform.ReadMmio32(kGcBase1, regCP_MES_CNTL, &mes_cntl);
  if (status != HSA_STATUS_SUCCESS) return status;
  if ((mes_cntl & (kCpMesCntlPipe0Active | kCpMesCntlPipe1Active)) !=
      (kCpMesCntlPipe0Active | kCpMesCntlPipe1Active)) {
    const uint32_t scheduler_version = ReadMesPipeVersion(platform, 0);
    const uint32_t kiq_version = ReadMesPipeVersion(platform, kMesKiqPipe);
    if (scheduler_version == 0 || kiq_version == 0) {
      if (options.trace) {
        std::fprintf(stderr,
                     "%s MES scheduler/KIQ pipe inactive CP_MES_CNTL=0x%08x "
                     "versions sched=0x%08x kiq=0x%08x; "
                     "run firmware bring-up first\n",
                     TracePrefix(options), mes_cntl, scheduler_version,
                     kiq_version);
      }
      return HSA_STATUS_ERROR;
    }
    if (options.trace) {
      std::fprintf(stderr,
                   "%s MES CP_MES_CNTL readback is 0x%08x but GP3 versions "
                   "are sched=0x%08x kiq=0x%08x; continuing\n",
                   TracePrefix(options), mes_cntl, scheduler_version,
                   kiq_version);
    }
  }

  hsa_status_t aperture_status = platform.EnsureDoorbellAperture();
  if (aperture_status != HSA_STATUS_SUCCESS) return aperture_status;

  DirectQueueLayout scheduler_layout{};
  DirectQueueLayout kiq_layout{};
  DirectQueueMemory scheduler_memory{};
  DirectQueueMemory kiq_memory{};
  status = PrepareQueueLayout(platform, framebuffer_base, kMesSchedulerBaseOffset,
                              &scheduler_layout, &scheduler_memory);
  if (status == HSA_STATUS_SUCCESS) {
    status = PrepareQueueLayout(platform, framebuffer_base, kMesKiqBaseOffset,
                                &kiq_layout, &kiq_memory);
  }
  if (status != HSA_STATUS_SUCCESS) {
    platform.FreeQueueMemory(&scheduler_memory);
    platform.FreeQueueMemory(&kiq_memory);
    return status;
  }

  DirectQueueMqd scheduler_mqd =
      BuildMesKernelQueueMqd(scheduler_layout, kMesSchedulerDoorbell,
                             kMesRingSize);
  DirectQueueMqd kiq_mqd =
      BuildMesKernelQueueMqd(kiq_layout, kMesKiqDoorbell, kMesRingSize);

  if (options.trace_verbose) {
    std::fprintf(stderr,
                 "%s MES layouts sched base=0x%llx mqd=0x%llx ring=0x%llx "
                 "rptr=0x%llx wptr=0x%llx doorbell=0x%x pq_ctl=0x%08x\n",
                 TracePrefix(options),
                 static_cast<unsigned long long>(scheduler_layout.base_gpu),
                 static_cast<unsigned long long>(scheduler_layout.mqd_gpu),
                 static_cast<unsigned long long>(scheduler_layout.ring_gpu),
                 static_cast<unsigned long long>(scheduler_layout.rptr_gpu),
                 static_cast<unsigned long long>(scheduler_layout.wptr_gpu),
                 kMesSchedulerDoorbell, scheduler_mqd[0x91]);
    std::fprintf(stderr,
                 "%s MES layouts kiq base=0x%llx mqd=0x%llx ring=0x%llx "
                 "rptr=0x%llx wptr=0x%llx doorbell=0x%x pq_ctl=0x%08x\n",
                 TracePrefix(options),
                 static_cast<unsigned long long>(kiq_layout.base_gpu),
                 static_cast<unsigned long long>(kiq_layout.mqd_gpu),
                 static_cast<unsigned long long>(kiq_layout.ring_gpu),
                 static_cast<unsigned long long>(kiq_layout.rptr_gpu),
                 static_cast<unsigned long long>(kiq_layout.wptr_gpu),
                 kMesKiqDoorbell, kiq_mqd[0x91]);
  }

  status = ZeroQueueMemory(platform, scheduler_layout, kMesRingSize);
  if (status == HSA_STATUS_SUCCESS) {
    status = ZeroQueueMemory(platform, kiq_layout, kMesRingSize);
  }
  if (status == HSA_STATUS_SUCCESS) {
    status = ZeroLayoutMemory(
        platform, scheduler_layout,
        scheduler_layout.base_offset + kMesSchedulerContextRelativeOffset,
        0x3000);
  }
  if (status == HSA_STATUS_SUCCESS) {
    status = ZeroLayoutMemory(platform, kiq_layout,
                              kiq_layout.base_offset +
                                  kMesSchedulerContextRelativeOffset,
                              0x3000);
  }
  if (status == HSA_STATUS_SUCCESS) {
    status = WriteMqdImage(platform, scheduler_layout, scheduler_mqd);
  }
  if (status == HSA_STATUS_SUCCESS) {
    status = WriteMqdImage(platform, kiq_layout, kiq_mqd);
  }
  if (status != HSA_STATUS_SUCCESS) {
    platform.FreeQueueMemory(&scheduler_memory);
    platform.FreeQueueMemory(&kiq_memory);
    return status;
  }

  DirectQueueState ring = MakeRingState(platform, scheduler_layout,
                                        kMesSchedulerDoorbell, kMesRingSize);
  DirectQueueState kiq =
      MakeRingState(platform, kiq_layout, kMesKiqDoorbell, kMesRingSize);
  if (ring.ring_cpu == nullptr || ring.rptr_cpu == nullptr ||
      ring.wptr_cpu == nullptr || ring.doorbell_cpu == nullptr ||
      kiq.ring_cpu == nullptr || kiq.rptr_cpu == nullptr ||
      kiq.wptr_cpu == nullptr || kiq.doorbell_cpu == nullptr) {
    platform.FreeQueueMemory(&scheduler_memory);
    platform.FreeQueueMemory(&kiq_memory);
    return HSA_STATUS_ERROR;
  }

  state = {};
  state.ring = ring;
  state.kiq = kiq;
  state.scheduler_memory = scheduler_memory;
  state.kiq_memory = kiq_memory;
  scheduler_memory = {};
  kiq_memory = {};
  state.sch_ctx_gpu =
      scheduler_layout.base_gpu + kMesSchedulerContextRelativeOffset;
  state.query_status_fence_gpu =
      scheduler_layout.base_gpu + kMesQueryFenceRelativeOffset;
  state.api_fence_gpu = scheduler_layout.base_gpu + kMesApiFenceRelativeOffset;
  state.api_fence_cpu = static_cast<volatile uint64_t*>(
      LayoutCpuPointer(platform, scheduler_layout,
                       scheduler_layout.base_offset +
                           kMesApiFenceRelativeOffset,
                       sizeof(uint64_t) * 2));
  state.cleaner_fence_gpu =
      scheduler_layout.base_gpu + kMesCleanerFenceRelativeOffset;
  state.kiq_sch_ctx_gpu =
      kiq_layout.base_gpu + kMesSchedulerContextRelativeOffset;
  state.kiq_query_status_fence_gpu =
      kiq_layout.base_gpu + kMesQueryFenceRelativeOffset;
  state.kiq_api_fence_gpu = kiq_layout.base_gpu + kMesApiFenceRelativeOffset;
  state.kiq_api_fence_cpu = static_cast<volatile uint64_t*>(
      LayoutCpuPointer(platform, kiq_layout,
                       kiq_layout.base_offset + kMesApiFenceRelativeOffset,
                       sizeof(uint64_t) * 2));
  state.kiq_cleaner_fence_gpu =
      kiq_layout.base_gpu + kMesCleanerFenceRelativeOffset;
  if (state.api_fence_cpu == nullptr || state.kiq_api_fence_cpu == nullptr) {
    ResetMesSchedulerState(platform, state);
    return HSA_STATUS_ERROR;
  }
  for (uint32_t i = 0; i < state.aggregated_doorbells.size(); ++i) {
    state.aggregated_doorbells[i] = kMesAggregatedDoorbellBase + i * 2;
  }

  uint32_t schedulers = 0;
  if (platform.ReadMmio32(kGcBase1, regRLC_CP_SCHEDULERS, &schedulers) ==
      HSA_STATUS_SUCCESS) {
    schedulers &= 0xffffff00u;
    schedulers |= (kMesKiqMe << 5) | (kMesKiqPipe << 3) | kMesKiqHqd |
                  0x80u;
    platform.WriteMmio32(kGcBase1, regRLC_CP_SCHEDULERS, schedulers);
  }

  status = ProgramMesQueueRegisters(platform, kMesKiqPipe, kiq_mqd, options,
                                    "MES KIQ", /*set_active=*/true);
  if (status != HSA_STATUS_SUCCESS) {
    ResetMesSchedulerState(platform, state);
    return status;
  }
  status = SyncMesRingPointersFromHardware(platform, state.kiq, options,
                                           "KIQ");
  if (status != HSA_STATUS_SUCCESS) {
    ResetMesSchedulerState(platform, state);
    return status;
  }

  const uint32_t kiq_version = ReadMesPipeVersion(platform, kMesKiqPipe);
  const uint32_t scheduler_version = ReadMesPipeVersion(platform, 0);

  if (std::getenv("ROCR_AMDGPU_LITE_MES_QUERY_FIRST") != nullptr) {
    std::array<uint32_t, kMesApiFrameDwords> kiq_query_first{};
    kiq_query_first[0] = MesHeader(kMesOpcodeQuerySchedulerStatus);
    status = SubmitMesApiFrameOnRing(
        platform, state.kiq, state.kiq_api_fence_cpu, state.kiq_api_fence_gpu,
        state.next_fence_value, kiq_query_first, 2, options,
        "KIQ QUERY_FIRST");
    if (status != HSA_STATUS_SUCCESS) {
      ResetMesSchedulerState(platform, state);
      return status;
    }
  }

  EnableUnmappedDoorbellHandling(platform);
  auto kiq_set_hw = BuildMesSetHwResourcesFrame(
      state, state.kiq_sch_ctx_gpu, state.kiq_query_status_fence_gpu, false,
      MesOversubscriptionTimer(kiq_version));
  status = SubmitMesApiFrameOnRing(
      platform, state.kiq, state.kiq_api_fence_cpu, state.kiq_api_fence_gpu,
      state.next_fence_value, kiq_set_hw, kMesApiStatusSetHwResourcesDw,
      options, "KIQ SET_HW_RESOURCES");
  if (status == HSA_STATUS_SUCCESS) {
    auto kiq_set_hw1 =
        BuildMesSetHwResources1Frame(state.kiq_cleaner_fence_gpu);
    status = SubmitMesApiFrameOnRing(
        platform, state.kiq, state.kiq_api_fence_cpu, state.kiq_api_fence_gpu,
        state.next_fence_value, kiq_set_hw1, kMesApiStatusSetHwResources1Dw,
        options,
        "KIQ SET_HW_RESOURCES_1");
  }
  // Program the scheduler ring's HQD image on pipe 0 (MQD registers + WPTR_POLL)
  // WITHOUT setting CP_HQD_ACTIVE: MES owns/activates it via the MAP_SCHEDULER
  // below (mirrors the proven Python direct_activate=False recipe). On
  // WDDM/passthrough the scheduler ring was otherwise never programmed on pipe 0,
  // so MES never serviced it and its rptr stayed 0 (SET_HW_RESOURCES timeout ->
  // MES map status=4096). Gated Windows-only.
  if (status == HSA_STATUS_SUCCESS && MesActivateSchedulerHqdEnabled()) {
    const uint32_t sched_pipe = MesPipeForRing(state.ring);
    status = ProgramMesQueueRegisters(platform, sched_pipe, scheduler_mqd,
                                      options, "MES SCHEDULER",
                                      /*set_active=*/false);
    if (status != HSA_STATUS_SUCCESS) {
      ResetMesSchedulerState(platform, state);
      return status;
    }
  }
  if (status == HSA_STATUS_SUCCESS) {
    auto map_scheduler = BuildMesMapLegacySchedulerFrame(scheduler_layout);
    status = SubmitMesApiFrameOnRing(
        platform, state.kiq, state.kiq_api_fence_cpu, state.kiq_api_fence_gpu,
        state.next_fence_value, map_scheduler, kMesApiStatusAddQueueDw,
        options, "KIQ MAP_SCHEDULER");
  }
  if (status != HSA_STATUS_SUCCESS) {
    ResetMesSchedulerState(platform, state);
    return status;
  }

  auto set_hw = BuildMesSetHwResourcesFrame(
      state, state.sch_ctx_gpu, state.query_status_fence_gpu, true,
      MesOversubscriptionTimer(scheduler_version));
  status = SubmitMesApiFrame(platform, state, set_hw,
                             kMesApiStatusSetHwResourcesDw, options,
                             "SET_HW_RESOURCES");
  if (status != HSA_STATUS_SUCCESS) {
    ResetMesSchedulerState(platform, state);
    return status;
  }
  auto set_hw1 = BuildMesSetHwResources1Frame(state.cleaner_fence_gpu);
  status = SubmitMesApiFrame(platform, state, set_hw1,
                             kMesApiStatusSetHwResources1Dw, options,
                             "SET_HW_RESOURCES_1");
  if (status != HSA_STATUS_SUCCESS) {
    ResetMesSchedulerState(platform, state);
    return status;
  }
  InitMesAggregatedDoorbells(platform, state);

  state.initialized = true;
  if (MesTeardownAtExitEnabled()) {
    // Register once: on clean process exit, deactivate this process's MES
    // scheduler HQD so the next process's bring-up is not wedged (#66).
    static std::once_flag mes_teardown_once;
    std::call_once(mes_teardown_once,
                   [] { std::atexit(TeardownAllMesSchedulersAtExit); });
  }
  if (options.trace) {
    uint32_t version = 0;
    platform.ReadMmio32(kGcBase1, regCP_MES_GP3_LO, &version);
    std::fprintf(stderr,
                 "%s MES scheduler initialized doorbell=0x%x ring=0x%llx "
                 "version=0x%08x\n",
                 TracePrefix(options), kMesSchedulerDoorbell,
                 static_cast<unsigned long long>(scheduler_layout.ring_gpu),
                 version);
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t MapLegacyQueueWithMes(const DirectQueuePlatform& platform,
                                   DirectQueueState& queue,
                                   const DirectQueueLayout& layout,
                                   uint64_t framebuffer_base,
                                   const DirectQueueOptions& options) {
  hsa_status_t status = EnsureMesScheduler(platform, framebuffer_base, options);
  if (status != HSA_STATUS_SUCCESS) return status;

  std::lock_guard<std::mutex> lock(MesSchedulerMutex());
  auto it = MesSchedulers().find(&platform);
  if (it == MesSchedulers().end() || !it->second.initialized) {
    return HSA_STATUS_ERROR;
  }
  MesSchedulerState& state = it->second;
  std::array<uint32_t, kMesApiFrameDwords> frame{};
  frame[0] = MesHeader(kMesOpcodeAddQueue);
  frame[18] = queue.doorbell_index;
  PutU64(frame.data(), 20, layout.mqd_gpu);
  PutU64(frame.data(), 22, layout.wptr_gpu);
  frame[28] = kMesQueueTypeCompute;
  frame[37] = kMesAddQueueMapLegacyKq;
  frame[50] = DirectQueuePipe(queue.queue_index);
  frame[51] = DirectQueueHqd(queue.queue_index);
  status = SubmitMesApiFrame(platform, state, frame, kMesApiStatusAddQueueDw,
                             options, "ADD_QUEUE");
  if (status != HSA_STATUS_SUCCESS) return status;

  queue.mes_backed = true;
  if (options.trace) {
    std::fprintf(stderr,
                 "%s MES mapped compute queue qid=%u index=%u pipe=%u hqd=%u "
                 "doorbell=0x%x mqd=0x%llx wptr=0x%llx\n",
                 TracePrefix(options), queue.queue_id, queue.queue_index,
                 DirectQueuePipe(queue.queue_index),
                 DirectQueueHqd(queue.queue_index), queue.doorbell_index,
                 static_cast<unsigned long long>(layout.mqd_gpu),
                 static_cast<unsigned long long>(layout.wptr_gpu));
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t UnmapLegacyQueueWithMes(const DirectQueuePlatform& platform,
                                     const DirectQueueState& queue,
                                     const DirectQueueOptions& options) {
  std::lock_guard<std::mutex> lock(MesSchedulerMutex());
  auto it = MesSchedulers().find(&platform);
  if (it == MesSchedulers().end() || !it->second.initialized) {
    return HSA_STATUS_ERROR;
  }
  MesSchedulerState& state = it->second;
  std::array<uint32_t, kMesApiFrameDwords> frame{};
  frame[0] = MesHeader(kMesOpcodeRemoveQueue);
  frame[1] = queue.doorbell_index;
  frame[4] = kMesRemoveQueueUnmapLegacy;
  frame[10] = DirectQueuePipe(queue.queue_index);
  frame[11] = DirectQueueHqd(queue.queue_index);
  frame[15] = kMesQueueTypeCompute;
  hsa_status_t status = SubmitMesApiFrame(platform, state, frame,
                                          kMesApiStatusRemoveQueueDw, options,
                                          "REMOVE_QUEUE");
  if (options.trace) {
    std::fprintf(stderr,
                 "%s MES unmap compute queue qid=%u index=%u status=%u\n",
                 TracePrefix(options), queue.queue_id, queue.queue_index,
                 status);
  }
  return status;
}

}  // namespace

hsa_status_t StartMesEngine(const DirectQueuePlatform& platform,
                            uint64_t mes_entry,
                            const DirectQueueOptions& options) {
  // Release the MES engine when the firmware autoloaded the ucode but left the
  // pipes halted (macOS/Windows do not run the kernel-driver MES start). Port of
  // ring_init.py::_enable_mes_from_ucode / wddmStartMes: the only programmable
  // input needed (PSP already staged the IC_BASE) is the ucode entry PC, written
  // to CP_MES_PRGRM_CNTR_START on both pipes. All MES registers live at base_idx
  // 1 (kGcBase1), matching the Windows base_idx=1 reference.
  uint32_t mes_cntl_pre = 0;
  platform.ReadMmio32(kGcBase1, regCP_MES_CNTL, &mes_cntl_pre);
  uint32_t version_pre = 0;
  platform.ReadMmio32(kGcBase1, regCP_MES_GP3_LO, &version_pre);
  if (options.trace) {
    std::fprintf(stderr,
                 "%s StartMesEngine entry=0x%llx PC=0x%llx CP_MES_CNTL(pre)="
                 "0x%08x GP3_LO=0x%08x\n",
                 TracePrefix(options),
                 static_cast<unsigned long long>(mes_entry),
                 static_cast<unsigned long long>(mes_entry >> 2), mes_cntl_pre,
                 version_pre);
  }

  // 1. RLC_CP_SCHEDULERS: route the KIQ (me=3 pipe=KIQ hqd=0) + enable bit.
  uint32_t schedulers = 0;
  hsa_status_t status =
      platform.ReadMmio32(kGcBase1, regRLC_CP_SCHEDULERS, &schedulers);
  if (status != HSA_STATUS_SUCCESS) return status;
  schedulers &= 0xFFFFFF00u;
  schedulers |= (kMesKiqMe << 5) | (kMesKiqPipe << 3) | kMesKiqHqd |
                kRlcCpSchedulersEnable;
  status = platform.WriteMmio32(kGcBase1, regRLC_CP_SCHEDULERS, schedulers);
  if (status != HSA_STATUS_SUCCESS) return status;

  // 2. CP_MES_CNTL: clear ACTIVE, set INVALIDATE_ICACHE + PIPE0/1_RESET + HALT.
  uint32_t val = 0;
  status = platform.ReadMmio32(kGcBase1, regCP_MES_CNTL, &val);
  if (status != HSA_STATUS_SUCCESS) return status;
  val &= ~(kCpMesCntlPipe0Active | kCpMesCntlPipe1Active);
  val |= (kCpMesCntlInvalidateIcache | kCpMesCntlPipe0Reset |
          kCpMesCntlPipe1Reset | kCpMesCntlHalt);
  status = platform.WriteMmio32(kGcBase1, regCP_MES_CNTL, val);
  if (status != HSA_STATUS_SUCCESS) return status;

  // 3. Per-pipe program counter (both MES pipe0 and pipe1 share the uni_mes
  //    entry, mirroring bringup.py MES/MES1 = mes_entry).
  uint32_t active_mask = 0;
  for (uint32_t pipe = 0; pipe < 2; ++pipe) {
    status = SelectHqd(platform, kMesKiqMe, pipe, kMesKiqHqd);
    if (status != HSA_STATUS_SUCCESS) {
      DeselectHqd(platform);
      return status;
    }
    status = platform.WriteMmio32(kGcBase1, regCP_MES_PRGRM_CNTR_START,
                                  static_cast<uint32_t>(mes_entry >> 2));
    if (status == HSA_STATUS_SUCCESS) {
      status = platform.WriteMmio32(
          kGcBase1, regCP_MES_PRGRM_CNTR_START_HI,
          static_cast<uint32_t>((mes_entry >> 2) >> 32));
    }
    if (status != HSA_STATUS_SUCCESS) {
      DeselectHqd(platform);
      return status;
    }
    active_mask |= (pipe == 0) ? kCpMesCntlPipe0Active : kCpMesCntlPipe1Active;
  }
  DeselectHqd(platform);

  // 4. CP_MES_CNTL release: clear reset/halt/icache + set PIPE0/1_ACTIVE.
  status = platform.ReadMmio32(kGcBase1, regCP_MES_CNTL, &val);
  if (status != HSA_STATUS_SUCCESS) return status;
  val &= ~(kCpMesCntlInvalidateIcache | kCpMesCntlPipe0Reset |
           kCpMesCntlPipe1Reset | kCpMesCntlHalt | kCpMesCntlPipe0Active |
           kCpMesCntlPipe1Active);
  val |= active_mask;
  status = platform.WriteMmio32(kGcBase1, regCP_MES_CNTL, val);
  if (status != HSA_STATUS_SUCCESS) return status;
  platform.SleepUs(1000);

  // Liveness check: HEADER_DUMP / INSTR_PNTR should change if MES executes.
  uint32_t hdr0 = 0;
  uint32_t ip0 = 0;
  platform.ReadMmio32(kGcBase1, regCP_MES_HEADER_DUMP, &hdr0);
  platform.ReadMmio32(kGcBase1, regCP_MES_INSTR_PNTR, &ip0);
  platform.SleepUs(200000);
  uint32_t hdr1 = 0;
  uint32_t ip1 = 0;
  uint32_t mes_cntl_post = 0;
  platform.ReadMmio32(kGcBase1, regCP_MES_HEADER_DUMP, &hdr1);
  platform.ReadMmio32(kGcBase1, regCP_MES_INSTR_PNTR, &ip1);
  platform.ReadMmio32(kGcBase1, regCP_MES_CNTL, &mes_cntl_post);

  const bool pipes_active =
      (mes_cntl_post & (kCpMesCntlPipe0Active | kCpMesCntlPipe1Active)) ==
      (kCpMesCntlPipe0Active | kCpMesCntlPipe1Active);
  const bool mes_alive = (hdr0 != hdr1) || (ip1 != 0);
  if (options.trace) {
    std::fprintf(stderr,
                 "%s StartMesEngine CP_MES_CNTL(post)=0x%08x pipes_active=%s "
                 "HEADER_DUMP 0x%08x->0x%08x INSTR_PNTR 0x%08x->0x%08x MES %s\n",
                 TracePrefix(options), mes_cntl_post,
                 pipes_active ? "set" : "NOT-set", hdr0, hdr1, ip0, ip1,
                 mes_alive ? "RUNNING" : "NOT visibly executing");
  }
  return pipes_active ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR;
}

uint32_t DirectQueuePipe(uint32_t queue_index) { return queue_index / 4; }

uint32_t DirectQueueHqd(uint32_t queue_index) { return queue_index % 4; }

uint32_t DirectQueueDoorbell(uint32_t queue_index, bool mec_doorbell) {
  // Doorbell-dead amdgpu_lite transport (Linux): kDirectComputeDoorbellBase (0x20)
  // is an UNASSIGNED doorbell slot -> the CP never receives the ring
  // (DOORBELL_HIT stays 0). Use the MEC ring doorbell (0x6 + idx*2), whose NBIO
  // routing the bring-up programs. macOS/Windows keep the 0x20 slot.
  if (mec_doorbell) {
    return 0x6u + queue_index * 0x2u;
  }
  return kDirectComputeDoorbellBase + queue_index * kDirectComputeDoorbellStride;
}

DirectQueueLayout BuildDirectQueueLayout(uint64_t framebuffer_base,
                                         uint32_t queue_index) {
  return BuildQueueLayoutAt(framebuffer_base,
                            kDirectComputeBaseOffset +
                                queue_index * kDirectComputeStride);
}

DirectQueueMqd BuildPm4DirectQueueMqd(const DirectQueueLayout& layout,
                                      uint32_t doorbell_index) {
  DirectQueueMqd mqd{};
  mqd[0] = 0xC0310800;
  mqd[1] = 1;
  constexpr uint32_t kStaticThreadMgmtDwords[] = {0x17u, 0x18u, 0x1Au, 0x1Bu};
  for (uint32_t dw : kStaticThreadMgmtDwords) mqd[dw] = 0xFFFFFFFFu;
  mqd[0x20] = 7;  // compute_misc_reserved (dword 0x2C was a static_thread_mgmt_se4 slip)

  const uint64_t eop_base_shifted = layout.eop_gpu >> 8;
  mqd[0xA5] = static_cast<uint32_t>(eop_base_shifted);
  mqd[0xA6] = static_cast<uint32_t>(eop_base_shifted >> 32);
  mqd[0xA7] = 9;  // bit_length(4 KiB / 4) - 2.

  mqd[0x80] = static_cast<uint32_t>(layout.mqd_gpu) & 0xFFFFFFFCu;
  mqd[0x81] = static_cast<uint32_t>(layout.mqd_gpu >> 32);
  mqd[0x82] = 1;
  mqd[0x84] = (kCpHqdPersistentStateDefault & ~(0x3FFu << 8)) | (0x55u << 8);

  const uint64_t pq_base_shifted = layout.ring_gpu >> 8;
  mqd[0x88] = static_cast<uint32_t>(pq_base_shifted);
  mqd[0x89] = static_cast<uint32_t>(pq_base_shifted >> 32);
  mqd[0x8B] = static_cast<uint32_t>(layout.rptr_gpu) & 0xFFFFFFFCu;
  mqd[0x8C] = static_cast<uint32_t>(layout.rptr_gpu >> 32) & 0xFFFFu;
  mqd[0x8D] = static_cast<uint32_t>(layout.wptr_gpu) & 0xFFFFFFF8u;
  mqd[0x8E] = static_cast<uint32_t>(layout.wptr_gpu >> 32) & 0xFFFFu;
  mqd[0x8F] = ((doorbell_index & 0x03FFFFFFu) << 2) | (1u << 30);

  // QUEUE_SIZE (CP_HQD_PQ_CONTROL[5:0]) must track the ring buffer, else the HQD
  // wraps rptr at the encoded size while ROCr writes the full ring -> stall at the
  // boundary (the ~14-dispatch ceiling: hardcoded 9=1024dw vs kDirectComputeRingSize=8192dw).
  mqd[0x91] = (kCpHqdPqControlPm4 & ~0x3fu) |
              (QueueSizeField(kDirectComputeRingSize) & 0x3fu);
  mqd[0x95] = 0x00300000;
  mqd[0xA2] = 0x100;
  mqd[0xB8] = 1u << 15;
  return mqd;
}

hsa_status_t CreateDirectQueue(const DirectQueuePlatform& platform,
                               DirectQueueState* queue,
                               uint32_t queue_index,
                               uint64_t framebuffer_base,
                               const DirectQueueOptions& options) {
  if (queue == nullptr) return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  hsa_status_t status = platform.EnsureDoorbellAperture();
  if (status != HSA_STATUS_SUCCESS) {
    if (options.trace) {
      std::fprintf(stderr, "%s create failed: doorbell aperture status=%u\n",
                   TracePrefix(options), status);
    }
    return status;
  }

  *queue = {};
  queue->queue_index = queue_index;
  queue->queue_id = queue_index + 1;
  queue->doorbell_index = DirectQueueDoorbell(queue_index, options.mec_doorbell);
  queue->ring_size_bytes = kDirectComputeRingSize;
  if (options.trace) {
    std::fprintf(stderr,
                 "%s create qid=%u index=%u doorbell=0x%x mode=%s\n",
                 TracePrefix(options), queue->queue_id, queue->queue_index,
                 queue->doorbell_index,
                 options.use_mes_queue ? "mes" : "direct");
  }

  const uint32_t pipe = DirectQueuePipe(queue->queue_index);
  const uint32_t hqd_queue = DirectQueueHqd(queue->queue_index);
  status = SelectHqd(platform, 1, pipe, hqd_queue);
  if (status != HSA_STATUS_SUCCESS) {
    if (options.trace) {
      std::fprintf(stderr,
                   "%s create failed: select compute HQD pipe=%u hqd=%u "
                   "status=%u\n",
                   TracePrefix(options), pipe, hqd_queue, status);
    }
    *queue = {};
    return status;
  }

  uint32_t active = 0;
  status = platform.ReadMmio32(kGcBase0, regCP_HQD_ACTIVE, &active);
  if (status != HSA_STATUS_SUCCESS) {
    if (options.trace) {
      std::fprintf(stderr,
                   "%s create failed: read active pipe=%u hqd=%u status=%u\n",
                   TracePrefix(options), pipe, hqd_queue, status);
    }
    DeselectHqd(platform);
    *queue = {};
    return status;
  }
  if (active != 0 && !options.force_reclaim && !options.use_mes_queue) {
    if (options.trace) {
      std::fprintf(stderr,
                   "%s create failed: active HQD pipe=%u hqd=%u active=0x%x\n",
                   TracePrefix(options), pipe, hqd_queue, active);
    }
    DeselectHqd(platform);
    *queue = {};
    return HSA_STATUS_ERROR;
  }
  if (active != 0 && !options.use_mes_queue) {
    status = ReclaimActiveHqd(platform, *queue, options, "activate-reclaim");
    if (status != HSA_STATUS_SUCCESS) {
      DeselectHqd(platform);
      *queue = {};
      return status;
    }
  }

  DirectQueueLayout layout{};
  DirectQueueMemory queue_memory{};
  status = PrepareQueueLayout(
      platform, framebuffer_base,
      kDirectComputeBaseOffset + queue->queue_index * kDirectComputeStride,
      &layout, &queue_memory);
  if (status != HSA_STATUS_SUCCESS) {
    if (options.trace) {
      std::fprintf(stderr,
                   "%s create failed: allocate queue memory qid=%u index=%u "
                   "status=%u\n",
                   TracePrefix(options), queue->queue_id, queue->queue_index,
                   status);
    }
    DeselectHqd(platform);
    *queue = {};
    return status;
  }

  const DirectQueueMqd mqd = BuildPm4DirectQueueMqd(layout, queue->doorbell_index);

  status = ZeroQueueMemory(platform, layout);
  if (status != HSA_STATUS_SUCCESS) {
    if (options.trace) {
      std::fprintf(stderr,
                   "%s create failed: zero queue memory base_off=0x%llx "
                   "status=%u\n",
                   TracePrefix(options),
                   static_cast<unsigned long long>(layout.base_offset), status);
    }
    platform.FreeQueueMemory(&queue_memory);
    DeselectHqd(platform);
    *queue = {};
    return status;
  }

  status = WriteMqdImage(platform, layout, mqd);
  if (status != HSA_STATUS_SUCCESS) {
    if (options.trace) {
      std::fprintf(stderr,
                   "%s create failed: write MQD base_off=0x%llx status=%u\n",
                   TracePrefix(options),
                   static_cast<unsigned long long>(layout.base_offset), status);
    }
    platform.FreeQueueMemory(&queue_memory);
    DeselectHqd(platform);
    *queue = {};
    return status;
  }

  // Retain the layout + framebuffer base so SetDirectQueueScratch can patch the
  // MQD scratch fields and re-map this queue later.
  queue->layout = layout;
  queue->framebuffer_base = framebuffer_base;

  auto* ring_cpu = static_cast<volatile uint32_t*>(
      LayoutCpuPointer(platform, layout, layout.ring_offset,
                       kDirectComputeRingSize));
  auto* rptr_cpu = static_cast<volatile uint64_t*>(
      LayoutCpuPointer(platform, layout, layout.rptr_offset, sizeof(uint64_t)));
  auto* wptr_cpu = static_cast<volatile uint64_t*>(
      LayoutCpuPointer(platform, layout, layout.wptr_offset, sizeof(uint64_t)));
  volatile uint64_t* doorbell_cpu = platform.DoorbellCpuPointer(queue->doorbell_index);
  if (ring_cpu == nullptr || rptr_cpu == nullptr || wptr_cpu == nullptr ||
      doorbell_cpu == nullptr) {
    if (options.trace) {
      std::fprintf(stderr,
                   "%s create failed: CPU pointers ring=%p rptr=%p wptr=%p "
                   "doorbell=%p\n",
                   TracePrefix(options), const_cast<uint32_t*>(ring_cpu),
                   const_cast<uint64_t*>(rptr_cpu),
                   const_cast<uint64_t*>(wptr_cpu),
                   const_cast<uint64_t*>(doorbell_cpu));
    }
    platform.FreeQueueMemory(&queue_memory);
    DeselectHqd(platform);
    *queue = {};
    return HSA_STATUS_ERROR;
  }

  queue->ring_gpu = layout.ring_gpu;
  queue->wptr = 0;
  queue->memory = queue_memory;
  queue_memory = {};
  queue->ring_cpu = ring_cpu;
  queue->rptr_cpu = rptr_cpu;
  queue->wptr_cpu = wptr_cpu;
  queue->doorbell_cpu = doorbell_cpu;

  if (options.use_mes_queue) {
    DeselectHqd(platform);
    status = MapLegacyQueueWithMes(platform, *queue, layout, framebuffer_base,
                                   options);
    if (status != HSA_STATUS_SUCCESS) {
      if (options.trace) {
        std::fprintf(stderr,
                     "%s create failed: MES map qid=%u index=%u status=%u\n",
                     TracePrefix(options), queue->queue_id, queue->queue_index,
                     status);
      }
      platform.FreeQueueMemory(&queue->memory);
      *queue = {};
      return status;
    }
    return HSA_STATUS_SUCCESS;
  }

  platform.WriteMmio32(kGcBase0, regCP_HQD_ACTIVE, 0);
  platform.WriteMmio32(kGcBase0, regCP_HQD_PQ_RPTR, 0);
  platform.WriteMmio32(kGcBase0, regCP_HQD_PQ_WPTR_LO, 0);
  platform.WriteMmio32(kGcBase0, regCP_HQD_PQ_WPTR_HI, 0);
  uint32_t vmid = 0;
  platform.ReadMmio32(kGcBase0, regCP_HQD_VMID, &vmid);
  platform.WriteMmio32(kGcBase0, regCP_HQD_VMID, vmid & ~0xFu);
  uint32_t doorbell_ctl = 0;
  platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_DOORBELL_CONTROL, &doorbell_ctl);
  platform.WriteMmio32(kGcBase0, regCP_HQD_PQ_DOORBELL_CONTROL,
                       doorbell_ctl & ~0x40000000u);
  platform.WriteMmio32(kGcBase0, regCP_MQD_BASE_ADDR, mqd[0x80]);
  platform.WriteMmio32(kGcBase0, regCP_MQD_BASE_ADDR_HI, mqd[0x81]);
  platform.WriteMmio32(kGcBase0, regCP_MQD_CONTROL, mqd[0xA2]);
  platform.WriteMmio32(kGcBase0, regCP_HQD_EOP_BASE_ADDR, mqd[0xA5]);
  platform.WriteMmio32(kGcBase0, regCP_HQD_EOP_BASE_ADDR_HI, mqd[0xA6]);
  platform.WriteMmio32(kGcBase0, regCP_HQD_EOP_CONTROL, mqd[0xA7]);
  platform.WriteMmio32(kGcBase0, regCP_HQD_PQ_BASE, mqd[0x88]);
  platform.WriteMmio32(kGcBase0, regCP_HQD_PQ_BASE_HI, mqd[0x89]);
  platform.WriteMmio32(kGcBase0, regCP_HQD_PQ_RPTR_REPORT_ADDR, mqd[0x8B]);
  platform.WriteMmio32(kGcBase0, regCP_HQD_PQ_RPTR_REPORT_ADDR_HI, mqd[0x8C]);
  platform.WriteMmio32(kGcBase0, regCP_HQD_PQ_CONTROL, mqd[0x91]);
  platform.WriteMmio32(kGcBase0, regCP_HQD_PQ_WPTR_POLL_ADDR, mqd[0x8D]);
  platform.WriteMmio32(kGcBase0, regCP_HQD_PQ_WPTR_POLL_ADDR_HI, mqd[0x8E]);
  platform.WriteMmio32(kGcBase0, regCP_HQD_PQ_DOORBELL_CONTROL, mqd[0x8F]);
  {
    if (options.poll_wptr) {
      // Linux amdgpu_lite userspace doorbell is live (unlike Windows WDDM); the
      // working Python HQD runs with DOORBELL_EN set. Enable it so the doorbell
      // ring kicks the CP.
      uint32_t dbc = 0;
      if (platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_DOORBELL_CONTROL, &dbc) ==
          HSA_STATUS_SUCCESS) {
        platform.WriteMmio32(kGcBase0, regCP_HQD_PQ_DOORBELL_CONTROL,
                             dbc | 0x40000000u);
      }
    }
  }
  platform.WriteMmio32(kGcBase0, regCP_HQD_PERSISTENT_STATE, mqd[0x84]);
  {
    // Full HQD register set the working Python HQD writes (ring_init.py:997-1001)
    // but the C++ direct activation omits. The firmware dequeue leaves these
    // cleared -> the CP fetches+drains the ring but the dispatch launches NO
    // waves (spi=0, data=0). Match Python so waves launch. Env-gated.
    if (options.poll_wptr) {
      platform.WriteMmio32(kGcBase0, 0x1FAEu, 0x2u);         // CP_HQD_PIPE_PRIORITY
      platform.WriteMmio32(kGcBase0, 0x1FAFu, 0xFu);         // CP_HQD_QUEUE_PRIORITY
      platform.WriteMmio32(kGcBase0, 0x1FB0u, 0x111u);       // CP_HQD_QUANTUM
      platform.WriteMmio32(kGcBase0, 0x1FBEu, 0x300000u);    // CP_HQD_IB_CONTROL 3<<20
      platform.WriteMmio32(kGcBase0, 0x1FC9u, 0x20004000u);  // CP_HQD_HQ_STATUS0
      platform.WriteMmio32(kGcBase0, 0x1FDEu, 0x0u);         // CP_HQD_AQL_CONTROL
    }
  }
  if (options.trace_verbose) {
    uint32_t mqd_base = 0;
    uint32_t mqd_base_hi = 0;
    uint32_t pq_base = 0;
    uint32_t pq_base_hi = 0;
    uint32_t pq_control = 0;
    uint32_t doorbell_control = 0;
    uint32_t persistent = 0;
    uint32_t selected_vmid = 0;
    uint32_t active_before = 0;
    platform.ReadMmio32(kGcBase0, regCP_HQD_ACTIVE, &active_before);
    platform.ReadMmio32(kGcBase0, regCP_MQD_BASE_ADDR, &mqd_base);
    platform.ReadMmio32(kGcBase0, regCP_MQD_BASE_ADDR_HI, &mqd_base_hi);
    platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_BASE, &pq_base);
    platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_BASE_HI, &pq_base_hi);
    platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_CONTROL, &pq_control);
    platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_DOORBELL_CONTROL, &doorbell_control);
    platform.ReadMmio32(kGcBase0, regCP_HQD_PERSISTENT_STATE, &persistent);
    platform.ReadMmio32(kGcBase0, regCP_HQD_VMID, &selected_vmid);
    std::fprintf(stderr,
                 "%s pre-active readback qid=%u active=0x%x "
                 "mqd=0x%08x:%08x pq=0x%08x:%08x pq_control=0x%08x "
                 "doorbell_control=0x%08x persistent=0x%08x vmid=0x%08x\n",
                 TracePrefix(options), queue->queue_id, active_before, mqd_base_hi,
                 mqd_base, pq_base_hi, pq_base, pq_control, doorbell_control,
                 persistent, selected_vmid);
  }
  // Enable wptr polling BEFORE activating: the CP latches poll behavior at the
  // CP_HQD_ACTIVE 0->1 edge. The working Python direct HQD activates with
  // CP_PQ_WPTR_POLL_CNTL=0x1 (RLC default); the C++ reclaim leaves it 0, so the
  // CP never polls the in-memory wptr the MMIO poke updates on the dead-doorbell
  // amdgpu_lite transport (cp=0). Env-gated so macOS's live doorbell is intact.
  {
    if (options.poll_wptr) {
      platform.WriteMmio32(kGcBase0, regCP_PQ_WPTR_POLL_CNTL, 1u);
    }
  }
  platform.WriteMmio32(kGcBase0, regCP_HQD_ACTIVE, 1);
  if (options.trace_verbose) {
    uint32_t active_immediate = 0;
    uint32_t pq_control = 0;
    uint32_t doorbell_control = 0;
    platform.ReadMmio32(kGcBase0, regCP_HQD_ACTIVE, &active_immediate);
    platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_CONTROL, &pq_control);
    platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_DOORBELL_CONTROL, &doorbell_control);
    std::fprintf(stderr,
                 "%s post-active-write readback qid=%u active=0x%x "
                 "pq_control=0x%08x doorbell_control=0x%08x\n",
                 TracePrefix(options), queue->queue_id, active_immediate, pq_control,
                 doorbell_control);
  }

  platform.SleepUs(options.activate_sleep_us);
  uint32_t post_active = 0;
  status = platform.ReadMmio32(kGcBase0, regCP_HQD_ACTIVE, &post_active);
  if (status != HSA_STATUS_SUCCESS) {
    DeselectHqd(platform);
    platform.FreeQueueMemory(&queue->memory);
    *queue = {};
    return status;
  }
  if (post_active == 0) {
    if (options.trace) {
      uint32_t mqd_base = 0;
      uint32_t mqd_base_hi = 0;
      uint32_t eop_base = 0;
      uint32_t eop_base_hi = 0;
      uint32_t eop_control = 0;
      uint32_t pq_base = 0;
      uint32_t pq_base_hi = 0;
      uint32_t pq_control = 0;
      uint32_t doorbell_control = 0;
      uint32_t rptr_report = 0;
      uint32_t rptr_report_hi = 0;
      uint32_t wptr_poll = 0;
      uint32_t wptr_poll_hi = 0;
      uint32_t persistent = 0;
      uint32_t selected_vmid = 0;
      uint32_t rptr = 0;
      uint32_t wptr = 0;
      uint32_t wptr_hi = 0;
      platform.ReadMmio32(kGcBase0, regCP_MQD_BASE_ADDR, &mqd_base);
      platform.ReadMmio32(kGcBase0, regCP_MQD_BASE_ADDR_HI, &mqd_base_hi);
      platform.ReadMmio32(kGcBase0, regCP_HQD_EOP_BASE_ADDR, &eop_base);
      platform.ReadMmio32(kGcBase0, regCP_HQD_EOP_BASE_ADDR_HI, &eop_base_hi);
      platform.ReadMmio32(kGcBase0, regCP_HQD_EOP_CONTROL, &eop_control);
      platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_BASE, &pq_base);
      platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_BASE_HI, &pq_base_hi);
      platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_CONTROL, &pq_control);
      platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_DOORBELL_CONTROL, &doorbell_control);
      platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_RPTR_REPORT_ADDR, &rptr_report);
      platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_RPTR_REPORT_ADDR_HI, &rptr_report_hi);
      platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_WPTR_POLL_ADDR, &wptr_poll);
      platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_WPTR_POLL_ADDR_HI, &wptr_poll_hi);
      platform.ReadMmio32(kGcBase0, regCP_HQD_PERSISTENT_STATE, &persistent);
      platform.ReadMmio32(kGcBase0, regCP_HQD_VMID, &selected_vmid);
      platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_RPTR, &rptr);
      platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_WPTR_LO, &wptr);
      platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_WPTR_HI, &wptr_hi);
      std::fprintf(stderr,
                   "%s activate failed qid=%u index=%u me=1 pipe=%u hqd=%u "
                   "doorbell=0x%x base_off=0x%llx ring=0x%llx active=0x0\n",
                   TracePrefix(options), queue->queue_id, queue->queue_index, pipe,
                   hqd_queue, queue->doorbell_index,
                   static_cast<unsigned long long>(layout.base_offset),
                   static_cast<unsigned long long>(layout.ring_gpu));
      std::fprintf(stderr,
                   "%s failed readback mqd=0x%08x:%08x eop=0x%08x:%08x "
                   "eop_ctl=0x%08x pq=0x%08x:%08x pq_ctl=0x%08x "
                   "doorbell_ctl=0x%08x rptr_report=0x%08x:%08x "
                   "wptr_poll=0x%08x:%08x persistent=0x%08x vmid=0x%08x "
                   "rptr=0x%x wptr=0x%08x:%08x\n",
                   TracePrefix(options), mqd_base_hi, mqd_base, eop_base_hi, eop_base,
                   eop_control, pq_base_hi, pq_base, pq_control, doorbell_control,
                   rptr_report_hi, rptr_report, wptr_poll_hi, wptr_poll, persistent,
                   selected_vmid, rptr, wptr_hi, wptr);
    }
    DeselectHqd(platform);
    platform.FreeQueueMemory(&queue->memory);
    *queue = {};
    return HSA_STATUS_ERROR;
  }

  if (options.trace) {
    uint32_t pq_base = 0;
    uint32_t pq_base_hi = 0;
    uint32_t pq_control = 0;
    uint32_t doorbell_control = 0;
    uint32_t rptr = 0;
    uint32_t wptr = 0;
    uint32_t wptr_hi = 0;
    platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_BASE, &pq_base);
    platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_BASE_HI, &pq_base_hi);
    platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_CONTROL, &pq_control);
    platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_DOORBELL_CONTROL, &doorbell_control);
    platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_RPTR, &rptr);
    platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_WPTR_LO, &wptr);
    platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_WPTR_HI, &wptr_hi);
    std::fprintf(stderr,
                 "%s activate qid=%u index=%u me=1 pipe=%u hqd=%u "
                 "doorbell=0x%x base_off=0x%llx ring=0x%llx active=0x%x "
                 "pq_base=0x%08x:%08x pq_control=0x%08x doorbell_control=0x%08x "
                 "rptr=0x%x wptr=0x%08x:%08x\n",
                 TracePrefix(options), queue->queue_id, queue->queue_index, pipe,
                 hqd_queue, queue->doorbell_index,
                 static_cast<unsigned long long>(layout.base_offset),
                 static_cast<unsigned long long>(layout.ring_gpu), post_active, pq_base_hi,
                 pq_base, pq_control, doorbell_control, rptr, wptr_hi, wptr);
  }
  DeselectHqd(platform);

  return HSA_STATUS_SUCCESS;
}

hsa_status_t DestroyDirectQueue(const DirectQueuePlatform& platform,
                                DirectQueueState& queue,
                                const DirectQueueOptions& options) {
  if (queue.queue_id == 0) return HSA_STATUS_SUCCESS;
  if (options.skip_destroy) {
    if (options.trace) {
      std::fprintf(stderr, "%s destroy skipped qid=%u index=%u\n",
                   TracePrefix(options), queue.queue_id, queue.queue_index);
    }
    return HSA_STATUS_SUCCESS;
  }
  if (queue.mes_backed) {
    hsa_status_t status = UnmapLegacyQueueWithMes(platform, queue, options);
    const hsa_status_t free_status = platform.FreeQueueMemory(&queue.memory);
    if (status == HSA_STATUS_SUCCESS) status = free_status;
    queue = {};
    return status;
  }

  const uint32_t pipe = DirectQueuePipe(queue.queue_index);
  const uint32_t hqd_queue = DirectQueueHqd(queue.queue_index);
  hsa_status_t status = SelectHqd(platform, 1, pipe, hqd_queue);
  if (status != HSA_STATUS_SUCCESS) return status;
  uint32_t active = 0;
  status = platform.ReadMmio32(kGcBase0, regCP_HQD_ACTIVE, &active);
  if (status != HSA_STATUS_SUCCESS) {
    DeselectHqd(platform);
    return status;
  }

  if (options.use_firmware_dequeue) {
    platform.WriteMmio32(kGcBase0, regCP_HQD_DEQUEUE_REQUEST,
                         kCpHqdDequeueResetWaves);
    platform.WriteMmio32(kGcBase0, regSPI_COMPUTE_QUEUE_RESET, 1);
    for (uint32_t i = 0; i < 100; ++i) {
      platform.ReadMmio32(kGcBase0, regCP_HQD_ACTIVE, &active);
      if (active == 0) break;
      platform.SleepUs(1000);
    }
    platform.WriteMmio32(kGcBase0, regCP_HQD_DEQUEUE_REQUEST, 0);
    // The firmware dequeue does not reliably clear the doorbell-enable bit
    // (gfx12: "may not stick"), which would leave WaitForDirectHqdIdle's
    // doorbell-off condition unmet -> timeout -> the queue is reported as not
    // destroyed (the historical macOS flake that forced SKIP_DESTROY, which then
    // leaks queues across processes). Clear it explicitly, mirroring the disable
    // path below, so the destroy is deterministic rather than reliant on the
    // dequeue happening to clear it.
    uint32_t dequeue_doorbell_ctl = 0;
    platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_DOORBELL_CONTROL,
                        &dequeue_doorbell_ctl);
    platform.WriteMmio32(kGcBase0, regCP_HQD_PQ_DOORBELL_CONTROL,
                         dequeue_doorbell_ctl & ~0x40000000u);
    status = WaitForDirectHqdIdle(platform, pipe, hqd_queue, "destroy", options);
    if (status != HSA_STATUS_SUCCESS) {
      DeselectHqd(platform);
      return status;
    }
  } else {
    platform.WriteMmio32(kGcBase0, regCP_HQD_ACTIVE, 0);
    platform.WriteMmio32(kGcBase0, regCP_HQD_PQ_RPTR, 0);
    platform.WriteMmio32(kGcBase0, regCP_HQD_PQ_WPTR_LO, 0);
    platform.WriteMmio32(kGcBase0, regCP_HQD_PQ_WPTR_HI, 0);
    uint32_t doorbell_ctl = 0;
    platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_DOORBELL_CONTROL, &doorbell_ctl);
    platform.WriteMmio32(kGcBase0, regCP_HQD_PQ_DOORBELL_CONTROL,
                         doorbell_ctl & ~0x40000000u);
  }
  if (options.trace) {
    uint32_t post_active = 0;
    uint32_t rptr = 0;
    platform.ReadMmio32(kGcBase0, regCP_HQD_ACTIVE, &post_active);
    platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_RPTR, &rptr);
    std::fprintf(stderr,
                 "%s destroy qid=%u index=%u pipe=%u hqd=%u mode=%s "
                 "pre_active=0x%x post_active=0x%x rptr=0x%x\n",
                 TracePrefix(options), queue.queue_id, queue.queue_index, pipe,
                 hqd_queue, options.use_firmware_dequeue ? "dequeue" : "disable",
                 active, post_active, rptr);
  }
  DeselectHqd(platform);
  const hsa_status_t free_status = platform.FreeQueueMemory(&queue.memory);
  queue = {};
  return free_status;
}

hsa_status_t SetDirectQueueScratch(const DirectQueuePlatform& platform,
                                   DirectQueueState& queue,
                                   uint64_t scratch_base_256,
                                   uint32_t tmpring_size,
                                   const DirectQueueOptions& options) {
  // The per-dispatch SET_SH_REG of COMPUTE_DISPATCH_SCRATCH_BASE/TMPRING is
  // clobbered when the MEC restores the queue's saved persistent state from the
  // MQD (which is zero) -> FLAT_SCRATCH stays 0 -> GCVM permission fault at VA0
  // for any spilling kernel. Patch the MQD's saved scratch fields so the restore
  // carries the real values. v12 compute MQD dwords: compute_dispatch_scratch_
  // base_lo/hi = 0x11/0x12 (backing VA >> 8), compute_tmpring_size = 0x19.
  if (ShMemApertureEnabled()) {  // both MES + direct paths: SH_MEM needed for any scratch
    hsa_status_t sh = ProgramShMemAllVmids(platform);
    if (options.trace) {
      std::fprintf(stderr,
                   "%s set-scratch program-shmem all-vmids bases=0x%08x "
                   "config=0x%08x status=%u\n",
                   TracePrefix(options), kShMemBasesGfx12,
                   kShMemConfigGfx12, sh);
    }
    if (sh != HSA_STATUS_SUCCESS) return sh;
  }
  hsa_status_t status = WriteLayoutMemory32(
      platform, queue.layout, queue.layout.mqd_offset + 0x11 * 4,
      static_cast<uint32_t>(scratch_base_256));
  if (status != HSA_STATUS_SUCCESS) return status;
  status = WriteLayoutMemory32(
      platform, queue.layout, queue.layout.mqd_offset + 0x12 * 4,
      static_cast<uint32_t>(scratch_base_256 >> 32));
  if (status != HSA_STATUS_SUCCESS) return status;
  status = WriteLayoutMemory32(
      platform, queue.layout, queue.layout.mqd_offset + 0x19 * 4, tmpring_size);
  if (status != HSA_STATUS_SUCCESS) return status;
  std::atomic_thread_fence(std::memory_order_seq_cst);
  status = platform.FlushHdp();
  if (status != HSA_STATUS_SUCCESS) return status;

  if (options.trace) {
    std::fprintf(stderr,
                 "%s set-scratch qid=%u base256=0x%llx tmpring=0x%x mes=%d\n",
                 TracePrefix(options), queue.queue_id,
                 static_cast<unsigned long long>(scratch_base_256), tmpring_size,
                 queue.mes_backed ? 1 : 0);
  }

  // Re-activate so the MEC reloads the patched MQD persistent state. For an
  // MES-backed queue that is REMOVE_QUEUE + ADD_QUEUE(map_legacy). (A direct
  // HQD would need a dequeue + re-activate; not used on the MES path.)
  if (queue.mes_backed) {
    // Skip the REMOVE/ADD remap when gated (#57): it resets the compute pipe's
    // wptr-poll and freezes the already-advancing HQD, and scratch is programmed
    // per-dispatch via SET_SH_REG (not via the MQD reload the remap would force,
    // see amd_windows_aql_queue.cpp:763-767). The MQD patch above still ran, so
    // any later state reload carries the correct scratch base.
    const bool skip_remap = SkipScratchRemapEnabled();
    if (!skip_remap) {
      status = UnmapLegacyQueueWithMes(platform, queue, options);
      if (status != HSA_STATUS_SUCCESS) return status;
      status = MapLegacyQueueWithMes(platform, queue, queue.layout,
                                     queue.framebuffer_base, options);
      if (status != HSA_STATUS_SUCCESS) return status;
    } else if (options.trace) {
      std::fprintf(stderr,
                   "%s set-scratch skip-remap (no REMOVE/ADD) qid=%u\n",
                   TracePrefix(options), queue.queue_id);
    }
    // The MES REMOVE/ADD remap resets CP_PQ_WPTR_POLL_CNTL on the compute
    // pipe, so the doorbell-dead MMIO wptr poke stops advancing the CP
    // (rptr freezes, no GPUVM fault). Re-assert the wptr-poll enable on the
    // re-mapped compute HQD so the poke advances the CP again. Windows-only.
    if (MesMmioWptrPokeEnabled()) {
      const uint32_t cpipe = DirectQueuePipe(queue.queue_index);
      const uint32_t chqd = DirectQueueHqd(queue.queue_index);
      if (SelectHqd(platform, 1, cpipe, chqd) == HSA_STATUS_SUCCESS) {
        platform.WriteMmio32(kGcBase0, regCP_PQ_WPTR_POLL_CNTL, 1u);
        DeselectHqd(platform);
        if (options.trace) {
          std::fprintf(stderr,
                       "%s set-scratch reassert wptr-poll me=1 pipe=%u hqd=%u\n",
                       TracePrefix(options), cpipe, chqd);
        }
      }
    }
    // Re-assert SH_MEM after the remap (proven recipe re-asserts post-map)
    // and clear the GCVM fault status so the per-submit probe captures a
    // FRESH scratch fault from the first spilling dispatch.
    if (ShMemApertureEnabled()) {
      status = ProgramShMemAllVmids(platform);
      if (status != HSA_STATUS_SUCCESS) return status;
      platform.WriteMmio32(kGcBase0, regGCVM_L2_PROTECTION_FAULT_STATUS, 0);
      if (options.trace) TraceScratchFault(platform, "post-remap", options);
    }
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t SubmitDirectQueue(const DirectQueuePlatform& platform,
                               DirectQueueState& queue,
                               const uint32_t* pm4,
                               size_t dword_count,
                               const DirectQueueOptions& options) {
  if (queue.queue_id == 0 || queue.ring_cpu == nullptr || queue.wptr_cpu == nullptr ||
      queue.doorbell_cpu == nullptr || pm4 == nullptr || dword_count == 0) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  const uint64_t ring_dw = queue.ring_size_bytes / sizeof(uint32_t);
  // A PM4 packet must not straddle the ring-end boundary: the CP reads each
  // packet as a contiguous run from rptr, so a dispatch block that wraps past
  // the end is split into a malformed run and the CP faults (HSA 0x1000).
  // Mirror the SubmitRingPm4 guard: when the block would cross the end, NOP-pad
  // from the current offset to the ring end, advance wptr to the boundary, then
  // write the real block at offset 0. Unlike the MES scheduler ring (whose CP
  // treats the pre-wrap remainder as already-consumed and tolerates a zero
  // fill), this is a live HQD-backed compute PQ, so the pad must be a real PM4
  // TYPE-3 NOP packet — a zero dword decodes as a TYPE-0 register write and
  // would itself fault. A single dispatch must fit the ring (dword_count <=
  // ring_dw); the observed 69-dword dispatches are far below the 1024-dword
  // ring, but reject the pathological case rather than corrupt the ring.
  if (dword_count > ring_dw) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  uint64_t wptr = queue.wptr;
  uint64_t start = wptr % ring_dw;
  if (start + dword_count > ring_dw) {
    const uint64_t tail = ring_dw - start;  // 1 <= tail < ring_dw (start < ring_dw)
    if (tail >= 2) {
      // PM4 TYPE-3 IT_NOP filling exactly `tail` dwords: 1 header + (tail-1)
      // ignored body dwords. Packet3's count field encodes (body_dwords - 1) =
      // tail - 2. This matches the gfx1201-proven NOP (0xC0001000 = count 0 =>
      // 2 dwords, used by the phase-9 probe) and the mainline ROCr nop-pad
      // (amd_aql_queue.cpp: PM4_HDR(NOP, n) header + zero body). Advance wptr to
      // the ring boundary so the real block lands contiguously at offset 0.
      queue.ring_cpu[start] = Packet3(kPacket3Nop, static_cast<uint32_t>(tail - 2));
      for (uint64_t i = 1; i < tail; ++i) {
        queue.ring_cpu[start + i] = 0;
      }
      wptr += tail;  // advance past the NOP tail so wptr % ring_dw == 0
    } else {
      // tail == 1: a TYPE-3 NOP needs >= 2 dwords and cannot fit the single
      // remaining slot, so fill it with a 1-dword TYPE-2 NOP (the CP consumes a
      // TYPE-2 packet as a no-op). Only reachable when a prior dispatch left
      // wptr exactly one dword short of the boundary (possible with mixed
      // dispatch sizes; the uniform 69-dword stride never hits it). The TYPE-2
      // encoding is not yet hardware-confirmed on gfx1201 (no wrap exercises it
      // in the current tests) — revisit if a mixed-size workload faults here.
      queue.ring_cpu[start] = 0x80000000u;  // PM4 TYPE-2 NOP (single dword)
      wptr += 1;  // advance to the ring boundary (wptr % ring_dw == 0)
    }
    start = 0;
  }
  for (size_t i = 0; i < dword_count; ++i) {
    queue.ring_cpu[(start + i) % ring_dw] = pm4[i];
  }
  std::atomic_thread_fence(std::memory_order_release);
  const uint64_t new_wptr = wptr + dword_count;
  *queue.wptr_cpu = new_wptr;
  std::atomic_thread_fence(std::memory_order_release);
  hsa_status_t status = platform.FlushHdp();
  if (status != HSA_STATUS_SUCCESS) return status;

  const uint32_t pipe = DirectQueuePipe(queue.queue_index);
  const uint32_t hqd_queue = DirectQueueHqd(queue.queue_index);
  if (options.trace) {
    const size_t sample = std::min<size_t>(dword_count, 8);
    std::fprintf(stderr,
                 "%s submit qid=%u index=%u pipe=%u hqd=%u doorbell=0x%x "
                 "wptr=%llu new_wptr=%llu dwords=%zu first_pm4=",
                 TracePrefix(options), queue.queue_id, queue.queue_index, pipe,
                 hqd_queue, queue.doorbell_index,
                 static_cast<unsigned long long>(wptr),
                 static_cast<unsigned long long>(new_wptr), dword_count);
    for (size_t i = 0; i < sample; ++i) {
      std::fprintf(stderr, "%s0x%08x", i == 0 ? "" : ",", pm4[i]);
    }
    std::fprintf(stderr, "\n");
  }

  if (queue.mes_backed) {
    // Windows doorbell-dead workaround (ROCR_WINDOWS_MES_MMIO_WPTR): the MES
    // mapped this compute queue onto a hardware MEC HQD slot via ADD_QUEUE/
    // MAP_LEGACY, which uses frame[50]/frame[51] = DirectQueuePipe/Hqd(
    // queue_index) on the MEC engine (me=1) -- the SAME slot a DIRECT queue of
    // the same queue_index occupies. The doorbell (0x20) is dead on this WDDM/
    // passthrough setup, so ringing it alone never makes the CP fetch the NOP.
    // The only mechanism proven to advance an HQD here is the direct MMIO wptr
    // poke (SubmitDirectQueue's me=1 path) and the scheduler-ring poke
    // (SubmitMesApiFrameOnRing's me=3 KIQ path). Mirror it for the mapped
    // compute HQD: SelectHqd(me=1, pipe, hqd) -> write CP_HQD_PQ_WPTR_LO/HI =
    // new_wptr so the MEC's CP picks up the wptr and fetches the ring without a
    // doorbell event. pipe/hqd_queue above are exactly DirectQueuePipe/Hqd(
    // queue_index), i.e. the slot the MES mapped. Default (flag unset) keeps the
    // doorbell-only behavior so macOS/Linux are unchanged.
    // Accepts ROCR_WINDOWS_MES_MMIO_WPTR (Windows) or generic ROCR_MES_MMIO_WPTR.
    const bool mes_mmio_wptr_poke = MesMmioWptrPokeEnabled();
    if (mes_mmio_wptr_poke) {
      status = SelectHqd(platform, 1, pipe, hqd_queue);
      if (status == HSA_STATUS_SUCCESS) {
        // compute HQD poll_cntl re-assert (non-scratch): the doorbell is
        // dead, so the CP must poll the in-memory wptr the poke updates.
        // The scratch path re-asserts this, but a non-scratch dispatch (a
        // blit) skips it -> poll_cntl=0 -> CP freezes mid-ring (rptr short
        // of wptr, no fault). Mirror the scratch-path re-assert here.
        platform.WriteMmio32(kGcBase0, regCP_PQ_WPTR_POLL_CNTL, 1u);
        status = platform.WriteMmio32(kGcBase0, regCP_HQD_PQ_WPTR_LO,
                                      static_cast<uint32_t>(new_wptr));
        if (status == HSA_STATUS_SUCCESS) {
          status = platform.WriteMmio32(kGcBase0, regCP_HQD_PQ_WPTR_HI,
                                        static_cast<uint32_t>(new_wptr >> 32));
        }
      }
      if (options.trace) {
        // Diagnostics: did the poke latch the compute HQD wptr, and does its
        // RPTR advance? RPTR advancing + compute NOP fence reaching 1 = win.
        uint32_t poke_active = 0;
        uint32_t poke_rptr = 0;
        uint32_t poke_wptr = 0;
        uint32_t poke_wptr_hi = 0;
        uint32_t poke_poll = 0;
        platform.ReadMmio32(kGcBase0, regCP_HQD_ACTIVE, &poke_active);
        platform.ReadMmio32(kGcBase0, regCP_PQ_WPTR_POLL_CNTL, &poke_poll);
        platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_RPTR, &poke_rptr);
        platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_WPTR_LO, &poke_wptr);
        platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_WPTR_HI, &poke_wptr_hi);
        std::fprintf(stderr,
                     "%s MES-backed mmio-wptr poke me=1 pipe=%u hqd=%u "
                     "new_wptr=%llu active=0x%x rptr=0x%x wptr=0x%08x:%08x "
                     "poll_cntl=0x%x status=%u\n",
                     TracePrefix(options), pipe, hqd_queue,
                     static_cast<unsigned long long>(new_wptr), poke_active,
                     poke_rptr, poke_wptr_hi, poke_wptr, poke_poll, status);
        // Settled stall+fault probe (#57): runs while the compute HQD is
        // still GRBM-selected (CP_HQD_PQ_RPTR readable). Polls the read
        // pointer until the CP drains to new_wptr or parks (settles), then
        // reads GCVM fault + GRBM_STATUS + ring[rptr..+8] AT that point -- so
        // the scratch dispatch's own probe captures the 1019 hang rather than
        // sampling at rptr=966 mid-consume. Merges the old TraceStallMaxInfo +
        // post-sleep TraceScratchFault into one settled read.
        TraceSettledStall(platform, queue, new_wptr, "mes-scratch", options);
      }
      DeselectHqd(platform);
      if (status != HSA_STATUS_SUCCESS) return status;
    }
    *queue.doorbell_cpu = new_wptr;
    queue.wptr = new_wptr;
    if (options.trace) {
      const uint64_t rptr =
          queue.rptr_cpu != nullptr ? *queue.rptr_cpu : 0;
      std::fprintf(stderr,
                   "%s submit MES-backed qid=%u doorbell=0x%x "
                   "wptr=%llu rptr=%llu\n",
                   TracePrefix(options), queue.queue_id, queue.doorbell_index,
                   static_cast<unsigned long long>(new_wptr),
                   static_cast<unsigned long long>(rptr));
    }
    // The settled stall+fault probe above (TraceSettledStall, under the
    // selected HQD) already captured the scratch dispatch hang at its settled
    // read pointer; the old fixed-3ms post-deselect probe only ever sampled
    // mid-consume (rptr=966) and is removed.
    return HSA_STATUS_SUCCESS;
  }

  status = SelectHqd(platform, 1, pipe, hqd_queue);
  if (status != HSA_STATUS_SUCCESS) return status;
  uint32_t selected_active = 0;
  status = platform.ReadMmio32(kGcBase0, regCP_HQD_ACTIVE, &selected_active);
  if (status != HSA_STATUS_SUCCESS) {
    DeselectHqd(platform);
    return status;
  }
  if (selected_active == 0) {
    if (options.trace) {
      std::fprintf(stderr,
                   "%s submit rejected inactive HQD qid=%u index=%u pipe=%u "
                   "hqd=%u doorbell=0x%x\n",
                   TracePrefix(options), queue.queue_id, queue.queue_index, pipe,
                   hqd_queue, queue.doorbell_index);
    }
    DeselectHqd(platform);
    return HSA_STATUS_ERROR;
  }
  // Optionally skip the MMIO wptr poke: tinygrad/Linux deliver wptr only via the
  // VRAM wptr + doorbell value, and writing CP_HQD_PQ_WPTR_* on a live HQD races
  // the CP's own use of the context window. Default preserves the existing poke;
  // ROCR_MACOS_DIRECT_QUEUE_MMIO_WPTR=0 skips it (doorbell value latches wptr).
  // Linux/vfio doorbell-dead: the direct HQD keeps DOORBELL_EN clear and the
  // firmware CP_PQ_WPTR_POLL_CNTL default is 0 under amdgpu_lite (Windows RLC
  // left it 0x1, the sole reason the poke worked there). Without polling, the
  // MMIO wptr poke never makes the CP fetch (cp=0, hw_rptr stuck). Enable wptr
  // polling so the CP picks up the in-memory wptr, mirroring the mes_backed
  // submit. Gated (default off) so macOS's live-doorbell direct path is intact.
  {
    if (options.poll_wptr) {
      platform.WriteMmio32(kGcBase0, regCP_PQ_WPTR_POLL_CNTL, 1u);
    }
  }
  const char* mmio_wptr_env = std::getenv("ROCR_MACOS_DIRECT_QUEUE_MMIO_WPTR");
  const bool skip_mmio_wptr =
      mmio_wptr_env != nullptr && std::strcmp(mmio_wptr_env, "0") == 0;
  if (!skip_mmio_wptr) {
    status = platform.WriteMmio32(kGcBase0, regCP_HQD_PQ_WPTR_LO,
                                  static_cast<uint32_t>(new_wptr));
    if (status == HSA_STATUS_SUCCESS) {
      status = platform.WriteMmio32(kGcBase0, regCP_HQD_PQ_WPTR_HI,
                                    static_cast<uint32_t>(new_wptr >> 32));
    }
  }
  if (options.trace) {
    uint32_t active = 0;
    uint32_t rptr = 0;
    uint32_t mmio_wptr = 0;
    uint32_t mmio_wptr_hi = 0;
    platform.ReadMmio32(kGcBase0, regCP_HQD_ACTIVE, &active);
    platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_RPTR, &rptr);
    platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_WPTR_LO, &mmio_wptr);
    platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_WPTR_HI, &mmio_wptr_hi);
    std::fprintf(stderr,
                 "%s after mmio-wptr qid=%u active=0x%x rptr=0x%x "
                 "wptr=0x%08x:%08x status=%u\n",
                 TracePrefix(options), queue.queue_id, active, rptr, mmio_wptr_hi,
                 mmio_wptr, status);
    // Direct-path scratch stall probe (#57 / macOS #15): the direct HQD had no
    // stall probe before (only the MES-backed submit did). Settle-poll the CP
    // while the compute HQD is still grbm-selected to capture where a spilling
    // wave parks -- rptr / GRBM idle bits / GCVM fault status / ring packet --
    // instead of sampling mid-consume.
    TraceSettledStall(platform, queue, new_wptr, "direct-scratch", options);
  }
  DeselectHqd(platform);
  if (status != HSA_STATUS_SUCCESS) return status;
  *queue.doorbell_cpu = new_wptr;
  queue.wptr = new_wptr;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t ReadDirectQueueRptr(const DirectQueuePlatform& platform,
                                 const DirectQueueState& queue,
                                 uint32_t* rptr) {
  if (queue.queue_id == 0 || rptr == nullptr) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  if (queue.mes_backed) {
    if (queue.rptr_cpu == nullptr) return HSA_STATUS_ERROR;
    *rptr = static_cast<uint32_t>(*queue.rptr_cpu);
    return HSA_STATUS_SUCCESS;
  }
  const uint32_t pipe = DirectQueuePipe(queue.queue_index);
  const uint32_t hqd_queue = DirectQueueHqd(queue.queue_index);
  hsa_status_t status = SelectHqd(platform, 1, pipe, hqd_queue);
  if (status != HSA_STATUS_SUCCESS) return status;
  status = platform.ReadMmio32(kGcBase0, regCP_HQD_PQ_RPTR, rptr);
  DeselectHqd(platform);
  return status;
}

}  // namespace lite
}  // namespace AMD
}  // namespace rocr
