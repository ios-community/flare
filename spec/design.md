# Architecture & Design Specification: FLARE

**Role:** Architect Directive → Senior Engineer Blueprint  
**Revision:** 1.0.0 | **Toolchain:** Rust 1.97.1 (Edition 2024)

---

## Architect Directives

- **Decoupling Rule:** `flare-core` is an independent crate maintaining `#![no_std] + alloc` compliance. The `flare-vector` and `flare-kv` crates depend exclusively on public `flare-core` abstractions. All hardware GPU interactions are strictly abstracted behind the `GpuSyncDriver` trait and implemented in `flare-ffi`.
- **Memory Ownership Constraint:** Nodes are allocated directly within a contiguous `FlatArena` addressed via 40-bit relative offsets. Raw virtual pointers (`*const T` / `*mut T`) must never be stored inside tree nodes. Higher-level modules interface with arena data via safe `ArenaRef<T>` wrappers.
- **Thread Model Requirement:** The read path is completely lock-free using 64-bit atomic operations and Hazard Eras epoch checks. The write path performs node mutations via 64-bit atomic CAS. The WAL persistence engine operates an asynchronous Leader-Follower Group Commit pipeline.
- **Feature Gating:**
  - `flare-core`: `default = ["std"]`, `std`, `alloc`.
  - `flare-vector`: `default = ["simd"]`, `simd`.
  - `flare-ffi`: `default = ["cpu"]`, `cuda`.

---

## Senior Engineer Specification

### 1. System Architecture

```
+-----------------------------------------------------------------------------------------------+
|                                    CARGO WORKSPACE LAYOUT                                     |
|                                                                                               |
|  +---------------------------+  +---------------------------+  +---------------------------+  |
|  |       flare-vector        |  |         flare-kv          |  |         flare-ffi         |  |
|  |  (IVF-PQ, Centroid Router,|  |  (Radix Attention, Prefix |  |  (C ABI, CudaSyncDriver,  |  |
|  |   ADC SIMD Distance)      |  |   Sharing, Clock Evict)   |  |   CUDA Epoch Fences)      |  |
|  +---------------------------+  +---------------------------+  +---------------------------+  |
|                │                              │                              │                |
|                └──────────────────────┬───────┴──────────────────────────────┘                |
|                                       ▼                                                       |
|  +-----------------------------------------------------------------------------------------+  |
|  |                                       flare-core                                        |  |
|  |  +-----------------------------------------------------------------------------------+  |  |
|  |  |                       Contiguous Memory Arena (A) + 4KB Slabs                     |  |  |
|  |  |  +--------------------+  +--------------------+  +-----------------------------+  |  |  |
|  |  |  | Bump Allocator     |  | 4KB Slab Pools     |  | Trait: GpuSyncDriver        |  |  |  |
|  |  |  | (Frontier Pointer) |  | (Classes 0 - 3)    |  | (CpuFallback / CudaDriver)  |  |  |  |
|  |  |  +--------------------+  +--------------------+  +-----------------------------+  |  |  |
|  |  +-----------------------------------------------------------------------------------+  |  |
|  |  | 64-bit Tagged Pointer | Adaptive Nodes (N4..N256) | Hazard Eras (Eg / Et)         |  |  |
|  |  | count_ones() Popcount | Physical Delta WAL (Async)| 56-bit Extended Leaf Inline   |  |  |
|  |  +-----------------------------------------------------------------------------------+  |  |
|  +-----------------------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------------------+
```

---

### 2. Module Structure & Responsibilities

| Crate | Path | Responsibility | Architect Constraint |
| --- | --- | --- | --- |
| `flare-core` | `crates/flare-core/src/lib.rs` | Crate root, re-exports, feature flags | `#![no_std]` compatible, `#![deny(unsafe_code)]` at root level |
| `flare-core` | `crates/flare-core/src/ptr/tagged.rs` | 64-bit Bitwise Polymorphic Tagged Pointer, Leaf Inlining | Zero UB, Safe Packing/Unpacking APIs |
| `flare-core` | `crates/flare-core/src/alloc/arena.rs` | Flat Arena byte array, bump frontier, 40-bit offsets | Confined `unsafe`, contiguous huge-page backing |
| `flare-core` | `crates/flare-core/src/alloc/slab.rs` | 4 KB Size-Classified Slab Pools (Classes 0–3) | Bitmask slot management via `count_ones` / `trailing_zeros` |
| `flare-core` | `crates/flare-core/src/tree/` | ART Nodes (`Node4`, `Node16`, `Node64`, `Node256`), CAS resizing | Lock-free traversal, popcount indexing via `count_ones()` |
| `flare-core` | `crates/flare-core/src/sync/hazard.rs` | Hazard Eras manager ($E_g, E_t$, Retired List) | Thread-safe era epoch tracking without locks |
| `flare-core` | `crates/flare-core/src/sync/gpu.rs` | `GpuSyncDriver` trait, `CpuFallbackDriver` | Zero-copy host memory abstractions |
| `flare-core` | `crates/flare-core/src/wal/delta.rs` | Physical Delta WAL, Leader-Follower Group Commit | Async flush pipeline, write ordering barrier |
| `flare-vector`| `crates/flare-vector/src/` | IVF-PQ, Centroid Tree, Shadow Re-clustering, ADC | SIMD-accelerated distance kernels with portable fallback |
| `flare-kv` | `crates/flare-kv/src/` | Radix Attention, LCP matching, 2-bit Slab Clock Eviction | Zero read-path allocation, $O(1)$ amortized memory eviction |
| `flare-ffi` | `crates/flare-ffi/src/` | C ABI bindings, `CudaSyncDriver` implementation | Strict ABI validation, FFI memory boundary isolation |

---

### 3. Concurrency, Memory & Hardware Invariants

#### A. Polymorphic 64-bit Tagged Pointer Bitwise Specification

```
 63             48 47 46    43 42                                 3 2      0
+-----------------+--+--------+------------------------------------+--------+
| Polymorphic     | T| Arena  |       Relative Array Offset        |  Node  |
| Field (16-bit)  |  | ID(4b) |          (40-bit Index)            |  Type  |
+-----------------+--+--------+------------------------------------+--------+
```

1. **Bit Layout Specification:**
   - `Type` (Bits 0..2, 3 bits): `000` = LeafInlined, `001` = LeafOffset, `010` = Node4, `011` = Node16, `100` = Node64, `101` = Node256.
   - `Offset` (Bits 3..42, 40 bits): Value range $[0, 2^{40}-1]$ (Addressing up to 1 TB per Arena instance).
   - `ArenaID` (Bits 43..46, 4 bits): Value range $[0, 15]$ (Routing across up to 16 arena instances, 16 TB aggregate).
   - `Tombstone (T)` (Bit 47, 1 bit): `1` = Logically Deleted, `0` = Active.
   - `Polymorphic Field` (Bits 48..63, 16 bits):
     - Node4 / Node16: 16-bit Inline Child Presence Bitmap.
     - Node64 / Node256: 8-bit RefCount + 8-bit Generation ID (ABA protection).
     - KV-Cache Node: 16-bit Token Sequence Length.

2. **56-bit Extended Leaf Inlining with Isolated Tombstone Routing:**
   - Bits 3..46 (44 bits): Payload Part 1 (Low 44 bits).
   - Bit 47 (1 bit): **Isolated Tombstone Flag** (Must remain isolated; data payloads cannot overwrite bit 47).
   - Bits 48..59 (12 bits): Payload Part 2 (High 12 bits).
   - Bits 60..63 (4 bits): Reserved / Unused.
   - For 64-bit identifiers, 8-bit MSB Truncation is applied on pack and zero-extended on unpack.

```rust
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct TaggedPointer(pub u64);

impl TaggedPointer {
    pub const MASK_TYPE: u64      = 0x0000_0000_0000_0007;
    pub const MASK_OFFSET: u64    = 0x0000_07FF_FFFF_FFF8;
    pub const MASK_ARENA: u64     = 0x0000_7800_0000_0000;
    pub const MASK_TOMBSTONE: u64 = 0x0000_8000_0000_0000;
    pub const MASK_POLY: u64      = 0xFFFF_0000_0000_0000;

    #[inline(always)]
    pub const fn pack(node_type: u8, offset: u64, arena_id: u8, tombstone: bool, poly: u16) -> Self {
        let t_bit = if tombstone { 1u64 } else { 0u64 };
        let raw = ((poly as u64) << 48)
            | (t_bit << 47)
            | (((arena_id as u64) & 0x0F) << 43)
            | ((offset & 0x00FF_FFFF_FFFF) << 3)
            | ((node_type as u64) & 0x07);
        Self(raw)
    }

    #[inline(always)]
    pub const fn node_type(self) -> u8 { (self.0 & Self::MASK_TYPE) as u8 }

    #[inline(always)]
    pub const fn offset(self) -> u64 { (self.0 & Self::MASK_OFFSET) >> 3 }

    #[inline(always)]
    pub const fn arena_id(self) -> u8 { ((self.0 & Self::MASK_ARENA) >> 43) as u8 }

    #[inline(always)]
    pub const fn is_tombstone(self) -> bool { (self.0 & Self::MASK_TOMBSTONE) != 0 }

    #[inline(always)]
    pub const fn polymorphic_field(self) -> u16 { (self.0 >> 48) as u16 }

    #[inline(always)]
    pub const fn pack_inline_payload(payload_56: u64, tombstone: bool) -> Self {
        let t_bit = if tombstone { 1u64 } else { 0u64 };
        let p_low = payload_56 & 0x0FFF_FFFF_FFFF;         // 44 bits
        let p_high = (payload_56 >> 44) & 0x0FFF;          // 12 bits
        let raw = (p_high << 48) | (t_bit << 47) | (p_low << 3) | 0x00; // Type 0 (LeafInline)
        Self(raw)
    }

    #[inline(always)]
    pub const fn unpack_inline_payload(self) -> u64 {
        let p_low = (self.0 & 0x0000_7FFF_FFFF_FFF8) >> 3;
        let p_high = (self.0 >> 48) & 0x0FFF;
        (p_high << 44) | p_low
    }
}
```

#### B. Popcount Index Resolution (Child Lookup Algorithm)

Using Rust's standard `#![no_std]` intrinsic `u16::count_ones()`, which compiles to a single-cycle hardware popcount instruction:

```rust
#[inline(always)]
pub fn resolve_child_index(tagged_ptr: TaggedPointer, nibble: u8) -> Option<usize> {
    if tagged_ptr.is_tombstone() {
        return None;
    }
    let presence_mask = tagged_ptr.polymorphic_field();
    let bit = 1u16 << (nibble & 0x0F);
    if (presence_mask & bit) == 0 {
        return None;
    }
    let mask_prior = bit - 1;
    let dense_index = (presence_mask & mask_prior).count_ones() as usize;
    Some(dense_index)
}
```

#### C. Contiguous Flat Arena & 4 KB Size-Classified Slab Model

- **Bump Frontier:** $H \leftarrow H + s$ executes in $O(1)$ for unallocated memory.
- **Slab Allocation Pools:** 4 KB slab chunks categorized by size classes (Class 0: `Node4`, Class 1: `Node16`, Class 2: `Node64`, Class 3: `Node256`).
- **Slot Discovery:** First free slot located instantly via `(!bitmap).trailing_zeros()`.

#### D. Physical Delta WAL Child-Before-Parent Ordering

To ensure crash consistency without complex high-level key parsing:
1. Allocate memory for child delta $A_1$ in the arena.
2. Append the WAL frame for child $A_1$ into the async group commit queue.
3. Emit a release memory barrier (`Ordering::Release`).
4. Append the WAL frame for the parent pointer modification $A_0$ into the queue.
5. Replay recovery sequentially parses frames and overlays raw bytes directly into memory via `memcpy`.

---

### 4. API Surface Contract

#### A. `flare-core::sync::gpu::GpuSyncDriver` Trait

```rust
pub trait GpuSyncDriver: Send + Sync {
    /// Publishes an epoch fence event to GPU streams to prevent reader warps from observing partial CAS writes.
    fn publish_epoch_fence(&self, epoch_id: u64) -> Result<(), FlareError>;
    
    /// Allocates host memory with zero-copy mapping properties (accessible by host and GPU via PCIe/NVLink).
    fn allocate_pinned_arena(&self, size_bytes: usize) -> Result<*mut u8, FlareError>;
    
    /// Deallocates host pinned memory.
    /// 
    /// # Safety
    /// Pointer must have been allocated via `allocate_pinned_arena` and must not be accessed concurrently.
    unsafe fn deallocate_pinned_arena(&self, ptr: *mut u8, size_bytes: usize) -> Result<(), FlareError>;
}
```

#### B. `flare-core::tree::FlareArtTree`

```rust
pub struct FlareArtTree<G: GpuSyncDriver> {
    arena: Arc<FlatArena>,
    hazard: Arc<HazardManager>,
    gpu_driver: G,
    root: AtomicU64, // Stores 64-bit TaggedPointer
}

impl<G: GpuSyncDriver> FlareArtTree<G> {
    pub fn new(arena: Arc<FlatArena>, hazard: Arc<HazardManager>, gpu_driver: G) -> Self;
    
    pub fn get(&self, key: &[u8]) -> Result<Option<u64>, FlareError>;
    pub fn insert(&self, key: &[u8], value: u64) -> Result<Option<u64>, FlareError>;
    pub fn delete(&self, key: &[u8]) -> Result<bool, FlareError>;
}
```

#### C. `flare-vector::IvfPqIndex`

```rust
pub struct IvfPqIndex<G: GpuSyncDriver> {
    centroid_tree: FlareArtTree<G>,
    codebooks: Arc<PqCodebooks>,
    dimension: usize,
    sub_vectors: usize,
}

impl<G: GpuSyncDriver> IvfPqIndex<G> {
    pub fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<SearchResult>, FlareError>;
    pub fn trigger_shadow_reclustering(&self) -> Result<(), FlareError>;
}
```

#### D. `flare-kv::RadixAttentionEngine`

```rust
pub struct RadixAttentionEngine<G: GpuSyncDriver> {
    tree: FlareArtTree<G>,
    slab_pool: Arc<SlabPool>,
}

impl<G: GpuSyncDriver> RadixAttentionEngine<G> {
    pub fn match_longest_common_prefix(&self, tokens: &[u32]) -> Result<PrefixMatch, FlareError>;
    pub fn insert_token_sequence(&self, tokens: &[u32], kv_offset: u64) -> Result<(), FlareError>;
    pub fn evict_clock_step(&self, target_slots: usize) -> Result<usize, FlareError>;
}
```

---

### 5. Documentation & Testing Strategy

- **Strict Rustdoc:** All `pub` items follow standard documentation sections: `# Summary`, `# Examples`, `# Errors`, `# Panics`, and `# Safety` (for all `unsafe` functions). CI enforces `-D warnings`.
- **Coverage Target ($\ge 95\%$):** Enforced via `cargo-llvm-cov`. Unit tests cover all bitwise packing combinations, node resizing cascades, Hazard Eras retirement queues, and WAL async group commit fault injection.
- **Regression Guardrail ($< 5\%$):** Headless Criterion benchmarks run in CI against the latest baseline. Any throughput degradation or latency regression $> 5\%$ fails the build.

---

### 6. Engineering Implementation Notes

- **Rust Edition 2024 Rules:** Leverage `let`-chains (`if let Some(x) = a && let Some(y) = b`), modern C-string literals, and explicit unsafe attribute annotations (`#[unsafe(no_mangle)]` where applicable).
- **Lint Policy:** Enforce `#![deny(clippy::all, clippy::pedantic, clippy::nursery)]`. Allow `#![allow(clippy::module_name_repetitions)]`.
- **`no_std` Isolation:** `flare-core` declares `#![no_std]` at `crates/flare-core/src/lib.rs` and exclusively imports `extern crate alloc;`.

---

## Senior Engineer Sign-Off

- [x] Architecture fully mapped across workspace crates (`flare-core`, `flare-vector`, `flare-kv`, `flare-ffi`)
- [x] Concurrency & Memory invariants formally verified with bitwise packing specifications
- [x] API surface & trait contracts defined
- [x] Strict test coverage ($\ge 95\%$) & regression guardrails ($< 5\%$) locked
