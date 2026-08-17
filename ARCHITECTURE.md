# Architecture & Design Specification: FLARE

This document describes the high-level architecture and design decisions behind FLARE (Flat Lock-free Arena Radix Engine).

## Module Overview

The project is structured as a multi-crate Cargo workspace with a strict dependency order: `flare-core` provides all memory and concurrency primitives, `flare-vector` and `flare-kv` build the two engines on top of it, and `flare-ffi` is the single integration boundary to the outside world.

```text
+-----------------------------------------------------------------------+
|                        Workspace Root (Cargo.toml)                    |
+-----------------------------------+-----------------------------------+
                                    |
                                    v
+-----------------------------------+-----------------------------------+
|                          flare-core (no_std + alloc)                  |
|   +---------------------------------------------------------------+   |
|   |  alloc/  Flat Arena, 4 KB Slab Pools (Classes 0-3), pinned    |   |
|   |          zero-copy host memory                                |   |
|   +-------------------------------+-------------------------------+   |
|   |  ptr/    64-bit Tagged Pointer, Extended Leaf Inlining,       |   |
|   |          Tombstone bit routing                                |   |
|   +-------------------------------+-------------------------------+   |
|   |  tree/   Node4/16/64/256 ART nodes, popcount child indexing,  |   |
|   |          single-CAS root publication                          |   |
|   +-------------------------------+-------------------------------+   |
|   |  sync/   Hazard Eras reclamation, GpuSyncDriver trait,        |   |
|   |          CpuFallbackDriver                                    |   |
|   +-------------------------------+-------------------------------+   |
|   |  wal/    Physical Delta WAL, Leader-Follower async group      |   |
|   |          commit, replay overlay recovery                      |   |
|   +---------------------------------------------------------------+   |
+-----------------------------------+-----------------------------------+
                                    |
                +-------------------+-----------------------+
                v                                           v
+---------------+-----------------+          +--------------+--------------+
|           flare-vector          |          |          flare-kv           |
|  - IVF-PQ index & centroid      |          |  - Radix attention engine   |
|    routing (O(log C))           |          |  - Longest Common Prefix    |
|  - K-means training             |          |    matching DAG             |
|  - Codebooks (8-bit PQ)         |          |  - Lock-free 2-bit slab     |
|  - SIMD ADC distance + scalar   |          |    clock eviction           |
|    fallback                     |          |                             |
|  - Shadow dual-tree             |          |                             |
|    re-clustering                |          |                             |
+-----------------+---------------+          +--------------+--------------+
                  |                                         |
                  +--------------------+--------------------+
                                       v
+----------------------------------------------------------------------+
|                            flare-ffi                                 |
|  - c_abi/  C function exports over opaque handles (cbindgen header)  |
|  - cuda/   CudaSyncDriver (feature = "cuda", dynamic runtime load)   |
+----------------------------------------------------------------------+
```

### 1. Core Primitives (`flare-core`)
The foundation crate, compiled as `#![no_std]` + alloc with **zero external dependencies**.
- **`alloc`** \
A contiguous `FlatArena` with $O(1)$ bump allocation for newly ingested frontiers, plus 4 KB size-classified slab pools (Node4, Node16, Node64, Node256) for instant slot recycling. Every node is addressed with a 40-bit relative offset; raw virtual pointers are never stored inside tree nodes.
- **`ptr`** \
The 64-bit polymorphic `TaggedPointer` layout — Type (3b), Offset (40b), ArenaID (4b), Tombstone (1b), Polymorphic Metadata (16b) — with bitwise pack/unpack routines and 56-bit extended leaf inlining.
- **`tree`** \
Adaptive radix nodes (`Node4`, `Node16`, `Node64`, `Node256`) with `count_ones`-based child index resolution and atomic single-CAS node resizing. Root publication is a CAS-with-retry loop (see Concurrency).
- **`sync`** \
Hazard Eras — a global epoch era, thread-local active-era tracking, and a retired queue that recycles deallocated slab slots without blocking readers. The `GpuSyncDriver` trait and its default `CpuFallbackDriver` live here.
- **`wal`** \
A binary physical delta write-ahead log enforcing a strict child-before-parent ordering barrier, with a leader-follower pipeline for asynchronous group commit and instant replay crash recovery via direct memory overlay.

### 2. Vector Search Engine (`flare-vector`)
Builds the IVF-PQ engine on top of `flare-core`. A flat radix tree routes queries to centroids in $O(\log C)$; vectors are compressed with 8-bit product quantization; asymmetric distance computation (ADC) runs on a runtime-dispatched SIMD kernel (AVX2) with a portable scalar fallback. All index state is arena-resident and published through a single atomic handoff word, so background shadow dual-tree re-clustering swaps the working snapshot lock-free.

### 3. Radix Attention Engine (`flare-kv`)
Manages LLM KV-caches. A `FlareArtTree` maps token-prefix byte keys to physical slot indices for $O(\text{depth})$ longest-common-prefix matching; a `SlabPool` provides physical 4 KB-classed slots; a lock-free 2-bit clock (bumped with `fetch_or`, `LIVE` bit included) performs sequential linear scans directly over physical slots for eviction. Reads never allocate and verify slot liveness with a double-read of the published KV offset.

### 4. Integration Boundary (`flare-ffi`)
The C ABI (`c_abi`) exposes the vector index and radix attention engine as plain C functions over opaque handles; `build.rs` regenerates `include/flare.h` via cbindgen. The `cuda` module (feature-gated) implements `GpuSyncDriver` by dynamic-loading the CUDA runtime — building never requires a CUDA toolkit, and a missing runtime surfaces as `FlareError::GpuDriverUnavailable`.

### 5. Benchmark Harness (`flare-bench`)
Criterion benchmarks for the three critical paths: 1D radix lookups (`tree_1d_lookup`), IVF-PQ search (`vector_search`), and radix attention TTFT (`radix_attention_ttft`). Not part of the publish gates.

## Key Architectural Decisions

### Pointer-less Memory Management
All tree nodes are addressed by 40-bit relative offsets into a contiguous arena. This removes external fragmentation, makes snapshots trivially relocatable, and keeps nodes small — but it means the arena must be sized with headroom: retry-orphaned path nodes from failed CAS attempts stay in the append-only arena and are never reclaimed.

### Thread Safety & Concurrency
The read path is lock-free: mutations are published with a single 64-bit CAS. **Root publication must be a CAS-with-retry loop** — a plain `store` publishes a stale root and silently loses concurrent inserts. Slab slot access state is a `fetch_or(1)` on a 2-bit clock; the effective reference count is {0, 1}, and `CacheCapacityExceeded` is only reachable when the reference count is $\ge 2$. Reclamation of retired slots is deferred through Hazard Eras, eliminating ABA without garbage collection.

### Error Handling
All fallible operations return `Result<T, FlareError>`. Arena exhaustion (`AllocationFailed`), cache capacity, journal size vs. codebook size mismatches (`InvalidParameter`), and GPU unavailability are all first-class error variants. Performance-critical internal paths use `debug_assert!` for boundary checks, avoiding runtime overhead in release builds.

### Portability & Safety
- `flare-core` compiles under `#![no_std] + alloc`; `std` is only linked for `std::error::Error` integration (feature `std`, default-on).
- Crate roots deny `unsafe_code`; every `unsafe` block is confined to internal memory/reclamation primitives with a documented invariant proof in its `# Safety` section.
- `flare-vector` relaxes `unsafe_code` only for the confined AVX2 distance kernels.
- The default `CpuFallbackDriver` executes `core::sync::atomic::fence(Ordering::SeqCst)` on `publish_epoch_fence` to guarantee sequential consistency across host readers.

### Test Sizing
Concurrency stress tests must size arenas with headroom: test helpers use `1 << 23` slab regions while `1 << 20` exhausts under load. Re-clustering trains on the journal; a journal smaller than the codebook size returns `InvalidParameter`, so concurrency tests around re-clustering retry until success.
