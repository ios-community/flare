# FLARE

_**Flat Lock-free Arena Radix Engine: unified 1D key-value, IVF-PQ vector search, and radix-attention LLM KV-cache management.**_

[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)
[![Rust](https://img.shields.io/badge/rust-1.97.1%2B-orange.svg)](https://github.com/ios-community/flare.git)

A lock-free, arena-resident storage engine written from scratch in Rust. FLARE unifies 1D key-value indexing, high-dimensional vector search (IVF-PQ), and LLM KV-cache management (radix attention) inside a single contiguous, fragmentation-free memory arena addressed exclusively by 40-bit relative offsets — raw virtual pointers are never stored in tree nodes.

## Overview

FLARE is a multi-crate Cargo workspace. `flare-core` provides the memory, pointer, tree, concurrency, and persistence primitives (`#![no_std]` + alloc, zero external dependencies); `flare-vector` and `flare-kv` build the IVF-PQ search engine and the radix attention KV-cache engine on top of it; `flare-ffi` exports the whole system over a C ABI with an optional CUDA synchronization driver.

The read path is lock-free: every tree mutation is published with a single 64-bit atomic Compare-And-Swap, and retired slab slots are reclaimed through Hazard Eras so readers are never blocked.

## Features

- **Flat Arena & Hybrid Slab Allocator** \
  A contiguous linear arena combining $O(1)$ bump allocation for ingested frontiers with size-classified 4 KB slab pools (Node4, Node16, Node64, Node256) for instant slot recycling without external fragmentation.
- **64-bit Polymorphic Tagged Pointer** \
  Bitwise pack/unpack of Type (3b), Offset (40b), ArenaID (4b), Tombstone (1b), and Polymorphic Metadata (16b), with 56-bit extended leaf inlining.
- **Adaptive Radix Tree (ART)** \
  Hardware-accelerated popcount child resolution (`count_ones`) and single-CAS node resizing for `Node4/16/64/256`.
- **Hazard Eras Reclamation** \
  Global epoch era, thread-local active-era tracking, and a retired queue for safe, lock-free slab slot recycling (ABA-prevented).
- **Physical Delta WAL (Async Group Commit)** \
  Binary physical delta write-ahead log with child-before-parent ordering, leader-follower async flush, and instant replay crash recovery.
- **IVF-PQ Vector Search** \
  Radix-tree centroid routing ($O(\log C)$), 8-bit product quantization, SIMD-accelerated asymmetric distance computation (ADC) with a portable scalar fallback, and lock-free shadow dual-tree re-clustering.
- **Radix Attention KV-Cache** \
  Longest-common-prefix matching DAG for token-prefix sharing plus lock-free 2-bit slab clock eviction scanning physical memory slots.
- **GPU Sync Abstraction** \
  `GpuSyncDriver` trait with a `CpuFallbackDriver` for deterministic CPU execution and an optional `CudaSyncDriver` (feature `cuda`) that dynamic-loads the CUDA runtime.
- **Strict Quality Gates** \
  `#![deny(unsafe_code)]` at the crate roots (unsafe confined to internal memory primitives), pedantic clippy, zero-warning rustdoc, and $\ge 95\%$ line coverage.

## Workspace Layout

| Crate | Responsibility | Depends On |
| --- | --- | --- |
| `flare-core` | Flat arena, 4 KB slabs, tagged pointer, ART nodes, hazard eras, delta WAL, `GpuSyncDriver` trait. `#![no_std]` + alloc, zero external deps. | — |
| `flare-vector` | IVF-PQ index, k-means training, codebooks, SIMD ADC distance kernels. | `flare-core` |
| `flare-kv` | Radix attention engine: prefix sharing, 2-bit slab clock eviction. | `flare-core` |
| `flare-ffi` | C ABI exports (generated `include/flare.h` via cbindgen) + optional `CudaSyncDriver`. | all |
| `flare-bench` | Criterion benchmarks (1D lookups, IVF-PQ search, radix attention TTFT). `publish = false`. | all |

The full design is specified in [`spec/`](spec/requirements.md) (requirements), [`spec/design.md`](spec/design.md) (design), and [`spec/tasks.md`](spec/tasks.md) (validation sequence & rollout checklist).

## Installation

Requires Rust 1.97.1 or newer (edition 2024):

```bash
git clone https://github.com/ios-community/flare.git
cd flare
cargo build --release
```

## Library Usage

### flare-core — 1D Radix Key-Value

```rust
use flare_core::alloc::arena::FlatArena;
use flare_core::sync::gpu::CpuFallbackDriver;
use flare_core::sync::hazard::HazardManager;
use flare_core::tree::FlareArtTree;
use std::sync::Arc;

let arena = Arc::new(FlatArena::new(1 << 20).expect("arena fits"));
let hazard = Arc::new(HazardManager::new());
let tree = FlareArtTree::new(arena, hazard, CpuFallbackDriver::default());
tree.insert(b"hello", 42).expect("insert succeeds");
let value = tree.get(b"hello").expect("lookup succeeds");
assert_eq!(value, Some(42));
```

### flare-vector — IVF-PQ Vector Search

```rust
use flare_core::sync::gpu::CpuFallbackDriver;
use flare_core::sync::hazard::HazardManager;
use flare_vector::{IvfPqIndex, SearchResult};
use std::sync::Arc;

let index = IvfPqIndex::new(
    4, 2, 2, 7, 1 << 20, Arc::new(HazardManager::new()), CpuFallbackDriver::default(),
).expect("index construction succeeds");
let samples: Vec<f32> = (0..512)
    .flat_map(|i| {
        let base = if i % 2 == 0 { 10.0 } else { -10.0 };
        [base, base, base, base]
    })
    .collect();
index.train(&samples).expect("training succeeds");
index.insert(&[10.5, 10.5, 10.5, 10.5]).expect("insert succeeds");
let hits: Vec<SearchResult> = index
    .search(&[10.4, 10.4, 10.4, 10.4], 1)
    .expect("search succeeds");
assert_eq!(hits.len(), 1);
```

### flare-kv — Radix Attention KV-Cache

```rust
use flare_core::sync::gpu::CpuFallbackDriver;
use flare_core::sync::hazard::HazardManager;
use flare_kv::RadixAttentionEngine;
use std::sync::Arc;

let engine = RadixAttentionEngine::new(
    1 << 20,
    1 << 20,
    Arc::new(HazardManager::new()),
    CpuFallbackDriver::default(),
)
.expect("construction succeeds");
engine.insert(&[1, 2, 3], 100).expect("insert succeeds");
let m = engine
    .match_common_prefix(&[1, 2, 3, 4])
    .expect("match succeeds")
    .expect("prefix found");
assert_eq!(m.token_len, 3);
assert_eq!(m.kv_offset, 100);
```

### flare-ffi — C ABI

The C header is generated by `build.rs` via cbindgen from `src/c_abi.rs` and lives at `crates/flare-ffi/include/flare.h`. The exported functions operate over opaque handles, never allocate on the C side, and return a `flare_status_t` status code where `0` means success. The `CudaSyncDriver` (feature `cuda`) dynamic-loads `nvcuda.dll`/`libcuda.so.1` at runtime, so building never requires a CUDA toolkit.

## Validation

The full validation sequence (tests, coverage, rustdoc, clippy, fmt) is documented in [TESTING.md](TESTING.md). In short:

```bash
cargo check --workspace --all-targets
cargo check --package flare-core --no-default-features   # no_std compliance
cargo test --workspace --all-features
cargo llvm-cov --workspace --all-features --exclude flare-ffi --fail-under-lines 95
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo fmt --check
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution workflow and quality gates, and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for community standards.

## License

This project is licensed under either of **MIT** or **Apache-2.0**, at your option — see [LICENSE](LICENSE).
