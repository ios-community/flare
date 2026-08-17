# Requirements Specification: FLARE (Flat Lock-free Arena Radix Engine)

**Role:** Architect Engineer → Senior Software Engineer  
**Status:** Frozen  
**Target Registry:** crates.io / Internal Enterprise Registry  
**MSRV:** Rust 1.97.1 | **Edition:** 2024  

---

## Architect Directives

- **Primary Objective:** Build an ultra-high-performance, pointer-less hybrid radix engine that unifies 1D Key-Value indexing, High-Dimensional Vector Search (IVF-PQ), and LLM KV-Cache Management (Radix Attention) within a single contiguous, fragmentation-free memory arena.
- **Crate Type & Workspace Architecture:** Multi-crate Cargo Workspace:
  - `flare-core`: Core primitives (*Flat Memory Arena, 4 KB Slab Pools, 64-bit Tagged Pointer, ART Nodes, Hazard Eras, Async WAL, GpuSyncDriver Trait*). Must support `#![no_std]` (with `extern crate alloc`).
  - `flare-vector`: High-Dimensional Vector Search extension (*IVF-PQ Centroid Routing, Shadow Dual-Tree Re-clustering, SIMD-accelerated Asymmetric Distance Computation*).
  - `flare-kv`: LLM KV-Cache Management extension (*Radix Attention Prefix Sharing, Lock-Free 2-bit Clock Eviction over physical slab memory slots*).
  - `flare-ffi`: C/C++ ABI exports and optional `CudaSyncDriver` implementation for GPU runtime synchronization.
- **Concurrency Model:** Lock-free read path utilizing 64-bit atomic Compare-And-Swap (CAS), Hazard Eras reclamation, and atomic Tombstone tagging.
- **Memory Safety:** Top-level crates strictly enforce `#![deny(unsafe_code)]`. All `unsafe` blocks are strictly confined to internal memory primitives (`flare-core::alloc` and `flare-core::ptr`) with formally documented invariant proofs.
- **Portability:** `flare-core` must compile under `#![no_std] + alloc`. Primary target triples: `x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu`.
- **Documentation & Quality Contract:** Strict rustdoc compliance (`RUSTDOCFLAGS="-D warnings"`), Line Test Coverage $\ge 95\%$, Performance Regression Threshold $< 5\%$.

---

## Functional Requirements (FR)

| ID | Requirement | Owner | Description |
| --- | --- | --- | --- |
| **FR-01** | **Contiguous Arena & Hybrid Slab Allocator** | Senior Eng | Implement a contiguous linear memory arena ($A$) combining $O(1)$ bump allocation for newly ingested frontiers with size-classified 4 KB slab pools (Node4, Node16, Node64, Node256) for instant slot recycling without external memory fragmentation. Must support `#![no_std] + alloc`. |
| **FR-02** | **64-bit Polymorphic Tagged Pointer** | Senior Eng | Implement a 64-bit polymorphic tagged pointer layout: Type (3b), Offset (40b), ArenaID (4b), Tombstone (1b), and Polymorphic Metadata (16b). Provide instant bitwise pack/unpack routines and 56-bit Extended Leaf Inlining with isolated Tombstone bit routing (Bit 47) and 8-bit MSB truncation for 64-bit identifiers. |
| **FR-03** | **Adaptive Radix Tree (ART) Core** | Senior Eng | Implement adaptive radix node structures (`Node4`, `Node16`, `Node64`, `Node256`) with hardware-accelerated popcount child index resolution via `u16::count_ones()` / `u64::count_ones()` and single 64-bit atomic CAS node resizing. |
| **FR-04** | **Hazard Eras Safe Memory Reclamation** | Senior Eng | Implement global epoch era ($E_g$), thread-local active era tracking ($E_t$), and a retired queue to safely recycle deallocated slab slots back to slab freelists without blocking concurrent reader threads. |
| **FR-05** | **Physical Delta WAL (Async Group Commit)** | Senior Eng | Implement a binary physical delta Write-Ahead Log enforcing a strict *Child-Before-Parent* write ordering barrier. Provide a Leader-Follower pipeline for asynchronous group commit flushing (`AsyncFlush`) and instant replay crash recovery via direct memory overlay. |
| **FR-06** | **Trait-Based GPU Sync & Affinity Abstraction** | Senior Eng | Define the `GpuSyncDriver` trait in `flare-core`. Provide a default `CpuFallbackDriver` (for pure CPU execution, CI, and determinism) and zero-copy host pinned memory allocation abstraction. Implement `CudaSyncDriver` in `flare-ffi` for publishing CUDA epoch fences to GPU streams. |
| **FR-07** | **IVF-PQ Vector Engine & Shadow Re-clustering** | Senior Eng | Implement Inverted File Product Quantization in `flare-vector`. Utilize the Flat Radix Tree as an $O(\log C)$ centroid router, provide SIMD-accelerated Asymmetric Distance Computation (ADC) with portable loop fallbacks, and support non-blocking background Shadow Dual-Tree dynamic re-clustering. |
| **FR-08** | **Radix Attention & Slab Clock Eviction** | Senior Eng | Implement a prefix-sharing KV-cache DAG in `flare-kv` with Longest Common Prefix (LCP) matching and a *Lock-Free 2-bit Clock Eviction* policy that executes sequential linear scans directly over physical slab memory slots. |

---

## Non-Functional Requirements (NFR)

| ID | Category | Constraint | Validation Method |
| --- | --- | --- | --- |
| **NFR-01** | **Performance** | Maximum allowed performance regression $< 5\%$ against release baselines. | Headless Criterion CI benchmark tracking. |
| **NFR-02** | **Safety & Invariants** | Zero Undefined Behavior (UB), zero data races under atomic CAS concurrency, strictly bounds-checked arena access. | ThreadSanitizer (TSAN), AddressSanitizer (ASAN), fuzz testing (`cargo-fuzz`), and `proptest`. |
| **NFR-03** | **Quality & Docs** | Line coverage $\ge 95\%$ (excluding hardware-level fatal OOM abort handlers). Zero warnings on documentation for all `pub` items. | `cargo llvm-cov --fail-under-lines 95`, `RUSTDOCFLAGS="-D warnings" cargo doc`. |
| **NFR-04** | **Portability & Toolchain** | Must compile on Rust 1.97.1 (Edition 2024). `flare-core` must compile cleanly without the standard library (`--no-default-features`). | `cargo check --package flare-core --no-default-features --target aarch64-unknown-linux-gnu`, `cargo check --workspace`. |

---

## Performance Targets

| Subsystem | Metric | Bound | Measurement |
| --- | --- | --- | --- |
| **1D Radix Lookup** | Zipfian Point Lookup Latency | $p50 \le 0.08\ \mu s,\ p99 \le 0.25\ \mu s$ | Criterion benchmark suite (100M keys) |
| **1D Radix Ingestion** | Bulk Insertion Throughput | $\ge 40.0\ \text{MOPS}$ | Multi-threaded insertion stress test |
| **IVF-PQ Vector Search** | Recall@10 Latency (SIFT1M) | $\le 0.50\ \text{ms}$ | Recall vs Latency benchmark harness |
| **KV-Cache Eviction** | Slab Clock Scan Overhead | $O(1)\ \text{amortized} \le 15\ \text{ns/slot}$ | Micro-benchmark eviction loop |
| **Physical Delta WAL** | Async Group Commit Throughput | $\ge 250,000\ \text{frames/sec}$ | Concurrent I/O stress harness |

---

## Out of Scope

- Triton / WebAssembly JIT runtime compilation layers.
- Multi-node distributed consensus (Raft/Paxos) across distinct network hosts (FLARE focuses on single-node NVLink/CXL multi-arena clusters).
- Dynamic on-the-fly CUDA kernel code generation (CUDA kernels are statically compiled in `flare-ffi`).

---

## Architect Sign-Off

- [x] Workspace architecture & `#![no_std]` constraints finalized
- [x] Concurrency & Trait-Based GPU sync boundaries defined
- [x] Code coverage target ($\ge 95\%$) & Performance regression limit ($< 5\%$) locked
- [x] Async Group Commit WAL durability model specified
