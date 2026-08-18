# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-17

### Added
- Initial release of FLARE (Flat Lock-free Arena Radix Engine), a multi-crate workspace with strict dependency order: `flare-core` → `flare-vector` / `flare-kv` → `flare-ffi`.
- `flare-core` \
`#![no_std]` + alloc primitives with zero external dependencies:
  - Contiguous `FlatArena` with $O(1)$ bump allocation and 4 KB size-classified slab pools (Node4, Node16, Node64, Node256) for instant slot recycling.
  - 64-bit polymorphic `TaggedPointer` (Type 3b, Offset 40b, ArenaID 4b, Tombstone 1b, Metadata 16b) with bitwise pack/unpack and 56-bit extended leaf inlining.
  - Adaptive radix tree (`Node4/16/64/256`) with popcount child resolution and single-CAS node resizing; CAS-with-retry root publication preserving concurrent inserts.
  - Hazard Eras safe memory reclamation (global era, thread-local tracking, retired queue, ABA prevention).
  - Physical delta WAL with child-before-parent ordering, leader-follower async group commit, and instant replay crash recovery.
  - `GpuSyncDriver` trait with default `CpuFallbackDriver` and zero-copy pinned memory allocation abstraction.
- `flare-vector` \
IVF-PQ index with radix centroid routing ($O(\log C)$), k-means training, 8-bit product-quantization codebooks, SIMD-accelerated ADC with portable scalar fallback, and lock-free shadow dual-tree re-clustering.
- `flare-kv` \
radix attention engine with longest-common-prefix matching DAG and lock-free 2-bit slab clock eviction over physical slots.
- `flare-ffi` \
C ABI exports over opaque handles with cbindgen-generated `include/flare.h`, plus optional `CudaSyncDriver` (feature `cuda`) with dynamic runtime loading.
- `flare-bench` \
Criterion benchmarks for 1D radix lookups, IVF-PQ vector search, and radix attention TTFT.
- `flare-cli` \
Interactive shell (`publish = false`): a `ratatui` TUI dashboard (`tui`) streaming live counters over three tabs (tree / vector / KV-cache), a reedline REPL (`repl`) with 14 commands covering every engine API, and a headless chaos arena (`chaos`) with multi-threaded insert storms, memory-pressure sweeps, and crash injection.
- Workspace quality gates: pedantic + nursery clippy denied, zero-warning rustdoc, `missing_docs` enforced, and line coverage $\ge 95\%$ (95.38% excluding the untestable CUDA driver and flare-cli TUI glue paths).
- Multi-threaded CAS stress tests (including `concurrent_inserts_are_all_present`), fault-injection crash-recovery tests, and synthetic-data benchmark suites.

### Fixed
- TUI workload now starts paused: the dashboard opens idle and only grows data after the user presses `SPACE`.
- TUI tab labels are numbered 1-5 (keys 1-5 select them); the header no longer shows 0-4.
- REPL prompt renders a single `flare> ` (reedline's built-in `> ` indicator suppressed) and command output starts on its own line.
- Chaos storm audit only verifies the key range the workers actually wrote, eliminating spurious "lost updates" reports when the keyspace exceeds the per-thread attempt budget.

### Known Blockers (publication)
- Crate name `flare-core` is taken on crates.io by an unrelated QUIC project; packaging verified with `--no-verify`, real publication requires a crate rename (decision pending).
- `flare-ffi` dry-run fails until `flare-kv`/`flare-vector` exist on the index (normal dependency-order constraint).
- Repository has no git commits yet; no release tag or pinned benchmark baseline.
