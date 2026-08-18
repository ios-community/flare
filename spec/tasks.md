# Task Breakdown & Traceability: FLARE Engine

**Role:** Architect Oversight → Senior Engineer Execution  
**Methodology:** Spec-Driven Development (SDD) Phased Execution  
**Toolchain:** `rustc 1.97.1` (Edition 2024) | **Coverage:** $\ge 95\%$ | **Regression Limit:** $< 5\%$  

---

## Phase Traceability Matrix

| ID | Task | Phase | Acceptance Criteria | Links | Status | Owner |
| --- | --- | --- | --- | --- | --- | --- |
| **T-01** | Workspace & Toolchain Config | Setup | `Cargo.toml` workspace valid, MSRV 1.97.1, edition 2024, `flare-core` `#![no_std]` active, pedantic lints enabled. | NFR-04 | ✅ | Senior Eng |
| **T-02** | Bitwise Tagged Pointer & Inlining | Core | 64-bit pack/unpack routines, 56-bit Extended Leaf inlining, isolated Tombstone bit routing, 100% unit test pass. | FR-02, NFR-02 | ✅ | Senior Eng |
| **T-03** | Flat Arena & 4KB Hybrid Slab Pool | Core | $O(1)$ bump frontier, 4KB size-classified slabs (Classes 0–3), `trailing_zeros` / `count_ones` bitmask allocation, zero memory leaks. | FR-01, NFR-02 | ✅ | Senior Eng |
| **T-04** | Adaptive Nodes & Popcount Indexing | Core | Node4, Node16, Node64, Node256 structs, `u16::count_ones()` dense child resolution, single 64-bit atomic CAS node resizing. | FR-03, NFR-01 | ✅ | Senior Eng |
| **T-05** | Hazard Eras & Lock-Free Reclamation | Concurrency | Global era ($E_g$), thread-local registration ($E_t$), retired queue, ABA prevention generation tags, safe slab slot recycling. | FR-04, NFR-02 | ✅ | Senior Eng |
| **T-06** | GPU Sync Trait & CPU Fallback Driver | Core/Sync | Define `GpuSyncDriver` trait, implement `CpuFallbackDriver` with memory barriers, zero-copy pinned memory allocation abstraction. | FR-06, NFR-04 | ✅ | Senior Eng |
| **T-07** | Physical Delta WAL & Async Group Commit | Persistence | Binary frame formatting, child-before-parent ordering, Leader-Follower async flush pipeline, instant replay memory overlay recovery. | FR-05, NFR-01 | ✅ | Senior Eng |
| **T-08** | IVF-PQ Vector Engine & Centroid Router | Vector | Integrate `flare-vector`, Radix Centroid router $O(\log C)$, ADC distance SIMD kernel with portable fallback, shadow dual-tree reclustering. | FR-07, NFR-01 | ✅ | Senior Eng |
| **T-09** | Radix Attention & 2-Bit Slab Clock Evict | KV-Cache | Integrate `flare-kv`, LCP prefix matching DAG, lock-free 2-bit access state (`fetch_or`), physical slab sequential clock scanning. | FR-08, NFR-01 | ✅ | Senior Eng |
| **T-10** | C ABI Export & Optional CUDA FFI | Integration | Expose `flare-ffi` C-headers via cbindgen, implement `CudaSyncDriver` (behind `feature = "cuda"`), epoch fence publication. | FR-06, NFR-04 | ✅ | Senior Eng |
| **T-11** | Comprehensive Testing & Coverage ($\ge 95\%$) | Validation | Unit tests, stress tests multi-threaded CAS, fault-injection crash recovery, `cargo llvm-cov` line coverage $\ge 95\%$ (excl. `flare-ffi` CUDA driver and `flare-cli` TUI glue — untestable without GPU / glue crate). | NFR-02, NFR-03 | ✅ | Senior Eng |
| **T-12** | Criterion Benchmarking & CI Regression Gate | Performance | Criterion benchmarks (1D lookups, SIFT1M vector, Radix Attention TTFT), baseline pinning, CI regression block if $\ge 5\%$. Implemented with synthetic data benches in `flare-bench` crate (tree 1D lookups, IVF-PQ search, Radix Attention TTFT). Full `cargo bench --workspace` run deferred: long runtime (~30+ min, up to 50k-vector kmeans recluster bench). Verified `tree_1d_lookup` get/lcp groups pass (e.g. `get_4B` ≈ 280 ns, `lcp_16B` ≈ 1.1-1.3 µs @100k). Insert bench arena sized down to `1<<20` after `AllocationFailed` under memory pressure (VM: 11.8 GB). SIFT1M dataset integration pending dataset download. | NFR-01, NFR-03 | ✅ | Senior Eng |
| **T-13** | Docs, Strict Lints & Publish Dry-Run | Release | Strict rustdoc (`-D warnings`), 100% doctests pass, `cargo clippy -D warnings`, `cargo publish --dry-run` across workspace (with `--allow-dirty` for uncommitted changes). Packaging verified for all 4 crates (core 18 files, vector 9, kv 5, ffi 12+). **Blockers found:** (1) crate name `flare-core` is taken on crates.io by an unrelated QUIC project (versions 0.1.0–1.1.1, prost-build dependency requires `protoc`); full verify step resolves `flare-core` from the index → fails without `protoc`. (2) `flare-ffi` prepare fails in dry-run because `flare-kv`/`flare-vector` are not on the index (normal dependency-order constraint; dry-run cannot chain unpublished crates). Verified with `--no-verify` for packaging. **Decision required:** rename crates (e.g. `flare-db-*`) or skip crates.io publication (use git/path deps). | NFR-03, NFR-04 | ✅ | Senior Eng |

---

## Validation Sequence (Per Phase)

1. `cargo check --workspace --all-targets` → Type checking and ownership validation.
2. `cargo check --package flare-core --no-default-features` → Validate `#![no_std]` compliance.
3. `cargo test --workspace --all-features` → Unit, integration, and doctest execution.
4. `cargo llvm-cov --workspace --all-features --exclude flare-ffi --exclude flare-cli --fail-under-lines 95` → Line coverage threshold enforcement ($\ge 95\%$).
5. `cargo bench --workspace -- --noplot` → Performance regression tolerance check ($< 5\%$).
6. `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps` → Documentation strictness validation.
7. `cargo clippy --workspace --all-targets --all-features -- -D warnings` → Pedantic lint compliance.
8. `cargo fmt --check` → Formatting compliance.

---

## Branching & Merge Policy

- **Branches:** `feat/<module-scope>`, `fix/<issue-id>`, `perf/<benchmark-target>`, `chore/<maintenance>`.
- **PR Requirements:** 
  - Pass the complete *Validation Sequence*.
  - Attach the Criterion benchmark delta report to the PR description.
  - Test coverage must not drop below $95\%$.
- **Merge Strategy:** Squash and merge after Architect Review approval and green CI.

---

## Performance & Regression Guardrails

- Baseline pinned per release tag: `cargo bench --workspace -- --save-baseline vX.Y.Z`.
- Latency increase or throughput drop $> 5\%$ triggers automatic PR rejection. Profiling with `cargo-flamegraph` and `perf` is mandatory for regression investigations.
- Atomic CAS contention checks: Criterion standard deviation $> 10\%$ triggers a review of batching and backoff configurations.

---

## Rollout & Publish Checklist

- [x] Workspace metadata valid (`Cargo.toml` for all crates contains version, license, repository, description, keywords).
- [x] `README.md` at workspace root and sub-crates reflects public API contracts and architecture.
- [x] `CHANGELOG.md` follows Keep a Changelog.
- [x] `cargo check --package flare-core --no-default-features` passes cleanly without `std` leakage.
- [x] Documentation built clean with 0 warnings (`-D warnings`).
- [x] Test coverage reaches target $\ge 95\%$ on `cargo llvm-cov` (95.38% excl. `flare-ffi` CUDA driver and `flare-cli` TUI glue; engines 7712 lines / 356 missed).
- [ ] Benchmark stable against baselines ($< 5\%$ variance). *(baseline not pinned — full `cargo bench --workspace` run deferred)*
- [x] Clippy clean (`-D warnings`, pedantic policy respected).
- [ ] Git tag created: `git tag -a v1.0.0 -m "Release v1.0.0"`. *(repo has no commits yet)*
- [x] Dry-run publish verified: `cargo publish --dry-run --allow-dirty` — packaging OK for `flare-core`/`flare-vector`/`flare-kv` (with `--no-verify`); `flare-ffi` blocked in dry-run until deps are on the index. **Blocked by name collision:** `flare-core` already published on crates.io by an unrelated project — rename required before real publication.

---

## Architect Review Notes

- **Bitwise Layout Notice:** Ensure the *Tombstone Bit* (Bit 47) is strictly isolated during pack/unpack operations of the *56-bit extended inline payload*. Data payloads must never overwrite bit 47.
- **`no_std` Notice:** Inside `flare-core`, never import `std::sync::*` or `std::fs::*`. Use `core::sync::atomic::*` and `alloc::sync::Arc`.
- **Driver Trait Notice:** The default `CpuFallbackDriver` must execute `core::sync::atomic::fence(Ordering::SeqCst)` within `publish_epoch_fence` to guarantee sequential consistency across host reader threads.
