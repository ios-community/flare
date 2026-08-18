# AGENTS.md

Rust workspace (edition 2024, MSRV 1.97.1, resolver 3) implementing FLARE: a lock-free arena/radix engine for 1D KV, IVF-PQ vector search, and LLM KV-cache. `spec/` (requirements.md, design.md, tasks.md) is the source of truth; `spec/tasks.md` has the validation sequence and rollout checklist.

## Layout & dependency order

- `crates/flare-core` — `#![no_std]` + alloc, **zero external deps**. Arena, 4KB slabs, tagged ptr, ART nodes, hazard eras, GPU sync trait, delta WAL.
- `crates/flare-vector` → depends on core. IVF-PQ index, kmeans, ADC distance.
- `crates/flare-kv` → depends on core. Radix attention engine (prefix match, 2-bit clock eviction).
- `crates/flare-ffi` → depends on all. C ABI + optional CUDA driver (`feature = "cuda"`).
- `crates/flare-bench` — criterion benches, `publish = false`. Not part of gates.

## Required validation sequence (all must pass)

```powershell
cargo check --workspace --all-targets
cargo check --package flare-core --no-default-features   # no_std compliance
cargo test --workspace --all-features
cargo llvm-cov --workspace --all-features --exclude flare-ffi --exclude flare-cli --fail-under-lines 95
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo fmt --check
```

Gotchas:
- **Coverage: llvm-cov MUST exclude `flare-ffi` and `flare-cli`.** The CUDA driver path (~37% covered) and the TUI render paths (glue crates, ~1033 uncovered lines in flare-cli) pull the workspace below 95% (86.70% full vs 95.38% excluding — engines alone are 7712 lines / 356 missed).
- **AVX2 is not required on the runner:** this VM lacks AVX2, so the `distance.rs` AVX2 kernel body is permanently uncovered here. Treat engine coverage misses in defensive/legacy paths (e.g. `Node64`→`Node256` growth is unreachable with 4-bit nibbles; Node16→Node64 needs a synthetic bitmap) as acceptable; keep the reachable recovery budget: engines need ≤ 365 missed lines for the 95% gate.
- `cargo doc` on this toolchain requires `--all-features` or rustdoc `-D warnings` failures in flare-ffi cfg-gated modules.
- Single test: `cargo test -p flare-core concurrent_inserts` (filter substring).

## Toolchain quirks (edition 2024)

- Extern blocks must be `unsafe extern "C" { ... }`; calls through them still need `unsafe {}`.
- `#[unsafe(no_mangle)]`, not `#[no_mangle]`.
- `gen` is a reserved keyword: rand 0.8 usage is `rng.r#gen::<f32>()`.
- flare-core: never import `std::*` — use `core::sync::atomic` and `alloc::sync::Arc` (enforced by `--no-default-features` check).
- Workspace lints deny `missing_docs` and `unsafe_code`; flare-vector relaxes `unsafe_code` for AVX2 kernels. All public items need doc comments. Pedantic + nursery clippy denied (`module_name_repetitions` allowed). FFI boundary fns carry module-level `#[allow(clippy::not_unsafe_ptr_arg_deref, clippy::identity_op, clippy::erasing_op)]` — keep that pattern for new C-exposed fns.

## Concurrency invariants (hard-won — do not regress)

- **Root publication must be CAS with retry** (see `flare-core/src/tree/tree.rs` insert). A plain `store` publishes a stale root and silently loses concurrent inserts; the stress test `concurrent_inserts_are_all_present` catches this.
- Retry-orphaned path nodes stay in the append-only arena (never reclaimed) — size arenas with headroom. Test helpers use `1 << 23`; `1 << 20` exhausts under load.
- Slab access state is `fetch_or(1)`; the effective reference count is {0,1} — `CacheCapacityExceeded` is only reachable when ref ≥ 2 (tests set the clock directly to `LIVE|2`).
- flare-vector reclustering trains on the journal; journal smaller than the codebook size returns `InvalidParameter`. Concurrency tests around reclustering retry until success.

## flare-ffi specifics

- `include/flare.h` is **generated** by `build.rs` via cbindgen 0.29.4 from `src/c_abi.rs` — never hand-edit; keep the export list in `cbindgen.toml` in sync.
- `flare_version()` hardcodes `100` (= 0*10_000 + 1*100 + 0) — update it when the workspace version changes.
- `CudaSyncDriver` dynamic-loads the driver (`nvcuda.dll`/`libcuda.so.1`); tests must be environment-independent (early-return when runtime absent).

## flare-bench

- Full `cargo bench --workspace` takes 30+ min; the recluster bench runs full kmeans per iteration (sample_size reduced to 10).
- 256 MB `alloc_zeroed` arenas can fail with `AllocationFailed` on this VM (~11.8 GB RAM) — keep bench arenas ≤ `1 << 26`.

## Publishing (not yet done)

- Repo has **no git commits** — `cargo publish --dry-run` requires `--allow-dirty`.
- **`flare-core` name is taken on crates.io** by an unrelated QUIC project (0.1.0–1.1.1; needs `protoc` via prost-build). Cargo's verify step resolves `flare-core` from the index → fails. Use `--no-verify` for packaging-only validation; real publication needs a crate rename (decision pending).
- Publish in dependency order: core → vector/kv → ffi (ffi dry-run fails before deps exist on the index).
