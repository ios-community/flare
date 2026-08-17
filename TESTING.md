# Testing & Validation Guide

This document explains how to run the FLARE test suite, generate coverage reports, validate documentation and lints, execute performance benchmarks, and understand the required validation sequence.

## Prerequisites

Ensure you have the Rust toolchain (1.97.1 or newer, edition 2024). Line coverage requires `cargo-llvm-cov`:

```bash
cargo install cargo-llvm-cov
```

---

## Required Validation Sequence (All Must Pass)

This is the gate used by CI and by every pull request. Run the steps in order:

```powershell
cargo check --workspace --all-targets
cargo check --package flare-core --no-default-features   # no_std compliance
cargo test --workspace --all-features
cargo llvm-cov --workspace --all-features --exclude flare-ffi --fail-under-lines 95
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo fmt --check
```

### 1. Type Checking
`cargo check --workspace --all-targets` validates type correctness and ownership across all crates including tests, examples, and benches.

### 2. `no_std` Compliance
`cargo check --package flare-core --no-default-features` verifies that `flare-core` compiles cleanly without the standard library. This enforces the hard rule that the crate never imports `std::*` — use `core::sync::atomic` and `alloc::sync::Arc` instead.

### 3. Unit and Integration Tests
`cargo test --workspace --all-features` runs unit tests (at the bottom of each module file), integration tests, and doctests with every feature enabled. Key concurrency invariants covered:

- **`concurrent_inserts_are_all_present`** — the stress test that catches a stale-root publication bug. A plain `store` publishes a stale root and silently loses concurrent inserts; the CAS-with-retry loop keeps every insert. Run a single test with a filter substring:
  ```bash
  cargo test -p flare-core concurrent_inserts
  ```

### 4. Line Coverage (≥ 95%)
```bash
cargo llvm-cov --workspace --all-features --exclude flare-ffi --fail-under-lines 95
```

**Gotcha:** `flare-ffi` MUST be excluded. Its CUDA driver path (~37% covered, untestable without a GPU) pulls the workspace below 95% (93.88% full vs 96.66% excluding).

### 5. Documentation Validation
```bash
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps
```

**Gotcha:** `cargo doc` on this toolchain requires `--all-features` or rustdoc `-D warnings` fails on cfg-gated modules in `flare-ffi`.

### 6. Clippy Lints
```bash
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

The workspace denies `all`, `pedantic`, and `nursery` lint groups (`module_name_repetitions` is allowed). All public items require doc comments (`missing_docs` = deny).

### 7. Formatting
```bash
cargo fmt --check
```

---

## Specialized Verification

### Concurrency & Safety Targets
Per `spec/requirements.md` (NFR-02), the project targets zero Undefined Behavior and zero data races. Multi-threaded CAS stress tests are part of the test suite; fault-injection crash-recovery tests cover the WAL replay path.

### Environment-Dependent Tests
`flare-ffi` CUDA tests must be environment-independent: `CudaSyncDriver` dynamic-loads the driver (`nvcuda.dll`/`libcuda.so.1`), and tests early-return when the runtime is absent.

---

## Benchmarking

The `flare-bench` crate (criterion, `publish = false`) measures the critical paths:

| Benchmark | Measures |
| --- | --- |
| `tree_1d_lookup` | 1D radix point lookup and LCP latency (Zipfian, 100M-key class) |
| `vector_search` | IVF-PQ search latency/recall (SIFT1M-class harness) |
| `radix_attention_ttft` | Radix attention time-to-first-token |

```bash
# Run a single benchmark group
cargo bench --workspace --bench tree_1d_lookup

# Pin a release baseline for regression tracking
cargo bench --workspace -- --save-baseline v0.1.0
```

Guidelines:

- **A full `cargo bench --workspace` takes 30+ minutes** — the recluster bench runs full k-means per iteration (its sample size is reduced to 10). Run individual benchmarks while developing.
- **Arena sizing:** 256 MB `alloc_zeroed` arenas can fail with `AllocationFailed` on constrained machines (~11.8 GB RAM) — keep bench arenas ≤ `1 << 26`.
- **Regression gate:** latency increase or throughput drop > 5% against the pinned baseline triggers PR rejection (NFR-01).
