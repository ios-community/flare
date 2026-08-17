# Contributing to FLARE

We welcome contributions from the scientific and open-source communities! To maintain high code quality and correctness, please follow these guidelines.

## How to Contribute

1. Fork the repository.
2. Create a branch for your change following the naming policy:
   - `feat/<module-scope>` — new functionality
   - `fix/<issue-id>` — bug fixes
   - `perf/<benchmark-target>` — performance work
   - `chore/<maintenance>` — maintenance
3. Make your changes and update the specs in `spec/` (`requirements.md`, `design.md`, `tasks.md`) when behaviour is affected — they are the source of truth.
4. Ensure the complete validation sequence passes (see below).
5. Submit a Pull Request with a detailed description of your changes. Attach the Criterion benchmark delta report when the change touches performance-critical paths.

## Required Validation Sequence

Before submitting your PR, run every step:

```bash
cargo check --workspace --all-targets
cargo check --package flare-core --no-default-features
cargo test --workspace --all-features
cargo llvm-cov --workspace --all-features --exclude flare-ffi --fail-under-lines 95
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo fmt --check
```

See [TESTING.md](TESTING.md) for detailed explanations and gotchas.

### PR Acceptance Criteria

- Complete validation sequence passes (coverage must not drop below 95%).
- For performance changes: benchmark delta against the pinned baseline must be `< 5%` regression, with the Criterion delta report attached.
- Concurrency invariants must not regress — in particular, keep the CAS-with-retry root publication in `flare-core/src/tree/tree.rs` intact and never introduce `unsafe` outside the internal memory primitives.

## Code Style & Lints

We enforce strict compiler lints and formatting rules:

```bash
cargo fmt --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

The workspace denies `all`, `pedantic`, and `nursery` clippy groups (`module_name_repetitions` allowed) and `missing_docs`. Note the edition 2024 quirks:

- Extern blocks must be `unsafe extern "C" { ... }`, and `#[unsafe(no_mangle)]` replaces `#[no_mangle]`.
- `gen` is reserved: use `rng.r#gen::<f32>()`.
- `flare-core` must never import `std::*` — use `core::sync::atomic` and `alloc::sync::Arc`.

## Documentation

All public APIs must be fully documented, and the docs must build without warnings:

```bash
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps
```

Keep the workspace-level `README.md` and this documentation set (`ARCHITECTURE.md`, `TESTING.md`, `CHANGELOG.md`) in sync with API changes.

## FFI Changes

If you modify `crates/flare-ffi/src/c_abi.rs`, the C header is **generated** by `build.rs` via cbindgen — never hand-edit `include/flare.h`. Keep the export list in `cbindgen.toml` in sync, and update `flare_version()` (currently `100`) when the workspace version changes.

## Code of Conduct

All contributions are governed by our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before participating.
