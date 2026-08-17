//! FLARE core primitives: flat memory arenas, slab pools, tagged pointers,
//! adaptive radix nodes, hazard eras, delta WAL, and GPU synchronisation.
//!
//! This crate provides the memory, pointer, tree, concurrency, and
//! persistence primitives that the [`flare-vector`](https://docs.rs/flare-vector),
//! [`flare-kv`](https://docs.rs/flare-kv), and [`flare-ffi`](https://docs.rs/flare-ffi)
//! crates are built upon. Every node is addressed with a 40-bit relative
//! offset inside a contiguous [`FlatArena`]; raw virtual pointers are never
//! stored inside tree nodes.
//!
//! # `no_std` Compliance
//!
//! The crate is compiled with `#![no_std]` and depends exclusively on
//! `core` plus `alloc` (via `extern crate alloc as alloc_crate`, because
//! the crate's own memory module is named `alloc`). The optional `std`
//! feature (enabled by default) additionally integrates
//! [`std::error::Error`] for [`FlareError`]. The module layout is:
//!
//! | Module | Responsibility |
//! | --- | --- |
//! | [`ptr`] | 64-bit polymorphic tagged pointer packing and unpacking |
//! | [`alloc`] | Bump [`FlatArena`] and 4 KB size-classified [`SlabPool`] |
//! | [`tree`] | Adaptive radix nodes and the [`FlareArtTree`] index |
//! | [`sync`] | Hazard Eras reclamation and the [`GpuSyncDriver`] trait |
//! | [`wal`] | Physical delta write-ahead log with group commit framing |
//!
//! # Memory Safety
//!
//! This crate root denies `unsafe_code`. All `unsafe` blocks are strictly
//! confined to the internal memory and reclamation primitives in [`alloc`],
//! [`ptr`], [`sync`], and [`wal`], each carrying a formally documented
//! invariant proof in its `# Safety` section; the tree and error layers are
//! entirely `safe` code.
//!
//! # Examples
//!
//! ```
//! # use flare_core::tree::FlareArtTree;
//! # use flare_core::alloc::arena::FlatArena;
//! # use flare_core::sync::hazard::HazardManager;
//! # use flare_core::sync::gpu::CpuFallbackDriver;
//! # use std::sync::Arc;
//! let arena = Arc::new(FlatArena::new(1 << 20).expect("arena fits"));
//! let hazard = Arc::new(HazardManager::new());
//! let tree = FlareArtTree::new(arena, hazard, CpuFallbackDriver::default());
//! let inserted = tree.insert(b"hello", 42).expect("insert succeeds");
//! assert_eq!(inserted, None);
//! let value = tree.get(b"hello").expect("lookup succeeds");
//! assert_eq!(value, Some(42));
//! ```

#![no_std]
#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]
#![deny(rustdoc::private_intra_doc_links)]
#![deny(rustdoc::missing_crate_level_docs)]
#![deny(rustdoc::invalid_codeblock_attributes)]
#![deny(rustdoc::invalid_html_tags)]
#![deny(rustdoc::invalid_rust_codeblocks)]
#![deny(rustdoc::bare_urls)]
#![deny(unsafe_code)]
#![deny(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::module_name_repetitions)]

#[cfg(any(test, feature = "std"))]
extern crate std;

extern crate alloc as alloc_crate;

pub mod alloc;
pub mod error;
pub mod ptr;
pub mod sync;
pub mod tree;
pub mod wal;

pub use alloc::{ArenaRef, FlatArena, SlabClass, SlabPool, SlabSlot};
pub use error::FlareError;
pub use ptr::{NodeType, TaggedPointer, resolve_child_index};
pub use sync::{CpuFallbackDriver, EraGuard, GpuSyncDriver, HazardManager};
pub use tree::{FlareArtTree, Node4, Node16, Node64, Node256, key_nibbles};
