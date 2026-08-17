//! Radix attention engine for LLM KV-cache management.
//!
//! This crate implements the `flare-kv` extension on top of `flare-core`:
//! a lock-free longest-common-prefix matching engine keyed by raw token
//! bytes, with engine-owned clock metadata and physical 2-bit slab clock
//! eviction.
//!
//! The engine keeps three lock-free structures:
//!
//! - a `flare_core::tree::FlareArtTree` mapping token-prefix byte keys to
//!   physical slot indices, providing `O(depth)` radix longest-prefix
//!   matching;
//! - a `flare_core::alloc::slab::SlabPool` of 4 KB-classed physical slots
//!   (one `Class0` slot per KV state), recycled through a hand-swept 2-bit
//!   clock;
//! - per-slot metadata owned by the engine: a 2-bit clock counter bumped
//!   with `fetch_or` and a `LIVE` bit, plus the published KV offset.
//!
//! Reads are lock-free and never allocate: a match walks the radix tree,
//! verifies the slot is live, touches its clock, and double-reads the KV
//! offset to detect eviction/reinitialisation races.
//!
//! # Examples
//!
//! ```
//! use flare_core::sync::gpu::CpuFallbackDriver;
//! use flare_core::sync::hazard::HazardManager;
//! use flare_kv::RadixAttentionEngine;
//! use std::sync::Arc;
//! let engine = RadixAttentionEngine::new(
//!     1 << 20,
//!     1 << 20,
//!     Arc::new(HazardManager::new()),
//!     CpuFallbackDriver::default(),
//! )
//! .expect("construction succeeds");
//! engine.insert(&[1, 2, 3], 100).expect("insert succeeds");
//! let m = engine
//!     .match_common_prefix(&[1, 2, 3, 4])
//!     .expect("match succeeds")
//!     .expect("prefix found");
//! assert_eq!(m.token_len, 3);
//! assert_eq!(m.kv_offset, 100);
//! ```
#![no_std]

extern crate alloc;

#[cfg(test)]
extern crate std;

pub mod engine;

pub use engine::{PrefixMatch, RadixAttentionEngine};
