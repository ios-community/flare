//! High-dimensional vector search extension for FLARE.
//!
//! This crate implements the IVF-PQ engine specified by the FLARE design
//! document: an inverted-file index whose centroids are routed through a
//! radix tree (`O(log C)`), whose vectors are compressed with product
//! quantization (8-bit codes per sub-vector), and whose asymmetric
//! distance computation (ADC) runs on a runtime-dispatched `SIMD` kernel
//! with a portable scalar fallback. All index state is arena-resident and
//! published through a single atomic handoff word, so shadow
//! re-clustering swaps the working snapshot lock-free.
//!
//! The crate is `#![no_std] + alloc` and depends exclusively on public
//! `flare-core` abstractions. The only `unsafe` blocks in the crate are
//! confined to the `AVX2` distance kernels in [`distance`].
//!
//! # Examples
//!
//! ```
//! use flare_core::sync::gpu::CpuFallbackDriver;
//! use flare_core::sync::hazard::HazardManager;
//! use flare_vector::{IvfPqIndex, SearchResult};
//! use std::sync::Arc;
//!
//! let index = IvfPqIndex::new(
//!     4, 2, 2, 7, 1 << 20, Arc::new(HazardManager::new()), CpuFallbackDriver::default(),
//! ).expect("index construction succeeds");
//! let samples: Vec<f32> = (0..512)
//!     .flat_map(|i| {
//!         let base = if i % 2 == 0 { 10.0 } else { -10.0 };
//!         [base, base, base, base]
//!     })
//!     .collect();
//! index.train(&samples).expect("training succeeds");
//! index.insert(&[10.5, 10.5, 10.5, 10.5]).expect("insert succeeds");
//! let hits: Vec<SearchResult> = index
//!     .search(&[10.4, 10.4, 10.4, 10.4], 1)
//!     .expect("search succeeds");
//! assert_eq!(hits.len(), 1);
//! ```
//!
//! # Feature Gates
//!
//! - `simd` (default): enables runtime `AVX2` detection and the
//!   intrinsic-based kernels. The portable scalar kernel is always
//!   available as the fallback.

#![no_std]
#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]
#![deny(rustdoc::private_intra_doc_links)]
#![deny(rustdoc::missing_crate_level_docs)]
#![deny(rustdoc::invalid_codeblock_attributes)]
#![deny(rustdoc::invalid_html_tags)]
#![deny(rustdoc::invalid_rust_codeblocks)]
#![deny(rustdoc::bare_urls)]

extern crate alloc;

#[cfg(test)]
extern crate std;

pub mod codebook;
pub mod distance;
pub mod index;
pub mod kmeans;
pub mod rng;

pub use index::{IvfPqIndex, SearchResult};
