//! Concurrency and synchronisation primitives.
//!
//! This module hosts the Hazard Eras reclamation manager
//! ([`hazard::HazardManager`]) and the trait-based GPU synchronisation
//! abstraction ([`gpu::GpuSyncDriver`]) with its default CPU fallback
//! driver ([`gpu::CpuFallbackDriver`]).
//!
//! # Unsafe Confinement
//!
//! `unsafe` appears here only inside the reclamation internals of
//! [`hazard::HazardManager`] (slot and retired-queue access behind
//! `UnsafeCell`) and in the trait-defined pinned-arena destructor of the
//! GPU driver boundary. Both shapes carry a formally documented invariant
//! proof in their `# Safety`/invariant sections.
#![allow(unsafe_code)]

pub mod gpu;
pub mod hazard;

pub use gpu::{CpuFallbackDriver, GpuSyncDriver};
pub use hazard::{EraGuard, HazardManager};
