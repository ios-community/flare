//! Pinned host memory blocks for GPU-accessible arenas.
//!
//! The blocks returned by [`allocate_pinned_block`] are 64-byte aligned
//! zeroed regions from the global allocator. On systems with an actual GPU
//! runtime these blocks are the host-side half of a zero-copy mapping; the
//! allocation itself is platform-neutral, so this module is usable in
//! `#![no_std]` contexts.
//!
//! This module is one of the two `unsafe`-confined corners of the crate
//! (together with [`super::arena`]); the safety contract is documented on
//! each function.

use crate::error::FlareError;
use alloc_crate::alloc::{Layout, alloc_zeroed, dealloc};

/// Alignment of pinned host blocks: one cache line (64 bytes).
const PINNED_ALIGN: usize = 64;

/// Allocates a zeroed, 64-byte-aligned host block of `size_bytes`.
///
/// The returned pointer is valid until [`deallocate_pinned_block`] is
/// called with the same pointer and size. The block is exclusively owned
/// by the caller and is not shared with any device until the caller
/// publishes it through a fence-ordered protocol.
///
/// # Errors
///
/// Returns [`FlareError::AllocationFailed`] when the request is zero-sized
/// or the global allocator cannot back it.
pub fn allocate_pinned_block(size_bytes: usize) -> Result<*mut u8, FlareError> {
    if size_bytes == 0 {
        return Err(FlareError::AllocationFailed);
    }
    let layout = Layout::from_size_align(size_bytes, PINNED_ALIGN)
        .map_err(|_| FlareError::AllocationFailed)?;
    // SAFETY: `layout` is guaranteed valid by `from_size_align` and the
    // global allocator contract for `alloc_zeroed`.
    let ptr = unsafe { alloc_zeroed(layout) };
    if ptr.is_null() {
        return Err(FlareError::AllocationFailed);
    }
    Ok(ptr)
}

/// Releases a host block previously returned by [`allocate_pinned_block`].
///
/// # Safety
///
/// The caller must guarantee that:
/// - `ptr` was returned by [`allocate_pinned_block`] with the exact same
///   `size_bytes`, and has not already been released;
/// - no thread (host or device) accesses the block concurrently with or
///   after this call.
pub unsafe fn deallocate_pinned_block(ptr: *mut u8, size_bytes: usize) {
    // SAFETY: upheld by the caller contract above; the layout matches the
    // one used at allocation time.
    unsafe {
        let layout = Layout::from_size_align_unchecked(size_bytes, PINNED_ALIGN);
        dealloc(ptr, layout);
    }
}

#[cfg(test)]
mod tests {
    use super::{allocate_pinned_block, deallocate_pinned_block};

    /// Verifies that blocks round-trip allocation and release.
    #[test]
    fn pinned_blocks_roundtrip() {
        let ptr = allocate_pinned_block(4096).expect("allocation succeeds");
        assert!(!ptr.is_null());
        // SAFETY: the block is exclusively owned by this test.
        unsafe {
            core::ptr::write_bytes(ptr, 0xAB, 4096);
            deallocate_pinned_block(ptr, 4096);
        }
    }

    /// Verifies that blocks are 64-byte aligned.
    #[test]
    fn pinned_blocks_are_cache_line_aligned() {
        let ptr = allocate_pinned_block(1).expect("allocation succeeds");
        assert_eq!(ptr as usize % 64, 0);
        // SAFETY: the block is exclusively owned by this test.
        unsafe {
            deallocate_pinned_block(ptr, 1);
        }
    }

    /// Verifies that zero-sized requests fail cleanly instead of
    /// dereferencing a null pointer.
    #[test]
    fn zero_sized_requests_fail() {
        assert!(allocate_pinned_block(0).is_err());
    }
}
