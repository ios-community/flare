//! GPU synchronisation abstraction.
//!
//! The [`GpuSyncDriver`] trait is the single boundary between the
//! platform-neutral engine and hardware GPU runtimes. Host crunching and
//! CI workloads use [`CpuFallbackDriver`]; CUDA deployments provide a
//! concrete driver in the `flare-ffi` crate.

use crate::alloc::pinned::{allocate_pinned_block, deallocate_pinned_block};
use crate::error::FlareError;
use core::sync::atomic::{AtomicPtr, AtomicU64, Ordering};

/// The maximum number of concurrent pinned blocks kept by the CPU fallback
/// driver.
pub const PINNED_SLOT_CAPACITY: usize = 64;

/// Abstract interface for GPU epoch synchronisation and pinned memory.
///
/// Implementors must satisfy the following contract:
///
/// - [`GpuSyncDriver::publish_epoch_fence`] must order all prior host
///   writes against every GPU reader stream, so reader warps can never
///   observe partial CAS writes.
/// - [`GpuSyncDriver::allocate_pinned_arena`] returns host memory mapped
///   for zero-copy device access (PCIe/NVLink), or a plain host block for
///   drivers pretending to be GPUs.
/// - Implementors are `Send + Sync` and may be shared across threads.
///
/// The raw pointer lifecycle in this trait crosses the crate's `unsafe`
/// confinement boundary: the actual block allocation and release live in
/// [`crate::alloc::pinned`], and this trait only forwards the ownership
/// edge mandated by the GPU memory-management section of the design.
#[allow(unsafe_code)]
pub trait GpuSyncDriver: Send + Sync {
    /// Publishes an epoch fence event to GPU streams.
    ///
    /// After this call returns, every GPU reader stream must observe the
    /// result of all host memory writes ordered before the fence. The
    /// epoch identifier is opaque guidance for the driver.
    ///
    /// # Errors
    ///
    /// Returns a driver-specific error when the GPU runtime fails to
    /// enqueue the fence.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flare_core::sync::gpu::CpuFallbackDriver;
    /// # use flare_core::sync::gpu::GpuSyncDriver;
    /// let driver = CpuFallbackDriver::default();
    /// driver.publish_epoch_fence(7).expect("fence succeeds");
    /// ```
    fn publish_epoch_fence(&self, epoch_id: u64) -> Result<(), FlareError>;

    /// Allocates host memory with zero-copy mapping properties.
    ///
    /// The returned pointer is 64-byte aligned and remains valid until
    /// [`Self::deallocate_pinned_arena`] is called with the same pointer
    /// and size. Implementors must return appropriately aligned memory
    /// and may only fail on allocation exhaustion.
    ///
    /// # Errors
    ///
    /// Returns [`FlareError::AllocationFailed`] on exhaustion, or a
    /// driver-specific error when the GPU runtime rejects the request.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flare_core::sync::gpu::CpuFallbackDriver;
    /// # use flare_core::sync::gpu::GpuSyncDriver;
    /// let driver = CpuFallbackDriver::default();
    /// let ptr = driver.allocate_pinned_arena(1024).expect("allocation succeeds");
    /// // SAFETY: the pointer belongs to `driver` and is not used concurrently.
    /// unsafe { driver.deallocate_pinned_arena(ptr, 1024).expect("deallocation succeeds") };
    /// ```
    fn allocate_pinned_arena(&self, size_bytes: usize) -> Result<*mut u8, FlareError>;

    /// Deallocates host pinned memory previously allocated with
    /// [`Self::allocate_pinned_arena`].
    ///
    /// # Safety
    ///
    /// The pointer must have been returned by a prior successful call to
    /// [`Self::allocate_pinned_arena`] on this driver, and must not be
    /// accessed concurrently with or after this call.
    ///
    /// # Errors
    ///
    /// Returns [`FlareError::UnknownPinnedPointer`] when the pointer was
    /// never allocated by this driver.
    unsafe fn deallocate_pinned_arena(
        &self,
        ptr: *mut u8,
        size_bytes: usize,
    ) -> Result<(), FlareError>;
}

/// The default driver: pure CPU execution with a sequential-consistency
/// epoch fence.
///
/// This driver is used for deterministic CPU-only deployments, CI, and
/// portability checks. Allocation properties:
///
/// - [`GpuSyncDriver::publish_epoch_fence`] executes
///   `core::sync::atomic::fence(Ordering::SeqCst)`, which is the strongest
///   host-side ordering and guarantees that prior host writes are visible
///   before the fence returns.
/// - Pinned arenas are 64-byte-aligned zeroed blocks from the global
///   allocator; on pure CPU systems this is already zero-copy, so no
///   device mapping is required.
///
/// # Examples
///
/// ```
/// # use flare_core::error::FlareError;
/// # use flare_core::sync::gpu::{CpuFallbackDriver, GpuSyncDriver};
/// let driver = CpuFallbackDriver::default();
/// driver.publish_epoch_fence(7).expect("fence succeeds");
/// let ptr = driver.allocate_pinned_arena(1024).expect("allocation succeeds");
/// // SAFETY: the pointer belongs to `driver` and is not used concurrently.
/// unsafe { driver.deallocate_pinned_arena(ptr, 1024).expect("deallocation succeeds") };
/// ```
#[derive(Debug)]
pub struct CpuFallbackDriver {
    slots: [AtomicPtr<u8>; PINNED_SLOT_CAPACITY],
    sizes: [AtomicU64; PINNED_SLOT_CAPACITY],
    last_epoch: AtomicU64,
}

impl CpuFallbackDriver {
    /// Returns the epoch identifier of the most recent published fence.
    ///
    /// Returns `u64::MAX` when no fence has been published yet.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flare_core::sync::gpu::CpuFallbackDriver;
    /// # use flare_core::sync::gpu::GpuSyncDriver;
    /// let driver = CpuFallbackDriver::default();
    /// assert_eq!(driver.last_epoch(), u64::MAX);
    /// driver.publish_epoch_fence(7).expect("fence succeeds");
    /// assert_eq!(driver.last_epoch(), 7);
    /// ```
    #[must_use]
    pub fn last_epoch(&self) -> u64 {
        self.last_epoch.load(Ordering::Relaxed)
    }

    /// Returns the number of pinned blocks currently registered.
    ///
    /// This is a diagnostic helper; allocation and deallocation run
    /// lock-free, so the count may drift while the pool is in use.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flare_core::sync::gpu::CpuFallbackDriver;
    /// # use flare_core::sync::gpu::GpuSyncDriver;
    /// let driver = CpuFallbackDriver::default();
    /// assert_eq!(driver.pinned_count(), 0);
    /// let ptr = driver.allocate_pinned_arena(1024).expect("allocation succeeds");
    /// assert_eq!(driver.pinned_count(), 1);
    /// // SAFETY: the pointer belongs to `driver` and is not used concurrently.
    /// unsafe { driver.deallocate_pinned_arena(ptr, 1024).expect("deallocation succeeds") };
    /// assert_eq!(driver.pinned_count(), 0);
    /// ```
    #[must_use]
    pub fn pinned_count(&self) -> usize {
        self.slots
            .iter()
            .filter(|slot| !slot.load(Ordering::Relaxed).is_null())
            .count()
    }
}

impl Default for CpuFallbackDriver {
    /// Creates a driver with an empty pinned-block registry.
    fn default() -> Self {
        Self {
            slots: core::array::from_fn(|_| AtomicPtr::new(core::ptr::null_mut())),
            sizes: core::array::from_fn(|_| AtomicU64::new(0)),
            last_epoch: AtomicU64::new(u64::MAX),
        }
    }
}

impl GpuSyncDriver for CpuFallbackDriver {
    fn publish_epoch_fence(&self, epoch_id: u64) -> Result<(), FlareError> {
        self.last_epoch.store(epoch_id, Ordering::Relaxed);
        // Architect directive: the CPU fallback must issue a
        // sequential-consistency fence so host reader threads observe all
        // prior writes in program order.
        core::sync::atomic::fence(Ordering::SeqCst);
        Ok(())
    }

    fn allocate_pinned_arena(&self, size_bytes: usize) -> Result<*mut u8, FlareError> {
        let ptr = allocate_pinned_block(size_bytes)?;
        // Register the block in the first free slot with a CAS so two
        // concurrent allocations never share a slot.
        for (slot, size_slot) in self.slots.iter().zip(self.sizes.iter()) {
            if slot
                .compare_exchange(
                    core::ptr::null_mut(),
                    ptr,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
            {
                size_slot.store(size_bytes as u64, Ordering::Relaxed);
                return Ok(ptr);
            }
        }
        // Registry full; the caller must fall back to a bump region.
        // SAFETY: the pointer was just allocated and is not yet shared.
        unsafe { deallocate_pinned_block(ptr, size_bytes) };
        Err(FlareError::AllocationFailed)
    }

    unsafe fn deallocate_pinned_arena(
        &self,
        ptr: *mut u8,
        size_bytes: usize,
    ) -> Result<(), FlareError> {
        for (slot, size_slot) in self.slots.iter().zip(self.sizes.iter()) {
            if slot.load(Ordering::Acquire) == ptr {
                let recorded = usize::try_from(size_slot.load(Ordering::Relaxed))
                    .expect("recorded size fits in usize");
                if recorded != size_bytes {
                    return Err(FlareError::UnknownPinnedPointer);
                }
                // The CAS guarantees only this caller releases the block.
                if slot
                    .compare_exchange(
                        ptr,
                        core::ptr::null_mut(),
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    )
                    .is_err()
                {
                    return Err(FlareError::UnknownPinnedPointer);
                }
                // SAFETY of this dealloc: the pointer and size were
                // recorded at allocation, the block is exclusively owned
                // by the caller, and it is detached from the registry
                // before release.
                unsafe { deallocate_pinned_block(ptr, size_bytes) };
                return Ok(());
            }
        }
        Err(FlareError::UnknownPinnedPointer)
    }
}

#[cfg(test)]
mod tests {
    use super::{CpuFallbackDriver, GpuSyncDriver, PINNED_SLOT_CAPACITY};
    use crate::error::FlareError;
    use alloc_crate::vec::Vec;

    /// Verifies that fences record the epoch and ride a `SeqCst` barrier.
    #[test]
    fn fence_records_epoch() {
        let driver = CpuFallbackDriver::default();
        assert_eq!(driver.last_epoch(), u64::MAX);
        driver.publish_epoch_fence(7).expect("fence succeeds");
        assert_eq!(driver.last_epoch(), 7);
    }

    /// Verifies that pinned arenas register and release cleanly.
    #[test]
    fn pinned_arenas_register_and_release() {
        let driver = CpuFallbackDriver::default();
        let ptr = driver
            .allocate_pinned_arena(1024)
            .expect("allocation succeeds");
        assert_eq!(driver.pinned_count(), 1);
        assert_eq!(ptr as usize % 64, 0);
        // SAFETY: the pointer belongs to the driver with the recorded size.
        unsafe {
            driver
                .deallocate_pinned_arena(ptr, 1024)
                .expect("deallocation succeeds");
        }
        assert_eq!(driver.pinned_count(), 0);
    }

    /// Verifies that unknown pointers are rejected on release.
    #[test]
    fn unknown_pointer_rejected() {
        let driver = CpuFallbackDriver::default();
        // SAFETY: the pointer is never dereferenced; the registry rejects it.
        unsafe {
            assert!(matches!(
                driver.deallocate_pinned_arena(core::ptr::null_mut(), 16),
                Err(FlareError::UnknownPinnedPointer)
            ));
        }
    }

    /// Verifies that size mismatches are refused on release.
    #[test]
    fn size_mismatch_rejected() {
        let driver = CpuFallbackDriver::default();
        let ptr = driver
            .allocate_pinned_arena(64)
            .expect("allocation succeeds");
        // SAFETY: the pointer is never dereferenced during the rejected call.
        unsafe {
            assert!(matches!(
                driver.deallocate_pinned_arena(ptr, 63),
                Err(FlareError::UnknownPinnedPointer)
            ));
            driver
                .deallocate_pinned_arena(ptr, 64)
                .expect("exact size releases");
        }
    }

    /// Verifies that the registry is capacity-bounded and recycles slots.
    #[test]
    fn registry_capacity_is_bounded() {
        let driver = CpuFallbackDriver::default();
        let mut blocks = Vec::with_capacity(PINNED_SLOT_CAPACITY);
        for _ in 0..PINNED_SLOT_CAPACITY {
            blocks.push(
                driver
                    .allocate_pinned_arena(1)
                    .expect("allocation succeeds"),
            );
        }
        assert!(driver.allocate_pinned_arena(1).is_err());
        assert_eq!(driver.pinned_count(), PINNED_SLOT_CAPACITY);
        for (i, ptr) in blocks.iter().enumerate() {
            // SAFETY: each pointer belongs to the driver with size 1.
            unsafe {
                let label = alloc_crate::format!("block {i} releases");
                driver.deallocate_pinned_arena(*ptr, 1).expect(&label);
            }
        }
        assert_eq!(driver.pinned_count(), 0);
        assert!(
            driver.allocate_pinned_arena(1).is_ok(),
            "recycled slot is reusable"
        );
    }
}
