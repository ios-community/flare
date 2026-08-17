//! Optional CUDA synchronization driver (`feature = "cuda"`).
//!
//! `CudaSyncDriver` implements the
//! `flare_core::sync::gpu::GpuSyncDriver` contract for
//! CUDA deployments. The CUDA Driver API is loaded dynamically at runtime
//! (never linked), so this module compiles on machines without a CUDA
//! toolkit and degrades gracefully to
//! `flare_core::error::FlareError::GpuDriverUnavailable` when no driver is
//! installed.
//!
//! Epoch fences are published by recording and synchronising a driver
//! event on the current context's legacy default stream, which orders all
//! prior host writes before the fence returns. Pinned arenas are mapped
//! with `CU_MEMHOSTALLOC_DEVICEMAP` for zero-copy device access.

use core::ffi::{c_char, c_int, c_void};
use core::ptr;
use core::sync::atomic::{AtomicPtr, AtomicU64, Ordering};
use flare_core::error::FlareError;
use flare_core::sync::gpu::{GpuSyncDriver, PINNED_SLOT_CAPACITY};

/// `CU_MEMHOSTALLOC_DEVICEMAP` from the CUDA Driver API.
const CU_MEMHOSTALLOC_DEVICEMAP: u32 = 0x02;

/// Platform glue for loading the CUDA Driver API at runtime.
#[cfg(windows)]
mod platform {
    use core::ffi::{c_char, c_void};

    /// UTF-16 library name with a null terminator.
    pub const LIBRARY_NAME: &[u16] = &[
        b'n' as u16,
        b'v' as u16,
        b'c' as u16,
        b'u' as u16,
        b'd' as u16,
        b'a' as u16,
        b'.' as u16,
        b'd' as u16,
        b'l' as u16,
        b'l' as u16,
        0,
    ];

    #[link(name = "kernel32")]
    unsafe extern "system" {
        pub fn LoadLibraryW(name: *const u16) -> *mut c_void;
        pub fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
    }
}

/// Platform glue for loading the CUDA Driver API at runtime.
#[cfg(all(unix, not(target_os = "macos")))]
mod platform {
    use core::ffi::{c_char, c_int, c_void};

    /// POSIX `dlopen` flags: resolve symbols lazily.
    pub const RTLD_LAZY: c_int = 1;
    /// SONAME of the NVIDIA driver library.
    pub const LIBRARY_NAME: &[u8] = b"libcuda.so.1\0";

    #[link(name = "dl")]
    unsafe extern "C" {
        pub fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
        pub fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    }
}

/// Platform glue for loading the CUDA Driver API at runtime.
#[cfg(target_os = "macos")]
mod platform {
    use core::ffi::{c_char, c_int, c_void};

    /// POSIX `dlopen` flags: resolve symbols lazily.
    pub const RTLD_LAZY: c_int = 1;
    /// SONAME of the NVIDIA driver library on macOS.
    pub const LIBRARY_NAME: &[u8] = b"libcuda.dylib\0";

    unsafe extern "C" {
        pub fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
        pub fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    }
}

/// Loads the CUDA Driver API library, returning a null handle when absent.
fn load_library() -> *mut c_void {
    #[cfg(windows)]
    {
        // SAFETY: `LIBRARY_NAME` is a valid null-terminated UTF-16 string.
        unsafe { platform::LoadLibraryW(platform::LIBRARY_NAME.as_ptr()) }
    }
    #[cfg(unix)]
    {
        // SAFETY: `LIBRARY_NAME` is a valid null-terminated C string.
        unsafe {
            platform::dlopen(
                platform::LIBRARY_NAME.as_ptr() as *const c_char,
                platform::RTLD_LAZY,
            )
        }
    }
}

/// Resolves one exported symbol, or null when the library is absent.
fn resolve(module: *mut c_void, name: &'static [u8]) -> *mut c_void {
    if module.is_null() {
        return ptr::null_mut();
    }
    #[cfg(windows)]
    {
        // SAFETY: `module` is a loaded library and `name` a C string.
        unsafe { platform::GetProcAddress(module, name.as_ptr() as *const c_char) }
    }
    #[cfg(unix)]
    {
        // SAFETY: `module` is a loaded library and `name` a C string.
        unsafe { platform::dlsym(module, name.as_ptr() as *const c_char) }
    }
}

/// Resolves one symbol as a typed function pointer.
///
/// # Safety
///
/// The caller must ensure `T` is the correct function-pointer type for the
/// exported symbol.
fn load_symbol<T: Copy>(module: *mut c_void, name: &'static [u8]) -> Option<T> {
    let raw = resolve(module, name);
    if raw.is_null() {
        return None;
    }
    // SAFETY: the OS loader returned a valid pointer for the named
    // function; function pointers and raw pointers share one machine word.
    Some(unsafe { core::mem::transmute_copy(&raw) })
}

/// The CUDA Driver API entry points used by the driver, all optional.
#[derive(Clone, Copy)]
struct Symbols {
    cu_init: Option<extern "C" fn(c_int) -> c_int>,
    cu_ctx_get_current: Option<extern "C" fn(*mut *mut c_void) -> c_int>,
    cu_event_create: Option<extern "C" fn(*mut *mut c_void, u32) -> c_int>,
    cu_event_record: Option<extern "C" fn(*mut c_void, *mut c_void) -> c_int>,
    cu_event_synchronize: Option<extern "C" fn(*mut c_void) -> c_int>,
    cu_event_destroy: Option<extern "C" fn(*mut c_void) -> c_int>,
    cu_mem_host_alloc: Option<extern "C" fn(*mut *mut c_void, usize, u32) -> c_int>,
    cu_mem_free_host: Option<extern "C" fn(*mut c_void) -> c_int>,
}

impl Symbols {
    /// Loads every entry point from `module`; missing symbols stay `None`.
    fn load(module: *mut c_void) -> Self {
        Self {
            cu_init: load_symbol(module, b"cuInit\0"),
            cu_ctx_get_current: load_symbol(module, b"cuCtxGetCurrent\0"),
            cu_event_create: load_symbol(module, b"cuEventCreate\0"),
            cu_event_record: load_symbol(module, b"cuEventRecord\0"),
            cu_event_synchronize: load_symbol(module, b"cuEventSynchronize\0"),
            cu_event_destroy: load_symbol(module, b"cuEventDestroy\0"),
            cu_mem_host_alloc: load_symbol(module, b"cuMemHostAlloc\0"),
            cu_mem_free_host: load_symbol(module, b"cuMemFreeHost\0"),
        }
    }
}

/// CUDA-backed [`GpuSyncDriver`] implementation.
///
/// Construction never fails and never touches the GPU: the driver library
/// is loaded lazily, and every operation reports
/// [`FlareError::GpuDriverUnavailable`] when the runtime (or an active CUDA
/// context) is absent. Pinned arenas are registered in a fixed-size
/// lock-free registry so that deallocation can validate ownership.
///
/// # Examples
///
/// ```
/// # use flare_core::error::FlareError;
/// # use flare_core::sync::gpu::GpuSyncDriver;
/// # use flare_ffi::cuda::CudaSyncDriver;
/// let driver = CudaSyncDriver::new();
/// if driver.is_available() {
///     driver.publish_epoch_fence(1).expect("fence succeeds");
/// }
/// ```
pub struct CudaSyncDriver {
    symbols: Symbols,
    last_epoch: AtomicU64,
    slots: [AtomicPtr<u8>; PINNED_SLOT_CAPACITY],
    sizes: [AtomicU64; PINNED_SLOT_CAPACITY],
}

impl CudaSyncDriver {
    /// Creates a driver and resolves the CUDA Driver API entry points.
    ///
    /// The constructor is infallible: a missing runtime is reported by the
    /// individual operations, never by construction.
    #[must_use]
    pub fn new() -> Self {
        let module = load_library();
        Self {
            symbols: Symbols::load(module),
            last_epoch: AtomicU64::new(u64::MAX),
            slots: core::array::from_fn(|_| AtomicPtr::new(ptr::null_mut())),
            sizes: core::array::from_fn(|_| AtomicU64::new(0)),
        }
    }

    /// Returns whether the CUDA Driver API could be resolved.
    #[must_use]
    pub fn is_available(&self) -> bool {
        self.symbols.cu_init.is_some()
    }

    /// Returns the epoch of the most recent published fence.
    ///
    /// Returns `u64::MAX` when no fence has been published yet.
    #[must_use]
    pub fn last_epoch(&self) -> u64 {
        self.last_epoch.load(Ordering::Relaxed)
    }

    /// Maps a non-zero CUDA status onto a driver error.
    fn driver_error() -> FlareError {
        FlareError::GpuDriverUnavailable {
            reason: "CUDA driver call failed",
        }
    }

    /// Registers a pinned block in the first free registry slot.
    fn register(&self, ptr: *mut u8, size_bytes: usize) -> Result<(), FlareError> {
        for (slot, size_slot) in self.slots.iter().zip(self.sizes.iter()) {
            if slot
                .compare_exchange(ptr::null_mut(), ptr, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                size_slot.store(size_bytes as u64, Ordering::Relaxed);
                return Ok(());
            }
        }
        Err(FlareError::AllocationFailed)
    }
}

impl Default for CudaSyncDriver {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuSyncDriver for CudaSyncDriver {
    fn publish_epoch_fence(&self, epoch_id: u64) -> Result<(), FlareError> {
        let Some(cu_init) = self.symbols.cu_init else {
            return Err(FlareError::GpuDriverUnavailable {
                reason: "CUDA runtime not loaded",
            });
        };
        let Some(cu_ctx_get_current) = self.symbols.cu_ctx_get_current else {
            return Err(Self::driver_error());
        };
        let Some(cu_event_create) = self.symbols.cu_event_create else {
            return Err(Self::driver_error());
        };
        let Some(cu_event_record) = self.symbols.cu_event_record else {
            return Err(Self::driver_error());
        };
        let Some(cu_event_synchronize) = self.symbols.cu_event_synchronize else {
            return Err(Self::driver_error());
        };
        let Some(cu_event_destroy) = self.symbols.cu_event_destroy else {
            return Err(Self::driver_error());
        };
        if cu_init(0) != 0 {
            return Err(Self::driver_error());
        }
        let mut context = ptr::null_mut();
        if cu_ctx_get_current(&mut context) != 0 || context.is_null() {
            return Err(FlareError::GpuDriverUnavailable {
                reason: "no current CUDA context",
            });
        }
        let mut event = ptr::null_mut();
        // SAFETY: `event` receives a driver-created handle.
        if cu_event_create(&mut event, 0) != 0 {
            return Err(Self::driver_error());
        }
        // SAFETY: `event` is a valid driver event; the null stream is the
        // current context's legacy default stream.
        let recorded = cu_event_record(event, ptr::null_mut());
        let synced = if recorded == 0 {
            // SAFETY: `event` is a valid driver event.
            cu_event_synchronize(event)
        } else {
            recorded
        };
        // SAFETY: `event` is a valid driver event; failure to destroy the
        // event is benign.
        let _ = cu_event_destroy(event);
        if synced != 0 {
            return Err(Self::driver_error());
        }
        self.last_epoch.store(epoch_id, Ordering::Relaxed);
        Ok(())
    }

    fn allocate_pinned_arena(&self, size_bytes: usize) -> Result<*mut u8, FlareError> {
        let Some(cu_mem_host_alloc) = self.symbols.cu_mem_host_alloc else {
            return Err(FlareError::AllocationFailed);
        };
        let Some(cu_mem_free_host) = self.symbols.cu_mem_free_host else {
            return Err(FlareError::AllocationFailed);
        };
        let mut ptr = ptr::null_mut();
        // SAFETY: `ptr` receives a driver-allocated host pointer.
        let status = cu_mem_host_alloc(&mut ptr, size_bytes, CU_MEMHOSTALLOC_DEVICEMAP);
        if status != 0 || ptr.is_null() {
            return Err(FlareError::AllocationFailed);
        }
        let host = ptr.cast::<u8>();
        match self.register(host, size_bytes) {
            Ok(()) => Ok(host),
            Err(error) => {
                // SAFETY: the block was just allocated and is unshared.
                let _ = cu_mem_free_host(ptr);
                Err(error)
            }
        }
    }

    unsafe fn deallocate_pinned_arena(
        &self,
        ptr: *mut u8,
        size_bytes: usize,
    ) -> Result<(), FlareError> {
        let Some(cu_mem_free_host) = self.symbols.cu_mem_free_host else {
            return Err(FlareError::UnknownPinnedPointer);
        };
        for (slot, size_slot) in self.slots.iter().zip(self.sizes.iter()) {
            if slot.load(Ordering::Acquire) == ptr {
                let recorded = usize::try_from(size_slot.load(Ordering::Relaxed))
                    .expect("recorded size fits in usize");
                if recorded != size_bytes {
                    return Err(FlareError::UnknownPinnedPointer);
                }
                if slot
                    .compare_exchange(ptr, ptr::null_mut(), Ordering::AcqRel, Ordering::Acquire)
                    .is_err()
                {
                    return Err(FlareError::UnknownPinnedPointer);
                }
                // SAFETY: ownership is exclusive per the trait contract,
                // and the block is detached before release.
                if cu_mem_free_host(ptr.cast()) != 0 {
                    return Err(Self::driver_error());
                }
                return Ok(());
            }
        }
        Err(FlareError::UnknownPinnedPointer)
    }
}

#[cfg(test)]
mod tests {
    use super::{CudaSyncDriver, GpuSyncDriver, PINNED_SLOT_CAPACITY};
    use alloc::vec::Vec;
    use flare_core::error::FlareError;

    /// Verifies the driver degrades gracefully when the runtime is absent.
    #[test]
    fn unavailable_runtime_is_reported() {
        let driver = CudaSyncDriver::new();
        if driver.is_available() {
            return;
        }
        assert_eq!(driver.last_epoch(), u64::MAX);
        assert!(matches!(
            driver.publish_epoch_fence(3),
            Err(FlareError::GpuDriverUnavailable { .. })
        ));
        assert!(matches!(
            driver.allocate_pinned_arena(64),
            Err(FlareError::AllocationFailed)
        ));
        // SAFETY: the pointer is never dereferenced.
        unsafe {
            assert!(matches!(
                driver.deallocate_pinned_arena(core::ptr::null_mut(), 64),
                Err(FlareError::UnknownPinnedPointer)
            ));
        }
    }

    /// Verifies the fence and pinned-arena paths against a live runtime;
    /// skipped silently when the runtime is absent.
    #[test]
    fn available_runtime_roundtrip() {
        let driver = CudaSyncDriver::new();
        if !driver.is_available() {
            return;
        }
        let ptr = driver
            .allocate_pinned_arena(1024)
            .expect("pinned allocation succeeds");
        assert_eq!(ptr as usize % 64, 0);
        // SAFETY: the pointer belongs to the driver with the recorded size.
        unsafe {
            driver
                .deallocate_pinned_arena(ptr, 1024)
                .expect("deallocation succeeds");
        }
        match driver.publish_epoch_fence(9) {
            Ok(()) => assert_eq!(driver.last_epoch(), 9),
            Err(FlareError::GpuDriverUnavailable { .. }) => {}
            Err(other) => panic!("unexpected fence error: {other}"),
        }
    }

    /// Verifies the pinned registry is capacity-bounded like the CPU
    /// fallback driver.
    #[test]
    fn registry_is_capacity_bounded() {
        let driver = CudaSyncDriver::new();
        if !driver.is_available() {
            return;
        }
        let mut blocks = Vec::with_capacity(PINNED_SLOT_CAPACITY);
        for _ in 0..PINNED_SLOT_CAPACITY {
            blocks.push(
                driver
                    .allocate_pinned_arena(1)
                    .expect("allocation succeeds"),
            );
        }
        assert!(driver.allocate_pinned_arena(1).is_err());
        for ptr in blocks {
            // SAFETY: each pointer belongs to the driver with size 1.
            unsafe {
                driver
                    .deallocate_pinned_arena(ptr, 1)
                    .expect("deallocation succeeds");
            }
        }
    }
}
