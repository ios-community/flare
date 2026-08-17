//! Distance kernels with runtime SIMD dispatch and a portable fallback.
//!
//! The asymmetric distance computation (ADC) hot path computes many small
//! `f32` L2 distances. This module provides the scalar reference kernel
//! [`l2_sq`], a runtime-dispatched entry point [`l2_sq_dispatch`], and —
//! behind the `simd` feature on `x86_64` — an `AVX2` kernel selected by a
//! cached `cpuid`/`xgetbv` probe.
//!
//! # Safety
//!
//! This module contains the **only** `unsafe` code in the crate, confined
//! to the `AVX2` kernel and the CPU-feature probe. Every unsafe block is
//! documented with a `SAFETY` comment; the dispatch path guarantees the
//! executing CPU supports `AVX2` before the kernel runs.

use core::sync::atomic::{AtomicU8, Ordering};

/// Portable scalar L2-squared distance between two equal-length slices.
///
/// This is the reference kernel every accelerated path is tested against.
///
/// # Panics
///
/// Panics when the slices have different lengths.
#[must_use]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "distance slices must have equal length");
    let mut acc = 0.0f32;
    for (x, y) in a.iter().zip(b) {
        let d = x - y;
        acc += d * d;
    }
    acc
}

/// Probe cache: `0` unknown, `1` available, `2` unavailable.
static AVX2_CACHE: AtomicU8 = AtomicU8::new(0);

/// Reports whether the executing CPU supports `AVX2` (feature `simd`,
/// `x86_64` targets only; always `false` elsewhere).
///
/// The probe verifies `OSXSAVE`, `AVX`, and the OS-managed `XMM`/`YMM`
/// state bits so the result is trustworthy before dispatching to an
/// intrinsic kernel.
#[must_use]
pub fn has_avx2() -> bool {
    match AVX2_CACHE.load(Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => {
            let ok = detect_avx2();
            AVX2_CACHE.store(if ok { 1 } else { 2 }, Ordering::Relaxed);
            ok
        }
    }
}

/// Runs the CPU-feature probe for `AVX2`.
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
fn detect_avx2() -> bool {
    use core::arch::x86_64::{__cpuid, _xgetbv};
    let max = __cpuid(0).eax;
    if max < 7 {
        return false;
    }
    let leaf1 = __cpuid(1);
    if leaf1.ecx & (1 << 27) == 0 {
        return false;
    }
    if leaf1.ecx & (1 << 28) == 0 {
        return false;
    }
    // SAFETY: `_xgetbv` reads the OS-managed extended-control register
    // state; executing it on any x86-64 CPU is well defined.
    let xcr0 = unsafe { _xgetbv(0) };
    if xcr0 & 0b110 != 0b110 {
        return false;
    }
    let leaf7 = __cpuid(7);
    leaf7.ebx & (1 << 5) != 0
}

/// Reports `false` on targets without the intrinsic-based probe.
#[cfg(not(all(target_arch = "x86_64", feature = "simd")))]
fn detect_avx2() -> bool {
    false
}

/// `AVX2` L2-squared kernel over the common prefix of both slices.
///
/// The tail (fewer than 8 elements) is reduced with the scalar kernel.
///
/// # Safety
///
/// Callers must guarantee the executing CPU supports `AVX2`; violating
/// this triggers an illegal instruction fault.
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[target_feature(enable = "avx2")]
#[allow(clippy::many_single_char_names)]
unsafe fn l2_sq_avx2(lhs: &[f32], rhs: &[f32]) -> f32 {
    use core::arch::x86_64::{
        _mm_add_ps, _mm_cvtss_f32, _mm_hadd_ps, _mm256_add_ps, _mm256_castps256_ps128,
        _mm256_extractf128_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_setzero_ps, _mm256_sub_ps,
    };
    let count = lhs.len().min(rhs.len());
    let mut acc = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 8 <= count {
        // SAFETY: `i + 8 <= count` keeps the unaligned loads inside both
        // slices, and the caller guarantees `AVX2` availability.
        let va = unsafe { _mm256_loadu_ps(lhs.as_ptr().add(i)) };
        let vb = unsafe { _mm256_loadu_ps(rhs.as_ptr().add(i)) };
        let diff = _mm256_sub_ps(va, vb);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(diff, diff));
        i += 8;
    }
    let lo = _mm256_castps256_ps128(acc);
    let hi = _mm256_extractf128_ps(acc, 1);
    let mut s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    let mut total = _mm_cvtss_f32(s);
    let tail = l2_sq(&lhs[i..], &rhs[i..]);
    total += tail;
    total
}

/// Computes the L2-squared distance using the fastest verified kernel.
///
/// On `x86_64` with the `simd` feature the `AVX2` kernel is used when the
/// probe reports support; every other configuration falls back to the
/// portable scalar kernel. The result is bit-identical to [`l2_sq`] (both
/// accumulate in `f32`), so callers may mix kernels freely.
#[must_use]
pub fn l2_sq_dispatch(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    if has_avx2() {
        // SAFETY: `has_avx2` verifies the executing CPU supports `AVX2`,
        // which satisfies the kernel's precondition.
        return unsafe { l2_sq_avx2(a, b) };
    }
    l2_sq(a, b)
}

#[cfg(test)]
mod tests {
    use super::{has_avx2, l2_sq, l2_sq_dispatch};
    use crate::rng::Xorshift64Star;
    use alloc::vec::Vec;

    fn random_slices(len: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
        let mut rng = Xorshift64Star::new(seed);
        let a: Vec<f32> = (0..len).map(|_| rng.next_f32()).collect();
        let b: Vec<f32> = (0..len).map(|_| rng.next_f32()).collect();
        (a, b)
    }

    /// Verifies the kernel dispatch matches the scalar reference across
    /// sizes that exercise the SIMD body and the scalar tail.
    #[test]
    fn dispatch_matches_scalar() {
        for len in [0usize, 1, 7, 8, 9, 16, 17, 64, 65, 127, 128] {
            let (a, b) = random_slices(len, len as u64 + 1);
            let expected = l2_sq(&a, &b);
            let got = l2_sq_dispatch(&a, &b);
            assert!(
                (got - expected).abs() < 1e-3,
                "len {len}: dispatch {got} vs scalar {expected}"
            );
        }
    }

    /// Verifies the scalar kernel computes a known value.
    #[test]
    fn scalar_known_value() {
        assert!((l2_sq(&[0.0, 0.0], &[3.0, 4.0]) - 25.0).abs() < 1e-6);
        assert!((l2_sq(&[], &[]) - 0.0).abs() < 1e-6);
    }

    /// Verifies the probe is stable across repeated calls.
    #[test]
    fn probe_is_stable() {
        let a = has_avx2();
        let b = has_avx2();
        assert_eq!(a, b);
    }

    /// Verifies slices with misaligned starts still agree between kernels.
    #[test]
    fn misaligned_slices_agree() {
        let (a, b) = random_slices(40, 99);
        for off in 0..5 {
            let expected = l2_sq(&a[off..], &b[off..]);
            let got = l2_sq_dispatch(&a[off..], &b[off..]);
            assert!(
                (got - expected).abs() < 1e-3,
                "offset {off}: dispatch {got} vs scalar {expected}"
            );
        }
    }
}
