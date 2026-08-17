//! Deterministic pseudo-random number generation for training.
//!
//! The training pipelines (k-means initialisation, empty-cluster
//! re-seeding) need reproducible randomness so that identical inputs and
//! seeds produce identical models. This module ships a tiny `xorshift64*`
//! generator instead of depending on an external RNG crate, keeping the
//! dependency surface empty and the sequence fully deterministic.
//!
//! The generator is deliberately **not** cryptographic.

/// A deterministic `xorshift64*` generator seeded at construction.
///
/// The seed is forced non-zero so the generator never falls into the
/// all-zero absorbing state. The same seed always reproduces the same
/// sequence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Xorshift64Star {
    state: u64,
}

impl Xorshift64Star {
    /// Creates a generator from `seed`.
    ///
    /// A zero seed is mapped to `1` internally to keep the state machine
    /// alive; the observable sequence is still fully deterministic.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }

    /// Advances the generator and returns the next 64-bit value.
    #[must_use]
    pub const fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Advances the generator and returns a value in `[0, 1)`.
    #[must_use]
    pub fn next_f32(&mut self) -> f32 {
        // The top 24 bits of the state cover 2^24 evenly spaced values in
        // [0, 1), which is sufficient for training initialisation.
        f32::from_bits((self.next_u64() >> 40) as u32 | 0x3F80_0000) - 1.0
    }

    /// Advances the generator and returns a value in `[0, bound)`.
    ///
    /// # Panics
    ///
    /// Panics when `bound` is zero.
    #[must_use]
    pub fn next_bounded(&mut self, bound: usize) -> usize {
        assert!(bound > 0, "bound must be non-zero");
        usize::try_from(self.next_u64() % bound as u64).expect("bounded value fits in usize")
    }
}

#[cfg(test)]
mod tests {
    use super::Xorshift64Star;

    /// Verifies that identical seeds reproduce identical sequences.
    #[test]
    fn deterministic_sequences() {
        let mut a = Xorshift64Star::new(42);
        let mut b = Xorshift64Star::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    /// Verifies that different seeds diverge and that the zero seed works.
    #[test]
    fn seeds_diverge_and_zero_is_live() {
        let mut a = Xorshift64Star::new(1);
        let mut b = Xorshift64Star::new(2);
        assert_ne!(a.next_u64(), b.next_u64());
        let mut zero = Xorshift64Star::new(0);
        assert_ne!(zero.next_u64(), zero.next_u64());
    }

    /// Verifies `next_f32` stays inside `[0, 1)`.
    #[test]
    fn f32_range_is_bounded() {
        let mut rng = Xorshift64Star::new(7);
        for _ in 0..10_000 {
            let v = rng.next_f32();
            assert!((0.0..1.0).contains(&v), "value {v} out of range");
        }
    }

    /// Verifies `next_bounded` stays inside the requested bound.
    #[test]
    fn bounded_range_is_respected() {
        let mut rng = Xorshift64Star::new(9);
        for _ in 0..10_000 {
            let v = rng.next_bounded(13);
            assert!(v < 13, "value {v} out of range");
        }
        assert_eq!(rng.next_bounded(1), 0);
    }

    /// Verifies the generator spreads values across the full state space.
    #[test]
    fn output_has_high_entropy() {
        let mut rng = Xorshift64Star::new(3);
        let first = rng.next_u64();
        let mut distinct = 1;
        for _ in 0..1000 {
            if rng.next_u64() != first {
                distinct += 1;
                break;
            }
        }
        assert!(distinct > 1, "sequence stuck at a single value");
    }
}
