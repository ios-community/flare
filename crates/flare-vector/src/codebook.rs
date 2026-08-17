//! Product-quantization codebooks for residual compression.
//!
//! Each sub-vector of a residual (the vector minus its centroid) is
//! quantized to one of 256 centroids, producing one 8-bit code byte per
//! sub-vector. The asymmetric distance computation (ADC) precomputes a
//! distance table between a query and every codebook centroid per
//! sub-vector, then scores a stored vector by summing table lookups —
//! `O(sub_vectors)` per candidate regardless of dimension.

use crate::distance::l2_sq_dispatch;
use crate::kmeans::kmeans_l2;
use alloc::vec::Vec;
use flare_core::FlareError;

/// Number of 8-bit codes per sub-vector; a `u8` index into the codebook.
pub const CODES_PER_SUBVECTOR: usize = 256;

/// Product-quantization codebooks: 256 centroids per sub-vector.
///
/// The centroid data is laid out as `[sub][code][component]` in a single
/// flat `f32` slice, so `sub * 256 * dims_per_sub + code * dims_per_sub +
/// component` addresses any centroid component.
#[derive(Debug, Clone, PartialEq)]
pub struct PqCodebooks {
    dimension: usize,
    sub_vectors: usize,
    dims_per_sub: usize,
    centroids: Vec<f32>,
}

impl PqCodebooks {
    /// Creates empty codebooks for `dimension` split into `sub_vectors`
    /// sub-vectors.
    ///
    /// # Errors
    ///
    /// Returns [`FlareError::InvalidParameter`] when `dimension` or
    /// `sub_vectors` is zero, or when `dimension` is not divisible by
    /// `sub_vectors`.
    ///
    /// # Examples
    ///
    /// ```
    /// use flare_vector::codebook::PqCodebooks;
    /// let books = PqCodebooks::new(128, 16).expect("valid shape");
    /// assert_eq!(books.dimension(), 128);
    /// assert_eq!(books.sub_vectors(), 16);
    /// ```
    pub const fn new(dimension: usize, sub_vectors: usize) -> Result<Self, FlareError> {
        if dimension == 0 {
            return Err(FlareError::InvalidParameter {
                reason: "dimension is zero",
            });
        }
        if sub_vectors == 0 {
            return Err(FlareError::InvalidParameter {
                reason: "sub-vector count is zero",
            });
        }
        if !dimension.is_multiple_of(sub_vectors) {
            return Err(FlareError::InvalidParameter {
                reason: "dimension not divisible by sub-vector count",
            });
        }
        Ok(Self {
            dimension,
            sub_vectors,
            dims_per_sub: dimension / sub_vectors,
            centroids: Vec::new(),
        })
    }

    /// Rebuilds the codebooks over a flat centroid slice.
    ///
    /// The slice must hold exactly `sub_vectors * 256 * dims_per_sub`
    /// `f32` values in `[sub][code][component]` order.
    pub(crate) fn from_centroids(
        dimension: usize,
        sub_vectors: usize,
        centroids: Vec<f32>,
    ) -> Result<Self, FlareError> {
        let dims_per_sub = dimension / sub_vectors;
        let expected = sub_vectors * CODES_PER_SUBVECTOR * dims_per_sub;
        if centroids.len() != expected {
            return Err(FlareError::InvalidParameter {
                reason: "codebook centroid slice has the wrong length",
            });
        }
        Ok(Self {
            dimension,
            sub_vectors,
            dims_per_sub,
            centroids,
        })
    }

    /// Trains every sub-vector codebook with Lloyd k-means over the
    /// sub-vector slices of `samples`.
    ///
    /// `samples` must hold whole rows of `dimension` values; each row's
    /// `s`-th sub-vector feeds the `s`-th codebook. The training is
    /// deterministic for a fixed `(samples, seed)` pair.
    ///
    /// # Errors
    ///
    /// Returns [`FlareError::VectorDimensionMismatch`] when `samples` is
    /// not a multiple of `dimension`, and
    /// [`FlareError::InvalidParameter`] when fewer than 256 rows are
    /// supplied or when the k-means parameters are rejected.
    ///
    /// # Examples
    ///
    /// ```
    /// use flare_vector::codebook::PqCodebooks;
    /// let mut books = PqCodebooks::new(4, 2).expect("valid shape");
    /// let samples: Vec<f32> = (0..600)
    ///     .flat_map(|i| {
    ///         let v = (i % 64) as f32;
    ///         [v, v, v, v]
    ///     })
    ///     .collect();
    /// books.train(&samples, 8, 3).expect("training succeeds");
    /// assert_eq!(books.sub_vectors(), 2);
    /// ```
    pub fn train(
        &mut self,
        samples: &[f32],
        iterations: usize,
        seed: u64,
    ) -> Result<(), FlareError> {
        if !samples.len().is_multiple_of(self.dimension) {
            return Err(FlareError::VectorDimensionMismatch {
                expected: self.dimension,
                got: samples.len(),
            });
        }
        let rows = samples.len() / self.dimension;
        if rows < CODES_PER_SUBVECTOR {
            return Err(FlareError::InvalidParameter {
                reason: "training set smaller than the codebook size",
            });
        }
        self.centroids =
            Vec::with_capacity(self.sub_vectors * CODES_PER_SUBVECTOR * self.dims_per_sub);
        let mut subdata = Vec::with_capacity(rows * self.dims_per_sub);
        for s in 0..self.sub_vectors {
            subdata.clear();
            for r in 0..rows {
                subdata.extend_from_slice(
                    &samples[r * self.dimension + s * self.dims_per_sub
                        ..r * self.dimension + (s + 1) * self.dims_per_sub],
                );
            }
            let centres = kmeans_l2(
                self.dims_per_sub,
                CODES_PER_SUBVECTOR,
                &subdata,
                iterations,
                seed.wrapping_add(s as u64 + 1),
            )?;
            self.centroids.extend_from_slice(&centres);
        }
        Ok(())
    }

    /// Quantizes a residual to one code byte per sub-vector.
    ///
    /// # Errors
    ///
    /// Returns [`FlareError::VectorDimensionMismatch`] when `residual` is
    /// not exactly `dimension` values long.
    ///
    /// # Panics
    ///
    /// The internal code index is bounded by `CODES_PER_SUBVECTOR` and
    /// always fits in `u8`; the cast cannot panic in practice.
    pub fn encode(&self, residual: &[f32]) -> Result<Vec<u8>, FlareError> {
        if residual.len() != self.dimension {
            return Err(FlareError::VectorDimensionMismatch {
                expected: self.dimension,
                got: residual.len(),
            });
        }
        let mut codes = Vec::with_capacity(self.sub_vectors);
        for s in 0..self.sub_vectors {
            let sub = &residual[s * self.dims_per_sub..(s + 1) * self.dims_per_sub];
            let base = s * CODES_PER_SUBVECTOR * self.dims_per_sub;
            let mut best = 0usize;
            let mut best_d = f32::INFINITY;
            for c in 0..CODES_PER_SUBVECTOR {
                let d = l2_sq_dispatch(
                    sub,
                    &self.centroids
                        [base + c * self.dims_per_sub..base + (c + 1) * self.dims_per_sub],
                );
                if d < best_d {
                    best_d = d;
                    best = c;
                }
            }
            codes.push(u8::try_from(best).expect("code index fits in u8"));
        }
        Ok(codes)
    }

    /// Builds the ADC distance table for `query` against every codebook
    /// centroid.
    ///
    /// The returned slice has `sub_vectors * 256` entries; entry
    /// `s * 256 + c` is the L2-squared distance between the `s`-th
    /// sub-vector of `query` and the `c`-th codebook centroid of
    /// sub-vector `s`. A stored code byte vector then scores in
    /// `O(sub_vectors)` table lookups.
    ///
    /// # Errors
    ///
    /// Returns [`FlareError::VectorDimensionMismatch`] when `query` is
    /// not exactly `dimension` values long.
    pub fn table(&self, query: &[f32]) -> Result<Vec<f32>, FlareError> {
        if query.len() != self.dimension {
            return Err(FlareError::VectorDimensionMismatch {
                expected: self.dimension,
                got: query.len(),
            });
        }
        let mut t = Vec::with_capacity(self.sub_vectors * CODES_PER_SUBVECTOR);
        for s in 0..self.sub_vectors {
            let sub = &query[s * self.dims_per_sub..(s + 1) * self.dims_per_sub];
            let base = s * CODES_PER_SUBVECTOR * self.dims_per_sub;
            for c in 0..CODES_PER_SUBVECTOR {
                t.push(l2_sq_dispatch(
                    sub,
                    &self.centroids
                        [base + c * self.dims_per_sub..base + (c + 1) * self.dims_per_sub],
                ));
            }
        }
        Ok(t)
    }

    /// Returns the full vector dimension of the codebooks.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns the number of sub-vectors (code bytes per vector).
    #[must_use]
    pub const fn sub_vectors(&self) -> usize {
        self.sub_vectors
    }

    /// Returns the number of components per sub-vector.
    #[must_use]
    pub const fn dims_per_sub(&self) -> usize {
        self.dims_per_sub
    }

    /// Returns the flat centroid storage for arena serialization.
    pub(crate) fn raw_centroids(&self) -> &[f32] {
        &self.centroids
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    use super::{CODES_PER_SUBVECTOR, PqCodebooks};
    use alloc::vec;
    use alloc::vec::Vec;
    use flare_core::FlareError;

    fn grid_samples(rows: usize) -> Vec<f32> {
        (0..rows)
            .flat_map(|i| {
                let v = (i % 64) as f32;
                [v, v, v, v]
            })
            .collect()
    }

    /// Verifies constructor validation.
    #[test]
    fn constructor_validation() {
        assert!(matches!(
            PqCodebooks::new(0, 2),
            Err(FlareError::InvalidParameter { .. })
        ));
        assert!(matches!(
            PqCodebooks::new(4, 0),
            Err(FlareError::InvalidParameter { .. })
        ));
        assert!(matches!(
            PqCodebooks::new(5, 2),
            Err(FlareError::InvalidParameter { .. })
        ));
        let books = PqCodebooks::new(8, 4).expect("valid shape");
        assert_eq!(books.dims_per_sub(), 2);
    }

    /// Verifies trained codebooks quantize a known residual and that
    /// encoding is stable across calls.
    #[test]
    fn train_and_encode_roundtrip() {
        let mut books = PqCodebooks::new(4, 2).expect("valid shape");
        books
            .train(&grid_samples(512), 8, 3)
            .expect("training succeeds");
        let residual = [3.0f32, 3.0, 3.0, 3.0];
        let a = books.encode(&residual).expect("encode succeeds");
        let b = books.encode(&residual).expect("encode succeeds");
        assert_eq!(a, b);
        assert_eq!(a.len(), 2);
        let table = books.table(&residual).expect("table succeeds");
        assert_eq!(table.len(), 2 * CODES_PER_SUBVECTOR);
        let scored = table[a[0] as usize] + table[CODES_PER_SUBVECTOR + a[1] as usize];
        assert!(scored.is_finite());
    }

    /// Verifies dimension validation on encode and table.
    #[test]
    fn rejects_dimension_mismatch() {
        let mut books = PqCodebooks::new(4, 2).expect("valid shape");
        books
            .train(&grid_samples(300), 4, 1)
            .expect("training succeeds");
        assert!(matches!(
            books.encode(&[1.0, 2.0, 3.0]),
            Err(FlareError::VectorDimensionMismatch { .. })
        ));
        assert!(matches!(
            books.table(&[1.0, 2.0]),
            Err(FlareError::VectorDimensionMismatch { .. })
        ));
    }

    /// Verifies training rejects undersized sample sets.
    #[test]
    fn rejects_undersized_training_set() {
        let mut books = PqCodebooks::new(4, 2).expect("valid shape");
        assert!(matches!(
            books.train(&grid_samples(100), 4, 1),
            Err(FlareError::InvalidParameter { .. })
        ));
        assert!(matches!(
            books.train(&[1.0, 2.0, 3.0], 4, 1),
            Err(FlareError::VectorDimensionMismatch { .. })
        ));
    }

    /// Verifies training is deterministic for a fixed seed.
    #[test]
    fn training_is_deterministic() {
        let mut a = PqCodebooks::new(4, 2).expect("valid shape");
        let mut b = PqCodebooks::new(4, 2).expect("valid shape");
        let samples = grid_samples(300);
        a.train(&samples, 6, 9).expect("training succeeds");
        b.train(&samples, 6, 9).expect("training succeeds");
        assert_eq!(a, b);
    }

    /// Verifies `from_centroids` round-trips the raw storage.
    #[test]
    fn from_centroids_roundtrip() {
        let mut books = PqCodebooks::new(4, 2).expect("valid shape");
        books
            .train(&grid_samples(300), 4, 2)
            .expect("training succeeds");
        let rebuilt = PqCodebooks::from_centroids(
            books.dimension(),
            books.sub_vectors(),
            books.raw_centroids().to_vec(),
        )
        .expect("rebuild succeeds");
        assert_eq!(rebuilt, books);
        assert!(PqCodebooks::from_centroids(4, 2, vec![1.0, 2.0]).is_err());
    }
}
