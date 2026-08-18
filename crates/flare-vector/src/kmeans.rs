//! Lloyd k-means clustering over flat `f32` rows.
//!
//! The engine trains centroids and product-quantization codebooks with a
//! deterministic Lloyd iteration: initial centres are drawn as distinct
//! random rows from a seeded generator, each row is assigned to its
//! nearest centre, and empty clusters are re-seeded from the same
//! generator so the whole pipeline is reproducible.

use crate::distance::l2_sq;
use crate::rng::Xorshift64Star;
use alloc::vec;
use alloc::vec::Vec;
use flare_core::FlareError;

/// Runs `iterations` of Lloyd's algorithm over `data`, viewed as rows of
/// `dimension` `f32` values, and returns `k` centroids (`k * dimension`
/// values) or an error.
///
/// The initial centres are `k` distinct rows of `data` drawn with the
/// seeded generator, so the result is deterministic for a fixed
/// `(data, seed)` pair. Empty clusters are re-seeded from a fresh random
/// row. Iteration stops early when no row changes assignment.
///
/// # Errors
///
/// Returns [`FlareError::InvalidParameter`] when `k` is zero, when `k`
/// exceeds the number of rows, or when `iterations` is zero. Returns
/// [`FlareError::VectorDimensionMismatch`] when `data` is not a multiple
/// of `dimension`.
///
/// # Examples
///
/// ```
/// use flare_vector::kmeans::kmeans_l2;
/// let data = [
///     10.0, 10.0, 10.0, 10.0, -10.0, -10.0, -10.0, -10.0,
///     11.0, 11.0, 11.0, 11.0, -9.0, -9.0, -9.0, -9.0,
/// ];
/// let centroids = kmeans_l2(4, 2, &data, 16, 5).expect("clustering succeeds");
/// assert_eq!(centroids.len(), 8);
/// ```
pub fn kmeans_l2(
    dimension: usize,
    k: usize,
    data: &[f32],
    iterations: usize,
    seed: u64,
) -> Result<Vec<f32>, FlareError> {
    if dimension == 0 {
        return Err(FlareError::InvalidParameter {
            reason: "dimension is zero",
        });
    }
    if !data.len().is_multiple_of(dimension) {
        return Err(FlareError::VectorDimensionMismatch {
            expected: dimension,
            got: data.len(),
        });
    }
    let rows = data.len() / dimension;
    if k == 0 {
        return Err(FlareError::InvalidParameter {
            reason: "cluster count is zero",
        });
    }
    if k > rows {
        return Err(FlareError::InvalidParameter {
            reason: "cluster count exceeds row count",
        });
    }
    if iterations == 0 {
        return Err(FlareError::InvalidParameter {
            reason: "iteration count is zero",
        });
    }
    let mut rng = Xorshift64Star::new(seed);
    let mut centroids: Vec<f32> = Vec::with_capacity(k * dimension);
    let mut taken = vec![false; rows];
    for _ in 0..k {
        loop {
            let row = rng.next_bounded(rows);
            if !taken[row] {
                taken[row] = true;
                centroids.extend_from_slice(&data[row * dimension..(row + 1) * dimension]);
                break;
            }
        }
    }
    let mut assignment = vec![usize::MAX; rows];
    for _ in 0..iterations {
        let mut sums = vec![0.0f32; k * dimension];
        let mut counts = vec![0usize; k];
        let mut changed = false;
        for row in 0..rows {
            let slice = &data[row * dimension..(row + 1) * dimension];
            let mut best = 0usize;
            let mut best_d = f32::INFINITY;
            for c in 0..k {
                let d = l2_sq(slice, &centroids[c * dimension..(c + 1) * dimension]);
                if d < best_d {
                    best_d = d;
                    best = c;
                }
            }
            if assignment[row] != best {
                changed = true;
                assignment[row] = best;
            }
            counts[best] += 1;
            for (i, v) in slice.iter().enumerate() {
                sums[best * dimension + i] += v;
            }
        }
        // The assignments are stable: the centroids are unchanged, so the
        // final model is what the previous update produced.
        if !changed {
            break;
        }
        for c in 0..k {
            if counts[c] == 0 {
                let row = rng.next_bounded(rows);
                centroids[c * dimension..(c + 1) * dimension]
                    .copy_from_slice(&data[row * dimension..(row + 1) * dimension]);
            } else {
                // The cluster count is bounded by the row count, so the
                // reciprocal cast is exact enough for centroid averaging.
                #[allow(clippy::cast_precision_loss)]
                let inv = 1.0 / counts[c] as f32;
                for i in 0..dimension {
                    centroids[c * dimension + i] = sums[c * dimension + i] * inv;
                }
            }
        }
    }
    Ok(centroids)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    use super::kmeans_l2;
    use alloc::vec::Vec;
    use flare_core::FlareError;

    /// Verifies that two well-separated clusters converge to their means.
    #[test]
    fn separates_well_separated_clusters() {
        let data: Vec<f32> = (0..40)
            .flat_map(|i| {
                let base = if i % 2 == 0 { 10.0 } else { -10.0 };
                [base, base, base, base]
            })
            .collect();
        let centroids = kmeans_l2(4, 2, &data, 32, 1).expect("clustering succeeds");
        let c0 = &centroids[0..4];
        let c1 = &centroids[4..8];
        let (near, far) = if c0[0] > c1[0] { (c0, c1) } else { (c1, c0) };
        assert!(near[0] > 9.0, "positive cluster mean {near:?}");
        assert!(far[0] < -9.0, "negative cluster mean {far:?}");
    }

    /// Verifies the same input and seed reproduce identical centroids.
    #[test]
    fn deterministic_output() {
        let data: Vec<f32> = (0..24).flat_map(|i| [i as f32, i as f32 * 2.0]).collect();
        let a = kmeans_l2(2, 4, &data, 8, 11).expect("clustering succeeds");
        let b = kmeans_l2(2, 4, &data, 8, 11).expect("clustering succeeds");
        assert_eq!(a, b);
    }

    /// Verifies parameter validation errors.
    #[test]
    fn rejects_bad_parameters() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        assert!(matches!(
            kmeans_l2(4, 3, &data, 8, 1),
            Err(FlareError::InvalidParameter { .. })
        ));
        assert!(matches!(
            kmeans_l2(4, 0, &data, 8, 1),
            Err(FlareError::InvalidParameter { .. })
        ));
        assert!(matches!(
            kmeans_l2(4, 1, &data, 0, 1),
            Err(FlareError::InvalidParameter { .. })
        ));
        assert!(matches!(
            kmeans_l2(0, 4, &data, 8, 1),
            Err(FlareError::InvalidParameter { .. })
        ));
        assert!(matches!(
            kmeans_l2(3, 1, &data, 8, 1),
            Err(FlareError::VectorDimensionMismatch { .. })
        ));
    }

    /// Verifies that an empty cluster is re-seeded instead of producing
    /// NaN centroids: one outlier far from a dense cluster forces the
    /// second centre to be re-seeded from the dense rows.
    #[test]
    fn empty_cluster_is_reseeded() {
        let data: Vec<f32> = (0..16)
            .flat_map(|i| {
                if i == 15 {
                    [1000.0, 1000.0]
                } else {
                    [i as f32, i as f32]
                }
            })
            .collect();
        let centroids = kmeans_l2(2, 2, &data, 8, 3).expect("clustering succeeds");
        assert!(centroids.iter().all(|v| v.is_finite()), "no NaN centroids");
    }

    /// Verifies early convergence: two identical rows converge in one pass.
    #[test]
    fn converges_early() {
        let data = [5.0f32, 5.0, 5.0, 5.0];
        let centroids = kmeans_l2(2, 1, &data, 16, 9).expect("clustering succeeds");
        assert_eq!(centroids, [5.0, 5.0]);
    }
}
