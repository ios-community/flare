use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use flare_core::sync::gpu::CpuFallbackDriver;
use flare_core::sync::hazard::HazardManager;
use flare_vector::IvfPqIndex;
use rand::Rng;
use rand::thread_rng;
use std::sync::Arc;

fn gen_vectors(count: usize, dim: usize) -> Vec<f32> {
    let mut rng = thread_rng();
    (0..count * dim).map(|_| rng.r#gen::<f32>()).collect()
}

fn build_index(
    vector_count: usize,
    dim: usize,
    n_centroids: usize,
    sub_vectors: usize,
) -> IvfPqIndex<CpuFallbackDriver> {
    let hazard = Arc::new(HazardManager::new());
    let driver = CpuFallbackDriver::default();

    let index = IvfPqIndex::new(dim, n_centroids, sub_vectors, 42, 1 << 26, hazard, driver)
        .expect("index construction succeeds");

    let training = gen_vectors(512, dim);
    index.train(&training).expect("training succeeds");

    let vectors = gen_vectors(vector_count, dim);
    for v in vectors.chunks(dim) {
        index.insert(v).expect("insert succeeds");
    }
    index
}

fn bench_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_search");
    for (vector_count, dim, n_centroids, sub_vectors, top_k) in [
        (10_000, 128, 256, 16, 10),
        (50_000, 128, 256, 16, 10),
        (10_000, 64, 128, 8, 20),
    ] {
        let index = build_index(vector_count, dim, n_centroids, sub_vectors);
        let query = vec![0.5f32; dim];

        group.bench_with_input(
            BenchmarkId::new(format!("search_{dim}D_{vector_count}"), vector_count),
            &(&index, query, top_k),
            |b, (index, query, top_k)| {
                b.iter(|| {
                    index
                        .search(black_box(query), *top_k)
                        .expect("search succeeds")
                });
            },
        );
    }
    group.finish();
}

fn bench_search_with_recluster(c: &mut Criterion) {
    let mut group = c.benchmark_group("search_with_recluster");
    group.sample_size(10);
    for vector_count in [10_000, 20_000] {
        let index = build_index(vector_count, 128, 256, 16);
        let query = vec![0.5f32; 128];

        group.bench_with_input(
            BenchmarkId::new("search_recluster", vector_count),
            &(&index, query, 10),
            |b, (index, query, top_k)| {
                b.iter(|| {
                    index
                        .search(black_box(query), *top_k)
                        .expect("search succeeds");
                    index.trigger_shadow_reclustering().expect("recluster");
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_search, bench_search_with_recluster);
criterion_main!(benches);
