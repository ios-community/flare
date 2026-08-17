use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use flare_core::alloc::arena::FlatArena;
use flare_core::sync::gpu::CpuFallbackDriver;
use flare_core::sync::hazard::HazardManager;
use flare_core::tree::FlareArtTree;
use std::sync::Arc;

fn build_tree(key_count: usize, key_len: usize) -> FlareArtTree<CpuFallbackDriver> {
    let arena = Arc::new(FlatArena::new(1 << 28).expect("arena fits"));
    let hazard = Arc::new(HazardManager::new());
    let driver = CpuFallbackDriver::default();
    let tree = FlareArtTree::new(arena, hazard, driver);

    for i in 0..key_count {
        let mut key = vec![0u8; key_len];
        key[0] = (i % 256) as u8;
        key[1] = ((i >> 8) % 256) as u8;
        if key_len > 2 {
            key[2] = ((i >> 16) % 256) as u8;
        }
        tree.insert(&key, i as u64).expect("insert succeeds");
    }
    tree
}

fn bench_get(c: &mut Criterion) {
    let mut group = c.benchmark_group("tree_get");
    for key_count in [1_000, 10_000, 100_000] {
        for key_len in [4, 8, 16] {
            let tree = build_tree(key_count, key_len);
            let mut query = vec![0u8; key_len];
            query[0] = 42;
            query[1] = 1;
            group.bench_with_input(
                BenchmarkId::new(format!("get_{key_len}B"), key_count),
                &(&tree, query),
                |b, (tree, query)| {
                    b.iter(|| tree.get(black_box(query)).expect("lookup succeeds"));
                },
            );
        }
    }
    group.finish();
}

fn bench_longest_prefix(c: &mut Criterion) {
    let mut group = c.benchmark_group("tree_longest_prefix");
    for key_count in [1_000, 10_000, 100_000] {
        for key_len in [4, 8, 16] {
            let tree = build_tree(key_count, key_len);
            let mut query = vec![0u8; key_len];
            query[0] = 42;
            query[1] = 1;
            group.bench_with_input(
                BenchmarkId::new(format!("lcp_{key_len}B"), key_count),
                &(&tree, query),
                |b, (tree, query)| {
                    b.iter(|| tree.longest_prefix(black_box(query)).expect("lcp succeeds"));
                },
            );
        }
    }
    group.finish();
}

fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("tree_insert");
    for key_len in [4, 8, 16] {
        group.bench_with_input(
            BenchmarkId::new(format!("insert_{key_len}B"), key_len),
            &key_len,
            |b, &key_len| {
                b.iter_batched(
                    || {
                        let arena = Arc::new(FlatArena::new(1 << 20).expect("arena fits"));
                        let hazard = Arc::new(HazardManager::new());
                        let driver = CpuFallbackDriver::default();
                        FlareArtTree::new(arena, hazard, driver)
                    },
                    |tree| {
                        let mut key = vec![0u8; key_len];
                        for i in 0..100 {
                            key[0] = (i % 256) as u8;
                            key[1] = ((i >> 8) % 256) as u8;
                            if key_len > 2 {
                                key[2] = ((i >> 16) % 256) as u8;
                            }
                            tree.insert(black_box(&key), black_box(i as u64))
                                .expect("insert succeeds");
                        }
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_get, bench_longest_prefix, bench_insert);
criterion_main!(benches);
