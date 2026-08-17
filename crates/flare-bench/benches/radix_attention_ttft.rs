use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use flare_core::sync::gpu::CpuFallbackDriver;
use flare_core::sync::hazard::HazardManager;
use flare_kv::RadixAttentionEngine;
use rand::Rng;
use rand::thread_rng;
use std::sync::Arc;

fn build_engine(
    capacity_bytes: usize,
    arena_capacity: usize,
) -> RadixAttentionEngine<CpuFallbackDriver> {
    let hazard = Arc::new(HazardManager::new());
    let driver = CpuFallbackDriver::default();
    RadixAttentionEngine::new(capacity_bytes, arena_capacity, hazard, driver)
        .expect("engine construction succeeds")
}

fn gen_tokens(count: usize) -> Vec<u32> {
    let mut rng = thread_rng();
    (0..count).map(|_| rng.r#gen::<u32>()).collect()
}

fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("radix_attention_insert");
    for capacity_bytes in [1 << 20, 1 << 22, 1 << 24] {
        let engine = build_engine(capacity_bytes, 1 << 22);
        let tokens = gen_tokens(1000);

        group.bench_with_input(
            BenchmarkId::new("insert", capacity_bytes),
            &(&engine, tokens, 1u64),
            |b, (engine, tokens, kv_base)| {
                let mut kv = *kv_base;
                b.iter(|| {
                    engine
                        .insert(black_box(tokens), black_box(kv))
                        .expect("insert succeeds");
                    kv += 1;
                });
            },
        );
    }
    group.finish();
}

fn bench_match(c: &mut Criterion) {
    let mut group = c.benchmark_group("radix_attention_match");
    for capacity_bytes in [1 << 20, 1 << 22, 1 << 24] {
        let engine = build_engine(capacity_bytes, 1 << 22);
        let tokens = gen_tokens(10000);
        for (i, chunk) in tokens.chunks(10).enumerate() {
            engine
                .insert(chunk, i as u64 + 1)
                .expect("prefill succeeds");
        }
        let query = gen_tokens(50);

        group.bench_with_input(
            BenchmarkId::new("match", capacity_bytes),
            &(&engine, query),
            |b, (engine, query)| {
                b.iter(|| {
                    engine
                        .match_common_prefix(black_box(query))
                        .expect("match succeeds")
                });
            },
        );
    }
    group.finish();
}

fn bench_mixed(c: &mut Criterion) {
    let mut group = c.benchmark_group("radix_attention_mixed");
    for capacity_bytes in [1 << 20, 1 << 22] {
        let engine = build_engine(capacity_bytes, 1 << 22);
        let tokens = gen_tokens(10000);
        for (i, chunk) in tokens.chunks(8).enumerate() {
            engine
                .insert(chunk, i as u64 + 1)
                .expect("prefill succeeds");
        }
        let queries: Vec<Vec<u32>> = (0..1000).map(|_| gen_tokens(8)).collect();

        group.bench_with_input(
            BenchmarkId::new("mixed_70_30", capacity_bytes),
            &(&engine, queries),
            |b, (engine, queries)| {
                let mut kv = 10000u64;
                b.iter(|| {
                    // 70% match, 30% insert
                    if rand::random::<f32>() < 0.7 {
                        let q = &queries[rand::random::<usize>() % queries.len()];
                        engine.match_common_prefix(black_box(q)).expect("match");
                    } else {
                        let t = gen_tokens(8);
                        engine.insert(black_box(&t), black_box(kv)).expect("insert");
                        kv += 1;
                    }
                });
            },
        );
    }
    group.finish();
}

fn bench_evict(c: &mut Criterion) {
    let mut group = c.benchmark_group("radix_attention_evict");
    for capacity_bytes in [1 << 20, 1 << 22] {
        let engine = build_engine(capacity_bytes, 1 << 22);

        group.bench_with_input(
            BenchmarkId::new("evict", capacity_bytes),
            &engine,
            |b, engine| {
                b.iter(|| {
                    black_box(engine.evict_clock_step(1024));
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_insert, bench_match, bench_mixed, bench_evict);
criterion_main!(benches);
