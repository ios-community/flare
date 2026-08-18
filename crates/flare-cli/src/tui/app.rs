//! TUI application state: the three FLARE engines, shared counters, the
//! telemetry ring, a background workload thread, and channel-fed results
//! from chaos scenarios launched from the Chaos tab.

use crate::chaos::crash_injector::{CrashConfig, CrashReport};
use crate::chaos::memory_pressure::PressureReport;
use crate::chaos::storm::StormReport;
use crate::telemetry::{Collector, EventKind, EventWord};
use flare_core::alloc::arena::FlatArena;
use flare_core::error::FlareError;
use flare_core::sync::gpu::CpuFallbackDriver;
use flare_core::sync::hazard::HazardManager;
use flare_core::tree::FlareArtTree;
use flare_core::wal::{MemoryWalSink, WalFrame, WalTransaction};
use flare_kv::RadixAttentionEngine;
use flare_vector::IvfPqIndex;
use flare_vector::distance::{l2_sq, l2_sq_dispatch};
use flare_vector::rng::Xorshift64Star;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::{Duration, Instant};

/// Vector dimension used by the TUI workload.
const VECTOR_DIM: usize = 64;
/// Sub-vector count of the TUI's IVF-PQ index.
const VECTOR_SUBVECTORS: usize = 4;
/// Centroid count of the TUI's IVF-PQ index.
const VECTOR_CENTROIDS: usize = 16;
/// Training sample count at startup.
const VECTOR_TRAIN_SAMPLES: usize = 256;
/// Hot keyspace size shared by all KV workload inserts.
const HOT_KEYS: usize = 1 << 10;
/// Maximum event log lines kept for the footer.
const LOG_LINES: usize = 6;

/// One worker-thread heartbeat: how much work to do before sleeping.
const TICK_INSERTS: u64 = 64;
const TICK_GETS: u64 = 128;
const TICK_VECTOR_INSERTS: u64 = 2;
const TICK_KV_INSERTS: u64 = 4;

/// Aggregated result of a chaos scenario launched from the Chaos tab.
#[derive(Debug, Clone)]
pub enum ChaosResult {
    /// Contention storm audit report.
    Storm(StormReport),
    /// Crash fault-injection audit report.
    Crash(CrashReport),
    /// Memory exhaustion audit report.
    Pressure(PressureReport),
}

/// Point-in-time counter values shared with the UI thread.
#[derive(Debug, Clone, Copy, Default)]
pub struct CounterSnapshot {
    /// Successful radix tree inserts.
    pub inserts: u64,
    /// Successful radix tree lookups.
    pub hits: u64,
    /// Missed radix tree lookups.
    pub misses: u64,
    /// CAS race probes observed.
    pub contended: u64,
    /// Vector appends.
    pub vector_ops: u64,
    /// KV-cache insert/match operations.
    pub kv_ops: u64,
    /// WAL batches flushed.
    pub wal_frames: u64,
    /// Slots evicted by the clock sweep.
    pub evictions: u64,
    /// Engine errors observed by the workload.
    pub errors: u64,
}

/// Engine counters shared between the workload and the UI thread.
#[derive(Debug)]
pub struct Counters {
    inserts: Arc<AtomicU64>,
    hits: Arc<AtomicU64>,
    misses: Arc<AtomicU64>,
    contended: Arc<AtomicU64>,
    vector_ops: Arc<AtomicU64>,
    kv_ops: Arc<AtomicU64>,
    wal_frames: Arc<AtomicU64>,
    evictions: Arc<AtomicU64>,
    errors: Arc<AtomicU64>,
}

impl Counters {
    /// Creates a set of zeroed counters.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inserts: Arc::new(AtomicU64::new(0)),
            hits: Arc::new(AtomicU64::new(0)),
            misses: Arc::new(AtomicU64::new(0)),
            contended: Arc::new(AtomicU64::new(0)),
            vector_ops: Arc::new(AtomicU64::new(0)),
            kv_ops: Arc::new(AtomicU64::new(0)),
            wal_frames: Arc::new(AtomicU64::new(0)),
            evictions: Arc::new(AtomicU64::new(0)),
            errors: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Reads all counters into an immutable snapshot.
    #[must_use]
    pub fn snapshot(&self) -> CounterSnapshot {
        CounterSnapshot {
            inserts: self.inserts.load(Ordering::Relaxed),
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            contended: self.contended.load(Ordering::Relaxed),
            vector_ops: self.vector_ops.load(Ordering::Relaxed),
            kv_ops: self.kv_ops.load(Ordering::Relaxed),
            wal_frames: self.wal_frames.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            errors: self.errors.load(Ordering::Relaxed),
        }
    }
}

impl Default for Counters {
    fn default() -> Self {
        Self::new()
    }
}

/// Complete state of the TUI application.
#[allow(clippy::struct_excessive_bools)]
pub struct TuiApp {
    /// Radix tree engine (arena inspected by the Memory tab).
    pub tree: Arc<FlareArtTree<CpuFallbackDriver>>,
    /// IVF-PQ vector index (Vector tab).
    pub vector: Arc<IvfPqIndex<CpuFallbackDriver>>,
    /// Radix KV-cache engine (KV-Cache tab).
    pub kv: Arc<RadixAttentionEngine<CpuFallbackDriver>>,
    /// WAL sink flushed by the workload.
    pub sink: Arc<MemoryWalSink>,
    /// Hazard manager (era + retired list sizes).
    pub hazard: Arc<HazardManager>,
    /// Shared engine counters.
    pub counters: Arc<Counters>,
    /// Telemetry ring shared with the workload.
    pub ring: Arc<Collector>,
    /// Last decoded telemetry lines for the footer.
    pub log: VecDeque<String>,
    /// Selected tab index.
    pub tab: usize,
    /// Workload paused indicator.
    pub paused: bool,
    /// Shutdown flag for the workload thread.
    pub shutdown: Arc<AtomicBool>,
    /// Pause flag honoured by the workload thread.
    paused_shared: Arc<AtomicBool>,
    /// Workload thread join handle.
    workload: Option<std::thread::JoinHandle<()>>,
    /// SIMD-vs-scalar benchmark result (nanoseconds per distance).
    pub bench_scalar_ns: f64,
    /// SIMD-vs-scalar benchmark result (nanoseconds per distance).
    pub bench_avx2_ns: f64,
    /// Published re-clustering generation.
    pub recluster_gen: u64,
    /// Chaos scenario results delivered from background threads.
    pub chaos_rx: Receiver<ChaosResult>,
    /// Chaos scenario sender kept alive for background threads.
    chaos_tx: Sender<ChaosResult>,
    /// Number of chaos scenarios currently running in the background.
    pub chaos_inflight: usize,
    /// Whether the contention storm scenario is running.
    pub storm_running: bool,
    /// Whether the crash fault-injection scenario is running.
    pub crash_running: bool,
    /// Whether the memory exhaustion scenario is running.
    pub pressure_running: bool,
    /// Quit requested by the user.
    pub quit: bool,
    /// Live per-second rates: inserts, hits, misses.
    pub rates: [f64; 3],
    /// Counter values of the previous rate sample.
    rates_prev: CounterSnapshot,
    /// Instant of the previous rate sample.
    rates_at: Instant,
    /// Last contention storm audit report.
    pub storm: Option<StormReport>,
    /// Last crash fault-injection audit report.
    pub crash: Option<CrashReport>,
    /// Last memory exhaustion audit report.
    pub pressure: Option<PressureReport>,
    /// Loaded configuration.
    pub config: crate::config::Config,
}

impl TuiApp {
    /// Constructs the engines, spawns the workload thread, and returns
    /// the ready-to-render application.
    ///
    /// # Errors
    ///
    /// Returns the underlying [`FlareError`] when an arena allocation
    /// fails.
    pub fn new(
        arena_capacity: usize,
        default_tab: usize,
        start_paused: bool,
        _refresh_ms: u64,
        config: crate::config::Config,
    ) -> Result<Self, FlareError> {
        let hazard = Arc::new(HazardManager::new());
        let tree = Arc::new(FlareArtTree::new(
            Arc::new(FlatArena::new(arena_capacity)?),
            Arc::clone(&hazard),
            CpuFallbackDriver::default(),
        ));
        let vector = Arc::new(IvfPqIndex::new(
            VECTOR_DIM,
            VECTOR_CENTROIDS,
            VECTOR_SUBVECTORS,
            0x7E57,
            arena_capacity,
            Arc::clone(&hazard),
            CpuFallbackDriver::default(),
        )?);
        let kv = Arc::new(RadixAttentionEngine::new(
            arena_capacity,
            arena_capacity,
            Arc::clone(&hazard),
            CpuFallbackDriver::default(),
        )?);
        let sink = Arc::new(MemoryWalSink::new());
        let counters = Arc::new(Counters::new());
        let ring = Arc::new(Collector::default());
        let shutdown = Arc::new(AtomicBool::new(false));
        let paused_shared = Arc::new(AtomicBool::new(start_paused));
        let (chaos_tx, chaos_rx) = channel();

        let workload = spawn_workload(
            Arc::clone(&tree),
            Arc::clone(&vector),
            Arc::clone(&kv),
            Arc::clone(&sink),
            Arc::clone(&counters),
            Arc::clone(&ring),
            Arc::clone(&shutdown),
            Arc::clone(&paused_shared),
        );

        Ok(Self {
            tree,
            vector,
            kv,
            sink,
            hazard,
            counters,
            ring,
            log: VecDeque::new(),
            tab: default_tab.min(4),
            paused: start_paused,
            shutdown,
            paused_shared,
            workload: Some(workload),
            bench_scalar_ns: 0.0,
            bench_avx2_ns: 0.0,
            recluster_gen: 0,
            chaos_rx,
            chaos_tx,
            chaos_inflight: 0,
            storm_running: false,
            crash_running: false,
            pressure_running: false,
            quit: false,
            rates: [0.0; 3],
            rates_prev: CounterSnapshot::default(),
            rates_at: Instant::now(),
            storm: None,
            crash: None,
            pressure: None,
            config,
        })
    }

    /// Stops the workload thread and waits for it to finish.
    pub fn shutdown(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        if let Some(handle) = self.workload.take() {
            let _ = handle.join();
        }
    }

    /// Drains new telemetry events into the footer log.
    pub fn pump_log(&mut self) {
        let mut events = Vec::new();
        self.ring.drain(&mut events, self.ring.len().min(64));
        for event in events {
            if event.kind != EventKind::CasContention && event.kind != EventKind::TreeHit {
                self.log.push_back(format!("{event}"));
            }
        }
        while self.log.len() > LOG_LINES {
            self.log.pop_front();
        }
    }

    /// Reads a finished chaos scenario, if any, and stores its report.
    pub fn pump_chaos(&mut self) {
        match self.chaos_rx.try_recv() {
            Ok(ChaosResult::Storm(report)) => {
                self.chaos_inflight = self.chaos_inflight.saturating_sub(1);
                self.storm_running = false;
                self.storm = Some(report);
            }
            Ok(ChaosResult::Crash(report)) => {
                self.chaos_inflight = self.chaos_inflight.saturating_sub(1);
                self.crash_running = false;
                self.crash = Some(report);
            }
            Ok(ChaosResult::Pressure(report)) => {
                self.chaos_inflight = self.chaos_inflight.saturating_sub(1);
                self.pressure_running = false;
                self.pressure = Some(report);
            }
            Err(_) => {}
        }
    }

    /// Recomputes the live per-second operation rates.
    #[allow(clippy::cast_precision_loss)]
    pub fn update_rates(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.rates_at).as_secs_f64();
        if elapsed < 0.1 {
            return;
        }
        let current = self.counters.snapshot();
        self.rates = [
            current.inserts.saturating_sub(self.rates_prev.inserts) as f64 / elapsed,
            current.hits.saturating_sub(self.rates_prev.hits) as f64 / elapsed,
            current.misses.saturating_sub(self.rates_prev.misses) as f64 / elapsed,
        ];
        self.rates_prev = current;
        self.rates_at = now;
    }

    /// Toggles the workload pause flag.
    pub fn toggle_pause(&mut self) {
        self.paused = !self.paused;
        self.paused_shared.store(self.paused, Ordering::Relaxed);
    }

    /// Requests a clean shutdown of the whole application.
    pub const fn request_quit(&mut self) {
        self.quit = true;
    }

    /// Starts the contention storm scenario on a background thread.
    pub fn launch_storm(&mut self) {
        self.chaos_inflight += 1;
        self.storm_running = true;
        let tx = self.chaos_tx.clone();
        std::thread::spawn(move || {
            let config = crate::chaos::storm::StormConfig {
                threads: 8,
                attempts: 200_000,
                keyspace: 1 << 12,
                arena_bytes: 1 << 26,
            };
            let report = crate::chaos::storm::contention_storm(config, None);
            let _ = tx.send(ChaosResult::Storm(report.expect("storm succeeds")));
        });
    }

    /// Starts the crash fault-injection scenario on a background thread.
    pub fn launch_crash(&mut self) {
        self.chaos_inflight += 1;
        self.crash_running = true;
        let tx = self.chaos_tx.clone();
        std::thread::spawn(move || {
            let report = crate::chaos::crash_injector::crash_fault_injection(CrashConfig::demo());
            let _ = tx.send(ChaosResult::Crash(report.expect("crash run succeeds")));
        });
    }

    /// Starts the memory exhaustion scenario on a background thread.
    pub fn launch_pressure(&mut self) {
        self.chaos_inflight += 1;
        self.pressure_running = true;
        let tx = self.chaos_tx.clone();
        std::thread::spawn(move || {
            let report = crate::chaos::memory_pressure::memory_exhaustion(1 << 20);
            let _ = tx.send(ChaosResult::Pressure(
                report.expect("pressure run succeeds"),
            ));
        });
    }

    /// Burst-inserts `count` keys into the tree from the UI thread.
    ///
    /// Returns the number of keys that were actually inserted; failures
    /// (arena exhaustion) simply reduce the count.
    pub fn burst_insert(&self, count: u64) -> u64 {
        let mut rng = Xorshift64Star::new(0xB57);
        let mut inserted = 0u64;
        for _ in 0..count {
            let key = format!("burst:{:08x}", rng.next_bounded(1 << 20));
            if self
                .tree
                .insert(key.as_bytes(), rng.next_u64() & 0xFF_FFFF)
                .is_ok()
            {
                inserted += 1;
            }
        }
        inserted
    }

    /// Runs the SIMD-vs-scalar distance benchmark once.
    #[allow(clippy::cast_precision_loss)]
    pub fn run_bench(&mut self) {
        let mut rng = Xorshift64Star::new(0xBE17);
        let a: Vec<f32> = (0..VECTOR_DIM).map(|_| rng.next_f32()).collect();
        let b: Vec<f32> = (0..VECTOR_DIM).map(|_| rng.next_f32()).collect();
        let runs = 100_000u32;
        let started = Instant::now();
        for _ in 0..runs {
            let _ = l2_sq(&a, &b);
        }
        let scalar_ns = started.elapsed().as_nanos() as f64 / f64::from(runs);
        let started_avx = Instant::now();
        for _ in 0..runs {
            let _ = l2_sq_dispatch(&a, &b);
        }
        let avx2_ns = started_avx.elapsed().as_nanos() as f64 / f64::from(runs);
        self.bench_scalar_ns = scalar_ns;
        self.bench_avx2_ns = avx2_ns;
    }

    /// Triggers a shadow re-clustering cycle.
    ///
    /// # Errors
    ///
    /// Returns the rendered engine error when the journal is too small.
    pub fn trigger_recluster(&mut self) -> Result<(), String> {
        self.vector
            .trigger_shadow_reclustering()
            .map_err(|e| e.to_string())?;
        self.recluster_gen += 1;
        self.ring.try_push(EventWord::recluster(self.recluster_gen));
        Ok(())
    }

    /// Inserts a fresh random chat prefix and reports its length.
    ///
    /// # Errors
    ///
    /// Returns the rendered engine error when the insert fails.
    pub fn new_chat_prefix(&self) -> Result<usize, String> {
        let mut rng = Xorshift64Star::new(0xC477);
        let tokens: Vec<u32> = (0..24).map(|_| bounded_u32(&mut rng, 1 << 16)).collect();
        let offset = 0x1000 + u64::from(bounded_u32(&mut rng, 1 << 12));
        self.kv.insert(&tokens, offset).map_err(|e| e.to_string())?;
        Ok(tokens.len())
    }

    /// Advances the clock sweep by one step.
    pub fn evict_step(&self, steps: usize) -> usize {
        let evicted = self.kv.evict_clock_step(steps);
        self.ring.try_push(EventWord::clock_evict(evicted));
        evicted
    }
}

/// Runs the background workload until `shutdown` flips.
///
/// The workload starts paused so the dashboard opens idle: press `SPACE`
/// in the TUI to start (or stop) the synthetic load.
#[allow(clippy::too_many_arguments)]
fn spawn_workload(
    tree: Arc<FlareArtTree<CpuFallbackDriver>>,
    vector: Arc<IvfPqIndex<CpuFallbackDriver>>,
    kv: Arc<RadixAttentionEngine<CpuFallbackDriver>>,
    sink: Arc<MemoryWalSink>,
    counters: Arc<Counters>,
    ring: Arc<Collector>,
    shutdown: Arc<AtomicBool>,
    paused: Arc<AtomicBool>,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let mut rng_state = Xorshift64Star::new(0x6F0D);
        let mut samples = Vec::with_capacity(VECTOR_TRAIN_SAMPLES * VECTOR_DIM);
        for _ in 0..VECTOR_TRAIN_SAMPLES {
            for _ in 0..VECTOR_DIM {
                samples.push(rng_state.next_f32());
            }
        }
        let _ = vector.train(&samples);
        let mut kv_tokens: Vec<u32> = Vec::new();
        let mut tick = 0u64;
        while !shutdown.load(Ordering::Relaxed) {
            if paused.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(20));
                continue;
            }
            for _ in 0..TICK_INSERTS {
                let key = format!("hot:{:08x}", rng_state.next_bounded(HOT_KEYS));
                let value = rng_state.next_u64() & 0xFF_FFFF;
                let seen = tree.get(key.as_bytes()).ok().flatten();
                match tree.insert(key.as_bytes(), value) {
                    Ok(old) => {
                        counters.inserts.fetch_add(1, Ordering::Relaxed);
                        if old != seen {
                            counters.contended.fetch_add(1, Ordering::Relaxed);
                        }
                        ring.try_push(EventWord::tree_insert(key.len(), value));
                    }
                    Err(_) => {
                        counters.errors.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
            for _ in 0..TICK_GETS {
                let key = format!("hot:{:08x}", rng_state.next_bounded(HOT_KEYS));
                match tree.get(key.as_bytes()) {
                    Ok(Some(_)) => counters.hits.fetch_add(1, Ordering::Relaxed),
                    Ok(None) => counters.misses.fetch_add(1, Ordering::Relaxed),
                    Err(_) => counters.errors.fetch_add(1, Ordering::Relaxed),
                };
            }
            for _ in 0..TICK_VECTOR_INSERTS {
                let mut vector_data = Vec::with_capacity(VECTOR_DIM);
                for _ in 0..VECTOR_DIM {
                    vector_data.push(rng_state.next_f32());
                }
                if vector.insert(&vector_data).is_ok() {
                    counters.vector_ops.fetch_add(1, Ordering::Relaxed);
                    ring.try_push(EventWord::vector_insert(VECTOR_DIM, tick));
                }
            }
            if tick.is_multiple_of(32) {
                let mut query = Vec::with_capacity(VECTOR_DIM);
                for _ in 0..VECTOR_DIM {
                    query.push(rng_state.next_f32());
                }
                if let Ok(hits) = vector.search(&query, 8) {
                    counters.kv_ops.fetch_add(1, Ordering::Relaxed);
                    ring.try_push(EventWord::vector_search(8, hits.len()));
                }
            }
            for _ in 0..TICK_KV_INSERTS {
                let len = 4 + rng_state.next_bounded(20);
                kv_tokens.clear();
                for _ in 0..len {
                    kv_tokens.push(bounded_u32(&mut rng_state, 1 << 16));
                }
                let offset = u64::from(bounded_u32(&mut rng_state, 1 << 16));
                if kv.insert(&kv_tokens, offset).is_ok()
                    && kv.match_common_prefix(&kv_tokens).is_ok()
                {
                    counters.kv_ops.fetch_add(1, Ordering::Relaxed);
                    ring.try_push(EventWord::kv_insert(len, offset));
                    ring.try_push(EventWord::kv_match(len));
                }
            }
            if tick.is_multiple_of(256)
                && let Ok(offset) = sink_alloc_and_flush(&tree, &sink, &mut rng_state)
            {
                ring.try_push(EventWord::wal_flush(offset));
                counters.wal_frames.fetch_add(1, Ordering::Relaxed);
            }
            tick += 1;
            std::thread::sleep(Duration::from_millis(4));
        }
    })
}

/// Draws a bounded value from the generator as a `u32`.
fn bounded_u32(rng: &mut Xorshift64Star, bound: usize) -> u32 {
    u32::try_from(rng.next_bounded(bound)).expect("bound fits in u32")
}

/// Allocates one word in the tree arena and flushes it as a WAL frame.
fn sink_alloc_and_flush(
    tree: &FlareArtTree<CpuFallbackDriver>,
    sink: &MemoryWalSink,
    rng: &mut Xorshift64Star,
) -> Result<u64, FlareError> {
    let arena = tree.arena();
    let offset = arena.alloc(8, 8)?;
    let value = rng.next_u64();
    arena.write_node(offset, &value)?;
    let tx = WalTransaction::new(
        vec![WalFrame::alloc(offset, 8)],
        WalFrame::update(offset, value.to_le_bytes().to_vec()),
    );
    tx.commit(sink)?;
    Ok(offset)
}

#[cfg(test)]
mod tests {
    use super::{CounterSnapshot, Counters, TuiApp};
    use crate::config::Config;
    use std::sync::atomic::Ordering;

    /// Verifies that counter snapshots accumulate work.
    #[test]
    fn counters_snapshot_accumulates() {
        let counters = Counters::new();
        counters.inserts.fetch_add(5, Ordering::Relaxed);
        counters.hits.fetch_add(3, Ordering::Relaxed);
        let snapshot: CounterSnapshot = counters.snapshot();
        assert_eq!(snapshot.inserts, 5);
        assert_eq!(snapshot.hits, 3);
        assert_eq!(snapshot.misses, 0);
    }

    /// Verifies that the app constructs and shuts down cleanly.
    #[test]
    fn app_constructs_and_shuts_down() {
        // Use smaller arena for faster test execution on CI
        let mut app = TuiApp::new(1 << 20, 0, true, 30, Config::default()).expect("app construction succeeds");
        // Extended timeout for CI environments (5 minutes)
        let deadline = std::time::Instant::now() + std::time::Duration::from_mins(5);
        let mut snapshot = app.counters.snapshot();
        assert_eq!(snapshot.inserts, 0, "workload must start paused");
        app.toggle_pause();
        while std::time::Instant::now() < deadline {
            snapshot = app.counters.snapshot();
            if snapshot.inserts > 0 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        app.shutdown();
        assert!(snapshot.inserts > 0, "workload must have inserted");
        assert!(snapshot.hits > 0, "workload must have hit");
    }

    /// Verifies that the burst insert reports its inserted count.
    #[test]
    fn burst_insert_counts_work() {
        let mut app = TuiApp::new(1 << 22, 0, true, 30, Config::default()).expect("app construction succeeds");
        let inserted = app.burst_insert(128);
        assert_eq!(inserted, 128);
        app.shutdown();
    }

    /// Verifies that launching every chaos scenario stays usable.
    #[test]
    fn chaos_scenarios_launch_and_report() {
        let mut app = TuiApp::new(1 << 22, 0, true, 30, Config::default()).expect("app construction succeeds");
        app.launch_storm();
        app.launch_pressure();
        let deadline = std::time::Instant::now() + std::time::Duration::from_mins(1);
        while std::time::Instant::now() < deadline {
            app.pump_chaos();
            if app.storm.is_some() && app.pressure.is_some() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(25));
        }
        assert!(app.storm.is_some(), "storm report must arrive");
        assert!(app.pressure.is_some(), "pressure report must arrive");
        app.shutdown();
    }
}
