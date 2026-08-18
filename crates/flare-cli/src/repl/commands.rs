//! Command surface of the REPL shell.
//!
//! The engine owns one instance of each FLARE engine (radix tree, IVF-PQ
//! index, radix KV-cache) plus the WAL sink, and executes line-oriented
//! commands into a text buffer. Keeping the engine separate from the
//! terminal makes every command unit-testable.

use core::fmt::Write as _;
use flare_core::alloc::arena::FlatArena;
use flare_core::error::FlareError;
use flare_core::sync::gpu::CpuFallbackDriver;
use flare_core::sync::hazard::HazardManager;
use flare_core::tree::FlareArtTree;
use flare_core::wal::{MemoryWalSink, WalFrame, WalTransaction};
use flare_kv::RadixAttentionEngine;
use flare_vector::IvfPqIndex;
use flare_vector::rng::Xorshift64Star;
use std::sync::Arc;

/// Every command recognised by the shell (also powers completion/help).
pub const COMMANDS: &[&str] = &[
    "kv-insert",
    "kv-get",
    "kv-delete",
    "inspect-pointer",
    "inspect-arena",
    "vec-train",
    "vec-insert",
    "vec-search",
    "kvcache-insert",
    "kvcache-match",
    "wal-flush",
    "status",
    "help",
    "exit",
];

/// Vector index dimension fixed by the shell.
const VECTOR_DIM: usize = 64;
/// Sub-vector count of the shell's IVF-PQ index.
const VECTOR_SUBVECTORS: usize = 4;
/// Centroid count of the shell's IVF-PQ index.
const VECTOR_CENTROIDS: usize = 16;
/// Seed of the deterministic vector generator.
const RNG_SEED: u64 = 0x1A7E_F1A7;

/// What the shell should do after executing a command line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplAction {
    /// Keep the shell loop running.
    Continue,
    /// Terminate the shell.
    Exit,
}

/// Stateful command engine: three FLARE engines plus a WAL sink.
pub struct ReplEngine {
    tree: FlareArtTree<CpuFallbackDriver>,
    vector: IvfPqIndex<CpuFallbackDriver>,
    kv: RadixAttentionEngine<CpuFallbackDriver>,
    sink: MemoryWalSink,
    rng: Xorshift64Star,
    last_kv_tokens: Vec<u32>,
    output: String,
}

impl ReplEngine {
    /// Constructs the engines with the given arena byte capacity.
    ///
    /// # Errors
    ///
    /// Returns the underlying [`FlareError`] when any arena allocation
    /// fails.
    pub fn new(arena_capacity: usize) -> Result<Self, FlareError> {
        let hazard = Arc::new(HazardManager::new());
        let tree_arena = Arc::new(FlatArena::new(arena_capacity)?);
        let tree = FlareArtTree::new(
            tree_arena,
            Arc::clone(&hazard),
            CpuFallbackDriver::default(),
        );
        let vector = IvfPqIndex::new(
            VECTOR_DIM,
            VECTOR_CENTROIDS,
            VECTOR_SUBVECTORS,
            RNG_SEED,
            arena_capacity,
            Arc::clone(&hazard),
            CpuFallbackDriver::default(),
        )?;
        let kv = RadixAttentionEngine::new(
            arena_capacity,
            arena_capacity,
            hazard,
            CpuFallbackDriver::default(),
        )?;
        Ok(Self {
            tree,
            vector,
            kv,
            sink: MemoryWalSink::new(),
            rng: Xorshift64Star::new(RNG_SEED),
            last_kv_tokens: Vec::new(),
            output: String::new(),
        })
    }

    /// Executes one line of shell input, appending text to the buffer.
    ///
    /// # Errors
    ///
    /// Returns the rendered error message when the line is malformed or an
    /// engine call fails; the buffer is left untouched in that case.
    pub fn execute(&mut self, line: &str) -> Result<ReplAction, String> {
        let line = line.trim();
        if line.is_empty() {
            return Ok(ReplAction::Continue);
        }
        let mut parts = line.split_whitespace();
        let command = parts.next().expect("line is non-empty");
        let args: Vec<&str> = parts.collect();
        match command {
            "kv-insert" => self.cmd_kv_insert(&args),
            "kv-get" => self.cmd_kv_get(&args),
            "kv-delete" => self.cmd_kv_delete(&args),
            "inspect-pointer" => self.cmd_inspect_pointer(&args),
            "inspect-arena" => self.cmd_inspect_arena(&args),
            "vec-train" => self.cmd_vec_train(&args),
            "vec-insert" => self.cmd_vec_insert(&args),
            "vec-search" => self.cmd_vec_search(&args),
            "kvcache-insert" => self.cmd_kvcache_insert(&args),
            "kvcache-match" => self.cmd_kvcache_match(&args),
            "wal-flush" => self.cmd_wal_flush(&args),
            "status" => self.cmd_status(&args),
            "help" => self.cmd_help(&args),
            "exit" | "quit" => {
                Self::require_no_args(&args)?;
                Ok(ReplAction::Exit)
            }
            other => Err(format!("unknown command '{other}' — type `help`")),
        }
    }

    /// Takes the accumulated output buffer, leaving it empty.
    #[must_use]
    pub fn take_output(&mut self) -> String {
        std::mem::take(&mut self.output)
    }

    /// Renders a `Command: args` header into the buffer.
    fn header(&mut self, command: &str, args: &[&str]) {
        self.output
            .push_str(format!("[CMD] {} {}", command, args.join(" ")).trim_end());
        self.output.push('\n');
    }

    /// Requires exactly one argument and returns it.
    fn one_arg<'a>(args: &[&'a str], command: &str) -> Result<&'a str, String> {
        match args {
            [arg] => Ok(*arg),
            _ => Err(format!("usage: {command} <arg>")),
        }
    }

    /// Rejects any trailing arguments.
    fn require_no_args(args: &[&str]) -> Result<(), String> {
        if args.is_empty() {
            Ok(())
        } else {
            Err(format!("unexpected argument(s): {}", args.join(" ")))
        }
    }

    fn cmd_kv_insert(&mut self, args: &[&str]) -> Result<ReplAction, String> {
        let (key, value) = match args {
            [key, value] => (
                *key,
                value.parse::<u64>().map_err(|_| "value must be a u64")?,
            ),
            _ => return Err("usage: kv-insert <key> <value>".into()),
        };
        self.header("kv-insert", args);
        let previous = self
            .tree
            .insert(key.as_bytes(), value)
            .map_err(|e| e.to_string())?;
        let _ = writeln!(
            self.output,
            "inserted '{key}' = {value} (previous: {previous:?})"
        );
        Ok(ReplAction::Continue)
    }

    fn cmd_kv_get(&mut self, args: &[&str]) -> Result<ReplAction, String> {
        let key = Self::one_arg(args, "kv-get")?;
        self.header("kv-get", args);
        let found = self.tree.get(key.as_bytes()).map_err(|e| e.to_string())?;
        match found {
            Some(value) => {
                let _ = writeln!(self.output, "'{key}' = {value}");
            }
            None => {
                let _ = writeln!(self.output, "'{key}' not found");
            }
        }
        Ok(ReplAction::Continue)
    }

    fn cmd_kv_delete(&mut self, args: &[&str]) -> Result<ReplAction, String> {
        let key = Self::one_arg(args, "kv-delete")?;
        self.header("kv-delete", args);
        let removed = self
            .tree
            .delete(key.as_bytes())
            .map_err(|e| e.to_string())?;
        let _ = writeln!(self.output, "deleted '{key}': {removed:?}");
        Ok(ReplAction::Continue)
    }

    fn cmd_inspect_pointer(&mut self, args: &[&str]) -> Result<ReplAction, String> {
        Self::require_no_args(args)?;
        self.header("inspect-pointer", args);
        let root = self.tree.root_tag();
        let _ = writeln!(
            self.output,
            "root tagged pointer: 0x{:016x}",
            root.to_bits()
        );
        Ok(ReplAction::Continue)
    }

    fn cmd_inspect_arena(&mut self, args: &[&str]) -> Result<ReplAction, String> {
        Self::require_no_args(args)?;
        self.header("inspect-arena", args);
        let arena = self.tree.arena();
        let _ = writeln!(
            self.output,
            "arena frontier: {} / {} bytes ({} free)",
            arena.frontier(),
            arena.capacity(),
            arena.remaining()
        );
        Ok(ReplAction::Continue)
    }

    fn cmd_vec_train(&mut self, args: &[&str]) -> Result<ReplAction, String> {
        let count = match args {
            [count] => count
                .parse::<usize>()
                .map_err(|_| "sample count must be a usize".to_string())?,
            _ => return Err("usage: vec-train <samples>".into()),
        };
        if count < 256 {
            return Err("training needs at least 256 samples (PQ codebook size)".into());
        }
        self.header("vec-train", args);
        let mut samples = Vec::with_capacity(count * VECTOR_DIM);
        for _ in 0..count {
            for _ in 0..VECTOR_DIM {
                samples.push(self.rng.next_f32());
            }
        }
        self.vector.train(&samples).map_err(|e| e.to_string())?;
        let _ = writeln!(self.output, "trained on {count} x {VECTOR_DIM}-dim samples");
        Ok(ReplAction::Continue)
    }

    fn cmd_vec_insert(&mut self, args: &[&str]) -> Result<ReplAction, String> {
        Self::require_no_args(args)?;
        self.header("vec-insert", args);
        let mut vector = Vec::with_capacity(VECTOR_DIM);
        for _ in 0..VECTOR_DIM {
            vector.push(self.rng.next_f32());
        }
        self.vector.insert(&vector).map_err(|e| e.to_string())?;
        let count = self.vector.vector_count().map_err(|e| e.to_string())?;
        let _ = writeln!(self.output, "inserted vector #{count} ({VECTOR_DIM} dims)");
        Ok(ReplAction::Continue)
    }

    fn cmd_vec_search(&mut self, args: &[&str]) -> Result<ReplAction, String> {
        let top_k = match args {
            [top_k] => top_k
                .parse::<usize>()
                .map_err(|_| "top-k must be a usize")?,
            _ => return Err("usage: vec-search <top-k>".into()),
        };
        self.header("vec-search", args);
        let mut query = Vec::with_capacity(VECTOR_DIM);
        for _ in 0..VECTOR_DIM {
            query.push(self.rng.next_f32());
        }
        let hits = self
            .vector
            .search(&query, top_k)
            .map_err(|e| e.to_string())?;
        let _ = writeln!(self.output, "top-{top_k} results: {hits:?}");
        Ok(ReplAction::Continue)
    }

    fn cmd_kvcache_insert(&mut self, args: &[&str]) -> Result<ReplAction, String> {
        let (count, offset) = match args {
            [count, offset] => (
                count
                    .parse::<usize>()
                    .map_err(|_| "token count must be a usize")?,
                offset.parse::<u64>().map_err(|_| "offset must be a u64")?,
            ),
            _ => return Err("usage: kvcache-insert <tokens> <kv-offset>".into()),
        };
        self.header("kvcache-insert", args);
        let mut tokens = Vec::with_capacity(count);
        for _ in 0..count {
            tokens.push((self.rng.next_u64() & 0xFFFF) as u32);
        }
        self.kv.insert(&tokens, offset).map_err(|e| e.to_string())?;
        self.last_kv_tokens = tokens;
        let _ = writeln!(self.output, "cached {count} tokens -> offset {offset}");
        Ok(ReplAction::Continue)
    }

    fn cmd_kvcache_match(&mut self, args: &[&str]) -> Result<ReplAction, String> {
        Self::require_no_args(args)?;
        self.header("kvcache-match", args);
        if self.last_kv_tokens.is_empty() {
            self.output
                .push_str("no prefix cached yet — use kvcache-insert first");
            return Ok(ReplAction::Continue);
        }
        let matched = self
            .kv
            .match_common_prefix(&self.last_kv_tokens)
            .map_err(|e| e.to_string())?;
        let _ = writeln!(self.output, "prefix match: {matched:?}");
        Ok(ReplAction::Continue)
    }

    fn cmd_wal_flush(&mut self, args: &[&str]) -> Result<ReplAction, String> {
        Self::require_no_args(args)?;
        self.header("wal-flush", args);
        let arena = self.tree.arena();
        let offset = arena.alloc(8, 8).map_err(|e| e.to_string())?;
        let value = self.rng.next_u64();
        arena
            .write_node(offset, &value)
            .map_err(|e| e.to_string())?;
        let tx = WalTransaction::new(
            vec![WalFrame::alloc(offset, 8)],
            WalFrame::update(offset, value.to_le_bytes().to_vec()),
        );
        tx.commit(&self.sink).map_err(|e| e.to_string())?;
        let _ = writeln!(
            self.output,
            "flushed tx @ offset {offset} ({} frames buffered)",
            self.sink.frame_count()
        );
        Ok(ReplAction::Continue)
    }

    fn cmd_status(&mut self, args: &[&str]) -> Result<ReplAction, String> {
        Self::require_no_args(args)?;
        self.header("status", args);
        let arena = self.tree.arena();
        let vector_count = self.vector.vector_count().unwrap_or(0);
        let _ = writeln!(
            self.output,
            "radix tree   : frontier {} / {} bytes",
            arena.frontier(),
            arena.capacity()
        );
        let _ = writeln!(
            self.output,
            "ivf-pq index : trained={} vectors={vector_count} (dim {VECTOR_DIM})",
            self.vector.is_trained()
        );
        let _ = writeln!(
            self.output,
            "kv-cache     : {} bytes, {} slots",
            self.kv.capacity_bytes(),
            self.kv.slot_count()
        );
        let _ = writeln!(
            self.output,
            "wal sink     : {} frames, {} bytes",
            self.sink.frame_count(),
            self.sink.len()
        );
        Ok(ReplAction::Continue)
    }

    fn cmd_help(&mut self, args: &[&str]) -> Result<ReplAction, String> {
        Self::require_no_args(args)?;
        self.header("help", args);
        for command in COMMANDS {
            let _ = writeln!(self.output, "  {command}");
        }
        Ok(ReplAction::Continue)
    }
}

#[cfg(test)]
mod tests {
    use super::{COMMANDS, ReplAction, ReplEngine};

    /// Creates a small engine for tests.
    fn engine() -> ReplEngine {
        ReplEngine::new(1 << 20).expect("engine construction succeeds")
    }

    /// Verifies the full KV round-trip through the command surface.
    #[test]
    fn kv_insert_get_delete_roundtrip() {
        let mut engine = engine();
        engine.execute("kv-insert hello 42").expect("insert parses");
        let output = engine.take_output();
        assert!(output.contains("inserted 'hello' = 42"));
        engine.execute("kv-get hello").expect("get parses");
        assert!(engine.take_output().contains("'hello' = 42"));
        engine.execute("kv-delete hello").expect("delete parses");
        assert!(engine.take_output().contains("deleted 'hello'"));
        engine.execute("kv-get hello").expect("get parses");
        assert!(engine.take_output().contains("not found"));
    }

    /// Verifies that malformed input yields usage errors, not panics.
    #[test]
    fn malformed_commands_report_usage() {
        let mut engine = engine();
        assert!(engine.execute("kv-insert").unwrap_err().contains("usage"));
        assert!(engine.execute("kv-insert a not-a-number").is_err());
        assert!(engine.execute("kv-get").unwrap_err().contains("usage"));
        assert!(
            engine
                .execute("bogus-command")
                .unwrap_err()
                .contains("unknown command")
        );
        assert!(
            engine
                .execute("exit extra")
                .unwrap_err()
                .contains("unexpected")
        );
    }

    /// Verifies that `exit` terminates the shell loop.
    #[test]
    fn exit_returns_exit_action() {
        let mut engine = engine();
        assert_eq!(engine.execute("exit"), Ok(ReplAction::Exit));
        assert_eq!(engine.execute("quit"), Ok(ReplAction::Exit));
    }

    /// Verifies arena inspection reports consistent numbers.
    #[test]
    fn inspect_arena_reports_numbers() {
        let mut engine = engine();
        engine.execute("inspect-arena").expect("inspect parses");
        let output = engine.take_output();
        assert!(output.contains("frontier:"));
        assert!(output.contains("bytes"));
    }

    /// Verifies the vector train/insert/search pipeline end to end.
    #[test]
    fn vector_pipeline_end_to_end() {
        let mut engine = engine();
        engine.execute("vec-train 512").expect("train parses");
        assert!(engine.take_output().contains("trained on 512"));
        engine.execute("vec-insert").expect("insert parses");
        assert!(engine.take_output().contains("inserted vector #"));
        engine.execute("vec-search 5").expect("search parses");
        let output = engine.take_output();
        assert!(output.contains("top-5 results: ["), "output was: {output}");
        assert!(output.contains(']'), "output was: {output}");
    }

    /// Verifies the KV-cache prefix pipeline.
    #[test]
    fn kvcache_insert_then_match() {
        let mut engine = engine();
        engine
            .execute("kvcache-insert 8 1024")
            .expect("insert parses");
        assert!(engine.take_output().contains("cached 8 tokens"));
        engine.execute("kvcache-match").expect("match parses");
        assert!(engine.take_output().contains("prefix match:"));
    }

    /// Verifies the WAL flush pipeline.
    #[test]
    fn wal_flush_reports_frame_count() {
        let mut engine = engine();
        engine.execute("wal-flush").expect("flush parses");
        assert!(engine.take_output().contains("flushed tx @ offset"));
        engine.execute("status").expect("status parses");
        let output = engine.take_output();
        assert!(
            output.contains("wal sink     : 1 frames, 40 bytes"),
            "output was: {output}"
        );
    }

    /// Verifies that the help listing covers every command.
    #[test]
    fn help_lists_every_command() {
        let mut engine = engine();
        engine.execute("help").expect("help parses");
        let output = engine.take_output();
        for command in COMMANDS {
            assert!(output.contains(command), "help must list '{command}'");
        }
    }

    /// Verifies pointer inspection renders a hexadecimal root.
    #[test]
    fn inspect_pointer_renders_hex() {
        let mut engine = engine();
        engine.execute("inspect-pointer").expect("inspect parses");
        assert!(engine.take_output().contains("0x"));
    }

    /// Verifies that empty lines are silently ignored.
    #[test]
    fn blank_lines_are_ignored() {
        let mut engine = engine();
        assert_eq!(engine.execute("   "), Ok(ReplAction::Continue));
        assert!(engine.take_output().is_empty());
    }
}
