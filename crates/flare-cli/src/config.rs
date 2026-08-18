//! Configuration for flare-cli: arena, workload, and TUI settings.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

/// Top-level configuration loaded from TOML file.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct Config {
    /// Arena configuration shared by all engines.
    pub arena: ArenaConfig,
    /// Background workload parameters.
    pub workload: WorkloadConfig,
    /// TUI appearance and behavior.
    pub tui: TuiConfig,
}

/// Arena capacity and slab settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ArenaConfig {
    /// Total byte capacity shared by all engines (suffix KB/MB/GB supported).
    pub capacity: String,
    /// Slab size in bytes (4KB default).
    pub slab_size: usize,
}

/// Workload tuning knobs.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct WorkloadConfig {
    /// Number of hot keys used by the synthetic workload.
    pub hot_keys: usize,
    /// Tree inserts per tick.
    pub tick_inserts: u64,
    /// Tree lookups per tick.
    pub tick_gets: u64,
    /// Vector inserts per tick.
    pub tick_vector_inserts: u64,
    /// KV inserts per tick.
    pub tick_kv_inserts: u64,
}

/// TUI appearance and behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TuiConfig {
    /// Default tab index on startup (0=Dashboard, 1=Memory, 2=Vector, 3=KV-Cache, 4=Chaos).
    pub default_tab: usize,
    /// Refresh interval in milliseconds.
    pub refresh_ms: u64,
    /// Theme name: dark, light, high-contrast, protanopia, deuteranopia, tritanopia.
    pub theme: String,
    /// Start with workload paused.
    pub start_paused: bool,
}

/// Persisted TUI runtime state (last tab, pause state, etc.).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct TuiState {
    /// Last active tab index.
    pub last_tab: usize,
    /// Whether the workload was paused on exit.
    pub was_paused: bool,
}

impl Default for ArenaConfig {
    fn default() -> Self {
        Self {
            capacity: "64MB".into(),
            slab_size: 4096,
        }
    }
}

impl Default for WorkloadConfig {
    fn default() -> Self {
        Self {
            hot_keys: 1 << 10,
            tick_inserts: 64,
            tick_gets: 128,
            tick_vector_inserts: 2,
            tick_kv_inserts: 4,
        }
    }
}

impl Default for TuiConfig {
    fn default() -> Self {
        Self {
            default_tab: 0,
            refresh_ms: 30,
            theme: "dark".into(),
            start_paused: true,
        }
    }
}

/// Load configuration from file, falling back to defaults.
pub fn load_config() -> Config {
    let paths = config_paths();
    for path in paths {
        if path.exists()
            && let Ok(content) = fs::read_to_string(&path)
            && let Ok(config) = toml::from_str(&content)
        {
            return config;
        }
    }
    Config::default()
}

/// Returns ordered list of config file paths to try (highest priority first).
fn config_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    // 1. Current working directory
    paths.push(PathBuf::from("flare.toml"));
    // 2. User config directory
    if let Some(config_dir) = dirs::config_dir() {
        paths.push(config_dir.join("flare").join("config.toml"));
    }
    // 3. Home directory fallback
    if let Some(home) = dirs::home_dir() {
        paths.push(home.join(".flare.toml"));
    }
    paths
}

/// Returns the path to the TUI state file.
fn tui_state_path() -> PathBuf {
    let mut path = dirs::data_local_dir().unwrap_or_else(std::env::temp_dir);
    path.push("flare");
    std::fs::create_dir_all(&path).ok();
    path.push("tui-state.toml");
    path
}

/// Loads the persisted TUI state.
pub fn load_tui_state() -> TuiState {
    let path = tui_state_path();
    if path.exists()
        && let Ok(content) = fs::read_to_string(&path)
        && let Ok(state) = toml::from_str(&content)
    {
        return state;
    }
    TuiState::default()
}

/// Saves the TUI state to disk.
pub fn save_tui_state(state: &TuiState) {
    let path = tui_state_path();
    if let Ok(content) = toml::to_string(state) {
        let _ = fs::write(&path, content);
    }
}

/// Parses a size string like `64MB` into bytes (re-export from main).
pub fn parse_arena(raw: &str) -> Result<usize, String> {
    let trimmed = raw.trim();
    let (number, factor) = [("KB", 10usize), ("MB", 20), ("GB", 30)]
        .into_iter()
        .find_map(|(suffix, shift)| {
            trimmed
                .strip_suffix(suffix)
                .map(|rest| (rest, 1usize << shift))
        })
        .unwrap_or((trimmed, 1));
    let value: usize = number
        .parse()
        .map_err(|_| format!("invalid arena size '{raw}'"))?;
    value
        .checked_mul(factor)
        .ok_or_else(|| format!("arena size '{raw}' overflows usize"))
}

/// Applies config to TUI app creation arguments.
pub fn apply_config_to_tui(config: &Config) -> TuiArgs {
    let capacity = parse_arena(&config.arena.capacity).unwrap_or(64 << 20);
    TuiArgs {
        arena_capacity: capacity,
        default_tab: config.tui.default_tab.min(4),
        refresh_ms: config.tui.refresh_ms,
        theme: config.tui.theme.clone(),
        start_paused: config.tui.start_paused,
    }
}

/// Arguments for TUI mode derived from config + CLI.
#[derive(Debug, Clone)]
pub struct TuiArgs {
    pub arena_capacity: usize,
    pub default_tab: usize,
    pub refresh_ms: u64,
    pub theme: String,
    pub start_paused: bool,
}

impl Default for TuiArgs {
    fn default() -> Self {
        Self {
            arena_capacity: 64 << 20,
            default_tab: 0,
            refresh_ms: 30,
            theme: "dark".into(),
            start_paused: true,
        }
    }
}