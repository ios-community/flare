//! Non-blocking telemetry: engine event definitions and the lock-free
//! ring-buffer collector that decouples engine threads from the UI thread.

pub mod collector;
pub mod events;

pub use collector::Collector;
pub use events::{EventKind, EventWord};
