//! Single-producer single-consumer lock-free ring buffer.
//!
//! The collector decouples engine workload threads from the UI thread:
//! producers publish [`EventWord`]s with release-store + release fetch-add
//! on the head, and the single consumer drains them with acquire loads on
//! the tail. A full buffer drops the incoming event (never blocks), so
//! telemetry can never slow down an engine path.

use super::events::EventWord;
use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Default ring capacity in event words (power of two).
const DEFAULT_CAPACITY: usize = 4096;

/// A lock-free SPSC ring buffer of encoded telemetry events.
///
/// # Threading contract
///
/// Exactly one thread may call the push side (`try_push`) and exactly one
/// thread may call the pop side (`try_pop` / [`Self::drain`]); both sides
/// may run concurrently. Any other use pattern is a data race.
pub struct Collector {
    buffer: Box<[AtomicU64]>,
    head: AtomicUsize,
    tail: AtomicUsize,
}

impl Collector {
    /// Creates a ring buffer holding `capacity` event words.
    ///
    /// # Panics
    ///
    /// Panics when `capacity` is smaller than 2, because a ring with fewer
    /// than two slots cannot distinguish full from empty.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        assert!(capacity >= 2, "ring capacity must be at least 2");
        let buffer = (0..capacity)
            .map(|_| AtomicU64::new(0))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self {
            buffer,
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
        }
    }

    /// Returns the number of event words the ring can hold.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.buffer.len()
    }

    /// Returns the number of unread events currently buffered.
    #[must_use]
    pub fn len(&self) -> usize {
        self.head
            .load(Ordering::Relaxed)
            .wrapping_sub(self.tail.load(Ordering::Relaxed))
    }

    /// Returns whether the ring holds no unread events.
    #[allow(dead_code)]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Publishes one event, dropping it when the ring is full.
    ///
    /// Returns `false` when the event was dropped.
    ///
    /// # Panics
    ///
    /// Panics when invoked from more than one thread (SPSC contract).
    pub fn try_push(&self, event: EventWord) -> bool {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Relaxed);
        if head.wrapping_sub(tail) == self.capacity() {
            return false;
        }
        self.buffer[head % self.capacity()].store(event.encode(), Ordering::Release);
        self.head.store(head.wrapping_add(1), Ordering::Release);
        true
    }

    /// Removes and returns the oldest event, or `None` when empty.
    ///
    /// # Panics
    ///
    /// Panics when invoked from more than one thread (SPSC contract).
    pub fn try_pop(&self) -> Option<EventWord> {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Acquire);
        if tail == head {
            return None;
        }
        let word = self.buffer[tail % self.capacity()].load(Ordering::Acquire);
        self.tail.store(tail.wrapping_add(1), Ordering::Relaxed);
        Some(EventWord::decode(word))
    }

    /// Drains up to `max` events into `out`, returning the drained count.
    pub fn drain(&self, out: &mut Vec<EventWord>, max: usize) -> usize {
        let mut drained = 0;
        while drained < max {
            let Some(event) = self.try_pop() else {
                break;
            };
            out.push(event);
            drained += 1;
        }
        drained
    }
}

impl Default for Collector {
    /// Creates a collector with [`DEFAULT_CAPACITY`] slots.
    fn default() -> Self {
        Self::new(DEFAULT_CAPACITY)
    }
}

#[cfg(test)]
mod tests {
    use super::{Collector, DEFAULT_CAPACITY};
    use crate::telemetry::events::EventWord;

    /// Verifies that events round-trip through the ring in FIFO order.
    #[test]
    fn push_pop_fifo_order() {
        let ring = Collector::new(16);
        for i in 0..8u64 {
            assert!(ring.try_push(EventWord::tree_insert(4, i)));
        }
        assert_eq!(ring.len(), 8);
        for i in 0..8u64 {
            let event = ring.try_pop().expect("event buffered");
            assert_eq!(event.kind, crate::telemetry::events::EventKind::TreeInsert);
            assert_eq!(event.b, u32::try_from(i).expect("fits in u32"));
        }
        assert!(ring.try_pop().is_none());
        assert!(ring.is_empty());
    }

    /// Verifies that a full ring drops incoming events instead of blocking.
    #[test]
    fn full_ring_drops_events() {
        let ring = Collector::new(4);
        for _ in 0..4 {
            assert!(ring.try_push(EventWord::tree_hit(1)));
        }
        assert!(!ring.try_push(EventWord::tree_hit(2)), "ring must be full");
        assert_eq!(ring.try_pop().expect("event buffered").a, 1);
        assert!(ring.try_push(EventWord::tree_hit(3)), "slot freed by pop");
    }

    /// Verifies that wrapping indices do not confuse the ring.
    #[test]
    fn indices_wrap_cleanly() {
        let ring = Collector::new(8);
        for round in 0..5usize {
            for i in 0..6usize {
                assert!(ring.try_push(EventWord::kv_match(round + i)));
            }
            for _ in 0..6 {
                assert!(ring.try_pop().is_some());
            }
        }
        assert!(ring.is_empty());
    }

    /// Verifies that `drain` respects its maximum and the default capacity.
    #[test]
    fn drain_respects_max() {
        let ring = Collector::default();
        assert_eq!(ring.capacity(), DEFAULT_CAPACITY);
        for i in 0..10u64 {
            assert!(ring.try_push(EventWord::wal_flush(i)));
        }
        let mut out = Vec::new();
        assert_eq!(ring.drain(&mut out, 4), 4);
        assert_eq!(ring.drain(&mut out, 64), 6);
        assert!(ring.is_empty());
    }
}
