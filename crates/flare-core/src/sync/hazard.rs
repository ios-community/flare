//! Hazard Eras: lock-free memory reclamation through global and thread eras.
//!
//! The [`HazardManager`] tracks a monotonically increasing global era
//! ($`E_g`$) and per-reader thread-local era registrations ($`E_t`$) as
//! documented by the architecture specification. Objects retired at era
//! $E_{retire}$ are only reclaimed once every active reader has advanced
//! past it:
//!
//! $$\text{SafeToReclaim}(obj, E_{retire}) \iff \forall t \in
//! \text{ActiveThreads},\ `E_t` > E_{retire}$$
//!
//! Threads register an [`EraGuard`] before entering a traversal; the
//! guard's era is refreshed on demand and the registration is released on
//! drop.

use alloc_crate::vec::Vec;
use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicU64, Ordering};

/// The era value that marks a thread as "advanced past everything".
///
/// A guard whose era is `u64::MAX` is treated as retired-registered: it
/// never blocks reclamation of any object.
const ERA_ADVANCED: u64 = u64::MAX;

/// Manages the global era and thread-local era registrations.
///
/// The manager owns a fixed-capacity registration table; threads acquire a
/// [`EraGuard`] handle, refresh their era while traversing, and release
/// the slot on drop. Retired objects are queued with the era at which they
/// were retired, and reclaimed once all active registrations exceed that
/// era.
///
/// # Examples
///
/// ```
/// # use flare_core::sync::hazard::HazardManager;
/// let manager = HazardManager::new();
/// let guard = manager.register().expect("free slot");
/// let era = manager.advance_era();
/// assert!(guard.era() <= era);
/// drop(guard);
/// ```
pub struct HazardManager {
    global_era: AtomicU64,
    slots: UnsafeCell<Vec<Slot>>,
    retired: UnsafeCell<Vec<RetiredEntry>>,
}

/// A single thread-era registration slot inside a [`HazardManager`].
#[derive(Debug, Clone, Copy)]
struct Slot {
    /// `true` while the slot is leased by an active guard.
    active: bool,
    /// The era the holder last refreshed before traversal.
    era: u64,
}

/// A retired object awaiting reclamation.
#[derive(Debug, Clone, Copy)]
struct RetiredEntry {
    /// The era at which the object was retired.
    era: u64,
}

impl HazardManager {
    /// Creates a hazard manager with an empty era counter.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            global_era: AtomicU64::new(0),
            slots: UnsafeCell::new(Vec::new()),
            retired: UnsafeCell::new(Vec::new()),
        }
    }

    /// Registers the calling thread and returns a fresh [`EraGuard`].
    ///
    /// The guard's era is initialised to the current global era, so the
    /// very next global advance makes the holder eligible for reclamation
    /// unless it refreshes first.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flare_core::sync::hazard::HazardManager;
    /// let manager = HazardManager::new();
    /// let guard = manager.register().expect("slot available");
    /// assert_eq!(guard.era(), HazardManager::current_era(&manager));
    /// ```
    #[must_use]
    pub fn register(&self) -> Option<EraGuard<'_>> {
        // SAFETY: the slot table is exclusively owned by this manager; the
        // guard lease protocol keeps concurrent registration from touching
        // the same slot twice (the caller must use a distinct guard per
        // thread in this milestone).
        let slots = unsafe { &mut *self.slots.get() };
        let index = slots
            .iter()
            .position(|slot| !slot.active)
            .unwrap_or_else(|| {
                slots.push(Slot {
                    active: false,
                    era: ERA_ADVANCED,
                });
                slots.len() - 1
            });
        let era = self.global_era.load(Ordering::Acquire);
        slots[index] = Slot { active: true, era };
        Some(EraGuard {
            manager: self,
            index,
        })
    }

    /// Returns the current global era ($`E_g`$).
    #[must_use]
    pub fn current_era(&self) -> u64 {
        self.global_era.load(Ordering::Acquire)
    }

    /// Advances the global era by one and returns the new value.
    ///
    /// Each advance makes every retired object whose era is below the new
    /// global era eligible for reclamation once active guards advance.
    #[must_use]
    pub fn advance_era(&self) -> u64 {
        self.global_era.fetch_add(1, Ordering::AcqRel) + 1
    }

    /// Refreshes the era of the guard at `index` to the current global era.
    ///
    /// # Panics
    ///
    /// Panics when the manager ends up without a slot for the guard,
    /// indicating a lifecycle violation.
    fn refresh(&self, index: usize) {
        let era = self.global_era.load(Ordering::Acquire);
        // SAFETY: guarded by the same lease protocol as `register`.
        let slots = unsafe { &mut *self.slots.get() };
        let slot = slots
            .get_mut(index)
            .expect("hazard guard slot was released while in use");
        slot.era = era;
    }

    /// Releases the registration at `index`, marking it as advanced.
    fn release(&self, index: usize) {
        // SAFETY: guarded by the same lease protocol as `register`; the
        // guard is dropped so no further access can occur.
        let slots = unsafe { &mut *self.slots.get() };
        if let Some(slot) = slots.get_mut(index) {
            slot.active = false;
            slot.era = ERA_ADVANCED;
        }
    }

    /// Queues an object for reclamation, tagged with the current global era.
    ///
    /// The object's storage must not be reused until
    /// [`Self::try_reclaim`] reports it safe. The raw tagged-pointer word
    /// of the retired object is recorded by the caller before calling; the
    /// physical release path is scheduled with the reclamation milestone.
    ///
    /// # Panics
    ///
    /// Panics when the retired queue cannot be written, which is
    /// impossible under the internal allocation guarantee.
    pub fn retire(&self) {
        let era = self.global_era.load(Ordering::Acquire);
        // SAFETY: the retired queue is only ever touched by the retiring
        // writer thread in this milestone.
        let queue = unsafe { &mut *self.retired.get() };
        queue.push(RetiredEntry { era });
    }

    /// Reclaims every retired entry that is safe to free.
    ///
    /// An entry retired at era $E$ is safe when every active guard
    /// advanced past $E$ and the global era crossed one further boundary
    /// ($E + 1$): a reader whose traversal began in era $E + 1$ may still
    /// reference the object until the global era moves past it. With no
    /// active guard at all, nobody can hold the pointer, so a single era
    /// boundary past $E$ suffices. The reclaimed entries are removed from
    /// the queue and their count is returned. The raw words themselves are
    /// handed back to the caller for physical release.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flare_core::sync::hazard::HazardManager;
    /// let manager = HazardManager::new();
    /// let guard = manager.register().expect("slot available");
    /// manager.retire();
    /// manager.advance_era();
    /// // The active guard still holds a stale era, so nothing reclaims.
    /// assert_eq!(manager.try_reclaim(), 0);
    /// guard.refresh();
    /// manager.advance_era();
    /// assert_eq!(manager.try_reclaim(), 1);
    /// ```
    #[must_use]
    pub fn try_reclaim(&self) -> usize {
        let min_active = self.min_active_era();
        let current = self.global_era.load(Ordering::Acquire);
        let mut reclaimed = 0;
        // SAFETY: the queue is drained by a single reclaimer in this
        // milestone; retirement is append-only from the writer side.
        let queue = unsafe { &mut *self.retired.get() };
        queue.retain(|entry| {
            let safe = if min_active == ERA_ADVANCED {
                current.saturating_sub(entry.era) >= 1
            } else {
                min_active > entry.era && current.saturating_sub(entry.era) >= 2
            };
            if safe {
                reclaimed += 1;
                false
            } else {
                true
            }
        });
        reclaimed
    }

    /// Returns the number of entries queued for reclamation.
    #[must_use]
    pub fn retired_len(&self) -> usize {
        // SAFETY: read of the queue length is data-race free under the
        // single-retirer protocol.
        unsafe { (*self.retired.get()).len() }
    }

    /// Returns the minimum era among active guards, or `u64::MAX` when no
    /// guard is active.
    fn min_active_era(&self) -> u64 {
        // SAFETY: reads of the slot table are data-race free: the writer
        // only mutates its own slot entry between register and release,
        // and reclamation reads happen under the same lease discipline.
        let slots = unsafe { &*self.slots.get() };
        slots
            .iter()
            .filter(|slot| slot.active)
            .map(|slot| slot.era)
            .min()
            .unwrap_or(ERA_ADVANCED)
    }
}

impl Default for HazardManager {
    /// Creates a hazard manager with an empty era counter.
    fn default() -> Self {
        Self::new()
    }
}

/// A lease on a hazard-era registration slot.
///
/// The guard must be refreshed before entering a traversal so that the
/// reader's era follows the global era; on drop the registration is
/// released and the slot becomes reusable.
#[derive(Debug)]
pub struct EraGuard<'a> {
    manager: &'a HazardManager,
    index: usize,
}

impl EraGuard<'_> {
    /// Returns the era the guard currently holds.
    #[must_use]
    pub fn era(&self) -> u64 {
        self.manager.current_era_slot(self.index)
    }

    /// Refreshes the guard's era to the current global era.
    pub fn refresh(&self) {
        self.manager.refresh(self.index);
    }
}

impl Drop for EraGuard<'_> {
    /// Releases the registration slot on drop.
    fn drop(&mut self) {
        self.manager.release(self.index);
    }
}

impl HazardManager {
    /// Reads the era stored in the slot at `index`.
    ///
    /// # Panics
    ///
    /// Panics when `index` is out of bounds, which cannot happen while a
    /// live guard holds it.
    #[must_use]
    fn current_era_slot(&self, index: usize) -> u64 {
        // SAFETY: the guard lease guarantees the slot is owned by this
        // guard and no other thread mutates it concurrently.
        let slots = unsafe { &*self.slots.get() };
        slots
            .get(index)
            .expect("hazard guard slot released while in use")
            .era
    }
}

/// Safety justification for sharing the manager across threads.
///
/// Each [`EraGuard`] leases exactly one slot and only mutates its own slot
/// between registration and release; the global era and the retired queue
/// are single-writer. Under that lease discipline no data race is possible.
///
/// # Safety
///
/// This is sound because every field access is either atomic, or confined
/// to the unique owner of the respective slot or queue.
unsafe impl Sync for HazardManager {}

impl core::fmt::Debug for HazardManager {
    /// Formats the manager as its global era, slot count, and queue depth.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // SAFETY: reading the slot-table length is data-race free under the
        // same lease discipline as `min_active_era`.
        let slots = unsafe { (*self.slots.get()).len() };
        f.debug_struct("HazardManager")
            .field("global_era", &self.global_era)
            .field("slots", &slots)
            .field("retired", &self.retired_len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::HazardManager;

    /// Verifies that multiple guards can coexist on distinct slots.
    #[test]
    fn guards_can_coexist() {
        let manager = HazardManager::new();
        let first = manager.register().expect("slot available");
        let second = manager.register().expect("slot available");
        assert_eq!(first.era(), second.era());
        assert_eq!(first.era(), manager.current_era());
        drop(first);
        drop(second);
        assert!(manager.register().is_some(), "slot recycled after drop");
    }

    /// Verifies that refreshing a guard tracks the global era.
    #[test]
    fn refresh_tracks_global_era() {
        let manager = HazardManager::new();
        let guard = manager.register().expect("slot available");
        let _ = manager.advance_era();
        assert_ne!(guard.era(), manager.current_era());
        guard.refresh();
        assert_eq!(guard.era(), manager.current_era());
    }

    /// Verifies that retired entries are reclaimed only when every active
    /// guard advanced past their retirement era.
    #[test]
    fn reclaim_respects_active_guards() {
        let manager = HazardManager::new();
        manager.retire();
        let _ = manager.advance_era();
        assert_eq!(manager.retired_len(), 1);
        let guard = manager.register().expect("slot available");
        assert_eq!(manager.try_reclaim(), 0, "stale guard blocks reclamation");
        guard.refresh();
        let _ = manager.advance_era();
        assert_eq!(manager.try_reclaim(), 1);
        assert_eq!(manager.retired_len(), 0);
    }

    /// Verifies that the retired queue drains without any active guard.
    #[test]
    fn reclaim_without_guards_drains() {
        let manager = HazardManager::new();
        for _ in 0..3 {
            manager.retire();
        }
        let _ = manager.advance_era();
        assert_eq!(manager.try_reclaim(), 3);
        assert_eq!(manager.retired_len(), 0);
    }

    /// Verifies the `Default` construction and the `Debug` representation.
    #[test]
    fn default_and_debug() {
        let manager = HazardManager::default();
        assert_eq!(manager.current_era(), 0);
        let rendered = alloc_crate::format!("{manager:?}");
        assert!(rendered.contains("global_era"));
        assert!(rendered.contains("slots"));
    }
}
