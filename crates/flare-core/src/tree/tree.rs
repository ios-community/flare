//! Adaptive radix node structures and the `FlareArtTree` index.
//!
//! Four adaptive node families are defined, sized to fit the 4 KB slab
//! slot classes of [`SlabPool`](crate::alloc::slab::SlabPool):
//!
//! | Node | Layout | Slot class |
//! | --- | --- | --- |
//! | [`Node4`] | 4 nibble keys + 4 child words + leaf word (48 B) | 0 |
//! | [`Node16`] | 16 nibble keys + 16 child words + leaf word (152 B) | 1 |
//! | [`Node64`] | 64-bit bitmap + 64 child words + leaf word (528 B) | 2 |
//! | [`Node256`] | 256 child words + leaf word (2056 B) | 3 |
//!
//! All child words are `AtomicU64` so the read path never takes a lock and
//! writer mutations are publishable with a single release-store of the
//! parent's tagged pointer.

use crate::alloc::arena::FlatArena;
use crate::error::FlareError;
use crate::ptr::{NodeType, TaggedPointer, resolve_child_index};
use crate::sync::gpu::GpuSyncDriver;
use crate::sync::hazard::HazardManager;
use crate::tree::{EMPTY_CHILD, key_nibbles};
use alloc_crate::sync::Arc;
use core::sync::atomic::{AtomicU64, Ordering};

/// The byte marker for an unused key slot inside `Node4`/`Node16`.
const EMPTY_KEY: u8 = 0xFF;

/// Returns `true` when the node type is one of the two leaf encodings.
const fn is_leaf_type(kind: NodeType) -> bool {
    matches!(kind, NodeType::LeafInlined | NodeType::LeafOffset)
}

/// An adaptive radix node holding up to 4 nibble-keyed children.
///
/// The 16-bit child presence bitmap lives in the parent's polymorphic
/// field; the local `keys` array stores nibble keys in dense order and
/// child words are packed behind them. The trailing `leaf` word holds the
/// terminal value of a key ending exactly at this node (absent =
/// [`EMPTY_CHILD`]).
///
/// The struct layout is 48 bytes, matching [`SlabClass::Class0`](crate::alloc::slab::SlabClass::Class0).
#[derive(Debug)]
pub struct Node4 {
    /// Dense nibble keys (unused slots carry [`EMPTY_KEY`]).
    keys: [u8; 4],
    /// Dense child words (absent = [`EMPTY_CHILD`]).
    children: [AtomicU64; 4],
    /// Terminal leaf word (absent = [`EMPTY_CHILD`]).
    leaf: AtomicU64,
}

impl Node4 {
    /// Creates a node from dense key/child pairs and a leaf word.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flare_core::tree::{Node4, EMPTY_CHILD};
    /// let node = Node4::new([1, 2, 3, 4], [9, 8, 7, 6], EMPTY_CHILD);
    /// assert_eq!(node.keys(), &[1, 2, 3, 4]);
    /// ```
    #[must_use]
    pub fn new(keys: [u8; 4], children: [u64; 4], leaf: u64) -> Self {
        Self {
            keys,
            children: core::array::from_fn(|i| AtomicU64::new(children[i])),
            leaf: AtomicU64::new(leaf),
        }
    }

    /// Returns the dense key array of this node.
    #[must_use]
    pub const fn keys(&self) -> &[u8; 4] {
        &self.keys
    }

    /// Returns the child words of this node.
    #[must_use]
    pub const fn children(&self) -> &[AtomicU64; 4] {
        &self.children
    }

    /// Returns the terminal leaf word of this node.
    #[must_use]
    pub const fn leaf(&self) -> &AtomicU64 {
        &self.leaf
    }
}

/// An adaptive radix node holding up to 16 nibble-keyed children.
///
/// Mirrors [`Node4`] with a 16-entry dense array and a 16-bit presence
/// bitmap in the parent's polymorphic field. The struct layout is 152
/// bytes, matching [`SlabClass::Class1`](crate::alloc::slab::SlabClass::Class1).
#[derive(Debug)]
pub struct Node16 {
    /// Dense nibble keys (unused slots carry [`EMPTY_KEY`]).
    keys: [u8; 16],
    /// Dense child words (absent = [`EMPTY_CHILD`]).
    children: [AtomicU64; 16],
    /// Terminal leaf word (absent = [`EMPTY_CHILD`]).
    leaf: AtomicU64,
}

impl Node16 {
    /// Creates a node from dense key/child pairs and a leaf word.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flare_core::tree::{Node16, EMPTY_CHILD};
    /// let node = Node16::new([0; 16], [EMPTY_CHILD; 16], 7);
    /// assert_eq!(node.keys(), &[0; 16]);
    /// ```
    #[must_use]
    pub fn new(keys: [u8; 16], children: [u64; 16], leaf: u64) -> Self {
        Self {
            keys,
            children: core::array::from_fn(|i| AtomicU64::new(children[i])),
            leaf: AtomicU64::new(leaf),
        }
    }

    /// Returns the dense key array of this node.
    #[must_use]
    pub const fn keys(&self) -> &[u8; 16] {
        &self.keys
    }

    /// Returns the child words of this node.
    #[must_use]
    pub const fn children(&self) -> &[AtomicU64; 16] {
        &self.children
    }

    /// Returns the terminal leaf word of this node.
    #[must_use]
    pub const fn leaf(&self) -> &AtomicU64 {
        &self.leaf
    }
}

/// An adaptive radix node holding up to 16 nibble-keyed children with a
/// 64-bit internal presence bitmap.
///
/// The bitmap (16 bits used by the nibble domain) doubles as the popcount
/// index: the dense child index of key `k` is `popcount(bitmap & ((1 << k) - 1))`.
/// The parent's polymorphic field carries an 8-bit reference count
/// and an 8-bit generation identifier for ABA protection. The struct
/// layout is 528 bytes, matching [`SlabClass::Class2`](crate::alloc::slab::SlabClass::Class2).
#[derive(Debug)]
pub struct Node64 {
    /// Internal 64-bit presence bitmap (nibble domain uses 16 bits).
    bitmap: AtomicU64,
    /// Dense child words (absent = [`EMPTY_CHILD`]).
    children: [AtomicU64; 64],
    /// Terminal leaf word (absent = [`EMPTY_CHILD`]).
    leaf: AtomicU64,
}

impl Node64 {
    /// Creates a node from a bitmap, its dense children, and a leaf word.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flare_core::tree::{Node64, EMPTY_CHILD};
    /// let mut children = [EMPTY_CHILD; 64];
    /// children[0] = 1;
    /// children[1] = 3;
    /// let node = Node64::new(0b1010, children, EMPTY_CHILD);
    /// assert_eq!(node.bitmap().load(core::sync::atomic::Ordering::Relaxed), 0b1010);
    /// assert_eq!(node.children()[1].load(core::sync::atomic::Ordering::Relaxed), 3);
    /// ```
    #[must_use]
    pub fn new(bitmap: u64, children: [u64; 64], leaf: u64) -> Self {
        Self {
            bitmap: AtomicU64::new(bitmap),
            children: core::array::from_fn(|i| AtomicU64::new(children[i])),
            leaf: AtomicU64::new(leaf),
        }
    }

    /// Returns the internal presence bitmap of this node.
    #[must_use]
    pub const fn bitmap(&self) -> &AtomicU64 {
        &self.bitmap
    }

    /// Returns the dense child words of this node.
    #[must_use]
    pub const fn children(&self) -> &[AtomicU64; 64] {
        &self.children
    }

    /// Returns the terminal leaf word of this node.
    #[must_use]
    pub const fn leaf(&self) -> &AtomicU64 {
        &self.leaf
    }
}

/// An adaptive radix node holding up to 16 nibble-keyed children indexed
/// directly by nibble position.
///
/// Lookup is `O(1)`: the child word of key `k` lives at `children[k]`.
/// The parent's polymorphic field carries an 8-bit reference count and an
/// 8-bit generation identifier. The struct layout is 2056 bytes, matching
/// [`SlabClass::Class3`](crate::alloc::slab::SlabClass::Class3).
#[derive(Debug)]
pub struct Node256 {
    /// Child words addressed directly by nibble (absent = [`EMPTY_CHILD`]).
    children: [AtomicU64; 256],
    /// Terminal leaf word (absent = [`EMPTY_CHILD`]).
    leaf: AtomicU64,
}

impl Node256 {
    /// Creates a node from direct-indexed children and a leaf word.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flare_core::tree::{Node256, EMPTY_CHILD};
    /// let mut children = [EMPTY_CHILD; 256];
    /// children[2] = 42;
    /// let node = Node256::new(children, EMPTY_CHILD);
    /// assert_eq!(node.children()[2].load(core::sync::atomic::Ordering::Relaxed), 42);
    /// ```
    #[must_use]
    pub fn new(children: [u64; 256], leaf: u64) -> Self {
        Self {
            children: core::array::from_fn(|i| AtomicU64::new(children[i])),
            leaf: AtomicU64::new(leaf),
        }
    }

    /// Returns the direct-indexed child words of this node.
    #[must_use]
    pub const fn children(&self) -> &[AtomicU64; 256] {
        &self.children
    }

    /// Returns the terminal leaf word of this node.
    #[must_use]
    pub const fn leaf(&self) -> &AtomicU64 {
        &self.leaf
    }
}

// ---------------------------------------------------------------------------
// FlareArtTree
// ---------------------------------------------------------------------------

/// A lock-free-read, single-writer adaptive radix tree over a [`FlatArena`].
///
/// The tree stores its root as an `AtomicU64` tagged pointer and addresses
/// every node through 40-bit arena offsets; raw virtual pointers never
/// appear inside nodes. The read path is completely lock-free:
///
/// - `get` follows acquire-loads of child words and never mutates state.
/// - `insert` and `delete` are single-writer for the current milestone:
///   node growth allocates a fresh node and republishes the parent's child
///   word with a release-store. Single-`CAS` node resizing (swapping
///   type/offset/polymorphic word atomically under concurrent writers) is
///   scheduled with the concurrency hardening milestone; until then
///   concurrent writers must be externally serialised.
///
/// Leaves inline the 56-bit payload (values `< 2^56`); larger values are
/// stored in an atomic 8-byte arena slot referenced by a `LeafOffset`
/// word. Tombstoned words are skipped by readers in `O(1)`.
pub struct FlareArtTree<G: GpuSyncDriver> {
    arena: Arc<FlatArena>,
    hazard: Arc<HazardManager>,
    gpu_driver: G,
    root: AtomicU64,
}

impl<G: GpuSyncDriver> FlareArtTree<G> {
    /// Creates an empty tree over the given arena, hazard manager, and GPU
    /// driver.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flare_core::alloc::arena::FlatArena;
    /// # use flare_core::sync::gpu::CpuFallbackDriver;
    /// # use flare_core::sync::hazard::HazardManager;
    /// # use flare_core::tree::FlareArtTree;
    /// # use std::sync::Arc;
    /// let tree = FlareArtTree::new(
    ///     Arc::new(FlatArena::new(1 << 20).expect("arena fits")),
    ///     Arc::new(HazardManager::new()),
    ///     CpuFallbackDriver::default(),
    /// );
    /// assert!(tree.get(b"missing").expect("lookup succeeds").is_none());
    /// ```
    #[must_use]
    pub const fn new(arena: Arc<FlatArena>, hazard: Arc<HazardManager>, gpu_driver: G) -> Self {
        Self {
            arena,
            hazard,
            gpu_driver,
            root: AtomicU64::new(EMPTY_CHILD),
        }
    }

    /// Returns the arena backing this tree.
    #[must_use]
    pub fn arena(&self) -> &FlatArena {
        &self.arena
    }

    /// Returns the hazard manager tracking this tree's reclamation.
    #[must_use]
    pub fn hazard(&self) -> &HazardManager {
        &self.hazard
    }

    /// Returns the tree root as a tagged pointer.
    ///
    /// The returned word is [`EMPTY_CHILD`] for an empty tree.
    #[must_use]
    pub fn root_tag(&self) -> TaggedPointer {
        TaggedPointer::from_bits(self.root.load(Ordering::Acquire))
    }

    /// Returns the value stored under `key`, if any.
    ///
    /// The lookup is lock-free and only reads arena-resident state through
    /// acquire-loads. A logically deleted entry or an absent path both
    /// report `None`.
    ///
    /// # Errors
    ///
    /// Returns [`FlareError::ArenaBoundsExceeded`] when a stored tagged
    /// pointer references a region outside the arena, indicating a
    /// lifecycle violation.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flare_core::alloc::arena::FlatArena;
    /// # use flare_core::sync::gpu::CpuFallbackDriver;
    /// # use flare_core::sync::hazard::HazardManager;
    /// # use flare_core::tree::FlareArtTree;
    /// # use std::sync::Arc;
    /// let tree = FlareArtTree::new(
    ///     Arc::new(FlatArena::new(1 << 20).expect("arena fits")),
    ///     Arc::new(HazardManager::new()),
    ///     CpuFallbackDriver::default(),
    /// );
    /// tree.insert(b"k", 1).unwrap();
    /// assert_eq!(tree.get(b"k").unwrap(), Some(1));
    /// assert_eq!(tree.get(b"absent").unwrap(), None);
    /// ```
    pub fn get(&self, key: &[u8]) -> Result<Option<u64>, FlareError> {
        let nibbles = key_nibbles(key);
        let root_bits = self.root.load(Ordering::Acquire);
        if root_bits == EMPTY_CHILD {
            return Ok(None);
        }
        let mut tag = TaggedPointer::from_bits(root_bits);
        if tag.is_tombstone() {
            return Ok(None);
        }
        match tag.node_type() {
            kind if is_leaf_type(kind) => {
                return if nibbles.is_empty() {
                    self.read_leaf(tag)
                } else {
                    Ok(None)
                };
            }
            _ => {}
        }
        for &nibble in &nibbles {
            let Some(bits) = self.lookup_child(tag, nibble)? else {
                return Ok(None);
            };
            let child = TaggedPointer::from_bits(bits);
            if child.is_tombstone() {
                return Ok(None);
            }
            tag = child;
        }
        self.read_node_leaf(tag)
    }

    /// Returns the longest stored key that is a prefix of `key` together
    /// with its value.
    ///
    /// The walk is lock-free and mirrors [`Self::get`], but remembers the
    /// deepest leaf value encountered along the queried path instead of
    /// requiring an exact match. A key stored at depth `d` matches any
    /// query whose first `d` bytes equal that key; the empty key, when
    /// present, is a prefix of every query. Tombstoned entries are skipped
    /// at every depth.
    ///
    /// The returned pair is `(matched_bytes, value)`; `None` is returned
    /// when no stored key is a prefix of `key`.
    ///
    /// # Errors
    ///
    /// Returns [`FlareError::ArenaBoundsExceeded`] when a stored tagged
    /// pointer references a region outside the arena, indicating a
    /// lifecycle violation.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flare_core::alloc::arena::FlatArena;
    /// # use flare_core::sync::gpu::CpuFallbackDriver;
    /// # use flare_core::sync::hazard::HazardManager;
    /// # use flare_core::tree::FlareArtTree;
    /// # use std::sync::Arc;
    /// let tree = FlareArtTree::new(
    ///     Arc::new(FlatArena::new(1 << 20).expect("arena fits")),
    ///     Arc::new(HazardManager::new()),
    ///     CpuFallbackDriver::default(),
    /// );
    /// tree.insert(b"ab", 1).unwrap();
    /// tree.insert(b"abcd", 2).unwrap();
    /// assert_eq!(tree.longest_prefix(b"abcde").unwrap(), Some((4, 2)));
    /// assert_eq!(tree.longest_prefix(b"abc").unwrap(), Some((2, 1)));
    /// assert_eq!(tree.longest_prefix(b"x").unwrap(), None);
    /// ```
    pub fn longest_prefix(&self, key: &[u8]) -> Result<Option<(usize, u64)>, FlareError> {
        let nibbles = key_nibbles(key);
        let root_bits = self.root.load(Ordering::Acquire);
        if root_bits == EMPTY_CHILD {
            return Ok(None);
        }
        let mut tag = TaggedPointer::from_bits(root_bits);
        if tag.is_tombstone() {
            return Ok(None);
        }
        if is_leaf_type(tag.node_type()) {
            if nibbles.is_empty() {
                return Ok(self.read_leaf(tag)?.map(|value| (0, value)));
            }
            return Ok(None);
        }
        let mut best: Option<(usize, u64)> = None;
        let mut consumed = 0usize;
        for &nibble in &nibbles {
            if let Some(value) = self.read_node_leaf(tag)? {
                best = Some((consumed / 2, value));
            }
            let Some(bits) = self.lookup_child(tag, nibble)? else {
                return Ok(best);
            };
            let child = TaggedPointer::from_bits(bits);
            if child.is_tombstone() {
                return Ok(best);
            }
            tag = child;
            consumed += 1;
        }
        if let Some(value) = self.read_node_leaf(tag)? {
            best = Some((consumed / 2, value));
        }
        Ok(best)
    }
    ///
    /// Values below `2^56` are inlined into the leaf word; larger values
    /// are stored in an atomic arena slot. An existing value is replaced
    /// and returned. The write path publishes the new root with a
    /// release-store after every arena region is initialised, and emits a
    /// GPU epoch fence so reader warps never observe partial writes.
    ///
    /// # Errors
    ///
    /// Returns arena capacity errors when the tree exhausts its bump
    /// frontier, or a driver error when the epoch fence fails.
    ///
    /// # Panics
    ///
    /// Panics if a stored tagged pointer violates the tree invariants,
    /// indicating memory corruption.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flare_core::alloc::arena::FlatArena;
    /// # use flare_core::sync::gpu::CpuFallbackDriver;
    /// # use flare_core::sync::hazard::HazardManager;
    /// # use flare_core::tree::FlareArtTree;
    /// # use std::sync::Arc;
    /// let tree = FlareArtTree::new(
    ///     Arc::new(FlatArena::new(1 << 20).expect("arena fits")),
    ///     Arc::new(HazardManager::new()),
    ///     CpuFallbackDriver::default(),
    /// );
    /// assert_eq!(tree.insert(b"k", 1).unwrap(), None);
    /// assert_eq!(tree.insert(b"k", 2).unwrap(), Some(1));
    /// ```
    pub fn insert(&self, key: &[u8], value: u64) -> Result<Option<u64>, FlareError> {
        let nibbles = key_nibbles(key);
        loop {
            let root_bits = self.root.load(Ordering::Acquire);
            let (old, new_root) = if root_bits == EMPTY_CHILD {
                (None, self.create_path(&nibbles, value)?)
            } else {
                let root_tag = TaggedPointer::from_bits(root_bits);
                match root_tag.node_type() {
                    kind if is_leaf_type(kind) => {
                        let old: Option<u64> = if root_tag.is_tombstone() {
                            None
                        } else {
                            self.read_leaf(root_tag)?
                        };
                        if nibbles.is_empty() {
                            (old, self.make_leaf(value)?)
                        } else {
                            let chain = self.create_path(&nibbles, value)?;
                            // Preserve the empty-key value in the chain root's
                            // leaf word.
                            self.write_node_leaf(chain, root_bits)?;
                            (old, chain)
                        }
                    }
                    _ => self.insert_nibbles(root_tag, &nibbles, value)?,
                }
            };
            if self
                .root
                .compare_exchange(
                    root_bits,
                    new_root.to_bits(),
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
            {
                self.gpu_driver.publish_epoch_fence(0)?;
                return Ok(old);
            }
            // Another insert published a fresh root while the path above
            // was being rebuilt; the orphaned nodes stay unreachable in
            // the append-only arena and the walk restarts against the new
            // root.
        }
    }

    /// Deletes the entry under `key`, reporting whether it existed.
    ///
    /// The leaf word of the terminal node is cleared with a release-store;
    /// nodes are not physically reclaimed in this milestone (see the
    /// module documentation). A missing key reports `false`.
    ///
    /// # Errors
    ///
    /// Returns arena bounds errors when a stored pointer is corrupted, or
    /// driver errors when the epoch fence fails.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flare_core::alloc::arena::FlatArena;
    /// # use flare_core::sync::gpu::CpuFallbackDriver;
    /// # use flare_core::sync::hazard::HazardManager;
    /// # use flare_core::tree::FlareArtTree;
    /// # use std::sync::Arc;
    /// let tree = FlareArtTree::new(
    ///     Arc::new(FlatArena::new(1 << 20).expect("arena fits")),
    ///     Arc::new(HazardManager::new()),
    ///     CpuFallbackDriver::default(),
    /// );
    /// tree.insert(b"k", 1).unwrap();
    /// assert_eq!(tree.delete(b"k").unwrap(), true);
    /// assert_eq!(tree.delete(b"k").unwrap(), false);
    /// ```
    pub fn delete(&self, key: &[u8]) -> Result<bool, FlareError> {
        let nibbles = key_nibbles(key);
        let root_bits = self.root.load(Ordering::Acquire);
        if root_bits == EMPTY_CHILD {
            return Ok(false);
        }
        let root_tag = TaggedPointer::from_bits(root_bits);
        if is_leaf_type(root_tag.node_type()) {
            if !nibbles.is_empty() {
                return Ok(false);
            }
            self.root.store(EMPTY_CHILD, Ordering::Release);
            self.gpu_driver.publish_epoch_fence(0)?;
            return Ok(true);
        }
        let mut tag = root_tag;
        for &nibble in &nibbles {
            let Some(bits) = self.lookup_child(tag, nibble)? else {
                return Ok(false);
            };
            let child = TaggedPointer::from_bits(bits);
            if child.is_tombstone() {
                return Ok(false);
            }
            tag = child;
        }
        let leaf = self.node_leaf(tag)?;
        let bits = leaf.load(Ordering::Acquire);
        if bits == EMPTY_CHILD || TaggedPointer::from_bits(bits).is_tombstone() {
            return Ok(false);
        }
        leaf.store(EMPTY_CHILD, Ordering::Release);
        self.gpu_driver.publish_epoch_fence(0)?;
        Ok(true)
    }

    // -- internals ---------------------------------------------------------

    /// Looks up the child word below `tag` for `nibble`.
    fn lookup_child(&self, tag: TaggedPointer, nibble: u8) -> Result<Option<u64>, FlareError> {
        match tag.node_type() {
            NodeType::Node4 => {
                let Some(index) = resolve_child_index(tag, nibble) else {
                    return Ok(None);
                };
                let node = self.node4(tag)?;
                if node.keys()[index] != nibble {
                    return Ok(None);
                }
                Ok(Some(node.children()[index].load(Ordering::Acquire)))
            }
            NodeType::Node16 => {
                let Some(index) = resolve_child_index(tag, nibble) else {
                    return Ok(None);
                };
                let node = self.node16(tag)?;
                if node.keys()[index] != nibble {
                    return Ok(None);
                }
                Ok(Some(node.children()[index].load(Ordering::Acquire)))
            }
            NodeType::Node64 => {
                let node = self.node64(tag)?;
                let bitmap = node.bitmap().load(Ordering::Acquire);
                let bit = 1u64 << u64::from(nibble);
                if bitmap & bit == 0 {
                    return Ok(None);
                }
                let dense = (bitmap & (bit - 1)).count_ones() as usize;
                Ok(Some(node.children()[dense].load(Ordering::Acquire)))
            }
            NodeType::Node256 => {
                let node = self.node256(tag)?;
                let bits = node.children()[nibble as usize].load(Ordering::Acquire);
                if bits == EMPTY_CHILD {
                    Ok(None)
                } else {
                    Ok(Some(bits))
                }
            }
            kind if is_leaf_type(kind) => Err(FlareError::TreeInvariantViolation {
                reason: "leaf word encountered mid-path",
            }),
            _ => Err(FlareError::InvalidNodeType(tag.node_type().discriminant())),
        }
    }

    /// Reads the terminal value of `tag` treating it as a leaf word.
    fn read_leaf(&self, tag: TaggedPointer) -> Result<Option<u64>, FlareError> {
        match tag.node_type() {
            NodeType::LeafInlined => Ok(Some(tag.unpack_inline_payload())),
            NodeType::LeafOffset => {
                let word = self.arena.atomic_word(tag.offset())?;
                Ok(Some(word.load(Ordering::Acquire)))
            }
            kind if is_leaf_type(kind) => Ok(None),
            _ => Err(FlareError::TreeInvariantViolation {
                reason: "internal node treated as a leaf",
            }),
        }
    }

    /// Reads the leaf word of the internal node addressed by `tag`.
    fn read_node_leaf(&self, tag: TaggedPointer) -> Result<Option<u64>, FlareError> {
        let leaf = self.node_leaf(tag)?;
        let bits = leaf.load(Ordering::Acquire);
        if bits == EMPTY_CHILD || TaggedPointer::from_bits(bits).is_tombstone() {
            return Ok(None);
        }
        self.read_leaf(TaggedPointer::from_bits(bits))
    }

    /// Returns the leaf word slot of the internal node addressed by `tag`.
    fn node_leaf(&self, tag: TaggedPointer) -> Result<&AtomicU64, FlareError> {
        match tag.node_type() {
            NodeType::Node4 => Ok(self.node4(tag)?.leaf()),
            NodeType::Node16 => Ok(self.node16(tag)?.leaf()),
            NodeType::Node64 => Ok(self.node64(tag)?.leaf()),
            NodeType::Node256 => Ok(self.node256(tag)?.leaf()),
            kind if is_leaf_type(kind) => Err(FlareError::TreeInvariantViolation {
                reason: "leaf word of a leaf word",
            }),
            _ => Err(FlareError::InvalidNodeType(tag.node_type().discriminant())),
        }
    }

    /// Overwrites the leaf word of the internal node addressed by `tag`.
    fn write_node_leaf(&self, tag: TaggedPointer, bits: u64) -> Result<(), FlareError> {
        self.node_leaf(tag)?.store(bits, Ordering::Release);
        Ok(())
    }

    /// Reads a typed node from the arena with bounds checking.
    fn typed_node<T>(&self, tag: TaggedPointer) -> Result<&T, FlareError> {
        let miss = FlareError::ArenaBoundsExceeded {
            offset: tag.offset(),
            length: core::mem::size_of::<T>(),
            capacity: self.arena.capacity(),
        };
        let Some(handle) = self.arena.node_ref::<T>(tag.offset()) else {
            return Err(miss);
        };
        handle.get().ok_or(miss)
    }

    /// Reads a typed `Node4` from the arena with bounds checking.
    fn node4(&self, tag: TaggedPointer) -> Result<&Node4, FlareError> {
        self.typed_node(tag)
    }

    /// Reads a typed `Node16` from the arena with bounds checking.
    fn node16(&self, tag: TaggedPointer) -> Result<&Node16, FlareError> {
        self.typed_node(tag)
    }

    /// Reads a typed `Node64` from the arena with bounds checking.
    fn node64(&self, tag: TaggedPointer) -> Result<&Node64, FlareError> {
        self.typed_node(tag)
    }

    /// Reads a typed `Node256` from the arena with bounds checking.
    fn node256(&self, tag: TaggedPointer) -> Result<&Node256, FlareError> {
        self.typed_node(tag)
    }

    /// Allocates and initialises an arena-resident `Node4`, returning its
    /// tagged pointer.
    fn build_node4(
        &self,
        keys: [u8; 4],
        children: [u64; 4],
        leaf: u64,
    ) -> Result<TaggedPointer, FlareError> {
        let offset = self.arena.alloc(core::mem::size_of::<Node4>(), 8)?;
        self.arena
            .write_node(offset, &Node4::new(keys, children, leaf))?;
        let poly = keys
            .iter()
            .filter(|k| **k != EMPTY_KEY)
            .fold(0u16, |acc, k| acc | (1u16 << u16::from(*k)));
        Ok(TaggedPointer::pack(NodeType::Node4, offset, 0, false, poly))
    }

    /// Allocates and initialises an arena-resident `Node16`.
    fn build_node16(
        &self,
        keys: [u8; 16],
        children: [u64; 16],
        leaf: u64,
    ) -> Result<TaggedPointer, FlareError> {
        let offset = self.arena.alloc(core::mem::size_of::<Node16>(), 8)?;
        self.arena
            .write_node(offset, &Node16::new(keys, children, leaf))?;
        let poly = keys
            .iter()
            .filter(|k| **k != EMPTY_KEY)
            .fold(0u16, |acc, k| acc | (1u16 << u16::from(*k)));
        Ok(TaggedPointer::pack(
            NodeType::Node16,
            offset,
            0,
            false,
            poly,
        ))
    }

    /// Allocates and initialises an arena-resident `Node64`.
    fn build_node64(
        &self,
        bitmap: u64,
        children: &[u64; 64],
        leaf: u64,
        generation: u16,
    ) -> Result<TaggedPointer, FlareError> {
        let offset = self.arena.alloc(core::mem::size_of::<Node64>(), 8)?;
        self.arena
            .write_node(offset, &Node64::new(bitmap, *children, leaf))?;
        let count = u16::try_from(bitmap.count_ones()).expect("bitmap count fits in u16");
        let poly = (count << 8) | (generation & 0xFF);
        Ok(TaggedPointer::pack(
            NodeType::Node64,
            offset,
            0,
            false,
            poly,
        ))
    }

    /// Allocates and initialises an arena-resident `Node256`.
    fn build_node256(
        &self,
        children: &[u64; 256],
        leaf: u64,
        generation: u16,
    ) -> Result<TaggedPointer, FlareError> {
        let offset = self.arena.alloc(core::mem::size_of::<Node256>(), 8)?;
        self.arena
            .write_node(offset, &Node256::new(*children, leaf))?;
        let count = u16::try_from(children.iter().filter(|c| **c != EMPTY_CHILD).count())
            .expect("child count fits in u16");
        let poly = (count << 8) | (generation & 0xFF);
        Ok(TaggedPointer::pack(
            NodeType::Node256,
            offset,
            0,
            false,
            poly,
        ))
    }

    /// Creates a chain of `Node4`s descending along `nibbles`, terminating
    /// in a node whose leaf word carries `value`.
    ///
    /// The terminal node stores the leaf word in its own `leaf` slot with
    /// empty keys, so a lookup consuming every nibble lands on a node whose
    /// [`Self::read_node_leaf`] resolves the payload; every ancestor
    /// re-packages the node below it as its single child.
    fn create_path(&self, nibbles: &[u8], value: u64) -> Result<TaggedPointer, FlareError> {
        let leaf_word = self.make_leaf(value)?.to_bits();
        let mut current = self.build_node4([EMPTY_KEY; 4], [EMPTY_CHILD; 4], leaf_word)?;
        for &nibble in nibbles.iter().rev() {
            let mut keys = [EMPTY_KEY; 4];
            keys[0] = nibble;
            let mut children = [EMPTY_CHILD; 4];
            children[0] = current.to_bits();
            current = self.build_node4(keys, children, EMPTY_CHILD)?;
        }
        Ok(current)
    }

    /// Builds the leaf word for `value` (inlined when possible).
    fn make_leaf(&self, value: u64) -> Result<TaggedPointer, FlareError> {
        const INLINE_LIMIT: u64 = 1 << 56;
        if value < INLINE_LIMIT {
            return Ok(TaggedPointer::pack_inline_payload(value, false));
        }
        let offset = self.arena.alloc(8, 8)?;
        self.arena
            .atomic_word(offset)?
            .store(value, Ordering::Release);
        Ok(TaggedPointer::pack(
            NodeType::LeafOffset,
            offset,
            0,
            false,
            0,
        ))
    }

    /// Inserts `value` below `tag`, returning the previous value (if any)
    /// and the (possibly new) tagged pointer to publish into the parent.
    fn insert_nibbles(
        &self,
        tag: TaggedPointer,
        nibbles: &[u8],
        value: u64,
    ) -> Result<(Option<u64>, TaggedPointer), FlareError> {
        if nibbles.is_empty() {
            let leaf = self.node_leaf(tag)?;
            let bits = leaf.load(Ordering::Acquire);
            let old: Option<u64> =
                if bits == EMPTY_CHILD || TaggedPointer::from_bits(bits).is_tombstone() {
                    None
                } else {
                    self.read_leaf(TaggedPointer::from_bits(bits))?
                };
            leaf.store(self.make_leaf(value)?.to_bits(), Ordering::Release);
            return Ok((old, tag));
        }
        let nibble = nibbles[0];
        if let Some(bits) = self.lookup_child(tag, nibble)? {
            let child = TaggedPointer::from_bits(bits);
            if child.is_tombstone() {
                let new_tag =
                    self.store_child(tag, nibble, self.create_path(&nibbles[1..], value)?)?;
                return Ok((None, new_tag));
            }
            let (old, new_child) = self.insert_nibbles(child, &nibbles[1..], value)?;
            if new_child == child {
                Ok((old, tag))
            } else {
                let new_tag = self.store_child(tag, nibble, new_child)?;
                Ok((old, new_tag))
            }
        } else {
            let chain = self.create_path(&nibbles[1..], value)?;
            let new_tag = self.store_child(tag, nibble, chain)?;
            Ok((None, new_tag))
        }
    }

    /// Installs `child` under `nibble` inside the node addressed by `tag`.
    ///
    /// Nodes are immutable after publication: the updated node is written
    /// to a fresh arena region, and its new tagged pointer is returned to
    /// the parent for a single word store. Growth cascades `Node4` →
    /// `Node16` → `Node64` → `Node256` when the current family is full.
    /// Replaces the child at `nibble` below `tag`, returning the tagged
    /// pointer of the (possibly resized) node.
    #[allow(clippy::too_many_lines)]
    fn store_child(
        &self,
        tag: TaggedPointer,
        nibble: u8,
        child: TaggedPointer,
    ) -> Result<TaggedPointer, FlareError> {
        match tag.node_type() {
            NodeType::Node4 => {
                let node = self.node4(tag)?;
                let mask = tag.polymorphic_field();
                let count = mask.count_ones();
                let mut children = [EMPTY_CHILD; 4];
                for (i, slot) in node.children().iter().enumerate() {
                    children[i] = slot.load(Ordering::Acquire);
                }
                let leaf = node.leaf().load(Ordering::Acquire);
                if let Some(position) = node.keys()[..count as usize]
                    .iter()
                    .position(|k| *k == nibble)
                {
                    children[position] = child.to_bits();
                    self.build_node4(*node.keys(), children, leaf)
                } else if count < 4 {
                    let index = (mask & ((1u16 << u16::from(nibble)) - 1)).count_ones() as usize;
                    let mut keys = *node.keys();
                    keys.copy_within(index..3, index + 1);
                    keys[index] = nibble;
                    children.copy_within(index..3, index + 1);
                    children[index] = child.to_bits();
                    self.build_node4(keys, children, leaf)
                } else {
                    let mut pairs: [(u8, u64); 5] = [(EMPTY_KEY, EMPTY_CHILD); 5];
                    for (i, slot) in node.children().iter().enumerate() {
                        pairs[i] = (node.keys()[i], slot.load(Ordering::Acquire));
                    }
                    pairs[4] = (nibble, child.to_bits());
                    pairs.sort_unstable_by_key(|(k, _)| *k);
                    let mut keys = [EMPTY_KEY; 16];
                    let mut children = [EMPTY_CHILD; 16];
                    for (i, (k, bits)) in pairs.iter().enumerate() {
                        keys[i] = *k;
                        children[i] = *bits;
                    }
                    self.build_node16(keys, children, leaf)
                }
            }
            NodeType::Node16 => {
                let node = self.node16(tag)?;
                let mask = tag.polymorphic_field();
                let count = mask.count_ones();
                let mut children = [EMPTY_CHILD; 16];
                for (i, slot) in node.children().iter().enumerate() {
                    children[i] = slot.load(Ordering::Acquire);
                }
                let leaf = node.leaf().load(Ordering::Acquire);
                if let Some(position) = node.keys()[..count as usize]
                    .iter()
                    .position(|k| *k == nibble)
                {
                    children[position] = child.to_bits();
                    self.build_node16(*node.keys(), children, leaf)
                } else if count < 16 {
                    let index = (mask & ((1u16 << u16::from(nibble)) - 1)).count_ones() as usize;
                    let mut keys = *node.keys();
                    keys.copy_within(index..15, index + 1);
                    keys[index] = nibble;
                    children.copy_within(index..15, index + 1);
                    children[index] = child.to_bits();
                    self.build_node16(keys, children, leaf)
                } else {
                    let mut pairs: [(u8, u64); 17] = [(EMPTY_KEY, EMPTY_CHILD); 17];
                    for (i, slot) in node.children().iter().enumerate() {
                        pairs[i] = (node.keys()[i], slot.load(Ordering::Acquire));
                    }
                    pairs[16] = (nibble, child.to_bits());
                    pairs.sort_unstable_by_key(|(k, _)| *k);
                    let mut bitmap = 0u64;
                    let mut children = [EMPTY_CHILD; 64];
                    for (i, (k, bits)) in pairs.iter().enumerate() {
                        bitmap |= 1u64 << u64::from(*k);
                        children[i] = *bits;
                    }
                    let generation = tag.polymorphic_field() & 0xFF;
                    self.build_node64(bitmap, &children, leaf, generation)
                }
            }
            NodeType::Node64 => {
                let node = self.node64(tag)?;
                let bitmap = node.bitmap().load(Ordering::Acquire);
                let count = bitmap.count_ones();
                let bit = 1u64 << u64::from(nibble);
                let mut children = [EMPTY_CHILD; 64];
                for (i, slot) in node.children().iter().enumerate() {
                    children[i] = slot.load(Ordering::Acquire);
                }
                let leaf = node.leaf().load(Ordering::Acquire);
                let generation = tag.polymorphic_field() & 0xFF;
                if bitmap & bit != 0 {
                    let dense = (bitmap & (bit - 1)).count_ones() as usize;
                    children[dense] = child.to_bits();
                    self.build_node64(bitmap, &children, leaf, generation)
                } else if count < 16 {
                    let index = (bitmap & (bit - 1)).count_ones() as usize;
                    children.copy_within(index..63, index + 1);
                    children[index] = child.to_bits();
                    self.build_node64(bitmap | bit, &children, leaf, generation)
                } else {
                    let mut children = [EMPTY_CHILD; 256];
                    for bit in 0_u8..16 {
                        if bitmap & (1u64 << u64::from(bit)) != 0 {
                            let dense =
                                (bitmap & ((1u64 << u64::from(bit)) - 1)).count_ones() as usize;
                            children[usize::from(bit)] =
                                node.children()[dense].load(Ordering::Acquire);
                        }
                    }
                    children[nibble as usize] = child.to_bits();
                    self.build_node256(&children, leaf, generation)
                }
            }
            NodeType::Node256 => {
                let node = self.node256(tag)?;
                let mut children = [EMPTY_CHILD; 256];
                for (i, slot) in node.children().iter().enumerate() {
                    children[i] = slot.load(Ordering::Acquire);
                }
                children[nibble as usize] = child.to_bits();
                let leaf = node.leaf().load(Ordering::Acquire);
                let generation = tag.polymorphic_field() & 0xFF;
                self.build_node256(&children, leaf, generation)
            }
            kind if is_leaf_type(kind) => Err(FlareError::TreeInvariantViolation {
                reason: "child insertion below a leaf",
            }),
            _ => Err(FlareError::InvalidNodeType(tag.node_type().discriminant())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{EMPTY_CHILD, FlareArtTree, NodeType, TaggedPointer};
    use crate::alloc::arena::FlatArena;
    use crate::sync::gpu::CpuFallbackDriver;
    use crate::sync::hazard::HazardManager;
    use alloc_crate::sync::Arc;
    use core::sync::atomic::Ordering;

    fn test_tree() -> FlareArtTree<CpuFallbackDriver> {
        FlareArtTree::new(
            Arc::new(FlatArena::new(1 << 20).expect("arena fits")),
            Arc::new(HazardManager::new()),
            CpuFallbackDriver::default(),
        )
    }

    /// Verifies that concurrent inserts never lose entries: the root is
    /// published with a compare-and-swap retry loop, so every path rebuilt
    /// from a stale root snapshot is retried against the winning root.
    ///
    /// The keys share a long zero prefix, maximising contention on the
    /// deep common chain (the same shape flare-kv queries create).
    #[test]
    fn concurrent_inserts_are_all_present() {
        use alloc_crate::vec::Vec;
        use std::sync::Barrier;
        let tree = Arc::new(FlareArtTree::new(
            Arc::new(FlatArena::new(1 << 23).expect("arena fits")),
            Arc::new(HazardManager::new()),
            CpuFallbackDriver::default(),
        ));
        let barrier = Arc::new(Barrier::new(5));
        let mut threads = Vec::new();
        for t in 0..4u64 {
            let tree = Arc::clone(&tree);
            let barrier = Arc::clone(&barrier);
            threads.push(std::thread::spawn(move || {
                barrier.wait();
                for i in 0..100u64 {
                    let key = t * 1000 + i;
                    let bytes = [key.to_le_bytes(), (key + 1).to_le_bytes()].concat();
                    tree.insert(&bytes, key + 1).expect("insert succeeds");
                }
            }));
        }
        barrier.wait();
        for thread in threads {
            thread.join().expect("worker finishes");
        }
        for t in 0..4u64 {
            for i in 0..100u64 {
                let key = t * 1000 + i;
                let bytes = [key.to_le_bytes(), (key + 1).to_le_bytes()].concat();
                assert_eq!(
                    tree.get(&bytes).expect("lookup succeeds"),
                    Some(key + 1),
                    "key {key}"
                );
            }
        }
    }

    /// Verifies that every inserted key round-trips through the nibble
    /// chain, including keys sharing long prefixes.
    #[test]
    fn insert_get_roundtrip() {
        let tree = test_tree();
        for key in [
            &b"hello"[..],
            &b"world"[..],
            &b"hell"[..],
            &b"hellfire"[..],
            &b""[..],
            &b"\x00\x00"[..],
            &b"\xFF\xFF\xFF"[..],
        ] {
            tree.insert(key, 7).expect("insert succeeds");
            assert_eq!(
                tree.get(key).expect("lookup succeeds"),
                Some(7),
                "key {key:?}"
            );
        }
        let tree = test_tree();
        let keys: &[&[u8]] = &[
            b"hello",
            b"world",
            b"hell",
            b"hellfire",
            b"",
            b"\x00\x00",
            b"\xFF\xFF\xFF",
            b"a very long key that spans many levels of the radix tree",
        ];
        for (i, key) in keys.iter().enumerate() {
            let value = i as u64 * 7 + 1;
            assert_eq!(tree.insert(key, value).expect("insert succeeds"), None);
            assert_eq!(tree.get(key).expect("lookup succeeds"), Some(value));
        }
        for (i, key) in keys.iter().enumerate() {
            let value = i as u64 * 7 + 1;
            assert_eq!(tree.get(key).expect("lookup succeeds"), Some(value));
        }
        assert_eq!(tree.get(b"absent").expect("lookup succeeds"), None);
    }

    /// Verifies that overwriting a key returns the previous value.
    #[test]
    fn overwrite_returns_previous() {
        let tree = test_tree();
        assert_eq!(tree.insert(b"k", 1).expect("insert succeeds"), None);
        assert_eq!(tree.insert(b"k", 2).expect("insert succeeds"), Some(1));
        assert_eq!(tree.insert(b"k", 3).expect("insert succeeds"), Some(2));
        assert_eq!(tree.get(b"k").expect("lookup succeeds"), Some(3));
    }

    /// Verifies that the empty key round-trips through insert and delete.
    #[test]
    fn empty_key_roundtrip() {
        let tree = test_tree();
        assert_eq!(tree.insert(b"", 41).expect("insert succeeds"), None);
        assert_eq!(tree.get(b"").expect("lookup succeeds"), Some(41));
        assert!(tree.delete(b"").expect("delete succeeds"));
        assert_eq!(tree.get(b"").expect("lookup succeeds"), None);
        assert!(!tree.delete(b"").expect("delete succeeds"));
    }

    /// Verifies longest-prefix resolution across nested key depths.
    #[test]
    fn longest_prefix_nested_depths() {
        let tree = test_tree();
        assert_eq!(tree.longest_prefix(b"abc").expect("walk succeeds"), None);
        tree.insert(b"ab", 1).expect("insert succeeds");
        tree.insert(b"abcd", 2).expect("insert succeeds");
        tree.insert(b"abcdef", 3).expect("insert succeeds");
        assert_eq!(
            tree.longest_prefix(b"abcdefgh").expect("walk succeeds"),
            Some((6, 3))
        );
        assert_eq!(
            tree.longest_prefix(b"abcde").expect("walk succeeds"),
            Some((4, 2))
        );
        assert_eq!(
            tree.longest_prefix(b"abc").expect("walk succeeds"),
            Some((2, 1))
        );
        assert_eq!(
            tree.longest_prefix(b"ab").expect("walk succeeds"),
            Some((2, 1))
        );
        assert_eq!(tree.longest_prefix(b"a").expect("walk succeeds"), None);
        assert_eq!(tree.longest_prefix(b"xyz").expect("walk succeeds"), None);
    }

    /// Verifies the empty key acts as a universal prefix and that a leaf
    /// root resolves only the empty key.
    #[test]
    fn longest_prefix_empty_key() {
        let tree = test_tree();
        tree.insert(b"", 5).expect("insert succeeds");
        assert_eq!(
            tree.longest_prefix(b"anything").expect("walk succeeds"),
            Some((0, 5))
        );
        assert_eq!(
            tree.longest_prefix(b"").expect("walk succeeds"),
            Some((0, 5))
        );
        tree.insert(b"ab", 9).expect("insert succeeds");
        assert_eq!(
            tree.longest_prefix(b"ab").expect("walk succeeds"),
            Some((2, 9))
        );
        assert_eq!(
            tree.longest_prefix(b"ac").expect("walk succeeds"),
            Some((0, 5))
        );
    }

    /// Verifies tombstoned prefixes are skipped during the walk.
    #[test]
    fn longest_prefix_skips_tombstones() {
        let tree = test_tree();
        tree.insert(b"ab", 1).expect("insert succeeds");
        tree.insert(b"abcd", 2).expect("insert succeeds");
        assert!(tree.delete(b"ab").expect("delete succeeds"));
        assert_eq!(tree.longest_prefix(b"abc").expect("walk succeeds"), None);
        assert_eq!(
            tree.longest_prefix(b"abcd").expect("walk succeeds"),
            Some((4, 2))
        );
        tree.insert(b"ab", 7).expect("insert succeeds");
        assert_eq!(
            tree.longest_prefix(b"abc").expect("walk succeeds"),
            Some((2, 7))
        );
    }

    /// Verifies that values at or above `2^56` fall back to the atomic
    /// `LeafOffset` slot and still round-trip.
    #[test]
    fn large_values_use_leaf_offset() {
        let tree = test_tree();
        let values = [1u64 << 56, u64::MAX, (1u64 << 56) + 5];
        for (i, value) in values.iter().enumerate() {
            let key = [b'v', b'0' + u8::try_from(i).expect("index fits in u8")];
            assert_eq!(tree.insert(&key, *value).expect("insert succeeds"), None);
            assert_eq!(tree.get(&key).expect("lookup succeeds"), Some(*value));
        }
    }

    /// Verifies that delete removes exactly the targeted entry and that
    /// re-deleting reports `false`.
    #[test]
    fn delete_roundtrip() {
        let tree = test_tree();
        tree.insert(b"alpha", 1).expect("insert succeeds");
        tree.insert(b"beta", 2).expect("insert succeeds");
        tree.insert(b"gamma", 3).expect("insert succeeds");
        assert!(tree.delete(b"beta").expect("delete succeeds"));
        assert_eq!(tree.get(b"beta").expect("lookup succeeds"), None);
        assert_eq!(tree.get(b"alpha").expect("lookup succeeds"), Some(1));
        assert_eq!(tree.get(b"gamma").expect("lookup succeeds"), Some(3));
        assert!(!tree.delete(b"beta").expect("delete succeeds"));
        assert!(!tree.delete(b"missing").expect("delete succeeds"));
    }

    /// Verifies that a fifth child sharing the first nibble cascades the
    /// node family from `Node4` to `Node16`.
    #[test]
    fn growth_n4_to_n16() {
        let tree = test_tree();
        let keys: [&[u8]; 8] = [
            b"\x60\x00",
            b"\x61\x00",
            b"\x62\x00",
            b"\x63\x00",
            b"\x64\x00",
            b"\x65\x00",
            b"\x66\x00",
            b"\x67\x00",
        ];
        for (i, key) in keys.iter().enumerate() {
            assert_eq!(
                tree.insert(key, i as u64).expect("insert succeeds"),
                None,
                "key {i}"
            );
        }
        assert_eq!(
            tree.root_tag().node_type().discriminant(),
            2,
            "root remains a Node4"
        );
        for (i, key) in keys.iter().enumerate() {
            assert_eq!(
                tree.get(key).expect("lookup succeeds"),
                Some(i as u64),
                "key {i}"
            );
        }
    }

    /// Verifies that inserts with equal trailing nibbles never corrupt
    /// sibling entries (tombstone isolation).
    #[test]
    fn tombstoned_branch_stays_isolated() {
        let tree = test_tree();
        tree.insert(b"cat", 1).expect("insert succeeds");
        tree.insert(b"car", 2).expect("insert succeeds");
        tree.insert(b"cap", 3).expect("insert succeeds");
        assert!(tree.delete(b"car").expect("delete succeeds"));
        assert_eq!(tree.get(b"cat").expect("lookup succeeds"), Some(1));
        assert_eq!(tree.get(b"car").expect("lookup succeeds"), None);
        assert_eq!(tree.get(b"cap").expect("lookup succeeds"), Some(3));
        tree.insert(b"car", 9).expect("reinsert succeeds");
        assert_eq!(tree.get(b"car").expect("lookup succeeds"), Some(9));
        assert_eq!(tree.get(b"cat").expect("lookup succeeds"), Some(1));
    }

    /// Verifies that a full `Node16` (sixteen distinct nibbles at one
    /// level) serves lookups and child replacement without corruption, and
    /// that the adaptive family stops there for nibble-keyed levels.
    #[test]
    fn node16_saturation() {
        let tree = test_tree();
        let keys: [&[u8]; 16] = [
            b"\x60", b"\x61", b"\x62", b"\x63", b"\x64", b"\x65", b"\x66", b"\x67", b"\x68",
            b"\x69", b"\x6A", b"\x6B", b"\x6C", b"\x6D", b"\x6E", b"\x6F",
        ];
        for (i, key) in keys.iter().enumerate() {
            assert_eq!(
                tree.insert(key, i as u64).expect("insert succeeds"),
                None,
                "key {i}"
            );
        }
        assert_eq!(
            tree.insert(b"\x65", 100).expect("overwrite succeeds"),
            Some(5),
            "Node16 child replacement returns the previous value"
        );
        assert_eq!(
            tree.insert(b"\x6A", 200).expect("overwrite succeeds"),
            Some(10),
            "Node16 child replacement returns the previous value"
        );
        for (i, key) in keys.iter().enumerate() {
            let expected = match i {
                5 => 100,
                10 => 200,
                _ => i as u64,
            };
            assert_eq!(
                tree.get(key).expect("lookup succeeds"),
                Some(expected),
                "key {i}"
            );
        }
        assert_eq!(
            tree.get(b"\x7F").expect("lookup succeeds"),
            None,
            "absent nibble reports None"
        );
        assert_eq!(
            tree.insert(b"\x70", 999).expect("insert succeeds"),
            None,
            "a sibling first nibble extends the root, not the saturated level"
        );
        assert_eq!(tree.get(b"\x70").expect("lookup succeeds"), Some(999));
        assert_eq!(tree.get(b"\x6F").expect("lookup succeeds"), Some(15));
    }

    /// Verifies that converting a leaf root into a chain for a deep insert
    /// preserves the empty-key value in the new root's leaf word.
    #[test]
    fn leaf_root_deep_insert_preserves_empty_value() {
        let tree = test_tree();
        tree.insert(b"", 5).expect("insert succeeds");
        tree.insert(b"ab", 7).expect("insert succeeds");
        assert_eq!(tree.get(b"").expect("lookup succeeds"), Some(5));
        assert_eq!(tree.get(b"ab").expect("lookup succeeds"), Some(7));
        tree.insert(b"ab", 8).expect("overwrite succeeds");
        assert_eq!(tree.get(b"").expect("lookup succeeds"), Some(5));
        assert_eq!(tree.get(b"ab").expect("lookup succeeds"), Some(8));
    }

    /// Verifies leaf-root interactions: non-matching lookups and deletes
    /// on a single-entry tree report `None` / `false` without corruption.
    #[test]
    fn leaf_root_interactions() {
        let tree = test_tree();
        tree.insert(b"k", 1).expect("insert succeeds");
        assert_eq!(tree.get(b"other").expect("lookup succeeds"), None);
        assert!(!tree.delete(b"other").expect("delete succeeds"));
        assert_eq!(tree.get(b"k").expect("lookup succeeds"), Some(1));
        assert!(tree.delete(b"k").expect("delete succeeds"));
        assert!(!tree.delete(b"k").expect("delete succeeds"));
        assert_eq!(tree.get(b"k").expect("lookup succeeds"), None);
    }

    /// Verifies empty-tree behaviour and the arena/hazard accessors.
    #[test]
    fn empty_tree_and_accessors() {
        let tree = test_tree();
        assert_eq!(tree.get(b"x").expect("lookup succeeds"), None);
        assert!(!tree.delete(b"x").expect("delete succeeds"));
        assert_eq!(tree.root_tag().to_bits(), crate::tree::EMPTY_CHILD);
        assert_eq!(tree.arena().capacity(), 1 << 20);
        assert_eq!(tree.hazard().current_era(), 0);
    }

    /// Verifies the `Node64` family machinery directly: dense bitmap
    /// construction, popcount-indexed lookups, in-place child replacement,
    /// sparse insertion, and the direct-indexed `Node256` behaviour.
    ///
    /// A nibble-keyed level admits at most sixteen distinct children, so
    /// these families are unreachable from the public API; the builders and
    /// accessors are driven here to pin their invariants.
    #[test]
    fn node64_family_machinery() {
        let tree = test_tree();
        let mut children = [EMPTY_CHILD; 64];
        children[0] = tree.make_leaf(10).expect("leaf").to_bits();
        children[1] = tree.make_leaf(20).expect("leaf").to_bits();
        children[2] = tree.make_leaf(30).expect("leaf").to_bits();
        let bitmap = (1u64 << 1) | (1u64 << 4) | (1u64 << 9);
        let tag = tree
            .build_node64(
                bitmap,
                &children,
                tree.make_leaf(7).expect("leaf").to_bits(),
                3,
            )
            .expect("node64 builds");
        assert_eq!(tag.node_type(), NodeType::Node64);
        assert_eq!(tag.polymorphic_field() & 0xFF, 3, "generation preserved");
        assert_eq!(tag.polymorphic_field() >> 8, 3, "refcount = 3 children");
        assert_eq!(
            tree.lookup_child(tag, 1).expect("lookup"),
            Some(children[0])
        );
        assert_eq!(
            tree.lookup_child(tag, 4).expect("lookup"),
            Some(children[1])
        );
        assert_eq!(
            tree.lookup_child(tag, 9).expect("lookup"),
            Some(children[2])
        );
        assert_eq!(tree.lookup_child(tag, 2).expect("lookup"), None);
        assert_eq!(tree.lookup_child(tag, 15).expect("lookup"), None);
        let new_leaf = tree.make_leaf(99).expect("leaf");
        let replaced = tree.store_child(tag, 4, new_leaf).expect("replacement");
        assert_eq!(
            replaced.node_type(),
            NodeType::Node64,
            "no resize on overwrite"
        );
        assert_eq!(
            tree.lookup_child(replaced, 4).expect("lookup"),
            Some(new_leaf.to_bits())
        );
        assert_eq!(
            tree.lookup_child(replaced, 1).expect("lookup"),
            Some(children[0])
        );
        assert_eq!(
            tree.lookup_child(replaced, 9).expect("lookup"),
            Some(children[2])
        );
        let extra = tree.make_leaf(77).expect("leaf");
        let grown = tree.store_child(replaced, 3, extra).expect("insert");
        assert_eq!(
            tree.lookup_child(grown, 3).expect("lookup"),
            Some(extra.to_bits())
        );
        assert_eq!(tree.lookup_child(grown, 15).expect("lookup"), None);
        let mut wide = [EMPTY_CHILD; 256];
        wide[1] = tree.make_leaf(11).expect("leaf").to_bits();
        wide[15] = tree.make_leaf(12).expect("leaf").to_bits();
        let direct = tree
            .build_node256(&wide, tree.make_leaf(13).expect("leaf").to_bits(), 9)
            .expect("node256 builds");
        assert_eq!(direct.node_type(), NodeType::Node256);
        assert_eq!(direct.polymorphic_field() & 0xFF, 9, "generation preserved");
        assert_eq!(direct.polymorphic_field() >> 8, 2, "refcount = 2 children");
        assert_eq!(tree.lookup_child(direct, 1).expect("lookup"), Some(wide[1]));
        assert_eq!(
            tree.lookup_child(direct, 15).expect("lookup"),
            Some(wide[15])
        );
        assert_eq!(tree.lookup_child(direct, 14).expect("lookup"), None);
        let fresh = tree.make_leaf(66).expect("leaf");
        let direct2 = tree.store_child(direct, 14, fresh).expect("insert");
        assert_eq!(
            tree.lookup_child(direct2, 14).expect("lookup"),
            Some(fresh.to_bits())
        );
        let direct3 = tree.store_child(direct2, 1, fresh).expect("replacement");
        assert_eq!(
            tree.lookup_child(direct3, 1).expect("lookup"),
            Some(fresh.to_bits())
        );
        assert_eq!(
            tree.lookup_child(direct3, 15).expect("lookup"),
            Some(wide[15])
        );
    }

    /// Verifies the leaf-root paths by forcing the root word to a leaf
    /// tag: lookups, overwrites, deep-insert chain conversion, and deletes
    /// must all honour the empty-key payload.
    #[test]
    fn leaf_root_paths() {
        let tree = test_tree();
        let leaf = TaggedPointer::pack_inline_payload(5, false).to_bits();
        tree.root.store(leaf, Ordering::Release);
        assert_eq!(tree.get(b"").expect("lookup succeeds"), Some(5));
        assert_eq!(tree.get(b"ab").expect("lookup succeeds"), None);
        assert!(!tree.delete(b"ab").expect("delete succeeds"));
        assert_eq!(tree.get(b"").expect("lookup succeeds"), Some(5));
        assert_eq!(tree.insert(b"", 6).expect("overwrite succeeds"), Some(5));
        assert_eq!(tree.get(b"").expect("lookup succeeds"), Some(6));
        tree.root.store(leaf, Ordering::Release);
        assert_eq!(tree.insert(b"ab", 7).expect("insert succeeds"), Some(5));
        assert_eq!(
            tree.get(b"").expect("lookup succeeds"),
            Some(5),
            "empty-key preserved"
        );
        assert_eq!(tree.get(b"ab").expect("lookup succeeds"), Some(7));
        assert!(tree.delete(b"").expect("delete succeeds"));
        assert_eq!(tree.get(b"").expect("lookup succeeds"), None);
        assert!(!tree.delete(b"").expect("delete succeeds"));
    }

    /// Verifies that tombstoned words are treated as absent on every read
    /// path, and that re-insertion replaces the tombstone with a live
    /// chain.
    #[test]
    fn tombstoned_child_is_absent_and_reinsertable() {
        let tree = test_tree();
        tree.insert(b"ab", 1).expect("insert succeeds");
        let root = tree.root_tag();
        let bits = tree.lookup_child(root, 6).expect("child").expect("present");
        let level2 = TaggedPointer::from_bits(bits);
        let bits = tree
            .lookup_child(level2, 1)
            .expect("child")
            .expect("present");
        let level3 = TaggedPointer::from_bits(bits);
        let bits = tree
            .lookup_child(level3, 6)
            .expect("child")
            .expect("present");
        let level4 = TaggedPointer::from_bits(bits);
        let node = tree.node4(level4).expect("node4 readable");
        let tombstone = TaggedPointer::pack_inline_payload(1, true).to_bits();
        node.children()[0].store(tombstone, Ordering::Relaxed);
        assert_eq!(
            tree.get(b"ab").expect("lookup succeeds"),
            None,
            "tombstone invisible"
        );
        assert!(!tree.delete(b"ab").expect("delete succeeds"));
        assert_eq!(tree.insert(b"ab", 9).expect("reinsert succeeds"), None);
        assert_eq!(tree.get(b"ab").expect("lookup succeeds"), Some(9));
    }

    /// Verifies the invariant-violation arms: leaves misused as internal
    /// nodes, internal nodes misused as leaves, and out-of-bounds node
    /// pointers all surface as errors instead of touching memory.
    #[test]
    fn invariant_violations_are_reported() {
        let tree = test_tree();
        tree.insert(b"k", 1).expect("insert succeeds");
        let leaf = TaggedPointer::pack_inline_payload(1, false);
        assert!(matches!(
            tree.lookup_child(leaf, 5),
            Err(crate::error::FlareError::TreeInvariantViolation { .. })
        ));
        assert!(matches!(
            tree.read_leaf(tree.root_tag()),
            Err(crate::error::FlareError::TreeInvariantViolation { .. })
        ));
        assert!(matches!(
            tree.node_leaf(leaf),
            Err(crate::error::FlareError::TreeInvariantViolation { .. })
        ));
        assert!(matches!(
            tree.store_child(leaf, 5, leaf),
            Err(crate::error::FlareError::TreeInvariantViolation { .. })
        ));
        let bogus = TaggedPointer::pack(NodeType::Node4, (1 << 40) - 1, 0, false, 1 << 3);
        assert!(matches!(
            tree.node4(bogus),
            Err(crate::error::FlareError::ArenaBoundsExceeded { .. })
        ));
        assert!(matches!(
            tree.node64(bogus),
            Err(crate::error::FlareError::ArenaBoundsExceeded { .. })
        ));
        assert!(matches!(
            tree.node256(bogus),
            Err(crate::error::FlareError::ArenaBoundsExceeded { .. })
        ));
        assert!(matches!(
            tree.lookup_child(bogus, 3),
            Err(crate::error::FlareError::ArenaBoundsExceeded { .. })
        ));
    }

    /// Verifies that a tombstone root reads as empty on every read path
    /// and that insert replaces it in place without a stale value.
    #[test]
    fn tombstone_root_reads_as_empty_and_replaces() {
        let tree = test_tree();
        tree.insert(b"ab", 1).expect("insert succeeds");
        let tombstone = TaggedPointer::pack_inline_payload(1, true).to_bits();
        tree.root.store(tombstone, Ordering::Release);
        assert_eq!(tree.get(b"ab").expect("lookup succeeds"), None);
        assert_eq!(tree.longest_prefix(b"ab").expect("walk succeeds"), None);
        assert_eq!(
            tree.insert(b"cd", 2).expect("insert succeeds"),
            None,
            "tombstone root is replaced without a stale previous value"
        );
        assert_eq!(tree.get(b"cd").expect("lookup succeeds"), Some(2));
    }

    /// Verifies longest-prefix resolution against a leaf root: the empty
    /// key resolves, any other key misses.
    #[test]
    fn leaf_root_longest_prefix() {
        let tree = test_tree();
        let leaf = TaggedPointer::pack_inline_payload(5, false);
        tree.root.store(leaf.to_bits(), Ordering::Release);
        assert_eq!(
            tree.longest_prefix(b"").expect("walk succeeds"),
            Some((0, 5))
        );
        assert_eq!(tree.longest_prefix(b"ab").expect("walk succeeds"), None);
    }

    /// Verifies that deleting the empty key from a leaf root empties the
    /// tree entirely.
    #[test]
    fn leaf_root_delete_empties_tree() {
        let tree = test_tree();
        let leaf = TaggedPointer::pack_inline_payload(5, false);
        tree.root.store(leaf.to_bits(), Ordering::Release);
        assert!(tree.delete(b"").expect("delete succeeds"));
        assert_eq!(tree.get(b"").expect("lookup succeeds"), None);
        assert!(!tree.delete(b"").expect("delete succeeds"));
    }

    /// Verifies that a tombstone child terminates a longest-prefix walk
    /// with the best match seen so far.
    #[test]
    fn longest_prefix_stops_at_tombstone_child() {
        let tree = test_tree();
        tree.insert(b"ab", 1).expect("insert succeeds");
        tree.insert(b"cb", 2).expect("insert succeeds");
        let root = tree.root_tag();
        let level2_bits = tree.lookup_child(root, 6).expect("child").expect("present");
        let level2 = TaggedPointer::from_bits(level2_bits);
        let node = tree.node4(level2).expect("node4 readable");
        let tombstone = TaggedPointer::pack_inline_payload(1, true).to_bits();
        node.children()[0].store(tombstone, Ordering::Relaxed);
        assert_eq!(
            tree.longest_prefix(b"ab").expect("walk succeeds"),
            None,
            "walk stops at the tombstoned child"
        );
        assert_eq!(
            tree.longest_prefix(b"cb").expect("walk succeeds"),
            Some((2, 2)),
            "sibling branch is unaffected"
        );
    }

    /// Verifies the `Node64` → `Node256` cascade once the bitmap is full.
    ///
    /// A `Node64` with sixteen children covers every nibble, so the growth
    /// branch is only reachable with a synthetic bitmap that keeps one
    /// nibble free while counting sixteen bits.
    #[test]
    fn node64_grows_to_node256_beyond_sixteen_children() {
        let tree = test_tree();
        let mut children = [EMPTY_CHILD; 64];
        let mut bitmap = 0u64;
        for i in 0..15u8 {
            children[usize::from(i)] = tree.make_leaf(200 + u64::from(i)).expect("leaf").to_bits();
            bitmap |= 1u64 << u64::from(i);
        }
        bitmap |= 1u64 << 16;
        let tag = tree
            .build_node64(
                bitmap,
                &children,
                tree.make_leaf(7).expect("leaf").to_bits(),
                3,
            )
            .expect("node64 builds");
        assert_eq!(tag.node_type(), NodeType::Node64);
        let extra = tree.make_leaf(77).expect("leaf");
        let grown = tree.store_child(tag, 15, extra).expect("growth succeeds");
        assert_eq!(grown.node_type(), NodeType::Node256);
        assert_eq!(
            tree.lookup_child(grown, 15).expect("lookup"),
            Some(extra.to_bits())
        );
        assert_eq!(
            tree.lookup_child(grown, 1).expect("lookup"),
            Some(children[1])
        );
    }

    /// Verifies longest-prefix walks read the leaf words of `Node64` and
    /// `Node256` nodes.
    #[test]
    fn prefix_walk_reads_wide_node_leaves() {
        let tree = test_tree();
        let mut children = [EMPTY_CHILD; 64];
        let bitmap = (1u64 << 1) | (1u64 << 4) | (1u64 << 9);
        children[0] = tree.make_leaf(10).expect("leaf").to_bits();
        children[1] = tree.make_leaf(20).expect("leaf").to_bits();
        children[2] = tree.make_leaf(30).expect("leaf").to_bits();
        let tag = tree
            .build_node64(
                bitmap,
                &children,
                tree.make_leaf(7).expect("leaf").to_bits(),
                3,
            )
            .expect("node64 builds");
        tree.root.store(tag.to_bits(), Ordering::Release);
        assert_eq!(
            tree.longest_prefix(&[2]).expect("walk succeeds"),
            Some((0, 7)),
            "node64 leaf word is resolved during the walk"
        );
        let mut wide = [EMPTY_CHILD; 256];
        wide[3] = tree.make_leaf(11).expect("leaf").to_bits();
        let direct = tree
            .build_node256(&wide, tree.make_leaf(13).expect("leaf").to_bits(), 9)
            .expect("node256 builds");
        tree.root.store(direct.to_bits(), Ordering::Release);
        assert_eq!(
            tree.longest_prefix(&[0]).expect("walk succeeds"),
            Some((0, 13)),
            "node256 leaf word is resolved during the walk"
        );
    }
}
