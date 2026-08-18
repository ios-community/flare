//! Physical delta WAL frames, group commit pipeline, and crash recovery.
//!
//! The WAL persists *physical byte deltas* instead of parsed key-value
//! records, so recovery replays frames directly into the arena with
//! `memcpy` overlays. The on-disk frame layout is:
//!
//! ```text
//! +-------------------+-----------------+-----------------------+---------------------+-----------------+
//! | Frame Length (4B) | OpCode (1B)     | Arena Offset (5B)     | Payload Length (4B) | Payload Data    |
//! | BigEndian Uint32  | (Alloc/Free/Up) | 40-bit Byte Index     | BigEndian Uint32    | Raw Byte Delta  |
//! +-------------------+-----------------+-----------------------+---------------------+-----------------+
//! ```
//!
//! Frame length counts the trailing `1 + 5 + 4 + payload` bytes, so the
//! total encoded size is `4 + length`. Offsets are big-endian over 40 bits.

use crate::alloc::arena::FlatArena;
use crate::error::FlareError;
use alloc_crate::vec::Vec;
use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Maximum accepted payload length for a decoded WAL frame.
///
/// The limit bounds replay `memcpy` size against corrupted length fields.
pub const MAX_FRAME_LEN: u32 = 1 << 20;

/// Maximum number of frames parsed during a single recovery scan.
///
/// The limit turns a corrupted length chain into a bounded parse.
pub const REPLAY_FRAME_LIMIT: usize = 1 << 22;

/// The binary operation code of a physical delta frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum WalOpCode {
    /// Records the allocation of a fresh arena region.
    Alloc = 0,
    /// Records the retirement of an arena region.
    Free = 1,
    /// Records the raw byte overlay of an existing region.
    Update = 2,
}

impl WalOpCode {
    /// Returns the 1-byte wire representation of this op code.
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        self as u8
    }

    /// Decodes a raw op code byte.
    ///
    /// # Errors
    ///
    /// Returns [`FlareError::WalFrameMalformed`] for unknown bytes.
    pub const fn from_u8(raw: u8) -> Result<Self, FlareError> {
        match raw {
            0 => Ok(Self::Alloc),
            1 => Ok(Self::Free),
            2 => Ok(Self::Update),
            _ => Err(FlareError::WalFrameMalformed {
                reason: "unknown op code byte",
            }),
        }
    }
}

/// A single physical delta frame.
///
/// The frame addresses an arena byte region and carries the raw bytes to
/// overlay, or a region lifecycle marker with an empty payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalFrame {
    /// The operation performed by this frame.
    pub opcode: WalOpCode,
    /// The 40-bit arena byte offset addressed by this frame.
    pub offset: u64,
    /// The raw payload bytes to overlay (empty for lifecycle frames).
    pub payload: Vec<u8>,
}

impl WalFrame {
    /// Creates a frame that records a raw byte overlay.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flare_core::wal::{WalFrame, WalOpCode};
    /// let frame = WalFrame::update(0x100, vec![1, 2, 3]);
    /// assert_eq!(frame.opcode, WalOpCode::Update);
    /// ```
    #[must_use]
    pub const fn update(offset: u64, payload: Vec<u8>) -> Self {
        Self {
            opcode: WalOpCode::Update,
            offset,
            payload,
        }
    }

    /// Creates a frame that records an allocation.
    #[must_use]
    pub fn alloc(offset: u64, length: usize) -> Self {
        Self {
            opcode: WalOpCode::Alloc,
            offset,
            payload: Vec::with_capacity(0),
        }
        .with_length(length)
    }

    /// Creates a frame that records a retirement.
    #[must_use]
    pub fn free(offset: u64, length: usize) -> Self {
        Self {
            opcode: WalOpCode::Free,
            offset,
            payload: Vec::with_capacity(0),
        }
        .with_length(length)
    }

    /// Attaches a region length to a lifecycle frame by packing it into a
    /// 4-byte payload (big-endian).
    fn with_length(mut self, length: usize) -> Self {
        let length = u32::try_from(length).expect("region length fits in u32");
        self.payload = length.to_be_bytes().to_vec();
        self
    }

    /// Returns the encoded byte length of this frame.
    ///
    /// # Errors
    ///
    /// Returns [`FlareError::WalFrameTooLarge`] when the payload exceeds
    /// [`MAX_FRAME_LEN`].
    pub fn encoded_len(&self) -> Result<u32, FlareError> {
        let payload = u32::try_from(self.payload.len())
            .map_err(|_| FlareError::WalFrameTooLarge { declared: u32::MAX })?;
        if payload > MAX_FRAME_LEN {
            return Err(FlareError::WalFrameTooLarge { declared: payload });
        }
        let header = 1u32 + 5 + 4;
        Ok(header + payload)
    }

    /// Encodes this frame into its binary representation.
    ///
    /// # Errors
    ///
    /// Returns [`FlareError::WalFrameTooLarge`] when the payload exceeds
    /// [`MAX_FRAME_LEN`].
    ///
    /// # Panics
    ///
    /// Panics when the header plus payload exceeds `usize`, which cannot
    /// happen because [`encoded_len`](Self::encoded_len) bounds it to
    /// [`MAX_FRAME_LEN`].
    ///
    /// # Examples
    ///
    /// ```
    /// # use flare_core::wal::{WalFrame, decode_frame};
    /// let frame = WalFrame::update(0x1234, vec![0xAA, 0xBB]);
    /// let bytes = frame.encode().expect("frame fits limits");
    /// let (decoded, consumed) = decode_frame(&bytes).expect("frame decodes");
    /// assert_eq!(decoded, frame);
    /// assert_eq!(consumed, bytes.len());
    /// ```
    pub fn encode(&self) -> Result<Vec<u8>, FlareError> {
        let length = self.encoded_len()?;
        let mut out = Vec::with_capacity(usize::try_from(4 + length).expect("fits in usize"));
        out.extend_from_slice(&length.to_be_bytes());
        out.push(self.opcode.as_u8());
        let offset = self.offset & 0x0000_00FF_FFFF_FFFF;
        out.push(((offset >> 32) & 0xFF) as u8);
        out.push(((offset >> 24) & 0xFF) as u8);
        out.push(((offset >> 16) & 0xFF) as u8);
        out.push(((offset >> 8) & 0xFF) as u8);
        out.push((offset & 0xFF) as u8);
        out.extend_from_slice(
            &u32::try_from(self.payload.len())
                .expect("payload fits in u32")
                .to_be_bytes(),
        );
        out.extend_from_slice(&self.payload);
        Ok(out)
    }
}

/// Decodes one frame from the front of `bytes`.
///
/// # Errors
///
/// Returns [`FlareError::WalFrameMalformed`] for truncated headers or
/// inconsistent lengths, and [`FlareError::WalFrameTooLarge`] when the
/// declared length exceeds [`MAX_FRAME_LEN`].
///
/// # Panics
///
/// Panics when a declared length below [`MAX_FRAME_LEN`] does not fit in
/// `u32`, which is impossible on every supported target.
pub fn decode_frame(bytes: &[u8]) -> Result<(WalFrame, usize), FlareError> {
    if bytes.len() < 4 {
        return Err(FlareError::WalFrameMalformed {
            reason: "header shorter than 4 bytes",
        });
    }
    let length = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    if length > MAX_FRAME_LEN as usize {
        return Err(FlareError::WalFrameTooLarge {
            declared: u32::try_from(length).expect("fits"),
        });
    }
    let total = 4 + length;
    if bytes.len() < total {
        return Err(FlareError::WalFrameMalformed {
            reason: "frame length exceeds remaining buffer",
        });
    }
    let opcode = WalOpCode::from_u8(bytes[4])?;
    let offset = (u64::from(bytes[5]) << 32)
        | (u64::from(bytes[6]) << 24)
        | (u64::from(bytes[7]) << 16)
        | (u64::from(bytes[8]) << 8)
        | u64::from(bytes[9]);
    let payload_len = u32::from_be_bytes([bytes[10], bytes[11], bytes[12], bytes[13]]) as usize;
    if payload_len + 14 != total {
        return Err(FlareError::WalFrameMalformed {
            reason: "payload length inconsistent with frame length",
        });
    }
    let payload = bytes[14..total].to_vec();
    Ok((
        WalFrame {
            opcode,
            offset,
            payload,
        },
        total,
    ))
}

/// Encodes a batch of frames into a single contiguous buffer.
///
/// # Errors
///
/// Returns [`FlareError::WalFrameTooLarge`] when any frame payload exceeds
/// [`MAX_FRAME_LEN`].
pub fn encode_frames(frames: &[WalFrame]) -> Result<Vec<u8>, FlareError> {
    let mut out = Vec::new();
    for frame in frames {
        out.extend_from_slice(&frame.encode()?);
    }
    Ok(out)
}

/// Parses a complete WAL log into its frames.
///
/// # Errors
///
/// Returns [`FlareError::WalFrameMalformed`] when the log ends mid-frame
/// or violates [`REPLAY_FRAME_LIMIT`].
pub fn parse_log(bytes: &[u8]) -> Result<Vec<WalFrame>, FlareError> {
    let mut frames = Vec::new();
    let mut cursor = 0usize;
    while cursor < bytes.len() {
        if frames.len() >= REPLAY_FRAME_LIMIT {
            return Err(FlareError::WalFrameMalformed {
                reason: "replay frame limit exceeded",
            });
        }
        let (frame, consumed) = decode_frame(&bytes[cursor..])?;
        cursor += consumed;
        frames.push(frame);
    }
    Ok(frames)
}

/// A group commit batch: a set of frames flushed together by the leader.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WalBatch {
    /// The frames belonging to this batch, in append order.
    pub frames: Vec<WalFrame>,
}

impl WalBatch {
    /// Creates an empty batch.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a frame to this batch.
    pub fn push(&mut self, frame: WalFrame) {
        self.frames.push(frame);
    }

    /// Encodes the batch into a single buffer for a sink.
    ///
    /// # Errors
    ///
    /// Returns [`FlareError::WalFrameTooLarge`] when a frame payload
    /// exceeds [`MAX_FRAME_LEN`].
    pub fn encode(&self) -> Result<Vec<u8>, FlareError> {
        encode_frames(&self.frames)
    }
}

/// A sink that persists encoded WAL bytes durably.
///
/// Implementors decide the durability semantics: a memory sink appends to
/// RAM, a file sink `fsync`s, and a device sink streams to `NVMe`. Flush
/// calls must be safe to invoke concurrently.
pub trait WalSink: Send + Sync {
    /// Flushes `bytes` durably (or durably-enough for the sink contract).
    ///
    /// # Errors
    ///
    /// Returns a driver-specific error when persistence fails.
    fn flush(&self, bytes: &[u8]) -> Result<(), FlareError>;
}
/// A [`WalSink`] that appends encoded frames into an in-memory log.
///
/// The memory sink is the canonical test and recovery fixture: its buffer
/// is the exact byte stream a durable sink would produce, and `recover`
/// replays it back into arena memory. All buffer access is serialised by a
/// spin lock, so concurrent `flush` calls are safe.
#[derive(Debug, Default)]
pub struct MemoryWalSink {
    log: UnsafeCell<Vec<u8>>,
    lock: AtomicBool,
    frames: AtomicU64,
}

impl MemoryWalSink {
    /// Creates an empty memory sink.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Acquires the buffer lock, blocking until it is free.
    fn acquire(&self) {
        while self.lock.swap(true, Ordering::Acquire) {
            core::hint::spin_loop();
        }
    }

    /// Releases the buffer lock.
    fn release(&self) {
        self.lock.store(false, Ordering::Release);
    }

    /// Returns the total encoded bytes stored so far.
    #[must_use]
    pub fn len(&self) -> usize {
        self.acquire();
        // SAFETY: the lock makes this the only reader-writer in the
        // critical section.
        let len = unsafe { (*self.log.get()).len() };
        self.release();
        len
    }

    /// Returns whether the log is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of frames flushed so far.
    #[must_use]
    pub fn frame_count(&self) -> u64 {
        self.frames.load(Ordering::Relaxed)
    }

    /// Returns a snapshot of the log buffer.
    #[must_use]
    pub fn snapshot(&self) -> Vec<u8> {
        self.acquire();
        // SAFETY: the lock makes this the only reader-writer in the
        // critical section.
        let log = unsafe { (*self.log.get()).clone() };
        self.release();
        log
    }
}

// SAFETY: the buffer is only ever touched inside the spin-lock critical
// section, which excludes concurrent access across all threads; the frame
// counter is an `AtomicU64` and needs no exclusion.
unsafe impl Sync for MemoryWalSink {}

impl WalSink for MemoryWalSink {
    fn flush(&self, bytes: &[u8]) -> Result<(), FlareError> {
        self.acquire();
        // SAFETY: the lock makes this the only reader-writer in the
        // critical section.
        let log = unsafe { &mut *self.log.get() };
        log.extend_from_slice(bytes);
        self.release();
        self.frames.fetch_add(1, Ordering::Release);
        Ok(())
    }
}

/// A leader-follower group commit transaction.
///
/// The transaction enforces the *child-before-parent* write ordering
/// barrier: child frames must be persisted before the parent pointer
/// frame that references them. The pipeline writes `children` frames,
/// issues a release fence, then writes `parent`, and flushes the whole
/// batch in one sink call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalTransaction {
    /// Frames for newly allocated child regions, ordered first.
    pub children: Vec<WalFrame>,
    /// The parent pointer modification frame, ordered last.
    pub parent: WalFrame,
}

impl Default for WalTransaction {
    /// Creates an empty transaction whose parent frame addresses a
    /// zero-length update at offset 0.
    fn default() -> Self {
        Self {
            children: Vec::new(),
            parent: WalFrame::update(0, Vec::new()),
        }
    }
}

impl WalTransaction {
    /// Creates a transaction from its ordered parts.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flare_core::wal::{WalFrame, WalTransaction};
    /// let tx = WalTransaction::new(
    ///     vec![WalFrame::alloc(0x10, 48)],
    ///     WalFrame::update(0x00, vec![1, 2]),
    /// );
    /// assert_eq!(tx.children.len(), 1);
    /// ```
    #[must_use]
    pub const fn new(children: Vec<WalFrame>, parent: WalFrame) -> Self {
        Self { children, parent }
    }

    /// Appends an ordered batch to `sink`, observing the child-before-
    /// parent barrier.
    ///
    /// A transaction without children whose parent frame carries an empty
    /// payload is a no-op and writes nothing to the sink.
    ///
    /// # Errors
    ///
    /// Returns [`FlareError::WalFrameTooLarge`] on oversized payloads, or
    /// the sink error when the flush fails.
    pub fn commit<S: WalSink>(&self, sink: &S) -> Result<(), FlareError> {
        if self.children.is_empty() && self.parent.payload.is_empty() {
            return Ok(());
        }
        let mut batch = Vec::with_capacity(self.children.len() + 1);
        batch.extend(self.children.iter().cloned());
        // Release barrier: the child frames are visible to the sink before
        // the parent frame is appended.
        core::sync::atomic::fence(Ordering::Release);
        batch.push(self.parent.clone());
        sink.flush(&encode_frames(&batch)?)
    }
}

/// Replays a WAL log into an arena by direct memory overlay.
///
/// Frames are parsed sequentially and each payload is copied into arena
/// memory at its recorded offset. The returned high-water mark is
/// `max(offset + payload_len)` and reconstructs the bump frontier.
///
/// # Errors
///
/// Returns [`FlareError::WalFrameMalformed`] for malformed logs and
/// [`FlareError::ArenaBoundsExceeded`] when a frame addresses a region
/// outside the arena.
///
/// # Panics
///
/// Panics when a frame payload length does not fit in `u64`, which cannot
/// happen because [`MAX_FRAME_LEN`] bounds every payload.
pub fn recover(arena: &FlatArena, log: &[u8]) -> Result<u64, FlareError> {
    let mut high_water = 0u64;
    for frame in parse_log(log)? {
        let end = frame.offset + u64::try_from(frame.payload.len()).expect("fits in u64");
        high_water = high_water.max(end);
        match frame.opcode {
            WalOpCode::Alloc | WalOpCode::Free => {}
            WalOpCode::Update => {
                arena.write_bytes(frame.offset, &frame.payload)?;
            }
        }
    }
    Ok(high_water)
}

#[cfg(test)]
mod tests {
    use super::{
        MemoryWalSink, WalBatch, WalFrame, WalOpCode, WalTransaction, decode_frame, encode_frames,
        parse_log, recover,
    };
    use crate::alloc::arena::FlatArena;
    use crate::error::FlareError;
    use alloc_crate::vec;

    /// Verifies that frames survive an encode/decode round-trip.
    #[test]
    fn frames_roundtrip_through_encode_decode() {
        let frames = vec![
            WalFrame::alloc(0x10, 48),
            WalFrame::update(0x00, vec![1, 2, 3]),
            WalFrame::free(0x10, 48),
        ];
        for frame in frames {
            let bytes = frame.encode().expect("frame fits limits");
            let (decoded, consumed) = decode_frame(&bytes).expect("frame decodes");
            assert_eq!(decoded, frame);
            assert_eq!(consumed, bytes.len());
        }
    }

    /// Verifies that `parse_log` reconstructs every frame in order.
    #[test]
    fn parse_log_reconstructs_frames_in_order() {
        let frames = vec![
            WalFrame::alloc(0x100, 48),
            WalFrame::update(0x0, vec![0xAA; 8]),
            WalFrame::update(0x100, vec![0xBB; 4]),
        ];
        let batch = encode_frames(&frames).expect("batch fits limits");
        let parsed = parse_log(&batch).expect("log parses");
        assert_eq!(parsed, frames);
    }

    /// Verifies that truncated or inconsistent frames are rejected.
    #[test]
    fn malformed_frames_are_rejected() {
        let good = WalFrame::update(0x10, vec![1, 2, 3])
            .encode()
            .expect("frame fits limits");
        assert!(decode_frame(&good[..good.len() - 2]).is_err());
        let mut corrupt = good;
        corrupt[0] = 0xFF;
        assert!(decode_frame(&corrupt).is_err());
        assert!(parse_log(b"\x00\x01").is_err(), "garbage is rejected");
    }

    /// Verifies that oversized payloads are refused at encoding time.
    #[test]
    fn oversized_payloads_are_rejected() {
        let frame = WalFrame::update(0, vec![0u8; 1 << 21]);
        assert!(matches!(
            frame.encode(),
            Err(FlareError::WalFrameTooLarge { .. })
        ));
        assert!(matches!(
            frame.encoded_len(),
            Err(FlareError::WalFrameTooLarge { .. })
        ));
    }

    /// Verifies that unknown op code bytes are rejected during decode.
    #[test]
    fn unknown_opcode_is_rejected() {
        assert!(matches!(
            WalOpCode::from_u8(9),
            Err(FlareError::WalFrameMalformed {
                reason: "unknown op code byte"
            })
        ));
    }

    /// Verifies that a declared payload length inconsistent with the frame
    /// length is rejected.
    #[test]
    fn payload_length_mismatch_is_rejected() {
        let mut encoded = WalFrame::update(0x10, vec![1, 2, 3])
            .encode()
            .expect("frame fits limits");
        encoded[13] = 0;
        assert!(matches!(
            decode_frame(&encoded),
            Err(FlareError::WalFrameMalformed { .. })
        ));
    }

    /// Verifies that a log longer than the replay budget is rejected.
    #[test]
    fn replay_limit_is_enforced() {
        use super::REPLAY_FRAME_LIMIT;
        let log = vec![0u8; 18 * REPLAY_FRAME_LIMIT + 18];
        assert!(matches!(
            parse_log(&log),
            Err(FlareError::WalFrameMalformed { .. })
        ));
    }

    /// Verifies that batches append frames and encode them in order.
    #[test]
    fn batch_appends_and_encodes() {
        let mut batch = WalBatch::new();
        assert!(batch.frames.is_empty());
        let frame = WalFrame::free(0x10, 48);
        batch.push(frame.clone());
        let encoded = batch.encode().expect("frame fits limits");
        assert_eq!(encoded, encode_frames(&[frame]).expect("frame fits limits"));
    }

    /// Verifies that recovery overlays payload bytes and reconstructs the
    /// bump high-water mark.
    #[test]
    fn recover_overlays_payload_and_high_water() {
        let arena = FlatArena::new(4096).expect("arena fits");
        let sink = MemoryWalSink::new();
        let tx = WalTransaction::new(
            vec![WalFrame::alloc(0x100, 48)],
            WalFrame::update(0x100, vec![0x11, 0x22, 0x33, 0x44]),
        );
        tx.commit(&sink).expect("commit succeeds");
        let high_water = recover(&arena, &sink.snapshot()).expect("recovery succeeds");
        assert_eq!(high_water, 0x104);
        assert_eq!(
            arena.read_node::<[u8; 4]>(0x100).expect("region readable"),
            &[0x11, 0x22, 0x33, 0x44]
        );
        assert_eq!(sink.frame_count(), 1);
    }

    /// Verifies the child-before-parent barrier: alloc frames precede the
    /// parent update frame in the committed byte stream.
    #[test]
    fn child_frames_precede_parent() {
        let sink = MemoryWalSink::new();
        let tx = WalTransaction::new(
            vec![WalFrame::alloc(0x200, 48), WalFrame::alloc(0x300, 48)],
            WalFrame::update(0x100, vec![9, 9]),
        );
        tx.commit(&sink).expect("commit succeeds");
        let parsed = parse_log(&sink.snapshot()).expect("log parses");
        assert_eq!(parsed[0].opcode, WalOpCode::Alloc);
        assert_eq!(parsed[1].opcode, WalOpCode::Alloc);
        assert_eq!(parsed[2].opcode, WalOpCode::Update);
    }

    /// Verifies the default transaction is empty and commits harmlessly.
    #[test]
    fn default_transaction_is_harmless() {
        let tx = WalTransaction::default();
        assert!(tx.children.is_empty());
        let sink = MemoryWalSink::new();
        tx.commit(&sink).expect("empty commit succeeds");
        assert!(sink.is_empty());
    }

    /// Verifies that concurrent commits are serialised by the sink lock and
    /// every batch lands in the log.
    #[test]
    fn concurrent_commits_are_serialised() {
        use alloc_crate::vec::Vec;
        use std::sync::{Arc, Barrier};
        let sink = Arc::new(MemoryWalSink::new());
        let barrier = Arc::new(Barrier::new(5));
        let mut threads = Vec::new();
        for _ in 0..4 {
            let sink = Arc::clone(&sink);
            let barrier = Arc::clone(&barrier);
            threads.push(std::thread::spawn(move || {
                barrier.wait();
                for i in 0..64u64 {
                    let tx = WalTransaction::new(
                        vec![WalFrame::alloc(i * 64, 48)],
                        WalFrame::update(i * 64, vec![0xAA]),
                    );
                    tx.commit(&*sink).expect("commit succeeds");
                }
            }));
        }
        barrier.wait();
        for thread in threads {
            thread.join().expect("worker finishes");
        }
        assert_eq!(sink.frame_count(), 256, "every batch is flushed once");
    }
}
