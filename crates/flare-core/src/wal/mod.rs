//! Physical delta write-ahead logging.
//!
//! The [`delta`] module implements the binary frame format, the
//! child-before-parent ordering barrier, the leader-follower group commit
//! queue, and the replay overlay that restores arena memory after a crash.
//!
//! # Unsafe Confinement
//!
//! `unsafe` appears only inside [`MemoryWalSink`], where the frame buffer
//! is reached through a single-owner `UnsafeCell`; the ownership proof is
//! documented in that type's invariant section.
#![allow(unsafe_code)]

pub mod delta;

pub use delta::{
    MAX_FRAME_LEN, MemoryWalSink, REPLAY_FRAME_LIMIT, WalBatch, WalFrame, WalOpCode, WalSink,
    WalTransaction, decode_frame, encode_frames, parse_log, recover,
};
