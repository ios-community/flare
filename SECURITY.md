# Security Policy

## Supported Versions

We actively maintain and patch security vulnerabilities for the following versions of FLARE:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take the security of our users and their data seriously. If you discover a security vulnerability in this project, please report it to us privately.

**Do not open a public GitHub issue for security vulnerabilities.**

Instead, please report vulnerabilities by emailing **dzulkiflianwar2@gmail.com**.

### What to Include in Your Report

To help us address the issue quickly, please include:
- A detailed description of the vulnerability.
- Steps to reproduce the issue (including a minimal working example or proof-of-concept code if possible).
- The potential impact of the vulnerability (e.g., memory safety, data corruption, or denial of service).

### Areas of Interest

Given the lock-free, arena-resident design of FLARE, we are especially interested in reports involving:
- Undefined Behavior (UB) or data races in the atomic CAS concurrency paths.
- Memory unsafety in the internal `unsafe` memory primitives (`alloc`, `ptr`, `sync`, `wal`).
- Out-of-bounds arena access or slab slot recycling bugs (use-after-free / ABA).
- WAL replay or crash-recovery correctness.
- CUDA dynamic symbol loading in `CudaSyncDriver`.

### Our Process

1. **Acknowledgement** \
We will acknowledge receipt of your report within 48 hours.
2. **Investigation** \
We will investigate the issue and determine its severity.
3. **Fix** \
We will develop a patch to address the vulnerability.
4. **Disclosure** \
We will coordinate a public disclosure date with you to ensure users have time to update before details are made public.