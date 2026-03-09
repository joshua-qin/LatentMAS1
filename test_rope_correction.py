#!/usr/bin/env python3
"""
Quick experiment to verify our ROPE unrotate/rotate correction is mathematically correct.

Tests:
1. Round-trip: apply RoPE -> unapply -> apply again should recover original (within dtype tolerance).
2. Two-segment fuse: two caches built with positions 0..L1-1 and 0..L2-1; after unrotate, concat, rotate
   with 0..(L1+L2)-1, the second segment should equal "rotate(segment2_raw, position_start=L1)".
"""

import torch

# RoPE helpers from models (same as used in hierarchical_fixed)
from models import (
    _rope_cos_sin,
    _apply_rotary,
    _unapply_rotary,
)

# Qwen3-4B-like config (no model load needed)
ROPE_CONFIG = {"rope_theta": 1000000.0, "head_dim": 128, "num_heads": 32}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


def test_roundtrip():
    """Unapply_rotary should be the exact inverse of apply_rotary for the same cos/sin."""
    seq_len = 64
    head_dim = ROPE_CONFIG["head_dim"]
    # key shape [batch, num_heads, seq_len, head_dim]
    key = torch.randn(1, 8, seq_len, head_dim, device=DEVICE, dtype=DTYPE)
    cos, sin = _rope_cos_sin(key, 0, ROPE_CONFIG, DEVICE, DTYPE)
    key_rot = _apply_rotary(key, cos, sin)
    key_recovered = _unapply_rotary(key_rot, cos, sin)
    err = (key - key_recovered).abs().max().item()
    assert err < 1e-5, f"Round-trip error too large: max |diff| = {err}"
    print(f"[PASS] Round-trip: max |K - unapply(apply(K))| = {err:.2e}")


def test_two_segment_fuse():
    """
    Segment1: keys built with positions 0..L1-1.
    Segment2: keys built with positions 0..L2-1.
    After unrotate(both, 0), concat, rotate(0..L1+L2-1):
    - Fused[0:L1] should equal segment1_rotated (positions 0..L1-1).
    - Fused[L1:L1+L2] should equal rotate(segment2_raw, position_start=L1).
    """
    L1, L2 = 40, 30
    head_dim = ROPE_CONFIG["head_dim"]
    # Raw keys (what each "agent" would have before RoPE in their own run)
    K1_raw = torch.randn(1, 8, L1, head_dim, device=DEVICE, dtype=DTYPE)
    K2_raw = torch.randn(1, 8, L2, head_dim, device=DEVICE, dtype=DTYPE)

    # Simulate what the model does: each segment gets RoPE 0..L-1
    cos1, sin1 = _rope_cos_sin(K1_raw, 0, ROPE_CONFIG, DEVICE, DTYPE)
    cos2, sin2 = _rope_cos_sin(K2_raw, 0, ROPE_CONFIG, DEVICE, DTYPE)
    K1_rot = _apply_rotary(K1_raw, cos1, sin1)
    K2_rot = _apply_rotary(K2_raw, cos2, sin2)

    # Our pipeline: unrotate each (with position_start=0), fuse, rotate with 0..L1+L2-1
    K1_unrot = _unapply_rotary(K1_rot, cos1, sin1)
    K2_unrot = _unapply_rotary(K2_rot, cos2, sin2)
    fused_raw = torch.cat([K1_unrot, K2_unrot], dim=-2)  # [1, 8, L1+L2, head_dim]
    cos_fuse, sin_fuse = _rope_cos_sin(fused_raw, 0, ROPE_CONFIG, DEVICE, DTYPE)
    fused_rot = _apply_rotary(fused_raw, cos_fuse, sin_fuse)

    # Check segment 1: fused_rot[0:L1] should equal K1_rot
    err1 = (fused_rot[:, :, :L1, :] - K1_rot).abs().max().item()
    assert err1 < 1e-5, f"Segment 1 mismatch: max |diff| = {err1}"

    # Check segment 2: fused_rot[L1:L1+L2] should equal rotate(K2_raw, position_start=L1)
    cos2_global, sin2_global = _rope_cos_sin(K2_raw, L1, ROPE_CONFIG, DEVICE, DTYPE)
    K2_expected = _apply_rotary(K2_raw, cos2_global, sin2_global)
    err2 = (fused_rot[:, :, L1 : L1 + L2, :] - K2_expected).abs().max().item()
    assert err2 < 1e-5, f"Segment 2 mismatch: max |diff| = {err2}"

    print(f"[PASS] Two-segment fuse: segment1 err = {err1:.2e}, segment2 err = {err2:.2e}")


def main():
    print("Testing ROPE correction (unrotate -> fuse -> rotate)...")
    print("Config:", ROPE_CONFIG)
    test_roundtrip()
    test_two_segment_fuse()
    print("All checks passed: ROPE correction math is correct.")


if __name__ == "__main__":
    main()
