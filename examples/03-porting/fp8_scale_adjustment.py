#!/usr/bin/env python3
"""Example: the fp8 weight-loading wrapper, and the one bug class that is unrecoverable.

**Status: annotated sketch. Not executed as part of the manual. No HPU was used to produce it.**
It is written to show the *shape* of a correct wrapper and to mark the hazard precisely. Read
chapter 3 §3.4 before using it.

WHY THIS FILE EXISTS
--------------------
Gaudi 2's native fp8 has a maximum finite value of **240**. Checkpoints ship OCP
`float8_e4m3fn`, whose maximum finite value is **448**. The two formats agree on every
`|v| <= 240` and disagree on exactly the top bin — 14 codes (unsigned-byte 120-126 and
248-254). Feeding raw out-of-range `e4m3fn` weights to the device decoder produces NaN/Inf
for those values.

**The device's fp8 is `e4m3fn` CODES (bias 7) truncated at 240 — it is NOT `e4m3fnuz`.**
`float8_e4m3fnuz` is not a supported dtype on HPU at all, and the device's own encoder emits
bias-7 codes (1.0 -> 0x38, 240 -> 0x77). `float8_e4m3fnuz` contributes exactly one useful
number to this stack: the ceiling 240. Do not reason from the fnuz *name* — under bias 8 you
would compute the poison set to be the OPPOSITE bytes. (Chapter 3 §3.4.1.)

The standard conversion multiplies fp8 weights by 0.5 and fp32 scales by 2.0. The real
(dequantized) value is preserved while the fp8 bit pattern fits the device's range.

THE HAZARD, STATED EXACTLY
--------------------------
The halving is **safe by LAYOUT, not by design**. It works only because it happens on a
**host / checkpoint tensor**, before the value ever meets the device decoder.

    On a DEVICE tensor the same expression is a NO-OP that PRESERVES THE POISON.

And the poison is **UNRECOVERABLE**:
  * `inf * 0.5 == inf`, `nan * 0.5 == nan`;
  * the post-decode map is many-to-one, with nothing downstream inverting it;
  * `0x78` is a FIXED POINT of the re-encode, and `0x79-0x7E` all collapse to one NaN code.

So the guard below is **structural**: it asserts the multiply happened on a host tensor. It is
NOT a scan for bad code values, because a scan is data-dependent, ambiguous, and detect-only.

TAGS
----
MEASURED / ASSUMED / INFERRED / UNKNOWN, per the manual's convention.
"""

from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# Configuration the wrapper needs. In the source project this is read from the
# library's own global config so that one flag governs the whole load path.
# ---------------------------------------------------------------------------
SCALE_ADJUSTMENT_ENABLED = True   # the library's `scale_adjustment` flag, default on

# Maximum finite value of the device's native fp8 format.
# MEASURED: 240 on Gaudi 2 (unsigned-byte code 0x77). The checkpoint format's maximum is 448.
DEVICE_FP8_MAX = 240.0
CHECKPOINT_FP8_MAX = 448.0

# Codes that disagree between the checkpoint format and the device format.
# MEASURED: unsigned-byte 120-126 -> +inf/NaN, 248-254 -> -inf/NaN on decode.
DISPUTED_CODES = tuple(range(120, 127)) + tuple(range(248, 255))


class ScaleAdjustmentError(RuntimeError):
    """Raised when the fp8 conversion would be applied where it is a no-op."""


def _assert_host_tensor(t: torch.Tensor, what: str) -> None:
    """Structural guard: the value must never have met the device decoder yet.

    This is the property that actually prevents the hazard. A code-value scan cannot
    substitute for it: a tensor whose amax < 256 passes a scan even on the broken path,
    so the scan's verdict moves with the data while this property does not.
    """
    if t.device.type != "cpu":
        raise ScaleAdjustmentError(
            f"{what}: expected a host/checkpoint tensor, got device={t.device!r}. "
            f"The x0.5 conversion is a NO-OP on a device tensor and preserves the poison, "
            f"which is unrecoverable. Move the conversion to the host-side load path."
        )
    # A meta tensor carries no data and cannot be poisoned, but it also cannot be converted.
    if t.is_meta:
        raise ScaleAdjustmentError(f"{what}: meta tensor has no data to convert.")


def decode_e8m0_bytes_to_scale(raw: torch.Tensor) -> torch.Tensor:
    """Decode raw e8m0 exponent BYTES into a real float scale.

    THE BUG THIS FIXES (MEASURED, and it cost real time)
    ----------------------------------------------------
    Some MoE checkpoints store the e8m0 expert scales as raw `uint8` bytes rather than as a
    real float, to preserve them for a vendor-specific execution path. Those bytes then fall
    into the `else` branch of the conversion below -- the `* 2.0` meant for fp32 scales -- and
    are silently corrupted (byte 115 -> 230).

    The fix is to decode first, so the `* 2.0` then applies to a real scale and is correct.

    Do NOT work around this by re-adding CPU dequant/requant passes, and do NOT force the MoE
    onto a different execution path. The native path works once the bytes are decoded.
    """
    if raw.dtype != torch.uint8:
        return raw
    # e8m0: value = 2 ** (byte - 127). Exponent-only, so it is an exact power of two.
    return torch.pow(2.0, raw.float() - 127.0)


def scale_adjust_fp8_weight(loaded_weight: torch.Tensor) -> torch.Tensor:
    """The x0.5 fp8 weight conversion, with the structural guard.

    Must run on a HOST/checkpoint tensor. See the module docstring.
    """
    _assert_host_tensor(loaded_weight, "fp8 weight scale adjustment")
    # Halving maps codes 120-126 -> 0x70-0x76 and 248-254 -> 0xF0-0xF6, so no FINITE value
    # lands in the broken top bin. The maximum representable code becomes ~224 <= 240.
    halved = loaded_weight.float() * 0.5
    return halved.to(torch.float8_e4m3fn)


def scale_adjust_fp32_scale(loaded_scale: torch.Tensor) -> torch.Tensor:
    """The x2.0 scale conversion. Runs on host; the e8m0 byte case is decoded first."""
    _assert_host_tensor(loaded_scale, "fp32 scale adjustment")
    decoded = decode_e8m0_bytes_to_scale(loaded_scale)
    return decoded * 2.0


def wrap_weight_loader(loaded_weight: torch.Tensor) -> torch.Tensor:
    """The full conversion, in the order that is safe.

    Order matters and is not interchangeable:

        1. decode raw e8m0 BYTES into a real scale        (only if dtype is uint8)
        2. fp8 weights: x0.5 on the HOST tensor
        3. fp32 scales: x2.0 on the HOST tensor

    Doing (2) on a device tensor is the unrecoverable case. Doing (1) after (3) corrupts the
    expert scales. See chapter 3 sections 3.4.3 and 3.4.4.
    """
    if not SCALE_ADJUSTMENT_ENABLED:
        return loaded_weight

    if loaded_weight.dtype == torch.float8_e4m3fn:
        return scale_adjust_fp8_weight(loaded_weight)

    # Everything else is treated as a scale. NOTE: this branch is exactly where the raw-uint8
    # e8m0 expert scales used to get corrupted -- decode before scaling.
    return scale_adjust_fp32_scale(loaded_weight)


# ---------------------------------------------------------------------------
# A cheap, honest checker you can run on the LOADED state (host side, no device needed).
# ---------------------------------------------------------------------------
def check_loaded_fp8_range(weight: torch.Tensor, name: str) -> dict:
    """Report whether a loaded fp8 weight tensor can be decoded without overflow.

    This is a DIAGNOSTIC, not the guard. It is reported here so a port can see the state of
    its loaded weights, but it must not be the thing that prevents the hazard: it is
    data-dependent (a tensor whose amax < 256 passes even on a broken path), ambiguous
    (0x7F is a legitimate e4m3fn NaN; 0x78 a legitimate checkpoint code), and detect-only
    (the poison is unrecoverable, so detecting it is too late).

    MEASURED basis: the device's fp8 value space has a maximum finite value of 240.
    """
    if weight.dtype != torch.float8_e4m3fn:
        return {"name": name, "applicable": False, "dtype": str(weight.dtype)}

    # Reinterpret the fp8 payload as unsigned bytes so we can inspect the raw codes.
    codes = weight.view(torch.uint8)
    disputed = 0
    for c in DISPUTED_CODES:
        disputed += int((codes == c).sum())

    # Decode on the host with fp32 to find the true magnitude range.
    as_float = weight.float()
    finite = as_float[torch.isfinite(as_float)]
    amax = float(finite.abs().max()) if finite.numel() else float("nan")

    return {
        "name": name,
        "applicable": True,
        "amax": amax,
        "exceeds_device_max": amax > DEVICE_FP8_MAX,
        "disputed_code_count": disputed,
        "verdict": (
            "OK (fits the device range)"
            if amax <= DEVICE_FP8_MAX and disputed == 0
            else "CHECK: values or codes outside the device's representable range"
        ),
    }


if __name__ == "__main__":
    # Self-demo on the HOST only. No device, no HPU.
    print("fp8 scale-adjustment demo (host only, no HPU)\n")

    # 1. A well-formed conversion: halving keeps the value inside the device range.
    w = (torch.randn(64, 64) * 100).to(torch.float8_e4m3fn)
    adjusted = wrap_weight_loader(w)
    print("adjusted dtype:", adjusted.dtype, "shape:", tuple(adjusted.shape))
    print(check_loaded_fp8_range(adjusted, "demo_weight"))

    # 2. The hazard, demonstrated: the same expression on a "device" tensor is refused.
    class _FakeDeviceTensor(torch.Tensor):
        # Stands in for a tensor that has already met the device decoder. We only need
        # `.device` to differ so the structural guard fires.
        @property
        def device(self):  # type: ignore[override]
            return torch.device("hpu:0")

    try:
        scale_adjust_fp8_weight(_FakeDeviceTensor(w))
    except ScaleAdjustmentError as e:
        print("\nstructural guard fired as intended:")
        print("  ", e)

    # 3. The e8m0 raw-uint8 case.
    raw = torch.tensor([127, 128, 126, 115], dtype=torch.uint8)
    print("\nraw e8m0 bytes :", raw.tolist())
    print("decoded scales :", decode_e8m0_bytes_to_scale(raw).tolist())
    print("after x2.0     :", (decode_e8m0_bytes_to_scale(raw) * 2.0).tolist())
    print("\n(without the decode, byte 115 would have become 230 -- a corrupted scale.)")
