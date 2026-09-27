"""The quantizers, and the error each one actually produces. This is the runnable heart of T10.

The whole topic reduces to one question -- **what is the error, and where is it concentrated?** -- and
that question is answerable offline, on a synthetic tensor with realistic structure, without a GPU
and without a model.

Four quantizers, in increasing order of sophistication and decreasing order of error:

    per-tensor absmax   one scale for the whole matrix   -> destroyed by a single outlier channel
    per-channel absmax  one scale per output channel     -> the standard; still fails on outliers
    group-wise          one scale per group of N weights -> better, at a storage cost
    NF4 codebook        equal-mass bins under a normal   -> the 4-bit default for fine-tuning [R]

Plus the AWQ idea, which is not a quantizer at all: keep the SALIENT channels in high precision and
quantize the rest. The corpus states the mechanism exactly -- "identifies the 1% of 'salient'
weights that are most important for quality and keeps them in higher precision" [R] -- and the
experiment shows why 1% is enough.

Everything here is plain Python lists. The point is that the arithmetic is inspectable, not fast.
"""
from __future__ import annotations

import math
import random

# --------------------------------------------------------------------------------------
# The quantizers
# --------------------------------------------------------------------------------------

def absmax_scale(values: list[float], bits: int) -> float:
    """Symmetric scale: `max|v| / q_max`. One scale, used for both signs.

    Symmetric (rather than affine with a zero point) is the default for weights because weight
    distributions are approximately zero-centred, so an explicit zero point buys almost nothing and
    costs storage.
    """
    q_max = (1 << (bits - 1)) - 1
    peak = max((abs(v) for v in values), default=0.0)
    if peak == 0.0:
        return 1.0
    return peak / q_max


def quantize_symmetric(values: list[float], bits: int) -> tuple[list[int], float]:
    """Round-to-nearest on a symmetric grid. Returns (codes, scale)."""
    q_max = (1 << (bits - 1)) - 1
    q_min = -(1 << (bits - 1))
    s = absmax_scale(values, bits)
    codes = [max(q_min, min(q_max, int(round(v / s)))) for v in values]
    return codes, s


def dequantize_symmetric(codes: list[int], scale: float) -> list[float]:
    return [c * scale for c in codes]


def quantize_per_tensor(w: list[list[float]], bits: int) -> list[list[float]]:
    """ONE scale for the entire matrix. The naive baseline, and it is catastrophic.

    A single large-magnitude channel -- and real LLM weight matrices have them -- raises the scale
    for every other weight, so the small weights all round to zero. This is the failure that makes
    the whole topic non-trivial, and it is visible in experiment 2 as an error that grows with
    outlier severity at CONSTANT bit width.
    """
    flat = [v for row in w for v in row]
    codes, s = quantize_symmetric(flat, bits)
    q_max = (1 << (bits - 1)) - 1
    q_min = -(1 << (bits - 1))
    out, i = [], 0
    for row in w:
        r = []
        for _ in row:
            r.append(max(q_min, min(q_max, codes[i])) * s)
            i += 1
        out.append(r)
    return out


def quantize_per_channel(w: list[list[float]], bits: int) -> list[list[float]]:
    """One scale per output channel (row). The standard for INT8 weights.

    Each row gets its own scale, so an outlier channel only degrades ITSELF. This is a large win
    over per-tensor and costs one fp16 scale per row -- negligible.
    """
    out = []
    for row in w:
        codes, s = quantize_symmetric(row, bits)
        out.append(dequantize_symmetric(codes, s))
    return out


def quantize_groupwise(w: list[list[float]], bits: int, group_size: int) -> list[list[float]]:
    """One scale per contiguous group of `group_size` weights within a row.

    The 4-bit standard. Shrinking the group shrinks the error -- because the scale is fitted to
    fewer, more homogeneous weights -- and costs `scale_bits/group_size` extra bits per weight.
    **This is the only genuine trade in the method**, and experiment 2 shows both sides of it.
    """
    if group_size < 1:
        raise ValueError("group_size must be >= 1")
    out = []
    for row in w:
        r = []
        for start in range(0, len(row), group_size):
            g = row[start:start + group_size]
            codes, s = quantize_symmetric(g, bits)
            r.extend(dequantize_symmetric(codes, s))
        out.append(r)
    return out


# --------------------------------------------------------------------------------------
# NF4 -- the codebook quantizer
# --------------------------------------------------------------------------------------

def nf4_levels() -> list[float]:
    """The 16 NormalFloat4 levels, normalised to [-1, 1].

    NF4 is not a uniform grid. Its levels are the QUANTILES of a standard normal distribution, so
    each bin holds an equal share of the probability mass. The corpus gives the reason [R]:
    "each quantization bin contains an equal number of values from the normal distribution. This
    prevents 'clustering' of weights and ensures that the model preserves as much information
    (entropy) as possible".

    The practical consequence is a grid that is DENSE near zero and SPARSE in the tails, which is
    exactly where LLM weights live. Compare with `int4_levels()` in experiment 4.
    """
    # The published NF4 table (Dettmers et al. 2023), normalised.
    raw = [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
           -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
           0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
           0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
           0.7229568362236023, 1.0]
    return raw


def int4_levels() -> list[float]:
    """A uniform 4-bit grid, for contrast. 16 evenly-spaced levels in [-1, 1]."""
    return [-1.0 + 2.0 * i / 15.0 for i in range(16)]


def quantize_nf4(w: list[list[float]], group_size: int = 64) -> list[list[float]]:
    """NF4 with per-group absmax normalisation, then nearest-codebook lookup.

    Two steps per group: normalise by the group's absmax so the values land in [-1, 1], then snap
    each to the nearest NF4 level and scale back. The absmax normalisation is what makes the
    codebook applicable to groups of any magnitude.
    """
    levels = nf4_levels()
    out = []
    for row in w:
        r = []
        for start in range(0, len(row), group_size):
            g = row[start:start + group_size]
            peak = max((abs(v) for v in g), default=0.0) or 1.0
            for v in g:
                x = v / peak
                r.append(min(levels, key=lambda L: abs(L - x)) * peak)
        out.append(r)
    return out


# --------------------------------------------------------------------------------------
# AWQ's idea -- not a quantizer, a policy
# --------------------------------------------------------------------------------------

def salience(w: list[list[float]]) -> list[float]:
    """Per-channel importance, proxied by L1 norm. Real AWQ uses activation statistics from a
    small calibration set; the L1 norm of the weight row is the offline stand-in.

    The corpus's mechanism [R]: AWQ "identifies which weights are the most 'salient' based on the
    actual activation values seen during a small calibration run. By preserving only these
    important weights (usually 1%) in higher precision and quantizing the rest, AWQ achieves
    better perplexity than GPTQ".
    """
    return [sum(abs(v) for v in row) for row in w]


def quantize_with_protected_channels(w: list[list[float]], bits: int, group_size: int,
                                     salient_frac: float) -> tuple[list[list[float]], list[int]]:
    """Quantize everything, then RESTORE the most salient channels at full precision.

    This is the AWQ pattern in its simplest honest form: a mixed-precision tensor. Returns the
    reconstructed matrix and the list of protected channel indices, so the caller can report what
    fraction of storage the protection cost.

    `salient_frac = 0.01` means the top 1% of channels are kept exact. The experiment shows that
    1% recovers most of the error -- which is the surprising part, and the reason the method works.
    """
    if not 0.0 <= salient_frac < 1.0:
        raise ValueError("salient_frac must be in [0, 1)")
    q = quantize_groupwise(w, bits, group_size)
    order = sorted(range(len(w)), key=lambda i: salience(w)[i], reverse=True)
    n_protect = int(round(len(w) * salient_frac))
    protected = sorted(order[:n_protect])
    for i in protected:
        q[i] = list(w[i])
    return q, protected


def protected_storage_overhead(salient_frac: float, protected_bits: int = 16,
                               quantized_bits: int = 4) -> float:
    """Extra bytes per weight as a fraction of the fully-quantized size.

    Protecting 1% of channels at 16 bits against 4 bits elsewhere costs
    `0.01 * (16-4)/4 = 3%` more storage. The experiment pairs this with the error it recovers --
    and the ratio is what makes AWQ worth doing.
    """
    return salient_frac * (protected_bits - quantized_bits) / quantized_bits


# --------------------------------------------------------------------------------------
# Error metrics
# --------------------------------------------------------------------------------------

def mse(a: list[list[float]], b: list[list[float]]) -> float:
    n = 0
    total = 0.0
    for ra, rb in zip(a, b):
        for x, y in zip(ra, rb):
            total += (x - y) ** 2
            n += 1
    return total / max(n, 1)


def snr_db(reference: list[list[float]], estimate: list[list[float]]) -> float:
    """Signal-to-noise ratio in dB. THE metric to compare quantizers on.

    MSE alone is not comparable across tensors of different magnitude; SNR is. A doubling of the
    standard deviation makes MSE 4x worse with no change in quantizer quality.

    Rule of thumb visible in the experiments: every additional bit buys about 6 dB, and every
    halving of the group size buys about 3 dB. Those two are the knobs.
    """
    sig = sum(x * x for row in reference for x in row)
    err = sum((x - y) ** 2 for ra, rb in zip(reference, estimate) for x, y in zip(ra, rb))
    if err == 0:
        return float("inf")
    return 10.0 * math.log10(sig / err)


def worst_channel_error(reference: list[list[float]], estimate: list[list[float]]) -> tuple[int, float]:
    """The channel with the largest relative error, and its error. Returns (index, relative error).

    THE TAIL MATTERS MORE THAN THE MEAN. Aggregate SNR can look healthy while a single channel is
    destroyed, and a destroyed channel is a destroyed feature. This is what per-channel and
    group-wise quantization fix, and what a mean-only report hides.
    """
    worst_i, worst_e = -1, 0.0
    for i, (ra, rb) in enumerate(zip(reference, estimate)):
        sig = sum(x * x for x in ra)
        err = sum((x - y) ** 2 for x, y in zip(ra, rb))
        rel = math.sqrt(err / sig) if sig > 0 else 0.0
        if rel > worst_e:
            worst_i, worst_e = i, rel
    return worst_i, worst_e


# --------------------------------------------------------------------------------------
# Synthetic weights with realistic structure
# --------------------------------------------------------------------------------------

def synthetic_weights(n_channels: int = 96, n_weights: int = 64, seed: int = 11,
                      outlier_frac: float = 0.04, outlier_scale: float = 12.0) -> list[list[float]]:
    """A weight matrix shaped like a real one: zero-centred normal, plus a few large channels.

    The outlier structure is not decoration. It is the single fact that makes quantization a
    research problem rather than a rounding exercise -- and it is why `per_tensor` fails while
    `per_channel` does not.
    """
    rng = random.Random(seed)
    w = [[rng.gauss(0.0, 1.0) for _ in range(n_weights)] for _ in range(n_channels)]
    n_out = max(1, int(round(n_channels * outlier_frac)))
    for i in rng.sample(range(n_channels), n_out):
        w[i] = [v * outlier_scale for v in w[i]]
    return w


def outlier_severity_sweep(severities: list[float], n_channels: int = 96, n_weights: int = 64,
                           seed: int = 11, bits: int = 4, group_size: int = 32) -> list[dict]:
    """Error as a function of HOW LARGE the outlier channels are, at fixed bit width.

    This is the demonstration that precision is not the whole story: the same 4 bits give wildly
    different errors depending on the tensor's dynamic range, which is why "4-bit is nearly free"
    is true on some models and false on others [T].
    """
    rows = []
    for sev in severities:
        w = synthetic_weights(n_channels, n_weights, seed, 0.04, sev)
        pt = quantize_per_tensor(w, bits)
        pc = quantize_per_channel(w, bits)
        gw = quantize_groupwise(w, bits, group_size)
        rows.append({
            "severity": sev,
            "per_tensor_snr": snr_db(w, pt),
            "per_channel_snr": snr_db(w, pc),
            "groupwise_snr": snr_db(w, gw),
            "worst_channel_tensor": worst_channel_error(w, pt)[1],
            "worst_channel_group": worst_channel_error(w, gw)[1],
        })
    return rows
