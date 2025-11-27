#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
π-normality research harness (experimental).
- Pulls decimal digits of π via mpmath.
- Runs:
  (1) Block and difference-mod-10 frequency tests
  (2) Weyl sums for {10^n π} via digit shifts
  (3) Star discrepancy of {10^n π} with truncated fractional parts
  (4) Finite-state martingale (Kelly log-likelihood vs uniform)
  (5) Approximate Gowers U^2 for digit-indicator functions
  (6) Resonant-base correlation test (hex digits via BBP -> decimal digit)

Notes:
- Nothing here can prove normality. It only provides evidence or hunts for obstructions.
- Hex BBP digit extraction is included but slow for large indices; keep samples modest.
"""

import math
import cmath
import random
from collections import defaultdict, Counter
from decimal import Decimal, getcontext

try:
    from mpmath import mp
    HAVE_MPMATH = True
except Exception:
    HAVE_MPMATH = False

# ---------- Utilities: π decimal digits ----------

def pi_decimal_digits(N, offset=0, extra_guard=25):
    """
    Return N decimal digits of π starting at position offset+1 after the decimal point.
    Requires mpmath. For N up to ~200k it is OK on a modern machine.
    """
    if not HAVE_MPMATH:
        raise RuntimeError("mpmath is required to compute π digits. pip install mpmath")
    total = N + offset
    mp.dps = total + extra_guard + 5
    txt = mp.nstr(mp.pi, n=total + 2)  # includes "3." + digits
    # Strip "3." and take digits
    digits = [int(ch) for ch in txt if ch.isdigit()][1:]  # drop the initial '3'
    if len(digits) < total:
        raise RuntimeError("Not enough digits produced. Increase dps or N.")
    return digits[offset:offset+N]

# ---------- Fractional parts from digits ----------

def fractional_from_digits(digs, start, K, as_float=True):
    """
    Approximate x_n = {10^n π} by truncating to K digits:
    x_n ≈ 0.d_{n+1} d_{n+2} ... d_{n+K}.
    """
    K = min(K, len(digs) - start)
    if K <= 0:
        return 0.0
    if as_float and K <= 16:
        # Fast path with double precision
        x = 0.0
        p = 0.1
        for i in range(K):
            x += digs[start+i] * p
            p *= 0.1
        return x
    else:
        # High-precision via Decimal
        getcontext().prec = K + 10
        x = Decimal(0)
        p = Decimal(1) / Decimal(10)
        for i in range(K):
            x += Decimal(digs[start+i]) * p
            p /= 10
        return float(x)

# ---------- Test 1: Block frequency and difference-mod-10 ----------

def block_frequencies(digs, k=1):
    """
    Count k-blocks in decimal digits and return normalized frequencies.
    """
    N = len(digs)
    if N < k:
        return {}
    counts = Counter()
    for i in range(N - k + 1):
        block = tuple(digs[i:i+k])
        counts[block] += 1
    total = N - k + 1
    freqs = {blk: c/total for blk, c in counts.items()}
    return freqs

def chi_square_uniform(freqs, alphabet_size):
    """
    Chi-square statistic vs uniform over a given alphabet size.
    freqs: dict of observed frequencies over observed outcomes.
    Missing outcomes are treated as zero.
    """
    expected = 1.0 / alphabet_size
    chi2 = 0.0
    for _ in range(alphabet_size):
        pass
    # If keys are tuples of digits, infer alphabet size from power of 10 when possible
    # For simple cases we accept alphabet_size provided by caller.
    observed_values = list(freqs.values())
    # Add zero for missing outcomes if appropriate
    if len(freqs) < alphabet_size:
        observed_values += [0.0] * (alphabet_size - len(freqs))
    for p in observed_values:
        chi2 += (p - expected) ** 2 / expected
    return chi2

def difference_mod10_test(digs, lag=1):
    """
    Compute distribution of (d_{n+lag} - d_n) mod 10.
    Returns frequencies and chi-square vs uniform.
    """
    N = len(digs) - lag
    counts = [0] * 10
    for i in range(N):
        diff = (digs[i+lag] - digs[i]) % 10
        counts[diff] += 1
    freqs = [c/N for c in counts]
    chi2 = sum((f - 0.1)**2 / 0.1 for f in freqs)
    return freqs, chi2

# ---------- Test 2: Weyl sums and star discrepancy for {10^n π} ----------

def weyl_sums_from_digits(digs, h_vals=(1,2,3,4,5), K=12, stride=1):
    """
    Compute S_h(N) = (1/N) sum_{n} exp(2π i h x_n), x_n ~ 0.d_{n+1}...d_{n+K}.
    Returns dict h -> (S_h, |S_h|).
    """
    N = (len(digs) - K)
    idxs = list(range(0, N, stride))
    res = {}
    for h in h_vals:
        s = 0+0j
        for i in idxs:
            x = fractional_from_digits(digs, i, K, as_float=True)
            s += cmath.exp(2j * math.pi * h * x)
        S = s / len(idxs)
        res[h] = (S, abs(S))
    return res

def star_discrepancy(xs):
    """
    Star discrepancy D_N* of a 1D point set xs in [0,1).
    """
    N = len(xs)
    if N == 0:
        return 0.0
    ys = sorted(xs)
    d = 0.0
    for i, y in enumerate(ys, start=1):
        d = max(d, abs(i/N - y), abs(y - (i-1)/N))
    return d

def discrepancy_from_digits(digs, K=8, stride=1):
    """
    Approximate {10^n π} by K-digit truncations and compute star discrepancy.
    """
    xs = []
    N = (len(digs) - K)
    for i in range(0, N, stride):
        xs.append(fractional_from_digits(digs, i, K, as_float=True))
    return star_discrepancy(xs)

# ---------- Test 3: Finite-state martingale (Kelly log-likelihood) ----------

def finite_state_martingale_llr(digs, order=2, alpha=1.0):
    """
    Online n-gram predictor with Laplace smoothing (alpha), returns cumulative
    log-likelihood ratio vs uniform: sum log(p_hat(next)/0.1).
    If this grows linearly positive, it would indicate a learnable bias.
    """
    N = len(digs)
    if N <= order:
        return 0.0, []
    counts = defaultdict(lambda: [alpha]*10)  # context -> counts
    llrs = []
    llr = 0.0
    for i in range(order, N):
        ctx = tuple(digs[i-order:i])
        total = sum(counts[ctx])
        probs = [c/total for c in counts[ctx]]
        nextd = digs[i]
        llr += math.log(probs[nextd] / 0.1)
        llrs.append(llr)
        counts[ctx][nextd] += 1.0
    return llr, llrs

# ---------- Test 4: Approximate Gowers U^2 for a digit-indicator ----------

def approx_gowers_U2(digs, digit=0, H=2000):
    """
    Approximate U^2 norm for f(n) = 1[d_n=digit] - 0.1.
    U^2^4 ≈ (1/(2H+1)) * sum_{|h|<=H} |E_n f(n) f(n+h)|^2
    We return U2 estimate.
    """
    N = len(digs)
    f = [1.0 if d==digit else 0.0 for d in digs]
    mu = sum(f)/N
    f = [x - 0.1 for x in f]
    H = min(H, N//4)
    # Precompute prefix sums for correlations
    # Naive O(NH) which is fine for moderate N,H
    def corr(h):
        if h >= 0:
            M = N - h
            s = 0.0
            for i in range(M):
                s += f[i] * f[i+h]
            return s / M
        else:
            return corr(-h)
    acc = 0.0
    count = 0
    for h in range(-H, H+1):
        c = corr(h)
        acc += (c*c)
        count += 1
    U2_4 = acc / count
    U2 = U2_4 ** 0.25
    return U2

# ---------- Optional: BBP hex digit extraction ----------

def _bbp_series(j, n):
    """
    Helper for BBP: fractional part of sum_{k=0}^∞ 16^{n-k} / (8k+j).
    Splits into modular sum (k<=n) and tail (k>n).
    """
    # Modular sum part
    s = 0.0
    for k in range(n+1):
        r = 8*k + j
        s = (s + pow(16, n - k, r) / r) % 1.0
    # Tail part (converges quickly)
    t = 0.0
    k = n + 1
    p = 16.0 ** (n - k)  # = 16^{-1} initially
    while True:
        term = (16.0 ** (n - k)) / (8*k + j)
        if term < 1e-17:
            break
        t += term
        k += 1
    return (s + t) % 1.0

def pi_hex_digit_fractional(n):
    """
    Fractional part {16^n * π} via BBP, n>=0.
    Then the (n+1)-th hex digit after the dot is floor(16 * fractional).
    """
    # Bailey–Borwein–Plouffe:
    x = (4.0 * _bbp_series(1, n)
         - 2.0 * _bbp_series(4, n)
         - 1.0 * _bbp_series(5, n)
         - 1.0 * _bbp_series(6, n)) % 1.0
    return x

def pi_hex_digit_at(pos):
    """
    pos>=1: return the hex digit at position pos after the point (0..15).
    """
    frac = pi_hex_digit_fractional(pos-1)
    return int(16.0 * frac)

def pi_hex_window(center_pos, w=1):
    """
    Return a tuple of hex digits around center_pos (inclusive), total length = 2*w+1.
    Uses BBP; slow for large positions—keep positions modest.
    """
    res = []
    for p in range(center_pos - w, center_pos + w + 1):
        if p < 1:
            res.append(0)
        else:
            res.append(pi_hex_digit_at(p))
    return tuple(res)

# ---------- Test 5: Resonant-base correlation ----------

def continued_fraction_convergents(x, max_den=200):
    """
    Compute convergents p/q of x with q<=max_den.
    """
    a = []
    y = x
    while True:
        ai = math.floor(y)
        a.append(ai)
        frac = y - ai
        if frac < 1e-18:
            break
        y = 1.0/frac
        if len(a) > 50:
            break
    # Build convergents
    convs = []
    p0, q0 = 1, 0
    p1, q1 = a[0], 1
    convs.append((p1, q1))
    for i in range(1, len(a)):
        p2 = a[i]*p1 + p0
        q2 = a[i]*q1 + q0
        convs.append((p2, q2))
        p0, q0, p1, q1 = p1, q1, p2, q2
    # Filter by denominator
    convs = [ (p,q) for (p,q) in convs if q <= max_den ]
    return convs

def resonant_pairs(max_M=60):
    """
    Find small (M,L) with 10^M ≈ 16^L via L/M ≈ ln(10)/ln(16).
    Returns list of (M,L, error) sorted by M.
    """
    alpha = math.log(10.0) / math.log(16.0)  # L/M ≈ alpha
    convs = continued_fraction_convergents(alpha, max_den=max_M)
    pairs = []
    for (p, q) in convs:
        L, M = p, q
        err = abs(M*math.log(10.0) - L*math.log(16.0))
        pairs.append((M, L, err))
    pairs.sort()
    return pairs

def resonant_base_correlation(digs_dec, J=80, w=1, max_M=40, train_frac=0.6, seed=1):
    """
    Try to predict decimal digit at position j*M from a small window of hex digits
    around position j*L for resonant (M,L).
    Returns results per (M,L): accuracy, best-pattern table size, and a naive p-value.
    """
    random.seed(seed)
    pairs = resonant_pairs(max_M=max_M)
    results = []
    for (M, L, err) in pairs:
        # Build dataset
        data = []
        max_pos_dec = len(digs_dec) - 1
        for j in range(1, J+1):
            pos_dec = j*M
            if pos_dec < 1 or pos_dec > max_pos_dec:
                continue
            y = digs_dec[pos_dec]  # target decimal digit at j*M (1-based after decimal)
            pos_hex = j*L
            # Keep hex positions modest for BBP speed
            if pos_hex < 1 or pos_hex > 5000:
                continue
            xpat = pi_hex_window(pos_hex, w=w)  # small feature
            data.append((xpat, y))
        if len(data) < 20:
            continue
        random.shuffle(data)
        split = int(train_frac * len(data))
        train, test = data[:split], data[split:]
        # Train: majority mapping pattern -> argmax digit
        table = defaultdict(lambda: [0]*10)
        for xpat, y in train:
            table[xpat][y] += 1
        predict = {}
        for xpat, counts in table.items():
            bestd = max(range(10), key=lambda d: counts[d])
            predict[xpat] = bestd
        # Test
        correct = 0
        for xpat, y in test:
            yhat = predict.get(xpat, random.randrange(10))
            if yhat == y:
                correct += 1
        acc = correct / len(test)
        # Naive binomial p-value vs 1/10 baseline
        n = len(test)
        p0 = 0.1
        # Upper tail p-value:
        logp = 0.0
        # Use simple normal approx when n large
        if n >= 30:
            mean = n*p0
            var = n*p0*(1-p0)
            z = (correct - mean) / math.sqrt(var)
            # one-sided
            from math import erf, sqrt
            pval = 0.5 * (1 - erf(z / math.sqrt(2)))
        else:
            # exact sum
            from math import comb
            pval = sum(comb(n, k) * (p0**k) * ((1-p0)**(n-k)) for k in range(correct, n+1))
        results.append({
            "M": M, "L": L, "err": err, "samples": n, "accuracy": acc,
            "table_size": len(table), "p_value_one_sided": pval
        })
    return results

# ---------- Main demo ----------

def main():
    # Parameters
    N = 60000         # number of decimal digits to fetch
    offset = 0
    print(f"Computing {N} decimal digits of π...")
    digs = pi_decimal_digits(N, offset=offset)

    # Test 1: block and differences
    print("\n[1] Block frequencies (k=1,2) and difference-mod-10")
    freqs1 = block_frequencies(digs, k=1)
    chi1 = chi_square_uniform(freqs1, 10)
    print(f"  1-digit chi^2 vs uniform: {chi1:.4f} (lower is better)")
    freqs2 = block_frequencies(digs, k=2)
    chi2 = chi_square_uniform(freqs2, 100)
    print(f"  2-digit chi^2 vs uniform: {chi2:.4f}")

    for lag in [1, 2, 5, 10]:
        freqs, chi = difference_mod10_test(digs, lag=lag)
        max_dev = max(abs(f-0.1) for f in freqs)
        print(f"  diff mod 10 (lag={lag}): chi^2={chi:.4f}, max|f-0.1|={max_dev:.4f}")

    # Test 2: Weyl sums and discrepancy
    print("\n[2] Weyl sums S_h(N) and star discrepancy for {10^n π}")
    h_vals = (1,2,3,4,5)
    weyl = weyl_sums_from_digits(digs, h_vals=h_vals, K=12, stride=3)
    for h in h_vals:
        Sh, mag = weyl[h]
        print(f"  h={h}: |S_h|={mag:.5f}")
    D = discrepancy_from_digits(digs, K=8, stride=2)
    print(f"  Star discrepancy (K=8) D* ≈ {D:.5f}")

    # Test 3: finite-state martingale
    print("\n[3] Finite-state martingale (order=2) log-likelihood ratio vs uniform")
    llr, trajectory = finite_state_martingale_llr(digs, order=2, alpha=1.0)
    rate = llr / max(1, len(trajectory))
    print(f"  Total LLR: {llr:.3f}, avg per step: {rate:.6f} (should hover near 0)")

    # Test 4: approximate Gowers U^2
    print("\n[4] Approximate Gowers U^2 for a few digits")
    for d in [0,1,2,3,4]:
        U2 = approx_gowers_U2(digs, digit=d, H=2000)
        print(f"  digit={d}: U^2 ≈ {U2:.4f} (random-like ≈ small, e.g. ~0.1-0.2)")

    # Test 5: resonant-base correlation (hex -> decimal)
    print("\n[5] Resonant-base correlation (hex BBP window -> decimal digit at j*M)")
    results = resonant_base_correlation(digs, J=60, w=1, max_M=40, train_frac=0.6, seed=1)
    if not results:
        print("  No usable resonant pairs within limits.")
    else:
        for r in results:
            print(f"  M={r['M']:>2}, L={r['L']:>2}, samples={r['samples']:>3}, "
                  f"acc={r['accuracy']:.3f}, p≈{r['p_value_one_sided']:.3f}, err={r['err']:.2e}")

if __name__ == "__main__":
    main()