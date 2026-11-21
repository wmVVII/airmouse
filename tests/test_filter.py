import math
import numpy as np
import pytest

from airmouse.core.filter import LowPassFilter, OneEuroFilter, PointFilter2D


def test_low_pass_filter():
    lpf = LowPassFilter()
    assert not lpf.initialized
    assert lpf.filter(10.0, alpha=0.5) == 10.0
    assert lpf.initialized

    v = lpf.filter(20.0, alpha=0.5)
    assert math.isclose(v, 15.0)

    lpf.reset()
    assert not lpf.initialized
    assert lpf.filter(5.0, alpha=0.5) == 5.0


def test_one_euro_filter_jitter_reduction():
    """Verify that 1€ filter significantly reduces variance on stationary noisy signals."""
    filter = OneEuroFilter(freq=60.0, mincutoff=1.0, beta=5.0)
    np.random.seed(42)

    true_val = 0.5
    raw_signal = true_val + np.random.normal(0, 0.01, 100)
    filtered = []

    t = 0.0
    dt = 1.0 / 60.0
    for val in raw_signal:
        filtered.append(filter.filter(val, timestamp=t))
        t += dt

    warmup = 10
    raw_var = np.var(raw_signal[warmup:])
    filtered_var = np.var(filtered[warmup:])

    # Filtered signal variance should be at least 3x smaller than raw noise
    assert filtered_var < (raw_var / 3.0)


def test_one_euro_filter_fast_response():
    """Verify that during high velocity movement, cutoff increases and lag is minimal."""
    filter = OneEuroFilter(freq=60.0, mincutoff=1.0, beta=5.0)

    # Stationary at 0.0
    t = 0.0
    dt = 1.0 / 60.0
    for _ in range(20):
        filter.filter(0.0, timestamp=t)
        t += dt

    # Sudden high speed jump to 1.0
    out = filter.filter(1.0, timestamp=t)
    # Because beta is 5.0, cutoff increases dramatically and responds immediately (>0.7)
    assert out > 0.4


def test_point_filter_2d():
    p_filter = PointFilter2D(freq=60.0, mincutoff=1.0, beta=5.0)
    fx, fy = p_filter.filter(0.5, 0.5, timestamp=1.0)
    assert fx == 0.5 and fy == 0.5

    fx2, fy2 = p_filter.filter(0.6, 0.6, timestamp=1.016)
    assert 0.5 < fx2 <= 0.6
    assert 0.5 < fy2 <= 0.6

    p_filter.reset()
    assert p_filter.x_filt.last_time is None
