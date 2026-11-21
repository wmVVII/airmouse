"""
One Euro (1€) Filter implementation for real-time, low-latency, jitter-free cursor tracking.

Reference:
    Casiez, G., Roussel, N. and Vogel, D. (2012).
    1 € Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems.
    ACM CHI 2012.
"""

from __future__ import annotations

import math
from typing import Tuple, Optional


class LowPassFilter:
    """First-order low-pass filter with exponential smoothing."""

    def __init__(self, init_val: float = 0.0) -> None:
        self.y: float = init_val
        self.initialized: bool = False

    def reset(self) -> None:
        self.initialized = False
        self.y = 0.0

    def filter(self, value: float, alpha: float) -> float:
        if not self.initialized:
            self.y = value
            self.initialized = True
            return value
        self.y = alpha * value + (1.0 - alpha) * self.y
        return self.y


class OneEuroFilter:
    """
    Adaptive low-pass filter with speed-based cutoff frequency adjustment.
    
    Provides high filtering (tremor suppression) during slow movements and
    virtually zero-delay tracking during fast movements.
    """

    def __init__(
        self,
        freq: float = 30.0,
        mincutoff: float = 1.0,
        beta: float = 5.0,
        dcutoff: float = 1.0,
    ) -> None:
        """
        Args:
            freq: Default sampling rate in Hz.
            mincutoff: Minimum cutoff frequency in Hz (> 0). Lower values reduce jitter at rest.
            beta: Speed coefficient (> 0). Scaled for normalized coordinates [0, 1].
                  Higher values eliminate lag during fast flicks.
            dcutoff: Cutoff frequency for derivative estimation in Hz.
        """
        self.freq: float = max(freq, 1e-4)
        self.mincutoff: float = mincutoff
        self.beta: float = beta
        self.dcutoff: float = dcutoff

        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()
        self.last_time: Optional[float] = None

    def _alpha(self, cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x: float, timestamp: Optional[float] = None) -> float:
        """Filter a single scalar value x at given timestamp (in seconds)."""
        if self.last_time is None or timestamp is None:
            dt = 1.0 / self.freq
        else:
            dt = timestamp - self.last_time
            if dt <= 0.0:
                dt = 1e-4

        self.last_time = timestamp

        # Estimate derivative
        if not self.x_filter.initialized:
            dx = 0.0
        else:
            dx = (x - self.x_filter.y) / dt

        edx = self.dx_filter.filter(dx, self._alpha(self.dcutoff, dt))
        cutoff = self.mincutoff + self.beta * abs(edx)
        return self.x_filter.filter(x, self._alpha(cutoff, dt))

    def reset(self) -> None:
        """Reset internal filter states."""
        self.x_filter.reset()
        self.dx_filter.reset()
        self.last_time = None


class PointFilter2D:
    """Convenience wrapper for filtering 2D coordinates (x, y) using OneEuroFilter."""

    def __init__(
        self,
        freq: float = 30.0,
        mincutoff: float = 1.0,
        beta: float = 5.0,
        dcutoff: float = 1.0,
    ) -> None:
        self.x_filt = OneEuroFilter(freq, mincutoff, beta, dcutoff)
        self.y_filt = OneEuroFilter(freq, mincutoff, beta, dcutoff)

    def filter(self, x: float, y: float, timestamp: Optional[float] = None) -> Tuple[float, float]:
        """Filters 2D position (x, y) and returns smoothed (x, y)."""
        fx = self.x_filt.filter(x, timestamp)
        fy = self.y_filt.filter(y, timestamp)
        return fx, fy

    def reset(self) -> None:
        """Resets both coordinate filters."""
        self.x_filt.reset()
        self.y_filt.reset()
