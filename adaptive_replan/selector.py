"""DVAC equations 6--9, with an explicit first-call initialization rule."""

import math
from collections import deque


class DVACSelector:
    def __init__(self, horizon=50, tail_length=5, window_size=5, alpha=2.0, min_steps=1):
        for name, value in (("horizon", horizon), ("tail_length", tail_length),
                            ("window_size", window_size), ("min_steps", min_steps)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if min_steps > horizon or not math.isfinite(alpha) or alpha < 0:
            raise ValueError("Invalid minimum horizon or alpha")
        self.horizon = horizon
        self.tail_length = tail_length
        self.alpha = float(alpha)
        self.min_steps = min_steps
        self.window = deque(maxlen=window_size)

    def reset(self):
        self.window.clear()

    def select(self, variance):
        values = tuple(float(v) for v in variance)
        if len(values) != self.horizon or any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("Expected one finite nonnegative variance per future action")
        count = len(self.window)
        mean = std = threshold = crossing = None
        if count == 0:
            chosen = self.min_steps
        else:
            previous = [v for call in self.window for v in call]
            mean = math.fsum(previous) / len(previous)
            std = math.sqrt(math.fsum((v - mean) ** 2 for v in previous) / len(previous))
            threshold = mean + self.alpha * std
            crossing = next((j for j, v in enumerate(values) if v > threshold), None)
            chosen = self.horizon if crossing is None else max(self.min_steps, crossing)
        result = {
            "chosen_k": chosen,
            "threshold": threshold,
            "first_crossing": crossing,
            "window_calls": count,
            "window_mean": mean,
            "window_std": std,
            "bootstrap": count == 0,
        }
        self.window.append(values)
        return result
