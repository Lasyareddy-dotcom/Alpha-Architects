"""Lightweight online Page-Hinkley drift detector."""
from __future__ import annotations

class PageHinkley:
    def __init__(self, delta: float = 0.002, threshold: float = 4.0, min_instances: int = 25):
        self.delta = delta
        self.threshold = threshold
        self.min_instances = min_instances
        self.reset()

    def reset(self):
        self.n = 0
        self.mean = 0.0
        self.cum = 0.0
        self.min_cum = 0.0

    def update(self, value: float) -> bool:
        self.n += 1
        self.mean += (value - self.mean) / self.n
        self.cum += value - self.mean - self.delta
        self.min_cum = min(self.min_cum, self.cum)
        if self.n >= self.min_instances and (self.cum - self.min_cum) > self.threshold:
            self.reset()
            return True
        return False
