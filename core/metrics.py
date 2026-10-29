"""路径长度与质量度量的统一口径。"""

from __future__ import annotations

import numpy as np


def tour_length(tour, dist: np.ndarray) -> float:
    """闭合路径长度。tour 不含重复起点。"""
    t = np.asarray(tour, dtype=np.int64)
    if t.size == 0:
        return float("inf")
    if t.size == 1:
        return 0.0
    return float(dist[t, np.roll(t, -1)].sum())


def gap(ours: float, ref: float) -> float:
    """相对差距（百分比）。ref 为 0 时返回 0。"""
    if ref is None or ref <= 0:
        return 0.0
    return (ours - ref) / ref * 100.0


def is_valid_tour(tour, n: int) -> bool:
    """校验：恰好访问每个城市一次。"""
    t = np.asarray(tour, dtype=np.int64)
    if t.size != n:
        return False
    return bool(np.array_equal(np.sort(t), np.arange(n)))


def summarize(results: dict[str, tuple[float, np.ndarray]], ref_len: float) -> list[dict]:
    """把 {方法名: (长度, 路径)} 整理成带 gap 的表格行。"""
    rows = []
    for name, (length, _tour) in results.items():
        rows.append(
            {
                "method": name,
                "length": round(float(length), 6),
                "gap": round(gap(length, ref_len), 3),
            }
        )
    return rows
