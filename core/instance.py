"""TSP 实例的生成与读取。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import DISTRIBUTIONS, InstanceConfig


@dataclass
class Instance:
    """一个 TSP 实例：城市坐标 + 距离矩阵。"""

    coords: np.ndarray  # (n, 2) float64
    dist: np.ndarray  # (n, n) float64
    name: str = ""
    optimal: float | None = None  # 已知最优解（TSPLIB 才会给）

    @property
    def n(self) -> int:
        return int(self.coords.shape[0])

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "n_cities": self.n,
            "coords": self.coords.tolist(),
            "optimal": self.optimal,
        }


def _build(coords: np.ndarray, name: str, optimal: float | None = None) -> Instance:
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff**2).sum(axis=-1))
    return Instance(coords=coords, dist=dist, name=name, optimal=optimal)


def generate(cfg: InstanceConfig) -> Instance:
    """按指定分布生成随机实例。"""
    n = int(cfg.n_cities)
    if not 3 <= n <= 200:
        raise ValueError(f"城市数必须在 3~200 之间，收到 {n}")
    if cfg.distribution not in DISTRIBUTIONS:
        raise ValueError(f"未知分布：{cfg.distribution}，可选 {DISTRIBUTIONS}")

    rng = np.random.default_rng(cfg.seed)

    if cfg.distribution == "uniform":
        coords = rng.random((n, 2))

    elif cfg.distribution == "cluster":
        # 3~5 个高斯簇，各簇散布程度不同
        k = int(rng.integers(3, 6))
        centers = rng.random((k, 2))
        assign = rng.integers(0, k, size=n)
        scales = rng.uniform(0.03, 0.10, size=(k, 1))
        coords = centers[assign] + rng.normal(size=(n, 2)) * scales[assign]
        coords = np.clip(coords, 0.0, 1.0)

    else:  # grid
        side = int(np.ceil(np.sqrt(n)))
        gx, gy = np.meshgrid(np.arange(side), np.arange(side))
        pts = np.stack([gx.ravel(), gy.ravel()], axis=1)[:n].astype(float)
        pts /= max(side - 1, 1)
        coords = np.clip(pts + rng.normal(scale=0.012, size=(n, 2)), 0.0, 1.0)

    return _build(coords, cfg.resolved_name())


def load_tsplib(path: str | Path) -> Instance:
    """解析 TSPLIB 格式（支持 EUC_2D）。"""
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="ignore")

    name = path.stem
    optimal = None
    dim = None
    coords: list[tuple[float, float]] = []
    in_section = False

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line == "EOF":
            continue
        upper = line.upper()

        if upper.startswith("NAME"):
            name = line.split(":", 1)[-1].strip()
        elif upper.startswith("DIMENSION"):
            dim = int(line.split(":", 1)[-1].strip())
        elif upper.startswith("BEST_KNOWN") or upper.startswith("OPTIMAL"):
            try:
                optimal = float(line.split(":", 1)[-1].strip())
            except ValueError:
                pass
        elif upper.startswith("NODE_COORD_SECTION"):
            in_section = True
        elif in_section:
            parts = line.split()
            if len(parts) >= 3:
                coords.append((float(parts[1]), float(parts[2])))

    if not coords:
        raise ValueError(f"{path} 中未找到 NODE_COORD_SECTION 数据")
    if dim is not None and len(coords) != dim:
        raise ValueError(f"{path} 声明 {dim} 个城市，实际解析到 {len(coords)} 个")

    arr = np.asarray(coords, dtype=float)
    # TSPLIB 坐标常是整数，归一化到单位正方形便于与随机实例统一绘制
    lo, hi = arr.min(axis=0), arr.max(axis=0)
    span = np.maximum(hi - lo, 1e-12)
    arr_norm = (arr - lo) / span

    inst = _build(arr_norm, name, optimal)
    # 保留原始距离矩阵，用于与官方最优解比对
    raw_diff = coords_arr_scale = arr[:, None, :] - arr[None, :, :]
    inst.dist = np.sqrt((raw_diff**2).sum(axis=-1))
    return inst


def list_tsplib() -> list[dict]:
    """列出 data/tsplib 下可用的标准实例。"""
    from config import TSPLIB_DIR

    out = []
    for p in sorted(TSPLIB_DIR.glob("*.tsp")):
        try:
            inst = load_tsplib(p)
            out.append({"name": inst.name, "n_cities": inst.n, "optimal": inst.optimal})
        except Exception:  # noqa: BLE001
            continue
    return out
