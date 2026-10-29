"""专家数据：专家解、监督标签、索引置换增广。

背景
----
DQN 在 N≥30 上的瓶颈是**样本效率** —— 实测样本/参数比只有 6.1（健康值 >100）。
监督学习没有这个问题：一个专家解就提供 N-1 个「(状态 → 应该走哪)」标签，
而这些标签几乎免费 —— rr2opt 在 N=50 上 0.4 秒就能拿到最优解。

增广的正确做法（这里纠正过一个方案）
------------------------------------
最初打算用「坐标随机旋转 + 翻转」做增广，**这是错的**：
状态由**距离矩阵**构造，而旋转/翻转是等距变换，**距离矩阵逐位不变**，
标签与状态也完全相同 —— 增广后一条新信息都不会带来，纯属白干。

真正有效的增广是**索引置换**（permutation）：
把城市编号整体重排，距离矩阵与专家路径同步置换，
得到一个几何形状相同但编号完全不同的新样本。
这直接打击当前网络的核心弱点 ——
one-hot 让网络能死记「17 号城市在哪」，置换增广逼它必须用距离特征。

规模（N=50 / 500 实例 × 8 次置换）：
    4000 条专家轨迹 = 196,000 个标签
    样本/参数比 6.1 → 约 200，跨进健康区
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

import config  # noqa: F401  必须在 import torch 之前
from config import DATA_DIR, InstanceConfig
from core.baselines import random_restart_2opt
from core.instance import generate
from core.metrics import tour_length


CACHE_DIR = DATA_DIR / "cache"


def _cache_path(
    n_cities: int,
    n_instances: int,
    restarts: int,
    permutations: int,
    seed: int | None,
    use_dist_feature: bool,
) -> Path:
    key = (
        f"n{n_cities}_k{n_instances}_r{restarts}_p{permutations}"
        f"_s{seed}_d{int(use_dist_feature)}"
    )
    return CACHE_DIR / f"expert_{key}.npz"


def save_dataset(ds: "ExpertDataset", path: Path) -> None:
    """把数据集落盘（专家解生成是最贵的一步，A/B 对照时不该重复跑）。

    写完顺手做一次清理：缓存目录总量超过 CACHE_MAX_BYTES 时，
    按修改时间从旧到新删，直到回到限额以内。
    WebUI 每换一组 (K, 置换次数) 就会生成一个几百 MB 的新缓存文件，
    不清理的话磁盘会被慢慢吃满。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        states=ds.states,
        actions=ds.actions,
        q_targets=ds.q_targets,
        n_cities=np.asarray(ds.n_cities),
        n_instances=np.asarray(ds.n_instances),
        n_permutations=np.asarray(ds.n_permutations),
        use_dist_feature=np.asarray(ds.use_dist_feature),
        expert_lengths=np.asarray(ds.expert_lengths, dtype=np.float64),
        instance_seeds=np.asarray(ds.instance_seeds, dtype=np.int64),
        dists=np.asarray(ds.dists, dtype=np.float64),
        tours=np.asarray(ds.tours, dtype=np.int64),
    )
    _prune_cache()


CACHE_MAX_BYTES = 3_000_000_000  # 缓存目录限额 3 GB


def _prune_cache() -> None:
    """按修改时间从旧到新删缓存文件，直到总量回到限额内。"""
    try:
        files = [p for p in CACHE_DIR.glob("expert_*.npz") if p.is_file()]
        if not files:
            return
        total = sum(p.stat().st_size for p in files)
        if total <= CACHE_MAX_BYTES:
            return
        files.sort(key=lambda p: p.stat().st_mtime)  # 最旧的先删
        for p in files:
            if total <= CACHE_MAX_BYTES:
                break
            try:
                size = p.stat().st_size
                p.unlink()
                total -= size
            except OSError:
                pass
    except Exception:  # noqa: BLE001
        pass  # 清理失败不影响主流程


def load_dataset(path: Path) -> "ExpertDataset | None":
    if not path.exists():
        return None
    try:
        z = np.load(path, allow_pickle=False)
        return ExpertDataset(
            states=z["states"],
            actions=z["actions"],
            q_targets=z["q_targets"],
            n_cities=int(z["n_cities"]),
            n_instances=int(z["n_instances"]),
            n_permutations=int(z["n_permutations"]),
            use_dist_feature=bool(z["use_dist_feature"]),
            expert_lengths=[float(x) for x in z["expert_lengths"]],
            dists=[d for d in z["dists"]],
            tours=[t for t in z["tours"]],
            instance_seeds=[int(x) for x in z["instance_seeds"]],
        )
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------------------------ 数据结构
@dataclass
class ExpertDataset:
    """一批专家监督样本。

    states    : (M, state_dim) float32，与 build_state 的口径完全一致
    actions   : (M,) int64，专家在每个状态下选择的动作
    q_targets : (M,) float32，专家 Q 回归的标签（已按 dmean 归一化）
    """

    states: np.ndarray
    actions: np.ndarray
    q_targets: np.ndarray

    n_cities: int
    n_instances: int
    n_permutations: int
    use_dist_feature: bool

    # 每个**原始**实例的专家解（用于参考长度 / 抽检）
    expert_lengths: list[float] = field(default_factory=list)
    dists: list[np.ndarray] = field(default_factory=list)
    tours: list[np.ndarray] = field(default_factory=list)
    instance_seeds: list[int] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return int(self.actions.shape[0])

    @property
    def state_dim(self) -> int:
        return int(self.states.shape[1])

    def summary(self) -> dict:
        return {
            "n_samples": self.n_samples,
            "state_dim": self.state_dim,
            "n_instances": self.n_instances,
            "n_permutations": self.n_permutations,
            "n_cities": self.n_cities,
            "sample_per_param_ratio_hint": self.n_samples,
        }


# ------------------------------------------------------------------ 状态构造
def build_states_np(
    cur: np.ndarray, visited: np.ndarray, dist: np.ndarray, use_dist_feature: bool
) -> np.ndarray:
    """批量构造状态，与 models/dqn.py 的 build_state 逐位对齐。

    差异只有一处：那边用 torch 且一次性算 dist_norm.mean()，
    这边是 numpy 的 dist.mean() —— 数值一致（置换不改变距离矩阵的元素集，
    因此 mean 也不变）。

    参数
    ----
    cur     : (B,) int64，当前城市下标
    visited : (B, N) bool，已访问掩码
    dist    : (N, N) float64，距离矩阵
    """
    b = cur.shape[0]
    n = visited.shape[1]
    width = 3 * n if use_dist_feature else 2 * n
    s = np.zeros((b, width), dtype=np.float32)

    s[np.arange(b), cur] = 1.0
    s[:, n : 2 * n] = visited.astype(np.float32)

    if use_dist_feature:
        dmean = float(dist.mean())
        dfeat = dist[cur] / dmean
        dfeat[visited] = 0.0
        s[:, 2 * n :] = dfeat

    return s


def remaining_costs(tour: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """专家路径上每一步的「剩余代价」。

    rem[t] = 沿专家路径从 tour[t] 走到 tour[0]（闭环收尾）的距离总和
    rem[N-1] = d(tour[N-1], tour[0])
    rem[t]   = d(tour[t], tour[t+1]) + rem[t+1]

    于是状态 t（cur=tour[t]）上专家动作 a=tour[t+1] 的 Q 标签 = -rem[t+1]。
    """
    n = tour.size
    rem = np.zeros(n, dtype=np.float64)
    rem[n - 1] = dist[tour[n - 1], tour[0]]
    for t in range(n - 2, -1, -1):
        rem[t] = dist[tour[t], tour[t + 1]] + rem[t + 1]
    return rem


# ------------------------------------------------------------------ 置换
def permute_instance(
    dist: np.ndarray, tour: np.ndarray, perm: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """索引置换：距离矩阵与专家路径同步重排。

    perm[i] = 旧城市 i 在新编号下的下标。
    返回 (new_dist, new_tour)。
    """
    inv = np.argsort(perm)
    new_dist = dist[np.ix_(perm, perm)]
    new_tour = inv[np.asarray(tour, dtype=np.int64)]
    return new_dist, new_tour


# ------------------------------------------------------------------ 主入口
def make_expert_dataset(
    n_cities: int,
    distribution: str = "uniform",
    n_instances: int = 500,
    restarts: int = 200,
    permutations: int = 8,
    seed: int | None = 0,
    use_dist_feature: bool = True,
    max_bytes: int = 1_200_000_000,
    on_progress: Callable[[dict], None] | None = None,
    stop_event=None,
    use_cache: bool = True,
) -> ExpertDataset:
    """生成专家监督数据集。

    种子空间用 ``seed * 1_000_000 + i`` 派生，保证不同用途（训练集/留出集）
    可以用不同的 base seed 得到**完全不重叠**的实例集合 —— 这是评估可信度的前提。

    permutations = 1 表示不做置换增广（只有原始编号）。

    use_cache=True 时结果会落盘到 data/cache/，重复调用直接读缓存。
    """
    n = int(n_cities)
    perms = max(1, int(permutations))
    state_dim = (3 if use_dist_feature else 2) * n

    cache_path = _cache_path(
        n, int(n_instances), int(restarts), perms, seed, bool(use_dist_feature)
    )
    if use_cache:
        cached = load_dataset(cache_path)
        if cached is not None:
            # 缓存命中也要报一次进度，否则前端看不到「专家数据」这一步
            if on_progress is not None:
                on_progress(
                    {
                        "type": "bc_data",
                        "done": cached.n_instances,
                        "total": cached.n_instances,
                        "samples": cached.n_samples,
                        "elapsed": 0.0,
                        "cached": True,
                    }
                )
            return cached

    # 内存护栏：样本数 × state_dim × 4 字节 超预算就自动削减置换次数
    per_instance_samples = (n - 1) * perms
    est_bytes = int(n_instances) * per_instance_samples * state_dim * 4
    if est_bytes > max_bytes and perms > 1:
        scale = max_bytes / max(est_bytes, 1)
        perms = max(1, int(perms * scale))
        est_bytes = int(n_instances) * (n - 1) * perms * state_dim * 4

    rng = np.random.default_rng(seed if seed is not None else np.random.default_rng().integers(0, 2**31))

    state_chunks: list[np.ndarray] = []
    action_chunks: list[np.ndarray] = []
    q_chunks: list[np.ndarray] = []

    expert_lengths: list[float] = []
    dists: list[np.ndarray] = []
    tours: list[np.ndarray] = []
    seeds: list[int] = []

    t0 = time.perf_counter()
    for i in range(n_instances):
        if stop_event is not None and stop_event.is_set():
            break

        base = (int(seed) if seed is not None else 0) * 1_000_000 + i
        inst = generate(
            InstanceConfig(n_cities=n, distribution=distribution, seed=base)
        )
        d = inst.dist

        tour = random_restart_2opt(
            d, restarts=restarts, seed=int(rng.integers(0, 2**31 - 1))
        )
        expert_lengths.append(float(tour_length(tour, d)))
        dists.append(d)
        tours.append(tour)
        seeds.append(base)

        # 原始编号 + 若干次索引置换
        variants = [(d, tour)]
        for _ in range(perms - 1):
            perm = rng.permutation(n)
            variants.append(permute_instance(d, tour, perm))

        for vd, vt in variants:
            rem = remaining_costs(vt, vd)
            dmean = float(vd.mean())
            cur = vt[:-1]
            # visited[t] = {vt[0], ..., vt[t]}
            visited = np.zeros((n - 1, n), dtype=bool)
            visited[np.arange(n - 1), vt[:-1]] = True
            visited = np.logical_or.accumulate(visited, axis=0)

            state_chunks.append(build_states_np(cur, visited, vd, use_dist_feature))
            action_chunks.append(vt[1:].astype(np.int64))
            q_chunks.append((-rem[1:] / dmean).astype(np.float32))

        if on_progress is not None and (i % 25 == 0 or i == n_instances - 1):
            on_progress(
                {
                    "type": "bc_data",
                    "done": i + 1,
                    "total": n_instances,
                    "samples": sum(a.shape[0] for a in action_chunks),
                    "elapsed": round(time.perf_counter() - t0, 1),
                }
            )

    if action_chunks:
        states = np.concatenate(state_chunks, axis=0)
        actions = np.concatenate(action_chunks, axis=0)
        q_targets = np.concatenate(q_chunks, axis=0)
    else:
        states = np.zeros((0, state_dim), dtype=np.float32)
        actions = np.zeros((0,), dtype=np.int64)
        q_targets = np.zeros((0,), dtype=np.float32)

    ds = ExpertDataset(
        states=states,
        actions=actions,
        q_targets=q_targets,
        n_cities=n,
        n_instances=len(expert_lengths),
        n_permutations=perms,
        use_dist_feature=use_dist_feature,
        expert_lengths=expert_lengths,
        dists=dists,
        tours=tours,
        instance_seeds=seeds,
    )
    if use_cache:
        try:
            save_dataset(ds, cache_path)
        except Exception:  # noqa: BLE001
            pass  # 缓存失败不影响主流程
    return ds


# ------------------------------------------------------------------ 抽检
def verify_expert_against_ortools(
    dist: np.ndarray, tour: np.ndarray, time_limit_s: int = 3
) -> dict:
    """用 OR-Tools 抽检专家解，量化「专家有多准」。

    这不是为了算 gap（gap 的分母用专家长度即可），
    而是为了验证「专家 = 最优」这个前提确实成立。
    """
    from core.baselines import solve_ortools

    expert_len = float(tour_length(tour, dist))
    ort = solve_ortools(dist, time_limit_s=time_limit_s)
    if ort is None:
        return {"expert_len": expert_len, "ortools_len": None, "diff_pct": None}
    ort_len = float(tour_length(ort, dist))
    diff = (expert_len - ort_len) / max(ort_len, 1e-12) * 100.0
    return {
        "expert_len": expert_len,
        "ortools_len": ort_len,
        "diff_pct": round(diff, 4),
    }
