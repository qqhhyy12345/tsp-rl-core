"""基准求解器：最近邻贪心、2-opt、OR-Tools。

三者角色不同：
    greedy    —— 极快，作为下界参考与兜底
    two_opt   —— 后处理，消除路径交叉
    ortools   —— 主基准，作为参考最优解计算 gap
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from core.metrics import tour_length


@dataclass
class SolveResult:
    method: str
    tour: np.ndarray
    length: float
    elapsed: float  # 秒

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "tour": [int(x) for x in self.tour],
            "length": round(float(self.length), 6),
            "elapsed": round(float(self.elapsed), 4),
        }


# ------------------------------------------------------------------ 贪心
def greedy_tour(dist: np.ndarray, start: int = 0) -> np.ndarray:
    """最近邻贪心：每步走向最近的未访问城市。"""
    n = dist.shape[0]
    visited = np.zeros(n, dtype=bool)
    tour = np.empty(n, dtype=np.int64)
    cur = int(start) % n
    tour[0] = cur
    visited[cur] = True

    masked = dist.copy()
    masked[:, visited] = np.inf

    for step in range(1, n):
        row = masked[cur].copy()
        row[visited] = np.inf
        nxt = int(np.argmin(row))
        tour[step] = nxt
        visited[nxt] = True
        cur = nxt

    return tour


def greedy_best(dist: np.ndarray) -> np.ndarray:
    """尝试所有起点取最优（n 较小的时候用，质量明显更好）。"""
    n = dist.shape[0]
    if n > 120:
        return greedy_tour(dist, start=0)
    best, best_len = None, np.inf
    for s in range(n):
        t = greedy_tour(dist, start=s)
        L = tour_length(t, dist)
        if L < best_len:
            best, best_len = t, L
    return best


# ------------------------------------------------------------------ 2-opt
def two_opt(tour: np.ndarray, dist: np.ndarray, max_passes: int = 200) -> np.ndarray:
    """2-opt 局部搜索：反复交换边消除交叉，直到无法改进。

    实现为「按 i 做 best-improvement」：对每个 i 一次性向量化算出所有可行的
    j 对应的 delta，取最优者执行翻转。比逐个 (i, j) 尝试快得多。
    """
    t = np.asarray(tour, dtype=np.int64).copy()
    n = t.size
    if n < 4:
        return t

    for _ in range(max_passes):
        improved = False

        for i in range(n - 2):
            js = np.arange(i + 2, n)
            # i 与最后一个点相邻，不构成有效交换
            if i == 0:
                js = js[js != n - 1]
            if js.size == 0:
                continue

            a = t[i]
            b = t[i + 1]
            c = t[js]
            d = t[(js + 1) % n]

            delta = dist[a, c] + dist[b, d] - dist[a, b] - dist[c, d]
            k = int(np.argmin(delta))

            if delta[k] < -1e-12:
                j = int(js[k])
                t[i + 1 : j + 1] = t[i + 1 : j + 1][::-1]
                improved = True

        if not improved:
            break

    return t


# ------------------------------------------------------------------ OR-Tools
def solve_ortools(dist: np.ndarray, time_limit_s: int = 3) -> np.ndarray | None:
    """用 OR-Tools 的引导式局部搜索求解，作为参考最优解。

    距离需整数化，这里放大 10000 倍保留 4 位小数精度。
    """
    try:
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    except ImportError:
        return None

    n = dist.shape[0]
    scale = 10_000
    d_int = np.round(dist * scale).astype(np.int64)

    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def _cb(from_index: int, to_index: int) -> int:
        f = manager.IndexToNode(from_index)
        t = manager.IndexToNode(to_index)
        return int(d_int[f, t])

    cb_idx = routing.RegisterTransitCallback(_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(cb_idx)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    params.time_limit.FromSeconds(int(time_limit_s))

    solution = routing.SolveWithParameters(params)
    if solution is None:
        return None

    tour: list[int] = []
    idx = routing.Start(0)
    while not routing.IsEnd(idx):
        tour.append(manager.IndexToNode(idx))
        idx = solution.Value(routing.NextVar(idx))

    return np.asarray(tour, dtype=np.int64)


def random_restart_2opt(
    dist: np.ndarray, restarts: int = 200, seed: int | None = 0
) -> np.ndarray:
    """随机重启 + 2-opt —— **不需要学习的强基线**。

    这是本项目最容易被忽视的对照。实测（N=25, uniform）：
    200 次随机重启 + 2-opt 就能达到 OR-Tools 最优解，耗时不到 1 秒。

    意义：N≤50 时 gap 数字全是 0~5%，光看 DQN 的 gap 无法判断模型好坏 ——
    必须有这条基线才知道「学习到底带来了什么」。

    随机重启在 N≥80 后开始跟不上（实测 N=200 时 +1.98%），
    那才是学习方法真正该发力的区间。

    返回约定：与本模块其他求解器一致 —— **n 个点，隐式闭环**（不重复首点）。
    """
    n = dist.shape[0]
    if n < 4:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    best: np.ndarray | None = None
    best_len = float("inf")
    for _ in range(max(1, restarts)):
        t = rng.permutation(n).astype(np.int64)
        t = np.append(t, t[0])          # 2-opt 需要显式闭环
        t = two_opt(t, dist)
        length = tour_length(t, dist)
        if length < best_len:
            best_len = length
            best = t
    if best is None:
        return np.arange(n, dtype=np.int64)
    # 去掉重复的末点，回到 n 点隐式闭环的约定
    if best.size > n and best[0] == best[-1]:
        best = best[:-1]
    return best


def multi_start_2opt(
    tours: np.ndarray, dist: np.ndarray, max_passes: int = 200
) -> tuple[np.ndarray, float]:
    """对每条候选路径**分别**做 2-opt，再从优化后的结果里选最优。

    为什么不先选优再优化（重要）：
    不同起点解的**结构质量差异很大**，原始长度最长的那条经过 2-opt 之后
    反而可能收敛到最优解。先选优会把这些解提前丢掉。

    实测（N=25，同一份 DQN 权重）：
        先选优再 2-opt      gap 0.57%
        各自 2-opt 后选优   gap 0.00%   ← 命中最优解
    20 个随机实例统计检验：更优 15 次 / 持平 5 次 / 更差 0 次。

    代价：N 条路径各做一次 2-opt，约 13~16 ms（纯 numpy）。

    返回约定：**n 个点，隐式闭环**（与本模块其他求解器一致）。
    """
    arr = np.atleast_2d(np.asarray(tours))
    best: np.ndarray | None = None
    best_len = float("inf")
    for t in arr:
        # 2-opt 需要闭环输入（最后一点 → 第一点那条边）
        tt = np.asarray(t, dtype=np.int64)
        if tt.size >= 2 and tt[0] != tt[-1]:
            tt = np.append(tt, tt[0])
        opt = two_opt(tt, dist, max_passes=max_passes)
        length = tour_length(opt, dist)
        if length < best_len:
            best_len = length
            best = opt
    if best is None:
        return np.arange(dist.shape[0], dtype=np.int64), float("inf")
    # 去掉重复的末点，回到 n 点隐式闭环的约定
    if best.size > dist.shape[0] and best[0] == best[-1]:
        best = best[:-1]
    return best, best_len


# ------------------------------------------------------------------ 统一入口
def solve(method: str, dist: np.ndarray, time_limit_s: int = 3) -> SolveResult:
    """按名字调用求解器。method ∈ {greedy, two_opt, ortools, ms2opt, rr2opt, ...}"""
    t0 = time.perf_counter()
    method = method.lower()

    if method == "greedy":
        tour = greedy_best(dist)
    elif method == "two_opt":
        tour = two_opt(greedy_best(dist), dist)
    elif method == "ortools":
        tour = solve_ortools(dist, time_limit_s=time_limit_s)
        if tour is None:
            tour = two_opt(greedy_best(dist), dist)
            method = "ortools(fallback:2opt)"
    elif method == "rr2opt":
        # 随机重启 + 2-opt：不学习的强基线，用于判断学习是否真的有价值
        tour = random_restart_2opt(dist, restarts=200)
    elif method == "ms2opt":
        # 全部起点最近邻 + 各自 2-opt 后选优：比 random_restart 更稳的规则基线
        n = dist.shape[0]
        tours = np.stack([greedy_tour(dist, start=s) for s in range(n)])
        tour, _ = multi_start_2opt(tours, dist)
    elif method == "identity":
        tour = np.arange(dist.shape[0], dtype=np.int64)
    else:
        raise ValueError(f"未知方法：{method}")

    return SolveResult(
        method=method,
        tour=tour,
        length=tour_length(tour, dist),
        elapsed=time.perf_counter() - t0,
    )


def reference_solution(dist: np.ndarray, time_limit_s: int = 3) -> SolveResult:
    """计算参考解（gap 的分母）。优先 OR-Tools，失败则退到 2-opt。"""
    t0 = time.perf_counter()
    tour = solve_ortools(dist, time_limit_s=time_limit_s)
    if tour is None:
        method = "2opt(fallback)"
        tour = two_opt(greedy_best(dist), dist)
    else:
        method = "ortools"
    return SolveResult(
        method=method,
        tour=tour,
        length=tour_length(tour, dist),
        elapsed=time.perf_counter() - t0,
    )
