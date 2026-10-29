"""逐起点 2-opt 后取最优 —— 验证「每条候选都做局部优化再选优」的价值。

背景
----
多起点 argmax 解码后统一做一次 2-opt，只优化了「最好的那条原始路径」。
但不同起点的路径结构质量差异很大：单起点 argmax 的原始 gap 高达 21%，
2-opt 之后却能到 0.00% —— 比多起点最优路径 2-opt 后的 0.57% 还低。

这说明：**先选优再优化，会丢掉那些原始长度不占优但结构更优的解。**
本脚本改为「每个起点各自 2-opt，再从优化后的结果里选最优」。

用法:
    .venv\\Scripts\\python.exe scripts\\per_start_2opt.py --ckpt data\\runs\\<job_id>.pt
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import DB_PATH, DQNConfig, InstanceConfig  # noqa: E402
from core.baselines import two_opt  # noqa: E402
from core.instance import generate  # noqa: E402
from core.metrics import gap as gap_fn  # noqa: E402
from core.metrics import tour_length  # noqa: E402
from models.dqn import DQNAgent  # noqa: E402


def parse_instance(name: str) -> tuple[int, str, int]:
    m = re.match(r"^([a-z]+)-n(\d+)-(?:s(\d+)|rand)$", name)
    if not m:
        raise SystemExit(f"无法解析实例名: {name}")
    return int(m.group(2)), m.group(1), int(m.group(3)) if m.group(3) else 0


@torch.no_grad()
def all_start_tours(agent: DQNAgent, dist: torch.Tensor, n: int) -> np.ndarray:
    """从每个起点各做一次 argmax 解码，返回 (n, n+1) 的 tours。"""
    device = dist.device
    cur = torch.arange(n, device=device)
    visited = torch.zeros(n, n, dtype=torch.bool, device=device)
    visited[torch.arange(n, device=device), cur] = True
    tours = torch.zeros(n, n, dtype=torch.long, device=device)
    tours[:, 0] = cur

    for step in range(1, n):
        q = agent.online(agent.make_state(cur, visited)).masked_fill(visited, float("-inf"))
        action = q.argmax(dim=1)
        tours[:, step] = action
        visited[torch.arange(n, device=device), action] = True
        cur = action

    tours = torch.cat([tours, tours[:, :1]], dim=1)
    return tours.cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ref", type=float, default=None)
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM runs WHERE ckpt LIKE ? OR job_id = ? ORDER BY created_at DESC LIMIT 1",
        (f"%{ckpt.name}", ckpt.stem),
    ).fetchone()
    if row is None:
        print(f"数据库里找不到: {ckpt.name}")
        return 1

    inst_name = row["instance"]
    ref_len = args.ref if args.ref is not None else row["ref_len"]
    n, dist_name, seed = parse_instance(inst_name)
    inst = generate(InstanceConfig(n_cities=n, distribution=dist_name, seed=seed))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(ckpt, map_location=device)
    input_dim = state["net.0.weight"].shape[1]
    use_dist = input_dim == 3 * n

    cfg = DQNConfig(n_cities=n, device=device, use_dist_feature=use_dist)
    agent = DQNAgent(n, cfg, device=device)
    agent.online.load_state_dict(state)
    agent.target.load_state_dict(state)
    agent.online.eval()
    dist = torch.as_tensor(inst.dist.astype(np.float32), device=device)
    if use_dist:
        agent.bind_dist(dist)

    print("=" * 78)
    print(f"  逐起点 2-opt 对比   实例={inst_name}   N={n}   状态={'3N' if use_dist else '2N'}")
    print(f"  OR-Tools 基准 = {ref_len:.4f}    训练中最优 = {row['best_len']:.4f}")
    print("=" * 78)

    t0 = time.perf_counter()
    tours = all_start_tours(agent, dist, n)
    t_decode = (time.perf_counter() - t0) * 1000

    raw_lens = np.array([tour_length(t, inst.dist) for t in tours])

    # 方案一：先选优再优化（当前实现）
    t0 = time.perf_counter()
    best_i = int(raw_lens.argmin())
    t1 = two_opt(tours[best_i], inst.dist)
    l1 = tour_length(t1, inst.dist)
    t_opt1 = (time.perf_counter() - t0) * 1000

    # 方案二：每条各自优化再选优
    t0 = time.perf_counter()
    opt_lens = np.zeros(n)
    opt_tours = []
    for i in range(n):
        ti = two_opt(tours[i], inst.dist)
        opt_tours.append(ti)
        opt_lens[i] = tour_length(ti, inst.dist)
    t_opt2 = (time.perf_counter() - t0) * 1000
    best_j = int(opt_lens.argmin())

    print(f"{'方案':<34}{'长度':>11}{'gap':>10}{'耗时':>10}")
    print("-" * 78)
    print(f"{'原始多起点最优（未优化）':<34}{raw_lens[best_i]:>11.4f}"
          f"{gap_fn(raw_lens[best_i], ref_len):>9.2f}%{t_decode:>9.1f}ms")
    print(f"{'方案一 先选优再 2-opt':<34}{l1:>11.4f}{gap_fn(l1, ref_len):>9.2f}%{t_opt1:>9.1f}ms")
    print(f"{'方案二 各自 2-opt 后选优':<34}{opt_lens[best_j]:>11.4f}"
          f"{gap_fn(opt_lens[best_j], ref_len):>9.2f}%{t_opt2:>9.1f}ms")
    print("-" * 78)

    gain = (l1 - opt_lens[best_j]) / l1 * 100
    print(f"方案二相对方案一 额外改善: {gain:.2f}%")
    print()
    print(f"每起点 2-opt 后 gap 分布（前 8 名）:")
    order = np.argsort(opt_lens)
    for k in order[:8]:
        raw_g = gap_fn(raw_lens[k], ref_len)
        print(f"   起点 {k:>3}   原始 {raw_lens[k]:.4f} (gap {raw_g:>6.2f}%)"
              f"  →  2-opt {opt_lens[k]:.4f} (gap {gap_fn(opt_lens[k], ref_len):>6.2f}%)")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
