"""推理策略对比：argmax 解码 vs 采样解码 vs 2-opt 后处理。

动机
----
训练中 ``best_len`` 来自 ε-greedy **采样**轨迹的偶然好解，
而「当前策略」是 argmax 贪心解码 —— 两者常常不等，
说明模型"见过"更好的解但 argmax 走不出来。

本脚本用同一份权重对比三种推理方式：
    A. 单起点 argmax
    B. 多起点 argmax（POMO 思路）
    C. 多起点 + 采样（推理时保留随机性）

用法:
    .venv\\Scripts\\python.exe scripts\\decode_strategies.py --ckpt data\\runs\\<job_id>.pt
"""

from __future__ import annotations

import argparse
import json
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
    """uniform-n25-s42 -> (25, 'uniform', 42)"""
    m = re.match(r"^([a-z]+)-n(\d+)-(?:s(\d+)|rand)$", name)
    if not m:
        raise SystemExit(f"无法解析实例名: {name}")
    dist = m.group(1)
    n = int(m.group(2))
    seed = int(m.group(3)) if m.group(3) else 0
    return n, dist, seed


def load_agent(ckpt: Path, n: int, device: str) -> tuple[DQNAgent, torch.Tensor]:
    """按权重形状自动判定 2N / 3N，并绑定距离矩阵。"""
    state = torch.load(ckpt, map_location=device)
    input_dim = state["net.0.weight"].shape[1]
    if input_dim == 3 * n:
        use_dist = True
    elif input_dim == 2 * n:
        use_dist = False
    else:
        raise SystemExit(f"权重输入维度异常: {input_dim}，N={n}")

    cfg = DQNConfig(n_cities=n, device=device, use_dist_feature=use_dist)
    agent = DQNAgent(n, cfg, device=device)
    agent.online.load_state_dict(state)
    agent.target.load_state_dict(state)
    agent.online.eval()
    return agent, use_dist


@torch.no_grad()
def sample_rollouts(
    agent: DQNAgent,
    dist: torch.Tensor,
    n: int,
    n_rollouts: int,
    eps: float,
    temperature: float,
    seed: int = 0,
) -> tuple[float, np.ndarray]:
    """推理时采样：每条 rollout 用 ε-greedy + 温度 softmax 选动作。

    全部 rollout 一次性并行推进（batch 维 = n_rollouts）。
    """
    device = dist.device
    # randint/rand 只接受 CPU generator，multinomial 要求与张量同设备，
    # 所以拆成两个生成器；numpy 负责起点，torch 负责 multinomial。
    g_cpu = torch.Generator(device="cpu")
    g_cpu.manual_seed(seed)
    g_dev = torch.Generator(device=device)
    g_dev.manual_seed(seed)

    b = n_rollouts
    # 每个 rollout 一个随机起点（长度必须正好是 b）
    cur = torch.randint(0, n, (b,), generator=g_cpu).to(device)
    visited = torch.zeros(b, n, dtype=torch.bool, device=device)
    visited[torch.arange(b, device=device), cur] = True
    tours = torch.zeros(b, n, dtype=torch.long, device=device)
    tours[:, 0] = cur

    for step in range(1, n):
        q = agent.online(agent.make_state(cur, visited)).masked_fill(visited, float("-inf"))
        # 温度 softmax（ε=0 时退化为 argmax）
        probs = torch.softmax(q / max(temperature, 1e-6), dim=1)
        rand = torch.rand(b, 1, generator=g_cpu).to(device)
        greedy = q.argmax(dim=1, keepdim=True)
        sampled = torch.multinomial(probs, 1, generator=g_dev).to(device)
        use_sample = (rand < eps)
        action = torch.where(use_sample, sampled, greedy).squeeze(1)
        tours[:, step] = action
        visited[torch.arange(b, device=device), action] = True
        cur = action

    # 闭环：补回起点
    tours = torch.cat([tours, tours[:, :1]], dim=1)
    lens = dist[tours[:, :-1], tours[:, 1:]].sum(dim=1)
    best_i = int(lens.argmin().item())
    return float(lens[best_i].item()), tours[best_i].cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--instance", default=None, help="默认从数据库按权重反查")
    ap.add_argument("--ref", type=float, default=None, help="参考解长度，默认查库/调 OR-Tools")
    ap.add_argument("--rollouts", type=int, default=64)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        print(f"找不到权重文件: {ckpt}")
        return 1

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM runs WHERE ckpt LIKE ? OR job_id = ? ORDER BY created_at DESC LIMIT 1",
        (f"%{ckpt.name}", ckpt.stem),
    ).fetchone()
    if row is None:
        print(f"数据库里找不到权重对应的记录: {ckpt.name}")
        return 1

    inst_name = args.instance or row["instance"]
    ref_len = args.ref if args.ref is not None else row["ref_len"]

    n, dist_name, seed = parse_instance(inst_name)
    inst = generate(InstanceConfig(n_cities=n, distribution=dist_name, seed=seed))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent, use_dist = load_agent(ckpt, n, device)
    dist = torch.as_tensor(inst.dist.astype(np.float32), device=device)
    if use_dist:
        agent.bind_dist(dist)

    if ref_len is None:
        from core.baselines import reference_solution

        ref_len = reference_solution(inst.dist, time_limit_s=3).length

    print("=" * 74)
    print(f"  推理策略对比   实例={inst_name}   N={n}   状态={'3N' if use_dist else '2N'}")
    print(f"  OR-Tools 基准 = {ref_len:.4f}")
    print(f"  训练中最优     = {row['best_len']:.4f}  (gap {gap_fn(row['best_len'], ref_len):.2f}%)")
    print("=" * 74)
    print(f"{'策略':<28}{'路径长度':>12}{'gap':>10}{'2-opt后gap':>12}{'耗时':>10}")
    print("-" * 74)

    rows = []

    # A. 单起点 argmax
    t0 = time.perf_counter()
    tour_a = agent.greedy_tour(dist, start=0)
    len_a = tour_length(tour_a, inst.dist)
    dt_a = (time.perf_counter() - t0) * 1000
    l2a = tour_length(two_opt(tour_a, inst.dist), inst.dist)
    print(f"{'A 单起点 argmax':<28}{len_a:>12.4f}{gap_fn(len_a, ref_len):>9.2f}%"
          f"{gap_fn(l2a, ref_len):>11.2f}%{dt_a:>9.1f}ms")
    rows.append(("A 单起点 argmax", len_a))

    # B. 多起点 argmax
    t0 = time.perf_counter()
    tour_b, len_b, used = agent.decode_multi_start(dist, n)
    dt_b = (time.perf_counter() - t0) * 1000
    l2b = tour_length(two_opt(tour_b, inst.dist), inst.dist)
    print(f"{f'B 多起点 argmax ({used}起点)':<28}{len_b:>12.4f}{gap_fn(len_b, ref_len):>9.2f}%"
          f"{gap_fn(l2b, ref_len):>11.2f}%{dt_b:>9.1f}ms")
    rows.append((f"B 多起点 argmax ({used})", len_b))

    # C. 采样解码
    for eps in (0.05, 0.1, 0.2, 0.3):
        t0 = time.perf_counter()
        len_c, tour_c = sample_rollouts(
            agent, dist, n, args.rollouts, eps, args.temperature
        )
        dt_c = (time.perf_counter() - t0) * 1000
        l2c = tour_length(two_opt(tour_c, inst.dist), inst.dist)
        print(f"{f'C 采样 eps={eps} x{args.rollouts}':<28}{len_c:>12.4f}{gap_fn(len_c, ref_len):>9.2f}%"
              f"{gap_fn(l2c, ref_len):>11.2f}%{dt_c:>9.1f}ms")
        rows.append((f"C 采样 eps={eps}", len_c))

    print("-" * 74)
    best_name, best_len = min(rows, key=lambda x: x[1])
    print(f"最优: {best_name}  {best_len:.4f}  (gap {gap_fn(best_len, ref_len):.2f}%)")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
