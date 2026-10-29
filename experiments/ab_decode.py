"""对比实验：单起点 vs 多起点贪心解码（POMO 思路）。

不改训练，只改推理，直接加载已保存的权重，因此几秒就能跑完。

用法：
    .venv\\Scripts\\python.exe scripts\\ab_decode.py --ckpt data\\runs\\10947137fe47.pt
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from config import DB_PATH, DQNConfig, InstanceConfig  # noqa: E402
from core.instance import generate  # noqa: E402
from core.metrics import gap as gap_fn  # noqa: E402
from core.metrics import tour_length  # noqa: E402
from models.dqn import DQNAgent  # noqa: E402


def resolve_device() -> str:
    import config  # noqa: F401  必须先导入以设置 HIP_VISIBLE_DEVICES

    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def parse_instance(name: str) -> tuple[int, str, int | None]:
    m = re.match(r"^([a-z]+)-n(\d+)-(?:s(\d+)|rand)$", name)
    if not m:
        raise ValueError(f"无法解析实例名: {name}")
    return int(m.group(2)), m.group(1), (int(m.group(3)) if m.group(3) else None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--instance", default="", help="默认从 ckpt 文件名推断")
    ap.add_argument("--ref", type=float, default=None, help="参考解长度")
    args = ap.parse_args()

    inst_name = args.instance
    best_sampled = None
    if not inst_name:
        import os
        import sqlite3

        want = os.path.basename(args.ckpt).lower()
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        row = None
        for cand in conn.execute("SELECT instance, ref_len, best_len, ckpt FROM runs WHERE ckpt IS NOT NULL ORDER BY created_at DESC"):
            if os.path.basename(cand["ckpt"] or "").lower() == want:
                row = cand
                break
        if row is None:
            print(f"找不到权重对应的记录: {args.ckpt}")
            print("提示：库中 ckpt 字段记录的是绝对路径，这里按文件名匹配。")
            return 1
        inst_name = row["instance"]
        if args.ref is None:
            args.ref = row["ref_len"]
        best_sampled = row["best_len"]

    n, dist_name, seed = parse_instance(inst_name)
    inst = generate(InstanceConfig(n_cities=n, distribution=dist_name, seed=seed))
    device = resolve_device()

    # 状态维度必须由权重形状反推，在建 agent 之前确定，
    # 否则 DQNAgent 会按默认配置建出维度不匹配的网络。
    state = torch.load(args.ckpt, map_location=device)
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

    dist_t = torch.as_tensor(inst.dist.astype(np.float32), device=device)
    if use_dist:
        agent.bind_dist(dist_t)
        print(f"[权重为 3N，已启用距离特征并绑定距离矩阵]")
    else:
        print(f"[权重为 2N，使用 2N 状态]")
    ref_len = args.ref
    if ref_len is None:
        from core.baselines import reference_solution

        ref_len = reference_solution(inst.dist, time_limit_s=3).length

    print("=" * 72)
    print(f"  解码对比实验   实例={inst_name}   N={n}")
    print(f"  参考解（OR-Tools）= {ref_len:.4f}")
    if best_sampled is not None:
        print(f"  训练中最优（采样）= {best_sampled:.4f}")
    print("=" * 72)
    print(f"{'起点数':>8}  {'最优解码':>12}  {'gap':>8}  {'耗时':>9}")
    print("-" * 72)

    for k in (1, 2, 4, 8, 16, n):
        t0 = time.perf_counter()
        tour, length, used = agent.decode_multi_start(dist_t, k)
        dt = time.perf_counter() - t0
        print(f"{used:>8}  {length:>12.4f}  {gap_fn(length, ref_len):>7.2f}%  {dt*1000:>7.1f}ms")

    print("-" * 72)
    t0 = time.perf_counter()
    t1 = agent.greedy_tour(dist_t, start=0)
    l1 = tour_length(t1, inst.dist)
    print(f"{'单点 start=0':>10}  {l1:>12.4f}  {gap_fn(l1, ref_len):>7.2f}%  "
          f"{(time.perf_counter()-t0)*1000:>7.1f}ms   ← 旧实现")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())