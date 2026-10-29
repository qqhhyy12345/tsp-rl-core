r"""验证「当前策略路径」功能。

检查三件事：
    1. 训练过程中确实推送了 policy 事件
    2. 策略路径长度随训练推进而下降（否则可视化没有意义）
    3. policy 路径本身是一条合法环路

用法：.venv\Scripts\python.exe scripts\test_policy.py --n 15 --episodes 300
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DQNConfig, InstanceConfig  # noqa: E402
from core.instance import generate  # noqa: E402
from core.metrics import is_valid_tour, tour_length  # noqa: E402
from train.train_dqn import train_dqn  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    inst = generate(InstanceConfig(n_cities=args.n, distribution="uniform", seed=args.seed))
    cfg = DQNConfig(n_cities=args.n, episodes=args.episodes, early_stop=args.episodes,
                    n_traj=8, device="auto", seed=args.seed)

    # 参考解用于换算 gap
    from core.baselines import reference_solution

    ref = reference_solution(inst.dist, time_limit_s=2)
    ref_len = ref.length

    events: list[dict] = []
    result = train_dqn(inst, cfg, ref_len=ref_len, on_event=events.append, show_progress=False)

    policies = [e for e in events if e.get("type") == "policy"]
    print("=" * 62)
    print(f"总事件 {len(events)} 条，其中 policy {len(policies)} 条")
    print("=" * 62)

    if not policies:
        print("失败：没有收到任何 policy 事件")
        return 1

    # 抽样打印，看路径长度是否单调改善
    print(f"{'回合':>6}  {'策略长度':>10}  {'相对基准':>10}  路径点数")
    print("-" * 62)
    step = max(1, len(policies) // 8)
    for e in policies[::step][:9]:
        L = e["len"]
        rel = (L - ref_len) / ref_len * 100
        print(f"{e['ep']:>6}  {L:>10.4f}  {rel:>9.2f}%  {len(e['tour'])}")

    first = policies[0]["len"]
    last = policies[-1]["len"]
    print("-" * 62)
    print(f"首回合策略长度 {first:.4f}  →  末回合 {last:.4f}   改善 {first - last:+.4f}")

    # 合法性校验
    ok = is_valid_tour(policies[-1]["tour"], args.n)
    print(f"末回合路径是否合法环路: {ok}")

    # 与训练最优对比
    print(f"训练最优 {result['best_len']:.4f}   最终策略 {last:.4f}   "
          f"差 {last - result['best_len']:+.4f}")

    improved = last < first
    print()
    print("结论:", "通过 —— 策略路径随训练演化，且是合法环路" if (improved and ok)
          else "未通过")
    return 0 if (improved and ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
