"""命令行入口：不启动 WebUI 也能验证各模块。

用法示例：
    py cli.py bench --n 20 --dist uniform --seed 42
    py cli.py bench --n 50 --dist cluster --seed 7 --ortools-time 15
    py cli.py sweep --n 20 --runs 5
"""

from __future__ import annotations

import config  # noqa: F401  必须第一位，负责设置 HIP_VISIBLE_DEVICES

import argparse
import sys

import numpy as np

from config import InstanceConfig
from core.baselines import reference_solution, solve
from core.instance import generate
from core.metrics import gap, is_valid_tour, tour_length


def cmd_bench(args: argparse.Namespace) -> int:
    cfg = InstanceConfig(
        n_cities=args.n, distribution=args.dist, seed=args.seed
    )
    inst = generate(cfg)
    print(f"\n实例: {inst.name}  城市数={inst.n}")
    print("=" * 62)

    rows = []
    for method in ("greedy", "two_opt", "ortools"):
        r = solve(method, inst.dist, time_limit_s=args.ortools_time)
        valid = is_valid_tour(r.tour, inst.n)
        rows.append((r.method, r.length, r.elapsed, valid))
        print(f"  {r.method:<18} 长度={r.length:9.4f}  耗时={r.elapsed:7.3f}s  合法={valid}")

    ref = min(r[1] for r in rows)
    print("-" * 62)
    for name, length, elapsed, valid in rows:
        print(f"  {name:<18} gap={gap(length, ref):6.2f}%")

    print("\n校验：")
    greedy_len = next(r[1] for r in rows if r[0] == "greedy")
    twoopt_len = next(r[1] for r in rows if r[0] == "two_opt")
    checks = [
        ("所有路径合法", all(r[3] for r in rows)),
        ("2-opt 不劣于 greedy", twoopt_len <= greedy_len + 1e-9),
        ("OR-Tools 不劣于 2-opt", ref <= twoopt_len + 1e-9),
    ]
    ok = True
    for label, passed in checks:
        print(f"  [{'OK' if passed else 'FAIL'}] {label}")
        ok = ok and passed

    return 0 if ok else 1


def cmd_sweep(args: argparse.Namespace) -> int:
    """多随机种子跑一轮，看基准解与 gap 的稳定性。"""
    print(f"\n随机实例扫描：N={args.n}, 分布={args.dist}, 次数={args.runs}")
    print("=" * 70)
    header = f"{'seed':>6} {'greedy':>10} {'2-opt':>10} {'ortools':>10} {'g→2o':>8} {'2o→opt':>8}"
    print(header)
    print("-" * 70)

    stats = []
    for seed in range(args.runs):
        inst = generate(
            InstanceConfig(n_cities=args.n, distribution=args.dist, seed=seed)
        )
        g = tour_length(solve("greedy", inst.dist).tour, inst.dist)
        t = tour_length(solve("two_opt", inst.dist).tour, inst.dist)
        o = reference_solution(inst.dist, time_limit_s=args.ortools_time).length
        stats.append((g, t, o))
        print(
            f"{seed:>6} {g:>10.4f} {t:>10.4f} {o:>10.4f} "
            f"{gap(t, g):>7.2f}% {gap(o, t):>7.2f}%"
        )

    arr = np.array(stats)
    print("-" * 70)
    print(f"平均: greedy={arr[:,0].mean():.4f}  2opt={arr[:,1].mean():.4f}  "
          f"ortools={arr[:,2].mean():.4f}")
    print(f"2-opt 相对 greedy 平均改进: {np.mean((arr[:,0]-arr[:,1])/arr[:,0])*100:.2f}%")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """在命令行跑一次 DQN 训练，不启动 WebUI。"""
    from config import DQNConfig
    from core.baselines import two_opt
    from train.train_dqn import train_dqn

    inst = generate(
        InstanceConfig(n_cities=args.n, distribution=args.dist, seed=args.seed)
    )
    print(f"\n实例: {inst.name}  城市数={inst.n}")

    print("计算 OR-Tools 参考解 ...")
    ref = reference_solution(inst.dist, time_limit_s=args.ortools_time)
    print(f"参考解: {ref.length:.6f}  ({ref.elapsed:.2f}s, {ref.method})")

    cfg = DQNConfig(
        n_cities=args.n,
        episodes=args.episodes,
        early_stop=args.early_stop,
        device=args.device,
        seed=args.seed,
        n_traj=args.n_traj,
    )
    print(f"训练: episodes={cfg.episodes}  n_traj={cfg.n_traj}  device={cfg.device}\n")

    result = train_dqn(
        inst, cfg, ref_len=ref.length, show_progress=True
    )

    # 对 DQN 最优路径做 2-opt 后处理
    dqn_tour = np.asarray(result["best_tour"], dtype=np.int64)
    opt_tour = two_opt(dqn_tour, inst.dist) if dqn_tour.size else dqn_tour
    opt_len = tour_length(opt_tour, inst.dist)

    print("\n" + "=" * 62)
    print(f"  停止原因      : {result['stop_reason']}")
    print(f"  实际回合      : {result['episodes_run']}")
    print(f"  耗时          : {result['elapsed']:.1f}s")
    print(f"  运行设备      : {result['device']}")
    print("-" * 62)
    print(f"  DQN 最优      : {result['best_len']:9.4f}   gap={gap(result['best_len'], ref.length):6.2f}%")
    print(f"  DQN 贪心解码  : {result['greedy_len']:9.4f}   gap={gap(result['greedy_len'], ref.length):6.2f}%")
    print(f"  2-opt 后处理  : {opt_len:9.4f}   gap={gap(opt_len, ref.length):6.2f}%")
    print(f"  参考解        : {ref.length:9.4f}")
    print(f"  2-opt 改进    : {(result['best_len'] - opt_len) / result['best_len'] * 100:.2f}%")
    print("=" * 62)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TSP RL 项目命令行工具")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_bench = sub.add_parser("bench", help="单实例基准对比")
    p_bench.add_argument("--n", type=int, default=20)
    p_bench.add_argument("--dist", default="uniform", choices=["uniform", "cluster", "grid"])
    p_bench.add_argument("--seed", type=int, default=42)
    p_bench.add_argument("--ortools-time", type=int, default=3)
    p_bench.set_defaults(func=cmd_bench)

    p_sweep = sub.add_parser("sweep", help="多随机种子扫描")
    p_sweep.add_argument("--n", type=int, default=20)
    p_sweep.add_argument("--dist", default="uniform", choices=["uniform", "cluster", "grid"])
    p_sweep.add_argument("--runs", type=int, default=5)
    p_sweep.add_argument("--ortools-time", type=int, default=2)
    p_sweep.set_defaults(func=cmd_sweep)

    p_train = sub.add_parser("train", help="训练 DQN（命令行模式）")
    p_train.add_argument("--n", type=int, default=20)
    p_train.add_argument("--dist", default="uniform", choices=["uniform", "cluster", "grid"])
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--episodes", type=int, default=2000)
    p_train.add_argument("--early-stop", type=int, default=800)
    p_train.add_argument("--n-traj", type=int, default=8)
    p_train.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p_train.add_argument("--ortools-time", type=int, default=2)
    p_train.set_defaults(func=cmd_train)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
