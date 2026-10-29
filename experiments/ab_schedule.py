"""学习率调度对比实验：分段线性 vs 余弦退火。

固定同一实例、同一 seed、同一回合数，只改 alpha 调度方式。
关注两个量：
    best_len        训练中找到的最好解（采样偶然撞到的）
    策略解码         多起点贪心解码的结果

用法：
    .venv\\Scripts\\python.exe scripts\\ab_schedule.py --schedule cosine
    .venv\\Scripts\\python.exe scripts\\ab_schedule.py --schedule linear
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DQNConfig, InstanceConfig  # noqa: E402
from core.instance import generate  # noqa: E402
from train.train_dqn import sched_alpha, train_dqn  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--schedule", default="cosine", choices=["linear", "cosine", "const"])
    ap.add_argument("--alpha-end", type=float, default=0.0005)
    ap.add_argument("--use-dist", dest="use_dist", action="store_true", default=True,
                    help="状态含距离特征（3N 维，默认开）")
    ap.add_argument("--no-dist", dest="use_dist", action="store_false",
                    help="状态不含距离特征（2N 维，原版）")
    ap.add_argument("--n-traj", type=int, default=8)
    ap.add_argument("--alpha-schedule-only", action="store_true",
                    help="只打印调度曲线，不训练")
    args = ap.parse_args()

    # 只看调度曲线
    if args.alpha_schedule_only:
        print(f"alpha 调度对比  a0=0.02 → a1={args.alpha_end}  共 {args.episodes} 回合")
        print()
        print(f"{'回合':>7}  {'linear':>10}  {'cosine':>10}  {'const':>10}")
        print("-" * 44)
        for ep in range(0, args.episodes + 1, args.episodes // 10):
            a = sched_alpha(ep, args.episodes, 0.02, args.alpha_end, 1/3, "linear")
            b = sched_alpha(ep, args.episodes, 0.02, args.alpha_end, 1/3, "cosine")
            c = sched_alpha(ep, args.episodes, 0.02, args.alpha_end, 1/3, "const")
            print(f"{ep:>7}  {a:>10.6f}  {b:>10.6f}  {c:>10.6f}")
        return 0

    from core.baselines import reference_solution

    inst = generate(InstanceConfig(n_cities=args.n, distribution="uniform", seed=args.seed))
    ref = reference_solution(inst.dist, time_limit_s=3)

    cfg = DQNConfig(
        n_cities=args.n,
        episodes=args.episodes,
        early_stop=args.episodes,
        n_traj=args.n_traj,
        device="auto",
        seed=args.seed,
        alpha_schedule=args.schedule,
        alpha_end=args.alpha_end,
        decode_starts=None,  # 全部起点
        use_dist_feature=args.use_dist,
    )

    result = train_dqn(inst, cfg, ref_len=ref.length, show_progress=False)

    out = {
        "schedule": args.schedule,
        "alpha_end": args.alpha_end,
        "use_dist": args.use_dist,
        "state_dim": (3 if args.use_dist else 2) * args.n,
        "n_traj": args.n_traj,
        "n": args.n,
        "episodes": args.episodes,
        "seed": args.seed,
        "best_len": result["best_len"],
        "best_gap": result.get("gap_best"),
        "decode_len": result["greedy_len"],
        "decode_gap": result.get("gap_greedy"),
        "ref_len": result["ref_len"],
        "elapsed": result["elapsed"],
        "decode_starts": result.get("decode_starts"),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))

    # 追加到结果文件，便于多次实验汇总
    res_file = Path(__file__).resolve().parent.parent / "data" / "ab_schedule.json"
    res_file.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(res_file.read_text(encoding="utf-8")) if res_file.exists() else []
    old = [x for x in old if not (x["schedule"] == out["schedule"]
                                   and x["alpha_end"] == out["alpha_end"]
                                   and x.get("use_dist", True) == out["use_dist"]
                                   and x["n"] == out["n"]
                                   and x["episodes"] == out["episodes"]
                                   and x["seed"] == out["seed"])]
    old.append(out)
    res_file.write_text(json.dumps(old, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已追加到 {res_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())