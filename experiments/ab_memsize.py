"""mem_size A/B 实验：同实例同参数，只改 mem_size（30000 → 300000）。

对照臂 = 2026-10-27 19:31:49 那轮 WebUI 记录（mem_size=30000，gap 7.57%），
本脚本跑 300000 臂，其余参数逐项复制。

用法：
    .venv/Scripts/python.exe scripts/ab_memsize.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8000"


def post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=120).read())


def get(path: str) -> dict:
    return json.loads(urllib.request.urlopen(BASE + path, timeout=30).read())


async def watch(job_id: str, timeout_s: float = 1800) -> list[dict]:
    try:
        import websockets
    except ImportError:
        raise SystemExit("需要 websockets 包")

    msgs: list[dict] = []
    uri = f"ws://127.0.0.1:8000/api/ws/train/{job_id}"
    async with websockets.connect(uri, max_size=2**22) as ws:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout_s:
            raw = await asyncio.wait_for(ws.recv(), timeout=timeout_s)
            msg = json.loads(raw)
            msgs.append(msg)
            if msg.get("type") in ("done", "error", "stopped"):
                break
    return msgs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mem-size", type=int, default=300_000)
    ap.add_argument("--lr-warmup", type=int, default=100)
    ap.add_argument("--sb", action="store_true", help="启用 POMO 共享 baseline")
    ap.add_argument("--ddqn", action="store_true", help="启用 Double DQN")
    ap.add_argument("--eps-start", type=float, default=0.9)
    ap.add_argument("--schedule", default="cosine", choices=["linear", "cosine", "const", "plateau"])
    ap.add_argument("--plateau-patience", type=int, default=300)
    ap.add_argument("--early-stop", type=int, default=800)
    ap.add_argument("--n", type=int, default=35)
    ap.add_argument("--seed", type=int, default=42, help="实例种子与训练种子（复测噪声带时换）")
    ap.add_argument("--episodes", type=int, default=2000)
    args = ap.parse_args()

    inst = post("/api/instance", {"n_cities": args.n, "distribution": "uniform", "seed": args.seed})
    print(f"实例 {inst['name']}（id={inst['id']}）")

    # 逐项复制 19:31:49 那轮 WebUI 参数，唯一差异 mem_size
    params = {
        "episodes": args.episodes,
        "early_stop": args.early_stop,
        "n_traj": 8,
        "gamma": 0.95,
        "alpha_start": 0.02,
        "alpha_end": 0.0005,
        "alpha_schedule": args.schedule,
        "plateau_patience": args.plateau_patience,
        "eps_start": args.eps_start,
        "eps_end": 0.001,
        "batch_size": 1024,
        "mem_size": args.mem_size,
        "tau": 0.005,
        "device": "auto",
        "seed": args.seed,
        "report_every": 10,
        "decode_starts": None,
        "use_dist_feature": True,
        "train_mode": "bc_rl",
        "bc_variant": "B",
        "bc_instances": 100,
        "bc_permutations": 128,
        "bc_epochs": 30,
        "bc_w_q": 0.1,
        "bc_restarts": 100,
        "bc_lr_warmup_epochs": 0,
        "lr_warmup_episodes": args.lr_warmup,
        "shared_baseline": args.sb,
        "double_dqn": args.ddqn,
    }

    start = post("/api/train/start", {"instance_id": inst["id"], "params": params, "time_limit_s": 2})
    job_id = start["job_id"]
    print(f"job={job_id}  mem={args.mem_size} warmup={args.lr_warmup} eps_start={args.eps_start} early_stop={args.early_stop}  ref={start['ref_len']:.4f}")

    msgs = asyncio.run(watch(job_id))

    counts: dict[str, int] = {}
    for m in msgs:
        counts[m.get("type", "?")] = counts.get(m.get("type", "?"), 0) + 1
        if m.get("type") == "metric" and (m.get("ep", 0) % 500 == 0):
            print(f"  ep {m['ep']:>5}: best {m.get('best_len')}  gap {m.get('gap')}%")

    res = get(f"/api/runs/{job_id}/result")
    print()
    print("========== 结果 ==========")
    print("消息统计:", counts)
    print(f"best      {res.get('best_len')}  gap_best {res.get('gap_best')}%")
    print(f"greedy    {res.get('greedy_len')}  gap_greedy {res.get('gap_greedy')}%")
    print(f"cand2opt  {res.get('cand2opt_len')}  gap {res.get('gap_cand2opt')}%")
    print(f"用时 {res.get('elapsed')}s")


if __name__ == "__main__":
    main()
