"""端到端冒烟测试：实例 → 基准求解 → 训练 → WebSocket 推送。

用法（需先启动服务）：
    .venv\\Scripts\\python.exe scripts\\smoke_test.py
    .venv\\Scripts\\python.exe scripts\\smoke_test.py --n 15 --episodes 300
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8000"


def _req(method: str, path: str, payload: dict | None = None, timeout: int = 60):
    url = BASE + path
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {path} -> {e.code}: {body}") from None


def get(path: str, timeout: int = 60):
    return _req("GET", path, None, timeout)


def post(path: str, payload: dict, timeout: int = 120):
    return _req("POST", path, payload, timeout)


async def watch(job_id: str, max_wait_s: float = 300.0) -> dict:
    """连接 WebSocket，收集所有 metric 与 policy 路径，返回最后一帧。"""
    import websockets

    url = f"ws://127.0.0.1:8000/api/ws/train/{job_id}"
    metrics: list[dict] = []
    policies: list[dict] = []
    last: dict = {}

    async with websockets.connect(url, open_timeout=15) as ws:
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=max_wait_s)
            msg = json.loads(raw)
            kind = msg.get("type")
            last = msg

            if kind == "hello":
                print(f"  [WS] 已连接，实例={msg.get('instance')}，共 {msg.get('total')} 回合")
            elif kind == "metric":
                metrics.append(msg)
                if len(metrics) % 10 == 1:
                    gap_txt = f"  gap={msg['gap']:6.2f}%" if "gap" in msg else ""
                    print(
                        f"  [WS] ep={msg['ep']:>5}/{msg['total']}  "
                        f"best={msg['best_len']:.4f}{gap_txt}  "
                        f"eps={msg['eps']:.3f}  loss={msg['loss']:.5f}"
                    )
            elif kind == "policy":
                policies.append(msg)
            elif kind in ("done", "closed"):
                print(f"  [WS] 结束: {kind}  最终 best={msg.get('best_len')}")
                break
            elif kind == "error":
                print(f"  [WS] 错误: {msg.get('message')}")
                break

    last["_metric_count"] = len(metrics)
    last["_policy_count"] = len(policies)
    if policies:
        last["_policy_first"] = policies[0].get("len")
        last["_policy_last"] = policies[-1].get("len")
        last["_policy_points"] = len(policies[0].get("tour") or [])
    return last


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--dist", default="uniform")
    parser.add_argument("--ortools-time", type=int, default=2)
    args = parser.parse_args()

    print("=" * 66)
    print("  端到端冒烟测试")
    print("=" * 66)

    # 1 环境
    env = get("/api/env")
    print(f"\n[1/5] 环境: python={env['python']} torch={env['torch']} "
          f"rocm={env['rocm']} gpu={env.get('gpu')}")

    # 2 实例
    inst = post(
        "/api/instance",
        {"n_cities": args.n, "distribution": args.dist, "seed": args.seed},
    )
    print(f"[2/5] 实例: {inst['name']}  id={inst['id']}  城市数={inst['n_cities']}")
    assert len(inst["coords"]) == args.n, "坐标数量与城市数不符"

    # 3 基准求解
    print("[3/5] 基准求解:")
    results = {}
    for method in ("greedy", "two_opt", "ortools"):
        r = post(
            "/api/solve",
            {
                "instance_id": inst["id"],
                "method": method,
                "time_limit_s": args.ortools_time,
            },
            timeout=120,
        )
        results[method] = r
        print(f"        {method:<9} len={r['length']:.4f}  {r['elapsed']:.3f}s")

    ref = results["ortools"]["length"]
    assert results["two_opt"]["length"] <= results["greedy"]["length"] + 1e-9, \
        "2-opt 不应劣于贪心"

    # 4 启动训练
    start = post(
        "/api/train/start",
        {
            "instance_id": inst["id"],
            "params": {
                "episodes": args.episodes,
                "early_stop": max(args.episodes // 2, 100),
                "n_traj": 8,
                "device": "auto",
                "seed": args.seed,
                "report_every": 10,
            },
            "time_limit_s": args.ortools_time,
        },
        timeout=180,
    )
    job_id = start["job_id"]
    print(f"\n[4/5] 训练已启动: job_id={job_id}  "
          f"参考解={start['ref_len']:.4f} ({start['ref_method']})")

    # 5 WebSocket 观察
    print("[5/5] WebSocket 实时推送:")
    last = asyncio.run(watch(job_id, max_wait_s=max(args.episodes * 2, 120)))

    # 6 结果
    final = get(f"/api/train/status/{job_id}")
    print("\n" + "=" * 66)
    print(f"  状态      : {final['status']}")
    print(f"  完成回合  : {final['ep']}/{final['total']}")
    print(f"  最优长度  : {final['best_len']}")
    print(f"  gap       : {final['gap']}%")
    print(f"  耗时      : {final['elapsed']}s")
    print(f"  收到推送  : {last.get('_metric_count')} 帧 metric")
    print(f"  策略路径  : {last.get('_policy_count')} 帧 policy  "
          f"（首 {last.get('_policy_first')} → 末 {last.get('_policy_last')}，"
          f"{last.get('_policy_points')} 个点）")

    try:
        res = get(f"/api/runs/{job_id}/result", timeout=120)
        print(f"  2-opt 后  : {res.get('two_opt_len')}  gap={res.get('gap_two_opt')}%")
    except Exception as exc:  # noqa: BLE001
        print(f"  2-opt 结果获取失败: {exc}")

    print("=" * 66)
    ok = final["status"] in ("done", "stopped") and last.get("_metric_count", 0) > 0
    print("冒烟测试通过" if ok else "冒烟测试失败")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
