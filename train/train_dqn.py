"""Track A · DQN 训练主循环（实时轨）。

与 MATLAB 版的对应关系：
    generateTrajectory  → collect_trajectories（改成了批量并行）
    getLearningRate     → linear_alpha
    getExplorationRate  → cosine_epsilon
    trainStep           → DQNAgent.train_step（PyTorch 自动微分）
    updateTargetNet     → DQNAgent.soft_update
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from config import DQNConfig
from core.instance import Instance
from core.metrics import gap, tour_length
from models.dqn import DQNAgent, build_state
from train.replay import ReplayBuffer

EventCallback = Callable[[dict], None]


# ------------------------------------------------------------------ 调度器
def cosine_epsilon(ep: int, total: int, eps0: float, eps1: float) -> float:
    """余弦退火探索率。"""
    return eps1 + (eps0 - eps1) * (1.0 + math.cos(math.pi * ep / max(total, 1))) / 2.0


def linear_alpha(ep: int, total: int, a0: float, a1: float, ratio: float) -> float:
    """分段学习率：前 ratio 部分线性衰减，之后保持。"""
    boundary = max(total * ratio, 1.0)
    if ep <= boundary:
        frac = ep / boundary
        return a0 * (1.0 - frac) + a1 * frac
    return a1


def cosine_alpha(ep: int, total: int, a0: float, a1: float, ratio: float) -> float:
    """余弦退火学习率：全程从 a0 平滑降到 a1。

    与分段线性的区别：
        分段线性 —— 前 1/3 降到 a1，之后**恒定不动**（学习率完全静止）
        余弦退火 —— 全程平滑下降，前慢后快，终点恰好是 a1

    注意：余弦退火后期学习率趋近 0，若停滞发生在训练后期，这个调度
    反而会让停滞来得更早。是否有效取决于缺口在「学习率太大」还是
    「argmax 解码陷入局部最优」。
    """
    frac = min(max(ep / max(total, 1), 0.0), 1.0)
    return a1 + (a0 - a1) * (1.0 + math.cos(math.pi * frac)) / 2.0


def sched_alpha(
    ep: int, total: int, a0: float, a1: float, ratio: float, schedule: str = "linear"
) -> float:
    """学习率调度总入口。schedule ∈ {linear, cosine, const}。"""
    if schedule == "cosine":
        return cosine_alpha(ep, total, a0, a1, ratio)
    if schedule == "const":
        return a0
    raise ValueError(f"未知调度: {schedule}")
    return linear_alpha(ep, total, a0, a1, ratio)


# ------------------------------------------------------------------ 轨迹采集
def collect_trajectories(
    agent: DQNAgent,
    dist_t: torch.Tensor,
    n: int,
    n_traj: int,
    epsilon: float,
    dmean: float,
    gen: torch.Generator,
    shared_baseline: bool = False,
):
    """一次前向并行推进 n_traj 条轨迹，返回 (tours, lengths, 经验列表)。

    shared_baseline=True 时启用 POMO 式共享 baseline：轨迹级优势
    adv_i = (mean_len − len_i) / dmean / n_steps 均摊到该轨迹每一步奖励。
    """
    dev = dist_t.device
    b = int(n_traj)

    starts = torch.randint(0, n, (b,), device=dev, generator=gen)
    cur = starts.clone()
    visited = torch.zeros(b, n, dtype=torch.bool, device=dev)
    visited[torch.arange(b, device=dev), starts] = True

    tours = torch.zeros(b, n + 1, dtype=torch.long, device=dev)
    tours[:, 0] = starts
    lengths = torch.zeros(b, device=dev)

    rows = torch.arange(b, device=dev)
    buf: list[tuple] = []

    for step in range(1, n):
        state = agent.make_state(cur, visited)
        actions = agent.select_actions(cur, visited, dist_t, epsilon, generator=gen)

        step_dist = dist_t[cur, actions]
        reward = -step_dist / dmean
        lengths += step_dist

        next_visited = visited.clone()
        next_visited[rows, actions] = True
        next_state = agent.make_state(actions, next_visited)

        is_last = step == n - 1
        done = torch.full((b,), bool(is_last), dtype=torch.bool, device=dev)

        buf.append(
            (
                state.cpu().numpy(),
                actions.cpu().numpy(),
                reward.cpu().numpy(),
                next_state.cpu().numpy(),
                next_visited.cpu().numpy(),
                done.cpu().numpy(),
            )
        )

        tours[:, step] = actions
        cur = actions
        visited = next_visited

    # 回到起点闭合路径
    back = dist_t[cur, starts]
    lengths += back
    tours[:, n] = starts

    # POMO 式共享 baseline：把轨迹级优势均摊到每一步奖励上
    if shared_baseline:
        adv = (lengths.mean() - lengths) / dmean / float(n - 1)
        adv_np = adv.detach().cpu().numpy()
        buf = [
            (s, a, r + adv_np, ns, nv, d) for (s, a, r, ns, nv, d) in buf
        ]

    tours_np = tours.cpu().numpy()
    lengths_np = lengths.cpu().numpy()
    return tours_np, lengths_np, buf


# ------------------------------------------------------------------ 主循环
def train_dqn(
    instance: Instance,
    cfg: DQNConfig,
    ref_len: float | None = None,
    on_event: EventCallback | None = None,
    stop_event=None,
    show_progress: bool = False,
    ckpt_path: str | Path | None = None,
    init_state: dict | None = None,
) -> dict:
    """训练 DQN，返回结果字典。

    ref_len  : 参考解长度（OR-Tools），用于计算 gap；None 则不算 gap
    on_event : 每 report_every 回合被调用一次，用于 WebSocket 推送
    stop_event: threading.Event，置位则提前停止
    ckpt_path: 训练结束后把 online 网络权重存到该路径（.pt），None 则不存
    init_state: 预训练权重（{"online": state_dict}），加载后再开始 RL。
                用于「专家预热 → RL 微调」的两阶段流程。
    """
    t_start = time.perf_counter()

    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

    # 设备解析（必须在创建 generator 之前完成，二者 device 必须一致）
    from config import resolve_device

    device = resolve_device(cfg.device)
    n = instance.n

    gen = torch.Generator(device=device)
    if cfg.seed is not None:
        gen.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    dist_np = instance.dist.astype(np.float32)
    dist_t = torch.as_tensor(dist_np, dtype=torch.float32, device=device)
    dmean = float(dist_np.mean())

    agent = DQNAgent(n, cfg, device=device)

    # 加载专家预热权重（两阶段流程的第一阶段产物）。
    # online 与 target 必须同步加载，否则 target 还在随机初始化，
    # 前几百个回合的 TD target 全是噪声，会把预热权重冲掉。
    if init_state:
        try:
            agent.online.load_state_dict(init_state["online"])
            tgt = init_state.get("target")
            agent.target.load_state_dict(tgt if tgt is not None else agent.online.state_dict())
            agent.target.eval()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"加载预训练权重失败：{type(exc).__name__}: {exc}") from exc

    # 状态里若含距离特征（use_dist_feature=True），必须先绑定距离矩阵。
    # 漏掉这步不会报错，但状态会悄悄退化成 2N 维 —— 静默失效，最难查。
    if cfg.use_dist_feature:
        assert agent._dist_norm is None, "dist 绑定时机不对"
    agent.bind_dist(dist_t)
    if cfg.use_dist_feature and agent.state_dim != 3 * n:
        raise RuntimeError(f"状态维度应为 {3 * n}，实际 {agent.state_dim}")
    # 经验池状态维度必须跟随实际 state_dim（2N 或 3N），不能写死 2*n
    buffer = ReplayBuffer(cfg.mem_size, agent.state_dim, torch.device(device), n_cities=n)

    best_len = float("inf")
    best_tour: np.ndarray | None = None
    no_gain = 0
    loss_last = 0.0
    history: list[dict] = []
    stop_reason = "max_episodes"
    ep = 0
    policy_len: float | None = None
    policy_tour: np.ndarray | None = None
    # plateau 调度的当前 LR（状态ful：随无改进次数自适应衰减）
    plateau_alpha = cfg.alpha_start

    iterator = range(1, cfg.episodes + 1)
    if show_progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="DQN 训练", unit="ep", ncols=88)
        except ImportError:
            pass

    for ep in iterator:
        if stop_event is not None and stop_event.is_set():
            stop_reason = "stopped_by_user"
            break

        eps = cosine_epsilon(ep, cfg.episodes, cfg.eps_start, cfg.eps_end)
        if cfg.alpha_schedule == "plateau":
            # ReduceLROnPlateau 思路：LR 不随时间衰减，只在
            # 连续 plateau_patience 回合无改进时乘 factor 衰减。
            alpha = plateau_alpha
        else:
            alpha = sched_alpha(
                ep, cfg.episodes, cfg.alpha_start, cfg.alpha_end,
                cfg.alpha_decay_ratio, cfg.alpha_schedule,
            )
        # YOLO 式学习率预热（默认关闭，lr_warmup_episodes=0）。
        # 前若干回合把 α 从 0 线性升到 alpha_start，之后交回正常调度。
        # 作用是「保险栓」：万一预热权重一进 RL 就被大梯度冲掉，用它稳住。
        if cfg.lr_warmup_episodes > 0 and ep <= cfg.lr_warmup_episodes:
            alpha = cfg.alpha_start * ep / float(cfg.lr_warmup_episodes)
        agent.update_learning_rate(alpha)

        tours, lengths, exp = collect_trajectories(
            agent, dist_t, n, cfg.n_traj, eps, dmean, gen,
            shared_baseline=cfg.shared_baseline,
        )

        # 写入经验池（把各步的数组按列拼起来）
        if exp:
            stacked = [np.concatenate(x, axis=0) for x in zip(*exp)]
            buffer.push(*stacked)

        # 梯度更新
        if len(buffer) >= cfg.batch_size:
            for _ in range(cfg.train_iters_per_ep):
                batch = buffer.sample(cfg.batch_size, rng)
                loss_last = agent.train_step(
                    *batch, gamma=cfg.gamma, double_dqn=cfg.double_dqn
                )
            agent.soft_update(cfg.tau)

        # 更新最优解
        idx_best = int(np.argmin(lengths))
        this_best = float(lengths[idx_best])
        improved = this_best < best_len - 1e-12
        if improved:
            best_len = this_best
            best_tour = tours[idx_best].copy()
            no_gain = 0
        else:
            no_gain += 1
            # plateau 调度：连续无改进达到 patience 就降 LR 并重置计数。
            # 每次降 LR 重置 no_gain → early_stop 只有在 LR 已降到
            # alpha_end 且仍无改进时才会触发，不会误杀「降 LR 后复活」。
            if (
                cfg.alpha_schedule == "plateau"
                and no_gain >= cfg.plateau_patience
                and plateau_alpha > cfg.alpha_end + 1e-12
            ):
                plateau_alpha = max(
                    plateau_alpha * cfg.plateau_factor, cfg.alpha_end
                )
                no_gain = 0

        record = {
            "ep": ep,
            "best_len": round(best_len, 6),
            "ep_best_len": round(this_best, 6),
            "eps": round(eps, 5),
            "alpha": round(alpha, 7),
            "loss": round(loss_last, 6),
            "no_gain": no_gain,
        }
        if ref_len:
            record["gap"] = round(gap(best_len, ref_len), 3)
        history.append(record)

        # 当前策略路径：每回合用 online 网络贪心解码。
        # 多起点取最优（POMO 推理阶段思路）—— 训练时轨迹起点是随机的，
        # 单起点 argmax 容易锁进固定局部最优，多起点能显著改善。
        policy_tour, policy_len, n_used = agent.decode_multi_start(dist_t, cfg.decode_starts)
        if on_event is not None:
            on_event(
                {
                    "type": "policy",
                    "ep": ep,
                    "tour": [int(x) for x in policy_tour],
                    "len": round(policy_len, 6),
                    "starts": n_used,
                }
            )

        # 事件推送
        if on_event is not None and (ep % max(cfg.report_every, 1) == 0):
            payload = {
                "type": "metric",
                "ep": ep,
                "total": cfg.episodes,
                "best_len": round(best_len, 6),
                "eps": round(eps, 4),
                "alpha": round(alpha, 7),
                "loss": round(loss_last, 6),
                "elapsed": round(time.perf_counter() - t_start, 2),
                "improved": improved,
            }
            if ref_len:
                payload["gap"] = round(gap(best_len, ref_len), 3)
            if improved and best_tour is not None:
                payload["tour"] = [int(x) for x in best_tour]
            on_event(payload)

        # 早停
        if no_gain >= cfg.early_stop:
            stop_reason = "early_stop"
            break

    elapsed = time.perf_counter() - t_start

    # 最终策略解码（与训练中的实时紫线同口径，多起点）
    greedy, greedy_len, n_used = agent.decode_multi_start(dist_t, cfg.decode_starts)
    greedy = _canonical(greedy)

    # 多起点候选各自 2-opt 后选优 —— 整套流水线最强的展示口径
    # （探针实测：N=50 留出集 0.2% vs 策略解码 15%；知识来自规则算法，
    #   展示时要诚实标注「+2-opt」）
    cand2opt_tour, cand2opt_len = None, None
    try:
        from core.baselines import two_opt

        tours_c, lens_c = agent.decode_candidates(dist_t, cfg.decode_starts)
        best_l = float("inf")
        best_t2: np.ndarray | None = None
        for i in range(tours_c.shape[0]):
            t2 = two_opt(tours_c[i, :n], instance.dist)
            l2 = tour_length(t2, instance.dist)
            if l2 < best_l:
                best_l, best_t2 = l2, t2
        if best_t2 is not None:
            cand2opt_tour = _canonical(best_t2)
            cand2opt_len = float(best_l)
    except Exception:  # noqa: BLE001  后处理失败不影响主结果
        cand2opt_tour, cand2opt_len = None, None

    result = {
        "best_len": round(best_len, 6),
        "best_tour": [int(x) for x in best_tour] if best_tour is not None else [],
        "greedy_tour": [int(x) for x in greedy],
        "greedy_len": round(greedy_len, 6),
        "policy_tour": [int(x) for x in policy_tour] if policy_tour is not None else [],
        "policy_len": round(policy_len, 6) if policy_len is not None else None,
        "episodes_run": ep,
        "stop_reason": stop_reason,
        "elapsed": round(elapsed, 2),
        "device": device,
        "history": history,
        "n_cities": n,
        "instance": instance.name,
        "coords": instance.coords.tolist(),
    }
    if ref_len:
        result["ref_len"] = round(ref_len, 6)
        result["gap_best"] = round(gap(best_len, ref_len), 3)
        result["gap_greedy"] = round(gap(greedy_len, ref_len), 3)
        if cand2opt_len is not None:
            result["cand2opt_tour"] = [int(x) for x in cand2opt_tour]
            result["cand2opt_len"] = round(cand2opt_len, 6)
            result["gap_cand2opt"] = round(gap(cand2opt_len, ref_len), 3)
    result["decode_starts"] = n_used

    # 权重落盘。存 CPU 副本，避免加载时依赖 GPU 环境
    result["ckpt"] = None
    if ckpt_path is not None:
        try:
            ckpt = Path(ckpt_path)
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            state = {k: v.detach().cpu() for k, v in agent.online.state_dict().items()}
            torch.save(state, ckpt)
            result["ckpt"] = str(ckpt)
        except Exception as exc:  # noqa: BLE001
            # 权重保存失败不应影响训练结果本身
            result["ckpt_error"] = f"{type(exc).__name__}: {exc}"

    if on_event is not None:
        on_event({"type": "done", **{k: v for k, v in result.items() if k != "history"}})

    return result


def _canonical(tour: np.ndarray) -> np.ndarray:
    """把路径旋转到以 0 开头，便于对比与去重（TSP 是环路，起点无关）。"""
    t = np.asarray(tour, dtype=np.int64)
    if t.size == 0:
        return t
    k = int(np.argmin(t))
    return np.roll(t, -k)
