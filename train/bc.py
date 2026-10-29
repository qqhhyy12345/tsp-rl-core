"""专家预热（Behaviour Cloning / 专家 Q 回归）—— 在 RL 之前用监督信号推权重。

定位
----
DQN 在 N≥30 的瓶颈是**样本效率**（样本/参数比 6.1）。监督学习没有这个问题。
本模块用规则算法（rr2opt）的最优解当专家，生成 (状态 → 动作) 标签，
先把网络推到「至少会走一条像样的路」，再交给 RL 微调。

三个损失变体
------------
A. 专家 Q 回归
     Q(s, a*) ← −(专家轨迹上从 a* 走回起点的剩余代价) / dmean
     预训练的就是 DQN 需要的**同一个 Q**，量纲一致（都按 dmean 归一化），
     预热完直接微调，语义无缝。

B. 行为克隆 CE
     把「下一个城市」当类别做交叉熵。更像 YOLO 的分类预训练，
     但末层的语义从分类切回归，接 RL 时需要一个过渡。

C. 混合（默认）
     loss = SmoothL1(Q(s)[a*], q*) + λ · CE(masked_logits, a*)
     回归项给 Q 一个正确的初值，CE 项强化动作排序。

关于 YOLO 式学习率预热
---------------------
YOLO 的 warmup 是为了防 SGD + 大学习率在训练初期炸掉。
本项目用 Adam + 目标网络，通常不需要，所以默认关闭（warmup_epochs=0），
做成开关以便 A/B 决定。
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from config import BCConfig
from core.expert import ExpertDataset
from models.dqn import DQNAgent

EventCallback = Callable[[dict], None]


# ------------------------------------------------------------------ 学习率
def warmup_lr(epoch: int, base_lr: float, warmup_epochs: int) -> float:
    """YOLO 式线性预热：前 warmup_epochs 轮把 lr 从 0 升到 base_lr。

    epoch 从 1 开始计数。warmup_epochs <= 0 时不做预热。
    """
    if warmup_epochs <= 0:
        return base_lr
    if epoch >= warmup_epochs:
        return base_lr
    return base_lr * epoch / float(warmup_epochs)


# ------------------------------------------------------------------ 损失
def bc_loss(
    agent: DQNAgent,
    states: torch.Tensor,
    actions: torch.Tensor,
    q_targets: torch.Tensor,
    variant: str,
    w_q: float,
) -> tuple[torch.Tensor, dict]:
    """混合损失。

    states 的第 [n:2n] 段是「已访问掩码」（2N / 3N 两种模式下都在这个位置），
    直接从状态里取，不用额外存一份。

    w_q 是**回归项的权重**（CE 的权重固定为 1）。
    这个方向很重要：实测把回归权重设成 0.5 会让 CE 的排序信号被压死
    （留出集 gap 从 43% 恶化到 180%），所以默认 w_q=0.1。
    """
    n = agent.n

    q_all = agent.online(states)  # (B, N)
    q_sa = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)

    parts: dict[str, float] = {}
    loss = torch.zeros((), device=states.device)

    if variant in ("A", "C"):
        loss_reg = agent.loss_fn(q_sa, q_targets)
        loss = loss + (loss_reg if variant == "A" else w_q * loss_reg)
        parts["loss_q"] = float(loss_reg.item())

    if variant in ("B", "C"):
        visited = states[:, n : 2 * n] > 0.5
        # 用 -1e9 而不是 -inf：交叉熵对 -inf 的梯度可能出现 0*(-inf)=nan
        logits = q_all.masked_fill(visited, -1e9)
        loss_ce = F.cross_entropy(logits, actions)
        loss = loss + loss_ce
        parts["loss_ce"] = float(loss_ce.item())

    parts["loss"] = float(loss.item())
    return loss, parts


# ------------------------------------------------------------------ 主循环
def pretrain_bc(
    agent: DQNAgent,
    dataset: ExpertDataset,
    cfg: BCConfig,
    on_event: EventCallback | None = None,
    stop_event=None,
) -> dict:
    """预热训练。返回训练摘要（历史 / 最终损失 / 耗时）。

    注意：这里直接改 agent.online 的权重，调用方负责在之后
    把权重同步给 target 网络（soft_update 或直接 copy）。
    """
    dev = agent.device
    n = agent.n
    variant = (cfg.variant or "C").upper()

    states_np = dataset.states
    actions_np = dataset.actions
    q_np = dataset.q_targets

    # 训练集内部再切一份验证集（只用于监控过拟合，与最终测试集无关）
    rng = np.random.default_rng((cfg.seed or 0) + 77)
    idx = rng.permutation(states_np.shape[0])
    n_val = int(states_np.shape[0] * max(0.0, min(cfg.val_ratio, 0.5)))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    states = torch.as_tensor(states_np, dtype=torch.float32, device=dev)
    actions = torch.as_tensor(actions_np, dtype=torch.long, device=dev)
    q_targets = torch.as_tensor(q_np, dtype=torch.float32, device=dev)

    tr_s, tr_a, tr_q = states[tr_idx], actions[tr_idx], q_targets[tr_idx]
    va_s, va_a, va_q = states[val_idx], actions[val_idx], q_targets[val_idx]

    history: list[dict] = []
    stop_reason = "max_epochs"
    t0 = time.perf_counter()

    # 早停状态：验证 loss 连续 patience 轮不降则停，并回滚到最优权重。
    # 2026-10-27 实测（N=40 同实例）：BC 100 轮过拟合，best 比 50 轮差 8%，
    # 所以 epochs 只是上限，实际停止由早停决定。
    best_val: float | None = None
    best_epoch = 0
    best_state: dict | None = None
    no_gain = 0

    agent.online.train()
    agent.update_learning_rate(cfg.lr)

    for epoch in range(1, cfg.epochs + 1):
        if stop_event is not None and stop_event.is_set():
            stop_reason = "stopped_by_user"
            break

        lr = warmup_lr(epoch, cfg.lr, cfg.lr_warmup_epochs)
        agent.update_learning_rate(lr)

        # 训练
        perm = torch.randperm(tr_s.shape[0], device=dev)
        tot, nb = 0.0, 0
        parts_acc: dict[str, float] = {}
        for start in range(0, tr_s.shape[0], cfg.batch_size):
            b = perm[start : start + cfg.batch_size]
            loss, parts = bc_loss(
                agent, tr_s[b], tr_a[b], tr_q[b], variant, cfg.w_q
            )
            agent.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.online.parameters(), max_norm=10.0)
            agent.optimizer.step()
            tot += float(loss.item())
            nb += 1
            for k, v in parts.items():
                parts_acc[k] = parts_acc.get(k, 0.0) + v

        train_loss = tot / max(nb, 1)
        train_parts = {k: v / max(nb, 1) for k, v in parts_acc.items()}

        # 验证（不更新权重）
        val_loss = float("nan")
        if va_s.shape[0] > 0:
            agent.online.eval()
            with torch.no_grad():
                vl, _ = bc_loss(
                    agent, va_s, va_a, va_q, variant, cfg.w_q
                )
            val_loss = float(vl.item())
            agent.online.train()

        # ---- 早停判定与最优权重快照 ----
        if not np.isnan(val_loss):
            improved = (
                best_val is None
                or val_loss < best_val - cfg.early_stop_min_delta
            )
            if improved:
                best_val = val_loss
                best_epoch = epoch
                no_gain = 0
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in agent.online.state_dict().items()
                }
            else:
                no_gain += 1
                if (
                    cfg.early_stop_patience > 0
                    and no_gain >= cfg.early_stop_patience
                ):
                    stop_reason = "early_stop(val)"
                    break

        rec = {
            "epoch": epoch,
            "lr": round(lr, 8),
            "train_loss": round(train_loss, 6),
            "val_loss": None if np.isnan(val_loss) else round(val_loss, 6),
        }
        rec.update({k: round(v, 6) for k, v in train_parts.items()})
        history.append(rec)

        if on_event is not None:
            payload = {
                "type": "bc",
                "phase": "pretrain",
                "epoch": epoch,
                "total": cfg.epochs,
                "lr": round(lr, 8),
                "train_loss": round(train_loss, 6),
                "val_loss": rec["val_loss"],
                "best_epoch": best_epoch,
                "elapsed": round(time.perf_counter() - t0, 2),
            }
            on_event(payload)

    elapsed = time.perf_counter() - t0

    # 回滚到验证 loss 最低的那一轮权重（早停的核心价值：
    # 用户填 100 轮也只会拿到最优的那一轮，而不是过拟合的最后一轮）
    if best_state is not None:
        agent.online.load_state_dict(
            {k: v.to(agent.device) for k, v in best_state.items()}
        )
        agent.online.eval()

    return {
        "variant": variant,
        "epochs_run": len(history),
        "stop_reason": stop_reason,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "elapsed": round(elapsed, 2),
        "n_train": int(tr_s.shape[0]),
        "n_val": int(va_s.shape[0]),
        "n_samples": dataset.n_samples,
        "final_train_loss": history[-1]["train_loss"] if history else None,
        "final_val_loss": history[-1]["val_loss"] if history else None,
        "history": history,
    }
