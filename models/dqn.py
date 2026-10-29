"""DQN 模型：Q 网络 + 智能体。

设计要点（与 PLAN.md §4.1 对应）：
    - 状态 2N 维：[当前城市 one-hot ‖ 已访问掩码]
    - 网络 2N → 256 → 256 → 128 → N，激活 tanh
    - 权重初始化必须手动设为 N(0, 0.1) + 零偏置，
      否则与 MATLAB 版收敛行为不可比（PyTorch 默认是 Kaiming Uniform）
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from config import DQNConfig


class QNet(nn.Module):
    """全连接 Q 网络。"""

    def __init__(self, state_dim: int, hidden: tuple[int, ...], n_actions: int):
        super().__init__()
        layers: list[nn.Module] = []
        prev = state_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.Tanh())
            prev = h
        layers.append(nn.Linear(prev, n_actions))
        self.net = nn.Sequential(*layers)
        self._manual_init()

    def _manual_init(self) -> None:
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_state(
    cur: torch.Tensor,
    visited: torch.Tensor,
    dist_norm: torch.Tensor | None = None,
) -> torch.Tensor:
    """构造批量状态矩阵。

    两种模式：
        dist_norm 为 None  → 2N 维：[one-hot 当前城市 ‖ 已访问掩码]
        dist_norm 有值    → 3N 维：追加「当前城市到各城市的归一化距离」，
                             已访问城市该维置 0，让模型能区分「不可选」与「很远」

    为什么加距离特征：2N 维时模型只能靠 one-hot 隐式逼近距离函数
    （实测与最近邻一致率 82.3%，说明学到了，但样本/参数仅 6.1）。
    显式给出距离 = 不必再学 N×N 的隐式映射，样本效率显著提升。
    这是 Kool 2019 / POMO 里 encoder 显式编码距离的等价做法。

    参数
    ----
    cur       : (B,) long，当前城市下标
    visited   : (B, N) bool
    dist_norm : (N, N) 已按均值归一化的距离矩阵，或 None
    """
    b = cur.size(0)
    n = visited.size(1)

    width = 2 * n if dist_norm is None else 3 * n
    s = torch.zeros(b, width, device=cur.device, dtype=torch.float32)
    s.scatter_(1, cur.unsqueeze(1), 1.0)
    s[:, n : 2 * n] = visited.to(torch.float32)

    if dist_norm is not None:
        # dist_norm[cur] → (B, N)，每行是当前城市到所有城市的距离
        dfeat = dist_norm[cur] / dist_norm.mean()
        s[:, 2 * n :] = dfeat.masked_fill(visited, 0.0)

    return s


class DQNAgent:
    """封装网络、目标网络、优化器与动作选择。"""

    def __init__(self, n_cities: int, cfg: DQNConfig, device: str = "cpu"):
        self.n = int(n_cities)
        self.cfg = cfg
        self.device = torch.device(device)
        # 3N 维（含距离特征）或 2N 维（原 MATLAB 版）
        self.state_dim = (3 if cfg.use_dist_feature else 2) * self.n
        self._dist_norm: torch.Tensor | None = None
        # CUDA 图解码缓存：bind_dist / load_weights 会使其失效
        self._decode_graph = None
        self._graph_disabled = False  # 捕获/校验失败后置位，避免每次调用重试
        self._g_cur: torch.Tensor | None = None
        self._g_vis: torch.Tensor | None = None
        self._g_tour: torch.Tensor | None = None

        self.online = QNet(self.state_dim, tuple(cfg.hidden), self.n).to(self.device)
        self.target = QNet(self.state_dim, tuple(cfg.hidden), self.n).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = torch.optim.Adam(
            self.online.parameters(), lr=cfg.alpha_start, betas=(0.9, 0.999), eps=1e-8
        )
        self.loss_fn = nn.SmoothL1Loss()

    # ------------------------------------------------------------ 状态构造
    def bind_dist(self, dist: torch.Tensor) -> None:
        """绑定距离矩阵。仅 use_dist_feature=True 时用到，训练开始时调用一次。

        换实例（不同 dist）必须重新调用本方法：CUDA 图里捕获了距离矩阵的
        指针，不失效重捕获的话会拿旧矩阵算出新路径 —— 静默错误。
        """
        self._dist_norm = dist
        self.invalidate_decode_graph()

    def invalidate_decode_graph(self) -> None:
        """使已捕获的解码图失效（下次 greedy_tour 时重新捕获）。"""
        self._decode_graph = None
        self._graph_disabled = False

    def make_state(self, cur: torch.Tensor, visited: torch.Tensor) -> torch.Tensor:
        """按当前配置构造状态。

        训练与推理都走这个入口，否则口径会走偏（比如训练用 3N、解码用 2N）。
        """
        d = self._dist_norm if self.cfg.use_dist_feature else None
        if self.cfg.use_dist_feature and d is None:
            raise RuntimeError(
                "use_dist_feature=True 但未调用 bind_dist()：状态会退化成 2N，"
                "与 3N 网络维度不匹配。训练入口 train_dqn 已自动绑定，"
                "独立加载权重时需手动 agent.bind_dist(dist)。"
            )
        return build_state(cur, visited, d)

    # ------------------------------------------------------------ 推理
    @torch.no_grad()
    def q_values(self, state: torch.Tensor) -> torch.Tensor:
        return self.online(state)

    @torch.no_grad()
    def select_actions(
        self,
        cur: torch.Tensor,
        visited: torch.Tensor,
        dist: torch.Tensor,
        epsilon: float,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """批量 ε-greedy 动作选择。

        探索时不是纯随机，而是按距离倒数采样（原项目的改进点之一）。
        """
        state = self.make_state(cur, visited)
        q = self.online(state)
        q = q.masked_fill(visited, float("-inf"))

        greedy = q.argmax(dim=1)

        if epsilon <= 0.0:
            return greedy

        b = cur.size(0)
        explore_mask = torch.rand(b, device=self.device, generator=generator) < epsilon
        if not explore_mask.any():
            return greedy

        rows = torch.nonzero(explore_mask, as_tuple=False).squeeze(1)
        sub_cur = cur[rows]
        sub_visited = visited[rows]

        d = dist[sub_cur]  # (k, N)
        d = d.masked_fill(sub_visited, float("inf"))
        w = 1.0 / (d + 1e-8)
        row_sum = w.sum(dim=1, keepdim=True)
        # 极端情况下（理论上不会发生）退回均匀分布
        bad = (row_sum <= 0) | ~torch.isfinite(row_sum)
        if bad.any():
            w[bad.squeeze(1)] = (~sub_visited[bad.squeeze(1)]).float()
            row_sum = w.sum(dim=1, keepdim=True).clamp_min(1e-12)
        probs = w / row_sum

        picked = torch.multinomial(probs, 1, generator=generator).squeeze(1)

        out = greedy.clone()
        out[rows] = picked
        return out

    # ------------------------------------------------------------ 训练
    def update_learning_rate(self, alpha: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = float(alpha)

    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        next_visited: torch.Tensor,
        dones: torch.Tensor,
        gamma: float,
        double_dqn: bool = False,
    ) -> float:
        """一次梯度更新。返回 loss。

        double_dqn=True 时用 Double DQN 目标（Van Hasselt 2016）：
        选动作用 online、评价值用 target，两套网络噪声不相关，
        抵消 max 算子的系统性高估（本项目实测 Q 平均高估 +0.207）。
        """
        self.online.train()
        self.optimizer.zero_grad(set_to_none=True)

        q_all = self.online(states)  # (B, N)
        q_sa = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if double_dqn:
                # Double DQN：online 选动作，target 评价值
                q_next_online = self.online(next_states)
                q_next_online = q_next_online.masked_fill(next_visited, float("-inf"))
                best_actions = q_next_online.argmax(dim=1, keepdim=True)
                q_next = self.target(next_states)
                q_next = q_next.masked_fill(next_visited, float("-inf"))
                max_next = q_next.gather(1, best_actions).squeeze(1)
            else:
                q_next = self.target(next_states)
                # 已访问城市不可再选
                q_next = q_next.masked_fill(next_visited, float("-inf"))
                max_next = q_next.max(dim=1).values
            # 终态（走完全部城市）不再自举，避免无穷累积
            max_next = torch.where(dones, torch.zeros_like(max_next), max_next)
            target = rewards + gamma * max_next

        loss = self.loss_fn(q_sa, target)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10.0)
        self.optimizer.step()
        return float(loss.item())

    @torch.no_grad()
    def soft_update(self, tau: float) -> None:
        for tp, op in zip(self.target.parameters(), self.online.parameters()):
            tp.data.mul_(1.0 - tau).add_(op.data, alpha=tau)
        for tb, ob in zip(self.target.buffers(), self.online.buffers()):
            tb.data.copy_(ob.data)

    @torch.no_grad()
    def greedy_tour(self, dist: torch.Tensor, start: int = 0) -> np.ndarray:
        """用当前策略走一条完整路径（推理用，不做探索）。

        推理加速三层叠加（2026-10-27 实测，N=50）：
            1. 全 GPU 无同步：逐步 .item() 会让 GPU 每步空等 CPU
               （32.3 ms → 16.5 ms，路径逐位一致）
            2. fp16 autocast：激活 RDNA 4 的 AI 加速单元（WMMA），
               纯 GEMM 快 6.7 倍，解码整体再快约 2 倍
            3. CUDA 图捕获：解码循环形状固定，可整图捕获消除内核
               启动开销（ROCm 栈下走 hipGraph，实测支持良好）
               三层合计 32.3 ms → 2.07 ms（15.6 倍）

        正确性守卫：
            - bind_dist() 换实例时图自动失效重捕获
            - 捕获后对所有起点做 fp16 vs fp32 路径逐位校验，
              不一致则永久回退 eager fp32（本 agent 生命周期内）
            - CPU 设备走 eager 路径
        """
        self.online.eval()
        n = self.n

        if self.device.type == "cuda" and torch.cuda.is_available():
            g = self._ensure_decode_graph()
            if g is not None:
                self._g_cur.fill_(start)
                self._g_vis.zero_()
                self._g_vis[0, start] = True
                g.replay()
                return self._g_tour.cpu().numpy()

        return self._greedy_tour_eager(dist, start)

    # ------------------------------------------------------------ 解码加速实现
    def _greedy_step_body(self, cur: torch.Tensor, visited: torch.Tensor) -> torch.Tensor:
        """贪心解码循环体（可被 eager 与图捕获共用）。返回 (N,) long。"""
        tour = [cur]
        for _ in range(self.n - 1):
            q = self.online(self.make_state(cur, visited))
            q = q.masked_fill(visited, float("-inf"))
            nxt = q.argmax(dim=1)
            tour.append(nxt)
            visited.scatter_(1, nxt.unsqueeze(1), True)
            cur = nxt
        return torch.cat(tour)

    def _greedy_tour_eager(self, dist: torch.Tensor, start: int = 0) -> np.ndarray:
        """全 GPU 无同步的 fp32 解码（回退路径，也是图校验的参照）。

        dist 参数仅为保持与 greedy_tour 相同的调用形态；实际距离矩阵
        经 make_state 使用 self._dist_norm。
        """
        cur = torch.zeros(1, dtype=torch.long, device=self.device)
        cur.fill_(start)
        visited = torch.zeros(1, self.n, dtype=torch.bool, device=self.device)
        visited[0, start] = True
        return self._greedy_step_body(cur, visited).cpu().numpy()

    def _ensure_decode_graph(self):
        """懒捕获 CUDA 解码图；失败时禁用并返回 None（回退 eager）。

        两级降级：先试 fp16 图（快，2 ms），所有起点校验不过再试
        fp32 图（6 ms，必然逐位一致）；fp32 也失败则禁用走 eager。
        校验失败的常见原因是 fp16 舍入在 argmax 平局处翻转——
        这类分歧虽小但不接受，保证「换实现不换结果」。

        捕获前必须在 side stream 上预热几轮：MIOpen 首次调用 kernel 时
        要构建查找表，不预热会把查找逻辑一起录进图里。
        """
        if self._graph_disabled:
            return None
        if self._decode_graph is not None:
            return self._decode_graph
        if self._dist_norm is None:
            return None
        n = self.n
        try:
            self._g_cur = torch.zeros(1, dtype=torch.long, device=self.device)
            self._g_vis = torch.zeros(1, n, dtype=torch.bool, device=self.device)
            self._g_tour = torch.zeros(n, dtype=torch.long, device=self.device)

            # ---- 预热（side stream，MIOpen 查找表）----
            warm_cur = torch.zeros(1, dtype=torch.long, device=self.device)
            warm_vis = torch.zeros(1, n, dtype=torch.bool, device=self.device)
            warm_vis[0, 0] = True
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    self._greedy_step_body(warm_cur, warm_vis.clone())
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()

            # ---- 两级尝试：fp16 → fp32 ----
            for dtype in (torch.float16, torch.float32):
                try:
                    graph = torch.cuda.CUDAGraph()
                    self._g_cur.fill_(0)
                    self._g_vis.zero_()
                    self._g_vis[0, 0] = True
                    with torch.cuda.graph(graph):
                        cur = self._g_cur.clone()
                        visited = self._g_vis.clone()

                        def _run(c: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                            if dtype is torch.float16:
                                with torch.autocast("cuda", dtype=torch.float16):
                                    return self._greedy_step_body(c, v)
                            return self._greedy_step_body(c, v)

                        self._g_tour.copy_(_run(cur, visited))
                except Exception:  # noqa: BLE001
                    continue

                # ---- 正确性校验：所有起点 vs fp32 eager 逐位比对 ----
                # 一次性成本 N×16ms（N=50 约 0.8 秒），换永久低延迟解码
                ok = True
                for s0 in range(n):
                    self._g_cur.fill_(s0)
                    self._g_vis.zero_()
                    self._g_vis[0, s0] = True
                    graph.replay()
                    if not np.array_equal(
                        self._g_tour.cpu().numpy(),
                        self._greedy_tour_eager(self._dist_norm, s0),
                    ):
                        ok = False
                        break
                if ok:
                    self._decode_graph = graph
                    return graph

            # 两级都失败 → 永久禁用（本实例生命周期内不再重试）
            self._graph_disabled = True
            return None
        except Exception:  # noqa: BLE001  图捕获失败不应影响训练主流程
            self._graph_disabled = True
            return None

    def decode_multi_start(
        self, dist: torch.Tensor, n_starts: int | None = None
    ) -> tuple[np.ndarray, float, int]:
        """多起点贪心解码，返回 (最优路径, 长度, 实际起点数)。

        为什么需要它：训练时轨迹起点是随机的（collect_trajectories 用
        randint），模型学的是「从任意起点出发都能走好」；但单起点 argmax 解码
        每步都选当前 Q 值最大的城市，容易锁进一个固定的局部最优。

        POMO（Kwon et al. 2021）的做法就是：多起点贪心解码取最好的那个。
        这里实现的是其推理阶段的核心思想，起点数可调以便对比。

        起点不足 n 个时按等距抽样，保证覆盖均匀。
        """
        self.online.eval()
        n = self.n
        k = n if n_starts is None else max(1, min(int(n_starts), n))

        if k >= n:
            starts_np = np.arange(n, dtype=np.int64)
        else:
            starts_np = np.unique(np.linspace(0, n - 1, k).astype(np.int64))

        dev = dist.device
        b = len(starts_np)
        starts = torch.as_tensor(starts_np, device=dev, dtype=torch.long)
        cur = starts.clone()
        rows = torch.arange(b, device=dev)
        visited = torch.zeros(b, n, dtype=torch.bool, device=dev)
        visited[rows, starts] = True

        tours = torch.zeros(b, n + 1, dtype=torch.long, device=dev)
        tours[:, 0] = starts

        with torch.no_grad():
            for step in range(1, n):
                q = self.online(self.make_state(cur, visited))
                q = q.masked_fill(visited, float("-inf"))
                actions = q.argmax(dim=1)
                tours[:, step] = actions
                visited[rows, actions] = True
                cur = actions
            tours[:, n] = tours[:, 0]  # 闭环

            # 闭环长度：dist[tour[i], tour[i+1]] 求和
            step_len = dist[tours[:, :-1], tours[:, 1:]].sum(dim=1)

        best = int(step_len.argmin().item())
        tour = tours[best].cpu().numpy()
        return tour, float(step_len[best].item()), b

    @torch.no_grad()
    def decode_candidates(
        self, dist: torch.Tensor, n_starts: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """多起点贪心解码，返回**全部**候选路径及其长度。

        decode_multi_start 只返回最优的那条，做不了「每条候选各自 2-opt
        后再选优」的后处理（实测这一步在 N=25 上把 gap 从 0.57% 压到 0.00%）。
        这个方法专门给后处理用。

        返回
        ----
        tours   : (b, n+1) int64，闭环（末点 = 首点）
        lengths : (b,) float64
        """
        self.online.eval()
        n = self.n
        k = n if n_starts is None else max(1, min(int(n_starts), n))

        if k >= n:
            starts_np = np.arange(n, dtype=np.int64)
        else:
            starts_np = np.unique(np.linspace(0, n - 1, k).astype(np.int64))

        dev = dist.device
        b = len(starts_np)
        starts = torch.as_tensor(starts_np, device=dev, dtype=torch.long)
        cur = starts.clone()
        rows = torch.arange(b, device=dev)
        visited = torch.zeros(b, n, dtype=torch.bool, device=dev)
        visited[rows, starts] = True

        tours = torch.zeros(b, n + 1, dtype=torch.long, device=dev)
        tours[:, 0] = starts

        for step in range(1, n):
            q = self.online(self.make_state(cur, visited))
            q = q.masked_fill(visited, float("-inf"))
            actions = q.argmax(dim=1)
            tours[:, step] = actions
            visited[rows, actions] = True
            cur = actions
        tours[:, n] = tours[:, 0]

        step_len = dist[tours[:, :-1], tours[:, 1:]].sum(dim=1)
        return tours.cpu().numpy(), step_len.cpu().numpy()
