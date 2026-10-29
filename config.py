"""全局配置与路径约定。

重要：本模块必须在任何 import torch 之前被导入。
原因见 PLAN.md §11.2 —— 本机存在核显（gfx115x）与独显（gfx1200）两块 AMD GPU，
而安装的 ROCm wheel 只为 gfx120X 编译了 kernel。两块同时可见时全部报
hipErrorInvalidImage，因此这里统一把可见设备限制为独显（索引 1）。

所有入口脚本（cli.py / run.py）都在第一行 import config，保证时序正确。
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------- 设备隔离
# 必须在 import torch 之前执行，否则不生效
os.environ.setdefault("HIP_VISIBLE_DEVICES", "1")

from dataclasses import asdict, dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402

# ---------------------------------------------------------------- 路径
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
TSPLIB_DIR = DATA_DIR / "tsplib"
RUNS_DIR = DATA_DIR / "runs"
EXPORT_DIR = DATA_DIR / "exports"

for _d in (DATA_DIR, TSPLIB_DIR, RUNS_DIR, EXPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DB_PATH = RUNS_DIR / "tsp_rl.sqlite"

# ---------------------------------------------------------------- 实例配置
DISTRIBUTIONS = ("uniform", "cluster", "grid")


@dataclass
class InstanceConfig:
    """一个 TSP 实例的生成参数。"""

    n_cities: int = 20
    distribution: str = "uniform"
    seed: int | None = None
    name: str = ""

    def resolved_name(self) -> str:
        if self.name:
            return self.name
        seed_part = "rand" if self.seed is None else f"s{self.seed}"
        return f"{self.distribution}-n{self.n_cities}-{seed_part}"


# ---------------------------------------------------------------- DQN 配置
@dataclass
class DQNConfig:
    """Track A（实时轨）的全部超参。

    默认值继承自原 MATLAB 版（tsp_dqn_plotfinalzsj.m），
    以便与课程报告结论对拍。
    """

    # 问题规模
    n_cities: int = 20

    # 训练长度
    episodes: int = 6000
    early_stop: int = 2500  # 连续无改进的回合上限

    # 折扣与学习率
    gamma: float = 0.95
    alpha_start: float = 0.02
    alpha_end: float = 0.0005
    # 分段线性衰减所用的比例（仅 alpha_schedule="linear" 时生效）
    alpha_decay_ratio: float = 1.0 / 3.0
    # 学习率调度：linear（分段线性）| cosine（余弦退火）| const（恒定）| plateau（平台期自适应）
    #
    # 默认用 cosine。N=25 / 2000 回合实测（seed 42，解码统一 25 起点）：
    #   cosine → 训练最优 gap 8.12%
    #   linear → 训练最优 gap 30.29%   （1/3 回合就降到 alpha_end，剩 2/3 躺平）
    #   const  → 训练最优 gap 44.89%   （始终 0.02，无法精细收敛）
    # 若要与原 MATLAB 版对拍，请选回 "linear"（原版即为分段线性）。
    alpha_schedule: str = "cosine"

    # plateau 调度参数（ReduceLROnPlateau 思路）：
    #   连续 plateau_patience 回合无改进 → alpha 乘 plateau_factor，
    #   下限 alpha_end；每次降 LR 重置 no_gain 计数（配合早停：
    #   只有 LR 已到底还无改进才触发 early_stop）。
    #   动机：N=35 ε=0.3 实测 ep500-1200 卡 24.3% 平台期，
    #   cosine 是「按时间盲降」，plateau 是「按进展降」。
    plateau_patience: int = 300
    plateau_factor: float = 0.5

    # 探索（余弦退火）
    eps_start: float = 0.9
    eps_end: float = 0.001

    # 网络与回放
    hidden: tuple[int, ...] = (256, 256, 128)
    batch_size: int = 1024
    mem_size: int = 30000
    n_traj: int = 4  # 每回合并行轨迹数
    train_iters_per_ep: int = 3  # 每回合做几次梯度更新
    tau: float = 0.005  # 目标网络软更新系数

    # 运行环境
    device: str = "auto"  # auto | cpu | cuda
    seed: int | None = None

    # 推送节流：每多少回合向前端推一次指标
    report_every: int = 10

    # 推理：贪心解码的起点数量。None 或 >= N 表示全部起点（POMO 思路）
    decode_starts: int | None = None

    # 状态表示：是否把「当前城市到各城市的归一化距离」显式放进状态
    #   False → 2N 维（one-hot 当前城市 ‖ 访问掩码），原 MATLAB 版做法
    #   True  → 3N 维（追加 N 维距离特征，已访问城市置 0）
    #
    # 动机（诊断结论）：2N 维时模型靠 one-hot 隐式逼近距离函数，与最近邻
    # 一致率 82.3% 说明学到了，但样本/参数仅 6.1，样本效率极低。
    use_dist_feature: bool = True

    # YOLO 式学习率预热：前 N 个回合把 α 从 0 线性升到 alpha_start。
    # 默认 0（关闭）。YOLO 需要它是因为 SGD + 大学习率会炸；本项目用 Adam
    # + 目标网络，通常不需要，做成开关以便 A/B 决定要不要。
    lr_warmup_episodes: int = 0

    # POMO 式共享 baseline（Kwon 2021 的核心技巧移植）：
    # 每回合 n_traj 条轨迹算完长度后，把轨迹级优势
    #   adv_i = (mean_len − len_i) / dmean / n_steps
    # 均摊加到该轨迹每一步的奖励上。Q 目标由此变成「相对同批平均水平」
    # 的尺度，轨迹间共同波动（运气好坏）被整体抵消，梯度方差显著下降。
    # 注意：这是 REINFORCE-with-baseline 思想在 DQN 上的实用化折算，
    # 会改变 Q 的绝对尺度（argmax 不受影响），结论展示时要如实标注。
    # 实验依据：γ/mem 实验排除其他假设后，训练-解码 gap 的头号嫌疑
    # 就是梯度方差。
    shared_baseline: bool = False

    # Double DQN（Van Hasselt 2016）：TD 目标里选动作用 online、评价值用
    # target，两套网络噪声不相关，抵消 max 的系统性高估（实测 Q 平均高估
    # +0.207）。改动只影响 train_step 的目标计算，风险极低。
    double_dqn: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BCConfig:
    """专家预热（Behaviour Cloning / 专家 Q 回归）配置。

    目的是在 RL 之前用监督信号把权重推到「至少会走一条像样的路」，
    解决 DQN 的样本效率瓶颈（样本/参数比 6.1 → 200+）。

    variant:
        A = 专家 Q 回归   Q(s,a) ← −(专家轨迹剩余代价)/dmean
        B = 行为克隆 CE   Q(s,·) 对 one-hot(专家动作) 做交叉熵
        C = 混合          loss = w_q · L_q + L_ce

    ⚠ 实测结论（N=50 / K=50 / 15 轮，同一留出集）：
        A  专家 Q 回归      留出集 gap 322.7%   ← 不可用
        B  行为克隆 CE      留出集 gap  43.1%
        C  混合 w_q=0.5     留出集 gap 180.3%
        （最近邻贪心 11.3% / 随机重启+2-opt −0.006%）

        **A 失败的原因是目标量级**：Q 目标 ≈ −剩余步数（量级 ~−50），
        被「还剩几步」完全主导，候选之间的边际差异被淹没到 1e-3 量级，
        argmax 等于在噪声里挑。所以 A 不能用来预热。
        C 若用 w_q=0.5 也会被回归项压死 CE，必须让 CE 主导。
        因此 w_q 默认取 **0.1**（只做尺度校准，不主导梯度）。

    ⚠ 实测结论 2（2026-10-27，N=40 同实例同种子对照）：
        BC 50 轮 + 2000 回合  best 6.99
        BC 100 轮 + 8000 回合 best 7.57  ← 更久反而更差
        BC 轮数超过 ~50 后在 K 个专家实例上过拟合，
        把网络推进对 RL 微调不利的区域。所以 epochs 语义是
        「上限」，验证集早停（early_stop_patience）负责实际停止，
        停止后回滚到 val loss 最低的那一轮权重。
    """

    variant: str = "C"
    n_instances: int = 500      # 专家实例数（K）
    restarts: int = 200         # 每个实例的 2-opt 重启次数
    permutations: int = 8       # 每个实例的索引置换增广次数
    epochs: int = 30            # 预热轮数**上限**（验证集早停会提前停）
    early_stop_patience: int = 5    # 验证 loss 连续 N 轮不降则早停
    early_stop_min_delta: float = 0.0  # 判定「不降」的最小改善量
    lr: float = 1e-3
    batch_size: int = 1024
    w_q: float = 0.1            # 混合损失里「回归项」的权重（CE 的权重固定为 1）
    val_ratio: float = 0.1      # 训练集内部留出的验证比例（不是测试集）
    seed: int | None = 0        # 专家实例的种子空间基准
    lr_warmup_epochs: int = 0   # YOLO 式学习率预热（默认关）
    max_samples: int = 2_000_000  # 样本数上限（内存护栏）

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- 训练轨道
TRACKS = ("dqn",)  # 后续加入 "attention" 时在此扩展


@dataclass
class TrainRequest:
    """一次训练请求的完整描述。"""

    track: str = "dqn"
    instance: InstanceConfig = field(default_factory=InstanceConfig)
    dqn: DQNConfig = field(default_factory=DQNConfig)


# ---------------------------------------------------------------- 杂项
def resolve_device(pref: str = "auto") -> str:
    """把 'auto' 解析为具体设备字符串。"""
    if pref != "auto":
        return pref
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"
