"""规模能力分级 —— 决定 WebUI 上「能实现 / 不能实现」的断点画在哪。

数据来源（全部为本项目实测，脚本 `scripts/ab_schedule.py` + `scripts/per_start_2opt.py`）：

| N   | DQN 训练最优 gap | DQN 策略解码 gap | 随机多起点+2-opt gap | 判定          |
|-----|------------------|------------------|----------------------|---------------|
| 15  | 0.00%            | 0.00%            | 0.00%                | saturated     |
| 20  | 0.00%            | 0.00%            | 0.00%                | saturated     |
| 25  | 0.00%            | 0.00%            | 0.00%                | saturated     |
| 30  | 11.66%           | 13.81%           | 0.00%                | marginal      |
| 40  | 15.98%           | 18.34%           | 0.00%                | marginal      |
| 50  | 48.21%           | 56.92%           | 0.00%                | failing       |
| 80  | —                | —                | +0.19%               | baseline-fail |
| 200 | —                | —                | +1.98%               | baseline-fail |

三个关键事实（决定了断点位置）：

1. **N ≤ 25 是饱和区**：随机路径 + 2-opt（200 次重启，<1 秒）也能达到 OR-Tools 最优解。
   此时 gap 数字全是 0~5%，**无法区分模型好坏**，「学习」本身没有意义。

2. **N = 30~40 是边际区**：DQN 与基线开始拉开真实差距，学习开始有意义，
   但绝对 gap 仍在 12~18%，需要配合后处理（各自 2-opt 后选优）才能收敛。

3. **N ≥ 50 是失效区**：MLP-DQN 的 256-256-128 网络无法表达 N×N 的距离关系，
   训练 4.5 分钟的结果**比 0.4 秒的规则算法还差 48 个百分点**。
   此时要竞争必须换 Attention Model（PLAN 的 M5）。

判定规则：
    saturated  : N <= 25   任务饱和，学习无意义
    marginal   : 25 < N <= 40   学习有意义但需后处理配合
    failing    : N > 40    MLP-DQN 失效，需更大模型
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CapabilityTier:
    """一个规模区间的判定。"""

    key: str
    label: str
    n_min: int
    n_max: int          # 闭区间上界
    summary: str
    detail: str
    tone: str           # ok | warn | danger
    measurable: bool    # 该区间内 gap 数字是否有区分度


TIERS: tuple[CapabilityTier, ...] = (
    CapabilityTier(
        key="saturated",
        label="饱和区",
        n_min=5,
        n_max=25,
        summary="任务已饱和，学习没有意义",
        detail=(
            "随机路径 + 2-opt（200 次重启，<1 秒）也能达到 OR-Tools 最优解。"
            "此时 gap 全在 0~5% 区间，**测不出模型好坏** —— "
            "对比表中若不加入基线，会误以为模型表现很好。"
        ),
        tone="warn",
        measurable=False,
    ),
    CapabilityTier(
        key="marginal",
        label="边际区",
        n_min=26,
        n_max=40,
        summary="学习开始有意义，但需配合后处理",
        detail=(
            "DQN 与规则基线拉开真实差距（gap 12~18%），学习确实起作用。"
            "必须用「每条候选各自 2-opt 后再选优」的后处理，否则会低估模型。"
        ),
        tone="ok",
        measurable=True,
    ),
    CapabilityTier(
        key="failing",
        label="失效区",
        n_min=41,
        n_max=200,
        summary="MLP-DQN 已失效，需换更大的模型",
        detail=(
            "N=50 实测：DQN 训练最优 gap +48.2%，而随机 + 2-opt 是 0.00% —— "
            "训练 4.5 分钟比不过 0.4 秒的规则算法。根因是 256-256-128 的 MLP "
            "无法表达 N×N 的距离关系。要在 N≥80 竞争必须用 Attention Model。"
        ),
        tone="danger",
        measurable=False,
    ),
)


def tier_for(n: int) -> CapabilityTier:
    """返回给定规模所属的区间判定。"""
    for t in TIERS:
        if t.n_min <= n <= t.n_max:
            return t
    return TIERS[-1]


def capability_report(n: int) -> dict:
    """给前端的完整能力报告。"""
    t = tier_for(n)
    return {
        "n_cities": n,
        "tier": t.key,
        "label": t.label,
        "summary": t.summary,
        "detail": t.detail,
        "tone": t.tone,
        "measurable": t.measurable,
        "range": [t.n_min, t.n_max],
    }


def all_tiers() -> list[dict]:
    """全部区间的列表，供前端画分段轴。"""
    return [
        {
            "key": t.key,
            "label": t.label,
            "n_min": t.n_min,
            "n_max": t.n_max,
            "summary": t.summary,
            "tone": t.tone,
            "measurable": t.measurable,
        }
        for t in TIERS
    ]
