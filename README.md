# TSP-RL-Core

基于深度强化学习（DQN）求解旅行商问题（TSP）的核心算法实现。

项目包含：DQN 求解器、专家引导的行为克隆（BC）、多起点贪心解码（POMO 思路）、
以及 Greedy / 2-opt / OR-Tools 等传统基线方法，用于对比评估。

## 项目结构

```
tsp-rl-core/
├── core/                  # 问题定义与基线求解
│   ├── instance.py        #   TSP 实例生成（均匀 / 聚类分布）
│   ├── baselines.py       #   Greedy、2-opt、OR-Tools 基线求解器
│   ├── expert.py          #   专家解生成（用于行为克隆）
│   ├── metrics.py         #   路径长度、合法性、gap 等评估指标
│   └── capability.py      #   求解器能力描述
├── models/                # 神经网络模型
│   └── dqn.py             #   Q 网络 + DQN 智能体
│                         #   状态: [当前城市 one-hot ‖ 已访问掩码] (2N 维)
│                         #   网络: 2N → 256 → 256 → 128 → N, tanh 激活
├── train/                 # 训练模块
│   ├── train_dqn.py       #   DQN 主训练循环
│   ├── bc.py              #   行为克隆（从专家解预训练）
│   └── replay.py          #   经验回放缓冲区
├── experiments/           # 实验与验证脚本
│   ├── check_env.py       #   环境自检
│   ├── smoke_test.py      #   冒烟测试
│   ├── test_policy.py     #   策略测试
│   ├── verify_baseline.py #   基线正确性验证
│   ├── ab_decode.py       #   消融：单起点 vs 多起点解码（POMO）
│   ├── ab_memsize.py      #   消融：回放缓冲区大小
│   ├── ab_schedule.py     #   消融：学习率/探索调度
│   ├── decode_strategies.py # 解码策略对比
│   └── per_start_2opt.py  #   每起点 2-opt 后处理
├── cli.py                 # 命令行入口：基线对比、批量扫描
└── config.py              # 全局配置（DQN 超参、实例参数、设备选择）
```

## 快速开始

```bash
pip install -r requirements.txt

# 环境自检
python experiments/check_env.py

# 基线方法对比（Greedy / 2-opt / OR-Tools）
python cli.py bench --n 20 --dist uniform --seed 42

# 批量扫描多个实例
python cli.py sweep --n 20 --runs 5
```

### 训练 DQN

```bash
python train/train_dqn.py --help   # 查看训练参数
```

### 消融实验

```bash
python experiments/ab_decode.py  --ckpt <权重文件>   # 多起点解码消融
python experiments/ab_schedule.py                    # 调度策略消融
python experiments/ab_memsize.py                     # 缓冲区大小消融
```

## 设计要点

- **状态表示**：`[当前城市 one-hot ‖ 已访问掩码]`，2N 维
- **网络结构**：2N → 256 → 256 → 128 → N，tanh 激活；
  权重手动初始化为 N(0, 0.1) + 零偏置（与 MATLAB 参考实现保持可比）
- **POMO 思路解码**：多起点并行贪心解码，取最优路径
- **行为克隆预训练**：先用专家解预热策略，再进行 RL 微调

## 许可证

本项目基于 [MIT License](LICENSE) 开源。
