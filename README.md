# Multi-UAV-Assisted ISAC: Joint Beamforming & Trajectory Optimization (复现)

Python 复现 *Tun et al., "Joint Beamforming and Trajectory Optimization for
Multi-UAV-Assisted Integrated Sensing and Communication Systems,"
arXiv:2503.16915v2, Apr. 2025*（论文 PDF 见仓库根目录）。

论文用多 UAV（每架带 M 天线均匀线阵的双功能雷达通信载荷）同时服务地面用户、
感知目标。目标是**最大化用户和速率**，受限于 CRB 感知约束、功率、能耗、速度、
避碰等。求解采用 **BCD 交替优化**：

- 通信 / 感知波束成形 → **分数规划 (FP) + SDR 凸松弛**（cvxpy）
- UAV 轨迹 → **DDPG 深度强化学习**（PyTorch）

复现目标为论文 **Fig. 1 (a)(b)(c)** 三张图。

---

## ⚠️ 关于数值精度的说明

论文 Table I **只列了部分参数**。以下关键参数在论文中**未给出**，本项目采用同类
UAV-ISAC 文献的典型值替代，并在 [`src/muav_isac/config.py`](src/muav_isac/config.py)
中**显式标注为可调**：

| 缺失参数 | 本项目默认值 | 依据 |
|---|---|---|
| 载波频率 / 波长 | 2 GHz / λ=0.15 m | sub-6GHz，与 M=3 小阵列匹配 |
| 天线间距 | λ/2 | 标准半波长 |
| **带宽 / 噪声功率 σ²** | **~250 MHz / −80 dBm** | 通用 ISAC 宽带基线；Fig.1(c) 脚本默认用 −75 dBm 做收敛图校准 |
| **功率预算 Pmax** | **70 dBm (10 kW)** | 论文 Fig.1(c) 扫描 {60, 70, 80} dBm；70 为默认（Table I 未列） |
| 时隙数 N / 周期 T | 60 / 60 s | a_max·τ=20 m，适配 500 m 场景 |
| 感知 NLoS 因子 κ | 0.1 | NLoS 回波远弱于 LoS |
| CRB 阈值 Γ | 默认 1e-5；Fig.1(b) 对数扫描 1e-6.5 ~ 1e-3 rad² | Fig.1(b) 横轴 |

因此**复现数值无法精确匹配原图**（参数缺失 + RL 随机性），但**趋势和量级**可复现：

- **Fig.1(b)**：和速率随 CRB 阈值放松而上升；proposed > BFWOT > TWOBF
- **Fig.1(c)**：~4–6 次 BCD 迭代收敛；Pmax 越大速率越高，脚本默认 `--sigma2-dBm -75` 以贴近论文 70–90 bps/Hz 的量级
- **Fig.1(a)**：DDPG 轨迹比 A2C 更贴近用户
- **Fig.1(d)**（本项目新增，非论文原图）：揭示噪声底/带宽对"功率-速率"趋势的影响

> **噪声底的选择（关键）**：论文未给带宽。若取窄带（10 MHz → σ²=−94 dBm），
> 60 dBm 已落入干扰受限区——FP 的和速率最大化会丢弃约一半弱用户（强信道集中），
> 功率增大反而用户数下降，**和速率对 Pmax 几乎不变**。通用默认值取宽带（~250 MHz →
> σ²=−80 dBm），可保持 Fig.1(b) 的紧 CRB 端可行；Fig.1(c) 为了让 60/70/80 dBm
> 曲线间距更贴近论文，脚本默认使用更保守的论文校准噪声底（~800 MHz → σ²=−75 dBm）。
> 所有 Fig.1 脚本都支持 `--sigma2-dBm` 继续做敏感性扫描；详见
> [`scripts/fig1d_noise_floor.py`](scripts/fig1d_noise_floor.py)。

> **数值稳定性注记**：Pmax 达 60–80 dBm 时，噪声归一化后的信号 `Tr(HG)~|h|²P/σ²`
> 可达 ~1e6，超出 SCS 求解器（eps=1e-3）可分辨范围。FP 求解器内做了两层 SINR-
> 不变的等价缩放（数据 `HH/=Pmax·mean_trace` + 求解变量 `G̃=G/Pmax`），保证
> 60–80 dBm 全程不崩。详见 [`beamforming/comm.py`](src/muav_isac/beamforming/comm.py)
> 与 [`beamforming/sensing.py`](src/muav_isac/beamforming/sensing.py)。

---

## 环境要求

- macOS / Linux（开发于 Mac Apple Silicon，PyTorch 走 MPS）
- Python 3.12（由 [uv](https://docs.astral.sh/uv/) 自动管理，无需手动安装）
- 依赖：`numpy scipy cvxpy torch gymnasium matplotlib`（运行）/ `pytest ruff`（开发）

## 快速开始

```bash
# 1. 安装依赖（uv 会自建 .venv 并锁版本到 uv.lock）
uv sync

# 2. 跑单元测试
uv run pytest

# 3. 生成图（写入 results/）
uv run python scripts/fig1c_convergence.py   # 最快，先验证收敛
uv run python scripts/fig1b_rate_vs_crb.py
uv run python scripts/fig1a_trajectories.py
uv run python scripts/fig1d_noise_floor.py   # 噪声底机制（本项目新增）
```

三个 Fig.1 脚本都支持 `--sigma2-dBm`；其中 Fig.1(c) 默认 `-75 dBm`，Fig.1(a)(b)
默认使用通用 `-80 dBm`。

## 项目结构

```
src/muav_isac/
├── config.py            # Table I 全部参数 + 推断默认值（@dataclass Params）
├── scenario.py          # UAV/用户/目标布局、初始直线轨迹
├── geometry.py          # 距离、仰角、LoS 概率 (Eq.1)
├── channel.py           # 信道增益 h_v[n] (Eq.2)、感知信道 (Eq.4)、导向矢量
├── sensing.py           # CRB(φ) (Eq.5) 及其 A_k,n 导数
├── energy.py            # 通信/感知能耗 (Eq.6)、飞行能耗 (Eq.7)
├── rate.py              # 数据率 R_v[n] (Eq.3)、SINR、干扰 Θ_v
├── beamforming/         # Algorithm 1 (通信 FP) + Algorithm 2 (感知 FP)，cvxpy
├── trajectory/          # Algorithm 3 (DDPG) + A2C baseline，Gymnasium + PyTorch
├── baselines.py         # TWOBF、BFWOT
└── bcd.py               # 外层 BCD：BF ↔ 轨迹 交替
scripts/                 # 三张图的生成脚本
tests/                   # 物理/凸子问题单调性单测
results/                 # 生成的 PNG + npz（git 忽略）
```

完整设计与各阶段细节见 [PLAN.md](PLAN.md)。

## 关键方法学

- **分数规划 (Shen & Yu 2018, [10])**：拉格朗日对偶变换 + 二次变换，将和速率最大化
  转为可凸化的 SDR，用 cvxpy 求解；χ、ψ 辅助变量闭式更新，G/I 由凸解得。
- **跨时隙能量耦合**：能量约束把所有 N 时隙耦合；采用对偶分解（单个 Lagrange 乘子）
  拆为每个 (u,n) 的小独立 SDP，避免上千 PSD 变量的大规模 SDP。
- **CRB 的 A_k,n 导数**：默认数值中心差分，M=3 解析式作单测交叉验证。
- **UAV 推进能耗**：采用公认的 Zeng & Zhang 模型，常量按论文 Table I 映射。

## 参考文献

1. 原文：arXiv:2503.16915v2
2. K. Shen & W. Yu, "Fractional programming for communication systems—Part I,"
   IEEE TSP, 2018. （分数规划 [10]）
3. Y. Zeng & R. Zhang, UAV 通信推进能耗模型。
