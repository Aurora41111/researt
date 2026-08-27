# 复现计划：Multi-UAV-Assisted ISAC 联合波束成形与轨迹优化

## Context

复现论文 *Tun et al., "Joint Beamforming and Trajectory Optimization for Multi-UAV-Assisted ISAC Systems," arXiv:2503.16915v2, 2025*（PDF 已在仓库根目录）。

论文研究多 UAV（每架带 M 天线均匀线阵的双功能雷达通信载荷）同时服务地面用户、感知目标的 ISAC 系统。核心是**最大化用户和速率**，受限于 CRB 感知约束、功率、能耗、速度、避碰等。求解用 **BCD 交替优化**：通信/感知波束成形用**分数规划 (FP) + CVX 凸松弛**，UAV 轨迹用 **DDPG 深度强化学习**。

目标：用 **Python (cvxpy + PyTorch)** 完整复现 **Fig.1 (a)(b)(c) 三张图**，工程上用 **uv** 管理依赖，运行于 Mac (Apple Silicon, MPS)。

**重要前提（写进 README）：** 论文 Table I 只列了部分参数，**载波频率/波长、带宽、噪声功率、时隙数 N、飞行周期 T、κ、CRB 阈值 Γ 等均未给出**。因此复现的**数值无法精确匹配原图**（RL 的随机性 + 参数缺失），但**趋势和量级**可以复现（Fig1b: 速率随 CRB 阈值放松而上升；Fig1c: ~6 次迭代收敛、功率越大速率越高；Fig1a: DDPG 轨迹比 A2C 更贴近用户）。缺失参数用同类 UAV-ISAC 文献的典型值替代，并在代码与文档中**显式标注为可调**。

---

## 技术栈与环境

| 项 | 选择 |
|---|---|
| 语言 | Python 3.12（uv 自建虚拟环境） |
| 包管理 | uv（已装 0.11.23） |
| 凸优化 | cvxpy + SCS（SDP 后端，免费），CVXOPT 备选 |
| 深度学习 | PyTorch（MPS 加速） |
| RL 环境 | Gymnasium（自定义 env） |
| 数值/绘图 | numpy, scipy, matplotlib |
| 测试/lint | pytest, ruff |
| 硬件 | Mac M5（MPS；无 NVIDIA GPU，DDPG 训练偏慢但可行） |

---

## 项目结构（greenfield，仓库当前为空）

```
RP-JBT-Opti-MUAV-ISAC/
├── PLAN.md                      # 本文件
├── README.md                    # 运行说明 + 参数缺失声明
├── pyproject.toml / .python-version / uv.lock
├── src/muav_isac/
│   ├── config.py                # @dataclass Params：Table I 全部 + 推断默认值
│   ├── scenario.py              # UAV/用户/目标布局，初始/终点位置，直线轨迹初始化
│   ├── geometry.py              # 距离 d、仰角 φ、LoS 概率 P_LoS (Eq.1)
│   ├── channel.py               # h_v[n] (Eq.2)、感知信道 ϕ_k (Eq.4)、导向矢量 a(φ)
│   ├── sensing.py               # CRB(φ) (Eq.5) 及 A_k,n 的数值/解析导数
│   ├── energy.py                # E_cs (Eq.6)、E_fly (Eq.7, Zeng&Zhang 推进模型)
│   ├── rate.py                  # R_v[n] (Eq.3)、SINR、干扰项 Θ_v（SDR 形式）
│   ├── beamforming/
│   │   ├── cvxbuild.py          # 由轨迹构造 H_v=h_v h_v†、(A†A) 矩阵；分块拆分
│   │   ├── comm.py              # Algorithm 1：FP 通信 BF（χ, ψ, G 交替）
│   │   └── sensing.py           # Algorithm 2：FP 感知 BF（Ω, I, ι 交替）
│   ├── trajectory/
│   │   ├── env.py               # Gymnasium env：状态/动作/奖励 (P8 → MDP)
│   │   ├── ddpg.py              # Algorithm 3：DDPG（actor/critic/target/replay）
│   │   └── a2c.py               # A2C baseline（仅 Fig1a 用）
│   ├── baselines.py             # TWOBF、BFWOT、直线轨迹/MRT 启发式
│   └── bcd.py                   # 外层 BCD：BF(Alg1+Alg2) → 轨迹(Alg3) 交替
├── scripts/
│   ├── fig1a_trajectories.py    # DDPG vs A2C 轨迹（2D/3D）
│   ├── fig1b_rate_vs_crb.py     # 扫 CRB 阈值 Γ；proposed vs TWOBF vs BFWOT
│   └── fig1c_convergence.py     # 3 个 P_max 下 BCD 收敛曲线
├── results/                     # 生成的 PNG + npz
└── tests/                       # 信道/CRB/能耗/凸子问题单调性单测
```

---

## 参数表（config.py）

**Table I 明确给出的**（直接用）：U=3, M=3；V_u∈[3,5]，K_u∈[2,4]；a_h∈[10,20] m/s；θ∈[−5π/12, 5π/12]；H∈[150,200] m；g0=−70 dB；C=11.95, D=0.136；Utip=120, ψ̃0=0.6；C0=798.6, C1=88.6；r̃=0.005, ρ=1.226；C2=11.5, σ_k=−17 dBsm；G=0.503, a0=4.3；replay buffer B=1600, batch=32。

**推断默认值（README 显式标注可调，因论文未给）**：

| 参数 | 默认值 | 依据 |
|---|---|---|
| 载波 f_c | 2 GHz → λ=0.15 m | sub-6GHz，与 M=3 小阵列/小场景匹配 |
| 天线间距 d_ant | λ/2 | 标准半波长 |
| 带宽 B_ch | ~250 MHz（Fig.1c 校准可取 ~800 MHz） | 论文未给；脚本暴露 `--sigma2-dBm` |
| 噪声 σ² | −80 dBm（Fig.1c 默认 −75 dBm） | N0=−174 dBm/Hz + 10log10(B) + NF≈10dB |
| 时隙数 N / 周期 T / τ | 60 / 60 s / 1 s | a_max·τ=20m 与 500m 场景匹配 |
| κ（感知 NLoS 因子） | 0.1 | NLoS 回波远弱于 LoS |
| CRB 阈值 Γ | 扫描范围 1e-6.5 ~ 1e-3 (rad²) | Fig1b 横轴 −6.5~−3（对数） |
| UAV 起点/终点 | 对角两端 (50,50)/(450,450) | Fig1a 示意 |
| 能量上限 E_th | 按满功率+全速飞行 1.2× 估 | 保证可行 |
| 场景 | 500m×500m | 论文明确 |

---

## 实现阶段（执行步骤）

### 阶段 0 — 脚手架
1. `uv init`，建 `pyproject.toml`（python=3.12），`uv add numpy scipy cvxpy torch gymnasium matplotlib`，`uv add --dev pytest ruff`。
2. 写 `README.md`（含参数缺失声明与运行命令）。
3. 建 `.gitignore`（`results/`, `.venv/`, `__pycache__/`；保留 `uv.lock`）。

### 阶段 1 — 系统模型（物理层）
按公式实现，每个模块配单测：
- `geometry.py`: `distance(o,l,H)`、`elevation(...)`、`plos(C,D,φ)` (Eq.1)。单测：d↑→P_LoS↓。
- `channel.py`: 导向矢量 `steer(φ,M,d_ant,λ)`；`h_gain(...)` (Eq.2，复 M×1)；`sensing_channel(...)` (Eq.4)。
- `sensing.py`: `beta(σ_k,d)`；`A_matrix(φ)`（**对 φ 求导**——优先**中心差分数值导数**，并对 M=3 解析式做一次等价性单测交叉验证）；`crb(...)` (Eq.5)。单测：sensing 功率↑→CRB↓。
- `energy.py`: `E_cs` (Eq.6, SDR 下 = τ·ΣTr)；`E_fly` (Eq.7，按 **Zeng & Zhang UAV 推进模型**实现，常量映射 C0=诱导、C1=叶型、C2=爬升)。单测：E_fly>0，悬停点 v=0 为有限值。
- `rate.py`: `sinr(G,I,H,σ²)`、`Θ_interference(...)`、`sum_rate(...)` (Eq.3 的 SDR 形式：用 Tr(H_v G_v) 替代 |h†g|²)。
- `scenario.py`: 随机布点（固定随机种子保证可复现），生成初始直线轨迹。

### 阶段 2 — 波束成形（Algorithm 1 & 2，cvxpy）
**关键数学（Shen & Yu 分数规划 [10]）：**

*通信 BF (Alg1)*，给定轨迹 o 与感知 I：
1. SDR：G_v[n]=g_v g_v† ⪰ 0，松弛 rank-1（论文附录 [11] 证明存在 rank-1 最优解）。
2. **拉格朗日对偶变换**引入 χ_v[n]，最优 χ* = Tr(H_v G_v)/Θ_v（闭式）。
3. **二次变换**引入 ψ_v[n]，最优 ψ* = √O_v/√P_v（闭式），其中 O_v=τ(1+χ)Tr(H_vG_v)，P_v=Tr(H_vG_v)+Θ_v。
4. χ,ψ 固定后，目标对 G 为**凹**（含 √(Tr(H_vG_v)) 项，concave）→ cvxpy 解凸问题，约束 (9c)(9d)(10c)。
5. 三步交替至收敛。

*感知 BF (Alg2)*，给定 o 与 G：
1. SDR：I_k[n] ⪰ 0。
2. 引入 ι_v[n] 与上界近似 (15)：Ω_v=Θ_v/ι_v，约束变凸。
3. cvxpy 解凸问题（目标仍是和速率，sensing 波束只通过干扰 Θ_v 影响速率；CRB 约束 (9b) 为线性 Tr((A†A)I_k)≥σ²/(2Γ|β|²)）。

**关键工程决策（跨时隙耦合）：** 能量约束 (9d) 把所有 N 个时隙耦合。直接拼成一个含 ~1260 个 PSD(3×3) 变量的大 SDP 会非常慢。**利用可分性**：除能量约束外 BF 问题按 (u,n) 时隙可分。对能量预算做**对偶分解**（单个 Lagrange 乘子 λ），把全局预算化为每时隙的等效功率上限，于是每个 (u,n) 是一个**小独立 SDP**（V_u+K_u 个 3×3 PSD 变量，约 7 个），SCS 秒级求解。λ 用二分/次梯度更新。

**rank-1 恢复**：松弛后的 G 一般天然接近 rank-1；用特征分解取主特征向量 + 高斯随机化（必要时）恢复 g_v。

### 阶段 3 — 轨迹优化（Algorithm 3，DDPG / PyTorch）
`trajectory/env.py`（Gymnasium `Env`）：
- **State** s(n)：当前所有 UAV 位置、用户/目标位置、由当前 BF 计算的各链路速率（或简化为信道增益）。
- **Action** a(n)（连续，每架 UAV 3 维）：a_h∈[10,20], θ∈[−5π/12,5π/12], H∈[150,200]；用 tanh 输出 + 线性缩放到区间。
- **Dynamics**：x[n+1]=x[n]+τ·a_h·cosθ，y[n+1]=y[n]+τ·a_h·sinθ（论文 Alg3 第 22-23 行）。
- **Reward** r(n)=Σ_u Σ_v R_v^u[n](o) − ξ（ξ 为越界惩罚：高度/速度/避碰/能耗/CRB 违背时罚）。
- **重要**：DDPG 训练阶段（一个外层 BCD 迭代内）BF 视为**固定**，reward 只用当前固定 G,I 代入速率公式（无需调 cvxpy），保证训练快。

`trajectory/ddpg.py`：标准 DDPG——actor μ(s)、critic Q(s,a)、target 网络软更新（δ）、replay buffer（B=1600）、batch=32、γ、OU 探索噪声 Ω。严格对照 Alg3 伪代码。
`trajectory/a2c.py`：A2C baseline（仅 Fig1a 对比用）。

### 阶段 4 — 外层 BCD 与 Baselines
`bcd.py`：`for outer in range(max_iter)`: ① 给定 o 跑 Alg1→Alg2 得 G,I；② 给定 G,I 跑 DDPG 得新 o；③ 记录和速率；④ 收敛判定（速率变化<ε 或达上限，预期 ~6 次）。输出每迭代速率（供 Fig1c）。
`baselines.py`：
- **TWOBF**（轨迹 w/o BF）：DDPG 优化轨迹，但 BF 用固定启发式（如最大比发射 MRT 到最强用户 + 等功率感知波束，不跑 FP）。
- **BFWOT**（BF w/o 轨迹）：FP 优化 BF，但轨迹固定为起点→终点直线。

### 阶段 5 — 三张图
- `fig1a_trajectories.py`：固定一个随机场景，分别用 DDPG 与 A2C 跑轨迹优化，画 2D 俯视 + 3D（含用户/目标/起终点）。期望：DDPG 轨迹更贴近用户。
- `fig1b_rate_vs_crb.py`：扫 Γ（对数横轴），对每个 Γ 跑完整 BCD（proposed）、TWOBF、BFWOT，画和速率 vs Γ。期望：proposed>BFWOT>TWOBF；速率随 Γ 放松单调上升。
- `fig1c_convergence.py`：P_max∈{0.1,0.5,1} W，各跑一次 BCD，画和速率 vs 外层迭代。期望：~6 次收敛；功率越大速率越高。
- 每个 script 把原始数据存 `results/*.npz`，图存 `results/*.png`，便于重绘/调参。

---

## 关键风险与对策

| 风险 | 对策 |
|---|---|
| **参数缺失致数值不匹配** | README/PLAN 显式声明；聚焦复现**趋势**，提供可调 config |
| **CRB 中 A_k,n 求导复杂易错** | 默认**数值中心差分**实现，解析式仅作单测交叉验证 |
| **大 SDP 过慢** | 能量约束对偶分解，按 (u,n) 拆成小独立 SDP |
| **DDPG 训练慢/不稳** | MPS 加速；先小规模（少 episode、小 N）调通再放大；reward 归一化、OU 噪声衰减 |
| **rank-1 松弛后 rank>1** | 特征分解取主向量 + 高斯随机化 |
| **cvxpy SDP 求解器报错/数值不稳** | SCS 为主，CVXOPT 备选；调 eps、max_iters；scale 功率量纲 |
| **UAV 推进能耗公式 (Eq.7) 排印不清** | 采用公认的 Zeng & Zhang 模型，常量按论文映射 |

---

## 验证（Verification）

**单元测试（`uv run pytest`）：**
- 信道：d↑→|h|²↓；导向矢量首元素=1、相位单调。
- CRB：sensing 功率↑→CRB↓；β 随 d↑↓。
- 能耗：E_fly(v)>0 且 v=0 有限；E_cs=τ·ΣTr。
- BF 凸子问题：目标随 FP 内迭代**单调不降**（FP 正确性标志）。
- BCD：外层目标和速率**单调不降**。

**端到端（逐图）：**
```
uv run python scripts/fig1c_convergence.py   # 最快，先验证收敛性
uv run python scripts/fig1b_rate_vs_crb.py
uv run python scripts/fig1a_trajectories.py
```
- Fig1c：确认 ~6 次迭代收敛、3 条曲线随 P_max 递增。
- Fig1b：确认 3 条曲线排序 proposed>BFWOT>TWOBF、随 Γ 上升。
- Fig1a：确认 DDPG 轨迹更贴近用户。
- 每张图存 npz+png，并手动核对量级合理（速率 bps/Hz 量级、CRB rad²）。

**完成判据：** 三张图趋势与论文一致；FP 与 BCD 单调性测试通过；README 可让他人一键复现。

---

## 执行顺序建议
阶段 0 → 1（+单测）→ 2（+单测，**先用单时隙小算例调通 FP**）→ 3（先用玩具 reward 调通 DDPG）→ `bcd.py` → 5（先 Fig1c，再 1b，最后 1a）→ 4（baselines 嵌入 1b）→ 调参美化。每个阶段完成后即可提交一次，便于回滚。
