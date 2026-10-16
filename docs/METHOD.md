# 实现与论文对照

论文：[BladeMaster, arXiv:2609.27342v2](https://arxiv.org/html/2609.27342v2)。

## 公式到代码

| 论文 | 内容 | 代码 |
|---|---|---|
| 式 1–2 | 参考构型 TLMPM P2G，固定三线性权重 | `geometry.stencil`，`material.p2g` |
| §III-B | fixed-corotated + von Mises 塑性（Hencky 返回映射），可选牛顿黏性 | `material.p2g`，`material.plastic_at`，`material.viscous_impulse` |
| 式 3–6 | 切割带、方向接触功、侧标签 | `contact.rigid_at`（D 更新），`contact.assign`（标签） |
| 式 7 | 三进制切割码 | `Solver._set_cut`，`material.compatible` |
| 式 8–9 | 逻辑网格反对称交换、网格速度 | `material.couple_and_velocity_at`，`material.couple_at` |
| §III-B | PIC/FLIP G2P + MUSL 重映射 | `material.g2p_remap`，`material.update_F_plastic` |
| 式 10–13 | 粒子间接触与切向历史 | `contact.pair_at`，`contact.friction` |
| 式 14–16 | 刀具接触、有效质量、反力 | `contact.rigid_at` |
| §III-F | linked-volume + 局部 dual contouring 表面 | `surface.py`，`surface_gpu.py`，`surface_ccd_gpu.py` |
| §III-G | 紧凑活动网格；外部刚体接口，区间平均 wrench | `Solver(grid_storage='compact')`，`coupling.ExternalToolPort` |

## 数值设置

| 量 | 值 | 来源 |
|---|---|---|
| 网格间距 Δx | 6 mm（香蕉） | 本地选择 |
| 粒子 | 每格每轴 2 个，香蕉 1452 个 | 本地选择 |
| 接触半径 | \(R_p = (V_p^0)^{1/3}/2\)，即 \(s_R = 1\) | 本地选择 |
| PIC/FLIP 比例 | 0.95 | DexLab |
| \(s_n\)，\(\gamma_k\)，\(\gamma_c\) | 0.1，0.05，0.1（在参考步长下） | DexLab |
| \(b_\text{lead}\)，\(b_\text{back}\)，\(u_\text{min}\) | \(2R_p\)，\(2\Delta x\)，0 | DexLab |
| \(h_k\) | \(6\Delta x\) | 论文 |
| 香蕉材料 | E=400 kPa，ν=0.45，ρ=1100，屈服 30 kPa，\(G_c\)=700，ζ=0.9，刀 0.1 kg，μ=0.2 | 论文 §IV-C |
| 香蕉步长 | 62.5 μs，接触参数以 25 μs 为参考 | 本地选择，见 RESULTS.md 步长扫描 |
| 机器人耦合间隔 | 1 ms（MuJoCo 步长） | 本地选择 |

## 接触刚度与步长

论文式 11/15 的法向刚度是 \(k = s_n m_\text{eff}/\Delta t^2\)，切向 \(k_t = \gamma_k m_\text{eff}/\Delta t^2\)。固定 \(s_n\) 时，步长减半，接触就硬 4 倍，所以力曲线不可能随步长收敛。

`Config.contact_reference_dt > 0` 时，`normal_scale` 和 `tangent_stiffness` 视为该参考步长下的值，并按 \((\Delta t/\Delta t_\text{ref})^2\) 缩放，从而保持 \(k\) 不变。缩放后的因子必须 ≤ 1（否则单步过冲），对香蕉（参考 25 μs，\(s_n=0.1\)）这意味着 \(\Delta t \le 79\) μs；默认 62.5 μs 时 \(s_n = 0.625\)。预设默认启用；`franka_run --contact-scaling per-step` 恢复论文的固定 \(s_n\) 写法。

## 与论文的差异

| 项 | 论文 | 这里 |
|---|---|---|
| 粒子位置更新 | 未写明 | 用插值网格速度推进（de Vaucorbeil 2020 TLMPM 式 13），速度用 PIC/FLIP |
| 接触刚度 | 每步常数 \(s_n\) | 默认按参考步长固定物理刚度，见上节 |
| 耦合区间内的刀具 | 位姿与速度保持不变 | 默认相同；`tool_substep='predict'` 用 6×6 移动性矩阵逐子步更新刀速并推进位姿 |
| 接触检测 | 未写明 | n ≤ 2048 时全对检测（每粒子 64 线程），否则 Warp HashGrid；接触集合相同 |
| 切割次数 | 不限 | int32 三进制码，单次运行最多 19 次切割 |
| 香蕉几何 | 实物扫描 | DexLab 合成 120 mm 香蕉 |
| 刀具 | 未公开 | 80×40×1 mm 楔形，5 mm 刃口斜面 |
| 地面 | 刚性支撑 | MPM 中为 z=0 无限平面；MuJoCo 中为有限砧板 |
| 控制器 | 未公开 | libfranka 笛卡尔阻抗示例公式，或 SERL++ `impedance.py`（未改动） |
