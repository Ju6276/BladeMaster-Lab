# 实现与论文对照

论文：[BladeMaster, arXiv:2609.27342v2](https://arxiv.org/html/2609.27342v2)。

## 公式到代码

| 论文 | 内容 | 代码 |
|---|---|---|
| 式 1–2 | 参考构型 TLMPM P2G，固定三线性权重 | `geometry.stencil`，`material.p2g` |
| §III-B | fixed-corotated + von Mises 塑性（Hencky 返回映射），可选牛顿黏性、Maxwell 黏弹性 | `material.p2g`，`material.plastic_at`，`material.viscous_impulse`，`material.maxwell_at` |
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
| 网格间距 Δx | 6 mm（香蕉），`--spacing` 可取任意值，不必整除香蕉尺寸 | 本地选择 |
| 粒子 | 每格每轴 2 个；规则采样 1452 个，默认 seed=1 扰动采样 1439 个 | 本地选择 |
| 接触半径 | 粒子—粒子、粒子—砧板：\(R_p = (V_p^0)^{1/3}/2 = \Delta x/4\)，即 \(s_R = 1\)；粒子—刀：香蕉固定 0.5 mm（`blade_contact_radius`），见 RESULTS.md 力的收敛；求解器下限 \(\Delta x/4\) − 刀半厚（6 mm 网格为 1.0 mm），否则刀落在两列粒子之间谁也碰不到 | 本地选择 |
| PIC/FLIP 比例 | 0.95 | DexLab |
| \(s_n\)，\(\gamma_k\)，\(\gamma_c\) | 0.1，0.05，0.1（在参考步长下） | DexLab |
| \(b_\text{lead}\)，\(b_\text{back}\)，\(u_\text{min}\) | 2×刀接触距离，\(2\Delta x\)，0 | 本地选择，论文定义符号但未给数值 |
| \(h_k\) | \(6\Delta x\) | 论文 |
| 香蕉材料 | E=400 kPa，ν=0.45，ρ=1100，屈服 30 kPa，\(G_c\)=700，ζ=0.9，刀 0.1 kg，μ=0.2 | 论文 §IV-C |
| 香蕉步长 | 62.5 μs，接触参数以 25 μs 为参考 | 本地选择，见 RESULTS.md 步长扫描 |
| 机器人耦合间隔 | 1 ms（MuJoCo 步长） | 本地选择 |
| 表面刀路 | 记录实际材料子步的刀位、切割开关和刀次；保持不动的子步只存首尾。默认每 40 ms 更新，可用 `--observation-interval` 加密；材料节点在观测间线性插值 | 本地选择 |
| 表面 QEF 边界 | 解约束在参考单元每侧内缩 0.001×表面格距的盒中；1.5 mm 格距时为 1.5 μm；多区域单元以 1e-3×Hermite 样本数弱正则到各区域质心 | 消除相邻单元边界上的重合折叠，材料参数与粒子运动不变 |

## 接触刚度与步长

论文式 11/15 的法向刚度是 \(k = s_n m_\text{eff}/\Delta t^2\)，切向 \(k_t = \gamma_k m_\text{eff}/\Delta t^2\)。固定 \(s_n\) 时，步长减半，接触就硬 4 倍，所以力曲线不可能随步长收敛。

`Config.contact_reference_dt > 0` 时，`normal_scale` 和 `tangent_stiffness` 视为该参考步长下的值，并按 \((\Delta t/\Delta t_\text{ref})^2\) 缩放，从而保持 \(k\) 不变。缩放后的因子必须 ≤ 1（否则单步过冲），对香蕉（参考 25 μs，\(s_n=0.1\)）这意味着 \(\Delta t \le 79\) μs；默认 62.5 μs 时 \(s_n = 0.625\)。预设默认启用；`franka_run --contact-scaling per-step` 恢复论文的固定 \(s_n\) 写法。

切割功阈值只通过 \(G_c(1-\zeta)/\zeta\) 使用切割能和锋利度。保持这个组合不变的参数对产生相同阈值；力曲线标定时固定其中一个，或用额外测量约束二者。损伤影响刃口接触分量，几何侧标签独立决定网格分离。

## Maxwell 黏弹性

`Config.maxwell_tau` \(\tau\) > 0 且 `maxwell_fraction` \(\beta\) > 0 时启用（广义标准线性固体）。弹性变形 \(f = F F_p^{-1}\) 分两支：平衡支为共旋弹性，\(\mu_\infty = (1-\beta)\mu\)、\(\lambda + \tfrac23\beta\mu\)；Maxwell 支 \(f_m = f F_v^{-1}\)，Kirchhoff 应力 \(2\beta\mu\,\mathrm{dev}\log V_m\)。两支之和的小应变剪切、体积模量与关闭时相同，长期剪切模量为 \((1-\beta)\mu\)，体积部分不松弛。每步在塑性投影之后把 \(f_m\) 的偏 Hencky 应变乘以 \(e^{-\Delta t/\tau}\)（应变保持不变时是精确解，对任意 \(\Delta t/\tau\) 稳定；\(\tau \ll \Delta t\) 时该支每步归零），\(F_v\) 按等容增量更新。每步增量 \((\Delta t/\tau)\,\varepsilon\) 远小于 \(F_v\) 在 float32 下的分辨率，所以 \(F_v - I\) 存为 float64。屈服仍按瞬时模量的 Hencky 应变判定，加载慢于 \(\tau\) 时流动应力为 \((1-\beta)\sigma_y\)。关闭时调用原来的 kernel。

## 与论文的差异

| 项 | 论文 | 这里 |
|---|---|---|
| 粒子位置更新 | 未写明 | 用插值网格速度推进（de Vaucorbeil 2020 TLMPM 式 13），速度用 PIC/FLIP |
| 接触刚度 | 每步常数 \(s_n\) | 默认按参考步长固定物理刚度，见上节 |
| 黏弹性 | 无 | 可选 Maxwell 分支，默认关闭，见上节 |
| 耦合区间内的刀具 | 位姿与速度保持不变 | 默认相同；`tool_substep='predict'` 用 6×6 移动性矩阵逐子步更新刀速并推进位姿 |
| 接触检测 | 未写明 | 默认用设备端条件重建邻居表；`hash` 每子步重建 Warp HashGrid，`all` 全对检测（每粒子 64 线程），按完整搜索核对接触集合 |
| 切割次数 | 不限 | int64 三进制码，上限由 \(3^{k+1}\cdot\)节点数 \(< 2^{63}\) 决定（香蕉 6 mm 网格 33 刀，3 mm 31 刀，2 mm 30 刀） |
| 香蕉几何 | 实物扫描 | DexLab 合成 120 mm 香蕉 |
| 刀具 | 未公开 | 80×40×1 mm 楔形，5 mm 刃口斜面 |
| 地面 | 刚性支撑 | 粒子：砧板顶面矩形内的罚函数接触。刀：Franka 由 MuJoCo 实体盒碰撞；其余刀（导向、位姿伺服、自由落体）共用 `tool.step` 的同一个刀顶点—砧板接触（速度约束 + 库仑摩擦），三者使用同一砧板矩形 |
| 控制器 | 未公开 | libfranka 笛卡尔阻抗示例公式，或保持 SERL++ 公式的本地移植（MuJoCo 四元数替代 scipy；测试对照固定版本源码） |
| 连续切割任务 | 论文演示多刀切割 | `CutSequence` 指定切点、速度和推开距离；按实测刀位到达底部触发推开，切割事件由下刀阶段启用 |
| 刃前带宽 \(b_\text{lead}\) | 定义于式 4，未给数值 | 2×刀接触距离 `blade_offset`：3 mm 香蕉为 1.0 mm，6 mm 香蕉为 2.0 mm；它与粒子半径可分别变化 |
| 刃压到砧板时的底层侧标签 | 只按切割带（式 6） | `contact.assign` 另外给刃高度 ≤ \(R_p\) 时刃两侧、中心在刃上方 \(R_p\) 以内且在砧板矩形内的底层粒子打侧标签：刀倾斜时这些粒子在刃平面以下，切割带永远不包含它们，块切不透（Fig. 3） |
| 砧板罚因子 | 与刀共用 \(s_n\) | 单独的 `board_normal_scale`；Fig. 3/4/7/8 在论文步长下刀与砧板都取 \(s_n\) = 1（0.1 时每步穿透约 \(N g\Delta t^2/s_n\)，底层沉入砧板），香蕉仍为 0.1 |
| 扰动采样 | 未写明 | 每轴 ±Δx/4 均匀扰动后，把低于 \(R_p\) = Δx/4 的粒子按 \(z \to R_p + |z - R_p|\) 反射到砧板以上（`geometry.points`）；否则初始穿透的恢复冲量把整块抬起（修正前 6 mm 香蕉最低粒子弹起 8.4 mm，3 mm 1.2 mm） |
| 场景设置 | Franka 操作 | Fig. 4/7/8 用规定刀路（`scenarios.py`，非 Franka）；Fig. 8 每刀后把切片推倒；Fig. 3 刀绕刀厚轴倾斜（论文只给角度）；Fig. 4/7/8 的 \(G_c\) = 10 J/m² 为本地取值；导向刀驱动 1000 N/m、20 N·s/m（论文材料预设，DexLab 250 N/m）；摩擦默认 0.3 / 0.2（DexLab），论文香蕉 0.2 / 0.2。详见 RESULTS |
| Fig. 5 独立对照 | 两刀三块、0.5 mm 刀厚、相同采样与弹性参数 | 新夹具使用 60×30×30 mm 方块、E=6 kPa、ν=0.4、ρ=1100、5 mm 网格、0.25 ms 步长；退刀后左块速度 −30 mm/s。切口用粒子列间距减初始间距，中块体积用内部粒子四面体，评价量与论文切口宽度/全块体积定义不同 |
| CRESSim-MPM 对照实现 | 论文使用 CRESSim-MPM | 直接编译原版 `09aa5009` 的 `CpuMlsMpmSolver`。CPU/CUDA 共用的 `computeStress` 共旋分支缺少 `break`，结果被线弹性覆盖；构建选项 `--fix-corotated-dispatch` 只在独立头文件副本中补上此行，原版模式也保留。质量单位扫描将固定网格质量阈值 1e-4 换算为 SI，并检查总动量 |
| TopoCut 基线损伤 | 按体积比阈值 | 先在重力下静置（默认 1 s）再开启损伤，\(J\) 相对静置后的构形计（`MlsSolver.start_damage`）：软材料（10 kPa）自重压缩本身会越过阈值，静置算作损伤时 Fig. 4 有 23% 粒子在刀动之前就损伤 |
| 已与论文核对 | — | ① 自接触：论文 §III-E"For self-contact, nearby reference neighbors are excluded … Pairs with conflicting cut-side codes are exempt"，没有给距离。代码对切割码兼容且参考距离 \(< 2\Delta x\) 的粒子对不算接触（`contact.pair_at`）：三线性权重下两粒子共享网格节点的条件是每轴参考距离 \(< 2\Delta x\)，这里用欧氏距离，对角方向共享节点的少数粒子对仍是接触候选。② 式 5：\(d_k\) 是"front–back direction … obtained by transforming their fixed tool-frame counterparts using the current tool pose"，即刀坐标系里固定的方向，代码取刀的 −z 轴（`resist_cut`、`assign`），与论文一致；\(u_{c,p}\) 默认用相对速度（`cut_work="relative"`），也与论文一致。论文的 \(u_\text{min}\) 没有给值，代码取 0（`advance > 0`） |
