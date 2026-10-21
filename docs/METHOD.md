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
| 网格刀面约束 | 外部刀位保持期间，补足网格传输后的刀面法向冲量 | `blade_constraint.py`，`Solver._material_transfers` |
| §III-F | linked-volume + 局部 dual contouring 表面 | `surface.py`，`surface_gpu.py`，`surface_ccd_gpu.py` |
| §III-G | 紧凑活动网格；外部刚体接口，区间平均 wrench | `Solver(grid_storage='compact')`，`coupling.ExternalToolPort` |

## 数值设置

| 量 | 值 | 来源 |
|---|---|---|
| 网格间距 Δx | 香蕉默认 6 mm；`--spacing` 可取任意有效值，不必整除香蕉尺寸。3 mm 子步 40 μs，6 mm 为 62.5 μs；物理接触刚度保持相同 | 本地选择 |
| 粒子 | 每格每轴 2 个；默认规则子单元中心（seed=0，6 mm 下 1452 个）；显式 seed=1 扰动采样为 1439 个 | 本地选择；全幅扰动在粗网格切透时放大侧向弹开，见 RESULTS |
| 接触半径 | 默认粒子—粒子、粒子—砧板、粒子—刀均用 \(R_p = (V_p^0)^{1/3}/2 = \Delta x/4\)，即 \(s_R = 1\)；刀距可用 `--blade-contact-radius` 单独指定，0 表示使用粒子半径。指定值的下限为 \(\Delta x/4\) − 刀半厚。旧香蕉预设使用固定 0.5 mm，在 6 mm 下被下限提高至 1.0 mm | 论文半径形式；旧固定距离和下限为本地选择 |
| PIC/FLIP 比例 | 0.95 | DexLab |
| \(s_n\)，\(\gamma_k\)，\(\gamma_c\) | 0.1，0.05，0.1（在参考步长下） | DexLab |
| \(b_\text{lead}\)，\(b_\text{back}\)，\(u_\text{min}\) | 2×刀接触距离，\(2\Delta x\)，0 | 本地选择，论文定义符号但未给数值 |
| \(h_k\) | \(6\Delta x\) | 论文 |
| 香蕉材料 | E=400 kPa，ν=0.45，ρ=1100，屈服 30 kPa，\(G_c\)=700，ζ=0.9，刀 0.1 kg，μ=0.2 | 论文 §IV-C |
| 香蕉步长 | 62.5 μs，接触参数以 25 μs 为参考 | 本地选择，见 RESULTS.md 步长扫描 |
| 机器人耦合间隔 | 1 ms（MuJoCo 步长） | 本地选择 |
| 表面刀路 | 记录实际材料子步的刀位、切割开关和刀次；保持不动的子步只存首尾。默认每 40 ms 更新，可用 `--observation-interval` 加密；材料节点在观测间线性插值 | 本地选择 |
| 表面 QEF 边界 | 解约束在参考单元每侧内缩 0.001×表面格距的盒中；多区域单元以 1e-3×Hermite 样本数弱正则到各区域质心。共享裂口前沿满足对应 Hermite 平面约束；平面与单元盒无公共解时保留有界 QEF。两区域单元复制同一个联合 QEF 代表点，保留两侧独立编号；近似平面的切口四边形按所属单元编号统一对角线 | 消除共享前沿偏离切面和两侧参考代表点错位；三区域及以上单元仍分别求解 |

## 接触刚度与步长

论文式 11/15 的法向刚度是 \(k = s_n m_\text{eff}/\Delta t^2\)，切向 \(k_t = \gamma_k m_\text{eff}/\Delta t^2\)。固定 \(s_n\) 时，步长减半，接触就硬 4 倍，所以力曲线不可能随步长收敛。

`Config.contact_reference_dt > 0` 时，`normal_scale` 和 `tangent_stiffness` 视为该参考步长下的值，并按 \((\Delta t/\Delta t_\text{ref})^2\) 缩放，从而保持 \(k\) 不变。缩放后的因子必须 ≤ 1（否则单步过冲），对香蕉（参考 25 μs，\(s_n=0.1\)）这意味着 \(\Delta t \le 79\) μs；默认 62.5 μs 时 \(s_n = 0.625\)。预设默认启用；`franka_run --contact-scaling per-step` 恢复论文的固定 \(s_n\) 写法。

切割功阈值只通过 \(G_c(1-\zeta)/\zeta\) 使用切割能和锋利度。保持这个组合不变的参数对产生相同阈值；力曲线标定时固定其中一个，或用额外测量约束二者。损伤影响刃口接触分量，几何侧标签独立决定网格分离。

## 外部刀位保持时的网格刀面约束

`external_tool=True, tool_substep="hold"` 在原有接触和弹性冲量完成 P2G 后、G2P 推进位置前检查刀身两侧平面。接触响应使用本步节点质量及相容场交换：`W = Bᵀ M⁻¹ C B`，其中 B 将各接触的法向冲量按粒子权重散到逻辑节点，C 是现有的场间动量交换。所有接触同时迭代，避免把单粒子的对角有效质量直接放大后产生过冲。补充冲量的反向合力和力矩返回原有机器人耦合接口。

刀位在这些子步中保持不动，因此补充约束使用实际网格速度和当前刀面；原有罚接触仍使用机器人提供的速度与移动性矩阵。刀身接触保存进入侧，绕过有限边缘后释放侧别。速度约束不附加回弹，粒子位置由约束后的网格速度平流。刃口斜面、损伤功、摩擦、粒子对接触和表面计算保持原实现；自由刀和 `predict` 模式仍走原有接触路径。

GPU 投影梯度按 48 次一批细化，速度残差目标 1 μm/s，最多 384 次；未收敛时耦合接口报错。无约束违反的子步跳过迭代。

## Maxwell 黏弹性

`Config.maxwell_tau` \(\tau\) > 0 且 `maxwell_fraction` \(\beta\) > 0 时启用（广义标准线性固体）。弹性变形 \(f = F F_p^{-1}\) 分两支：平衡支为共旋弹性，\(\mu_\infty = (1-\beta)\mu\)、\(\lambda + \tfrac23\beta\mu\)；Maxwell 支 \(f_m = f F_v^{-1}\)，Kirchhoff 应力 \(2\beta\mu\,\mathrm{dev}\log V_m\)。两支之和的小应变剪切、体积模量与关闭时相同，长期剪切模量为 \((1-\beta)\mu\)，体积部分不松弛。每步在塑性投影之后把 \(f_m\) 的偏 Hencky 应变乘以 \(e^{-\Delta t/\tau}\)（应变保持不变时是精确解，对任意 \(\Delta t/\tau\) 稳定；\(\tau \ll \Delta t\) 时该支每步归零），\(F_v\) 按等容增量更新。每步增量 \((\Delta t/\tau)\,\varepsilon\) 远小于 \(F_v\) 在 float32 下的分辨率，所以 \(F_v - I\) 存为 float64。屈服仍按瞬时模量的 Hencky 应变判定，加载慢于 \(\tau\) 时流动应力为 \((1-\beta)\sigma_y\)。关闭时调用原来的 kernel。

## 可微弹性与固定接触

`differentiable.ElasticTrajectory` 在未切割单场上复用 P2G、网格速度和共转应力；G2P 与 MUSL 写入下一步状态。每一步保存独立数组，分段 VJP 在反向时重算段内状态。E 使用可求导数组；参考采样、密度、ν 与步长保持固定。

令 \(F=U\Sigma V^T\)、\(R=UV^T\)。极分解旋转对传入矩阵伴随 G 的 VJP 为 \(UBV^T\)，其中 \(B_{ii}=0\)，\(B_{ij}=[(U^TGV)_{ij}-(U^TGV)_{ji}]/(\sigma_i+\sigma_j)\)。对正行列式 F，重复正奇异值处这个分母仍非零。`elastic_stress_grad` 用该 VJP 和行列式、逆矩阵导数计算共转应力的伴随；正向仍调用原 `corotated_stress`。

`differentiable_contact.PrescribedContact` 复用刀距离、预测法向罚函数和带历史的库仑摩擦；外部刀的移动性为零，切向历史逐步保存。梯度沿本次执行的接触和粘滑分支传播。切换扫描与轨迹优化的实验安排见 [DIFFERENTIABLE.md](DIFFERENTIABLE.md)。

接触的 `checkpoint_vjp` 在段边界同时保存 x、v、F 和刀/砧板的切向历史。反向重算一段后，把这四类状态的伴随传给上一段；每段累加工具参数与 E 的梯度。逐步反力罚项在所属段求导，末端位置损失在最后一段求导。`tool_at_step` 和 `objective` 回调在正向与重算时执行相同计算。

`scripts.diff_optimize` 将分段刀速作为优化量，并在 Warp kernel 中积分刀位，使位姿与速度同时进入梯度链。目标是指定粒子区域的末端位置，刀速用变量边界约束，反力超限用平方罚项。每条最优刀路都交给原 `Solver` 重放。接触核扫描分别记录接触建立处的导数折点、静动摩擦切换的力跳变；整段轨迹差分同时统计接触/摩擦分支变化。

`differentiable_material.MaterialTrajectory` 逐步保存 Fp 和投影后的应力，复用原 Hencky von Mises 投影的正向公式。投影后伸长为 t、投影前为 s；伴随对称项使用 `(t_i-t_j)/(s_i-s_j)`，相对伸长差 ≤1e-4 时使用对称极限形式，在相等处为 `a*t_i/s_i`，其中 `a=屈服半径/偏应变范数`。非屈服分支原样传回弹性变形并保持 Fp。E、屈服应力、F 和 Fp 均进入梯度链。

固定侧码下，式 8 的相容场交换是常系数线性映射。`FixedTopology` 根据参考质量与相容关系生成稀疏行，动量、冲量和 MUSL 重映射共用该映射；反向通过同一系数的转置传播。参考采样、粒子质量及侧码在一段求导中保持固定。

`differentiable_cutting.FixedCutContact` 读取原求解器逐步产生的侧码、损伤释放标记和接触粒子对。粒子对按稳定 ID 对应前一步的切向历史；刀、板和粒子对的状态分别写入新数组。刀位、刀速、粒子状态和摩擦系数参与求导。固定接触标记时，两侧差分评估相同的法向闭合、粘滑和屈服分支公式；`--natural-branches` 改为每次重新判断这些局部分支。刀刃最近特征、切割计功带的几何判断继续随位姿计算。

`FixedCutContact.checkpoint_vjp` 保存 x、v、F、Fp、应力及刀/板/粒子对的摩擦历史和损伤。网格约束逐次保存投影迭代，保持原求解器的迭代公式与批次；进入侧和求解批数作为离散状态。

`Solver(precision="float64")` 与 `MaterialTrajectory(precision="float64")` 分别运行原正向流程和可微流程。`precision.py` 将同一份 Warp 函数的浮点类型及常量提升到双精度；分支和公式保持相同。原求解器继续使用动态多场注册与场交换，可微流程使用固定分支的稀疏映射；同精度对照检验两套推进结果。

## 损伤释放的步内积分

`EventCutContact` 使用每粒子阈值 Q = G_c V₀^(2/3) (1−ζ)/ζ（J）。将本步接触核算出的刃向冲量记为 J，正向相对推进速度记为 u，当前损伤为 D；步内恒定力和 u 下，完整步损伤增量为 Ju/Q，剩余断裂功为 R=(1−D)Q。若 Ju≥R，释放时刻 τ=Δt R/(Ju)，保留刃向冲量 R/u，并移除余下的 J−R/u；该步损伤功取 R，损伤变为 1。已释放粒子继续撤除原计功带内的刃向推力。其余方向的冲量和摩擦历史使用现有公式，粒子通过原一阶 MPM 步更新。

τ 的导数通过冲量进入刀力和粒子状态，自动微分与有限差分评估同一正向公式。`CutHistory.damage_work` 记录各粒子的损伤计功，核对 ΣW = QΣΔD；整体系的输入功、储能、塑性及摩擦耗散另行统计。侧码及粒子对集合由调用者给定；`cut_hold=False` 使释放前后的刃口接触法向规则保持一致。

事件积分思路参考 [DiffTaichi 的 TOI 实现](https://github.com/taichi-dev/difftaichi/blob/master/examples/mass_spring.py)。[Saltation matrices](https://arxiv.org/abs/2306.06862) 给出跨事件的状态灵敏度更新；当前实现对上述一阶离散积分直接求导。事件时间、终态与梯度的步长对照使用 `scripts.diff_release_validate`。

## 与论文的差异

| 项 | 论文 | 这里 |
|---|---|---|
| 粒子位置更新 | 未写明 | 用插值网格速度推进（de Vaucorbeil 2020 TLMPM 式 13），速度用 PIC/FLIP |
| 刀身穿透处理 | 粒子罚接触 | 外部 `hold` 模式补充网格刀面法向约束，见上节 |
| 接触刚度 | 每步常数 \(s_n\) | 默认按参考步长固定物理刚度，见上节 |
| 黏弹性 | 无 | 可选 Maxwell 分支，默认关闭，见上节 |
| 损伤释放积分 | 损伤达到阈值后移除刃口推力 | 原 `Solver` 和 M4 在子步边界切换；M5 `EventCutContact` 根据剩余断裂功积分释放步的刃向冲量，使释放时刻进入状态导数，见上节 |
| 耦合区间内的刀具 | 位姿与速度保持不变 | 默认相同；`tool_substep='predict'` 用 6×6 移动性矩阵逐子步更新刀速并推进位姿 |
| 接触检测 | 未写明 | 默认用设备端条件重建邻居表；`hash` 每子步重建 Warp HashGrid，`all` 全对检测（每粒子 64 线程），按完整搜索核对接触集合 |
| 切割次数 | 不限 | int64 三进制码，上限由 \(3^{k+1}\cdot\)节点数 \(< 2^{63}\) 决定（香蕉 6 mm 网格 33 刀，3 mm 31 刀，2 mm 30 刀） |
| 香蕉几何 | 实物扫描 | DexLab 合成 120 mm 香蕉 |
| 刀具 | 未公开 | 80×40×1 mm 楔形，5 mm 刃口斜面 |
| 地面 | 刚性支撑 | 粒子：砧板顶面矩形内的罚函数接触。刀：Franka 由 MuJoCo 实体盒碰撞；其余刀（导向、位姿伺服、自由落体）共用 `tool.step` 的同一个刀顶点—砧板接触（速度约束 + 库仑摩擦），三者使用同一砧板矩形 |
| 控制器 | 未公开 | libfranka 笛卡尔阻抗示例公式，或保持 SERL++ 公式的本地移植（MuJoCo 四元数替代 scipy；测试对照固定版本源码） |
| 连续切割任务 | 论文演示多刀切割 | `CutSequence` 指定切点、速度和推开距离；按实测刀位到达底部触发推开，切割事件由下刀阶段启用 |
| Franka 中片抬升 | Fig. 4 两次径向切割后抬起中片 | 复用 Fig. 4 圆柱与材料；机器人耦合步长 1 ms，接触刚度以场景原来的 2 ms 为参考。100×80×1 mm、0.1 kg 刀具通过 115 mm 偏置支架安装，旋转刚度 80 Nm/rad，保留刀—板碰撞。刀面转平后沿对角方向托住扇形块，目标抬高 40 mm；实测位姿到达后才进入下一阶段 |
| 刃前带宽 \(b_\text{lead}\) | 定义于式 4，未给数值 | 2×刀接触距离 `blade_offset`：默认 3 mm 香蕉为 1.5 mm，6 mm 香蕉为 3.0 mm；显式刀距可单独改变 |
| 刃压到砧板时的底层侧标签 | 只按切割带（式 6） | `contact.assign` 另外给刃高度 ≤ \(R_p\) 时刃两侧、中心在刃上方 \(R_p\) 以内且在砧板矩形内的底层粒子打侧标签：刀倾斜时这些粒子在刃平面以下，切割带永远不包含它们，块切不透（Fig. 3） |
| 砧板罚因子 | 与刀共用 \(s_n\) | 单独的 `board_normal_scale`；Fig. 3/4/7/8 在论文步长下刀与砧板都取 \(s_n\) = 1（0.1 时每步穿透约 \(N g\Delta t^2/s_n\)，底层沉入砧板），香蕉仍为 0.1 |
| 扰动采样 | 未写明 | 每轴 ±Δx/4 均匀扰动后，把低于 \(R_p\) = Δx/4 的粒子按 \(z \to R_p + |z - R_p|\) 反射到砧板以上（`geometry.points`）；否则初始穿透的恢复冲量把整块抬起（修正前 6 mm 香蕉最低粒子弹起 8.4 mm，3 mm 1.2 mm） |
| 场景设置 | Franka 操作 | Fig. 4/7/8 用规定刀路（`scenarios.py`，非 Franka）；Fig. 8 每刀后把切片推倒；Fig. 3 刀绕刀厚轴倾斜（论文只给角度）；Fig. 4/7/8 的 \(G_c\) = 10 J/m² 为本地取值；导向刀驱动 1000 N/m、20 N·s/m（论文材料预设，DexLab 250 N/m）；摩擦默认 0.3 / 0.2（DexLab），论文香蕉 0.2 / 0.2。详见 RESULTS |
| Fig. 5 独立对照 | 两刀三块、0.5 mm 刀厚、相同采样与弹性参数；内部四面体估计中块体积 | 默认 60×30×30 mm 方块、E=6 kPa、ν=0.4、ρ=1100、5 mm 网格、0.25 ms 步长；退刀后实际左块速度 −30 mm/s，其余速度保持。每刀完全插入时冻结材料成员，第二刀细分余块，并核对机械连通块。切口指标为粒子列开口增量；主要体积指标为内部四面体及其覆盖率，另用同块 26 邻居重建 Jacobian 积分完整材料单元，秩亏、跨切口和翻转时输出无效。最新共同实验将步长改为 0.125 ms、刀半长改为 60 mm，见 RESULTS |
| CRESSim-MPM 对照实现 | 论文使用 CRESSim-MPM | 编译原版 `09aa5009` 的 `CpuMlsMpmSolver`。共旋分支缺少 `break`；`--fix-corotated-dispatch` 在独立头文件副本中补此行，保留原版模式。质量单位换算固定网格阈值 1e-4，计算域余量可配置；每子步检查完整二次 B 样条支撑，粒子中心安全区为原点 +2.5h 至原点 +(N−3.5)h。当前刀面恰落网格节点时，侧别和结果对网格原点敏感 |
| TopoCut 基线损伤 | 按体积比阈值 | 先在重力下静置（默认 1 s）再开启损伤，\(J\) 相对静置后的构形计（`MlsSolver.start_damage`）：软材料（10 kPa）自重压缩本身会越过阈值，静置算作损伤时 Fig. 4 有 23% 粒子在刀动之前就损伤 |
| 已与论文核对 | — | ① 自接触：论文 §III-E"For self-contact, nearby reference neighbors are excluded … Pairs with conflicting cut-side codes are exempt"，没有给距离。代码对切割码兼容且参考距离 \(< 2\Delta x\) 的粒子对不算接触（`contact.pair_at`）：三线性权重下两粒子共享网格节点的条件是每轴参考距离 \(< 2\Delta x\)，这里用欧氏距离，对角方向共享节点的少数粒子对仍是接触候选。② 式 5：\(d_k\) 是"front–back direction … obtained by transforming their fixed tool-frame counterparts using the current tool pose"，即刀坐标系里固定的方向，代码取刀的 −z 轴（`resist_cut`、`assign`），与论文一致；\(u_{c,p}\) 默认用相对速度（`cut_work="relative"`），也与论文一致。论文的 \(u_\text{min}\) 没有给值，代码取 0（`advance > 0`） |
