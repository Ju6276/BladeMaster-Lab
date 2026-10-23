# 实现与论文对照

论文：[BladeMaster, arXiv:2609.27342v2](https://arxiv.org/html/2609.27342v2)。

## 公式到代码

| 论文 | 内容 | 代码 |
|---|---|---|
| 式 1–2 | 参考构型 TLMPM P2G，固定三线性权重 | `geometry.stencil`，`material.p2g` |
| §III-B | fixed-corotated + von Mises 塑性（等容增量变分返回映射），可选牛顿黏性、Maxwell 黏弹性 | `material.p2g`，`material.plastic_at`，`plasticity.py`，`material.viscous_impulse`，`material.maxwell_at` |
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

`Config.maxwell_tau` \(\tau\) > 0 且 `maxwell_fraction` \(\beta\) > 0 时启用（广义标准线性固体）。弹性变形 \(f = F F_p^{-1}\) 分两支：平衡支为共旋弹性，\(\mu_\infty = (1-\beta)\mu\)、\(\lambda + \tfrac23\beta\mu\)；Maxwell 支 \(f_m = f F_v^{-1}\)，Kirchhoff 应力 \(2\beta\mu\,\mathrm{dev}\log V_m\)。两支之和的小应变剪切、体积模量与关闭时相同，长期剪切模量为 \((1-\beta)\mu\)，体积部分不松弛。每步在塑性投影之后把 \(f_m\) 的偏 Hencky 应变乘以 \(e^{-\Delta t/\tau}\)（应变保持不变时是精确解，对任意 \(\Delta t/\tau\) 稳定；\(\tau \ll \Delta t\) 时该支每步归零），\(F_v\) 按等容增量更新。每步增量 \((\Delta t/\tau)\,\varepsilon\) 远小于 \(F_v\) 在 float32 下的分辨率，所以 \(F_v - I\) 存为 float64。塑性返回使用瞬时模量对应的共旋应力；小应变、加载慢于 \(\tau\) 时流动应力趋于 \((1-\beta)\sigma_y\)。关闭时调用原来的 kernel。

## 材料能量统计

`energy.material_energy` 用独立 NumPy 计算粒子动能、重力势能和弹性储能。令弹性变形 \(F_e=FF_p^{-1}\) 的奇异值为 \(s_i\)、行列式为 \(J_e\)，参考体积能量密度为 \(\Psi=\mu\sum_i(s_i-1)^2+\frac12\lambda(J_e-1)^2\)。三项分别为 \(\sum_p\frac12m_p|v_p|^2\)、\(-\sum_p m_p g\cdot x_p\)、\(\sum_p V_p^0\Psi_p\)。固定 \(F_p\) 时，储能对 \(F\) 的方向导数与求解器第一 Piola 应力的虚功逐项核对。

自然轨迹报告记录起终点的三项材料能量，以及 `knife_work_j`。每个实际积分段的刀具输入功为 `Σ J_p·(v_tool+ω×r_p)`，其中 J 是施加给粒子的刀接触冲量，r 是同一接触查询的世界坐标力臂；等价于负刀反力/反力矩与平移/转动速度的功率积分。网格刀面约束另加 `J_grid·v_tool+L_grid·ω`，L 是该约束冲量的矩。接触点速度与广义力的配对见 [MuJoCo 接触雅可比](https://mujoco.readthedocs.io/en/stable/computation/index.html#contact)，这里直接用本求解器的冲量计功。

`CutHistory.tool_work` 记录本步输入功（J），正值表示刀向材料系统输入能量。中点积分取终点更新实际使用的中点冲量；事件积分只累计接受的分段，求根试算和半步预测不重复计功。损伤释放后的冲量参与计功，`damage_work` 保持独立统计。

`energy_ledger.EnergyLedger` 记录 `GeometryEventContact` 接受的分段。法向接触储能为 `½kₙ min(gap,0)²`，切向储能为 `½kₜ|h|²`；粒子对使用约化质量对应的刚度。塑性耗散为同一终点 F 在返回映射前后的储能差。摩擦按下文恒等式分为阻尼、滑动和时间积分三项；切平面投影、脱离清零与接触对丢失的储能差单独记录。PIC 项为 `τλΣm v_stage·(v_stage−v_flow)`，其中 `v_flow=Σw v_grid`。

原始收支 `global_balance_residual_j` 为动能、材料及接触储能变化加上述耗散，减去刀与重力外功，包含分侧和换刀造成的离散能量变化。另记录法向/摩擦/释放冲量与 `(v_flow−v_contact)` 的有符号功差、释放前后冲量功差及网格约束功差；这些量用于定位传递误差。`integration_residual_j` 从逐段原始残差中扣除摩擦、释放和网格约束功差，保留时间积分误差。损伤计功按 `QΔD` 独立核对，不重复加入耗散。检查点反向重算关闭账本，求根试算只在接受段落时入账。

M5 中点阶段的 `MaterialTrajectory.flow_velocity` 对当前速度先 P2G，再按侧码耦合，最后插值回接触点。若 W 为粒子插值、M 为网格质量、C 为多场动量耦合，接触速度为 `u=W M⁻¹ C Wᵀ m v`。材料接触加速度使用相同传递，故 `Σm v·a_contact=Σf_contact·u`。刚体与粒子对的相对速度、切向历史增量和损伤推进都使用 u；PIC 阻尼仍作用于存储速度 v。未切开、部分切开和交叉侧码的虚功及转置导数用独立 NumPy 映射核对。

`MaterialTrajectory.step(grid_load=...)` 接收每个逻辑网格槽的附加冲量（N·s），在相容场交换 C 之前与 P2G 冲量相加。网格力 `f_g` 对应粒子加速度 `a=W M⁻¹ C f_g`，与其共轭的网格速度为 `u_g=M⁻¹ C Wᵀ m v`；现有质量加权耦合满足 `Cᵀ M⁻¹=M⁻¹ C`，因此 `Σm v·a=Σf_g·u_g`。界面两侧用相同插值及其转置分别取得速度、散射等反作用力，可直接核对界面功。显式裂纹的双侧运动与牵引组织参考 [CRAMP 的裂纹牵引说明](https://osupdocs.forestry.oregonstate.edu/index.php?title=Traction_Laws)，这里的多场功配对由现有 C 推导。

## 内聚界面元件

`cohesive.cohesive_response` 使用各开裂方向相同的刚度 K（Pa/m）、强度 σ（Pa）和断裂能 G_c（J/m²）。起裂/完全断裂位移为 `δ₀=σ/K`、`δ_f=2G_c/σ`，`CohesiveParameters` 检查 `0<δ₀<δ_f`。开口 g、当前法向 n 的压缩量为 `s=min(g·n,0)`，损伤驱动量为 `a=g−sn`；历史 `κ=max(κ_old,|a|)`，中间段损伤 `D=δ_f(κ−δ₀)/(κ(δ_f−δ₀))`，两端分别为 0、1。参考面积上的储能 `ψ=K[(1−D)|a|²+s²]/2`，累计断裂耗散 `Γ=G_c clamp((κ−δ₀)/(δ_f−δ₀),0,1)`。牵引 `T=K[(1−D)g+Dsn]` 保留断裂后的压缩响应。加载—卸载切换处力连续、切线有折点；测试分别核对两侧解析导数。

界面变形 F_Γ 映射两侧当前面的平均切向量，`n=normalize(cofactor(F_Γ)N)`。元件同时输出固定 D 时的 `P_Γ=∂ψ/∂F_Γ`（N/m），使功率为 `T·ġ+P_Γ:Ḟ_Γ`。双侧参考插值 N_i 与切向梯度 ∇_ΓN_i 对应网格力 `A₀[±N_i T−P_Γ∇_ΓN_i/2]`，相同映射推进面位置与切向量。参考面积 A₀ 保持固定；刚体旋转时两项功相消。

`cohesive_contact_response` 计算单边约束 `g·n≥0` 的共轭力：给定压力乘子 `p≤0`（Pa），输出 `T_c=p n` 和 `P_c=∂(p g·n)/∂F_Γ`（N/m）。`surface_normal_pullback` 对归一化余因子法向求转置导数，与上述弹性界面共用。两侧面位置和平均切向量同时推进；纯刚体转动下 `T_c·(Ωg)+P_c:(ΩF_Γ)=0`。压力乘子由耦合约束求解，局部核不新增材料参数。

`cohesive.cohesive_proximal_response` 求解初始刚性内聚增量：在 `g·n≥0` 下最小化 `|g−q|²/(2λ)+ψ+Γ`，其中 `λ` 的单位为 m/Pa。`δ_f=2G_c/σ`、`κ=max(κ_old,|g|)`；`0<κ<δ_f` 时 `ψ=σ(1−κ/δ_f)|g|²/(2κ)`，其余为 0；`Γ=σ min(κ,δ_f)/2`。闭合界面承载时 `g=0`、牵引 `T=q/λ`，储能和断裂耗散均为 0。`ExtrinsicCohesiveParameters` 检查正参数及 `λσ<δ_f`，后者使局部增量势严格凸。

多界面原型以 ADMM 约束 `g=q+B u`，粒子修正的惯性能为 `uᵀm u/(2Δt²)`，面能按参考面积 A 积分。原始界面迁移矩阵允许秩亏；线性子问题使用 `I+ρΔt² sqrt(A) B m⁻¹ Bᵀ sqrt(A)`，单位阵来自惯性项。局部开口更新调用上述近端核，牵引按 B 的转置回传粒子；原型检查动量—开口一致性至相对误差 `1e-11`。重复积分点时将面积等分，核对相同粒子结果。

## 可微弹性与固定接触

`differentiable.ElasticTrajectory` 在未切割单场上复用 P2G、网格速度和共转应力；G2P 与 MUSL 写入下一步状态。每一步保存独立数组，分段 VJP 在反向时重算段内状态。E 使用可求导数组；参考采样、密度、ν 与步长保持固定。

令 \(F=U\Sigma V^T\)、\(R=UV^T\)。极分解旋转对传入矩阵伴随 G 的 VJP 为 \(UBV^T\)，其中 \(B_{ii}=0\)，\(B_{ij}=[(U^TGV)_{ij}-(U^TGV)_{ji}]/(\sigma_i+\sigma_j)\)。对正行列式 F，重复正奇异值处这个分母仍非零。`elastic_stress_grad` 用该 VJP 和行列式、逆矩阵导数计算共转应力的伴随；正向仍调用原 `corotated_stress`。

`differentiable_contact.PrescribedContact` 复用刀距离、预测法向罚函数和带历史的库仑摩擦；外部刀的移动性为零，切向历史逐步保存。梯度沿本次执行的接触和粘滑分支传播。切换扫描与轨迹优化的实验安排见 [DIFFERENTIABLE.md](DIFFERENTIABLE.md)。

接触的 `checkpoint_vjp` 在段边界同时保存 x、v、F 和刀/砧板的切向历史。反向重算一段后，把这四类状态的伴随传给上一段；每段累加工具参数与 E 的梯度。逐步反力罚项在所属段求导，末端位置损失在最后一段求导。`tool_at_step` 和 `objective` 回调在正向与重算时执行相同计算。

`scripts.diff_optimize` 将分段刀速作为优化量，并在 Warp kernel 中积分刀位，使位姿与速度同时进入梯度链。目标是指定粒子区域的末端位置，刀速用变量边界约束，反力超限用平方罚项。每条最优刀路都交给原 `Solver` 重放。接触核扫描分别记录接触建立处的导数折点、静动摩擦切换的力跳变；整段轨迹差分同时统计接触/摩擦分支变化。

`differentiable_material.MaterialTrajectory` 逐步保存 Fp 和投影后的应力，与原求解器共用 `plasticity.corotated_return`。设试算弹性主伸长为 s，等容对数增量为 d，更新后伸长为 `t=exp(log(s)+d)`；返回映射求解 `min Σ(t_i−1)²/2+r‖d‖`、`Σd_i=0`，其中 `r=√(2/3)σ_y/(2μ)`。等容约束使体积储能项保持不变。弹性分支条件为 `‖dev(s(s−1))‖≤r`，塑性分支满足 `dev(t(t−1))+r d/‖d‖=0`，并更新 `Fp=inv(U diag(t) Vᵀ)F`。

返回映射以 float64 求解，使用带 Armijo 回溯的修正 Newton 方向；在塑性增量的径向/正交基中求逆，避免增量趋于零时丢失径向曲率。方向中的正定化只用于寻找下降步，最终检查原驻点方程，残差目标为 `2e-13 max(1,‖dev(s(s−1))‖)`，最多 80 次。未收敛时报错。`Ψ_trial−Ψ_new≥√(2/3)σ_y‖d‖` 与 `det(Fp_new)=det(Fp_old)` 分别检查塑性耗散和体积保持。

伴随由驻点方程的隐式导数计算；对称谱项 `(t_i−t_j)/(s_i−s_j)` 用两条驻点方程相减得到的等价式计算，重复伸长处使用对数差商的连续极限。E、屈服应力、F 和 Fp 均进入梯度链。

固定侧码下，式 8 的相容场交换是常系数线性映射。`FixedTopology` 根据参考质量与相容关系生成稀疏行，动量、冲量和 MUSL 重映射共用该映射；反向通过同一系数的转置传播。参考采样、粒子质量及侧码在一段求导中保持固定。

`differentiable_cutting.FixedCutContact` 读取原求解器逐步产生的侧码、损伤释放标记和接触粒子对。粒子对按稳定 ID 对应前一步的切向历史；刀、板和粒子对的状态分别写入新数组。刀位、刀速、粒子状态和摩擦系数参与求导。固定接触标记时，两侧差分评估相同的法向闭合、粘滑和屈服分支公式；`--natural-branches` 改为每次重新判断这些局部分支。刀刃最近特征、切割计功带的几何判断继续随位姿计算。

`FixedCutContact.checkpoint_vjp` 保存 x、v、F、Fp、应力及刀/板/粒子对的摩擦历史和损伤。网格约束逐次保存投影迭代，保持原求解器的迭代公式与批次；进入侧和求解批数作为离散状态。启用 `timings` 时，逐项核对重算的连续状态、切割侧码、接触对、刀面进入侧、接触模式、网格求解批数及几何事件边界状态。

`differentiable_natural.NaturalCutContact` 每步从当前 x/v 和三进制侧码生成接触对，调用原 `contact.assign` 在步末更新侧码。候选对用 CPU 空间树查找，半径 `2R + 2Δt max|v|` 覆盖预测接触；原求解器与可微流程在 GPU 上共用 `contact.pair_activity` 筛选。切向历史按 `p*n+q` 继承，脱离后重新接触从零开始。每次扰动重新计算侧码、接触集合、接触模式和屈服分支；Warp 对本次执行的连续运算求导。

`CutHistory` 保存侧码、排序后的粒子对 ID 和刀次；检查点重算同时核对连续状态及这些离散状态。同刀暂停保留 D，新刀按原求解器规则清零 D 并保留历史侧码。`release="boundary"` 使用原步末释放；`release="integrated"` 使用下节的步内刃向冲量积分。`scripts.diff_natural_validate` 从首次分侧附近的同一状态开始，逐步对照原求解器、比较不同检查点长度，分别记录差分误差及侧码/接触/释放的变化数。

`checkpoint_device="cpu"` 将检查点的全部粒子、接触及事件状态复制到主机，每段反向前恢复到计算设备；完成该段后释放已处理的检查点。保存到 CPU、反向时取回的存储策略可参照 [PyTorch save_on_cpu](https://docs.pytorch.org/docs/stable/autograd.html#torch.autograd.graph.save_on_cpu)，这里使用 `warp.clone` 搬移数组。时间步、事件求根与 VJP 公式共用同一实现，CPU/GPU 检查点均与完整 Tape 核对。

## 几何分侧的步内积分

`GeometryEventContact` 使用分侧区域各平面的有符号距离 g。RK2 二次位置延拓与三个时刻的 g 给出进入候选，随后在实际 TLMPM 推进 `Φτ(state)` 上用带括区间的割线迭代求解 `g(Φτ(state), knife(t+τ))=0`。根处检查完整区域的所有不等式，再写入该粒子的侧码并重新生成多场交换，推进余下时间。求根、事件前后两段状态、接触历史和时长加权刀力均由同一正向计算进入 Warp Tape。

默认中点积分的均匀自由落体单步为 `v_new=v+τg`、`x_new=x+τv+½τ²g`；`integration="symplectic"` 的位置项为 `τ²g`。解析事件测试分别核对两种离散流的根，跨时间步的状态与导数另做收敛检查。

刀位在完整步内按给定线速度、角速度推进，中点积分的接触计算使用各阶段的实际刀姿。求根空间容差是 `32×机器精度×坐标尺度`，最多 48 次迭代；括区间丢失、根落在区域外或事件排序失败时直接报错并要求减小完整步长。`CutHistory.geometry_events` 保存本步粒子 ID 与可求导的发生时间，扰动实验同时比较左右差分、事件集合和顺序。

刀刃平面穿越后的第一个积分段使用事件后的计功分支：`CutHistory.edge_departed` 保存刚发生穿越的粒子，停止刃前计功和刃前保持；接触几何、粒子位置及法向罚力继续使用实际状态。一次正时长推进后清除此标记，恰好发生在步末的标记进入下一步及检查点。这是混合系统在事件边界选择后继模式的处理，避免求根残差的舍入符号重新选择事件前模式。

运行 `scripts.diff_natural_validate --geometry event` 使用该积分；`--geometry boundary` 使用原步末分侧。两者共用材料、接触和损伤公式，完整步长及接触/PIC 参考时间分别记录在配置中。

`Solver(precision="float64")` 与 `MaterialTrajectory(precision="float64")` 分别运行原正向流程和可微流程。`precision.py` 将同一份 Warp 函数的浮点类型及常量提升到双精度；分支和公式保持相同。原求解器继续使用动态多场注册与场交换，可微流程使用固定分支的稀疏映射；同精度对照检验两套推进结果。

`MaterialTrajectory.correct_displacement` 接受全局约束求解得到的网格诱导位移增量 `u=Δt·Δv`，完成同一一阶 MUSL 步：`x=x_free+u`、`v=v_free+u/Δt`，F 增量由 u 的粒子到网格重映射与相容场交换计算，塑性返回从段初 Fp 更新。它与先散射约束冲量再执行 `step(grid_load=...)` 代数等价，直接使用求解位移可避免从约束反力重构位移时的抵消；13 项状态和导数对照覆盖未切开、部分切开、交叉侧码、弹性/塑性和两档 PIC/FLIP。该入口使用一阶 MUSL 的位置与 F 更新，中点 RK2 仍由原阶段入口推进。

## 事件分段的中点积分

`GeometryEventContact` 默认采用 `integration="midpoint"`：对参考网格上的半离散动力学执行显式中点 RK2。第一阶段推进半段，第二阶段用中点的速度、应力、刀位和接触力推进整段。粒子位置率为插值网格速度，变形率为参考梯度作用于当前速度，速度率包含材料/接触加速度及固定单位时间的 PIC 残差阻尼。塑性 Fp 在预测点和终点分别从段初 Fp 返回映射。各阶段、分段时长及返回映射均进入同一 Warp Tape。

接触阶段使用该时刻的间隙计算法向罚力；切向力和历史增量共同满足下节的 Kelvin–Voigt–滑动元件方程。终点将历史投影到新的切平面，接触脱离时清零。粒子对使用相对速度及两粒子的有效质量。刀力积分采用中点力乘实际分段时长。`integration="symplectic"` 调用原 MUSL 步进；M4 固定分支和原 `Solver` 使用此步进。

切向历史只在阶段实际接触时更新；未接触或脱离时清零。中点实际接触对重新搜索，与初始候选集合合并，按粒子 ID 映射段初历史。第二阶段使用段初弹性历史和中点相对速度求整个分段的增量，避免重复累计预测半步。粒子位置由网格插值速度推进，因此初始粒子速度的预测查询可能漏掉中点接触。

中点阶段的刀位连续推进，网格刀面约束使用 `v_grid − v_tool − ω×r` 的法向分量；原 `hold` 子步仍使用固定刀位约束。相对接触速度的构造与 [MuJoCo 接触雅可比](https://mujoco.readthedocs.io/en/stable/computation/index.html#contact) 一致，投影求解仍复用本仓库的相容场算子。

阶段网格的约束候选速度为 `v_base_grid + τ a_stage`，投影增量加回同一阶段的力增量；段末速度由段初速度加这些增量得到。这样中点预测阶段已施加的投影不会使终点漏掉法向冲量。反力累计使用终点实际增加的冲量除以分段时长。

中点 RK2 的系数对照 [PETSc TSRK2B](https://petsc.org/release/manualpages/TS/TSRK2B/)，切向接触采用下节的本构积分。MPM 的传递与时间积分误差分析见 [Berzins 2021](https://www.scipedia.com/wd/images/e/e1/Draft_Content_867464886p1127.pdf)。这里在 Warp 中实现这些阶段和返回映射。

`diff_natural_refine --phases 0 .25 .5` 固定同一初态和结束时刻，改变第一步长度，从而比较时间网格起点对状态及梯度的影响。检查点保存每步时长并按相同区间重算。

几何候选保留二次轨迹与各切割区域平面的交点，同时检查交点之间是否位于整个区域内；旋转查询按角跨度 ≤π/2 分段，使用各段开始、中点和结束的实际姿态。该划分只用于搜索括区。入区后又离开的轨迹即使两端均在区域外也参与求根。候选的插值时刻用于选择括区，最终事件时刻和状态由实际材料推进计算。

## 应力与塑性伴随核对

应力使用本仓库的解析 VJP，塑性返回映射使用驻点方程的隐式 VJP，其余连续算子由 Warp Tape 生成反向。旋转导数的分母取正奇异值之和；塑性谱导数通过稳定对数差商处理重复主伸长。`tests/test_constitutive_adjoint.py` 独立用 NumPy SVD、SciPy 增量能量最小化和第一 Piola 应力检查 24 组旋转、重复/近重复伸长及非对称 Fp，并以两档差分核对 F、Fp、μ、λ、屈服半径的 VJP（rtol=2e-6、atol=1e-7）。`tests/test_plasticity.py` 另检查 400 组随机体积/偏形变的驻点、耗散与塑性体积，以及屈服边界和强压缩的矩阵方向导数。

初始塑性增量为零时，线搜索中的线性项用恒等式 `g·s+r|s|=(r−|g|)|s|+½|g||s||g/|g|+s/|s||²` 计算，避免近屈服时弹性功与塑性功抵消。屈服条件和驻点容差保持原值；1792 个距离屈服边界不超过四个相邻浮点间距的状态检查返回收敛、塑性体积和应力。

时间展开参考 DiffTaichi，塑性组织参考 PlasticineLab，分段反向参考 Warp 的 checkpoint 示例；对应链接见 DIFFERENTIABLE 的方法来源。TLMPM 传递、专用伴随和切割事件积分均在本仓库实现；双精度验证使用相同公式生成的内核。

## 摩擦弹性历史与滑移耗散

切向接触由 Kelvin–Voigt 元件（弹簧和阻尼器并联）串联库仑滑动元件组成。记段初切平面上的弹性位移为 h₀，阶段相对切向速度为 vₜ，分段时长为 τ，弹性增量为 Δh，滑移为 Δs，则 `τvₜ=Δh+Δs`，`fₜ=−kₜ(h₀+θΔh)−cₜΔh/τ`。阻尼器作用于弹性位移率；滑动时总相对速度还包含滑移率。

`friction.tangential_return` 先计算 `f_trial=−kₜ(h₀+θτvₜ)−cₜvₜ`。若 `|f_trial|≤μ_s f_n`，取 `Δs=0`；滑动时取 `fₜ=μ_d f_n f_trial/|f_trial|`、`Δs=τ(fₜ−f_trial)/(cₜ+θτkₜ)`，两者均返回 `h₁=h₀+τvₜ−Δs`。零库仑半径取 `fₜ=0`。静动摩擦系数相等时，两条分支的力与历史连续；静摩擦系数较大时保留力跳变。

原 `Solver`、M3/M4 和自然单步默认 `friction_update="implicit"`，θ=1；M5 中点积分使用 `friction_update="midpoint"`，θ=1/2。θ 只选择本构时间积分，刚度、阻尼和摩擦系数由同一配置确定。终点切平面投影与脱离清零会减少储能，完整轨迹账本分别记录历史传输/重置耗散。

固定法向下，切向输入功满足下面的离散恒等式：

`−fₜ·τvₜ = ½kₜ(|h₁|²−|h₀|²) + (cₜ/τ)|Δh|² − fₜ·Δs + (θ−½)kₜ|Δh|²`。

右侧依次为弹簧储能变化、阻尼耗散、滑动耗散和积分耗散；自然粘滑分支、`0≤μ_d≤μ_s`、`θ≥1/2` 时后三项非负。`energy.contact_storage` 和 `energy.friction_increment_energy` 用 NumPy 独立核对这些量。自动测试覆盖旋转法向、零/非零阻尼、零摩擦、边界连续性、状态与参数 VJP、解析滑动松弛、砧板振子的二阶状态/梯度收敛及检查点。

## 损伤释放的步内积分

`EventCutContact` 使用每粒子阈值 Q = G_c V₀^(2/3) (1−ζ)/ζ（J）。将本步接触核算出的刃向冲量记为 J，正向相对推进速度记为 u，当前损伤为 D；步内恒定力和 u 下，完整步损伤增量为 Ju/Q，剩余断裂功为 R=(1−D)Q。若 Ju≥R，释放时刻 τ=Δt R/(Ju)，保留刃向冲量 R/u，并移除余下的 J−R/u；该步损伤功取 R，损伤变为 1。已释放粒子继续撤除原计功带内的刃向推力。其余方向的冲量和摩擦历史使用现有公式，粒子通过原一阶 MPM 步更新。

同一个 `cut_index` 内，D 累积并保留到后续子步，暂停或退刀不清零；`Solver._set_cut` 切换切割编号时重置各刀的 D 和释放标记。历史三进制侧码及塑性形变 Fp 保留，分别记忆旧切口和塑性变形。D 表示当前这一刀的断裂功进度；`EventCutContact` 将它作为逐步状态和检查点状态传递。

τ 的导数通过冲量进入刀力和粒子状态，自动微分与有限差分评估同一正向公式。`CutHistory.damage_work` 记录各粒子的损伤计功，核对 ΣW = QΣΔD；整体系的输入功、储能、塑性及摩擦耗散另行统计。侧码及粒子对集合由调用者给定；`cut_hold=False` 使释放前后的刃口接触法向规则保持一致。

事件积分思路参考 [DiffTaichi 的 TOI 实现](https://github.com/taichi-dev/difftaichi/blob/master/examples/mass_spring.py)。[Saltation matrices](https://arxiv.org/abs/2306.06862) 给出跨事件的状态灵敏度更新；当前实现对上述一阶离散积分直接求导。事件时间、终态与梯度的步长对照使用 `scripts.diff_release_validate`。

## PIC 传递耗散的时间尺度

PIC/FLIP 的混合速度为 `v_new = PIC_new + α (v_old − PIC_old)`。固定每步 α 时，粒子速度残差的有效耗散率为 `(1−α)/Δt`，步长减半会使该速率加倍。[NairnMPM 的 PIC damping 推导](https://osupdocs.forestry.oregonstate.edu/index.php?title=Damping_Options#PIC_Damping) 将这一混合解释为对传递残差的数值阻尼。

`Config.flip_reference_dt > 0` 时，本实现固定 `λ=(1−flip_fraction)/flip_reference_dt`，每步取 `α=1−λΔt`。这是残差阻尼的显式积分，要求 `0≤λΔt≤1`；粒子位置仍使用插值网格速度推进。原求解器、弹性和塑性可微求解器共用该系数。`flip_reference_dt=0` 保持每步固定混合比例，`flip_fraction=1` 为纯 FLIP。

`MaterialTrajectory.step` 和 `FixedCutContact.step` 接受可求导的单元素 `duration` 数组（秒）。分段时长 τ 同时进入重力/应力冲量、P2G/G2P/MUSL、塑性、摩擦历史、损伤计功、网格约束及反力换算。接触系数以完整步的值乘 `(τ/config.dt)²`，包括砧板系数，保持物理刚度；PIC 参考时间取显式 `flip_reference_dt`，未设置时取完整步长。分段时长取 `0<τ≤config.dt`。

`board_normal_scale` 是每步无量纲罚因子。固定刚度的步长实验将显式砧板系数同步乘 `(新步长/基准步长)²`；其默认值 0 继续使用由 `contact_reference_dt` 换算的刀具法向系数。

`scripts.diff_release_validate --transfers fixed-mix fixed-rate flip` 从同一初态对照三种传递设置；`fixed-rate` 用该网格的基准步长作为参考，6/4 mm 分别得到 λ=3200/3800 s⁻¹。网格间固定同一 λ 的对照需要显式使用同一参考步长。

## 与论文的差异

| 项 | 论文 | 这里 |
|---|---|---|
| 粒子位置更新 | 未写明 | 用插值网格速度推进（de Vaucorbeil 2020 TLMPM 式 13），速度用 PIC/FLIP |
| 传递耗散 | PIC/FLIP 混合 | 默认每步固定比例；步长对照可用 `flip_reference_dt` 固定单位时间的残差阻尼，避免细化步长时同时改变耗散率 |
| 刀身穿透处理 | 粒子罚接触 | 外部 `hold` 模式补充固定刀面的网格法向约束；M5 中点阶段按相对运动刀面速度约束，见上节 |
| 事件间时间积分 | MUSL 单步推进、预测步末接触间隙 | M5 `GeometryEventContact` 默认使用中点材料阶段与 θ=1/2 切向本构积分，在阶段实际间隙上取法向罚力，终点运输/重置历史，以改善事件分段的状态和梯度收敛 |
| 接触运动学速度 | 接触律使用粒子速度 | M5 中点阶段使用位置推进对应的插值网格速度，原 `Solver`、M4 和一阶模式保留存储粒子速度。参考 [Kakouris 等的 MPM 接触虚功与力映射，式 17、29、36](https://arxiv.org/html/2403.13534v1)，本地推导现有多场耦合算子的功配对；原因是两种速度混用产生不随步长消失的摩擦功差 |
| 接触刚度 | 每步常数 \(s_n\) | 默认按参考步长固定物理刚度，见上节 |
| 塑性返回映射 | 共旋弹性与 von Mises 型投影，具体返回映射未给出 | 保留共旋储能，使用与其功共轭应力一致的等容增量变分返回；参考 [Miehe、Apel、Lambrecht 的增量最小化框架](https://doi.org/10.1016/S0045-7825(02)00438-3)，本地推导共旋势的驻点、Newton 求解和隐式 VJP。原因是对数应变径向返回在强压缩下会增加共旋储能，数字见 RESULTS |
| 共旋应力数值计算 | 固定共旋本构 | 保持 `2μ(f−R)+λ(J−1)J f⁻ᵀ`。参考 [Shakeri 等稳定有限应变计算](https://arxiv.org/html/2401.13196v2)及其 [Julia 作者代码，Zenodo 11113499](https://doi.org/10.5281/zenodo.11113499)：以 H=f−I 计算 `2E=H+Hᵀ+HᵀH` 和 J−1 的三维行列式展开。本地推导共旋项的等价式 `f−R=R(2E)(S+I)⁻¹`，S=Rᵀf；SVD 旋转保留一次 [Higham 正交细化](https://eprints.maths.manchester.ac.uk/694/1/high86p.pdf)。原求解器、可微材料及材料参数反向使用相同项；作者论文给出 Neo-Hookean/Mooney–Rivlin/Ogden，本仓库保留共旋材料。F 状态的存储精度相同 |
| 极旋转反向数值计算 | 正向 SVD | 按 [Gawlik–Leok，式 66](https://arxiv.org/pdf/1608.04491)的 Sylvester 方程 `SΩ+ΩS=RᵀA−AᵀR` 计算极旋转 VJP。本地实现将三维反对称矩阵转为轴向量，解 `(tr(S)I−S)ω=vee(RᵀA−AᵀR)`，返回 RΩ；R 从 f 与稳定计算的 f−R 恢复，避免近单位形变时 SVD 基变换产生的弱导数舍入。特征值仍为 σᵢ+σⱼ，重复正伸长处保持正则 |
| 摩擦历史积分 | 用当前历史计算试力，粘住时累加位移，滑动时回算历史 | 采用弹簧/阻尼器并联后串联滑动元件的接触网络，见 [Computational Analysis for Industrial Brush，图 2](https://journals.utm.my/aej/article/download/15382/6897)。本地推导 θ 时间积分和离散能量恒等式；原求解器 θ=1，M5 中点 θ=1/2。原因是按总相对速度回算阻尼历史在零输入功时可生成弹簧储能，数字见 RESULTS |
| 黏弹性 | 无 | 可选 Maxwell 分支，默认关闭，见上节 |
| 可微求解 | 正向 TLMPM 切割与工具双向耦合 | 增加粒子、材料、接触历史、切割事件和刀路的自动微分与伴随；当前可微刀路由外部给定 |
| 损伤释放积分 | 方向接触功的经验阈值；达到阈值后移除刃向冲量，材料应力保持原更新 | 原 `Solver` 和 M4 在子步边界切换；M5 `EventCutContact` 根据剩余断裂功积分释放步的刃向冲量，使释放时刻进入状态导数，见上节 |
| E5 内聚元件 | 方向接触功损伤，几何侧码独立分离网格 | 已实现独立双线性内聚元件，形式参考 [Camanho–Dávila/MOOSE](https://mooseframework.inl.gov/releases/moose/2022-06-10/source/materials/cohesive_zone_model/BiLinearMixedModeTraction.html)，大变形虚功参考 [Reinoso–Paggi](https://arxiv.org/abs/1507.05168)。本地推导平均面法向、共轭表面应力与现有多场的力映射；K、σ、G_c 为独立材料量。原因是已释放刃向投影力的闭合路径功非零。元件与 TLMPM 耦合算例已核对，自然切口入口的面积生成、插入牵引平衡和历史接入待完成；动态插入的牵引初始化参考 [Sierra/SM](https://www.sandia.gov/files/sierra/SM_Users_5_20/main/materials/cohesive.html) |
| E5 受载插入原型 | 几何事件直接更新侧码 | 初始刚性内聚局部核为 `cohesive_proximal_response`，移动面压力核为 `cohesive_contact_response`；全局平衡与隐式反向已整理为 `differentiable_cohesive.solve`，CPU 非线性方程位于 `_cohesive_solve.py`；粒子推进原型在 `runs/e5_release_audit/extrinsic_*.py`。依据 [Petrie–Hirmand–Papoulia](https://arxiv.org/html/2202.06359) 的初始刚性、卸载至原点、单边接触及 ADMM 形式；本地推导 3D 径向闭式核、TLMPM 功配对、惯性子问题与 `p g·n` 的面转动共轭力。论文使用小应变 FEM/固定界面，原型使用有限应变 TLMPM/预置参考面和一阶半隐式推进。移动面联合约束另用 CPU SciPy 主动集初值与 NumPy 驻点细化，Warp 在 GPU 上推进材料和计算局部响应；按 [OptNet](https://proceedings.mlr.press/v70/amos17a.html) 的 KKT 隐式求导思路实现 VJP。统一原型使用 `g=prox(g+λT)` 与粒子动量平衡连接 κ=0 承载和 κ>0 卸载，CPU 解半光滑方程，隐式反向直接对 Warp 实际残差构建 Jacobian 并解伴随方程，使零压力边界的分支与前向一致。 求解位移投影到 B 和切面梯度 G 的共同作用空间；其正交动量块为单位阵，反向保留该块的完整右端项，并核对原空间的伴随残差。CPU 组装局部 9×9 约束 Hessian，再映射至约化空间。该代数消元保持原方程、面积权重与容差。固定法向 ADMM 或光滑能量驻点提供初值；零开口且牵引未达强度时解线性约束。λ 是数值分裂尺度，0.5/1/2 倍对照核对同一解。局部差分对 `F=F_free+G u` 同时保存原始相减与基态/增量分别相减的结果，避免弱导数被单位基态的舍入覆盖；公式和阈值相同。连续粒子轨迹、自然切口与永久历史继续接入 |
| E5 隐式梯度核对 | 正向求解 | 对照 [DiffPD 的收敛状态伴随矩阵与反向残差](https://github.com/mit-gfx/diff_pd_public/blob/1594aa07a6983ff9a9c107e9f0d83d07fa99c8d1/cpp/core/src/fem/deformable_newton_backward.cpp)、[Nimble 的最小范数反力与完整伪逆导数](https://github.com/keenon/nimblephysics/blob/c405b056fc35068027e03e0c384e84e12870b475/dart/neural/ConstrainedGroupGradientMatrices.cpp#L210)及 [Petrie 等按面积归一化的原始/对偶压力残差，式 21–26](https://arxiv.org/html/2202.06359)。实际 Warp 残差的逐步秩检查位于 `extrinsic_rank_probe.py`；`extrinsic_canonical_reaction.py` 在同一动量平衡和非拉伸压力约束下，以参考面积加权选择最小范数反力，本地实现用于检查冗余反力对隐式梯度的影响。等价反力选择已整理到 `_cohesive_solve.canonical_reaction`；几何精度和轨迹差分结果见 RESULTS |
| E5 界面几何精度 | 双精度状态与运算 | 参考 [Ogita–Rump–Oishi 的求和与点积误差补偿](https://www.tuhh.de/ti3/paper/rump/OgRuOi05.pdf)。`cohesive_geometry.InterfaceGeometry` 在 CPU longdouble 中累加开口与面形变，`solve_geometry` 细化同一界面平衡残差；几何状态保存为双精度高位和舍入余量，快照和重算同时保存两部分。高位和余量表示同一物理变量，VJP 对总值求导，材料与伴随使用 float64。实现要求 CPU longdouble 的尾数位数大于 float64；与论文同精度 TwoSum/TwoProduct 的实现方式不同。初始零历史、零开口且牵引未达强度时，按线性闭合约束细化，响应与未激活的断裂参数无关。物理方程和差分阈值相同 |
| E5 参考面与积分原型 | 几何侧码与粒子接触面积 | 自然参考面映射参考 [Crook–Homel，式 18–22](https://www.osti.gov/servlets/purl/3003988)及作者 [GEOS `czUpdate` 分支 e068c207 的参考状态保存、位移与力映射](https://github.com/GEOS-DEV/GEOS/blob/e068c20772fe157428b5509c363146f0efd2e8ac/src/coreComponents/physicsSolvers/solidMechanics/SolidMechanicsMPM.cpp#L10404)。作者使用 CPDI 和预设内聚界面，本仓库计划在自然切割事件插入 TLMPM 参考面。积分对照参考 [Schellekens–de Borst 的线性界面节点面积积分](https://doi.org/10.1002/nme.1620360104)：`extrinsic_nodal_surface.py` 从现有迹基函数 N 计算 Aₙ=Σq A_q N_qn，开口使用节点迹，面梯度使用面积加权平均，力使用同一映射的转置。面梯度平均为本地推导；16 个 Gauss 点与 9 个节点代表两种离散，分别检查能量、旋转和差分。主线切割尚使用原释放公式 |
| 接触事件导数对照 | 步末处理接触 | [Zhong 等 TOI-Velocity](https://proceedings.mlr.press/v211/zhong23a.html) 在刚体碰撞时刻重算法向和碰前速度，同时修正碰后位置；已核对作者 [PyTorch 实现](https://github.com/DesmondZhong/diff_sim_improve_grads/blob/31a78e2e2c94bb7607057caef966825c725ee946/single_collision/single_torch.py#L18)。本仓库 E1 对自然几何事件分段并传播时长导数；E5 内聚原型的界面闭合仍用步末联合约束。接入主线时按同一事件时刻核对界面法向、材料状态、接触历史及速度更新 |
| 几何分侧积分 | 子步更新侧码 | `GeometryEventContact` 用 RK2 二次位置延拓和三个时刻的几何间隙生成入区候选；旋转查询按角跨度 ≤π/2 分段，在实际材料推进上验证括区、求根并切分，按事件时间求导；使用连续的刀后 `2Δx` 区域，原步末查询的 `max(相对推进速度×Δt,0)` 回看扩展由步内穿越查询替代 |
| 耦合区间内的刀具 | 位姿与速度保持不变 | 默认相同；`tool_substep='predict'` 用 6×6 移动性矩阵逐子步更新刀速并推进位姿 |
| 接触检测 | 未写明 | 默认用设备端条件重建邻居表；`hash` 每子步重建 Warp HashGrid，`all` 全对检测（每粒子 64 线程），按完整搜索核对接触集合 |
| 切割次数 | 不限 | int64 三进制码，上限由 \(3^{k+1}\cdot\)节点数 \(< 2^{63}\) 决定（香蕉 6 mm 网格 33 刀，3 mm 31 刀，2 mm 30 刀） |
| 香蕉几何 | 实物扫描 | DexLab 合成 120 mm 香蕉 |
| 刀具 | 未公开 | 80×40×1 mm 楔形，5 mm 刃口斜面 |
| 地面 | 刚性支撑 | 粒子：砧板顶面矩形内的罚函数接触。刀：Franka 由 MuJoCo 实体盒碰撞；其余刀（导向、位姿伺服、自由落体）共用 `tool.step` 的同一个刀顶点—砧板接触（速度约束 + 库仑摩擦），三者使用同一砧板矩形 |
| 控制器与机器人接口 | 外部刚体系统接收反力/力矩，展示 Franka 任务；控制器和刚体引擎未给出 | 自由刀与位姿伺服使用本地刚体积分；Franka 使用 MuJoCo，控制器采用 libfranka 笛卡尔阻抗示例公式或 SERL++ 公式的本地移植（MuJoCo 四元数替代 scipy；测试对照固定版本源码） |
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
