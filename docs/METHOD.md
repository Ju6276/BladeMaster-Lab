# 方法与实现

论文：[BladeMaster, arXiv:2609.27342v2](https://arxiv.org/html/2609.27342v2)。当前版本和验收范围见 [DIFFERENTIABLE](DIFFERENTIABLE.md)，数值结果见 [RESULTS](RESULTS.md)。

## 正向模型与代码

| 内容 | 入口 |
|---|---|
| 参考构型 TLMPM、三线性权重、P2G/MUSL/G2P | `geometry.stencil`、`material.py`、`solver.py` |
| 共旋弹性与塑性返回 | `material.py`、`plasticity.corotated_return` |
| 方向接触功与几何分侧 | `contact.rigid_at`、`contact.assign` |
| 持久三进制侧码与相容场交换 | `Solver._set_cut`、`material.compatible`、`material.couple_at` |
| 刀/板/粒子对接触与摩擦历史 | `contact.py`、`friction.py` |
| 刀面网格约束 | `blade_constraint.py` |
| 外部刚体及 Franka 耦合 | `coupling.ExternalToolPort`、`franka.py` |
| 显示表面 | `surface.py`、`surface_gpu.py`、`surface_ccd_gpu.py` |

论文的每刀进度 D 由方向接触功更新，阈值为 `Q = G_c V₀^(2/3)(1−ζ)/ζ`。D 达到 1 后释放刃向阻力。几何扫掠独立更新侧码，侧码改变下一步网格耦合；D 不修改材料应力。换刀时重置本刀 D，保留旧侧码与塑性 Fp。

## 材料与反向

弹性变形 `F_e=F Fp⁻¹` 的主伸长为 s、行列式为 J，参考体积能量密度为 `Ψ=μΣ(s_i−1)²+λ(J−1)²/2`。

极分解旋转的 VJP 使用主伸长之和作分母，重复正主伸长处保持有限。`MaterialTrajectory` 保存独立 x/v/F/Fp/应力状态；其余连续算子使用 Warp Tape。

塑性返回求解 `min Σ(t_i−1)²/2+r‖d‖`，约束 `Σd_i=0`，其中 `t=exp(log(s)+d)`、`r=√(2/3)σ_y/(2μ)`。驻点为 `dev(t(t−1))+r d/‖d‖=0`，Fp 按投影后弹性变形更新。反向对同一驻点方程隐式求导。检查项包括塑性体积保持、非负耗散、重复主伸长和屈服边界。

## 接触、耗散与能量

论文罚接触的 `k_n=s_n m_eff/Δt²`、`k_t=γ_k m_eff/Δt²`。固定无量纲系数而改变步长会改变接触刚度。`contact_reference_dt` 将系数定义在参考步长，细化时按步长平方缩放，保持 k 不变。

切向本构使用弹簧/阻尼器与库仑滑动串联：`τv_t=Δh+Δs`，`f_t=−k_t(h₀+θΔh)−c_t Δh/τ`。`friction.tangential_return` 同时返回力与历史；原单步 θ=1，中点阶段 θ=1/2。固定法向时满足：

`−f_t·τv_t = k_t(|h₁|²−|h₀|²)/2 + (c_t/τ)|Δh|² − f_t·Δs + (θ−1/2)k_t|Δh|²`。

四项分别为储能变化、阻尼耗散、滑动耗散和积分耗散。历史转动、脱离清零和接触 ID 丢失的能量变化另计。

PIC/FLIP 速度为 `v_new=PIC_new+α(v_old−PIC_old)`。固定每步 α 的有效耗散率随步长变化；`flip_reference_dt>0` 时取 `η=(1−flip_fraction)/flip_reference_dt`、`α=1−ηΔt`，要求 `0≤ηΔt≤1`。时间细化固定 η；空间细化也显式固定相同参考时间。

`energy.material_energy` 独立计算动能、重力势能和共旋储能。刀具外功由实际冲量与接触点速度配对，含反力矩与角速度；塑性耗散取同一终点 F 在返回映射前后的储能差。`EnergyLedger` 只累计接受的自然事件分段，检查点重算和求根试算不重复计功。

自然中点接触使用功共轭的插值速度。若 W 为插值、M 为网格质量、C 为相容场交换，接触速度为 `u=W M⁻¹ C Wᵀ m v`，接触加速度使用同一映射，核对 `Σm v·a=Σf·u`。

`global_balance_residual_j` 保留完整外功/能量/耗散差；传递与释放冲量的功差分别输出。`damage_work=ΣQΔD` 是独立的方向计功量。原释放律的整体能量失败及闭环功见 RESULTS。

无摩擦、固定损伤的刀刃圆角小例中，记相对位置为 `(x,z)`、`r=√(x²+z²)`、接触距离为 a。释放后的力为 `F_x=k(a/r−1)x`、`F_z=0`，因此 `∂F_x/∂z=−kaxz/r³` 而 `∂F_z/∂x=0`。该分支存在非零闭合路径功；这项来自方向分量删除的力学定义，损伤计功在固定 D 时为零。

## 自然事件与检查点

`FixedCutContact` 使用记录的侧码、释放标记和接触集合。`NaturalCutContact` 从当前粒子状态重新搜索接触，按稳定粒子 ID 传递历史；`GeometryEventContact` 在实际材料流上定位几何事件并推进事件前后两段。

`GeometryEventContact` 默认采用显式中点 RK2。候选由二次轨迹与切割区域平面求交生成，最终时间在实际流 `g(Φτ(state),knife(t+τ))=0` 上求根。分段时长进入粒子、塑性、接触历史、损伤和刀力求导。空间求根容差为 `32ε×坐标尺度`；一次事件后选择后继模式，侧码进入后续网格交换。

`EventCutContact` 的步内释放按剩余方向功 `R=(1−D)Q` 计算。恒定刃向冲量率与推进速度 u 下，若完整步功 Ju 达到 R，则保留冲量 `R/u`，释放时刻 `τ=Δt R/(Ju)`。该时长及参数参与实际状态推进的导数。

每次自然差分重新生成侧码、接触、释放与屈服分支。横穿事件处检查时刻导数；切触、事件顺序变化等边界记录单侧导数及事件序列。固定分支、事件分段和软代理梯度按各自定义比较。

检查点同时保存连续状态、摩擦/损伤历史、侧码、接触 ID、刀次和事件边界；反向重算逐项核对。`checkpoint_device="cpu"` 使用 `warp.clone` 搬移数组，材料与 VJP 公式保持相同。

## 内聚实现与统一路线

统一路线保留几何分侧以构造双侧自由度，切面由刀刃实际扫掠在线生成。初始刚性内聚关系在强度内维持连续，达到强度后软化并保存永久最大开口；刀、板及切后重新接触使用独立几何接触。新切面从零损伤出生，旧切面历史保留。当前主树与候选的接入状态见 DIFFERENTIABLE。

本地主树包含双线性单元、初始刚性单元和隐式平衡。双线性单元使用 K（Pa/m）、σ（Pa）、G_c（J/m²）；初始刚性单元使用 σ、G_c，完整断裂开口 `δ_f=2G_c/σ`，永久历史 `κ=max(κ_old,|g|)`。

`differentiable_cohesive.solve` / `solve_force` 计算界面平衡及其隐式反向；后者以积分总力处理面积。闭合冗余反力按参考面积选择最小范数，反向包含固定秩伪逆的零空间项。解析闭合反向用于正面积、闭合且反力严格位于强度内部的分支；其余分支使用相应残差/KKT 反向。迁移回归的修复保存在本地 review 工作树，包含闭合解析入口恢复和 `(3N,0)` 零维映射构造。

`reference_surface`、`reference_cut`、`reference_grid` 构造扫掠切面、参考积分与双侧映射。`CohesiveHistory` 保存切面出生朝向和永久损伤；`cohesive_stages.advance_affine` 使用独立平移/仿射速度及 ARS 两阶段推进。补偿数组保存同一变量的主值与余量，CPU 求根和线性代数使用两分量运算。

4090 的接触、碰撞、内聚分离和事件分段候选另见 HANDOFF。当前自然轨迹入口仍使用 `GeometryEventContact`，新界面的完整自然切割接入待完成。

## 与论文及参考实现的差异

| 项目 | 原工作 | 本地实现、原因与来源 |
|---|---|---|
| 自动微分 | BladeMaster 提供正向 TLMPM 与刚体耦合 | Warp 时间展开、专用材料 VJP 与检查点；参考 [DiffTaichi](https://github.com/taichi-dev/difftaichi/blob/master/examples/diffmpm.py)、[Warp checkpoint](https://github.com/NVIDIA/warp/blob/main/warp/examples/optim/example_fluid_checkpoint.py)；TLMPM 与事件链在本仓库实现 |
| 塑性 | BladeMaster 给出共旋弹性与 von Mises 型塑性 | 共旋储能一致的等容变分返回与隐式 VJP；替代应力/能量不一致的返回，解析与独立最小化核对在 `test_constitutive_adjoint.py`、`test_plasticity.py` |
| 时间积分 | BladeMaster 显式 MUSL | 自然事件使用中点 RK2；内聚候选使用显式材料/隐式界面 ARS 阶段，分别处理事件时间与刚性约束；[RK2](https://petsc.org/release/manualpages/TS/TSRK2B/)、[IMEX RK](https://doi.org/10.1016/S0168-9274(97)00056-1) |
| 事件导数 | 原正向在子步更新标签/阻力 | 在实际离散流上定位事件并求分段导数；参考 [DiffTaichi TOI](https://github.com/taichi-dev/difftaichi/blob/master/examples/mass_spring.py)、[Saltation matrices](https://arxiv.org/abs/2306.06862)。几何根保留位置容差并取进入侧，避免外侧舍入残差产生重复极短子步；终点选择参考 [CVODE rootfinding](https://sundials.readthedocs.io/en/v7.8.0/cvode/Mathematics_link.html#rootfinding)，CVODE 使用时间区间容差 |
| 接触与历史 | 原文方向冲量和切向历史 | 固定物理刚度的步长对照、力与历史联合返回、功共轭速度；依据本构离散功恒等式，本地实现于 `friction.py` 与 `energy.py` |
| PIC 时间尺度 | 固定每步 PIC/FLIP 混合 | 固定传递残差耗散率用于时间细化；参考 [NairnMPM PIC damping](https://osupdocs.forestry.oregonstate.edu/index.php?title=Damping_Options#PIC_Damping) |
| 断裂模型 | 方向接触功控制阻力，几何侧码独立分离；论文将阻力律定义为经验关系 | 采用初始刚性内聚关系：新增物理强度 σ，G_c 为参考切面单位面积的断裂功，永久最大开口控制卸载/再加载；数值近端柔度用于求解，材料初始柔度为零。[Petrie–Hirmand–Papoulia](https://arxiv.org/html/2202.06359) 使用准静态、小应变 FEM 与单元界面，本地扩展至动态 TLMPM、移动面和在线切面；[Nairn 切割模型](https://www.cof.orst.edu/cof/wse/faculty/Nairn/papers/MPMCutting.pdf) 的内聚面沿给定路径布置，借鉴其牵引功与断裂能的独立核对 |
| 移动界面功 | 固定或小变形界面的内聚关系 | 本地由平均面法向推导开口力与面转动共轭力，参考 [Reinoso–Paggi](https://arxiv.org/abs/1507.05168) 的大变形虚功 |
| 隐式伴随 | 收敛方程的导数 | 本地求同一残差/KKT 的转置方程；参考 [OptNet](https://proceedings.mlr.press/v70/amos17a.html)、[DiffPD 反向实现](https://github.com/mit-gfx/diff_pd_public/blob/1594aa07a6983ff9a9c107e9f0d83d07fa99c8d1/cpp/core/src/fem/deformable_newton_backward.cpp)、[Nimble 约束梯度](https://github.com/keenon/nimblephysics/blob/c405b056fc35068027e03e0c384e84e12870b475/dart/neural/ConstrainedGroupGradientMatrices.cpp#L210) 的约束与伪逆导数 |
| 数值精度 | 原 GPU 正向使用普通浮点状态 | 梯度对照采用 float64；内聚候选含 CPU 两分量算术、残差细化与独立高精度参考，依据 [Ogita–Rump–Oishi](https://www.tuhh.de/ti3/paper/rump/OgRuOi05.pdf)。补偿乘积采用 [TwoProductFMA](https://www.tuhh.de/ti3/paper/rump/Ru12.pdf)，通过 LLVM FMA 获取一次舍入的乘积余量；其余运算保持原累加顺序与 `fastmath=False`。线性求解先用 [SciPy SuperLU](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html) 分解主值，再用完整两分量矩阵/右端计算残差并修正解；方法依据 [Carson–Higham](https://eprints.maths.manchester.ac.uk/2629/) 和[作者实现](https://github.com/eccarson/ir3)，本地采用 float64 分解与两分量残差。局部实现持续精化到相对后向误差不再下降，保留满足原残差界的最佳解，避免首次达到误差界就丢失右端低位；主值奇异或细化不满足原残差界时，使用完整两分量分解 |
| 非线性求根与冗余约束 | 内聚 ADMM 初值及约束平衡 | 本地残差求根、坐标缩放和最小范数反力；依据 [Lewis–Overton 弱 Wolfe 搜索](https://cs.nyu.edu/overton/papers/pdffiles/nsoquasi.pdf)、[LAPACK DGEEQUB](https://www.netlib.org/lapack/explore-html/df/d0a/group__geequb_gaef3a3015d58122ab3d4d018422b93b90.html)、[KINSOL](https://sundials.readthedocs.io/en/v7.7.0/kinsol/Mathematics_link.html)、[Golub–Pereyra 固定秩伪逆导数](https://epubs.siam.org/doi/10.1137/0710036)；候选最终检查原方程和相同活跃约束 |
| 内聚与重新接触 | BladeMaster 按侧码选择粒子对接触 | 远端候选分开内聚牵引与普通法向接触，避免重复承载；参考 [GEOS 接触分支](https://github.com/GEOS-DEV/GEOS/blob/e068c20772fe157428b5509c363146f0efd2e8ac/src/coreComponents/physicsSolvers/solidMechanics/SolidMechanicsMPM.cpp#L13812)、[MOOSE](https://mooseframework.inl.gov/source/userobjects/BilinearMixedModeCohesiveZoneModel.html)、[Collins-Craft–Acary 作者代码](https://zenodo.org/records/16911870)；历史重排与事件分段为本地实现 |
| 刀具查询 | 本地箱形/楔形解析几何 | 远端联合求解保留有限刀具几何链式导数；参考 [Newton 碰撞几何](https://github.com/newton-physics/newton/blob/f121272e7af8795d94994b3b7478f990ffd02c27/newton/_src/geometry/collision_primitive.py) |
| 连续损伤与软梯度对照 | [DiSECt](https://diff-cutting-sim.github.io/) 沿预先给定切面软化；[DiffMJX/CFD](https://arxiv.org/html/2506.14186v2) 使用接触代理梯度 | 作为方法比较来源；正式刀路任务仍每次生成自然切口，导数定义分别记录 |
| 网格刀面约束 | BladeMaster 使用粒子接触冲量 | `hold` 刀位下增加相容网格速度约束，防止传递后穿刀；将反力/矩返回刚体接口 |
| 切割标签与采样 | 论文给出切割带及几何分侧 | `contact.assign` 增加触板底层标签；扰动采样将初始粒子反射至砧板上方；用于处理底层残连及初始穿透，网格/刀路核对见 TODO |
| 场景与控制 | 论文扫描物体与机器人任务 | 香蕉沿用 DexLab 合成几何；规定刀路、自由刀与本地伺服，Franka 使用 MuJoCo 和 libfranka/SERL 形式控制器；尺寸和参数见 PARAMETERS |
| 黏弹性 | BladeMaster 有可选牛顿黏性 | Maxwell 为本地可选扩展、默认关闭；正向内部状态 `Fv`，参数为 `maxwell_tau` 与 `maxwell_fraction` |
