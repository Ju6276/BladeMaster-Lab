# 可微 TLMPM 实施计划

当前主线：给现有 TLMPM 增加可核对的反向传播，再用刀路优化比较梯度的收益。真机标定放在轨迹优化之后。

目标是由实际刀路决定切口位置，梯度穿过物体内部变形、接触和材料分离。当前 M1–M3 使用未切割的弹性方块，全部粒子的 x、v、F 参与求导；刀路控制量是各时间段的刀速，刀位由速度积分。M4 固定一次正向运行得到的离散切割分支，核对塑性、多场和切块接触；M5 再处理刀路变化引起的侧码、切开位置与事件时刻变化。刀路参数化与预设材料切面是两个独立设置。

## 实现与运行

4090 工作目录：`~/Projects/BladeMaster-Warp-diff-20261021`。基础验证为 `1857020`；长轨迹、接触切换和首轮推压优化为 `32f0378`。接触分段重算和 8/16 控制量实验为 `023d30d`，报告目录 `runs/diff_023d30d_{contact_horizon,optimize_q8,optimize_q16,optimize_q16_full_ad}`，均为 dirty=false。

- `differentiable.ElasticTrajectory`：纯弹性、未切割的固定参考网格；复用原 P2G 和网格速度 kernel，PIC/FLIP、MUSL 使用独立的下一步数组。
- 共转应力的正向调用 `material.corotated_stress`；反向通过极分解求导，分母为奇异值之和，覆盖单位阵和重复正伸长。
- `checkpoint_vjp`：保存分段边界，反向时重算一段，返回初始位置、速度、变形梯度和 E 的 VJP。
- `differentiable_contact.PrescribedContact`：刀与砧板的预测罚接触、粘着/滑动摩擦，逐步保存切向历史；复用 `contact.blade_query`、`contact.penalty_impulse`。
- `PrescribedContact.checkpoint_vjp`：分段边界保存粒子状态和两组切向历史，返回初始状态、历史、工具参数和 E 的梯度；支持逐步损失和末端损失。
- `PrescribedContact` 使用外部位姿、零移动性及单场弹性轨迹。
- `differentiable_material.MaterialTrajectory`：逐步保存 Fp 和投影后应力，Hencky 投影专用伴随，固定侧码的相容场交换；E、屈服应力与初始状态方向导数检查。
- `differentiable_cutting.FixedCutContact`：固定侧码、损伤释放标记和粒子对 ID，逐步保存刀、板和粒子对的摩擦历史；支持记录的法向闭合、粘滑及屈服分支。
- `differentiable_constraint.solve`：保留原网格刀面约束的 48 次迭代批次、残差和投影公式，各迭代写入独立数组；支持固定正向记录的求解批数。
- `differentiable_events`：固定法向平面的步内穿越时刻/刀刃位置，以及损伤阈值的线性穿越时刻。`FixedCutContact.fracture_energy` 支持 G_c 求导和分段重算，`CutHistory.trial_damage` 保存截断前损伤。
- `differentiable_cutting.EventCutContact`：按剩余断裂功积分释放步的刃口冲量，释放标记由逐步损伤决定；事件时刻进入粒子状态和刀力的梯度。侧码与粒子对 ID 由调用者提供，使用 `cut_hold=False`。
- `FixedCutContact.checkpoint_vjp`：保存 x、v、F、Fp、应力、刀/板/粒子对摩擦历史和损伤；支持接触对数量变化及逐步反力损失。
- `--ordered-transfers` 固定粒子到网格的累加顺序；`--precision float64 --reference-precision float64` 从同一份 Warp 函数定义生成双精度核，分别运行可微与原求解器的推进流程。默认计算精度为 float32。

M4 验收通过，数值提交 `fe6a422`（dirty=false）。两网格三材料的 12 个切割窗口全部通过正向及材料、摩擦、刀路方向导数核对；刀路每例至少两档误差 ≤2%，最差第二档误差 0.00950%。完整切透、退刀与推块为 1.05 s、67200 步、两块各 726 粒子；刀路三档差分误差 0.0232/0.00308/0.00137%，同精度原求解器最大位置差 6.21e-15 m、Fp 差 2.53e-13，32 步分段的全部状态重算差为 0。CUDA 数组内存池峰值 4711.3 MiB。报告为 `runs/m4_fe6a422_{windows,full}/report.json`，视频为 `runs/m4_fe6a422_full/differentiable_cutting.mp4`。

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_differentiable.py -q
PYTHONPATH=. .venv/bin/python -m scripts.diff_validate \
  --spacings 0.006 0.004 --steps 1 12 32 --output runs/diff_<commit>
```

`report.json` 包含 commit、配置、命令、正向最大误差、自动微分与各档有限差分结果、分段重算梯度误差。数值见 RESULTS 的“可微 TLMPM”。

## 分阶段验收

| 阶段 | 工作 | 验收 | 状态 |
|---|---|---|---|
| M1 单步弹性 | E、初始速度、变形梯度进入求导链；不覆盖中间状态 | 单位阵、等向伸长、重复伸长、剪切、旋转；解析小应变与有限差分；两个网格与原求解器正向对照 | 已实现，4090 测试 |
| M2 多步与检查点 | 完整状态保存；分段重算 VJP | 两个网格，32/128/512 步 E、初始速度和 F 导数；分段长度 8/32/128 的梯度、计时、显存 | 已完成本轮扫描 |
| M3a 接触与摩擦 | 外部刀和砧板接触；独立摩擦历史 | 两个网格，接触建立/脱离、粘滑切换的单侧差分；推压轨迹记录分支变化 | 已完成切换扫描 |
| M3b 轨迹优化 | 优化工具推压轨迹，使材料达到目标形状 | AD + Adam、FD + 同配置 Adam、CMA-ES，同预算五次初始化 | 4/8/16 控制量、两网格和时域、另一初始刀位的对照完成；长任务完整/分段 AD 在 20 s 内均未达 0.05 mm |
| M4 固定切割分支 | 固定侧码、相容场交换、接触集合；接入塑性与刀路梯度 | 多场正向一致；E、屈服、摩擦和刀路的方向导数；换材料与刀路 | 已完成：12 个窗口、网格约束、完整状态检查点及 67200 步切割/退刀/推块通过 |
| M5 切开事件 | 处理损伤阈值、接触集合和拓扑变化的梯度 | 阈值扰动扫描；事件时间导数或平滑方案对照；完整切割轨迹优化 | 进行中：事件定位、自然扫描、释放状态导数及固定耗散率的窗口步长检查完成；待处理几何分侧、接触集合变化和完整轨迹优化 |

M1–M3 小算例使用 float32，M4 的最终正向和梯度验收使用 float64。正向最大误差目标：位置 1 μm、速度 0.1 mm/s、F/Fp 为 1e-5、刀力 1e-4 N。弹性 E 梯度对至少两档有限差分误差 ≤ 1%，接触梯度 ≤ 2%；接近零的导数同时报告绝对误差。检查点 E 梯度误差 ≤ 0.1%，初始状态 VJP 逐数组比较。

## 推压任务与运行

```bash
PYTHONPATH=. .venv/bin/python -m scripts.diff_horizon --output runs/diff_<commit>_horizon
PYTHONPATH=. .venv/bin/python -m scripts.diff_contact_scan --output runs/diff_<commit>_contact
PYTHONPATH=. .venv/bin/python -m scripts.diff_contact_horizon --output runs/diff_<commit>_contact_horizon
PYTHONPATH=. .venv/bin/python -m scripts.diff_optimize --output runs/diff_<commit>_optimize
```

`diff_contact_horizon` 默认扫描 6/4 mm、128/512 步、8/16 控制量，比较完整 Tape 与 32/128 步分段。每种配置在独立进程预热后测三次，使用相同目标数组。`diff_optimize --segment-steps 32` 启用接触分段重算，0 为完整 Tape；`--controls 16 --steps 512 --spacing 0.004 --tool-offset -0.0003 0 0` 改变控制维度、时域、网格和初始刀位，偏移单位为 m。

推压使用 6 mm 网格、288 粒子、128 步（25.6 ms）。优化四段恒定 x 向刀速，归一化控制量在 [0,1]，对应 0–0.2 m/s；刀位由速度积分。目标是左侧四分之一粒子的末端三维位置，由控制量 [0.7,0.5,0.8,0.4] 的同模型正向运行生成。

目标函数为 `(位置 RMSE / 1 mm)^2 + 10 × mean(max(刀力 / 2 N - 1, 0)^2)`。成功条件是位置 RMSE ≤ 0.05 mm 且峰值刀力 ≤ 2 N。每种方法使用相同五组初始刀路和 20 s 墙钟预算；AD 与 FD 共用 Adam，学习率 0.04，FD 步长 0.001，CMA-ES 初始标准差 0.2。

预算包含前向、反向、数据读回和优化器更新。编译预热、目标生成、初始梯度扫描和原求解器重放在预算外执行。预算内完成的所有目标评估参与最优值统计，包括 FD 扰动评估；超时才完成的评估不参与。每个方法在独立进程运行，显存记录 CUDA 内存池实际使用量的峰值，结果重放使用原 `Solver`。

接触建立处有导数折点；静摩擦系数 0.3 转到动摩擦 0.2 时，切向力发生跳变。自动微分沿实际执行分支传播；报告分别记录差分两侧的接触/摩擦模式、单侧差分和中心差分。

## 切割窗口

```bash
PYTHONPATH=. .venv/bin/python -m scripts.diff_cut_validate \
  --spacings 0.006 0.004 --materials banana soft plastic \
  --prefixes 1200 2400 --steps 12 --grid-constraint \
  --ordered-transfers --precision float64 --reference-precision float64 \
  --control-epsilons .0001 .00003 .00001 --output runs/diff_<commit>_cut
```

原求解器从完整香蕉开始下刀，先运行 1200/2400 步，再重放随后 12 步；步长 62.5 μs。刀速为 `[0, 0.012, -0.12] m/s`，初始刃口高 27 mm、x=0.4 mm。三种材料的 `(E, 屈服应力)` 分别为 `(400000, 30000)`、`(20000, 2000)`、`(20000, 200)` Pa。报告记录各状态及刀力误差、屈服粒子数、粒子对数、分支变化和三档差分。

刀路控制量分别改变初始 x 偏移、下刀速度和 y 向速度；位置由速度积分。窗口目标为平均刀反力在 `[0.3,0.2,-1]` 上的投影，加末端 `20 mean(|v|²) + 0.1 mean(|Fp|²)`。每次扰动从同一窗口初态开始。默认记录接触与屈服分支；`--natural-branches` 比较重新判断局部分支的结果。

`--parameters controls --control-epsilons 0.001 0.0003 0.0001 0.00005 0.00003 --repeats 3` 细扫刀路差分，并从同一窗口初态重复计算 AD，记录损失和梯度的重复性。

快照保存在 `captures/*.npz`，包含原求解器的窗口初态、应力、摩擦历史、侧码、接触对和逐步输出。float32 正向误差与原求解器自身重复性一起核对；双精度方向导数使用同一分支公式。

## 完整切割轨迹

```bash
PYTHONPATH=. .venv/bin/python -m scripts.diff_cut_trajectory \
  --duration 1.05 --frame-stride 320 \
  --backward --segment-steps 32 --epsilons .0001 .00003 .00001 \
  --ordered-transfers --precision float64 --reference-precision float64 \
  --video --output runs/diff_<commit>_trajectory
```

0–0.24 s 下刀，0.24–0.48 s 退刀，0.48–0.75 s 移位和下降，0.75–0.85 s 横向推块，0.85–1.05 s 再退刀。三项控制量是初始 x 偏移、z 向行程比例和 x 向推移比例。`--duration .48` 运行切透与退刀；`--render-only` 从保存的可微粒子状态渲染视频。

`trajectory.npz` 保存可微求解器的 x/v/F、粒子块号和刀反力；`reference_x/reference_v/reference_F/reference_force` 保存原求解器独立推进的对应量，`pair_count` 保存采样时的接触对数。`branches.npz` 保存分支序列。原求解器提供侧码、释放标记、接触集合和约束求解批数。反向按分段重算，报告逐状态重算误差、梯度和 CUDA 内存池峰值。

用 `--comparison-video` 替换 `--video`，左右并排显示原求解器与可微求解器；两侧共用材质、刀路、步长和相机。默认播放速度为 0.1 倍，正常速度重绘命令如下。表面更新仍读取全部记录帧，播放速度只改变视频采样。

```bash
PYTHONPATH=. .venv/bin/python -m scripts.diff_cut_trajectory \
  --render-only --comparison-video --playback-speed 1 \
  --output runs/diff_<commit>_trajectory
```

输出为 `solver_comparison.mp4`（0.1 倍速）和 `solver_comparison_1x.mp4`（正常速度）。M4 的横推为 16 mm / 0.1 s，即 0.16 m/s；原 Franka 三刀入口的 `SequenceSettings.push_speed_m_s` 为 0.02 m/s，两套任务使用不同刀路。

固定法向的单粒子粘着接触中，以 `(v_t, h_t/dt)` 为状态，切向更新矩阵为 `[[1-2γc√γk, -γk], [1, 1]]`。当前 γc=0.1，稳定条件为 γk<0.04；保持参考接触刚度时对应 dt<22.36 μs。62.5/31.25/15.625 μs 的单步谱半径分别为 1.09576/1.01105/0.99578。MPM 传递和粘滑切换进入完整轨迹的梯度检查。

提交 `c697950` 起，完整轨迹入口从网格 CFL 步长开始，调用 `presets.tangent_history_dt` 逐次减半直到满足粘着历史的稳定条件；当前香蕉默认得到 15.625 μs。`--dt` 显式指定步长用于对照。参考接触刚度和材料参数保持相同。

## M5：事件定位与扰动扫描

```bash
PYTHONPATH=. .venv/bin/python -m scripts.diff_cut_events \
  --spacing .006 --duration .24 --output runs/m5_<commit>_events_6mm
PYTHONPATH=. .venv/bin/python -m pytest tests/test_differentiable_events.py -q
```

扫描复用 M4 前 0.24 s 下刀轨迹，逐次只改变 x 偏移、z 行程或 G_c。每个子步由原求解器自然更新侧码和释放标记；报告首次分侧/释放时刻、最终分块、反力冲量、刀做功及损伤计功。`*.npz` 保存逐粒子的事件步号、侧码、损伤和终态。

外部指定刀路下，`contact.assign` 的几何分侧由刀刃位置触发；`contact.resist_cut` 在累计损伤达到 1 后撤除刃前推力。两类事件分别测量。`differentiable_events` 对线性步内模型求穿越分数 `(level-before)/(after-before)`，损伤使用截断前的 `trial_damage`。平面法向在该子步内固定；输出穿越时刻及该时刻的刀刃位置。

`EventCutContact` 在子步内按剩余断裂功求刃口推力的作用时长，同时改变正向冲量和反向导数，公式见 METHOD。释放后粒子的 x/v/F/Fp、应力和摩擦历史继续参与时间展开及分段反向。

```bash
PYTHONPATH=. .venv/bin/python -m scripts.diff_release_validate \
  --spacings .006 .004 --energies 7 70 --dt-divisors 1 2 4 \
  --output runs/m5_<commit>_release
PYTHONPATH=. .venv/bin/python -m pytest tests/test_differentiable_release.py -q
```

释放窗口从原求解器首个损伤事件之前的共同状态起步，保持 2 ms 物理时长，比较三档步长。G_c=7/70 J/m² 用于激活损伤释放；默认香蕉仍为 700 J/m²。每次扰动重新判断损伤释放、刚体接触、粘滑和屈服分支；报告事件步号及刚体接触模式变化。侧码和粒子对集合在窗口内保持不变，原求解器重放逐步核对这两个条件。目标包含平均刀力和末端 v/Fp；比较完整 Tape、16 步分段反向以及五档自然有限差分，同时核对损伤功、终态及事件时刻的步长收敛。

`--transfers fixed-mix fixed-rate flip` 对照每步固定混合比例、固定单位时间耗散率和纯 FLIP。三组及其细化步长共用原求解器在基准步长下生成的初态，报告 `pic_rate_s_inv`；`fixed-rate` 的参考步长写入配置，数值公式见 METHOD。只运行一组可指定 `--transfers fixed-rate`。

提交 `abd6b02` 的三档扫描加 `--dt-divisors 4 8 --epsilons .00001 .000001`，共覆盖 16 种配置。每个刀路/G_c 方向至少两档差分误差 ≤2%；完整 Tape 与分段反向的控制量/G_c 最大差为 4.89e-15/1.74e-17，全部状态重算差为 0。4 mm、G_c=70 的粗步长在 ε=1e-6 时跨过砧板接触分支，刀路差分误差 18.86%；补充 ε=1e-7/1e-8 后误差降至 7.59e-7%/1.44e-5%，分支变化为 0。

固定每步混合比例的步长收敛验收失败：最细相邻两档的刀路方向导数相对差为 5.04–22.52%，G_c 导数为 1.95–57.69%；6/4 mm、G_c=70 在 2 ms 窗口内的释放数分别从 4/3 变为 3/5。

提交 `1377995` 的 `--transfers fixed-rate --dt-divisors 1 2 4 8 --epsilons 1e-7 1e-8` 共 16 例通过窗口检查：最细相邻两档刀路/G_c 导数相对差 ≤0.856%/0.261%，位置距离差 ≤0.616 μm，事件时间差 ≤1.307 μs，释放粒子集合一致。两档自然差分最大误差为 0.000294%，扰动时释放步号和刚体接触模式变化均为 0；分段重算全部状态差为 0。固定每步 5% PIC 对应的耗散率随步长减半而加倍，固定速率的公式与单位见 METHOD；报告与命令见 RESULTS。

## 下一步

1. M5 动力学：将窗口中通过检查的传递设置扩展到完整自然事件轨迹，核对状态、力、功和梯度的步长收敛。
2. M5 拓扑：处理几何侧码和接触集合变化，比较事件求导与连续损伤/连接过渡方案。改变正向物理的方案独立提交，记录力、损伤、切口和能量差异。
3. 在改变切口位置、切块大小的香蕉刀路上核对梯度，再做完整切割轨迹优化。

切割力网格收敛与阈值/能量核对配合 M5，按 HANDOFF 第 4 节执行。

## 参考实现与方法

- [Newton soft body](https://github.com/newton-physics/newton/blob/main/newton/examples/diffsim/example_diffsim_soft_body.py)：Warp Tape、逐步状态和软体接触优化。
- [Warp fluid checkpoint](https://github.com/NVIDIA/warp/blob/main/warp/examples/optim/example_fluid_checkpoint.py)：分段重算与 VJP 传递。
- [DiffTaichi MPM](https://github.com/taichi-dev/difftaichi/blob/master/examples/diffmpm.py)：MPM 时间展开和反向组织。
- [DiffTaichi mass-spring TOI](https://github.com/taichi-dev/difftaichi/blob/master/examples/mass_spring.py)：事件前后分别按作用时长积分。`EventCutContact` 将这一思路用于剩余断裂功决定的刃口冲量，采用步内恒力、恒推进速度的一阶积分。
- [PlasticineLab MPM](https://github.com/hzaskywalker/PlasticineLab/blob/master/plb/engine/mpm_simulator.py)：参考其 Hencky 塑性投影及自定义 SVD 反向的组织；本项目对完整投影写专用伴随，重复伸长使用有限极限。
- [Warp 可微文档](https://nvidia.github.io/warp/stable/user_guide/differentiability.html)：数组访问检查、自定义伴随和有限差分检查。
- [接触梯度与碰撞时刻修正](https://arxiv.org/abs/2305.00092)、[Saltation Matrices](https://arxiv.org/abs/2306.06862)：事件发生时刻与事件前后状态变化的导数。
- [DiSECt](https://diff-cutting-sim.github.io/)：沿切面的连接逐渐软化，作为连续损伤方案的对照。
- [APIC](https://doi.org/10.1145/2766996)：粒子—网格传递的数值耗散及局部仿射速度表示，用于分析当前 PIC/FLIP 的步长敏感性。
- [NairnMPM PIC damping](https://osupdocs.forestry.oregonstate.edu/index.php?title=Damping_Options#PIC_Damping)：PIC/FLIP 混合与传递残差阻尼的关系；据此实现固定耗散率的步长对照。

首版使用普通 kernel launch；梯度核对通过后再测 CUDA Graph 的前反向调度。
