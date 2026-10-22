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
- `differentiable_material.MaterialTrajectory`：逐步保存 Fp 和投影后应力，共旋储能的增量变分返回与隐式伴随，固定侧码的相容场交换；E、屈服应力与初始状态方向导数检查。
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
| M5 切开事件 | 处理损伤阈值、接触集合和拓扑变化的梯度 | 阈值扰动扫描；事件时间导数或平滑方案对照；完整切割轨迹优化 | E1–E3 原版本通过；共用公式更新后的完整复验待补齐，E4 当前公式复验与 E5 能量修复进行中，E6–E7 待完成 |

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

固定法向的单粒子粘着接触中，以 `(v_t, h_t/dt)` 为状态，当前隐式历史更新矩阵为 `[[1−γk−d, −γk], [1, 1]]`，其中 `d=2γc√γk`。严格稳定条件为 `d>0`、`γk+2d<4`。`presets.tangent_history_dt` 从网格 CFL 步长开始，逐次减半直到满足此条件；`--dt` 可在完整轨迹入口显式指定步长。

此前 M4/E3/E4 报告的 6 mm 基准步长为 **15.625 μs**，对应旧摩擦积分。当前固定法向条件在默认香蕉的 **62.5 μs** 已满足；完整切割的梯度与步长对照继续单独验收。与旧版本做数值对照时显式固定步长及 PIC 参考时间，避免同时改变单位时间耗散率。

## M5：事件定位与扰动扫描

完整自然轨迹从未切割的初态建立材料与接触历史，所有扰动重新搜索接触并定位切割事件。`--trajectory cut-withdraw-push` 复用上述六段刀路，在速度变化时刻切分时间步；控制量三项分别为 x 偏移、下刀行程比例和横推行程比例。`--state-parameters` 额外检查初始粒子位置和材料参数到末端的导数。

```bash
PYTHONPATH=. .venv/bin/python -m scripts.diff_natural_validate \
  --output runs/natural_<commit>_trajectory --geometry event --from-initial \
  --trajectory cut-withdraw-push --duration .24 --spacings .006 \
  --adjoint checkpoint --segment-steps 16 32 128 --epsilons 1e-6 1e-7 \
  --state-parameters initial_z_mm young_scale fracture_scale
PYTHONPATH=. .venv/bin/python -m scripts.diff_natural_refine \
  --output runs/natural_<commit>_refine --from-initial \
  --trajectory cut-withdraw-push --duration .24 --spacings .006 .004 \
  --dt-divisors 1 2 4 --adjoint checkpoint --segment-steps 32 --epsilons 1e-7
```

`--duration 1.05` 延长到横推后的退刀；`--cutting-energy 70` 或 `7` 检查损伤释放活跃的材料组合。报告包含初态梯度范数、自然差分、释放粒子数、刀力峰值、损伤功及检查点的前向/重算/反向耗时和显存。

`scripts.diff_contact_convergence` 将各材料场的粒子放在同一法向位置、赋予统一法向速度，使形变率为零，得到刀、板和两组粒子相撞的单自由度解析对照。粒子对参考刚度按同一切向位置的接触重数计算。扫描固定刚度下的步长及时间网格相位，分别记录位置/速度导数对解析解的误差和对自然差分的误差。

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

`diff_natural_validate` 和 `diff_natural_refine` 的 `--adjoint checkpoint --segment-steps 32 128` 使用分段反向：独立前向核对末态和损失，不保留完整 Tape；两种分段长度核对刀路、材料和初态导数。首个分段结果作为梯度基准，其比较误差为 `null`；后续分段标明实际的梯度参照来源。默认 `--adjoint full` 同时比较完整 Tape。`--parameters knife_x_mm knife_depth_scale knife_lateral_scale` 选择三项刀路控制量的差分扫描。

## 剩余工作与计划实验（2026-10-22）

自然正向入口 `NaturalCutContact` 已接入逐步侧码生成、预测接触对搜索及摩擦历史传递；调用者提供初态和刀路。`CutInstruction` 控制切割开关和刀次，完整 Tape 与分段重算共用同一流程。事件变化窗口的核对命令：

```bash
PYTHONPATH=. .venv/bin/python -m scripts.diff_natural_validate \
  --output runs/m5_natural --spacings .006 .004 --materials banana soft plastic \
  --steps 64 --segment-steps 16 32 --release boundary \
  --epsilons .05 .005 .0005 .00005
```

`--shapes block cylinder` 更换初始几何；`--release integrated` 接入损伤释放步内积分。`--geometry event` 按实际材料推进求解分侧时刻，切换多场后积分余下时长；默认 `boundary` 保留原步末分侧。报告包含正向误差、检查点误差、左右差分、接触模式各通道、几何事件时刻与顺序变化。

分段时长测试覆盖塑性、接触历史、损伤功、活跃网格约束及固定刚度/耗散率。几何事件的 84 项测试覆盖离散自由落体的解析根、6/4 mm、20/400 kPa、方块/圆柱、直刀/斜刀/转动刀；从完整材料自然生成两侧不相容场，核对事件时刻、全部粒子状态、E 与刀位的 VJP 和分段重算。抛物线粒子与旋转半圈/整圈的刀补充两端区外、步内进入再离开的覆盖。E1 时间步对照与 E2 跨接触切换的状态导数检查均已通过，数字见 RESULTS。

`--friction-update midpoint` 对切向弹性历史采用 θ=1/2 本构积分；`implicit` 采用 θ=1。刀、砧板和粒子对共用 Kelvin–Voigt–滑动元件更新，并核对储能与阻尼/滑动耗散恒等式，公式与来源见 METHOD。自然窗口分别记录梯度、接触集合和粘滑切换。

`4fe7e4a` 的 8/8 自然窗口，两项刀路梯度各有至少两档差分 ≤2%，最差第二档为 0.004304%；全部 112 组扰动仍有 47 组偏差 >2%、6 组符号相反。时间步对照入口从同一窗口初态出发，固定接触参考时间及单位时间 PIC 耗散率，保存各档末端状态、材料/刀路梯度、事件时刻、刀冲量和损伤功：

```bash
PYTHONPATH=. .venv/bin/python -m scripts.diff_natural_refine \
  --output runs/m5_natural_refine --spacings .006 --materials banana \
  --duration .001 --dt-divisors 1 2 4 8 --segment-steps 16 32 \
  --epsilons .0000005 .00000005
```

`--flip-reference-dt` 指定共同 PIC 参考时间；默认采用每个场景的基础步长。报告逐项比较相邻时间步，记录完整 Tape 与检查点梯度差，以及 `QΔD` 和累计损伤功之差。

时间步扫描的 `--epsilons` 不带数值时，只运行状态、力、功、AD 梯度及检查点对照，报告的 `finite_difference` 为空。完整轨迹的自然差分使用 `diff_natural_validate` 独立运行。

两个入口均可使用 `--checkpoint-device cpu` 将检查点存到主机，反向时逐段恢复到 GPU；默认值为 `cuda`。`--segment-steps` 控制每段 Tape 的长度。CPU 存储减少驻留显存，增加数据搬运；物理配置和梯度公式相同，报告逐段核对连续状态与离散事件。

提交 `acead71` 的 8 个自然分侧窗口正向侧码/接触集合与原求解器一致，16/32 步检查点全部状态重算差为 0。80 组差分中 65 组改变记录事件，49 组导数偏差 >2%、8 组符号相反；跨事件验收失败。详细配置、命令和数字见 RESULTS。

M1/M2 基础验收、M3 计划实验、M4 固定切割分支验收及 M5 E1–E3 在 RESULTS 所列提交完成；M3 长推压的 AD 优化在原预算内失败。M5 从未切割初态展开，每次扰动重新生成侧码、损伤释放和接触对。E3 的 `fb379d4`、`39741bf`、`a55b185` 覆盖 6/4 mm、50 ms 释放、0.24 s 下刀及 1.05 s 完整轨迹：50 组自然差分全部通过，最大误差 0.9185%；17 组检查点状态重算差为 0。6 mm 完整轨迹最细两档刀路梯度差 ≤0.0516%；4 mm 完整轨迹的刀路/G_c 两档差分最大误差 0.6647%。

以下 E1–E7 是已有 M5 与数值核对待办的实验拆分。E5 修改共用塑性与摩擦公式后，当前版本的 E1–E3 完整复验待补齐；`109fe3f` 已完成 2 ms 自然事件窗口的 36 组差分，最大误差 0.034311%，检查点状态重算差为 0。复验顺序为 M3 接触/M4 塑性与固定分支对照、E1–E2 事件与步长、E3 活跃释放和 6/4 mm 完整轨迹；保留原扰动档位与通过标准。E4 当前公式的进度见下，E5 完成自然切割接入后重复受影响组合，再进行 E6 网格与场景泛化、E7 切割优化。

E4 长轨迹计算期间并行准备 E5 的分项计功与解析小例、E6 的网格/材料/形状配置，以及 E7 的目标和控制接口。E5 静置与压入、E6 原正向网格对照可独立运行；涉及多刀的对照使用通过 E4 的版本。E7 先测单次梯度耗时并开展小任务，完整优化验收使用完成对应能量及收敛核对的模型。单张 4090 的长批次按实测吞吐安排，代码准备与 GPU 队列分开推进。

E5 的 `energy.material_energy` 按实际粒子状态独立计算动能、重力势能和共旋弹性储能。自然轨迹报告保存 `material_energy_initial`、`material_energy_final` 和包含平移/转动及网格约束的刀具输入功 `knife_work_j`。解析检查覆盖刚体转动、均匀膨胀、带塑性状态的储能—应力虚功关系，以及三档步长的自由落体能量。`energy.contact_storage` 和 `friction_increment_energy` 已提供接触储能与切向分项能量核算；`energy_ledger.EnergyLedger` 已接入自然事件的接受分段，9 项测试覆盖解析能量、观察前后状态/梯度一致性和检查点去重。损伤计功单独记录，按离散更新核对其与接触功的关系。M5 中点接触已使用与材料推进共轭的插值速度，未切开/部分切开/交叉侧码的虚功与转置导数通过独立核对；受载长轨迹继续验收。

```bash
PYTHONPATH=. .venv/bin/python -m scripts.diff_energy_validate \
  --output runs/e5_energy --shape block --material plastic --duration .05 \
  --scenes rest press cut --dt-divisors 1 2 4 --cutting-energies 7 70 700 1400
```

报告保留原始能量残差、各项功差和相邻步长的状态差；若轨迹未接触刀具或切割轨迹未产生分侧，命令报错。

E5 的 50 ms、G_c=7/70/700/1400 与三档步长扫描共 30 例已完成，其中 24 例未释放、6 例活跃释放。塑性方块压入与切割的最细步长能量残差分别为 **0.000375%/0.004001%**；香蕉 G_c=7/70 释放组合残差为 **23.60%/7.31%**，细化步长后仍存在。零摩擦、固定已释放状态的单接触闭合路径净功为 **7.786e-5 J**，反向路径改变净功符号：移除刃向力分量的规则本身不满足被动能量收支。下一项分开断裂阻力与刀面压缩接触，核对储能、不可逆耗散和梯度后接入自然切割，再复测扫描。参考 [DiSECt 的切割弹簧与独立接触力实现](https://github.com/NVlabs/DiSECt/blob/ff90165070335bb8033000373d2d20c6aac13edd/dflex/sim.py)及[界面损伤与压缩接触分开的 CZM](https://doc.comsol.com/6.4/doc/com.comsol.help.sme/sme_ug_modeling.05.147.html)，保持刀路自然生成切口。

`cohesive.cohesive_response` 已实现带不可逆历史、独立压缩响应和旋转表面应力的双线性界面元件。64 粒子的双侧 TLMPM 算例中，完全断裂耗散等于 `G_c A₀`；三相位细化的最细步长相对能量残差 ≤**2.327e-7**。旋转/剪切算例的轨迹旋转相对差 ≤**5.22e-15**，六类状态/材料输入的两档差分最大相对误差 **8.144e-7**。下一项接入自然切口的参考面积、双侧映射、插入牵引平衡与检查点历史，再复测完整刀路的能量和梯度。

受载插入原型已通过零开口承载、四档断裂功和 256 个状态导数检查。64 粒子、4 界面采用完整 TLMPM 耦合矩阵，256 次约束迭代的平衡残差 **4.101e-16**；历史导数与独立隐式方程核对一致。当前继续接入自然切口生成与移动面约束，完整切割仍待这一替换及能量复验，方法与数字见 METHOD/RESULTS。

初始刚性局部核已实现为 `cohesive.cohesive_proximal_response`，14 项测试通过。全局 ADMM 原型处理了 16/32 积分点、秩 9 的冗余耦合；固定法向的实际粒子推进通过 9 例能量细化及 24/160 步轨迹导数。移动面的共轭压力已实现为 `cohesive_contact_response`，4 项独立虚功/旋转/导数检查通过；旋转受压算例的三档相对能量残差为 **−0.006207/−0.003104/−0.001552**。

移动面与接触压力的联合约束原型已实现主动集求解及隐式反向，10 类局部输入各有两档自然差分通过。160 步原型采用稳定增量求和与约束位移直接更新：8 类输入各至少两档相对误差 ≤**2e-5**，最差第二档 **1.507e-5**；旋转归一化状态差 **5.641e-11**，通过 **1e-10** 标准。初速度两档较大扰动误差仍为 **0.4045/0.6498**，四档原始数字保存在报告。位移更新已整理为 `MaterialTrajectory.correct_displacement`，13 项与网格冲量推进的状态/导数对照通过；干净 `a7df3f4` 的整段复跑已通过，数字与原型相同。`ca283ec` 修复已损伤界面在零开口处的卸载导数，新增 7 项解析/单侧差分检查通过，全量 1083 项通过。统一近端固定点原型已连接零开口承载与已有损伤：16 组正向/分裂尺度对照通过，4 类状态 × 10 类输入的局部隐式导数各有至少两档相对误差 ≤2e-5，最差第二档 1.352e-7。统一原型的 160 步初速度复验失败：四档误差 0.40448/0.64982/4.592e-5/2.937e-6，达到 2e-5 的档位为 1/4，要求至少 2 档；正在核对约束求解精度。随后接自然参考面、永久历史和接触去重，再复测完整能量扫描。


统一界面的闭合承载、G_c=7/70 张开及已有损伤旋转受压，共 **12** 例三档步长能量检查通过。四组最细步长残差幅值分别为 **0.004835%/0.10884%/0.10046%/0.15525%**，步长减半后约减半；闭合保持零开口、零损伤，G_c=7 完全断裂耗散为 **0.001008 J=G_c A₀**。共旋应力采用相同公式的稳定浮点计算，256 组小形变解析检查的两条调用路径通过，全量 **1085 passed、8 skipped、18 subtests passed**。160 步初速度轨迹差分仍失败：统一求解四档误差 **0.40449/0.64982/4.6294e-5/2.2108e-5**，要求至少两档 ≤2e-5；光滑约束对照的两档小扰动误差为 **2.9951e-5/1.4668e-5**。下一项逐步核对冗余约束的秩、解精度与伴随；自然切口接入和完整 30 例扫描随后运行。

E4 执行计划：

1. 方块/圆柱的短轨迹：核对同刀暂停、换编号后的损伤与旧侧码、塑性保留；只扰动第一刀，目标从第二刀开始累计。
2. 用完整 Tape 与 CPU 检查点核对跨刀梯度，检查点分别落在换刀边界和边界之外；每个控制量做两档自然差分。
3. 6 mm 完整物体：暂停续切、第二刀、交叉切口、三刀后推块。4 mm 对照交叉事件窗口；记录每刀损伤、释放、分块和检查点误差。
4. E5 在静置、压入、切割上核对整体能量，扫描 G_c；E6 固定物理配置做空间细化与场景对照。对应失败修复后重复梯度与物理核对。
5. E7 在共同耗时预算下比较 AD/FD/CMA-ES，独立重放优化刀路。各阶段全量测试通过后提交，实验使用干净提交。

E4 表中长轨迹数字对应 `8d35d5c` / `f60a5c4`；E5 塑性和摩擦公式更新后，受影响的轨迹与 E1–E3 梯度/步长组合重新验收。

E3 当前公式复验与 E4 并行运行在干净 `ca90da9`。6 mm、50 ms 活跃释放已通过：12 组自然差分最大误差 0.61668%，完整 Tape 与 16/32/128 步检查点状态重算差为 0；4 mm、50 ms 活跃释放的三档步长共 12 组自然差分通过，最大误差 0.00043623%，检查点状态重算差为 0；6 mm 下刀正在运行。队列依次执行 6 mm 活跃释放的完整 Tape/16/32/128 步检查点、4 mm 释放步长细化、6/4 mm 下刀与完整轨迹，以及 6 mm 下刀/完整轨迹步长细化。6/4 mm 的基准步长及 PIC 参考时间分别固定为 **15.625/13.157895 μs**，接触参考时间保持 **25 μs**；每项自然差分保存原始结果并按 **2%** 检查。命令为 `PYTHONPATH=. .venv/bin/python runs/e3_ca90da9/e3_queue.py`，工作树、状态文件见 HANDOFF。

`43a939a` 的 6 mm 交叉两刀复验失败：横移两档差分误差 **8.03%/13.94%**，深度两档为 **7.53%/0.01368%**；检查点重算状态差 **0**。逐段对照确认网格刀面约束全程未触发；横移小扰动在 **0.696765625 s** 跨过接触起始点，旧摩擦端点回算引入 **1.372 μm** 的历史跳变。当前公式已删除该回算，零载荷极限的 **6** 项解析检查通过，并在 `ca90da9` 上按 **15.625 μs** 步长和相同 PIC 参考时间重验完整轨迹，命令见下。当前 6 mm 完整交叉、暂停续切及 4 mm 交叉窗口各四项自然差分通过，最大误差分别为 **0.03716%/0.005113%/0.009204%**；检查点状态重算差及暂停损伤增量均为 **0**。平行两刀四项自然差分也已通过，最大误差 **0.07879%**，32/127 步检查点状态差 **0**；三刀后推块正在运行。

E4 完整轨迹使用同一自然事件求解器，控制量为第一刀横向偏移和下切深度（mm）；第二刀及之后的刀路不含这两个控制量。暂停续切沿用同一编号和控制量。轨迹各段交界进入时间网格，损伤计功按每刀累计，检查点覆盖编号切换。

多刀 NPZ 使用 `event_times_by_cut[cut_index, particle]` 保存各刀事件时刻，未发生事件的位置为 NaN。差分输入、导数和检查点状态必须为有限数；多刀入口要求每个控制量至少两档不同扰动，误差超过 2% 时保存报告并以非零退出码结束。

```bash
PYTHONPATH=. .venv/bin/python -m scripts.diff_multicut_validate \
  --output runs/e4_multicut --spacings .006 \
  --cases resume two-cuts crossing three-cuts-push \
  --dt .000015625 --flip-reference-dt .000015625 \
  --segment-steps 32 127 --checkpoint-device cpu --epsilons 1e-6 1e-7
```

| 项目 | 当前状态 | 实验设置 | 验收方式 |
|---|---|---|---|
| E1 几何分侧与事件后状态 | 原版本通过，当前公式完整复验待补齐：中点分段积分、摩擦历史终点投影及步内双穿越查询；84 项几何测试通过，27 档时间步对照的最细两档刀路梯度差 ≤0.1279% | 6/4 mm × banana/soft/plastic；时间网格相位 0、1/4、1/2；方块/圆柱、直刀/斜刀/转动刀；抛物线粒子和半圈/整圈刀的步内进入再离开 | 核对事件时刻、侧码、x/v/F/Fp、力与目标梯度。每条扰动刀路自然更新分侧；可导区间至少两档差分误差 ≤2%，切换点记录左右差分和跳变量 |
| E2 接触切换与历史 | 原版本通过，当前公式完整复验待补齐：18 项接触、粘滑与刀缘检查；72 例解析碰撞各有两档差分 ≤2%，最差第二档 0.005707%；最细步长位置梯度误差 ≤0.4461% | 6/4 mm 上逐项测刀/板接触建立与脱离、粘滑切换、两块碰撞后分开、刀面/刀缘切换；加入活跃网格刀面约束 | 检查候选对覆盖、物理接触集合、动量、反力及历史继承/重置；核对事件前后状态导数。候选对用离散搜索生成，物理接触和其历史进入梯度检查 |
| E3 从初态展开的完整轨迹 | 原版本通过，当前 ca90da9 完整复验运行中：50 组自然差分最大误差 0.9185%，17 组检查点状态重算差为 0；6 mm 完整轨迹最细两档刀路梯度差 ≤0.0516%，4 mm 活跃释放短轨迹刀路/G_c 梯度差 ≤0.468%/1.426% | 从 t=0 开始跑 50 ms 活跃释放、0.24 s 下刀及 1.05 s 切透—退刀—推块；6/4 mm，G_c=700 及 70/7 的释放活跃组合；固定 λ 与接触刚度比较 dt、dt/2、dt/4 | 短例完整 Tape 对照 16/32/128 步分段；6 mm 完整轨迹比较 16/32/128 步分段，4 mm 完整轨迹使用 CPU/32 步检查点，逐段核对连续状态和离散事件。记录状态/力/功/事件随步长变化，检查粒子初态、材料和刀路到末端目标的梯度链 |
| E4 多刀与永久历史 | 短轨迹通过；6 mm 完整暂停续切/平行两刀/交叉两刀共 12 组自然差分最大误差 0.01160%/0.1855%/0.07377%，32/127 步检查点重算差为 0。4 mm 交叉窗口 4 组差分最大误差 0.02062%、检查点重算差 0；三刀后推块 4 组差分最大误差 0.14894%、检查点重算差 0。当前 ca90da9 的完整交叉/暂停续切/4 mm 交叉窗口通过，最大误差 0.03716%/0.005113%/0.009204%，检查点重算差 0；平行两刀也已通过，最大误差 0.07879%、检查点状态差 0；三刀后推块继续复验 | 暂停—退刀—同编号续切；新编号第二刀；两刀相交；三刀后推块。先 6 mm 完整任务，再用 4 mm 核对交叉事件窗口 | 同刀 D 延续、新刀 D 按现有规则重置、旧侧码及 Fp 保留；检查点跨编号重算一致。扰动第一刀，核对第二刀之后的状态和损失梯度 |
| E5 整体能量与阈值 | 已核对损伤计功与 QΔD；材料能量、储能—应力虚功及自由落体共 28 项基础检查通过。已接入包含平移/转动及网格约束的刀具输入功；已改用共旋增量变分塑性返回，400 组形变的耗散/屈服/体积及两档自然差分通过。切向接触已改用储能/耗散一致的更新；已接入逐段接触储能、耗散及几何分侧能量账本，中点接触改用与材料推进共轭的插值速度，受载长轨迹继续验收 | 静置/无切割压入/切割三个层次；G_c=7/70/700/1400；分别统计刀与重力做功、动能、弹性及接触储能、塑性/摩擦/接触阻尼/PIC 耗散、损伤计功和分侧前后能量变化 | 先按离散更新推导各项功的归属，避免将损伤计功重复算作耗散；记录能量残差及其步长趋势、新切面与计功关系。默认香蕉 G_c=700 的几何分侧与损伤释放分别统计 |
| E6 网格与场景泛化 | 原正向切割力尚未网格收敛；M5 局部释放主要覆盖香蕉 | 按 HANDOFF 第 4 节跑 2.5/2/1.5 mm × 5 种采样种子 × 默认/固定 0.5 mm 刀接触距离，共 30 例正向对照；趋势明确后再加 1 mm。可微事件先做 6/4 mm × banana/soft/plastic，再选 Fig. 8 材料与薄切轨迹 | 网格比较固定同一 λ、材料与参考接触刚度；报告分项刀力、功、切口/块质量、事件和梯度。逐项考察接触距离与切割带宽的网格依赖 |
| E7 完整切割优化 | 已有推压 AD/FD/CMA-ES 实验；完整切割轨迹优化待实现 | 任务 A：达到切透终点同时优化下切/横切力；任务 B：改变切割位置达到目标切片。先 4/8 控制量，再 16；三种方法共用五组初始刀路和目标 | 优化连续的位姿/形变/力目标，分块及切透作正向任务检查。先测预热后的单次梯度耗时，再固定共同墙钟预算；记录成功率、最终误差、力/功、评估次数、耗时和显存。最优刀路使用同一正向模型，重新生成自然事件并独立无梯度重放 |

E1 使用中点材料阶段；当前切向历史采用 θ=1/2 本构积分，公式及来源见 METHOD。独立接触振子的状态与梯度误差在步长减半后下降 >3.8 倍；三材料与两个网格的 1 ms 自然事件窗口固定同一初态、刚度及耗散率，比较时间步及起点相位。最细相邻两档位置差 ≤1.041 nm、事件时刻差 ≤0.2691 ns，事件及最终侧码集合一致；命令和梯度数字见 RESULTS。

E1 使用几何事件时刻求根及前后分段的实际 TLMPM 积分；E2 已核对接触切换及历史重置的状态导数，罚接触使用阶段间隙及切向本构积分。连续损伤或连接过渡作为方法对照。数值方法独立提交公式与实现，按 E5/E6 对照力、功和网格依赖。

验收分开记录：① AD 与同一正向模型的有限差分；② 步长/网格与能量的一致性；③ 优化任务的实际成功率。可导区间沿用至少两档差分 ≤2% 的目标；真实切换点记录单侧导数和事件序列。M3 的长推压在 20 s 内完整/分段 AD 均为 0/5，作为 E7 的失败对照保留。

自然事件的 **1.05 s、4 mm、4892 粒子**轨迹，RTX 4090 上独立正向为 **665.00 s**；CPU 存储、32 步分段的检查点正向/重算/反向为 **690.48/740.65/289.56 s**，峰值 CUDA 数组内存 **5.431 GiB**。E7 在预热后按同一目标与配置测量单次梯度耗时，再设置共同优化预算。调度或存储优化在分阶段计时后独立提交。

## 现有正向求解器的后续可微覆盖

当前可微接触使用外部给定位姿的刀；材料支持共旋弹性和等容 von Mises 塑性。以下接口与模型在当前实现之外，按应用需求接入：

- 牛顿黏性：将应变率产生的黏性应力纳入求导，核对耗散和黏度梯度；当前构造器要求 `dynamic_viscosity=0`。
- Maxwell 黏弹性作为可选扩展：增加松弛内部状态及其反向，核对保持形变时的应力衰减和耗散；当前构造器要求 `maxwell_tau=0`。
- 自由刚体刀、位姿伺服及 Franka 双向动力学：增加刚体/机器人状态和耦合求导，再核对动量与控制梯度；目前刀路是外部控制输入。
- 密度、泊松比、锋利度、几何等其他参数的梯度接口与验收；当前已有 E、屈服应力、摩擦、G_c、初始状态和选定刀路参数的检查。

功能来源与论文的对应关系见 METHOD 的差异表。原论文描述正向动力学，以上模型与接口的求导由本项目实现。

真机参数标定、CRESSim/MLS 方法比较、显示表面修补、网页与 SpaceMouse 性能继续按 TODO 对应章节安排。显示网格与渲染不参与当前物理损失求导。

## 参考实现与方法

- [Newton soft body](https://github.com/newton-physics/newton/blob/main/newton/examples/diffsim/example_diffsim_soft_body.py)：Warp Tape、逐步状态和软体接触优化。
- [Warp fluid checkpoint](https://github.com/NVIDIA/warp/blob/main/warp/examples/optim/example_fluid_checkpoint.py)：分段重算与 VJP 传递。
- [DiffTaichi MPM](https://github.com/taichi-dev/difftaichi/blob/master/examples/diffmpm.py)：MPM 时间展开和反向组织。
- [DiffTaichi mass-spring TOI](https://github.com/taichi-dev/difftaichi/blob/master/examples/mass_spring.py)：事件前后分别按作用时长积分。`EventCutContact` 将这一思路用于剩余断裂功决定的刃口冲量，采用步内恒力、恒推进速度的一阶积分。
- [PlasticineLab MPM](https://github.com/hzaskywalker/PlasticineLab/blob/master/plb/engine/mpm_simulator.py)：参考其塑性投影与自定义 SVD 反向的组织；本项目的共旋变分返回和隐式 VJP 见 METHOD。
- [Warp 可微文档](https://nvidia.github.io/warp/stable/user_guide/differentiability.html)：数组访问检查、自定义伴随和有限差分检查。
- [接触梯度与碰撞时刻修正](https://arxiv.org/abs/2305.00092)、[Saltation Matrices](https://arxiv.org/abs/2306.06862)：事件发生时刻与事件前后状态变化的导数。
- [DiSECt](https://diff-cutting-sim.github.io/)：沿切面的连接逐渐软化，作为连续损伤方案的对照。
- [APIC](https://doi.org/10.1145/2766996)：粒子—网格传递的数值耗散及局部仿射速度表示，用于分析当前 PIC/FLIP 的步长敏感性。
- [NairnMPM PIC damping](https://osupdocs.forestry.oregonstate.edu/index.php?title=Damping_Options#PIC_Damping)：PIC/FLIP 混合与传递残差阻尼的关系；据此实现固定耗散率的步长对照。
- [LAMMPS tangential history](https://docs.lammps.org/pair_granular.html)：先增量、后库仑投影的摩擦历史积分；M5 用于核对粘滑边界的历史连续性。

首版使用普通 kernel launch；梯度核对通过后再测 CUDA Graph 的前反向调度。
