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
- `differentiable_material.MaterialTrajectory`：逐步保存 Fp，Hencky 投影专用伴随，固定侧码的相容场交换；E、屈服应力与初始状态方向导数检查。
- `differentiable_cutting.FixedCutContact`：固定侧码、损伤释放标记和粒子对 ID，逐步保存刀、板和粒子对的摩擦历史；支持记录的法向闭合、粘滑及屈服分支。
- 切割窗口使用完整 Tape，外部刀为 hold 模式，关闭网格刀面约束。网格刀面约束与完整切割轨迹的验收继续推进。

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
| M4 固定切割分支 | 固定侧码、相容场交换、接触集合；接入塑性与刀路梯度 | 多场正向一致；E、屈服、摩擦和刀路的方向导数；换材料与刀路 | 材料、多场和粒子对接触已实现；香蕉切割窗口扫描、几何切换与完整轨迹验收进行中 |
| M5 切开事件 | 处理损伤阈值、接触集合和拓扑变化的梯度 | 阈值扰动扫描；事件时间导数或平滑方案对照；完整切割轨迹优化 | 待做 |

当前小算例使用 float32。正向最大误差目标：位置 1 μm、速度 0.1 mm/s、F 为 1e-5、刀力 1e-4 N。弹性 E 梯度对至少两档有限差分误差 ≤ 1%，接触梯度 ≤ 2%；接近零的导数同时报告绝对误差。检查点 E 梯度误差 ≤ 0.1%，初始状态 VJP 逐数组比较。

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

## 切割窗口与下一步

```bash
PYTHONPATH=. .venv/bin/python -m scripts.diff_cut_validate \
  --spacings 0.006 0.004 --materials banana soft plastic \
  --prefixes 1200 2400 --steps 12 --output runs/diff_<commit>_cut
```

原求解器从完整香蕉开始下刀，先运行 1200/2400 步，再重放随后 12 步；步长 62.5 μs。刀速为 `[0, 0.012, -0.12] m/s`，初始刃口高 27 mm、x=0.4 mm。三种材料的 `(E, 屈服应力)` 分别为 `(400000, 30000)`、`(20000, 2000)`、`(20000, 200)` Pa。报告记录各状态及刀力误差、屈服粒子数、粒子对数、分支变化和三档差分。

刀路控制量分别改变初始 x 偏移、下刀速度和 y 向速度；位置由速度积分。窗口目标为平均刀反力在 `[0.3,0.2,-1]` 上的投影，加末端 `20 mean(|v|²) + 0.1 mean(|Fp|²)`。每次扰动从同一窗口初态开始。默认记录接触与屈服分支；`--natural-branches` 比较重新判断局部分支的结果。

1. M4 几何：定位刀路差分跨越刀刃最近特征与计功带时的变化，补齐固定几何分支的方向导数；检查 float32 重放的分支重复性。
2. M4 接触：接入网格刀面约束；延长切割窗口，覆盖默认香蕉配置和切块再次碰撞的完整轨迹。
3. M4 梯度：汇总材料、摩擦与刀路的方向导数；分开统计保持分支和改变分支的扰动。
4. M5 事件：处理刀路变化引起的几何侧码、损伤阈值、接触集合与切开时刻变化，比较事件求导与连续损伤/连接过渡方案。改变正向物理的方案独立提交，记录力、损伤、切口和能量差异。
5. 换初始刀位和刀路，验证切口随动作改变，再做完整香蕉切割的轨迹优化。

切割力网格收敛与阈值/能量核对服务于 M4–M5，按 HANDOFF 第 4 节执行。M1–M3 的固定网格梯度验证先行。

## 参考实现

- [Newton soft body](https://github.com/newton-physics/newton/blob/main/newton/examples/diffsim/example_diffsim_soft_body.py)：Warp Tape、逐步状态和软体接触优化。
- [Warp fluid checkpoint](https://github.com/NVIDIA/warp/blob/main/warp/examples/optim/example_fluid_checkpoint.py)：分段重算与 VJP 传递。
- [DiffTaichi MPM](https://github.com/taichi-dev/difftaichi/blob/master/examples/diffmpm.py)：MPM 时间展开和反向组织。
- [PlasticineLab MPM](https://github.com/hzaskywalker/PlasticineLab/blob/master/plb/engine/mpm_simulator.py)：参考其 Hencky 塑性投影及自定义 SVD 反向的组织；本项目对完整投影写专用伴随，重复伸长使用有限极限。
- [Warp 可微文档](https://nvidia.github.io/warp/stable/user_guide/differentiability.html)：数组访问检查、自定义伴随和有限差分检查。

首版使用普通 kernel launch；梯度核对通过后再测 CUDA Graph 的前反向调度。
