# 可微 TLMPM 实施计划

当前主线：给现有 TLMPM 增加可核对的反向传播，再用刀路优化比较梯度的收益。真机标定放在轨迹优化之后。

## 实现与运行

- `differentiable.ElasticTrajectory`：纯弹性、未切割的固定参考网格；复用原 P2G 和网格速度 kernel，PIC/FLIP、MUSL 使用独立的下一步数组。
- 共转应力的正向调用 `material.corotated_stress`；反向通过极分解求导，分母为奇异值之和，覆盖单位阵和重复正伸长。
- `checkpoint_vjp`：保存分段边界，反向时重算一段，返回初始位置、速度、变形梯度和 E 的 VJP。
- `differentiable_contact.PrescribedContact`：刀与砧板的预测罚接触、粘着/滑动摩擦，逐步保存切向历史；复用 `contact.blade_query`、`contact.penalty_impulse`。
- 接触刀具采用外部位姿、零移动性；材料为纯弹性单场。塑性、切割码、粒子自接触和网格刀面约束在后续阶段接入。

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
| M2 多步与检查点 | 完整状态保存；分段重算 VJP | 1/12/32 步 E 梯度；32 步分段长度 1/7/32 的 VJP 与完整 Tape 对照 | 已实现短轨迹；长时域资源测试待做 |
| M3a 接触与摩擦 | 外部刀和砧板接触；独立摩擦历史 | 两个网格、低速粘着与高速滑动正向对照；1/8 步刀位梯度核对 | 已实现基础接触；切换边界扫描待做 |
| M3b 轨迹优化 | 优化工具推压轨迹，使材料达到目标形状 | AD + Adam、FD + 同配置 Adam、CMA-ES，同预算多次初始化 | 下一步 |
| M4 固定切割分支 | 固定侧码、相容场交换、接触集合；接入塑性与刀路梯度 | 多场正向一致；E、屈服、摩擦和刀路的方向导数；换材料与刀路 | 待做 |
| M5 切开事件 | 处理损伤阈值、接触集合和拓扑变化的梯度 | 阈值扰动扫描；事件时间导数或平滑方案对照；完整切割轨迹优化 | 待做 |

当前小算例使用 float32。正向最大误差目标：位置 1 μm、速度 0.1 mm/s、F 为 1e-5、刀力 1e-4 N。弹性 E 梯度对至少两档有限差分误差 ≤ 1%，接触梯度 ≤ 2%；接近零的导数同时报告绝对误差。检查点 E 梯度误差 ≤ 0.1%，初始状态 VJP 逐数组比较。

## 下一轮 4090 实验

1. 固定物理尺寸，扫描 32/128/512 步与分段长度，记录前向、反向、重算耗时和峰值显存。核对 E、初始速度及初始 F 的梯度。
2. 扫描刀与材料间隙及切向速度，覆盖接触建立、脱离、粘滑切换；记录分支变化和单侧差分。切换点与分支内部结果分开统计。
3. 不切割的推压任务：材料参数固定，以轨迹控制点为优化变量；目标为指定粒子区域的位置，约束刀速和最大反力。先用少量控制点跑通，再增加变量数和时域。
4. 使用相同初始轨迹、参数边界、目标和物理模型。AD 与 FD 使用同一 Adam 设置；CMA-ES 单列优化器差异。至少 5 个初始化，报告目标误差随总墙钟时间、达标时间、成功率、约束违反、前向次数和峰值显存。反向与重算时间计入总预算。
5. 若接触平滑改变正向物理，独立提交，并同时对比原接触与平滑接触。原始前向求解器也重放优化结果。

切割力网格收敛与阈值/能量核对服务于 M4–M5，按 HANDOFF 第 4 节执行。M1–M3 的固定网格梯度验证先行。

## 参考实现

- [Newton soft body](https://github.com/newton-physics/newton/blob/main/newton/examples/diffsim/example_diffsim_soft_body.py)：Warp Tape、逐步状态和软体接触优化。
- [Warp fluid checkpoint](https://github.com/NVIDIA/warp/blob/main/warp/examples/optim/example_fluid_checkpoint.py)：分段重算与 VJP 传递。
- [DiffTaichi MPM](https://github.com/taichi-dev/difftaichi/blob/master/examples/diffmpm.py)：MPM 时间展开和反向组织。
- [Warp 可微文档](https://nvidia.github.io/warp/stable/user_guide/differentiability.html)：数组访问检查、自定义伴随和有限差分检查。

首版使用普通 kernel launch；梯度核对通过后再测 CUDA Graph 的前反向调度。
