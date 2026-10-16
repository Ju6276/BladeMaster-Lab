# 实验结果

硬件：tams61，RTX 4060 8 GB，Warp 1.18.0，MuJoCo 3.3.7。运行目录在 tams61 的 `runs/` 下（不进仓库）。

## 2026-10-16 香蕉单刀：接触刚度与步长

commit `c3ca5c6`；`franka_run --motion feedback --controller serl --material paper-banana --duration 12 --record`；`runs/dt_sweep_20261016/`。

| 步长 | 接触参数 | 下刀段平均 Fz [N] | 下刀段冲量 [N·s] | 峰值 [N] | 结果 |
|---|---|---|---|---|---|
| 25 μs | 物理刚度 | 0.160 | 0.408 | 0.755 | 两块 |
| 50 μs | 物理刚度 | 0.158 | 0.402 | 0.801 | 两块 |
| 76.9 μs | 物理刚度 | 0.158 | 0.399 | 0.829 | 两块 |
| 50 μs | 固定 \(s_n\)（论文写法） | 0.130 | 0.329 | 0.609 | 两块 |
| 25 μs | 物理刚度 + 区间内刀具预测 | 0.160 | 0.407 | 0.756 | 两块 |

固定物理刚度后，平均力在 25–77 μs 之间相差 < 2%；按上表舍入数值，冲量相差约 2.2%。固定 \(s_n\) 时 50 μs 的冲量低约 19%。峰值仍随步长变化（单个区间的尖峰）。20 mm/s 下刀时，区间内刀具预测与保持不变没有可见差别。

## 2026-10-16 性能

单子步材料耗时（`scripts/probe_substep.py`，外部刀具，不切割）：

| 粒子 | 接触检测 | 修改前 | 修改后 |
|---|---|---|---|
| 1452 | HashGrid | 101 μs | 101 μs |
| 1452 | 全对，每粒子 64 线程 | — | 73 μs |
| 11520 | HashGrid | 215 μs | 217 μs |

1452 粒子时 GPU 远未填满：逐 kernel 计时为接触对 37 μs、p2g 15 μs、刀具接触 11 μs、update_F+塑性 10.5 μs，都是单个线程内的串行延迟（SVD、原子操作、哈希探测），合并 kernel 本身几乎不省时间。

香蕉 + Franka + CUDA 表面完整运行（5.36 s 仿真，`--output` 不带 `--record`）：

| 版本 | 步长 | 实际耗时 | 实时倍率 | 材料 | 机器人 | 观测+表面 |
|---|---|---|---|---|---|---|
| 修改前 `3a6d458` | 25 μs | 29.8 s | 0.180× | 22.4 s | 3.4 s | 3.8 s |
| `1743d71` | 62.5 μs（默认） | 13.5 s | 0.397× | 7.0 s | 2.5 s | 3.8 s |
| `2e30ca7` | 76.9 μs | 13.0 s | 0.413× | 6.4 s | 2.6 s | 3.9 s |

`robot_step` 每步从约 535 μs 降到约 360 μs：SERL 控制器直接传入目标四元数，libfranka 专用的姿态误差只在该分支计算。

## 2026-10-16 论文 Table I：重力刀（Fig. 3）

commit `1743d71`；`python -m blademaster_warp.paper_freefall`；8960 粒子，2 ms，1 s 仿真。

wall/sim = 0.47（RTX 4060）；论文 0.33（RTX 5090）。1 s 内刀转动 65°，材料仍为一块。

## 2026-10-16 香蕉连续切割（论文 Fig. 1 形式）

commit `a52ec9b`；`franka_run --motion sequence --controller libfranka --stiffness 1000 --material paper-banana --sampling-seed 1 --surface --surface-embedding cuda --surface-ccd cuda --record`；`runs/fig1_sequence/`。

![连续三刀](media/fig1_sequence.png)

切点 x = +40、+20、0 mm，每刀后推开 12 mm。刀具原点到达底部目标（误差 ≤ 0.5 mm 并保持 0.1 s）后才开始推。

| 时间 [s] | 事件 | 粒子分块 |
|---|---|---|
| 3.52 | 第 1 刀到底 | 140 / 1299 |
| 9.16 | 第 2 刀到底 | 140 / 213 / 1086 |
| 14.92 | 第 3 刀到底 | 140 / 213 / 274 / 812 |
| 17.68–18.15 | 退刀后保持 | 4 块不变，刀上无材料 |

峰值材料反力 7.4 N，最大跟踪误差 8.1 mm（下刀时刀被材料托住），控制力矩无饱和。

18.15 s 仿真用时 32.1 s（0.56×，含逐帧记录）：材料 19.8 s（每子步约 68 μs）、观测与表面 9.6 s（其中 CPU 网格构建 3.4 s、记录 1.9 s）、机器人 2.3 s。

## 2026-10-16 当前提交复核

commit `872c633`；89 项测试通过（16.73 s）。三刀重跑命令：

```bash
.venv/bin/python -m blademaster_warp.franka_run --motion sequence --controller libfranka --stiffness 1000 --material paper-banana --duration 30 --surface --surface-embedding cuda --surface-ccd cuda --output /tmp/blademaster-review-20261016
```

18.164 s 仿真耗时 30.674 s（0.592×），关闭录制和显示，保留物理、观测及表面。四个机械连通块为 878 / 245 / 178 / 138 粒子，退刀后保持；峰值材料力 7.716 N，最大跟踪误差 7.392 mm，最小 det(F)=0.9275，GPU 错误和 MuJoCo 警告均为零。计时排除 JIT、显示初始化和客户端渲染。
