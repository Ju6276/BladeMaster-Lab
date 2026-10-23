# BladeMaster-Lab

[BladeMaster](https://arxiv.org/html/2609.27342v2) 的 Warp/CUDA 实现：拓扑感知 TLMPM 切割、MuJoCo Franka 双向耦合、表面重建和 SpaceMouse 遥操作。

- 香蕉三刀得到四块；6 mm 网格下含机器人、表面和录制为 **0.59× 实时**（RTX 4060，提交 `8fad2c5`）。
- Fig. 4 中片抬升：多场 **37.7 mm**，单场最高 **4.3 mm**。
- Fig. 3 静平衡起步：锋利刀到板后仍有角部连接；钝刀停在离板 **14.7 mm**。
- 切割力尚有网格依赖：2.5 / 2 / 1.5 mm 网格的稳态力为 **0.818 / 0.625 / 0.453 N**。

## 安装与测试

Linux + NVIDIA GPU：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
PYTHONPATH=. .venv/bin/python -m pytest tests -q
```

## 在线运行

```bash
.venv/bin/python -m blademaster_warp.franka_run --live \
  --motion feedback --controller serl --material paper-banana --duration 12 \
  --surface --surface-embedding cuda --surface-ccd cuda
```

打开 <http://127.0.0.1:8765>，点击“开始”。远程机器通过 SSH 转发端口：

```bash
ssh -N -L 18765:127.0.0.1:8765 dong@100.70.97.92
```

转发后打开 <http://127.0.0.1:18765>。SpaceMouse、材料选择、夹持方式和全部运行参数见 [运行指南](docs/RUNNING.md)。香蕉默认规则采样，6 mm 材料网格、62.5 μs 子步。

## 记录与诊断

离线运行去掉 `--live`，加 `--output runs/<名称>`；再加 `--record` 保存逐帧数据，使用 `python -m blademaster_warp.render_franka runs/<名称>` 渲染。

| 入口 | 用途 |
|---|---|
| `paper_freefall`、`paper_impact`、`scenarios` | 重力刀、多刀冲击、Fig. 4/7/8 刀路；模块前缀为 `blademaster_warp` |
| `scripts.force_convergence`、`scripts.indentation`、`scripts.cut_force` | 网格与采样扫描、压入与松弛、切割力曲线 |
| `scripts.fig5_compare`、`scripts.topocut_compare` | CRESSim-MPM 与 TopoCut 式基线对照 |
| `scripts.diff_validate` | 可微弹性、检查点和刀/砧板接触的正向与梯度核对 |
| `scripts.diff_horizon`、`scripts.diff_contact_horizon` | 弹性/含摩擦轨迹的完整保存与分段重算：梯度、耗时、显存 |
| `scripts.diff_contact_scan`、`scripts.diff_optimize` | 接触切换导数、AD/FD/CMA-ES 推压刀路优化 |
| `scripts.diff_contact_convergence` | 刀/板/粒子对弹性碰撞：解析状态和梯度、自然差分、步长与时间网格相位对照 |
| `scripts.diff_cut_validate` | 香蕉切割窗口的塑性、多场及接触梯度；可选双精度原求解器正向对照 |
| `scripts.diff_cut_trajectory` | 完整下刀、退刀、推块的分段反向；`--comparison-video` 同镜头比较原/可微求解器，`--playback-speed 1` 按真实时间播放 |
| `scripts.diff_cut_events` | 自然切割分支的刀位、下刀深度和断裂能扫描；记录分侧、释放时刻、刀力和功 |
| `scripts.diff_release_validate` | 香蕉释放窗口：自然梯度、分段反向、损伤功、步长与 PIC/FLIP 耗散对照 |
| `scripts.diff_natural_validate` | 自然分侧、梯度、检查点和起终点材料能量；`--from-initial --trajectory cut-withdraw-push --duration 1.05` 从完整初态验证切割、退刀和推块 |
| `scripts.diff_natural_refine` | 同初态、固定接触刚度与 PIC 耗散率的步长对照；`--adjoint checkpoint` 分段反向，避免保存完整 Tape；`--phases 0 .25 .5` 比较时间网格起点 |
| `scripts.diff_multicut_validate` | 暂停续切、第二刀、交叉切口及三刀后推块；第一刀控制量到后续目标的自然差分与跨刀检查点核对 |
| `scripts.diff_energy_validate` | 静置、压入和自然切割的逐段能量账本、G_c 扫描及固定物理参数的步长对照 |
| `scripts.probe_substep`、`scripts.live_latency` | 材料子步分阶段计时、输入到 HTTP 状态帧延迟 |
| `scripts.stress_teleop`、`scripts.stress_multi_tool`、`scripts.check_cut_release` | 会话与多刀压力测试、切透退刀检查 |
| `scripts.surface_accuracy`、`scripts.surface_deformed`、`scripts.surface_observations` | 表面精度、自交与观测间隔检查 |
| `scripts.scene_surfaces`、`scripts.surface_closeup`、`scripts.fig3_freefall`、`scripts.fig4_compare`、`scripts.render_compare` | 论文场景表面、录制近景及对比图/视频 |

`report.json` 记录材料、机器人、表面与发布耗时；`runs/` 不进 Git。数值扫描使用干净提交，配置或版本变化时换输出目录。

`blademaster_warp.differentiable_cohesive.solve` 提供移动内聚界面的全局平衡与隐式反向；`cohesive_geometry.InterfaceGeometry` 保存几何舍入余量和检查点。运行 `PYTHONPATH=. .venv/bin/python -m pytest tests/test_differentiable_cohesive.py -q` 核对内力、旋转、面积拆分、输入导数及检查点。自然切口接入进度见可微分计划。

## 文档

- [运行指南](docs/RUNNING.md)：遥操作、论文场景、录制与分析命令。
- [实验结果](docs/RESULTS.md)、[剩余工作](TODO.md)：数字、提交与下一步。
- [方法对照](docs/METHOD.md)、[参数来源](docs/PARAMETERS.md)：公式、参数和实现差异。
- [可微分计划](docs/DIFFERENTIABLE.md)：当前主线、M1–M5 验收与轨迹优化对照。
- [交接说明](docs/HANDOFF.md)、[研究计划](docs/PAPER_PLAN.md)：机器访问与研究方向。
- [第三方来源](NOTICE.md)、[贡献规范](AGENTS.md)。
