# BladeMaster-Lab

[BladeMaster](https://arxiv.org/html/2609.27342v2) 的 Warp/CUDA 实现，包含 TLMPM 切割、持久侧标签、切面重新接触、MuJoCo Franka 耦合、表面显示和遥操作。

可微分主线已有材料、接触、固定分支和旧模型自然多刀的梯度结果；新内聚模型前 8 步的刀位梯度通过自然差分。当前推进统一版本、完整切透/退刀/再接触及刀路优化，状态与验收条件见 [可微分计划](docs/DIFFERENTIABLE.md)。

## 安装与测试

Linux + NVIDIA GPU：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
PYTHONPATH=. .venv/bin/python -m pytest tests -q
```

## 运行

```bash
.venv/bin/python -m blademaster_warp.franka_run --live --motion feedback --controller serl \
  --material paper-banana --duration 12 --surface --surface-embedding cuda --surface-ccd cuda
```

打开 <http://127.0.0.1:8765>。SpaceMouse、远程转发、论文场景和录制命令见 [运行指南](docs/RUNNING.md)。离线运行去掉 `--live`，加 `--output runs/<名称>`；`runs/` 不进 Git。

## 可复用工具

| 模块组 | 用途与入口 |
|---|---|
| 正向实验 | `blademaster_warp.paper_freefall`、`blademaster_warp.paper_impact`、`blademaster_warp.scenarios` |
| 力与物理核对 | `scripts.force_convergence`、`scripts.indentation`、`scripts.cut_force` |
| 材料/接触梯度 | `scripts.diff_validate`、`scripts.diff_horizon`、`scripts.diff_contact_scan`、`scripts.diff_contact_horizon`、`scripts.diff_contact_convergence` |
| 推压优化 | `scripts.diff_optimize`：AD/FD/CMA-ES 对照 |
| 固定分支切割 | `scripts.diff_cut_validate`、`scripts.diff_cut_trajectory`；后者支持同刀路并排视频 |
| 可微分自然任务 | `scripts.diff_natural_task --case examples/natural_contact.json --output runs/natural_contact`：统一刀路、接触历史、内聚状态、梯度、检查点与能量记录 |
| 对照/诊断/显示 | `scripts.fig5_compare`、`scripts.topocut_compare`、`scripts.probe_substep` 等，完整索引见运行指南 |

## 文档

- [当前计划](docs/DIFFERENTIABLE.md)、[待办](TODO.md)：目标、阶段和验收范围。
- [实验结果](docs/RESULTS.md)：版本、命令、关键数字与失败项。
- [方法](docs/METHOD.md)、[参数](docs/PARAMETERS.md)：实现、公式与来源。
- [交接](docs/HANDOFF.md)：机器访问、候选源码和记录位置。
- [第三方来源](NOTICE.md)、[工作规范](AGENTS.md)。
