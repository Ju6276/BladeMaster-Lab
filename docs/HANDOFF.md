# 交接说明

2026-11-01。目标与验收条件见 [DIFFERENTIABLE.md](DIFFERENTIABLE.md)，工作规范见 [AGENTS.md](../AGENTS.md)，实验数字见 [RESULTS.md](RESULTS.md)。

## 机器与版本

- Mac：`/Users/judong/Desktop/Papers/Code/BladeMaster-Warp`，分支 `sync-20261020`，当前提交 `4624c6a`，主工作区干净。
- 4090：`dong@tamsgpu9`，工作树位于 `/home/dong/Projects/`；使用项目 `.venv/bin/python`。
- 跳板 tams61：`dong@100.70.97.92`。登录后执行 `ssh dong@tamsgpu9`；认证信息不写入文件。
- GitHub：`git@github.com:Ju6276/BladeMaster-Lab.git`。
- 数值进程设置 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`。长实验记录命令、配置、commit + dirty、逐步输出和退出码；运行中的源码保持冻结。

## 当前源码与验收

下列工作树在 Mac 和 4090 使用相同名称；候选基线均为 `4624c6a`。

| 工作树或相对目录 | 内容 | 验收记录 |
|---|---|---|
| `BladeMaster-Warp-cache-20261101` | 统一内聚核心、缩放停止条件、提前 Newton、GMRES、零乘子反力选择及批量求值 | `runs/weak_full_20261101/`：2137 passed、8 skipped、18 subtests passed，退出 0 |
| cache 的 `runs/numerical_weak_source/` | 核心修复去掉性能改动 | `runs/numerical_weak_full_20261101/`：2103 passed、8 skipped、18 subtests passed，退出 0 |
| `BladeMaster-Warp-task-20261101` | 当前核心加连续刀路、任务目标、能量与检查点入口 | `runs/task_full_20261101/`：2162 项全量运行中 |
| cache 的 `runs/accelerated_source/` | 任务版本加 16 点分块与输入设备驱动的 CUDA 补偿乘法 | 39 项定向检查通过；`runs/accelerated_full_complete_20261101/`：2201 项全量运行中 |
| `BladeMaster-Warp-batched-20261101` | 自然下刀与捕获平衡方程实验 | 当前续跑 `runs/natural_weak_energy_20261101/` 使用冻结 cache 核心 |

8 项跳过检查均依赖独立 CRESSim 上游构建。全量状态以对应 `full_result.json`、`failures.txt` 和进程为准。临时源码的测试驱动必须核对 `blademaster_warp.__file__`；加速候选链接 task 的 vendor，并使用 task 的完整脚本与测试资源。

## 已有实验结果

- 自然下刀前 8 步：64 粒子、200 μs 步长、8 个分侧事件；刀位三档自然差分误差 0.277232% / 0.0000598127% / 0.0000149532%，E 两档误差 2.2772e−10 / 4.9092e−10。检查点与完整反向的刀路梯度逐项一致。
- 任务入口七步承载案例：完整/检查点状态和刀路梯度逐项一致，两档自然差分误差 2.7393e−8 / 7.0570e−11；该案例分侧事件和损伤为零。
- 同一七步任务固定参数，200/100/50 μs 的净能量残差为 1.3132% / 1.1284% / 0.9612%；分阶段绝对残差和为 2.6611e−6 / 9.0017e−7 / 2.0322e−7 J。
- 同一 3312 阶平衡方程：分块加 CUDA 单次求值 7.2728→2.4901 s；完整求解 1670.76→561.52 s。主位移高低分量与独立参考解逐项一致，接触集合相同，相对残差 1.09097e−31。结果在 batched 的 `runs/cuda_equation_20261101/` 和 `runs/cuda_captured_20261101/`。

## 当前完整轨迹

- 原冻结核心：batched 的 `runs/natural_weak_energy_20261101/`，从第 8 步检查点继续 60 步下刀，当前第 9 步求解中；保留该运行至取得对应结果。
- 加速候选：cache 的 `runs/natural_complete_accelerated_20261101/`，正式入口 `task/scripts/diff_natural_task.py`；由未切割初态下刀 12 ms、退刀 12 ms、静置 6 ms，dt=200 μs。配置 `cache/runs/natural_complete_case.json`，每步保存能量、事件和检查点；前 8 步完成，8 个自然分侧事件，能量残差 0.80403%，继续第 9 步。
- 完整切透、退刀、切面重新接触、活跃断裂能量、完整任务自然差分和优化验收继续执行。

优化公共驱动在 cache 的 `runs/optimization_source`：13 项解析/预算检查通过；三步非零刀力任务的控制映射梯度检查通过，AD/自然差分误差 6.1740e−11 / 1.6367e−9，记录 `runs/optimization_adapter_20261101/`。

## 下一步与提交

1. 核对三套全量退出码及完整自然轨迹；失败定位、修复和复测。
2. 数值改动与性能改动分开提交。核心拆分补丁为 cache 的 `runs/current_unified_20261101/performance_split_scaled.patch`；数值全量通过后再提交，任务入口和新增 CUDA 性能改动独立提交。
3. 将通过的源码归并到主工作树，删除被替代入口；同一干净提交重跑完整任务验收。
4. 固定任务、五组初始化和墙钟预算，执行 AD / FD / CMA-ES 刀路优化与独立重放。

待归并的早期 review 位于 `BladeMaster-Warp-e5-review-20261101`。历史实验目录在 `runs/`，版本与结果索引在 RESULTS；淘汰代码由 Git 历史保留。显示网格修补不恢复，归档 tag 为 `archive/before-cleanup-20261020`。

## 文档分工

README / RUNNING：运行方式；METHOD：公式与方法差异；PARAMETERS：单位与参数来源；RESULTS：实验数字；DIFFERENTIABLE / TODO：当前验收与待办；NOTICE：第三方来源与许可。
