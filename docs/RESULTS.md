# 实验结果

本页保留有结论的实验与当前失败项。各结果绑定所列版本；后续计划见 [DIFFERENTIABLE](DIFFERENTIABLE.md)，公式与模型差异见 [METHOD](METHOD.md)。GPU 数值实验使用 RTX 4090，另列硬件的条目除外。运行记录在对应工作树的 `runs/`，完整过程可从 Git 历史查询。

命令在对应版本的仓库根目录执行，统一前缀为 `PYTHONPATH=. .venv/bin/python`。以下使用 `python` 简写。旧 `runs/*.py` 是该次实验记录中的入口。

## 2026-11-01：释放力定义与内聚本构核对

干净 `4d5a534`，RTX 4090，工作树 `BladeMaster-Warp-goal-20261101`。命令 `python runs/method_audit_20261101/release_field.py`：三位置 × 释放前后 × 三档差分，AD 与解析最大绝对差 **1.66e-10 N/m**；相邻 `1e-8/1e-9 m` 差分的最大相对差 **5.33e-10**。释放后两项交叉导数分别为 **7838.232679 N/m / 0**（另一个位置为 **5393.126095 N/m / 0**），释放前相等至 **9.10e-13 N/m**。

同脚本闭合路径每边 128/256/512 点：释放前净功 **1.034e-10/2.584e-11/6.460e-12 J**；释放后 **7.785586e-5/7.785585e-5/7.785584e-5 J**。固定损伤、无摩擦、接触分支保持相同，能量失败来自方向力定义。

同提交 `python -m pytest tests/test_cohesive_proximal.py::test_loaded_birth_and_exact_fracture_work tests/test_cohesive_proximal.py::test_kkt_rotation_and_natural_gradients -q`：**5 passed / 0.43 s**；覆盖四档 G_c 的牵引功、永久历史、转动和自然差分。记录在 `runs/method_audit_20261101/`。

`e169578` 源码，`python runs/method_audit_20261101/model_comparison.py`：恢复完整几何法向力后，同一点的力由 **(4.137092, 0) N** 变为 **(4.137092, −2.955066) N**；闭环净功见上文未释放分支。该接触势能修复提供弹性接触，材料断裂另用内聚关系。σ=25/50 kPa × G_c=7/70/700/1400 × 两个开口方向的 **16/16** 加载/卸载/再闭合例通过；每单位面积完整牵引功等于 G_c，最大分段能量误差 **1.137e-13 J/m²**，最大牵引误差 **1.111e-11 Pa**；完全断裂后闭合牵引为零，断裂历史保持。

## 2026-11-01：内聚反向的补偿求和开销

RTX 4090，`4d5a534` 对照 `e169578`。命令 `python runs/method_audit_20261101/cohesive_cost.py` / `cohesive_cost_after.py`；64 粒子、16 个界面积分点、dt=2 μs，预热后三次中位数，计时均在 profiler 关闭时完成。

| 状态 | 正向改前 / 改后 | 反向改前 / 改后 |
|---|---|---|
| 闭合 | 17.37 / 16.60 ms | 68.99 / 43.93 ms |
| 预开口 | 182.06 / 181.59 ms | 2131.81 / 1052.94 ms |
| 加载 | 181.41 / 181.35 ms | 2130.55 / 1053.54 ms |

编译后的求和保持累加顺序与两分量运算，160 组高/低分量逐位一致。`python -m pytest tests/test_compensated_products.py tests/test_compensated_state.py tests/test_compensated_linalg.py tests/test_cohesive_momentum.py tests/test_cohesive_stages.py -q`：**80 passed / 248.87 s**。阶段分析定位到 `_implicit_pullback` 的逐行反向与 CPU 补偿运算；完整自然切割仍待接入。

相同源码全量：`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest tests -q`：**1686 passed、8 skipped、18 subtests passed / 739.17 s**，退出码 0。测试时为 `4d5a534` + 三文件改动，提交前核对与 `e169578` 三文件内容完全一致；记录 `full_tests.log`、`full_result.json`。

## 2026-11-01：候选异常退出的诊断线程复现

RTX 4090 主机 Python **3.12.15**，`e169578` 源码。`gdb --batch -ex run -ex 'thread apply all bt 16' --args .venv/bin/python runs/method_audit_20261101/watchdog_probe.py timer`：CPU 数组运算小例在定时调用栈输出线程触发 **SIGSEGV**，原生栈为 `dump_frame → dump_traceback → _Py_DumpTracebackThreads → faulthandler_thread`；相同 30000 次运算关闭定时输出正常完成。

impact 候选 `34304b0` 加原工作树改动，`pytest tests/test_cohesive_stages.py::test_default_damping_rate_and_restart -q -o faulthandler_timeout=0.5` 在 GDB 中复现相同诊断线程崩溃；改为默认 `faulthandler_timeout=0`：**1 passed / 203.75 s**，退出码 0。数值代码、断言与输入相同。日志在核对工作树 `runs/method_audit_20261101/{watchdog_timer,watchdog_control,impact_timer_gdb,impact_without_timer}.log`；运行方式见 RUNNING。[CPython 上游问题](https://github.com/python/cpython/issues/158200) 也记录了定时诊断线程访问帧元数据的崩溃。

同候选 `python /home/dong/Projects/BladeMaster-Warp-goal-20261101/runs/method_audit_20261101/candidate_forward_profile.py`：64 粒子、16 积分点、一个预开口推进的 profiler 耗时 **17.69 s**，其中残差细化 **14.92 s**、残差/Jacobian 求值 **136 次**。归并前检查已接近根的初值是否仍需完整正则化延拓；对照保持原方程与原残差判据。

## 2026-11-01：文档整理的求解器基线检查

求解器为干净 `34304b0`，RTX 4090。命令 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m pytest tests -q`：**1669 passed、8 skipped、18 subtests passed**，**841.27 s**，退出码 0。8 项跳过为 CRESSim 上游构建未配置。

日志与退出记录：4090 主仓库 `runs/docs_cleanup_20261101/pytest.log`、`result.json`。文档本地链接、命令参数和模块入口检查通过；文档整理提交 `4d5a534` 的求解器/测试源码与该基线相同。E5 待验收改动保存在独立 review 工作树，状态如下。

## 2026-11-01 复核：E5 候选与闭合回归

以下复核读取 2026-10-24 批次已有日志。候选根目录和源码位置见 [HANDOFF](HANDOFF.md)。

| 源码版本 | 日志（`runs/grid_update/`） | 结束结果 |
|---|---|---|
| `34304b0` + state 工作树改动 | `closed_kkt_full.log` | 1713 passed、26 failed、8 skipped、18 subtests passed；21355.83 s |
| `34304b0` + blade 工作树改动 | `blade_pair_full.log` | 1769 passed、26 failed、8 skipped、18 subtests passed；22674.56 s |
| `34304b0` + impact 工作树改动 | `impact_active_full.log` | 异常结束，全量结果缺失 |
| `34304b0` + adhesion 工作树改动 | `separated_spatial_full.log` | 异常结束，全量结果缺失 |
| `34304b0` + events 工作树改动 | `contact_event_derivatives.log` | 异常结束，完整事件梯度批次结果缺失 |
| events 工作树进入事件单例复跑 | `contact_event_crash_gdb.log` | 1 passed；2353.62 s；进程正常退出 |

全量命令：`python -m pytest tests -q`；异常批次的完整调用和捕获输入保留在上述候选的 `runs/`。8 项跳过对应 CRESSim 上游构建未配置。审查时上述批次均无运行进程。

闭合回归的三组节点为：

```bash
python -m pytest tests/test_differentiable_cohesive.py::test_unequal_area_partition_preserves_pressure_and_history_derivative tests/test_differentiable_cohesive.py::test_zero_opening_map_preserves_history_energy_and_adjoint tests/test_differentiable_cohesive.py::test_closed_area_limit_retains_displacement_and_pressure_derivatives -q
```

干净 `34304b0` 为 **36 passed**；迁移后的候选为 **26 failed、10 passed**。24 项零导数误差幅值为 **4.119e-20…1.230e-8**；2 项为零维映射 `reshape(-1,0)` 错误。根因是迁移流程关闭闭合解析 VJP、零余量表示阻断入口，以及零维形状构造错误。

修复候选 **36 passed**，扩大组合 **117 passed / 1278.39 s**，边界组合 **19 passed / 50.60 s**。日志为 `closed_main_regression.log`、`closed_regression_extended.log`、`closed_regression_boundaries.log`；原失败对照为 `closed_regression_baseline.log` 和 `partition_regression_current.log`。本地 review 工作树保存定向修复；统一最终版本的全量复验待完成。

## E5：新界面候选的模块证据

2026-10-24，`34304b0` 加各候选改动。命令前缀另设 `OPENBLAS_NUM_THREADS=1`；日志均相对对应候选的 `runs/grid_update/`。

| 工作树 / 检查范围 | 命令入口与记录 | 结果 |
|---|---|---|
| adhesion：内聚/法向接触分离 | `pytest tests/test_separated_adhesion.py tests/test_compensated_linalg.py tests/test_cohesive_adhesion.py`；`separated_adhesion_singleton.log` | 75 passed；24 个载荷例覆盖四档 G_c、三种历史、两种位移；16 组方向差分及 8 组面积差分 |
| adhesion：材料阶段与检查点 | `pytest tests/test_separated_adhesion_stages.py`、`runs/separated_adhesion_energy.py`；`separated_adhesion_stages.log` | 5/5，含 18 组自然差分；局部四档 G_c × 三档步长 12/12，最细能量残差幅值 ≤2.897e-9 |
| adhesion：边界与既有伴随 | `pytest tests/test_cohesive_force.py tests/test_cohesive_adjoint_refinement.py tests/test_cohesive_closed_residual.py tests/test_cohesive_mass_coordinates.py tests/test_compensated_row_action.py tests/test_joint_affine_contact.py tests/test_joint_pair_contact.py tests/test_rigid_joint_contact.py tests/test_combined_contacts.py -q`；`separated_adhesion_affected.log` | 136 passed；独立边界组合 84 passed，`boundary_formal_units.log` |
| adhesion：空间接触阶段 | `spatial_pair_mixed_fixed.log`，调用记录见候选 `runs/` | 空间筛选、储能和自然差分 8/8，接触出生检查点 1 项通过 |
| events：事件定位与分段 | `runs/spatial_event_probe.py`、`runs/spatial_event_gradients.py entry` / `exit`；`spatial_pair_first_event_energy.log` | 单事件 12/12 组差分；进入接触最细能量残差 −7.765e-11，退出三档均 −1.020e-13 |
| events：通用驱动 | `pytest tests/test_contact_event_stages.py`；具体节点见候选日志 | 进入、退出、错开与一步内再退出 4/4，373.88 s |

新界面完整自然切割入口、整体 30 例能量扫描及受影响 E1–E4 复验待完成。

## E5：原释放模型的能量失败

2026-10-22，`ec7bcb4`、dirty=false。50 ms，固定 PIC 参考时间 15.625 μs，三档步长为 15.625/7.8125/3.90625 μs。

```bash
python -m scripts.diff_energy_validate --output runs/e5_current/block_plastic --shape block --material plastic \
  --scenes rest press cut --duration .05 --dt-divisors 1 2 4 --cutting-energies 7 70 700 1400
python -m scripts.diff_energy_validate --output runs/e5_current/banana_release --shape banana \
  --material banana --scenes cut --duration .05 --dt-divisors 1 2 4 --cutting-energies 7 70 700 1400
```

| 工况 | 三档相对能量残差 | 释放粒子 |
|---|---|---|
| 塑性方块压入 | 0.001116% / 0.000758% / 0.000375% | 0 |
| 塑性方块切割，四档 G_c | 0.014090% / 0.007980% / 0.004001% | 0 |
| 香蕉 G_c=7 | 23.59953% / 23.59991% / 23.60007% | 每档 4 |
| 香蕉 G_c=70 | 7.30799% / 7.30844% / 7.30862% | 每档 4 |
| 香蕉 G_c=700/1400 | 幅值 0.000594% / 0.000182% / 0.0000579% | 0 |

30 例完成；活跃释放的能量残差不随步长缩小，检查失败。G_c=7 的释放前后刃向冲量功差约 **3.94165e-4 J**，几何分侧能量跳变约 **6.75e-16 J**。

同提交单接触、固定 D、无摩擦的闭合矩形路径，每边 128/256/512 个采样：未释放净功 **1.034e-10/2.584e-11/6.460e-12 J**，已释放净功 **7.785586e-5/7.785585e-5/7.785584e-5 J**，反向路径变号。入口 `python runs/e5_release_cycle.py`。力分量移除与储能/历史更新的一致性，以及释放规则本身的闭环功，进入方法重订。

## E3–E4：旧释放模型的完整自然轨迹

2026-10-23，干净 `ca90da9`。完整轨迹从未切割初态开始，每次扰动重新生成切口和接触。

| E3 工况 | 自然差分 | 检查点 | 性能 |
|---|---|---|---|
| 6 mm、1452 粒子、1.05 s、G_c=700、67200 步 | 刀路与屈服共 8 组通过，最大 0.0668341% | 16/32/128 步状态重算差 0 | 峰值数组显存 3.4193 GiB |
| 4 mm、4892 粒子、1.05 s、G_c=70、79800 步 | 刀路与 G_c 共 8 组通过，最大 0.0927014% | 32 步 CPU 检查点状态重算差 0 | 独立正向 701.26 s；检查点正向/重算/反向 732.55/770.44/300.66 s；显存 5.6271 GiB |

命令：`python runs/e3_ca90da9/run_case.py full_h6` / `full_h4`。工作树 `~/Projects/BladeMaster-Warp-e3-current-20261022`，报告分别为 `runs/e3_ca90da9/full_h6/report.json`、`full_h4/report.json`。

E4 五种组合共 **20** 组自然差分全部通过；检查点状态重算差与暂停损伤增量均为 **0**。

| 组合 | 最大梯度相对误差 |
|---|---|
| 暂停续切 | 0.005113% |
| 平行两刀 | 0.078791% |
| 交叉两刀 | 0.037159% |
| 三刀后推块 | 0.094683% |
| 4 mm 交叉窗口 | 0.009204% |

三刀推块为 2.18 s / 139520 步，最终四块 **380/346/302/424** 粒子，峰值数组显存 **3.4224 GiB**。命令：

```bash
python -m scripts.diff_multicut_validate --output runs/e4_ca90da9/three-cuts-push --cases three-cuts-push \
  --spacings .006 --dt .000015625 --flip-reference-dt .000015625 --segment-steps 32 127 \
  --checkpoint-device cpu --epsilons 1e-6 1e-7
```

对应的旧错误：`43a939a` 交叉两刀横移差分为 **8.02884%/13.94012%**。接触建立时旧摩擦端点回算产生 **1.372 μm** 历史跳变；移除该回算并固定 dt/PIC 参考时间后，`ca90da9` 同两项降至 **0.00094923%/0.00033584%**。报告 `runs/e4_ca90da9/crossing/report.json`；解析边界检查 `python -m pytest tests/test_friction_energy.py -q`。

## E1–E2：事件与接触时间细化

2026-10-22，`0a8fd13` 的几何事件中点积分：三材料、两网格和三个时间相位共 **27** 档，最细相邻两档刀路/材料梯度差最大 **0.1279%/0.06259%**，位置差 **1.041 nm**、事件时刻差 **0.2691 ns**；自然差分最大误差 **0.003862%**，检查点状态重算差 **0**。

```bash
python -m scripts.diff_natural_refine --output runs/e1_0a8fd13_phases --spacings .006 --materials banana \
  --dt-divisors 8 16 32 64 --phases 0 .25 .5 --adjoint checkpoint --segment-steps 32 --epsilons 5e-9 \
  --integration midpoint
```

2026-10-22，干净 `fb379d4`，刀/板/粒子对 × 6/4 mm × 三时间相位 × 四步长共 **72** 例弹性碰撞，每例至少两档自然差分 ≤2%，最差第二档 **0.005707%**。最细位置误差 **0.1249 nm**，位置梯度误差 ≤**0.4461%**。12 例在较大的 `1e-8 m` 扰动下跨接触切换，中心差分最大偏差 **63.87%**；更小局部扰动通过。板接触速度梯度误差随时间相位非单调，dt/32→dt/64 的最大误差 **0.5386→1.4098 s⁻¹**。

```bash
python -m scripts.diff_contact_convergence --output runs/e23_fb379d4/contact --spacings .006 .004 \
  --dt-divisors 1 8 32 64 --phases 0 .25 .5
```

补充局部差分调用 `scripts.diff_contact_convergence.evaluate(surface, spacing, dt, .004, phase, [3e-10])`，覆盖上述跨分支工况。

## M4：固定记录分支的切割

2026-10-21，干净 `fe6a422`。6/4 mm × banana/soft/plastic × 两前缀共 **12/12** 窗口通过正向和梯度检查，刀路每例至少两档误差 ≤2%，最差第二档 **0.00950%**。

```bash
python -m scripts.diff_cut_validate --grid-constraint --ordered-transfers --precision float64 \
  --reference-precision float64 --control-epsilons .0001 .00003 .00001 --output runs/m4_fe6a422_windows
python -m scripts.diff_cut_trajectory --duration 1.05 --frame-stride 320 --ordered-transfers \
  --precision float64 --reference-precision float64 --output runs/m4_fe6a422_full
python -m scripts.diff_cut_trajectory --reuse --backward --epsilons .0001 .00003 .00001 --output runs/m4_fe6a422_full
```

完整轨迹 1.05 s、67200 步、1452 粒子，分成 **726/726** 两块。侧码、释放与接触集合来自原求解器记录。与原求解器最大位置差 **6.21e-15 m**、Fp 差 **2.53e-13**；刀路三档差分误差 **0.023243/0.003080/0.001372%**，32 步检查点状态差 **0**，峰值数组显存 **4711.3 MiB**。

## M1–M3：材料、检查点与推压优化

2026-10-21，干净 `1857020`，6/4 mm、288/972 粒子。`python -m scripts.diff_validate --output runs/diff_clean`：E/刀位梯度最大差分误差 **0.201%/0.00367%**，分段与完整 E 梯度差 **0.000797%**；原求解器位置/速度最大差 **7.46e-9 m / 5.07e-7 m/s**。

同日干净 `32f0378`，`python -m scripts.diff_horizon --output runs/diff_32f0378_horizon`：两个网格、32/128/512 步，22 组完整/分段对照；E/初速度/初始 F 差分最大误差 **0.812%/0.0137%/0.0419%**。972 粒子、512 步时完整/32 步分段数组显存 **112.34/9.21 MiB**。

推压优化使用五组初值，每组 20 s；成功条件为末端位置 RMSE ≤0.05 mm、刀力 ≤2 N。

| 版本与任务 | 完整 AD | 分段 AD | FD | CMA-ES |
|---|---|---|---|---|
| `32f0378`，四控制量、288 粒子、128 步 | 5/5 | — | 2/5 | 4/5 |
| `023d30d`，八控制量、288 粒子、128 步 | — | 5/5 | 2/5 | 5/5 |
| `023d30d`，十六控制量、972 粒子、512 步、后移 0.3 mm | 0/5 | 0/5 | 0/5 | 1/5 |

```bash
python -m scripts.diff_optimize --output runs/diff_32f0378_optimize
python -m scripts.diff_optimize --controls 8 --segment-steps 32 --output runs/diff_023d30d_optimize_q8
python -m scripts.diff_optimize --controls 16 --steps 512 --spacing .004 --tool-offset -.0003 0 0 \
  --segment-steps 32 --output runs/diff_023d30d_optimize_q16
python -m scripts.diff_optimize --methods ad --controls 16 --steps 512 --spacing .004 --tool-offset -.0003 0 0 \
  --output runs/diff_023d30d_optimize_q16_full_ad
```

四控制量 AD 最终 RMSE 中位数 **0.03681 mm**，达标时间中位数 **8.35 s**。十六控制量完整/分段 AD 最终 RMSE 中位数 **0.10020/0.17009 mm**，检查失败；960 档初始刀路差分中 956 档改变接触/摩擦模式，五组分段 AD 对中心差分误差中位数 **1.26%–37.36%**。

`023d30d` 的 `python -m scripts.diff_contact_horizon --output runs/diff_023d30d_contact_horizon`：972 粒子、512 步，完整/32 步分段整次梯度耗时 **308.7/474.9 ms**，数组显存 **141.06/11.88 MiB**；24 组检查点对照中 32 步分段控制梯度最大相对 L2 差 **0.0890%**。

## 正向切割力：网格与阈值

2026-10-19，`51ac3ee`、dirty=true，`runs/force_seed_scan_20261019/rows.jsonl`：三网格各 20 个种子，稳态切割力 **0.818±0.020 / 0.625±0.015 / 0.453±0.009 N**，网格分别为 2.5/2/1.5 mm；2→1.5 mm 再下降约 28%，空间收敛失败。步长减半的力变化在 3% 内。

```bash
python -m scripts.force_convergence --spacings .0025 .002 .0015 --seeds 20 --blade-contact-radius 0 \
  --output runs/force_particle_radius
python -m scripts.force_convergence --spacings .0025 .002 .0015 --seeds 5 --blade-contact-radius .0005 \
  --output runs/force_fixed_radius
```

同一旧版本固定刀距 0.5 mm、各 5 种子时约 **0.580/0.626/0.651 N**，记录 `runs/radius_series_fine_20261019/rows.jsonl`。6 mm 网格横移刀位 0→1.5 mm，峰值力 **0.81→8.07 N**，后者刃口法向占 **7.90 N**。统一版本上的固定物理参数复验待完成。

2026-10-21，`79d176e`，2.5 mm、seed=1，G_c=700/1400 时平台力 **0.83729/0.83694 N**，相差 **0.041%**，均无粒子达到释放阈值，均分成两块。调用 `scripts.force_convergence.cut_trace(.0025, 1, 0.0)`，分别以 `dataclasses.replace(banana_config(), cutting_energy=...)` 设置材料。

## 正向论文场景摘要

下表保留早期复现数字；所列旧汇总中缺少实验提交的项目标为“旧汇总”。配置差异集中在 METHOD/PARAMETERS。

| 场景 | 日期 / 版本 | 结果 | 命令入口 |
|---|---|---|---|
| Fig. 1 香蕉三刀 | 旧汇总 | 四块 778/294/236/144，峰值材料反力 4.80 N | `python -m blademaster_warp.franka_run --motion sequence --controller libfranka --stiffness 1000 --material paper-banana --duration 30` |
| Fig. 3 静平衡落刀 | 2026-10-21 / `673ff77` | 锋利度 0.9 刃口离板 0.08 mm、仍一块，顶角 2 粒子残连；0.2 停在 14.7 mm | `python -m blademaster_warp.paper_freefall --duration 1 --sharpness .9`；另取 `.2` |
| Fig. 4 中片抬升 | 2026-10-20 / `32a6d4b` | 多场 37.7 mm；单场最高 4.3 mm、最终 −0.6 mm | `python -m blademaster_warp.scenarios --scenario fig4`；单场加 `--single-field` |
| Fig. 5 两刀三块 | 旧汇总 | 开口 0.706 mm、体积变化 0.924%；推前左块已有 −10.5 mm/s 滑行 | `python -m scripts.fig5_compare run --backend blademaster --output runs/fig5` |
| Fig. 6 端面射入 | 2026-10-21 / `673ff77` | 初速 0.5/0.8/1.1 m/s 对应深度 29.9/49.3/68.8 mm | `python -m blademaster_warp.paper_impact --duration 1 --launch end` |
| Fig. 7 浅/深划痕 | 旧汇总 | 6 mm 浅划与 20 mm 深划，两次均未切透 | `python -m blademaster_warp.scenarios --scenario fig7-shallow` / `fig7-deep` |
| Fig. 8 连续薄切 | 旧汇总 | 四片各 952 粒子，倾角 90°/84°/61°/42° | `python -m blademaster_warp.scenarios --scenario fig8` |

CRESSim（上游 `09aa5009`，2026-10-20 本仓库 `3c3d380`，RTX 4060 主机 CPU）四档步长 62.5/31.25/15.625/7.8125 μs：相邻最终位置 RMS **0.78/1.08/1.15 mm**，步长收敛失败；开口约 0，退刀后左块只移动约 0.05 mm。入口 `scripts.fig5_compare build-cressim` / `run --backend cressim_cpu`，记录 `~/Projects/CRESSim-MPM-runs/dt_scan_20261020`。

TopoCut 式本地 MLS-MPM 对照的旧汇总：Fig. 4 在 0.25 ms 下多场抬升 **38.4 mm**，MLS-MPM（损伤剪切倍数 0.1）为 **0.0 mm**；Fig. 8 的 MLS-MPM 在 1 ms 下发散，0.25 ms 下几何分块为 **2380/2380/2142/952**。入口 `python -m scripts.topocut_compare --scenario fig4 --dt .00025 --damaged-shear .1 --output runs/topocut`，Fig. 8 改 `--scenario fig8`。

2026-10-20，`8fad2c5`，RTX 4060，Fig. 1 含机器人、表面与录制：仿真 **21.04 s**，墙钟 **35.54 s**，速度 **0.59×**；材料 **24.8 s**、表面 **5.2 s**。入口同上 Fig. 1，增加 `--surface --surface-embedding cuda --surface-ccd cuda --record`。可微完整轨迹耗时见 E3 条目。
