# 实验结果

本页保留有结论的实验与当前失败项。各结果绑定所列版本；后续计划见 [DIFFERENTIABLE](DIFFERENTIABLE.md)，公式与模型差异见 [METHOD](METHOD.md)。GPU 数值实验使用 RTX 4090，另列硬件的条目除外。运行记录在对应工作树的 `runs/`，完整过程可从 Git 历史查询。

命令在对应版本的仓库根目录执行，统一前缀为 `PYTHONPATH=. .venv/bin/python`。以下使用 `python` 简写。旧 `runs/*.py` 是该次实验记录中的入口。

## 2026-11-01：统一核心提交

数值核心提交 `2f4bddf` 对应 `runs/numerical_weak_full_20261101/` 的 **2103 passed、8 skipped、18 subtests passed / 9732.87 s**；批量求值提交 `20fc706` 对应 `runs/weak_full_20261101/` 的 **2137 passed、8 skipped、18 subtests passed / 4786.21 s**。RTX 4090，命令均为对应源码目录的 `python -m pytest tests -q`。8 项跳过依赖独立 CRESSim 构建。313 个数值源码/测试/脚本文件核对一致，末尾空白整理后的映射检查 **24 passed / 4.33 s**。两项提交已推送至 `sync-20261020`。

## 2026-11-01：完整任务入口的前八步

cache 工作树 `4624c6a`，dirty=true，源码 `runs/accelerated_source`。`python runs/run_accelerated_task.py --case runs/natural_complete_case.json --output runs/natural_complete_accelerated_20261101`：150 步任务的前 8 步 **1033.01 s**，**8 个自然分侧事件**、945 个积分点、参考切面面积 **18.1245 mm²**；损伤历史和断裂面积为零。

与 batched 的 `runs/natural_board_cut_20261101/checkpoint_step_8.pkl` 比较 41 组数组：粒子位置及侧码逐项一致，速度最大差 **5.5511e−17 m/s**、应力 **1.1369e−13 Pa**、参考法向 **8.6814e−13**。比较 `runs/natural_complete_accelerated_20261101/prefix8_comparison.json`。前 8 步外功 **1.499249689e−4 J**，能量残差 **−1.205441545e−6 J（0.80403%）**；完整轨迹继续推进。

优化比较驱动位于 cache 的 `runs/optimization_source`。`PYTHONPATH=runs/optimization_source python -m pytest runs/optimization_source/test_natural_optimization.py -q`：**13 passed / 0.53 s**，覆盖有界解析解、AD/FD/CMA-ES 预算一致性、超时结果排除、种子与数值失败传播。AD/FD 共用 [SciPy L-BFGS-B](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html)，CMA-ES 使用 [pycma ask/tell](https://cma-es.github.io/apidocs-pycma/cma.evolution_strategy.CMAEvolutionStrategy.html)；实际刀路映射检查 **1 passed / 287.59 s**：三步承载接触任务，总目标 **0.0047023428**、刀力项 **0.0040131217**；归一化控制方向 AD **0.00151724775893**，两档自然差分相对误差 **6.1740e−11 / 1.6367e−9**，重复初值返回相同目标。记录 `runs/optimization_adapter_20261101/`。

## 2026-11-01：CUDA 补偿矩阵乘法

cache 工作树 `4624c6a`，dirty=true；`PYTHONPATH=.:runs python -m pytest runs/test_cuda_expansion_product.py -q`：**17 passed / 0.91 s**。本地 CUDA 原型按 CPU 相同顺序执行 TwoSum/TwoProductFMA，使用 [CUDA 显式舍入指令](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__DOUBLE.html)；覆盖大数抵消、空内维、批量乘积及次正规数，高低分量逐项一致。

batched 中 `python runs/benchmark_cuda_equation.py`，同一 3312 阶方程交错三次：原构造、16 点分块、CUDA 乘法、分块加 CUDA 的中位数 **7.2728 / 4.8529 / 4.8022 / 2.4901 s**。残差、Jacobian 和全部开口高低分量逐项一致，耗时含分配与传输。记录 `runs/cuda_equation_20261101/report.json`；`python runs/check_cuda_captured.py`：完整求解 **561.52 s**，相对残差 **1.09097e−31**，主位移高低分量与独立参考解逐项一致，接触集合相同；原对应求解 **1670.76 s**。结果 `runs/cuda_captured_20261101/report.json`。


设备派发候选位于 cache 的 `runs/accelerated_source`。`python runs/check_accelerated.py runs/accelerated_source/tests -q`：**39 passed / 4.92 s**，包含 CPU/GPU 前向与反向逐项一致、嵌套设备上下文恢复及分块顺序对照。完整 2201 项测试位于 `runs/accelerated_full_complete_20261101/`。

## 2026-11-01：内聚 Jacobian 分块存储

cache 工作树 `4624c6a`，dirty=true；batched 中 `python runs/profile_current_equation.py` 对同一 3312 阶捕获平衡方程预热后测量：求值 **7.643 s**，其中内聚构造 **7.401 s**。结果 `runs/current_equation_profile_20261101/report.json`。

`python runs/benchmark_chunked_equation.py` 交错三次对照，16 个积分点一组的主块组装保持加法顺序；原构造、16 点分块、64 点分块的中位数为 **7.6133 / 4.9996 / 5.4856 s**。残差、Jacobian 和全部开口的高低分量逐项一致。`PYTHONPATH=.:runs python -m pytest runs/test_chunked_cohesive_batch.py -q`：**16 passed / 0.72 s**，覆盖 0/1/17/65 点、自然/固定分支、法向约束开关及四档分块大小。候选位于 cache 的 `runs/chunked_cohesive_batch.py`，测量记录在 batched 的 `runs/chunked_equation_20261101/report.json`。

## 2026-11-01：统一任务入口与目标函数反向

cache 工作树 `4624c6a`，dirty=true，源码 `runs/task_source`。`python runs/task_source/scripts/diff_natural_task.py --case runs/task_press_case.json --output runs/natural_task_cli_20261101`：5×200 μs 无切割压入 **70.11 s**。与原同配置观测器相比，41 组状态数组中的位置完全相同，速度最大差 **1.3878e−17 m/s**、应力最大差 **1.4211e−14 Pa**；逐步外功最大差 **3.3881e−21 J**、能量残差最大差 **1.6941e−21 J**。比较记录 `runs/natural_task_cli_20261101/comparison.json`。

新位置目标的动态循环归一化反向产生 Inf；传入接触伴随的首个 RHS 已含 NaN。将动态求和移至可重放函数，除法与刀力平方保留在调用核内。`PYTHONPATH=runs/task_source python -m pytest runs/test_task_losses.py -q`：CPU/RTX 4090、1/7/19 项归约共 **12 passed / 2.18 s**，核对位置、目标、权重、刀力及阶段时长的解析梯度。非零刀力压入 `python -m pytest runs/test_natural_task_loaded.py -q -s`：**1 passed / 307.43 s**；刀力项 **0.0041094141**、总目标 **0.0048164432**；完整/检查点终态逐项一致，刀路梯度逐项一致，E 梯度最大绝对差 **3.1019e−25**。控制方向 AD **−0.009011476714981714**，扰动 1e−4/1e−5 的自然差分相对误差 **1.0247e−9 / 2.7275e−10**。结果 `runs/natural_task_loaded_20261101/report.json`。

三步新切面案例在刀具承载前结束，位置/刀力目标的刀路 AD 与差分均为零；相对误差报告除零，测试失败。下刀 1 ms、退刀/侧移各 0.2 ms 的七步任务，`python -m pytest runs/test_natural_task.py -q -s`：**1 passed / 1260.39 s**；完整/检查点终态逐项一致，刀路梯度逐项一致，E 梯度最大绝对差 **4.1359e−25**。控制方向 AD **−0.013974853011616148**，两档自然差分相对误差 **2.7393e−8 / 7.0570e−11**。记录 `runs/natural_task_loaded_cut_20261101/report.json`。

同一配置 `python runs/task_source/scripts/diff_natural_task.py --case runs/task_loaded_cut_case.json --output runs/natural_task_cut_energy_20261101`：**149.52 s**，正向目标 **0.002527383531411171** 与完整反向逐项相同。外功 **2.1830e−5 J**、能量残差 **2.8667e−7 J（1.3132%）**，峰值刀力 **0.38176 N**，参考切面面积 **14.5907 mm²**；分侧事件 **0**、损伤 **0**。固定全部物理参数，将 `--dt` 改为 **0.0001**：14 步、外功 **2.1430e−5 J**、残差 **2.4183e−7 J（1.1284%）**；积分点数 **555→1101**。50 μs：28 步，残差 **2.0319e−7 J（0.9612%）**，积分点 **2265**。三档分阶段残差绝对和为 **2.6611e−6 / 9.0017e−7 / 2.0322e−7 J**；粗步长下的下刀/退刀残差符号相反。相邻终态位置 RMS 差 **0.3611 / 0.2142 μm**，三档正向记录耗时 **148.19 / 373.35 / 1314.57 s**。汇总 `runs/natural_task_energy_refinement_20261101.json`。

## 2026-11-01：连续刀路与接受阶段能量记录

cache 工作树 `4624c6a`，dirty=true，连续刀路候选 `runs/prescribed_path.py`。`PYTHONPATH=.:runs python -m pytest runs/test_prescribed_path.py -q`：**3 passed / 0.83 s**；四段六维世界速度，步长 400/200/100 μs，最终位姿对照解析平移和独立 SciPy 旋转，控制梯度对照解析值及旋转差分。`PYTHONPATH=.:runs:tests python -m pytest runs/test_path_checkpoint.py -q`：**1 passed / 191.94 s**；下刀、反向退刀、侧移三步自然推进生成新切面，检查点终态逐项一致，完整反向与检查点控制梯度及两档自然差分通过。

接受阶段记录候选 `runs/task_source` 保存接触修正量和阶段时长，由 `natural_energy.measure_step` 读取。`PYTHONPATH=runs/task_source:. python runs/natural_measured_task.py --dt 9.999999999999999e-05 --steps 10 --knife-z .01801 --knife-speed -1 --no-cutting --output runs/accepted_energy_20261101`：**138.81 s**，最终 **41 组状态/历史数组**与原观测器记录逐项一致。外功 **2.94395447e−5 J**，能量残差 **−3.29122124e−7 J**；与原观测器的最大逐步能量残差差 **8.4703e−22 J**。比较记录 `runs/accepted_energy_20261101/comparison.json`。

## 2026-11-01：冗余界面线性精化

cache 工作树 `4624c6a` 加缩放停止条件修复，dirty=true，`PYTHONPATH=.:runs/scaled_full_20261101 python -m pytest tests -q -p run_status --durations=20`：全量中的 4 个积分力/冗余反力组合失败于线性求解。捕获的 36×36 系统、μ=1e−32，原精化后向误差 **5.9119e−25**；90 位 Decimal 参考转回两分量后为 **1.0761e−33**，记录 `runs/scaled_force_failure_20261101/linear_probe.json`。

右预处理 GMRES 候选 `runs/krylov_fresh_source` 从原 LU 初解开始校正：同一线性系统 **14 次迭代 / 0.0222 s**，原后向误差 **1.3847e−32**。`python runs/krylov_fresh_targeted_20261101/check.py <四个失败节点> runs/test_compensated_krylov.py -q --tb=short`，PYTHONPATH 为 `runs/krylov_fresh_source:.`：**7 passed / 99.73 s**，包括三组 RHS 尺度 2^−900/1/2^900。使用停滞精化解作为 GMRES 初值的候选在开口几何差分中仍失败，后向误差 2.1449e−28；采用原 LU 初解。

## 2026-11-01：零乘子冗余接触与归并复测

cache 工作树 `4624c6a`，dirty=true。缩放停止条件候选全量已执行部分为 **1794 passed、9 failed、8 skipped、18 subtests passed / 3606.30 s**，中断退出 2；其中 6 项失败于线性精化，3 项失败于固定集合 Newton 超限。合并 GMRES 与提前 Newton 后，两个冗余界面出现反力差分失配：位移、开口、损伤相同，反力选择漏掉零乘子的闭合约束。

`PYTHONPATH=runs/weak_contact_source:tests python runs/weak_contact_targeted_20261101/check.py runs/weak_contact_source/blademaster_warp/__init__.py runs/test_weak_contact_selection_gaps.py -q --tb=short`：**54 passed / 0.62 s**。解析最小压力解覆盖不同初始反力分配、相同/不同映射、2^−30/1/2^30 尺度及 0/2^−80/2^−20 正开口。原 18 项闭合解析例修复前全部失败；两个捕获差分样本修复后的平衡坐标相对差 **1.6873e−31 / 6.4483e−32**。

`python runs/weak_contact_failures_20261101/check.py <command.json 所列节点> -q --tb=short -p run_status`：原 9 项全量失败与 18 项解析例共 **27 passed / 148.47 s**。修复已归并至 cache 工作树。`python runs/weak_full_20261101/check.py tests -q -p run_status --tb=short --durations=20`：**2137 passed、8 skipped、18 subtests passed / 4786.21 s**，退出码 **0**；8 项跳过均依赖独立 CRESSim 上游构建。记录 `runs/weak_full_20261101/`。去除性能改动后，`PYTHONPATH=runs/numerical_weak_source python runs/numerical_weak_full_20261101/check.py runs/numerical_weak_source/tests -q`：**2103 passed、8 skipped、18 subtests passed / 9732.87 s**，退出码 **0**；记录 `runs/numerical_weak_full_20261101/`。

## 2026-11-01：实际第 9 步的平衡求解

batched 工作树 `ae963d3`，dirty=true，源码 `runs/scaled_newton`。内外层使用一致缩放残差，提前求解固定接触集合的 Newton 候选；`runs/scaled_newton_captured_20261101` 捕获问题为 1023 个积分点、3312 个未知量。求解 **1670.76 s**，原相对残差 **1.0910e−31**，位移与独立延拓参考解逐项一致，接触集合一致。包含零乘子修复的完整第 9 步及后续自然推进使用 cache 主候选，输出 `batched/runs/natural_weak_energy_20261101`；其中第 13/14/15 次平衡求解 **1492.36 / 1324.80 / 1418.31 s**，完整时间步继续求解。

## 2026-11-01：方程尺度与停止条件

batched 工作树 `ae963d3`，dirty=true，源码 `runs/pivot_budget_source`。标量方程 `s(x³+x−2)=0`，s=1e−40/1/1e40、初值 x=0/0.25：原活跃集合候选 **2 failed、4 passed / 0.52 s**，失败根 x=2/1.710526，正确根 x=1。运行 `pytest runs/test_scaled_active.py -q`，执行器显式将该源码置于导入路径首位；记录 `runs/scaled_active_units_baseline_20261101.log`。

缩放残差、固定集合搜索及内外层一致判据候选，`PYTHONPATH=runs/pivot_budget_source:. python runs/check_scaled_balanced.py <9 个接触/闭合测试文件> runs/test_scaled_active.py -q`：**149 passed / 45.77 s**。同一修正配合提前活跃集合求解，8 个弱约束解析例最大位移相对误差 **1.5510e−12**，记录 `runs/weak_roots_scaled_outer_20261101/result.json`。正式回归位于 `tests/test_equilibrium_scaling.py`。

## 2026-11-01：接触方程内存访问测量

batched 工作树 `ae963d3`，dirty=true，源码 `runs/pivot_budget_source`。`PYTHONPATH=runs/pivot_budget_source:. python runs/benchmark_tiled_sum.py`：保留逐列加法顺序、按 64 列分块访问后，1024×59049 数组求和三次中位数 **1.2843→0.1869 s**；实际 3312 阶接触方程求值 **10.1963→8.9565 s**。残差、Jacobian 和全部开口的高/低分量逐项相同。记录 `runs/tiled_sum_20261101/result.json`。

`NUMBA_NUM_THREADS=4 PYTHONPATH=runs/pivot_budget_source:.:tests python runs/check_memory_kernels.py -q tests/test_compensated_products.py tests/test_compensated_linalg.py tests/test_compensated_state.py tests/test_joint_closed_adjoint.py`：并行矩阵乘法和分块求和的算术、尺度及弱约束检查 **125 passed / 8.93 s**。预热后四种方程求值方式交错运行三次，中位数为串行 **10.1280 s**、并行乘法 **11.1956 s**、分块求和 **9.1748 s**、两者组合 **12.9456 s**；记录 `runs/memory_kernels_20261101/result.json`。保留分块求和候选，并行组合不合入。

## 2026-11-01：自然下刀前八步梯度

batched 工作树 `ae963d3`，dirty=true，源码 `runs/opening_jacobian_source`。`PYTHONPATH=runs/opening_jacobian_source:. python runs/profile_colored_backward.py --mode ad --steps 8 --output runs/natural_colored_backward_20261101`：64 粒子、dt=200 μs，共 **1.6 ms、8 个自然分侧事件**；目标值 **0.05450356130673139**，正向 **1224.44 s**，反向 **1820.91 s**，退出码 0。损伤历史为零，切透与活跃断裂检查继续使用完整下刀任务。

`PYTHONPATH=runs/opening_jacobian_source:. python runs/compare_natural_gradients.py --ad-directory runs/natural_colored_backward_20261101 --output runs/natural_colored_comparison_20261101.json`：刀位方向尺度 1 mm，AD **0.4475394653441809**；自然差分扰动 **5/1/0.5 μm**，相对误差 **0.277232% / 0.0000598127% / 0.0000149532%**，三档均通过原 2% 判据。最粗档正负扰动分别发生 12/4 个事件，其余两档均为 8/8；六次前向退出码均为 0。

相同源码与任务，`python runs/natural_opening_directional.py --mode fd --parameter young --offset <±0.001 或 ±0.0005> --steps 8 --output <目录>`，PYTHONPATH 同上。E=20 kPa，方向尺度 2000 Pa，实际扰动 **±2/±1 Pa**；四次前向均退出 0，各有 8 个几何事件。`python runs/compare_natural_young.py` 对照缩放 AD **−0.0002413881548054975**，中心差分相对误差 **2.2772e−10 / 4.9092e−10**，两档通过原 2% 判据。记录 `runs/natural_young_comparison_20261101.json`。

cache 工作树 `4624c6a` 加归并候选，dirty=true，`PYTHONPATH=.:runs python runs/checkpoint_board_task.py --steps 8 --segment-steps 4 --output runs/natural_checkpoint8_20261101` 与 `python runs/compare_checkpoint8.py`：每 4 步保存 CPU 检查点；重算边界、事件与终态逐项相同，刀位/刀速梯度逐项相同，E 梯度相对 L2 差 **1.0966e−15**，内聚参数梯度均为零。目标值差 **6.94e−18**，总耗时 **3232.39 s**，其中重算 **1232.53 s**、反向 **1923.46 s**；退出码 0。记录 `runs/natural_checkpoint8_20261101/{report,comparison}.json`，实现为 `runs/natural_checkpoint.py`。同一实现的无切割与新切面两步对照 **2 passed / 68.33 s**，覆盖全部连续初态、时长、E 和内聚参数梯度。

## 2026-11-01：统一候选全量与界面反向构造

cache 工作树 `4624c6a` 加统一数值修复、接触集合更新、数组缓存修复及反向构造加速，dirty=true。`PYTHONPATH=.:runs/current_unified_20261101 python -m pytest tests -q -p run_status --durations=20`：**2074 passed、8 skipped、18 subtests passed / 4716.42 s**，退出码 0；Mac 与 4090 的源码和测试逐文件相同。记录 `runs/current_unified_20261101/{full.log,full_result.json}`。拆分提交的纯数值源码位于 `runs/numerical_commit_source`，独立全量记录在 `runs/numerical_commit_full_20261101/`。

unified 工作树 `843ddd6` 加统一数值修复，dirty=true，源码 `runs/integrated_source`。`PYTHONPATH=runs/integrated_source:runs/integrated_full_20261101:tests:. python runs/check_integrated_full_20261101.py tests runs/test_nested_contacts.py runs/test_sparse_stopping.py -q -p run_status --durations=20`：**2032 passed、8 skipped、18 subtests passed / 9314.63 s**，退出码 0。记录 `runs/integrated_full_20261101/{full.log,full_result.json}`。

batched 工作树 `ae963d3` 加统一数值、批量求值及嵌套接触修复，dirty=true。`PYTHONPATH=.:runs/worktree_full_20261101 python -m pytest tests -q -p run_status --durations=20`：**2034 passed、8 skipped、18 subtests passed / 4512.17 s**，退出码 0。插件核对源码位置并记录逐项结果；根目录与冻结的 `runs/nested_source` 共 **120** 个 Python 源码文件逐项相同，命令和配置保存在该结果目录的 `manifest.json`。

同一基线的反向构造候选 `runs/opening_jacobian_source`，按界面点独立性用三个反向种子构造开口残差块。`PYTHONPATH=runs/opening_jacobian_source:tests:. python runs/check_opening_source.py runs/test_opening_jacobian.py tests/test_joint_closed_adjoint.py tests/test_joint_adjoint_scaling.py tests/test_separated_adhesion.py -q`：**87 passed / 96.93 s**。四种界面状态、两种力表示、法向约束开/关及普通/补偿映射共 **32** 例与逐行反向对照通过。

相同源码的全量：`PYTHONPATH=runs/opening_jacobian_source:runs/opening_full_20261101:tests:. python runs/check_opening_source.py tests runs/test_opening_jacobian.py -q -p run_status --durations=20`，**2066 passed、8 skipped、18 subtests passed / 5060.46 s**，退出码 0。记录 `runs/opening_full_20261101/{full.log,full_result.json}`。

`python runs/profile_opening_jacobian.py` 使用相同候选：64 粒子、32/128 积分点时，开口 Jacobian 构造 **0.40692→0.01527 s / 1.87557→0.01959 s**，矩阵逐项相同。32 积分点的完整局部反向 `python runs/profile_small_colored.py`：**0.6290→0.2335 s**，16 组输入梯度逐项相同。记录 `runs/opening_jacobian_benchmark_20261101/` 与 `runs/pullback_small_colored_20261101/`。

`PYTHONPATH=runs/opening_jacobian_source:. python runs/profile_colored_backward.py --mode ad --steps 8 --output runs/natural_colored_backward_20261101`：正向目标值 **0.05450356130673139**，41 组状态、切面、历史及刀具数组与原版逐项相同。第一组 867 积分点、**2826×2826** 伴随矩阵及右端高/低分量逐项相同，单组反向 **287.28→35.90 s**；记录该目录的 `forward_state_comparison.json`、`first_adjoint_system_comparison.json` 与 `profile.log`。

## 2026-11-01：数组编译缓存的进程启动崩溃

RTX 4090 主机、Numba 0.68.0，batched 候选的 `python runs/probe_compiled_divide.py raw` 与 `wrapper` 均在 `1÷2` 处以 **−11** 退出；隔离缓存串行构建、第二进程复用均以 **0** 退出。缓存主体导出 `array_divide…v5`，包装器引用 `array_divide…v3`；串行缓存两者均为 `v5`。记录 `runs/compiled_cache_20261101/{result,symbol_versions}.json`。Numba 上游有[并发 gufunc 缓存崩溃记录](https://github.com/numba/numba/issues/4807)。

`4447a85` 加修复，dirty=true：三个数组 gufunc 关闭磁盘缓存，保留即时编译及原运算顺序。新测试在七个独立进程中改变编译顺序、并发启动并模拟部分缓存失效，同时用 Decimal 核对加法、乘法和除法。`python -m pytest tests/test_compensated_processes.py tests/test_compensated_linalg.py tests/test_compensated_products.py -q`：**80 passed / 4.49 s**；同一进程测试在 `ae963d3` 上复现 **−11**，**1 failed / 1.24 s**。记录 cache 工作树的 `runs/cache_fix_20261101/`。

相同源码 `python -m pytest tests -q`：**1715 passed、8 skipped、18 subtests passed / 829.67 s**，退出码 0；记录 `runs/cache_fix_20261101/{full.log,full_result.json}`。

## 2026-11-01：自然事件加速等价与整体能量步长检查

batched 工作树 `ae963d3` 加统一候选，dirty=true。嵌套 `CombinedContacts` 保留叶子接触律，修复刀/板接触再与粒子对组合时的递归调用。`PYTHONPATH=runs/nested_source:tests:. python runs/check_nested.py runs/test_nested_contacts.py -q`：**2 passed / 7.64 s**，平铺/嵌套的力、历史及全部输入反向逐项相同；同入口检查 `tests/test_combined_contacts.py tests/test_mixed_contact_stages.py tests/test_spatial_pair_stages.py tests/test_contact_stage_checkpoint.py`：**16 passed / 646.54 s**。

unified 的修复/缓存快照 `843ddd6`、dirty=true，`PYTHONPATH=runs/natural_repaired_source:. python runs/natural_repaired_events.py`，与 batched 的 `PYTHONPATH=. python runs/natural_batched_events.py`：从未切割初态完成 **8 步、12 个自然分侧事件**，末态 **1467** 个积分点；**41** 组检查点数组、切面有理坐标、事件顺序和时刻全部相同。第 8 步单次耗时 **2320.88→1695.56 s**。对照命令与结果见 batched 的 `runs/natural_batched_20261101/{full_state_comparison,events_comparison}.json`。

batched `runs/nested_source`，`PYTHONPATH=runs/nested_source:. python runs/natural_energy_observer.py --dt <步长> --steps <步数> --no-cutting --output <目录>`：64 粒子、12 mm 方块、h=6 mm、E=20 kPa；接触参数固定为 kn=1000 N/m、kt=500 N/m、c=0.2 N·s/m、μs/μd=0.3/0.2。独立记录动能、仿射动能、重力势能、弹性/接触储能、刀功、黏性与滑动耗散；独立接触力重算差 **0 N**，摩擦功分解最大误差 **3.47e−18 W**。

步长 **200/100/50 μs**：静置推进 **0.6 ms** 的能量收支残差为 **−1.334e−11 / −1.890e−12 / −2.417e−13 J**。增加 `--knife-z .01801 --knife-speed -1`，无切割压入 **1 ms**，步长 **200/100/50/25 μs** 的残差为 **−1.187e−6 / −3.291e−7 / +1.176e−7 / −1.728e−8 J**，绝对值占输入功 **3.963% / 1.118% / 0.405% / 0.0591%**；相邻时间步的最大位置差 **1.140→0.409→0.0395 μm**。记录 `runs/energy_audit_20261101.json`，各运行的 `manifest.json` 保存完整参数和命令。

同一候选，能量观测器用两分量历史差计算切向速率，修复极短事件段先转为 float64 丢失历史增量的问题。`PYTHONPATH=runs/nested_source:. python runs/check_energy_history_precision.py`：时间 **1e−4/1e−12/1e−18/1e−22 s** 的四个解析黏着例通过，黏性功率均为 **2e−5 W**。`python runs/natural_energy_observer.py --dt .0002 --steps 8 --knife-z .01801 --knife-speed -1 --output runs/energy_cut_precise_20261101` 使用相同 PYTHONPATH：**8 个自然分侧事件**，观测前后 **41** 组状态/历史数组完全相同；摩擦功分解最大误差 **2.530e−22 W**。输入功 **1.49925e−4 J**，能量残差 **−1.20544e−6 J（0.8040%）**，累计断裂耗散 **0 J**。记录 `runs/energy_cut_precise_20261101/{energy_totals,state_comparison}.json`；活跃断裂的整体能量检查继续随完整下刀推进。

## 2026-11-01：保留稀疏精化中仍可解析的低位载荷

`843ddd6` 加停止条件修复，linear 工作树 `python -m pytest tests/test_compensated_linalg.py tests/test_compensated_products.py tests/test_compensated_state.py -q`：**91 passed / 1.04 s**。六个单位矩阵小例保留右端 `2^-80 / 2^-110 / 2^-150` 低位。旧实现首次达到残差界时返回，丢失后两档低位；修复继续到相对后向误差不再下降。

unified 候选 `843ddd6`、dirty=true，`PYTHONPATH=runs/repair_source:. python runs/check_repair.py tests/test_cohesive_displacement_precision.py::test_empty_interface_outputs_and_pullback_use_total_gap tests/test_cohesive_force.py::test_nonpositive_canonical_contact_reaction_with_binding_constraint -q`：原 **10 failed**，修复后 **10 passed / 27.16 s**。关闭稀疏加速的独立对照 **10 passed / 24.53 s**；旧除法对照仍 **10 failed**。记录 `runs/unified_failures_20261101/`。

linear 工作树 `python runs/refinement_stopping_20261101/benchmark.py`：同一 **4032** 阶实际系统，五次中位数 **0.18176→0.21625 s**；主值解相同，原方程相对后向误差均 **2.86749e-33**。

相同源码 `PYTHONPATH=. .venv/bin/python -m pytest tests -q`：**1714 passed、8 skipped、18 subtests passed / 723.58 s**，退出码 0。本地与 4090 源码/测试差异逐字节相同。记录 `runs/refinement_stopping_20261101/full.log`、`full_result.json`。


## 2026-11-01：两分量数组除法编译

goal 接触快照 `c20b3dc`、dirty=true，`PYTHONPATH=runs/natural_sparse_source:. python runs/profile_natural_sparse.py`：1311 积分点推进的性能采样中，**186** 次内聚残差/Jacobian 累计 **586.58 s**，推进累计 **660.08 s**；数组除法调用 **3401660** 次，累计 **161.12 s**。采样记录为 `runs/natural_sparse_profile_20261101/profile.pstats`。

linear 工作树 `d3f1d92`，`python runs/ufunc_dispatch_20261101/division.py`：**40** 组标量/向量/矩阵、广播、二进制尺度 −500/−100/0/100/500 的高低分量逐项相同；3 元素除法 **25.163→1.981 μs**，512² 元素 **13.103→3.584 ms**，预热后三次中位数。正式改动复用 LU 内的同一两轮除法修正，`python -m pytest tests/test_compensated_linalg.py tests/test_compensated_state.py tests/test_compensated_products.py -q`：**85 passed / 2.44 s**，含新增六组 Decimal 与广播核对。

自然接触推进 `PYTHONPATH=runs/natural_division_source:. python runs/natural_division_flow.py`：**310.06 s**，18 个状态、余量和历史数组与采样基线完全相同；记录 `runs/natural_division_flow_20261101/`。

`d3f1d92` 加编译除法和六项测试，`python -m pytest tests -q`：**1708 passed、8 skipped、18 subtests passed / 789.28 s**，退出码 0。本地与 4090 源码/测试差异逐项相同；记录 `runs/compiled_division_full_20261101/`。

## 2026-11-01：稀疏分解与两分量残差细化

RTX 4090 主机 CPU，`113b344` 加稀疏细化改动。goal 工作树 `python runs/natural_linear_benchmark_20261101/benchmark.py`：实际自然切割的 **4032×4032** 系统有 **436113** 个非零项；FMA 两分量分解 **21.354 s**，稀疏分解 **0.06948 s**、求解及两轮残差修正 **0.11779 s**。分解填充由 **9014193** 降至 **535779** 项；最终后向误差 **2.86749e-33**，解主值相同、完整两分量解的相对差 **2.06887e-31**，使用原残差判据。

linear 工作树 `python runs/sparse_refinement_full_20261101/small_benchmark.py`：1/3/16 阶系统增加约 **50–62 μs**，64 阶 **0.3705→0.3140 ms**、256 阶 **14.842→3.457 ms**，预热后五次中位数。`python -m pytest tests/test_compensated_products.py tests/test_compensated_linalg.py tests/test_compensated_state.py -q`：**79 passed / 0.67 s**，包含低位独立主元、极小/极大载荷及 Decimal 解核对。

相同源码 `python -m pytest tests -q`：**1702 passed、8 skipped、18 subtests passed / 734.20 s**，退出码 0。本地与 4090 源码/测试差异逐项相同；日志与退出记录为 `runs/sparse_refinement_full_20261101/full.log`、`full_result.json`。

goal 接触快照 `c20b3dc`、dirty=true，`PYTHONPATH=runs/natural_fma_source:. python runs/natural_refined_flow.py`：保存的 **1311** 积分点、**10** 个已分侧粒子状态完成一次接触推进，**361.60 s**，55 次 4032 阶求解均以两轮修正达到原残差界；结果为 1 个阶段、0 个新接触事件。完整自然切割仍在推进。

## 2026-11-01：FMA 补偿乘积

RTX 4090 主机，`4556cfa` 加 FMA 改动，`python runs/fma_probe_20261101/probe.py`：**87809** 组跨数量级有限乘积的主值/余量与原拆分算法相同，**200** 组独立 Decimal 余量核对通过；32/128/256/512 阶 LU 主值、余量与置换相同，512 阶分解 **0.51657→0.10575 s**。生成指令为 `vmulsd` 与 `vfmsub213sd`。

`python -m pytest tests/test_compensated_products.py tests/test_compensated_linalg.py tests/test_compensated_state.py -q`：**71 passed / 2.42 s**，含新增 8 组正常数、次正规数和下溢边界。自然入口 `PYTHONPATH=runs/natural_fma_source:. python runs/natural_fma_comparison.py`：第 5 步 **41** 组状态/历史数组与原实现相同，步骤耗时 **27.03→24.34 s**；记录分别在 linear 工作树 `runs/fma_probe_20261101/`、`runs/fma_full_20261101/` 和 goal 工作树 `runs/natural_fma_step_20261101/`。

`python -m pytest tests -q`：**1694 passed、8 skipped、18 subtests passed / 763.57 s**，退出码 0；测试源码为 `4556cfa` 加 FMA 与边界测试两文件改动。

## 2026-11-01：批量两分量算术编译

RTX 4090 主机，`6591d00` 加数组 TwoSum/TwoProduct 编译。单独原型 `python runs/compiled_expansion_probe.py`：100 组标量/数组/广播及二进制尺度对照，主值和余量逐项相同；3 元素乘积中位耗时 **6.577→0.713 μs**，1908² 元素乘积 **101.65→46.63 ms**。

自然入口编译快照 `PYTHONPATH=runs/natural_array_source:. python runs/natural_array_comparison.py`：第 5 步的 **41** 组材料、切面、历史和刀具检查点数组与原运算逐项相同；进程 **28.33 s**，记录 `runs/natural_array_compile_20261101/`。显式核对导入路径的 `python runs/check_natural_array_source.py`：**89 passed / 1.13 s**；基线算术检查 **63 passed / 2.54 s**。

`6591d00` 加上述两文件改动，`python -m pytest tests -q`：**1686 passed、8 skipped、18 subtests passed / 775.24 s**。本地与 4090 的源码差异完全相同；记录 `BladeMaster-Warp-linear-20261101/runs/compiled_expansions_full_20261101/`。

## 2026-11-01：几何事件终点的浮点分侧

接触归并快照 `c20b3dc`、dirty=true，加编译候选与自然入口。`PYTHONPATH=runs/formal_natural_source:. python runs/root_side_probe.py`：首个根为 **0.00013838755389802664 s**，两个粒子的边界残差同为 **+1.7347235e-18 m**；下一浮点时刻的两残差均为 **−1.7347235e-18 m**。原停止条件接受外侧残差，使第二个粒子再次推进 **1.8797595e-18 s**。

保持原位置容差并要求根位于进入侧。`PYTHONPATH=runs/root_inside_source:. python runs/root_inside_tests.py`：已有几何事件、自然差分及检查点 **86 passed / 61.58 s**，**1574** 次根均满足 `−tolerance <= residual <= 0`。相同进入侧断言在原实现的旋转刀具速度差分用例失败，残差 **+2.1684043e-18 m**；输入和梯度判据保持不变。

`998c18b` 加进入侧修复，`python -m pytest tests -q`：**1686 passed、8 skipped、18 subtests passed / 793.59 s**。本地与 4090 源码/测试差异相同；记录 `runs/entered_root_full_20261101/`。

## 2026-11-01：固定主元顺序的线性代数编译

RTX 4090 主机，`c20b3dc` 加 `perf/compensated-linear-20261101` 编译改动。`python runs/check_linear_equivalence.py`：24 组密集/三角矩阵、1/3/8/16 阶、二进制尺度 −400/0/400，LU 主值/余量、置换、解和残差逐项相同。主元顺序、补偿运算和残差停止条件保持不变。

`python runs/check_linear_actual.py`：实际接触矩阵 **1908×1908**，原分解 **109.1455 s**、编译分解 **0.10024 s**；原求解 **0.84195 s**、编译求解 **0.01276 s**。LU、置换和解的主值/余量逐项相同，记录 `runs/linear_compilation_20261101/actual_comparison.json`。

`python -m pytest tests -q`：**1686 passed、8 skipped、18 subtests passed / 813.80 s**。本地与 4090 源码差异逐项一致；日志与退出码位于 `runs/linear_compilation_20261101/`。

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
