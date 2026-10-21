# 交接说明（2026-10-21，提交 b03cfe2 之后）

给接手的代码代理（Codex 等）。先读本文件，再读 `AGENTS.md`。

## 1. 机器与访问

| 机器 | 角色 | 地址 | 仓库路径 |
|---|---|---|---|
| Mac（用户本机） | 写代码、提交、**唯一 push 的地方**；没有 `.venv`、没有 ffmpeg | — | `/Users/judong/Desktop/Papers/Code/BladeMaster-Warp` |
| tams61（RTX 4060，8 GB） | 跳板机；有 ffmpeg；CRESSim 源码和运行记录在这里 | `dong@100.70.97.92`（Tailscale） | `~/Projects/BladeMaster-Warp` |
| tamsGPU9（RTX 4090，24 GB） | 主要计算机；有 ffmpeg；`runs/` 实验结果都在这里 | 只能从 tams61 进：主机名 `tamsgpu9` | `~/Projects/BladeMaster-Warp` |

### 连接命令

```bash
# Mac → 4060（密钥登录）
ssh -o BatchMode=yes dong@100.70.97.92 '<命令>'

# Mac → 4060 → 4090（复用 4060 上已建立的 SSH 主连接，不需要密码）
ssh -o BatchMode=yes dong@100.70.97.92 \
  'ssh -o BatchMode=yes -o ControlPath=/tmp/sshcm/tamsgpu9 dong@tamsgpu9 "<命令>"'
```

- 4060 上的主连接是这样建的：`ssh -o ControlMaster=yes -o ControlPath=/tmp/sshcm/tamsgpu9 -o ControlPersist=24h -fN dong@tamsgpu9`。空闲超过 24 小时会断开。断开后重建需要**用户本人输入 4090 的密码**；代理不能也不应该这样做。如果上面的命令报 `Permission denied` 或 `Control socket connect ... No such file`，停下来请用户重建。
- **密码绝不写进仓库、文档、commit message 或回复。**
- 4090 上的文件传回 Mac 要经过 4060 中转：

```bash
ssh -o BatchMode=yes dong@100.70.97.92 'mkdir -p /tmp/bmv && scp -q -o BatchMode=yes \
  -o ControlPath=/tmp/sshcm/tamsgpu9 "dong@tamsgpu9:Projects/BladeMaster-Warp/runs/<路径>" /tmp/bmv/'
scp -q -o BatchMode=yes 'dong@100.70.97.92:/tmp/bmv/*' <Mac 上的目标目录>/
```

### 环境

- 4090：Python 3.12.15，Warp 1.18.0，`.venv/bin/python`，硬盘剩余约 470 GB。
- 运行方式：`cd ~/Projects/BladeMaster-Warp && PYTHONPATH=. .venv/bin/python -m <模块>`。
- 长任务用 `nohup bash <脚本> > /dev/null 2>&1 &` 放到后台，结果写进 `runs/<名字>_<commit>/`，进度写一个 `done.txt`；查看时用短命令轮询，不要在一条命令里 `sleep` 很久（工具会中断）。

### 注意事项

- 不要用 `pkill -f <模式>`：模式会匹配到你自己的 ssh 命令行，把会话杀掉。要停进程，先用 `ps -eo pid,args | grep -E "[r]ender_franka"` 找到 PID，再按 PID `kill`。
- GPU 机器是共享的：不要杀别人的进程，开跑前用 `nvidia-smi` 看一眼占用。
- 不要在 GPU 机器上启动 `franka_run --live`，除非任务就是测试遥操作。
- 同步流程：Mac 上提交，`git push origin sync-20261020`；然后在 4060 和 4090 上各执行一次 `git pull`（它们通过 HTTPS 拉取）。

## 2. 仓库与文档

- GitHub：`git@github.com:Ju6276/BladeMaster-Lab.git`，工作分支 `sync-20261020`。Mac 上另有一个 remote `tams` 指向 4060 的仓库，不要用它 push。
- `runs/` 不进 git，只在 GPU 机器上。任务视频在 Mac 的 `~/Desktop/BladeMaster_videos/`。

| 文件 | 内容 |
|---|---|
| `AGENTS.md` | 代码与文档规范（必须遵守） |
| `README.md` | 怎么运行，模块索引 |
| `docs/RUNNING.md` | 完整命令、遥操作选项、录制与诊断 |
| `TODO.md` | 未完成项（按论文图编号） |
| `docs/METHOD.md` | 论文公式到代码的对照；与论文的差异表 |
| `docs/PARAMETERS.md` | 论文给出的实验参数，以及哪些是本地取值 |
| `docs/RESULTS.md` | 实验命令、提交号、关键数字 |
| `docs/PAPER_PLAN.md` | 投稿计划、相关工作开源情况、腕部力/力矩标定的做法 |
| `docs/HANDOFF.md` | 本文件 |
| `NOTICE.md` | 第三方代码来源与许可（DexLab Apache-2.0 的署名必须保留） |

### 工作规则（用户要求，必须遵守）

用户的规则 `algorithm-not-demo`：

- 修仿真、接触、表面、力或时间步时，改的是算法在所有材料、网格和刀路上的行为。某个 demo、某一帧、某个种子上的数字只用来定位错误，不写进代码。
- 从守恒、拓扑、求交、步长这些不变量出发找错误，再改对应的算法。同一处修改要用解析例子核对，并换一条刀路或另一种材料再核对。
- 常数要来自离散本身（网格、刀厚、已有容差），并写明它为什么对所有情况都成立。
- 不做：为某一帧、某个种子、某种材料写分支；毫米上限、经验系数、事后把顶点推回去；用特判掩盖没通过的验收。
- 一个改动如果只在引出它的那个例子上变好，换材料、刀路或网格就不成立，就撤掉。
- 不重调 \(s_n\) 或接触律来"修"结果。
- 本项目和 WowSkin 无关，不要把两者联系起来。

提交前跑测试：`PYTHONPATH=. .venv/bin/python -m pytest tests -q`（在 4090 上跑；Mac 上只能 `python3 -m py_compile`）。

## 3. 当前进展

复现情况（详见 `TODO.md`、`docs/RESULTS.md`）：

| 图 | 状态 |
|---|---|
| Fig. 1 香蕉三刀 | 已复现，四块 |
| Fig. 3 重力落刀 | 钝刀停在离板 14.7 mm；锋利刀到板，但顶角 2 个粒子没拿到侧码，仍是一块 |
| Fig. 4 中片抬升 | 已复现，多场 37.7 mm，单场最高 4.3 mm |
| Fig. 5 两刀三块 | 切口 0.706 mm、体积 0.924%（论文 0.654 mm、0.506%）；左块在推之前就有 −10.5 mm/s 的滑行 |
| Fig. 6 三把飞刀 | 按项目页视频改为平放、刀尖朝前射入端面：29.9、49.3、68.8 mm |
| Fig. 7、Fig. 8 | 已复现 |
| Fig. 9 | 缺真机数据 |

其他：

- 遥操作可选 `--material fig4-cake|fig7-tofu|fig8-luncheon`；网页和离线视频的粒子按切开的块着色（`topology.StablePieces`、`material_colors.py`）。
- 全部论文任务视频已按网页默认相机渲染，在 Mac 桌面的 `BladeMaster_videos/`。
- 基线：CRESSim-MPM（固定上游 `09aa5009` 编译）、TopoCut 式单场 MLS-MPM（`mlsmpm.py`，按论文重写）、单场 TLMPM（`--single-field`）。

## 4. 下一步：切割力的网格收敛诊断（优先）

### 现象

香蕉竖直切，默认接触半径 \(R_p = (V_p^0)^{1/3}/2 = \Delta x/4\)（论文式 10 的形式，\(s_R = 1\)），每种网格 20 个种子（`runs/force_seed_scan_20261019`，4090）：

| 网格 | 稳态切割力 | 单次耗时 |
|---|---|---|
| 2.5 mm | 0.818 N | 约 7 s |
| 2 mm | 0.625 N | 约 15–30 s |
| 1.5 mm | 0.453 N | 约 220–490 s |

### 已有的线索（尚未写进 RESULTS）

`runs/radius_series_fine_20261019/rows.jsonl`：同样的切法，但刀接触距离固定为绝对值 0.5 mm（`--blade-contact-radius 0.0005`），每种网格 5 个种子：

| 网格 | 稳态切割力（5 个种子平均） |
|---|---|
| 2.5 mm | 约 0.580 N |
| 2 mm | 约 0.626 N |
| 1.5 mm | 约 0.651 N |

相邻两档的差值从 0.046 N 缩小到 0.025 N，看起来在收敛。这说明主要原因很可能是"刀接触距离随网格变"。但这组数据来自旧提交 `51ac3ee`，而且当时工作区有未提交改动（dirty），**必须在当前代码上重跑确认**。另外 `runs/force_components_20261020/components.json` 显示，6 mm 网格的峰值力里摩擦约占 0.545 N（总共 0.812 N），所以摩擦分量要单独看。

### 要做的事

1. 先固定网格、种子、刀路与接触设置，改变切割阈值组合 \(G_c(1-\zeta)/\zeta\)，记录力、损伤达阈值粒子数与首次达阈值时刻。2026-10-21 的 2.5 mm、seed=1 检查中，\(G_c\) 加倍时平台均值只变化 0.04%，两次均无粒子达到损伤阈值，见 RESULTS。
2. 在当前提交上重跑：默认半径与固定 0.5 mm 两组，各取 2.5、2、1.5 mm 网格 × 5 个种子；固定半径那组再加 1 mm 网格（每次约 30 分钟）。入口为 `scripts/force_convergence.py`，两组使用不同输出目录。
3. 力按分量分别统计：刃口法向、刀面法向、摩擦、网格约束（参考 `force_components_20261020` 的输出），看收敛与否主要由哪个分量决定。
4. 再逐项固定另外两个随网格变的量：刃前带宽 \(b_\text{lead}\)（现在取 2 倍接触距离）、切割带宽 \(h_k = 6\Delta x\)（论文取值）。
5. 能量核对分两层：先核对每粒子的方向接触功、损伤增量和阈值截断；再统计刀与重力输入功、动能、弹性储能、塑性与摩擦耗散及剩余数值误差。`directional_work_j` 是损伤驱动的计功量；几何侧标签可在损伤未达阈值时分开材料，因此新增切面面积与这份计功的关系需要单独测量。
6. 换材料或刀路再核对一次（例如 Fig. 8 午餐肉薄切）。
7. 结论写进 `docs/RESULTS.md`"BladeMaster 力的收敛"一节，并更新 `TODO.md` 第 3 节。

判断结果的归属：

- 是代码 bug：修掉。
- 是本地取值（接触距离、\(b_\text{lead}\)）：改成与网格无关的取法，写进 `docs/METHOD.md` 的差异表。
- 是论文自己的设计（式 10 的 \(R_p\) 随粒子体积缩放、\(h_k = 6\Delta x\)）：这就是论文算法部分的切入点（见 `docs/PAPER_PLAN.md`）。

## 5. 可微分计划（可选方向，在第 4 节之后）

目的：对材料参数（E、屈服应力、切割阈值、摩擦）和刀路求梯度，用于标定和轨迹优化。正向仿真、遥操作和渲染都不变。

### Warp 自动微分的规则（官方文档）

- 要求导的数组用 `requires_grad=True`；把正向包在 `wp.Tape()` 里；最后调用 `tape.backward(loss)`。
- 反向需要正向时的中间值。**数组被某个 kernel 读过之后不能再被覆盖**，否则梯度会悄悄算错。每一步要单独保存状态，或者做检查点（Warp 不提供，要自己实现；参考 Warp 的 fluid checkpointing 示例）。
- 支持原地 `+=`、`-=`，不支持原地 `*=`、`/=`。
- 调试时设 `wp.config.verify_autograd_array_access = True`，它会报出"读后写"的问题。
- 参考实现：Newton 的 `newton/examples/diffsim/example_diffsim_soft_body.py`（优化 Lamé 参数），以及用 `check_grad()` 和有限差分对比的 `example_diffsim_spring_cage.py`。

### 本仓库里要改的地方

- `solver.py`、`coupling.py` 用了 CUDA Graph（`wp.ScopedCapture`、`wp.capture_launch`），状态是两份数组来回交替（接触历史按 `step_count` 的奇偶切换）。求导模式下要关掉 CUDA Graph，改为逐步保存状态或检查点。
- 原子操作：
  - `material.py:42`（哈希插入 `atomic_cas`）、`material.py:75/818` 和 `contact.py:801`（链表 `atomic_exch`）只用来建索引，反向时当常数即可。
  - `contact.py:1029/1091`、`blade_constraint.py:71` 是标志位，不参与求导。
  - `blade_constraint.py`（持刀时的网格面约束，论文里没有）是迭代求解，用 `atomic_max` 判断残差收敛。要么在求导模式下关掉它（`solver.Config.blade_grid_constraint=False`），要么手写伴随（`Tape.record_func`）。
- 侧码、切开标志（`cut_released`）、静摩擦/滑动摩擦切换、接触开关都是离散的：在 M4 中把它们当作前向确定的常数。

### 里程碑

| 步骤 | 内容 | 验证 | 估计 |
|---|---|---|---|
| M1 | 单步弹性 TLMPM 可微（不切、不接触） | 有限差分；方块在重力下的静下沉量与 E 的解析关系 | 约 1 周 |
| M2 | 多步，加检查点 | 用已知 E 生成假数据，再用梯度反推出 E | 约 1 周 |
| M3 | 刀、砧板接触，摩擦 | 有限差分；在穿透量约等于 0 附近测梯度是否稳定 | 1–2 周 |
| M4 | 固定侧码与接触分支，对连续材料参数求切割力梯度 | 用已知参数生成力曲线，检查 E、屈服应力、摩擦的有限差分与参数辨识 | 1–2 周 |
| M5 | 损伤达阈值与切开事件的梯度 | 先用阈值扫描确认事件会改变力，再比较事件时间求导、平滑损伤等方案。\(G_c\) 在现有反力路径中只改变离散分支，固定事件后其力梯度为零；标定组合 \(G_c(1-\zeta)/\zeta\) 需要处理这些事件 | 研究阶段，方案待选 |

每一步都在 `tests/` 里加有限差分测试，结果写进 `docs/RESULTS.md`。

## 6. 参考资料

论文与项目：

- BladeMaster：https://arxiv.org/html/2609.27342v2 ，项目页 https://jango6324.github.io/blademaster/ （代码未开源）
- TLMPM：de Vaucorbeil, Nguyen, Hutchinson 2020（本仓库粒子更新沿用其式 13）
- MLS-MPM / CPIC：Hu 等，SIGGRAPH 2018，代码 https://github.com/yuanming-hu/taichi_mpm （MIT）
- CRESSim-MPM：https://github.com/yafei-ou/CRESSim-MPM （BSD-3，本仓库固定 `09aa5009`）
- TopoCut：https://arxiv.org/abs/2509.19712 （未找到官方代码；它基于 https://github.com/zhouxian/FluidLab ）
- DiSECt：https://github.com/NVlabs/DiSECt （FEM 可微切割，切面预先给定；仓库内是 LS-DYNA 仿真数据）
- RoboNinja：https://github.com/real-stanford/roboninja （Taichi 可微 MLS-MPM；分离由单场 MLS-MPM 隐式处理，没有切口标记；刀与核的碰撞用软化损失）
- EMPM：https://github.com/rai-opensource/empm （同 BladeMaster 组，从 RGB-D 标定可微 MPM）
- PhysTwin：https://github.com/jianghanxiao/phystwin （Warp 弹簧质点模型；拓扑参数用 CMA-ES，稠密参数用梯度）
- Newton：https://github.com/newton-physics/newton （基于 Warp，Apache-2.0；有可微 FEM 示例，MPM 示例只有正向）
- Warp 自动微分：https://nvidia.github.io/warp/stable/user_guide/differentiability.html
- 梯度质量：Suh 等，*Do Differentiable Simulators Give Better Policy Gradients?*，ICML 2022
- 切割真机数据：Jamdagni & Jia，IROS 2019、ICRA 2021、T-RO 2024（论文表里有 E、ν、断裂韧性，数据未公开）；Quantified Knife Project https://github.com/seattleultrasonics/Quantified-Knife-Project （公开，只有电子秤测的竖直力）
- 网格依赖：Bažant & Oh 1983，裂纹带模型（断裂能随网格变化的经典问题）
