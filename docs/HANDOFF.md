# 交接说明（2026-10-21，M4 完成，M5 进行中）

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
- `runs/` 不进 git；完整记录保留在 GPU 机器，本机保留主要报告、快照和视频。任务视频在 Mac 的 `~/Desktop/BladeMaster_videos/`。

| 文件 | 内容 |
|---|---|
| `AGENTS.md` | 代码与文档规范（必须遵守） |
| `README.md` | 怎么运行，模块索引 |
| `docs/RUNNING.md` | 完整命令、遥操作选项、录制与诊断 |
| `docs/DIFFERENTIABLE.md` | 当前可微分主线、验收和优化对照 |
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

## 4. 切割力的网格收敛诊断（配合 M4–M5）

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

## 5. 当前主线：可微 TLMPM

具体实现、命令与验收见 [DIFFERENTIABLE.md](DIFFERENTIABLE.md)。先验证梯度，再做轨迹优化；真机标定在其后。第 4 节的切割力检查配合 M4–M5。

| 阶段 | 内容 | 当前状态 |
|---|---|---|
| M1 | 单步弹性、应力伴随、解析与有限差分核对 | 已实现，4090 测试 |
| M2 | 多步状态与分段重算 VJP | 两网格、32/128/512 步的梯度、耗时和显存扫描完成 |
| M3 | 刀与砧板接触、摩擦、推压轨迹优化对照 | 接触分段重算及 8/16 控制量扩展；八控制量 AD/FD/CMA-ES 成功 5/2/5（各五组），长轨迹见 RESULTS |
| M4 | 固定切割分支、多场、塑性和刀路梯度 | 已完成：12 个窗口的正向与梯度、活跃网格约束、完整检查点及 67200 步切割/退刀/推块通过 |
| M5 | 损伤达阈值和切开事件的梯度 | 已接入步内事件定位、G_c 导数和分段重算；推进自然事件扫描及事件后状态导数 |

当前实现：`differentiable.py`、`differentiable_contact.py`、`differentiable_material.py`、`differentiable_cutting.py`、`differentiable_constraint.py`。`precision.py` 从同一份 Warp 函数定义生成双精度核。原求解器作为正向对照。记录器、表面显示和遥操作不参与梯度链。

M1–M3 算例是 24×18×18 mm 弹性方块的侧向推压。M4 已加入原求解器实际下刀产生的香蕉切割窗口，多场、Fp、刀/板/粒子对的接触进入梯度链，侧码与释放标记来自该次正向运行。目标由实际刀路决定切口。

4090 可微实验工作树：`~/Projects/BladeMaster-Warp-diff-20261021`，环境复用主仓库的 `.venv`。基础报告在 `runs/diff_clean/report.json`；`32f0378` 的报告在 `runs/diff_32f0378_{horizon,contact,optimize}/report.json`。接触分段重算与扩展推压提交为 `023d30d`，报告在 `runs/diff_023d30d_{contact_horizon,optimize_q8,optimize_q16,optimize_q16_full_ad}/report.json`。数值见 RESULTS 的“可微 TLMPM”。

M4 数值验收提交 `fe6a422`，dirty=false。保留塑性投影应力、固定传递顺序、双精度原求解器对照、网格约束与完整检查点均已接入。`624dafe` 增加活跃网格约束接入粒子推进的完整 Tape/分段重算/差分检查；提交前全量 **662 passed、8 skipped、18 subtests passed**（227.52 s）。8 项跳过为 CRESSim 上游构建未配置。

最终报告：`runs/m4_fe6a422_windows` 的 12/12 窗口通过正向、刀路、材料及摩擦导数；刀路每例至少两档 ≤2%，最差第二档误差 **0.00950%**。`runs/m4_fe6a422_full` 为 1.05 s、67200 步、两块 **726/726**；刀路三档误差 **0.0232/0.00308/0.00137%**，原求解器与可微求解器的双精度最大 x/Fp/刀力差为 **6.21e-15 m / 2.53e-13 / 6.37e-12 N**。分段全部状态重算差为 **0**，CUDA 数组内存池峰值 **4711.3 MiB**。视频 `~/Desktop/BladeMaster_videos/M4_differentiable_banana_fe6a422.mp4` 来自可微求解器，含切透、退刀、推块和再次退刀，211 帧、20 fps。

同镜头视频对照为 `05a9327`、`runs/m4_05a9327_compare`。两套求解器在同刀路、材质和步长下的最大粒子位置距离 **4.37e-15 m**，回弹、碰撞与刀面带起左块的运动一致。正常速度及 0.1 倍速视频在 `~/Desktop/BladeMaster_videos/M4_original_vs_differentiable[_1x]_05a9327.mp4`；生成方式见 DIFFERENTIABLE。M4 横推速度为 **0.16 m/s**，旧 Franka 三刀任务为 **0.02 m/s**。

完整轨迹入口按网格 CFL 和切向接触历史稳定条件取步长，当前香蕉为 **15.625 μs**，参考接触刚度保持相同。62.5 μs 长轨迹梯度验收失败（放大至 1e18）；步长对照在 `runs/m4_a6ede54_{default,half,quarter}_dt`。原求解器 float32 的 Fp 重复极差达到 **1.475e-5**，记录为 `runs/m4_a6ede54_precision_repeat`；最终正向使用同精度 float64 对照。

M5 已启动。`differentiable_events` 定位固定法向平面穿越和损伤阈值的步内时刻，`FixedCutContact.fracture_energy` 接入 G_c 的 Tape/分段重算导数；10 项新检查覆盖刀位、粒子位移、初始损伤与 G_c。自然事件扫描入口为 `scripts.diff_cut_events`，原推进仍在子步边界切换；下一步将事件时刻导数传入后续粒子状态。

完整轨迹入口 `scripts.diff_cut_trajectory`；窗口入口 `scripts.diff_cut_validate`，命令见 DIFFERENTIABLE。固定接触/屈服标记时差分评估同一分支公式，`--natural-branches` 重算局部分支。固定事件后，现有反力路径对 G_c 的梯度为零。外部指定刀路时，几何分侧与损伤释放是两个独立触发条件，M5 分别测量。

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
