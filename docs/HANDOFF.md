# 交接说明（2026-10-22，M4 完成，M5 进行中）

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
- 用反向计算的峰值显存安排任务。4090 上的 4 mm 完整切割反向验收单独运行；启动时的低占用不等于检查点重算时的峰值。
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

可微分主线的长期要求保存在 [AGENTS.md](../AGENTS.md) 的“用户的长期要求”中。

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
| M5 | 损伤达阈值和切开事件的梯度 | E1–E3 原版本通过，E3 当前公式复验进行中；E4 的 ca90da9 五种长轨迹通过；继续 E5 能量与界面接入 |

当前实现：`differentiable.py`、`differentiable_contact.py`、`differentiable_material.py`、`differentiable_cutting.py`、`differentiable_constraint.py`、`differentiable_events.py`；M5 自然事件与阶段积分位于 `differentiable_natural.py`、`differentiable_geometry.py`、`differentiable_midpoint.py`。`precision.py` 从同一份 Warp 函数定义生成双精度核。原求解器作为正向对照。记录器、表面显示和遥操作不参与梯度链。

M1–M3 算例是 24×18×18 mm 弹性方块的侧向推压。M4 已加入原求解器实际下刀产生的香蕉切割窗口，多场、Fp、刀/板/粒子对的接触进入梯度链，侧码与释放标记来自该次正向运行。目标由实际刀路决定切口。

4090 可微实验工作树：`~/Projects/BladeMaster-Warp-diff-20261021`，环境复用主仓库的 `.venv`。基础报告在 `runs/diff_clean/report.json`；`32f0378` 的报告在 `runs/diff_32f0378_{horizon,contact,optimize}/report.json`。接触分段重算与扩展推压提交为 `023d30d`，报告在 `runs/diff_023d30d_{contact_horizon,optimize_q8,optimize_q16,optimize_q16_full_ad}/report.json`。数值见 RESULTS 的“可微 TLMPM”。

E1 的 `runs/e1_0a8fd13_*` 报告在上述工作树；`runs/e1_7419b30_{materials,shapes,refine}` 在 `~/Projects/BladeMaster-Warp-e1-candidates-20261022`，本机 `runs/` 保存对应 `report.json`。

E2/E3 数值报告：`~/Projects/BladeMaster-Warp-e23-20261022/runs/e23_fb379d4`；后续长轨迹与独立时间步对照在 `~/Projects/BladeMaster-Warp-e3-refine-20261022/runs/e3_39741bf`。这些工作树固定提交运行，修改在 Mac 和测试工作树完成。时间步扫描的空 `--epsilons` 单独比较状态、力、功和 AD 梯度，自然差分由验证入口完成。 CPU 检查点的工作树为 `~/Projects/BladeMaster-Warp-e3-cpu-20261022`（`a55b185`），报告在 `runs/e3_a55b185`；4 mm 完整轨迹使用 `--checkpoint-device cpu --segment-steps 32`，长反向任务独占 GPU。相同 4 mm、50 ms、8 步分段的 GPU/CPU 显存峰值为 **1.906/0.791 GiB**，状态差 **0**、刀路梯度差 ≤**1.53e-14**。该提交全量测试 **863 passed、8 skipped、18 subtests passed**。

M4 数值验收提交 `fe6a422`，dirty=false。保留塑性投影应力、固定传递顺序、双精度原求解器对照、网格约束与完整检查点均已接入。`624dafe` 增加活跃网格约束接入粒子推进的完整 Tape/分段重算/差分检查；提交前全量 **662 passed、8 skipped、18 subtests passed**（227.52 s）。8 项跳过为 CRESSim 上游构建未配置。

最终报告：`runs/m4_fe6a422_windows` 的 12/12 窗口通过正向、刀路、材料及摩擦导数；刀路每例至少两档 ≤2%，最差第二档误差 **0.00950%**。`runs/m4_fe6a422_full` 为 1.05 s、67200 步、两块 **726/726**；刀路三档误差 **0.0232/0.00308/0.00137%**，原求解器与可微求解器的双精度最大 x/Fp/刀力差为 **6.21e-15 m / 2.53e-13 / 6.37e-12 N**。分段全部状态重算差为 **0**，CUDA 数组内存池峰值 **4711.3 MiB**。视频 `~/Desktop/BladeMaster_videos/M4_differentiable_banana_fe6a422.mp4` 来自可微求解器，含切透、退刀、推块和再次退刀，211 帧、20 fps。

同镜头视频对照为 `05a9327`、`runs/m4_05a9327_compare`。两套求解器在同刀路、材质和步长下的最大粒子位置距离 **4.37e-15 m**，回弹、碰撞与刀面带起左块的运动一致。正常速度及 0.1 倍速视频在 `~/Desktop/BladeMaster_videos/M4_original_vs_differentiable[_1x]_05a9327.mp4`；生成方式见 DIFFERENTIABLE。M4 横推速度为 **0.16 m/s**，旧 Franka 三刀任务为 **0.02 m/s**。

完整轨迹入口按网格 CFL 和切向接触历史稳定条件取步长，当前香蕉为 **15.625 μs**，参考接触刚度保持相同。62.5 μs 长轨迹梯度验收失败（放大至 1e18）；步长对照在 `runs/m4_a6ede54_{default,half,quarter}_dt`。原求解器 float32 的 Fp 重复极差达到 **1.475e-5**，记录为 `runs/m4_a6ede54_precision_repeat`；最终正向使用同精度 float64 对照。

M5 当前有三层：`EventCutContact` 按剩余断裂功积分释放冲量；`NaturalCutContact` 从当前状态生成侧码、接触对与历史；`GeometryEventContact` 在实际材料推进中求分侧时刻，按时刻切分状态推进与多场交换。时长进入粒子、塑性、接触历史、损伤功和刀力的求导链。刀刃平面穿越后保留后继计功分支，避免根残差的舍入符号重新触发刃前计功；同时穿越和 ±1 浮点间隔检查已接入。

2026-10-22 复核 `d3124a0`（dirty=false）：全量 **785 passed、8 skipped、18 subtests passed**，8 项跳过为未配置的 CRESSim 上游构建。修复双精度向量入口舍入及固定刚度实验的显式砧板系数换算；独立 NumPy 核对 24 个本构状态的五组 VJP，几何事件测试 **55** 项。M1–M4 的短窗、长轨迹和分段反向已复测；M4 67200 步的三档刀路差分误差 **0.023270/0.002956/0.001330%**，状态重算差 **0**。

M5 损伤释放的 **16/16** 窗口通过，最细两档刀路/G_c 梯度差 ≤**0.856%/0.261%**，损伤功平衡误差 ≤**1.29e-19 J**。8 个自然分侧窗口的局部差分通过，原七档 **112** 组扰动仍有 **49** 组误差 >2%、**6** 组符号相反。全部命令和数据见 RESULTS 的“M1–M5 代码复核”。

M5 E1 已通过：`0a8fd13` 用显式中点 RK2 和摩擦历史终点投影推进事件分段；三材料、6/4 mm、三个时间网格相位共 **27** 档对照，最细相邻两档刀路/材料梯度差 ≤**0.1279%/0.06259%**，位置差 ≤**1.041 nm**、事件时刻差 ≤**0.2691 ns**，事件与最终侧码集合一致。`7419b30` 补充二次轨迹候选及旋转分段查询，抛物线粒子、半圈刀和整圈刀的步内进入再离开测试通过。方法和命令见 METHOD、RESULTS。

M5 E2 已通过：**18** 项接触、粘滑与刀缘检查；刀/板/粒子对、两网格、三相位、四档步长共 **72** 例解析碰撞，各有两档自然差分误差 ≤2%，最差第二档 **0.005707%**。`a062351` 的全量测试为 **850 passed、8 skipped、18 subtests passed**。

M5 E3 在 `fb379d4`、`39741bf`、`a55b185` 通过：6/4 mm 从未切割初态运行 50 ms 释放、0.24 s 下刀及 1.05 s 完整轨迹，**50** 组自然差分最大误差 **0.9185%**，**17** 组检查点状态重算差为 **0**。6 mm、G_c=700 完整轨迹得到 **726/726** 两块，16/32/128 步分段通过，两档刀路差分最大误差 **0.0460%**；最细相邻时间步的刀路梯度差 ≤**0.0516%**。E5 共用公式更新后的完整 E3 组合待复验；当前已复测代码测试及自然事件短窗口，顺序与配置见 DIFFERENTIABLE。

4 mm、G_c=70 完整轨迹得到 **2446/2446** 两块，**5** 个粒子损伤释放；三项刀路与 G_c 各两档自然差分最大误差 **0.6647%**，CPU/32 步检查点峰值数组显存 **5.431 GiB**。4 mm、G_c=7 的释放短轨迹最细两档刀路/G_c 梯度差 ≤**0.468%/1.426%**。下一项是 E4 多刀与永久历史；整体能量、网格/场景泛化和切割优化分别为 E5–E7。

E4 完整暂停续切、平行两刀和交叉两刀通过，报告在 `~/Projects/BladeMaster-Warp-e4-20261022/runs/e4_8d35d5c/`。交叉两刀的 4 组自然差分最大误差 **0.07377%**，32/127 步检查点状态重算差 **0**。4 mm、0.64 s 交叉窗口也通过：4 组差分最大误差 **0.02062%**、检查点重算差 **0**。三刀后推块 4 项差分最大误差 **0.14894%**、检查点重算差 **0**；后两者使用 `f60a5c4`，工作树为 `~/Projects/BladeMaster-Warp-e4-audit-20261022`。

旧版 E4 状态在 `runs/e4_f60a5c4/status.json`，日志为 `crossing_h4.log` 和 `three_cuts_h6.log`，均已完成。

`f60a5c4` 全量 **900 passed、8 skipped、18 subtests passed**。验收拒绝非有限差分和检查点状态，多刀事件按切割编号导出；物理推进和梯度公式与 `8d35d5c` 相同。

E5 能量基准工作树为 `~/Projects/BladeMaster-Warp-e5-20261022`（`33a7001`），接触速度修正工作树为 `~/Projects/BladeMaster-Warp-e5-transfer-20261022`。材料能量 **28** 项、刀具计功 **18** 项、塑性返回 **27** 项及接触能量 **27** 项检查通过；全量 **1003 passed、8 skipped、18 subtests passed**（271.00 s）。自然报告保存起终点动能、重力势能和弹性储能，以及包含旋转与网格约束的 `knife_work_j`。

共旋储能配套的增量变分返回已接入原求解器和可微流程；驻点求导使用隐式 VJP。强压缩两例的储能分别下降 **1399.9545/10030.5700 J/m³**；400 组随机形变的耗散、屈服与塑性体积核对通过，屈服边界测试扰动到 **1e-12**。

切向力与弹性历史共用 Kelvin–Voigt–滑动元件更新，θ=1/2 用于 M5 中点，θ=1 用于原求解器/M4。零摩擦同参数小例的虚增储能从 **5e-5 J** 降为 **0**，三档步长相同；随机能量恒等式残差 ≤**2e-18 J**。公式、来源与步长条件见 METHOD。旧 `lagged/projected` 选项已替换为 `implicit/midpoint`。

公式更新后的自然窗口重验已完成，干净工作树 `~/Projects/BladeMaster-Warp-e5-friction-20261022`，日志 `runs/e5_109fe3f/banana_h6.log`。6 mm 香蕉、2 ms、三相位、dt=15.625/7.8125/3.90625 μs，固定 PIC 参考时间 15.625 μs，比较 32/127 步 CPU 检查点及两档自然差分。`43a939a` 的塑性交叉轨迹已结束，报告为 `~/Projects/BladeMaster-Warp-e4-energy-20261022/runs/e4_43a939a/crossing/report.json`：横移两档误差 **8.03%/13.94%**，深度两档 **7.53%/0.01368%**，3 项失败，队列退出。32/127 步检查点重算状态差均为 **0**。`runs/e4_43a939a/trace_crossing.py` 从未切初态重放基准及横移 ±1e-7 mm，逐段保存状态和约束统计；基准目标精确复现，全部阶段的网格约束求解次数 **0**。

当前公式的 6 mm 交叉两刀复验工作树为 `~/Projects/BladeMaster-Warp-e4-current-20261022`（干净 `ca90da9`），日志 `runs/e4_ca90da9/crossing.log`，报告目录 `runs/e4_ca90da9/crossing/`。命令显式使用 `--dt .000015625 --flip-reference-dt .000015625`，其余配置与上面的长轨迹相同。四项自然差分全部通过：横移 ε=1e-6/1e-7 mm 误差 **0.00094923%/0.00033584%**，深度 **0.0112984%/0.0371589%**。32/127 步检查点状态重算差 **0**，刀路梯度差 **2.297e-15**，暂停损伤增量 **0**。双精度、固定累加、中点积分及事件求根修复均保留；相关 **105** 项复核通过。`ca90da9` 全量 **1017 passed、8 skipped、18 subtests passed**（273.94 s）。

旧轨迹 `runs/e4_43a939a/trace_jump.py` 已定位横移 ε=1e-7 mm 的一个跳变：第 44593 步，粒子 1167 的刀距跨过 0，旧摩擦端点回算使历史从 0 跳到 **1.372 μm**。`109fe3f` 已替换该公式；新增 6 项接触起始处的解析连续性检查通过，完整轨迹结果继续以当前公式报告核对。

自然窗口的 36 项差分最大误差 **0.034311%**，32/127 步检查点重算状态差 **0**，刀路梯度差 ≤**7.61e-15**；最细两档梯度差 ≤**0.17884%**。

E5 逐段能量账本已接入静置、压入和自然切割，包含接触储能、塑性/摩擦/PIC 分项和几何事件能量。9 项账本测试、4 项虚功/转置导数及非均匀速度场能量测试通过。M5 中点接触使用与材料推进共轭的插值速度；塑性初始线搜索采用代数等价的稳定能量差，1792 个近屈服状态全部通过。当前数值提交 `ec7bcb4`，全量 **1017 passed、8 skipped、18 subtests passed**（290.38 s）。

G_c=7/70/700/1400 与三档步长的 50 ms 扫描已完成，报告在 `e5-transfer` 工作树的 `runs/e5_current/{block_plastic,banana_release}`，共 **18/12** 例。方块压入/切割的最细步长残差 **0.000375%/0.004001%**，各方块组合均未释放。香蕉 G_c=7/70 各释放 **4** 粒子，残差 **23.60%/7.31%**，三档步长不下降；G_c=700/1400 未释放，残差幅值从 **0.000594%** 降为 **0.0000579%**。`runs/e5_release_cycle.py` 的零摩擦闭合路径核对得到释放后净功 **7.786e-5 J**，反向路径净功变号；未释放净功在采样细化后降至 **6.46e-12 J**。下一项按独立断裂阻力与压缩接触推导实际力、储能和耗散，完成局部能量/梯度核对后接入自然事件，重跑扫描及受影响的 E1–E4。

`e5-transfer` 的 `runs/e5_release_audit/cohesive_unit.py` 已完成双线性内聚单元的独立验证：四档 G_c 的加载—卸载总功误差 ≤**2.615e-11 J/m²**，192 组状态的 AD/FD 最大相对误差 **1.653e-8**，保留损伤与断后压缩接触。来源、测试参数和数字见 RESULTS。

当前界面工作树为 `~/Projects/BladeMaster-Warp-e5-cohesive-20261022`。`d4bbb6f` 增加多场交换前的网格冲量入口，7 项独立传递/功配对/转置导数检查通过；全量 **1030 passed、8 skipped、18 subtests passed**（261.66 s）。干净提交下 `runs/e5_release_audit/cohesive_grid.py`、`cohesive_grid_refine.py` 完成 24 例平面双侧算例；`cohesive_surface_grid.py` 完成旋转/剪切算例。0.125 μs 三相位的最大相对能量残差 **2.327e-7**，旋转轨迹差 ≤**5.22e-15**，六类输入的两档梯度误差 ≤**8.144e-7**。`769dd89` 已在干净提交下完成旋转/剪切耦合复验，新增 15 项界面测试后的全量为 **1045 passed、8 skipped、18 subtests passed**（249.28 s）。界面响应已整理到 `cohesive.py`；`tests/test_cohesive.py` 包含断裂功、不可逆历史、虚功、客观性、自然差分和加载边界两侧切线。下一项是自然切口的参考面积、双侧映射、插入牵引平衡与检查点历史，再接回完整刀路；插入受载界面时同时核对受力连续性和能量。

同一 `cut_index` 内损伤 D 累积；新刀重置本刀 D，旧侧码与塑性 Fp 保留。M4 使用记录的离散分支；M5 的自然流程在每条扰动轨迹上重新生成侧码、接触对和损伤释放。`scripts.diff_natural_validate --from-initial` 从未切割初态展开，默认窗口模式从原求解器选定的共同状态起步。

受载插入原型保存在 `e5-cohesive` 工作树的 `runs/e5_release_audit/extrinsic_{prox,dynamics,coupled}.{py,json}`，干净 `769dd89`。零开口承载/零插入能、四档断裂功、256 状态梯度、12 例质量—界面步长对照及实际 TLMPM 的 4 界面全矩阵约束均已核对，数字见 RESULTS。原型保留起裂折点和全部差分步长；弱历史方向另有独立隐式方程导数核对。下一项接自然参考面和移动法向的共轭力，再接入完整自然轨迹和检查点；主切割流程仍使用原损伤释放律。

`extrinsic_admm.py` 已处理 16/32 积分点、秩 9 的冗余约束，面积等分复制和 ρ 变化对照通过。`extrinsic_particle.py` 将其接入真实粒子推进：9 例步长细化的能量残差约随步长线性下降，受载闭合例面能/损伤为 0，24/160 步的四类轨迹梯度通过两档自然差分。局部近端核已整理为 `cohesive.cohesive_proximal_response`；全局 ADMM 原型与平面耦合尚在 `runs/`。下一项是随物体运动的面法向及其共轭力、自然生成参考面与永久界面历史；同时统一界面压缩约束与现有粒子接触，避免重复计力。

干净 `6361c84` 下已复跑 ADMM 和粒子推进两份报告，目录不变。局部近端核提交前全量测试 **1059 passed、8 skipped、18 subtests passed**（274.88 s），日志在 `e5-cohesive` 的 `runs/cohesive_prox_full_tests.log`。

移动面压力核 `cohesive_contact_response` 与共享法向导数已实现，新增 4 项测试；全量 **1063 passed、8 skipped、18 subtests passed**（300.46 s）。`e5-cohesive/runs/e5_release_audit/extrinsic_surface.json` 的三档能量残差约随步长线性下降；`extrinsic_surface_short.json` 保存 160 步旋转一致性和闭合前 24 步八类输入导数。

移动面联合约束与隐式反向原型为 `extrinsic_monolithic.py`，局部 10 类输入自然差分通过。`11fbbd5` 下的 `extrinsic_primal_validate.json` 记录 160 步直接位移原型：8 类输入各至少两档误差 ≤2e-5，最差第二档 **1.507e-5**，旋转归一化状态差 **5.641e-11**；全部 32 组差分保留。位移更新源码为 `a7df3f4` 的 `MaterialTrajectory.correct_displacement`，13 项状态/导数等价检查通过，全量 **1076 passed、8 skipped、18 subtests passed**（265.79 s）。干净源码复验工作树 `~/Projects/BladeMaster-Warp-e5-correction-20261022`，日志 `runs/e5_release_audit/extrinsic_primal_validate.log`，32 组差分及旋转检查已通过。下一项合并零开口承载、自然参考面、永久历史和接触去重。完整自然切割仍使用原损伤释放律。

`e4-current` 的 6 mm 完整交叉、暂停续切及 4 mm 交叉窗口已通过，各四项自然差分的最大误差为 **0.03716%/0.005113%/0.009204%**；检查点状态重算差和暂停损伤增量均为 **0**。报告为 `runs/e4_ca90da9/{crossing,resume,crossing_h4_640ms}/report.json`。`continue_cases.py` 已完成平行两刀和三刀后推块；状态在 `runs/e4_ca90da9/continue_status.json`。运行中工作树保持固定提交。

E3 与 E4 并行复验的工作树为 `~/Projects/BladeMaster-Warp-e3-current-20261022`，固定干净 `ca90da9`。启动命令 `PYTHONPATH=. .venv/bin/python runs/e3_ca90da9/e3_queue.py`；状态 `runs/e3_ca90da9/status.json`，逐项日志和报告位于同目录。8 个队列项覆盖 6 mm 活跃释放完整 Tape、4 mm 活跃释放细化、6/4 mm 下刀和完整轨迹、6 mm 下刀与完整轨迹细化。6/4 mm 基准步长及 PIC 参考时间为 15.625/13.157895 μs，接触参考时间 25 μs；每次扰动自然重建事件，梯度保留 2% 标准，失败后队列停止并保留报告。细化报告的相邻梯度和状态趋势另行核对。

`e3-current/runs/e3_ca90da9/release_h6/report.json` 已通过：12 组自然差分最大误差 0.61668%，4 个粒子释放，完整 Tape 与 16/32/128 步检查点状态重算差为 0。当前队列项为 `cut_h6_refine`。

`ca283ec` 修复内聚近端核在零开口、零历史边界的导数；7 项新增解析与单侧差分通过，全量 **1083 passed、8 skipped、18 subtests passed**（286.64 s）。当前统一界面原型工作树为 `~/Projects/BladeMaster-Warp-e5-unified-20261023`，原型 `runs/e5_release_audit/extrinsic_semismooth{,_adjoint,_probe,_validate}.py`；它用近端固定点方程连接零开口承载和张开后的移动面，16 组正向/分裂尺度和 40 类局部输入方向已通过，160 行差分保存在 `extrinsic_semismooth_validate.json`（a7df3f4、dirty=true，对应 ca283ec 源码）。该工作树已核对源码并固定干净 ca283ec；`extrinsic_unified_{surface,trajectory}.py` 接入连续粒子轨迹，包含原有 160 步正历史算例及零历史承载/起裂。自然界面生成、历史接入和完整能量扫描仍待完成。

完整轨迹入口 `scripts.diff_cut_trajectory`；窗口入口 `scripts.diff_cut_validate`，命令见 DIFFERENTIABLE。固定接触/屈服标记时差分评估同一分支公式，`--natural-branches` 重算局部分支。M4 固定释放事件时反力对 G_c 的梯度为零；M5 释放窗口由累计损伤更新事件，G_c 梯度进入后续状态。外部指定刀路时，几何分侧与损伤释放是两个独立触发条件。

## 6. 参考资料

论文与项目：

- BladeMaster：https://arxiv.org/html/2609.27342v2 ，项目页 https://jango6324.github.io/blademaster/ （代码未开源）
- TLMPM：de Vaucorbeil, Nguyen, Hutchinson 2020（本仓库粒子更新沿用其式 13）
- MLS-MPM / CPIC：Hu 等，SIGGRAPH 2018，代码 https://github.com/yuanming-hu/taichi_mpm （MIT）
- CRESSim-MPM：https://github.com/yafei-ou/CRESSim-MPM （BSD-3，本仓库固定 `09aa5009`）
- TopoCut：https://arxiv.org/abs/2509.19712 （未找到官方代码；它基于 https://github.com/zhouxian/FluidLab ）
- DiSECt：https://github.com/NVlabs/DiSECt （FEM 可微切割，切面预先给定；含 `dflex/` 求解器、`cutting/` 接口及 `dataset/` 的 LS-DYNA 仿真数据）
- RoboNinja：https://github.com/real-stanford/roboninja （Taichi 可微 MLS-MPM；分离由单场 MLS-MPM 隐式处理，没有切口标记；刀与核的碰撞用软化损失）
- EMPM：https://github.com/rai-opensource/empm （同 BladeMaster 组，从 RGB-D 标定可微 MPM）
- PhysTwin：https://github.com/jianghanxiao/phystwin （Warp 弹簧质点模型；拓扑参数用 CMA-ES，稠密参数用梯度）
- Newton：https://github.com/newton-physics/newton （基于 Warp，Apache-2.0；有可微 FEM 示例，MPM 示例只有正向）
- Warp 自动微分：https://nvidia.github.io/warp/stable/user_guide/differentiability.html
- 梯度质量：Suh 等，*Do Differentiable Simulators Give Better Policy Gradients?*，ICML 2022
- 切割真机数据：Jamdagni & Jia，IROS 2019、ICRA 2021、T-RO 2024（论文表里有 E、ν、断裂韧性，数据未公开）；Quantified Knife Project https://github.com/seattleultrasonics/Quantified-Knife-Project （公开，只有电子秤测的竖直力）
- 网格依赖：Bažant & Oh 1983，裂纹带模型（断裂能随网格变化的经典问题）

统一原型局部检查在干净 `ca283ec` 上通过 40 个方向、160 行差分。共旋应力的浮点修正已完成：80 位参考核对的最大应力误差 6.9860e-12 Pa，256 组小形变两条调用路径通过，全量 1085 passed、8 skipped、18 subtests passed（304.68 s）。方法见 METHOD，实验命令与数字见 RESULTS。

`extrinsic_energy_state.py` 的统一界面 12 例能量细化已通过；闭合承载保持零开口/损伤，G_c=7/70 张开及旋转受压的能量残差随步长减半约减半。报告 `runs/e5_release_audit/extrinsic_unified_energy.json`，日志 `extrinsic_energy_stress_fixed.log`。应力对照的原标准保持 `atol=1e-10, rtol=1e-12`。

统一界面原型工作树 `e5-unified-20261023` 固定干净 `541c913`。零历史承载/张开 160 步各六类输入、各四档自然差分全部通过；已有损伤的其余七类输入各至少两档通过，报告 `canonical_preload160_scan.json`、`unified_remaining_inputs_scan.json`。零历史张开完整检查为 `canonical_opening160_scan.json`。已有损伤的初速度方向仍失败；`extrinsic_input_scan.py` 收集全部输入和四档原始差分，各输入仍要求至少两档相对误差 ≤2e-5。

`extrinsic_extended_reference.py` 独立实现扩展精度 MUSL/界面前向，两档初速度差分与 AD 的误差为 3.9201e-6/3.7684e-6。`--round-state deformation/velocity/interface/all` 分别核对状态舍入，输出 `extended_{deformation,velocity,interface,all}64/`。`stable_corotated_probe.py` 对照作者稳定应变计算思路与 80 位参考；`e5-stable-20261023` 工作树验证对应源码和极旋转 VJP，全量 **1089 passed、8 skipped、18 subtests passed**，日志 `runs/stable_stress_full.log`；源码初速度轨迹仍为 1/4 档通过，报告 `runs/e5_release_audit/stable_source_trajectory.json`。方法来源、作者源码和本地推导见 METHOD。

当前原型工作树 `~/Projects/BladeMaster-Warp-e5-stable-20261023` 固定干净 `66537e3`，报告位于 `runs/e5_release_audit/`。混合精度几何的 160 步初速度两档误差 **9.54776e-6/1.14416e-6**；`precise_geometry160_remaining.json` 的七类输入均完成，加上初速度共八类输入均至少两档通过。节点原型的已有损伤八类输入、零历史承载/张开各六类输入均通过。6/4/3/2.4/2 mm 空间对照中，节点积分的内部最大负间隙依次约为 **0/1.554/2.088/1.973/1.541 μm**；Gauss 的 6/4/3 mm 积分点负间隙幅值 ≤**1.458e-19 m**，三个网格均通过至少两档自然差分。2 mm 节点四档差分全部通过，最大相对误差 4.02542e-6。

全局界面接口已整理为 `differentiable_cohesive.solve`，非线性方程位于 `_cohesive_solve.py`，测试为 `tests/test_differentiable_cohesive.py`。测试工作树 `~/Projects/BladeMaster-Warp-e5-interface-20261023`，`runs/interface_extract/` 保存 10 项单步测试、24 步八类输入与三种 160 步逐步状态对照；后者与原型的状态/能量/外功差均为 **0**。对应源码全量 **1099 passed、8 skipped、18 subtests passed**，日志 `full.log`。

下一项采用 Gauss 积分接自然参考面积/位置，加入永久界面历史、接触去重、RK2 事件分段和检查点，再复测完整能量扫描及受影响的 E1–E4。几何舍入余量接口为 `cohesive_geometry.InterfaceGeometry` 与 `solve_geometry`；主线自然切割当前使用原释放公式。Crook–Homel 作者 GEOS `feature/crook5/czUpdate` 分支 `e068c207` 的参考状态、节点面积、位移/力映射已核对，文件位于本机 `runs/reference_sources/GEOS/czUpdate/`，固定源码链接及本地推导见 METHOD。

E4 当前 `ca90da9` 平行两刀已通过，报告 `runs/e4_ca90da9/two-cuts/report.json`；四项自然差分最大 **0.07879%**、32/127 步检查点状态差 **0**，最终 **380/648/424** 三块，暂停损伤增量 **0**。`continue_cases.py` 已完成 `three-cuts-push`，四项差分最大误差 **0.09468%**、32/127 步检查点状态差 **0**；最终 **380/346/302/424** 四块。当前 `ca90da9` 五种 E4 组合共 **20** 项差分全部通过。E3 的 `release_h4_refine` 已完成，12 组自然差分最大误差 **0.00043623%**、检查点状态差 **0**；三档侧码/释放/接触集合相同。深度梯度相邻变化 **0.24505%/0.27433%**，补充更细步长和事件相位核对；6 mm、0.24 s 下刀已完成，四组自然差分最大误差 **0.0224614%**、32 步检查点状态重算差 **0**；4 mm 下刀已完成十组自然差分，最大误差 **0.37406%**、32/128 步检查点状态差 **0**；6 mm、1.05 s 完整轨迹八组差分已通过，最大误差 **0.0668341%**，16/32/128 步检查点状态差 **0**；4 mm、1.05 s 完整轨迹八组差分通过，最大误差 **0.0927014%**，32 步检查点状态差 **0**，释放 **5** 粒子；`cut_h6_refine` 已完成，三档共 12 组差分最大误差 **0.009032%**，检查点状态差 **0**；最细相邻横向/深度梯度差 **0.011813%/0.11009%**。`full_h6_refine` 三档细化已完成：128 步检查点状态差 **0**，最细相邻三项刀路梯度差 **0.010024%/0.080899%/0.133284%**，事件集合一致、时刻差 **23.994 ns**。`release_h4_more.py` 的 dt/8 已完成：释放 5 粒子，四组自然差分最大相对误差 **4.921e-7**，32 步检查点状态重算差 **0**；最细相邻横向/深度梯度差 **0.013614%/0.219121%**。dt/16 已完成，位置差缩至 **0.0545 nm**，横向/深度梯度差缩至 **0.0019415%/0.0269824%**，检查点状态差 **0**；侧码、释放、接触与事件集合一致。

补偿几何接口在 `e5-interface-20261023` 的 `b7048c7`、dirty=true 源码上完成 **23** 项局部测试（14.83 s），包括 11 步界面轨迹的 3/4 步检查点状态与梯度对照；全量 **1112 passed、8 skipped、18 subtests passed**（287.52 s）。日志为 `runs/interface_precision/{unit,full}.log`。三种 160 步实际 TLMPM 轨迹与扩展精度原型的归一化状态差 ≤**7.661e-15**，报告 `compare160.json`。初速度两档小扰动误差 **6.42872e-6/9.62676e-8**，报告 `velocity160.json`；其余七类输入也已全部通过原标准，最差第二档 **1.39787e-6**，报告 `remaining160.json`；该工作树已同步至干净 `f507dc0`。主线的自然参考面、永久历史、接触去重及 RK2 接入仍待完成。

E5 参考表面测试工作树为 `~/Projects/BladeMaster-Warp-e5-reference-20261023`，源码基于 `f507dc0`。`reference_interface.reconstruct` 的 **7** 项测试通过；全量 **1119 passed、8 skipped、18 subtests passed**（284.42 s），日志 `runs/reference_interface/{unit,full}.log`。`kinematics.json` 核对静态表面重建和实际 TLMPM 推进：统一仿射状态的重建间隙 ≤5.067e-18 m；分侧后旋转推进仍产生 1.465e-6 m 间隙，未切场为 2.108e-18 m。`closed_grid.json` 的 12 例闭合约束对照全部通过，位置差 0，速度差 ≤5.552e-17 m/s，均匀牵引误差 ≤3.184e-12 Pa。方法推导见 METHOD。

`7f31460` 的统一网格原型完成 **12** 例 200 μs 能量检查，四组残差随步长减半约减半，最细档幅值 **0.017552%/0.002262%/0.007322%/0.348344%**。24 步闭合/已有损伤/受载断裂的六/八/六类输入各至少两档误差 ≤2e-5。报告在同工作树 `runs/reference_interface/grid_{energy,preload24,existing24,loaded24}.json`，命令和配置见 RESULTS；原型入口为 `runs/e5_release_audit/extrinsic_grid_surface.py`。可复用的闭合投影已整理到 `grid_constraints.project_velocity`，**3** 项单元测试通过；提交前全量 **1122 passed、8 skipped、18 subtests passed**（330.79 s），日志 `runs/reference_interface/grid_projection_full.log`。完整网格推进入口及后续接入项见下。

正式首阶接口为 `cohesive_grid_update.advance_material`，工作树 `~/Projects/BladeMaster-Warp-e5-grid-20261023`。新增 **6** 项测试，加网格载荷为 **13 passed**（25.94 s）；全量 **1128 passed、8 skipped、18 subtests passed**（330.77 s），日志 `runs/grid_update/full.log`。`runs/grid_update/compare160.json` 的三种 160 步轨迹与原型状态/能量/外功差均为 **0**；`gradient24.json` 的八类输入均至少两档通过，最差第二档 **2.74149e-6**。参考工作树干净 `22d421a` 的 `grid_library_{existing,loaded}160` 已完成：已有损伤八类/受载断裂六类输入均至少两档通过，最差第二档误差 **4.93259e-6/2.74636e-7**。日志和报告位于 `runs/reference_interface/`。

`reference_cut.quadrature` 的 **14** 项解析和导数检查通过（0.40 s），日志 `runs/grid_update/reference_cut_unit.log`；对应全量 **1142 passed、8 skipped、18 subtests passed**（279.81 s），日志 `reference_cut_final_full.log`。12 mm 方块的 0/0.3/1/2/3 mm 切平面偏移均得到参考面积 **1.44e-4 m²**。`reference_grid.build_trace` 已用多项式延拓补齐双侧支撑，六项测试覆盖三线性重现、切面位置/法向导数和四步材料耦合。固定步长/质量的自定义反向只登记可求导数组；新增闭合状态差分通过。相关 **16 passed**（83.01 s），日志 `reference_grid_coupled.log`；提交前全量 **1149 passed、8 skipped、18 subtests passed**（335.95 s），日志 `reference_grid_full.log`。

`reference_extension_energy.json` 的偏移 1 mm、200 μs、三档步长共九例能量残差随步长减半约减半；旋转/预拉伸/受载断裂最细档为 **0.0000406%/0.005247%/0.002257%**。五网格、八相位的一步旋转共 **40** 例保持零损伤；与未切场的位置 RMS 差最大值为 **32.9/42.0/27.3/19.5/13.1 nm**，报告 `reference_grid_phase.json`。有限刀面裁剪及 `blade_quadrature` 位姿/尺寸接口已完成，几何与网格映射相关 **34 passed**（62.74 s）；6/4 mm 网格、双向法向、两个刀面偏移和十二个刀高共 **96** 个配置符合解析面积及质心，日志 `runs/grid_update/reference_blade_unit.log`。提交前全量 **1163 passed、8 skipped、18 subtests passed**（334.54 s），日志 `reference_blade_full.log`。下一项接永久界面历史、接触去重与 RK2 事件分段，再重复完整能量扫描及受影响的 E1–E4。

累计参考切面入口为 `reference_surface.swept_edge`，面积扣除为 `ReferenceSurface.subtract`，同一工作树的 `reference_surface_unit.log` 为 **41 passed**（9.08 s）；提交前全量 **1176 passed、8 skipped、18 subtests passed**（334.29 s），日志 `reference_surface_full.log`。`reference_surface_measure.json` 中旋转刀刃空间/时间共同细化误差为 **7.57245e-4/1.90330e-4/4.76464e-5**；退刀重入的累计面积为 **2.22e-5/6.66e-5/6.66e-5/8.14e-5 m²**。旧前沿落在粒子域边界处的面积导数已核对。零测度边界片保留有向面积导数，最小浮点权重 **−5.482e-22 m²**；下一项核对其物理装配零面积极限，再接永久损伤状态、接触去重和 RK2 事件推进。

`differentiable_cohesive._closed_pullback` 对闭合线性约束及所选反力直接求固定秩导数，保留 κ=0 的历史单侧导数。`closed_adjoint_final_unit.log` 为 **38 passed**（16.76 s），全量 `closed_adjoint_full.log` 为 **1191 passed、8 skipped、18 subtests passed**（352.79 s）。`cohesive_area_analytic.json` 的四种状态、五档面积共 **20** 例符合解析解。受压总反力在 A→0 时保持有限，张开和卸载总力趋零；下一步在实际自然分侧状态下核对约束建立、零面积积分项和永久历史。

`reference_grid.build_trace` 的部分侧码映射按同节点相容场的参考质量加权，随后采用现有多项式延拓。CRAMP 来源及本地推导见 METHOD。`runs/grid_update/partial_trace_unit.log` 为 **18 passed**（61.01 s），全量 `partial_trace_full.log` 为 **1203 passed、8 skipped、18 subtests passed**（413.32 s）；`natural_surface_support.json` 的四种初态/刀角共 **31** 个接受段、**24** 个非空面、**1716** 个积分点全部建立映射；基线的十次缺失侧码错误已修复。

全共享场 Q=0 的内聚平衡已处理零自由度和冗余牵引，开口残差使用代数等价的牵引形式。`shared_solve_unit.log` 为 **65 passed**（52.30 s）；全量 `shared_solve_full.log` 为 **1223 passed、8 skipped、18 subtests passed**（366.31 s）。`shared_map_limit.json` 的映射梯度与解析解一致，四档差分最大误差 **1.38103e-13**；`shared_energy.json` 的十二例能量残差随步长减半约减半，四组最细档幅值 **0.017552%/0.002262%/0.007322%/0.348344%**。`shared_existing160.json` 的已有损伤八类输入均至少两档误差 ≤**2e-5**，最差第二档 **1.07941e-5**，共 **32** 行差分。

永久状态入口 `cohesive_history.CohesiveHistory` 的追加、材料推进、快照及分段反向已接好，方法来源及本地映射见 METHOD。`permanent_history_unit.log` 为 **7 passed**（12.75 s）；七类输入共 **14** 组自然差分通过。`permanent_history_measure.json` 的四组自然几何共 **31** 个接受段、**1716** 个积分点，旧点状态差 **0**；材料加载后的旧断裂能 **1.26e-6 J**，追加前后旧历史差及恢复后的粒子状态差均为 **0**。下一项将永久状态接入自然双侧码、零面积物理装配、接触去重及 RK2 分段。

提交前全量 `permanent_history_full.log` 为 **1230 passed、8 skipped、18 subtests passed**（380.51 s）。

闭合分支按参考面积选择最小压力范数反力，位移及其 VJP 保持同一约束；固定法向 ADMM 统一提供初值，初值牵引采用近端核的直接表达。`weighted_area_limit.json` 的 14 组正向、42 组单侧扰动全部完成，最小 1e-12 m 档相对误差 ≤**6.24702e-7**。`weighted_compare160.json` 三种轨迹与 `e4cf414` 求解器的归一化状态差 ≤**1.95528e-14**、能量差 ≤**4.337e-19 J**。新增 22 项解析及面积拆分检查；全量 `weighted_area_full.log` 为 **1252 passed、8 skipped、18 subtests passed**（366.51 s）。同一 `e5-grid` 工作树的 `weighted_existing160.json/.log` 已完成八类输入长梯度，每类至少两档通过，最差第二档误差 **1.07018e-6**。`weighted_energy.json/.log` 的十二例残差随步长减半约减半，最细档幅值 **0.017552%/0.002262%/0.007322%/0.348344%**。

`95eaf2d` 的零面积片被 κ=0 闭合投影当作完整约束：6/4 mm 的完全断裂后退刀例产生额外动能损失 **4.62607e-6/2.46949e-6 J**，报告 `runs/grid_update/zero_area_history_projection.json`。`cohesive_measure_proximal_response` 以积分总力为输出，局部零面积运动、正面积等价性及单侧面积导数共 **18** 项检查通过；连同原局部核为 **39 passed**（0.58 s），日志 `measure_proximal_unit.log`。下一项连接总力形式的全局平衡、闭合投影与自然出生事件，再接自然双侧码、接触去重及 RK2 分段。

提交前全量 `measure_proximal_full.log` 为 **1270 passed、8 skipped、18 subtests passed**（357.68 s）。

积分总力全局入口 `solve_force` 与 `solve_geometry(..., integrated_force=True)` 共用当前半光滑方程，含零面积、扩展精度残差细化和反力选择的 VJP。`force_global_unit7.log` 为 **132 passed**（74.98 s）；补充 **6** 例载荷尺度检查通过（1.80 s），日志 `force_scaled_constraint2.log`。`force_compare160.json` 比较共用代码更新前后的首阶轨迹，三种 160 步状态归一化差 ≤**6.006e-13**、能量差 ≤**8.882e-16 J**。`cohesive_grid_update.advance_material` 已连接总力接口与正面积闭合掩码；参考单元边界和精确裁剪已接入并保留面积导数；下一项连接自然双侧码、接触去重及 RK2 分段。

插入失败对照位于 `e5-interface-20261023/runs/interface_insertion/patch.json`：平移在 1.096e-23～4.386e-23 m 的间隙舍入处触发半光滑线搜索失败；刚体旋转 dt=2 μs 的假断裂耗散为 6.240e-6 J。该对照对应 `b7048c7`、dirty=true 的 `f507dc0` 源码。入口 `runs/e5_release_audit/extrinsic_insertion_patch.py`。

积分总力全局接口提交前全量 `force_global_final_full.log` 为 **1330 passed、8 skipped、18 subtests passed**（413.20 s）；新增 60 项检查。

总力材料装配的相关检查 `force_material_unit.log` 为 **41 passed**（107.00 s），新增 **5** 项零面积运动、状态梯度及面积单侧导数检查。`force_material_compare160.json` 比较 `016b255` 牵引装配与总力装配，闭合承载/已有损伤/受载断裂三种 160 步的归一化状态差 ≤**8.994e-15**、能量差 ≤**8.674e-19 J**、外功差 ≤**5.422e-20 J**。参考采样坐标与精确裁剪已接入，退刀新增零面积及实际材料运动检查通过；下一项连接自然切割的侧码和事件推进。

总力材料装配提交前全量 `force_material_full.log` 为 **1335 passed、8 skipped、18 subtests passed**（427.09 s）。E3 补充任务 `e3-current/runs/e3_ca90da9/after_queue.py` 已完成，退出码 **0**：4 mm、50 ms、G_c=7 的 dt/8 两档刀路差分通过；dt/16 的梯度与状态细化完成。状态文件 `after_queue_status.json`，日志 `release_h4_more.log`。

自然参考面采用 `ReferenceCoordinates.sampled` 与精确交点构造，材料入口使用 `model.reference_coordinates`；解析 VJP 与参考坐标公式见 METHOD。`exact_reference_unit2.log` 为 **59 passed**（82.01 s），`exact_reference_material2.log` 的实际退刀材料推进为 **1 passed**（1.16 s）。`exact_reference_material.json` 中旧 **6** 个正面积点保留，退刀新增 **45** 个点的面积全为 **0**；x/v/F/Fp/应力差 **0**，动能同为 **9.504e-6 J**。`reference_sampling_compare.json` 的 18 种配置与 `4e5a892` 逐位一致。下一项是自然多刀侧码、接触去重与 RK2 分段，然后复测完整能量扫描和受影响的 E1–E4。

精确参考面提交前全量 **1375 passed、8 skipped、18 subtests passed**（487.88 s），日志 `e5-grid-20261023/runs/grid_update/exact_reference_full.log`。

有限切面多刀映射入口为 `CohesiveHistory.reference_trace` 与 `advance(...)`；`append(..., tool_rotation=...)` 保存出生正法向侧码，也支持显式侧码参数。`visibility_unit.log` 为 **36 passed**（93.80 s）。`visibility_natural.json` 的完整/部分交叉扫掠为 **612/462** 积分点，完整交叉与解析分区的权重差 **0**、三线性重现误差 ≤**8.882e-16**。`visibility_energy.json` 的两组 32 μs、12 点轨迹，2/1/0.5 μs 能量残差约随步长减半，最细档相对残差 **0.00072282%/0.075308%**。上述报告在 `e5-grid-20261023/runs/grid_update/`。下一项接自然事件、接触去重及 RK2；有限面判定的时间见报告，调度/存储优化单独提交。

四步双切面材料轨迹的八类输入各至少两档自然差分通过 **2e-5** 标准，最差第二档 **4.647e-6**；48 组原始数据及可见性分支对照为 `visibility_gradient.json`、`visibility_branches.json`。提交前全量 **1383 passed、8 skipped、18 subtests passed**（502.67 s），日志 `visibility_full_final.log`。精确判定的区间过滤原型为 `runs/e5_release_audit/visibility_filtered.py`，`visibility_filter.json` 的完整/部分交叉及三种尺度随机三角片共五例与精确判定的可见性差均为 **0**；完整/部分交叉耗时由 **8.960/4.746 s** 降至 **1.346/0.719 s**。继续核对退化/窄片/大坐标，再单独整理性能提交。

区间筛选已整理到 `_interval_predicates.py`，材料映射公式保持原值。`visibility_filter_unit.log` 为 **15 passed**（3.65 s），包含退化、窄片、大坐标和独立精确重心求交；`visibility_filter_source.json` 的完整/部分交叉耗时 **8.556→1.294 s / 4.735→0.7021 s**，五例判定差均为 **0**。提交前全量 **1390 passed、8 skipped、18 subtests passed**（485.87 s），日志 `visibility_filter_full.log`。切面出生朝向已保存，下一项接入自然事件、接触去重和 RK2。


出生朝向由 `n_ref·F_end⁻¹R_end e_x` 的符号确定并随历史保存。`birth_orientation_unit.log` 为 **31 passed**（23.14 s），其中新增六项；全量 `birth_orientation_full.log` 为 **1396 passed、8 skipped、18 subtests passed**（489.97 s）。3.2 ms 的两种采样、三种刀路共六组自然事件几何观察全部通过，累计 **196** 个接受段，末端分别为 **2952/2292/2292/2637/471/471** 个积分点；三线性重现误差 ≤**1.333e-15**、恢复权重差 **0**。脚本 `runs/e5_release_audit/birth_orientation_natural16_fixed.py`，报告 `runs/grid_update/birth_orientation_natural16_fixed.json`。这些观察沿用原自然切割力学；内聚力、接触去重和 RK2 的联合推进继续接入。

闭合运动按矩形约束矩阵直接求解，保留节点增量的因式表示；伸出名义网格的参考面在边界单元多项式上按实际坐标取值。通用非线性入口为 `_cohesive_initial.fixed_normal_initial` 和 `_cohesive_linear.solve_blocks`，由 `_cohesive_solve` 调用；伴随通过完整 Warp 残差细化。新增 **42** 项初值/分块解析检查通过，相关 **194 passed**（97.09 s），全量 **1483 passed、8 skipped、18 subtests passed**（519.75 s），日志 `runs/grid_update/adaptive_general_full.log`。

576 点自然界面从零初值完成正式材料一步推进，正向/反向 **109.7196/7.3879 s**，`natural_generic_material.json` 的两档初速度差分误差 ≤**3.674e-14**。`natural_generic_validate.json` 的固定参考几何自由开口/强度各两档差分通过，最大误差 **4.8591e-8**；`adaptive_general_energy.json` 的十二例材料—界面能量残差随步长减半约减半，最细档 **0.017552%/0.002262%/0.007322%/0.348344%**。报告在 `e5-grid-20261023/runs/grid_update/`。下一项将界面力连接自然分段、接触去重及永久历史，再完成整体能量扫描和受影响的 E1–E4 复验；完整自然切割当前仍使用原释放律。

E5 界面冲量入口 `cohesive_impact` 的 **20** 项检查通过，涵盖同时碰撞、重复约束、行缩放、移动法向及输入导数。平面碰撞的四种时间划分均得到 **115 μs** 事件时刻和解析动能损失 **1.333333e-5 J**；五类输入共十组差分最大相对误差 **2.539e-7**。分阶段材料入口与验收结果见下段，完整自然碰撞和整体能量扫描继续接入。

`reference_affine` 已提供仿射速度传递、独立速度梯度编码、动能及虚功配对；6/4 mm 方块/圆柱与扰动采样的 **13** 项检查通过，共享场的开口映射及切面位置导数严格为零。参考仿射分阶段原型在 32 μs、dt=2/1/0.5 μs 的纯旋转和预载旋转中保持零开口、零损伤；受载开裂的相邻速度差随步长减半约缩小四倍。三类八步轨迹的 **58** 组差分中非零导数最大相对误差 **3.669e-6**；切面位置、法向、平移/仿射速度及面积到八步材料状态的十组差分最大误差 **7.090e-7**。自然事件生成的 288/576/1152 点参考界面在八步旋转检查中均保持零开口、零损伤。自然切割中的接触和出生事件继续连接。

`cohesive_stages.advance_affine` 已接入独立仿射状态、两阶段材料/内聚求解和永久损伤；`CohesiveHistory.advance` 支持该状态。**11** 项检查通过，覆盖自由落体查询中点与时长导数、逐位重启、分段反向、出生侧码、仿射动能和活跃塑性。两类八步轨迹 **44** 组输入差分及塑性 **6** 组差分通过；公开入口 **15** 例能量/步长检查完成。提交前全量 **1527 passed、8 skipped、18 subtests passed**（568.50 s）。固定率耗散已接入；下一项连接刀/板与界面闭合接触、在线切面出生和自然事件重算。

`reference_damping.decay` 的固定率耗散已接入正式分阶段推进，`damping_energy` 计入两半步完整动能变化。六种网格/形状的 **48** 组差分最大误差 **8.691e-8**，八步联合推进的速度/仿射矩/时长/速率 **8** 组差分通过；零率导数、大衰减解析解、重启和分段反向通过。正式入口 **18** 例能量/步长检查完成，随机速度例耗散约 **9.622e-7 J**、能量相对残差 ≤**2.05e-12**。刀/板接触及自然切面出生继续连接。

闭合界面非均匀速度检查的开口/损伤为零，但耗散后的闭合速度残差在 dt=2/1/0.5 μs 为 **1.6626e-5/8.3128e-6/4.1563e-6 m/s**。下一步优先核对约束子空间中的耗散生成算子和固定秩反向，再连接刀/板接触；当前分裂的差分正确性与此速度约束误差分别记录。报告 `runs/grid_update/damping_closed_probe.json`。

本批全量为 **1540 passed、8 skipped、18 subtests passed**（603.60 s），日志 `runs/grid_update/damping_full.log`。闭合子空间耗散原型 `damping_constrained_probe.py` 将开口速度残差降至 ≤**4.99e-16 m/s**，三档能量残差为 **7.6074e-11/1.9002e-11/4.7483e-12**。`constrained_damping_adjoint.py` 的 **32** 组差分最大误差 **7.106e-8**，与 SciPy Fréchet 方向导数差 ≤**7.801e-15**；原型和报告均在 runs 下。下一项正式化一般三维约束下的指数作用量与固定秩 VJP，覆盖零/大时间率积、独立/重复约束和联合材料差分，再接刀/板接触及在线切面出生。
