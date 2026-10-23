# 结果

本页列出 BladeMaster（arXiv 2609.27342）各实验的复现数字，以及几种 MPM 的对比。待办见 [TODO.md](../TODO.md)；公式对照见 [METHOD.md](METHOD.md)；参数来源见 [PARAMETERS.md](PARAMETERS.md)。2026-10-20 之前的完整过程记录在 tag `archive/before-cleanup-20261020` 的同名文件里。

论文没有给出的量，在各节写明是本地取值。所有命令都在仓库根目录运行。

## Fig. 1：香蕉三刀

```bash
python -m blademaster_warp.franka_run --motion sequence --controller libfranka --stiffness 1000 \
  --material paper-banana --duration 30 --surface --surface-embedding cuda --surface-ccd cuda --record --output <目录>
```

- 设置：6 mm 网格，1452 个粒子，Franka 阻抗控制。
- 结果：三刀切成四块（778/294/236/144），退刀后保持。
- 刀与材料：峰值材料反力 4.80 N，最大跟踪误差 5.3 mm，GPU 错误为 0。

![Fig. 1](media/fig1_sequence.png)

## Fig. 3 / Table I：重力落刀与锋利度

```bash
python -m blademaster_warp.paper_freefall --output <目录> --duration 1 --sharpness <0.9|0.5|0.2> [--surface] [--no-settle]
```

设置：8960 个粒子，1 cm 网格，2 ms 步长，种子 1。

起始状态（2026-10-21 起默认）：先在重力下求静平衡，刀扶在释放位置不动，再松刀。平衡用动能阻尼（每当动能过峰就把速度清零，没有阻尼系数），收敛判据是从静止走一步、任何粒子的加速度都小于 1% g。0.314 s 收敛，残差 0.72% g。之前从未受力的形状起步，果冻在重力下下沉约 7 mm 后以 ±5 mm、约 0.14 s 周期振荡，几乎不衰减，所以视频一开始就晃。

| 锋利度 | 静平衡起步（`673ff77`） | 未受力起步（`d663eb6`，`--no-settle`） |
|---|---|---|
| 0.9 | 刀到板（刃口离板 0.08 mm），但仍是一整块 | 切成两块 4308/4652，刃口离板 0.105 mm |
| 0.2 | 一整块，刃口停在离板 14.7 mm | 一整块，刃口离板 26.0 mm |

- 钝刀不切透，与论文一致；静平衡起步后停得更接近论文 Fig. 3(d) 的中间深度。
- 锋利刀在静平衡起步下没有切成两块。连着两半的是顶面角上的 2 个粒子（参考位置 y ≈ −48 mm、z ≈ 80 mm）：刃口扫过时它们离刀面 3.1 mm 和 7.7 mm，没有被标上侧码，于是两侧经它们连着。之前的两块结果依赖起步时果冻还在下沉振荡。待查：切割带在块角处的横向宽度与刀在下落中的转动（26.7°）。
- 网格因素仍在：1 cm 网格下粒子间距 5 mm，远大于 1 mm 刀厚，被挡住的粒子横移几毫米就绕到刃后。

机制扫描（每次只改一项，`runs/mechanism_scan_20261019`）：

| 改动 | 结果 |
|---|---|
| 关掉刃前计功带 | 锋利度 0.9 也不再付损伤 |
| 关掉触板底层标签 | 刀到板，但两半仍连着 |
| 打开绕刃 | 刀到板，但仍是一块 |

![Fig. 3](media/fig3_freefall.png)

## Fig. 4：中片抬升，多场与单场

```bash
python -m blademaster_warp.scenarios --scenario fig4 [--single-field]
python -m scripts.fig4_compare <多场目录> <单场目录> --output <图>
```

- 设置：
  - 圆柱 Ø160 × 56 mm，1 cm 网格，8932 粒子；E = 10 kPa，ν = 0.4。
  - 两刀径向切出 120° 扇形片，刀放平从片下滑入，再抬 40 mm。
  - G_c = 10 J/m² 是本地取值。
- 单场对照只在粒子与网格之间传递时忽略切割码，分块与多场完全相同（5940/2992）。
- 刀面网格约束（`blade_constraint.py`，论文没有）假设刀两面的材料在不同的场里。单场时两面共用网格节点，这条约束收不到 1 μm/s，所以 `--single-field` 会关掉它。
  - 多场开着和关掉（`--no-blade-grid-constraint`）这条约束，结果逐位相同。

| 方法 | 片抬升 |
|---|---|
| BladeMaster 多场 | 37.7 mm |
| 单场 TLMPM | 最高 4.3 mm，随后刀从片中穿出，最终 −0.6 mm |

（2026-10-20，提交 `32a6d4b`，`runs/fig4_constraint_20261020`。抬升取片上粒子高度相对 t = 0 的中位数，包含重力下的沉降。）

![Fig. 4](media/fig4_slice_lift.png)

## Fig. 5：两刀三块，推左块

```bash
python -m scripts.fig5_compare run --backend blademaster --output <目录>
python -m scripts.fig5_compare build-cressim --source <CRESSim-MPM 09aa5009> --output <构建目录>
python -m scripts.fig5_compare run --backend cressim_cpu --cressim-build <构建目录> --output <目录>
python -m scripts.fig5_compare compare <bm目录> <cressim目录> --output <图>
```

| | 半深切口开口 | 中块体积变化 |
|---|---|---|
| 论文 BladeMaster | 0.654 mm | 0.506% |
| 本仓库 BladeMaster | 0.706 mm | 0.924% |
| 论文 CRESSim | 4.059 mm | 8.746% |
| 本仓库 CRESSim（上游未改，62.5 / 31.25 μs） | 0.257 / 0.102 mm | 中块有非正四面体，体积无效 |

左块的滑行：

- 第一刀落到板附近时（1.068 s），左块得到 −10.5 mm/s 的水平速度，之后整块滑行。
- 推动阶段扣除这份滑行后，三块位移都在 10⁻⁶ m 量级，所以推动本身不带动其余两块。
- 滑行的来源：
  - 底排粒子标上侧码后，网格交换停止，这一步的刀冲量全部留给左块。
  - 相邻粒子列的参考位置已伸进接触壳 0.25 mm。
- 接触刚度固定、步长减半后，滑行仍在（−5.96 → −6.28 mm/s）。
- 罚函数没有改。

CRESSim 基线（上游提交 `09aa5009`，MLS 头、求解器、本构头与上游逐字节相同）。2026-10-20 在 4060 的 CPU 上用提交 `3c3d380` 重新编译，四个步长各跑完整刀路（`~/Projects/CRESSim-MPM-runs/dt_scan_20261020`）：

| 步长 | 用时 | 半深开口 | 中块体积变化 | 左块（初速 −0.03 m/s）实际位移 |
|---|---|---|---|---|
| 62.5 μs | 245 s | −0.003 mm | −1.41% | 0.05 mm |
| 31.25 μs | 493 s | 0.000 mm | −0.55% | 0.05 mm 量级 |
| 15.625 μs | 974 s | 0.000 mm | +0.15% | 0.05 mm 量级 |
| 7.8125 μs | 1949 s | 0.000 mm | +0.34% | 0.05 mm 量级 |

相邻两档步长的最终位置 RMS：

| 比较 | 最终 RMS | 全程最大 RMS |
|---|---|---|
| 62.5 vs 31.25 μs | 0.78 mm | 0.96 mm |
| 31.25 vs 15.625 μs | 1.08 mm | 1.10 mm |
| 15.625 vs 7.8125 μs | 1.15 mm | 1.15 mm |

- 步长减半时差别不缩小，所以没有随步长收敛；但也没有塌缩，块高从 27.5 mm 变到 28.0–28.7 mm。
- 四个步长下切口都没有打开。左块的初速按匀速应走 12 mm，实际只动了约 0.05 mm，被另外两块拉住。
- 原因在 CRESSim 自己的切割方式。夹具用的是它的 `QuadSlicer`：
  - 粒子与网格传递时，若粒子和节点分在刀片平面两侧，就不传速度（`checkOnDifferentSides`，`mls_mpm_algorithm.h`）。
  - 这个判断只看刀片当前的位置，不记住已经切过的地方。刀片抽走后两侧重新共用节点，又连在一起。
  - BladeMaster 用持久的切割码记住切口，这正是两者的差别。
- 论文里的 CRESSim 切口是 4.059 mm、体积变化 8.746%，我们在半深处测到的开口约为 0。刀片半厚 0.25 mm，离切面最近的粒子列在 1.25 mm 外，没有被推开。论文给 CRESSim 用的参数或测量定义可能和这里不同，还没对上。
- 10-19 记录的 62.5 vs 31.25 μs RMS 是 7.36 mm，那次运行的目录已经不在了。这次用同一份源码重跑没有复现，原因还没查。

## Fig. 6：三把飞刀

```bash
python -m blademaster_warp.paper_impact --output <目录> --duration 1 [--launch end|side|top] [--cut-hold]
```

默认 `--launch end`（2026-10-21 起）：按项目页视频，三把刀平放、并排相距 15 mm，刀尖朝前沿块的长轴射入 50 mm 宽的端面。刀尖用 10 mm 宽的前刃表示，刀长 60 mm，刀重 20 g（论文没有给，是本地取值）。块先在重力下求静平衡，方法同 Fig. 3（0.06 s 收敛，残差 0.96% g）。

1 s 时的切深（块长 100 mm，提交 `673ff77`）：

| 初速 | 0.5 m/s | 0.8 m/s | 1.1 m/s | 块 |
|---|---|---|---|---|
| 端面射入，论文式 (5) | 29.9 mm（0.10 s 停） | 49.3 mm（0.65 s 停） | 68.8 mm（0.57 s 停） | 一整块 |

三把刀都停在块里，初速越大越深，与论文一致。

之前的侧面射入（`--launch side`）：刀面水平，前刃 24 mm 宽从前侧面射入，三把刀相距 30 mm，未受力起步。1 s 时的切深（块深 50 mm，提交 `d663eb6`，`runs/fig6_release_20261020`）：

| 实现 | 0.5 m/s | 0.8 m/s | 1.1 m/s | 块 |
|---|---|---|---|---|
| 论文式 (5) | 22.2 mm | 36.8 mm | 51.4 mm | 一整块 |
| `cut_hold` | 14.0 mm | 28.8 mm | 41.5 mm | 一整块 |

- 侧面射入时三把刀也都被减速、初速越大越深。
- 论文式 (5) 下，0.5 m/s 的刀到 1 s 还有 16 mm/s，没有完全停下；最快的刀刚到块背面。
- 每把自由刀只有在本刀、本次切割付清断裂阈值后才打开切口。在这条规则修正之前（各刀共用刀 0 的标记，换刀号也不清零），切深是 20.0/33.8/51.5 mm 和 13.9/26.6/41.7 mm。
- G_c 从 30 到 1000 J/m² 切深都不变：这块材料的阻力来自塑性流动和刀面摩擦，不是切割能。

![Fig. 6](media/fig6_side.png)

## Fig. 7 / Fig. 8：刀尖划痕与连续薄切

```bash
python -m blademaster_warp.scenarios --scenario fig7-shallow|fig7-deep|fig8 --surface --surface-embedding cuda --surface-ccd cuda
```

Fig. 7：

- 设置：块 80 × 80 × 70 mm，3584 个粒子，E = 6 kPa。刀倾斜 30°，十字划两刀。
- 浅划（6 mm）只切开顶层，表面是细痕。
- 深划（20 mm）切开 4 层，两刀在中心相交。
- 两次都不切透。

Fig. 8：

- 设置：块 100 × 50 × 42 mm，7854 个粒子。每隔 12 mm 切一刀，共 4 刀。
- 结果：4 片各 952 个粒子，最终倾角 90°/84°/61°/42°，后两片斜靠在前一片上。

Franka 版本（`franka_run --motion fig7-shallow|fig7-deep|fig8`）结果相同，都通过任务判据。

![Fig. 7](media/fig7_cross_scores.png)
![Fig. 8](media/fig8_sequential_slicing.png)

## 论文任务视频

渲染只改显示，不改记录的粒子、表面和刀位。

- 视角：
  - 机械臂任务用网页遥操作界面打开时的默认相机（`web/app.js`），即从机械臂正前方顺着它的伸出方向往回看。左边是整机，右边是近景表面和粒子。
  - 无机械臂的 Fig. 3 / 5 / 6 用同一个近景方向和俯角，距离按场景外接球取。Fig. 6 再朝刀射入的一侧转一半，否则刀被块挡住。
  - 灯光随相机方位一起转，另加一盏头灯，让朝向相机的侧面也照得到。
- 颜色从项目页视频取：
  - 香蕉：奶油色，配绿板。
  - 豆腐：米白。
  - 午餐肉：粉色，带按参考坐标生成的斑纹，切面上也有。
  - 抹茶蛋糕：按参考高度 60% 分两层，上深绿、下浅绿。
  - 果冻：深红半透明。
  - Fig. 5：奶油色粒子。
  - 颜色表在 `material_colors.py`，离线渲染和网页共用。
- 粒子按切开的块着色，和网页一样：
  - 块由保存的侧码算出（`topology.components`），块号跨刀保持（`topology.StablePieces`）；
  - 未切开时整块是材料色；每切下一块，新块换一种颜色，原块保留材料色；
  - 粒子面板角上显示块数。
  - 表面保持材料色。Fig. 5 的 CRESSim 没有侧码，两边都还是奶油色。

| 视频 | 运行 | 显示 |
|---|---|---|
| Fig. 1 香蕉三刀 | `franka_run --motion sequence --material paper-banana --duration 30` | 表面 + 粒子 |
| Fig. 7 豆腐浅划 / 深划 | `franka_run --motion fig7-shallow` / `fig7-deep` | 表面 + 粒子 |
| Fig. 8 午餐肉薄切 | `franka_run --motion fig8` | 表面 + 粒子 |
| Fig. 4 抹茶蛋糕多场 / 单场 | `franka_run --motion slice-lift [--single-field]` | 表面 + 粒子 |
| Fig. 3 锋利 / 钝刀 | `paper_freefall --sharpness 0.9 / 0.2 --surface` | 半透明表面 |
| Fig. 6 三把飞刀 | `paper_impact --duration 1` | 粒子 |
| Fig. 5 BladeMaster 对 CRESSim | `scripts.fig5_compare run`，两个后端 | 粒子 |

Franka 运行都加 `--controller libfranka --stiffness 1000 --record --surface --surface-embedding cuda --surface-ccd cuda`，Fig. 4/7/8 再加 `--attachment fixed --layout paper`。渲染命令：

- `python -m blademaster_warp.render_franka <运行>`
- `python -m blademaster_warp.render_free fig3|fig6 <运行>`
- `python -m blademaster_warp.render_free fig5 <BM 运行> <CRESSim 运行>`

`franka_run --single-field` 是 Fig. 4 的单场对照：刀不在切时，粒子与网格的传递忽略侧码，规则与 `scenarios --single-field` 相同。

Fig. 3 和 Fig. 6 有自由刀，表面记录器只在求解器放开切口（`cut_released`）后才沿刃口切开显示表面。Fig. 6 是三把刀，表面记录器目前只接一把，所以还只画粒子。

## MPM 对比：BladeMaster 与 TopoCut 式 MLS-MPM

```bash
python -m scripts.topocut_compare --scenario fig4|fig8 --output <目录> [--dt 0.00025] --damaged-shear 0.1
```

实现与设置：

- TopoCut 式基线在 Warp 里重写，`mlsmpm.py` + `topocut_topology.py`：
  - 单速度场 MLS-MPM；
  - 按体积变化标损伤；
  - 每刀后用 SDF 和 Marching Cubes 分块；
  - 用 Laplace–Beltrami 谱奖励评价。
- 两种求解器共用粒子、材料、步长、重力和接触，只差场的表示和分离方式。
- 损伤后剪切模量的倍数论文没有给，0、0.1、1 三种都跑了。

Fig. 4 片抬升（满分谱奖励 2）：

| 步长 | 求解器 | 片抬升 | 其余最大 | 谱奖励 |
|---|---|---|---|---|
| 2 ms（论文） | BladeMaster | 38.9 mm | 0.8 mm | 1.98 |
| 2 ms | MLS-MPM ×0 | 发散 | | |
| 2 ms | MLS-MPM ×0.1 | 18.4 mm | 9.4 mm | 1.83 |
| 2 ms | MLS-MPM ×1 | 0.8 mm | 1.0 mm | 1.89 |
| 0.25 ms | BladeMaster | 38.4 mm | 0.4 mm | 1.98 |
| 0.25 ms | MLS-MPM ×0.1 | 0.0 mm | 0.5 mm | 1.91 |

Fig. 8 连续薄切（满分 5）：

| 步长 | 求解器 | 分块 | 谱奖励 |
|---|---|---|---|
| 1 ms（论文） | BladeMaster | 4046 + 4×952 | 3.0 |
| 1 ms | MLS-MPM（三种倍数） | 只受重力就在 0.33 s 发散 | |
| 0.25 ms | BladeMaster | 4046 + 4×952 | 4.6 |
| 0.25 ms | MLS-MPM ×0.1 / ×1 | 2380/2380/2142/952 | −72 / −67 |

结论：

- 论文步长下 MLS-MPM 不稳定，BladeMaster 在 1–2 ms 下稳定。
- 单速度场下，切开的两侧仍共享网格节点，切片和主体粘在一起：
  - Fig. 8 推片时整块跟着动，后续刀落错位置；
  - Fig. 4 抬片时拖动周围材料。
- 两种方法分块纯度都 ≥ 0.995：分块由刀路决定，差别在力学上能不能分开。

![Fig. 8，0.25 ms](media/topocut_fig8_dt025.png)

## BladeMaster 力的收敛

```bash
python -m scripts.force_convergence --spacings .0025 .002 .0015 --seeds 20 \
  --blade-contact-radius 0 --output runs/force_particle_radius
python -m scripts.force_convergence --spacings .0025 .002 .0015 --seeds 5 \
  --blade-contact-radius .0005 --output runs/force_fixed_radius
```

以下 20 种子结果来自 `runs/force_seed_scan_20261019/rows.jsonl`（2026-10-19，commit `51ac3ee`，dirty=true）。所有组都切成两块，GPU 错误为 0。

| 网格 | 稳态切割力（20 个种子） | 变异系数 |
|---|---|---|
| 2.5 mm | 0.818 ± 0.020 N | 2.5% |
| 2 mm | 0.625 ± 0.015 N | 2.5% |
| 1.5 mm | 0.453 ± 0.009 N | 2.0% |

- 步长：每种网格步长减半，压入力、切割力和体积比的变化都在 3% 以内。
  - 1.5 mm 网格：压入力 1.385 / 1.383 N，切割力 0.452 / 0.456 N。
- 网格：没有收敛。2 → 1.5 mm 时切割力再降约 28%。
- 6 mm 交互网格：刀位横移 0 → 1.5 mm 时，峰值力从 0.81 N 升到 8.07 N。
  - 8.07 N 里 7.90 N 来自刃口法向。
  - 这是一整列粒子同时进入刃口接触壳造成的，属于粗网格离散效应。

![力收敛](media/force_convergence.png)

### 阈值灵敏度

2026-10-21，commit `79d176e`，RTX 4090；2.5 mm、seed=1、粒子半径接触。\(G_c=700/1400\) 时，0.4–1.4 s 平均竖直力为 **0.83729 / 0.83694 N**，相差 0.041%；达损伤阈值粒子数均为 0，两次均分成两块。该工况下平台力对这次阈值加倍变化很小。

```bash
PYTHONPATH=. .venv/bin/python - <<'PY'
from dataclasses import replace
from scripts import force_convergence as study
preset = study.banana_config
for energy in (700.0, 1400.0):
    study.banana_config = lambda: replace(preset(), cutting_energy=energy)
    run = study.cut_trace(.0025, 1, 0.0)
    force = study.summarize([run])["by_spacing"]["2.5 mm"]["steady_force_n"]["mean"]
    print(energy, force, run["damaged_particles"], run["pieces"], run["revision"])
PY
```

同日核对两组旧记录：默认半径 60 次、固定半径 15 次，均为 `51ac3ee` dirty=true。72 次达阈值粒子数为 0，3 次为 1；75 次均分成两块。下一步同时测方向计功、损伤事件及力的阈值响应，再建立输入功、储能与耗散的分项统计。

## 可微 TLMPM

2026-10-21，提交 `1857020`（dirty=false），RTX 4090，Warp 1.18.0。24×18×18 mm 弹性块，6/4 mm 网格，288/972 粒子，步长 0.2 ms：6 组弹性实验（1/12/32 步），4 组接触实验（8 步，切向速度 0.01/2 m/s）。

```bash
PYTHONPATH=. .venv/bin/python -m scripts.diff_validate --output runs/diff_clean
```

- AD 对各档有限差分的最大相对误差：E 梯度 **0.201%**；刀位梯度 **0.00367%**。分段重算与完整反传的 E 梯度差 **0.000797%**。
- 与原求解器的最大正向差：位置 **7.46e-9 m**、速度 **5.07e-7 m/s**、F **2.39e-7**、刀力 **4.58e-7 N**。
- 27 项可微测试通过：应力的单位阵/重复伸长/剪切/旋转导数，初始速度与 F 方向导数，两个网格的正向结果，分段长度 1/7/32 的 VJP，粘着与滑动接触。
- 4090 工作目录 `~/Projects/BladeMaster-Warp-diff-20261021`，报告 `runs/diff_clean/report.json`。

### 长轨迹与接触切换

2026-10-21，提交 `32f0378`（dirty=false），RTX 4090。两个网格、32/128/512 步自由弹性运动（重力为零），22 组完整/分段反传：

```bash
PYTHONPATH=. .venv/bin/python -m scripts.diff_horizon --output runs/diff_32f0378_horizon
PYTHONPATH=. .venv/bin/python -m scripts.diff_contact_scan --output runs/diff_32f0378_contact
```

- AD 对两档有限差分的最大相对误差：E **0.812%**、初始速度方向 **0.0137%**、初始 F 方向 **0.0419%**。检查点对完整反传：E **0.00171%**，初始 F 的相对 L2 差 **0.00152%**。
- 972 粒子、512 步（102.4 ms）：完整 Tape 峰值 CUDA 数组内存 **112.34 MiB**，前向 **0.1255 s**、反向 **0.0546 s**；32 步分段为 **9.21 MiB**，前向 **0.1076 s**、重算 **0.1306 s**、反向 **0.0637 s**。
- 接触核的 116 档差分中，88 档保持同一分支，非零导数最大相对误差 **0.0475%**；零导数最大绝对误差 **4.81e-11**。其余 28 档跨接触或粘滑切换。零历史、压入 0.3 mm 时，临界切向速度 **1.00623 m/s**，静摩擦转动摩擦的切向力下降 **33.33%**。

### 推压刀路优化

2026-10-21，提交 `32f0378`（dirty=false），RTX 4090。288 粒子、128 步（25.6 ms）、四段刀速，目标为左侧四分之一粒子的末端位置；各方法共用五组初始刀路，初始 RMSE 为 **2.04–2.72 mm**。每组预算 20 s，成功条件为位置 RMSE ≤ 0.05 mm、刀力 ≤ 2 N，完整设置见 DIFFERENTIABLE。

```bash
PYTHONPATH=. .venv/bin/python -m scripts.diff_optimize --output runs/diff_32f0378_optimize
```

| 方法 | 成功数 | 最终 RMSE 中位数 | 达标时间中位数 | 前向次数中位数 | 峰值 CUDA 数组内存 |
|---|---|---|---|---|---|
| AD + Adam | 5/5 | 0.03681 mm | 8.35 s | 258 | 10.64 MiB |
| FD + Adam | 2/5 | 0.05142 mm | >20 s | 746 | 0.157 MiB |
| CMA-ES | 4/5 | 0.04915 mm | 8.23 s | 745 | 0.157 MiB |

达标时间中位数将未成功样本记为 >20 s。AD 每次评估的前向/反向中位耗时约 **48.2/22.4 ms**；无梯度前向约 **26.7 ms**。15 条最优刀路的最大刀力 **0.392 N**；原求解器重放的最大位置差 **0.486 μm**。各初始刀路的四档差分与接触分支变化保存在报告 `initial_gradient_check`；接触与摩擦公式保持原值。

### 接触轨迹的分段重算

2026-10-21，提交 `023d30d`（dirty=false），RTX 4090。6/4 mm 网格、128/512 步、8/16 段刀速，完整 Tape 与 32/128 步分段共 24 组，每组独立进程预热后测三次：

```bash
PYTHONPATH=. .venv/bin/python -m scripts.diff_contact_horizon --output runs/diff_023d30d_contact_horizon
```

32/128 步分段对完整 Tape 的控制量梯度最大相对 L2 差为 **0.0890% / 0.1327%**；完整 Tape 三次重复的最大梯度差为 **0.1371%**。末端位置最大差 **0.124 μm**。分段边界传递 x、v、F、刀和砧板的摩擦历史；逐步反力罚项与末端位置损失均参与反传。

972 粒子、512 步、16 个控制量，以下为三次耗时中位数；显存是 CUDA 数组内存池实际使用量峰值：

| 保存方式 | 显存 | 前向 | 重算 | 反向 | 整次梯度评估 |
|---|---|---|---|---|---|
| 完整 Tape | 141.06 MiB | 186.5 ms | 0 | 90.1 ms | 308.7 ms |
| 每 32 步分段 | 11.88 MiB | 170.0 ms | 182.3 ms | 104.9 ms | 474.9 ms |
| 每 128 步分段 | 36.37 MiB | 168.7 ms | 183.7 ms | 104.1 ms | 475.0 ms |

32 步分段节省 **91.6%** 数组显存，整次梯度评估耗时增加 **53.8%**。整次耗时包含数组构造、读回、清理及 Python 调用；分阶段列分别取中位数。

### 八控制量推压

2026-10-21，提交 `023d30d`（dirty=false），RTX 4090。6 mm、288 粒子、128 步、八段刀速；AD 每 32 步分段重算，FD/CMA-ES 使用无梯度前向。每种方法相同五组初始刀路、20 s 预算，位置成功阈值 0.05 mm，力上限 2 N。初始位置 RMSE **2.136–2.496 mm**。

```bash
PYTHONPATH=. .venv/bin/python -m scripts.diff_optimize --controls 8 --segment-steps 32 --output runs/diff_023d30d_optimize_q8
```

| 方法 | 成功数 | 最终 RMSE 中位数 | 达标时间中位数 | 前向次数中位数 | 峰值 CUDA 数组内存 |
|---|---|---|---|---|---|
| AD + Adam，32 步分段 | 5/5 | 0.02988 mm | 5.69 s | 172 | 2.99 MiB |
| FD + Adam | 2/5 | 0.05040 mm | >20 s | 749 | 0.157 MiB |
| CMA-ES | 5/5 | 0.01500 mm | 6.66 s | 740 | 0.157 MiB |

15 条最优刀路的最大刀力 **0.405 N**，原求解器重放最大位置差 **0.00559 μm**，错误计数均为零。此任务中 AD 与 CMA-ES 成功率相同，CMA-ES 最终误差中位数更低。

### 十六控制量、细网格与后移刀位

2026-10-21，提交 `023d30d`（dirty=false），RTX 4090。4 mm、972 粒子、512 步（102.4 ms）、16 段刀速，初始刀位沿 x 后移 0.3 mm；各方法同五组初始化、20 s 预算。初始位置 RMSE **8.299–9.593 mm**，成功条件仍为位置 RMSE ≤ 0.05 mm、刀力 ≤ 2 N。

```bash
PYTHONPATH=. .venv/bin/python -m scripts.diff_optimize --controls 16 --steps 512 --spacing 0.004 --tool-offset -0.0003 0 0 --segment-steps 32 --output runs/diff_023d30d_optimize_q16
PYTHONPATH=. .venv/bin/python -m scripts.diff_optimize --methods ad --controls 16 --steps 512 --spacing 0.004 --tool-offset -0.0003 0 0 --output runs/diff_023d30d_optimize_q16_full_ad
```

| 方法 | 成功数 | 最终 RMSE 中位数 | 前向次数中位数 | 峰值 CUDA 数组内存 |
|---|---|---|---|---|
| AD + Adam，完整 Tape | 0/5 | 0.10020 mm | 62 | 141.06 MiB |
| AD + Adam，32 步分段 | 0/5 | 0.17009 mm | 43 | 11.88 MiB |
| FD + Adam | 0/5 | 5.29368 mm | 188 | 0.529 MiB |
| CMA-ES | 1/5 | 0.12350 mm | 188 | 0.529 MiB |

这组任务中，AD 与 FD 均未达到位置阈值。完整 Tape 比分段 AD 多完成约 **44%** 的梯度评估，最终误差中位数低 **41.1%**。20 条最优刀路的最大刀力 **0.302 N**，原求解器重放最大位置差 **0.639 μm**，错误计数均为零。

分段 AD/FD/CMA-ES 的初始刀路共 960 档参数差分，**956 档改变接触/摩擦模式**；五组分段 AD 的梯度与中心差分相对差中位数为 **1.26%–37.36%**，分母取两者绝对值的较大者。单侧差分与模式变化计数保存在报告。

## 计时

Table I 场景的本地计时使用 RTX 4060，只计材料推进，不含表面；论文使用 RTX 5090。

| 场景 | 本地粒子 | 论文粒子 | 本地 wall/sim |
|---|---|---|---|
| Fig. 3 重力刀 | 8960 | 8960 | 0.42 |
| Fig. 4 片抬升 | 8932 | 8960 | 0.174 |
| Fig. 7 划痕 | 3584 | 3584 | 0.158 |
| Fig. 8 薄切 | 7854 | 7922 | 0.232 |
| Fig. 6 多刀 | 7854 | 7922 | 0.55 |

单位为墙钟时间 / 仿真时间。下一次与论文比较性能时，先对齐 Fig. 4/6/8 的粒子数并记录步长与完整配置。

Fig. 1 三刀，含 Franka、表面和录制（2026-10-20，提交 `8fad2c5`，RTX 4060）：

- 仿真 21.04 s，墙钟 35.54 s，0.59×。
- 材料 24.8 s，表面 5.2 s。

材料步本身已长于仿真时间，单靠加快表面到不了 1×。4090 上每 1 ms 仿真的 16 个子步要约 1.2 ms。

## 固定切割分支的可微 TLMPM

2026-10-21，提交 `26fff83`，RTX 4090，dirty=false。命令 `PYTHONPATH=. .venv/bin/python -m scripts.diff_cut_validate --output runs/diff_26fff83_cut`。6/4 mm 网格，1452/4892 粒子，三种材料 × 两个下刀阶段，共 12 个 12 步窗口（每步 62.5 μs）。原求解器先从完整香蕉运行 1200/2400 步；侧码、释放标记和粒子对来自该轨迹。

- 正向最大绝对差：位置 **0.0149 μm**、速度 **0.0831 mm/s**、F **8.345e-7**、Fp **9.54e-6**、刀力 **1.223e-4 N**。一个窗口刀力超过 1e-4 N 目标；原求解器错误计数均为 0。
- 三档相对差分为 0.01/0.003/0.001。每例至少两档误差：E 的 12/12 例 ≤ **0.883%**；活跃屈服应力的 4/4 例 ≤ **1.52%**；摩擦的 12/12 例 ≤ **0.119%**。八个未屈服窗口的屈服应力 AD 为 0；低屈服材料每步最多 **1826** 粒子屈服，窗口内最大 Fp 变化 **0.00257**。
- 刀路在上述三档中有 **6/12** 例满足至少两档误差 ≤2%。单步命令增加 `--spacings 0.006 --materials soft plastic --prefixes 2400 --steps 1 --output runs/diff_26fff83_cut_single`，两个窗口的刀路误差最大 **0.00727%**。
- 自然分支对照增加 `--spacings 0.006 --natural-branches --output runs/diff_26fff83_cut_natural`：6 个窗口中刀路均未达到两档误差 ≤2%。香蕉窗口的粒子对数均为 0；粒子对及摩擦历史由已切分压紧块测试覆盖。

2026-10-21，提交 `f5f1e30`，dirty=false。命令 `PYTHONPATH=. .venv/bin/python -m scripts.diff_cut_validate --spacings 0.006 0.004 --materials soft plastic --prefixes 1200 2400 --steps 12 --parameters controls --control-epsilons .001 .0003 .0001 .00005 .00003 --repeats 3 --output runs/diff_8a_refine`。

- 刀路细扫 **7/8** 例在至少两档步长下误差 ≤2%。失败为 **6 mm、plastic、2400 步后**：五档误差 **119.36/3.53/20.46/3.91/8.28%**；固定同一初态和分支，三次 AD 的最大相对 L2 变化为 **0.000856%**，最大损失极差 **3.54e-7**。该窗口差分的两个方向均未改变活跃刀面特征、cut-hold 或计功带标记，差异原因继续核对。
- 细扫中最大 Fp 正向误差 **1.135e-5**（4 mm、plastic、2400 步后），超过 1e-5 目标。

### 网格约束、完整检查点与精度核对

2026-10-21，提交 `de61363`，RTX 4090，dirty=false。命令 `PYTHONPATH=. .venv/bin/python -m scripts.diff_cut_validate --grid-constraint --control-epsilons .0003 .0002 .0001 --output runs/m4_de61363_windows_f32`。12 个窗口的 float32 正向最大差：x **0.0112 μm**、v **0.0341 mm/s**、F **8.345e-7**、Fp **1.538e-5**、刀力 **2.178e-5 N**。Fp 超过 1e-5 目标；其余状态及刀力通过原目标。刀路三档中至少两档 ≤2% 的窗口为 **10/12**。

2026-10-21，提交 `a6ede54`，dirty=false。命令 `PYTHONPATH=. .venv/bin/python -m scripts.diff_cut_validate --grid-constraint --ordered-transfers --precision float64 --control-epsilons .0001 .00003 .00001 --output runs/m4_a6ede54_windows_f64`。两网格三材料 **12/12** 窗口通过方向导数检查，三档最大相对误差：刀路 **0.5694%**、E **0.001311%**、活跃屈服应力 **0.000121%**、摩擦 **0.006371%**。8 个未屈服窗口的屈服梯度为 0，FD 最大绝对值 **5.56e-14**。

2026-10-21，提交 `a6ede54`，dirty=false。命令 `PYTHONPATH=. .venv/bin/python runs/m4_a6ede54_precision_repeat/replay.py`，输入为 `m4_de61363_windows_f32/captures` 中保存的同一初态。原求解器重复 6 次，4 mm/plastic/2400 窗口的 Fp 最大极差 **1.475e-5**，与保存轨迹最大差 **1.342e-5**；可微 float32 与保存轨迹最大差 **1.531e-5**。6 mm/plastic 的原求解器极差 **5.96e-6**；6 mm/soft 的 Fp 极差为 **0**。

### 完整可微香蕉轨迹

2026-10-21，提交 `de61363`，dirty=false。命令 `PYTHONPATH=. .venv/bin/python -m scripts.diff_cut_trajectory --duration .48 --frame-stride 80 --backward --epsilons .0001 .00003 .00001 --ordered-transfers --precision float64 --video --output runs/m4_de61363_cut_6mm`。6 mm、1452 粒子，切透与退刀 **7680 步**、两块 **726/726**。刀路方向导数 AD **−0.1282190**，三档 FD 相对误差 **0.17382/0.04465/0.04568%**。32 步分段的 x/v/F/Fp/应力/接触历史/损伤重算最大差均为 **0**。独立原求解器 float32 与可微 float64 的整段最大 x/刀力差 **12.48 μm / 0.01988 N**。视频 `differentiable_cutting.mp4` 为可微求解器状态，97 帧、20 fps。

2026-10-21，提交 `a6ede54`，dirty=false。完整轨迹命令 `PYTHONPATH=. .venv/bin/python -m scripts.diff_cut_trajectory --duration 1.05 --dt <dt> --frame-stride 320 --backward --epsilons .0001 .00003 .00001 --ordered-transfers --precision float64 --output runs/m4_a6ede54_<档位>_dt`。62.5 μs 对照使用 `--frame-stride 160 --epsilons .00001`。

| dt | 步数 | 刀路 AD | FD 最大相对误差 | 原求解器/可微求解器最大位置差 | CUDA 内存池峰值 |
|---|---:|---:|---:|---:|---:|
| 62.5 μs（default） | 16800 | −1.397e18 | **100%：失败** | 954.1 μm | 1254.7 MiB |
| 31.25 μs（half） | 33600 | −0.4073895 | **0.04851%** | 21.47 μm | 2415.6 MiB |
| 15.625 μs（quarter） | 67200 | −0.6307439 | **0.08627%** | 69.78 μm | 4718.5 MiB |

31.25 μs 三档误差 **0.04851/0.000210/0.01365%**；最多 **24** 个粒子接触对。15.625 μs 三档误差 **0.08627/0.003527/0.001124%**，最多 **20** 对；视频 **211 帧 / 20 fps**。切后均为两块 **726/726**。三档全部检查点状态重算误差为 **0**，原求解器错误计数为 **0**。固定法向粘着接触的单步谱半径在 62.5/31.25/15.625 μs 下为 **1.09576/1.01105/0.99578**；完整轨迹的梯度放大对时间步敏感。三档独立进程并发运行。

2026-10-21，提交 `c697950`。完整轨迹入口默认用 `tangent_history_dt` 约束时间步；香蕉从 62.5 μs 细分到 **15.625 μs**，接触刚度保持相同。命令 `PYTHONPATH=. .venv/bin/python -m pytest tests/test_tangent_history_step.py -q`：**6 passed**。四种切向阻尼 0.1/0.5/1/2，接触核展开 32 步的正向及 VJP 与解析矩阵幂一致（rtol **1e-11**）。

### M4 最终验收

2026-10-21，提交 `fe6a422`，RTX 4090，dirty=false。命令 `PYTHONPATH=. .venv/bin/python -m scripts.diff_cut_validate --grid-constraint --ordered-transfers --precision float64 --reference-precision float64 --control-epsilons .0001 .00003 .00001 --output runs/m4_fe6a422_windows`。6/4 mm × banana/soft/plastic × 1200/2400 步前缀，**12/12** 窗口通过原正向目标；最大差 x **2.09e-17 m**、v **1.03e-13 m/s**、F **2.11e-15**、Fp **2.07e-14**、刀力 **5.28e-14 N**。

12/12 窗口刀路导数至少两档误差 ≤2%，最差第二档为 **0.00950%**。4 mm/plastic/2400 的三档误差 **2.58226/0.009496/0.000745%**，最粗档失败。材料和摩擦三档最大误差：E **0.001298%**、活跃屈服 **0.0000408%**、摩擦 **0.000584%**；8 个未屈服窗口的屈服 AD 为 0，差分最大绝对值 **5.56e-14**。

2026-10-21，提交 `fe6a422`，dirty=false。完整轨迹命令：

```bash
PYTHONPATH=. .venv/bin/python -m scripts.diff_cut_trajectory \
  --duration 1.05 --frame-stride 320 --ordered-transfers \
  --precision float64 --reference-precision float64 --video \
  --output runs/m4_fe6a422_full
PYTHONPATH=. .venv/bin/python -m scripts.diff_cut_trajectory \
  --reuse --backward --epsilons .0001 .00003 .00001 \
  --output runs/m4_fe6a422_full
```

本次三档差分分别在 `runs/m4_fe6a422_fd_{0,1,2}` 并发运行，共用完整轨迹的分支文件；各自命令与提交记录保存在报告，汇总为 `runs/m4_fe6a422_full/report.json`。1.05 s、**67200 步**、dt **15.625 μs**、1452 粒子，切后两块 **726/726**、最多 **20** 个粒子接触对。原求解器与可微求解器均为 float64，整段最大差 x **6.21e-15 m**、v **5.85e-13 m/s**、F **2.46e-13**、Fp **2.53e-13**、应力 **8.27e-7 Pa**、刀力 **6.37e-12 N**；原求解器错误计数 **0**。

刀路方向导数 AD **−0.5909453**，三档 FD 相对误差 **0.023243/0.003080/0.001372%**，全部通过。三个独立进程的控制量梯度最大绝对差 **7.11e-15**；32 步分段中 x/v/F/Fp/应力/接触历史/损伤重算差全部为 **0**。CUDA 数组内存池峰值 **4711.3 MiB**。视频 `differentiable_cutting.mp4` 使用可微求解器的粒子状态，含切透、退刀、推块及再次退刀，**211 帧 / 20 fps**。香蕉轨迹的网格约束求解次数为 0；活跃约束由 `test_active_grid_constraint_in_complete_checkpoint` 覆盖完整推进与反向。

### 原求解器与可微求解器的视频对照

2026-10-21，提交 `05a9327`，RTX 4090，dirty=false。命令 `PYTHONPATH=. .venv/bin/python -m scripts.diff_cut_trajectory --duration 1.05 --frame-stride 320 --ordered-transfers --precision float64 --reference-precision float64 --comparison-video --output runs/m4_05a9327_compare`。两侧分别读取原/可微求解器独立推进的 x/v/F，共用刀路、材质、时间步和相机；67200 步最大位置分量差 **4.07e-15 m**、粒子位置距离 **4.37e-15 m**、速度分量差 **6.02e-13 m/s**、刀力分量差 **9.24e-12 N**。回弹、碰撞与退刀时的切块抬升在两边一致。

0.630 s 有 **14** 对块间接触；0.765 s 有 **8** 对，0.775 s 右块质心 x 速度 **0.2523 m/s**。0.85–1.05 s 无块间接触，左块质心从 **11.636 mm** 升至 **33.663 mm**；末端 z 速度 **0.11999 m/s**，刀上升速度 **0.12 m/s**，反力 **0.21153 N** 接近该块重量 **0.21153 N**。当前材料黏度和 Maxwell 分量均为 0；M4 横推速度 **0.16 m/s**，Franka 三刀任务为 **0.02 m/s**。

`solver_comparison.mp4` 为 **0.1 倍速、211 帧**；命令增加 `--render-only --playback-speed 1` 生成 `solver_comparison_1x.mp4`（**22 帧、20 fps**）。正常速度的六次循环版通过 `ffmpeg -stream_loop 5 -i solver_comparison_1x.mp4 -c copy -movflags +faststart solver_comparison_1x_loop.mp4` 生成。两版视频在 Mac 的 `~/Desktop/BladeMaster_videos/M4_original_vs_differentiable[_1x]_05a9327.mp4`。

### M5：事件定位与自然切割扫描

2026-10-21，提交 `26c9fe5`，RTX 4090，dirty=false。命令 `PYTHONPATH=. .venv/bin/python -m scripts.diff_cut_events --spacing <.006|.004> --duration .24 --output runs/m5_26c9fe5_events_<6|4>mm`。两网格各 12 例，扫描刀位 ±0.05/±2 mm、下刀行程 ±1%、G_c 的 0.01/0.1/0.99/1.01/10 倍；粒子数 **1452/4892**，步长 **15.625/13.158 μs**。

G_c=700 J/m² 时，两网格均有 **0** 个损伤释放事件，最大损伤 **0.13835/0.19641**，几何分侧得到 **726/726、2446/2446** 两块。G_c=70 时释放 **4/5** 个粒子，G_c=7 时释放 **4/9** 个。6 mm 下 G_c±1% 的终态位置距离差 ≤ **3.13e-16 m**；该区间反力对 G_c 的局部导数为零。

刀向 x 正向偏移 **0.05 mm** 时，两网格分别有 **263/1817** 个粒子的分侧时刻改变，最大变化 **46.875/276.316 μs**，最终分侧集合相同。偏移 **−2 mm** 时切块为 **674/778、2330/2562**，侧码改变 **118/314** 个；6 mm 下还触发 **4** 个损伤释放事件。事件时间和分侧集合分别进入 M5 后续动力学导数。

6/4 mm 的默认刀做功为 **9.596/7.876 mJ**，损伤计功为 **0.34478/0.26292 mJ**。从保存的损伤计算 `threshold × sum(damage)`，默认差值 < **2e-18 J**；G_c=7 时，计功超过截断后损伤对应值 **0.269/0.294 μJ**，来自跨越阈值的子步增量。

### M5：损伤释放后的状态与刀力导数

2026-10-21，提交 `abd6b02`，RTX 4090，dirty=false。`EventCutContact` 按剩余断裂功积分释放步的刃口冲量；6/4 mm、G_c=7/70、四档步长共 **16** 种 **2 ms** 窗口，每个刀路/G_c 方向至少两档自然差分误差 ≤**2%**。初始粒子状态及损伤固定，窗口内自然更新损伤释放、接触及粘滑分支。

```bash
PYTHONPATH=. .venv/bin/python -m scripts.diff_release_validate --output runs/m5_abd6b02_release
PYTHONPATH=. .venv/bin/python -m scripts.diff_release_validate \
  --dt-divisors 4 8 --epsilons .00001 .000001 --output runs/m5_abd6b02_release_refined
PYTHONPATH=. .venv/bin/python -m scripts.diff_release_validate \
  --spacings .004 --energies 70 --dt-divisors 1 --epsilons .0000001 .00000001 .000000001 \
  --output runs/m5_abd6b02_release_small_eps
```

完整 Tape 与 16 步分段的刀路/G_c/初始 x 梯度最大绝对差为 **4.89e-15/1.74e-17/4.55e-13**；全部状态重算差为 **0**。损伤计功与 `Q × sum(ΔD)` 的最大差为 **1.77e-19 J**。相同 dt 下，步内释放与原边界释放的末端位置分量差 ≤**0.693 μm**。

自然差分在跨接触分支时不单调：4 mm、G_c=70、原步长的刀路差分在 ε=1e-6 时改变 **30** 个砧板接触模式，误差 **18.86%**；ε=1e-7/1e-8 时模式变化均为 **0**，误差 **7.59e-7%/1.44e-5%**。

**固定每步 PIC 比例的步长收敛验收失败。** 从四分之一步长继续减半到八分之一步长，末端位置距离差仍为 **15.84–19.48 μm**，梯度和释放集合继续变化。相邻两档导数相对差用 `abs(a-b)/max(abs(a),abs(b))` 计算：

| 网格 / G_c | 窗口释放数，dt/4 → dt/8 | 刀路方向导数相对差 | G_c 导数相对差 |
|---|---:|---:|---:|
| 6 mm / 7 | 4 → 4 | 5.04% | 1.95% |
| 6 mm / 70 | 4 → 3 | 22.52% | 31.17% |
| 4 mm / 7 | 5 → 5 | 18.75% | 6.61% |
| 4 mm / 70 | 3 → 5 | 9.61% | 57.69% |

### M5：固定传递耗散率的步长对照

2026-10-21，提交 `1377995`，RTX 4090，dirty=false。固定 PIC 残差耗散率后，**16** 种 **2 ms** 释放窗口的最细相邻两档刀路/G_c 导数相对差 ≤**0.856%/0.261%**，释放粒子集合一致。6/4 mm 分别固定 λ=**3200/3800 s⁻¹**，材料、接触刚度及每组共同初态与上表相同；`flip_reference_dt` 对应 **15.625/13.1579 μs**。每步固定 5% PIC 则会在每次减半步长时使 λ 加倍。

```bash
PYTHONPATH=. .venv/bin/python -m scripts.diff_release_validate \
  --spacings .006 .004 --energies 7 70 --dt-divisors 1 2 4 8 \
  --transfers fixed-rate --epsilons 1e-7 1e-8 --output runs/m5_1377995_fixed_rate
```

下表比较 dt/4 → dt/8；四档步长的释放集合均一致：

| 网格 / G_c | 窗口释放数 | 位置最大距离差 | 事件时间最大差 | 刀路方向导数相对差 | G_c 导数相对差 |
|---|---:|---:|---:|---:|---:|
| 6 mm / 7 | 4 → 4 | 0.375 μm | 0.470 μs | 0.856% | 0.160% |
| 6 mm / 70 | 4 → 4 | 0.470 μm | 1.307 μs | 0.650% | 0.0264% |
| 4 mm / 7 | 5 → 5 | 0.615 μm | 0.720 μs | 0.509% | 0.261% |
| 4 mm / 70 | 3 → 3 | 0.589 μm | 0.702 μs | 0.427% | 0.0240% |

两档自然差分最大误差 **0.000294%**，扰动时释放步号与刚体接触模式变化均为 **0**。完整 Tape/分段反向的刀路/G_c/初始 x 最大差为 **7.99e-15/1.04e-17/2.28e-13**，全部状态重算差 **0**，损伤计功平衡误差 ≤**8.14e-20 J**。

同提交的纯 FLIP（λ=0）对照共 **8** 例，dt/4 → dt/8 的刀路/G_c 导数相对差 ≤**0.998%/0.0818%**，位置距离差 ≤**0.239 μm**，事件时间差 ≤**0.965 μs**，四组释放集合一致；自然差分最大误差 **0.00218%**，损伤计功平衡误差 ≤**1.43e-19 J**。两种固定 λ 设置均通过这些窗口的步长检查。

```bash
PYTHONPATH=. .venv/bin/python -m scripts.diff_release_validate \
  --spacings .006 .004 --energies 7 70 --dt-divisors 4 8 \
  --transfers flip --epsilons 1e-7 1e-8 --output runs/m5_1377995_flip
```

下一步处理几何侧码、接触集合变化及完整自然切割轨迹的收敛与优化。

## M5 自然分侧、接触集合与跨事件差分（2026-10-21）

提交 `acead71`（dirty=false），RTX 4090、float64。首次分侧附近各 64 步：香蕉几何 6/4 mm × banana/soft/plastic 共 6 例，6 mm 方块/圆柱 soft 共 2 例。每例从同一中间状态开始，各条扰动轨迹重新计算侧码、接触对、摩擦和释放分支。

8 例原/可微正向侧码与接触对差异均为 **0**；最大 x/v/Fp/刀力差为 **2.08e-17 m / 4.13e-14 m/s / 5.46e-14 / 2.43e-14 N**。新增分侧粒子分别为 **68/54/42/35/42/13/48/32**；这些窗口的粒子对数均为 **0**。16/32 步检查点的刀路/初始 x 梯度最大差为 **2.78e-16 / 8.18e-14**，全部状态重算差 **0**。

刀位与下刀深度各五档扰动，共 **80** 组中心差分；其中 **65** 组改变记录的分侧、接触模式或释放事件，**49** 组相对偏差 >**2%**，**8** 组 AD 与差分符号相反。记录事件不变的 **15** 组最大相对误差 **1.17e-7**。6 mm 香蕉的深度扰动 ε=5e-5：AD **0.0191941**、FD **0.00304626**，相对偏差 **84.13%**；6 mm soft 在 ε=5e-7 时分侧不变、接触模式改变，深度导数偏差 **13.15%**。跨事件验收失败；步边界分侧的事件时间导数及接触切换灵敏度继续接入。

```bash
PYTHONPATH=. .venv/bin/python -m scripts.diff_natural_validate \
  --output runs/m5_acead71_natural_materials --spacings .006 .004 \
  --materials banana soft plastic --steps 64 --segment-steps 16 32 \
  --release boundary --epsilons .05 .005 .0005 .00005 .0000005
PYTHONPATH=. .venv/bin/python -m scripts.diff_natural_validate \
  --output runs/m5_acead71_natural_shapes --spacings .006 --shapes block cylinder \
  --materials soft --steps 64 --segment-steps 16 32 \
  --release boundary --epsilons .05 .005 .0005 .00005 .0000005
```

### M5 几何分侧的步内状态积分（2026-10-21，`3e8f777`）

8 个窗口、112 组自然差分，**63 组误差 >2%、19 组符号相反**；每个方向至少两档差分 ≤2% 的窗口为 **5/8**。原五档扰动对应的 80 组中 **60** 组误差 >2%。分侧时刻已进入实际前后分段及多场状态，接触切换验收失败。6 mm 香蕉的深度微扰 `5e-8` 在第 21 步、粒子 852 的砧板粘滑模式由 **11→12**；AD **0.10329855**，左右差分 **1.26101778/0.10351683**。微扰 `5e-9` 时模式不变，中央差分相对误差 **3.07e-7**。

两网格三材料及方块/圆柱的新分侧数为 **68/55/42/37/42/14/48/32**；窗口粒子对峰值均为 **0**。16/32 步检查点的状态重算差为 **0**，控制量/初始位置梯度最大差 **1.16e-13/1.05e-11**。相对原步末积分，最大位置/速度/Fp/刀力差为 **1.427 μm/0.02156 m/s/3.18e-5/0.01645 N**。报告：`runs/m5_3e8f777_geometry_{materials,shapes}/report.json`，dirty=false。

```bash
PYTHONPATH=. .venv/bin/python -m scripts.diff_natural_validate \
  --output runs/m5_3e8f777_geometry_materials --spacings .006 .004 \
  --materials banana soft plastic --steps 64 --segment-steps 16 32 \
  --release integrated --geometry event \
  --epsilons .05 .005 .0005 .00005 .0000005 .00000005 .000000005
PYTHONPATH=. .venv/bin/python -m scripts.diff_natural_validate \
  --output runs/m5_3e8f777_geometry_shapes --spacings .006 --shapes block cylinder \
  --materials soft --steps 64 --segment-steps 16 32 \
  --release integrated --geometry event \
  --epsilons .05 .005 .0005 .00005 .0000005 .00000005 .000000005
```

## 测试

2026-10-21，提交 `3e8f777`：全量 **740 passed、8 skipped、18 subtests passed**，142.37 s。新增 5 项分段时长检查和 28 项几何事件检查：固定刚度/PIC 速率、解析穿越、两网格与刚度、方块/圆柱、转动刀、自然生成两侧场、塑性及接触全状态 VJP、材料/刀路导数和检查点。8 项跳过为 CRESSim 上游构建未配置。

2026-10-21，提交 `acead71`：全量 **707 passed、8 skipped、18 subtests passed**，142.84 s。新增 14 项覆盖方块/圆柱和转动刀的自然侧码对照、预测接触集合与原全对搜索、摩擦历史继承/重入清零、刀路改变侧码、同刀暂停/新刀重置、两种释放积分的自然轨迹梯度及分段重算。8 项跳过仍为 CRESSim 上游构建未配置。

2026-10-21，提交 `1377995`：全量 **693 passed、8 skipped、18 subtests passed**，134.49 s。新增 13 项覆盖 PIC 耗散率配置、动量守恒、传递动能耗散、原/可微求解器对照、速度伴随及解析指数衰减的步长细化。8 项跳过为 CRESSim 上游构建未配置。

2026-10-21，提交 `abd6b02`：全量 **680 passed、8 skipped、18 subtests passed**，135.57 s。新增 8 项覆盖释放冲量/动量、损伤计功、自然释放后的 G_c/刀路梯度及分段反向、未释放时与 M4 一致性。`72f553e` 将三刀编号置换检查设为 float64，并按机器精度比例收紧容差；原 float32 相同编号重复运行的位置差可达 **0.101 μm**，超过 **0.075 μm** 的检查阈值。8 项跳过为 CRESSim 上游构建未配置。

2026-10-21，提交 `26c9fe5`：全量 **672 passed、8 skipped、18 subtests passed**，136.72 s。新增 10 项覆盖平面穿越时刻/位置、损伤阈值时刻对刀位/初始损伤/G_c 的导数，以及 G_c 的完整 Tape/分段重算/双档差分。8 项跳过为 CRESSim 上游构建未配置。

2026-10-21，提交 `05a9327`：提交前全量 **662 passed、8 skipped、18 subtests passed**，135.46 s。新增原求解器状态记录、同镜头并排视频与播放速度选项；两档视频已检查下刀、块间接触、横推及退刀帧。

2026-10-21，提交 `624dafe`：提交前全量 **662 passed、8 skipped、18 subtests passed**，227.52 s。新增活跃刀面网格约束与粒子推进的集成测试：完整 Tape 和 2 步分段的刀位梯度一致（rtol **1e-9**），两档有限差分通过 **2%** 目标，约束速度残差 ≤ **1e-6 m/s**。8 项跳过为 CRESSim 上游构建未配置。

2026-10-21，提交 `fe6a422`：全量 **661 passed、8 skipped、18 subtests passed**，142.64 s。同精度双精度多场塑性对照采用 **1e-10** 绝对容差；原求解器双精度外部耦合的 CUDA Graph 与直接启动结果一致。删除未被求解流程调用的 `separate_embedded` 及其单独测试。

2026-10-21，提交 `c697950`：全量 **657 passed、8 skipped、18 subtests passed**，151.00 s。新增 6 项为接触历史稳定步长、刚度保持和四种阻尼的解析正反向核对。

2026-10-21，提交 `a6ede54`：全量 **651 passed、8 skipped、18 subtests passed**，180.64 s。新增检查覆盖网格刀面约束的正向与 x/刀位/网格速度梯度，双精度本构对 NumPy SVD，固定顺序传递，以及含塑性应力、接触对变化和摩擦历史的完整状态检查点。

2026-10-21，提交 `f5f1e30`：提交前全量 **637 passed、8 skipped、18 subtests passed**，134.27 s。新增 28 项覆盖 Hencky 投影及重复伸长、E/屈服/F/Fp/速度梯度、多场交换、粒子对历史与动量守恒、刀路/摩擦梯度、切割计功和释放。8 项跳过仍为 CRESSim 上游构建未配置。

2026-10-21，提交 `023d30d`：提交前全量 **609 passed、8 skipped、18 subtests passed**，133.47 s。8 项跳过为 CRESSim 上游构建未配置。新增 6 项覆盖接触分段重算的初始状态、摩擦历史、E 与刀路梯度，含非整除分段及反力罚项生效的情况。

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests -q
```

测试只依赖本仓库，覆盖：

- 网格交换（论文式 8）、刀距离、摩擦（论文式 12–13）、粒子传输的 FP64 参考；
- 动量守恒、刚体平移、切割标签和接触的不变量。

### M5 投影摩擦与自然窗口（2026-10-22，`4fe7e4a`）

8/8 窗口的刀位/行程梯度各有至少两档差分误差 ≤2%，最差第二档 **0.004304%**。全部 112 组扰动中 **47** 组偏差 >2%、**6** 组符号相反；原五档 80 组子集对应 **47/6**。新分侧数 **68/55/42/37/42/14/48/32**，粒子对峰值均为 0。16/32 步检查点状态重算差 **0**，控制量/初始位置梯度最大差 **1.11e-14/1.47e-12**。相对原求解器最大位置/速度/Fp/刀力差 **4.915 μm/0.01928 m/s/7.53e-5/0.01640 N**。

```bash
PYTHONPATH=. .venv/bin/python -m scripts.diff_natural_validate \
  --output runs/m5_4fe7e4a_projected_materials --spacings .006 .004 \
  --materials banana soft plastic --steps 64 --segment-steps 16 32 \
  --release integrated --geometry event --friction-update projected \
  --epsilons .05 .005 .0005 .00005 .0000005 .00000005 .000000005
PYTHONPATH=. .venv/bin/python -m scripts.diff_natural_validate \
  --output runs/m5_4fe7e4a_projected_shapes --spacings .006 --shapes block cylinder \
  --materials soft --steps 64 --segment-steps 16 32 \
  --release integrated --geometry event --friction-update projected \
  --epsilons .05 .005 .0005 .00005 .0000005 .00000005 .000000005
```

两个报告均为 dirty=false。全量 `PYTHONPATH=. .venv/bin/python -m pytest tests -q`：**772 passed、8 skipped、18 subtests passed**，147.52 s；新增 8 项摩擦与 24 项投影摩擦/几何事件组合检查。8 项跳过为 CRESSim 上游构建未配置。

### M5 自然分侧的时间步对照（2026-10-22，`3276aa3`）

**下刀行程梯度的时间步收敛失败。** 6 mm 香蕉、同一 1 ms 初态窗口、固定接触刚度和 PIC 速率 3200 s⁻¹，时间步 **15.625/7.8125/3.90625/1.953125/0.9765625/0.48828125 μs**。对应行程梯度 **0.0716204/0.2122731/0.0921054/0.0285121/−0.0104185/−0.0179385**；最细相邻两档相差 **41.92%**，刀位梯度相差 **0.1911%**。相邻位置最大分量差依次 **0.4975/0.3218/0.1870/0.09888/0.05162 μm**，末两档速度差 **0.0004047 m/s**、事件时刻差 **0.06751 μs**；各档最终侧码、释放及事件粒子集合一致。

```bash
PYTHONPATH=. .venv/bin/python -m scripts.diff_natural_refine \
  --output runs/m5_3276aa3_refine --spacings .006 --materials banana \
  --duration .001 --dt-divisors 1 2 4 8 16 32 --segment-steps 32 \
  --epsilons .00000005
```

报告 dirty=false。32 步检查点状态重算差 **0**，控制量/初态梯度最大差 **4.38e-14/1.59e-12**，损伤功平衡误差 ≤**1.99e-19 J**。该扰动在前五档的行程差分误差 ≤**0.01781%**，最细档 **5.895%**，正向扰动同时改变板/刀接触模式和几何事件步号。全量测试 **772 passed、8 skipped、18 subtests passed**，145.29 s。

分项对照仍用同一初态及前四档时间步：关闭网格刀面约束的行程梯度最大变化 **6.22e-12**；静动摩擦均为 0 时为 **−0.102746/0.078288/−0.046175/−0.024107**。去除全部接触、保留材料推进和几何分侧时为 **0.036907/−0.008764/0.016565/−0.002596**。下一步核对事件分段的时间离散误差及接触切换。命令：`PYTHONPATH=. .venv/bin/python runs/m5_contact_diagnosis/probe.py` 与 `probe_free_flow.py`；脚本和输出保存在该 runs 目录。

### M1–M5 代码复核（2026-10-22，`d3124a0`）

**修复双精度入口舍入、刀刃事件后的计功分支及固定刚度实验的砧板系数换算。** 全量命令 `PYTHONPATH=. .venv/bin/python -m pytest tests -q`：**785 passed、8 skipped、18 subtests passed**，150.35 s；8 项跳过为 CRESSim 上游构建。独立 NumPy 本构核对覆盖 24 个旋转/重复伸长/塑性状态及 F、Fp、μ、λ、屈服半径五组 VJP；55 项几何测试包含同时穿越及刀位 ±1 浮点间隔。分段反向入口对照完整 Tape 的三项刀路和材料梯度通过。

- `python -m scripts.diff_validate --output runs/audit_d3124a0_m1_m3`：两网格、1/12/32 步共 6 个弹性案例，E 差分最大误差 **0.2032%**；4 个接触案例刀位差分最大误差 **0.003785%**。
- `python -m scripts.diff_horizon --output runs/audit_d3124a0_m2_horizon`：两网格、32/128/512 步共 22 组完整/分段计算；36 档 E/v/F 差分最大误差 **0.8241%**，检查点 E/状态梯度相对差 ≤**0.000810%/0.001502%**。
- `python -m scripts.diff_contact_horizon --repeats 1 --output runs/audit_d3124a0_m3_horizon`：两网格、128/512 步、8/16 控制量共 24 组；完整/分段梯度最大相对 L2 差 **0.1371%**，末端位置最大差 **0.1509 μm**。 同提交的 `diff_contact_horizon --spacings .004 --steps 512 --controls 8 --repeats 3 --output runs/audit_d3124a0_m3_repeat` 中，完整 Tape 三次重复自身的梯度差为 **0.1370%**，两种分段的最大差为 **0.002614%**。
- `python -m scripts.diff_cut_validate --spacings .006 .004 --materials banana soft plastic --prefixes 1200 2400 --steps 12 --grid-constraint --ordered-transfers --precision float64 --reference-precision float64 --control-epsilons .0001 .00003 .00001 --output runs/audit_d3124a0_m4_windows`：**12/12** 窗口的刀路/E/屈服/摩擦均有至少两档差分通过；刀路最差第二档 **0.009495%**。原求解器对照最大 x/Fp/力差 **2.09e-17 m/2.27e-14/6.79e-14 N**。
- `python -m scripts.diff_release_validate --output runs/audit_d3124a0_m5_release --transfers fixed-rate --dt-divisors 1 2 4 8 --epsilons 1e-7 1e-8`：**16/16** 释放窗口通过；自然差分最大误差 **0.001564%**，最细相邻两档刀路/G_c 梯度差 ≤**0.856%/0.261%**，末端位置差 ≤**0.616 μm**，释放集合一致。状态重算差 **0**，检查点梯度最大绝对差 **3.98e-13**，损伤功平衡误差 ≤**1.29e-19 J**。
- `python -m scripts.diff_natural_validate --geometry event --friction-update projected --spacings .006 .004 --materials banana soft plastic --epsilons .05 .005 .0005 .00005 .000005 .0000005 .00000005 --output runs/audit_d3124a0_m5_materials`，另以 `--spacings .006 --materials banana --shapes block cylinder` 输出到 `runs/audit_d3124a0_m5_shapes`。8 个窗口的局部刀路导数各有两档误差 ≤2%，最差第二档 **0.3854%**；4 mm 塑性材料的局部检查另用 `--parameters knife_depth_scale --segment-steps 32 --epsilons 5e-8 1e-8 5e-9 --output runs/audit_d3124a0_m5_plastic_local`，三档误差 **0.0001071/0.0000809/0.004076%**。原七档共 **112** 组扰动中仍有 **49** 组误差 >2%、**6** 组符号相反。16/32 步检查点状态重算差 **0**，刀路/初态梯度最大差 **1.15e-14/4.87e-12**。
- `python -m scripts.diff_cut_trajectory --duration 1.05 --frame-stride 320 --backward --segment-steps 32 --epsilons .0001 .00003 .00001 --ordered-transfers --precision float64 --reference-precision float64 --output runs/audit_d3124a0_m4_full`：**67200** 步、两块 **726/726**，原求解器错误计数 **0**；最大 x/Fp/力差 **4.71e-15 m/2.73e-13/3.89e-12 N**。刀路 AD **−0.59092516**，三档差分误差 **0.023270/0.002956/0.001330%**；全部状态重算差 **0**，峰值 CUDA 数组内存 **4805.8 MiB**。
- `python -m scripts.diff_natural_refine --output runs/audit_d3124a0_m5_refine --spacings .006 --materials banana --duration .001 --dt-divisors 1 2 4 8 16 32 64 128 --segment-steps 32 --adjoint checkpoint --epsilons 5e-9`：**时间步收敛失败**。最细两档 **0.244140625/0.1220703125 μs** 的行程梯度为 **−0.02249150/−0.02701883**，相差 **16.76%**；刀位梯度差 **0.02728%**，最大位置分量差 **13.43 nm**，事件时刻差 **17.53 ns**，末端侧码/释放/接触对及事件粒子集合一致。最细档刀位/行程的自然差分误差 **0.000176%/0.000133%**，全部状态重算差 **0**。此实验的梯度取 32 步分段结果，独立核对来自自然差分；完整 Tape/不同分段长度对照由局部测试及上面的自然窗口实验覆盖。
- `python runs/audit_20261022/event_condition.py`：基准步长及 dt/128 各 **68** 个入区根，`|∂g/∂τ|` 分别为 **0.11914–0.14325 / 0.11915–0.14308 m/s**，本窗口没有接近零的穿越速度。下一步比较事件段的应力更新时间与宏步分割相位，核对材料推进的状态/梯度收敛阶，再逐项恢复接触切换。

报告比较来源修正为 `03fd308`：checkpoint-only 首段作为梯度基准，对应误差字段为 `null`；后续段数或完整 Tape 的比较记录 `gradient_reference`。全量测试仍为 **785 passed、8 skipped、18 subtests passed**，147.34 s。

### M5 E1：中点积分与步内穿越（2026-10-22）

`0a8fd13`：事件分段采用显式中点 RK2，摩擦历史在阶段终点按库仑界限投影。固定材料、接触刚度和 PIC 速率的 **27** 档时间步对照中，最细相邻两档的三项刀路梯度相对差最大 **0.1279%**，材料梯度最大 **0.06259%**；位置最大分量差 **1.041 nm**，事件时刻差 **0.2691 ns**，最终侧码、释放、接触对及事件粒子集合一致。自然差分最大误差 **0.003862%**，检查点状态重算差 **0**。

| 材料 / 网格 | 时间网格相位 | dt/32 → dt/64 行程梯度 | 相对差 |
|---|---:|---:|---:|
| banana / 6 mm | 0 | −0.02506961 → −0.02504539 | 0.09661% |
| banana / 6 mm | 1/4 | −0.02504662 → −0.02504461 | 0.008040% |
| banana / 6 mm | 1/2 | −0.02512598 → −0.02509385 | 0.1279% |
| banana / 4 mm | 0 | 0.21444212 → 0.21443270 | 0.004393% |
| soft / 6 mm | 0 | 0.006233852 → 0.006233193 | 0.01058% |
| plastic / 6 mm | 0 | 0.008598364 → 0.008605933 | 0.08796% |
| soft / 4 mm | 0 | 0.002880771 → 0.002880311 | 0.01597% |
| plastic / 4 mm | 0 | 0.007109533 → 0.007109035 | 0.007006% |

```bash
PYTHONPATH=. .venv/bin/python -m scripts.diff_natural_refine \
  --output runs/e1_0a8fd13_phases --spacings .006 --materials banana \
  --dt-divisors 8 16 32 64 --phases 0 .25 .5 --adjoint checkpoint \
  --segment-steps 32 --epsilons 5e-9 --integration midpoint
PYTHONPATH=. .venv/bin/python -m scripts.diff_natural_refine \
  --output runs/e1_0a8fd13_refine_4mm --spacings .004 --materials banana \
  --dt-divisors 16 32 64 --adjoint checkpoint --segment-steps 32 \
  --parameters knife_depth_scale --epsilons 5e-9 --integration midpoint
PYTHONPATH=. .venv/bin/python -m scripts.diff_natural_refine \
  --output runs/e1_0a8fd13_refine_materials --spacings .006 .004 --materials soft plastic \
  --dt-divisors 16 32 64 --adjoint checkpoint --segment-steps 32 \
  --parameters knife_depth_scale --epsilons 5e-9 --integration midpoint
```

`7419b30`：候选查询加入 RK2 二次位置延拓及旋转分段搜索。抛物线粒子、半圈刀和整圈刀的两端均在区外、步内进入再离开案例通过，事件时刻及导数与解析式/独立求根对照。粘着接触振子与矩阵指数解相比，步长减半的状态和梯度误差均下降 **>3.8 倍**；滑动接触符合恒定库仑加速度。全量 `PYTHONPATH=. .venv/bin/python -m pytest tests -q`：**820 passed、8 skipped、18 subtests passed**，201.68 s。8 项跳过为 CRESSim 上游构建。以上正式报告均为 dirty=false，运行于 RTX 4090。

同提交的 **8/8** 自然窗口，每项刀路梯度均有至少两档差分误差 ≤2%，最差第二档 **0.0003088%**；三档局部扰动共 **48** 组的最大误差 **0.08092%**。完整 Tape 与 16/32 步检查点的控制量/初始位置梯度最大差 **4.09e-14/2.12e-12**，状态重算差 **0**，损伤功平衡误差 ≤**9.75e-20 J**。包含大扰动的全部 **128** 组中，**35** 组误差 >2%、**2** 组反号；反号均在行程 ±5%，同时改变分侧与接触集合。E2 将接触建立/脱离及历史切换分解为单事件，核对连续性和单侧导数，区分有限扰动的非线性与状态跳变。

```bash
PYTHONPATH=. .venv/bin/python -m scripts.diff_natural_validate \
  --output runs/e1_7419b30_materials --geometry event --integration midpoint \
  --friction-update projected --spacings .006 .004 --materials banana soft plastic \
  --epsilons .05 .005 .0005 .00005 .000005 .0000005 .00000005 .000000005 \
  --segment-steps 16 32
PYTHONPATH=. .venv/bin/python -m scripts.diff_natural_validate \
  --output runs/e1_7419b30_shapes --geometry event --integration midpoint \
  --friction-update projected --spacings .006 --materials banana --shapes block cylinder \
  --epsilons .05 .005 .0005 .00005 .000005 .0000005 .00000005 .000000005 \
  --segment-steps 16 32
```

`7419b30` 同一 6 mm 香蕉窗口、相位 0 的 **dt/64→dt/128** 行程梯度为 **−0.02504539→−0.02505073**，相差 **0.02130%**（`d3124a0` 同两档为 16.76%）；刀位梯度差 **0.0003992%**，位置差 **0.06026 nm**，事件时刻差 **0.05289 ns**。最终侧码、释放、接触对及事件粒子集合一致。

三个时间网格相位 **0/1⁄4/1⁄2**、三档步长共 **9** 档复测，dt/128 行程梯度分别为 **−0.02505073/−0.02505046/−0.02506833**；相邻 dt/64→dt/128 的相对差分别为 **0.02130%/0.02335%/0.1017%**，同一最细步长的相位间差 **0.07128%**。刀路自然差分最大误差 **0.001385%**，材料梯度的最细两档相对差 ≤**0.002841%**，状态重算差 **0**，损伤功平衡误差 ≤**1.02e-18 J**。

```bash
PYTHONPATH=. .venv/bin/python -m scripts.diff_natural_refine \
  --output runs/e1_7419b30_refine --spacings .006 --materials banana \
  --dt-divisors 32 64 128 --phases 0 .25 .5 --adjoint checkpoint \
  --segment-steps 32 --epsilons 5e-9 --integration midpoint
```

### 2026-10-22：M5 E2 接触修复与 E3 初态释放轨迹

`819cb13` / `09e52b4`：中点阶段重搜实际粒子接触对，按 ID 补齐阶段历史；接触外历史积分率为零，零摩擦返回映射消去试探切向力及其历史导数。移动刀面的网格约束使用相对速度，并投影 `v_base_grid + τ a_stage`。`7b1fb8e` 的 16 项接触测试覆盖穷举接触对、平移/转动刀面、动量与反力、进入侧释放及刀面到刀缘的完整轨迹导数。另加两档步长的滑动到粘着轨迹，核对完整 Tape、32 步检查点及两档自然差分。共 18 项接触检查通过；全量 `PYTHONPATH=. .venv/bin/python -m pytest tests -q`：**850 passed、8 skipped、18 subtests passed**，351.56 s；8 项跳过为 CRESSim 上游构建。

`fb379d4`，RTX 4090、dirty=false：**6 mm、1452 粒子、G_c=7 J/m²、15.625 μs、3200 步 / 50 ms**，从未切割初态自然生成 **68** 个侧码，**4** 个粒子损伤释放。完整 Tape 与 **16/32/128** 步检查点的控制量/初始位置梯度最大差为 **5.47e-15 / 6.24e-13**，全部状态重算差 **0**；损伤功平衡误差 **2.03e-20 J**，进程 CUDA 数组内存池峰值 **13.38 GiB**。

刀位、行程、初始高度、E、摩擦系数与 G_c 共 **12** 组自然差分，最大相对误差 **0.9185%**；刀路与材料各自最大误差 **0.02604% / 0.0002062%**。初态有粒子位于板接触边界，负向高度扰动在首步建立接触；高度扰动 `1e-7 mm` 时 AD **−0.00886663175**、左差分 **−0.00903107089**、右差分 **−0.00886658247**，AD 与右差分相差 **0.000556%**。

```bash
PYTHONPATH=. .venv/bin/python -m scripts.diff_natural_validate \
  --output runs/e23_fb379d4/initial_50ms_gc7 --geometry event --from-initial \
  --trajectory cut-withdraw-push --duration .05 --cutting-energy 7 --spacings .006 \
  --adjoint full --segment-steps 16 32 128 --epsilons 1e-6 1e-7 \
  --state-parameters initial_z_mm young_scale friction_scale fracture_scale
```

同提交的 **6 mm、G_c=700、0.24 s / 15360 步** 自然下刀，侧码新增 **1096**，损伤释放 **0**，峰值刀力 **1.0963 N**。32 步检查点的全部状态重算差 **0**；刀位/行程梯度为 **−0.00949831376 / 0.111050458**，`1e-7` 自然差分的相对误差为 **0.00001653% / 0.002260%**。行程负向扰动改变一个粒子的分侧步号。损伤功平衡误差 **2.17e-19 J**，进程 CUDA 数组内存池峰值 **2.74 GiB**。

```bash
PYTHONPATH=. .venv/bin/python -m scripts.diff_natural_validate \
  --output runs/e23_fb379d4/initial_240ms_gc700 --geometry event --from-initial \
  --trajectory cut-withdraw-push --duration .24 --spacings .006 \
  --adjoint checkpoint --segment-steps 32 --epsilons 1e-7
```

### 2026-10-22：M5 E2 接触验收

`fb379d4`，RTX 4090、dirty=false：刀/板/粒子对 × 6/4 mm × 时间相位 0/1⁄4/1⁄2 × 步长 50/6.25/1.5625/0.78125 μs，共 **72** 例弹性碰撞。每例位置、速度对初始间隙的导数均有两档自然差分误差 ≤2%，最差第二档 **0.005707%**。`1e-8 m` 扰动中 **12** 例跨过离散接触切换，中央差分最大偏差 **63.87%**；这些例子的 `1e-10/3e-10 m` 局部差分通过。

50→0.78125 μs 时，解析位置误差最大值 **498.3→0.1249 nm**，速度误差 **8.496e-5→3.715e-8 m/s**。最细步长的位置梯度相对误差 ≤**0.4461%**；板接触速度梯度误差 **1.410 s⁻¹ / 0.9213%**，刀/粒子对的解析速度梯度为 0，绝对误差分别 ≤**1.458/2.533 s⁻¹**。接触切换的时间网格相位使梯度误差非单调：板接触 dt/32→dt/64 的最大速度梯度误差为 **0.5386→1.4098 s⁻¹**。

```bash
PYTHONPATH=. .venv/bin/python -m scripts.diff_contact_convergence \
  --output runs/e23_fb379d4/contact --spacings .006 .004 \
  --dt-divisors 1 8 32 64 --phases 0 .25 .5
PYTHONPATH=. .venv/bin/python - <<'PY'
from scripts.diff_contact_convergence import evaluate
cases = [('blade', .25, 64), ('blade', .5, 32), ('blade', .5, 64),
         ('pair', 0., 32), ('pair', .25, 64), ('pair', .5, 64)]
for surface, phase, divisor in cases:
    for spacing in (.006, .004):
        print(evaluate(surface, spacing, .00005 / divisor, .004, phase, [3e-10]))
PY
```

`a062351` 增加检查点的刀面进入侧、接触模式和网格求解批数核对。全量 `PYTHONPATH=. .venv/bin/python -m pytest tests -q`：**850 passed、8 skipped、18 subtests passed**，477.88 s；8 项跳过为 CRESSim 上游构建。

### 2026-10-22：M5 E3 细网格初态释放的时间步对照

`fb379d4`，RTX 4090、dirty=false：**4 mm、4892 粒子、G_c=7、50 ms**，步长 **13.1579/6.57895/3.28947 μs**。三档均新增 **98** 个侧码并释放 **5** 个粒子；六组刀路自然差分最大误差 **0.00001197%**，32 步检查点的全部连续状态重算差 **0**，损伤功平衡误差 ≤**3.05e-20 J**，进程 CUDA 数组内存池峰值 **2.609 GiB**。

最细相邻两档的刀位/行程梯度差为 **0.04425%/0.4679%**，E/摩擦/G_c 梯度差为 **0.08733%/0.001695%/1.4252%**；位置差 **41.23 nm**、速度差 **10.88 μm/s**、事件时刻差 **18.48 ns**、刀冲量差 **1.57e-8 N·s**。侧码、释放、接触对和几何事件粒子集合一致。行程梯度差从粗两档的 **2.869%** 降到 **0.4679%**。

```bash
PYTHONPATH=. .venv/bin/python -m scripts.diff_natural_refine \
  --output runs/e23_fb379d4/refine_50ms_h4_gc7 --from-initial \
  --trajectory cut-withdraw-push --duration .05 --spacings .004 --cutting-energy 7 \
  --dt-divisors 1 2 4 --adjoint checkpoint --segment-steps 32 --epsilons 1e-7
```

### 2026-10-22：M5 E3 完整自然切割基线

`fb379d4`，RTX 4090、dirty=false：**6 mm、G_c=700、0.24 s** 下刀的步长 **15.625/7.8125/3.90625 μs** 对照，刀位/行程梯度相邻差从 **0.1460%/0.9766%** 降为 **0.01013%/0.2428%**。最细两档末端位置差 **25.33 nm**、速度差 **2.597 μm/s**、事件时刻差 **31.57 ns**、刀冲量差 **2.556e-7 N·s**；E/摩擦梯度差 **0.005017%/0.004374%**。侧码、释放、接触对及事件粒子集合一致，均为 **726/726** 两块。细两档四组自然差分最大误差 **0.03185%**，32 步检查点状态重算差 **0**，损伤功平衡误差 ≤**8.68e-19 J**。

```bash
PYTHONPATH=. .venv/bin/python -m scripts.diff_natural_refine \
  --output runs/e23_fb379d4/refine_240ms_h6_gc700 --from-initial \
  --trajectory cut-withdraw-push --duration .24 --spacings .006 \
  --dt-divisors 2 4 --adjoint checkpoint --segment-steps 32 --epsilons 1e-7
```

`fb379d4`，RTX 4090、dirty=false：**6 mm、1452 粒子、G_c=700、1.05 s / 67200 步**，从未切割初态完成下刀、退刀、推块和再次退刀。末端按参考网格与侧码的机械连通性得到 **726/726** 两块；侧码新增 **1096**，损伤释放 **0**，峰值候选粒子接触对 **25**，峰值刀力 **12.3579 N**，`max|Fp−I|=0.137919`。

刀位/行程/横推梯度为 **−0.00869876627 / −1.88067358 / 0.532864613**；`1e-7` 自然差分误差为 **0.001071% / 0.0006306% / 0.01955%**，`1e-8` 为 **0.04597% / 0.0004850% / 0.001143%**。行程负向扰动改变一个粒子的分侧步号。**16/32/128** 步检查点的控制量/初始位置梯度最大差为 **5.49e-14 / 2.29e-12**，全部连续状态重算差 **0**；损伤功平衡误差 **2.17e-19 J**，进程 CUDA 数组内存池峰值 **5.485 GiB**。

屈服应力 **30 kPa**，末端目标对其倍数的梯度为 **−1.15993703**；`1e-7/1e-8` 自然差分误差为 **0.00005520%/0.0004225%**，扰动后的末端侧码及释放集合一致。

```bash
PYTHONPATH=. .venv/bin/python -m scripts.diff_natural_validate \
  --output runs/e23_fb379d4/initial_1050ms_gc700 --geometry event --from-initial \
  --trajectory cut-withdraw-push --duration 1.05 --spacings .006 \
  --adjoint checkpoint --segment-steps 16 32 128 --epsilons 1e-7 \
  --parameters knife_x_mm knife_depth_scale knife_lateral_scale
PYTHONPATH=. .venv/bin/python runs/e23_fb379d4/full_path_extra_fd.py
PYTHONPATH=. .venv/bin/python runs/e23_fb379d4/full_path_yield_fd.py
```

### 2026-10-22：M5 E3 完整轨迹的时间步对照

`fb379d4` 的基础步长与 `39741bf` 的细两档，均为 RTX 4090、dirty=false：**6 mm、G_c=700、1.05 s**，步长 **15.625/7.8125/3.90625 μs**。刀路梯度的最大相邻差由 **0.1848%** 降为 **0.05160%**；最细两档刀位/行程/横推分别相差 **0.005032%/0.008205%/0.05160%**，E/屈服应力/摩擦梯度差为 **0.006049%/0.0007495%/0.005536%**。

最细两档位置差 **0.8320 μm**、速度差 **2.078e-8 m/s**、Fp 差 **1.902e-6**、事件时刻差 **31.57 ns**、刀冲量差 **2.917e-6 N·s**。侧码、释放、接触对及事件粒子集合一致；损伤释放均为 **0**。128 步检查点的连续状态重算差 **0**，损伤功平衡误差 ≤**8.68e-19 J**，进程 CUDA 数组内存池峰值 **4.568 GiB**。

```bash
PYTHONPATH=. .venv/bin/python -m scripts.diff_natural_refine \
  --output runs/e3_39741bf/refine_1050ms_h6_gc700 --from-initial \
  --trajectory cut-withdraw-push --duration 1.05 --spacings .006 --dt-divisors 2 4 \
  --adjoint checkpoint --segment-steps 128 --epsilons \
  --parameters knife_x_mm knife_depth_scale knife_lateral_scale
```

### 2026-10-22：CPU 检查点存储

`a55b185`，RTX 4090、dirty=false：**4 mm、4892 粒子、G_c=7、50 ms / 3800 步、每段 8 步**，GPU/CPU 检查点的末端 x/v/F/Fp/应力/损伤差、事件时刻差及刀冲量差均为 **0**，两组均释放 **5** 个粒子。刀路梯度最大差 **1.53e-14**，材料梯度最大相对差 **4.41e-15**，全部检查点状态重算差 **0**。

峰值 CUDA 数组内存从 **1.906 GiB** 降到 **0.791 GiB**（**−58.5%**）；检查点正向、重算、反向累计为 **98.50/98.81 s**。CPU 存储通过 13 项新增检查；全量 `PYTHONPATH=. .venv/bin/python -m pytest tests -q`：**863 passed、8 skipped、18 subtests passed**，249.61 s。8 项跳过为 CRESSim 上游构建。

```bash
for storage in cuda cpu; do
  PYTHONPATH=. .venv/bin/python -m scripts.diff_natural_refine \
    --output runs/e3_a55b185/checkpoint_memory_${storage} --from-initial \
    --trajectory cut-withdraw-push --duration .05 --spacings .004 --cutting-energy 7 \
    --dt-divisors 1 --adjoint checkpoint --segment-steps 8 --epsilons \
    --checkpoint-device "$storage"
done
```

### 2026-10-22：M5 E3 细网格完整下刀

`39741bf`，RTX 4090、dirty=false：**4 mm、4892 粒子、G_c=70、0.24 s / 18240 步**，新增 **2632** 个侧码，**5** 个粒子损伤释放，峰值刀力 **0.715172 N**。刀位、行程、初始高度、E 和 G_c 各两档自然差分共 **10** 组通过，最大相对误差 **0.1014%**；刀路及材料各自最大误差 **0.01648%/0.004101%**。

32/128 步检查点的刀路/初始位置梯度最大差 **1.93e-14/1.36e-12**，连续状态重算差及损伤功平衡误差均为 **0**。进程 CUDA 数组内存池峰值 **15.02 GiB**。初始高度扰动 `1e-8 mm` 时，AD/左差分/右差分为 **−0.000710194/−0.000708936/−0.000710168**。

```bash
PYTHONPATH=. .venv/bin/python -m scripts.diff_natural_validate \
  --output runs/e3_39741bf/initial_240ms_h4_gc70 --geometry event --from-initial \
  --trajectory cut-withdraw-push --duration .24 --spacings .004 --cutting-energy 70 \
  --adjoint checkpoint --segment-steps 32 128 --epsilons 1e-7 1e-8 \
  --state-parameters initial_z_mm young_scale fracture_scale
```

### 2026-10-22：M5 E3 细网格完整轨迹的 CPU 检查点

`a55b185`，RTX 4090、dirty=false：**4 mm、4892 粒子、G_c=70、1.05 s / 79800 步**，从未切割初态完成下刀、退刀、推块和再次退刀。机械连通性得到 **2446/2446** 两块；新增 **2632** 个侧码，**5** 个粒子损伤释放，峰值候选接触对 **37**，峰值刀力 **11.3819 N**，`max|Fp−I|=0.192143`。

刀位/行程/横推梯度为 **−0.00684463284 / −1.80985156 / −0.152145023**，G_c 梯度为 **−1.52513171e-5**。CPU 存储、每段 **32** 步的检查点连续状态重算差及损伤功平衡误差均为 **0**；初始位置梯度范数 **268.677**。独立正向 **665.00 s**，检查点正向/重算/反向为 **690.48/740.65/289.56 s**；进程 CUDA 数组内存池峰值 **5.431 GiB**。

刀位/行程/横推/G_c 倍数的 `1e-7` 自然差分误差为 **0.005291% / 0.003057% / 0.00007528% / 0.07878%**，`1e-8` 为 **0.006858% / 0.0009211% / 0.001340% / 0.6647%**，八组全部通过。扰动后的末端侧码及释放集合一致；独立差分程序的未扰动末态核对通过，目标值差 **4.44e-16**。

```bash
PYTHONPATH=. .venv/bin/python -m scripts.diff_natural_refine --dt-divisors 1 \
  --output runs/e3_a55b185/initial_1050ms_h4_gc70 --geometry event --from-initial \
  --trajectory cut-withdraw-push --duration 1.05 --spacings .004 --cutting-energy 70 \
  --adjoint checkpoint --segment-steps 32 --checkpoint-device cpu --epsilons \
  --parameters knife_x_mm knife_depth_scale knife_lateral_scale
for task_index in 0 1 2 3; do
  PYTHONPATH=. .venv/bin/python runs/e3_a55b185/full_path_h4_fd.py "$task_index" &
done
wait
```

E3 汇总：上述 `fb379d4`、`39741bf`、`a55b185` 的 **50 组自然差分全部 ≤2%**，最大误差 **0.9185%**；**17** 组检查点的连续状态重算差均为 **0**。完整轨迹、活跃损伤释放及时间步对照通过，下一项为 E4 多刀与永久历史。

### 2026-10-22：M5 E4 跨刀历史与短轨迹梯度

`8d35d5c`，RTX 4090：方块/圆柱 × 平行/60°第二刀，共 **4** 条 24 步轨迹。第一刀的横向偏移与下切深度各做两档自然差分，**16** 组误差均 ≤**2%**；目标仅包含第二刀开始后的力及末端状态。完整 Tape 与 CPU 检查点（4/7 步）共 **8** 组梯度对照通过，连续状态重算差 **0**，旧切口侧码保留。

同刀暂停时损伤值与导数保留，新刀损伤值和旧损伤导数归零，既有塑性状态保留。多刀报告累计各刀损伤增量，计功平衡误差 <**1e-14 J**。新增 **11** 项检查；全量测试 **874 passed、8 skipped、18 subtests passed**，202.33 s；8 项跳过为 CRESSim 上游构建。

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests -q --tb=short
```

### 2026-10-22：M5 E4 完整暂停续切与平行两刀

`8d35d5c`，RTX 4090、dirty=false，**6 mm、1452 粒子、G_c=700**。暂停续切 **0.74 s / 47360 步**，最终 **726/726** 两块；第一刀横移/深度各两档自然差分共 **4** 组，最大误差 **0.01160%**。暂停、退刀和再次进入阶段的损伤总和保持 **0.5706745558166016**。

平行两刀 **1.04 s / 66560 步**，最终 **380/648/424** 三块；第一刀控制量到第二刀及末端目标的 **4** 组自然差分最大误差 **0.1855%**。第一刀释放 **1** 个粒子，第二刀当前损伤按规则重置，旧切口侧码逐步核对通过。

两任务的 **32/127 步 CPU 检查点**连续状态重算差均为 **0**，刀路梯度最大差 **5.84e-16**，初态位置梯度最大差 **7.43e-13**。暂停损伤变化 **0**，损伤计功平衡误差 ≤**5.64e-18 J**。独立正向耗时 **248.35/359.94 s**，峰值 CUDA 数组内存 **3.561/3.180 GiB**。报告和末态数组的有限数检查通过。

```bash
PYTHONPATH=. .venv/bin/python -m scripts.diff_multicut_validate \
  --output runs/e4_8d35d5c/resume --cases resume --spacings .006 \
  --segment-steps 32 127 --checkpoint-device cpu --epsilons 1e-6 1e-7
PYTHONPATH=. .venv/bin/python -m scripts.diff_multicut_validate \
  --output runs/e4_8d35d5c/two-cuts --cases two-cuts --spacings .006 \
  --segment-steps 32 127 --checkpoint-device cpu --epsilons 1e-6 1e-7
```

### 2026-10-22：M5 验收判定与多刀事件导出

`f60a5c4`，RTX 4090：新增 **26** 项检查，覆盖差分中的 NaN/Inf、缺失或重复扰动尺度、超差退出、检查点保存/重算状态中的 NaN、同粒子多刀事件导出及自定义路径的完整初态。定向 **40 passed**；全量 **900 passed、8 skipped、18 subtests passed**，229.27 s。8 项跳过为 CRESSim 上游构建。

已有 `8d35d5c` 暂停续切、平行两刀的报告用新判定复核通过，末态 x/v/F/Fp/应力/损伤及全部刀力样本均为有限数。多刀事件存入 `event_times_by_cut`，物理推进与梯度公式保持相同。

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests -q --tb=short
```

### 2026-10-22：M5 E4 完整交叉两刀

`8d35d5c`，RTX 4090、dirty=false，**6 mm、1452 粒子、G_c=700、1.04 s / 66560 步**。第二刀转动 **60°**；第一刀横移/深度各两档自然差分共 **4** 组，最大误差 **0.07377%**。最终分块粒子数 **726/354/372**，第二刀释放 **19** 个粒子，塑性变形 `max|Fp−I|=0.210871`。

**32/127 步 CPU 检查点**连续状态重算差均为 **0**，刀路梯度最大差 **5.90e-15**，初态位置梯度最大差 **1.34e-11**。暂停损伤变化 **0**，损伤计功 **0.0203520 J**，计功平衡误差 **2.78e-17 J**。独立正向 **363.09 s**，峰值 CUDA 数组内存 **3.774 GiB**；报告与末态数组通过 `f60a5c4` 的有限数及差分判定复核。

```bash
PYTHONPATH=. .venv/bin/python -m scripts.diff_multicut_validate \
  --output runs/e4_8d35d5c/crossing --cases crossing --spacings .006 \
  --segment-steps 32 127 --checkpoint-device cpu --epsilons 1e-6 1e-7
```

### 2026-10-22：M5 E5 材料能量与解析检查

`53e417b`，RTX 4090：新增 **25** 项能量检查通过。独立 NumPy 计算动能、重力势能和共旋弹性储能；4 组含塑性状态的形变，各用两档扰动核对储能方向导数与求解器应力虚功。自然切割报告记录起终点材料能量。

自由落体 **2 ms**、步长 **0.2/0.1/0.05 ms**：中点积分的总能量偏差均 ≤**1e-16 J**；半隐式积分的偏差符合 \(-\frac12 M|g|^2T\Delta t\)。全量 **925 passed、8 skipped、18 subtests passed**，310.43 s；8 项跳过为 CRESSim 上游构建。下一项为接触储能、刀具输入功、塑性与摩擦及传递耗散的分项收支。

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_energy.py -q --tb=short
PYTHONPATH=. .venv/bin/python -m pytest tests -q --tb=short
```

### 2026-10-22：M5 E5 刀具输入功与返回映射核对

`76f9a23`，RTX 4090：刀具输入功新增 **18** 项检查通过，覆盖静止/平移/旋转刀、网格约束冲量矩、事件分段累计、完整 Tape/CPU 检查点及两档自然差分。自然报告增加 `knife_work_j`；全量 **943 passed、8 skipped、18 subtests passed**，**322.54 s**。

塑性耗散核对失败：固定 F、初始 Fp=I，E=20 kPa、ν=0.35、屈服=200 Pa；F 的主伸长 **[0.55,0.4,0.3] / [0.2,0.1,0.05]**，J=**0.066 / 0.001**，当前投影令共旋储能分别增加 **94.0385 / 574.0021 J/m³**。同参数 **[1.1,0.95,0.95]** 的储能下降 **110.0832 J/m³**。

独立 NumPy/SciPy 对照按等容对数塑性增量最小化“共旋储能＋塑性耗散距离”，上述两组强压缩的储能下降 **1399.9545 / 10030.5700 J/m³**，塑性体积误差 ≤**5e-16**，驻点残差 ≤**1.15e-8 Pa**。方法依据为 [Miehe 等的增量能量最小化](https://doi.org/10.1016/S0045-7825(02)00438-3)；下一项为 Warp 返回映射及屈服/重复主伸长/耗散/导数核对，再复测受影响的切割轨迹。

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_tool_work.py -q
PYTHONPATH=. .venv/bin/python -m pytest tests -q
PYTHONPATH=. .venv/bin/python runs/e5_plastic_probe.py
PYTHONPATH=. .venv/bin/python runs/e5_variational_probe.py
```

### 2026-10-22：M5 E4 从初态展开的 4 mm 交叉窗口

`f60a5c4`，RTX 4090、dirty=false，**4 mm、4892 粒子、G_c=700、0.64 s / 48640 步**。完整第一刀、退刀与 60° 转动后，推进第二刀前 **80 ms**；第一刀横移/深度各两档自然差分共 **4** 组，最大误差 **0.02062%**。两刀分侧事件分别 **2632/162** 个；CPU/32 步检查点连续状态重算差 **0**。

暂停损伤变化 **0**；第二刀末累计损伤 **1.989058**、释放 **1** 个粒子，`max|Fp−I|=0.033190`。损伤计功 **0.000909503 J**，计功平衡误差 **2.17e-18 J**；独立正向 **511.85 s**，峰值 CUDA 数组内存 **5.046 GiB**。报告有限数、四组差分及逐刀事件与最终侧码对应关系复核通过。

```bash
PYTHONPATH=. .venv/bin/python runs/e4_f60a5c4/crossing_window.py
```

### 2026-10-22：M5 E5 共旋能量一致的塑性返回

`43a939a`，RTX 4090：原求解器与可微流程共用等容增量变分返回；以驻点方程隐式求导。新增 **27** 项返回映射检查及 **3** 项实际材料更新能量检查通过。400 组随机体积/偏形变检查耗散、屈服和体积保持；3 组屈服边界的扰动从 **1e-6** 到 **1e-12**；旋转、重复/近重复伸长及强压缩的矩阵 VJP 用两档自然差分核对。原/可微求解器状态对照、塑性切割事件梯度和可选 Maxwell 压入回归通过。

干净提交上的 E=20 kPa、ν=0.35、屈服=200 Pa 材料更新：主伸长 **[0.55,0.4,0.3] / [0.2,0.1,0.05]** 的储能下降 **1399.9545 / 10030.5700 J/m³**；**[1.1,0.95,0.95]** 下降 **110.0823 J/m³**，与独立 SciPy 最小化一致。全量 **973 passed、8 skipped、18 subtests passed**，**300.35 s**；8 项跳过为 CRESSim 上游构建。

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_plasticity.py tests/test_energy.py -q
PYTHONPATH=. .venv/bin/python -m pytest tests -q --tb=short
PYTHONPATH=. .venv/bin/python runs/e5_plastic_probe.py
```


### 2026-10-22：M5 E5 切向摩擦储能与耗散

`109fe3f`，RTX 4090：原求解器与可微接触共用 Kelvin–Voigt–滑动元件更新，原/M4 使用 θ=1，M5 中点使用 θ=1/2。**27** 项接触能量检查通过，覆盖随机法向和状态、三档阻尼、零摩擦、解析松弛、两档状态/参数差分与实际接触核；随机增量的储能/耗散恒等式残差 ≤**2e-18 J**。接触振子状态与梯度的步长减半误差下降 **>3.8 倍**，粘滑切换、自然事件及检查点检查通过。

同参数单接触：m=0.01 kg、kₙ=1000 N/m、kₜ=500 N/m、阻尼比 0.1、间隙 −0.1 mm、切向速度 0.5 m/s。`43a939a` 的零摩擦输入功为 0，新增储能 **5e-5 J**；`109fe3f` 干净提交在 **100/50/25 μs** 三档步长下的输入功、储能和耗散均为 **0**。μ=0.2、100 μs 时，输入功 **1e-6 J**、储能 **4.48459e-9 J**、耗散 **9.95515e-7 J**。

切向步长检查使用新更新矩阵，稳定条件为 `d>0、γk+2d<4`，d=2γc√γk；新增一个边界测试拒绝谱半径大于 1 的步长。旋转做功梯度以基准做功归一化，两个自然差分仍使用 rtol=2e-4。全量 **1003 passed、8 skipped、18 subtests passed**，**271.00 s**；8 项跳过为 CRESSim 上游构建。整体轨迹分项能量和 G_c 扫描继续执行。

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_friction_energy.py tests/test_differentiable_midpoint.py tests/test_tangent_history_step.py -q
PYTHONPATH=. .venv/bin/python -m pytest tests -q
PYTHONPATH=. .venv/bin/python runs/e5_friction_probe.py
```

## 2026-10-22：摩擦更新后的自然事件窗口复验（`109fe3f`）

RTX 4090，干净提交，6 mm 香蕉 1452 粒子、2 ms 窗口，dt=15.625/7.8125/3.90625 μs，时间网格相位 0/0.25/0.5，固定 PIC 速率 3200 s⁻¹。36 项自然差分的最大相对误差 **0.034311%**；32/127 步检查点重算状态差 **0**，刀路梯度差 ≤**7.61e-15**。最细两档刀路梯度差 ≤**0.17884%**、位置差 ≤**1.303e-9 m**。9 份末态数组均为有限值；每个窗口新分侧 68 粒子，其余粒子的事件时刻按报告约定为空。

```bash
PYTHONPATH=. .venv/bin/python -m scripts.diff_natural_refine --output runs/e5_109fe3f/banana_h6 --spacings .006 --materials banana --duration .002 --dt-divisors 4 8 16 --phases 0 .25 .5 --adjoint checkpoint --segment-steps 32 127 --checkpoint-device cpu --epsilons 1e-6 1e-7 --integration midpoint --friction-update midpoint --flip-reference-dt .000015625
```

## 2026-10-22：E4 完整三刀后推块（`f60a5c4`）

RTX 4090，干净提交，6 mm 香蕉 1452 粒子，2.18 s / 139520 子步，最终四块 **380/346/302/424** 粒子。第一刀横移与深度的两档自然差分共 4 项，最大误差 **0.148941%**；32/127 步 CPU 检查点重算状态差 **0**，刀路梯度差 **2.193e-15**。每刀标签 **968/971/954**；暂停段损伤变化 **0**，最终本刀释放 **1** 粒子，Fp−I 最大值 **0**。累计损伤功 **0.0064744463 J**，QΔD 残差 **1.735e-18 J**；峰值刀力 **3.14566 N**，峰值活动显存池 **3.344 GiB**。

```bash
PYTHONPATH=. .venv/bin/python -m scripts.diff_multicut_validate --output runs/e4_f60a5c4/three-cuts-push --cases three-cuts-push --spacings .006 --segment-steps 32 127 --checkpoint-device cpu --epsilons 1e-6 1e-7
```

## 2026-10-22：E5 接触功配对与屈服边界修复（`ec7bcb4`）

中点接触使用位置推进的插值速度，未切开/部分切开/交叉侧码的虚功和转置导数通过独立 NumPy 核对；非均匀速度场在 12.5/6.25/3.125 μs 下的能量误差逐档下降超过 **3.5 倍**。塑性初始线搜索保留原阈值和驻点容差，采用避免浮点抵消的等价能量差；1792 个近屈服状态从 **7** 个失败变为 **0**。全量 **1017 passed、8 skipped、18 subtests passed**（290.38 s）。

RTX 4090、干净提交、50 ms 塑性方块：静置 dt=15.625/7.8125/3.90625 μs 的总能量残差分别为 **−6.687e-12/−4.252e-12/−1.385e-12 J**；压入前两档为 **1.770e-10/1.202e-10 J**（**0.001116%/0.000758%**）。`33a7001` 同参数压入的三档残差约 **−3.31e-7 J（−2.17%）**，主要来自摩擦计算与材料传递的速度功差。

损伤释放能量核对失败：6 mm 香蕉 1452 粒子、G_c=7、50 ms、4 粒子释放，dt=15.625/7.8125 μs 的总残差分别为 **3.941532e-4/3.941602e-4 J（23.5995%/23.5999%）**。释放前后刃向冲量的功差为 **3.941671e-4/3.941648e-4 J**；几何分侧能量跳变约 **6.75e-16 J**。原因是刃向冲量移除后，接触储能和摩擦历史仍按移除前的接触律更新。下一项在单接触上统一释放后的实际力、储能与耗散，再复测自然梯度和阈值扫描。

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests -q
PYTHONPATH=. .venv/bin/python -m scripts.diff_energy_validate --output runs/e5_current/block_plastic --shape block --material plastic --scenes rest press cut --duration .05 --dt-divisors 1 2 4 --cutting-energies 7 70 700 1400
PYTHONPATH=. .venv/bin/python -m scripts.diff_energy_validate --output runs/e5_current/banana_release --shape banana --material banana --scenes cut --duration .05 --dt-divisors 1 2 4 --cutting-energies 7 70 700 1400
```

## 2026-10-22：E4 塑性交叉复验与旧修复核对

`43a939a`，RTX 4090、dirty=false，6 mm 香蕉、1.04 s / 66560 步。横移 ε=1e-6/1e-7 mm 的 AD/FD 相对误差为 **8.02884%/13.94012%**，深度对应 **7.53189%/0.013684%**，3 项失败。32/127 步检查点重算状态差 **0**，控制量梯度差 **5.684e-14**。横移小扰动的 AD/左差分/右差分为 **−0.01440889/−0.01445391/−0.01034664**。逐段重放的基准目标精确复现，所有材料阶段的网格约束求解次数 **0**。

```bash
PYTHONPATH=. .venv/bin/python -m scripts.diff_multicut_validate --output runs/e4_43a939a/crossing --cases crossing --spacings .006 --segment-steps 32 127 --checkpoint-device cpu --epsilons 1e-6 1e-7
PYTHONPATH=. .venv/bin/python runs/e4_43a939a/trace_crossing.py
```

`43a939a` 保留 `f60a5c4` 的有序 P2G、双精度转换、中点推进和几何事件求根。当前公式的精度/中点/几何/材料 VJP 复核 **105 passed**（37.32 s）。`ca90da9` 增加显式 dt/PIC 参考时间参数，固定接触刚度及 PIC 速率核对通过；全量 **1017 passed、8 skipped、18 subtests passed**（273.94 s）。

三条完整横移轨迹定位到 **0.696765625 s / 第 44593 步**：粒子 1167 的基准/负扰动/正扰动刀距为 **−4.564e-12/−1.374e-11/+4.609e-12 m**，旧端点摩擦回算产生 **1.372e-6/1.372e-6/0 m** 的历史位移。接触起始处的历史跳变传入后续刀力；此时三条轨迹的塑性、释放及侧码数量相同。`109fe3f` 已删除这条端点回算，当前接触网络在零初始历史下满足 `h₁=Δt μFₙ t/(c+θΔt kₜ)`，随法向载荷趋零；θ=1/2、1 与三档阻尼共 **6** 个解析连续性检查通过。命令：`PYTHONPATH=. .venv/bin/python runs/e4_43a939a/trace_jump.py`、`PYTHONPATH=. .venv/bin/python -m pytest tests/test_friction_energy.py -q`（**33 passed**）。

`ca90da9`，RTX 4090、dirty=false，固定 `--dt .000015625 --flip-reference-dt .000015625` 的当前公式复验：横移 ε=1e-6/1e-7 mm 的 AD/FD 相对误差为 **0.00094923%/0.00033584%**，对应旧值 **8.02884%/13.94012%**。深度两档误差 **0.0112984%/0.0371589%**，四项全部通过；32/127 步检查点状态重算差 **0**，刀路/初始位置梯度差 **2.297e-15/3.898e-12**。暂停损伤增量 **0**，终态三块 **726/360/366** 粒子、释放 **36** 粒子。报告：`runs/e4_ca90da9/crossing/report.json`，命令其余参数同本节的 `diff_multicut_validate`。

`d221ad4`、dirty=true（追加上述 6 项测试及文档），RTX 4090：`PYTHONPATH=. .venv/bin/python -m pytest tests -q`，**1023 passed、8 skipped、18 subtests passed**（254.58 s）。

## 2026-10-22：E5 完整短轨迹阈值扫描与释放力闭合路径

`ec7bcb4`，RTX 4090、dirty=false，50 ms、dt=15.625/7.8125/3.90625 μs，固定 PIC 参考时间 15.625 μs。塑性方块静置/压入及 G_c=7/70/700/1400 切割共 **18** 例完成；压入相对能量残差 **0.001116%/0.000758%/0.000375%**，各阈值的切割残差 **0.014090%/0.007980%/0.004001%**，释放数均为 **0**。香蕉四阈值共 **12** 例完成：G_c=7/70 各释放 **4** 粒子，三档残差分别 **23.59953%/23.59991%/23.60007%** 和 **7.30799%/7.30844%/7.30862%**；G_c=700/1400 未释放，残差幅值均为 **0.000594%/0.000182%/0.0000579%**。

```bash
PYTHONPATH=. .venv/bin/python -m scripts.diff_energy_validate --output runs/e5_current/block_plastic --shape block --material plastic --scenes rest press cut --duration .05 --dt-divisors 1 2 4 --cutting-energies 7 70 700 1400
PYTHONPATH=. .venv/bin/python -m scripts.diff_energy_validate --output runs/e5_current/banana_release --shape banana --material banana --scenes cut --duration .05 --dt-divisors 1 2 4 --cutting-energies 7 70 700 1400
```

同提交单接触：μ=0、kₙ=4752 N/m、接触距离 1.5 mm，刃尖前方矩形路径 x=0.3–0.4 mm、z=−0.3 至 −0.2 mm。保持 D 不变，每边 128/256/512 个采样；未释放闭环净功 **1.034e-10/2.584e-11/6.460e-12 J**，已释放为 **7.785586e-5/7.785585e-5/7.785584e-5 J**。反向路径净功变号，释放力规则的能量失败来自移除刃向分量。下一项将断裂阻力与刀面压缩接触分开，先核对储能、非负耗散和局部导数，再接入自然轨迹。

```bash
PYTHONPATH=. .venv/bin/python runs/e5_release_cycle.py
```

## 2026-10-22：E5 双线性内聚界面单元

`ec7bcb4`，RTX 4090、dirty=false，独立单元使用 [Camanho–Dávila 双线性界面形式及 MOOSE 实现](https://mooseframework.inl.gov/releases/moose/2022-06-10/source/materials/cohesive_zone_model/BiLinearMixedModeTraction.html)：最大开裂位移保存不可逆损伤，压缩刚度独立保留。测试刚度 **5e8 Pa/m**、强度 **5e4 Pa**，G_c=7/70/700/1400 的加载—卸载—完全断裂功分别等于 **7/70/700/1400 J/m²**，最大能量绝对残差 **2.615e-11 J/m²**；断裂后的 −0.03 mm 压缩对应 **−15 kPa**。

192 组状态的开裂位移、历史和三个材料参数自然差分（ε=1e-5/1e-6/1e-7）最大相对误差 **1.653e-8**；两步加载历史到卸载力的梯度误差 **5.979e-9**。128 组刚体旋转与储能虚功通过。混合开裂/剪切路径采用 128/256/512 个分段时，积分功残差 **−2.872e-4/−7.180e-5/−1.795e-5 J/m²**，每次细化约降 4 倍。下一项将界面连接运行时生成的切口，核对插入能量、双侧力及刀路导数。命令：`PYTHONPATH=. .venv/bin/python runs/e5_release_audit/cohesive_unit.py`。

## 2026-10-22：E5 双侧界面与 TLMPM 耦合

`d4bbb6f`，RTX 4090、dirty=false，64 粒子、12 mm 方块、6 mm 网格、E=20 kPa、纯 FLIP、无重力。平面界面 4 个积分点、参考面积 **1.44e-4 m²**，测试 K=5e8 Pa/m、σ=5e4 Pa；初始双侧分离速度 4 m/s、动能 **0.0038016 J**。G_c=7/70、0.4 ms、dt=2/1/0.5 μs 共 6 例，G_c=7 完全断裂耗散为 **0.001008 J=G_c A₀**。再取 dt=0.5/0.25/0.125 μs、三档时间网格相位共 18 例，G_c=7 的最大相对能量残差为 **7.659e-6/1.520e-6/2.131e-7**，G_c=70 为 **7.213e-6/1.482e-6/2.326e-7**。160 步轨迹的初速度、内聚参数和 E 自然差分最大相对误差 **2.400e-7**。

```bash
PYTHONPATH=. .venv/bin/python runs/e5_release_audit/cohesive_grid.py
PYTHONPATH=. .venv/bin/python runs/e5_release_audit/cohesive_grid_refine.py
```

`769dd89`，RTX 4090、dirty=false，旋转/剪切界面：初始开口 **(−0.03,0.2,0.1) mm**、角速度 **(30,−20,200) s⁻¹**，0.2 ms，dt=1/0.5/0.25 μs。G_c=7 的相对能量残差 **−1.787e-5/−1.936e-7/+1.372e-7**，G_c=70 为 **−1.565e-5/−7.232e-7/−1.817e-7**。160 步整体旋转对照的状态相对差 ≤**5.22e-15**；卸载初态 κ=0.3 mm 下，初速度、界面位置/变形、损伤历史、内聚参数及 E 的两档自然差分最大相对误差 **8.144e-7**（E 用 ε=1e-4/1e-5，其余 1e-5/1e-6）。

```bash
PYTHONPATH=. .venv/bin/python runs/e5_release_audit/cohesive_surface_grid.py
```

同提交 192 个有限变形界面状态：力/表面应力与增量势导数的相对差 **1.577e-16/2.622e-16**，刚体旋转虚功绝对残差 ≤**1.137e-13**。加载边界 κ=0.2/0.3/1 mm 的左右导数分别符合卸载刚度与软化斜率；两者不同，力连续。命令：`PYTHONPATH=. .venv/bin/python runs/e5_release_audit/cohesive_frame.py`、`PYTHONPATH=. .venv/bin/python runs/e5_release_audit/cohesive_loading_corner.py`。

`d4bbb6f`、dirty=true（新增 `cohesive.py` 与 15 项测试），RTX 4090：`PYTHONPATH=. .venv/bin/python -m pytest tests -q`，**1045 passed、8 skipped、18 subtests passed**（249.28 s）。

## 2026-10-22：E5 受载界面生成与耦合约束原型

`769dd89`，RTX 4090、dirty=false。初始刚性内聚原型在零开口下传递低于强度的拉/剪载荷及压缩载荷，新增储能与耗散均为 **0**。σ=50 kPa、G_c=7/70/700/1400 的加载—卸载—断裂功分别为 **7/70/700/1400 J/m²**，最大功残差 **5.685e-14 J/m²**。256 个随机状态的归一化最小能量解误差 ≤**3.787e-8**，平衡残差 ≤**5.335e-15**；状态、法向、历史和参数在 ε=1e-6/1e-7 下的 AD/FD 最大相对误差 **1.764e-9**。ε=1e-5 的参数扰动跨过 1 个起裂分支，误差 **0.4221%**；起裂点左右切线与解析式一致，牵引连续。

双侧质量折合 **2 kg/m²**，先施加低于强度的预载，再拉开至断裂；四档 G_c、每特征时间 128/256/512 步，共 **12** 例后向 Euler 推进。最大相对能量残差 **2.376e-5/4.138e-6/8.360e-7**，材料、质量、步长的两档轨迹梯度最大误差 **7.607e-8**。

64 粒子、4 个界面的实际 TLMPM 传力矩阵功配对误差 **5.552e-17**。全矩阵耦合求解在 32/64/128/256 次迭代的平衡相对残差为 **1.138e-3/6.276e-6/3.651e-10/4.101e-16**。自由开口、法向、历史、材料及耦合矩阵均通过至少两档差分；ε=1e-4/1e-5 最大误差 **3.211e-7**。历史方向的 AD/独立隐式方程导数为 **0.02862841554856842/0.02862841554856843**；细差分 ε=1e-6/1e-7 的相对误差为 **4.415e-6/5.285e-5**。上述原型位于 `runs/`；下一项将参考面生成、移动面法向和多界面约束接入自然切割。

```bash
PYTHONPATH=. .venv/bin/python runs/e5_release_audit/extrinsic_prox.py
PYTHONPATH=. .venv/bin/python runs/e5_release_audit/extrinsic_dynamics.py
PYTHONPATH=. .venv/bin/python runs/e5_release_audit/extrinsic_coupled.py
```

## 2026-10-22：E5 冗余界面约束与粒子推进

`6361c84`，RTX 4090、dirty=false。64 粒子、16 个面积分点的 TLMPM 矩阵秩为 **9**；ADMM 的开口—动量平衡相对残差 **3.873e-17**。将积分点复制为 32 个并各分一半面积，粒子速度保持一致（绝对差 **4.441e-16 m/s**）；ρ 乘 0.5/1.5 后同样一致。速度、材料、插值、面积、质量、时长的两档自然差分最大相对误差 **9.213e-9**。局部核现为 `cohesive_proximal_response`，14 项单元测试覆盖有限预载、断裂功、约束驻点、旋转和梯度。

同一 64 粒子、12 mm 方块，16 点平面、E=20 kPa、纯 FLIP，接入实际 `MaterialTrajectory.step`。0.2 ms、dt=2/1/0.5 μs，G_c=7/70 拉开及 G_c=70 受载闭合共 **9** 例。拉开相对能量残差分别 **0.005789/0.002888/0.001443** 和 **−0.004990/−0.002493/−0.001246**；G_c=7 完整断裂耗散为 **0.001008 J=G_c A₀**。初始 F_xx=1.05 的受载例保持开口/损伤/面能为 **0**，整体能量残差 **−0.0004931/−0.0002466/−0.0001233**。每步约束 16–48 次迭代，最大平衡相对残差 ≤**1e-11**。

24/160 步轨迹的初速度、内聚参数、E 和时长在 ε=1e-4/1e-5 下的最大梯度相对误差 **1.347e-6**；报告保留四档 ε=1e-4 至 1e-7。24 步的弱 E 导数在后两档误差为 **7.550e-5/6.756e-4**，160 步为 **6.609e-6/3.583e-5**。

```bash
PYTHONPATH=. .venv/bin/python runs/e5_release_audit/extrinsic_admm.py
PYTHONPATH=. .venv/bin/python runs/e5_release_audit/extrinsic_particle.py
PYTHONPATH=. .venv/bin/python -m pytest tests/test_cohesive_proximal.py -q
```

`769dd89`、dirty=true（新增局部近端核及 14 项测试），RTX 4090：`PYTHONPATH=. .venv/bin/python -m pytest tests -q`，**1059 passed、8 skipped、18 subtests passed**（274.88 s）。

## 2026-10-22：E5 移动面压力与闭合约束

`6361c84`、dirty=true（新增移动面压力核及 4 项测试），RTX 4090。`cohesive_contact_response` 的压力牵引和面应力通过独立增量势虚功、64 组旋转、全部输入自然差分与秩 2 面映射检查。全量 `PYTHONPATH=. .venv/bin/python -m pytest tests -q`：**1063 passed、8 skipped、18 subtests passed**（300.46 s）。

64 粒子、16 点界面、G_c=70、0.2 ms，初始切向开口为 **0.2/0.1 mm**，法向初速度满足闭合约束，两侧加速度为 **±2000 m/s²**，叠加角速度 **(30,−20,200) rad/s**。dt=2/1/0.5 μs 的相对能量残差为 **−0.006207/−0.003104/−0.001552**；去掉面转动共轭力时为 **−0.009538/−0.006444/−0.004896**。160 步整体坐标旋转的归一化状态差 ≤**9.736e-15**。闭合前 24 步轨迹的八类输入在 ε=1e-4/1e-5 下梯度误差 ≤**4.997e-6**；弱 E 导数在 ε=1e-7 下误差为 **1.704e-4**，四档差分均保留。

移动面 160 步自然差分失败：初速度扰动轨迹第 **46** 步开始闭合，ADMM 512/8192 次迭代的平衡相对残差为 **2.532e-8/2.406e-8**。同一输入、同一增量势的 SciPy SLSQP 与 NumPy 驻点细化得到 **9/16** 个承压点，其余点正常开口 **5.424e-12 至 1.434e-10 m**，动量—开口残差 **3.026e-19**。下一项将主动集求解和对应导数接入原型，重跑该扰动及完整梯度。

```bash
PYTHONPATH=. .venv/bin/python runs/e5_release_audit/extrinsic_surface.py
PYTHONPATH=. .venv/bin/python runs/e5_release_audit/extrinsic_surface_validate.py --steps 24 --output runs/e5_release_audit/extrinsic_surface_short.json
PYTHONPATH=. .venv/bin/python runs/e5_release_audit/extrinsic_surface_validate.py --steps 160
PYTHONPATH=. .venv/bin/python runs/e5_release_audit/surface_failed_primal.py
```


## 2026-10-22：E4 当前公式的暂停续切与 4 mm 交叉窗口

`ca90da9`，dirty=false，RTX 4090。6 mm 暂停续切为 1452 粒子、0.74 s、47360 步，横移 ε=1e-6/1e-7 mm 的自然差分误差 **0.0007693%/0.0051131%**，深度 **0.0024713%/0.0002709%**。32/127 步检查点的状态重算差 **0**，刀路梯度差 **1.960e-16**；暂停损伤增量 **0**，最终两块各 **726** 粒子。

4 mm 交叉窗口为 4892 粒子、0.64 s、48640 步，包含第一刀、退刀、转向和第二刀前 80 ms。横移两档自然差分误差 **0.0069135%/0.0092038%**，深度 **0.0059460%/0.0088622%**；32 步 CPU 检查点状态重算差和暂停损伤增量均为 **0**，**2** 粒子释放，最大塑性变形量 **0.053064**，数组显存峰值 **5.198 GiB**。

```bash
PYTHONPATH=. .venv/bin/python -m scripts.diff_multicut_validate --output runs/e4_ca90da9/resume --cases resume --spacings .006 --dt .000015625 --flip-reference-dt .000015625 --segment-steps 32 127 --checkpoint-device cpu --epsilons 1e-6 1e-7
PYTHONPATH=. .venv/bin/python runs/e4_ca90da9/crossing_window.py
```

## 2026-10-22：E5 移动面联合约束与轨迹核对

`11fbbd5`，dirty=false，RTX 4090，原型位于 `runs/e5_release_audit/`。64 粒子、16 面积分点的移动面联合约束在一个承压点的局部算例中，对开口、面映射、参考法向、历史、材料、插值、面梯度、面积、质量和时长共 **10** 类输入各通过两档自然差分；最差第二档相对误差 **9.184e-9**。原第 46 步停滞的初速度扰动完整推进 **160** 步，最大动量平衡相对残差 **2.647e-19**。

同一增量势采用代数等价的稳定能量差，SLSQP 提供初值、Newton 驻点保持 **1e-11** 的最终残差要求。该 160 步正向算例耗时 **11.70 s**，与直接能量相减版本的终态能量差 ≤**6.1e-18 J**。源代码全量测试 **1063 passed、8 skipped、18 subtests passed**（305.99 s）。

整段梯度验收失败：160 步初速度方向 ε=1e-4/1e-5/1e-6/1e-7 的自然差分相对误差为 **0.40448/0.64981/5.704e-6/4.443e-5**，原标准要求至少两档 ≤**2e-5**。坐标旋转后的归一化速度差 **2.464e-10**，超过 **1e-10**。下一项逐步核对接触集合、约束条件数及隐式导数，再完成全部输入扫描；自然切口接入和完整能量扫描随后进行。

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python runs/e5_release_audit/extrinsic_monolithic_validate.py
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python runs/e5_release_audit/extrinsic_monolithic_replay.py --steps 160 --initial-tolerance 1e-8 --output runs/e5_release_audit/extrinsic_monolithic_seed8.json
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python runs/e5_release_audit/extrinsic_trajectory_gradients.py --steps 160 --solver monolithic --initial-tolerance 1e-8
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python runs/e5_release_audit/extrinsic_rotation_audit.py
PYTHONPATH=. .venv/bin/python -m pytest tests -q
```

## 2026-10-22：E5 约束位移更新与移动面轨迹

`11fbbd5`、dirty=false，RTX 4090，`extrinsic_primal_validate.py`：64 粒子、16 面积分点、160 步，8 类输入各四档自然差分，至少两档相对误差 ≤**2e-5**；最差第二档为初速度的 **1.5073e-5**。初速度四档误差为 **0.404486/0.649818/4.2691e-6/1.5073e-5**；旋转归一化状态差 **5.6409e-11**，标准 **1e-10**。原型用稳定增量求和及全局约束位移直接更新粒子，每步与网格冲量推进对照 x/v/F/Fp/应力，原四档扰动和验收阈值保持不变。

`a7df3f4` 整理 `MaterialTrajectory.correct_displacement`；提交前在 `11fbbd5`、dirty=true 的对应源码上，13 项未切开/部分切开/交叉侧码、弹性/塑性及 PIC/FLIP 状态与 VJP 对照通过。全量 **1076 passed、8 skipped、18 subtests passed**（265.79 s）。8 项跳过为 CRESSim 上游构建。干净 `a7df3f4` 的整段界面复跑已启动；自然切口接入及整体能量扫描随后运行。

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python runs/e5_release_audit/extrinsic_primal_validate.py
PYTHONPATH=. .venv/bin/python -m pytest tests/test_displacement_correction.py -q
PYTHONPATH=. .venv/bin/python -m pytest tests -q
```

## 2026-10-23：当前公式的 E3 活跃释放与 E5 源码复跑

干净 `ca90da9`，RTX 4090，6 mm、50 ms、G_c=7、dt/PIC 参考时间 15.625 μs：**4** 个粒子损伤释放，**12** 组刀路/初态/材料/摩擦/G_c 自然差分全部通过，最大误差 **0.61668%**。完整 Tape 与 16/32/128 步检查点的控制量/初始位置梯度最大差 **5.912e-15/7.704e-13**，状态重算差 **0**；CUDA 数组峰值 **13.710 GiB**。报告 `runs/e3_ca90da9/release_h6/report.json`；命令 `PYTHONPATH=. .venv/bin/python runs/e3_ca90da9/run_case.py release_h6`。4 mm 释放与后续完整轨迹继续运行。

干净 `a7df3f4` 的 `extrinsic_primal_validate.py` 源码复跑完成：160 步、8 类输入共 32 组自然差分，各类至少两档相对误差 ≤**2e-5**，最差第二档 **1.5073e-5**；旋转归一化状态差 **5.6409e-11**。报告 `e5-correction` 工作树的 `runs/e5_release_audit/extrinsic_primal_validate.json`，命令同上一节。

## 2026-10-23：内聚界面闭合时的卸载导数

`ca283ec` 保留零开口卸载的解析增益 `κ/[κ+λσ(1−κ/δ_f)]`，并用同一有理式计算牵引与储能。原零长度分支在已损伤闭合状态返回开口导数 **0**，解析值分别为 **0.76923/1**（部分损伤/完全断裂）。新增 **6** 个闭合切线算例和 **1** 个零历史单侧导数算例通过；内聚近端核与移动面压力共 **25 passed**（1.37 s）。

提交前的 `a7df3f4`、dirty=true 对应源码在 RTX 4090 全量 **1083 passed、8 skipped、18 subtests passed**（286.64 s）；8 项跳过为 CRESSim 上游构建。命令：`PYTHONPATH=. .venv/bin/python -m pytest tests/test_cohesive_proximal.py tests/test_cohesive_contact.py -q`、`PYTHONPATH=. .venv/bin/python -m pytest tests -q`。

## 2026-10-23：初始刚性界面的统一局部求解与导数

`a7df3f4`、dirty=true（对应 `ca283ec` 的内聚核修复），RTX 4090，闭合/零历史起裂/混合历史/已有损伤各 4 档载荷共 **16** 例通过；解析残差 Jacobian 的独立差分最大相对误差 **1.3821e-9**，数值分裂尺度 0.5/1/2 倍给出相同粒子解。4 类状态 × 10 类输入的 **40** 个方向各有至少两档自然差分相对误差 ≤**2e-5**，最差第二档 **1.3519e-7**；完整 **160** 行结果保存原始差分以及 `F_free+G u` 的基态/增量分开相减值。连续粒子轨迹随后核对。

命令：`OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python runs/e5_release_audit/extrinsic_semismooth_probe.py`、`OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python runs/e5_release_audit/extrinsic_semismooth_validate.py`。报告为同名 `.json`，工作树见 HANDOFF。

## 2026-10-23：当前公式的 E4 平行两刀

干净 `ca90da9`，RTX 4090，6 mm、1.04 s、66560 步，dt/PIC 参考时间 15.625 μs。横移两档自然差分相对误差 **0.0002968%/0.0001523%**，深度 **0.07879%/0.07261%**；32/127 步检查点状态重算差 **0**、控制量/初始位置梯度差 **3.903e-16/7.434e-13**。最终 **380/648/424** 三块，暂停损伤增量 **0**，CUDA 数组峰值 **3.288 GiB**。

命令：`PYTHONPATH=. .venv/bin/python -m scripts.diff_multicut_validate --output runs/e4_ca90da9/two-cuts --cases two-cuts --spacings .006 --dt .000015625 --flip-reference-dt .000015625 --segment-steps 32 127 --checkpoint-device cpu --epsilons 1e-6 1e-7`。报告 `runs/e4_ca90da9/two-cuts/report.json`。

## 2026-10-23：统一内聚残差的反向与连续轨迹复验

干净 `ca283ec`，RTX 4090，隐式反向 Jacobian 直接由 Warp 残差构建，局部 **40** 个方向、**160** 行自然差分复跑通过；最差第二档 **1.3519e-7**。160 步旋转归一化状态差 **1.5409e-11**，通过 1e-10。初速度轨迹差分失败：四档相对误差 **0.404484/0.649817/4.5917e-5/2.9373e-6**，低于 2e-5 的档位为 **1/4**，要求至少 2 档；后续输入组合尚未运行。下一项对照约束位移/压力及自然差分的逐步精度。

命令：`OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python runs/e5_release_audit/extrinsic_semismooth_validate.py`、`OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python runs/e5_release_audit/extrinsic_unified_trajectory.py`。报告/日志位于同目录；工作树见 HANDOFF。

## 2026-10-23：共旋应力的浮点一致性与统一界面能量

`ca283ec`、dirty=true，RTX 4090。两种等价粒子推进的 F 差 **1.6263e-19**；80 位十进制独立计算的应力差 **1.4055e-15 Pa**。原 SVD 主应力表达的状态差 **1.0415e-10 Pa**；直接使用 f、det(f) 并细化极旋转后为 **7.4735e-12 Pa**，相对高精度参考的最大绝对误差 **6.9860e-12 Pa**。256 组随机对称小形变分别覆盖初始弹性与材料更新路径；材料/能量/位移及导数 **96 passed**（57.11 s），全量 **1085 passed、8 skipped、18 subtests passed**（304.68 s）。

统一界面使用补偿求和推进开口，64 粒子、16 界面积分点、200 μs，dt=2/1/0.5 μs。闭合承载、G_c=7/70 张开及已有损伤旋转受压共 **12** 例通过步长细化检查。四组相对能量残差（%）依次为 **0.019345/0.009671/0.004835**、**0.434820/0.217200/0.108840**、**−0.402324/−0.200999/−0.100459**、**−0.620727/−0.310450/−0.155245**。闭合开口/损伤均为 0，G_c=7 的断裂耗散为 **0.001008 J=G_c A₀**；全部步骤的位移/网格冲量状态对照通过原 `atol=1e-10, rtol=1e-12`。

160 步旋转状态差 **7.8360e-11**，通过 1e-10。初速度轨迹差分失败：统一求解四档误差 **0.404488/0.649819/4.62939e-5/2.21084e-5**，低于 2e-5 的档位 **0/4**；光滑约束对照的 ε=1e-6/1e-7 误差 **2.9951e-5/1.4668e-5**，为 **1/2**。逐步状态已保存，下一项核对约束秩、零空间及反向残差。

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python runs/e5_release_audit/stress_roundoff.py
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python runs/e5_release_audit/extrinsic_energy_state.py
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python runs/e5_release_audit/extrinsic_unified_trajectory.py
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python runs/e5_release_audit/extrinsic_trace_solvers.py
PYTHONPATH=. .venv/bin/python -m pytest tests -q
```

## 2026-10-23：当前公式的 E3 四毫米释放步长对照

干净 `ca90da9`，RTX 4090，4 mm、4892 粒子、50 ms、G_c=7；dt=13.157895/6.578947/3.289474 μs，固定 PIC 参考时间 13.157895 μs、接触参考时间 25 μs。三档各释放 **5** 粒子，共 **12** 组刀路自然差分通过，最大相对误差 **0.00043623%**；32 步 CPU 检查点状态重算差 **0**。侧码、释放、接触及事件集合相同；最细两档位置差 **7.721e-10 m**、事件时刻差 **3.766e-9 s**，E/G_c 梯度变化 **0.037864%/0.021458%**。深度梯度的两次相邻变化为 **0.24505%/0.27433%**；继续核对更细步长及事件相位。

命令：`PYTHONPATH=. .venv/bin/python runs/e3_ca90da9/run_case.py release_h4_refine`。报告：`runs/e3_ca90da9/release_h4_refine/report.json`；队列已进入 `cut_h6`。

## 2026-10-23：统一界面闭合处的反力秩与轨迹差分

干净 `541c913`，RTX 4090，64 粒子、16 界面积分点、dt=2 μs。第 47–48 步的实际 Warp 残差 Jacobian 为 240×240、秩 **239**，右零空间的粒子位移分量最大 **1.223e-15/6.087e-15**；包含全部闭合点的法向约束秩 **14/16**。最小范数反力满足原动量方程，闭合第一步压力范围 **−544703.43～−544691.57 Pa**。±1e-6 初速度扰动在第 48 步出现最大 **2.480e-13/1.083e-13 m** 的局部张开，±1e-7 的对应点仍闭合。

按参考面积加权选择等价最小范数反力的 160 步对照仍失败：AD **−1.6345582086**，四档自然差分相对误差 **0.404479/0.649813/3.07919e-5/6.60599e-6**，通过 **1/4** 档，要求至少两档 ≤**2e-5**。使用能量算例相同的补偿求和，两档小扰动误差为 **2.78117e-5/7.97212e-6**，仍为 **1/4** 档通过。扩展精度残差核对显示闭合步骤仍有约 **1e-19 m** 的位移修正；下一项迭代细化整段对照。

命令：`OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python runs/e5_release_audit/extrinsic_rank_probe.py`、`OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python runs/e5_release_audit/extrinsic_canonical_reaction.py`、`OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python runs/e5_release_audit/extrinsic_canonical_flow_trajectory.py`。报告 `runs/e5_release_audit/rank_probe/report.json`，轨迹日志与入口同名；方法来源见 METHOD。

文档提交前，干净 `541c913` 全量 **1085 passed、8 skipped、18 subtests passed**（336.09 s）。命令 `PYTHONPATH=. .venv/bin/python -m pytest tests -q`，日志 `runs/research_541c913_full_tests.log`；8 项跳过为 CRESSim 上游构建。

## 2026-10-23：当前公式的 E3 六毫米下刀

干净 `ca90da9`，RTX 4090，6 mm、1452 粒子、0.24 s、15360 步、G_c=700。四组自然差分最大相对误差 **0.0224614%**；32 步 CPU 检查点状态重算差 **0**。深度扰动 ε=1e-6 时两侧分别改变 **26/14** 个几何事件，ε=1e-7 时为 **2/2**；最终释放数 **0**。命令 `PYTHONPATH=. .venv/bin/python runs/e3_ca90da9/run_case.py cut_h6`，报告 `runs/e3_ca90da9/cut_h6/report.json`；4 mm 下刀继续运行。

## 2026-10-23：统一界面轨迹的扩展精度对照

干净 `541c913`，RTX 4090 原型、64 粒子、16 界面积分点、dt=2 μs、160 步。零历史承载采用补偿求和，六类输入各四档自然差分全部通过，最大相对误差 **1.00254e-6**；材料方向的解析梯度与差分均为 **0**。零历史张开同样六类输入 × 四档全部通过，最大误差 **1.25666e-6**；已有损伤的其余七类输入各至少两档通过，最差第二档 **3.84909e-6**，报告 `canonical_opening160_scan.json`、`unified_remaining_inputs_scan.json`。命令 `OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python runs/e5_release_audit/extrinsic_input_scan.py --solver canonical --mode preload --steps 160 --output runs/e5_release_audit/canonical_preload160_scan.json`。

已有损伤轨迹仅细化约束根仍失败：初速度四档差分通过 **1/4**。独立 CPU 扩展精度前向采用相同 MUSL、极分解应力和内聚方程，ε=1e-6/1e-7 的差分为 **−1.6345646164/−1.6345520491**，与 AD **−1.6345582088** 的相对误差 **3.9201e-6/3.7684e-6**，两档均低于 **2e-5**；双精度与扩展精度基态的最终位置/速度最大差 **3.1580e-17 m/1.7056e-12 m/s**。这支持继续定位前向舍入误差；原双精度轨迹检查仍需通过原标准。

命令：`OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python runs/e5_release_audit/extrinsic_refinement_trajectory.py`、`OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python runs/e5_release_audit/extrinsic_extended_reference.py`。CPU 使用 NumPy longdouble，尾数位数 **63**；报告 `runs/e5_release_audit/extended_reference/report.json`。

## 2026-10-23：共旋应力与极旋转反向的稳定计算

RTX 4090，`541c913` 原公式与等价应变表达对照，共 **512** 组形变，独立 **80** 位十进制参考。128 个实际轨迹状态的最大应力误差 **6.9860e-12 → 7.1670e-16 Pa**；1e-12/1e-8/1e-4/0.1 四档随机形变的相对应力范数误差分别为 **1.8724e-16/2.1223e-16/2.1846e-16/2.0300e-16**。有限转动小应变与大伸长的误差为 **1.1500e-12/1.7015e-15**。命令 `OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python runs/e5_release_audit/stable_corotated_probe.py`，报告为同名 `.json`；方法来源与本地推导见 METHOD。

源码核对使用 `e5-stable` 工作树的 `541c913`、dirty=true。解析简单剪切覆盖 float32/float64、弹性/材料更新两条路径和八档幅值；四项新检查同时核对应力、μ/λ 及形变导数，材料应力与导数共 **12 passed**（6.71 s）；相同四项新检查在干净 `541c913` 上全部失败。命令 `PYTHONPATH=. .venv/bin/python -m pytest tests/test_constitutive_adjoint.py -q`，日志 `runs/stable_stress_adjoint.log`。原型的应力计算改写后，160 步初速度自然差分仍通过 **1/4** 档，两档小扰动误差 **2.89455e-5/1.90705e-5**；状态累加和约束求解的舍入继续分项核对。

分项扩展精度对照中，只将粒子 F 或速度在步末舍入至双精度，两档误差最大分别为 **8.674e-6/6.035e-6**；只舍入界面开口/损伤历史/面形变时，两档误差 **2.2363e-6/2.83044e-5**。界面状态舍入能重现小扰动超标，下一项分开核对这三类状态。命令为 `extrinsic_extended_reference.py --round-state deformation/velocity/interface/all --output runs/e5_release_audit/extended_<类别>64`，对应报告位于各输出目录。

提交前对应源码全量 **1089 passed、8 skipped、18 subtests passed**（397.18 s）；8 项跳过为 CRESSim 上游构建。命令 `PYTHONPATH=. .venv/bin/python -m pytest tests -q`，工作树 `~/Projects/BladeMaster-Warp-e5-stable-20261023`，日志 `runs/stable_stress_full.log`。源码原型的 160 步初速度四档仍通过 **1/4**，报告 `runs/e5_release_audit/stable_source_trajectory.json`；E5 继续核对界面状态精度和自然切口接入。

## 2026-10-23：界面几何精度与节点面积积分对照

干净 `66537e3` 上的独立原型，RTX 4090，64 粒子、dt=2 μs。仅补偿开口的 160 步初速度检查失败，两档小扰动误差 **7.52765e-6/2.01040e-5**，共 **1/4** 档通过；同时补偿开口与面形变后，四档误差为 **0.404478/0.649813/9.54776e-6/1.14416e-6**，达到 **2/4** 档。24 步八类输入各至少两档通过。原标准保持每类至少两档误差 ≤**2e-5**，160 步其余输入继续检查。

节点面积积分使用 9 节点，总面积 **1.44000e-4 m²**；采样闭合步骤的约束秩 **9/9**、最小奇异值 ≥**0.1799735**。与 Gauss 算例使用完全相同的初速度和扰动方向，将节点输出插值回原积分点计算同一目标，AD **−2.1800175420**；ε=1e-4/1e-5/1e-6/1e-7 的自然差分误差为 **2.62309e-6/7.66659e-12/7.03000e-9/4.72492e-10**，四档通过。原 Gauss 离散的对应 AD 为 **−1.6345582079**；两种离散的导数变化继续用界面内部间隙及空间细化核对。独立随机目标四档也通过，最大误差 **4.09853e-7**；160 步旋转归一化状态差 **4.55191e-15**，通过 1e-10。

节点原型的 200 μs、dt=2/1/0.5 μs 能量检查共 **12** 例通过。闭合承载、G_c=7/70 张开、已有损伤旋转受压的最细步长残差分别为 **0.004835%/0.108840%/−0.100459%/−0.155245%**，相邻步长残差约减半；闭合开口/损伤为 **0**，完全断裂耗散 **0.001008 J=G_c A₀**。零历史承载六类输入 × 四档自然差分通过；其余轨迹检查继续运行。

命令均在 `e5-stable-20261023` 使用 `OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python`：

```bash
runs/e5_release_audit/extrinsic_precise_geometry_scan.py --solver unified --mode existing --steps 160 --inputs initial_velocity --output runs/e5_release_audit/precise_geometry160_velocity.json
runs/e5_release_audit/extrinsic_nodal_common_velocity.py
runs/e5_release_audit/extrinsic_nodal_rank.py
runs/e5_release_audit/extrinsic_nodal_energy.py
runs/e5_release_audit/extrinsic_nodal_rotation.py
runs/e5_release_audit/extrinsic_nodal_scan.py --solver canonical --mode preload --steps 160 --output runs/e5_release_audit/nodal_preload160.json
```

报告分别为 `precise_geometry160_velocity.json`、`nodal_common_velocity.json`、`nodal_rank.json`、`nodal_energy.json`、`nodal_rotation.json`、`nodal_preload160.json`。方法与作者源码见 METHOD；本次记录未改主求解器源码，对应全量结果仍为 **1089 passed、8 skipped、18 subtests passed**。

## 2026-10-23：全局内聚接口与空间积分对照

RTX 4090，独立原型基于干净 `66537e3`，固定 12 mm 方块、E=20 kPa、ρ=1100 kg/m³、dt=2 μs、160 步及同一解析初速度/目标。6/4/3 mm Gauss 离散分别为 64/216/512 粒子、16/36/64 积分点，四档差分通过 **4/4、4/4、2/4**，最差第二档误差 **8.784e-8**；积分点最大负间隙幅值 ≤**1.458e-19 m**。节点离散在 4/3/2.4/2 mm 的内部 Gauss 位置测得负间隙 **1.554/2.088/1.973/1.541 μm**；前三档分别通过 **2/4、2/4、3/4**，2 mm 差分继续运行。空间收敛继续核对，自然切口接入采用 Gauss 积分。

命令：`OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python runs/e5_release_audit/extrinsic_fast_mesh_validate.py --spacing .003 --quadrature gauss --output runs/e5_release_audit/gauss_mesh_h003.json`。替换 spacing 和 quadrature 得到各档，报告为 `gauss_mesh_h{006,004,003}.json`、`nodal_mesh_h{006,004,003,0024,002}.json`；节点 6/4/3 mm 使用同方程的 `extrinsic_mesh_validate.py`。

可复用全局接口的验证源码基于 `7456cbc`、dirty=true，工作树 `e5-interface-20261023`。**10** 项单步测试通过（5.73 s），覆盖四类状态 × 十类输入 × 四档差分、总内力、旋转、面积拆分和冗余约束反向。24 步八类输入各至少两档 ≤**2e-5**，最差第二档 **4.84531e-6**；已有损伤、零历史承载、零历史张开三种 160 步 TLMPM 轨迹与原型的逐步状态/能量/外功差均为 **0**。命令 `PYTHONPATH=. .venv/bin/python -m pytest tests/test_differentiable_cohesive.py -q`、`OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python runs/e5_release_audit/extrinsic_library_scan.py --solver canonical --mode existing --steps 24 --output runs/interface_extract/trajectory24.json`、`OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python runs/e5_release_audit/extrinsic_library_compare.py`。

提交前全量 **1099 passed、8 skipped、18 subtests passed**（277.54 s），命令 `PYTHONPATH=. .venv/bin/python -m pytest tests -q`，日志 `runs/interface_extract/full.log`；8 项跳过为 CRESSim 上游构建。全局接口已整理，下一项接入自然参考面、永久历史、接触去重与 RK2，并重复完整能量扫描。

## 2026-10-23：补偿界面几何与检查点

RTX 4090，`b7048c7`、dirty=true。`cohesive_geometry.InterfaceGeometry` 同时保存开口、面形变及舍入余量；普通与补偿存储共 **23 passed**（14.83 s）。11 步界面轨迹的 3/4 步检查点重算状态完全一致，初态和参数梯度按 rtol=atol=1e-10 通过。全量 **1112 passed、8 skipped、18 subtests passed**（287.52 s）；8 项跳过为 CRESSim 上游构建。

64 粒子、16 Gauss 点、dt=2 μs、160 步，已有损伤、零历史承载和零历史张开三种实际 TLMPM 轨迹与扩展精度原型的最大归一化状态差分别为 **7.661e-15/5.552e-16/0**，能量差 ≤**4.337e-19 J**。已有损伤的初速度 AD **−1.634558208340304**，四档差分相对误差 **0.404478/0.649813/6.42872e-6/9.62676e-8**，达到原要求的至少两档 ≤2e-5；其余输入继续复核。

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_differentiable_cohesive.py -q
PYTHONPATH=. .venv/bin/python -m pytest tests -q
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python runs/e5_release_audit/extrinsic_geometry_compare.py
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python runs/e5_release_audit/extrinsic_library_geometry_scan.py --solver unified --mode existing --steps 160 --inputs initial_velocity --output runs/interface_precision/velocity160.json
```

日志和报告位于 `runs/interface_precision/`，工作树见 HANDOFF。

干净 `66537e3` 的扩展精度原型已完成 160 步其余七类输入，加上初速度共 **8/8** 类均至少两档通过；其余七类的最差第二档误差 **1.42430e-6**。2 mm 节点空间对照的四档差分全部通过，最大误差 **4.02542e-6**，内部 Gauss 位置的最小间隙 **−1.54109 μm**。命令分别为 `extrinsic_precise_geometry_scan.py --solver unified --mode existing --steps 160 --inputs initial_gap initial_history initial_frame material young duration acceleration --output runs/e5_release_audit/precise_geometry160_remaining.json` 和 `extrinsic_fast_mesh_validate.py --spacing .002 --quadrature nodal --output runs/e5_release_audit/nodal_mesh_h002.json`，均使用 `OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python runs/e5_release_audit/<入口>`。

## 2026-10-23：当前公式的 E3 四毫米下刀

干净 `ca90da9`，RTX 4090，4 mm、4892 粒子、0.24 s、18240 步、G_c=70，dt/PIC 参考时间 13.157895 μs。刀路、初始高度、E/G_c 共 **10** 组自然差分通过，最大相对误差 **0.374054%**；32/128 步检查点状态重算差 **0**，控制量/初始位置梯度差 **3.238e-14/8.881e-13**。释放 **5** 粒子，CUDA 数组峰值 **15.154 GiB**。命令 `PYTHONPATH=. .venv/bin/python runs/e3_ca90da9/run_case.py cut_h4`，报告 `runs/e3_ca90da9/cut_h4/report.json`；队列进入 6 mm 完整轨迹。

## 2026-10-23：参考表面重建与闭合场插入

RTX 4090，`f507dc0`、dirty=true。`reference_interface.reconstruct` 的 **7 passed**（1.01 s），覆盖统一/双侧仿射运动、面形变的空间导数、六类输入的两档差分和整体平移虚功。实际 TLMPM 参考权重上的平移、旋转、拉伸、剪切状态，初始重建间隙 ≤**5.067e-18 m**、面形变误差 ≤**3.331e-16**。旋转状态在分侧网格推进后仍产生 **1.465e-6 m** 间隙，未切场为 **2.108e-18 m**，需要统一完整闭合场的网格运动更新。

网格质量内积下的闭合投影同时用于旧速度、新速度和 MUSL 重映射。四种运动 × dt=2/1/0.5 μs 共 **12** 例通过：与未切物体的位置差 **0**、速度差 ≤**5.552e-17 m/s**、F 差 ≤**6.939e-18**、应力差 ≤**5.685e-14 Pa**。平移/旋转的牵引为 **0**；拉伸/剪切的牵引最大 **1604.938/370.486 Pa**，与均匀材料应力差 ≤**3.184e-12 Pa**。下一项连接闭合约束与损伤后的内聚响应。

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_reference_interface.py -q
PYTHONPATH=. .venv/bin/python -m pytest tests -q
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python runs/e5_release_audit/reference_kinematics_probe.py
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python runs/e5_release_audit/closed_grid_insertion.py
```

全量 **1119 passed、8 skipped、18 subtests passed**（284.42 s）；8 项跳过为 CRESSim 上游构建。日志/报告 `runs/reference_interface/{unit.log,full.log,kinematics.json,closed_grid.json}`，工作树见 HANDOFF。

插入对照使用 `b7048c7`、dirty=true（对应 `f507dc0` 源码）。普通粒子界面更新在平移的约 **1e-23 m** 间隙舍入处求解失败，旋转 dt=2 μs 产生假断裂耗散 **6.240e-6 J**。命令 `OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python runs/e5_release_audit/extrinsic_insertion_patch.py`，报告 `runs/interface_insertion/patch.json`。

相同源码的 160 步几何精度核对已完成八类输入，各至少两档误差 ≤**2e-5**；其余七类的最差第二档误差 **1.39787e-6**。命令 `OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python runs/e5_release_audit/extrinsic_library_geometry_scan.py --solver unified --mode existing --steps 160 --inputs initial_gap initial_history initial_frame material young duration acceleration --output runs/interface_precision/remaining160.json`。

## 2026-10-23：闭合网格与内聚响应的统一推进

RTX 4090，独立原型基于干净 `7f31460`，64 粒子、16 Gauss 点。24 步平移/旋转保持零开口、零损伤。200 μs、dt=2/1/0.5 μs 共 **12** 例，闭合承载、G_c=7/70 受载断裂、已有损伤的能量残差随步长减半约减半，最细档幅值 **0.017552%/0.002262%/0.007322%/0.348344%**。受载断裂使用双侧 ±2e5 m/s²，强度 50 kPa；16 个积分点均完全断裂，耗散 **0.001008/0.01008 J=G_c A₀**。

24 步闭合/已有损伤/受载断裂分别检查六/八/六类输入，每类至少两档自然差分误差 ≤**2e-5**；三组最差第二档 **1.81311e-8/2.74149e-6/3.23489e-7**。命令前缀 `OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python`：

```bash
runs/e5_release_audit/extrinsic_grid_energy.py
runs/e5_release_audit/extrinsic_grid_scan.py --solver unified --mode preload --steps 24 --output runs/reference_interface/grid_preload24.json
runs/e5_release_audit/extrinsic_grid_scan.py --solver unified --mode existing --steps 24 --output runs/reference_interface/grid_existing24.json
runs/e5_release_audit/extrinsic_grid_loaded_scan.py --solver unified --mode opening --steps 24 --output runs/reference_interface/grid_loaded24.json
```

`7f31460`、dirty=true 的 `grid_constraints.project_velocity` 单元测试 **3 passed**（0.39 s），覆盖质量加权共享速度、动量、投影能量、旋转、三类输入反向和冗余约束。命令 `PYTHONPATH=. .venv/bin/python -m pytest tests/test_grid_constraints.py -q`，日志 `runs/reference_interface/grid_projection_unit.log`。

提交前全量 **1122 passed、8 skipped、18 subtests passed**（330.79 s），8 项跳过为 CRESSim 上游构建。命令 `PYTHONPATH=. .venv/bin/python -m pytest tests -q`，日志 `runs/reference_interface/grid_projection_full.log`。

## 2026-10-23：E4 当前公式的五种长轨迹复验

干净 `ca90da9`，RTX 4090。6 mm 三刀后推块 **2.18 s、139520 步、1452 粒子**，四组自然差分最大误差 **0.0946834%**；32/127 步检查点状态重算差 **0**，控制量/初始位置梯度差 **2.759e-14/7.152e-11**。最终 **380/346/302/424** 四块，暂停损伤增量 **0**，推块后最大塑性形变偏差 **0.0953027**，峰值数组显存 **3.4224 GiB**。

`ca90da9` 的暂停续切、平行两刀、交叉两刀、三刀后推块和 4 mm 交叉窗口共 **20** 组自然差分全部通过；各组最大误差 **0.005113%/0.078791%/0.037159%/0.094683%/0.009204%**，检查点状态重算差及暂停损伤增量均为 **0**。E5 界面接入后复测受影响组合。

命令 `PYTHONPATH=. .venv/bin/python -m scripts.diff_multicut_validate --output runs/e4_ca90da9/three-cuts-push --cases three-cuts-push --spacings .006 --dt .000015625 --flip-reference-dt .000015625 --segment-steps 32 127 --checkpoint-device cpu --epsilons 1e-6 1e-7`。报告 `runs/e4_ca90da9/three-cuts-push/report.json`，队列 `continue_status.json` 的全部任务返回码为 **0**。

## 2026-10-23：内聚网格与材料推进接口

RTX 4090，`22d421a`、dirty=true。`cohesive_grid_update.advance_material` 将闭合约束、内聚响应和首阶 MUSL 接入同一步推进。新增 **6** 项测试核对四类完整运动、完全断裂的 G_c A 耗散、卸载历史、状态重启及四类输入的两档梯度；连同网格载荷测试为 **13 passed**（25.94 s）。命令 `PYTHONPATH=. .venv/bin/python -m pytest tests/test_cohesive_grid_update.py tests/test_grid_load.py -q`。

闭合承载、已有损伤、受载断裂三种 **160** 步实际 TLMPM 轨迹，正式接口与原型的逐步粒子/界面状态、能量和外功差均为 **0**。24 步八类输入各至少两档自然差分误差 ≤**2e-5**，最差第二档 **2.74149e-6**。命令前缀 `OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python`：`runs/e5_release_audit/extrinsic_grid_library_compare.py`；`runs/e5_release_audit/extrinsic_grid_library_scan.py --solver unified --mode existing --steps 24 --output runs/grid_update/gradient24.json`。工作树 `~/Projects/BladeMaster-Warp-e5-grid-20261023`，报告 `runs/grid_update/{compare160,gradient24}.json`。

提交前全量 **1128 passed、8 skipped、18 subtests passed**（330.77 s）；8 项跳过为 CRESSim 上游构建。命令 `PYTHONPATH=. .venv/bin/python -m pytest tests -q`，日志 `runs/grid_update/full.log`。

## 2026-10-23：切面参考面积及双侧支撑

RTX 4090，`28c08cb`、dirty=true。`reference_cut.quadrature` 的 **14 passed**（0.40 s），包括解析切面面积/二次矩、仿射变换、六类输入的两档差分及真实粒子网格三个坐标方向、正反法向的共享面计数。12 mm 方块、6 mm 网格，0/0.3/1/2/3 mm 切平面偏移均得到 **1.44e-4 m²**。直接切点网格插值在 0.3/1 mm 偏移时缺少一侧 **5%/16.67%** 权重；完整双侧参考迹与实际材料推进的转置力映射继续接入。

命令：`PYTHONPATH=. .venv/bin/python -m pytest tests/test_reference_cut.py -q`；`OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python runs/e5_release_audit/reference_cut_support.py`。日志和报告位于 `e5-grid-20261023/runs/grid_update/reference_cut_{unit.log,support.json}`。

干净 `22d421a` 的 160 步已有损伤/受载断裂扫描已完成，分别为八/六类输入、32/24 行自然差分；每类均至少两档误差 ≤**2e-5**，最差第二档 **4.93259e-6/2.74636e-7**。命令前缀 `OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python`：`runs/e5_release_audit/extrinsic_grid_scan.py --solver unified --mode existing --steps 160 --output runs/reference_interface/grid_library_existing160.json`；`runs/e5_release_audit/extrinsic_grid_loaded_scan.py --solver unified --mode opening --steps 160 --output runs/reference_interface/grid_library_loaded160.json`。

提交前全量 **1142 passed、8 skipped、18 subtests passed**（279.81 s）；8 项跳过为 CRESSim 上游构建。命令 `PYTHONPATH=. .venv/bin/python -m pytest tests -q`，日志 `runs/grid_update/reference_cut_final_full.log`。

## 2026-10-23：偏移切面的双侧网格映射

RTX 4090，`b899437`、dirty=true。`reference_grid.build_trace` 六项测试通过，覆盖四个轴向/斜向切面的三线性多项式及梯度重现、切面位置/法向经过几何与网格映射的两档差分，以及四步内聚材料推进的两档差分（rtol=2e-5、atol=1e-9）。固定步长、固定网格质量的自定义反向登记已修复，闭合材料推进的初速度/材料输入差分通过。相关测试共 **16 passed**（83.01 s）：`PYTHONPATH=. .venv/bin/python -m pytest tests/test_reference_grid.py tests/test_cohesive_grid_update.py tests/test_grid_constraints.py -q`，日志 `runs/grid_update/reference_grid_coupled.log`。

干净 `b899437` 的延拓原型：12 mm 方块、6 mm 网格、64 粒子、96 积分点，切面偏移 **1 mm**、推进 **200 μs**，dt=2/1/0.5 μs。旋转能量相对残差 **1.62335e-6/8.11860e-7/4.05976e-7**；预拉伸幅值 **2.09824e-4/1.04934e-4/5.24726e-5**；受载断裂 **8.62933e-5/4.44269e-5/2.25651e-5**。断裂参数 G_c=7 J/m²、强度 50 kPa，双侧加速度 ±2e5 m/s²；最终耗散 **0.001008 J=G_c A₀**。命令 `OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python runs/e5_release_audit/reference_extension_energy.py`，报告 `runs/grid_update/reference_extension_energy.json`。

正式网格映射与 NumPy 原型在 6/4/3/2.4/2 mm 网格上相同（映射 atol=1e-15、梯度 atol=1e-12）。固定 1 mm 刀位、dt=2 μs 的一步旋转与未切场对照，位置 RMS 差 **44.0/42.0/17.3/5.83/5.22e-12 nm**。八个相位 −1/2…3/8 网格间距的 **40** 例均保持零损伤；各网格最大位置 RMS 差 **32.9/42.0/27.3/19.5/13.1 nm**，最大 F 分量 RMS 差 **2.2205e-5/2.8809e-5/2.4950e-5/2.2316e-5/1.1404e-5**。6→4 mm 不单调，后四档位置 RMS 上界下降；完整空间收敛继续核对。命令 `OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python runs/e5_release_audit/reference_grid_mesh.py` 及同入口 `--phase-scan`，报告 `runs/grid_update/reference_grid_{mesh,phase}.json`。

提交前全量 **1149 passed、8 skipped、18 subtests passed**（335.95 s）；8 项跳过为 CRESSim 上游构建。命令 `PYTHONPATH=. .venv/bin/python -m pytest tests -q`，日志 `runs/grid_update/reference_grid_full.log`。

## 2026-10-23：有限刀面裁剪与位姿导数

RTX 4090，`fb687af`、dirty=true。`reference_cut.quadrature` 增加逐半平面裁剪，`blade_quadrature` 将刀具位置、四元数和半尺寸接入有限矩形刀面。几何和双侧网格映射共 **34 passed**（62.74 s）。新增解析矩形/三角形面积、二次矩、相切/空交集与仿射不变性核对；6/4 mm 网格、正反法向、两个刀面偏移和十二个刀高共 **96** 个配置符合解析面积及质心。一般裁剪八类输入、刀具/粒子七类输入各两档差分按 rtol=2e-6、atol=1e-8 通过。

命令 `PYTHONPATH=. .venv/bin/python -m pytest tests/test_reference_cut.py tests/test_reference_clip.py tests/test_blade_quadrature.py tests/test_reference_grid.py -q`，工作树 `~/Projects/BladeMaster-Warp-e5-grid-20261023`，日志 `runs/grid_update/reference_blade_unit.log`。

提交前全量 **1163 passed、8 skipped、18 subtests passed**（334.54 s）；8 项跳过为 CRESSim 上游构建。命令 `PYTHONPATH=. .venv/bin/python -m pytest tests -q`，日志 `runs/grid_update/reference_blade_full.log`。

## 2026-10-23：当前公式的 E3 六毫米完整轨迹

干净 `ca90da9`，RTX 4090，6 mm、1452 粒子、1.05 s、67200 步、G_c=700；dt/PIC 参考时间 15.625 μs。三类刀路控制量与屈服应力共 **8** 组自然差分通过，最大相对误差 **0.0668341%**。16/32/128 步 CPU 检查点状态重算差 **0**；控制量/初始位置梯度最大差 **7.550e-15/3.383e-12**。释放数 **0**，峰值刀力 **12.5424 N**，峰值 CUDA 数组 **3.4193 GiB**。

命令 `PYTHONPATH=. .venv/bin/python runs/e3_ca90da9/run_case.py full_h6`，工作树 `~/Projects/BladeMaster-Warp-e3-current-20261022`，报告 `runs/e3_ca90da9/full_h6/report.json`；队列继续执行 `full_h4`。

## 2026-10-23：材料参考刀刃扫掠与重复面积扣除

RTX 4090，`0447687`、dirty=true。`reference_surface.swept_edge`、`ReferenceSurface.subtract` 的新旧几何检查共 **41 passed**（9.08 s）：6/4 mm 网格的解析面积与二次矩、刚体共动零面积、退刀重入与交叉切面、十一类输入的两档差分（rtol=2e-6、atol=1e-8）。旧前沿位于 **6/6.3 mm** 时，新增面积对前沿高度的导数均为 **0.0074 m**；两个网格、四个前沿高度的共享边界面积导数只计一次。

刀刃宽 **4 mm**、下降 **6 mm**、转角 **0.6 rad** 的螺旋面：空间/时间共同取 4/8/16 段，相对面积误差 **7.57245e-4/1.90330e-4/4.76464e-5**；固定刀刃四段、时间细化至同一空间离散极限的误差 **5.48573e-5/1.37447e-5/3.43806e-6**，均约二阶。宽 **7.4 mm** 刀刃前沿依次为 15/9/3/9/1 mm，累计面积 **2.22e-5/6.66e-5/6.66e-5/8.14e-5 m²**，旧顶点保持原值。退刀后保留零测度边界片用于面积导数；最小有符号积分权重为舍入量 **−5.482e-22 m²**。物理装配继续核对零面积极限、永久损伤与自然事件推进。

命令 `PYTHONPATH=. .venv/bin/python -m pytest tests/test_reference_surface.py tests/test_reference_cut.py tests/test_reference_clip.py tests/test_blade_quadrature.py -q`；`OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python runs/e5_release_audit/reference_surface_measure.py`。工作树 `~/Projects/BladeMaster-Warp-e5-grid-20261023`，日志与报告 `runs/grid_update/reference_surface_{unit.log,measure.json}`。

提交前全量 **1176 passed、8 skipped、18 subtests passed**（334.29 s）；8 项跳过为 CRESSim 上游构建。命令 `PYTHONPATH=. .venv/bin/python -m pytest tests -q`，日志 `runs/grid_update/reference_surface_full.log`。

## 2026-10-23：闭合界面的小面积伴随

RTX 4090，`4646fa0`、dirty=true。闭合约束直接使用固定秩伪逆 VJP，面积从 **4e-6** 缩小到 **4e-14 m²** 时，闭合位移的面积导数误差为 **0**；相同最小面积下，基线通用伴随的绝对误差为 **7.45058e-8 m/m²**。解析受压小例的总反力始终为 **−2.5 N**、单侧位移 **−10 μm**，压力随面积倒数变化；张开与卸载总力随面积减小趋零。四种状态、五档面积共 **20** 例，最大开口/总力误差 **3.389e-21 m / 4.962e-16 N**。零面积处的约束集合继续与自然插入核对。

界面测试 **38 passed**（16.76 s），覆盖十类输入差分、重复积分点、检查点、12 例小面积解析导数及三例损伤历史单侧导数。命令前缀 `OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python`：`-m pytest tests/test_differentiable_cohesive.py -q`；`runs/e5_release_audit/cohesive_area_analytic.py`。日志/报告在 `e5-grid-20261023/runs/grid_update/{closed_adjoint_final_unit.log,cohesive_area_analytic.json}`。

提交前全量 **1191 passed、8 skipped、18 subtests passed**（352.79 s）；8 项跳过为 CRESSim 上游构建。命令 `PYTHONPATH=. .venv/bin/python -m pytest tests -q`，日志 `runs/grid_update/closed_adjoint_full.log`。

## 2026-10-23：自然切面与部分侧码的网格映射

RTX 4090，`9765655`、dirty=true。6/4 mm 方块/圆柱、共享场/单侧/交叉侧码的多项式重现、界面—粒子虚功、总内力与位置/法向导数共 **18** 项检查通过（61.01 s）。同节点相容场按参考质量取值，系数及来源见 METHOD；全共享场分离映射严格为 **0**。

两种采样、直刀/0.2 rad 斜刀各推进两步，共 **31** 个接受段、**24** 个非空参考切面和 **1716** 个积分点，双侧映射全部建立。干净 `9765655` 的同组输入有 **10** 次缺失侧码错误；首次扫掠面积 **5.92095e-7 m²**，当时全部 **64** 个粒子的侧码仍为 0。下一项处理 Q=0 共享场、零面积项的物理装配和永久界面历史。

命令前缀 `OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python`：`-m pytest tests/test_reference_grid.py -q`；`runs/e5_release_audit/natural_surface_support.py`。工作树 `~/Projects/BladeMaster-Warp-e5-grid-20261023`，日志/报告 `runs/grid_update/{partial_trace_unit.log,natural_surface_support.json,natural_surface_support_baseline.json}`。

提交前全量 **1203 passed、8 skipped、18 subtests passed**（413.32 s）；8 项跳过为 CRESSim 上游构建。命令 `PYTHONPATH=. .venv/bin/python -m pytest tests -q`，日志 `runs/grid_update/partial_trace_full.log`。

## 2026-10-23：共享场的内聚平衡与稳定开口残差

RTX 4090，`35153a7`、dirty=true。共享场 B=0 使用有限近端尺度；q=B=κ=0 的冗余约束选择零牵引；开口残差按等价牵引形式计算。新增 **20** 项检查覆盖混合历史、空位移块、映射导数、追加共享点和 6 mm 方块/4 mm 圆柱的六步材料状态与梯度对照，相关测试 **65 passed**（52.30 s）。已有损伤、有限开口而 B=G=0 的解析映射梯度为 **−2.769216607024814e-6**；四档自然差分最大相对误差 **1.38103e-13**。

材料—界面推进的闭合承载、G_c=7/70 受载断裂及已有损伤四组 200 μs 轨迹，在 dt=2/1/0.5 μs 共 **12** 例的能量残差均随步长减半约减半。最细档幅值 **0.017552%/0.002262%/0.007322%/0.348344%**；完全断裂耗散 **0.001008/0.01008 J=G_c A₀**，闭合承载保持零开口和零损伤。

命令前缀 `OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python`：`-m pytest tests/test_differentiable_cohesive.py tests/test_cohesive_grid_update.py -q`；`runs/e5_release_audit/shared_map_limit.py`；`runs/e5_release_audit/shared_energy.py`。工作树 `~/Projects/BladeMaster-Warp-e5-grid-20261023`，日志/报告 `runs/grid_update/{shared_solve_unit.log,shared_map_limit.json,shared_energy.json}`。

提交前全量 **1223 passed、8 skipped、18 subtests passed**（366.31 s）；8 项跳过为 CRESSim 上游构建。命令 `PYTHONPATH=. .venv/bin/python -m pytest tests -q`，日志 `runs/grid_update/shared_solve_full.log`。

## 2026-10-23：稳定开口残差的长轨迹复验

RTX 4090，`35153a7`、dirty=true，对应 `6d2ffb5` 数值源码。已有损伤的 **160** 步材料—界面轨迹，初速度、开口、历史、面形变、内聚材料、杨氏模量、时长和外加载荷共 **8** 类输入、**32** 行自然差分全部完成；每类至少两档相对误差 ≤**2e-5**，最差第二档 **1.07941e-5**。初速度的两档小扰动误差为 **2.96208e-6/1.07941e-5**。

命令 `OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python runs/e5_release_audit/extrinsic_grid_library_scan.py --solver unified --mode existing --steps 160 --output runs/grid_update/shared_existing160.json`，工作树 `~/Projects/BladeMaster-Warp-e5-grid-20261023`，报告/日志为 `shared_existing160.json/.log`。

## 2026-10-23：追加参考切面的永久损伤状态

RTX 4090，`6d2ffb5`、dirty=true。`CohesiveHistory` 的追加、首阶材料推进和快照共 **7** 项检查通过（12.75 s）。6/4 mm 退刀重入与交叉参考面保留旧积分点和损伤；跨追加时刻的完整反向/分段重算按 rtol=atol=2e-9 通过。三步材料轨迹的旧历史、开口、粒子形变、材料、外载、时长和新增曲面平移七类输入各两档自然差分按 rtol=2e-5、atol=1e-7 通过。

四组自然事件几何累计 **31** 个接受段、**1716** 个积分点，追加时旧点状态差 **0**。16 步加载例的三个旧积分点全部断裂，耗散 **1.26e-6 J=G_c A₀**；追加三个新点时旧历史差 **0**、新历史 **0**，卸载一步的旧历史保持单调，检查点恢复后的粒子状态差 **0**。旧断裂能对前沿高度的导数为 **−0.037 J/m**，两档重建几何差分通过。

命令前缀 `OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python`：`-m pytest tests/test_cohesive_history.py -q`；`runs/e5_release_audit/permanent_history_measure.py`。同一工作树报告/日志为 `runs/grid_update/permanent_history_{unit.log,measure.json}`。

提交前全量 **1230 passed、8 skipped、18 subtests passed**（380.51 s）；8 项跳过为 CRESSim 上游构建。命令 `PYTHONPATH=. .venv/bin/python -m pytest tests -q`，日志 `runs/grid_update/permanent_history_full.log`。

## 2026-10-23：闭合界面不等面积拆分的历史导数

RTX 4090，`6d2ffb5`、dirty=true（内聚公式同该提交）。总面积 **4e-6 m²**、自由开口 **1 μm** 的同位置积分点按 **60/40、65/35** 拆分，闭合分支的位移不变，零历史处的共同历史导数却为 **0.6510417/0.6868132**，解析值均为 **0.625**。1e-11 m 单侧差分为 **0.6249961**，两组误差 **4.0006%/9.0006%**，检查失败。闭合分支按未加权约束分配反力，使牵引依赖拆分权重；下一项统一面积加权反力与相应 VJP，并复测不等面积、零面积极限及长轨迹。1e-12 m 历史扰动还触发光滑初值的 KKT 细化失败，另核对其病态驻点求解。

命令 `OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python runs/e5_release_audit/unequal_area_limit.py`，工作树 `~/Projects/BladeMaster-Warp-e5-grid-20261023`，报告/日志 `runs/grid_update/unequal_area_limit.json/.log`。14 组正向、42 组单侧扰动逐项保存数值或求解错误。

## 2026-10-23：面积加权闭合反力与统一近端初值

RTX 4090，`e4cf414`、dirty=true。同位置积分点按七种面积比例拆分，受拉/受压共 **14** 组的位移差 **0**，牵引均为 **±31250 Pa**，共同历史导数为 **0.625/0**；**42** 组单侧扰动无求解失败，1e-12 m 档最大相对误差 **6.24702e-7**。新增 **22** 项检查覆盖不等面积的十类输入伴随，以及 1e-10/1e-12 m 正历史的串联弹簧解析状态和导数。

闭合承载、已有损伤和受载断裂三种 **160** 步轨迹，与 `e4cf414` 求解器逐步对照：归一化状态差 ≤**1.95528e-14**、能量差 ≤**4.337e-19 J**、外功差 ≤**2.711e-20 J**。闭合反力按参考面积取最小压力范数；固定法向 ADMM 提供统一初值，直接近端牵引避免小耦合下的相减损失。方法和 VJP 见 METHOD。

命令前缀 `OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python`：`runs/e5_release_audit/unequal_area_limit.py --output runs/grid_update/weighted_area_limit.json`；`runs/e5_release_audit/weighted_solver_compare.py`；`-m pytest tests -q`。同一 `e5-grid-20261023` 工作树报告/日志为 `runs/grid_update/{weighted_area_limit.json,weighted_compare160.json,weighted_area_full.log}`。提交前全量 **1252 passed、8 skipped、18 subtests passed**（366.51 s）；8 项跳过为 CRESSim 上游构建。

## 2026-10-23：面积加权反力的长轨迹与能量复测

RTX 4090，`e4cf414`、dirty=true（求解源码对应 `95eaf2d`）。160 步、八类输入、四档自然差分，每类至少两档相对误差 ≤**2e-5**，最差第二档 **1.07018e-6**；32 组中 **28** 组通过。闭合承载、G_c=7/70 受载断裂、已有损伤四组各用 dt=2/1/0.5 μs，十二例能量残差随步长减半约减半，最细档幅值 **0.017552%/0.002262%/0.007322%/0.348344%**；完全断裂耗散为 **0.001008/0.01008 J=G_c A₀**。

命令前缀 `OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python`：`runs/e5_release_audit/extrinsic_grid_library_scan.py --solver unified --mode existing --steps 160 --output runs/grid_update/weighted_existing160.json`；`runs/e5_release_audit/weighted_energy.py`。报告为同一工作树 `runs/grid_update/weighted_existing160.json`、`weighted_energy.json`。

## 2026-10-23：零面积局部响应与退刀闭合投影

RTX 4090，`95eaf2d`、dirty=true。积分总力局部核新增 **18** 项检查通过，覆盖 64 个正面积状态的牵引形式等价性、零面积运动、张开/卸载/完全断裂的力与能量面积导数，以及五类输入各两档差分。包含原局部核共 **39 passed**（0.58 s）。正自由间隙的三组零面积全局原型与删除该积分点的位移差 ≤**1.165e-21 m**，十二组单侧差分全部 ≤**2e-5**。

完全断裂后退刀的闭合投影失败：6/4 mm 追加的 **957/1572** 个退化积分点被零历史掩码当作完整约束，动能损失 **4.62607e-6/2.46949e-6 J**。局部总力核已令零面积保持自由运动；全局平衡、闭合投影与出生事件继续接入。

命令前缀 `OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python`：`-m pytest tests/test_cohesive_measure.py tests/test_cohesive_proximal.py -q`；`runs/e5_release_audit/zero_measure_opening.py`；`runs/e5_release_audit/zero_area_history_projection.py`。报告/日志位于 `runs/grid_update/{measure_proximal_unit.log,zero_measure_opening.json,zero_area_history_projection.json}`。提交前全量 `-m pytest tests -q` 为 **1270 passed、8 skipped、18 subtests passed**（357.68 s），日志 `measure_proximal_full.log`；8 项跳过为 CRESSim 上游构建。

## 2026-10-23：积分总力全局平衡与反力选择的伴随

RTX 4090，`77d5721`、dirty=true。普通和补偿几何的总力接口接受零面积，输出积分力及积分面力矩；相关 **132 passed**（74.98 s）。覆盖正面积与牵引接口一致性、零面积自由运动、十类输入反向、面积单侧差分、冗余闭合/张开/混合状态的独立反力目标、压缩乘子非正及强度球边界。三种重复积分点的面积方向，AD 为 **8.4736433281/0.0066720124212/8.4113422350**，两档差分相对误差均 ≤**7.1e-10**。接触力尺度 **1e-9/1/1e3** 的普通/补偿接口共 **6** 例通过（1.80 s），含非零自由间隙方向导数。

共用近端和反力代码修改前后的首阶 MUSL 对照为闭合承载、已有损伤、受载断裂三种 **160** 步轨迹。与 `77d5721` 的归一化状态差 ≤**6.006e-13**、能量差 ≤**8.882e-16 J**、外功差 ≤**5.422e-20 J**。两条轨迹均使用当前材料入口的牵引接口；总力形式的材料装配继续接入。

命令前缀 `OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python`：`-m pytest tests/test_cohesive_force.py tests/test_differentiable_cohesive.py -q`；`-m pytest tests/test_cohesive_force.py -k nonpositive -q`；`runs/e5_release_audit/force_partition_adjoint.py`；`runs/e5_release_audit/force_solver_compare.py`。同一 `e5-grid-20261023` 工作树报告/日志为 `runs/grid_update/{force_global_unit7.log,force_scaled_constraint2.log,force_partition_adjoint3.log,force_compare160.json}`。

提交前全量 `PYTHONPATH=. .venv/bin/python -m pytest tests -q` 为 **1330 passed、8 skipped、18 subtests passed**（413.20 s），日志 `runs/grid_update/force_global_final_full.log`；新增 **60** 项总力及反力选择检查，8 项跳过为 CRESSim 上游构建。

## 2026-10-23：当前公式的 E3 四毫米完整轨迹

干净 `ca90da9`，RTX 4090，4 mm、4892 粒子、1.05 s / 79800 步、G_c=70；dt/PIC 参考时间 13.157895 μs。三类刀路与 G_c 倍数各两档自然差分，共 **8** 组全部通过，最大相对误差 **0.0927014%**。32 步 CPU 检查点连续状态重算差 **0**；释放 **5** 粒子，峰值刀力 **11.4993 N**，CUDA 数组峰值 **5.6271 GiB**。

独立正向 **701.26 s**，检查点正向/重算/反向 **732.55/770.44/300.66 s**；完整差分任务耗时 **14700.59 s**。命令 `PYTHONPATH=. .venv/bin/python runs/e3_ca90da9/run_case.py full_h4`，工作树 `~/Projects/BladeMaster-Warp-e3-current-20261022`，报告 `runs/e3_ca90da9/full_h4/report.json`。队列继续 6 mm 下刀与完整轨迹的步长细化。

## 2026-10-23：积分总力材料装配与零面积自由运动

RTX 4090，`016b255`、dirty=true。`advance_material` 使用积分总力接口，初态和 MUSL 重映射仅约束正面积、零历史界面。相关 **41 passed**（107.00 s），新增 **5** 项检查覆盖零面积点的分离/压缩、零/已有损伤历史、三步粒子状态及其反向，以及张开处面积从零增大的力导数。

闭合承载、已有损伤和受载断裂三种 **160** 步轨迹，与 `016b255` 的牵引装配逐步比较：归一化状态差 ≤**8.994e-15**、能量差 ≤**8.674e-19 J**、外功差 ≤**5.422e-20 J**；受载闭合三项差均为 **0**。

命令前缀 `OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python`：`-m pytest tests/test_cohesive_grid_update.py tests/test_cohesive_history.py tests/test_reference_grid.py -q`；`runs/e5_release_audit/force_material_compare.py`。报告/日志为 `e5-grid-20261023/runs/grid_update/{force_material_unit.log,force_material_compare160.json,force_material_compare.log}`。

提交前全量 `PYTHONPATH=. .venv/bin/python -m pytest tests -q` 为 **1335 passed、8 skipped、18 subtests passed**（427.09 s），日志 `runs/grid_update/force_material_full.log`；8 项跳过为 CRESSim 上游构建。

## 2026-10-23：精确参考面与退刀后的材料运动

RTX 4090，`4e5a892`、dirty=true。几何/历史/参考坐标共 **59 passed**（82.01 s），覆盖 6/4 mm 共享边界、零面积单侧导数、宽度 2⁻³⁰/2⁻⁶⁰ 的正面积、11 类输入差分、分段反向，以及方块/圆柱/香蕉 × 原位/扰动采样 × 直/斜刀。18 种采样配置与 `4e5a892` 的粒子坐标、原点和网格尺寸差均为 **0**。

64 粒子的已断裂材料退刀例：旧 **6** 个积分点的面积为 **1.5e-6 m²**，新增 **45** 点的面积严格为 **0**；实际材料推进的 x/v/F/Fp/应力与自由推进差均为 **0**，动能同为 **9.504e-6 J**。该项 **1 passed**（1.16 s）。

命令前缀 `OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python`：`-m pytest tests/test_reference_coordinates.py tests/test_reference_surface.py tests/test_cohesive_history.py -q`；`-m pytest tests/test_cohesive_history.py -k withdrawal_zero -q`；`runs/e5_release_audit/reference_sampling_compare.py`；`runs/e5_release_audit/exact_reference_material_measure.py`。报告/日志位于 `e5-grid-20261023/runs/grid_update/{exact_reference_unit2.log,exact_reference_material2.log,reference_sampling_compare.json,exact_reference_material.json}`。

提交前全量 `PYTHONPATH=. .venv/bin/python -m pytest tests -q` 为 **1375 passed、8 skipped、18 subtests passed**（487.88 s），日志 `runs/grid_update/exact_reference_full.log`；新增 **40** 项检查，8 项跳过为 CRESSim 上游构建。

## 2026-10-23：有限切面可见性与交叉界面材料耦合

RTX 4090，原型干净 `a211caf`、接口测试 `a211caf` dirty=true。相关 **36 passed**（93.80 s），新增 **8** 项检查。64 粒子完整/部分交叉扫掠为 **612/462** 积分点；完整交叉映射与解析分区的权重差 **0**，三线性值/空间梯度重现误差 ≤**8.882e-16/1.706e-13**。两点交叉例中沿用所属粒子的另一刀侧码产生最大权重差 **1.05215**；有限面映射与解析值一致，位置两档差分误差 ≤**7.013e-11**。原型两种扫掠映射分别耗时 **11.57/6.52 s**。

实际双切面 **12** 积分点、**32 μs** 的预应变/受载两组、**2/1/0.5 μs** 三档能量对照：相对残差幅值分别为 **0.0027535%/0.0013976%/0.00072282%** 和 **0.29614%/0.14903%/0.075308%**。粒子推进、界面储能/断裂功与网格中点外功共同核算。

命令前缀 `OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python`：`-m pytest tests/test_reference_visibility.py tests/test_reference_grid.py tests/test_cohesive_history.py -q`；`runs/e5_release_audit/visibility_trace_validate.py`；`runs/e5_release_audit/visibility_natural.py`；`runs/e5_release_audit/visibility_energy.py`。报告/日志在 `e5-grid-20261023/runs/grid_update/visibility_{unit.log,trace.json,natural.json,energy.json}`。

`runs/e5_release_audit/visibility_gradient.py` 的双切面四步材料轨迹覆盖八类输入、六档扰动，共 **48** 组自然差分；每类至少两档相对误差 ≤**2e-5**，最差第二档 **4.647e-6**。两刀位的 ε=0.1/0.03 跨过可见性分支，ε≤0.01 的分支保持一致；`--branches` 保存逐步支撑集合变化。杨氏模量方向 AD 为 **7.39318e-11**，ε=0.1/0.03/0.01 的误差为 **3.322e-6/4.647e-6/6.751e-6**。同一目标用分量差的扩展精度求和核对，报告同时保留标量相减结果及全部六档差分：`visibility_gradient.json`、`visibility_branches.json`。

提交前全量 `PYTHONPATH=. .venv/bin/python -m pytest tests -q` 为 **1383 passed、8 skipped、18 subtests passed**（502.67 s），日志 `runs/grid_update/visibility_full_final.log`；8 项跳过为 CRESSim 上游构建。

## 2026-10-23：有限面精确判定的区间筛选

RTX 4090，`4d6e6ac`、dirty=true。相关 **15 passed**（3.65 s），新增 **7** 项，覆盖 1e-150～1e150 尺度的独立有理数重心求交对照、大坐标下 2⁻⁶⁰ 偏移、零面积和零长度。完整/部分交叉的 **38784/29184** 条射线与 `4d6e6ac` 的精确判定差均为 **0**，耗时从 **8.556/4.735 s** 降至 **1.294/0.7021 s**；三种尺度随机三角片的判定差也均为 **0**。

命令前缀 `OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python`：`-m pytest tests/test_reference_visibility.py -q`；`runs/e5_release_audit/visibility_filter_validate.py`。报告/日志为 `e5-grid-20261023/runs/grid_update/visibility_filter_{unit.log,source.json,source.log}`。提交前全量 **1390 passed、8 skipped、18 subtests passed**（485.87 s），日志 `visibility_filter_full.log`；8 项跳过为 CRESSim 上游构建。


## 2026-10-23：切面出生朝向与永久侧码

RTX 4090，`c0e09f7`、dirty=true。新增 **6** 项朝向检查通过，覆盖 6/4 mm、仿射形变与整体旋转、退刀重入、新刀追加及状态恢复；保存朝向与显式侧码的材料状态逐位一致，初速度伴随按 rtol=**2e-13**、atol=**1e-13** 通过。连同历史和可见性检查为 **31 passed**（23.14 s）。六组自然事件的 **1.6 ms** 几何观察中，四组生成 **1728/1140/1140/1464** 个积分点，两组未生成参考片；旧点朝向和损伤保持不变，三线性重现误差 ≤**1.333e-15**，恢复后的映射权重差 **0**。观察使用原自然切割力学推进，内聚力与 RK2 的联合接入继续进行。

命令前缀 `OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python`：`-m pytest tests/test_cohesive_orientation.py tests/test_cohesive_history.py tests/test_reference_visibility.py -q`；`runs/e5_release_audit/birth_orientation_natural.py`。报告/日志在 `e5-grid-20261023/runs/grid_update/birth_orientation_{unit.log,natural.json,natural8.log}`。提交前全量 `-m pytest tests -q` 为 **1396 passed、8 skipped、18 subtests passed**（489.97 s），日志 `birth_orientation_full.log`；8 项跳过为 CRESSim 上游构建。


## 2026-10-23：当前公式的 E3 六毫米下刀步长细化

RTX 4090，干净 `ca90da9`，命令 `PYTHONPATH=. .venv/bin/python runs/e3_ca90da9/run_case.py cut_h6_refine`。0.24 s、15.625/7.8125/3.90625 μs 的 **12** 组自然差分全部通过，最大误差 **0.009032%**；三档 32 步检查点状态重算差 **0**。相邻位置最大差 **5.513→1.591 nm**，横向/深度梯度相邻差 **0.11836%→0.011813% / 0.18223%→0.11009%**。报告 `e3-current-20261022/runs/e3_ca90da9/cut_h6_refine/report.json`；队列继续完整轨迹细化及 4 mm 活跃释放补充检查。

## 2026-10-23：闭合界面弱约束与参考边界

RTX 4090，`0b279f6`、dirty=true。相关 **27 passed**（64.68 s），新增九项：6/4 mm 的平移、旋转和非均匀运动与自由材料的状态/初速度伴随一致；奇异值 **1、2⁻²⁰、2⁻³⁵** 的解析闭合约束及反向通过；外侧参考积分点保持三线性值、空间梯度和位置伴随。六组 **3.2 ms** 自然事件几何观察通过，累计 **196** 个接受段，末端 **2952/2292/2292/2637/471/471** 个积分点，多项式重现误差 ≤**1.333e-15**，恢复权重差 **0**。

实际自然事件生成的 **288** 点、39 网格槽界面完成一次 **2 μs** 材料推进，耗时 **0.1533 s**，最大历史 **0**。**576** 点、47 槽例的非线性线搜索在第九次迭代失败，归一化残差 **2.55145e-14**；直接闭合候选超出原强度界。残差缩放和约束零空间原型均未解决该例，继续核对承载可行性与活跃约束切换。

命令前缀 `OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python`：`-m pytest tests/test_cohesive_closed_motion.py tests/test_reference_grid.py -q`；`runs/e5_release_audit/birth_orientation_natural16_fixed.py`；`runs/e5_release_audit/natural_coupling_profile.py --segments 4 --output runs/grid_update/natural_coupling_factored4_final.json`（576 点使用 `--segments 8`）。报告/日志在 `e5-grid-20261023/runs/grid_update/`。提交前全量 `-m pytest tests -q` 为 **1405 passed、8 skipped、18 subtests passed**（507.17 s），日志 `closed_boundary_full.log`；8 项跳过为 CRESSim 上游构建。

## 2026-10-23：闭合残差与近起裂牵引的反向精度

RTX 4090，`9ddcd27`、dirty=true。新增 **20** 项解析检查在旧提交全部失败，当前全部通过：8 项闭合残差的反力导数严格为零、历史单侧导数保留；12 项牵引/总力在起裂距离外 **2⁻²⁰/2⁻³⁵/2⁻⁴⁵** 相对增量处，径向力与储能导数符合解析值，数值分裂尺度覆盖 λAσ/δ_f=**0.17/1e-12**。连同近端和面积响应测试为 **59 passed**（1.78 s）。

576 点固定输入的自适应 ADMM、归一化最小二乘和约束分块原型完成平衡求解，扩展精度残差 **6.77012e-29**，最大开口 **4.73853e-18 m**；材料强度和原求解容差保持原值。界面位移目标的完整伴随检查通过原 **1e-10** 缩放标准。实际材料一步推进的原型通过一次伴随残差细化，将残差从 **0.0130086** 降至 **4.01624e-5**，相对右端尺度为 **3.33e-13**；正向/反向耗时 **10.8963/7.3093 s**，状态及速度梯度有限。自适应非线性求解和材料伴随细化仍在原型中，继续整理通用入口及参数差分。

命令前缀 `OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python`：`-m pytest tests/test_cohesive_small_opening.py tests/test_cohesive_closed_residual.py tests/test_cohesive_proximal.py tests/test_cohesive_measure.py -q`；`runs/e5_release_audit/natural_adaptive_admm.py --resume --tolerance 1e-10 --iterations 200000 --name natural_adaptive_precise`；`runs/e5_release_audit/natural_adaptive_newton.py --seed natural_adaptive_precise --scaled --nullspace`；`runs/e5_release_audit/natural_interface_adjoint.py`；`runs/e5_release_audit/natural_material_coupling.py --refine-adjoint`。报告在 `e5-grid-20261023/runs/grid_update/`。提交前全量 `-m pytest tests -q` 为 **1425 passed、8 skipped、18 subtests passed**（505.85 s），日志 `small_opening_full.log`；8 项跳过为 CRESSim 上游构建。

## 2026-10-23：内聚材料伴随残差细化

RTX 4090，`b28744c`、dirty=true。**16** 项链式界面的解析梯度检查通过（0.57 s），覆盖 1/4 个界面、1e-12/1e-18 m 开口、1e-10/1e-13 m/N 分裂尺度及压力/总力接口；相同检查在旧提交为 **12 failed、4 passed**。解析值由消去反力后的弹性切线计算，包含位移和反力目标。

576 点自然界面的材料一步推进已使用正式伴随入口通过，正向/反向 **11.0320/7.3650 s**，状态和初速度梯度有限，最大初速度伴随 **2.54044**。初速度方向的 ε=1e-3/1e-4 两档差分相对误差 **3.673e-14/3.496e-14**。固定参考几何的界面自由开口/强度各两档差分全部通过，最大相对误差 **4.852e-8**；每次扰动重新求解平衡及活跃集合，单组差分耗时 **73.49–731.84 s**。正向仍使用非线性原型和收敛解热启动；正式非线性入口和完整自然事件接入继续完成。

命令前缀 `OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python`：`-m pytest tests/test_cohesive_adjoint_refinement.py -q`；`runs/e5_release_audit/adjoint_refinement_old.py`；`runs/e5_release_audit/natural_material_coupling.py`；`runs/e5_release_audit/natural_material_difference.py`；`runs/e5_release_audit/natural_difference_adaptive.py`。日志及报告在 `e5-grid-20261023/runs/grid_update/`，对应 `adjoint_refinement_unit2`、`adjoint_refinement_old_test`、`natural_material_source_refined`、`natural_material_difference`、`natural_difference_adaptive`。

提交前全量 `PYTHONPATH=. .venv/bin/python -m pytest tests -q` 为 **1441 passed、8 skipped、18 subtests passed**（511.56 s），日志 `runs/grid_update/adjoint_refinement_full.log`；8 项跳过为 CRESSim 上游构建。


## 2026-10-23：通用内聚非线性入口与零初值复验

RTX 4090，`1664492`、dirty=true。自适应固定法向初值、移动法向平衡及反力块消元已进入正式求解器；新增 **42** 项解析检查通过，相关检查 **194 passed**（97.09 s）。自然事件生成的 **576** 点、47 网格槽界面从零初值完成材料一步推进，正向/反向 **109.7196/7.3879 s**；初速度两档差分最大相对误差 **3.674e-14**。固定参考几何的自由开口和强度各两档差分全部通过，最大相对误差 **4.8591e-8**；平衡扩展残差 **1.00993e-28**。每次扰动独立从零初值求解。

闭合承载、G_c=7/70 受载断裂、已有损伤四组 200 μs 材料轨迹共 **12** 例能量检查通过；dt=2/1/0.5 μs 的残差随步长减半约减半，最细档幅值 **0.017552%/0.002262%/0.007322%/0.348344%**。闭合例开口及损伤为 **0**，完全断裂耗散 **0.001008/0.01008 J=G_c A₀**。

命令前缀 `OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python`：`-m pytest tests/test_cohesive_initial.py tests/test_cohesive_linear.py tests/test_differentiable_cohesive.py tests/test_cohesive_force.py tests/test_cohesive_grid_update.py -q`；`runs/e5_release_audit/natural_generic_validate.py`；`runs/e5_release_audit/natural_generic_material.py`；`runs/e5_release_audit/adaptive_energy.py`。报告/日志在 `e5-grid-20261023/runs/grid_update/`，对应 `adaptive_general_unit3`、`natural_generic_validate`、`natural_generic_material`、`adaptive_general_energy`。提交前全量 `PYTHONPATH=. .venv/bin/python -m pytest tests -q` 为 **1483 passed、8 skipped、18 subtests passed**（519.75 s），日志 `adaptive_general_full.log`。下一项连接自然分段、接触去重和永久历史，再完成整体能量扫描及受影响的 E1–E4 复验。


## 2026-10-23：当前公式的 E3 完整轨迹步长细化

RTX 4090，干净 `ca90da9`，命令 `PYTHONPATH=. .venv/bin/python runs/e3_ca90da9/run_case.py full_h6_refine`。6 mm、1.05 s、15.625/7.8125/3.90625 μs 三档的 128 步检查点状态重算差均为 **0**；最细相邻位置最大差 **17.426 nm**，三项刀路梯度差 **0.010024%/0.080899%/0.133284%**。侧码、接触对与事件集合一致，事件时刻差 **23.994 ns**，三档均未达到损伤释放。报告 `e3-current-20261022/runs/e3_ca90da9/full_h6_refine/report.json`；4 mm 活跃释放继续 dt/8、dt/16 细化。

## 2026-10-23：移动界面速度冲量与分阶段原型

RTX 4090，`7c9c470`、dirty=true。新增 **20** 项碰撞检查通过（0.93 s），覆盖不等质量、同时接触、重复/零/分离约束、1e-8～1e8 行缩放、动量/动能损失、移动法向及两档输入差分。正式速度投影接入平面碰撞原型后，7/16/32/64 步均得到 **115 μs** 事件时刻，动量差 **0**，动能损失 **1.333333e-5 J** 与解析值一致；位置、速度、质量、法向及总时长共十组差分最大相对误差 **2.539e-7**。

`7c9c470`、dirty=false 的 ARS 材料原型：闭合/已有损伤/受载开裂三类各 8 步，29 个输入方向共 **58** 组差分完成；非零导数最大相对误差 **2.392e-6**，零导数最大绝对差 **1.241e-10**。32 μs、dt=2/1/0.5 μs 的九例能量残差均随步长减小，闭合开口及历史为 **0**。已有损伤的速度相邻差 **6.796e-8→1.699e-8 m/s**；受载开裂相邻差 **0.002365→0.001076 m/s**，继续核对起裂事件。解析受载双点例在 8/16/32/64/128 步下速度误差 **1.0298e-4→4.0307e-7 m/s**，每次细化约缩小四倍。粒子算例去掉初始旋转时也呈二阶收敛；原旋转场经当前分侧网格映射后产生 **0.225/0.09 m/s** 切向速度差。继续核对仿射运动重现和出生时刻的速度相容性，分阶段推进及自然碰撞连接仍在原型中。

命令前缀 `OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python`：`-m pytest tests/test_cohesive_impact.py -q`；`runs/e5_release_audit/cohesive_impact_probe.py`；`runs/e5_release_audit/cohesive_ars_material.py`；`runs/e5_release_audit/cohesive_ars_material_grad.py`。报告/日志在 `e5-grid-20261023/runs/grid_update/`，对应 `impact_tests`、`cohesive_impact_probe`、`cohesive_ars_material`、`cohesive_ars_material_grad`。

刚体旋转检查失败：dt=2 μs、32 μs 轨迹产生 **1.5471 μm** 损伤历史；dt=1 μs 在第 28 步反力选择报约束不相容。纯旋转经原型的分侧速度映射产生非零切向跳量，下一项核对仿射重现和界面速度/功的共同映射。命令为 `runs/e5_release_audit/cohesive_ars_rigid.py`，报告及复现输入为 `cohesive_ars_rigid.json`、`ars_rigid_failure.npz`。

提交前全量 `PYTHONPATH=. .venv/bin/python -m pytest tests -q` 为 **1503 passed、8 skipped、18 subtests passed**（454.94 s），日志 `runs/grid_update/impact_full.log`；8 项跳过为 CRESSim 上游构建。


## 2026-10-23：参考仿射速度传递及共享场的精确消元

RTX 4090，`4f29254`、dirty=true。`reference_affine` 新增 **13** 项检查，连同参考迹和弱约束检查为 **40 passed**（46.64 s）。覆盖 6/4 mm 方块/圆柱、扰动采样、仿射运动、线动量/参考角动量、网格传递动能、虚功及移动切面导数。同节点完全相容场先合并动量，迹按等价速度场合并可见质量；共享开口映射与切面位置导数严格为 **0**。自然事件生成的 **288/576/1152** 点参考界面在八步旋转检查中均保持零开口、零损伤。

32 μs、dt=2/1/0.5 μs 的纯旋转、预载旋转、已有损伤与受载开裂共 **12** 例完成。前两组开口/损伤为 **0**；受载开裂能量相对残差 **8.80347e-8/1.10049e-8/1.37588e-9**，相邻速度最大差 **6.14264e-5→1.53614e-5 m/s**。三类八步轨迹共 **58** 组差分，非零导数最大相对误差 **3.669e-6**，零导数最大绝对差 **2.997e-11**。切面位置、法向、平移/仿射速度和面积到八步材料状态的十组差分最大相对误差 **7.090e-7**。

同版 200 μs 塑性原型中，64 粒子均发生塑性更新，屈服应力 200 Pa。dt=2/1/0.5 μs 的能量相对残差为 **5.65980e-5/2.82956e-5/1.41469e-05**，塑性耗散增量非负，界面损伤为 **0**；能量收支包含每步固定总形变下塑性返回的储能下降。报告 `affine_plastic_probe.json`，命令 `runs/e5_release_audit/affine_plastic_probe.py`。

命令前缀 `OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python`：`-m pytest tests/test_reference_affine.py tests/test_reference_grid.py tests/test_cohesive_closed_motion.py -q`；`runs/e5_release_audit/affine_source_validate.py`；`runs/e5_release_audit/affine_material_grad_source.py`；`runs/e5_release_audit/affine_chain_probe.py`；`runs/e5_release_audit/natural_affine_patch.py`。报告/日志在 `e5-grid-20261023/runs/grid_update/`，同名 JSON 及 `affine_unit.log`。下一项连接分阶段材料推进、自然事件与接触历史，再核对完整切割能量。

提交前全量 `OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m pytest tests -q` 为 **1516 passed、8 skipped、18 subtests passed**（496.71 s），日志 `runs/grid_update/affine_full_final.log`；8 项跳过为 CRESSim 上游构建。

## 2026-10-23：E3 活跃释放的 4 mm 追加细化

干净 `ca90da9`，RTX 4090，命令 `PYTHONPATH=. .venv/bin/python runs/e3_ca90da9/release_h4_more.py`。50 ms、G_c=7、dt=1.64473684 μs，释放 **5** 粒子；四组自然差分最大相对误差 **4.921e-7**，32 步检查点全部状态重算差 **0**。与 dt=3.28947368 μs 相比，位置差 **0.184 nm**，横向/深度梯度差 **0.013614%/0.219121%**；侧码、释放及事件集合一致，事件时刻差 **1.022 ns**。报告 `e3-current-20261022/runs/e3_ca90da9/release_h4_more/report.json`。

同报告的 dt/16=**0.822368421 μs**、60800 步细化已完成，队列退出码 **0**。32 步检查点状态重算差 **0**，释放仍为 **5** 粒子；最细相邻位置差 **0.0545 nm**，横向/深度梯度差 **0.0019415%/0.0269824%**，E/G_c 梯度差 **0.0280664%/0.0209865%**；侧码、释放、接触对及事件集合一致，事件时刻差 **0.2658 ns**。该档执行状态/梯度细化，刀路有限差分在 dt/8 执行。独立正向 **725.48 s**，检查点正向/重算/反向 **524.78/546.86/216.52 s**。

## 2026-10-23：仿射材料分阶段推进与永久历史入口

RTX 4090，`5212332`、dirty=true。`cohesive_stages.advance_affine` 和 `CohesiveHistory.advance` 已连接独立仿射状态、两阶段内聚求解及永久损伤。自由落体的端点/查询中点/时长导数符合解析解；三类轨迹重启逐位一致；分段反向保留应力、仿射速度及界面伴随。已有损伤/受载开裂的八步轨迹共 **44** 组输入差分通过（rtol=2e-5、atol=2e-8），活跃塑性的 E/屈服应力/仿射速度共 **6** 组差分通过（rtol=2e-5、atol=1e-8）；64 粒子全部塑性更新，塑性体积比按 rtol=2e-12 通过，逐步塑性耗散非负。

公开入口的 **15** 例能量/步长检查完成：纯旋转及预载旋转均零开口、零损伤；受载开裂在 dt=2/1/0.5 μs 的能量相对残差 **8.80347e-8/1.10049e-8/1.37588e-9**，相邻速度差 **6.14264e-5→1.53614e-5 m/s**。200 μs 塑性例的残差 **5.65980e-5/2.82956e-5/1.41469e-5**，耗散 **2.04170e-6/2.03326e-6/2.02904e-6 J**；动能包含独立仿射矩。

命令前缀 `OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python`：`-m pytest tests/test_cohesive_stages.py -q`；`-m pytest tests/test_cohesive_stages.py -k 'retained_surface or active_plastic' -q`；`runs/e5_release_audit/stages_source_validate.py`。定向检查 **8 passed**（139.91 s）及追加 **2 passed**（5.67 s）；日志/报告为 `e5-grid-20261023/runs/grid_update/stages_{unit.log,extra.log,source_validate.json}`。下一项接入固定耗散率、刀/板与界面闭合接触、在线切面出生和自然事件重算。

提交前全量 `OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m pytest tests -q` 为 **1527 passed、8 skipped、18 subtests passed**（568.50 s），日志 `runs/grid_update/stages_full.log`；8 项跳过为 CRESSim 上游构建。


## 2026-10-23：固定率仿射传递耗散原型

RTX 4090，`5212332`、dirty=true。6/4 mm 方块、扰动方块和圆柱六例，η=**2000 s⁻¹**，200 μs 的随机速度/仿射矩耗散均为正；一次推进与四次等分最大差 **1.5544e-15 m/s**，全局仿射场最大变化 **5.8634e-16 m/s**。速度、仿射矩、时长、速率共 **48** 组两档差分最大相对误差 **8.691e-8**。

对称半步组合的纯旋转、受载开裂、塑性及含随机速度的已有损伤共 **12** 例完成。已有损伤三档的实际传递耗散 **9.6220254e-7/9.6220258e-7/9.6220259e-7 J**，能量相对残差 ≤**2.05e-12**，最细相邻速度差 **2.6805e-8 m/s**，永久历史保持 **1e-3 m**。仿射运动例的计算耗散处于舍入尺度（单步最小 **−3.69e-18 J**），按带符号原值计入账本。方法和本地推导见 METHOD；组合的时长链及接触联合推进继续核对。

命令前缀 `OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python`：`runs/e5_release_audit/affine_damping_probe.py`；`runs/e5_release_audit/stages_damping_validate.py`；`runs/e5_release_audit/stages_damping_opened.py`。报告在 `e5-grid-20261023/runs/grid_update/`，同名 JSON。


## 2026-10-23：固定率耗散接入分阶段推进

RTX 4090，`f2506bf`、dirty=true。`reference_damping.decay` 已连接 `cohesive_stages.advance_affine`，输出两半步的动能变化，半步时长、耗散率和速度矩保留反向链。六种网格/形状的 **48** 组差分最大相对误差 **8.691e-8**；200 μs 一次推进与四次等分的速度差 ≤**1.5544e-15 m/s**，全局仿射场变化 ≤**5.8634e-16 m/s**。

正式入口的 **18** 例能量/步长检查完成，η=2000 s⁻¹、dt=2/1/0.5 μs。含随机速度的已有损伤例：耗散 **9.6220254e-7/9.6220258e-7/9.6220259e-7 J**，能量相对残差 ≤**2.05e-12**，相邻速度差 **1.07227e-7→2.68050e-8 m/s**。纯旋转、预载旋转均零开口、零损伤；受载开裂最细档能量相对残差 **1.37588e-9**；塑性三档残差 **5.65980e-5/2.82956e-5/1.41469e-5**。仿射场的最小单步计算耗散为 **−5.32e-18 J**，账本保留舍入后的带符号原值。

命令前缀 `OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python`：`runs/e5_release_audit/damping_source_gradient.py`；`runs/e5_release_audit/damping_source_energy.py`；`-m pytest tests/test_reference_damping.py tests/test_cohesive_stages.py -q`。报告/日志在 `e5-grid-20261023/runs/grid_update/damping_*`。下一项连接刀/板接触、界面闭合碰撞和自然切面出生，再核对整体切割能量。

定向检查 **23 passed**（174.08 s）；追加分段反向核对 **2 passed**（12.71 s），覆盖零/非零速率下速度矩、时长及速率梯度。八步联合推进新增 **8** 组输入差分通过（rtol=2e-5、atol=2e-8），同时对端点、查询中点、界面状态和耗散能量求导。对角解析算子覆盖零时长、零速率及 ηt=200；两类守恒量和重启检查通过。

闭合界面非均匀速度检查：dt=2/1/0.5 μs 的开口与损伤均为 **0**，能量相对残差幅值 **1.8342e-8/9.3787e-9/4.7413e-9**；末端开口速度残差 **1.6626e-5/8.3128e-6/4.1563e-6 m/s**，随步长呈一阶下降。耗散半步不保持闭合速度约束；第 6 项将约束和耗散联合推进，再核对约束残差、能量与反向。命令 `runs/e5_release_audit/damping_closed_probe.py`，报告 `runs/grid_update/damping_closed_probe.json`。

提交前全量 `OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m pytest tests -q` 为 **1540 passed、8 skipped、18 subtests passed**（603.60 s），日志 `runs/grid_update/damping_full.log`；8 项跳过为 CRESSim 上游构建。

## 2026-10-23：闭合约束子空间耗散原型

RTX 4090，`f2506bf`、dirty=true。闭合非均匀速度使用 ΠLΠ 的生成算子，在 dt=2/1/0.5 μs 下保持零开口、零损伤，末端开口速度残差 ≤**4.99e-16 m/s**；能量相对残差幅值 **7.6074e-11/1.9002e-11/4.7483e-12**。相邻速度差 **5.0937e-10→1.2736e-10 m/s**，比值约 4。固定秩约束映射、速度、时长、速率共 **32** 组差分最大相对误差 **7.106e-8**；约束方向导数与 SciPy `expm_frechet` 对照的相对差 ≤**7.801e-15**，4/8 点积分的伴随矩阵最大差 **1.364e-12**。

命令前缀 `OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python`：`runs/e5_release_audit/damping_constrained_probe.py`；`runs/e5_release_audit/constrained_damping_adjoint.py`。报告 `runs/grid_update/{damping_constrained_probe,constrained_damping_adjoint}.json`。该固定闭合投影在 κ→0⁺ 的连续性检查失败；下一项核对耗散与内聚反力的联合隐式推进。

## 2026-10-23：近闭合界面的弱导数与残差表达

RTX 4090，`bf0c2c9`、dirty=true。反力 Jacobian 直接使用 −Dprox；卸载时按操作数大小选择等价残差，加载保留牵引差。κ/(λσ) 从 **1e-18** 到 **1e3** 的八例解析核对通过；最小比值处旧表达的反力块相对误差 **1**，修复后所有比值及压力/积分总力表达的最大误差 **2.171e-16**。材料势、强度和最终求解容差保持原值。

命令前缀 `OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python`：`-m pytest tests/test_cohesive_closed_residual.py tests/test_cohesive_adjoint_refinement.py tests/test_cohesive_small_opening.py tests/test_cohesive_linear.py -q` 为 **55 passed**（2.53 s）；`runs/e5_release_audit/residual_jacobian_compare.py` 保存新旧表达对照。全量 `-m pytest tests -q` 为 **1548 passed、8 skipped、18 subtests passed**（608.00 s），8 项跳过为 CRESSim 上游构建。报告/日志在 `runs/grid_update/{residual_jacobian_compare.json,cohesive_residual_tests.log,cohesive_residual_full_final.log}`。

## 2026-10-23：近相关压缩约束的线搜索

RTX 4090，`05b5dcc`、dirty=true。线搜索加入舍入尺度后，64 粒子非均匀速度场的四例近闭合状态通过分裂尺度 1/0.25 的解一致性核对；位移 rtol=**2e-8**、atol=**1e-20 m**，反力 rtol=**2e-8**、atol=**1e-12 N**。原线搜索停滞的 κ=**1e-13/1e-15 m** 两例分别用 **14/7** 次半光滑迭代收敛，最终残差 **7.2124e-25/5.7700e-25**。最终残差标准保持原值，方法来源见 METHOD。

命令前缀 `OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python`：`-m pytest tests/test_cohesive_globalization.py -q` 为 **4 passed**（14.52 s）；全量 `-m pytest tests -q` 为 **1552 passed、8 skipped、18 subtests passed**（623.98 s）。日志 `runs/grid_update/globalization_{unit,full}.log`；新旧线搜索对照为 `runs/e5_release_audit/globalization_generated_cases.py`，报告 `runs/grid_update/globalization_generated_cases.json`。

耗散与内聚反力联合推进的极小历史梯度检查仍失败：κ=**1e-15 m** 时，时长到速度的三档差分相对误差 **2.36e-10/2.17e-9/2.03e-8**，时长到反力为 **0.258/0.707/1.177**。报告 `runs/grid_update/birth_channel_gradient.json`。下一项核对变形增量与反力的舍入误差传播，完成第 6 项联合求解后再接在线切面出生。

## 2026-10-23：联合推进的高精度参考与隐式方向导数

RTX 4090 主机，`eab0d7d`、dirty=false；Warp 对照在 GPU，35/55 位参考求解在 CPU。64 粒子、16 个界面积分点，dt=**2 μs**、η=**2000 s⁻¹**、κ=**1e-15 m**。固定相同局部接触输入后，历史梯度与 55 位差分的相对误差为 **7.892e-8**；保留 H 位移梯度的两阶段原型仍有反力差分失败，耗散坐标变换与几何输入的精度继续影响联合结果。

原坐标联合方程的 55 位参考计算完成两阶段推进，时长与历史的六组中心差分最大相对误差 **1.046e-6**；扰动 1e-2/1e-3/1e-4 时，时长误差 **4.567e-9/4.567e-11/4.567e-13**，历史误差 **1.046e-6/1.046e-8/1.046e-10**。全部扰动通过活跃集合不等式核对。35/55 位的最大反力差 **8.712e-28 N**，两类方向导数的相对差 **3.883e-24/5.781e-21**。

命令前缀 `OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python`：`runs/e5_release_audit/birth_precision_local.py`、`birth_precision_rollout.py`、`birth_precision_check.py`；报告 `runs/grid_update/birth_precision_{local,rollout,check}.json`。下一项为通用求解接口接入 D、补偿几何及一致的隐式反向，以该参考解核对零损伤、加载、卸载、塑性和接触组合；第 6 项仍进行中。正式源码沿用已通过 **1552 passed、8 skipped、18 subtests passed** 的 `eab0d7d`。

## 2026-10-23：联合惯性算子与界面隐式反向

RTX 4090，`f7990be`、dirty=true。`ReferenceInertia` 接入压力/积分总力和补偿几何接口，**29** 项检查通过：独立稠密解对照、零时长/零速率导数、闭合/张开/混合/已断裂四类输入差分、重复及零面积约束、零历史处三档惯性参数的解析导数。动量绝对残差检查 **2e-17**；一般输入差分 rtol=**2e-5**、atol=**2e-7**，零历史解析导数 rtol=**2e-10**。

命令 `OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m pytest tests/test_reference_inertia.py -q -x`：**29 passed**（13.08 s），日志 `runs/grid_update/inertia_unit_final.log`。联合推进的极小历史反力精度继续核对；E5 第 6 项进行中，第 7–8 项待完成。

两阶段联合入口 `runs/e5_release_audit/native_inertia_stages.py` 通过已有 **14** 项阶段检查（210.48 s），包括活跃塑性、重启、分段反向及耗散参数梯度。κ=**1e-15 m** 时，反力与 55 位参考的最大差 **1.378e-10 N**，时长/历史方向导数的相对差 **2.1492%/0.072923%**；六组双精度差分仍失败。补偿惯性后的自由几何并保留 H 的弹性对照，时长梯度误差为 **1.6583%**，分块伴随对照为 **1.6577%**。下一项核对各阶段固定局部输入的高精度反力与伴随，定位状态误差和局部求解误差的放大。

命令前缀同上：`runs/e5_release_audit/native_inertia_stage_tests.py`、`native_inertia_precision.py`、`native_increment_precision.py`、`native_block_adjoint_precision.py`；日志及 JSON 位于 `runs/grid_update/native_*`。高精度参考使用相同的 P/Pᵀ 分解构造 D；弹性增量对照遇到活跃塑性即报错，正式塑性入口保持完整返回映射。

提交前全量 `OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m pytest tests -q`：**1581 passed、8 skipped、18 subtests passed**（653.50 s），日志 `runs/grid_update/inertia_full.log`；8 项跳过为 CRESSim 上游构建。

## 2026-10-23：几何与惯性反向的求和精度

RTX 4090，`09cf584`、dirty=true。界面运动 VJP 和 D⁻¹ 的参数收缩保留 CPU longdouble 中间量。四个相消余量的解析检查从 **4 failed** 变为通过；时长逆作用导数的解析值 **−0.1875**，旧计算为 **0**。定向检查 **56 passed**（14.80 s）。两阶段 κ=1e-15 m 的时长反力梯度相对高精度参考误差从 **2.1492%** 降为 **0.0019595%**；历史误差 **0.072923%**，小扰动反力差分仍失败。

固定各阶段相同输入的十组局部方向导数最大误差 **8.754e-8**，反力差 ≤**6.234e-15 N**。第一阶段自由间隙舍入差 **3.597e-24 m**；以高精度间隙替换后，参考历史方向导数从 **1.1504991e-5** 恢复到完整公式的 **1.1486097e-5**，单独替换面形变不改变结果。下一项保持惯性逆作用与几何乘积的补偿精度，并核对完整两阶段差分。

命令前缀 `OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python`：`-m pytest tests/test_cohesive_geometry_precision.py tests/test_reference_inertia.py tests/test_cohesive_adjoint_refinement.py tests/test_cohesive_closed_motion.py -q`；`runs/e5_release_audit/native_stage_local.py`、`native_stage_inputs.py`、`chain_precision_rollout.py`。报告/日志为 `runs/grid_update/{native_stage_local,native_stage_inputs,chain_precision_*}`。

55 位惯性逆作用与几何合并计算原型，使两阶段时长/历史反力梯度相对参考误差降为 **1.551e-7/1.640e-7**，最大反力差 **2.590e-14 N**。三档中心差分相对误差：时长 **3.833e-6/3.565e-5/1.164e-4**，历史 **3.356e-6/1.103e-4/4.318e-3**；四个较小扰动仍超过原 rtol=2e-5。入口 `runs/e5_release_audit/precise_fused_rollout.py`，报告 `runs/grid_update/precise_fused_rollout.json`。下一项核对局部平衡解的舍入误差并实现通用补偿作用量。

提交前全量 `OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m pytest tests -q`：**1585 passed、8 skipped、18 subtests passed**（634.54 s），日志 `runs/grid_update/chain_precision_full.log`；8 项跳过为 CRESSim 上游构建。

## 2026-10-23：通用近端残差与补偿作用量对照

RTX 4090 主机，`9250750`、dirty=false；状态与隐式反向在 GPU，高精度残差对照在 CPU。通用近端方程的 **10** 组参考面旋转/压力或总力算例通过，覆盖闭合、加载、混合历史、卸载、完全断裂；**16** 组时长/历史差分最大相对误差 **7.190e-8**。命令 `OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python runs/e5_release_audit/generic_precise_branches.py`。

两个 99 维阶段 Jacobian 的补偿 GMRES 细化，经三次修正与 55 位解的相对差为 **2.653e-30/1.815e-31**，各耗时约 **0.032 s**。P/Pᵀ 补偿惯性逆作用的零时长、零速率、ηt=0.004/20 四例通过，最大相对误差 **2.811e-31**，各耗时 **0.020–0.022 s**。命令前缀同上：`runs/e5_release_audit/compensated_linear_probe.py`、`compensated_inertia_probe.py`。全部报告位于 `runs/grid_update/` 下同名 JSON。

两阶段弹性原型保留广义速度线性更新的 55 位副本，κ=1e-15 m、dt=2 μs、η=2000 s⁻¹。原三档扰动 1e-2/1e-3/1e-4 共 **6/6** 组差分通过 rtol=2e-5：时长相对误差 **6.168e-7/6.123e-7/6.124e-7**，历史 **1.038e-6/3.172e-9/3.065e-8**。基准反力与独立高精度参考最大差 **2.169e-19 N**；时长/历史隐式梯度误差 **6.123e-7/7.196e-9**。命令 `runs/e5_release_audit/shadow_moment_rollout.py`，报告 `runs/grid_update/shadow_moment_rollout.json`。仅提高局部残差精度或另保留材料 H 时，最小历史扰动误差均约 **4.017e-4**；阶段间速度余量需同时保留。

下一项：将补偿作用量、局部平衡细化及广义速度余量接入正式联合推进，核对塑性、重复约束、接触和能量。E5 第 6 项继续，第 7–8 项待完成。正式源码 `9250750` 的全量结果为 **1585 passed、8 skipped、18 subtests passed**，上述原型和报告保存在 runs。

## 2026-10-23：补偿状态与惯性／几何接口

RTX 4090，`f555c3f`、dirty=true。`CompensatedArray` 保存高位/余量、线性增量和快照；`ReferenceInertia.solve_state` 及 `InterfaceGeometry.advance/extrapolate` 使用两分量作用量。原型中替换惯性／自由几何计算后的六组差分 **6/6** 通过，最大相对误差 **1.132e-6**；基准反力与 55 位参考最大差 **2.169e-19 N**。时长/历史梯度相对参考误差 **1.132e-6/7.198e-9**。命令 `OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python runs/e5_release_audit/formal_geometry_rollout.py`；报告 `runs/grid_update/formal_geometry_rollout.json`。

局部残差/Jacobian 的两分量原型通过 **10** 组本构分支、**16** 组时长/历史差分，最大相对误差 **7.190e-8**。两阶段基准的反力差 **2.169e-19 N**，时长/历史梯度误差 **2.492e-7/7.196e-9**；99 维 Jacobian 与 55 位表达式的相对差 ≤**5.458e-33**。该对照的基底构造和阶段速度副本仍用 55 位计算。命令前缀同上：`runs/e5_release_audit/compensated_residual_branches.py`、`compensated_residual_rollout.py`；报告为 `runs/grid_update/` 下同名 JSON。下一项接入补偿局部平衡、位移余量与完整阶段状态。

正式代码的解析检查覆盖 **2⁸⁰** 相消余量、**2⁻¹⁰²⁰/2¹⁰²⁰** 输入的乘积、90 位 Decimal 独立惯性解、零率导数和状态重启。全量 `OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m pytest tests -q`：**1599 passed、8 skipped、18 subtests passed**（712.26 s），日志 `runs/grid_update/compensated_state_full.log`；8 项跳过为 CRESSim 上游构建。

局部基底构造、残差和 Jacobian 全部使用两分量算术的原型，通过 **10** 组分支、**16** 组参数方向及两阶段 **6/6** 组差分。时长三档误差 **1.989e-6/1.994e-6/1.994e-6**，历史 **1.039e-6/2.998e-9/9.750e-10**；反力与独立参考差 **1.085e-19 N**。命令 `runs/e5_release_audit/paired_equilibrium_branches.py`、`paired_equilibrium_fd.py`，报告为 `runs/grid_update/` 下同名 JSON。阶段速度副本仍由 55 位对照传递；下一项将局部求解及位移余量纳入正式阶段状态，并核对重复/零面积约束与塑性。

## 2026-10-23：局部位移余量和最小范数条件

RTX 4090，`95efe9d`、dirty=true。界面响应新增 `displacement_state`，保留局部细化位移的低位并进入 `CompensatedArray` 增量、快照及反向链。D=I 时的闭合位移采用扩展精度行空间再正交化；解析两粒子例原高位误差 **1.059e-22 m**，修复后 **2⁻⁸¹ m** 余量及两档总间隙差分通过。新增 **8 passed**（0.61 s），相关约束／惯性／梯度 **55 passed**（26.10 s）。命令前缀 `OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python`：`-m pytest tests/test_cohesive_displacement_precision.py -q`；相关组合加 `tests/test_cohesive_linear.py tests/test_cohesive_closed_motion.py tests/test_reference_inertia.py`。日志 `runs/grid_update/displacement_precision_{final_unit,qr_unit}.log`。

`95efe9d`、dirty=false 的两分量非线性原型重复约束检查为 **42 passed、1 failed**（71.31 s，遇首个失败停止）。失败位于三条相关接触约束：三档载荷的浮点自由间隙兼容残差为 **−4.930e-32/5.294e-23/5.421e-20 m**；70 位互补解在后两档选择另一接触集合。保留自由间隙余量的三档反力及六组面积差分通过原标准。命令 `runs/e5_release_audit/constrained_pair_tests.py`、`constrained_contact_case.py`、`constrained_contact_compensated.py`；日志 `runs/grid_update/constrained_pair_newton_tests.log`、`constrained_contact_compensated.log`。下一项保持切面运动的几何关系，处理非线性求解中的接触集合切换，再连接完整阶段状态。

70 位独立接触对照，在相关约束的自由间隙增加 **±2⁻⁸⁰ m**，活跃集合分别为 `{0,1}` 和 `{1,2}`。单点反力差 **0.002 N**，粒子合力差 **1.379e-19 N**，位移差 **5.515e-25 m**；两侧动量残差为 0。命令 `runs/e5_release_audit/contact_boundary_reference.py`，报告 `runs/grid_update/contact_boundary_reference.json`（`95efe9d`、dirty=true）。下一项在保留几何关系的状态参数化下核对冗余反力梯度，并对独立间隙引起的接触切换核对单侧响应及集合重建。

提交前全量 `OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m pytest tests -q`：**1607 passed、8 skipped、18 subtests passed**（707.95 s），日志 `runs/grid_update/displacement_precision_full.log`。8 项跳过为 CRESSim 上游构建。

## 2026-10-23：界面输出与伴随的总几何

RTX 4090，`74971d6`、dirty=true，工作树 `~/Projects/BladeMaster-Warp-e5-state-20261023`。本构输出和局部伴随共用补偿后的间隙、面形变及位移；零面积行保留自由几何。**±2⁻⁸⁰/±2⁻¹⁰⁰ m** 四例的间隙、法向间隙、损伤历史、储能和断裂计功与解析值一致，解析导数及两档差分通过。相关检查 **31 passed**（2.38 s）。命令 `OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m pytest tests/test_cohesive_displacement_precision.py tests/test_cohesive_closed_motion.py tests/test_cohesive_small_opening.py -q`；日志 `runs/grid_update/geometry_output_unit.log`。

`74971d6` 的非线性原型在三档不相容自由间隙上与 70 位接触参考解一致，原方程相对残差为 **9.923e-31/9.910e-31/9.949e-31**；十组旋转面／本构分支和 16 组参数差分通过，最大相对误差 **7.190e-8**。重复切面加载的面积加权反力选择通过 **18** 组输入方向差分（124.46 s）；加载、混合历史、卸载和强度边界四例的反力差 ≤**2.443e-15 N**。强度边界两入口共 **8** 组差分通过（15.69 s）；三档载荷的面积／载荷方向共 **12** 组差分通过，载荷方向最大相对误差 **2.454e-12**，面积方向最大绝对误差 **1.557e-15**。原型以递减正则项生成初值，接受解仍核对原方程；下一项是其余冗余约束导数、边界接触集合与完整阶段状态。命令前缀同上，入口 `runs/e5_release_audit/regularized_contact_nonmonotone.py`、`regularized_contact_large.py`、`regularized_equilibrium_branches.py`、`regularized_reaction_tests.py`、`regularized_forward_checks.py`、`regularized_boundary_tests.py`、`regularized_contact_compensated.py`；报告／日志在 `runs/grid_update/` 下同名文件。接触与分支报告 dirty=false，反力选择报告 dirty=true。

提交前全量 `OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m pytest tests -q`：**1611 passed、8 skipped、18 subtests passed**（936.22 s）。日志 `runs/grid_update/geometry_output_full.log`；8 项跳过为 CRESSim 上游构建。

## 2026-10-24：总间隙分支与联合推进数值核对

RTX 4090，`1ecc419`、dirty=true。共享界面判断改用推进后的总间隙；压力／积分力形式、主值／余量两种存储共四例的平衡残差和解析导数通过，相关检查 **7 passed**（0.58 s）。命令 `PYTHONPATH=. .venv/bin/python -m pytest tests/test_cohesive_geometry_precision.py -q`；日志 `runs/grid_update/shared_remainder_after.log`。

近端尺度平衡原型的两阶段时长／历史差分 **6/6** 通过，最大相对误差 **1.0464e-6**；反力与高精度参考最大差 **1.0843e-19 N**，两项 AD 与参考导数误差分别为 **1.4622e-7/1.9010e-12**。命令 `OPENBLAS_NUM_THREADS=1 PYTHONPATH=.:runs/e5_release_audit .venv/bin/python runs/e5_release_audit/scaled_equilibrium_fd.py`；报告 `runs/grid_update/scaled_equilibrium_fd_balanced_response.json`。活跃塑性 8 步的 16 个前向平衡通过，反向在隐式伴随残差检查失败（**1.1934e6**）；继续核对其矩阵尺度、秩和原方程 VJP。入口 `runs/e5_release_audit/scaled_plastic_stages.py`，日志 `runs/grid_update/scaled_plastic_stages_dual_forcing.log`。

仿射状态补偿原型的速度往返差 **0**，仿射梯度往返差 **2.308e-32 s⁻¹**；与 90 位十进制矩乘参考差 **6.878e-35 m/s**，虚功转置差 **7.100e-29**。状态快照保留全部主值／余量，坐标转换反向与恒等映射一致。命令前缀同上，入口 `runs/e5_release_audit/persistent_affine_probe.py`；报告 `runs/grid_update/persistent_affine_probe.json`。两步轨迹的连续／恢复状态逐项一致，包含位置、速度、仿射梯度、形变及界面余量；入口 `runs/e5_release_audit/persistent_stage_probe.py`。两步完整／检查点反向的速度、仿射梯度、时长和耗散率导数按 rtol=atol=1e-10 通过，入口 `runs/e5_release_audit/persistent_stage_backward.py`。下一项为完整塑性伴随和联合接触。

联合伴随原型对历史 **1e-5/1e-15/1e-22 m** 的两粒子解析梯度均通过，最大相对误差 **1.234e-16**；最小历史例的原方程伴随残差 **1.939e-11**。实际塑性轨迹捕获的 **99×99** 系统在反力收缩及伴随状态都保留余量后，残差 **1.987e-10**，右端尺度 **5.506e6**，通过原 `1e-10 × RHS` 检查。命令前缀同上，入口 `runs/e5_release_audit/adjoint_scale_analytic.py`、`replay_compensated_momentum.py runs/grid_update/implicit_failure_1213893.npz`；报告 `runs/grid_update/adjoint_scale_analytic.json`、`implicit_failure_1213893_compensated_momentum.json`。完整塑性参数差分及冗余约束复验继续运行。

提交前全量 `OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m pytest tests -q`：**1615 passed、8 skipped、18 subtests passed**（984.20 s）。日志 `runs/grid_update/full_shared_geometry.log`；8 项跳过为 CRESSim 上游构建。

## 2026-10-24：补偿伴随正式接口

RTX 4090，`482504f`、dirty=true。界面动量的转置收缩和隐式伴随保留两分量；稀小开口、独立／重复约束、面积、材料、时长和惯性参数共 **204 passed**（121.26 s）。其中新增 11 个线性代数例、3 个动量例和 6 个解析伴随例；保留小至 **2⁻⁸⁰** 的矩阵主元，覆盖 **2⁻¹⁰⁰⁰…2⁹⁰⁰** 右端缩放、秩亏最小范数、32 组输入差分及大幅相反种子的余量。命令 `OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m pytest tests/test_compensated_linalg.py tests/test_cohesive_momentum.py tests/test_cohesive_adjoint_refinement.py tests/test_differentiable_cohesive.py tests/test_reference_inertia.py tests/test_cohesive_force.py -q -x`；日志 `runs/grid_update/formal_compensated_adjoint_unit.log`。

正式接口在捕获的 99 维塑性伴随上通过原残差检查，与两分量原型的十二类输入梯度最大绝对差 **1.103e-39**。命令前缀同上，入口 `runs/e5_release_audit/replay_formal_adjoint.py runs/grid_update/implicit_failure_1213893.npz`；报告 `runs/grid_update/implicit_failure_1213893_formal.json`。保留跨步余量的塑性原型已通过 8 步基准的前向、塑性体积约束和完整反向；六组参数差分及逐步能量核对继续运行，入口 `runs/e5_release_audit/persistent_plastic_stages.py`。

提交前全量 `OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m pytest tests -q`：**1635 passed、8 skipped、18 subtests passed**（1573.45 s）。日志 `runs/grid_update/formal_compensated_adjoint_full.log`；8 项跳过为 CRESSim 上游构建。

## 2026-10-24：约束反力与矩阵运算原型

RTX 4090 工作站，`482504f`、dirty=true。保持原运算顺序的 Numba 矩阵循环在 **30** 组形状／尺度上与 Python 实现逐位一致，100 位十进制参考最大绝对差 **8.528e-32**。同一个 **99** 维平衡算例的求解时间 **28.771→8.848 s（3.252×）**，主值、余量及 Newton 迭代记录完全一致。入口 `compiled_expansion_probe.py`、`compiled_equilibrium_probe.py runs/grid_update/scaled_stage_input_1213893.npz`，报告在 `runs/grid_update/` 下同名 JSON；脚本位于 `runs/e5_release_audit/`。命令前缀为 `OPENBLAS_NUM_THREADS=1 PYTHONPATH=.:runs/e5_release_audit:../BladeMaster-Warp-e5-grid-20261023/runs/e5_release_audit .venv/bin/python`。

同版本的反力选择原型在 **2⁻⁸⁰ m** 开口余量上保留输出导数 **1**，普通几何重分类得到 **0**；压力／积分力两种接口均通过解析核对。强度边界的两种总力接口及 **8** 组面积／强度差分通过（6.85 s）；三种旋转下拉伸、剪切、压缩剪切共 **9** 例与 90 位解析反力最大差 **3.689e-32 N**。入口 `canonical_remainder_probe.py`、`canonical_scaled_strength.py`、`constrained_projection_oracle.py`；日志 `canonical_scaled_strength_consistent.log`，其余报告为同名 JSON。

`65361b8`、dirty=false。普通浮点构造的三档相关间隙与理想相容条件偏差 **−4.930e-32/5.294e-23/5.421e-20 m**；直接套用理想反力的测试在中档失败，单点反力差 **0.002 N**。对实际输入重算 70 位接触解，三档反力最大差 **1.141e-30 N**。保留构造余量后，三档解析反力及 **12** 组面积／载荷差分全部通过，载荷导数最大相对误差 **2.440e-12**、面积导数最大绝对差 **1.756e-32**。入口 `canonical_contact_reference.py`、`canonical_compensated_contact.py`；报告位于 `runs/grid_update/` 下同名 JSON。下一项核对其余重复界面导数和完整塑性轨迹，再整理联合推进接口。

## 2026-10-24：有序补偿矩阵乘法正式接口

RTX 4090 工作站，`6402282`、dirty=true。Numba 编译有序两分量矩阵循环，**30** 组形状／尺度与该提交的 Python 实现逐位一致；同一个 **99** 维平衡算例的求解时间 **28.62454→8.81470 s（3.247×）**，主值、余量、开口及 Newton 迭代记录完全一致。命令 `OPENBLAS_NUM_THREADS=1 PYTHONPATH=.:runs/e5_release_audit:../BladeMaster-Warp-e5-grid-20261023/runs/e5_release_audit .venv/bin/python runs/e5_release_audit/formal_compiled_products.py runs/grid_update/scaled_stage_input_1213893.npz`；报告 `runs/grid_update/formal_compiled_products.json`。

新增 **23** 例覆盖 100 位十进制参考、广播、空维度、大幅相反项及 **2⁻¹⁰⁰⁰…2⁹⁰⁰** 尺度。相关 **78 passed**（2.46 s）；提交前全量 `OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m pytest tests -q`：**1658 passed、8 skipped、18 subtests passed**（857.14 s），日志 `runs/grid_update/compiled_products_full.log`；8 项跳过为 CRESSim 上游构建。
