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

### 2026-10-22：CPU 检查点存储

`a55b185`，RTX 4090、dirty=false：**4 mm、4892 粒子、G_c=7、50 ms / 3800 步、每段 8 步**，GPU/CPU 检查点的末端 x/v/F/Fp/应力/损伤差、事件时刻差及刀冲量差均为 **0**，两组均释放 **5** 个粒子。刀路梯度最大差 **1.53e-14**，材料梯度最大相对差 **4.42e-15**，全部检查点状态重算差 **0**。

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
