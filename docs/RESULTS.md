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
python -m scripts.force_seed_scan ...
python -m scripts.timestep_reference ...
python -m scripts.force_convergence ...
```

所有组都切成两块，GPU 错误为 0。

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

## 计时

Table I 场景（RTX 4060，只计材料推进，不含表面）。论文数字来自 RTX 5090，不是同卡对比。

| 场景 | 粒子 | 本仓库 | 论文 |
|---|---|---|---|
| Fig. 3 重力刀 | 8960 | 0.42 | 0.33 |
| Fig. 4 片抬升 | 8932 | 0.174 | 0.36 |
| Fig. 7 划痕 | 3584 | 0.158 | 0.30 |
| Fig. 8 薄切 | 7854 | 0.232 | 0.56 |
| Fig. 6 多刀 | 7854 | 0.55 | 0.62 |

单位与论文 Table I 相同。

Fig. 1 三刀，含 Franka、表面和录制（2026-10-20，提交 `8fad2c5`，RTX 4060）：

- 仿真 21.04 s，墙钟 35.54 s，0.59×。
- 材料 24.8 s，表面 5.2 s。

材料步本身已长于仿真时间，单靠加快表面到不了 1×。4090 上每 1 ms 仿真的 16 个子步要约 1.2 ms。

## 测试

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests -q
```

测试只依赖本仓库，覆盖：

- 网格交换（论文式 8）、刀距离、摩擦（论文式 12–13）、粒子传输的 FP64 参考；
- 动量守恒、刚体平移、切割标签和接触的不变量。
