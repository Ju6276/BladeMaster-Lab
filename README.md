# BladeMaster-Lab

[BladeMaster](https://arxiv.org/html/2609.27342v2)（拓扑感知 TLMPM 切割仿真）的 Warp/CUDA 实现，带 MuJoCo Franka 双向耦合、linked-volume 表面重建和浏览器实时显示。

- 香蕉单刀（论文 §IV-C 材料）+ Franka + 表面：RTX 4060 上 **0.40× 实时**。
- 连续三刀 + 推开 + 表面：**四块，0.80× 实时**（6 mm 网格，无录制）；3 mm 网格（11485 粒子）约 0.31×。
- 论文 Table I 重力刀（8960 粒子，2 ms）：**0.42 wall/sim**，切透成两块（论文 RTX 5090 为 0.33）。
- 接触刚度与步长解耦，25–77 μs 下刀阶段平均力相差 < 2%。

详细数字见 [docs/RESULTS.md](docs/RESULTS.md)，公式对照与数值设置见 [docs/METHOD.md](docs/METHOD.md)，论文各实验参数、DexLab 与文献数据见 [docs/PARAMETERS.md](docs/PARAMETERS.md)，剩余工作见 [TODO.md](TODO.md)。

![连续三刀，3 mm 网格](docs/media/fig1_sequence_3mm.png)

## 安装

Linux + NVIDIA GPU。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## 在线运行

```bash
.venv/bin/python -m blademaster_warp.franka_run --live \
  --motion feedback --controller serl --material paper-banana --duration 12 \
  --surface --surface-embedding cuda --surface-ccd cuda
```

服务只监听 `127.0.0.1:8765`。在 Mac 上转发：

```bash
ssh -N -L 18765:127.0.0.1:8765 dong@100.70.97.92
```

打开 <http://127.0.0.1:18765>，点"开始"；可暂停、继续、结束、重新开始。三个视图是整臂、表面和粒子，粒子颜色表示切割侧标签。

### 用 SpaceMouse 遥操作 Franka 切

```bash
# 服务器（tams61）
.venv/bin/python -m blademaster_warp.franka_run --live --motion teleop --controller libfranka --stiffness 1000 \
  --material paper-banana --duration 120 --surface --surface-embedding cuda --surface-ccd cuda
# Mac：转发端口后运行客户端（读取 SpaceMouse 用的是 WowSkin-Lab 里的 HID 代码）
ssh -N -L 18765:127.0.0.1:8765 dong@100.70.97.92
python -m blademaster_warp.spacemouse_client --port 18765 --seconds 120 --wowskin-root ~/Desktop/WowSkin-Lab
```

SpaceMouse 六轴给刀的世界系线速度和角速度；左键 `begin_cut` 开始新的一刀，右键 `end_cut` 结束，两键同按停住（6 mm 网格最多 33 刀）。加 `--cut-events auto` 则不用按键：刀每次离开材料后再进入就是新的一刀（用刀面推开切块时仍应用按键模式，自动模式会把推的动作也当成切）。只有开始一刀之后刀刃经过的材料才会被切开，切口位置和形状完全由刀的实际轨迹决定。不加 `--live` 时使用固定的模拟输入序列（`teleop_input.mock_command`），用于测试。

`--attachment grasp` 让夹爪夹住带柄的自由刀（默认每指 30 N），刀只靠摩擦保持；默认 `fixed` 为刀固定在法兰安装座上，手指合在安装座上。

![遥操作（模拟输入），两刀未切透](docs/media/franka_teleop_mock.png)

## 记录与离线分析

```bash
# 只写 manifest.json 和 report.json
.venv/bin/python -m blademaster_warp.franka_run --motion feedback --controller serl \
  --material paper-banana --duration 12 --surface --surface-embedding cuda --surface-ccd cuda \
  --output runs/banana_01

# 加 --record 保存逐帧状态和表面；--video 再导出视频（需要 VTK、ffmpeg）
.venv/bin/python -m blademaster_warp.franka_run ... --record --output runs/banana_02
.venv/bin/python -m blademaster_warp.render_franka runs/banana_02
```

论文 Fig. 1 形式的连续切割（3 刀，每刀切到砧板后在切口内把切下的一段推开，再退刀；`paper` 版式左边粒子、右边重建表面）：

```bash
.venv/bin/python -m blademaster_warp.franka_run --motion sequence --controller libfranka --stiffness 1000 \
  --material paper-banana --duration 30 --surface --surface-embedding cuda --surface-ccd cuda \
  --record --output runs/fig1_sequence
.venv/bin/python -m blademaster_warp.render_franka runs/fig1_sequence --layout paper
```

`--cuts` 指定各刀相对起始位置的 x 坐标（m，按顺序切），`--push-distance` 指定推开距离。加 `--spacing 0.003` 用 3 mm 网格（11485 粒子，与论文 Table I 的粒子数同一量级；步长自动取 40 μs），外形和切面明显更圆润，RTX 4060 上约 0.31× 实时。

常用选项：

| 选项 | 作用 |
|---|---|
| `--dt` | 材料步长，必须整除 1 ms；香蕉默认 \(\min(62.5\,\mu s,\ 0.5\,\Delta x/c)\)，6 mm 为 62.5 μs，3 mm 为 40 μs |
| `--contact-scaling per-step` | 使用论文的固定 \(s_n\)（接触刚度随步长变化） |
| `--tool-substep predict` | 耦合区间内用移动性矩阵更新刀具状态 |
| `--pair-search {auto,hash,all}` | 接触检测方式；auto 在 ≤ 2048 粒子时用全对检测 |
| `--controller {serl,libfranka}` | 笛卡尔阻抗控制器 |

`report.json` 里的 `time_s` 给出材料、机器人、观测各阶段耗时，`task_evaluation` 给出切割结果。

## 其他入口

| 模块 | 用途 |
|---|---|
| `blademaster_warp.paper_freefall` | 论文 Fig. 3 / Table I 重力刀 |
| `blademaster_warp.paper_impact` | 论文 Fig. 6 三把自由刀（`Config.tool_count`）；论文式 (5) 下刀未被减速，`--cut-hold` 变体见 RESULTS |
| `blademaster_warp.scenarios` | 论文 Fig. 4/7/8 的切割历史场景（规定刀路）；`fig4`（`--single-field` 为单场对照）、`fig7-shallow`/`fig7-deep`、`fig8` |
| `blademaster_warp.convergence` | 步长、耦合间隔、分辨率扫描 |
| `blademaster_warp.run`，`blademaster_warp.teleop` | 独立导向刀与不带机械臂的刀遥操作（刀—板接触待统一到 Franka 路径） |
| `scripts/probe_substep.py` | 单子步耗时测量 |
| `scripts/topocut_compare.py` | 同一刀路下 BladeMaster 与 TopoCut 式单场 MLS-MPM（`mlsmpm.py` + `topocut_topology.py`）对比，谱奖励评价 |
| `scripts/surface_deformed.py` | 已知变形下的表面嵌入误差与自交 |
| `scripts/live_latency.py` | 在线遥操作路径延迟（对运行中的 `franka_run --live --motion teleop`；只有给 `--start`、`--restart`、`--stop` 才改变运行状态） |

## 测试

```bash
PYTHONPATH=/path/to/Dexlab/src .venv/bin/python -m pytest tests -q
```

部分测试以 [DexLab](https://github.com/huangkiki/Dexlab) `b955d85` 作为数值参考。

## 代码结构

- `solver.py`、`material.py`、`contact.py`：材料、网格传输、接触、切割标签。
- `franka.py`、`coupling.py`、`franka_run.py`、`cut_cycle.py`：机器人、外部刚体接口、运行循环。
- `surface*.py`：表面重建（CPU 参考 + CUDA 部分）。
- `live.py`、`web/`：浏览器实时显示。
- `render*.py`、`replay_style.py`、`robot_replay.py`：离线渲染。
- `vendor/`：固定版本的 Franka 模型和控制器。

检查已录制的表面与粒子近景：

```bash
.venv/bin/python -m scripts.surface_closeup runs/fig1_sequence --times 0 3.6 18
```

输出 `surface_closeup.png`，依赖 VTK 和 Pillow（同离线渲染）。

第三方来源与许可见 [NOTICE.md](NOTICE.md)。贡献规范见 [AGENTS.md](AGENTS.md)。
