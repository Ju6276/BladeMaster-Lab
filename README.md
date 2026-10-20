# BladeMaster-Lab

[BladeMaster](https://arxiv.org/html/2609.27342v2)（拓扑感知 TLMPM 切割仿真）的 Warp/CUDA 实现，带 MuJoCo Franka 双向耦合、linked-volume 表面重建和浏览器在线显示。

- SpaceMouse 遥操作 Franka 切香蕉（论文 §IV-C 材料，默认 6 mm 材料网格、62.5 μs 子步、规则粒子采样），浏览器支持开始、暂停、结束和重启，并分别显示材料与表面的网格间距。表面保留实际子步刀路。
- 连续三刀 + 推开 + 表面 + 录制：**四块，0.56× 实时**（6 mm，21.039 s 仿真 / 37.756 s 计算，RTX 4060；不含客户端渲染）。遥操作切到底、保持、退刀的固定输入测试为 0.48×。最新验证与计时见 RESULTS。
- 论文 Table I 重力刀（8960 粒子，2 ms）：**0.42–0.44 wall/sim**，切透成两块（论文 RTX 5090 为 0.33）。
- 已实现 Fig. 1、3、4、6、7、8 对应场景。Fig. 5（CRESSim-MPM 独立操作对照）和 Fig. 3(d) 的停刀深度仍在 TODO；低锋利度切不透使用 `cut_hold` 对照开关。逐项见 RESULTS。
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

服务只监听 `127.0.0.1`，端口默认 8765（`--port` 改）。在 Mac 上转发：

```bash
ssh -N -L 18765:127.0.0.1:8765 dong@100.70.97.92
```

打开 <http://127.0.0.1:18765>，点"开始"；可暂停、继续、结束、重新开始。三个视图是整臂、表面和粒子，粒子颜色表示切割侧标签。

### 用 SpaceMouse 遥操作 Franka 切

```bash
# 服务器（SpaceMouse 插在哪台机器上，客户端就在哪台运行；这里两者都在 tams61）
.venv/bin/python -m blademaster_warp.franka_run --live --port 18765 --motion teleop --cut-events auto \
  --controller libfranka --stiffness 1000 --material paper-banana --spacing 0.006 --duration 1800 \
  --surface --surface-embedding cuda --surface-ccd cuda
.venv/bin/python -m blademaster_warp.spacemouse_client --port 18765 --seconds 1800
# Mac：只转发页面
ssh -N -L 18765:127.0.0.1:18765 dong@100.70.97.92
```

SpaceMouse 六轴给刀的线速度和角速度。按键默认 `--buttons session`：左键开始（结束后再按为重新开始），右键结束，两键同按暂停，与页面按钮相同；`--buttons cut` 改为左键 `begin_cut`、右键 `end_cut`，配合 `--cut-events manual` 手动开、关每一刀（6 mm 网格最多 33 刀）。`--cut-events auto`（推荐）不用按键分刀：刀每次离开材料后再进入就是新的一刀；用刀面推开切块时应改用手动模式，否则推的动作也会被当成切。只有切割开启时刀刃经过的材料才会被切开，切口位置和形状完全由刀的实际轨迹决定。

- `--spacing` 是材料网格间距，单位 m，可取 0.003、0.004、0.0045、0.006 等任意有效值。默认 6 mm（粒子间距 3 mm）；3 mm 细网格更慢。默认恢复规则采样和粒子半径接触，减轻随机采样造成的切透弹开；旧配置可用 `--sampling-seed 1 --blade-contact-radius .0005` 重现。
- `--teleop-view front|behind`：操作者站在机械臂对面（默认，往前推即朝向机械臂）或身后。
- `--teleop-rotation yaw|full`：`yaw`（默认）只允许绕竖直轴转，刀始终竖直；`full` 用三个转轴。
- `--surface-precision`：在线显示默认用 float32 的 CUDA 表面嵌入，录制时用 float64。

客户端自己读 SpaceMouse（`blademaster_warp/spacemouse.py`，依赖 `hidapi`），不需要别的仓库。Linux 上自动改用 hidapi 的 hidraw 后端，并需要一条 udev 规则让普通用户读写设备（插拔一次后生效）：

```bash
echo 'KERNEL=="hidraw*", ATTRS{idVendor}=="256f", MODE="0666", TAG+="uaccess"' | sudo tee /etc/udev/rules.d/99-spacemouse.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

不加 `--live` 时使用固定的模拟输入序列（`teleop_input.mock_command`），用于测试和计时。

`--attachment grasp` 让夹爪夹住带柄的自由刀（默认每指 30 N），刀只靠摩擦保持；默认 `fixed` 为刀固定在法兰安装座上，手指合在安装座上。

![遥操作模拟输入示例](docs/media/franka_teleop_mock.png)

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

`--cuts` 指定各刀相对起始位置的 x 坐标（m，按顺序切），`--push-distance` 指定推开距离。加 `--spacing 0.003` 用 3 mm 网格（规则采样 11520 粒子；步长自动取 40 μs），外形和切面更圆润。

Fig. 4 形式的 Franka 中片抬升：两次径向切割，转平刀具，插入片块下方并抬高 40 mm。使用圆柱材料、10 mm 材料网格和 1 ms 子步；成功评价检查片块分离、实际抬升、持续托持及余块位移。

```bash
.venv/bin/python -m blademaster_warp.franka_run --motion slice-lift --live --port 18766 \
  --surface --surface-embedding cuda --surface-ccd cuda
```

去掉 `--live`、加 `--record --output runs/franka_slice_lift` 保存完整任务；用 `render_franka runs/franka_slice_lift --layout paper` 导出粒子/表面对照视频。

常用选项：

| 选项 | 作用 |
|---|---|
| `--observation-interval` | 表面与状态观测间隔，默认 0.04 s；须为机器人步长的整数倍，可用 0.01 s 检查插值误差 |
| `--dt` | 材料步长，必须整除 1 ms；香蕉默认 \(\min(62.5\,\mu s,\ 0.5\,\Delta x/c)\)，6 mm 为 62.5 μs，3 mm 为 40 μs |
| `--contact-scaling per-step` | 使用论文的固定 \(s_n\)（接触刚度随步长变化） |
| `--tool-substep predict` | 耦合区间内用移动性矩阵更新刀具状态 |
| `--pair-search {auto,list,hash,all}` | 接触检测方式；auto 使用带位移余量的邻居表，hash 每子步重建，all 完整搜索 |
| `--controller {serl,libfranka}` | 笛卡尔阻抗控制器 |

`report.json` 的 `time_s` 分列材料、机器人、刀路记录、观测和状态发布耗时；`surface_phase_s` 再分节点嵌入、链接扫掠、网格构建和顶点嵌入。`material_interval_phase_s` 的读回时间包含等待 GPU 完成本区间，`task_evaluation` 给出切割结果。

表面帧的 `tool_path` 保存完整刀路，列定义在 `surface/method.json`。`scripts.force_convergence` 在干净的 Git 提交上运行；复用缓存时核对配置、试验协议和提交，换参数使用新的输出目录。已有历史汇总仍可用 `--plot` 读取。

规则采样的刀位敏感性对照：`python -m scripts.force_convergence --spacings .006 .004 .003 --sampling-seeds 0 --x-offset-fractions 0 .125 .25 .375 .5 --output runs/force_offsets`。偏移值以粒子间距 h/2 为单位；输出包含各刀位的力、冲量和方向切割功。

## 其他入口

| 模块 | 用途 |
|---|---|
| `blademaster_warp.paper_freefall` | Fig. 3 / Table I 重力刀；`--grid-spacing`、`--dt`、`--contact-reference-dt` 扫描分辨率和步长，结果包含刃口高度、方向功与损伤计数 |
| `blademaster_warp.paper_impact` | 论文 Fig. 6 三把自由刀（`Config.tool_count`），默认按论文图从前侧面水平射入（`--launch side`，20 g）：刀被减速停住，初速越大越深；`--launch top` 为竖直射入 |
| `blademaster_warp.scenarios` | 论文 Fig. 4/7/8 的切割历史场景（规定刀路）；`fig4`（`--single-field` 为单场对照）、`fig7-shallow`/`fig7-deep`、`fig8` |
| `blademaster_warp.convergence` | 步长、耦合间隔、分辨率扫描 |
| `blademaster_warp.run`，`blademaster_warp.teleop` | 不带机械臂的导向刀与刀遥操作；刀—砧板接触与 Franka 路径相同，由 `tool.step` 处理 |
| `scripts/probe_substep.py` | 单子步耗时测量 |
| `scripts/check_cut_release.py` | Franka 以 60 mm/s 输入切到底、保持、退刀，比较实际片块平移、反力和动能；如 `python -m scripts.check_cut_release --surface --output runs/release-6mm`；`--seed 1 --blade-contact-radius .0005` 重现旧设置 |
| `scripts/topocut_compare.py` | 同一刀路下 BladeMaster 与 TopoCut 式单场 MLS-MPM（`mlsmpm.py` + `topocut_topology.py`）对比，谱奖励评价；`scripts/render_compare.py` 把结果渲染成三栏对比视频 |
| `scripts/surface_deformed.py` | 已知变形下的表面嵌入误差与自交 |
| `scripts/surface_observations.py` | 保留完整刀路，对比不同材料观测间隔；默认 CUDA，可逐帧检查几何 |
| `scripts/stress_teleop.py` | 六轴输入、暂停、停止、重启和断线重连回放；`--seeds 0 17 913 --near-capacity --restart-cycles 2` 检查多种种子和切割历史容量 |
| `scripts/stress_multi_tool.py` | 同一材料上的多刀接触、旋转交叉、换序、独立损伤、动量及 int64 编号容量检查；`--seeds 0 17 913` 批量运行 |
| `scripts/fig5_compare.py` | Fig. 5 两刀三块、退刀后独立推动；构建固定版本的 CRESSim-MPM 独立对照 |
| `scripts/live_latency.py` | 状态帧确认的命令到 HTTP 帧的延迟，`--output` 保存统计；异常时关闭输入流，`--stop` 在正常或异常退出时结束会话 |

浏览器的 `window.liveDiagnostics` 记录 WebGL 提交、显示循环间隔、收到的状态帧间隔和已绘制的命令序号。模拟输入的延迟测量应使用单独的服务端口，避免与 SpaceMouse 客户端同时发送序号。

## 测试

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests -q
```

测试只依赖本仓库：网格交换、刀距离、摩擦、粒子传输按论文公式写出 FP64 参考，再加动量守恒和刚体平移等解析检查。需要 CUDA。

## 代码结构

- `solver.py`、`material.py`、`contact.py`：材料、网格传输、接触、切割标签。
- `franka.py`、`coupling.py`、`franka_run.py`、`cut_cycle.py`、`franka_task.py`：机器人、外部刚体接口、切割与抬片任务。
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
