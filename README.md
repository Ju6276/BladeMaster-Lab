# BladeMaster on Warp / CUDA

基于 BladeMaster 公开论文的独立实现：Warp/CUDA 材料求解 + MuJoCo Franka 双向动力学耦合。**不是作者源码，尚未完成论文复现验收，也尚未实时。**

当前可运行 Franka 香蕉单刀、撤刀和分块检查；可直接在浏览器看当前仿真，不依赖离屏渲染或回放视频。RTX 4060 上最新两次完整计算为约 **0.172–0.180× 实时**（5.36 s 仿真 / 31.18、29.79 s 实际计算，含表面，未连接浏览器），在Mac浏览器经SSH显示的完整运行约 **0.166×**（5.362 s / 32.299 s，排除人工暂停），开始/暂停/继续和结束状态已实测。不要把视频帧率当作物理速度。

- [TODO：7 类未完成工作及验收条件](TODO.md)
- [论文依据、数值选择、差异与验证记录](REPRODUCTION_ZH.md)
- [第三方来源与许可](NOTICE.md)
- [DexLab 历史审计](AUDIT_ZH.md)：只用于来源及数值故障对照，不是当前运行后端。

## 在线运行（推荐）

在 TAMS 上运行；现有环境为 `/home/dong/Projects/BladeMaster-Warp/.venv`，不需要 Isaac Sim：

```bash
ssh dong@100.70.97.92
cd /home/dong/Projects/BladeMaster-Warp
.venv/bin/python -m blademaster_warp.franka_run \
  --live --motion feedback --controller serl --material paper-banana \
  --duration 12 --surface --surface-embedding cuda --surface-ccd cuda
```

另开一个 Mac 终端建立转发，保持终端运行：

```bash
ssh -N -L 18765:127.0.0.1:8765 dong@100.70.97.92
```

打开 <http://127.0.0.1:18765>。初始化完成后点击“开始”；可以暂停、继续、结束，运行结束后点击“重新开始”。三个视图为整臂、表面、材料粒子，均可拖动旋转、缩放。粒子颜色代表切割历史，不等于独立块数。

这是当前物理状态的 WebGL 显示，不从视频或历史轨迹读数据。网络慢时只显示最新状态，不删物理子步。页面显示仿真时间、实际计算时间和实时倍率；人工暂停时间不计入计算耗时。浏览器渲染和网络延迟未计入服务器倍率。**当前仍是单刀反馈演示；新 Franka 入口尚未接入 SpaceMouse 任意刀路遥操作。**

默认不创建输出目录，也不保存图像、视频或逐帧表面快照。`--port` 可以改变远端监听端口；服务只监听 `127.0.0.1`。

## 记录与检查

只保存配置与最终统计：

```bash
.venv/bin/python -m blademaster_warp.franka_run \
  --motion feedback --controller serl --material paper-banana --duration 12 \
  --surface --surface-embedding cuda --surface-ccd cuda \
  --output runs/banana_summary_01
```

只产生 `manifest.json` 和 `report.json`。输出目录必须是新的，不覆盖历史记录。

只有需要逐帧分析、数值对照或离线视频时，显式启用 `--record`：

```bash
.venv/bin/python -m blademaster_warp.franka_run \
  --motion feedback --controller serl --material paper-banana --duration 12 \
  --surface --surface-embedding cuda --surface-ccd cuda \
  --record --output runs/banana_record_01

.venv/bin/python -m blademaster_warp.check_surface runs/banana_record_01
.venv/bin/python -m blademaster_warp.check_material_geometry runs/banana_record_01
```

记录包含机器人状态/目标/力矩、材料 wrench、粒子位置/速度/F/Fp/切割历史、表面、阶段和逐列定义。`--record` 与 `--live` 可以一起用；指定了输出目录的在线会话只录一段，再录请换新目录启动。默认无记录的在线会话可直接重新运行。

需要导出视频时再执行（可选，需要 VTK/ffmpeg；与在线显示无关）：

```bash
.venv/bin/python -m blademaster_warp.render_franka runs/banana_record_01
```

`report.json` 中 `status=completed` 只表示循环正常结束；另看 `motion_status` 和 `task_evaluation.success`。单刀诊断要求实际运动完成、最终保持阶段始终为两个机械连通部分、刀已脱离材料。它是本项目公开的诊断条件，不是论文官方评分。

## 数值设置与性能对照

Franka 与独立刀具共用 120 mm 合成香蕉、80×40×1 mm 楔形刀以及材料/接触核心。几何来自 DexLab 演示，不是论文原资产。`paper-banana` 指论文材料值：E=400 kPa、ν=0.45、ρ=1100 kg/m³、屈服30 kPa、Gc=700、锋利度0.9；25 μs 步长和其他离散/接触设置的来源见复现说明。

- `--grid-storage compact`（Franka 默认）：将活动网格状态映射到连续索引，固定参考模板及切割码未变时复用映射。设备端活动计数控制计算范围；预留容量不限制被处理的接触或节点数。保留哈希作为查找表，未声称完全消除哈希。
- `--grid-storage hash`：原始存储路径，用于同物理条件对照。它与 compact 共用材料公式。
- 表面节点场和有限边 CCD 可选 CUDA；局部 QEF、三角形构建和最终插值尚在 CPU。
- `--motion feedback` 根据实际刀位切换下刀/撤刀阶段。20 mm/s 下刀、25 mm/s 撤刀、0.5 mm 到位容差、0.1/0.5 s 保持均是明确的本地演示设置，不是作者控制器。不会按期望切块数修改运动或材料。
- `--controller serl` 使用固定的上游 SERL 控制函数；`libfranka` 为另一官方示例控制公式适配。刀固定安装；`--attachment grasp` 是单独的摩擦持刀诊断，未完成接触约束下的反力验收。

本轮优化只调整存储、缓存和 GPU 调度，没有放大时间步、软化材料、删除接触或预设断裂。当前反力仍未收敛，不能以“切成两块”证明物理准确性。

复跑数值对照：

```bash
.venv/bin/python -m blademaster_warp.convergence --output runs/convergence_01
.venv/bin/python -m blademaster_warp.benchmark_surface runs/banana_record_01 \
  --cuda-ccd --output runs/surface_comparison_01.json
```

## 安装、测试、代码结构

Linux + NVIDIA CUDA GPU；现有 `.venv` 中使用 Warp 1.18.0、MuJoCo 3.3.7。新机器安装：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

数值 oracle 测试需要单独的 DexLab 固定版本 b955d85，仅测试使用，不是运行时依赖：

```bash
PYTHONPATH=/home/dong/Projects/Dexlab/src .venv/bin/python -m pytest tests -q
```

代码组织：

- `blademaster_warp/solver.py`、`material.py`、`contact.py`：材料、相容网格传输、接触和切割历史。
- `franka.py`、`coupling.py`、`franka_run.py`：机器人、冲量交换、共用运行循环。
- `surface*.py`：表面重建及 CUDA 后端；不反过来改变材料物理。
- `live.py`、`web/`：实时状态传输与浏览器显示，渲染与物理分开。
- `render*.py`：显式调用的离线导出；在线模式不导入 VTK。
- `scripts/`：独立的历史诊断与性能分析，不进入运行路径。
- `tests/`：方程、动量、切割历史、CPU/CUDA 等价性、机器人、记录与在线协议测试。
- `vendor/`：固定版本机器人/控制器来源；`web/vendor/` 为带许可的 three.js r169。
- `runs/`：用户明确选择的记录目录，不放源代码；一次性预览无需长期保留。

旧 `run` 导向刀的高度止挡、旧独立 `teleop` 的本地刚体积分、Franka 的 MuJoCo 碰撞尚未统一；不能把材料核心共用说成全部动力学相同。旧入口仍保留作已声明的诊断，新的演示入口使用本页命令。详细剩余工作见 [TODO](TODO.md)。
