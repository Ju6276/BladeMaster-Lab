# 交接说明

2026-11-01：旧持续目标已删除；新目标执行方法核对、完整自然单刀验收和刀路优化。先读 [AGENTS.md](../AGENTS.md) 和 [DIFFERENTIABLE.md](DIFFERENTIABLE.md)。

## 机器与访问

| 机器 | 用途 | 地址与仓库 |
|---|---|---|
| Mac | 编辑、提交、推送 | `/Users/judong/Desktop/Papers/Code/BladeMaster-Warp` |
| tams61 / RTX 4060 | SSH 跳板、CRESSim 与旧记录 | `dong@100.70.97.92`；`~/Projects/BladeMaster-Warp` |
| tamsGPU9 / RTX 4090 | 数值实验与 CUDA 测试 | 从 tams61 连接 `dong@tamsgpu9`；`~/Projects/BladeMaster-Warp` |

```bash
ssh -o BatchMode=yes dong@100.70.97.92 '<命令>'
ssh -t dong@100.70.97.92 'ssh -t dong@tamsgpu9'
```

第二跳使用已授权的认证方式；SSH 复用连接过期后重新认证。密码不写入仓库、命令日志或文档。

4090 使用项目 `.venv/bin/python`；依赖版本以 `requirements.txt` 为准。开跑前检查 `nvidia-smi` 和现有任务。长批次保存命令、提交、日志与退出码；复查进程和退出码后更新结果。运行中保持该源码快照不变。

当前已提交 `6591d00` 求解器全量测试：1686 passed、8 skipped、18 subtests passed，793.59 s，日志在 `BladeMaster-Warp-linear-20261101/runs/entered_root_full_20261101/`。文档整理基线的 1669 项结果保留在主仓库 `runs/docs_cleanup_20261101/`。

GitHub：`git@github.com:Ju6276/BladeMaster-Lab.git`，分支 `sync-20261020`。Mac 推送后，干净的远端工作树再拉取；候选工作树先核对本地差异。`runs/` 不进 Git。

## 源码与结果位置

2026-11-01 整理：主工作区在 `34304b0` 求解器基础上归并同顺序补偿求和、线性代数编译及几何进入侧求根，最新提交 `6591d00`。本地待验收改动位于 `/Users/judong/Desktop/Papers/Code/BladeMaster-Warp-e5-review-20261101`，分支 `review/e5-pending-20261101`，包含 20 项已跟踪源码/测试变更和 13 个未跟踪文件；迁移前后逐文件内容及 Git 差异一致。

下表 4090 候选也基于 `34304b0`，各自有不同的未提交源码。原 Mac `/tmp/BladeMaster-Warp-e5-*` 工作树已不存在，4090 副本仍在。

4090 根目录：`/home/dong/Projects/`。

当前归并与运行快照如下；运行中的源码保持不变。

| 位置（4090） | 内容 | 结果目录 |
|---|---|---|
| `BladeMaster-Warp-linear-20261101` | `6591d00` 加两分量数组运算编译；等价性检查通过，全量 1686 passed / 775.24 s | `runs/compiled_expansions_full_20261101/` |
| `BladeMaster-Warp-core-compiled-20261101` | `6591d00` 加核心归并及数组编译；原四项反力/强度失败已通过，全量复验中 | `runs/core_compiled_20261101/` |
| `BladeMaster-Warp-goal-20261101` | 接触归并快照；112 项定向检查通过，全量复验中 | `runs/contact_integration_20261101/` |
| `BladeMaster-Warp-performance-20261101` | 接触归并加零项消除、线性代数及刀具几何编译 | `runs/formal_performance_20261101/` |
| goal 目录 `runs/natural_array_source/` | 自然事件、在线参考切面及数组编译；从第 7 步检查点续跑切割 | `runs/natural_array_events_20261101/` |

Mac 核心归并工作树为 `BladeMaster-Warp-core-20261101`；接触与自然入口稿在其 `runs/events_merge/`、`runs/natural_source/`。核心旧运行快照 `BladeMaster-Warp-core-fixed-20261101` 已由编译快照接替，保留其结果目录。方法小例与性能分析在 goal 工作树 `runs/method_audit_20261101/`；数组编译对照在 `runs/natural_array_compile_20261101/`。

| 目录 | 内容 | 现有全量日志状态 |
|---|---|---|
| `BladeMaster-Warp-e5-state-20261023` | 主树对应的补偿状态、闭合 KKT 等 | `closed_kkt_full.log`：1713 passed、26 failed、8 skipped |
| `BladeMaster-Warp-e5-blade-20261024` | 有限刀、刀/板及粒子对联合接触 | `blade_pair_full.log`：1769 passed、26 failed、8 skipped |
| `BladeMaster-Warp-e5-impact-20261024` | 仿射闭合碰撞与接触归属 | `impact_active_full.log`：异常结束 |
| `BladeMaster-Warp-e5-adhesion-20261024` | 内聚/法向接触分离与空间配对 | `separated_spatial_full.log`：异常结束 |
| `BladeMaster-Warp-e5-events-20261024` | 接触事件分段与闭合回归修复候选 | 36/117/19 项定向修复通过；完整事件批次异常结束，进入事件单例复跑通过 |

日志相对各工作树的位置均为 `runs/grid_update/`。其他 `e5-*` 工作树按 Git 差异检查其内容；源码归并后再删除被替代目录。

闭合回归原因是解析反向入口被迁移流程关闭、零余量表示阻断入口，以及 `(3N,0)` 映射的 reshape 错误。本地 review 工作树包含定向修复。impact 原失败节点已定位到 Python 定时诊断线程段错误，关闭定时输出后通过；运行设置见 RUNNING。统一版本的全量和事件批次复验待完成。

上表 `runs/grid_update/` 历史批次已结束；当前归并批次以进程、退出码和日志末尾为准。

## 文档分工

| 文件 | 内容 |
|---|---|
| `README.md` / `docs/RUNNING.md` | 安装、运行、脚本索引 |
| `docs/DIFFERENTIABLE.md` | 当前状态、持续目标、验收条件 |
| `TODO.md` | 主线待办和保留的正向问题 |
| `docs/METHOD.md` | 公式、方法来源、本地差异 |
| `docs/PARAMETERS.md` | 参数来源与单位 |
| `docs/RESULTS.md` | 日期、版本、命令、关键结果 |
| `NOTICE.md` | 第三方来源与许可 |

原标定/投稿草案从当前文档删除，历史可从 Git 查阅。已撤回的显示网格修补见 tag `archive/before-cleanup-20261020`。任务视频仍在 Mac 的 `~/Desktop/BladeMaster_videos/`；实验报告保留在 GPU 工作树的 `runs/`。
