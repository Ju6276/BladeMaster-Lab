# 给代码代理（Codex / Cursor 等）的工作规范

本仓库是 BladeMaster（arXiv:2609.27342）的 Warp/CUDA 复现。以下规则适用于所有代码、文档和实验记录。

## 版本与实验记录

- 所有改动都用 git 提交，小步提交，英文 commit message。不要再用"手工记录每个文件的 sha256"来代替版本历史。
- 运行记录（`runs/`）不进仓库。每个 manifest 只需记录 `provenance.git_revision()`（commit + dirty），配置和命令行参数。
- 有意义的实验结论写进 `docs/RESULTS.md`：日期、commit、命令、关键数字。一条实验几行，不写过程叙述。

## 文档写法

- 文档只回答：怎么运行、实现了什么、结果数字是多少、下一步做什么。
- **不要写免责声明。** 禁止"不能宣称…""不是作者…""不代表…""仅为诊断…""未经验证…"之类的句子，也不要在每段末尾重复限制条件。与论文的差异只在 `docs/METHOD.md` 的差异表里写一次，用事实陈述（"论文用 X，这里用 Y，原因 Z"）。
- 结论先行，用数字说话。失败就直接写失败和原因，不要绕开。
- 不要写过程日志（"先试了 A，后来发现 B，又更正…"）。过程在 git 历史里。
- README 保持在一屏到两屏以内。

## 代码写法

- docstring 和注释只写代码本身表达不了的约束（单位、坐标系、稳定性条件）。不写 "NOT a paper parameter""not author code""no silent fallback" 这类声明。
- 不要在 manifest/report 里塞 `limitations`、`acceptance`、`author_asset: false` 之类的说明性字段。
- 正常的可读性优先：一行一个语义，不要把十几个参数和语句压成一行。

## 不保留过程产物

- 一次性诊断脚本、预览图、视频、临时 benchmark 结果不进仓库。确实需要复用的脚本放 `scripts/`，并在 README 里写清楚用途。
- 被替代的入口和代码直接删除，不要以"历史对照"名义保留。需要时从 git 历史找。

## 性能工作

- 先测再改：用 `scripts/probe_substep.py` 或 `report.json` 的分阶段时间定位瓶颈，再动手。
- 和论文比性能时用同一场景、同一步长、同一粒子数（论文 Table I），不要拿不同场景的数字互相比较。
- 改数值方法（步长、接触参数）和改调度/存储要分开提交，并分别给出对照数字。

## 测试

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests -q
```

提交前必须全部通过。
