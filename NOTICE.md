# Third-party sources

- [DexLab](https://github.com/huangkiki/Dexlab) `b955d8536e219e7ba3de1150cfc19aa65878cfb3`, Apache-2.0 (`vendor/DexLab-LICENSE`): reference sampling and stencil in `geometry.py`, cut-push trajectory in `run.py`, scene style in `replay_style.py`.
- [MuJoCo Menagerie Franka Panda](https://github.com/google-deepmind/mujoco_menagerie/tree/62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35/franka_emika_panda), Apache-2.0 (`vendor/mujoco_menagerie`).
- [libfranka Cartesian impedance example](https://github.com/frankarobotics/libfranka/blob/e88d08f64cbf842b8f53c0b551963684064038d6/examples/cartesian_impedance_control.cpp), Apache-2.0 (`vendor/libfranka`).
- [SERL++ impedance.py](https://github.com/liusong-0086/serl-plus-plus/blob/a1c541ad77c3d55c3e7b2ddc568773a449987e7f/infra/sim/controllers/impedance.py), Apache-2.0 (`vendor/serl_controller`).
- [three.js r169](https://github.com/mrdoob/three.js/tree/r169), MIT (`blademaster_warp/web/vendor`).
- [CRESSim-MPM](https://github.com/yafei-ou/CRESSim-MPM/tree/09aa5009b8580351f516b6df7660e87821fc5eb6), BSD-3-Clause: `scripts.fig5_compare build-cressim` downloads this pinned source and compiles its original CPU solver with the local headless driver; upstream code remains in the chosen source directory.
