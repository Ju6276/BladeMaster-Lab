# Sources and scope

This is an **independent implementation**, not the BladeMaster authors' code.

- Method: [BladeMaster, arXiv:2609.27342v2](https://arxiv.org/html/2609.27342v2), equations 1–16.
- GPU runtime: [NVIDIA Warp](https://github.com/NVIDIA/warp), version 1.18.0.
- Reference implementation: [huangkiki/Dexlab](https://github.com/huangkiki/Dexlab), commit `b955d8536e219e7ba3de1150cfc19aa65878cfb3`.
- `geometry.py` sampling and trilinear stencil, `run.py` cut-push trajectory, baseline parameters, and the guided blade board-stop convention are adapted from DexLab under Apache-2.0. The original license is in `vendor/DexLab-LICENSE`. The GPU kernels are a separate implementation, checked against the CPU reference and paper equations.
- `replay_style.py` adapts the light-wood texture, palette, lighting and caption layout of DexLab's `cutting_showcase.py` under the same Apache-2.0 license. It renders saved simulation vertices and particles; it does not import DexLab's preassigned banana display seams or change physical states.
- Its optional `paper` palette and `render_franka.py` use BladeMaster Fig. 1 as a visual reference (pale background and green board), not author rendering code/assets. `robot_replay.py` draws the pinned Menagerie meshes with recorded joint poses and upstream mesh normals; no physics is stepped while rendering. Recorded geometry is unchanged.
- `presets.py` shares the published banana material values and declared numerical settings across the standalone and Franka drivers. `geometry.py` also shares the DexLab demo's synthetic 80×40×1 mm wedge with a 5 mm bevel height; the former robot-only 4 mm bevel was a local deviation and is removed. Synthetic fixtures are separate from this blade geometry.
- Franka assets: [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35/franka_emika_panda), pinned files/Apache-2.0 license and checksums in `vendor/mujoco_menagerie`. Source assets are unchanged; runtime actuator conversion and experimental knife mounting are in `franka.py`.
- Default controller equation: [libfranka Cartesian impedance example](https://github.com/frankarobotics/libfranka/blob/e88d08f64cbf842b8f53c0b551963684064038d6/examples/cartesian_impedance_control.cpp). The MuJoCo implementation uses full bias compensation; the source example/license/hash are in `vendor/libfranka`.
- Optional controller: [SERL++ impedance.py](https://github.com/liusong-0086/serl-plus-plus/blob/a1c541ad77c3d55c3e7b2ddc568773a449987e7f/infra/sim/controllers/impedance.py), unchanged function and Apache-2.0 license in `vendor/serl_controller`. Default gains and pseudo-inverse branch are preserved; no WowSkin teleoperation constraints are imported.

The banana geometry is synthetic. Material parameters, trajectory, reference solver, measured data, and reconstructed surfaces must not be conflated. No real measured force data or author mesh is bundled. Rendering particles does not implement the paper's linked-volume surface reconstruction.

- Live browser display uses [three.js r169](https://github.com/mrdoob/three.js/tree/r169), under MIT. Pinned module, OrbitControls, license and source metadata are in `blademaster_warp/web/vendor/`. The browser only renders the current physics state; it does not integrate the robot or modify material geometry.
