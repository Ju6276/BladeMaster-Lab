"""GPU-only contact instrumentation for the Fig. 5 BladeMaster runner."""

import numpy as np
import warp as wp

from .solver import Solver
from .fig5 import piece_ids


@wp.kernel
def collect_contact_impulses(
    pieces: wp.array(dtype=int),
    rigid: wp.array(dtype=wp.vec3),
    total: wp.array(dtype=wp.vec3),
    board: wp.array(dtype=wp.vec3),
    pair_sum: wp.array(dtype=wp.vec3),
    board_sum: wp.array(dtype=wp.vec3),
):
    p = wp.tid()
    wp.atomic_add(pair_sum, pieces[p], total[p] - rigid[p])
    if p == 0:
        board_sum[0] = board_sum[0] + board[0]


class Fig5Solver(Solver):
    def __init__(self, config, cuts, device="cuda:0"):
        super().__init__(config, device=device)
        self.fig5_pieces = wp.array(
            piece_ids(self.reference, cuts).astype(np.int32), dtype=int, device=self.device
        )
        self.fig5_rigid = wp.zeros(self.n, dtype=wp.vec3, device=self.device)
        self.fig5_pair_sum = wp.zeros(3, dtype=wp.vec3, device=self.device)
        self.fig5_board_sum = wp.zeros(1, dtype=wp.vec3, device=self.device)

    def particle_contacts(self):
        wp.copy(self.fig5_rigid, self.impulses)
        super().particle_contacts()
        self.launch(
            collect_contact_impulses,
            self.n,
            [
                self.fig5_pieces,
                self.fig5_rigid,
                self.impulses,
                self.board,
                self.fig5_pair_sum,
                self.fig5_board_sum,
            ],
        )

    def reset_contact_measurement(self):
        self.fig5_pair_sum.zero_()
        self.fig5_board_sum.zero_()
