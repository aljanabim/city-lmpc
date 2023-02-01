import casadi as ca
import numpy as np
from controllers.solo import SoloMPC, SoloRelaxedLMPC
from models.solo import SoloFrenetModel
from simulators.solo import SoloMPCSimulator, SoloRelaxedLMPCSimulator
from utils import sim_util, vis_util

from controllers.obstacle import ObstacleLMPC
from simulators.solo import SoloMPCSimulator, SoloRelaxedLMPCSimulator
from controllers.solo import SoloMPC
from simulators.obstacle import ObstacleLMPCSimulator, ObstacleMPCSimulator
from utils import sim_util, vis_util

if __name__ == "__main__":
    """ """
    R = (0, 0)
    L = 200
    N = 7
    load_until = 9

    EXP_NAME = f"obstacle-R{R}L{L}N{N}"
    model, mpc, track_vis, track_ctrl, track_J0, xub_lmpc = sim_util.setup_solo(
        Controller=SoloMPC, Q=ca.diag((1, 300, 200, 20))
    )
    N_POINTS = 217
    track_J0.update_n_points(217)
    s_obs = track_ctrl.length / 2
    simulator = ObstacleMPCSimulator(model, mpc, track_vis, track_J0)
    simulator.EXP_NAME = EXP_NAME
    simulator.S_OBS = s_obs

    traj_0 = simulator.load(iteration=0)
    if traj_0 is None:  # if J0 is not stored, run simulator, save, then load it
        traj_0 = simulator.run().save().load()
    # quit()

    vis_util.compare_iterations(EXP_NAME, 0, 9, track_vis, step=1, s_obs=s_obs)
    quit()
    trajectories = [simulator.load(j) for j in range(0, load_until + 1)]
    for traj in trajectories:
        print(traj["inputs"][:, :-5])
    quit()
    # # # Start LMPC learning
    lmpc = ObstacleLMPC(
        model,
        Q=None,
        R=ca.diag(R),
        xlb=-xub_lmpc,
        xub=xub_lmpc,
        ulb=-mpc.uub,
        uub=mpc.uub,
        N=N,
    )
    lmpc.L = L
    simulator = ObstacleLMPCSimulator(
        model, lmpc, track_vis, track_ctrl, trajectories, max_iter=10
    )
    simulator.EXP_NAME = EXP_NAME
    simulator.S_OBS = s_obs
    # simulator.track_ctrl.update_n_points(155)
    simulator.run()
