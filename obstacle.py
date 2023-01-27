import casadi as ca
import numpy as np
from controllers.obstacle import ObstacleLMPC, ObstacleMPC
from simulators.solo import SoloMPCSimulator, SoloRelaxedLMPCSimulator
from controllers.solo import SoloMPC
from simulators.obstacle import ObstacleLMPCSimulator, ObstacleMPCSimulator
from utils import sim_util, vis_util


if __name__ == "__main__":
    EXP_NAME = "obstacle"
    # use same model track and solo as in the solo scenario
    model, mpc, track, track_J0, xub_lmpc = sim_util.setup_solo(SoloMPC)
    s_obs = track.length / 2
    simulator = ObstacleMPCSimulator(model, mpc, track_J0)
    simulator.EXP_NAME = EXP_NAME
    simulator.S_OBS = s_obs

    traj_0 = simulator.load(iteration=0)
    if traj_0 is None:  # if J0 is not stored, run simulator, save, then load it
        traj_0 = simulator.run().save().load()
    quit()

    # vis_util.compare_iterations(EXP_NAME, 0, 0, track, step=1, s_obs=s_obs)
    # quit()
    load_until = 0
    trajectories = [simulator.load(j) for j in range(0, load_until + 1)]

    # # # # Start LMPC learning
    # x0 = model.x0
    # x0[0, 0] = track.length - 0.75
    # point_idx = np.argmin(np.abs(track.points_array[2, :] - x0[0, 0]))
    # phi0 = track.points[point_idx].phi_s
    # x0[2, 0] = 0.5
    # x0[-1, 0] = phi0  # set heading angle
    # model.set_x0(x0)
    # time_step = np.argmin(np.abs(traj_0["states"][0, :] - x0[0, 0]))

    # TODO mention in report that 0.1 for v_u == penalize acceleration
    # turning is more expensive
    # Kepp u_diff i LMPC cost, because of min time, we just want smoother control
    R = ca.diag((10, 50))
    lmpc = ObstacleLMPC(
        model, Q=None, R=R, xlb=-mpc.xub, xub=mpc.xub, ulb=-mpc.uub, uub=mpc.uub
    )
    # Use track not track_J0 in simulator because it has a shorter end (symmetric)
    simulator = ObstacleLMPCSimulator(model, lmpc, track, trajectories, max_iter=10)
    simulator.EXP_NAME = EXP_NAME
    simulator.S_OBS = s_obs
    # simulator.time_step = time_step
    simulator.run()
