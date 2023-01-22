import casadi as ca
import numpy as np
from controllers.obstacle import ObstacleLMPC
from simulators.solo import SoloMPCSimulator, SoloRelaxedLMPCSimulator
from simulators.obstacle import ObstacleLMPCSimulator, ObstacleMPCSimulator
from utils import sim_util, vis_util
import solo


if __name__ == "__main__":
    """
    Totally works only weird thing is that the J1-11 vehicles don't go all the
    way to xf and stop a little bit short, must be something related to relaxation
    of constraints, but I'm not sure exactly what it is.

    Tried to let the cost-to-go also include the full values with lambda, but got similar
    results for J1 and then a bunch of bugs for further iterations.
    """
    EXP_NAME = "obstacle"
    # use same model track and solo as in the solo scenario
    model, mpc, track = sim_util.setup_solo()
    s_obs = track.length / 2
    simulator = ObstacleMPCSimulator(model, mpc, track)
    simulator.EXP_NAME = EXP_NAME
    simulator.S_OBS = s_obs

    traj_0 = simulator.load(iteration=0)
    if traj_0 is None:  # if J0 is not stored, run simulator, save, then load it
        traj_0 = simulator.run().save().load()

    # print(traj_0["states"].shape)  # => (4, 317) => T0 = 316
    # # track.length => 10.587873909802939
    # # traj_0["states"][0, -1] => 10.59435249

    # vis_util.compare_iterations(EXP_NAME, 0, 11, track, step=1)
    # load_until = 11
    # trajectories = [simulator.load(j) for j in range(0, load_until + 1)]

    # # # # Start LMPC learning
    # lmpc = SoloRelaxedLMPC(model, Q=None, R=None, xlb=-mpc.xub, xub=mpc.xub, ulb=-mpc.uub, uub=mpc.uub)
    # simulator = SoloRelaxedLMPCSimulator(model, lmpc, track, trajectories, max_iter=15)
    # simulator.EXP_NAME = EXP_NAME
    # simulator.S_OBS = s_obs
    # simulator.run()
