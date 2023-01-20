import casadi as ca
import numpy as np
from controllers.solo import SoloMPC, SoloRelaxedLMPC
from models.solo import SoloFrenetModel
from simulators.solo import SoloMPCSimulator, SoloRelaxedLMPCSimulator
from utils import sim_util, vis_util


if __name__ == "__main__":
    """
    Totally works only weird thing is that the J1-11 vehicles don't go all the
    way to xf and stop a little bit short, must be something related to relaxation
    of constraints, but I'm not sure exactly what it is.

    Tried to let the cost-to-go also include the full values with lambda, but got similar
    results for J1 and then a bunch of bugs for further iterations.
    """
    EXP_NAME = "solo"
    lane_width = 0.5
    yaw0 = ca.pi
    track = sim_util.create_track(0, 0, yaw0)

    # Get Model
    x = ca.vertcat(0, -lane_width / 2, 0, yaw0)
    model = SoloFrenetModel(x)

    # Input and state constraints for MPC, +0.5 on s to allow reaching the target
    xub = ca.vertcat(track.length + 0.5, lane_width - model.WB / 2, 1.5, ca.inf)
    uub = ca.vertcat(0.75, ca.pi / 4)

    # Get trajectory for initial iteration
    mpc = SoloMPC(
        model,
        Q=ca.diag((1, 300, 200, 20)),
        R=ca.diag((100, 4)),
        xlb=-xub,
        xub=xub,
        ulb=-uub,
        uub=uub,
    )
    simulator = SoloMPCSimulator(model, mpc, track)
    simulator.EXP_NAME = EXP_NAME
    traj_0 = simulator.load(iteration=0)
    if traj_0 is None:  # if J0 is not stored, run simulator, save, then load it
        traj_0 = simulator.run().save().load()

    print(traj_0["states"].shape)  # => (4, 317) => T0 = 316
    # track.length => 10.587873909802939
    # traj_0["states"][0, -1] => 10.59435249

    vis_util.compare_iterations(EXP_NAME, 0, 11, track, step=1)
    load_until = 11
    trajectories = [simulator.load(j) for j in range(0, load_until + 1)]

    # # # Start LMPC learning
    lmpc = SoloRelaxedLMPC(model, Q=None, R=None, xlb=-xub, xub=xub, ulb=-uub, uub=uub)
    simulator = SoloRelaxedLMPCSimulator(model, lmpc, track, trajectories, max_iter=15)
    simulator.EXP_NAME = EXP_NAME
    # simulator.run()
