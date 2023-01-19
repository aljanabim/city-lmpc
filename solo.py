import casadi as ca
import numpy as np
from controllers.solo import SoloMPC, SoloRelaxedLMPC
from models.solo import SoloFrenetModel
from simulators.solo import SoloMPCSimulator, SoloLMPCSimulator
from utils import sim_util

if __name__ == "__main__":
    lane_width = 0.5
    yaw0 = ca.pi
    track = sim_util.create_track(0, 0, yaw0)

    # Get Model
    x = ca.vertcat(0, -lane_width / 2, 0, yaw0)
    model = SoloFrenetModel(x)

    # Input and state constraints for MPC, +0.5 on s to allow reaching the target
    xub = ca.vertcat(track.length + 0.5, lane_width, 1.5, ca.inf)
    # uub = ca.vertcat(1, ca.pi / 5)
    uub = ca.vertcat(1, ca.pi / 4)

    # Get trajectory for initial iteration
    mpc = SoloMPC(
        model,
        Q=ca.diag((1, 300, 200, 20)),
        R=ca.diag((5, 4)),
        xlb=-xub,
        xub=xub,
        ulb=-uub,
        uub=uub,
    )
    simulator = SoloMPCSimulator(model, mpc, track)
    simulator.EXP_NAME = "solo"
    traj_0 = simulator.load(iteration=0)
    if traj_0 is None:  # if J0 is not stored, run simulator, save, then load it
        traj_0 = simulator.run().save().load()

    # # Start LMPC learning
    lmpc = SoloRelaxedLMPC(model, Q=None, R=None, xlb=-xub, xub=xub, ulb=-uub, uub=uub)
    simulator = SoloLMPCSimulator(model, lmpc, track, [traj_0], 10)
    simulator.EXP_NAME = "solo"
    simulator.run()
