import casadi as ca
import numpy as np
from controllers.solo import SoloMPC, SoloRelaxedLMPC
from controllers.bicycle import SoloBicycleMPC
from models.solo import SoloFrenetModel
from simulators.bicycle import SoloMPCSimulator
from utils import sim_util, vis_util

from models.solo import SoloBicycleModel

if __name__ == "__main__":
    lane_width = 0.5
    yaw0 = ca.pi
    track = sim_util.create_track(0, 0, yaw0, J0=True)

    # Get Model
    x = ca.vertcat(0, 0, 0, yaw0)
    model = SoloBicycleModel(x)

    # Input and state constraints for MPC, +0.5 on s to allow reaching the target and the margin +0.5 for J0
    xub = ca.vertcat(ca.inf, ca.inf, 0.5, ca.inf)
    uub = ca.vertcat(1.7, ca.pi / 4)

    # Get trajectory for initial iteration
    mpc = SoloBicycleMPC(
        model,
        Q=ca.diag((20, 20, 0.1, 0.1)),  # (1, 300, 200, 20)
        R=ca.diag((2, 1)),  # (100, 4)
        xlb=-xub,
        xub=xub,
        ulb=-uub,
        uub=uub,
    )
    simulator = SoloMPCSimulator(model, mpc, track, track)
    simulator.run()
