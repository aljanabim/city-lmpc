import casadi as ca
from controllers.solo import SoloMPC
from models.solo import SoloFrenetModel
from simulators.solo import SoloMPCSimulator
from utils import sim_util

if __name__ == "__main__":
    lane_width = 0.5
    yaw0 = ca.pi
    track = sim_util.create_track(0, 0, yaw0)

    x = ca.vertcat(0, -lane_width / 2, 0, yaw0)
    # +0.5 on s to allow reaching the target
    xub = ca.vertcat(track.length + 0.5, lane_width, 1.5, ca.inf)
    uub = ca.vertcat(1, ca.pi / 5)
    model = SoloFrenetModel(x)
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
    # simulator.run()
    # simulator.save()
    simulator.load()
