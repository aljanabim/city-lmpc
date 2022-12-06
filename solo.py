from controllers.solo import SoloMPC
from models.example import ExampleModel
from models.solo import SoloFrenetModel
from models.track import Track
from controllers.example import ExampleMPC
import casadi as ca
from utils.sim_util import (
    ExpMeta,
    compute_ref_trajectory,
    create_track,
    get_data_folder,
)
from utils.vis_util import VehicleData, COLORS, animate_trajectory

if __name__ == "__main__":
    lane_width = 0.5

    track = create_track(0, 0, ca.pi / 4)

    x = ca.vertcat(0, lane_width / 2, 0, 0)
    xub = ca.vertcat(track.length, lane_width, 1.5, 2 * ca.pi)
    uub = ca.vertcat(1, ca.pi / 5)
    model = SoloFrenetModel(x)
    mpc = SoloMPC(
        model,
        Q=ca.diag((1, 300, 200, 20)),
        R=ca.diag((5, 1)),
        xlb=-xub,
        xub=xub,
        ulb=-uub,
        uub=uub,
    )
    exp_meta = ExpMeta("solo", 0, 0)
    dest_folder = get_data_folder(exp_meta)

    # while x0 - x_ref > 0.01:
    while x[0, 0] < track.length:
        x_ref, curvature, s_0_arc, phi_0_arc = compute_ref_trajectory(
            x=x, track=track, dt=model.dt, N=mpc.N, v_ref=0.5, e_ref=lane_width / 2
        )
        uopt, xopt = mpc.get_ctrl(x, x_ref, curvature, s_0_arc, phi_0_arc)
        u = ca.vertcat(uopt, curvature[0], s_0_arc[0], phi_0_arc[0])
        # State Feedback Loop
        x = model.sim(1, u=u, input_noise=False, state_noise=False)
        # states, inputs = model.get_trajectory()
        vehicles = [VehicleData("example", COLORS["ego"], x, u)]  # states, inputs),
        # print("s:", x[0, 0])
        animate_trajectory(dest_folder, track, vehicles, "J0")

    # states, inputs = model.get_trajectory()
    # print(states, inputs)
