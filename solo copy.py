import casadi as ca
from controllers.solo import SoloMPC
from models.solo import SoloFrenetModel
from utils.vis_util import VehicleData, COLORS, animate_trajectory, plot_trajectory
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

    while x[0, 0] < track.length:

        x_ref, curvature, s_0_arc, phi_0_arc = sim_util.compute_ref_trajectory(
            x=x, track=track, dt=model.dt, N=mpc.N, v_ref=0.5, e_ref=-lane_width / 2
        )
        uopt, xopt = mpc.get_ctrl(x, x_ref, curvature, s_0_arc, phi_0_arc)
        u = ca.vertcat(uopt, curvature[0], s_0_arc[0], phi_0_arc[0])
        # State Feedback Step
        x = model.sim(1, u=u, input_noise=False, state_noise=False)
        states, inputs = model.get_trajectory()
        vehicles = [VehicleData("example", COLORS["ego"], states, inputs)]
        plot_trajectory(track, vehicles, plot_input=True)

    # Code for dumping and loading the trajectory
    # exp_meta = sim_util.ExpMeta("solo", 0, 0)
    # sim_util.dump_trajectory(exp_meta, "J0", states, inputs)
    # traj = sim_util.load_trajectory(exp_meta, "J0")
    # animate_trajectory(
    #     exp_meta,
    #     track,
    #     [VehicleData("loaded", COLORS["ego"], traj["states"], traj["inputs"])],
    # )
