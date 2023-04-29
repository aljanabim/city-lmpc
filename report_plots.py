import casadi as ca
from utils.sim_util import create_track
from utils.vis_util import draw_arrow
from utils import sim_util, vis_util
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from models.solo import SoloFrenetModel

# Load scenario code
from simulators.solo import SoloMPCSimulator, SoloRelaxedLMPCSimulator
from controllers.solo import SoloMPC, SoloRelaxedLMPC

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"
matplotlib.pyplot.title(r"ABC123 vs $\mathrm{ABC123}^{123}$")

BLUE = "#00A2FF"
RED = "#FF644E"
GREEN = "#3BC449"
YELLOW = "#F9B900"
GRAY = "#585B56"


track = create_track(0, 0, np.pi)

# PLOT SHOWING THE DIFFERENCE BETWEEN CARTESIAN AND FRENET FRAMES
def cart_vs_frenet():
    plt.figure()
    lane_r = create_track(0, 0, np.pi)
    lane_r.e_shift = -0.5
    lane_r.update_track()

    lane_l = create_track(0, 0, np.pi)
    lane_l.e_shift = 0.5
    lane_l.update_track()

    plt.subplot(1, 2, 1)
    plt.title("Cartesian Frame")
    plt.plot(track.cartesian[0], track.cartesian[1], color=GRAY, alpha=0.5)
    plt.plot(lane_r.cartesian[0], lane_r.cartesian[1], color=BLUE, zorder=0)
    plt.plot(lane_l.cartesian[0], lane_l.cartesian[1], color=RED, zorder=0)
    # start point
    plt.plot(track.cartesian[0][0], track.cartesian[1][0], "*", color=GREEN)
    plt.plot(track.cartesian[0][-1], track.cartesian[1][-1], "P", color=YELLOW)
    # end point
    plt.plot()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")

    c_idx = np.argmin(np.abs(track.points_array[4] - 3 * np.pi / 4))
    c_point = track.points[c_idx]
    l = 1.25
    phi_s = c_point.phi_s
    draw_arrow(
        plt,
        (c_point.x_s, c_point.y_s),
        (
            c_point.x_s + l * np.cos(phi_s),
            c_point.y_s + l * np.sin(phi_s),
        ),
    )
    plt.annotate(
        "s",
        xy=(c_point.x_s + (l + 0.5) * np.cos(phi_s), c_point.y_s + l * np.sin(phi_s)),
    )
    draw_arrow(
        plt,
        (c_point.x_s, c_point.y_s),
        (
            c_point.x_s + l * np.cos(phi_s),
            c_point.y_s - l * np.sin(phi_s),
        ),
    )
    plt.annotate(
        "e",
        xy=(c_point.x_s + (l + 0.5) * np.cos(phi_s), c_point.y_s - l * np.sin(phi_s)),
    )

    plt.subplot(1, 2, 2)
    plt.title("Frenet Frame")
    plt.plot(track.points_array[2], track.points_array[2] * 0, color=GRAY, alpha=0.5)
    plt.plot(track.points_array[2], track.points_array[2] * 0 + 0.5, color=RED)
    plt.plot(track.points_array[2], track.points_array[2] * 0 - 0.5, color=BLUE)
    plt.plot(0, 0, "*", color=GREEN, label="Start")
    plt.plot(track.length, 0, "P", color=YELLOW, label="Finish")
    plt.xlabel("s")
    plt.ylabel("e")
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    # plt.show()
    plt.savefig("plots/cart_vs_frenet.pdf")
    plt.close()


# PLOT ALL THREE SCENARIOS SIDE-BY-SIDE
titlesize = 30


def scenarios():
    # PLOT SCENARIO SIDE BY SIDE
    set_dpi()
    plt.subplot(1, 3, 1)

    vehicles = [
        vis_util.VehicleData(
            "Ego",
            vis_util.COLORS["ego"],
            ego_solo_trajectories[0]["states"][:, :1],
            ego_solo_trajectories[0]["inputs"][:, :1],
        ),
    ]
    vis_util.plot_trajectory(
        track, vehicles, plot_axis=False, show=False, plot_legend=False
    )
    plt.title("Single Vehicle", fontsize=titlesize)

    plt.subplot(1, 3, 2)
    vehicles = [
        vis_util.VehicleData(
            "Ego",
            vis_util.COLORS["ego"],
            ego_solo_trajectories[0]["states"][:, :1],
            ego_solo_trajectories[0]["inputs"][:, :1],
        ),
        obs_vehicle,
    ]
    vis_util.plot_trajectory(
        track, vehicles, plot_axis=False, show=False, plot_legend=False
    )
    plt.title("Stationary Obstacle", fontsize=titlesize)

    plt.subplot(1, 3, 3)
    vehicles = [
        vis_util.VehicleData(
            "Ego",
            vis_util.COLORS["ego"],
            ego_solo_trajectories[0]["states"][:, :1],
            ego_solo_trajectories[0]["inputs"][:, :1],
        ),
        obs_vehicle,
        vis_util.VehicleData(
            "Onc",
            vis_util.COLORS["onc"],
            onc_trajectory["states"][:, :1],
            onc_trajectory["inputs"][:, :1],
        ),
    ]
    vis_util.plot_trajectory(
        track, vehicles, plot_axis=False, show=False, plot_legend=False
    )
    plt.title("Oncoming Traffic", fontsize=titlesize)
    plt.legend(fontsize=16, loc="upper right")
    plt.tight_layout()
    plt.savefig("plots/scenarios.pdf")
    plt.close()


def trajectory_results(trajectories, plot_name, vehicles=[], skip_even=False):
    plt.figure()
    alphas = np.linspace(20, 80, len(trajectories), dtype=int)
    for i, (solo_traj, alpha) in enumerate(zip(trajectories, alphas)):
        if skip_even and i % 2 == 0 and i != 0:
            continue
        vehicles.append(
            vis_util.VehicleData(
                f"Iteration {i}",
                "#000000" + str(alpha)
                if i < len(trajectories) - 1
                else vis_util.COLORS["ego"],
                solo_traj["states"],
                solo_traj["inputs"],
            )
        )
    vis_util.plot_trajectory(
        track, vehicles, plot_axis=False, show=False, plot_legend=False
    )
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"plots/{plot_name}_iterations.pdf")
    plt.close()

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(trajectories[0]["inputs"][0, :], label="Iteration 0")
    plt.plot(
        trajectories[-1]["inputs"][0, :],
        label=f"Iteration {len(trajectories)-1}",
    )
    plt.ylabel(r"Input velocity, $v_u$ [m/s]")
    # plt.xlabel(r"Time step, t")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(trajectories[0]["inputs"][1, :], label="Iteration 0")
    plt.plot(
        trajectories[-1]["inputs"][1, :],
        label=f"Iteration {len(trajectories)-1}",
    )
    plt.ylabel(r"Steering input, $\delta$ [rad]")
    plt.xlabel(r"Time step, t")
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"plots/{plot_name}_inputs.pdf")
    plt.close()


def set_dpi():
    my_dpi = 100
    plt.figure(figsize=(1320 / my_dpi, 500 / my_dpi), dpi=my_dpi)


def onc_results():
    print([traj["states"].shape[1] for traj in ego_onc_sneak_trajectories], "sneak")
    print([traj["states"].shape[1] for traj in ego_onc_wait_trajectories], "wait")
    set_dpi()

    time_idxs = [1, 240, 300, 386]
    fontsize_step = 20
    plt.suptitle("Ego waits for Onc to pass", fontsize=titlesize)
    for i, time in enumerate(time_idxs):
        plt.subplot(1, 4, i + 1)
        vehicles = [
            obs_vehicle,
            vis_util.VehicleData(
                "Ego",
                vis_util.COLORS["ego"],
                ego_onc_wait_trajectories[0]["states"][:, :time],
                ego_onc_wait_trajectories[0]["inputs"][:, :time],
            ),
            vis_util.VehicleData(
                "Onc",
                vis_util.COLORS["onc"],
                onc_trajectory["states"][:, :time],
                onc_trajectory["inputs"][:, :time],
            ),
        ]
        vis_util.plot_trajectory(
            track, vehicles, plot_axis=False, show=False, plot_legend=False
        )
        plt.title(f"Step {time}", fontsize=fontsize_step)
    plt.legend(fontsize=16, loc="upper right")
    plt.tight_layout()
    plt.savefig("plots/onc_j0_wait.pdf")
    plt.close()

    set_dpi()
    time_idxs = [1, 80, 140, 240]
    plt.suptitle("Ego moves before Onc passes", fontsize=titlesize)
    for i, time in enumerate(time_idxs):
        plt.subplot(1, 4, i + 1)
        vehicles = [
            obs_vehicle,
            vis_util.VehicleData(
                "Ego",
                vis_util.COLORS["ego"],
                ego_onc_sneak_trajectories[0]["states"][:, :time],
                ego_onc_sneak_trajectories[0]["inputs"][:, :time],
            ),
            vis_util.VehicleData(
                "Onc",
                vis_util.COLORS["onc"],
                onc_trajectory["states"][:, :time],
                onc_trajectory["inputs"][:, :time],
            ),
        ]
        vis_util.plot_trajectory(
            track, vehicles, plot_axis=False, show=False, plot_legend=False
        )
        plt.title(f"Step {time}", fontsize=fontsize_step)
    # plt.legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    plt.savefig("plots/onc_j0_sneak.pdf")
    plt.close()

    set_dpi()
    time_idxs = [1, 80, 140, 145]
    plt.suptitle("Time-optimal trajectory", fontsize=titlesize)
    for i, time in enumerate(time_idxs):
        plt.subplot(1, 4, i + 1)
        vehicles = [
            obs_vehicle,
            vis_util.VehicleData(
                "Ego",
                vis_util.COLORS["ego"],
                ego_onc_sneak_trajectories[3]["states"][:, :time],
                ego_onc_sneak_trajectories[3]["inputs"][:, :time],
            ),
            vis_util.VehicleData(
                "Onc",
                vis_util.COLORS["onc"],
                onc_trajectory["states"][:, :time],
                onc_trajectory["inputs"][:, :time],
            ),
        ]
        vis_util.plot_trajectory(
            track, vehicles, plot_axis=False, show=False, plot_legend=False
        )
        plt.title(f"Step {time}", fontsize=fontsize_step)
    # plt.legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    plt.savefig("plots/onc_jopt.pdf")
    plt.close()

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(ego_onc_wait_trajectories[0]["inputs"][0, :], label="Iteration 0 (wait)")
    plt.plot(
        ego_onc_sneak_trajectories[0]["inputs"][0, :], label="Iteration 0 (no wait)"
    )
    plt.plot(ego_onc_sneak_trajectories[3]["inputs"][0, :], label="Iteration 4")
    plt.ylabel(r"Input velocity, $v_u$ [m/s]")
    # plt.xlabel(r"Time step, t")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(ego_onc_wait_trajectories[0]["inputs"][1, :], label="Iteration 0 (wait)")
    plt.plot(
        ego_onc_sneak_trajectories[0]["inputs"][1, :], label="Iteration 0 (no wait)"
    )
    plt.plot(ego_onc_sneak_trajectories[3]["inputs"][1, :], label="Iteration 4")

    plt.ylabel(r"Steering input, $\delta$ [rad]")
    plt.xlabel(r"Time step, t")
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"plots/onc_inputs.pdf")
    plt.close()


def onc_safety_condition():
    x0 = ca.vertcat(1, -0.5 / 2, 0, ca.pi)
    model = SoloFrenetModel(x0)
    N = 15
    rect_onc = []
    rect_obs = []
    # Collision avoidance with rect ellipse for oncoming vehicle as obstacle
    deg = 2
    time_steps = ego_onc_sneak_trajectories[3]["states"].shape[1]
    for time_step in range(time_steps):
        # Check collision avoidance with rect ellipse
        L = model.LENGTH
        W = model.WIDTH
        dL = (2 ** (1 / deg) - 1) * L
        dW = W / L * dL
        # center of ego
        ego_c = (
            ego_onc_sneak_trajectories[3]["states"][0, time_step]
            + model.LENGTH / 2
            - model.BACKTOWHEEL
        )
        # center of obs
        obs_c = obs_vehicle.states[0, 0] + model.LENGTH / 2 - model.BACKTOWHEEL
        # lateral deviations of ego
        ego_e = ego_onc_sneak_trajectories[3]["states"][1, time_step]
        # lateral deviations of obs
        obs_e = -0.25
        # rect ellipse
        rectellipse_s = (2 * (ego_c - obs_c) / (2 * L + dL)) ** deg
        rectellipse_e = (2 * (ego_e - obs_e) / (2 * W + dW)) ** deg
        rectellipse = rectellipse_s + rectellipse_e
        rect_obs.append(float(rectellipse))

        # Collision avoidance with rect ellipse for oncoming vehicle as obstacle
        t_onc_lower = min(
            time_step, ego_onc_sneak_trajectories[3]["states"].shape[1] - 2
        )
        t_onc_upper = min(
            time_step + N + 1, ego_onc_sneak_trajectories[3]["states"].shape[1] - 1
        )
        c_onc_0 = (
            onc_trajectory["states"][0, t_onc_lower]
            - model.LENGTH / 2
            + model.BACKTOWHEEL
        )
        c_onc_N = (
            onc_trajectory["states"][0, t_onc_upper]
            - model.LENGTH / 2
            + model.BACKTOWHEEL
        )
        L_onc = c_onc_0 - c_onc_N + 2 * model.LENGTH
        c_onc = (c_onc_N + c_onc_0) / 2
        dL_onc = (2 ** (1 / deg) - 1) * L_onc
        dW_onc = W / L_onc * dL_onc

        # lateral deviations of obs
        e_onc = onc_trajectory["states"][1, t_onc_upper]
        # rect ellipse
        rectellipse_s_onc = (2 * (ego_c - c_onc) / (2 * L_onc + dL_onc)) ** deg
        rectellipse_e_onc = (2 * (ego_e - e_onc) / (2 * W + dW_onc)) ** deg
        rectellipse_onc = rectellipse_s_onc + rectellipse_e_onc
        rect_onc.append(rectellipse_onc)
    plt.figure()
    plt.plot(
        rect_obs,
        label=r"Collision condition with Obs: $l(x_{k|t}^j)\succcurlyeq 1$",
        color=vis_util.COLORS["obs"],
    )
    plt.plot(
        rect_onc,
        label=r"Collision condition with Onc: $g(x_{k|t}^j)\preccurlyeq 0$",
        color=vis_util.COLORS["onc"],
    )
    plt.annotate(
        r"Safety Line $(y=1)$", [0, 2], color=vis_util.COLORS["ground"], fontsize=12
    )
    plt.hlines(1, 0, len(max(rect_obs, rect_onc)), colors=vis_util.COLORS["ground"])
    plt.legend()
    plt.xlabel("Time steps, t")
    plt.ylabel(r"Evaluating $l(x_{k|t}^j)$ or $g(x_{k|t}^j)$")
    plt.savefig("plots/onc_collisions.pdf")
    plt.close()
    print("obs", rect_obs)
    print("onc", rect_onc)


if __name__ == "__main__":
    # Load SOLO trajectories
    R = (0, 0)
    L = 100
    N = 15
    load_until = 6
    EXP_NAME = f"solo-R{R}L{L}N{N}"
    exp_meta = sim_util.ExpMeta(EXP_NAME, 0, 0)
    ego_solo_trajectories = [
        sim_util.load_trajectory(exp_meta, f"J{j}") for j in range(0, load_until + 1)
    ]

    # Load OBS trajectories
    R = (0, 0)
    L = 200
    N = 7
    load_until = 9
    EXP_NAME = f"obstacle-R{R}L{L}N{N}"
    exp_meta = sim_util.ExpMeta(EXP_NAME, 0, 0)
    ego_obs_trajectories = [
        sim_util.load_trajectory(exp_meta, f"J{j}") for j in range(0, load_until + 1)
    ]
    obs_vehicle = vis_util.VehicleData(
        "Obs",
        vis_util.COLORS["obs"],
        ca.DM([track.length / 2, -0.25, 0, ca.pi / 2]),
        None,
    )

    # Load both wait and sneak ONC trajectories
    (
        # ego
        _,
        _,
        _,
        # onc
        _,
        track_J0_onc,
        _,
        # lmpc
        _,
        track_ctrl,
        _,
        _,
    ) = sim_util.setup_oncoming_solo(Controller=SoloMPC, Q=ca.diag((1, 300, 200, 20)))

    # Load onc wait trajectories
    L = 200
    N = 15
    mode = "wait"
    T_START = 80
    load_until = 2
    EXP_NAME = f"oncoming-solo-L{L}N{N}T{T_START}-{mode}"
    exp_meta = sim_util.ExpMeta(EXP_NAME, 0, 0)
    ego_onc_wait_trajectories = [
        sim_util.load_trajectory(exp_meta, f"J{j}") for j in range(0, load_until + 1)
    ]
    # track_J0_onc has an extra length of 1m on end. And the s_ego starts at 1.
    ego_onc_wait_trajectories[0]["states"][0, :] -= (
        track_J0_onc.length - track_ctrl.length
    ) / 2
    onc_trajectory = sim_util.load_trajectory(exp_meta, "J0-onc")
    onc_trajectory["states"][0, :] = (
        track_J0_onc.length
        # Flip s for onc
        # track_J0_onc has an extra length of 1m on end. And the s_onc starts at 1.
        # we want to flip and shift the s of onc so that it goes
        # from length of track_ctrl to negative values
        - onc_trajectory["states"][0, :]
        # track_J0_onc has an extra length of 1m on end. And the s_ego starts at 1.
        - (track_J0_onc.length - track_ctrl.length) / 2
    )

    # Load onc sneak trajectories
    mode = "sneak"
    load_until = 6
    EXP_NAME = f"oncoming-solo-L{L}N{N}T{T_START}-{mode}"
    exp_meta = sim_util.ExpMeta(EXP_NAME, 0, 0)
    ego_onc_sneak_trajectories = [
        sim_util.load_trajectory(exp_meta, f"J{j}") for j in range(0, load_until + 1)
    ]
    # track_J0_onc has an extra length of 1m on end. And the s_ego starts at 1.
    ego_onc_sneak_trajectories[0]["states"][0, :] -= (
        track_J0_onc.length - track_ctrl.length
    ) / 2
    onc_trajectory = sim_util.load_trajectory(exp_meta, "J0-onc")
    onc_trajectory["states"][0, :] = (
        track_J0_onc.length
        # Flip s for onc
        # track_J0_onc has an extra length of 1m on end. And the s_onc starts at 1.
        # we want to flip and shift the s of onc so that it goes
        # from length of track_ctrl to negative values
        - onc_trajectory["states"][0, :]
        # track_J0_onc has an extra length of 1m on end. And the s_ego starts at 1.
        - (track_J0_onc.length - track_ctrl.length) / 2
    )

    cart_vs_frenet()
    scenarios()
    trajectory_results(ego_solo_trajectories, plot_name="solo")
    trajectory_results(
        ego_obs_trajectories, vehicles=[obs_vehicle], plot_name="obs", skip_even=True
    )
    onc_results()
    onc_safety_condition()
