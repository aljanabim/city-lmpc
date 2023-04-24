import casadi as ca
from utils.sim_util import create_track
from utils.vis_util import draw_arrow
from utils import sim_util, vis_util
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

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


# PLOT ALL THREE SCENARIOS SIDE-BY-SIDE
def scenarios():
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
    obs_vehicle = (
        vis_util.VehicleData(
            "obs",
            vis_util.COLORS["obs"],
            ca.DM([track.length / 2, -0.25, 0, ca.pi / 2]),
            None,
        ),
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
        xub_lmpc,
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
    ego_onc_wait_trajectories[0]["states"][0, :] -= (
        track_J0_onc.length - track_ctrl.length
    ) / 2
    onc_trajectory = sim_util.load_trajectory(exp_meta, "J0-onc")
    onc_trajectory["states"][0, :] = (
        track_J0_onc.length
        - onc_trajectory["states"][0, :]
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
    ego_onc_sneak_trajectories[0]["states"][0, :] -= (
        track_J0_onc.length - track_ctrl.length
    ) / 2
    onc_trajectory = sim_util.load_trajectory(exp_meta, "J0-onc")
    onc_trajectory["states"][0, :] = (
        track_J0_onc.length
        - onc_trajectory["states"][0, :]
        - (track_J0_onc.length - track_ctrl.length) / 2
    )

    vehicles = [
        # vis_util.VehicleData(
        #     "ego",
        #     vis_util.COLORS["ego"],
        #     traj["states"][:4, :],
        #     traj["inputs"],
        # ),
        # vis_util.VehicleData(
        #     "obs",
        #     vis_util.COLORS["obs"],
        #     ca.DM([track_vis.length / 2, -0.25, 0, ca.pi / 2]),
        #     None,
        # ),
    ]

    vis_util.plot_trajectory(track, vehicles, plot_axis=True, show=True)

    # vis_util.animate_trajectory(
    #     simulator.exp_meta,
    #     track_vis,
    #     [
    #         vis_util.VehicleData(
    #             "ego",
    #             vis_util.COLORS["ego"],
    #             traj["states"][:4, :],
    #             traj["inputs"],
    #         ),
    #         # vis_util.VehicleData(
    #         #     "obs",
    #         #     vis_util.COLORS["obs"],
    #         #     ca.DM([track_vis.length / 2, -0.25, 0, ca.pi / 2]),
    #         #     None,
    #         # ),
    #     ],
    #     plot_input=True,
    #     save=True,
    #     animation_filename="J0",
    # )
    plt.show()


if __name__ == "__main__":
    # cart_vs_frenet()
    scenarios()
