import casadi as ca
import numpy as np
from controllers.solo import SoloMPC, SoloRelaxedLMPC
from models.solo import SoloFrenetModel
from simulators.solo import SoloMPCSimulator, SoloRelaxedLMPCSimulator
from utils import sim_util, vis_util

from controllers.obstacle import ObstacleLMPC
from simulators.solo import SoloMPCSimulator, SoloRelaxedLMPCSimulator
from controllers.solo import SoloMPC
from controllers.oncoming import OncomingSoloLMPC
from simulators.obstacle import ObstacleLMPCSimulator, ObstacleMPCSimulator
from simulators.oncoming import (
    OncomingEgoMPCSimulator,
    OncomingOncMPCSimulator,
    OncomingSoloLMPCSimulator,
)
from utils import sim_util, vis_util

if __name__ == "__main__":
    """
    For wait J0-J1 had L=50, then L=100
    """
    L = 200  # "inf" i.e. not slack didn't get far (t 41)
    # on left L 1 on mid L1 sneak on right L200 sneak
    N = 15  # running with slack on onc_ellipse
    # mode = "wait"  # "wait" or "sneak"
    mode = "wait"  # "wait" or "sneak"
    # make sure wait works and use the same set of parameters for sneak afterwards
    # at the moment there's no finished trajectory for sneak
    T_START = 80

    EXP_NAME = f"oncoming-solo-L{L}N{N}T{T_START}-{mode}"

    load_until = 2

    AWAIT_ONC = True if mode == "wait" else False
    # Generate Initial Trajectory on ONC vehicle
    (
        # ego
        model_ego,
        track_J0_ego,
        mpc_ego,
        # onc
        model_onc,
        track_J0_onc,
        mpc_onc,
        # lmpc
        track_vis,
        track_ctrl,
        _,
        xub_lmpc,
    ) = sim_util.setup_oncoming_solo(Controller=SoloMPC, Q=ca.diag((1, 300, 200, 20)))

    ##################
    ##### ONC J0 #####
    ##################
    track_J0_onc.update_n_points(236)
    s_obs = track_ctrl.length / 2
    simulator = OncomingOncMPCSimulator(model_onc, mpc_onc, track_J0_onc, track_J0_onc)
    # how many time-steps to wait before moving onc
    simulator.T_START = T_START
    simulator.EXP_NAME = EXP_NAME
    simulator.S_OBS = s_obs

    traj_0_onc = simulator.load(iteration="0-onc")
    if traj_0_onc is None:  # if J0 is not stored, run simulator, save, then load it
        traj_0_onc = simulator.run().save("0-onc").load("0-onc")

    # Flip s for onc
    # track_J0_onc has an extra length of 1m on end. And the s_onc starts at 1.
    # we want to flip and shift the s of onc so that it goes
    # from length of track_ctrl to negative values
    traj_0_onc["states"][0, :] = (
        track_J0_onc.length
        - traj_0_onc["states"][0, :]
        # - (track_J0_onc.length - track_ctrl.length) / 2
    )

    ##################
    ##### EGO J0 #####
    ##################
    track_J0_ego.update_n_points(236)
    s_obs = track_J0_ego.length / 2
    simulator = OncomingEgoMPCSimulator(
        model_ego, mpc_ego, track_J0_ego, track_J0_ego, traj_0_onc
    )
    simulator.AWAIT_ONC = AWAIT_ONC
    simulator.EXP_NAME = EXP_NAME
    simulator.S_OBS = s_obs

    traj_0_ego = simulator.load(iteration=0)
    if traj_0_ego is None:  # if J0 is not stored, run simulator, save, then load it
        traj_0_ego = simulator.run().save().load()

    # Shift s for ego
    # track_J0_onc has an extra length of 1m on end. And the s_ego starts at 1.
    traj_0_ego["states"][0, :] -= (track_J0_onc.length - track_ctrl.length) / 2
    traj_0_onc["states"][0, :] -= (track_J0_onc.length - track_ctrl.length) / 2

    #####################
    ### Test plot onc ###
    #####################
    # vis_util.animate_trajectory(
    #     simulator.exp_meta,
    #     track_vis,
    #     [
    #         vis_util.VehicleData(
    #             "ego",
    #             vis_util.COLORS["ego"],
    #             traj_0_ego["states"],
    #             traj_0_ego["inputs"],
    #         ),
    #         vis_util.VehicleData(
    #             "onc",
    #             vis_util.COLORS["onc"],
    #             traj_0_onc["states"],
    #             traj_0_onc["inputs"],
    #         ),
    #     ],
    #     plot_input=True,
    # )
    # quit()

    # vis_util.compare_iterations(EXP_NAME, 0, 9, track_vis, step=1, s_obs=s_obs)
    # quit()

    trajectories = [traj_0_ego] + [simulator.load(j) for j in range(1, load_until + 1)]

    traj = trajectories[-1]
    vis_util.animate_trajectory(
        simulator.exp_meta,
        track_vis,
        [
            vis_util.VehicleData(
                "ego",
                vis_util.COLORS["ego"],
                traj["states"][:4, :],
                traj["inputs"],
            ),
            vis_util.VehicleData(
                "onc",
                vis_util.COLORS["onc"],
                traj_0_onc["states"],
                traj_0_onc["inputs"],
            ),
            vis_util.VehicleData(
                "obs",
                vis_util.COLORS["obs"],
                ca.DM([track_vis.length / 2, -0.25, 0, ca.pi / 2]),
                None,
            ),
        ],
        plot_input=True,
        save=True,
        # animation_filename="J0-sneak",
        animation_filename="J2",
    )
    quit()

    #####################
    ### LMPC Training ###
    #####################

    model_ego.x0[0, 0] = 0
    model_ego.reset_trajectory()

    lmpc = OncomingSoloLMPC(
        model_ego,
        Q=None,
        R=None,  # ca.diag(R),
        xlb=-xub_lmpc,
        xub=xub_lmpc,
        ulb=-mpc_onc.uub,
        uub=mpc_onc.uub,
        N=N,
    )
    lmpc.L = L
    simulator = OncomingSoloLMPCSimulator(
        model_ego,
        lmpc,
        track_vis,
        track_ctrl,
        trajectories,
        max_iter=10,
        onc_traj=traj_0_onc,
    )
    simulator.EXP_NAME = EXP_NAME
    s_obs = track_vis.length / 2
    simulator.track_ctrl.update_n_points(200)
    simulator.S_OBS = s_obs
    print(simulator.T)
    # simulator.run()
