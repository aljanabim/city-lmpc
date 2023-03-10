import casadi as ca
import numpy as np
from controllers.solo import SoloMPC, SoloRelaxedLMPC
from models.solo import SoloFrenetModel
from simulators.solo import SoloMPCSimulator, SoloRelaxedLMPCSimulator
from utils import sim_util, vis_util

from controllers.obstacle import ObstacleLMPC
from simulators.solo import SoloMPCSimulator, SoloRelaxedLMPCSimulator
from controllers.solo import SoloMPC
from controllers.oncoming import OncomingSoloLMPC, OncomingDuoLMPC
from simulators.oncoming import (
    OncomingEgoMPCSimulator,
    OncomingOncMPCSimulator,
    OncomingSoloLMPCSimulator,
    OncomingDuoLMPCSimulator,
)
from models.oncoming import DuoFrenetModel
from utils import sim_util, vis_util

if __name__ == "__main__":
    """ """
    L = 200
    N = 7
    mode = "wait"  # "wait" or "sneak"
    # mode = "sneak"  # "wait" or "sneak"
    T_START = 80  # back to 80

    EXP_NAME = f"oncoming-duo-L{L}N{N}T{T_START}-{mode}"

    load_until = 0
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
        track_ctrl_flipped,
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
    traj_0_onc["states"][0, :] = track_J0_onc.length - traj_0_onc["states"][0, :]
    traj_0_onc["states"][3, :] -= 2 * ca.pi

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

    # Combine ego and onc trajectories into extended model
    ego_states = traj_0_ego["states"]
    ego_inputs = traj_0_ego["inputs"]

    onc_states = ca.DM.ones(ego_states.shape) * traj_0_onc["states"][:, -1]
    onc_states[:, : traj_0_onc["states"].shape[1]] = traj_0_onc["states"]

    onc_inputs = ca.DM.zeros(ego_inputs.shape)
    onc_inputs[:, : traj_0_onc["inputs"].shape[1]] = traj_0_onc["inputs"]

    traj_0 = {
        "states": ca.vertcat(ego_states, onc_states),  # (8, 407)
        "inputs": ca.vertcat(
            # first stack the u's then the p's
            ego_inputs[:2, :],  # u_ego
            onc_inputs[:2, :],  # u_onc
            ego_inputs[2:, :],  # p_ego
            onc_inputs[2:, :],  # p_onc
        ),  # (10, 406)
    }

    trajectories = [traj_0] + [simulator.load(j) for j in range(1, load_until + 1)]
    # print("s_ego", traj_0["states"][0, :])
    # print("s_onc", traj_0["states"][4, :])

    # print("phi_ego", traj_0["states"][3, :])
    # print("phi_onc", traj_0["states"][7, :])

    #####################
    ### LMPC Training ###
    #####################
    model_duo = DuoFrenetModel(traj_0["states"][:, 0])

    lmpc = OncomingDuoLMPC(
        model_duo,
        Q=None,
        R=None,  # ca.diag(R),
        xlb=-ca.vertcat(xub_lmpc, xub_lmpc),
        xub=ca.vertcat(xub_lmpc, xub_lmpc),
        ulb=-ca.vertcat(mpc_onc.uub, mpc_onc.uub),
        uub=ca.vertcat(mpc_onc.uub, mpc_onc.uub),
        N=N,
    )
    lmpc.L = L
    simulator = OncomingDuoLMPCSimulator(
        model_duo,
        lmpc,
        track_vis,
        track_ctrl,
        track_ctrl_flipped,
        trajectories,
        max_iter=10,
    )
    simulator.EXP_NAME = EXP_NAME
    s_obs = track_vis.length / 2
    simulator.track_ctrl.update_n_points(200)
    simulator.track_ctrl_flipped.update_n_points(200)
    simulator.S_OBS = s_obs
    simulator.run()
