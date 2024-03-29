from dataclasses import dataclass
import sys
from os import path, mkdir, remove
from models.track import Track
import numpy as np
from matplotlib import pyplot as plt
import pickle
import casadi as ca
from models.base import BaseModel
from models.track import ArcByAngle, ArcByLength, Track

root = path.abspath(path.join(path.dirname(path.abspath(__file__)), ".."))


@dataclass
class ExpMeta:
    exp_name: str  # name of experiment
    s_obs: float  # position of obstacle vehicle along the s_axis
    s_ego_0: float  # initial position of ego vehicle


def get_obs_state(exp_meta: ExpMeta, track: Track):
    point_idx = np.argmin(np.absolute(track.points_array[2, :] - exp_meta.s_obs))

    point = track.points[point_idx]
    return ca.vertcat(point.s, -track.L_LANE / 2, 0, point.phi_s)


def get_data_folder(exp_meta: ExpMeta):
    exp_name = exp_meta.exp_name
    data_folder = path.join(root, "trajectories", exp_name)
    return data_folder


def create_experiment(exp_meta: ExpMeta, s_0_onc: int = None):
    exp_name = exp_meta.exp_name
    exp_folder = path.join(root, "trajectories", exp_name)
    try:
        mkdir(exp_folder)
    except OSError as error:
        print(error)

    # create meta data file
    with open(path.join(exp_folder, "meta.txt"), "w+") as file:
        for k, v in exp_meta.__dict__.items():
            file.write(f"{k}: {v}\n")

    if s_0_onc is not None:
        data_folder = path.join(exp_folder, f"s_onc_{s_0_onc}")
        try:
            mkdir(data_folder)
        except OSError as error:
            print(error)


def dump_trajectory(exp_meta: ExpMeta, trajectory_filename: str, states, inputs):
    create_experiment(exp_meta)
    data_folder = get_data_folder(exp_meta)

    with open(path.join(data_folder, trajectory_filename + ".npx"), "wb") as file:
        if isinstance(states, ca.DM):
            states = states.full()
        if isinstance(states, ca.DM):
            states = states.full()
        trajectory_dict = {"states": states, "inputs": inputs}
        pickle.dump(trajectory_dict, file)
    print("Dumped trajectories for", trajectory_filename, "to", data_folder)


def load_trajectory(exp_meta: ExpMeta, trajectory_filename: str):
    data_folder = get_data_folder(exp_meta)

    try:
        with open(path.join(data_folder, trajectory_filename + ".npx"), "rb") as file:
            return pickle.load(file)
    except IOError as error:
        print(error)
        return None


def check_trajectory(exp_meta: ExpMeta, trajectory_filename: str):
    """Checks whether a certain trajectory file exists.

    Args:
        exp_meta (ExpMeta): Meta data about experiment. Used to get the experiment name.
        trajectory_filename (str): File name for trajectory file to be checked.

    Returns:
        bool: Indicates whether trajectory files exists or not
    """
    data_folder = get_data_folder(exp_meta)
    file_path = path.join(data_folder, trajectory_filename + ".npx")
    print("Found", file_path)
    return path.exists(file_path)


def compute_ref_trajectory(
    x: ca.DM,
    track: Track,
    dt: float,
    N: int,
    v_ref: float,
    e_ref: float,
    flipped_ref: bool = False,
    look_ahead_base=0,
    look_ahead_factor=0,
):
    """
    Computes the trajectory of the next N points starting from the state x, on the track.

    Args:
        x (ca.DM): Starting state
        track (Track): Track Model
        dt (float): size of one discretization step
        N (int): Number of reference points to compute
        v_ref (float): reference velocity
        e_ref (float): reference lateral deviation from track center
        flipped_ref (float): indicated whether reference should be generated for an oncoming vehicle.
    Returns:
        x_ref (ca.DM): the reference states the vehicle should follow during the N+1 next points
        curvature (ca.DM): curvature of the road for the reference points
        s_0_arc (ca.DM): starting s-coordinate of the arc section for the reference points
        phi_0_arc (ca.DM): starting phi-coordinate of the arc section for the reference points
    """
    v = x[2, 0]  # assume velocity is the third state
    look_ahead_dist = look_ahead_base + look_ahead_factor * v * N * dt

    i = np.argmin(
        np.abs(track.points_array[2, :] - x[0, 0] - look_ahead_dist)
    )  # i_with_look_ahead
    # Find out whether to get the last points in the track only
    get_last = i + N >= len(track.points)

    n_states = x.shape[0]
    x_ref = ca.DM.zeros(n_states, N + 1)
    curvature = ca.DM.zeros(1, N)
    phi_0_arc = ca.DM.zeros(1, N)
    s_0_arc = ca.DM.zeros(1, N)
    if get_last:
        x_ref[0, :] = track.points_array[2, -1]
    else:
        x_ref[0, :] = track.points_array[2, i : i + N + 1]

    x_ref[1, :] = e_ref
    x_ref[2, :] = v_ref + 0.01  # + correction term

    i = np.argmin(np.abs(track.points_array[2, :] - x[0, 0]))
    get_last = i + N >= len(track.points)
    if flipped_ref:
        if get_last:
            x_ref[-1, :] = track.points_array[4, -1] - np.pi
            curvature[0, :] = -track.points_array[3, -1]
        else:
            x_ref[-1, :] = track.points_array[4, i : i + N + 1] - np.pi
            curvature[0, :] = -track.points_array[3, i : i + N]
    else:
        if get_last:
            x_ref[-1, :] = track.points_array[4, -1]
            curvature[0, :] = track.points_array[3, -1]
        else:
            x_ref[-1, :] = track.points_array[4, i : i + N + 1]
            curvature[0, :] = track.points_array[3, i : i + N]
    if get_last:
        phi_0_arc[0, :] = track.points_array[6, -1]
        s_0_arc[0, :] = track.points_array[5, -1]
    else:
        phi_0_arc[0, :] = track.points_array[6, i : i + N]
        s_0_arc[0, :] = track.points_array[5, i : i + N]

    return x_ref, curvature, s_0_arc, phi_0_arc


def create_track(x0, y0, phi0, arcs=None, both=False, J0=False, flip=False) -> Track:
    """
    Creates the track used in all scenarios.

    Args:
        x0 (float): x-position of starting point in track
        y0 (float): y-position of starting point in track
        phi0 (float): rotation angle of starting point in track
        arcs (list of Arcs): arcs that make up the track

    Returns:
        (Track): instance of track
    """
    if arcs is None:
        arcs = [
            ArcByLength(0, 1.35),
            ArcByLength(0, 0.50),
            ArcByAngle(-1 / np.sqrt(2) / 0.65, 90),
            ArcByLength(0, 4),
            ArcByAngle(1 / np.sqrt(2) / 0.65, 90),
            ArcByLength(0, 0.50),
            ArcByLength(0, 1.35),
        ]

        if J0:  # add an extra 2 meters for J0 vehicle
            if both:
                arcs.insert(0, ArcByLength(0, 1))
                arcs.append(ArcByLength(0, 1))
            else:
                arcs.append(ArcByLength(0, 2))
    return Track(arcs, x_s0=x0, y_s0=y0, phi_s0=phi0, flip=flip)


def setup_solo(Controller, Q=None, R=None):
    from models.solo import SoloFrenetModel

    lane_width = 0.5
    yaw0 = ca.pi
    track_vis = create_track(0, 0, yaw0)
    track_ctrl = create_track(0, 0, yaw0)
    track_J0 = create_track(0, 0, yaw0, J0=True)

    # Get Model
    x = ca.vertcat(0, -lane_width / 2, 0, yaw0)
    model = SoloFrenetModel(x)

    # Input and state constraints for MPC, +0.5 on s to allow reaching the target and the margin +0.5 for J0
    xub = ca.vertcat(track_ctrl.length * 2, lane_width - model.WB / 2, 0.5, ca.inf)
    uub = ca.vertcat(1.7, ca.pi / 4)

    xub_lmpc = ca.vertcat(track_ctrl.length * 2, lane_width - model.WB / 2, 0.7, ca.inf)
    # Get trajectory for initial iteration
    mpc = Controller(
        model,
        Q=Q if Q is not None else ca.diag((1, 500, 100, 20)),  # (1, 300, 200, 20)
        R=R if R is not None else ca.diag((2, 1)),  # (100, 4)
        xlb=-xub,
        xub=xub,
        ulb=-uub,
        uub=uub,
    )
    return model, mpc, track_vis, track_ctrl, track_J0, xub_lmpc


def setup_oncoming_solo(Controller, Q=None, R=None):
    from models.solo import SoloFrenetModel

    lane_width = 0.5
    yaw0 = ca.pi

    track_vis = create_track(0, 0, yaw0)
    track_ctrl = create_track(0, 0, yaw0)
    track_ctrl_flipped = create_track(0, 0, yaw0, flip=True)

    track_J0_ego = create_track(0, 0, yaw0, J0=True, both=True)
    x_ego = ca.vertcat(1, -lane_width / 2, 0, yaw0)
    model_ego = SoloFrenetModel(x_ego)

    track_J0_onc = create_track(0, 0, yaw0, J0=True, both=True, flip=True)
    x_onc = ca.vertcat(1, lane_width / 2, 0, yaw0 + ca.pi)
    model_onc = SoloFrenetModel(x_onc)

    # Input and state constraints for MPC, +0.5 on s to allow reaching the target and the margin +0.5 for J0
    xub = ca.vertcat(track_ctrl.length * 2, lane_width - model_ego.WB / 2, 0.5, ca.inf)
    uub = ca.vertcat(1.7, ca.pi / 4)

    xub_lmpc = ca.vertcat(
        track_ctrl.length * 2, lane_width - model_ego.WB / 2, 0.7, ca.inf
    )

    # Get trajectory for initial iteration
    mpc_ego = Controller(
        model_ego,
        Q=Q if Q is not None else ca.diag((1, 500, 100, 20)),  # (1, 300, 200, 20)
        R=R if R is not None else ca.diag((2, 1)),  # (100, 4)
        xlb=-xub,
        xub=xub,
        ulb=-uub,
        uub=uub,
    )

    mpc_onc = Controller(
        model_onc,
        Q=Q if Q is not None else ca.diag((1, 500, 100, 20)),  # (1, 300, 200, 20)
        R=R if R is not None else ca.diag((2, 1)),  # (100, 4)
        xlb=-xub,
        xub=xub,
        ulb=-uub,
        uub=uub,
    )
    return (
        # ego
        model_ego,
        track_J0_ego,
        mpc_ego,
        # onc
        model_onc,
        track_J0_onc,
        mpc_onc,
        # Solo LMPC
        track_vis,
        track_ctrl,
        track_ctrl_flipped,
        xub_lmpc,
    )
