from dataclasses import dataclass
import sys
from os import path, mkdir, remove

# system path management (to import from adjacent directories)
root = path.abspath(path.join(path.dirname(path.abspath(__file__)), ".."))
from dataclasses import dataclass
import sys
from os import path, mkdir, remove
from models.track import Track
import numpy as np
from matplotlib import pyplot as plt
import pickle
from casadi import DM, vertcat


@dataclass
class ExpMeta:
    exp_name: str  # name of experiment
    s_obs: float  # position of obstacle vehicle along the s_axis
    s_lane_obs: float  # position of back to main lane constraint along the s-axis
    s_ego_0: float  # initial position of ego vehicle


# @dataclass
# class TrajectoryMeta:
#     n_vehicle: int  # amount of vehicles in the platoon
#     x0: float  # offset on the starting position of the entire platoon in the starting position of the
#     merge_x: float  # the postion of the merge point between the platoon and the intruder
#     length_highway: float  # the length of the highway in meters
#     length_onramp: float  # the length of the onramp lane in meters
#     hw: float  # the time headway in seconds
#     r: float  # the standstill distance between the platooning vehicles in meters
#     v_ref: float  # the reference velocity of both platoon and intruder
#     v_min: float  # not in use
#     v_max: float  # not in use
#     # the id of the vehicle which ends up tracking the intruder according to the fifo algorithm
#     tracking_id: int = 0


# def meta_to_filename(meta_dict):
#     # exp_name, s_lane_obs, s_obs, s_0_ego, s_0_onc = meta_dict.values()
#     # exp_name, s_lane_obs, s_obs, s_0_ego, s_0_onc = exp_name, ffloat(
#     #     s_lane_obs), ffloat(s_obs), ffloat(s_0_ego), ffloat(s_0_onc)
#     n, x0, mx, lh, lo, hw, r, vre, vmi, vma, tr = meta_dict.values()
#     n, x0, mx, lh, lo, hw, r, vre, vmi, vma, tr = str(n), ffloat(x0), ffloat(mx), ffloat(lh), ffloat(
#         lo), ffloat(hw), ffloat(r), ffloat(vre), ffloat(vmi), ffloat(vma), ffloat(tr)

#     return f"n_{n}_tr_{tr}_x_{x0}_mx_{mx}_lh_{lh}_lo_{lo}_hw_{hw}_r_{r}_vre_{vre}_vmi_{vmi}_vma_{vma}"


# def get_trajectory_root(sim_name, trajectory_meta: TrajectoryMeta):
#     traj_name = f"{sim_name}_{meta_to_filename(trajectory_meta.__dict__)}"
#     return path.join(root, "trajectories", traj_name)


def ffloat(num):
    return str(num).replace(".", ".")


def get_obs_state(exp_meta: ExpMeta, track: Track):
    point_idx = np.argmin(np.absolute(track.points_array[2, :] - exp_meta.s_obs))

    point = track.points[point_idx]
    return vertcat(point.s, -track.L_LANE / 2, 0, point.phi_s)


def get_data_folder(exp_meta: ExpMeta, s_0_onc: int):
    exp_name = exp_meta.exp_name
    exp_folder = path.join(root, "trajectories", exp_name)
    data_folder = path.join(exp_folder, f"s_onc_{s_0_onc}")
    return data_folder


def create_experiment(exp_meta: ExpMeta, s_0_onc: int = None):
    exp_name = exp_meta.exp_name
    exp_folder = path.join(root, "trajectories", exp_name)
    try:
        mkdir(exp_folder)
        with open(path.join(exp_folder, "meta.txt"), "w+") as file:
            for k, v in exp_meta.__dict__.items():
                file.write(f"{k}: {v}\n")
    except OSError as error:
        print(error)

    if s_0_onc is not None:
        data_folder = path.join(exp_folder, f"s_onc_{s_0_onc}")
        try:
            mkdir(data_folder)
        except OSError as error:
            print(error)


def dump_trajectory(
    exp_meta: ExpMeta, s_0_onc: int, trajectory_filename: str, states, inputs
):
    create_experiment(exp_meta, s_0_onc)
    data_folder = get_data_folder(exp_meta, s_0_onc)

    with open(path.join(data_folder, trajectory_filename + ".npx"), "wb") as file:
        if isinstance(states, DM):
            states = states.full()
        if isinstance(states, DM):
            states = states.full()
        trajectory_dict = {"states": states, "inputs": inputs}
        pickle.dump(trajectory_dict, file)
    print("Dumped trajectories for", trajectory_filename, "to", data_folder)


def load_trajectory(exp_meta: ExpMeta, s_0_onc: int, trajectory_filename: str):
    data_folder = get_data_folder(exp_meta, s_0_onc)

    try:
        with open(path.join(data_folder, trajectory_filename + ".npx"), "rb") as file:
            return pickle.load(file)
    except IOError as error:
        print(error)
        trajectory_dict = {"states": None, "inputs": None}
        return trajectory_dict


def check_trajectory(exp_meta: ExpMeta, s_0_onc: int, trajectory_filename: str):
    data_folder = get_data_folder(exp_meta, s_0_onc)
    file_path = path.join(data_folder, trajectory_filename + ".npx")
    print("Found", file_path)
    return path.exists(file_path)


# def animate_trajectory(sim_name, trajectory_meta: TrajectoryMeta, platoon_state_trajectory, intruder_state_trajectory, intruder_input_trajectory, spv, save=False):

if __name__ == "__main__":
    quit()
    tracking_id = 2
    x0_offset = 4.0  # 4.0 -> 2 tracking | 2.0 -> 1 tracking | 0.0 -> 0 tracking
    trajectory_meta = TrajectoryMeta(
        3, x0_offset, 8.0, 25, 5, 0.5, 0.3, 0.5, 0, 0.5, tracking_id
    )

    # dump_trajectory("lmpc", trajectory_meta, [0.4, 0], [0, 0])
    intruder_state, intruder_input = load_trajectory(
        "intruder", trajectory_meta
    ).values()

    for i in range(3):
        plt.suptitle(f"Vehicle {i}")
        for j in range(0, 4):
            if j == 0:
                p_state, p_input = load_trajectory(
                    "fifo", trajectory_meta, J=j
                ).values()
            else:
                p_state, p_input = load_trajectory(
                    "lmpc", trajectory_meta, J=j
                ).values()
            if i == 0:
                print(f"J={j}", p_state.shape, p_input.shape)
            plt.subplot(3, 2, i * 2 + 1)
            plt.title("Velocities")
            plt.plot(p_state[i * 3 + 1, :], label=f"V={i+1} J={j}")
            plt.legend()

            plt.subplot(3, 2, i * 2 + 2)
            plt.title("Inputs")
            plt.plot(p_input[i, :], label=f"V={i+1} J={j}")
            plt.legend()
    plt.show()
    # animate_trajectory("lmpc", trajectory_meta, p_state,
    #                    intruder_state, intruder_input, 3)

    quit()

    ffifo_state, ffifo_input = load_trajectory("ffifo", trajectory_meta).values()
    print(fifo_state.shape, ffifo_state.shape)
    quit()
    print(ffifo_state[:, -3:])
    print(fifo_state[:, -3:])
    print(np.abs(ffifo_state[:, -3:] - fifo_state[:, -3:]))
    # animate_trajectory("fifo", trajectory_meta, platoon_state,
    #                    intruder_state, intruder_input, 3)
    quit()
    # print(np.max(fifo_state, axis=1))
    print(np.max(fifo_input, axis=1))
    print(np.min(fifo_input, axis=1))
    quit()

    # animate_trajectory(
    #     "lmpc",
    #     trajectory_meta,
    #     platoon_state,
    #     intruder_state,
    #     intruder_input,
    #     3,
    #     save=False,
    # )
    quit()
    x, u = load_trajectory("lmpc", trajectory_meta, J=1).values()
    np.set_printoptions(threshold=sys.maxsize)
    y_outside_mask = np.abs(intruder_state[1, :] - 0.5) < (0.5 + 0.324) / 2
    for i in range(3):
        dist = np.sqrt(
            (x[i * 3, y_outside_mask] - intruder_state[0, y_outside_mask]) ** 2
        ) > (0.586)
        print(dist)
    quit()
    err = np.linalg.norm(x[::3, :] - platoon_state[::3, :], axis=0)
    err1 = np.linalg.norm(x - platoon_state, axis=0)
    tol = 0.08
    print(err.size)
    print(np.sum(err < tol))
    print(np.sum(err1 < tol))

    # for i, v in enumerate(err):
    #     print(i, v)

    # print(x.shape[1])
    # print([f"{i} {v}\n" for i, v in enumerate(err)])
    # print(platoon_state[0::3, 80])
    # print(x[:, 80])
