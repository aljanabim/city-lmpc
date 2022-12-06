# system path management (to import from adjacent directories)
from dataclasses import dataclass
import sys
from os import path, mkdir, remove

import numpy as np
import math
import matplotlib.pyplot as plt
from dataclasses import dataclass
import casadi as ca
import imageio
from models.track import Track
from tqdm import tqdm

# SVEA Vehicle parameters
LENGTH = 0.586  # [m]
WIDTH = 0.2485  # [m]
BACKTOWHEEL = 0.16  # [m]
WHEEL_LEN = 0.03  # [m]
WHEEL_WIDTH = 0.02  # [m]
TREAD = 0.07  # [m]
WB = 0.324  # [m]

chassis_height = 0.06  # [m] approx.
top_height = 0.22  # [m] approx.

COLORS = {"onc": "#000000", "ego": "#00c0b3", "obs": "#FF0000"}


@dataclass
class VehicleData:
    label: str
    color: str
    states: ca.DM = None
    inputs: ca.DM = None


def plot_car(x, y, yaw, steer=0.0, color="-k", plot=True, label=""):
    """
    Plotting function from PythonRobotics MPC

    :param x: Current x position of car in [m]
    :type x: float
    :param y: Current y position of car in [m]
    :type y: float
    :param yaw: Current yaw of car in [rad]
    :type yaw: float
    :param steer: Current steering angle of car's front wheels [rad]
    :type steer: float
    :param color: Color of plotted vehicle works with matplotlib colors
    :type color: str
    """

    outline = np.matrix(
        [
            [
                -BACKTOWHEEL,
                (LENGTH - BACKTOWHEEL),
                (LENGTH - BACKTOWHEEL),
                -BACKTOWHEEL,
                -BACKTOWHEEL,
            ],
            [WIDTH / 2, WIDTH / 2, -WIDTH / 2, -WIDTH / 2, WIDTH / 2],
        ]
    )

    fr_wheel = np.matrix(
        [
            [WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
            [
                -WHEEL_WIDTH - TREAD,
                -WHEEL_WIDTH - TREAD,
                WHEEL_WIDTH - TREAD,
                WHEEL_WIDTH - TREAD,
                -WHEEL_WIDTH - TREAD,
            ],
        ]
    )

    rr_wheel = np.copy(fr_wheel)

    fl_wheel = np.copy(fr_wheel)
    fl_wheel[1, :] *= -1
    rl_wheel = np.copy(rr_wheel)
    rl_wheel[1, :] *= -1

    Rot1 = np.matrix([[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]])
    Rot2 = np.matrix(
        [[math.cos(steer), math.sin(steer)], [-math.sin(steer), math.cos(steer)]]
    )

    fr_wheel = (fr_wheel.T * Rot2).T
    fl_wheel = (fl_wheel.T * Rot2).T
    fr_wheel[0, :] += WB
    fl_wheel[0, :] += WB

    fr_wheel = (fr_wheel.T * Rot1).T
    fl_wheel = (fl_wheel.T * Rot1).T

    outline = (outline.T * Rot1).T
    rr_wheel = (rr_wheel.T * Rot1).T
    rl_wheel = (rl_wheel.T * Rot1).T

    outline[0, :] += x
    outline[1, :] += y
    fr_wheel[0, :] += x
    fr_wheel[1, :] += y
    rr_wheel[0, :] += x
    rr_wheel[1, :] += y
    fl_wheel[0, :] += x
    fl_wheel[1, :] += y
    rl_wheel[0, :] += x
    rl_wheel[1, :] += y

    if plot:
        plt.plot(
            np.array(outline[0, :]).flatten(), np.array(outline[1, :]).flatten(), color
        )
        plt.plot(
            np.array(fr_wheel[0, :]).flatten(),
            np.array(fr_wheel[1, :]).flatten(),
            color,
        )
        plt.plot(
            np.array(rr_wheel[0, :]).flatten(),
            np.array(rr_wheel[1, :]).flatten(),
            color,
        )
        plt.plot(
            np.array(fl_wheel[0, :]).flatten(),
            np.array(fl_wheel[1, :]).flatten(),
            color,
        )
        plt.plot(
            np.array(rl_wheel[0, :]).flatten(),
            np.array(rl_wheel[1, :]).flatten(),
            color,
            label=label,
        )
        # plt.plot(x, y, "r*")
        return

    x = (
        np.array(outline[0, :]).flatten()
        + np.array(fr_wheel[0, :]).flatten()
        + np.array(rr_wheel[0, :]).flatten()
        + np.array(fl_wheel[0, :]).flatten()
        + np.array(rl_wheel[0, :]).flatten()
    )
    y = (
        np.array(outline[1, :]).flatten()
        + np.array(fr_wheel[1, :]).flatten()
        + np.array(rr_wheel[1, :]).flatten()
        + np.array(fl_wheel[1, :]).flatten()
        + np.array(rl_wheel[1, :]).flatten()
    )
    return x, y


def plot_highway(
    n_lanes=1,
    lane_width=0.5,
    length_highway=20,
    start_onramp=10,
    length_onramp=5,
    color="-k",
    color_dashed="#808080",
    plot=True,
):
    points_per_meter = 10

    # bottom line of onramp lane
    x_onramp_lower = np.linspace(
        start_onramp,
        length_onramp + start_onramp,
        int(np.ceil(points_per_meter * length_onramp)),
    )
    y_onramp_lower = np.zeros(len(x_onramp_lower)) - lane_width / 2

    # top part of onramp lane
    solid_line = 0.5
    # hiding code for printing small fork section in the begining (it's purely estetic)
    # x_onramp_fork = np.linspace(start_onramp, start_onramp + solid_line, int(
    #     np.ceil(solid_line * points_per_meter)))
    # fork_width = 0.05
    # y_onramp_lower = np.linspace(
    #     lane_width / 2 - fork_width, lane_width / 2, len(x_onramp_fork))
    # plt.plot(x_onramp_fork, y_onramp_lower, color)
    x_solid_line = np.linspace(
        0,
        start_onramp + solid_line,
        int(np.ceil((start_onramp + solid_line) * points_per_meter)),
    )
    y_solid_line = np.zeros(len(x_solid_line)) + lane_width / 2

    # end section off onramp lane
    end_length = 2.0
    y_onramp_end = np.linspace(
        -lane_width / 2, lane_width / 2, int(np.ceil(lane_width * points_per_meter))
    )
    x_onramp_end = np.linspace(
        start_onramp + length_onramp,
        start_onramp + length_onramp + end_length,
        len(y_onramp_end),
    )

    # solid line before onramp
    x_solid_onramp = np.linspace(
        0, start_onramp, int(np.ceil((start_onramp) * points_per_meter))
    )
    y_solid_onramp = np.zeros(len(x_solid_onramp)) + lane_width / 2

    # solid line after onramp
    x_solid_onramp = np.linspace(
        start_onramp + length_onramp + end_length,
        length_highway,
        int(
            np.ceil(
                (length_highway - length_onramp - end_length - start_onramp)
                * points_per_meter
            )
        ),
    )
    y_solid_onramp = np.zeros(len(x_solid_onramp)) + lane_width / 2

    # onramp dashed lines
    x_onramp_dashed = np.linspace(
        start_onramp + solid_line,
        start_onramp + length_onramp + end_length,
        int(np.ceil((length_onramp + end_length - solid_line) * points_per_meter)),
    )
    y_onramp_dashed = np.zeros(len(x_onramp_dashed)) + lane_width / 2

    # plot lanes
    x_lane = np.linspace(
        0, length_highway, int(np.ceil(length_highway * points_per_meter))
    )
    y_lane = np.zeros(len(x_lane)) + lane_width / 2

    if plot:
        plt.plot(x_onramp_lower, y_onramp_lower, color)
        plt.plot(x_solid_line, y_solid_line, color)
        plt.plot(x_onramp_end, y_onramp_end, color)
        plt.plot(x_solid_onramp, y_solid_onramp, color)
        plt.plot(x_solid_onramp, y_solid_onramp, color)
        plt.plot(x_onramp_dashed, y_onramp_dashed, color_dashed, linestyle=(0, (5, 10)))
    for i in range(n_lanes):
        # solid line for final lane
        if i + 1 < n_lanes:
            plt.plot(
                x_lane, y_lane + (i + 1) * lane_width, color_dashed, linestyle="--"
            )
        # dashed lines for lanes in between
        else:
            plt.plot(x_lane, y_lane + (i + 1) * lane_width, color)


def animate_trajectory(
    dest_folder: str,
    track: Track,
    vehicles: list[VehicleData],
    animation_filename: str,
    save=False,
    plot_axis=False,
    plot_input=False,
    file_format="gif",
):
    gif_name = animation_filename + "." + file_format
    dpi = 100

    all_time_steps = [vehicle.states.shape[1] for vehicle in vehicles]
    time_steps = np.max(all_time_steps)

    if save:
        print("Creating GIF of trajectory at\n", path.join(dest_folder, gif_name))
    filenames = []
    for t in tqdm(range(time_steps)):
        if save:
            plt.figure(figsize=(1920 / dpi, 1080 / dpi), dpi=dpi)

        if plot_input:
            plt.subplot(2, 1, 1)
        plt.cla()
        for vehicle in vehicles:
            # Set default values
            x, y, steering = 0, 0, 0
            # Get real state values
            if vehicle.states is not None:
                k = min(t, vehicle.states.shape[1] - 1)  # get last index for vehicle
                s = vehicle.states[0, k]
                e = vehicle.states[1, k]
                x, y = track.to_global(s=s, e=e)
                yaw = vehicle.states[-1, k]
            # Get real input values
            if vehicle.inputs is not None:
                k = min(t, vehicle.inputs.shape[1] - 1)  # get last index for vehicle
                steering = vehicle.inputs[1, k]
            plot_car(
                x,
                y,
                yaw,
                steering,
                color=vehicle.color,
                label=vehicle.label,
            )
        # Plot track
        x_track, y_track = track.cartesian
        plt.plot(x_track, y_track)

        plt.axis("equal")
        plt.legend()
        if plot_axis:
            plt.axis("on")
        else:
            plt.axis("off")

        if plot_input:
            plt.subplot(2, 1, 2)
            plt.cla()
            for vehicle in vehicles:
                if vehicle.inputs is not None:
                    plt.plot(
                        vehicle.inputs[0, :t],
                        label=f"{vehicle.label} " + r"$v_u$",
                        color=vehicle.color,
                    )
                    plt.plot(
                        vehicle.inputs[1, :t],
                        "--",
                        color=vehicle.color,
                        label=f"{vehicle.label} " + r"$\delta$",
                    )
                    plt.legend()

        # save frame
        if save:
            filename = f"{t}.png"
            filenames.append(filename)
            plt.savefig(path.join(dest_folder, filename), dpi=dpi)
            plt.close()
        else:
            plt.pause(0.05)
    if save:
        # Prolong the final frame
        for i in range(10):
            filenames.append(filenames[-1])
        print("Compiling images into a GIF")
        with imageio.get_writer(path.join(dest_folder, gif_name), mode="I") as writer:
            for filename in tqdm(filenames):
                image = imageio.imread(path.join(dest_folder, filename))
                writer.append_data(image)
        # Remove files
        print("Removing images")
        for filename in tqdm(set(filenames)):
            remove(path.join(dest_folder, filename))
        print("Finished creating GIF at\n", path.join(dest_folder, gif_name))
