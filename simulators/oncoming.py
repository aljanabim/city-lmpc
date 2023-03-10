import numpy as np
import casadi as ca
from utils import sim_util, vis_util
from models.solo import SoloFrenetModel
from controllers.solo import SoloMPC, SoloRelaxedLMPC
from models.track import Track
from simulators.base import BaseSimulator, BaseLMPCSimulator
from controllers.obstacle_old import ObstacleLMPC
from simulators.solo import SoloMPCSimulator
from controllers.oncoming import OncomingSoloLMPC, OncomingDuoLMPC
from models.oncoming import DuoFrenetModel
from typing import Dict


class OncomingOncMPCSimulator(SoloMPCSimulator):  # Using a Frenet Model
    T_START = 0

    def step(self):
        x_ref, curvature, s_0_arc, phi_0_arc = sim_util.compute_ref_trajectory(
            x=self.x,
            track=self.track_ctrl,
            dt=self.model.dt,
            N=self.controller.N,
            v_ref=0.5,
            e_ref=self.model.x0[1, 0],
            flipped_ref=True,
        )
        if self.time_step < self.T_START:  # wait condition
            uopt = ca.DM.zeros(self.controller.n_inputs, 1)  # Zero velocity
        else:
            uopt, _, _ = self.controller.get_ctrl(
                self.x, x_ref, curvature, s_0_arc, phi_0_arc, u_prev=self.u_prev
            )
        self.u_prev = uopt
        u = ca.vertcat(uopt, curvature[0], s_0_arc[0], phi_0_arc[0])
        # State Feedback Step
        self.x = self.model.sim(1, u=u, input_noise=False, state_noise=False)

        states, inputs = self.model.get_trajectory()
        self.vehicles = [
            vis_util.VehicleData("car", vis_util.COLORS["ego"], states, inputs)
        ]
        print(self.x[2, 0], "m/s")


class OncomingEgoMPCSimulator(BaseSimulator):  # Using a Frenet Model
    AWAIT_ONC = False

    def __init__(
        self, model: SoloFrenetModel, controller: SoloMPC, track, track_ctrl, onc_traj
    ):
        # Override the types for model and controller (Any by default)
        self.model = model
        self.controller = controller
        super().__init__(model, controller, track, track_ctrl)
        self.phi_obs = None
        self.u_prev = ca.DM.zeros(self.controller.n_inputs, 1)
        self.onc_traj = onc_traj

    def step(self):
        if (
            self.S_OBS - self.model.LENGTH - 1  # 1 [m] margin
            < self.x[0, 0]
            < self.S_OBS + self.model.LENGTH / 2
        ):  # switch condition
            e_ref = -self.model.x0[1, 0]
        else:
            e_ref = self.model.x0[1, 0]

        x_ref, curvature, s_0_arc, phi_0_arc = sim_util.compute_ref_trajectory(
            x=self.x,
            track=self.track_ctrl,
            dt=self.model.dt,
            N=self.controller.N,
            v_ref=0.5,
            e_ref=e_ref,
        )
        t_onc = min(self.time_step, self.onc_traj["inputs"].shape[1] - 1)
        if self.AWAIT_ONC:
            if (
                self.onc_traj["states"][0, t_onc] > self.x[0, 0]
                # We are 1.2m from OBS
                and self.S_OBS - self.model.LENGTH - 1.2 < self.x[0, 0]
            ):  # wait condition, behind obs
                uopt = ca.DM.zeros(self.controller.n_inputs, 1)  # Zero velocity
            else:
                uopt, _, _ = self.controller.get_ctrl(
                    self.x, x_ref, curvature, s_0_arc, phi_0_arc, u_prev=self.u_prev
                )
        else:
            uopt, _, _ = self.controller.get_ctrl(
                self.x, x_ref, curvature, s_0_arc, phi_0_arc, u_prev=self.u_prev
            )

        self.u_prev = uopt
        u = ca.vertcat(uopt, curvature[0], s_0_arc[0], phi_0_arc[0])
        # State Feedback Step
        self.x = self.model.sim(1, u=u, input_noise=False, state_noise=False)

        states, inputs = self.model.get_trajectory()
        obs_states = ca.DM(states.shape[0], 1)
        obs_states[0, 0] = self.S_OBS  # set s
        if self.phi_obs is None:
            point_idx = np.argmin(np.abs(self.track.points_array[2, :] - self.S_OBS))
            self.phi_obs = self.track.points[point_idx].phi_s
        obs_states[1, 0] = self.model.x0[1, 0]
        obs_states[-1, 0] = self.phi_obs  # set heading angle
        obs_inputs = ca.DM(inputs.shape[0], 1)
        self.vehicles = [
            vis_util.VehicleData("ego", vis_util.COLORS["ego"], states, inputs),
            vis_util.VehicleData("obs", vis_util.COLORS["obs"], obs_states, obs_inputs),
            vis_util.VehicleData(
                "onc",
                vis_util.COLORS["onc"],
                self.onc_traj["states"][:, [t_onc]],
                self.onc_traj["inputs"][:, [t_onc]],
            ),
        ]

        # Check collision avoidance with rect ellipse

        deg = 2
        L = self.model.LENGTH
        W = self.model.WIDTH
        dL = (2 ** (1 / deg) - 1) * L
        dW = W / L * dL
        # center of ego
        ego_c = self.x[0, :] + self.model.LENGTH / 2 - self.model.BACKTOWHEEL
        # center of obs
        obs_c = self.S_OBS + self.model.LENGTH / 2 - self.model.BACKTOWHEEL
        # lateral deviations of ego
        ego_e = self.x[1, :]
        # lateral deviations of obs
        obs_e = self.model.x0[1, 0]
        # rect ellipse
        rectellipse_s = (2 * (ego_c - obs_c) / (2 * L + dL)) ** deg
        rectellipse_e = (2 * (ego_e - obs_e) / (2 * W + dW)) ** deg
        rectellipse = rectellipse_s + rectellipse_e

        assert rectellipse >= 1, f"Rectellipse for obs is violated {rectellipse}"

        # Collision avoidance with rect ellipse for oncoming vehicle as obstacle
        t_onc_lower = min(self.time_step, self.onc_traj["states"].shape[1] - 2)
        t_onc_upper = min(
            self.time_step + self.controller.N + 1, self.onc_traj["states"].shape[1] - 1
        )
        c_onc_0 = (
            self.onc_traj["states"][0, t_onc_lower]
            - self.model.LENGTH / 2
            + self.model.BACKTOWHEEL
        )
        c_onc_N = (
            self.onc_traj["states"][0, t_onc_upper]
            - self.model.LENGTH / 2
            + self.model.BACKTOWHEEL
        )
        L_onc = c_onc_0 - c_onc_N + 2 * self.model.LENGTH
        c_onc = (c_onc_N + c_onc_0) / 2
        dL_onc = (2 ** (1 / deg) - 1) * L_onc
        dW_onc = W / L_onc * dL_onc

        # lateral deviations of obs
        e_onc = self.onc_traj["states"][1, t_onc_upper]
        # rect ellipse
        rectellipse_s_onc = (2 * (ego_c - c_onc) / (2 * L_onc + dL_onc)) ** deg
        rectellipse_e_onc = (2 * (ego_e - e_onc) / (2 * W + dW_onc)) ** deg
        rectellipse_onc = rectellipse_s_onc + rectellipse_e_onc
        assert (
            rectellipse_onc >= 1
        ), f"Rectellipse for onc is violated {rectellipse_onc}"

        print(self.x[2, 0], "m/s")


class OncomingSoloLMPCSimulator(BaseLMPCSimulator):  # Using a Frenet Model
    def __init__(
        self,
        model: SoloFrenetModel,
        controller: OncomingSoloLMPC,
        track,
        track_ctrl,
        trajectories,
        max_iter,
        onc_traj,
    ):
        # Override the types for model and controller (Any by default)
        self.model = model
        self.controller = controller
        super().__init__(model, controller, track, track_ctrl, trajectories, max_iter)
        self.n_points = self.controller.N
        self.n_points_shift = 7

        self.phi_obs = None
        self.u_prev = ca.DM.zeros(self.controller.n_inputs, 1)
        self.onc_traj = onc_traj

    def reset(self):
        super().reset()
        self.slack_norm = ca.inf
        self.u_prev = ca.DM.zeros(self.controller.n_inputs, 1)

    def keep_running(self):
        # For some reason this LMPC become infeasible outside this range
        return np.abs(self.x[0, 0] - self.track_ctrl.length) > 0.45

    def get_stored_data(self):
        stored_cost_to_go = ca.DM()
        stored_states = ca.DM()

        for i in range(self.iteration):
            cost_to_go_i = self.cost_to_go[i]
            states_i = self.SSx[i]

            # Limit ranges to ensure they don't exceed size of stored data
            # it_idx_lower = min(self.compute_it_idx(i), cost_to_go_i.shape[1] - 1)
            point_idx = (
                np.argmin(np.abs(states_i[0, :] - self.x[0, 0])) + self.n_points_shift
            )
            it_idx_lower = max(min(point_idx, cost_to_go_i.shape[1] - 1), 0)
            it_idx_upper = min(it_idx_lower + self.n_points, cost_to_go_i.shape[1])
            print(
                "idx_lower",
                it_idx_lower,
                "idx_upper",
                it_idx_upper,
                "total",
                cost_to_go_i.shape[1],
                "cost",
                cost_to_go_i[0],
            )

            stored_cost_to_go = ca.horzcat(
                stored_cost_to_go, cost_to_go_i[:, it_idx_lower:it_idx_upper]
            )
            stored_states = ca.horzcat(
                stored_states, states_i[:, it_idx_lower:it_idx_upper]
            )

        return stored_cost_to_go, stored_states

    def step(self):
        _, curvature, s_0_arc, phi_0_arc = sim_util.compute_ref_trajectory(
            x=self.x,
            track=self.track_ctrl,
            dt=self.model.dt,
            N=self.controller.N,
            v_ref=0.5,
            e_ref=self.model.x0[1, 0],
        )
        stored_cost_to_go, stored_states = self.get_stored_data()
        # no need to rebuild controller optimizer here with self.controller.build_optimizer()

        results = []
        results_infeasible = []
        i = 0
        t_onc_lower = min(self.time_step, self.onc_traj["states"].shape[1] - 2)
        t_onc_upper = min(
            self.time_step + self.controller.N + 1, self.onc_traj["states"].shape[1] - 1
        )
        onc_states = ca.horzcat(
            self.onc_traj["states"][:, t_onc_lower],
            self.onc_traj["states"][:, t_onc_upper],
        )
        for k in range(stored_cost_to_go.shape[1]):
            try:
                uopt, _, cost, slack_norm = self.controller.get_ctrl(
                    self.x,
                    curvature,
                    s_0_arc,
                    phi_0_arc,
                    terminal_cost=stored_cost_to_go[:, k],
                    terminal_state=stored_states[:, k],
                    s_obs=self.S_OBS,
                    s_final=self.track.length,
                    u_prev=self.u_prev,
                    onc_states=onc_states,
                )
                results.append((uopt, cost, slack_norm))
                print(
                    "Solving for x-xN=",
                    stored_states[:, k] - self.x,
                    "k",
                    k,
                    "index",
                    i,
                )
                i += 1
            except:
                # print(
                #     "Unable to solve for x-xN=",
                #     stored_states[:, k] - self.x,
                #     "with k",
                #     k,
                # )
                uopt = self.controller.opti.value(self.controller.u[:, [0]])
                cost = self.controller.opti.value(self.controller.cost)
                results_infeasible.append((uopt, cost, ca.inf))

        # Find the best results
        if len(results) == 0:
            print("All solutions are infeasible")
            exit()
            results = results_infeasible

        costs = [result[1] for result in results]
        opt_idx = np.argmin(costs)
        print("Best cost", results[opt_idx][1], "index", opt_idx)
        uopt = results[opt_idx][0]
        self.u_prev = uopt
        self.slack_norm = results[opt_idx][2]

        u = ca.vertcat(uopt, curvature[0], s_0_arc[0], phi_0_arc[0])
        # State Feedback Step
        self.x = self.model.sim(1, u=u, input_noise=False, state_noise=False)

        states, inputs = self.model.get_trajectory()
        obs_states = ca.DM(states.shape[0], 1)
        obs_states[0, 0] = self.S_OBS  # set s
        if self.phi_obs is None:
            point_idx = np.argmin(np.abs(self.track.points_array[2, :] - self.S_OBS))
            self.phi_obs = self.track.points[point_idx].phi_s
        obs_states[1, 0] = self.model.x0[1, 0]
        obs_states[-1, 0] = self.phi_obs  # set heading angle
        obs_inputs = ca.DM(inputs.shape[0], 1)
        onc_time_step = min(self.time_step, self.onc_traj["inputs"].shape[1] - 1)
        self.vehicles = [
            vis_util.VehicleData("ego", vis_util.COLORS["ego"], states, inputs),
            vis_util.VehicleData("obs", vis_util.COLORS["obs"], obs_states, obs_inputs),
            vis_util.VehicleData(
                "onc",
                vis_util.COLORS["onc"],
                self.onc_traj["states"][:, [onc_time_step]],
                self.onc_traj["inputs"][:, [onc_time_step]],
            ),
        ]

    def post_step(self):
        super().post_step()
        print(
            "J-1 time",
            self.cost_to_go[self.iteration - 1][0, 0],
            "J time",
            self.time_step,
            "slack norm",
            self.slack_norm,
            "dist",
            self.track.length - self.x[0, 0],
            "v",
            self.x[2, 0],
            "m/s",
        )


class OncomingDuoLMPCSimulator(BaseLMPCSimulator):  # Using a Frenet Model
    def __init__(
        self,
        model: DuoFrenetModel,
        controller: OncomingDuoLMPC,
        track,
        track_ctrl,
        track_ctrl_flipped,
        trajectories,
        max_iter,
    ):
        # Override the types for model and controller (Any by default)
        self.model = model
        self.controller = controller
        self.track_ctrl_flipped = track_ctrl_flipped
        super().__init__(model, controller, track, track_ctrl, trajectories, max_iter)

        self.n_points = 10
        self.n_points_shift = 5

        self.phi_obs = None
        self.u_prev = ca.DM.zeros(self.controller.n_inputs, 1)

    def reset(self):
        super().reset()
        self.slack_norm = ca.inf
        self.u_prev = ca.DM.zeros(self.controller.n_inputs, 1)

    def add_trajectory(self, trajectory: Dict, iteration: int):
        super().add_trajectory(trajectory, iteration)
        # Update number of points in track_ctrl the track used for the control references and vehicle parameters
        opt_time = int(np.min([states.shape[1] for states in self.SSx.values()]))
        self.track_ctrl_flipped.update_n_points(opt_time)

    # def keep_running(self):
    #     # For some reason this LMPC become infeasible outside this range
    #     return np.abs(self.x[0, 0] - self.track_ctrl.length) > 0.45

    def get_stored_data(self):
        stored_cost_to_go = ca.DM()
        stored_states = ca.DM()

        for i in range(self.iteration):
            cost_to_go_i = self.cost_to_go[i]
            states_i = self.SSx[i]

            # Limit ranges to ensure they don't exceed size of stored data
            # it_idx_lower = min(self.compute_it_idx(i), cost_to_go_i.shape[1] - 1)
            point_idx = max(
                # If ego is further back
                np.argmin(np.abs(states_i[0, :] - self.x[0, 0])) + self.n_points_shift,
                # If onc is further back
                np.argmin(np.abs(states_i[4, :] - self.x[4, 0])) + self.n_points_shift,
            )
            it_idx_lower = max(min(point_idx, cost_to_go_i.shape[1] - 1), 0)
            it_idx_upper = min(it_idx_lower + self.n_points, cost_to_go_i.shape[1])
            print(
                "idx_lower",
                it_idx_lower,
                "idx_upper",
                it_idx_upper,
                "total",
                cost_to_go_i.shape[1],
                "cost",
                cost_to_go_i[0],
            )

            stored_cost_to_go = ca.horzcat(
                stored_cost_to_go, cost_to_go_i[:, it_idx_lower:it_idx_upper]
            )
            stored_states = ca.horzcat(
                stored_states, states_i[:, it_idx_lower:it_idx_upper]
            )

        return stored_cost_to_go, stored_states

    def step(self):
        _, curvature_ego, s_0_arc_ego, phi_0_arc_ego = sim_util.compute_ref_trajectory(
            x=self.x,
            track=self.track_ctrl,
            dt=self.model.dt,
            N=self.controller.N,
            v_ref=0.5,
            e_ref=self.model.x0[1, 0],
        )
        _, curvature_onc, s_0_arc_onc, phi_0_arc_onc = sim_util.compute_ref_trajectory(
            x=self.x,
            track=self.track_ctrl,
            dt=self.model.dt,
            N=self.controller.N,
            v_ref=0.5,
            e_ref=self.model.x0[5, 0],
            flipped_ref=True,
        )
        stored_cost_to_go, stored_states = self.get_stored_data()
        # no need to rebuild controller optimizer here with self.controller.build_optimizer()

        results = []
        results_infeasible = []
        i = 0
        for k in range(stored_cost_to_go.shape[1]):
            try:
                uopt, _, cost, slack_norm = self.controller.get_ctrl(
                    self.x,
                    curvature_ego,
                    s_0_arc_ego,
                    phi_0_arc_ego,
                    curvature_onc,
                    s_0_arc_onc,
                    phi_0_arc_onc,
                    terminal_cost=stored_cost_to_go[:, k],
                    terminal_state=stored_states[:, k],
                    s_obs=self.S_OBS,
                    s_final=self.track.length,
                    u_prev=self.u_prev,
                )
                results.append((uopt, cost, slack_norm))
                print(
                    "Solving for x-xN=",
                    stored_states[:, k] - self.x,
                    "k",
                    k,
                    "index",
                    i,
                )
                i += 1
            except:
                # print(
                #     "Unable to solve for x-xN=",
                #     stored_states[:, k] - self.x,
                #     "with k",
                #     k,
                # )
                uopt = self.controller.opti.value(self.controller.u[:, [0]])
                cost = self.controller.opti.value(self.controller.cost)
                results_infeasible.append((uopt, cost, ca.inf))

        # Find the best results
        if len(results) == 0:
            print("All solutions are infeasible")
            exit()
            results = results_infeasible

        costs = [result[1] for result in results]
        opt_idx = np.argmin(costs)
        print("Best cost", results[opt_idx][1], "index", opt_idx)

        uopt = results[opt_idx][0]
        self.u_prev = uopt
        self.slack_norm = results[opt_idx][2]

        u = ca.vertcat(
            uopt,
            curvature_ego[0],
            s_0_arc_ego[0],
            phi_0_arc_ego[0],
            curvature_onc[0],
            s_0_arc_onc[0],
            phi_0_arc_onc[0],
        )
        # State Feedback Step
        self.x = self.model.sim(1, u=u, input_noise=False, state_noise=False)

        states, inputs = self.model.get_trajectory()
        obs_states = ca.DM(states.shape[0], 1)
        obs_states[0, 0] = self.S_OBS  # set s
        if self.phi_obs is None:
            point_idx = np.argmin(np.abs(self.track.points_array[2, :] - self.S_OBS))
            self.phi_obs = self.track.points[point_idx].phi_s
        obs_states[1, 0] = self.model.x0[1, 0]
        obs_states[-1, 0] = self.phi_obs  # set heading angle
        obs_inputs = ca.DM(inputs.shape[0], 1)
        self.vehicles = [
            vis_util.VehicleData(
                "ego", vis_util.COLORS["ego"], states[:4, :], inputs[0:2, :]
            ),
            vis_util.VehicleData("obs", vis_util.COLORS["obs"], obs_states, obs_inputs),
            vis_util.VehicleData(
                "onc",
                vis_util.COLORS["onc"],
                states[4:, :],
                inputs[2:4, :],
            ),
        ]

    def post_step(self):
        super().post_step()
        print(
            "J-1 time",
            self.cost_to_go[self.iteration - 1][0, 0],
            "J time",
            self.time_step,
            "slack norm",
            self.slack_norm,
            "dist",
            self.track.length - self.x[0, 0],
            "v",
            self.x[2, 0],
            "m/s",
        )
