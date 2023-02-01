import numpy as np
import casadi as ca
from utils import sim_util, vis_util
from models.solo import SoloFrenetModel
from controllers.solo import SoloMPC, SoloRelaxedLMPC
from models.track import Track
from simulators.base import BaseSimulator, BaseLMPCSimulator
from controllers.obstacle_old import ObstacleLMPC


class ObstacleMPCSimulator(BaseSimulator):  # Using a Frenet Model
    def __init__(self, model: SoloFrenetModel, controller: SoloMPC, track, track_ctrl):
        # Override the types for model and controller (Any by default)
        self.model = model
        self.controller = controller
        super().__init__(model, controller, track, track_ctrl)
        self.phi_obs = None
        self.u_prev = ca.DM.zeros(self.controller.n_inputs, 1)

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
        if rectellipse < 1:
            print(
                "OOOOopps rectellipse constraint not satisfied",
                rectellipse,
                rectellipse >= 1,
            )
            exit()

        print(self.x[2, 0], "m/s")


class ObstacleLMPCSimulator(BaseLMPCSimulator):  # Using a Frenet Model
    def __init__(
        self,
        model: SoloFrenetModel,
        controller: ObstacleLMPC,
        track,
        track_ctrl,
        trajectories,
        max_iter,
    ):
        # Override the types for model and controller (Any by default)
        self.model = model
        self.controller = controller
        super().__init__(model, controller, track, track_ctrl, trajectories, max_iter)
        self.n_points = 20
        self.n_points_shift = 5
        # self.n_included_iterations = self.max_iter
        self.phi_obs = None
        self.u_prev = ca.DM.zeros(self.controller.n_inputs, 1)

    def reset(self):
        super().reset()
        self.slack_norm = ca.inf

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
            it_idx_lower = min(point_idx, cost_to_go_i.shape[1] - 1)
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
        self.vehicles = [
            vis_util.VehicleData("ego", vis_util.COLORS["ego"], states, inputs),
            vis_util.VehicleData("obs", vis_util.COLORS["obs"], obs_states, obs_inputs),
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
