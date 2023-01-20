import casadi as ca
from utils import sim_util, vis_util
from models.solo import SoloFrenetModel
from controllers.solo import SoloMPC, SoloRelaxedLMPC
from models.track import Track
from simulators.base import BaseSimulator, BaseLMPCSimulator


class SoloMPCSimulator(BaseSimulator):  # Using a Frenet Model
    def __init__(self, model: SoloFrenetModel, controller: SoloMPC, track):
        # Override the types for model and controller (Any by default)
        self.model = model
        self.controller = controller
        super().__init__(model, controller, track)

    def step(self):
        x_ref, curvature, s_0_arc, phi_0_arc = sim_util.compute_ref_trajectory(
            x=self.x,
            track=self.track,
            dt=self.model.dt,
            N=self.controller.N,
            v_ref=0.5,
            e_ref=self.model.x0[1, 0],
        )
        uopt, _, _ = self.controller.get_ctrl(
            self.x, x_ref, curvature, s_0_arc, phi_0_arc
        )
        u = ca.vertcat(uopt, curvature[0], s_0_arc[0], phi_0_arc[0])
        # State Feedback Step
        self.x = self.model.sim(1, u=u, input_noise=False, state_noise=False)

        states, inputs = self.model.get_trajectory()
        self.vehicles = [
            vis_util.VehicleData("car", vis_util.COLORS["ego"], states, inputs)
        ]


class SoloRelaxedLMPCSimulator(BaseLMPCSimulator):  # Using a Frenet Model
    def __init__(
        self,
        model: SoloFrenetModel,
        controller: SoloRelaxedLMPC,
        track,
        trajectories,
        max_iter,
    ):
        # Override the types for model and controller (Any by default)
        self.model = model
        self.controller = controller
        super().__init__(model, controller, track, trajectories, max_iter)

    def reset(self):
        super().reset()
        self.slack_norm = ca.inf

    def compute_it_idx(self, iteration):
        T_opt = min(self.T.values())
        # int to ensure it can be used as index
        return int(min(self.T[iteration] + self.time_step - T_opt, self.T[iteration]))

    def get_stored_data(self):
        stored_cost_to_go = ca.DM()
        stored_states = ca.DM()
        for j, (cost_to_go_j, states_j) in enumerate(
            zip(self.cost_to_go.values(), self.SSx.values())
        ):
            it_idx = self.compute_it_idx(j)
            print("iteration", j, "k", it_idx)
            stored_cost_to_go = ca.horzcat(stored_cost_to_go, cost_to_go_j[:, it_idx:])
            stored_states = ca.horzcat(stored_states, states_j[:, it_idx:])
        return stored_cost_to_go, stored_states

    def keep_running(self):
        return self.slack_norm > 1e-8

    def step(self):
        _, curvature, s_0_arc, phi_0_arc = sim_util.compute_ref_trajectory(
            x=self.x,
            track=self.track,
            dt=self.model.dt,
            N=self.controller.N,
            v_ref=0.5,
            e_ref=self.model.x0[1, 0],
        )
        stored_cost_to_go, stored_states = self.get_stored_data()

        # rebuild controller optimizer with new LAMBDA_SIZE
        self.controller.LAMBDA_SIZE = stored_cost_to_go.shape[1]
        self.controller.build_optimizer()

        uopt, _, _, self.slack_norm = self.controller.get_ctrl(
            self.x, curvature, s_0_arc, phi_0_arc, stored_cost_to_go, stored_states
        )
        u = ca.vertcat(uopt, curvature[0], s_0_arc[0], phi_0_arc[0])
        # State Feedback Step
        self.x = self.model.sim(1, u=u, input_noise=False, state_noise=False)

        states, inputs = self.model.get_trajectory()
        self.vehicles = [
            vis_util.VehicleData("car", vis_util.COLORS["ego"], states, inputs)
        ]

    def post_step(self):
        super().post_step()
        print(
            "J-1 time",
            self.cost_to_go[self.iteration - 1][0, 0],
            "J time",
            self.time_step,
            "LAMBDA_SIZE",
            self.controller.LAMBDA_SIZE,
            "slack norm",
            self.slack_norm,
            "dist",
            self.track.length - self.x[0, 0],
        )
