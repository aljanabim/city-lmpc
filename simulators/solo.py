import casadi as ca
from utils import sim_util, vis_util
from models.solo import SoloFrenetModel
from controllers.solo import SoloMPC, SoloRelaxedLMPC
from models.track import Track
from simulators.base import BaseSimulator, BaseLMPCSimulator


class SoloMPCSimulator(BaseSimulator):  # Using a Frenet Model
    EXP_NAME = "solo"

    def __init__(self, model: SoloFrenetModel, controller: SoloMPC, track: Track):
        # Reassigned to make sure that the types are correct and not Any as in the BaseSimulator class
        self.model = model
        self.controller = controller
        self.track = track
        super().__init__(model, controller, track)

    def keep_running(self):
        return self.x[0, 0] < self.track.length

    def step(self):
        x_ref, curvature, s_0_arc, phi_0_arc = sim_util.compute_ref_trajectory(
            x=self.x, track=self.track, dt=self.model.dt, N=self.controller.N, v_ref=0.5, e_ref=self.model.x0[1, 0]
        )
        uopt, xopt = self.controller.get_ctrl(
            self.x, x_ref, curvature, s_0_arc, phi_0_arc)
        u = ca.vertcat(uopt, curvature[0], s_0_arc[0], phi_0_arc[0])
        # State Feedback Step
        self.x = self.model.sim(1, u=u, input_noise=False, state_noise=False)

        states, inputs = self.model.get_trajectory()
        self.vehicles = [vis_util.VehicleData(
            "car", vis_util.COLORS["ego"], states, inputs)]


class SoloLMPCSimulator(BaseLMPCSimulator):  # Using a Frenet Model
    EXP_NAME = "solo"

    def __init__(self, model: SoloFrenetModel, controller: SoloRelaxedLMPC, track: Track, iteration: int):
        # Reassigned to make sure that the types are correct and not Any as in the BaseSimulator class
        self.model = model
        self.controller = controller
        self.track = track
        self.iteration = iteration
        super().__init__(model, controller, track)

    def keep_running(self):
        return self.x[0, 0] < self.track.length

    def get_stored_cost_to_go(self):
        return ca.DM()

    def get_stored_states(self):
        return ca.DM()

    def step(self):
        x_ref, curvature, s_0_arc, phi_0_arc = sim_util.compute_ref_trajectory(
            x=self.x, track=self.track, dt=self.model.dt, N=self.controller.N, v_ref=0.5, e_ref=self.model.x0[1, 0]
        )
        stored_cost_to_go = self.get_stored_cost_to_go()
        stored_states = self.get_stored_states()
        uopt, xopt = self.controller.get_ctrl(
            self.x, curvature, s_0_arc, phi_0_arc, stored_cost_to_go, stored_states)
        u = ca.vertcat(uopt, curvature[0], s_0_arc[0], phi_0_arc[0])
        # State Feedback Step
        self.x = self.model.sim(1, u=u, input_noise=False, state_noise=False)

        states, inputs = self.model.get_trajectory()
        self.vehicles = [vis_util.VehicleData(
            "car", vis_util.COLORS["ego"], states, inputs)]
