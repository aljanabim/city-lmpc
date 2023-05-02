import numpy as np
import casadi as ca
from utils import sim_util, vis_util
from models.solo import SoloFrenetModel, SoloBicycleModel
from controllers.solo import SoloMPC, SoloRelaxedLMPC
from controllers.bicycle import SoloBicycleMPC
from models.track import Track
from simulators.base import BaseSimulator, BaseLMPCSimulator
from matplotlib import pyplot as plt


class SoloMPCSimulator(BaseSimulator):  # Using a Bicycle Model
    def __init__(
        self, model: SoloBicycleModel, controller: SoloBicycleModel, track, track_ctrl
    ):
        # Override the types for model and controller (Any by default)
        self.model = model
        self.controller = controller
        super().__init__(model, controller, track, track_ctrl)
        self.u_prev = ca.DM.zeros(self.controller.n_inputs, 1)
        self.next_waypoint = False

    def get_ref(self, x):
        waypoint_1 = [1, 3]
        waypoint_2 = [10, 10]
        dist = ((x[0, 0] - waypoint_1[0]) ** 2 + (x[1, 0] - waypoint_1[1]) ** 2) ** (
            1 / 2
        )
        print(dist)
        if dist < 1e-1:
            self.next_waypoint=True
        if self.next_waypoint:
            return waypoint_2
        else:
            return waypoint_1

    def step(self):
        self.way_point = self.get_ref(self.x)
        x_ref = ca.DM.zeros(self.controller.n_states, self.controller.N + 1)
        x_ref[0, :] = self.way_point[0]
        x_ref[1, :] = self.way_point[1]

        uopt, _, _ = self.controller.get_ctrl(self.x, x_ref)
        u = ca.vertcat(uopt)
        # State Feedback Step
        self.x = self.model.sim(1, u=u, input_noise=False, state_noise=False)

        states, inputs = self.model.get_trajectory()
        self.vehicles = [
            vis_util.VehicleData("car", vis_util.COLORS["ego"], states, inputs)
        ]

    def run(self, plot=True):
        print("======= Running", self.__class__.__name__, "=======")
        while self.keep_running():
            self.pre_step()
            self.step()
            self.post_step()
            if plot:
                vis_util.plot_trajectory(
                    self.track,
                    self.vehicles,
                    plot_input=True,
                    use_bicycle=True,
                    show=False,
                )
                plt.subplot(2, 1, 1)
                plt.plot(self.way_point[0], self.way_point[1], "r*")
                plt.pause(0.01)
        print("=======", self.__class__.__name__, "Finished =======")
        return self
