from typing import Tuple
import casadi as ca
from utils import sim_util, vis_util
from models.solo import SoloFrenetModel
from controllers.solo import SoloMPC
from models.track import Track


class BaseSimulator:
    EXP_NAME = "base"

    def __init__(self, model, controller, track):
        self.model = model
        self.controller = controller
        self.track = track

        self.x = self.model.x0
        self.exp_meta = sim_util.ExpMeta(self.EXP_NAME, 0, self.x[0, 0])
        self.vehicles: Tuple[vis_util.VehicleData] = None

    def keep_alive(self):
        """
        Stop condition for the simulator. Returns False, when the simulator should stop.
        """
        return True

    def step(self):
        """
        The most critical method in the simulator, it's unique for each type of simulator that inherits from
        the BaseSimulator class.

        It must include:
        1. Update self.x
        2. Update self.vehicles (using updated trajectories)
        """
        pass

    def pre_run(self):
        pass

    def post_run(self):
        pass

    def run(self, plot=True):
        self.pre_run()
        while self.keep_alive():
            self.step()
            if plot:
                vis_util.plot_trajectory(
                    self.track, self.vehicles, plot_input=True)
        self.post_run()

    def save(self, it=0):
        states, inputs = self.model.get_trajectory()
        sim_util.dump_trajectory(self.exp_meta, f"J{it}", states, inputs)

    def load(self, it=0):
        """Load trajectory from dumps based on the iteration number, it. The file loaded must be J{it}.npx
        Args:
            it (int): Iteration number of the file to be loaded

        Returns:
            traj (dict of states and inputs or None): Loaded states and inputs in a dictionary, None if file not found.
        """
        traj = sim_util.load_trajectory(self.exp_meta, f"J{it}")
        return traj

    def animate(self, it=0, save=False):
        vis_util.animate_trajectory(
            self.exp_meta,
            self.track,
            self.vehicles,
            animation_filename=f"{self.EXP_NAME}_J{it}",
            save=save
        )
