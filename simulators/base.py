from typing import Tuple
from utils import sim_util, vis_util
from models.base import BaseModel
from controllers.base import BaseMPC
from models.track import Track


class BaseSimulator:
    EXP_NAME = "base"

    # def __init__(self, model: BaseModel, controller: BaseMPC, track: Track):
    def __init__(self, model, controller, track: Track):
        self.model = model
        self.controller = controller
        self.track = track

        self.x = self.model.x0
        self.exp_meta = sim_util.ExpMeta(self.EXP_NAME, 0, self.x[0, 0])
        self.vehicles: Tuple[vis_util.VehicleData] = None
        self.time_step = 0

    def keep_running(self):
        """
        Stop condition for the simulator. Returns False, when the simulator should stop.
        """
        return False

    def step(self):
        """
        The most critical method in the simulator, it's unique for each type of simulator that inherits from
        the BaseSimulator class.

        It must include:
        1. Update self.x
        2. Update self.vehicles (using updated trajectories)
        """
        pass

    def pre_step(self):
        """
        Sub-routines executed before the simulator does a step
        """
        pass

    def post_step(self):
        """
        Sub-routines executed after the simulator does a step
        """
        self.time_step += 1

    def pre_run(self):
        """
        Sub-routines executed before the simulator starts running
        """
        pass

    def post_run(self):
        """
        Sub-routines executed after the simulator starts running
        """
        pass

    def run(self, plot=True):
        self.pre_run()
        print("======= Running", self.__class__.__name__, "=======")
        while self.keep_running():
            self.pre_step()
            self.step()
            self.post_step()
            if plot:
                vis_util.plot_trajectory(
                    self.track, self.vehicles, plot_input=True)
        self.post_run()
        print("=======", self.__class__.__name__, "Finished =======")

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


class BaseLMPCSimulator(BaseSimulator):
    def __init__(self, model, controller, track: Track, iteration: int):
        self.iteration = iteration
        super().__init__(model, controller, track)

    def keep_iterating(self):
        return False

    def pre_iteration(self):
        pass

    def post_iteration(self):
        pass

    def run(self, plot=True):
        while self.keep_iterating():
            print("Starting iteration", self.iteration)
            super().run(plot)
            print("Finished iteration", self.iteration)
