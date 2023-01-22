import casadi as ca
import numpy as np
from typing import Tuple, Dict
from utils import sim_util, vis_util
from models.base import BaseModel
from controllers.base import BaseMPC
from models.track import Track


class BaseSimulator:
    _EXP_NAME = "base"
    _S_OBS = 0.0

    def __init__(self, model: BaseModel, controller: BaseMPC, track: Track):
        self.model = model
        self.controller = controller
        self.track = track

        self.reset()
        self.exp_meta = sim_util.ExpMeta(self._EXP_NAME, self._S_OBS, self.x[0, 0])
        self.vehicles: Tuple[vis_util.VehicleData] = None

    @property
    def EXP_NAME(self):
        return self._EXP_NAME

    @EXP_NAME.setter
    def EXP_NAME(self, name: str):
        self._EXP_NAME = name
        self.exp_meta = sim_util.ExpMeta(self._EXP_NAME, self._S_OBS, self.x[0, 0])

    @property
    def S_OBS(self):
        return self._S_OBS

    @S_OBS.setter
    def S_OBS(self, s_obs: float):
        self._S_OBS = s_obs
        self.exp_meta = sim_util.ExpMeta(self._EXP_NAME, self._S_OBS, self.x[0, 0])

    def reset(self):
        """
        Reset model and self.x
        """
        self.model.reset_trajectory()
        self.x = self.model.x0
        self.time_step = 0

    def keep_running(self):
        """
        Stop condition for the simulator. Returns False, when the simulator should stop.
        """
        return self.x[0, 0] <= self.track.length

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
        Must include increment of time step number, self.time_step, by 1
        """
        self.time_step += 1

    # def pre_run(self):
    #     """
    #     Sub-routines executed before the simulator starts running
    #     """
    #     pass

    # def post_run(self):
    #     """
    #     Sub-routines executed after the simulator starts running
    #     """
    # pass

    def run(self, plot=True):
        print("======= Running", self.__class__.__name__, "=======")
        while self.keep_running():
            self.pre_step()
            self.step()
            self.post_step()
            if plot:
                vis_util.plot_trajectory(self.track, self.vehicles, plot_input=True)
        print("=======", self.__class__.__name__, "Finished =======")
        return self

    def save(self, iteration=0):
        states, inputs = self.model.get_trajectory()
        sim_util.dump_trajectory(self.exp_meta, f"J{iteration}", states, inputs)
        return self

    def load(self, iteration=0):
        """Load trajectory from dumps based on the iteration number. The file loaded must be J{iteration}.npx
        Args:
            iteration (int): Iteration number of the file to be loaded

        Returns:
            traj (dict of states and inputs or None): Loaded states and inputs in a dictionary, None if file not found.
        """
        traj = sim_util.load_trajectory(self.exp_meta, f"J{iteration}")
        return traj

    def animate(self, iteration=0, save=False):
        vis_util.animate_trajectory(
            self.exp_meta,
            self.track,
            self.vehicles,
            animation_filename=f"{self.EXP_NAME}_J{iteration}",
            save=save,
        )


class BaseLMPCSimulator(BaseSimulator):
    """
    Adds iteration functionality to the BaseSimulator in the run method
    """

    def __init__(
        self,
        model,
        controller,
        track: Track,
        trajectories: Tuple[Dict],
        max_iter: int,
    ):
        super().__init__(model, controller, track)
        self.trajectories = trajectories
        self.max_iter = max_iter
        self._parse_trajectories()

    def _parse_trajectories(self):
        self.SSx = {}
        self.SSu = {}
        self.cost_to_go = {}
        self.T = {}  # maps iteration number -> iteration length

        for j, traj_j in enumerate(self.trajectories):
            self.add_trajectory(traj_j, j)
        self.iteration = j + 1

    def add_trajectory(self, trajectory: Dict, iteration: int):
        """
        Takes a trajectory and an iteration number and updates the following instance attributes
        SSx[iteration] = states
        SSu[iteration] = inputs
        cost_to_go[iteration] = [T_iteration, T_iteration - 1, ..., 0]
        self.T[iteration] = T_iteration

        Args:
            trajectory (dict of "states" and "inputs"): Dictionary of stored states and inputs for a certain iteration
            iteration (int): Number of the iteration
        """

        self.SSx[iteration] = trajectory["states"]
        self.SSu[iteration] = trajectory["inputs"]

        total_time = trajectory["states"].shape[1]
        self.cost_to_go[iteration] = ca.DM.zeros(1, total_time)
        cost = 0
        for t in range(total_time - 1, -1, -1):
            s = self.SSx[iteration][0, t]
            if s < self.track.length:
                cost += 1
            self.cost_to_go[iteration][0, t] = cost

        # if trajectories are guaranteed to stop before hitting track length
        # can replace the for loop above with the following
        # T_iteration = trajectory["states"].shape[1] - 1
        # self.cost_to_go[iteration] = np.arange(T_iteration, -1, -1)[np.newaxis, :]

        self.T[iteration] = self.cost_to_go[iteration][0, 0]

    def keep_iterating(self):
        return self.iteration <= self.max_iter

    def pre_iteration(self):
        """
        Sub-routines executed before the simulator starts an iteration
        """
        pass

    def post_iteration(self):
        """
        Sub-routines executed after the simulator starts an iteration
        Must include increment of iteration number, self.iteration, by 1
        """
        self.iteration += 1

    def run(self, plot=True, save=True):
        while self.keep_iterating():
            self.pre_iteration()
            print("Starting iteration", self.iteration)
            super().run(plot)
            print("Finished iteration", self.iteration)
            states, inputs = self.model.get_trajectory()
            trajectory = {"states": states, "inputs": inputs}
            self.add_trajectory(trajectory, self.iteration)
            if save:
                self.save(self.iteration)
            self.post_iteration()
            self.reset()
