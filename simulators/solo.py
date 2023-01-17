import casadi as ca
from utils import sim_util, vis_util
from models.solo import SoloFrenetModel
from controllers.solo import SoloMPC
from models.track import Track


class SoloMPCSimulator:  # Using a Frenet Model
    EXP_NAME = "solo"

    def __init__(self, model: SoloFrenetModel = None, controller: SoloMPC = None, track: Track = None):
        self.model = model
        if model is not None:
            self.x = self.model.x0
            self.exp_meta = sim_util.ExpMeta(self.EXP_NAME, 0, self.x[0, 0])
        else:
            self.x = ca.DM()
            self.exp_meta = sim_util.ExpMeta(self.EXP_NAME, 0, 0)
        self.controller = controller
        self.track = track

    def check_stop(self):
        assert self.track is not None, "Track must be set"
        return self.x[0, 0] > self.track.length

    def step(self):
        assert self.model is not None, "Model must be set"
        assert self.controller is not None, "Controller must be set"
        assert self.track is not None, "Track must be set"

        x_ref, curvature, s_0_arc, phi_0_arc = sim_util.compute_ref_trajectory(
            x=self.x, track=self.track, dt=self.model.dt, N=self.controller.N, v_ref=0.5, e_ref=self.model.x0[1, 0]
        )
        uopt, xopt = self.controller.get_ctrl(
            self.x, x_ref, curvature, s_0_arc, phi_0_arc)
        u = ca.vertcat(uopt, curvature[0], s_0_arc[0], phi_0_arc[0])
        # State Feedback Step
        self.x = self.model.sim(1, u=u, input_noise=False, state_noise=False)

    def run(self):
        assert self.model is not None, "Model must be set"
        assert self.controller is not None, "Controller must be set"
        assert self.track is not None, "Track must be set"

        while not self.check_stop():
            self.step()
            states, inputs = self.model.get_trajectory()
            vehicles = [vis_util.VehicleData(
                "car", vis_util.COLORS["ego"], states, inputs)]
            vis_util.plot_trajectory(self.track, vehicles, plot_input=True)

    def save(self, it=0):
        assert self.model is not None, "Model must be set"
        states, inputs = self.model.get_trajectory()
        sim_util.dump_trajectory(self.exp_meta, f"J{it}", states, inputs)

    def load(self, it=0):
        """Load trajectory from dumps based on the iteration number, it. The file loaded must be J{it}.npx
        Args:
            it (int): Iteration number of the file to be loaded

        Returns:
            traj (dict of states and inputs or None): Loaded states and inputs in a dictionary, None if file not found.
        """
        assert self.track is not None, "Track must be set"
        traj = sim_util.load_trajectory(self.exp_meta, f"J{it}")
        return traj

    def animate(self, it=0, save=False, from_dump=False):
        if from_dump:
            traj = self.load(it)
            states = traj["states"]
            inputs = traj["inputs"]
        else:
            states, inputs = self.model.get_trajectory()

        vis_util.animate_trajectory(
            self.exp_meta,
            self.track,
            [vis_util.VehicleData("car", vis_util.COLORS["ego"],
                                  states, inputs)],
            animation_filename=f"{self.EXP_NAME}_J0",
            save=save
        )
