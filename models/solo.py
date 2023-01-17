from models.base import BaseModel
import numpy as np


class SoloBicycleModel(BaseModel):
    TAU = 0.1
    WB = 0.324  # [m] Wheelbase of vehicle
    INPUT_NOISE_STD = 0.05
    STATE_NOISE_STD = 0.01
    LENGTH = 0.586  # [m]
    WIDTH = 0.2485  # [m]
    BACKTOWHEEL = 0.16  # [m]

    def __init__(self, x0, dt=0.1):
        super().__init__(x0, dt)

    def __str__(self) -> str:
        with np.printoptions(precision=2, suppress=True):
            x, u = self.get_trajectory()
            out = "Trajectory of platoon"
            out += f"\nx: {x[0,:]}\n"
            if len(self.input_trajectory) > 0:
                out += f"u: {u[0,:]}"
            return out

    def _build_dae(self):
        # add states
        x = self.dae.add_x("x")
        y = self.dae.add_x("y")
        v = self.dae.add_x("v")
        phi = self.dae.add_x("phi")

        # add inputs
        v_u = self.dae.add_u("v_u")
        delta = self.dae.add_u("delta")

        x_dot = v * np.cos(phi)
        y_dot = v * np.sin(phi)
        v_dot = (v_u - v) / self.TAU
        phi_dot = v * np.tan(delta) / self.WB

        self.dae.add_ode("x_dot", x_dot)
        self.dae.add_ode("y_dot", y_dot)
        self.dae.add_ode("v_dot", v_dot)
        self.dae.add_ode("phi_dot", phi_dot)

        # set units
        self.dae.set_unit("x", "m")
        self.dae.set_unit("y", "m")
        self.dae.set_unit("v", "m/s")
        self.dae.set_unit("phi", "rad")


class SoloFrenetModel(BaseModel):
    TAU = 0.1
    WB = 0.324  # [m] Wheelbase of vehicle
    INPUT_NOISE_STD = 0.05
    STATE_NOISE_STD = 0.01
    LENGTH = 0.586  # [m]
    WIDTH = 0.2485  # [m]
    BACKTOWHEEL = 0.16  # [m]

    def __init__(self, x0, dt=0.1):
        super().__init__(x0, dt)

    def __str__(self) -> str:
        with np.printoptions(precision=2, suppress=True):
            x, u = self.get_trajectory()
            out = "Trajectory of platoon"
            out += f"\nx: {x[0,:]}\n"
            if len(self.input_trajectory) > 0:
                out += f"u: {u[0,:]}"
            return out

    def _build_dae(self):
        # add states
        s = self.dae.add_x("s")
        e = self.dae.add_x("e")
        v = self.dae.add_x("v")
        phi = self.dae.add_x("phi")

        # add inputs
        v_u = self.dae.add_u("v_u")
        delta = self.dae.add_u("delta")

        # add parameters (count towards inputs)
        curvature = self.dae.add_p("curvature")
        s_0_arc = self.dae.add_p("s_0_arc")
        phi_0_arc = self.dae.add_p("phi_0_arc")

        # differential equations of model
        phi_s = curvature * (s - s_0_arc) + phi_0_arc
        # phi_s = np.arctan2(np.sin(phi_s), np.cos(phi_s))
        # phi_con = np.arctan2(np.sin(phi), np.cos(phi))
        dphi = phi - phi_s
        # dphi = np.arctan2(np.sin(dphi), np.cos(dphi))

        # dphi = np.pi - norm_2(norm_2(phi_con - phi_s) - np.pi)

        s_dot = v * np.cos(dphi) / (1 - curvature * e)
        e_dot = v * np.sin(dphi)
        v_dot = (v_u - v) / self.TAU
        phi_dot = v * np.tan(delta) / self.WB

        self.dae.add_ode("s_dot", s_dot)
        self.dae.add_ode("e_dot", e_dot)
        self.dae.add_ode("v_dot", v_dot)
        self.dae.add_ode("phi_dot", phi_dot)

        # set units
        self.dae.set_unit("s", "m")
        self.dae.set_unit("e", "m")
        self.dae.set_unit("v", "m/s")
        self.dae.set_unit("phi", "rad")


if __name__ == "__main__":
    x0 = [0, 0, 0, np.pi]
    model = SoloBicycleModel(x0)
    t = 20
    u = np.zeros((2, t))
    u[0, :] = 0.5
    u[1, :] = np.pi / 4

    model.sim(t, u)
    x, u = model.get_trajectory()
    print(x, u)
