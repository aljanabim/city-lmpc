from models.base import BaseModel
import numpy as np


class DuoFrenetModel(BaseModel):
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
        for label in ["ego", "onc"]:
            # add states
            s = self.dae.add_x(f"s_{label}")
            e = self.dae.add_x(f"e_{label}")
            v = self.dae.add_x(f"v_{label}")
            phi = self.dae.add_x(f"phi_{label}")

            # add inputs
            v_u = self.dae.add_u(f"v_u_{label}")
            delta = self.dae.add_u(f"delta_{label}")

            # add parameters (count towards inputs)
            curvature = self.dae.add_p(f"curvature_{label}")
            s_0_arc = self.dae.add_p(f"s_0_arc_{label}")
            phi_0_arc = self.dae.add_p(f"phi_0_arc_{label}")

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

            self.dae.add_ode(f"s_dot_{label}", s_dot)
            self.dae.add_ode(f"e_dot_{label}", e_dot)
            self.dae.add_ode(f"v_dot_{label}", v_dot)
            self.dae.add_ode(f"phi_dot_{label}", phi_dot)

            # set units
            self.dae.set_unit(f"s_{label}", "m")
            self.dae.set_unit(f"e_{label}", "m")
            self.dae.set_unit(f"v_{label}", "m/s")
            self.dae.set_unit(f"phi_{label}", "rad")


if __name__ == "__main__":
    pass
    # x0 = [0, 0, 0, np.pi]
    # model = SoloBicycleModel(x0)
    # t = 20
    # u = np.zeros((2, t))
    # u[0, :] = 0.5
    # u[1, :] = np.pi / 4

    # model.sim(t, u)
    # x, u = model.get_trajectory()
    # print(x, u)
