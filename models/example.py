from models.base import BaseModel
import casadi as ca
import numpy as np

# Example of how to use the BaseModel


class ExampleModel(BaseModel):
    NEW_PARAM = 0.1

    def __init__(self, x0=[0], dt=0.1):
        super().__init__(x0, dt)

    def __str__(self) -> str:
        with np.printoptions(precision=2, suppress=True):
            x, u = self.get_trajectory()
            out = "Trajectory of platoon"
            out += f"\nx: {x[0,:]}\n"
            if len(self.input_trajectory) > 0:
                out += f"u: {u[:,:]}"
            return out

    def _build_dae(self):
        # add states
        x = self.dae.add_x("x")
        # add inputs
        u = self.dae.add_u("u")
        # add parameters (count towards inputs)
        p = self.dae.add_p("p")
        # differential equations of model
        x_dot = u / self.NEW_PARAM + p
        # add ode to dae
        self.dae.add_ode("x_dot", x_dot)
        # set units
        self.dae.set_unit("x", "m")


if __name__ == "__main__":
    model = ExampleModel()
    model.sim(2, ca.horzcat(ca.vertcat(1, 0.6), ca.vertcat(1, 0.2)))
    print(model)
