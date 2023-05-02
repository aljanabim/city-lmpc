import casadi as ca
from controllers.base import BaseMPC


class SoloBicycleMPC(BaseMPC):
    # Uses the SoloFrenetModel
    def set_parameters(self):
        self.x_ref = self.opti.parameter(self.n_states, self.N + 1)

    def set_cost(self):
        self.cost = 0

        n_sub = self.n_states - 1  # number of states ignore the yaw
        for i in range(self.N):
            err = self.x[:n_sub, i] - self.x_ref[:n_sub, i]
            self.cost += (
                err.T @ self.Q[:n_sub, :n_sub] @ err
                + self.u[:, i].T @ self.R @ self.u[:, i]
            )
            # Calculate the angle difference towards the smallest angle error
            # from https://math.stackexchange.com/questions/341749/how-to-get-the-minimum-angle-between-two-crossing-lines
            angle_err = ca.pi - ca.norm_2(
                ca.norm_2(self.x[-1, i] - self.x_ref[-1, i]) - ca.pi
            )
            self.cost += angle_err**2 * self.Q[-1, -1]

        errN = self.x[:n_sub, self.N] - self.x_ref[:n_sub, self.N]
        self.cost += errN.T @ self.Q[:n_sub, :n_sub] @ errN
        angle_err = ca.pi - ca.norm_2(
            ca.norm_2(self.x[-1, self.N] - self.x_ref[-1, self.N]) - ca.pi
        )
        self.cost += angle_err**2 * self.Q[-1, -1]

    def set_linear_constraints(self):
        # Set control rate constraints
        # uub_rate = self.model.dt * ca.DM([1, 0.7] * int(self.n_inputs / 2)).reshape(
        #     (self.n_inputs, 1)
        # )
        # u_rate = ca.sqrt((self.u[:, :-1] - self.u[:, 1:]) ** 2)
        # u_rate = ca.horzcat(ca.sqrt((self.u_prev - self.u[:, 0]) ** 2), u_rate)
        # self.opti.subject_to(self.opti.bounded(-uub_rate, u_rate, uub_rate))
        # Set control and state constraints
        return super().set_linear_constraints()

    def get_ctrl(self, x0, x_ref):
        self.opti.set_value(self.x0, x0)
        self.opti.set_value(self.x_ref, x_ref)
        return self.solve()
