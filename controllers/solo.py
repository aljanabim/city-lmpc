import casadi as ca
from controllers.base import BaseMPC
from models.solo import SoloFrenetModel


class SoloMPC(BaseMPC):
    def set_parameters(self):
        self.x_ref = self.opti.parameter(self.n_states, self.N + 1)
        self.curvature = self.opti.parameter(1, self.N)
        self.s_0_arc = self.opti.parameter(1, self.N)
        self.phi_0_arc = self.opti.parameter(1, self.N)
        return self.curvature, self.s_0_arc, self.phi_0_arc

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

    def get_ctrl(self, x0, x_ref, curvature, s_0_arc, phi_0_arc):
        self.opti.set_value(self.x0, x0)
        self.opti.set_value(self.x_ref, x_ref)
        self.opti.set_value(self.curvature, curvature)
        self.opti.set_value(self.s_0_arc, s_0_arc)
        self.opti.set_value(self.phi_0_arc, phi_0_arc)
        return self.solve()


class SoloLMPC(SoloMPC):
    def set_parameters(self):
        self.curvature = self.opti.parameter(1, self.N)
        self.s_0_arc = self.opti.parameter(1, self.N)
        self.phi_0_arc = self.opti.parameter(1, self.N)
        self.terminal_cost = self.opti.parameter(1, 1)
        self.terminal_state = self.opti.parameter(self.n_states, 1)
        return self.curvature, self.s_0_arc, self.phi_0_arc

    def set_cost(self):
        self.cost = self.N + self.terminal_cost

    def set_nonlinear_constraints(self):
        self.opti.subject_to(self.x[:, self.N] == self.terminal_state)

    def get_ctrl(
        self, x0, curvature, s_0_arc, phi_0_arc, terminal_cost, terminal_state
    ):
        self.opti.set_value(self.x0, x0)
        self.opti.set_value(self.curvature, curvature)
        self.opti.set_value(self.s_0_arc, s_0_arc)
        self.opti.set_value(self.phi_0_arc, phi_0_arc)
        self.opti.set_value(self.terminal_cost, terminal_cost)
        self.opti.set_value(self.terminal_state, terminal_state)
        return self.solve()
