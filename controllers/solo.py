import casadi as ca
from controllers.generic import GenericMPC
from models.solo import SoloFrenetModel


class SoloMPC(GenericMPC):
    def set_parameters(self):
        self.x_ref = self.opti.parameter(self.n_states, self.N + 1)
        self.curvature = self.opti.parameter(1, self.N)
        self.s_0_arc = self.opti.parameter(1, self.N)
        self.phi_0_arc = self.opti.parameter(1, self.N)
        return self.curvature, self.s_0_arc, self.phi_0_arc

    def set_cost(self):
        self.cost = 0
        for i in range(self.N):
            err = self.x[:, i] - self.x_ref[:, i]
            self.cost += err.T @ self.Q @ err + self.u[:, i].T @ self.R @ self.u[:, i]
        errN = self.x[:, self.N] - self.x_ref[:, self.N]
        self.cost += errN.T @ self.Q @ errN

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
