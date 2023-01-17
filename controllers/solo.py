import casadi as ca
from controllers.base import BaseMPC


class SoloMPC(BaseMPC):
    # Uses the SoloFrenetModel
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


class SoloRelaxedLMPC(BaseMPC):
    # Uses the SoloFrenetModel
    LAMBDA_SIZE = 10

    def set_variables(self):
        self.x = self.opti.variable(self.n_states, self.N + 1)
        self.u = self.opti.variable(
            self.n_inputs, self.N)  # For multi-shooting
        self.lam = self.opti.variable(1, self.LAMBDA_SIZE)

    def set_parameters(self):
        self.curvature = self.opti.parameter(1, self.N)
        self.s_0_arc = self.opti.parameter(1, self.N)
        self.phi_0_arc = self.opti.parameter(1, self.N)
        self.terminal_cost = self.opti.parameter(1, 1)
        # self.terminal_state = self.opti.parameter(self.n_states, 1)
        self.stored_cost_to_go = self.opti.parameter(1, self.LAMBDA_SIZE)
        # self.terminal_state = self.opti.parameter(self.n_states, 1)
        self.stored_states = self.opti.parameter(
            self.n_states, self.LAMBDA_SIZE)
        return self.curvature, self.s_0_arc, self.phi_0_arc

    def set_cost(self):
        self.cost = self.N + self.lam @ self.stored_cost_to_go.T

    def set_nonlinear_constraints(self):
        # All lambda must be greater or equal to zero
        self.opti.subject_to(0 <= self.lam)
        # The convex hulls parameters must add up to
        self.opti.subject_to(ca.sum2(self.lam) == 1)
        # The last state must be in the convex hull of the stored states
        self.opti.subject_to(self.x[:, self.N] ==
                             self.stored_states @ self.lam.T)

    def get_ctrl(
        self, x0, curvature, s_0_arc, phi_0_arc, stored_cost_to_go, stored_states
    ):
        self.opti.set_value(self.x0, x0)
        self.opti.set_value(self.curvature, curvature)
        self.opti.set_value(self.s_0_arc, s_0_arc)
        self.opti.set_value(self.phi_0_arc, phi_0_arc)
        self.opti.set_value(self.stored_cost_to_go, stored_cost_to_go)
        self.opti.set_value(self.stored_states, stored_states)
        return self.solve()
