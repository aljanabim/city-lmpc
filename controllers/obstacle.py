import casadi as ca
from controllers.base import BaseMPC

# For iteration 0 we use SoloMPC in this scenario too


class ObstacleLMPC(BaseMPC):
    # Uses the SoloFrenetModel
    LAMBDA_SIZE = 10

    def set_variables(self):
        self.x = self.opti.variable(self.n_states, self.N + 1)
        self.u = self.opti.variable(self.n_inputs, self.N)  # For multi-shooting
        self.lam = self.opti.variable(1, self.LAMBDA_SIZE)
        self.lam_slack = self.opti.variable(self.n_states, 1)

    def set_parameters(self):
        self.curvature = self.opti.parameter(1, self.N)
        self.s_0_arc = self.opti.parameter(1, self.N)
        self.phi_0_arc = self.opti.parameter(1, self.N)
        # self.terminal_cost = self.opti.parameter(1, 1)
        # self.terminal_state = self.opti.parameter(self.n_states, 1)
        self.stored_cost_to_go = self.opti.parameter(1, self.LAMBDA_SIZE)
        self.stored_states = self.opti.parameter(self.n_states, self.LAMBDA_SIZE)
        return self.curvature, self.s_0_arc, self.phi_0_arc

    def set_cost(self):
        self.cost = (
            self.N
            + self.lam @ self.stored_cost_to_go.T
            + 1e2 * (self.lam_slack.T @ self.lam_slack)
        )

    def set_linear_constraints(self):
        self.opti.subject_to(self.opti.bounded(self.ulb, self.u, self.uub))
        self.opti.subject_to(self.opti.bounded(self.xlb, self.x, self.xub))

    def set_nonlinear_constraints(self):
        # All lambda must be greater or equal to zero
        self.opti.subject_to(0 <= self.lam)
        # The convex hulls parameters must add up to
        self.opti.subject_to(ca.sum2(self.lam) == 1)
        # The last state must be in the convex hull of the stored states
        self.opti.subject_to(
            self.lam_slack + self.x[:, self.N] == self.stored_states @ self.lam.T
        )

    def get_ctrl(
        self, x0, curvature, s_0_arc, phi_0_arc, stored_cost_to_go, stored_states
    ):
        self.opti.set_value(self.x0, x0)
        self.opti.set_value(self.curvature, curvature)
        self.opti.set_value(self.s_0_arc, s_0_arc)
        self.opti.set_value(self.phi_0_arc, phi_0_arc)
        self.opti.set_value(self.stored_cost_to_go, stored_cost_to_go)
        self.opti.set_value(self.stored_states, stored_states)
        u_pred, x_pred, cost = self.solve()
        lambda_slack = self.opti.value(self.lam_slack)
        slack_norm = lambda_slack.T @ lambda_slack
        return u_pred, x_pred, cost, slack_norm
