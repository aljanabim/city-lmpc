import casadi as ca
from controllers.base import BaseMPC

# For iteration 0 we use SoloMPC in this scenario too


class ObstacleMPC(BaseMPC):
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
                err.T
                @ self.Q[:n_sub, :n_sub]
                @ err
                # + self.u[:, i].T @ self.R @ self.u[:, i]
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

        if self.R is not None:
            for t in range(self.N - 1):
                u_diff = self.u[:, t + 1] - self.u[:, t]
                self.cost += u_diff.T @ self.R @ u_diff

    def get_ctrl(self, x0, x_ref, curvature, s_0_arc, phi_0_arc):
        self.opti.set_value(self.x0, x0)
        self.opti.set_value(self.x_ref, x_ref)
        self.opti.set_value(self.curvature, curvature)
        self.opti.set_value(self.s_0_arc, s_0_arc)
        self.opti.set_value(self.phi_0_arc, phi_0_arc)
        return self.solve()


class ObstacleLMPC(BaseMPC):
    # Uses the SoloFrenetModel

    def set_variables(self):
        self.x = self.opti.variable(self.n_states, self.N + 1)
        self.u = self.opti.variable(self.n_inputs, self.N)  # For multi-shooting
        self.terminal_state_slack = self.opti.variable(self.n_states, 1)

    def set_parameters(self):
        self.curvature = self.opti.parameter(1, self.N)
        self.s_0_arc = self.opti.parameter(1, self.N)
        self.phi_0_arc = self.opti.parameter(1, self.N)
        self.terminal_cost = self.opti.parameter(1, 1)
        self.terminal_state = self.opti.parameter(self.n_states, 1)
        self.s_obs = self.opti.parameter(1, 1)
        self.s_final = self.opti.parameter(1, 1)
        return self.curvature, self.s_0_arc, self.phi_0_arc

    def set_cost(self):
        self.cost = 0
        k = -4
        for t in range(self.N):
            s_err = self.x[0, t] - self.s_final
            # Use algebraic sigmoid to define a smooth cost function=11
            self.cost += (1 / 2) * (k * s_err / ca.sqrt(1 + (k * s_err) ** 2) + 1)
        # Input cost
        # if self.R is not None:
        #     for t in range(self.N):
        #         self.cost += self.u[:, t].T @ self.R @ self.u[:, t]
        if self.R is not None:
            for t in range(self.N - 1):
                u_diff = self.u[:, t + 1] - self.u[:, t]
                self.cost += u_diff.T @ self.R @ u_diff

        # terminal cost
        self.cost += self.terminal_cost

        # self.cost = self.N + self.terminal_cost
        self.cost += 1e3 * (self.terminal_state_slack.T @ self.terminal_state_slack)

    def set_nonlinear_constraints(self):
        # Terminal state constraint
        # self.opti.subject_to(self.x[:, self.N] == self.terminal_state)
        self.opti.subject_to(
            self.terminal_state_slack +
            #
            self.x[:, self.N]
            == self.terminal_state
        )
        # Collision avoidance with rect ellipse
        deg = 2
        L = self.model.LENGTH
        W = self.model.WIDTH
        dL = (2 ** (1 / deg) - 1) * L
        dW = W / L * dL

        # center of ego
        ego_c = self.x[0, :] + self.model.LENGTH / 2 - self.model.BACKTOWHEEL
        # center of obs
        obs_c = self.s_obs + self.model.LENGTH / 2 - self.model.BACKTOWHEEL
        # lateral deviations of ego
        ego_e = self.x[1, :]
        # lateral deviations of obs
        obs_e = self.model.x0[1, 0]
        # rect ellipse
        rectellipse_s = (2 * (ego_c - obs_c) / (2 * L + dL)) ** deg
        rectellipse_e = (2 * (ego_e - obs_e) / (2 * W + dW)) ** deg
        rectellipse = rectellipse_s + rectellipse_e
        self.opti.subject_to(rectellipse >= 1)

    def get_ctrl(
        self,
        x0,
        curvature,
        s_0_arc,
        phi_0_arc,
        terminal_cost,
        terminal_state,
        s_obs,
        s_final,
    ):
        self.opti.set_value(self.x0, x0)
        self.opti.set_value(self.curvature, curvature)
        self.opti.set_value(self.s_0_arc, s_0_arc)
        self.opti.set_value(self.phi_0_arc, phi_0_arc)
        self.opti.set_value(self.terminal_cost, terminal_cost)
        self.opti.set_value(self.terminal_state, terminal_state)
        self.opti.set_value(self.s_obs, s_obs)
        self.opti.set_value(self.s_final, s_final)
        u_pred, x_pred, cost = self.solve(show_infeasibilities=False)
        slack_value = self.opti.value(self.terminal_state_slack)
        slack_norm = slack_value.T @ slack_value
        # slack_norm = 0
        return u_pred, x_pred, cost, slack_norm
