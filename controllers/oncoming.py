import casadi as ca
from controllers.base import BaseMPC
from controllers.solo import SoloMPC


# For iteration 0 we use SoloMPC in this scenario too


class OncomingSoloLMPC(SoloMPC):
    L = 200

    # Uses the SoloFrenetModel

    def set_variables(self):
        self.x = self.opti.variable(self.n_states, self.N + 1)
        self.u = self.opti.variable(self.n_inputs, self.N)  # For multi-shooting
        self.terminal_state_slack = self.opti.variable(self.n_states, 1)
        self.onc_slack = self.opti.variable(1, 1)

    def set_parameters(self):
        self.curvature = self.opti.parameter(1, self.N)
        self.s_0_arc = self.opti.parameter(1, self.N)
        self.phi_0_arc = self.opti.parameter(1, self.N)
        self.terminal_cost = self.opti.parameter(1, 1)
        self.terminal_state = self.opti.parameter(self.n_states, 1)
        self.s_obs = self.opti.parameter(1, 1)
        self.s_final = self.opti.parameter(1, 1)
        self.u_prev = self.opti.parameter(self.n_inputs, 1)
        self.onc_states = self.opti.parameter(self.n_states, 2)

        return self.curvature, self.s_0_arc, self.phi_0_arc

    def set_cost(self):
        self.cost = 0
        k = -4
        for t in range(self.N):
            s_err = self.x[0, t] - self.s_final
            # Use algebraic sigmoid to define a smooth cost function=11
            self.cost += (1 / 2) * (k * s_err / ca.sqrt(1 + (k * s_err) ** 2) + 1)
            if self.R is not None:
                self.cost += self.u[:, t].T @ self.R @ self.u[:, t]

        self.cost += self.terminal_cost
        self.cost += self.L * (self.terminal_state_slack.T @ self.terminal_state_slack)
        self.cost += self.L * 10 * self.onc_slack**2

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

        # Collision avoidance with rect ellipse for oncoming vehicle as obstacle
        c_onc_0 = self.onc_states[0, 0] - self.model.LENGTH / 2 + self.model.BACKTOWHEEL
        c_onc_N = (
            self.onc_states[0, -1] - self.model.LENGTH / 2 + self.model.BACKTOWHEEL
        )
        L_onc = c_onc_0 - c_onc_N + 2 * self.model.LENGTH
        c_onc = (c_onc_N + c_onc_0) / 2
        dL_onc = (2 ** (1 / deg) - 1) * L_onc
        dW_onc = W / L_onc * dL_onc

        # lateral deviations of obs
        e_onc = self.onc_states[1, -1]
        # rect ellipse
        rectellipse_s_onc = (2 * (ego_c - c_onc) / (2 * L_onc + dL_onc)) ** deg
        # rectellipse_e_onc = (2 * (ego_e - e_onc) / (2 * W + dW_onc)) ** deg
        rectellipse_e_onc = (2 * (ego_e - e_onc) / (2 * W + dW_onc)) ** deg
        rectellipse_onc = rectellipse_s_onc + rectellipse_e_onc

        self.opti.subject_to(self.onc_slack + rectellipse_onc >= 1)

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
        u_prev,
        onc_states,
    ):
        self.opti.set_value(self.x0, x0)
        self.opti.set_value(self.curvature, curvature)
        self.opti.set_value(self.s_0_arc, s_0_arc)
        self.opti.set_value(self.phi_0_arc, phi_0_arc)
        self.opti.set_value(self.terminal_cost, terminal_cost)
        self.opti.set_value(self.terminal_state, terminal_state)
        self.opti.set_value(self.s_obs, s_obs)
        self.opti.set_value(self.s_final, s_final)
        self.opti.set_value(self.u_prev, u_prev)
        self.opti.set_value(self.onc_states, onc_states)
        u_pred, x_pred, cost = self.solve(show_infeasibilities=False)
        slack_value = self.opti.value(self.terminal_state_slack)
        slack_norm = slack_value.T @ slack_value
        slack_norm += self.opti.value(self.onc_slack) ** 2
        # slack_norm = 0
        return u_pred, x_pred, cost, slack_norm


# For OncomingDuoLMPC() use the x0_ego x0_onc and terminal states for XN_onc and XN_ego


class OncomingDuoLMPC(SoloMPC):
    L = 200

    # Uses the SoloFrenetModel

    def set_variables(self):
        self.x = self.opti.variable(self.n_states, self.N + 1)
        self.u = self.opti.variable(self.n_inputs, self.N)  # For multi-shooting
        self.terminal_state_slack = self.opti.variable(self.n_states, 1)

    def set_parameters(self):
        self.curvature_ego = self.opti.parameter(1, self.N)
        self.s_0_arc_ego = self.opti.parameter(1, self.N)
        self.phi_0_arc_ego = self.opti.parameter(1, self.N)

        self.curvature_onc = self.opti.parameter(1, self.N)
        self.s_0_arc_onc = self.opti.parameter(1, self.N)
        self.phi_0_arc_onc = self.opti.parameter(1, self.N)

        self.terminal_cost = self.opti.parameter(1, 1)
        self.terminal_state = self.opti.parameter(self.n_states, 1)

        self.s_obs = self.opti.parameter(1, 1)
        self.s_final = self.opti.parameter(1, 1)
        self.u_prev = self.opti.parameter(self.n_inputs, 1)

        return (
            self.curvature_ego,
            self.s_0_arc_ego,
            self.phi_0_arc_ego,
            self.curvature_onc,
            self.s_0_arc_onc,
            self.phi_0_arc_onc,
        )

    def set_cost(self):
        self.cost = 0
        k = -4
        for t in range(self.N):
            s_err = self.x[0, t] - self.s_final
            # Use algebraic sigmoid to define a smooth cost function=11
            self.cost += (1 / 2) * (k * s_err / ca.sqrt(1 + (k * s_err) ** 2) + 1)
            if self.R is not None:
                self.cost += self.u[:, t].T @ self.R @ self.u[:, t]

        self.cost += self.terminal_cost
        self.cost += self.L * (self.terminal_state_slack.T @ self.terminal_state_slack)

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
        # center of onc
        onc_c = self.x[4, :] - self.model.LENGTH / 2 + self.model.BACKTOWHEEL
        # center of obs
        obs_c = self.s_obs + self.model.LENGTH / 2 - self.model.BACKTOWHEEL
        # lateral deviations of ego
        ego_e = self.x[1, :]
        # lateral deviations of onc
        onc_e = self.x[5, :]
        # lateral deviations of obs
        obs_e = self.model.x0[1, 0]
        # rect ellipse
        rectellipse_s = (2 * (ego_c - obs_c) / (2 * L + dL)) ** deg
        rectellipse_e = (2 * (ego_e - obs_e) / (2 * W + dW)) ** deg
        rectellipse = rectellipse_s + rectellipse_e
        self.opti.subject_to(rectellipse >= 1)

        # Collision avoidance with rect ellipse for ego vehicle with onc as obstacle
        c_onc_0 = self.x[4, 0] - self.model.LENGTH / 2 + self.model.BACKTOWHEEL
        c_onc_N = (
            self.terminal_state[4, 0] - self.model.LENGTH / 2 + self.model.BACKTOWHEEL
        )
        L_onc = c_onc_0 - c_onc_N + 2 * self.model.LENGTH
        c_onc = (c_onc_N + c_onc_0) / 2
        dL_onc = (2 ** (1 / deg) - 1) * L_onc
        dW_onc = W / L_onc * dL_onc

        # lateral deviations of obs
        e_onc = self.terminal_state[5, -1]
        # rect ellipse
        rectellipse_s_onc = (2 * (ego_c - c_onc) / (2 * L_onc + dL_onc)) ** deg
        rectellipse_e_onc = (2 * (ego_e - e_onc) / (2 * W + dW_onc)) ** deg
        rectellipse_onc = rectellipse_s_onc + rectellipse_e_onc
        self.opti.subject_to(rectellipse_onc >= 1)

        # Collision avoidance with rect ellipse for onc vehicle with ego as obstacle
        c_ego_0 = self.x[0, 0] + self.model.LENGTH / 2 - self.model.BACKTOWHEEL
        c_ego_N = (
            self.terminal_state[0, 0] + self.model.LENGTH / 2 - self.model.BACKTOWHEEL
        )
        L_ego = c_ego_0 - c_ego_N + 2 * self.model.LENGTH
        c_ego = (c_ego_N + c_ego_0) / 2
        dL_ego = (2 ** (1 / deg) - 1) * L_ego
        dW_ego = W / L_ego * dL_ego

        # lateral deviations of obs
        e_ego = self.terminal_state[1, 0]
        # rect ellipse
        rectellipse_s_ego = (2 * (onc_c - c_ego) / (2 * L_ego + dL_ego)) ** deg
        rectellipse_e_ego = (2 * (onc_e - e_ego) / (2 * W + dW_ego)) ** deg
        rectellipse_ego = rectellipse_s_ego + rectellipse_e_ego
        self.opti.subject_to(rectellipse_ego >= 1)

    def get_ctrl(
        self,
        x0,
        curvature_ego,
        s_0_arc_ego,
        phi_0_arc_ego,
        curvature_onc,
        s_0_arc_onc,
        phi_0_arc_onc,
        terminal_cost,
        terminal_state,
        s_obs,
        s_final,
        u_prev,
    ):
        self.opti.set_value(self.x0, x0)

        self.opti.set_value(self.curvature_ego, curvature_ego)
        self.opti.set_value(self.s_0_arc_ego, s_0_arc_ego)
        self.opti.set_value(self.phi_0_arc_ego, phi_0_arc_ego)

        self.opti.set_value(self.curvature_onc, curvature_onc)
        self.opti.set_value(self.s_0_arc_onc, s_0_arc_onc)
        self.opti.set_value(self.phi_0_arc_onc, phi_0_arc_onc)

        self.opti.set_value(self.terminal_cost, terminal_cost)
        self.opti.set_value(self.terminal_state, terminal_state)
        self.opti.set_value(self.s_obs, s_obs)
        self.opti.set_value(self.s_final, s_final)
        self.opti.set_value(self.u_prev, u_prev)

        u_pred, x_pred, cost = self.solve(show_infeasibilities=False)
        slack_value = self.opti.value(self.terminal_state_slack)
        slack_norm = slack_value.T @ slack_value
        # slack_norm = 0
        return u_pred, x_pred, cost, slack_norm
