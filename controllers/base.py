import numpy as np
import casadi as ca
from models.base import BaseModel


class BaseMPC:
    def __init__(
        self,
        model: BaseModel,
        Q: ca.DM,
        R: ca.DM,
        xlb: ca.DM,
        xub: ca.DM,
        ulb: ca.DM,
        uub: ca.DM,
        N=7,
        state_noise=False,
        input_noise=False,
    ) -> None:
        self.model = model
        # Number of states and inputs
        self.n_states = len(self.model.dae.x)
        self.n_inputs = len(self.model.dae.u)
        # Number of control intervals
        self.N = N
        # MPC Parameters
        self.Q = Q
        self.R = R
        self.xlb = xlb
        self.xub = xub
        self.ulb = ulb
        self.uub = uub
        # Noise settings
        self.state_noise = state_noise
        self.input_noise = input_noise

        # setup optimizer
        self.opti = self._create_optimizer()
        # Build variables, parameters, cost,
        self.build_optimizer()

    def _create_optimizer(self):
        opti = ca.Opti()
        p_opts = {
            "verbose": False,
            "expand": True,
            "print_in": False,
            "print_out": False,
            "print_time": False,
        }
        s_opts = {
            "max_iter": 150,
            "print_level": 1,
            #   "mu_strategy": "adaptive",
            #   "mu_init": 1e-5,
            #   "mu_min": 1e-15,
            "fixed_variable_treatment": "make_constraint",
            "barrier_tol_factor": 1,
        }
        opti.solver("ipopt", p_opts, s_opts)
        return opti

    def set_variables(self):
        """
        Adds decision variables that will be optimized throughout the control
        horizon of the solver. For optimal results, set initial (guess) values with
        self.opti.set_value(self.x, values)
        where values are the guess through out the horizon
        """
        # TODO self.x = self.opti.variable(self.n_states, self.N + 1)
        # TODO self.u = self.opti.variable(self.n_inputs, self.N) # For multi-shooting
        self.x = self.opti.variable(self.n_states, self.N + 1)
        self.u = self.opti.variable(
            self.n_inputs, self.N)  # For multi-shooting

    def set_parameters(self):
        """
        Adds non-decision variables that remain constant throughout the
        decision horizon. The value must set when getting the control through
        self.opti.set_value(self.param, value)
        ASSUMES: u_params are provided for the entire horizon self.N

        Returns:
            list of params: all opti parameters that should be applied after u in the model
        """
        # TODO self.x_ref = self.opti.parameter(self.n_states, self.N + 1)
        # TODO self.param1 = self.opti.parameter(1,self.N)
        # TODO self.param2 = self.opti.parameter(1,self.N)
        # TODO return self.param1, self.param2
        pass

    def set_cost(self):
        """
        Adds cost to be optimized, for example

        self.cost = 0
        for i in range(self.N):
            err = self.x[:, i] - self.x_ref
            self.cost += err.T @ self.Q @ err + self.u[:,i].T @ self.R @ self.u[:,i]
        errN = self.x[:, self.N] - self.x_ref
        self.cost += errN.T @ self.Q @ errN
        """
        pass

    def set_nonlinear_constraints(self):
        pass

    def build_optimizer(self):
        self.set_variables()

        # Handle parameters that should be concatenated with the input into the model update
        u_params = self.set_parameters()
        if u_params is None:
            u_params = []
        if isinstance(u_params, tuple):
            u = ca.vertcat(self.u, *u_params)
        else:
            u = ca.vertcat(self.u, u_params)

        # add the parameter for the initial value
        self.x0 = self.opti.parameter(self.n_states, 1)
        self.set_cost()
        self.opti.minimize(self.cost)

        # set dynamics constraints
        for t in range(self.N):
            x_next, _ = self.model.f(
                # curvature is included as a parameter in the simulation
                self.x[:, t],
                u[:, t],
                state_noise=self.state_noise,
                input_noise=self.input_noise,
            )
            self.opti.subject_to(self.x[:, t + 1] == x_next)
        # set linear constraints
        self.opti.subject_to(self.opti.bounded(self.ulb, self.u, self.uub))
        self.opti.subject_to(self.opti.bounded(self.xlb, self.x, self.xub))
        self.opti.subject_to(self.x[:, [0]] == self.x0)
        # set nonlinear constraints
        self.set_nonlinear_constraints()

        # self.mpc = opti.to_function("MPC", [x0, x_ref, curvature], [u[:, 0], self.cost, x[:, N], u], [
        #     "xt", "xN", "qN", "intruder_c"], ["u_opt", "cost", "xN", "u"])

    def solve(self):
        try:
            self.opti.solve()
        except RuntimeError as e:
            self.opti.debug.show_infeasibilities()
            raise (e)
        # u_opt = np.expand_dims(self.opti.value(self.u[:, 0]), axis=1)
        u_opt = self.opti.value(self.u[:, [0]])
        x_pred = self.opti.value(self.x)
        return u_opt, x_pred

    def get_ctrl(self, x0, *parameters):
        # TODO self.opti.set_value(self.x0,x0)
        # TODO self.opti.set_value(self.param,param[0])
        # TODO return self.solve()
        pass
