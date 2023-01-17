import casadi as ca
from controllers.generic import GenericMPC
from models.example import ExampleModel


class ExampleMPC(GenericMPC):
    # This part is only necessary incase the init of ExampleMPC takes different arguments than GenericMPC
    # def __init__(
    #     self,
    #     model: ExampleModel,
    #     Q: ca.DM,
    #     R: ca.DM,
    #     xlb: ca.DM,
    #     xub: ca.DM,
    #     ulb: ca.DM,
    #     uub: ca.DM,
    #     N=7,
    #     state_noise=False,
    #     input_noise=False,
    # ):
    #     super().__init__(model, Q, R, xlb, xub, ulb, uub, N, state_noise, input_noise)

    def set_parameters(self):
        self.x_ref = self.opti.parameter(self.n_states, 1)
        self.p = self.opti.parameter(1, self.N)
        return self.p

    def set_cost(self):
        self.cost = 0
        for i in range(self.N):
            err = self.x[:, i] - self.x_ref
            self.cost += err.T @ self.Q @ err + self.u[:, i].T @ self.R @ self.u[:, i]
        errN = self.x[:, self.N] - self.x_ref
        self.cost += errN.T @ self.Q @ errN

    def get_ctrl(self, x0, x_ref, p):
        self.opti.set_value(self.x0, x0)
        self.opti.set_value(self.x_ref, x_ref)
        self.opti.set_value(self.p, p)
        return self.solve()
