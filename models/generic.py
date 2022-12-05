from typing import Optional, Tuple
import numpy as np
import casadi as ca


class GenericModel:
    """
    Generic Model
    Exposes:
        build_model() // builds DAE, Integrator, and Simulator
        reset_trajectory() // Resets trajectory arrays and sets x0 as initial state
        get_trajectroy() // Return state, input trajectory, respectively
        set_x0() // sets initial state and resets the trajectory
        f() // executes a one step integration of the model
        sim() // executes a t-steps integration of the model, adds steps to trajectory
    """

    # TODO EDIT NOISE STD
    INPUT_NOISE_STD = 0.01
    STATE_NOISE_STD = 0.01

    def __init__(self, x0: Optional[Tuple[float]] = (0), dt: float = 0.1) -> None:
        # Set single integration step
        self.dt = dt

        self.x0 = ca.DM(x0)
        # Build Dae, Integrator and Simulator
        self.build_model()
        # Init trajectory logging
        self.reset_trajectory()

    def __str__(self) -> str:
        with np.printoptions(precision=3, suppress=True):
            x, u = self.get_trajectory()
            if len(self.input_trajectory) > 0:
                # TODO
                return f"Trajectory of vehicle \n  x:  {x[0,-1]}\n  u:  {u[0,-1]}"
            else:
                # TODO
                return f"Trajectory of vehicle \n  x:  {x[0,-1]}"

    @property
    def state(self):
        return self.state_trajectory[-1]

    @property
    def u(self):
        if len(self.input_trajectory) > 0:
            return self.input_trajectory[-1]
        else:
            n_inputs = len(self.dae.u)
            return ca.DM.zeros(n_inputs)

    def _build_dae(self):
        # add states
        # TODO self.dae.add_x("x")

        # add inputs
        # TODO self.dae.add_u("u")

        # add parameters (count towards inputs)
        # TODO self.dae.add_p("p")

        # differential equations of model
        # TODO x_dot =

        # add ode to dae
        # TODO self.dae.add_ode("x_dot", x_dot)

        # set units
        # TODO self.dae.set_unit("x", "m")
        pass

    def _build_integrator(self):
        # Extract states from DaeBuilder
        x = ca.vertcat(*self.dae.x)
        # Putting together the non-chaging parameters during an interation step, both input and road slope will be non-chaning
        # road_slope is concidered an input here
        u = ca.vertcat(*self.dae.u, *self.dae.p)
        # Extract ode from DaeBuilder
        ode = ca.vertcat(*self.dae.ode)
        # Options for the CVODES integrator
        # options_cvodes = {"abstol": 1e-5, "reltol": 1e-9, "max_num_steps": 100,
        #   "tf": self.dt}
        options_rk = {"simplify": True, "number_of_finite_elements": 40, "tf": self.dt}

        # DAE according to the Casadi Integrator API
        dae = {"x": x, "p": u, "ode": ode}
        self.integrator = ca.integrator("integrator", "rk", dae, options_rk)

    def _build_simulator(self):
        # Extract states from DaeBuilder
        x = ca.vertcat(*self.dae.x)
        # Putting together the non-changing parameters during an integration step
        u = ca.vertcat(*self.dae.u, *self.dae.p)
        # Evaluate the integrator symbolically
        res = self.integrator(x0=x, p=u)
        # Abstract out the next step from the integrator object
        x_next = res["xf"]
        # Create a function of the form (x_t, u_t, p) -> (x_t+1)
        self._F = ca.Function("F", [x, u], [x_next], ["x", "u"], ["x_next"])

    def build_model(self):
        # Setup DAE of platoon model and initial condition
        self.dae = ca.DaeBuilder()
        self._build_dae()

        # Setup integrator of platoon model
        self.integrator = None
        self._build_integrator()

        # Setup simulator for the platoon
        self._F = None
        self._build_simulator()

    def reset_trajectory(self):
        self.state_trajectory = [self.x0]
        self.input_trajectory = []

    def get_trajectory(self):
        if len(self.input_trajectory) > 0:
            return np.hstack(self.state_trajectory), np.hstack(self.input_trajectory)
        else:
            return np.hstack(self.state_trajectory), []

    def set_x0(self, x0):
        self.x0 = ca.DM(x0)

    def f(self, x, u, state_noise=False, input_noise=False) -> Tuple[float, float]:
        if input_noise:
            n_inputs = len(self.dae.u)
            u[:n_inputs, 0] += np.random.normal(
                0, self.INPUT_NOISE_STD, n_inputs
            )  # apply noise to inputs only not parameters
        x_next = self._F(x=x, u=u)["x_next"]
        if state_noise:
            n_states = len(self.dae.x)
            return x_next + np.random.normal(0, self.STATE_NOISE_STD, n_states), u
        return x_next, u

    def sim(
        self,
        time_steps,
        u,
        x0=None,
        state_noise=False,
        input_noise=False,
        is_logging=True,
    ):
        assert (
            time_steps == u.shape[1]
        ), "Input second dimension {} must match time_steps {}".format(
            u.shape[1], time_steps
        )
        if type(x0) == type(None):
            x0 = self.state_trajectory[-1]
        n_states = len(self.dae.x)
        res = ca.DM.zeros((n_states, time_steps))
        x_next = x0
        for t in range(time_steps):
            # output can injected noise, that is the one to be logged
            x_next, u_applied = self.f(
                # brackets on t to make the selection a column vector (n_states,1) instead of (n_states,)
                x_next,
                u[:, [t]],
                state_noise,
                input_noise,
            )
            # brackets on t to make the selection a column vector (n_states,1) instead of (n_states,)
            res[:, [t]] = x_next
            if is_logging:
                self.state_trajectory.append(x_next.full())
                self.input_trajectory.append(u_applied)
        return res.full()
        # previous approach using mapaccum but with static roadslopes
        # road_slopes = road_slope * np.ones((len(u), time_steps))
        # combined_u = np.vstack([u, road_slopes])
        # return horzcat(x0, self._sim.mapaccum(time_steps)(self.x0, combined_u))
