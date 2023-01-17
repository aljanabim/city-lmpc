# City LMPC

An LMPC implementation applied to smart-city scenarios.

## Scenarios

City-LMPC demonstrates the application of LMPC on several smart city scenarios with varying focus and complexities. Each scenario has a dedicated model, controller, simulator, and stored trajectories found in:

-   **Model**: `city_lmpc/models/[scenario].py`
-   **Controller**: `city_lmpc/controller/[scenario].py`
-   **Simulator**: `city_lmpc/[scenario].py`
-   **Trajectories**: `city_lmpc/trajectories/[scenario]`

### Model

A model defines the kinematic model of all agents involved in the demo. Every model is a class with inherits the `BaseModel` class found under `city_lmpc/models/base.py`.

### Controller

A controller builds the control variables, control parameters, cost function, and constraints which are then provided to an IPOPT solver through the `CasADi` package.

### Simulator

A simulator puts together the model and controller to generate the initial, feasible trajectory for the LMPC, J0, and to continue the iterative learning until a convergence criteria is met.

### Trajectory

Hold the recorded data from the state and input trajectories of each iteration executed during simulation of the scenario.
