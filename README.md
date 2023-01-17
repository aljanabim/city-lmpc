# City LMPC

An LMPC implementation applied to smart-city scenarios.

## Scenarios

City-LMPC demonstrates the application of LMPC on several smart city scenarios with varying focus and complexities. Each scenario has a dedicated model, controller, simulator, and stored trajectories found in:

-   **Model**: `city_lmpc/models/[scenario].py`
-   **Controller**: `city_lmpc/controllers/[scenario].py`
-   **Simulator**: `city_lmpc/simulators/[scenario].py`
-   **Trajectories**: `city_lmpc/trajectories/[scenario]`
-   **main**: `city_lmpc/[scenario].py`

LMPC requires that we have an initial feasible trajectory. In the City-LMPC project we create these initial feasible trajectories using other MPC controllers and simulators which lives in the same file that corresponds to the scenario. In other words, each controller and simulator of a scenario contain both the LMPC controller and the corresponding initial controller which generates the first feasible trajectory.

### Model

A model defines the kinematic model of all agents involved in the demo. Every model is a class with inherits the `BaseModel` class found under `city_lmpc/models/base.py`.

### Controller

A controller builds the control variables, control parameters, cost function, and constraints which are then provided to an IPOPT solver through the `CasADi` package.

### Simulator

A simulator puts together the model and controller to generate the initial, feasible trajectory for the LMPC, J0, and to continue the iterative learning until a convergence criteria is met.

### Trajectory

Hold the recorded data from the state and input trajectories of each iteration executed during simulation of the scenario.

### Main

Contains the main run file corresponding to each scenario.
