import casadi as ca
import numpy as np
from controllers.solo import SoloMPC, SoloRelaxedLMPC
from models.solo import SoloFrenetModel
from simulators.solo import SoloMPCSimulator, SoloRelaxedLMPCSimulator
from utils import sim_util, vis_util


if __name__ == "__main__":
    """
    Totally works only weird thing is that the J1-11 vehicles don't go all the
    way to xf and stop a little bit short, must be something related to relaxation
    of constraints, but I'm not sure exactly what it is.

    Tried to let the cost-to-go also include the full values with lambda, but got similar
    results for J1 and then a bunch of bugs for further iterations.
    TODO
    1. Update cost function in LMPC using the ObstacleLMPC cost (with algebraic Sigmoid)
    2. Update LMPC simulator to give s_final as parameter which is equal to track length
    2. Use track_J0 to generate J0
    3. Rerun the LMPC simulator
    """
    # solo+R20L20 converged
    # solo+R0L100 converged
    # solo+R0L50 converged
    # solo+R0L20 converged Best cost 151
    # solo++R0L20 converged Best cost 150. It uses it_idx as in paper not as in obstacle simulator
    # Nothing else has non-decreasing performance
    # solo+R0L18 infeasible
    # solo+R0L15 does not have decreasing performance
    # solo+R0L10 does not have decreasing performance
    # solo+R0L0 does not have decreasing performance

    # solo-R(0, 0)L100N7 converged from 216 to 155

    # Conclusion on N, I tried N=12,15,20 and all lead to infeasible solutions. Somehow 7 seems to be the sweet spot

    R = (0, 0)
    L = 100
    N = 15
    load_until = 6

    EXP_NAME = f"solo-R{R}L{L}N{N}"
    model, mpc, track_vis, track_ctrl, track_J0, xub_lmpc = sim_util.setup_solo(
        Controller=SoloMPC
    )
    track_J0.update_n_points(205)
    simulator = SoloMPCSimulator(model, mpc, track_J0, track_J0)
    simulator.EXP_NAME = EXP_NAME
    traj_0 = simulator.load(iteration=0)
    if traj_0 is None:  # if J0 is not stored, run simulator, save, then load it
        traj_0 = simulator.run().save().load()

    # vis_util.compare_iterations(EXP_NAME, 0, 6, track_vis, step=1)
    # quit()
    trajectories = [simulator.load(j) for j in range(0, load_until + 1)]

    traj = trajectories[0]

    vis_util.animate_trajectory(
        simulator.exp_meta,
        track_vis,
        [
            vis_util.VehicleData(
                "ego",
                vis_util.COLORS["ego"],
                traj["states"][:4, :],
                traj["inputs"],
            ),
            # vis_util.VehicleData(
            #     "obs",
            #     vis_util.COLORS["obs"],
            #     ca.DM([track_vis.length / 2, -0.25, 0, ca.pi / 2]),
            #     None,
            # ),
        ],
        plot_input=True,
        save=True,
        animation_filename="J0",
    )

    quit()
    # # # Start LMPC learning
    lmpc = SoloRelaxedLMPC(
        model,
        Q=None,
        R=ca.diag(R),
        xlb=-xub_lmpc,
        xub=xub_lmpc,
        ulb=-mpc.uub,
        uub=mpc.uub,
        N=N,
    )
    lmpc.L = L
    simulator = SoloRelaxedLMPCSimulator(
        model, lmpc, track_vis, track_ctrl, trajectories, max_iter=10
    )
    simulator.EXP_NAME = EXP_NAME
    simulator.track_ctrl.update_n_points(155)
    print(simulator.T)
    # simulator.run()
