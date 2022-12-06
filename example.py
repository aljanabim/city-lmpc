from models.example import ExampleModel
from models.track import Track
from controllers.example import ExampleMPC
import casadi as ca
from utils.sim_util import ExpMeta, get_data_folder
from utils.vis_util import VehicleData, COLORS, animate_trajectory

if __name__ == "__main__":
    model = ExampleModel(x0=[10])
    model.NEW_PARAM = 1
    controller = ExampleMPC(
        model,
        Q=ca.diag(1),
        R=ca.diag(1.2),
        xlb=-ca.inf,
        xub=ca.inf,
        ulb=-5,
        uub=5,
    )

    x0 = model.x0
    x_ref = ca.DM(1)
    # when |p| > uub / dt, the controller is insufficient
    p = ca.DM.zeros((1, controller.N)) + 10
    while x0 - x_ref > 0.01:
        uopt, xopt = controller.get_ctrl(x0, x_ref, p)
        # State Feedback Loop
        x0 = model.sim(
            1, u=ca.vertcat(uopt, p[0, 0]), input_noise=False, state_noise=False
        )
        print(uopt, x0)
    states, inputs = model.get_trajectory()
    # print(states, inputs)

    quit()
    s_onc_0 = 0
    track = Track([])
    exp_meta = ExpMeta("example", 0, 0, 0)
    dest_folder = get_data_folder(exp_meta, s_onc_0)
    vehicles = [
        VehicleData("example", COLORS["ego"], states, inputs),
    ]

    animate_trajectory(
        dest_folder,
        track,
        vehicles,
        animation_filename="final",
        save=False,
        file_format="gif",
        plot_input=True,
    )
