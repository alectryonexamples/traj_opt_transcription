import numpy as np
import matplotlib.pyplot as plt
from transcription import *

class DoubleIntegratorMinimumForce(ProblemBase):
    def __init__(self):
        pass

    def T(self):
        return 1.

    def x_init(self):
        return np.zeros(2)

    def x_end(self):
        return np.array([1., 0])

    def state_dim(self):
        return 2

    def control_dim(self):
        return 1

    def f(self, state, control):
        deriv = np.array([state[1], control[0]])
        return deriv

    def obj_func(self, state, control):
        cost = control[0]**2
        return cost

class PendulumMinimumTorque(object):
    def __init__(self):
        self.grav = 9.81
        self.length = 1.

    def T(self):
        return 10.

    def x_init(self):
        return np.zeros(2)

    def x_end(self):
        return np.array([np.pi, 0])

    def state_dim(self):
        return 2

    def control_dim(self):
        return 1

    def f(self, state, control):
        deriv = np.zeros(2)
        deriv[0] = state[1]
        deriv[1] = (-self.grav / self.length) * np.sin(state[0]) + control[0]
        return deriv

    def obj_func(self, state, control):
        cost = control[0]**2
        return cost

def plot_block(u, x, t_u, t_x):
    analytic_t = np.linspace(0, 1, 100)
    analytic_control = 6 - 12*analytic_t
    analytic_pos = 3*np.square(analytic_t) - 2*np.power(analytic_t, 3)
    analytic_vel = 6*analytic_t - 6*np.square(analytic_t)

    plt.figure(1)
    plt.subplot(3, 1, 1)
    plt.plot(analytic_t, analytic_control, label='analytic')
    plt.plot(t_u, u[:, 0], linestyle="--", label='optimizer')
    plt.ylabel("control (m/s/s)")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(analytic_t, analytic_pos, label='analytic')
    plt.plot(t_x, x[:, 0], linestyle="--", label='optimizer')
    plt.ylabel("position (m)")

    plt.subplot(3, 1, 3)
    plt.plot(analytic_t, analytic_vel, label='analytic')
    plt.plot(t_x, x[:, 1], linestyle="--", label='optimizer')
    plt.ylabel("velocity (m)")

    plt.xlabel("time (s)")

    plt.show()

def plot_pendulum(u, x, t_u, t_x):
    plt.figure(1)
    plt.subplot(3, 1, 1)
    plt.plot(t_u, u[:, 0], linestyle="--")
    plt.ylabel("control (rad/s/s)")

    plt.subplot(3, 1, 2)
    plt.plot(t_x, x[:, 0], linestyle="--")
    plt.ylabel("theta (rad)")

    plt.subplot(3, 1, 3)
    plt.plot(t_x, x[:, 1], linestyle="--")
    plt.ylabel("theta/s (rad/s)")

    plt.xlabel("time (s)")

    plt.show()

if __name__ == '__main__':
    import argparse
    import time
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--problem", 
                        action="store",
                        type=str,
                        choices=["block", "pendulum"],
                        default="block",
                        help="increase output verbosity")
    parser.add_argument("-N", "--N",
                        action="store",
                        type=int,
                        default=31,
                        help="Number of state knot points")
    parser.add_argument("-i", "--int",
                        action="store",
                        type=str,
                        choices=["trap", "hermite"],
                        default="trap", 
                        help="Integration type")
    parser.add_argument("--iter",
                        action="store",
                        type=int,
                        default=0,
                        help="Number of times to run the optimizer to time it")
    args = parser.parse_args()

    if args.problem == "block":
        problem = DoubleIntegratorMinimumForce()
    else:
        problem = PendulumMinimumTorque()
    N = args.N
    if args.int == "trap":
        int_type = IntegrationType.TRAPEZOIDAL
    else:
        int_type = IntegrationType.HERMITE_SIMPSON

    transcribed_problem = TranscribedProblem(N, problem, int_type)
    
    variables = transcribed_problem.initialize_variables()
    constraints_dict = {'type': 'eq',
                        'fun': transcribed_problem.sys_constraints}

    start_time = time.process_time()
    for _ in range(args.iter):
        sol = scipy.optimize.minimize(
            fun=transcribed_problem.obj_func,
            x0=variables,
            method="SLSQP",
            constraints=constraints_dict)
    end_time = time.process_time()
    comp_time = end_time - start_time
    
    if args.iter != 0:
        print("Solving problem " + str(args.iter) + " times with " + str(N) + " knot points took " + str(comp_time) + " seconds")
        print("Avg time: " + str(comp_time/args.iter) + " seconds")

    sol = scipy.optimize.minimize(
        fun=transcribed_problem.obj_func,
        x0=variables,
        method="SLSQP",
        constraints=constraints_dict)
    if not sol.success:
        print("Solution not found.")
        print(sol.message)
        exit()
    print("Solution found in " + str(sol.nit) + " iterations.")
    sol_vars = sol.x
    u, x, t_u, t_x = transcribed_problem.parse_variables(sol_vars)


    if args.problem == "block":
        plot_block(u, x, t_u, t_x)
    else:
        plot_pendulum(u, x, t_u, t_x)



