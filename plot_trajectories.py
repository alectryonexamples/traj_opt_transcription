import numpy as np
import matplotlib.pyplot as plt
from functools import partial
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



def plot_block(transcribed_problem, sol_vars):
    # values of u and x at knot points
    u, x, t_u, t_x = transcribed_problem.parse_variables(sol_vars)

    # the actual piecewise trajectory the solution takes on
    u_func, x_func = transcribed_problem.construct_piecewise_trajectory(sol_vars)

    times = np.linspace(0, 1, 100)

    traj_u = np.array(list(map(u_func, times)))
    traj_x = np.array(list(map(x_func, times)))

    # analytic solution to the double integrator problem
    analytic_control = 6 - 12*times
    analytic_pos = 3*np.square(times) - 2*np.power(times, 3)
    analytic_vel = 6*times - 6*np.square(times)


    plt.figure(1)
    plt.subplot(3, 1, 1)
    plt.plot(times, analytic_control, c='steelblue', label='analytic')
    plt.plot(times, traj_u, c='salmon', linestyle='--', label='analytic')
    plt.scatter(t_u, u[:, 0], c='salmon', label='optimized')
    plt.ylabel("control (m/s/s)")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(times, analytic_pos, label='analytic')
    plt.plot(times, traj_x[:, 0], c='salmon', linestyle='--', label='analytic')
    plt.scatter(t_x, x[:, 0], c='salmon', label='optimized')
    plt.ylabel("position (m)")

    plt.subplot(3, 1, 3)
    plt.plot(times, analytic_vel, label='analytic')
    plt.plot(times, traj_x[:, 1], c='salmon', linestyle='--', label='analytic')
    plt.scatter(t_x, x[:, 1], c='salmon', label='optimized')
    plt.ylabel("velocity (m)")

    plt.xlabel("time (s)")

    plt.show()

def plot_pendulum(transcribed_problem, sol_vars):
    # values of u and x at knot points
    u, x, t_u, t_x = transcribed_problem.parse_variables(sol_vars)

    # the actual piecewise trajectory the solution takes on
    u_func, x_func = transcribed_problem.construct_piecewise_trajectory(sol_vars)

    times = np.linspace(0, problem.T(), 100)
    traj_u = np.array(list(map(u_func, times)))
    traj_x = np.array(list(map(x_func, times)))

    plt.figure(1)
    plt.subplot(3, 1, 1)
    plt.plot(times, traj_u[:, 0], c='salmon', linestyle='--')
    plt.scatter(t_u, u[:, 0], c='salmon')
    plt.ylabel("control (rad/s/s)")

    plt.subplot(3, 1, 2)
    plt.plot(times, traj_x[:, 0], c='salmon', linestyle='--')
    plt.scatter(t_x, x[:, 0], c='salmon')
    plt.ylabel("theta (rad)")

    plt.subplot(3, 1, 3)
    plt.plot(times, traj_x[:, 1], c='salmon', linestyle='--')
    plt.scatter(t_x, x[:, 1], c='salmon')
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
    if args.problem == "block":
        plot_block(transcribed_problem, sol_vars)
    else:
        plot_pendulum(transcribed_problem, sol_vars)



