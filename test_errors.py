import time
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


def get_analytic_solution(t):
    u = 6 - 12*t
    pos = 3*np.square(t) - 2*np.power(t, 3)
    vel = 6*t - 6*np.square(t)
    return u, pos, vel

def get_errs(t, x, v):
    analytic_u, analytic_x, analytic_v = get_analytic_solution(t)
    h = t[1:] - t[:-1]

    pos_err_abs = np.abs(analytic_x - x)
    pos_err = np.sum((pos_err_abs[1:] + pos_err_abs[:-1]) * h)
    vel_err_abs = np.abs(analytic_v - v)
    vel_err = np.sum((vel_err_abs[1:] + vel_err_abs[:-1]) * h)
    return pos_err, vel_err


def solve_problem(N, int_type):
    problem = DoubleIntegratorMinimumForce()
    transcribed_problem = TranscribedProblem(N, problem, int_type)

    variables = transcribed_problem.initialize_variables()
    constraints_dict = {'type': 'eq',
                        'fun': transcribed_problem.sys_constraints}

    num_iter = 4
    start_time = time.process_time()
    for _ in range(num_iter):
        sol = scipy.optimize.minimize(
            fun=transcribed_problem.obj_func,
            x0=variables,
            method="SLSQP",
            constraints=constraints_dict)
    end_time = time.process_time()
    comp_time = end_time - start_time
    avg_time = comp_time / num_iter

    if not sol.success:
        return np.Inf, np.Inf, 0

    sol_vars = sol.x
    u, x, t_u, t_x = transcribed_problem.parse_variables(sol_vars)
    pos_err, vel_err = get_errs(t_x, x[:, 0], x[:, 1])
    return pos_err, vel_err, avg_time


if __name__ == '__main__':
    import matplotlib.pyplot as plt


    N_list = np.arange(11, 41, 2)

    trap_pos_errs = []
    hermite_pos_errs = []

    trap_vel_errs = []
    hermite_vel_errs = []

    trap_time = []
    hermite_time = []

    for N in N_list:
        pos_err, vel_err, avg_time = solve_problem(N, IntegrationType.TRAPEZOIDAL)
        trap_pos_errs.append(pos_err)
        trap_vel_errs.append(vel_err)
        trap_time.append(avg_time)

        pos_err, vel_err, avg_time = solve_problem(N, IntegrationType.HERMITE_SIMPSON)
        hermite_pos_errs.append(pos_err)
        hermite_vel_errs.append(vel_err)
        hermite_time.append(avg_time)

    plt.figure(1)
    plt.subplot(3, 1, 1)
    plt.title("Double Integrator errors/timing")
    plt.plot(N_list, trap_pos_errs, label="trapezoidal")
    plt.plot(N_list, hermite_pos_errs, label="hermite-simpson")
    plt.gca().set_yscale('log')
    plt.grid()
    plt.ylabel("pos err")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(N_list, trap_vel_errs, label="trapezoidal")
    plt.plot(N_list, hermite_vel_errs, label="hermite-simpson")
    plt.gca().set_yscale('log')
    plt.grid()
    plt.ylabel("vel err")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(N_list, trap_time, label="trapezoidal")
    plt.plot(N_list, hermite_time, label="hermite-simpson")
    plt.ylabel("time (s)")
    plt.grid()
    plt.xlabel("N")
    plt.legend()

    plt.show()



