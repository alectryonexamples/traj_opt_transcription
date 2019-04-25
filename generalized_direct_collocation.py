import numpy as np 
import scipy.optimize


class DoubleIntegratorMinimumForce(object):
    def __init__(self):
        pass

    def x_init(self):
        return np.zeros(2)

    def x_end(self):
        return np.array([1., 0])

    def n_states(self):
        return 2

    def n_controls(self):
        return 1

    def deriv(self, state, control):
        """
        @brief      differential equation for double integrator
        pos_dot = vel
        vel_dot = control
        
        @param      state    length 2 np.array of [pos, vel]
        @param      control  length 1 np.array of [control]
        
        @return     returns length 2 np.array of [pos_dot, vel_dot]
        """
        deriv = np.array([state[1], control[0]])
        return deriv

    def obj_func(self, state, control):
        cost = control[0]**2
        return cost


class TranscribedProblem(object):
    def __init__(self, N, sys):
        self.N = N
        self.sys = sys

        self.t = np.linspace(0, 1, N)
        self.h = self.t[1:] - self.t[:-1]

    def obj_func(self, variables):
        N = self.N
        n_states = self.sys.n_states()
        n_controls = self.sys.n_controls()

        u = variables[0:N*n_controls]
        x = variables[N*n_controls:]

        x_init = x[:n_states]
        x_end = x[-n_states:]
        obj = 0
        for i in range(N-1):
            idx1 = i*n_states
            idx2 = (i+1)*n_states
            idx3 = (i+2)*n_states

            x_i = x[idx1:idx2]
            x_ip1 = x[idx2:idx3]

            u_i = u[i*n_controls:(i+1)*n_controls]
            u_ip1 = u[(i+1)*n_controls:(i+2)*n_controls]
            obj += self._trapezoidal(self.h[i], self.sys.obj_func, x_i, x_ip1, u_i, u_ip1)
        return obj

    def sys_constraints(self, variables):
        """
        @brief      the equality constraints that come from system consistency
        
        @param      variables  The decision variables 
        [0:N*n_control] controls u
        [N*n_control:N*(n_control + n_state)] states x
        controls and states are arranged such that the first
        n_controls or n_states values in their section are the control and states
        for the first timestep

        @return     { description_of_the_return_value }
        """
        N = self.N
        n_states = self.sys.n_states()
        n_controls = self.sys.n_controls()

        u = variables[0:N*n_controls]
        x = variables[N*n_controls:]

        # initial and end constraints
        constraints = np.zeros(2*n_states + (N-1)*n_states)
        constraints[0:n_states] = x[0:n_states] - self.sys.x_init()
        constraints[n_states:2*n_states] = x[-n_states:] - self.sys.x_end()
        offset = (2*n_states)

        # system dynamic constraints
        for i in range(N-1):
            idx1 = i*n_states
            idx2 = (i+1)*n_states
            idx3 = (i+2)*n_states

            x_i = x[idx1:idx2]
            x_ip1 = x[idx2:idx3]

            u_i = u[i*n_controls:(i+1)*n_controls]
            u_ip1 = u[(i+1)*n_controls:(i+2)*n_controls]

            constraints[offset+idx1:offset+idx2] = (x_ip1 - x_i) - \
                self._trapezoidal(self.h[i], self.sys.deriv, x_i, x_ip1, u_i, u_ip1)

        return constraints

    def initialize_variables(self):
        N = self.N
        n_states = self.sys.n_states()
        n_controls = self.sys.n_controls()
        variables = np.zeros(N*(n_controls + n_states))

        initial_states = np.linspace(self.sys.x_init(), self.sys.x_end(), N)
        offset = N*n_controls
        for i in range(n_states):
            variables[offset+i::n_states] = initial_states[:, i]
        return variables

    def _trapezoidal(self, h, f, x_i, x_ip1, u_i, u_ip1):
        integral = 0.5*h*(f(x_i, u_i) + f(x_ip1, u_ip1))
        return integral

    def _hermite_simpson(self, h, f, x_i, x_ip1, u_i, u_ip1):
        f_i = f(x_i, u_i)
        f_ip1 = f(x_ip1, u_ip1)
        x_iphalf = 0.5*(x_i + x_ip1) + (h/8)*(f_i - f_ip1)
        f_iphalf = f(x_iphalf, u_iphalf)
        integral = (1./6)*h*(f_i + 4*f_iphalf + f_ip1)
        return integral

if __name__ == '__main__':
    N = 21

    problem = DoubleIntegratorMinimumForce()
    transcribed_problem = TranscribedProblem(N, problem)
    variables = transcribed_problem.initialize_variables()

    constraints_dict = {'type': 'eq',
                        'fun': transcribed_problem.sys_constraints}

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

    import matplotlib.pyplot as plt

    analytic_t = np.linspace(0, 1, 100)
    analytic_control = 6 - 12*analytic_t
    analytic_pos = 3*np.square(analytic_t) - 2*np.power(analytic_t, 3)
    analytic_vel = 6*analytic_t - 6*np.square(analytic_t)

    plt.figure(1)
    plt.subplot(3, 1, 1)
    plt.plot(analytic_t, analytic_control, label='analytic')
    plt.plot(transcribed_problem.t, sol_vars[0:N], linestyle="--", label='optimizer')
    plt.ylabel("control (m/s/s)")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(analytic_t, analytic_pos, label='analytic')
    plt.plot(transcribed_problem.t, sol_vars[N::2], linestyle="--", label='optimizer')
    plt.ylabel("position (m)")

    plt.subplot(3, 1, 3)
    plt.plot(analytic_t, analytic_vel, label='analytic')
    plt.plot(transcribed_problem.t, sol_vars[N+1::2], linestyle="--", label='optimizer')
    plt.ylabel("velocity (m)")

    plt.xlabel("time (s)")

    plt.show()


