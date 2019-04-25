import numpy as np 
import scipy.optimize

class DoubleIntegratorMinimumForceTrapezoidal(object):
    def __init__(self, N):
        self.N = N

        self.t = np.linspace(0, 1, N)
        self.h = self.t[1:] - self.t[:-1]
        

    def obj_func(self, variables):
        """
        @brief      the function to minimize
        sum_{k=0}^{N-2} 0.5*(t_{k+1} - t_k) * (u_k^2+u_{k+1}^2)

        
        @param      variables  The decision variables
        length 3*N np.array 
        [0:N] controls u
        [N::2] positions x
        [N+1::2] velocities v
        @param      t knot points t0, ..., t_{N-1}

        @return     { description_of_the_return_value }
        """
        N = self.N
        u = variables[0:self.N]

        obj = 0
        for i in range(N-1):
            obj += 0.5*(self.h[i]) * (u[i]**2 + u[i+1]**2)
        return obj

    def eq_constraint_func(self, variables):
        """
        @brief      the equality constraints 
        x[0] = 0
        v[0] = 0
        x[N-1] = 1
        v[N-1] = 0
        x_{k+1}-x_k = 0.5*(t_{k+1}-t_k)*(v_{k+1} + v_k) for k=0...N-2
        v_{k+1}-v_k = 0.5*(t_{k+1}-t_k)*(u_{k+1} + u_k) for k=0...N-2
        
        @param      variables  The decision variables
        length 3*N np.array 
        [0:N] controls u
        [N::2] positions x
        [N+1::2] velocities v
        @param      t knot points t0, ..., t_{N-1}
        
        @return     { description_of_the_return_value }
        """
        N = self.N
        u = variables[0:N]
        pos = variables[N::2]
        vel = variables[N+1::2]


        constraints = np.zeros(4 + (N-1)*2)
        constraints[0] = pos[0] # x[0] = 0
        constraints[1] = vel[0] # v[0] = 0
        constraints[2] = pos[N-1] - 1 # x[N] = 1
        constraints[3] = vel[N-1] # v[N] = 0
        for i in range(N-1):
            constraints[4 + i] = (pos[i+1]-pos[i]) - 0.5*(self.h[i])*(vel[i] + vel[i+1])
            constraints[4 + (N-1) + i] = (vel[i+1]-vel[i]) - 0.5*(self.h[i])*(u[i] + u[i+1])

        return constraints

    def initialize_variables(self):
        """
        @brief      initializes variables for the optimization problem
        u is initialized to all zeros
        x is set to be the same as t
        v is set to be 1 for all time
        
        @return     Initial variables
        length 3*N np.array 
        [0:N] controls u
        [N::2] positions x
        [N+1::2] velocities v
        """
        N = self.N
        variables = np.zeros(3*N)

        # populating positions
        variables[N::2] = self.t
        variables[N+1::2] = 1

        return variables


if __name__ == '__main__':
    N = 21
    problem = DoubleIntegratorMinimumForceTrapezoidal(21)
    variables = problem.initialize_variables()

    constraints_dict = {'type': 'eq',
                        'fun': problem.eq_constraint_func}

    sol = scipy.optimize.minimize(
        fun=problem.obj_func,
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
    plt.plot(problem.t, sol_vars[0:N], linestyle="--", label='optimizer')
    plt.ylabel("control (m/s/s)")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(analytic_t, analytic_pos, label='analytic')
    plt.plot(problem.t, sol_vars[N::2], linestyle="--", label='optimizer')
    plt.ylabel("position (m)")

    plt.subplot(3, 1, 3)
    plt.plot(analytic_t, analytic_vel, label='analytic')
    plt.plot(problem.t, sol_vars[N+1::2], linestyle="--", label='optimizer')
    plt.ylabel("velocity (m)")

    plt.xlabel("time (s)")

    plt.show()


