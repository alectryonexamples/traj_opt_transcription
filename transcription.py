import numpy as np 
import scipy.optimize
import enum
import math
from functools import partial

class ProblemBase(object):
    """
    @brief      Interface for problems to be transcribed into discrete 
                trajectory optimization (see TranscribedProblem)
    
    Represents a continuous problem of the form
    \min_{u(t), x(t)} \int_{t=0}^{T} obj_func(x(t), u(t)) dt
    s.t. x(0) = x_init, x(T) = x_end
         \dot{x}(t) = f(x(t), u(t))

    where f is system dynamics (derivative of state)
    """
    def __init__(self):
        pass

    def T(self):
        """
        @brief      returns the Ending Time for the problem
        """
        raise Exception("Not yet implemented.")

    def x_init(self):
        """
        @brief      Returns the initial state for the problem
        """
        raise Exception("Not yet implemented.")

    def x_end(self):
        """
        @brief      Returns the ending state for the problem
        """
        raise Exception("Not yet implemented.")

    def state_dim(self):
        """
        @brief      returns the dimension of the system state
        """
        raise Exception("Not yet implemented.")

    def control_dim(self):
        """
        @brief      returns the dimension of the control input
        """
        raise Exception("Not yet implemented.")

    def f(self, state, control):
        """
        @brief      returns the derivative of the state (system dynamics)

        @param      state    np.array of length state_dim
        @param      control  np.array of length control_dim
        
        @return     np.array of length state_dim which is the controls
        """
        raise Exception("Not yet implemented.")


    def obj_func(self, state, control):
        """
        @brief      the objective function to be minimized
        @param      state    np.array of length state_dim
        @param      control  np.array of length control_dim
        
        @return     scalar value of objective function
        """
        raise Exception("Not yet implemented.")


class IntegrationType(enum.Enum):
    """
    @brief      Enum class for type of integration approximation used
    """
    TRAPEZOIDAL = 0
    HERMITE_SIMPSON = 1

class TranscribedProblem(object):
    """
    @brief      Transcribes a problem of type ProblemBase into a discrete optimization problem

    @details    Turns the continuous problem (see ProblemBase) into the following
    \min{\{u_0, \hdots, u_{N-1}\}, \{x_0, \hdots, x_{N-1}\}, \{v_0, \hdots, v_{N-1}\}} 
        \sum_{k=0}^{N-1} \int_{t_k}^{t_{k+1}} obj_func(x(t), u(t)) dt

    s.t. x_{0} = x_init, x_{N-1} = x_end
         x_{k+1} - x_{k} = \int_{t_k}^{t_{k+1}} f(x(t), u(t)) dt

    where the integrals are approximated with either the trapezoidal or hermite-simpson method

    If hermite-simpson is used, the problem gets extra decision variables u_{i+1/2} for each original u_i except the last
    

    We will now discuss how the variables for the transcribed problem is layed out.
    All decision variables will be represented by one big vector
    The vector contains all the control variables, u, in the begining, and the state variables, x, in the end.

    For the control variables portion, variables[0:control_dim] represents u_0
    and variables[control_dim:2*control_dim] represents either u_1 or u_{0+1/2} depending what kind of integration method is used
    Let the total number of control variable paramteres be state_offset.
    The state x_0 is represented as variables[state_offset:state_offset+state_dim]
    """

    def __init__(self, N, problem, int_list=None):
        """
        @brief     Constructor for TranscribedProblem
        
        @param      N           Number of state knot points to be used
        @param      problem     The original continuous problem of type ProblemBase
        @param      int_list    Represents what sort of integration to be used at each state knot point
                                This can either be None, which defaults to TRAPEZOIDAL for all knot points
                                or you can give a list (of length N-1) of IntegrationType where the i'th entry represents
                                the integration method to be used between the i'th and (i+1)'th knot point
                                For convinience, if instead of a list, a pure IntegrationType variable is passed in,
                                it will use that type for every knot point      
        """

        self.N = N
        self.problem = problem

        # validating/generating int_list
        if int_list is None:
            self.int_list = [IntegrationType.TRAPEZOIDAL for _ in range(N-1)]
        else:
            if type(int_list) == IntegrationType:
                self.int_list = [int_list for _ in range(N-1)]
            else:
                if len(int_list) != (N-1):
                    raise Exception("Collation list not right size.")
                self.int_list = int_list


        # controls_idx is a list of length N.
        # The i'th idx of the list tells us that controls_idx[i] * control_dim
        # is the idx for which x_i corresponds with in the variables vector.
        # This is to handle extra control variables per state timestep for higher order methods.
        self.controls_idx = [0]
        for collation_type in self.int_list:
            if collation_type == IntegrationType.TRAPEZOIDAL:
                self.controls_idx.append(self.controls_idx[-1] + 1)
            elif collation_type == IntegrationType.HERMITE_SIMPSON:
                self.controls_idx.append(self.controls_idx[-1] + 2)

        # this tells us where the state variables begin
        # no matter what integration method is used, 
        # the last control input will not have any additional knot points after it
        self.state_offset = self.controls_idx[-1] + 1

        # generating time steps for control knot points (and all intermediate knot points)
        dt = problem.T() / (self.N - 1)
        self.t_u = [0]
        for collation_type in self.int_list:
            if collation_type == IntegrationType.TRAPEZOIDAL:
                self.t_u.append(self.t_u[-1] + dt)
            elif collation_type == IntegrationType.HERMITE_SIMPSON:
                self.t_u.append(self.t_u[-1] + dt * 0.5)
                self.t_u.append(self.t_u[-1] + dt * 0.5)
            else:
                raise Exception("Not valid collation type.")
        self.t_u = np.array(self.t_u)

        # times for state knot points
        self.t = np.linspace(0, problem.T(), N)

        # the delta times, t_i - t_{i-1}
        self.h = self.t[1:] - self.t[:-1]

    def obj_func(self, variables):
        """
        @brief      Takes in the decision variables vector and returns the objective_function
        
        @param      variables  The decision variables
        
        @return     scalar objective function
        """
        N = self.N
        state_dim = self.problem.state_dim()
        control_dim = self.problem.control_dim()

        u = variables[0:self.state_offset*control_dim]
        x = variables[self.state_offset*control_dim:]
        obj = 0
        for i in range(N-1):
            j = self.controls_idx[i]

            x_i = x[i*state_dim:(i+1)*state_dim]
            x_ip1 = x[(i+1)*state_dim:(i+2)*state_dim]

            if self.int_list[i] == IntegrationType.TRAPEZOIDAL:
                u_i = u[j*control_dim:(j+1)*control_dim]
                u_ip1 = u[(j+1)*control_dim:(j+2)*control_dim]
                obj += self._trapezoidal(self.h[i], self.problem.obj_func, x_i, x_ip1, u_i, u_ip1)
            elif self.int_list[i] == IntegrationType.HERMITE_SIMPSON:
                u_i = u[j*control_dim:(j+1)*control_dim]
                u_iphalf = u[(j+1)*control_dim:(j+2)*control_dim]
                u_ip1 = u[(j+2)*control_dim:(j+3)*control_dim]

                obj += self._hermite_simpson(self.h[i], self.problem.obj_func, x_i, x_ip1, u_i, u_ip1, u_iphalf)
            else:
                raise Exception("Not valid collation type.")

        return obj

    def sys_constraints(self, variables):
        """
        @brief      returns the system equality constraints vector
        @details    returns a (2*state_dim + (N-1)*state_dim) vector of constraints g 
                    where the problem constraints are cast so that the solution must have g(variables) = 0
        
        @param      variables  The decision variables
        
        @return     vector of equality constraint values
        """
        N = self.N
        state_dim = self.problem.state_dim()
        control_dim = self.problem.control_dim()

        u = variables[0:self.state_offset*control_dim]
        x = variables[self.state_offset*control_dim:]

        # initial and end constraints
        constraints = np.zeros(2*state_dim + (N-1)*state_dim)
        constraints[0:state_dim] = x[0:state_dim] - self.problem.x_init()
        constraints[state_dim:2*state_dim] = x[-state_dim:] - self.problem.x_end()
        offset = (2*state_dim)


        for i in range(N-1):
            j = self.controls_idx[i]

            x_i = x[i*state_dim:(i+1)*state_dim]
            x_ip1 = x[(i+1)*state_dim:(i+2)*state_dim]

            if self.int_list[i] == IntegrationType.TRAPEZOIDAL:
                u_i = u[j*control_dim:(j+1)*control_dim]
                u_ip1 = u[(j+1)*control_dim:(j+2)*control_dim]

                constraint_val = (x_ip1 - x_i) - \
                    self._trapezoidal(self.h[i], self.problem.f, x_i, x_ip1, u_i, u_ip1)
            elif self.int_list[i] == IntegrationType.HERMITE_SIMPSON:
                u_i = u[j*control_dim:(j+1)*control_dim]
                u_iphalf = u[(j+1)*control_dim:(j+2)*control_dim]
                u_ip1 = u[(j+2)*control_dim:(j+3)*control_dim]

                constraint_val = (x_ip1 - x_i) - \
                    self._hermite_simpson(self.h[i], self.problem.f, x_i, x_ip1, u_i, u_ip1, u_iphalf)
            
            constraints[offset+(i*state_dim):offset+((i+1)*state_dim)] = constraint_val

        return constraints

    def initialize_variables(self):
        """
        @brief      Returns a decision variable vector by linearly interpolating the states
                    from x_init to x_end and setting all control inputs to 0
        """
        N = self.N
        state_dim = self.problem.state_dim()
        control_dim = self.problem.control_dim()


        num_controls = self.state_offset
        variables = np.zeros(N*state_dim + num_controls*control_dim)

        initial_states = np.linspace(self.problem.x_init(), self.problem.x_end(), N)
        offset = num_controls*control_dim
        for i in range(state_dim):
            variables[offset+i::state_dim] = initial_states[:, i]
        return variables


    def parse_variables(self, variables):
        """
        @brief      Takes a decision variable vector
                    and returns (u, x, t_u, t_x)
                    where t_u is a vector of times that correspond with the control input
                    and t_x is a vector of times that correspond with the state 
                    u is a (N by control_dim) matrix where the i'th row is the control input at time t_u[i]
                    x is a (N by state_dim) matrix where the i'th row is the state at time t_x[i]
        """
        N = self.N
        state_dim = self.problem.state_dim()
        control_dim = self.problem.control_dim()

        u = variables[0:self.state_offset*control_dim]
        x = variables[self.state_offset*control_dim:]

        u = np.reshape(u, (self.state_offset, control_dim))
        x = np.reshape(x, (N, state_dim))
        return u, x, self.t_u, self.t


    def construct_piecewise_trajectory(self, variables, return_func=True):
        """
        @brief      Constructs a piece wise trajectory from the variables

        @details    Trapezoidal approximation will yield linear control/quadratic state pieces
                    Hermite-Simpson will yield quadratic control/cubic state pieces
        
                    if return_func is True,
                    Then this function will return (u_func, x_func)
                    where u_func is a function that takes in a time, t (from 0 to problem.T),
                    and spits out a np.array of length control_dim that corresponds with u(t)
                    simlarily, x_func takes in a time, t, 
                    and spits out a np.array of length state_dim that corresponds with x(t)

                    if return_func is False,
                    it will return (t_list, u_poly, x_poly) 
                    which t_list are the time knot points, and u_poly and x_poly are lists of control and state coefficients 
                    that can be used with poly_func (see poly_func for a detailed description)

        @param      variables    The variables
        @param      return_func  if True, return a python function for the trajectory. If False, return coefficients
        """
        N = self.N
        state_dim = self.problem.state_dim()
        control_dim = self.problem.control_dim()

        u, x, t_u, t_x = self.parse_variables(variables)

        u_polynomials = []
        x_polynomials = []
        for i, collation_type in enumerate(self.int_list):
            j = self.controls_idx[i]
        
            x_i = x[i, :]
            x_ip1 = x[i+1, :]
            h = self.h[i]

            if collation_type == IntegrationType.TRAPEZOIDAL:
                u_j = u[j, :]
                u_jp1 = u[j+1, :]

                u_poly = np.zeros((2, control_dim))
                u_poly[0, :] = (u_jp1 - u_j) / h
                u_poly[1, :] = u_j

                x_poly = np.zeros((3, state_dim))
                f_i = self.problem.f(x_i, u_j)
                f_ip1 = self.problem.f(x_ip1, u_jp1)

                x_poly[0, :] = (f_ip1 - f_i) / (2 * h)
                x_poly[1, :] = f_i
                x_poly[2, :] = x_i

                u_polynomials.append(u_poly)
                x_polynomials.append(x_poly)

            elif collation_type == IntegrationType.HERMITE_SIMPSON:
                u_j = u[j, :]
                u_jphalf = u[j+1, :]
                u_jp1 = u[j+2, :]

                f_i = self.problem.f(x_i, u_j)
                f_ip1 = self.problem.f(x_ip1, u_jp1)
                x_iphalf = (0.5*(x_i + x_ip1)) + ((h/8)*(f_i - f_ip1))
                f_iphalf = self.problem.f(x_iphalf, u_jphalf)

                u_poly = np.zeros((3, control_dim))
                u_poly[0, :] = (1.0/(h**2))*(2*u_j - 4*u_jphalf + 2*u_jp1)
                u_poly[1, :] = (1.0/h)*(-3*u_j + 4*u_jphalf - u_jp1)
                u_poly[2, :] = u_j

                # NOTE: I believe Equation 4.13 in  
                # http://www.matthewpeterkelly.com/research/MatthewKelly_IntroTrajectoryOptimization_SIAM_Review_2017.pdf
                # is incorrect. the linear, quadratic, and cubic terms are off by a factor of h
                x_poly = np.zeros((4, state_dim))
                x_poly[0, :] = (1.0/(h**2))*(1.0/3)*(2*f_i - 4*f_iphalf + 2*f_ip1)
                x_poly[1, :] = (1.0/h)*(0.5)*(-3*f_i + 4*f_iphalf - f_ip1)
                x_poly[2, :] = f_i
                x_poly[3, :] = x_i

                u_polynomials.append(u_poly)
                x_polynomials.append(x_poly)
            else:
                raise Exception("Not a valid integration type.")

        if return_func:
            return partial(poly_func, t_x, u_polynomials), partial(poly_func, t_x, x_polynomials)
        else:
            return t_x, u_polynomials, x_polynomials


    def _trapezoidal(self, h, f, x_i, x_ip1, u_i, u_ip1):
        integral = 0.5*h*(f(x_i, u_i) + f(x_ip1, u_ip1))
        return integral

    def _hermite_simpson(self, h, f, x_i, x_ip1, u_i, u_ip1, u_iphalf):
        f_i = f(x_i, u_i)
        f_ip1 = f(x_ip1, u_ip1)

        x_iphalf = (0.5*(x_i + x_ip1)) + ((h/8)*(f_i - f_ip1))
        f_iphalf = f(x_iphalf, u_iphalf)
        integral = (1./6)*h*(f_i + 4*f_iphalf + f_ip1)

        return integral



def poly_func(t_list, poly_list, t):
    """
    @brief      evaluates a piecewise polynomial function (vector valued), represented by t_list and poly_list
                at a specific time, t
    
    @details    poly_list[i] corresponds with the coefficients to be used between times t_list[i] and t_list[i+1]
                an entry in poly_list is a np.array of size (p by m)
                where (p+1) is the order of the polynomial and m is the dimension of the function output
                Thus, poly_list[i][-1, :] is a np.array of size m where each entry is the constant component of the polynomial function
                for a specific dimension

                The polynomial coefficients must be defined such that they take in delta times, dt.
                where poly_list[i] will operate on dt = (t - t_list[i])
    
    @param      t_list     time knot points of length N
    @param      poly_list  coefficient list of length (N-1)
    @param      t          time to evaluate function at
    """
    assert(len(poly_list) > 0)
    assert(len(poly_list) == len(t_list)-1)

    # idx is the index into t_list such that
    # t_list[idx] <= t < t_list[idx+1]
    idx = 0
    while (t >= t_list[idx]):
        idx += 1

        # if past last segment
        # set t to be last time possible
        if (idx >= len(t_list)):
            idx = len(t_list)-1
            t = t_list[-1]
            break;
    idx -= 1

    # if we are before the first time, return the f(0)
    if idx < 0:
        return poly_list[0][-1, :]


    dt = t - t_list[idx] 
    val = np.zeros(poly_list[0].shape[1])
    for power, coeff in enumerate(reversed(poly_list[idx])):
        val += coeff * (math.pow(dt, power))

    return val






