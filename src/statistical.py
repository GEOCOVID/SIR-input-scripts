import numpy as np
import scipy.integrate as spi
import scipy.optimize as spo
from .input import get_state_params

def ode(vars, t, beta, gamma):

    S, I, R, Nw = vars # getting variables values

    dS = -beta*S*I
    dI = beta*S*I - gamma*I
    dR = gamma*I

    dNw = beta*S*I

    return [dS, dI, dR, dNw]


def ode_initial_conditions(params):

    # ---------  Signature of variables unpacked by ode
    # S, I, R, Nw = vars # getting variables values

    vars0 = np.array([1-params[-1],params[-1],0,0])

    return vars0


def get_containers(params, predef_param, observed, meta, t):

    N = meta['pop'] # getting city population

    vars0 = ode_initial_conditions(params) # getting ode random initial conditions

    params = np.concatenate((params[:-1], predef_param))

    res = spi.odeint(ode, vars0, t, args=tuple(params))

    return res*N


def cost_function(params, observed, meta, t, predef_param):

    # getting containers
    res = get_containers(params, predef_param, observed, meta, t)

    I_stipulated = res[1:,-1] # getting all integrated time series for Infections
    I_observed = observed[3] # temporal observed Infected population

    # getting temporal deviations for specific populations
    err_I = (I_stipulated-I_observed)/np.sqrt(I_stipulated+1)

    return np.r_[err_I]


def stipulation(thr, extra_days, lsq_tries, observed, meta):

    N = meta['pop'] # unpacking state metadata
    t_lth = observed[4] # getting time series length

    boundaries = np.array([
        [.0,        1.], # beta
        [0.,     10./N], # I --> Initial Condition !!
    ]).T

    predef_param = (
        1/6, # gamma
    )

    t = np.arange(0,1+t_lth) # timespan based on days length

    best_cost   = np.inf  # holding minimized cost set of parameters
    best_params = (None,) # tuple holding set of best parameters

    for tries in range(lsq_tries):

        params0 = np.random.rand(boundaries.shape[1])
        params0 = boundaries[0] + params0*(boundaries[1]-boundaries[0])

        res = spo.least_squares(cost_function, params0, bounds=boundaries, kwargs={'observed':observed, 'meta':meta, 't':t, 'predef_param':predef_param})

        if res.status > 0 and best_cost > res.cost: # accepting only better converged parameters
            best_cost   = res.cost
            best_params = res.x
            # print(f'[{thr}]:{meta[2]}/{meta[3]} Found cost {best_cost} with params:\n{best_params}\n')


    t = np.arange(0,1+extra_days+t_lth) # timespan based on days length and a week ahead
    # getting containers
    containers = get_containers(best_params, predef_param, observed, meta, t)

    return best_params, predef_param, best_cost, containers


def stipulation_city(thr, extra_days, lsq_tries, observed, meta):

    best_params = get_state_params(meta)
    predef_param = (
        best_params[1], # getting state gamma
    )

    N = meta['pop'] # unpacking state metadata
    # fixing initial condition for city's first case instead of state's one
    # and removing predefined params
    best_params = np.array([best_params[0], observed[3][0]/N])
    # best_params = np.array([best_params[0], best_params[2]])

    t_lth = observed[4] # getting time series length

    # t = np.arange(0,1+extra_days+t_lth) # timespan based on days length and a week ahead
    t = np.arange(1,1+extra_days+t_lth) # timespan based on days length and a week ahead

    # getting containers
    containers = get_containers(best_params, predef_param, observed, meta, t)
    # prepending data before initial report
    containers = np.concatenate(([[N,0,0,0]], containers))

    best_cost = np.nan

    return best_params, predef_param, best_cost, containers
