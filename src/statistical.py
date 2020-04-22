import numpy as np
import scipy.integrate as spi
import scipy.optimize as spo


def beta_fun(t, t_thresh, beta1, beta2):
    k = 50. # code defined variable...
    hh = lambda t: 1./(1.+np.exp(-2.*k*t, dtype=np.float128))
    return beta1*hh(t_thresh-t) + beta2*hh(t-t_thresh)

def ode(vars, t, beta1, beta2, delta, ha, ksi, t_thresh, kappa, p, gamma_asym, gamma_sym, gamma_H, gamma_U, mi_H, mi_U, omega):

    S, I_asym, I_sym, E, H, U, R, D, Nw = vars # getting variables values

    beta = beta_fun(t, t_thresh, beta1, beta2)

    dS = -beta*S*(I_sym+delta*I_asym)
    dE = beta*S*(I_sym+delta*I_asym) - kappa*E
    dI_asym = (1-p)*kappa*E - gamma_asym*I_asym
    dI_sym = p*kappa*E - gamma_sym*I_sym
    dH = ha*ksi*gamma_sym*I_sym + (1-mi_U)*gamma_U*U - gamma_H*H
    dU = ha*(1-ksi)*gamma_sym*I_sym + (1-mi_H)*omega*gamma_H*H - gamma_U*U
    dR = gamma_asym*I_asym + (1-mi_H)*(1-omega)*gamma_H*H + (1-ha)*gamma_sym*I_sym
    dD = mi_H*gamma_H*H + mi_U*gamma_U*U
    dNw = p*kappa*E

    return [dS, dI_asym, dI_sym, dE, dH, dU, dR, dD, dNw]


def ode_initial_conditions(params):

    # ---------  Signature of variables unpacked by ode
    # S, I_asym, I_sym, E, H, U, R, D, Nw = vars # getting variables values

    vars0 = params[-3:]
    vars0 = np.concatenate(([1-np.sum(vars0)], vars0, [0, 0, 0, 0, 0])) # prepending resulting Susceptible & appending lasting containers: Hospitalized, UTI, Recovered, and Dead, respectively; and Nw inital value

    return vars0


def get_containers(params, predef_param, observed, meta, t):

    N = meta['pop'] # getting city population

    vars0 = ode_initial_conditions(params) # getting ode random initial conditions

    params = np.concatenate((params[:-3], predef_param)) # appending non randomized parameters

    res = spi.odeint(ode, vars0, t, args=tuple(params))

    return res*N


def cost_function(params, observed, meta, t, predef_param):

    N = meta['pop'] # unpacking state metadata

    # getting containers
    res = get_containers(params, predef_param, observed, meta, t)

    I_stipulated = res[1:,-1] # getting all integrated time series for Infections
    D_stipulated = res[1:,-2] # getting all integrated time series for Deaths

    I_observed = observed[3] # temporal observed Infected population
    D_observed = observed[1] # temporal observed Dead population


    # getting temporal deviations for specific populations
    err_I = (I_stipulated-I_observed)/np.sqrt(I_stipulated+1)
    err_D = (D_stipulated-D_observed)/np.sqrt(D_stipulated+1)

    return np.r_[err_I, err_D]


def stipulation(thr, extra_days, lsq_tries, observed, meta):

    N = meta['pop'] # unpacking state metadata
    t_lth = observed[4] # getting time series length

    m = (5, 5) # margins after beginning and before today for intervention fitting
    if t_lth-sum(m) <= 0:
        m = (m[0]+(t_lth-sum(m)-.01) , m[1]) # shifting lower bound to fit inside time interval. upper bound remain intact

    boundaries = np.array([
        [.0,           1.], # beta1
        [.0,           1.], # beta2
        [.0,           1.], # delta
        [.05,          .2], # ha
        [.48,         .53], # ksi
        [m[0], t_lth-m[1]], # t_thresh
        [0.,        10./N], # I_asym --> Initial Condition !!
        [0.,        10./N], # I_sym  --> Initial Condition !!
        [0.,        10./N]  # E      --> Initial Condition !!
    ]).T

    predef_param = (
        .25,   # kappa
        .15,   # p
        1/3.5, # gamma_asym
        1/6,   # gamma_sym
        1/9,   # gamma_H
        1/5.5, # gamma_U
        .15,   # mi_H
        .35,   # mi_U
        .14    # omega
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
