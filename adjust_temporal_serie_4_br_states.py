from xonsh_py import *
import pandas as pd
import numpy as np
import scipy.integrate as spi
import scipy.optimize as spo
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("seaborn-darkgrid")
from sys import exit

def beta_fun(t, t_thresh, beta1, beta2):
    k = 50. # code defined variable...
    hh = lambda t: 1./(1.+np.exp(-2.*k*t, dtype=np.float128))
    return beta1*hh(t_thresh-t) + beta2*hh(t-t_thresh)

def ode(vars, t, beta1, beta2, delta, ha, ksi, t_thresh, kappa, p, gamma_asym, gamma_sym, gamma_H, gamma_U, mi_H, mi_U, omega):

    S, E, I_asym, I_sym, H, U, R, D, Nw = vars # getting variables values

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

    return [dS, dE, dI_asym, dI_sym, dH, dU, dR, dD, dNw]


def ode_initial_conditions(params):

    # ---------  Signature of variables unpacked by ode
    # S, E, I_asym, I_sym, H, U, R, D, Nw = vars # getting variables values

    vars0 = params[-3:]
    vars0 = np.concatenate(([1-np.sum(vars0)], vars0, [0, 0, 0, 0, 0])) # prepending resulting Susceptible & appending lasting containers: Hospitalized, UTI, Recovered, and Dead, respectively; and Nw inital value

    return vars0


def cost_function(params, observed, meta, t, predef_param):

    N, _ = meta # unpacking state metadata

    vars0 = ode_initial_conditions(params) # getting ode random initial conditions

    params = np.concatenate((params[:-3], predef_param)) # appending non randomized parameters

    res = spi.odeint(ode, vars0, t, args=tuple(params))

    I_stipulated = res[1:,-1] # getting all integrated time series for Infections
    D_stipulated = res[1:,-2] # getting all integrated time series for Deaths

    I_observed = observed[3] # temporal observed Infected population
    D_observed = observed[1] # temporal observed Dead population


    # getting temporal deviations for specific populations
    err_I = (N*I_stipulated-I_observed)/np.sqrt(N*I_stipulated+1)
    err_D = (N*D_stipulated-D_observed)/np.sqrt(N*D_stipulated+1)

    return np.r_[err_I, err_D]


def stipulation(observed, meta):

    N, _      = meta # unpacking state metadata
    thresh_ub = len(observed[0]) # last day of time series

    # thrs -> ini + 5, fin - 5

    boundaries = np.array([
        [.0,        1.], # beta1
        [.0,        1.], # beta2
        [.0,        1.], # delta
        [.0,       .15], # ha
        [.2,        .5], # ksi
        [ 1, thresh_ub], # t_thresh # até o dia atual???????   ou até final de março??? planilhas com datas???
        [0.,     10./N], # I_asym --> Initial Condition !!
        [0.,     10./N], # I_sym  --> Initial Condition !!
        [0.,     10./N]  # E      --> Initial Condition !!
    ]).T

    predef_param = (
        .25,   # kappa
        .15,   # p
        .2,    # gamma_asym
        .2,    # gamma_sym
        .1,    # gamma_H
        .1,    # gamma_U
        .2,    # mi_H
        .55,   # mi_U
        .04    # omega
    )

    t = np.arange(0,1+len(observed[0])) # timespan based on days length

    best_cost   = np.inf  # holding minimized cost set of parameters
    best_params = (None,) # tuple holding set of best parameters

    for tries in range(40):

        params0 = np.random.rand(boundaries.shape[1])
        params0 = boundaries[0] + params0*(boundaries[1]-boundaries[0])

        res = spo.least_squares(cost_function, params0, bounds=boundaries, kwargs={'observed':observed, 'meta':meta, 't':t, 'predef_param':predef_param})

        if res.status > 0 and best_cost > res.cost: # accepting only better converged parameters
            best_cost   = res.cost
            best_params = res.x
            print(f'Found cost {best_cost} with params:\n{best_params}\n')

            # if best_cost < 50: break

        # if tries % 250 == 0 and not tries == 0: print(f'Tried {tries} times...')

    return best_params, predef_param, best_cost


def observed_data():
    raw = pd.read_csv('https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-states.csv')

    # USANDO APENAS COLUNAS date, deaths, totalCases e newCases da tabela do wesley
    raw.loc[raw.state == 'TOTAL','state'] = 'BR'

    states = set(raw.state) # getting unique regions/states

    raw_state = dict()

    for state in states:
        state_lines = raw[raw.state == state]

        raw_state[state] = (
            state_lines.date.values,       # [0] -> date       vector
            state_lines.deaths.values,     # [1] -> deaths     vector
            state_lines.newCases.values,   # [2] -> newCases   vector
            state_lines.totalCases.values, # [3] -> totalCases vector
        )

    return raw_state


def state_metadata():
    # TODO: get latest file based on date
    meta = pd.read_csv('estado_sumario_12-04-20.csv', index_col=0)
    meta.rename(index={'TOTAL':'BR'}, inplace=True)

    metadata = dict()
    for state in meta.index:
        metadata[state] = (
            meta.loc[state,'populacao'], # [0] -> N population scalar
            state                        # [1] -> state's name
        )

    return metadata


def plot_compare(params, predef_param, observed, meta):

    vars0 = ode_initial_conditions(params) # getting ode random initial conditions

    params = np.concatenate((params[:-3], predef_param)) # appending non randomized parameters

    t = np.arange(0,1+len(observed[0])) # timespan based on days length

    y = spi.odeint(ode, vars0, t, args=tuple(params))[1:]

    plot_english(t, y, params, observed, meta)
    plot_portuguese(t, y, params, observed, meta)


def plot_english(t, y, params, observed, meta):

    N, state = meta # unpacking state metadata

    ini = np.nonzero(observed[1])[0][0]
    dates = observed[0]

    # plot step
    fig = plt.figure(0)

    # plotting Deaths
    plt.plot(t[1+ini:], N*y[ini:,-2], '-r', label='Stipulated Death', lw=1)
    plt.plot(t[1+ini:], observed[1][ini:], '.k', label='Observed Death', ms=5)

    plt.plot(t[1:], N*y[:,-1], '-b', label='Stipulated Infection', lw=1)
    plt.plot(t[1:], observed[3], '.k', label='Observed Infection', ms=5)

    plt.axvline(params[5], c='r', ls='--', label='Start of Intervention')

    plt.legend(loc='upper left', fontsize=10)

    plt.xlabel('Date', fontsize=17)
    plt.ylabel('Reported cases', fontsize=17)

    plt.suptitle(f'Covid-19 - {state}')

    l = np.int32(np.floor(np.linspace(0, dates.size-1, 10)))
    plt.xticks(t[l], dates[l], rotation=45, rotation_mode='anchor', ha='right')

    outpath = f'output/{state}'
    mkdir_p([outpath])
    plt.savefig(f'{outpath}/english.svg',dpi=600)

    plt.close(fig)


def plot_portuguese(t, y, params, observed, meta):

    N, state = meta # unpacking state metadata

    ini = np.nonzero(observed[1])[0][0]
    dates = observed[0]

    # plot step
    fig = plt.figure(0)

    # plotting Deaths
    plt.plot(t[1+ini:], N*y[ini:,-2], '-r', label='Mortes Estipuladas', lw=1)
    plt.plot(t[1+ini:], observed[1][ini:], '.k', label='Mortes Reportadas', ms=5)

    plt.plot(t[1:], N*y[:,-1], '-b', label='Infecções Estipuladas', lw=1)
    plt.plot(t[1:], observed[3], '.k', label='Infecções Reportadas', ms=5)

    plt.axvline(params[5], c='r', ls='--', label='Início da Intervenção')

    plt.legend(loc='upper left', fontsize=10)

    plt.xlabel('Data', fontsize=17)
    plt.ylabel('Casos Reportados', fontsize=17)

    plt.suptitle(f'Covid-19 - {state}')

    l = np.int32(np.floor(np.linspace(0, dates.size-1, 10)))
    plt.xticks(t[l], dates[l], rotation=45, rotation_mode='anchor', ha='right')

    outpath = f'output/{state}'
    mkdir_p([outpath])
    plt.savefig(f'{outpath}/portuguese.svg',dpi=600)

    plt.close(fig)


def param_df_create():
    df = pd.DataFrame(columns=['state','population','beta1','beta2','delta','h','ksi','t_thresh','kappa','p','gamma_asym','gamma_sym','gamma_H','gamma_U','mi_H','mi_U','omega','initial_I_asym','initial_I_sym','initial_E','best_cost'])

    return df


def param_df_app(params, predef_param, best_cost, meta, df):

    N, state = meta # unpacking state metadata

    data = np.concatenate(([state], [N], params[:-3], predef_param, params[-3:], [best_cost]))
    cols = df.columns

    return df.append(dict(zip(cols,data)), ignore_index=True)

def param_df_out(df, observations):

    last_day = observations['BR'][0][-1] # getting last date

    outpath = f'output/'
    mkdir_p([f'{outpath}'])

    df.set_index('state', inplace=True)
    df.to_csv(f'{outpath}/states_summary_{last_day}.csv')


def main():

    # collecting data
    observed = observed_data()
    state_meta = state_metadata()

    # TODO: verify if there are unmatching state keys between previous dictionaries

    par_df = param_df_create()

    # stipulation per state
    for state in observed:
    # for state in ['BA']:
       print(f'Started {state}...\n')
       params, predef_param, best_cost = stipulation(observed[state], state_meta[state])

       plot_compare(params, predef_param, observed[state], state_meta[state])

       par_df = param_df_app(params, predef_param, best_cost, state_meta[state], par_df)

    # outputing state parameters
    param_df_out(par_df, observed)


main()
