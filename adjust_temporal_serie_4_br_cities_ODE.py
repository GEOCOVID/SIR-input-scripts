# import concurrent.futures as cf
# import threading
import time
import multiprocessing as mp
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

    N = meta[0] # unpacking state metadata
    
    # getting containers
    res = get_containers(params, predef_param, observed, meta, t)

    I_stipulated = res[1:,-1] # getting all integrated time series for Infections
    D_stipulated = res[1:,-2] # getting all integrated time series for Deaths

    I_observed = observed[3] # temporal observed Infected population
    D_observed = observed[1] # temporal observed Dead population


    # getting temporal deviations for specific populations
    err_I = (N*I_stipulated-I_observed)/np.sqrt(N*I_stipulated+1)
    err_D = (N*D_stipulated-D_observed)/np.sqrt(N*D_stipulated+1)

    return np.r_[err_I, err_D]


def stipulation(thr, observed, meta):

    N = meta[0] # unpacking state metadata
    thresh_ub = len(observed[0]) # last day of time series

    # thrs -> ini + 5, fin - 5

    boundaries = np.array([
        [.0,        1.], # beta1
        [.0,        1.], # beta2
        [.0,        1.], # delta
        [.0,       .15], # ha
        [.2,        .5], # ksi
        [ 1, thresh_ub], # t_thresh
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

    # for tries in range(40):
    for tries in range(5):

        params0 = np.random.rand(boundaries.shape[1])
        params0 = boundaries[0] + params0*(boundaries[1]-boundaries[0])

        res = spo.least_squares(cost_function, params0, bounds=boundaries, kwargs={'observed':observed, 'meta':meta, 't':t, 'predef_param':predef_param})

        if res.status > 0 and best_cost > res.cost: # accepting only better converged parameters
            best_cost   = res.cost
            best_params = res.x
            # print(f'[{thr}]:{meta[2]}/{meta[3]} Found cost {best_cost} with params:\n{best_params}\n')


    # getting containers
    containers = get_containers(best_params, predef_param, observed, meta, t)

    return best_params, predef_param, best_cost, containers[-1]

def get_containers(params, predef_param, observed, meta, t):

    vars0 = ode_initial_conditions(params) # getting ode random initial conditions

    params = np.concatenate((params[:-3], predef_param)) # appending non randomized parameters

    res = spi.odeint(ode, vars0, t, args=tuple(params))

    return res


def observed_data():

    raw = pd.read_csv('https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-cities-time.csv')

    raw = raw[raw.state != 'TOTAL'] # removing summarized rows

    ids = set(raw.ibgeID) # getting unique cities

    cities = dict()

    for id in ids:
        city_lines = raw[raw.ibgeID == id]

        # skipping time series with less than 7 points
        if city_lines.shape[0] < 7: continue

        cities[id] = (
            city_lines.date.values,       # [0] -> date       vector
            city_lines.deaths.values,     # [1] -> deaths     vector
            city_lines.newCases.values,   # [2] -> newCases   vector
            city_lines.totalCases.values, # [3] -> totalCases vector
        )

    return cities


def city_metadata():
    last_date = get_last_date('.', 'popBR_')
    df = pd.read_excel(f'popBR_{last_date}.xlsx', index_col=1)

    metadata = dict()
    for id in df.index:
        city_row = df.loc[id]
        metadata[id] = (
            city_row.Pt,   # [0] -> N population scalar
            id,            # [1] -> city's ibgeID
            city_row.Name, # [2] -> city's name
            city_row.UF    # [3] -> state
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

    N, _, city, state = meta # unpacking state metadata

    deaths = np.nonzero(observed[1])[0]
    ini = deaths[0] if len(deaths) else -1
    dates = observed[0]

    # plot step
    fig = plt.figure(0)

    # plotting Deaths
    if ini > 0:
        plt.plot(t[1+ini:], N*y[ini:,-2], '-r', label='Stipulated Death', lw=1)
        plt.plot(t[1+ini:], observed[1][ini:], '.k', label='Observed Death', ms=5)

    plt.plot(t[1:], N*y[:,-1], '-b', label='Stipulated Infection', lw=1)
    plt.plot(t[1:], observed[3], '.k', label='Observed Infection', ms=5)

    plt.axvline(params[5], c='r', ls='--', label='Start of Intervention')

    plt.legend(loc='upper left', fontsize=10)

    plt.xlabel('Date', fontsize=17)
    plt.ylabel('Reported cases', fontsize=17)

    plt.suptitle(f'Covid-19 - {city}/{state}')

    l = np.int32(np.floor(np.linspace(0, dates.size-1, 10)))
    plt.xticks(t[l], dates[l], rotation=45, rotation_mode='anchor', ha='right')

    outpath = f'{out_folder()}/{state}/{city}'
    mkdir_p([outpath])
    plt.savefig(f'{outpath}/english.svg',dpi=600)

    plt.close(fig)


def plot_portuguese(t, y, params, observed, meta):

    N, _, city, state = meta # unpacking state metadata

    deaths = np.nonzero(observed[1])[0]
    ini = deaths[0] if len(deaths) else -1
    dates = observed[0]

    # plot step
    fig = plt.figure(0)

    # plotting Deaths
    if ini > 0:
        plt.plot(t[1+ini:], N*y[ini:,-2], '-r', label='Mortes Estipuladas', lw=1)
        plt.plot(t[1+ini:], observed[1][ini:], '.k', label='Mortes Reportadas', ms=5)

    plt.plot(t[1:], N*y[:,-1], '-b', label='Infecções Estipuladas', lw=1)
    plt.plot(t[1:], observed[3], '.k', label='Infecções Reportadas', ms=5)

    plt.axvline(params[5], c='r', ls='--', label='Início da Intervenção')

    plt.legend(loc='upper left', fontsize=10)

    plt.xlabel('Data', fontsize=17)
    plt.ylabel('Casos Reportados', fontsize=17)

    plt.suptitle(f'Covid-19 - {city}/{state}')

    l = np.int32(np.floor(np.linspace(0, dates.size-1, 10)))
    plt.xticks(t[l], dates[l], rotation=45, rotation_mode='anchor', ha='right')

    outpath = f'{out_folder()}/{state}/{city}'
    mkdir_p([outpath])
    plt.savefig(f'{outpath}/portuguese.svg',dpi=600)

    plt.close(fig)


def param_df_out(data, observations, cities):

    data = np.vstack(data)

    df = pd.DataFrame(data, columns=['ibgeID','city','state','population','best_cost','beta1','beta2','delta','h','ksi','t_thresh','kappa','p','gamma_asym','gamma_sym','gamma_H','gamma_U','mi_H','mi_U','omega','initial_I_asym','initial_I_sym','initial_E','S','E','I_asym','I_sym','H','U','R','D','Nw'])

    id = cities[0] # getting any city (suposing everyone has the same last date)
    last_day = observations[id][0][-1] # getting last date

    outpath = f'{out_folder()}'
    mkdir_p([f'{outpath}'])

    df.set_index('ibgeID', inplace=True)
    df.sort_index(inplace=True)
    df.to_csv(f'{outpath}/cities_summary_{last_day}.csv')


def get_last_date(path, wild):

    l = lsgrep(path, [wild], with_path=False)

    dates = [x.split('_')[-1].split('.')[0] for x in l]
    dates = np.array(list(map(np.datetime64, dates)))
    return np.max(dates)

def out_folder():
    return f'output/{np.datetime64("today")}'


def hard_work(q, thr, cities, city_meta, observed):

    c = 0; tot = len(cities)
    # stipulation per state
    for city in cities:
        c+=1
        print(f'{thr}:[{c}/{tot}] Started {city}, {city_meta[city][2]}/{city_meta[city][3]}...\n')

        params, predef_param, best_cost, containers = stipulation(thr, observed[city], city_meta[city])

        plot_compare(params, predef_param, observed[city], city_meta[city])

        # adding numpy row to queue
        q.put(return_put(params, predef_param, best_cost, containers, city_meta[city]))


def return_put(params, predef_param, best_cost, containers, meta):

    N, id, city, state = meta # unpacking state metadata

    data = np.concatenate(([id, city, state, N, best_cost], params[:-3], predef_param, params[-3:], containers))

    return data


def main():

    # collecting data
    observed = observed_data()
    city_meta = city_metadata()

    # TODO: verify if there are unmatching state keys between previous dictionaries

    cities = np.array(list(set(observed.keys()) & set(city_meta.keys())))
    cities = cities[:6]
    tot = cities.shape[0]
    print(f'{tot} cities to be processed...\n')

    n_thread = 4
    l = tot//n_thread
    r = tot%n_thread

    tts = []
    que = mp.Queue()
    for thr in range(n_thread):
        if thr < r:
            beg = thr*(l+1)
            end = beg + l + 1
        else:
            beg = thr*l + r
            end = beg + l

        tt = mp.Process(target=hard_work, args=(que, thr, cities[beg:end], city_meta, observed))

        tts.append(tt)
        tt.start()


    # concat each thread work
    res = []
    # while not que.empty():
    #     res.append(que.get())
    while True:
        if not que.empty():
            res.append(que.get())

        a = sum([tt.is_alive() for tt in tts])
        if a:
            print(f'tem thread ativa! {a}')
        else:
            break
        time.sleep(2)


    for tt in tts:
        tt.join()
        tt.close()

    # outputing state parameters
    param_df_out(res, observed, cities)


main()
