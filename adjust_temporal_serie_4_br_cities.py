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

    N = meta[0] # getting city population

    vars0 = ode_initial_conditions(params) # getting ode random initial conditions

    params = np.concatenate((params[:-3], predef_param)) # appending non randomized parameters

    res = spi.odeint(ode, vars0, t, args=tuple(params))

    return res*N


def cost_function(params, observed, meta, t, predef_param):

    N = meta[0] # unpacking state metadata

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


def stipulation(thr, extra_days, observed, meta):

    N = meta[0] # unpacking state metadata
    t_lth = observed[4] # getting time series length

    # thrs -> ini + 5, fin - 5
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

    # for tries in range(25):
    for tries in range(1):

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


def observed_data():

    raw = pd.read_csv('https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-cities-time.csv')

    raw = raw[raw.state != 'TOTAL'] # removing summarized rows

    ids = set(raw.ibgeID) # getting unique cities

    cities = dict()

    for id in ids:
        city_lines = raw[raw.ibgeID == id]

        # skipping time series with less than 10 points
        if city_lines.shape[0] < 10: continue

        cities[id] = (
            city_lines.date.values,       # [0] -> date       vector
            city_lines.deaths.values,     # [1] -> deaths     vector
            city_lines.newCases.values,   # [2] -> newCases   vector
            city_lines.totalCases.values, # [3] -> totalCases vector
            city_lines.shape[0]           # [4] -> length of time series
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

def save_time_series(params, predef_param, containers, meta, extra_days):

    _, ibgeID, city, state = meta

    beta1, beta2, delta, ha, ksi, t_thresh = params[:-3]
    kappa, p, gamma_asym, gamma_sym, gamma_H, gamma_U, mi_H, mi_U, omega = predef_param


    outpath = f'{out_folder()}/{state}/{city}/time_series_extraDays={extra_days}.csv'

    np.savetxt(outpath, containers, delimiter=',', newline='\n', header='S,I_A,I_S,E,H,U,R,D,Nw', fmt=['%d','%d','%d','%d','%d','%d','%d','%d','%d'])

def plot_compare(params, predef_param, containers, observed, meta, extra_days):

    N, _, city, state = meta # unpacking state metadata

    y = containers[1:]
    t = np.arange(1,1+y.shape[0]) # timespan based on days length

    deaths = np.nonzero(observed[1])[0]
    ini = deaths[0] if len(deaths) else -1
    dates = observed[0]

    # plot step
    fig = plt.figure(0)

    vl = plt.axvline(params[5], c='r', ls='--', label='Start of Intervention')
    vL = plt.axvline(t.shape[0]-extra_days, c='#6d4848', ls=':', label='Time Series Limit')

    # plotting Deaths
    if ini > 0:
        od, = plt.plot(t[ini:-extra_days], observed[1][ini:], 'o', mfc='none', mec='#f47f18', label='Observed Death', ms=4, mew=1.)
        sd, = plt.plot(t[ini:], y[ini:,-2], '-r', label='Stipulated Death', lw=1)

    si, = plt.plot(t, y[:,-1], '-b', label='Stipulated Infection', lw=1)
    oi, = plt.plot(t[:-extra_days], observed[3], '.k', label='Observed Infection', ms=5)


    # r, = plt.plot(t, y[:,-3], '-g', label='Stipulated Recovery', lw=1)
    # u, = plt.plot(t, y[:,-4], '-', c='#98b7e2', label='Stipulated UTI', lw=1)
    # h, = plt.plot(t, y[:,-5], '-', c='#ed84b8', label='Stipulated Hospitalization', lw=1)
    # e, = plt.plot(t, y[:,-6], '-', c='#25f9ef', label='Stipulated Exposed', lw=1)
    # i, = plt.plot(t, y[:,-7], '-k', label='Stipulated Symptomatic', lw=1)
    # a, = plt.plot(t, y[:,-8], '-y', label='Stipulated Asymptomatic', lw=1)
    # s, = plt.plot(t, y[:,-9], '-', c='#f925eb', label='Stipulated Susceptible', lw=1)


    plt.legend(loc='upper left', fontsize=5)

    plt.xlabel('Date', fontsize=17)
    plt.ylabel('Reported cases', fontsize=17)

    plt.title(f'Covid-19 - {city}/{state}')

    l = np.int32(np.floor(np.linspace(0, dates.size-1, 10)))
    L = np.int32(np.ceil(np.linspace(0, extra_days, 3)[1:]))
    ex_dates = [str(np.datetime64(dates[-1])+np.timedelta64(x,'D')) for x in L]

    ticks = np.r_[t[l],t[L-extra_days-1]]
    lbls  = np.r_[dates[l],ex_dates]
    tks = plt.xticks(ticks, lbls, rotation=45, rotation_mode='anchor', ha='right')
    tks[1][-1].set_color('r') # setting stipulated days as red
    tks[1][-2].set_color('r') # setting stipulated days as red

    plt.tight_layout()

    outpath = f'{out_folder()}/{state}/{city}'
    mkdir_p([outpath])
    plt.savefig(f'{outpath}/english.svg',dpi=600)

    # relabeling in portuguese
    # plotting Deaths
    if ini > 0:
        sd.set_label('Mortes Estipuladas')
        od.set_label('Mortes Reportadas')
    si.set_label('Infecções Estipuladas')
    oi.set_label('Infecções Reportadas')
    vl.set_label('Início da Intervenção')
    vL.set_label('Fim da Série Temporal')
    plt.legend(loc='upper left', fontsize=10)
    plt.xlabel('Data', fontsize=17)
    plt.ylabel('Casos Reportados', fontsize=17)
    plt.title(f'Covid-19 - {city}/{state}')
    plt.savefig(f'{outpath}/portuguese.svg',dpi=600)

    plt.close(fig)


def param_df_out(data, observations, cities):

    data = np.vstack(data)

    df = pd.DataFrame(data, columns=['ibgeID','city','state','population','series_length','best_cost','beta1','beta2','delta','h','ksi','t_thresh','kappa','p','gamma_asym','gamma_sym','gamma_H','gamma_U','mi_H','mi_U','omega','initial_I_asym','initial_I_sym','initial_E','S','I_asym','I_sym','E','H','U','R','D','Nw'])

    id = cities[0] # getting any city (suposing everyone has the same last date)
    last_day = observations[id][0][-1] # getting last date

    outpath = f'{out_folder()}'
    mkdir_p([f'{outpath}'])

    df.set_index('ibgeID', inplace=True)
    df.sort_index(inplace=True)
    df.to_csv(f'{outpath}/cities_summary_{last_day}.csv')


def return_put(params, predef_param, best_cost, containers, meta, t_lth):

    N, id, city, state = meta # unpacking state metadata

    data = np.concatenate(([id, city, state, N, t_lth, best_cost], params[:-3], predef_param, params[-3:], containers))

    return data


def get_last_date(path, wild):

    l = lsgrep(path, [wild], with_path=False)

    dates = [x.split('_')[-1].split('.')[0] for x in l]
    dates = np.array(list(map(np.datetime64, dates)))
    return np.max(dates)

def out_folder():
    return f'output/{np.datetime64("today")}'


def hard_work(q, thr, cities, city_meta, observed):
    extra_days = 7

    c = 0; tot = len(cities)
    # stipulation per state
    for city in cities:
        c+=1
        print(f'{thr}:[{c}/{tot}] Started {city}, {city_meta[city][2]}/{city_meta[city][3]}...\n')

        params, predef_param, best_cost, containers = stipulation(thr, extra_days, observed[city], city_meta[city])

        plot_compare(params, predef_param, containers, observed[city], city_meta[city], extra_days)

        save_time_series(params, predef_param, containers, city_meta[city], extra_days)

        # adding numpy row to queue
        q.put(return_put(params, predef_param, best_cost, containers[-1-extra_days], city_meta[city], observed[city][4]))


def main():

    # collecting data
    observed = observed_data()
    city_meta = city_metadata()

    # TODO: verify if there are unmatching state keys between previous dictionaries

    cities = np.array(list(set(observed.keys()) & set(city_meta.keys())))
    # cities = cities[:11]
    cities = np.array([2927408])
    # cities = np.array([4305454])
    tot = cities.shape[0]
    print(f'{tot} cities to be processed...\n')

    n_thread = 1
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
    while True:
        if not que.empty():
            res.append(que.get())

        a = sum([tt.is_alive() for tt in tts])
        if not a: break
        time.sleep(2)


    for tt in tts:
        tt.join()
        tt.close()

    # outputing state parameters
    param_df_out(res, observed, cities)


main()
