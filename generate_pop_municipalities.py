import pandas as pd
import numpy as np
import scipy.integrate as spi
from datetime import date
from xonsh_py import *
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


def get_param():

    last_date = get_last_date('output','states_summary_')

    df = pd.read_csv(f'output/states_summary_{last_date}.csv', index_col=0)

    return df.to_dict(orient='index')


def get_last_date(path, wild):

    l = lsgrep(path, [wild], with_path=False)

    dates = [x.split('_')[-1].split('.')[0] for x in l]
    dates = np.array(list(map(np.datetime64, dates)))
    return np.max(dates)


def muni_metadata():
    last_date = get_last_date('.', 'popBR_')
    df = pd.read_excel(f'popBR_{last_date}.xlsx')

    states = set(df.UF)

    meta = dict()
    for state in states:
        munis = df.loc[df.UF == state]

        meta[state] = dict()

        for muni in range(munis.shape[0]):
            row = munis.iloc[muni]
            name = row.loc['id']

            meta[state][name] = (
                row.to_dict(), # dict with keys: UF, id, Name, Ps, Pi, Pr, Pt, Lon, Lat
            )

    return meta

def generate_municipality_containers(meta, params, first_day):

    duration = np.int32(np.datetime64('today') - first_day[0])
    t = np.arange(1,duration+1)

    I_sym = first_day[1]
    N = meta[0]['Pt'] - I_sym

    vars0 = (N, 0, 0, I_sym, 0, 0, 0, 0, 0)

    par = [params[k] for k in ['beta1', 'beta2', 'delta', 'h', 'ksi', 't_thresh', 'kappa', 'p', 'gamma_asym', 'gamma_sym', 'gamma_H', 'gamma_U', 'mi_H', 'mi_U', 'omega']]

    # TODO: check usage of t_thresh
    # some cities didn't have intervention neither reached an infection time longer

    res = spi.odeint(ode, vars0, t, args=tuple(par))

    return res[-1]


def get_muni_time_series():
    # last_date = get_last_date('', 'municipios_')
    df = pd.read_csv('https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-cities-time.csv')

    df = df[df.state != 'TOTAL']

    t_series = dict()
    for state in set(df.state):
        row_cities = df.loc[df.state == state]

        ids = set(row_cities.ibgeID)

        t_series[state] = dict()

        for id in ids:
            dates = row_cities.loc[row_cities.ibgeID == id].date
            first = np.min(list(map(np.datetime64,dates.values)))

            infect = row_cities.loc[(row_cities.ibgeID == id) & (row_cities.date == str(first))].totalCases.values[0]


            t_series[state][id] = (
                first, # first date              # [0] -> first date
                infect # quantity of infected    # [1] -> infection amount
            )

    return t_series

def create_containers_cities():
    df = pd.DataFrame(columns=['ibgeID', 'state', 'S', 'E', 'I_asym', 'I_sym', 'H', 'U', 'R', 'D', 'Nw'])
    return df


def append_containers_cities(containers, meta, df):
    data = np.concatenate(([meta[0]['id']], [meta[0]['UF']], containers))
    cols = df.columns
    return df.append(dict(zip(cols,data)), ignore_index=True)


def save_containers_cities(df):
    today = np.datetime64('today')
    df.set_index('ibgeID', inplace=True)
    df.to_csv(f'output/cities_containers_{today}.csv')


def main():

    # parameters -> 'state', 'population', 'beta1', 'beta2', 'delta', 'h', 'ksi', 't_thresh', 'kappa', 'p', 'gamma_asym', 'gamma_sym', 'gamma_H', 'gamma_U', 'mi_H', 'mi_U', 'omega', 'initial_I_asym', 'initial_I_sym', 'initial_E', 'best_cost'
    params = get_param()

    # getting municipal containers populations to output
    # metadata -> UF, id, Name, Ps, Pi, Pr, Pt, Lon, Lat
    muni_meta = muni_metadata()

    # info -> first epidemic day, quantity of infected
    muni_first_day = get_muni_time_series()

    cities_df = create_containers_cities()

    for state in muni_first_day:
        par_state = params[state]

        common_cities = set(muni_meta[state].keys()) & set(muni_first_day[state].keys())

        print(f'Actual state: {state} ; cities -> {len(common_cities)}')
        for muni in common_cities:
            containers = generate_municipality_containers(muni_meta[state][muni], par_state, muni_first_day[state][muni])

            cities_df = append_containers_cities(containers, muni_meta[state][muni], cities_df)

    save_containers_cities(cities_df)


main()
