import numpy as np
import pandas as pd
from .xonsh_py import *


# def out_folder():
#     return f'output/{np.datetime64("today")}'
def out_folder():
    return f'output/{out_folder.today}'

def check_done_work(meta):
    city = meta['name']; state = meta['state']

    outpath = f'{out_folder()}/{state}{"/"+city if city else ""}'

    return existOldData(f'{outpath}/summary.csv')


def return_put(params, predef_param, best_cost, containers, meta, t_lth):

    N = meta['pop']; state = meta['state']

    start = [meta[x] for x in ['id', 'name'] if meta[x]]

    data = np.concatenate((start+[state, N, t_lth, best_cost], params[:-3], predef_param, params[-3:], containers))

    return data


def param_df_out_single(data, meta):

    city = meta['name']; state = meta['state']

    data = np.vstack([data])

    start = [y for y,x in zip(['ibgeID','city'],['id', 'name']) if meta[x]]

    df = pd.DataFrame(data, columns=start+['state','population','series_length','best_cost','beta1','beta2','delta','ha','gamma_H','gamma_U','t_thresh','kappa','p','gamma_asym','gamma_sym','ksi','mi_H','mi_U','omega_H','omega_U','initial_I_asym','initial_I_sym','initial_E','S','I_asym','I_sym','E','H','U','R','D','Nw'])

    # getting last date
    today = out_folder.today

    outpath = f'{out_folder()}/{state}{"/"+city if city else ""}'
    mkdir_p([f'{outpath}'])

    df.set_index('ibgeID' if meta['id'] else 'state', inplace=True)
    df.sort_index(inplace=True)
    df.to_csv(f'{outpath}/summary.csv')


def param_df_out(data, observations, meta):

    data = np.vstack(data)

    start = [y for y,x in zip(['ibgeID','city'],['id', 'name']) if meta[x]]

    df = pd.DataFrame(data, columns=start+['state','population','series_length','best_cost','beta1','beta2','delta','ha','gamma_H','gamma_U','t_thresh','kappa','p','gamma_asym','gamma_sym','ksi','mi_H','mi_U','omega_H','omega_U','initial_I_asym','initial_I_sym','initial_E','S','I_asym','I_sym','E','H','U','R','D','Nw'])

    # getting any city (suposing everyone has the same last date)
    # id = meta['id'] if meta['id'] else meta['state']
    # last_day = observations[id][0][-1] # getting last date
    today = out_folder.today

    outpath = f'{out_folder()}'
    mkdir_p([f'{outpath}'])

    df.set_index('ibgeID' if meta['id'] else 'state', inplace=True)
    df.sort_index(inplace=True)
    df.to_csv(f'{outpath}/{"cities" if meta["id"] else "states"}_summary_{today}.csv')


def save_time_series(params, predef_param, containers, meta, extra_days):

    city = meta['name']; state = meta['state']

    outpath = f'{out_folder()}/{state}{"/"+city if city else ""}/time_series_extraDays={extra_days}.csv'

    np.savetxt(outpath, containers, delimiter=',', newline='\n', header='S,I_A,I_S,E,H,U,R,D,Nw', fmt=['%d','%d','%d','%d','%d','%d','%d','%d','%d'])


def merge_output(cities, observed, meta):

    l = []
    for id in cities:

        city  = meta[id]['name']
        state = meta[id]['state']

        outpath = f'{out_folder()}/{state}{"/"+city if city else ""}'
        df = pd.read_csv(f'{outpath}/summary.csv')

        l.append(df)


    today = out_folder.today

    df = pd.concat(l)
    df.to_csv(f'{out_folder()}/{"cities" if meta[cities[0]]["id"] else "states"}_summary_{today}.csv', index=False)
