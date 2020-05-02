import operator
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


def return_put(params, predef_param, best_cost, containers, meta, I_rep, t_lth):

    N = meta['pop']; state = meta['state']

    start = [meta[x] for x in ['id', 'name'] if meta[x]]

    data = np.concatenate((start+[state, N, I_rep, t_lth, best_cost], params[:-1], predef_param, params[-1:], containers))

    return data


def param_df_out_single(data, meta):

    city = meta['name']; state = meta['state']

    data = np.vstack([data])

    start = [y for y,x in zip(['ibgeID','city'],['id', 'name']) if meta[x]]

    df = pd.DataFrame(data, columns=start+['state','population','I_reported','series_length','best_cost','beta','gamma','initial_I','S','I','R','Nw'])

    # getting last date
    today = out_folder.today

    outpath = f'{out_folder()}/{state}{"/"+city if city else ""}'
    mkdir_p([f'{outpath}'])

    df.set_index('ibgeID' if meta['id'] else 'state', inplace=True)
    df.sort_index(inplace=True)
    df.to_csv(f'{outpath}/summary.csv')


def save_time_series(params, predef_param, containers, meta, extra_days):

    city = meta['name']; state = meta['state']

    outpath = f'{out_folder()}/{state}{"/"+city if city else ""}/time_series_extraDays={extra_days}.csv'

    np.savetxt(outpath, containers, delimiter=',', newline='\n', header='S,I,R,Nw', fmt=['%d','%d','%d','%d'])


def merge_output(cities, cities_nofit, observed_nofit, meta):

    l = []
    for id in cities:

        city  = meta[id]['name']
        state = meta[id]['state']

        outpath = f'{out_folder()}/{state}{"/"+city if city else ""}'
        df = pd.read_csv(f'{outpath}/summary.csv')

        l.append(df)


    today = out_folder.today

    df = pd.concat(l) if l else pd.DataFrame()

    df_nofit = create_nofit_rows(cities, cities_nofit, observed_nofit, meta)

    df = df.append(df_nofit, ignore_index=True, sort=False)

    outpath = f'{out_folder()}'
    mkdir_p([outpath])
    cid = cities_nofit[0] if len(cities_nofit) else cities[0]
    df.to_csv(f'{outpath}/{"cities" if meta[cid]["id"] else "states"}_summary_{today}.csv', index=False)


def create_nofit_rows(cities, cities_nofit, observed_nofit, meta):
    # ibgeID, city, state, populatin, I_reported, series_length

    # getting only remaining dictionaries
    data = []
    for id in cities_nofit:
        if id in observed_nofit:
            data.append(dict(
                meta[id],
                **{'I_reported': observed_nofit[id][3][-1],
                'series_length': observed_nofit[id][4]}
            ))
        else:
            data.append(dict(
                meta[id],
                **{'I_reported': 0,
                'series_length': 0}
            ))

    df = pd.DataFrame(data)
    cid = cities_nofit[0] if len(cities_nofit) else cities[0]
    if not meta[cid]['id'] and df.shape[0]:
        df.drop(axis=1,columns=['id','name'], inplace=True)
    df.rename(columns={'id':'ibgeID','name':'city','pop':'population'}, inplace=True)

    return df
