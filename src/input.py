import numpy as np
import pandas as pd
from .xonsh_py import *

def observed_data(url, removeTotal, keyIndex):

    raw = pd.read_csv(url)

    if removeTotal:
        raw = raw[raw.state != 'TOTAL'] # removing summarized rows
    else:
        raw.loc[raw.state == 'TOTAL','state'] = 'BR'

    ids = set(raw[keyIndex]) # getting unique cities

    cities = dict()

    for id in ids:
        city_lines = raw[raw.loc[:,keyIndex] == id]

        # skipping time series with less than 10 points
        if city_lines.shape[0] < 10 or city_lines.totalCases.values[-1] < 75:
            continue

        cities[id] = (
            city_lines.date.values,       # [0] -> date       vector
            city_lines.deaths.values,     # [1] -> deaths     vector
            city_lines.newCases.values,   # [2] -> newCases   vector
            city_lines.totalCases.values, # [3] -> totalCases vector
            city_lines.shape[0]           # [4] -> length of time series
        )

    today = cities[list(cities.keys())[0]][0][-1] # getting last date

    return cities, today

def city_metadata(base, wild, i_col, renameBR):
    last_date = get_last_date(base, wild)
    df = pd.read_csv(f'{base}/{wild}{last_date}.csv', index_col=i_col)

    if renameBR:
        df.rename(index={'TOTAL':'BR'}, inplace=True)

    metadata = dict()
    for id in df.index:
        city_row = df.loc[id]
        metadata[id] = {
            'pop': np.int64(city_row.Pt if 'Pt' in df.columns else city_row.populacao),   # [0] -> N population scalar
            'id': id if 'id' == df.index.name else None, # [1] -> city's ibgeID
            'name': city_row.Name if 'Name' in df.columns else None, # [2] -> city's name
            'state': city_row.UF if 'UF' in df.columns else id  # [3] -> state
        }

    return metadata


def get_last_date(path, wild):

    l = lsgrep(path, [wild], with_path=False)

    dates = [x.split('_')[-1].split('.')[0] for x in l]
    dates = np.array(list(map(np.datetime64, dates)))
    return np.max(dates)
