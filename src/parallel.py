import multiprocessing as mp
import time
import numpy as np
from .statistical import stipulation, stipulation_city
from .output import save_time_series, return_put, param_df_out_single, check_done_work
from .plot import plot_compare

def hard_work_no_queue(thr, cities, city_meta, observed, extra_days, lsq_tries, tseries_limit):

    stipul = stipulation_city if city_meta[cities[0]]['id'] else stipulation

    c = 0; tot = len(cities)
    # stipulation per state
    for city in cities:
        c+=1
        print(f'{thr}:[{c}/{tot}] Started {city}{", "+city_meta[city]["name"]+"/"+city_meta[city]["state"] if city_meta[city]["id"] else ""}...\n')

        if check_done_work(city_meta[city], tseries_limit):
            print(f'{thr}:[{c}/{tot}] Existing data for {city}{", "+city_meta[city]["name"]+"/"+city_meta[city]["state"] if city_meta[city]["id"] else ""}!\n')
            continue

        params, predef_param, best_cost, containers = stipul(thr, extra_days, lsq_tries, tseries_limit, observed[city], city_meta[city])

        plot_compare(params, predef_param, containers, observed[city], city_meta[city], extra_days, tseries_limit)

        save_time_series(params, predef_param, containers, city_meta[city], extra_days, tseries_limit)

        # writing its data into a single file
        res = return_put(params, predef_param, best_cost, containers[-1-extra_days], city_meta[city], observed[city][3][-1], observed[city][4])

        param_df_out_single(res, city_meta[city], tseries_limit)


def parallelizer_no_queue(n_thread, cities, city_meta, observed, extra_days, lsq_tries, tseries_limit):
    tot = cities.shape[0]
    print(f'{tot} cities to be processed until {tseries_limit} points...\n')

    l = tot//n_thread
    r = tot%n_thread

    tts = []
    for thr in range(n_thread):
        if thr < r:
            beg = thr*(l+1)
            end = beg + l + 1
        else:
            beg = thr*l + r
            end = beg + l

        tt = mp.Process(target=hard_work_no_queue, args=(thr, cities[beg:end], city_meta, observed, extra_days, lsq_tries, tseries_limit))

        tts.append(tt)
        tt.start()

    for tt in tts:
        tt.join()
        tt.close()

    return None
