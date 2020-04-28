import multiprocessing as mp
import time
import numpy as np
from .statistical import stipulation
from .output import save_time_series, return_put, param_df_out_single, check_done_work
from .plot import plot_compare

def hard_work(q, thr, cities, city_meta, observed, extra_days, lsq_tries):

    c = 0; tot = len(cities)
    # stipulation per state
    for city in cities:
        c+=1
        print(f'{thr}:[{c}/{tot}] Started {city}{", "+city_meta[city]["name"]+"/"+city_meta[city]["state"] if city_meta[city]["id"] else ""}...\n')

        params, predef_param, best_cost, containers = stipulation(thr, extra_days, lsq_tries, observed[city], city_meta[city])

        plot_compare(params, predef_param, containers, observed[city], city_meta[city], extra_days)

        save_time_series(params, predef_param, containers, city_meta[city], extra_days)

        # adding numpy row to queue
        # q.put(return_put(params, predef_param, best_cost, containers[-1-extra_days], city_meta[city], observed[city][4]))

        # writing its data into a single file
        res = return_put(params, predef_param, best_cost, containers[-1-extra_days], city_meta[city], observed[city][4])

        param_df_out_single(res, city_meta[city])


def parallelizer(n_thread, cities, city_meta, observed, extra_days, lsq_tries):
    tot = cities.shape[0]
    print(f'{tot} cities to be processed...\n')

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

        tt = mp.Process(target=hard_work, args=(que, thr, cities[beg:end], city_meta, observed, extra_days, lsq_tries))

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

    return res


def hard_work_no_queue(thr, cities, city_meta, observed, extra_days, lsq_tries):

    c = 0; tot = len(cities)
    # stipulation per state
    for city in cities:
        c+=1
        print(f'{thr}:[{c}/{tot}] Started {city}{", "+city_meta[city]["name"]+"/"+city_meta[city]["state"] if city_meta[city]["id"] else ""}...\n')

        if check_done_work(city_meta[city]):
            print(f'{thr}:[{c}/{tot}] Existing data for {city}{", "+city_meta[city]["name"]+"/"+city_meta[city]["state"] if city_meta[city]["id"] else ""}!\n')
            continue

        params, predef_param, best_cost, containers = stipulation(thr, extra_days, lsq_tries, observed[city], city_meta[city])

        plot_compare(params, predef_param, containers, observed[city], city_meta[city], extra_days)

        save_time_series(params, predef_param, containers, city_meta[city], extra_days)

        # writing its data into a single file
        res = return_put(params, predef_param, best_cost, containers[-1-extra_days], city_meta[city], observed[city][3][-1], observed[city][4])

        param_df_out_single(res, city_meta[city])


def parallelizer_no_queue(n_thread, cities, city_meta, observed, extra_days, lsq_tries):
    tot = cities.shape[0]
    print(f'{tot} cities to be processed...\n')

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

        tt = mp.Process(target=hard_work_no_queue, args=(thr, cities[beg:end], city_meta, observed, extra_days, lsq_tries))

        tts.append(tt)
        tt.start()

    for tt in tts:
        tt.join()
        tt.close()

    return None
