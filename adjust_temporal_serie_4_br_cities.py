import numpy as np
from src.input import observed_data, city_metadata
from src.parallel import parallelizer_no_queue
from src.output import out_folder, merge_output
from sys import exit


def main():

    # collecting data
    # observed, today = observed_data('https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-cities-time.csv', True, 'ibgeID')
    observed, observed_nofit, today = observed_data('input/cities.csv', True, 'ibgeID')
    city_meta = city_metadata('metadata', 'popBR_', 1, False)

    cities = np.array(list(set(observed.keys()) & set(city_meta.keys())))
    cities_nofit = np.array(list(set(city_meta.keys())-set(cities)))
    # cities = np.array([2927408])

    # defining static variable for out_folder function
    out_folder.today = today

    n_thread = 4
    lsq_tries = 40

    #padding = (np.datetime64('today')-np.datetime64(today)).astype(np.int64)
    padding=0
    extra_days = padding+7

    res = parallelizer_no_queue(n_thread, cities, city_meta, observed, extra_days, lsq_tries)
    # TODO: Verificar se todos os ids existem para o Merge ou se há falta.
    merge_output(cities, cities_nofit, observed_nofit, city_meta)


main()
