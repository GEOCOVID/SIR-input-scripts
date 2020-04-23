import numpy as np
from src.input import observed_data, city_metadata
from src.parallel import parallelizer
from src.output import param_df_out
from sys import exit


def main():

    # collecting data
    observed = observed_data('https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-states.csv', False, 'state')
    city_meta = city_metadata('metadata', 'estado_sumario_', 0, True)

    cities = np.array(list(set(observed.keys()) & set(city_meta.keys())))
    # cities = np.array(['BA'])

    n_thread = 4
    extra_days = 7
    lsq_tries = 40

    res = parallelizer(n_thread, cities, city_meta, observed, extra_days, lsq_tries)

    # outputing state parameters
    param_df_out(res, observed, city_meta[cities[0]])


main()
