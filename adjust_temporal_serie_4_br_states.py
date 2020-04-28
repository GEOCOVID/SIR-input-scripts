import numpy as np
from src.input import observed_data, city_metadata
from src.parallel import parallelizer_no_queue
from src.output import param_df_out, out_folder, merge_output
from sys import exit


def main():

    # collecting data
    # observed, today = observed_data('https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-states.csv', False, 'state')
    observed, today = observed_data('geocovid_pulled_database/states.csv', False, 'state')
    city_meta = city_metadata('metadata', 'estado_sumario_', 0, True)

    cities = np.array(list(set(observed.keys()) & set(city_meta.keys())))
    # cities = np.array(['BA', 'PA', 'RJ'])

    # defining static variable for out_folder function
    out_folder.today = today

    n_thread = 4
    extra_days = 7
    lsq_tries = 40

    res = parallelizer_no_queue(n_thread, cities, city_meta, observed, extra_days, lsq_tries)

    # outputing state parameters
    # param_df_out(res, observed, city_meta[cities[0]])

    # TODO: Verificar se todos os ids existem para o Merge ou se há falta.

    merge_output(cities, observed, city_meta)


main()
