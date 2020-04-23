import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("seaborn-darkgrid")
from .xonsh_py import *
from .output import out_folder


def plot_compare(params, predef_param, containers, observed, meta, extra_days):

    N = meta['pop']; id = meta['id']; city = meta['name']; state = meta['state']

    y = containers[1:]
    t = np.arange(1,1+y.shape[0]) # timespan based on days length

    deaths = np.nonzero(observed[1])[0]
    ini = deaths[0] if len(deaths) else -1
    dates = observed[0]

    # plot step
    fig = plt.figure(0)

    vl = plt.axvline(params[-4], c='r', ls='--', label='Start of Intervention')
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

    plt.title(f'Covid-19 - {"" if city == None else city+"/"}{state}')

    l = np.int32(np.floor(np.linspace(0, dates.size-1, 10)))
    L = np.int32(np.ceil(np.linspace(0, extra_days, 3)[1:]))
    ex_dates = [str(np.datetime64(dates[-1])+np.timedelta64(x,'D')) for x in L]

    ticks = np.r_[t[l],t[L-extra_days-1]]
    lbls  = np.r_[dates[l],ex_dates]
    tks = plt.xticks(ticks, lbls, rotation=45, rotation_mode='anchor', ha='right')
    tks[1][-1].set_color('r') # setting stipulated days as red
    tks[1][-2].set_color('r') # setting stipulated days as red

    plt.tight_layout()

    outpath = f'{out_folder()}/{state}{"" if city == None else "/"+city}'
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
    plt.title(f'Covid-19 - {"" if city == None else city+"/"}{state}')
    plt.savefig(f'{outpath}/portuguese.svg',dpi=600)

    plt.close(fig)
