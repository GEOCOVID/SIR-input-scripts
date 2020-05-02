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

    dates = observed[0]

    # plot step
    fig = plt.figure(0)


    vL = plt.axvline(t.shape[0]-extra_days, c='#6d4848', ls=':', label='Time Series Limit')

    si, = plt.plot(t, y[:,-1], '-b', label='Stipulated Infection', lw=1)
    oi, = plt.plot(t[:-extra_days], observed[3], '.k', label='Observed Infection', ms=5)
    r, = plt.plot(t, y[:,2], '-g', label='Stipulated Recovery', lw=1)


    plt.legend(loc='upper left', fontsize=10)

    plt.xlabel('Date', fontsize=17)
    plt.ylabel('Reported cases', fontsize=17)

    plt.title(f'Covid-19 - {"" if city == None else city+"/"}{state}')



    ratio = dates.size/(dates.size+extra_days)
    l = np.int32(np.floor(np.linspace(0, dates.size-1, np.floor(ratio*22))))
    L = np.int32(np.ceil(np.linspace(0, extra_days, np.ceil((1-ratio)*22))[1:]))
    ex_dates = [str(np.datetime64(dates[-1])+np.timedelta64(x,'D')) for x in L]

    ticks = np.r_[t[l],t[L-extra_days-1]]
    lbls  = np.r_[dates[l],ex_dates]
    tks = plt.xticks(ticks, lbls, rotation=45, rotation_mode='anchor', ha='right')
    for i in range(1,len(L)+1):
        tks[1][-i].set_color('r') # setting stipulated days as red

    plt.tight_layout()

    outpath = f'{out_folder()}/{state}{"" if city == None else "/"+city}'
    mkdir_p([outpath])
    plt.savefig(f'{outpath}/english.svg',dpi=600)

    # relabeling in portuguese
    si.set_label('Infecções Estipuladas')
    oi.set_label('Infecções Reportadas')
    vL.set_label('Fim da Série Temporal')
    plt.legend(loc='upper left', fontsize=10)
    plt.xlabel('Data', fontsize=17)
    plt.ylabel('Casos Reportados', fontsize=17)
    plt.title(f'Covid-19 - {"" if city == None else city+"/"}{state}')
    plt.savefig(f'{outpath}/portuguese.svg',dpi=600)

    plt.close(fig)
