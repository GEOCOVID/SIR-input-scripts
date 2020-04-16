import pandas as pd
import numpy as np
import scipy.integrate as spi
import scipy.optimize as spo
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("seaborn-darkgrid")
from sys import exit

def beta_fun(t, t_thresh, beta1, beta2):
    k = 50. # code defined variable...
    hh = lambda t: 1./(1.+np.exp(-2.*k*t, dtype=np.float128))
    return beta1*hh(t_thresh-t) + beta2*hh(t-t_thresh)

def ode(vars, t, beta1, beta2, delta, ha, ksi, t_thresh, kappa, p, gamma_asym, gamma_sym, gamma_H, gamma_U, mi_H, mi_U, omega):

    S, E, I_asym, I_sym, H, U, R, D, Nw = vars # getting variables values

    beta = beta_fun(t, t_thresh, beta1, beta2)

    dS = -beta*S*(I_sym+delta*I_asym)
    dE = beta*S*(I_sym+delta*I_asym) - kappa*E
    dI_asym = (1-p)*kappa*E - gamma_asym*I_asym
    dI_sym = p*kappa*E - gamma_sym*I_sym
    dH = ha*ksi*gamma_sym*I_sym + (1-mi_U)*gamma_U*U - gamma_H*H
    dU = ha*(1-ksi)*gamma_sym*I_sym + (1-mi_H)*omega*gamma_H*H - gamma_U*U
    dR = gamma_asym*I_asym + (1-mi_H)*(1-omega)*gamma_H*H + (1-ha)*gamma_sym*I_sym
    dD = mi_H*gamma_H*H + mi_U*gamma_U*U
    dNw = p*kappa*E

    return [dS, dE, dI_asym, dI_sym, dH, dU, dR, dD, dNw]

def ode_initial_conditions(params):

    # ---------  Signature of variables unpacked by ode
    # S, E, I_asym, I_sym, H, U, R, D, Nw = vars # getting variables values

    vars0 = params[-3:]
    vars0 = np.concatenate(([1-np.sum(vars0)], vars0, [0, 0, 0, 0, 0])) # prepending resulting Susceptible & appending lasting containers: Hospitalized, UTI, Recovered, and Dead, respectively; and Nw inital value

    return vars0


def cost_function(params, observed, meta, t, predef_param):

    N, _ = meta # unpacking state metadata

    vars0 = ode_initial_conditions(params) # getting ode random initial conditions

    params = np.concatenate((params[:-3], predef_param)) # appending non randomized parameters

    res = spi.odeint(ode, vars0, t, args=tuple(params))

    I_stipulated = res[1:,-1] # getting all integrated time series for Infections
    D_stipulated = res[1:,-2] # getting all integrated time series for Deaths

    I_observed = observed[3] # temporal observed Infected population
    D_observed = observed[1] # temporal observed Dead population


    # getting temporal deviations for specific populations
    err_I = (N*I_stipulated-I_observed)/np.sqrt(N*I_stipulated+1)
    err_D = (N*D_stipulated-D_observed)/np.sqrt(N*D_stipulated+1)

    return np.r_[err_I, err_D]


def stipulation(observed, meta):

    N, _      = meta # unpacking state metadata

    boundaries = np.array([
        [.0,        1.], # beta1
        [.0,        1.], # beta2
        [.0,        1.], # delta
        [.0,       .15], # ha
        [.2,        .5], # ksi
        [ 1, thresh_ub], # t_thresh # até o dia atual???????   ou até final de março??? planilhas com datas???
        [0.,     10./N], # I_asym --> Initial Condition !!
        [0.,     10./N], # I_sym  --> Initial Condition !!
        [0.,     10./N]  # E      --> Initial Condition !!
    ]).T

    predef_param = (
        .25,   # kappa
        .15,   # p
        .2,    # gamma_asym
        .2,    # gamma_sym
        .1,    # gamma_H
        .1,    # gamma_U
        .2,    # mi_H
        .55,   # mi_U
        .04    # omega
    )

    t = np.arange(0,1+len(observed[0])) # timespan based on days length

    best_cost   = np.inf  # holding minimized cost set of parameters
    best_params = (None,) # tuple holding set of best parameters

    for tries in range(40):

        params0 = np.random.rand(boundaries.shape[1])
        params0 = boundaries[0] + params0*(boundaries[1]-boundaries[0])

        res = spo.least_squares(cost_function, params0, bounds=boundaries, kwargs={'observed':observed, 'meta':meta, 't':t, 'predef_param':predef_param})

        if res.status > 0 and best_cost > res.cost: # accepting only better converged parameters
            best_cost   = res.cost
            best_params = res.x
            print(f'Found cost {best_cost} with params:\n{best_params}\n')

            # if best_cost < 50: break

        # if tries % 250 == 0 and not tries == 0: print(f'Tried {tries} times...')

    return best_params, predef_param


def observed_data():
    raw = pd.read_csv('https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-states.csv')

    # USANDO APENAS COLUNAS date, deaths, totalCases e newCases da tabela do wesley
    raw.loc[raw.state == 'TOTAL','state'] = 'BR'

    states = set(raw.state) # getting unique regions/states

    raw_state = dict()

    # for state in states:
    for state in ['BA']:
        state_lines = raw[raw.state == state]

        raw_state[state] = (
            state_lines.date.values,       # [0] -> date       vector
            state_lines.deaths.values,     # [1] -> deaths     vector
            state_lines.newCases.values,   # [2] -> newCases   vector
            state_lines.totalCases.values, # [3] -> totalCases vector
        )
        break

    return raw_state

def state_metadata():
    # TODO: get latest file based on date
    meta = pd.read_csv('estado_sumario_12-04-20.csv', index_col=0)
    meta.rename(index={'TOTAL':'BR'}, inplace=True)

    metadata = dict()
    for state in meta.index:
        metadata[state] = (
            meta.loc[state,'populacao'], # [0] -> N population scalar
            state                        # [1] -> state's name
        )

    return metadata


def plot_compare(params, predef_param, observed, meta):

    N, state = meta # unpacking state metadata

    vars0 = ode_initial_conditions(params) # getting ode random initial conditions

    params = np.concatenate((params[:-3], predef_param)) # appending non randomized parameters

    t = np.arange(0,1+len(observed[0])) # timespan based on days length

    y = spi.odeint(ode, vars0, t, args=tuple(params))[1:]

    ini = np.where(observed[1] == 1)[0][0]
    dates = observed[0]

    # plot step
    fig = plt.figure(0)

    # plotting Deaths
    plt.plot(t[1+ini:], N*y[ini:,-2], '-r', label='Stipulated Death', lw=1)
    plt.plot(t[1+ini:], observed[1][ini:], '.k', label='Observed Death', ms=5)

    plt.plot(t[1:], N*y[:,-1], '-b', label='Stipulated Infection', lw=1)
    plt.plot(t[1:], observed[3], '.k', label='Observed Infection', ms=5)

    plt.axvline(params[5], c='r', ls='--', label='Start of Intervention')

    plt.legend(loc='upper left', fontsize=10)

    plt.xlabel('Date', fontsize=17)
    plt.ylabel('Reported cases', fontsize=17)

    plt.suptitle(f'Covid-19 - {state}')

    l = np.int32(np.floor(np.linspace(0, dates.size-1, 10)))
    plt.xticks(t[l], dates[l], rotation=45, rotation_mode='anchor', ha='right')

    plt.show()

    plt.close(fig)


def main():

    # collecting data
    observed = observed_data()
    state_meta = state_metadata()

    # TODO: verify if there are unmatching state keys between previous dictionaries

    # stipulation per state
    for state in observed:
        params, predef_param = stipulation(observed[state], state_meta[state])

        plot_compare(params, predef_param, observed[state], state_meta[state])

    # getting municipal containers populations to output

main()
exit(1)

#-------------------------------------------------
#-------------------------------------------------
#-------------------------------------------------
#-------------------------------------------------
#-------------------------------------------------
#-------------------------------------------------
#-------------------------------------------------
#-------------------------------------------------
#-------------------------------------------------
#-------------------------------------------------
#-------------------------------------------------

def unique_list(l):
    x = []
    for a in l:
        if a not in x:
            x.append(a)
    return x



#Opening Raw_data States

#Aqui vc só cria um txt pra receber os dados do site do wesley

req = requests.get('https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-states.csv')
url_content = req.content
csv_file = open('C:\\Users\\Daniel\\Google Drive\\Dados Epideiologicos\\CoronaVairus\\CoronaData1.txt', 'wb')

csv_file.write(url_content)
csv_file.close()
raw_data=[]
with open('C:\\Users\\Daniel\\Google Drive\\Dados Epideiologicos\\CoronaVairus\\CoronaData1.txt', encoding="utf8") as data:
    data_reader = csv.reader(data, delimiter='\t')
    for data in data_reader:
        raw_data.append(data)
    for i in range(len(raw_data)):
        raw_data[i]=raw_data[i][0].split (",")
for i in range(len(raw_data)-1):
    raw_data[i+1][5]=int(raw_data[i+1][5])
    raw_data[i+1][6]=int(raw_data[i+1][6])
    raw_data[i+1][7]=int(raw_data[i+1][7])

    if raw_data[i+1][2]=='TOTAL':
        raw_data[i+1][2]= 'Brazil'

#Separating States in the Raw Data
t=[]
dates=[]
data_state=[]
states=[]
for i in range(len(raw_data)-1):
    t.append(raw_data[i+1][0])
    data_state.append(raw_data[i+1][2])

states=unique_list(data_state)
dates=unique_list(t)




#Separating the data for each state
epidemic={}
for s in states:
    state_dic=[[],[],[],[],[]]
    for i in range(len(raw_data)-1):
        if raw_data[i+1][2] == s :
            state_dic[0].append(raw_data[i+1][0])
            state_dic[1].append(raw_data[i+1][6])
            state_dic[2].append(raw_data[i+1][7])
            state_dic[3].append(raw_data[i+1][5])



    epidemic[s]= state_dic

    # Isso aqui é um dicionário, vamos por exemplo a bahia.
    # epidemic['BA'][0] são as datas,epidemic['BA'][2] são os casos acumulados e epidemic['BA'][3] são as mortes





#global parameters


N=14873064

k=50

#delta = 1/2

kappa = 1/4

p = 0.15

gammaA = 1/5

gammaS = 1/5

gammaH = 1/10

gammaU = 1/10

muH = 0.2

muU = 0.55

#h =  0.12

#xi = 0.53

omega = 0.04


#Define where is your data from
s='BA'

#Find when the death starts
for i in range(len(epidemic[s][3])):
    if epidemic[s][3][i] != 0:
        ti=i
        break




def H(t):
    h = 1.0/(1.0+ np.exp(-2.0*k*t))
    return h

def beta(t,t1,b,b1):
    beta = b*H(t1-t) + b1*H(t-t1)
    return beta


def SEIRHU(f,t,parametros):
    #parameters
    b, b1, xi, delta, h, t1 = parametros
    #variables
    S = f[0]
    E = f[1]
    IA = f[2]
    IS = f[3]
    H = f[4]
    U = f[5]
    R = f[6]
    D = f[7]
    Nw = f[8]
    #equations
    dS_dt = - beta(t,t1,b,b1)*S*(IS + delta*IA)
    dE_dt = beta(t,t1,b,b1)*S*(IS + delta*IA) - kappa * E
    dIA_dt = (1-p)*kappa*E - gammaA*IA
    dIS_dt = p*kappa*E - gammaS*IS
    dH_dt = h*xi*gammaS*IS + (1-muU)*gammaU*U -gammaH*H
    dU_dt = h*(1-xi)*gammaS*IS +(1-(muH))*omega*gammaH*H -gammaU*U
    dR_dt = gammaA*IA + (1-(muH))*(1-omega)*gammaH*H + (1-h)*gammaS*IS
    dD_dt = muH*gammaH*H + muU*gammaU*U


    #epidemic curve
    dNw_dt = p*kappa*E


    return [dS_dt,dE_dt,dIA_dt,dIS_dt,dH_dt,dU_dt,dR_dt,dD_dt,dNw_dt]


def lq_SEIRHU(pars, ts0):
    #assaing the Least square given parameters to the numerical integration for the error calculation
    b, b1, xi, delta, h, ia0, is0, e0, t1 = pars
    #initial conditions
    q0 = [1-ia0 - is0 -e0,e0,ia0,is0,0,0,0,0,0]
    #parameters
    parode = b, b1, xi, delta, h,t1
    #calls integrator
    qs = odeint(SEIRHU,q0, ts0,args=(parode,),mxstep=1000000)

    #sinf the epidemic curve
    #sdth the death curve

    sinf = qs[:,-1]
    sdth = qs[:,-2]
    #define the standardized residuals
    erri = (N*sinf - infec) / np.sqrt(N*sinf+1)
    errd = (N*sdth - dth) / np.sqrt(N*sdth+1)
    return np.r_[erri,errd]


# epidemic['BA'][0] são as datas,epidemic['BA'][2] são os casos acumulados e epidemic['BA'][3] são as mortes

#define your data
infec = np.array(epidemic[s][2])

#assing an array of zeros to the days before the first death
dth1 = np.array(epidemic[s][3])[ti:]
dth= np.concatenate((np.zeros(len(infec[:ti])),dth1))

#create an array for the time series starting in t=1
#we need it to start in 1 because we need to integrate from 0-1 to define the number of cases of the first day.
#This helps us because we can set the numer of new cases in t=0 equal 0, than there is no need to fit this parameter.
ts0 = np.array(list(range(len(infec))))+1



#DEEFINE THE INTERVALS OF PARS b, b1, xi, delta, h, ia0, is0, e0, t1
intervals = np.array(   [ [0., 1.],#b
                          [0., 1.],#b1
                          [0.2, 0.5],#xi
                          [0., 1.],#delta
                          [0., 0.15],#h
                          [0.,10./N],#ia0
                          [0.,10./N],#is0
                          [0.,10./N],#e0
                          [0.,18]#t1
                          ])



#NUMBER OF FITS
n_tries = 40
#best error starts at inft
best_err = np.inf
#aj the number of best fists
aj = 0

for i in range(n_tries):
    #create a set of parameters in the interval
    par0 = np.random.rand(len(intervals))
    par0 = intervals[:,0] + par0 * (intervals[:,1] - intervals[:,0])
    #try to fit

    res = least_squares(lambda pars: lq_SEIRHU(pars, ts0), par0, bounds=(intervals[:,0], \
                                                 intervals[:,1]))
    ier = res.status
        #if converges
    if ier >= 1 and ier <= 4:
        if res.cost < best_err:
                #best_err = erro
            best_err = res.cost
                #best_pop = parametro
            best_pop = res.x
                #+1 best fit
            aj = aj +1
            print('achou',aj)



#put best pop in these parameters
b, b1, xi, delta, h, ia0, is0, e0, t1 = best_pop
parode =b, b1, xi, delta, h, t1

q0 = [1-ia0 - is0 -e0,e0,ia0,is0,0,0,0,0,0]

#plot the data
qs = odeint(SEIRHU,q0, ts0,args=(parode,),mxstep=1000000)
pl.figure()
qq =qs[:,-1]
qd =qs[:,-2]




# Create figure and plot space
fig, ax = pl.subplots(figsize=(10, 7))

# Add x-axis and y-axis

ax.plot(epidemic[s][0], N*qq,'b-', linewidth=2.5,label='Casos Previstos Pelo Modelo')
ax.plot(epidemic[s][0][ti:], N*qd[ti:],'r-', linewidth=2.5,label='Mortes Previstas Pelo Modelo')

ax.plot(epidemic[s][0], infec, 'ko',label='Casos Reportados ')
ax.plot(epidemic[s][0][ti:], dth[ti:], 'ko',label='Mortes Reportadas ')

ax.axvline(t1, 0, 600,c='r',linestyle='--',label='Intervenção')



# Set title and labels for axes
fig.suptitle("Epidemia Covid-19 " + '(' + s + ')',fontsize=20)


ax.set_xlabel('Data',fontsize=17)
ax.set_ylabel('Casos Reportados',fontsize=17)


ax.legend(loc='upper left',fontsize=15)

# Rotate tick marks on x-axis
ax.xaxis.set_major_locator(pl.MaxNLocator(10))
pl.setp(ax.get_xticklabels(), rotation=45)

pl.show()
