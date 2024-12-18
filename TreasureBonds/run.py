# libraries 
import numpy as np # type: ignore
import seaborn as sns# type: ignore
import pandas as pd# type: ignore
import matplotlib.pyplot as plt# type: ignore
import os
from tqdm import tqdm# type: ignore
from pytz import timezone # type: ignore
from scipy.integrate import simps as simps# type: ignore
import pickle# type: ignore
from datetime import datetime, timedelta# type: ignore
import matplotlib.cm as cm# type: ignore
# import functions
from funs import *


bonds_names = ['US', 'FV', 'TY', 'TU']# import data
data = {}
for file in os.listdir('./TB_out_pickled_data'):
    try:
        with open('./TB_out_pickled_data/{}'.format(file), 'rb') as f:
            loaded_data = pickle.load(f)
        data.update({file[:-7] :loaded_data})
    except:
        None

# Define start and end times
start_time = datetime.strptime('00:00', '%H:%M')
end_time = datetime.strptime('23:55', '%H:%M')

# Generate list of time strings every 5 minutes
time_list = []
current_time = start_time

while current_time <= end_time:
    time_list.append(current_time.strftime('%H:%M'))
    current_time += timedelta(minutes=5)

# open_time_list = np.array(time_list[48:])
open_time_list = np.array(time_list[131:])


_hs = []; _ht = []; _hm = []; _R0_mean = []; _R0_std = []

bonds_names = ['US', 'FV', 'TY', 'TU']

for selected_bond in bonds_names[:-1]:
    fig, axs  = plt.subplots(figsize = (16,3.5), ncols = 3)
    fig.suptitle(selected_bond, fontsize = 17,y=1.1)    
    years = np.arange(2003, 2015)
    
    hs = []; ht = []; hm = []; R0_mean = []; R0_std = []
    for i, (selected_year, c) in tqdm(enumerate(zip(years,  plt.cm.viridis(np.linspace(0,1,years.size))))) :
        # import data
        Rs  = data['{}_{}'.format(selected_bond, selected_year)]['opening_values']
        Ts = data['{}_{}'.format(selected_bond, selected_year)]['times']
            
        # keep only samples of full length (excludes eg holidays), add artificial noise
        sizes = [_.size for _ in Rs]
        ixs_toKeep1 = np.where(np.array(sizes) == np.median(sizes))[0]
        Rs = np.array([_.ravel() + giveMeNoise(int(np.median(sizes))).ravel() for i, _ in enumerate(Rs) if i in ixs_toKeep1])
        # Rs = np.array([_.ravel() for i, _ in enumerate(Rs) if i in ixs_toKeep1])
        Ts = np.array([_  for i, _ in enumerate(Ts) if i in ixs_toKeep1])
        
        # keep only values when market is open
        strTs = [[ time.strftime('%H:%M') for time in T] for T in Ts] 
        Rs =np.array([ [r for r, t in zip(R,T) if t in open_time_list ] for R,T in zip(Rs,strTs)])
        Ts =np.array([ [_t for _t, t in zip(_T,T) if t in open_time_list ] for _T,T in zip(Ts,strTs)])
    
        # axes for plotting
        _where = np.array([ i for i,_ in enumerate(time_list) if _ in [time.strftime('%H:%M') for time in Ts[0]] ])
    
        # run estimation
        results = estimate_FVM(Rs ,h_qv = .65, h_mean = .65)
        hat_sigma = results['LLS_Der_mean_emp_QV']**.5
        hat_theta =((hat_sigma**2 - results['LLS_Der_variance'])/(2*results['LLS_variance']))
        hat_theta = np.array([_ if _>0 else 0 for _ in hat_theta]) 
        
        allhatmu = ((results['LLS_Der_mean'] + hat_theta*results['LLS_mean'])/hat_theta)[hat_theta>0]
        lower_bound = np.percentile(allhatmu, 25); upper_bound = np.percentile(allhatmu, 75);
        hat_mu = np.nanmedian(allhatmu[(allhatmu >= lower_bound) & (allhatmu <= upper_bound)])
    
        hs.append(hat_sigma); ht.append(hat_theta); hm.append(hat_mu)
        R0_mean.append(Rs[:,0].mean()); R0_std.append(Rs[:,0].std())
        
        axs[0].set_title('$\sigma(\cdot)$');axs[1].set_title('$\\theta(\cdot)$'); axs[2].set_title('$\mu$')
        axs[0].plot(_where,  hat_sigma, color=c, label = '{}'.format(selected_year));
        axs[1].plot(_where, hat_theta, color=c, label = '{}'.format(selected_year));
        axs[2].scatter(selected_year, hat_mu, color=c, label = '{}'.format(selected_year));

    every= 18
    datetime_with_dummy_date = [ datetime.combine(datetime(2000, 1, 1),t) for t in Ts[0]]
    my_delta = timedelta(hours=1, minutes=115)
    resulting_datetime = [t - my_delta for t in datetime_with_dummy_date]
    resulting_time = [ t.time() for t in resulting_datetime]
    
    strf_time  = np.array([ _ for _ in time_list if _ in [time.strftime('%H:%M') for time in resulting_time] ])[::every]
    
    axs[1].set_xticks(_where[::every]); axs[0].set_xticks(_where[::every]); 
    axs[0].set_xticklabels(strf_time ,rotation = 45); axs[1].set_xticklabels(strf_time,rotation = 45) 
    axs[0].set_ylim(0.4,1.41); axs[1].set_ylim(-.02,0.35); axs[2].set_ylim(100, 175)    
    _hs.append(np.array(hs).mean(0)) ; _ht.append(np.array(ht).mean(0)) ; _hm.append(np.array(hm).mean(0)) 
    _R0_mean.append(np.mean(R0_mean)); _R0_std.append(np.mean(R0_std))
    
    plt.savefig('fig/{}.png'.format(selected_bond), bbox_inches='tight')
    plt.show()

import pickle
IX = 1

# pickle.dump({'sigma': _hs[IX], 'theta': _ht[IX], 'mu':_hm[IX], 'grid_size': _ht[IX].size, 'R0_mean' : _R0_mean[IX],  'R0_std' : _R0_std[IX], 'xticklabs' : strf_time},file)

fig, axs  = plt.subplots(figsize = (16,2.5), ncols = 5)
axs[0].plot(_hs[IX]); axs[1].plot(_ht[IX]);  axs[2].scatter(0,_hm[IX]);  axs[3].scatter(0,_R0_mean[IX]);  axs[4].scatter(0,_R0_std[IX]); 



# libraries 
import numpy as np # type: ignore
import seaborn as sns# type: ignore
import pandas as pd# type: ignore
import matplotlib.pyplot as plt# type: ignore
import os
from tqdm import tqdm# type: ignore
from pytz import timezone # type: ignore
from scipy.integrate import simps as simps# type: ignore
import pickle# type: ignore
from datetime import datetime, timedelta# type: ignore
import matplotlib.cm as cm# type: ignore
# import functions
from funs import *



selected_year = 2009
selected_bond = 'US'

data = {}
for file in os.listdir('./data_treasure_bonds'):
    try:
        data.update({file[:-4] : pd.read_csv('./data_treasure_bonds/{}'.format(file)).sort_values(['Date', 'Time'])})
    except:
        None
data.keys()

df = data[selected_bond].copy()
df['Date'] = pd.to_datetime(df['Date']).copy(); df['Time'] = pd.to_timedelta( df.Time + ':00'); df.loc[:,'Datetime'] =  df.Date + df.Time 


df_year =df[df.Datetime.dt.year == selected_year].sort_values('Datetime')
    
set_of_dates = df_year.Datetime.dt.date.unique()

Rs = []; dates = []; Ts = []
for date in (set_of_dates):
    temp_df = df_year.loc[df_year.Datetime.dt.date==date].copy(); 
    Rs.append(temp_df.Open.values); 
    dates.append(temp_df.Datetime.values);
    Ts.append(temp_df.Datetime.dt.time.values)


# keep only samples of full length (excludes eg holidays), add artificial noise
sizes = [_.size for _ in Rs]
ixs_toKeep1 = np.where(np.array(sizes) == np.median(sizes))[0]

Rs = np.array([_.ravel() + giveMeNoise(int(np.median(sizes))).ravel() for i, _ in enumerate(Rs) if i in ixs_toKeep1])
Ts = np.array([_  for i, _ in enumerate(Ts) if i in ixs_toKeep1])

# keep only values when market is open
strTs = [[ time.strftime('%H:%M') for time in T] for T in Ts] 
Rs =np.array([ [r for r, t in zip(R,T) if t in open_time_list ] for R,T in zip(Rs,strTs)])
Ts =np.array([ [_t for _t, t in zip(_T,T) if t in open_time_list ] for _T,T in zip(Ts,strTs)])

dates = np.array([_  for i, _ in enumerate(dates) if i in ixs_toKeep1])



from datetime import datetime, timedelta
from matplotlib.ticker import ScalarFormatter# type: ignore
nrows = 1
fig, axs  = plt.subplots(figsize = (16,nrows*3), ncols = 8,nrows=nrows, sharey=True); axs= axs.ravel()
fig.suptitle('Some consecutive trajectories', fontsize = 22)
cmp = plt.cm.viridis(np.linspace(0,1,axs.size))
for i, (ax, R, date) in enumerate(zip(axs,Rs,dates)):
    ax.plot(np.linspace(0,1,R.size), R, c='k'); 
    axs[i].set_title(pd.to_datetime(date)[0].strftime("%B %d"), fontsize = 14)
    axs[i].set_xticks(np.linspace(0,1,len(strf_time[::2]))); axs[i].set_xticklabels(strf_time[::2], rotation = 45)
    # axs[i].yaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=True))

fig.suptitle('{} - {}\nsome consecutive curves'.format(selected_bond, selected_year),fontsize = 17)
plt.tight_layout()
fig.savefig('fig/consecutive_trajectories'); plt.show()