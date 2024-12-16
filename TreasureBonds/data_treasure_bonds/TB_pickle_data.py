# cd /Users/santoro/Desktop/EPFL/__research/Func_Vasicek_Model/financial_application
# python TB_pickle_data.py


# libraries 
import numpy as np
import pandas as pd
import os
import datetime
import pickle
from tqdm import tqdm
#import concurrent.futures


data = {}
for file in os.listdir('./data_treasure_bonds'):
    try:
        data.update({file[:-4] : pd.read_csv('./data_treasure_bonds/{}'.format(file)).sort_values(['Date', 'Time'])})
    except:
        None




def pickle_data(key):

    data = {}
    for file in os.listdir('./data_treasure_bonds'):
        try:
            data.update({file[:-4] : pd.read_csv('./data_treasure_bonds/{}'.format(file)).sort_values(['Date', 'Time'])})
        except:
            None

    df = data[key].copy(); 
    df['Date'] = pd.to_datetime(df['Date']).copy()
    df['Time'] = pd.to_timedelta( df.Time + ':00')
    df.loc[:,'Datetime'] =  df.Date + df.Time 
    df.loc[:,'Datetime'] = df.loc[:,'Datetime'].copy() - datetime.timedelta(hours=-7)  
    df = df.loc[ (df.Datetime.dt.year <= 1995)].copy()
   
    for year in tqdm(df.Datetime.dt.year.unique()):
        df_year = df.loc[df.Datetime.dt.year == year].copy()
        
        set_of_dates = df_year.Datetime.dt.date.unique()
        
        opening_values = []; times = [];
        for date in (set_of_dates):
            temp_df = df_year.loc[df_year.Datetime.dt.date==date].copy(); 
            opening_values.append(temp_df.Open.values); times.append(temp_df.Datetime.dt.time.values);
        
        with open('TB_out_pickled_data/{}_{}.pickle'.format(key,year), 'wb') as file:
            pickle.dump({'opening_values': opening_values, 'times':times}, file)

    return 

#for key in data.keys():
#    pickle_data(key)

import concurrent.futures
with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.map(pickle_data, data.keys())


