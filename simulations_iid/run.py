

N_ITERS = 50


import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))



MODEL_NAME = 'model_1'
import importlib.util

spec = importlib.util.spec_from_file_location(MODEL_NAME, f"{MODEL_NAME}.py")
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)




# to identify experiment
from datetime import datetime
filename = MODEL_NAME + '{}'.format(int(datetime.timestamp( datetime.now())))
pickle_file_path = './out_{}/{}.pickle'.format(MODEL_NAME, filename)


# import functions
import sys
sys.path.append('../') 
from fun import *

# running the simulations and storing the result

results = []
from tqdm import tqdm # type: ignore
for it in tqdm(np.arange(N_ITERS)):
    Rs = np.array([ EulerMaruyama_approx_sol(model.dt,model.T, model.mu, model.theta_fun ,model.sg_fun, model.R0()) for i in range(model.N)])
    results.append(runs(Rs))

# Storing the results as a pickle file
import pickle
with open(pickle_file_path, 'wb') as pickle_file:
    pickle.dump(results, pickle_file)