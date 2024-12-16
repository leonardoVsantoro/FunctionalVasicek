
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

MODEL_NAME = 'model'
import importlib.util
spec = importlib.util.spec_from_file_location(MODEL_NAME, f"{MODEL_NAME}.py")
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)


# to identify experiment
from datetime import datetime
filename = MODEL_NAME + '{}'.format(int(datetime.timestamp( datetime.now())))
pickle_file_path = './out/{}.pickle'.format(filename)


# import functions
import sys
sys.path.append('../') 
from sampe_funs import *

# running the simulations and storing the result
def task(it):
    # Rs = []; Rs.append(EulerMaruyama_approx_sol(model.dt,model.T, model.mu, model.theta_fun ,model.sg_fun,  model.R0(m=model.mu)))
    # for i in range(model.N-1):
    #     Rs.append(EulerMaruyama_approx_sol(model.dt,model.T, model.mu, model.theta_fun ,model.sg_fun, model.R0( Rs[i][-1] ) ) )
    # Rs = np.array(Rs)
    Rs =  np.array([EulerMaruyama_approx_sol(model.dt,model.T, model.mu, model.theta_fun ,model.sg_fun,  model.R0(m=model.mu)) for _ in range(model.N)])
    return runs(Rs, model.T)

import time
start_time = time.time()

N_ITERS = 250; items = np.arange(N_ITERS)
N_cores = 6

print(f'{N_ITERS} runs distributed over {N_cores} cores')
print('computing...')

import concurrent.futures  
executor = concurrent.futures.ThreadPoolExecutor(N_cores)
results = [result for result in map(task, items)] 
end_time = time.time()
runtime = end_time - start_time

print(f"Total runtime: {int(runtime)} seconds")
print(f"Average runtime per run: {int(runtime)/N_ITERS} seconds")

# Storing the results as a pickle file
import pickle
with open(pickle_file_path, 'wb') as pickle_file:
    pickle.dump(results, pickle_file)


