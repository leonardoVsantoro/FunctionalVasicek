import os
import pickle
import seaborn as sns
from sampe_funs import *

MODEL_NAME = 'model'
import importlib.util
spec = importlib.util.spec_from_file_location(MODEL_NAME, f"{MODEL_NAME}.py")
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)

def pred_bands(data, ax, kernel_size = 2, alpha = 5):
    kernel = np.ones(kernel_size) / kernel_size
    lower = np.array([ np.percentile(data[:,i],[alpha/2,100-alpha/2])[0] for i in range(len(data[0])) ])
    upper = np.array([ np.percentile(data[:,i],[alpha/2,100-alpha/2])[1] for i in range(len(data[0])) ])
    lower = np.convolve(lower, kernel, mode='same'); upper = np.convolve(upper, kernel, mode='same')
    ax.fill_between(model.T,lower, upper, alpha=.5, label='{}% CI'.format(100-alpha))
def plot_samplevstruth():
    # plot realisations + truth
    fig,[axNO,axNE,axSE] = plt.subplots(figsize = (16, 3.5),  ncols= 3, nrows=1 )
    # fig.suptitle('95% Confidence Bands vs Truth - {} MC simulations'.format(len(sim_data)), fontsize = 24, y=1.1)
    # ====
    axNO.set_title('$\sigma(\cdot)$', fontsize = 20)
    axNO.plot(model.T, [model.sg_fun(t) for t in model.T], lw=2, color = 'r', label='Truth')
    pred_bands(all_hat_sigma, axNO)
    axNO.legend()
    axNO.set_xlabel('Time')

    # ====
    axNE.set_title('$\\theta(\cdot)$', fontsize = 20)
    axNE.plot(model.T, [model.theta_fun(t) for t in model.T], lw=2, color = 'r', label='Truth')
    pred_bands(all_hat_theta, axNE)
    # axNE.plot(model.T,np.array(all_hat_theta).T, color = 'k', alpha=.1 )
    axNE.legend()
    axNE.set_xlabel('Time')

    # ====
    # # ====
    axSE.set_title('$\mu$', fontsize = 20)
    axSE.boxplot(all_hat_mu)
    axSE.axhline(model.mu, lw=2, color = 'r', label='Truth')
    axSE.legend()
    # ====
    plt.subplots_adjust(hspace=0.3)
    ylim_min =.4; ylim_max =1.36
    axNE.set_ylim(ylim_min*.5, ylim_max); axNO.set_ylim(ylim_min*.5, ylim_max)

    fig.savefig('CI');plt.show()
    folder_path = 'out'

files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) ]
sim_data = []
for filename in files:
    with open('{}/{}'.format(folder_path,filename), 'rb') as pickle_file:
        try:
            data = pickle.load(pickle_file)
            for _ in data:
                sim_data.append(_)
        except:
            print(filename) 

all_hat_sigma = [];all_hat_theta = [];all_hat_mu = []
for _ in sim_data:
    hat_sigma = _['LLS_Der_mean_emp_QV']**.5
    all_hat_sigma.append(hat_sigma)
    
    hat_theta =((hat_sigma**2 - _['LLS_Der_variance'])/(2*_['LLS_variance'])); hat_theta[hat_theta<0] = 1e-10
    all_hat_theta.append(hat_theta)
    
    all_hat_mu.append(np.median((_['LLS_Der_mean'] + hat_theta*_['LLS_mean'])/hat_theta))  
all_hat_sigma =  np.array(all_hat_sigma); all_hat_theta =  np.array(all_hat_theta); all_hat_mu =  np.array(all_hat_mu)

plot_samplevstruth()
