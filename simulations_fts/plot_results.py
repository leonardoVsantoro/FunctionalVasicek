MODEL_NAME = 'model'
import importlib.util
spec = importlib.util.spec_from_file_location(MODEL_NAME, f"{MODEL_NAME}.py")
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)

localN = 50

# generate curves 
Rs = np.array([ EulerMaruyama_approx_sol(model.dt,model.T, model.mu, model.theta_fun ,model.sg_fun, model.R0())
               for i in range(50)])


from datetime import datetime, timedelta
nrows = 4
fig, axs  = plt.subplots(figsize = (16,nrows*2.2), ncols = 8,nrows=nrows, sharey=False); axs= axs.ravel()
fig.suptitle('Some consecutive curves', fontsize = 22)
cmp = plt.cm.viridis(np.linspace(0,1,axs.size))
for i, (ax, R, c) in enumerate(zip(axs,Rs,cmp)):
    ax.plot(model.T, R, c='k'); 
    current_date = datetime(2024, 1, 1) + timedelta(days=i)
    axs[i].set_title(current_date.strftime("%B %d"), fontsize = 14)
    axs[i].set_xticks(np.linspace(0,1,len(model.xticklabs[::2]))); axs[i].set_xticklabels(model.xticklabs[::2], rotation = 45)
plt.tight_layout()
fig.savefig('fig/consecutive_trajectories'.format(MODEL_NAME));plt.show()