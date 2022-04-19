import json
import numpy as np
import assimilation_tools as at

class SettingsGolem(object):
	def __init__(self, settings_to_override=None):
		with open('experiment_settings.json') as f:
			self.settings = json.load(f)
		if settings_to_override is not None:
			keys = list(settings_to_override.keys())
			values = list(settings_to_override.values())
			for key,value in zip(keys,values):
				self.settings[key]['value'] = value
	def getSetting(self,key):
		return self.settings[key]['value']
	def makeDistMat(self):
		n_x = self.getSetting('n_x')
		n_y = self.getSetting('n_y')
		distmat = np.zeros((n_x,n_y,n_x,n_y))
		for x1 in range(n_x):
			for y1 in range(n_y):
				for x2 in range(n_x):
					for y2 in range(n_y):
						distmat[x1,y1,x2,y2] = np.sqrt((x2-x1)**2+(y2-y1)**2)
		return distmat

#Velocity as a function of time
def vel(timestep,timescale=100,maxval=10,noise=None):
	if noise:
		return maxval*np.cos((timestep/timescale)*2*np.pi)+(noise*((np.random.rand()*2)-1))
	else:
		return maxval*np.cos((timestep/timescale)*2*np.pi)


def make_emis_t(emis_variability,emis_timescale_short,emis_timescale_long):
	def emis_t(timestep,base_field):
		return (emis_variability*base_field)*(np.sin((timestep/emis_timescale_short)*2*np.pi)+np.sin((timestep/emis_timescale_long)*2*np.pi))+base_field
	return emis_t

def const_emis_t(timestep,base_field):
	return base_field

#computes the next time step given matrices representing values at

def makeInitialEnsEmisFields():
	ensfields = []
	for i in range(n_ens):
		ensfields.append(base_emis_field*(np.random.rand(n_x,n_y)+0.5))
		#ensfields.append(base_emis_field*(i+(n_ens/2))/n_ens)
	return ensfields


def run_with_assimilation(radius,inflation,scalingInflator,freq_obs,window,end_time,nature_vel_data,model_vel_data,nature_bias=0, nature_err=1,errtype='absolute',lifetime=500,nature_emis_func = None, model_emis_func = const_emis_t):
	ensfields = makeInitialEnsEmisFields()
	tstart = 0
	tend = window
	cur_time = 0
	initial_conditions = None
	nature_list = []
	ens_list = []
	emis_list = []
	time_list = []
	emis_list.append(ensfields)
	while cur_time < end_time:
		print(f'Running model from {tstart} to {tend}')
		nature,ensemble,timevals = compute_nature_and_ens(nature_vel_data=nature_vel_data,model_vel_data=model_vel_data,ens_emis_fields = ensfields, initial_conditions = initial_conditions, time_start = tstart, lifetime=lifetime, time_end=tend,nature_emis_func = nature_emis_func, model_emis_func = model_emis_func)
		cur_time = tend
		nature_list.append(nature)
		ens_list.append(ensemble)
		time_list.append(timevals)
		obs_times = np.arange(tstart,tend,freq_obs)
		assim = at.Assimilator(nature,ensemble,timevals, obs_times,ensfields,distmat,nature_bias=nature_bias,nature_err=nature_err)
		print(f'Assimilating data at time {cur_time}')
		ens_conc,ens_emis = assim.update(radius,inflation,scalingInflator)
		ensfields = []
		for i in range(np.shape(ens_emis)[0]):
			ensfields.append(ens_emis[i,:,:])
		emis_list.append(ensfields)
		initial_conditions = [nature[-1,:,:],ens_conc]
		tstart = tend
		tend += window
	nature_list = np.concatenate(nature_list,axis=0)
	ens_list = np.concatenate(ens_list,axis=1)
	time_list = np.concatenate(time_list)
	for i in range(len(emis_list)):
		emis_list[i] = np.stack(emis_list[i])
	emis_list = np.stack(emis_list,axis=1)
	return [nature_list,ens_list,emis_list,time_list]




