import json
import numpy as np
import assimilation_tools as at
import plume_tools as pt
import matplotlib.pyplot as plt
import scipy.stats as ss
import scipy.spatial.distance as ssd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class SettingsGolem(object):
	def __init__(self, settings_to_override=None, distmat = None):
		with open('experiment_settings.json') as f:
			self.settings = json.load(f)
		if settings_to_override is not None:
			keys = list(settings_to_override.keys())
			values = list(settings_to_override.values())
			for key,value in zip(keys,values):
				self.settings[key]['value'] = value
		if distmat is None:
			self.distmat = self.makeDistMat()
		else: 
			self.distmat = distmat
		self.initializeErrorCov()
	def getSetting(self,key):
		if type(key) is list:
			to_return = []
			for val in key:
				to_return.append(self.settings[val]['value'])
			return to_return
		else:
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
	def initializeErrorCov(self):
		n_x,n_y = self.getSetting(['n_x','n_y'])
		self.error_cov = {}
		if self.settings['nature_err']['isCorrelated'] == "True":
			correlation_scale = self.settings['nature_err']['corrDist']
			halg = self.getSetting('halg')
			if (halg == 'all') or (type(halg) is list) and (type(halg[0]) is list):
				cov = self.makeBaseCovMat(n_x,n_y,correlation_scale)
			elif (type(halg) is list) and (type(halg[0]) is int):
				cov = self.makeBaseCovMat(halg[0],halg[1],correlation_scale)
			else:
				raise ValueError('Specified halg is incompatible with error covariance structure')
			self.error_cov['obs'] = cov
		else:
			self.error_cov['obs'] = None
		if (self.settings['velocity_data']['noisy_nature'] == "True") and (self.settings['velocity_data']['nature_err_corr']['useCorrelatedError'] == "True"):
			correlation_scale = self.settings['velocity_data']['nature_err_corr']['correlationDistance']
			cov = self.makeBaseCovMat(n_x,n_y,correlation_scale)
			self.error_cov['nature_vel'] = cov
		else:
			self.error_cov['nature_vel'] = None
		if (self.settings['velocity_data']['noisy_model'] == "True") and (self.settings['velocity_data']['model_err_corr']['useCorrelatedError'] == "True"):
			correlation_scale = self.settings['velocity_data']['model_err_corr']['correlationDistance']
			cov = self.makeBaseCovMat(n_x,n_y,correlation_scale)
			self.error_cov['model_vel'] = cov
		else:
			self.error_cov['model_vel'] = None
	def makeBaseCovMat(self,nx,ny,corrdist):
		X,Y = np.meshgrid(np.arange(nx),np.arange(ny))
		# Create a vector of cells
		XY = np.column_stack((np.ndarray.flatten(X),np.ndarray.flatten(Y)))
		# Calculate a matrix of distances between the cells
		dist = ssd.pdist(XY)
		dist = ssd.squareform(dist)
		# Convert the distance matrix into a covariance matrix
		cov = np.exp(-dist**2/(2*corrdist)) # This will do as a covariance matrix
		return cov
	def getDistMat(self):
		return self.distmat
	def makeBaseField(self):
		n_x = self.getSetting('n_x')
		n_y = self.getSetting('n_y')
		if self.settings["emis_field"]['interpret_as'] == 'location':
			base_emis_field = np.zeros((n_x,n_y))
			to_replace = self.settings["emis_field"]['value']
			for loc,amp in zip(to_replace['location'],to_replace['value']):
				base_emis_field[loc[0],loc[1]] = amp
		elif self.settings["emis_field"]['interpret_as'] == 'function':
			base_emis_field = eval(self.settings["emis_field"]['value'])
		else:
			raise ValueError('Could not interpret base field instructions.')
		return base_emis_field
	def makeVel(self, dimension,isModel):
		veldata = self.getSetting('velocity_data')
		fulldetails = self.settings['velocity_data']
		if isModel:
			useNoise = fulldetails['noisy_model'] == "True"
			isNoiseCorrelated = fulldetails['model_err_corr']['useCorrelatedError'] == "True"
			corrDist = fulldetails['model_err_corr']['correlationDistance']
			cov = self.error_cov['model_vel']
		else:
			useNoise = fulldetails['noisy_nature'] == "True"
			isNoiseCorrelated = fulldetails['nature_err_corr']['useCorrelatedError'] == "True"
			corrDist = fulldetails['nature_err_corr']['correlationDistance']
			cov = self.error_cov['nature_vel']
		if veldata['function'] == 'circle':
			if dimension == 'x':
				maxval = veldata['parameters']["vel_x_max"]
				if useNoise:
					noise = veldata['parameters']["vel_x_noise"]
				else:
					noise = 0
				timescale = veldata['parameters']["vel_x_timescale"]
			elif dimension == 'y':
				maxval = veldata['parameters']["vel_y_max"]
				if useNoise:
					noise = veldata['parameters']["vel_y_noise"]
				else:
					noise = 0
				timescale = veldata['parameters']["vel_y_timescale"]
			else:
				raise ValueError('Dimension must be x or y')
			if isNoiseCorrelated and useNoise:
				n_x,n_y = self.getSetting(['n_x','n_y'])
				noisefield = ss.multivariate_normal.rvs(mean = np.zeros(np.shape(cov)[0]),cov = (noise**2)*cov).reshape((n_x,n_y))
				def vel(timestep,x,y):
					return maxval*np.cos((timestep/timescale)*2*np.pi)+noisefield[x,y]
			elif not isNoiseCorrelated and useNoise:
				def vel(timestep,x,y):
					return maxval*np.cos((timestep/timescale)*2*np.pi)+np.random.normal(0,noise)
			else:
				def vel(timestep,x,y):
					return maxval*np.cos((timestep/timescale)*2*np.pi)
			return vel
		elif veldata['function'] == 'const':
			if dimension == 'x':
				maxval = veldata['parameters']["vel_x_max"]
				if useNoise:
					noise = veldata['parameters']["vel_x_noise"]
				else:
					noise = 0
			elif dimension == 'y':
				maxval = veldata['parameters']["vel_y_max"]
				if useNoise:
					noise = veldata['parameters']["vel_y_noise"]
				else:
					noise = 0
			else:
				raise ValueError('Dimension must be x or y')
			if isNoiseCorrelated and useNoise:
				n_x,n_y = self.getSetting(['n_x','n_y'])
				noisefield = ss.multivariate_normal.rvs(mean = np.zeros(np.shape(cov)[0]),cov = (noise**2)*cov).reshape((n_x,n_y))
				def vel(timestep,x,y):
					return maxval+noisefield[x,y]
			elif not isNoiseCorrelated and useNoise:
				def vel(timestep,x,y):
					return maxval+np.random.normal(0,noise)
			else:
				def vel(timestep,x,y):
					return maxval
			return vel
		else:
			raise ValueError('Velocity function unsupported')
	def makeEmis(self, functype='full'):
		if functype=='const':
			def emis_t(timestep,base_field):
				return base_field
			return emis_t
		elif functype=='full':
			emis_variability,timescales,amps = self.getSetting(["emis_variability","emis_timescales","emis_amplitudes"])
			timescales = np.array(timescales)
			amps = np.array(amps)
			normed_amps = amps/np.sum(amps)
			def emis_t(timestep,base_field):
				fourier_terms = 0
				for timescale,amp in zip(timescales,normed_amps):
					fourier_terms+=amp*np.sin((timestep/timescale)*2*np.pi)
				return (emis_variability*base_field*fourier_terms)+base_field
			return emis_t
		elif type(functype) is int:
			emis_variability,timescales,amps = self.getSetting(["emis_variability","emis_timescales","emis_amplitudes"])
			timescales = np.array(timescales)[0:functype] #use only first fourier components
			amps = np.array(amps)[0:functype]
			normed_amps = amps/np.sum(amps)
			def emis_t(timestep,base_field):
				fourier_terms = 0
				for timescale,amp in zip(timescales,normed_amps):
					fourier_terms+=amp*np.sin((timestep/timescale)*2*np.pi)
				return (emis_variability*base_field*fourier_terms)+base_field
			return emis_t
		else:
			raise ValueError('Unacceptable function type.')
	def makeInitialEnsEmisFields(self, base_emis_field):
		n_x,n_y,n_ens = self.getSetting(['n_x','n_y','n_ens'])
		ensfields = []
		for i in range(n_ens):
			ensfields.append(base_emis_field*(np.random.rand(n_x,n_y)+0.5))
			#ensfields.append(base_emis_field*(i+(n_ens/2))/n_ens)
		self.initial_ens_std = np.std(np.stack(ensfields),axis=0)
		return ensfields
	def makeObsCorrelatedErrorField(self, bias,error, outshape):
		cov = self.error_cov['obs']
		if cov is None:
			return None
		else:
			if (type(error) is int) or (type(error) is float):
				noisefield = ss.multivariate_normal.rvs(mean = bias*np.ones(np.shape(cov)[0]),cov = (error**2)*cov).reshape(outshape)
			else:
				newcov = np.outer(error,error)*cov #Scale covariance matrix appropriately for relative error case
				noisefield = ss.multivariate_normal.rvs(mean = bias,cov = newcov).reshape(outshape)
			return noisefield
	def dumpSettings(self,outfile):
		with open(outfile, "w") as write_file:
			json.dump(self.settings, write_file)

def run_with_assimilation(setting_golem):
	base_emis_field = setting_golem.makeBaseField()
	ensfields = setting_golem.makeInitialEnsEmisFields(base_emis_field)
	window,freq_obs,end_time = setting_golem.getSetting(['window',"frequency","endtime"])
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
		nature,ensemble,timevals = pt.compute_nature_and_ens(settings_golem=setting_golem,ens_emis_fields = ensfields,initial_conditions = initial_conditions,time_start = tstart, time_end=tend)
		cur_time = tend
		nature_list.append(nature)
		ens_list.append(ensemble)
		time_list.append(timevals)
		obs_times = np.arange(tstart,tend,freq_obs)
		assim = at.Assimilator(setting_golem,nature,ensemble,timevals, obs_times,ensfields)
		print(f'Assimilating data at time {cur_time}')
		ens_conc,ens_emis = assim.update()
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

def make_emis_postprocess_plot(settings_golem,emis_list,time_list,xloc,yloc):
	window,emisModel = settings_golem.getSetting(['window','emis_model_type'])
	plottime = np.unique(time_list)
	if emisModel == 'const':
		model_emis = np.repeat(emis_list[:,:,xloc,yloc],window,axis=1)
	else:
		model_emis = np.zeros((np.shape(emis_list)[0],len(plottime)))
		emis_t_model = settings_golem.makeEmis(functype=emisModel)
		timestamp = 0
		for emisind in range(np.shape(emis_list)[1]-1):
			timeend = timestamp+window
			for ensnum in range(np.shape(emis_list)[0]):
				timerange = np.arange(timestamp,timeend)
				basefield = emis_list[ensnum,emisind,:,:]
				model_emis[ensnum,timestamp:timeend] = np.array([emis_t_model(t, basefield)[xloc,yloc] for t in timerange])
			timestamp = timeend
	base_emis_field = settings_golem.makeBaseField()
	emis_t = settings_golem.makeEmis(functype='full')
	nature_emis = np.array([emis_t(t, base_emis_field) for t in plottime])
	endval = settings_golem.getSetting('endtime')+1
	plt.rcParams.update({'font.size': 20})
	for i in range(settings_golem.getSetting('n_ens')):
		plt.plot(plottime[0:endval-1],model_emis[i,0:endval-1],label='ensemble')
	plt.plot(plottime[0:endval-1],nature_emis[0:endval-1,xloc,yloc],label='nature')
	plt.xlabel('Time')
	plt.ylabel('Emissions')

def animate_field(data):
	if len(np.shape(data))==3:
		def animate(i):
			im.set_array(data[i,:,:])
			return [im]

		fig, ax = plt.subplots()
		im = ax.imshow(data[0,:,:], animated=True)
		im.set_clim(np.min(data), np.max(data))
		plt.colorbar(im);
		anim = animation.FuncAnimation(fig, animate,np.shape(data)[0],interval=50, blit=False)
	else:
		def animate(i):
			im.set_array(np.mean(data[:,i,:,:],axis=0))
			return [im]

		fig, ax = plt.subplots()
		im = ax.imshow(np.mean(data[:,0,:,:],axis=0), animated=True)
		im.set_clim(np.min(np.mean(data,axis=0)), np.max(np.mean(data,axis=0)))
		plt.colorbar(im);
		anim = animation.FuncAnimation(fig, animate,np.shape(data)[1],interval=50, blit=False)
	return anim 
