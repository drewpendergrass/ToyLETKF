import numpy as np
import scipy.linalg as la
import pickle
import sys

#Constants for the problem 
h = 15 #m, grid spacing
delta_t = 1 #timestep, in minutes
diffusion_const = 0.01 #m^2/min, diffusion constant

#Grid settings
n_x = 30 #number of x coordinates
n_y = 30 #y coordinates
n_ens = 32


base_emis_field = np.zeros((n_x,n_y))
#base_emis_field = 10*np.random.rand(n_x,n_y)
base_emis_field[14,14] = 10/delta_t
#base_emis_field[9,9] = 10/delta_t
#base_emis_field[9,19] = 5/delta_t
#base_emis_field[19,9] = 15/delta_t
#base_emis_field[19,19] = 25/delta_t

#Velocity as a function of time
def vel(timestep,timescale=100,maxval=10,noise=None):
	if noise:
		return maxval*np.cos((timestep/timescale)*2*np.pi)+(noise*((np.random.rand()*2)-1))
	else:
		return maxval*np.cos((timestep/timescale)*2*np.pi)


distmat = np.zeros((n_x,n_y,n_x,n_y))
for x1 in range(n_x):
	for y1 in range(n_y):
		for x2 in range(n_x):
			for y2 in range(n_y):
				distmat[x1,y1,x2,y2] = np.sqrt((x2-x1)**2+(y2-y1)**2)

def make_emis_t(emis_variability,emis_timescale_short,emis_timescale_long):
	def emis_t(timestep,base_field):
		return (emis_variability*base_field)*(np.sin((timestep/emis_timescale_short)*2*np.pi)+np.sin((timestep/emis_timescale_long)*2*np.pi))+base_field
	return emis_t

def const_emis_t(timestep,base_field):
	return base_field

#computes the next time step given matrices representing values at
#current step and step immediately previous. We are given the values of 
#timestep 0 and 1, so this method will just be called repeatedly.
def compute_next_step(cur_step,timeval,base_field=base_emis_field,emis_func = None,lifetime=500,vel_x_timescale = 100,vel_y_timescale = 100,vel_x_max = 10,vel_y_max = 5,vel_x_noise=None,vel_y_noise=None):
	arr_dim = np.shape(cur_step)
	next_step = np.zeros(arr_dim)
	for j in range(arr_dim[0]):
		for k in range(arr_dim[1]):
			#Do the normal calculation
			c_n_j_k = cur_step[j,k]
			#j+1, k
			if j+1==n_x:
				c_n_jplus1_k = 0 #0 boundary conditions
			else:
				c_n_jplus1_k = cur_step[j+1,k]
			#j,k+1
			if k+1==n_y:
				c_n_j_kplus1 = 0
			else:
				c_n_j_kplus1 = cur_step[j,k+1]
			#j-1, k
			if j==0:
				c_n_jminus1_k = 0
			else:
				c_n_jminus1_k = cur_step[j-1,k]
			#j,k-1
			if k==0:
				c_n_j_kminus1 = 0
			else:
				c_n_j_kminus1 = cur_step[j,k-1]
			#Compute the next step
			timeder_nextstep = c_n_j_k
			diffusion_nextstep = (delta_t**2*diffusion_const/h**2)*(c_n_jplus1_k+c_n_j_kplus1-(4*c_n_j_k)+c_n_jminus1_k+c_n_j_kminus1)
			vx = vel(timeval,vel_x_timescale,vel_x_max,vel_x_noise)
			if vx>=0:
				vel_x_nextstep = (-1*vx*delta_t**2/h)*(c_n_j_k-c_n_jminus1_k) #backwards discretization
			else:
				vel_x_nextstep = (-1*vx*delta_t**2/h)*(c_n_jplus1_k-c_n_j_k) #forwards discretization
			vy = vel(timeval,vel_y_timescale,vel_y_max,vel_y_noise)
			if vy>=0:
				vel_y_nextstep = (-1*vy*delta_t**2/h)*(c_n_j_k-c_n_j_kminus1) #backwards discretization
			else:
				vel_y_nextstep = (-1*vy*delta_t**2/h)*(c_n_j_kplus1-c_n_j_k) #forwards discretization
			lambda_nextstep = (-1*delta_t**2/lifetime)*c_n_j_k
			next_step[j,k] = timeder_nextstep + diffusion_nextstep + vel_x_nextstep + vel_y_nextstep + lambda_nextstep
	#Add emissions
	next_step += emis_func(timeval,base_field)
	return next_step,timeval+delta_t

	#Save out a timeseries. Time values given in units of delta_t
def compute_ts(velocity_data, initial_conditions = None, time_start = 0, time_end=100,lifetime=500,base_field=base_emis_field,emis_func=const_emis_t):
	if initial_conditions is None:
		step0 = np.zeros((n_x,n_y))
		step0 += emis_func(0,base_field=base_field)
		cur_time = 0
		cur_step = step0
	else: 
		cur_time = time_start*delta_t
		cur_step = initial_conditions
	ts = [cur_step]
	ts_times = [cur_time]
	while cur_time < time_end*delta_t:
		next_step, next_time = compute_next_step(cur_step,cur_time,base_field=base_field,lifetime=lifetime,emis_func=emis_func,**velocity_data)
		cur_step = np.copy(next_step)
		cur_time = np.copy(next_time)
		ts.append(cur_step)
		ts_times.append(cur_time)
	#Time in first entry
	ts = np.stack(ts)
	ts_times = np.array(ts_times)
	return [ts,ts_times]

def makeInitialEnsEmisFields():
	ensfields = []
	for i in range(n_ens):
		ensfields.append(base_emis_field*(np.random.rand(n_x,n_y)+0.5))
		#ensfields.append(base_emis_field*(i+(n_ens/2))/n_ens)
	return ensfields

#Initial conditions is list in form [nature, ensemble], with no time
def compute_nature_and_ens(nature_vel_data,model_vel_data,ens_emis_fields,initial_conditions = None,time_start = 0, time_end=100,lifetime=500,nature_emis_func = None, model_emis_func = const_emis_t):
	if initial_conditions is None:
		nature,timevals = compute_ts(velocity_data=nature_vel_data,time_start = time_start, time_end=time_end,lifetime=lifetime,emis_func=nature_emis_func)
		natureshape = np.shape(nature)
		ensemble = np.zeros((n_ens,natureshape[0],natureshape[1],natureshape[2]))
		for i in range(n_ens):
			ensemble[i,:,:,:],_ = compute_ts(velocity_data=model_vel_data,time_start = time_start, time_end=time_end,lifetime=lifetime,base_field=ens_emis_fields[i],emis_func=model_emis_func)
	else:
		natureinit = initial_conditions[0]
		ensinit = initial_conditions[1]
		nature,timevals = compute_ts(velocity_data=nature_vel_data,initial_conditions = natureinit, time_start = time_start, time_end=time_end,lifetime=lifetime,emis_func=nature_emis_func)
		natureshape = np.shape(nature)
		ensemble = np.zeros((n_ens,natureshape[0],natureshape[1],natureshape[2]))
		for i in range(n_ens):
			ensemble[i,:,:,:],_ = compute_ts(velocity_data=model_vel_data,initial_conditions = ensinit[i,:,:],time_start = time_start, time_end=time_end,base_field=ens_emis_fields[i],lifetime=lifetime,emis_func=model_emis_func)
	return [nature,ensemble,timevals]


#Observation operator class

class Obs_Operator(object):
	def __init__(self,nature,model_ens,timevals, obs_times, distmat, nature_bias=0, nature_err=1, halg = 'all',errtype='absolute'):
		self.nature = nature
		self.model_ens = model_ens
		self.timevals = timevals
		self.obs_times = obs_times
		self.obs_time_inds = np.array([np.where(ot==self.timevals)[0][0] for ot in self.obs_times])
		self.bias=nature_bias
		self.err=nature_err
		self.halg = halg
		self.distmat = distmat
		self.errtype = errtype
		if self.halg=='all':
			self.xinds = np.tile(np.arange(0,n_x),n_y)
			self.yinds = np.repeat(np.arange(0,n_x),n_y)
		self.obsMean,self.obsPert,self.obsDiff,self.obsErr=self.obsMeanPertDiff()
	#Model ens has same times as in nature.
	def obsMeanPertDiff(self):
		lenobs = len(self.obs_time_inds)*len(self.xinds)
		nature_obs = np.zeros(lenobs)
		ens_obs = np.zeros((np.shape(self.model_ens)[0],lenobs))
		nature_err = np.zeros(lenobs)
		for ind,timeind in enumerate(self.obs_time_inds):
			nature_obs[(ind*len(self.xinds)):((ind+1)*len(self.xinds))],nature_err[(ind*len(self.xinds)):((ind+1)*len(self.xinds))] = self.H(self.nature[timeind,:,:],allowErrors=True)
			for i in range(np.shape(self.model_ens)[0]):
				ens_obs[i,(ind*len(self.xinds)):((ind+1)*len(self.xinds))],_ = self.H(self.model_ens[i,timeind,:,:],allowErrors=False)
		obsMean = np.mean(ens_obs,axis = 0)
		obsPert = np.zeros(np.shape(ens_obs))
		for j in range(np.shape(ens_obs)[0]):
			obsPert[j,:] = ens_obs[j,:]-obsMean
		obsDiff = nature_obs-obsMean
		return [obsMean,obsPert,obsDiff,nature_err]
	def H(self,conc2D, allowErrors=False):
		if self.halg=='all':
			to_return = conc2D.flatten()
		if (self.bias is not None) and (self.err is not None) and allowErrors:
			if self.errtype=='absolute':
				errorvec = np.repeat(self.err, len(to_return))
				to_return += np.random.normal(self.bias, self.err, np.shape(to_return))
			elif self.errtype=='relative':
				biasvec = to_return*self.bias
				errorvec = to_return*self.err
				to_return += np.random.normal(biasvec, errorvec, np.shape(to_return))
			else:
				raise ValueError('Error type must be relative or absolute.')
			to_return[to_return<0] = 0
		else:
			errorvec = np.zeros(len(to_return))
		return [to_return,errorvec]
	#expect radius in ind space
	def subsetByLoc(self,radius,xind,yind):
		inds_in_range = np.where(self.distmat[xind,yind,:,:]<=radius)
		xindrep = np.tile(self.xinds,len(self.obs_time_inds))
		yindrep = np.tile(self.yinds,len(self.obs_time_inds))
		matchinds = np.concatenate([np.where((xi==xindrep) & (yi==yindrep))[0] for xi,yi in zip(*inds_in_range)])
		return [self.obsMean[matchinds],self.obsPert[:,matchinds],self.obsDiff[matchinds],self.obsErr[matchinds]]

#Assimilator class

class Assimilator(object):
	def __init__(self,nature,model_ens,timevals, obs_times,emissions,distmat,nature_bias=0, nature_err=1, halg = 'all',errtype='absolute'):
		self.xinds = np.tile(np.tile(np.arange(0,n_x),n_y),2) #flatten concentrations for first half, emissions
		self.yinds = np.tile(np.repeat(np.arange(0,n_x),n_y),2)
		emissions = np.stack(emissions)
		self.ens_mean = np.concatenate((np.mean(model_ens[:,-1,:,:],axis=0).flatten(),np.mean(emissions,axis=0).flatten()))
		self.ens_pert = np.zeros((np.shape(model_ens)[0],len(self.ens_mean)))
		self.backgroundEnsemble = np.zeros((np.shape(model_ens)[0],len(self.ens_mean)))
		for i in range(np.shape(model_ens)[0]):
			background = np.concatenate([model_ens[i,-1,:,:].flatten(),emissions[i,:,:].flatten()])
			self.ens_pert[i,:] = background-self.ens_mean
			self.backgroundEnsemble[i,:] = background
		self.nature_err = nature_err
		self.distmat = distmat
		self.obs_op = Obs_Operator(nature,model_ens,timevals, obs_times, self.distmat, nature_bias, nature_err, halg)
	def getMatchInds(self,radius,xind,yind):
		inds_in_range = np.where(self.distmat[xind,yind,:,:]<=radius)
		matchinds = np.concatenate([np.where((xi==self.xinds) & (yi==self.yinds))[0] for xi,yi in zip(*inds_in_range)])
		matchindcolfull = np.where((xind==self.xinds)&(yind==self.yinds))[0]
		matchindcolsubset = np.where((xind==self.xinds[matchinds])&(yind==self.yinds[matchinds]))[0]
		return [matchinds,matchindcolfull,matchindcolsubset]
	def subsetByLoc(self,matchinds,radius,xind,yind):
		to_return = [self.ens_mean[matchinds],self.ens_pert[:,matchinds],self.backgroundEnsemble[:,matchinds]]+self.obs_op.subsetByLoc(radius,xind,yind)
		return to_return
	def LETKF(self,radius,inflation,scalingInflator):
		ens_analysis = np.zeros(np.shape(self.ens_pert))
		for x in range(n_x):
			for y in range(n_y):
				matchinds,matchindcolfull,matchindcolsubset = self.getMatchInds(radius,x,y)
				ensmean,enspert,backgroundEnsemble,obsmean,obspert,obsdiff,obserr = self.subsetByLoc(matchinds,radius,x,y)
				R = np.diag(obserr**2)
				C = obspert @ la.inv(R)
				cyb = C @ np.transpose(obspert)
				k = np.shape(obspert)[0]
				iden = (k-1)*np.identity(k)/(1+inflation)
				PtildeAnalysis = la.inv(iden+cyb)
				WAnalysis = la.sqrtm((k-1)*PtildeAnalysis)
				WbarAnalysis = PtildeAnalysis @ C @ obsdiff
				for i in range(k):
					WAnalysis[:,i]+=WbarAnalysis
				analysisEnsemble = np.zeros(np.shape(enspert))
				for i in range(k):
					analysisEnsemble[i,:] = np.transpose(enspert).dot(WAnalysis[:,i]) +ensmean
				#Inflate scalings to the X percent of the background standard deviation, per Miyazaki et al 2015
				if ~np.isnan(scalingInflator):
					analysisScalefactor = np.copy(analysisEnsemble)[:,1::2] #odd entries are scale factors
					backgroundScalefactor = np.copy(backgroundEnsemble)[:,1::2]
					analysis_std = np.std(analysisScalefactor,axis=0)
					background_std = np.std(backgroundScalefactor,axis=0)
					ratio = np.empty(len(analysis_std))
					ratio[:] = np.nan
					nonzero = np.where(background_std!=0)[0]
					ratio[nonzero] = analysis_std[nonzero]/background_std[nonzero]
					whereToInflate = np.where(ratio<scalingInflator)[0] #nans show up as false, removed
					if len(whereToInflate>0):
						new_std = scalingInflator*background_std[whereToInflate]
						for i in whereToInflate:
							meanrebalance = np.mean(analysisScalefactor[:,i])*((new_std/analysis_std[i])-1)
							analysisScalefactor[:,i] = analysisScalefactor[:,i]*(new_std/analysis_std[i])-meanrebalance
						analysisEnsemble[:,1::2] = analysisScalefactor
				for i in range(k):
					ens_analysis[i,matchindcolfull] = analysisEnsemble[i,matchindcolsubset]
		return ens_analysis
	def update(self,radius,inflation,scalingInflator):
		ens_analysis = self.LETKF(radius,inflation,scalingInflator)
		len_ens = np.shape(ens_analysis)[1]
		ens_conc = ens_analysis[:,0:int(len_ens/2)].reshape((n_ens,n_x,n_y))
		ens_emis = ens_analysis[:,int(len_ens/2):len_ens].reshape((n_ens,n_x,n_y))
		return [ens_conc,ens_emis]

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
		assim = Assimilator(nature,ensemble,timevals, obs_times,ensfields,distmat,nature_bias=nature_bias,nature_err=nature_err)
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


windows_to_test = np.array([2,5,10,25,50,125])
freq_obs_to_test = [[1],[1,2],[1,2,5],[2,5,10],[5,10,25],[10,25,50]]
#radii_to_test = np.array(1,2,5,10)
radius = 5 
#scaling_inflator_to_test = np.array([0.3,0.5,0.9])
scalingInflator = 0.9
#nature_relerr_to_test = np.array([0.01,0.05,0.1,0.25,0.5])
nature_relerr_to_test = 0.25

index = int(sys.argv[1]) #which simulation set to do, ranges from 0 to 20 inclusive

lifetimes_to_test = np.array([5,10,25,50,100,200,500])
#lifetimes_to_test = np.array([25])
emis_timescales_to_test = np.array([5,10,25,50,100,200,500]) #Will set long timescale to 1e6 to prevent any effect
emis_ind = index % 7
emis_timescales_to_test = emis_timescales_to_test[emis_ind:(emis_ind+1)]
#emis_timescales_to_test = np.array([200])
vel_timescales_to_test = np.array([5,50,100])
vel_ind = int(np.floor(index/7))
vel_timescales_to_test = vel_timescales_to_test[vel_ind:(vel_ind+1)]
#vel_timescales_to_test = np.array([100])
#vel_x_noise_to_test = np.array([0.1,0.5,1,5])
vel_x_noise = 2
#vel_y_noise_to_test = np.array([0.05,0.25,0.5,2.5])
vel_y_noise = 1
vel_x_max=10
vel_y_max=5
emis_variability = 0.25
inflation=0.2
endtime = 250 
naturebias = 0

misc_info = {'radius':radius,'scalingInflator':scalingInflator,'nature_relerr':nature_relerr_to_test,'vel_x_noise':vel_x_noise,'vel_y_noise':vel_y_noise,'vel_x_max':vel_x_max,'vel_y_max':vel_y_max,'emis_variability':emis_variability,'inflation':inflation,'endtime':endtime,'naturebias':naturebias}
for window,freqsubset in zip(windows_to_test,freq_obs_to_test):
	for freq in freqsubset:
		for lifetime in lifetimes_to_test:
			for emistime in emis_timescales_to_test:
				for veltime in vel_timescales_to_test:
					emis_t = make_emis_t(emis_variability,emistime,1e8)
					nature_vel_data = {"vel_x_timescale":veltime,"vel_y_timescale":veltime,"vel_x_max":vel_x_max,"vel_y_max":vel_y_max,"vel_x_noise":vel_x_noise,"vel_y_noise":vel_y_noise}
					model_vel_data = {"vel_x_timescale":veltime,"vel_y_timescale":veltime,"vel_x_max":vel_x_max,"vel_y_max":vel_y_max,"vel_x_noise":None,"vel_y_noise":None}
					assim_settings = {"nature_vel_data":nature_vel_data,"model_vel_data":model_vel_data,"radius":radius,"inflation":inflation,"scalingInflator":scalingInflator,"freq_obs":freq,"window":window,"end_time":endtime,"nature_bias":naturebias,"nature_err":nature_relerr_to_test,"errtype":'relative',"lifetime":lifetime,"nature_emis_func" : emis_t, "model_emis_func" : const_emis_t}
					nature_list,ens_list,emis_list,time_list = run_with_assimilation(**assim_settings)
					model_emis = np.repeat(emis_list[:,:,14,14],assim_settings['window'],axis=1)
					nature_emis = np.array([emis_t(t, base_emis_field) for t in np.unique(time_list)])
					to_save = {"nature":nature_list,"ensemble":ens_list,"time":time_list,"model_emis":model_emis,"nature_emis":nature_emis,"veltime":veltime,"emistime":emistime,"lifetime":lifetime,"window":window,"frequency":freq,'misc_info':misc_info}
					filename = f"/n/holyscratch01/jacob_lab/dpendergrass/toyDA/{window}/rundata_window-{window}_freq-{freq}_lifetime-{lifetime}_emis-{emistime}_vel-{veltime}.pkl"
					with open(filename, 'wb') as f:
						pickle.dump(to_save, f)
