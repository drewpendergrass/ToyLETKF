import numpy as np
import scipy.linalg as la
import pickle
import sys
import skimage.util as ski


#Observation operator class

class Obs_Operator(object):
	def __init__(self,nature,model_ens,timevals, obs_times, distmat, settings_golem):
		self.nature = nature
		self.model_ens = model_ens
		self.timevals = timevals
		self.obs_times = obs_times
		self.settings_golem = settings_golem
		self.obs_time_inds = np.array([np.where(ot==self.timevals)[0][0] for ot in self.obs_times])
		n_x,n_y,self.bias,self.err,self.halg = settings_golem.getSetting(['n_x','n_y','nature_bias','nature_err','halg'])
		self.distmat = distmat
		self.errtype = settings_golem.settings["nature_err"]["type"]
		if self.halg=='all':
			self.xinds = np.tile(np.arange(0,n_x),n_y)
			self.yinds = np.repeat(np.arange(0,n_x),n_y)
		#We only observe the mean of the entire state.
		elif self.halg=='one':
			self.xinds = np.array([int(np.round(np.mean([0,n_x-1])))])
			self.yinds = np.array([int(np.round(np.mean([0,n_y-1])))])
		#List of tuple coordinates, which we observe
		elif (type(self.halg) is list) and (type(self.halg[0]) is list):
			self.xinds = np.array([val[0] for val in self.halg])
			self.yinds = np.array([val[1] for val in self.halg])
		#Observations aggregated to grid of specified dimension
		elif (type(self.halg) is list) and (type(self.halg[0]) is int):
			if (n_x % self.halg[0] == 0) and (n_y % self.halg[1] == 0):
				self.aggregator,self.xinds,self.yinds = makeAggregator(n_x,n_y,self.halg[0],self.halg[1])
			else:
				raise ValueError('Observation grid does not partition advection-diffusion grid.')
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
			outshape = np.shape(conc2D)
		if self.halg=='one':
			to_return = np.array([np.mean(conc2D)])
			outshape = (1,1)
		if (type(self.halg) is list) and (type(self.halg[0]) is list):
			to_return = conc2D[self.xinds,self.yinds]
			outshape = np.shape(conc2D)
		if (type(self.halg) is list) and (type(self.halg[0]) is int):
			to_return = self.aggregator(conc2D)
			outshape = (self.halg[0],self.halg[1])
		if (self.bias is not None) and (self.err is not None) and allowErrors:
			if self.errtype=='absolute':
				noisefield = self.settings_golem.makeObsCorrelatedErrorField(self.bias, self.err, outshape = outshape)
				if noisefield is None:
					to_return += np.random.normal(self.bias, self.err, np.shape(to_return))
				else:
					if (type(self.halg) is list) and (type(self.halg[0]) is list):
						to_return += noisefield[self.xinds,self.yinds]
					else:
						to_return += noisefield.flatten()
			elif self.errtype=='relative':
				biasvec = to_return*self.bias
				errorvec = to_return*self.err
				errorvec[errorvec<=0.01] = 0.01 #so that we can invert
				noisefield = self.settings_golem.makeObsCorrelatedErrorField(biasvec, errorvec,outshape = outshape)
				if noisefield is None:
					to_return += np.random.normal(biasvec, errorvec, np.shape(to_return))
				else:
					if (type(self.halg) is list) and (type(self.halg[0]) is list):
						to_return += noisefield[self.xinds,self.yinds]
					else:
						to_return += noisefield.flatten()
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

def makeAggregator(nx_big,ny_big,nx_small,ny_small):
	X,Y = np.meshgrid(np.arange(nx_big),np.arange(ny_big),indexing='ij')
	aggsize_x = int(nx_big/nx_small)
	aggsize_y = int(ny_big/ny_small)
	xinds = np.round(ski.view_as_blocks(X, (aggsize_x,aggsize_y)).mean(axis=(2,3)).flatten()).astype(int)
	yinds = np.round(ski.view_as_blocks(Y, (aggsize_x,aggsize_y)).mean(axis=(2,3)).flatten()).astype(int)
	def AggregateToObsGrid(observed_values):
		return ski.view_as_blocks(observed_values, (aggsize_x,aggsize_y)).mean(axis=(2,3)).flatten()
	return AggregateToObsGrid,xinds,yinds


#Assimilator class

class Assimilator(object):
	def __init__(self,settings_golem,nature,model_ens,timevals, obs_times,emissions):
		self.settings_golem = settings_golem
		self.distmat = settings_golem.getDistMat()
		n_x,n_y = settings_golem.getSetting(['n_x','n_y'])
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
		self.obs_op = Obs_Operator(nature,model_ens,timevals, obs_times, self.distmat, settings_golem)
	def getMatchInds(self,radius,xind,yind):
		inds_in_range = np.where(self.distmat[xind,yind,:,:]<=radius)
		matchinds = np.concatenate([np.where((xi==self.xinds) & (yi==self.yinds))[0] for xi,yi in zip(*inds_in_range)])
		matchindcolfull = np.where((xind==self.xinds)&(yind==self.yinds))[0]
		matchindcolsubset = np.where((xind==self.xinds[matchinds])&(yind==self.yinds[matchinds]))[0]
		return [matchinds,matchindcolfull,matchindcolsubset]
	def subsetByLoc(self,matchinds,radius,xind,yind):
		to_return = [self.ens_mean[matchinds],self.ens_pert[:,matchinds],self.backgroundEnsemble[:,matchinds]]+self.obs_op.subsetByLoc(radius,xind,yind)
		return to_return
	def LETKF(self):
		radius,inflation,scalingInflator,n_x,n_y,gamma = self.settings_golem.getSetting(["radius","inflation","scalingInflator","n_x","n_y","gamma"]) 
		ens_analysis = np.zeros(np.shape(self.ens_pert))
		for x in range(n_x):
			for y in range(n_y):
				matchinds,matchindcolfull,matchindcolsubset = self.getMatchInds(radius,x,y)
				ensmean,enspert,backgroundEnsemble,obsmean,obspert,obsdiff,obserr = self.subsetByLoc(matchinds,radius,x,y)
				k = np.shape(obspert)[0]
				if len(obserr)==0:
					#overwrite with background, no observation available
					for i in range(k):
						ens_analysis[i,matchindcolfull] = backgroundEnsemble[i,matchindcolsubset]
					continue #skip this round, no observatons
				R = np.diag(obserr**2)
				R *= gamma**-1 #scale R by inverse gamma
				C = obspert @ la.inv(R)
				cyb = C @ np.transpose(obspert)
				iden = (k-1)*np.identity(k)/(1+inflation)
				PtildeAnalysis = la.inv(iden+cyb)
				WAnalysis = la.sqrtm((k-1)*PtildeAnalysis)
				WbarAnalysis = PtildeAnalysis @ C @ obsdiff
				for i in range(k):
					WAnalysis[:,i]+=WbarAnalysis
				analysisEnsemble = np.zeros(np.shape(enspert))
				for i in range(k):
					analysisEnsemble[i,:] = np.transpose(enspert).dot(WAnalysis[:,i]) +ensmean
				#Inflate scalings to the X percent of the initial standard deviation, per Miyazaki et al 2015
				if ~np.isnan(scalingInflator):
					analysisScalefactor = np.copy(analysisEnsemble)[:,1::2] #odd entries are scale factors
					analysis_std = np.std(analysisScalefactor,axis=0)
					background_std = self.settings_golem.initial_ens_std.flatten()[matchinds[matchinds<np.shape(self.ens_pert)[1]/2]] #Subset using the first half of matchinds 
					ratio = np.empty(len(analysis_std))
					ratio[:] = np.nan
					nonzero = np.where(background_std!=0)[0]
					ratio[nonzero] = analysis_std[nonzero]/background_std[nonzero]
					whereToInflate = np.where(ratio<scalingInflator)[0] #nans show up as false, removed
					if len(whereToInflate>0):
						new_std = scalingInflator*background_std[whereToInflate]
						for ind, i in enumerate(whereToInflate):
							meanrebalance = np.mean(analysisScalefactor[:,i])*((new_std[ind]/analysis_std[i])-1)
							analysisScalefactor[:,i] = analysisScalefactor[:,i]*(new_std[ind]/analysis_std[i])-meanrebalance
						analysisEnsemble[:,1::2] = analysisScalefactor
				for i in range(k):
					ens_analysis[i,matchindcolfull] = analysisEnsemble[i,matchindcolsubset]
		return ens_analysis
	def update(self):
		n_ens,n_x,n_y = self.settings_golem.getSetting(["n_ens","n_x","n_y"]) 
		ens_analysis = self.LETKF()
		len_ens = np.shape(ens_analysis)[1]
		ens_conc = ens_analysis[:,0:int(len_ens/2)].reshape((n_ens,n_x,n_y))
		ens_emis = ens_analysis[:,int(len_ens/2):len_ens].reshape((n_ens,n_x,n_y))
		return [ens_conc,ens_emis]
