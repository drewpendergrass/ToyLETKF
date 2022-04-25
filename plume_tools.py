import numpy as np
import scipy.linalg as la
import pickle
import sys


#current step and step immediately previous. We are given the values of 
#timestep 0 and 1, so this method will just be called repeatedly.
def compute_next_step(cur_step,timeval,base_field,emis_func,settings_golem,isModel = False):
	arr_dim = np.shape(cur_step)
	next_step = np.zeros(arr_dim)
	delta_t,diffusion_const,h,n_x,n_y = settings_golem.getSetting(["delta_t","diffusion_const","h","n_x","n_y"])
	velx = settings_golem.makeVel('x',isModel = isModel)
	vely = settings_golem.makeVel('y',isModel = isModel)
	if isModel:
		lifetime = settings_golem.getSetting('lifetime_model')
	else:
		lifetime = settings_golem.getSetting('lifetime_nature')
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
			vx = velx(timeval,j,k)
			if vx>=0:
				vel_x_nextstep = (-1*vx*delta_t**2/h)*(c_n_j_k-c_n_jminus1_k) #backwards discretization
			else:
				vel_x_nextstep = (-1*vx*delta_t**2/h)*(c_n_jplus1_k-c_n_j_k) #forwards discretization
			vy = vely(timeval,j,k)
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
def compute_ts(settings_golem, base_field,emis_func, isModel = False, initial_conditions = None, time_start = 0, time_end=100):
	n_x,n_y,delta_t = settings_golem.getSetting(["n_x","n_y","delta_t"])
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
		next_step, next_time = compute_next_step(cur_step,cur_time,base_field=base_field,emis_func=emis_func,settings_golem = settings_golem, isModel=isModel)
		cur_step = np.copy(next_step)
		cur_time = np.copy(next_time)
		ts.append(cur_step)
		ts_times.append(cur_time)
	#Time in first entry
	ts = np.stack(ts)
	ts_times = np.array(ts_times)
	return [ts,ts_times]

#Initial conditions is list in form [nature, ensemble], with no time
#vel_funcs is a list of two functions, first for x second for y
def compute_nature_and_ens(settings_golem,ens_emis_fields,initial_conditions = None,time_start = 0, time_end=100):
	base_emis_field = settings_golem.makeBaseField()
	n_ens,emisModel = settings_golem.getSetting(['n_ens','emis_model_type'])
	nature_emis_func = settings_golem.makeEmis(functype=='full')
	model_emis_func = settings_golem.makeEmis(functype=emisModel)
	if initial_conditions is None:
		nature,timevals = compute_ts(settings_golem, base_emis_field,emis_func=nature_emis_func,isModel = False, initial_conditions = None, time_start = time_start, time_end=time_end)
		natureshape = np.shape(nature)
		ensemble = np.zeros((n_ens,natureshape[0],natureshape[1],natureshape[2]))
		for i in range(n_ens):
			ensemble[i,:,:,:],_ = compute_ts(settings_golem,base_field=ens_emis_fields[i],emis_func=model_emis_func,isModel = True, initial_conditions = None, time_start = time_start, time_end=time_end)
	else:
		natureinit = initial_conditions[0]
		ensinit = initial_conditions[1]
		nature,timevals = compute_ts(settings_golem, base_emis_field,emis_func=nature_emis_func,isModel = False, initial_conditions = natureinit, time_start = time_start, time_end=time_end)
		natureshape = np.shape(nature)
		ensemble = np.zeros((n_ens,natureshape[0],natureshape[1],natureshape[2]))
		for i in range(n_ens):
			ensemble[i,:,:,:],_ = compute_ts(settings_golem,base_field=ens_emis_fields[i],emis_func=model_emis_func,isModel = True, initial_conditions = ensinit[i,:,:], time_start = time_start, time_end=time_end)
	return [nature,ensemble,timevals]

