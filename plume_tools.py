import numpy as np
import scipy.linalg as la
import pickle
import sys


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

