import numpy as np
import scipy.linalg as la
import pickle
import sys
import helper_tools as ht




windows_to_test = np.array([2,5,10,25,50,125])
#freq_obs_to_test = [[1],[1,2],[1,2,5],[2,5,10],[5,10,25],[10,25,50]]
freq_obs_to_test = [[1],[2],[5],[10],[25],[50]]
gamma_
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
