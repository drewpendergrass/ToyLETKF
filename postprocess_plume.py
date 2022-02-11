import numpy as np
import scipy.linalg as la
import pickle
import sys

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
emis_timescales_to_test = np.array([5,10,25,50,100,200,500]) #Will set long timescale to 1e6 to prevent any effect
emis_ind = index % 7
emis_timescales_to_test = emis_timescales_to_test[emis_ind:(emis_ind+1)]
vel_timescales_to_test = np.array([5,50,100])
vel_ind = int(np.floor(index/7))
vel_timescales_to_test = vel_timescales_to_test[vel_ind:(vel_ind+1)]
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


bias = np.mean(subset['Predicted PM']-subset['Actual PM'])
medbias = np.median(subset['Predicted PM']-subset['Actual PM'])
print(f'{country_name} {freqname} Mean bias: {np.round(bias,4)}')
print(f'{country_name} {freqname} Median bias: {np.round(medbias,4)}')
pm_abs_errors = np.abs(subset['Actual PM']-subset['Predicted PM'])
print(f'{country_name} {freqname} Mean abs. err.: {np.round(np.mean(pm_abs_errors),4)}')
print(f'{country_name} {freqname} Med. abs. err.: {np.round(np.median(pm_abs_errors),4)}')
mape_raw = 100 * (pm_abs_errors / subset['Actual PM'])
mape = np.mean(mape_raw[np.isfinite(mape_raw)])
print(f'{country_name} {freqname} Mean abs. % err.: {np.round(mape,4)}%')
medape = np.median(mape_raw[np.isfinite(mape_raw)])
print(f'{country_name} {freqname} Med. abs. % err: {np.round(medape,4)}%')
slope, intercept, r_value, p_value, std_err = linregress(subset['Actual PM'],subset['Predicted PM'])
print(f'{country_name} {freqname} R: {np.round(r_value,5)}')
print(f'{country_name} {freqname} R2: {np.round(r_value**2,5)}')
rmse = mean_squared_error(subset['Actual PM'],subset['Predicted PM'],squared=False)
print(f'{country_name} {freqname} RMSE: {np.round(rmse,4)}')
rrmse = 100*(rmse/np.mean(subset['Actual PM']))
print(f'{country_name} {freqname} RRMSE (normalized by observed mean): {np.round(rrmse,4)}%')
