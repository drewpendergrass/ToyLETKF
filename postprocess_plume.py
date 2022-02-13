import numpy as np
import pickle
import pandas as pd
from scipy.stats import linregress
from sklearn.metrics import mean_squared_error
from glob import glob

df = pd.DataFrame(columns = ['Window', 'Frequency', 'Lifetime', 'Emistime','Veltime','ConcBias','ConcR2','ConcRMSE','ConcRRMSE','ConcMeanAbsError','ConcMeanAbsPercentError','EmisBias','EmisR2','EmisRMSE','EmisRRMSE','EmisMeanAbsError','EmisMeanAbsPercentError'])
  
files_to_open = glob('/n/holyscratch01/jacob_lab/dpendergrass/toyDA/**/rundata_*.pkl', recursive=True)
for file in files_to_open:
	with open(file, 'rb') as handle:
		results = pickle.load(handle)
	ensmean = np.mean(results['ensemble'],axis=(0,2,3))
	naturemean = np.mean(results['nature'],axis=(1,2))
	ConcBias = np.mean(ensmean-naturemean)
	slope, intercept, r_value, p_value, std_err = linregress(naturemean,ensmean)
	ConcR2 = r_value**2
	ConcRMSE = mean_squared_error(naturemean,ensmean,squared=False)
	ConcRRMSE = ConcRMSE/np.mean(naturemean)
	ConcMeanAbsError = np.mean(np.abs(ensmean-naturemean))
	ConcMeanAbsPercentError = 100*(ConcMeanAbsError/np.mean(naturemean))
	endval = 251
	ensemismean = np.mean(results['model_emis'], axis=0)[0:endval]
	natureemis = results['nature_emis'][:,14,14]
	EmisBias = np.mean(ensemismean-natureemis)
	slope, intercept, r_value, p_value, std_err = linregress(natureemis,ensemismean)
	EmisR2 = r_value**2
	EmisRMSE = mean_squared_error(natureemis,ensemismean,squared=False)
	EmisRRMSE = EmisRMSE/np.mean(natureemis)
	EmisMeanAbsError = np.mean(np.abs(ensemismean-natureemis))
	EmisMeanAbsPercentError = 100*(EmisMeanAbsError/np.mean(natureemis))
	df = df.append({'Window' : results['window'], 'Frequency' : results['frequency'], 'Lifetime' : results['lifetime'], 'Emistime' : results['emistime'], 'Veltime' : results['veltime'],'ConcBias':ConcBias,'ConcR2':ConcR2,'ConcRMSE':ConcRMSE,'ConcRRMSE':ConcRRMSE,'ConcMeanAbsError':ConcMeanAbsError,'ConcMeanAbsPercentError':ConcMeanAbsPercentError,'EmisBias':EmisBias,'EmisR2':EmisR2,'EmisRMSE':EmisRMSE,'EmisRRMSE':EmisRRMSE,'EmisMeanAbsError':EmisMeanAbsError,'EmisMeanAbsPercentError':EmisMeanAbsPercentError},ignore_index = True)

df.to_csv('/n/holyscratch01/jacob_lab/dpendergrass/toyDA/accuracy_output.csv')
