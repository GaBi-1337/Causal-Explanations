import numpy as np 

files = ['BI_SI.txt', 'DPI_BI.txt', 'DPI_HPI.txt', 'DPI_RI.txt', 'DPI_SI.txt', 'HPI_BI.txt', 'HPI_RI.txt', 'HPI_SI.txt', 'JI_BI.txt', 'JI_DPI.txt', 'JI_HPI.txt', 'JI_RI.txt', 'JI_SI.txt', 'RI_BI.txt', 'RI_SI.txt']

for file in files:
	f = open('scores/'+file, 'r')
	y = f.readlines()
	y = [float(item) for item in y]
	print(file)
	print("mean = %f" % (np.mean(y)))
	print("std = %f" % (np.std(y)))
	print("min = %f" % (np.min(y)))