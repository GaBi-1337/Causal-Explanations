import numpy as np 

files = ['BI_SI.txt', 'DPI_BI.txt', 'DPI_HPI.txt', 'DPI_RI.txt', 'DPI_SI.txt', 'HPI_BI.txt', 'HPI_RI.txt', 'HPI_SI.txt', 'JI_BI.txt', 'JI_DPI.txt', 'JI_HPI.txt', 'JI_RI.txt', 'JI_SI.txt', 'RI_BI.txt', 'RI_SI.txt']

for file in files:
	f = open('scores/'+file, 'r')
	y = f.readlines()
	y = [[float(item) for item in line.split()] for line in y]
	print(file)
	print("mean = ", (np.mean(y, axis = 0)))
	print("std = ", (np.std(y, axis = 0)))
	print("min = ", (np.min(y, axis = 0)))